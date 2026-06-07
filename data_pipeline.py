from __future__ import annotations

import math
import re
import ast
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder


YC_DATASETS = {
    "y-combinator-directory": "miguelcorraljr/y-combinator-directory",
    "ycombinator-all-funded-companies-dataset": "sashakorovkina/ycombinator-all-funded-companies-dataset",
    "y-combinator-jobs-enriched": "lazarun/y-combinator-jobs-enriched",
}
STARTUP_DATASET_DIR = "startup-investments"
STARTUP_DATASET_SLUG = "justinas/startup-investments"
STARTUP_REQUIRED_FILES = {
    "objects.csv",
    "acquisitions.csv",
    "ipos.csv",
    "relationships.csv",
    "degrees.csv",
    "funding_rounds.csv",
}


def load_dashboard_data(base_dir: str | Path = ".") -> dict:
    base = Path(base_dir)
    raw_dir = base / "data" / "raw"
    readiness = dataset_readiness(base)

    yc = None
    if readiness["yc_ready"]:
        yc = load_yc_ecosystem(raw_dir)

    startup = None
    if readiness["startup_ready"]:
        startup = load_startup_intelligence(raw_dir / STARTUP_DATASET_DIR)

    return {
        "yc": yc,
        "startup": startup,
        "readiness": readiness,
    }


def dataset_readiness(base_dir: str | Path = ".") -> dict:
    base_path = Path(base_dir)
    raw_dir = Path(base_dir) / "data" / "raw"
    fetch_command = "./.venv/bin/python scripts/fetch_data.py" if (base_path / ".venv" / "bin" / "python").exists() else "python scripts/fetch_data.py"

    yc_status = []
    for folder, slug in YC_DATASETS.items():
        path = raw_dir / folder
        csvs = sorted(p.name for p in path.rglob("*.csv")) if path.exists() else []
        yc_status.append(
            {
                "name": slug,
                "folder": str(path),
                "ready": bool(csvs),
                "files": csvs,
            }
        )

    startup_path = raw_dir / STARTUP_DATASET_DIR
    startup_files = {p.name for p in startup_path.rglob("*.csv")} if startup_path.exists() else set()
    missing_startup = sorted(STARTUP_REQUIRED_FILES - startup_files)

    return {
        "raw_dir": str(raw_dir),
        "yc_status": yc_status,
        "yc_ready": all(item["ready"] for item in yc_status),
        "missing_yc": [item for item in yc_status if not item["ready"]],
        "startup_folder": str(startup_path),
        "startup_ready": not missing_startup,
        "startup_files": sorted(startup_files),
        "missing_startup_files": missing_startup,
        "fetch_command": fetch_command,
    }


def load_yc_ecosystem(raw_dir: Path) -> dict[str, pd.DataFrame | str]:
    frames = _load_named_frames(raw_dir, YC_DATASETS.keys())
    companies = _normalize_yc_companies(frames)
    founders = _normalize_yc_founders(frames)

    if companies.empty:
        return {
            "companies": companies,
            "founders": founders,
            "source": "YC Kaggle datasets loaded, but no company table matched expected columns.",
        }

    companies = _prepare_yc_features(companies)
    founders = _attach_founder_context(founders, companies)

    return {
        "companies": companies,
        "founders": founders,
        "source": "YC Kaggle datasets: " + ", ".join(YC_DATASETS.values()),
    }


def load_startup_intelligence(dataset_dir: Path) -> dict[str, pd.DataFrame | dict | str]:
    companies = _load_startup_companies(dataset_dir)
    founders = _load_startup_founder_features(dataset_dir)
    rounds = _load_startup_round_features(dataset_dir)

    model_df = companies.merge(founders, left_on="id", right_index=True, how="left").merge(
        rounds, left_on="id", right_index=True, how="left"
    )

    fill_zero_cols = [
        "founder_count",
        "founder_degree_records",
        "founders_with_degree",
        "founders_with_mba",
        "founders_with_phd",
        "founders_with_elite_school",
        "founder_prior_company_links",
        "observed_rounds",
        "total_raised_rounds",
        "median_round_usd",
        "avg_participants",
        "max_participants",
        "observed_funding_span",
    ]
    for col in fill_zero_cols:
        if col not in model_df:
            model_df[col] = 0
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce").fillna(0)

    ml_df = model_df[
        model_df["founded_year"].notna()
        & ((model_df["funding_rounds"].fillna(0) > 0) | (model_df["founder_count"] > 0))
    ].copy()

    model_result = _fit_startup_exit_model(ml_df)
    if not model_result["predictions"].empty:
        ml_df = ml_df.merge(model_result["predictions"], on="id", how="left")
    else:
        ml_df["predicted_exit_probability"] = np.nan

    return {
        "companies": companies,
        "model_df": model_df,
        "ml_df": ml_df,
        "model": model_result["model"],
        "model_features": model_result["features"],
        "metrics": model_result["metrics"],
        "importance": model_result["importance"],
        "signal_summary": _startup_signal_summary(ml_df),
        "sector_summary": _startup_sector_summary(ml_df),
        "country_summary": _startup_country_summary(ml_df),
        "source": STARTUP_DATASET_SLUG,
    }


def _load_named_frames(raw_dir: Path, dataset_folders: Iterable[str]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for folder in dataset_folders:
        root = raw_dir / folder
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.csv")):
            key = f"{folder}/{path.stem.lower()}"
            try:
                frames[key] = pd.read_csv(path, low_memory=False, encoding_errors="replace")
            except Exception:
                continue
    return frames


def _normalize_yc_companies(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    candidates = []
    for key, df in frames.items():
        lower_cols = {str(c).lower(): c for c in df.columns}
        has_company = any(c in lower_cols for c in ["company", "company_name", "name", "startup", "title"])
        has_yc_signal = any(c in lower_cols for c in ["batch", "yc_batch", "status", "industry", "industries", "tags"])
        if has_company and has_yc_signal:
            tmp = df.copy()
            tmp["_source_table"] = key
            candidates.append(tmp)

    if not candidates:
        return _empty_yc_companies()

    merged = pd.concat(candidates, ignore_index=True, sort=False)
    out = pd.DataFrame(index=merged.index)
    out["name"] = _first_present(merged, ["name", "company", "company_name", "startup", "title"])
    out["slug"] = _first_present(merged, ["slug"], default="")
    missing_slug = out["slug"].fillna("").astype(str).str.strip().eq("")
    out.loc[missing_slug, "slug"] = _slugify_series(out.loc[missing_slug, "name"])
    out["batch"] = _first_present(merged, ["batch", "yc_batch", "yc batch"], default="Unknown")
    out["status"] = _first_present(merged, ["status", "company_status"], default="Unknown")
    out["tags_text"] = _first_present(merged, ["tags", "tag", "badges", "keywords"], default="").map(_list_to_text)
    out["industry"] = _first_present(merged, ["industry", "industries", "category", "sector", "market"], default="")
    missing_industry = out["industry"].fillna("").astype(str).str.strip().isin(["", "nan", "None", "[]"])
    out.loc[missing_industry, "industry"] = out.loc[missing_industry, "tags_text"].map(_first_tag).replace("", "Unspecified")
    out["subindustry"] = _first_present(merged, ["subindustry", "sub_industry", "sub industry"], default="")
    out["region"] = _first_present(merged, ["region", "regions", "country", "location", "company_location", "all_locations", "address"], default="Unknown")
    out["one_liner"] = _first_present(merged, ["one_liner", "oneliner", "oneLiner", "description", "short_description", "tagline"], default="")
    out["website"] = _first_present(merged, ["website", "url", "company_url"], default="")
    out["team_size"] = _to_number(_first_present(merged, ["team_size", "teamsize", "teamSize", "team size", "employees", "employee_count"], default=0))
    open_jobs = _to_number(_first_present(merged, ["number_of_open_jobs", "open_jobs_count"], default=0))
    out["is_hiring"] = _to_bool(_first_present(merged, ["isHiring", "is_hiring", "hiring"], default=False)) | open_jobs.gt(0)
    out["top_company"] = _to_bool(_first_present(merged, ["top_company", "top company", "is_top"], default=False))
    out["source_table"] = merged["_source_table"]

    return out.dropna(subset=["name"]).drop_duplicates("slug").reset_index(drop=True)


def _empty_yc_companies() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "name",
            "slug",
            "batch",
            "status",
            "industry",
            "subindustry",
            "region",
            "one_liner",
            "website",
            "team_size",
            "is_hiring",
            "top_company",
            "tags_text",
            "source_table",
            "batch_year",
            "batch_era",
            "radar_score",
            "radar_tier",
            "recognition_score",
            "momentum_score",
            "topic_x",
            "topic_y",
        ]
    )


def _normalize_yc_founders(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    founder_rows = []
    for key, df in frames.items():
        cols = {str(c).lower(): c for c in df.columns}
        founder_col = _find_col(cols, ["founder", "founder_name", "name", "full_name"])
        if founder_col is None or not any(token in key for token in ["founder", "school", "prior", "education"]):
            continue

        company_col = _find_col(cols, ["company", "company_name", "startup", "name_company"])
        school_col = _find_col(cols, ["school", "university", "education", "institution"])
        prior_col = _find_col(cols, ["prior_company", "previous_company", "employer", "company_before"])

        tmp = pd.DataFrame(
            {
                "founder": df[founder_col].astype(str),
                "company": df[company_col].astype(str) if company_col else "",
                "school": df[school_col].astype(str) if school_col else "",
                "prior_company": df[prior_col].astype(str) if prior_col else "",
                "source_table": key,
            }
        )
        founder_rows.append(tmp)

    if not founder_rows:
        return _normalize_yc_bookface_founders(frames)

    founders = pd.concat(founder_rows, ignore_index=True).replace({"nan": "", "None": ""})
    founders["company_slug"] = _slugify_series(founders["company"])
    return founders.drop_duplicates()


def _normalize_yc_bookface_founders(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    founders = frames.get("ycombinator-all-funded-companies-dataset/founders")
    if founders is None or founders.empty:
        return pd.DataFrame(columns=["founder", "company", "school", "prior_company", "company_slug"])

    out = pd.DataFrame(index=founders.index)
    first = founders["first_name"].fillna("").astype(str) if "first_name" in founders else ""
    last = founders["last_name"].fillna("").astype(str) if "last_name" in founders else ""
    out["founder"] = (first + " " + last).str.strip()
    out["company"] = _first_present(founders, ["current_company"], default="")
    out["company_slug"] = _first_present(founders, ["company_slug"], default="").fillna("").astype(str)
    out["hnid"] = _first_present(founders, ["hnid"], default="").fillna("").astype(str)

    schools = frames.get("ycombinator-all-funded-companies-dataset/schools")
    if schools is not None and not schools.empty and {"hnid", "school"}.issubset(schools.columns):
        school_summary = (
            schools.assign(hnid=schools["hnid"].fillna("").astype(str), school=schools["school"].fillna("").astype(str).str.strip())
            .query("school != ''")
            .groupby("hnid")["school"]
            .agg(lambda s: ", ".join(sorted(set(s))[:4]))
        )
        out = out.merge(school_summary.rename("school"), left_on="hnid", right_index=True, how="left")
    else:
        out["school"] = ""

    prior = frames.get("ycombinator-all-funded-companies-dataset/prior_companies")
    if prior is not None and not prior.empty and {"hnid", "company"}.issubset(prior.columns):
        prior_summary = (
            prior.assign(hnid=prior["hnid"].fillna("").astype(str), company=prior["company"].fillna("").astype(str).str.strip())
            .query("company != ''")
            .groupby("hnid")["company"]
            .agg(lambda s: ", ".join(sorted(set(s))[:5]))
        )
        out = out.merge(prior_summary.rename("prior_company"), left_on="hnid", right_index=True, how="left")
    else:
        out["prior_company"] = ""

    out = out.drop(columns=["hnid"]).fillna("").replace({"nan": "", "None": ""})
    out = out[out["company_slug"].astype(str).str.len().gt(0) | out["company"].astype(str).str.len().gt(0)]
    return out[["founder", "company", "school", "prior_company", "company_slug"]].drop_duplicates()


def _prepare_yc_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["batch"] = out["batch"].fillna("Unknown").replace("", "Unknown")
    out["status"] = out["status"].fillna("Unknown").replace("", "Unknown")
    out["industry"] = out["industry"].fillna("Unspecified").replace("", "Unspecified").astype(str)
    out["region"] = out["region"].fillna("Unknown").replace("", "Unknown").map(_primary_location)
    out["team_size"] = _to_number(out["team_size"]).fillna(0).clip(lower=0)
    out["batch_year"] = out["batch"].map(_batch_year).fillna(0).astype(int)
    out["batch_era"] = out["batch_year"].map(_batch_era)

    industry_counts = out["industry"].value_counts()
    region_counts = out["region"].value_counts()
    batch_industry_counts = out.groupby(["batch_year", "industry"], dropna=False).size()

    out["sector_momentum"] = out.apply(
        lambda row: batch_industry_counts.get((row["batch_year"], row["industry"]), 0), axis=1
    )
    out["geo_novelty"] = out["region"].map(lambda r: 1 / math.sqrt(region_counts.get(r, 1)))
    out["industry_density"] = out["industry"].map(lambda r: industry_counts.get(r, 1))
    out["hiring_signal"] = out["is_hiring"].astype(float)
    out["recognition_score"] = (
        0.55 * _norm(out["team_size"])
        + 0.30 * out["top_company"].astype(float)
        + 0.15 * _norm(out["industry_density"])
    )
    out["momentum_score"] = (
        0.35 * out["hiring_signal"]
        + 0.25 * _norm(out["team_size"] / (2027 - out["batch_year"].clip(lower=2005)))
        + 0.25 * _norm(out["sector_momentum"])
        + 0.15 * _norm(out["geo_novelty"])
    )
    out["underrecognition_signal"] = (1 - out["recognition_score"]).clip(0, 1)
    out["raw_opportunity_score"] = (
        0.45 * out["momentum_score"]
        + 0.35 * out["underrecognition_signal"]
        + 0.20 * _norm(out["team_size"])
    )
    out["radar_score"] = (100 * out["raw_opportunity_score"].rank(pct=True, method="average")).round(1)
    out["radar_tier"] = pd.cut(
        out["radar_score"],
        bins=[-1, 40, 60, 75, 101],
        labels=["Watch", "Emerging", "High Potential", "Breakout"],
    ).astype(str)
    return _add_text_projection(out).sort_values("radar_score", ascending=False).reset_index(drop=True)


def _load_startup_companies(dataset_dir: Path) -> pd.DataFrame:
    obj_cols = [
        "id",
        "entity_type",
        "name",
        "category_code",
        "status",
        "founded_at",
        "closed_at",
        "country_code",
        "state_code",
        "city",
        "region",
        "first_funding_at",
        "last_funding_at",
        "funding_rounds",
        "funding_total_usd",
        "milestones",
        "relationships",
    ]
    objects = pd.read_csv(dataset_dir / "objects.csv", usecols=obj_cols, low_memory=False, encoding_errors="replace")
    companies = objects[objects["entity_type"].eq("Company")].copy()
    companies["founded_year"] = pd.to_datetime(companies["founded_at"], errors="coerce").dt.year
    companies["first_funding_year"] = pd.to_datetime(companies["first_funding_at"], errors="coerce").dt.year
    companies["last_funding_year"] = pd.to_datetime(companies["last_funding_at"], errors="coerce").dt.year
    companies["funding_span_years"] = (companies["last_funding_year"] - companies["first_funding_year"]).clip(lower=0)

    acq = pd.read_csv(dataset_dir / "acquisitions.csv", usecols=["acquired_object_id"], low_memory=False)
    ipos = pd.read_csv(dataset_dir / "ipos.csv", usecols=["object_id"], low_memory=False)
    exit_ids = set(acq["acquired_object_id"].dropna().astype(str)) | set(ipos["object_id"].dropna().astype(str))
    companies["success_exit"] = companies["id"].isin(exit_ids) | companies["status"].isin(["acquired", "ipo"])

    for col in ["funding_total_usd", "funding_rounds", "milestones", "relationships"]:
        companies[col] = pd.to_numeric(companies[col], errors="coerce").fillna(0)
    return companies.reset_index(drop=True)


def _load_startup_founder_features(dataset_dir: Path) -> pd.DataFrame:
    rels = pd.read_csv(
        dataset_dir / "relationships.csv",
        usecols=["person_object_id", "relationship_object_id", "title"],
        low_memory=False,
        encoding_errors="replace",
    )
    rels["title_str"] = rels["title"].fillna("").astype(str)
    founder_rels = rels[
        rels["title_str"].str.contains("founder", case=False, na=False)
    ][["relationship_object_id", "person_object_id"]].dropna().drop_duplicates()

    founder_count = founder_rels.groupby("relationship_object_id")["person_object_id"].nunique().rename("founder_count")

    all_person_company = rels[["person_object_id", "relationship_object_id"]].dropna().drop_duplicates()
    person_company_counts = all_person_company.groupby("person_object_id")["relationship_object_id"].nunique().rename("person_company_count")
    founder_prior = founder_rels.merge(person_company_counts, on="person_object_id", how="left")
    founder_prior["prior_links"] = (founder_prior["person_company_count"].fillna(1) - 1).clip(lower=0)
    prior_company_links = founder_prior.groupby("relationship_object_id")["prior_links"].sum().rename("founder_prior_company_links")

    deg = pd.read_csv(
        dataset_dir / "degrees.csv",
        usecols=["object_id", "degree_type", "institution"],
        low_memory=False,
        encoding_errors="replace",
    )
    deg["degree_type"] = deg["degree_type"].fillna("").astype(str)
    deg["institution"] = deg["institution"].fillna("").astype(str)
    founder_degrees = founder_rels.merge(deg, left_on="person_object_id", right_on="object_id", how="left")
    founder_degrees["has_degree"] = founder_degrees["degree_type"].str.len().gt(0) | founder_degrees["institution"].str.len().gt(0)
    founder_degrees["has_mba"] = founder_degrees["degree_type"].str.contains("mba", case=False, na=False)
    founder_degrees["has_phd"] = founder_degrees["degree_type"].str.contains("phd|ph.d|doctor", case=False, na=False, regex=True)
    elite_regex = r"stanford|harvard|mit|massachusetts institute|berkeley|yale|princeton|caltech|carnegie mellon|oxford|cambridge"
    founder_degrees["has_elite_school"] = founder_degrees["institution"].str.contains(elite_regex, case=False, na=False, regex=True)

    edu = founder_degrees[["relationship_object_id", "person_object_id", "has_degree", "has_mba", "has_phd", "has_elite_school"]].copy()
    for flag in ["has_degree", "has_mba", "has_phd", "has_elite_school"]:
        edu[f"person_{flag}"] = edu["person_object_id"].where(edu[flag])

    founder_edu = edu.groupby("relationship_object_id").agg(
        founder_degree_records=("has_degree", "sum"),
        founders_with_degree=("person_has_degree", "nunique"),
        founders_with_mba=("person_has_mba", "nunique"),
        founders_with_phd=("person_has_phd", "nunique"),
        founders_with_elite_school=("person_has_elite_school", "nunique"),
    )

    out = pd.concat([founder_count, founder_edu, prior_company_links], axis=1)
    return out


def _load_startup_round_features(dataset_dir: Path) -> pd.DataFrame:
    fr = pd.read_csv(
        dataset_dir / "funding_rounds.csv",
        usecols=["object_id", "funded_at", "raised_amount_usd", "participants"],
        low_memory=False,
        encoding_errors="replace",
    )
    fr["raised_amount_usd"] = pd.to_numeric(fr["raised_amount_usd"], errors="coerce").fillna(0)
    fr["participants"] = pd.to_numeric(fr["participants"], errors="coerce").fillna(0)
    fr["funded_year"] = pd.to_datetime(fr["funded_at"], errors="coerce").dt.year

    return fr.groupby("object_id").agg(
        observed_rounds=("object_id", "size"),
        total_raised_rounds=("raised_amount_usd", "sum"),
        median_round_usd=("raised_amount_usd", "median"),
        avg_participants=("participants", "mean"),
        max_participants=("participants", "max"),
        observed_funding_span=("funded_year", lambda s: max(s.max() - s.min(), 0) if s.notna().any() else 0),
    )


def _fit_startup_exit_model(ml_df: pd.DataFrame) -> dict[str, pd.DataFrame | dict]:
    max_model_rows = 12000
    max_importance_rows = 2500
    features = [
        "category_code",
        "country_code",
        "founded_year",
        "funding_total_usd",
        "funding_rounds",
        "milestones",
        "relationships",
        "funding_span_years",
        "founder_count",
        "founder_degree_records",
        "founders_with_degree",
        "founders_with_mba",
        "founders_with_phd",
        "founders_with_elite_school",
        "founder_prior_company_links",
        "observed_rounds",
        "total_raised_rounds",
        "median_round_usd",
        "avg_participants",
        "max_participants",
        "observed_funding_span",
    ]
    empty = {
        "metrics": {"accuracy": np.nan, "roc_auc": np.nan, "rows": len(ml_df), "success_exits": 0, "success_rate": np.nan},
        "importance": pd.DataFrame(columns=["feature", "importance", "signal_group"]),
        "predictions": pd.DataFrame(columns=["id", "predicted_exit_probability"]),
        "model": None,
        "features": features,
    }
    if len(ml_df) < 100 or ml_df["success_exit"].nunique() < 2:
        return empty

    if len(ml_df) > max_model_rows:
        sampled_parts = []
        for _, part in ml_df.groupby("success_exit", sort=False):
            share = len(part) / len(ml_df)
            n = max(1, int(round(max_model_rows * share)))
            sampled_parts.append(part.sample(n=min(n, len(part)), random_state=7))
        model_df = pd.concat(sampled_parts).sample(frac=1, random_state=7).reset_index(drop=True)
    else:
        model_df = ml_df.copy()

    X = model_df[features].copy()
    y = model_df["success_exit"].astype(int)
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    fund_like = ["funding_total_usd", "total_raised_rounds", "median_round_usd"]
    regular_num = [c for c in num_cols if c not in fund_like]

    preprocess = ColumnTransformer(
        [
            (
                "funding_log",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                        ("log", FunctionTransformer(_log_nonnegative, feature_names_out="one-to-one")),
                    ]
                ),
                fund_like,
            ),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), regular_num),
            (
                "cat",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=30))]),
                cat_cols,
            ),
        ]
    )

    model = Pipeline(
        [
            ("preprocess", preprocess),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=70,
                    random_state=7,
                    class_weight="balanced",
                    max_depth=11,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    if len(X_test) > max_importance_rows:
        importance_index = X_test.sample(n=max_importance_rows, random_state=7).index
        X_importance = X_test.loc[importance_index]
        y_importance = y_test.loc[importance_index]
    else:
        X_importance = X_test
        y_importance = y_test
    perm = permutation_importance(model, X_importance, y_importance, n_repeats=1, random_state=7, n_jobs=1, scoring="roc_auc")
    importance = pd.DataFrame({"feature": features, "importance": perm.importances_mean}).sort_values("importance", ascending=False)
    importance["signal_group"] = np.select(
        [
            importance["feature"].str.contains("founder|degree|elite|mba|phd", case=False, regex=True),
            importance["feature"].str.contains("fund|round|participants", case=False, regex=True),
            importance["feature"].str.contains("country|category", case=False, regex=True),
        ],
        ["Founder/team", "Funding maturity", "Market/geography"],
        default="Company metadata",
    )

    all_proba = model.predict_proba(ml_df[features].copy())[:, 1]
    predictions = pd.DataFrame({"id": ml_df["id"].to_numpy(), "predicted_exit_probability": all_proba})

    return {
        "metrics": {
            "accuracy": float(accuracy_score(y_test, pred)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "rows": int(len(ml_df)),
            "model_rows": int(len(model_df)),
            "success_exits": int(ml_df["success_exit"].astype(int).sum()),
            "success_rate": float(ml_df["success_exit"].astype(int).mean()),
        },
        "importance": importance,
        "predictions": predictions,
        "model": model,
        "features": features,
    }


def _startup_signal_summary(ml_df: pd.DataFrame) -> pd.DataFrame:
    if ml_df.empty:
        return pd.DataFrame(columns=["segment", "dimension", "companies", "exit_rate", "median_funding_usd", "median_founders", "exit_rate_pct", "median_funding_m"])
    signal_df = ml_df.copy()
    signal_df["team_structure"] = np.select(
        [signal_df["founder_count"] >= 3, signal_df["founder_count"] == 2, signal_df["founder_count"] == 1],
        ["3+ known founders", "2 known founders", "1 known founder"],
        default="No known founder record",
    )
    signal_df["education_coverage"] = np.select(
        [signal_df["founders_with_elite_school"] > 0, signal_df["founders_with_degree"] > 0],
        ["At least one elite-school record", "At least one degree record"],
        default="No known education record",
    )
    signal_df["funding_maturity"] = np.select(
        [signal_df["funding_rounds"] >= 3, signal_df["funding_rounds"] == 2, signal_df["funding_rounds"] == 1],
        ["3+ funding rounds", "2 funding rounds", "1 funding round"],
        default="No funding-round record",
    )

    def summarize(frame: pd.DataFrame, col: str, name: str) -> pd.DataFrame:
        out = frame.groupby(col).agg(
            companies=("id", "size"),
            exit_rate=("success_exit", "mean"),
            median_funding_usd=("funding_total_usd", "median"),
            median_founders=("founder_count", "median"),
        ).reset_index().rename(columns={col: "segment"})
        out["dimension"] = name
        return out

    out = pd.concat(
        [
            summarize(signal_df, "team_structure", "Team structure"),
            summarize(signal_df, "education_coverage", "Education coverage"),
            summarize(signal_df, "funding_maturity", "Funding maturity"),
        ],
        ignore_index=True,
    )
    out["exit_rate_pct"] = out["exit_rate"] * 100
    out["median_funding_m"] = out["median_funding_usd"] / 1_000_000
    return out.sort_values(["dimension", "exit_rate"], ascending=[True, False])


def _startup_sector_summary(ml_df: pd.DataFrame) -> pd.DataFrame:
    if ml_df.empty:
        return pd.DataFrame(columns=["category_code", "companies", "exit_rate", "median_funding_usd", "median_founders", "exit_rate_pct", "median_funding_m"])
    out = ml_df.groupby("category_code").agg(
        companies=("id", "size"),
        exit_rate=("success_exit", "mean"),
        median_funding_usd=("funding_total_usd", "median"),
        median_founders=("founder_count", "median"),
    ).query("companies >= 200").sort_values("exit_rate", ascending=False).reset_index()
    out["exit_rate_pct"] = out["exit_rate"] * 100
    out["median_funding_m"] = out["median_funding_usd"] / 1_000_000
    return out


def _startup_country_summary(ml_df: pd.DataFrame) -> pd.DataFrame:
    if ml_df.empty:
        return pd.DataFrame(columns=["country_code", "companies", "exit_rate", "median_funding_usd", "median_founders", "exit_rate_pct", "median_funding_m"])
    out = ml_df.groupby("country_code").agg(
        companies=("id", "size"),
        exit_rate=("success_exit", "mean"),
        median_funding_usd=("funding_total_usd", "median"),
        median_founders=("founder_count", "median"),
    ).query("companies >= 100").sort_values("exit_rate", ascending=False).reset_index()
    out["exit_rate_pct"] = out["exit_rate"] * 100
    out["median_funding_m"] = out["median_funding_usd"] / 1_000_000
    return out


def _attach_founder_context(founders: pd.DataFrame, companies: pd.DataFrame) -> pd.DataFrame:
    if founders.empty:
        return founders
    context = companies[["slug", "name", "industry", "batch", "radar_score"]].rename(columns={"name": "company_name"})
    return founders.merge(context, left_on="company_slug", right_on="slug", how="left").drop(columns=["slug"], errors="ignore")


def _add_text_projection(df: pd.DataFrame) -> pd.DataFrame:
    text = (df["industry"].fillna("") + " " + df["tags_text"].fillna("") + " " + df["one_liner"].fillna("")).str.strip()
    if len(df) < 3 or text.str.len().sum() == 0:
        df["topic_x"] = 0
        df["topic_y"] = 0
        return df

    try:
        matrix = TfidfVectorizer(max_features=80, min_df=2, stop_words="english").fit_transform(text)
        coords = PCA(n_components=2, random_state=7).fit_transform(matrix.toarray())
        df["topic_x"] = coords[:, 0]
        df["topic_y"] = coords[:, 1]
    except Exception:
        df["topic_x"] = 0
        df["topic_y"] = 0
    return df


def _first_present(df: pd.DataFrame, names: Iterable[str], default=np.nan) -> pd.Series:
    normalized = {str(c).lower(): c for c in df.columns}
    result = pd.Series([np.nan] * len(df), index=df.index, dtype="object")
    for name in names:
        col = normalized.get(str(name).lower())
        if col is not None:
            values = df[col]
            missing = result.isna() | result.astype(str).str.strip().isin(["", "nan", "None"])
            result.loc[missing] = values.loc[missing]
    missing = result.isna() | result.astype(str).str.strip().isin(["", "nan", "None"])
    result.loc[missing] = default
    return result


def _find_col(cols: dict[str, str], names: Iterable[str]) -> str | None:
    for name in names:
        if name.lower() in cols:
            return cols[name.lower()]
    return None


def _slugify_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).map(lambda s: re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-"))


def _to_number(series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0)


def _to_bool(series) -> pd.Series:
    if not isinstance(series, pd.Series):
        return pd.Series([bool(series)])
    return series.map(lambda x: str(x).strip().lower() in {"true", "1", "yes", "y", "hiring"})


def _list_to_text(value) -> str:
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return "" if pd.isna(value) else str(value)


def _first_tag(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text in {"[]", "nan", "None"}:
        return ""
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            for item in parsed:
                item_text = str(item).strip()
                if item_text:
                    return item_text
    except (SyntaxError, ValueError):
        pass
    return text.split(",")[0].strip().strip("[]'\"")


def _primary_location(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return "Unknown"
    parts = [p.strip() for p in re.split(r"[;,]", value) if p.strip()]
    if not parts:
        return "Unknown"
    return parts[-1] if len(parts) > 1 else parts[0]


def _batch_year(batch: str) -> float:
    match = re.search(r"(20\d{2}|19\d{2}|\d{2})", str(batch))
    if not match:
        return np.nan
    year = int(match.group(1))
    return 2000 + year if year < 100 else year


def _batch_era(year: int) -> str:
    if year <= 0:
        return "Unknown"
    if year < 2012:
        return "2005-2011"
    if year < 2018:
        return "2012-2017"
    if year < 2023:
        return "2018-2022"
    return "2023+"


def _norm(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0).to_numpy().reshape(-1, 1)
    if len(values) == 0 or float(values.max()) == float(values.min()):
        return pd.Series(np.zeros(len(values)), index=series.index)
    return pd.Series(MinMaxScaler().fit_transform(values).ravel(), index=series.index)


def _log_nonnegative(values):
    return np.log1p(np.clip(np.asarray(values, dtype=float), 0, None))
