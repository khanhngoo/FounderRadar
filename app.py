from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from shiny import App, reactive, render, ui
from starlette.applications import Starlette
from starlette.responses import RedirectResponse, Response
from starlette.routing import Mount, Route

from data_pipeline import load_dashboard_data


BASE_DIR = Path(__file__).parent
DATA = load_dashboard_data(BASE_DIR)
READINESS = DATA["readiness"]
YC = DATA["yc"]
STARTUP = DATA["startup"]
YC_READY = bool(YC and not YC["companies"].empty)
STARTUP_READY = bool(STARTUP and not STARTUP["ml_df"].empty)

YC_COMPANIES = YC["companies"] if YC_READY else pd.DataFrame()
STARTUP_ML = STARTUP["ml_df"] if STARTUP_READY else pd.DataFrame()
STARTUP_MODEL = STARTUP.get("model") if STARTUP_READY else None
STARTUP_MODEL_FEATURES = STARTUP.get("model_features", []) if STARTUP_READY else []

YC_ORANGE = "#ff6600"
INK = "#1f2933"
MUTED = "#667085"
TEAL = "#14b8a6"
BLUE = "#2563eb"
ROSE = "#e11d48"
MISSING_LABELS = {"", "nan", "none", "null", "[]", "unspecified", "unknown", "not provided"}
RADAR_TIER_ORDER = ["Watch", "Emerging", "High Potential", "Breakout"]
RADAR_TIER_COLORS = {
    "Watch": "#94a3b8",
    "Emerging": BLUE,
    "High Potential": TEAL,
    "Breakout": YC_ORANGE,
}


def fig_html(fig: go.Figure, height: int = 360) -> ui.HTML:
    fig.update_layout(
        height=height,
        margin=dict(l=14, r=14, t=42, b=20),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Aptos, Segoe UI, sans-serif", size=12, color=INK),
        legend_title_text="",
    )
    fig.update_xaxes(gridcolor="rgba(102,112,133,.16)", zerolinecolor="rgba(102,112,133,.28)")
    fig.update_yaxes(gridcolor="rgba(102,112,133,.16)", zerolinecolor="rgba(102,112,133,.28)")
    return ui.HTML(fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False}))


def _is_missing_label(value) -> bool:
    return str(value).strip().lower() in MISSING_LABELS


def choices(series: pd.Series | None, limit: int | None = None, include_all: bool = True) -> list[str]:
    prefix = ["All"] if include_all else []
    if series is None or series.empty:
        return prefix
    vals = sorted(
        v
        for v in series.dropna().astype(str).map(str.strip).unique()
        if not _is_missing_label(v)
    )
    return prefix + (vals[:limit] if limit else vals)


def default_choice(series: pd.Series | None, fallback: str = "") -> str:
    vals = choices(series, include_all=False)
    if not vals:
        return fallback
    mode = series.dropna().astype(str).map(str.strip)
    mode = mode[~mode.str.lower().isin(MISSING_LABELS)]
    return str(mode.mode().iloc[0]) if not mode.empty else vals[0]


def median_num(series: pd.Series | None, fallback: float = 0) -> float:
    if series is None or series.empty:
        return fallback
    values = pd.to_numeric(series, errors="coerce")
    value = values.replace([np.inf, -np.inf], np.nan).dropna().median()
    return float(value) if pd.notna(value) else fallback


def money(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"${value / 1_000:.0f}K"
    return f"${value:.0f}"


def pct(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{100 * value:.1f}%"


def readiness_notice(page: str) -> ui.Tag:
    missing_items: list[ui.Tag] = []
    if page == "yc":
        for item in READINESS["missing_yc"]:
            missing_items.append(ui.tags.li(f"{item['name']} -> {item['folder']}"))
    else:
        missing_items.append(ui.tags.li(f"{READINESS['startup_folder']}"))
        for filename in READINESS["missing_startup_files"]:
            missing_items.append(ui.tags.li(filename))

    return ui.div(
        {"class": "notice"},
        ui.h2("Kaggle data required"),
        ui.p("This dashboard intentionally blocks fallback data so the analysis matches the project datasets."),
        ui.h3("Missing data"),
        ui.tags.ul(*missing_items),
        ui.h3("Run"),
        ui.tags.pre(READINESS["fetch_command"]),
        ui.p("This uses kagglehub.dataset_download and syncs the downloaded files into data/raw."),
    )


def metric_card(label: str, value, note: str = "") -> ui.Tag:
    return ui.div(
        {"class": "metric"},
        ui.div({"class": "metric-title"}, label),
        ui.div({"class": "metric-value"}, value),
        ui.div({"class": "metric-note"}, note),
    )


def source_badge(label: str, ok: bool) -> ui.Tag:
    return ui.div({"class": "source-badge ok" if ok else "source-badge missing"}, label)


def page_head(eyebrow: str, title: str, lede: str) -> ui.Tag:
    return ui.div(
        {"class": "page-head"},
        ui.div(
            ui.div({"class": "eyebrow"}, eyebrow),
            ui.h1(title),
            ui.div({"class": "lede"}, lede),
        ),
    )


def yc_filter_bar() -> ui.Tag:
    return ui.div(
        {"class": "filter-bar yc-filter"},
        ui.input_select("yc_batch_era", "Batch era", choices(YC_COMPANIES.get("batch_era") if YC_READY else None)),
        ui.input_select("yc_industry", "Industry", choices(YC_COMPANIES.get("industry") if YC_READY else None, 80)),
        ui.input_select("yc_region", "Region", choices(YC_COMPANIES.get("region") if YC_READY else None, 80)),
        ui.input_select("yc_status", "Status", choices(YC_COMPANIES.get("status") if YC_READY else None)),
        ui.input_select("yc_tier", "Radar tier", choices(YC_COMPANIES.get("radar_tier") if YC_READY else None)),
        ui.input_checkbox("yc_hiring", "Hiring only", value=False),
        ui.input_text("yc_search", "Search", placeholder="Company, tag, description"),
    )


def startup_filter_bar() -> ui.Tag:
    return ui.div(
        {"class": "filter-bar startup-filter"},
        ui.input_select("si_sector", "Sector", choices(STARTUP_ML.get("category_code") if STARTUP_READY else None, 80)),
        ui.input_select("si_country", "Country", choices(STARTUP_ML.get("country_code") if STARTUP_READY else None, 80)),
        ui.input_slider("si_probability", "Minimum exit probability", 0, 100, 20, step=5),
        ui.input_text("si_search", "Search", placeholder="Company, sector, country"),
    )


def ml_input_bar() -> ui.Tag:
    if not STARTUP_READY:
        return ui.div()
    current_year = 2026
    return ui.div(
        {"class": "predictor-grid"},
        ui.input_select("ml_category", "Sector", choices(STARTUP_ML.get("category_code"), 120, include_all=False), selected=default_choice(STARTUP_ML.get("category_code"))),
        ui.input_select("ml_country", "Country", choices(STARTUP_ML.get("country_code"), 120, include_all=False), selected=default_choice(STARTUP_ML.get("country_code"))),
        ui.input_numeric("ml_founded_year", "Founded year", value=int(median_num(STARTUP_ML.get("founded_year"), 2010)), min=1980, max=current_year, step=1),
        ui.input_numeric("ml_funding_total_usd", "Funding total USD", value=round(median_num(STARTUP_ML.get("funding_total_usd"), 1_000_000)), min=0, step=250000),
        ui.input_numeric("ml_funding_rounds", "Funding rounds", value=round(median_num(STARTUP_ML.get("funding_rounds"), 1)), min=0, step=1),
        ui.input_numeric("ml_milestones", "Milestones", value=round(median_num(STARTUP_ML.get("milestones"), 0)), min=0, step=1),
        ui.input_numeric("ml_relationships", "Relationship count", value=round(median_num(STARTUP_ML.get("relationships"), 0)), min=0, step=1),
        ui.input_numeric("ml_funding_span_years", "Funding span years", value=round(median_num(STARTUP_ML.get("funding_span_years"), 0)), min=0, step=1),
        ui.input_numeric("ml_founder_count", "Founder count", value=round(median_num(STARTUP_ML.get("founder_count"), 1)), min=0, step=1),
        ui.input_numeric("ml_founder_degree_records", "Founder degree records", value=round(median_num(STARTUP_ML.get("founder_degree_records"), 0)), min=0, step=1),
        ui.input_numeric("ml_founders_with_degree", "Founders with degree", value=round(median_num(STARTUP_ML.get("founders_with_degree"), 0)), min=0, step=1),
        ui.input_numeric("ml_founders_with_mba", "Founders with MBA", value=round(median_num(STARTUP_ML.get("founders_with_mba"), 0)), min=0, step=1),
        ui.input_numeric("ml_founders_with_phd", "Founders with PhD", value=round(median_num(STARTUP_ML.get("founders_with_phd"), 0)), min=0, step=1),
        ui.input_numeric("ml_founders_with_elite_school", "Founders with elite school", value=round(median_num(STARTUP_ML.get("founders_with_elite_school"), 0)), min=0, step=1),
        ui.input_numeric("ml_founder_prior_company_links", "Prior company links", value=round(median_num(STARTUP_ML.get("founder_prior_company_links"), 0)), min=0, step=1),
        ui.input_numeric("ml_observed_rounds", "Observed rounds", value=round(median_num(STARTUP_ML.get("observed_rounds"), 1)), min=0, step=1),
        ui.input_numeric("ml_total_raised_rounds", "Total raised in rounds USD", value=round(median_num(STARTUP_ML.get("total_raised_rounds"), 1_000_000)), min=0, step=250000),
        ui.input_numeric("ml_median_round_usd", "Median round USD", value=round(median_num(STARTUP_ML.get("median_round_usd"), 250000)), min=0, step=100000),
        ui.input_numeric("ml_avg_participants", "Average participants", value=round(median_num(STARTUP_ML.get("avg_participants"), 1), 1), min=0, step=0.5),
        ui.input_numeric("ml_max_participants", "Max participants", value=round(median_num(STARTUP_ML.get("max_participants"), 1)), min=0, step=1),
        ui.input_numeric("ml_observed_funding_span", "Observed funding span", value=round(median_num(STARTUP_ML.get("observed_funding_span"), 0)), min=0, step=1),
    )


def yc_page_layout() -> ui.Tag:
    return ui.div(
        {"class": "page-wrap"},
        page_head(
            "YC Ecosystem",
            "Y Combinator opportunity map",
            "A compact view of YC company density, hiring momentum, batch evolution, and under-recognized opportunities from the three Kaggle YC datasets.",
        ),
        yc_filter_bar(),
        ui.div(
            {"class": "metric-grid"},
            metric_card("Companies", ui.output_text("yc_kpi_companies"), "Filtered YC companies"),
            metric_card("High Radar", ui.output_text("yc_kpi_high"), "Score >= 75"),
            metric_card("Hiring", ui.output_text("yc_kpi_hiring"), "Active hiring signal"),
            metric_card("Median Team", ui.output_text("yc_kpi_team"), "Team size"),
        ),
        ui.div({"class": "panel"}, ui.div({"class": "panel-title"}, "Opportunity Quadrant Map"), ui.div({"class": "panel-note"}, "Recognition vs momentum, using median recognition and upper-quartile momentum as the opportunity benchmark."), ui.output_ui("yc_opportunity")),
        ui.div(
            {"class": "grid-2"},
            ui.div({"class": "panel"}, ui.div({"class": "panel-title"}, "YC Batch Pulse Streamgraph"), ui.div({"class": "panel-note"}, "Tracks how top industries move across YC batch years. Placeholder categories are excluded."), ui.output_ui("yc_stream")),
            ui.div({"class": "panel"}, ui.div({"class": "panel-title"}, "Industry x Region Matrix"), ui.div({"class": "panel-note"}, "Reveals where sectors concentrate geographically and where radar scores cluster. Placeholder categories are excluded."), ui.output_ui("yc_matrix")),
        ),
        ui.div(
            {"class": "grid-2"},
            ui.div({"class": "panel"}, ui.div({"class": "panel-title"}, "Radar Tier Distribution"), ui.div({"class": "panel-note"}, "Shows the filtered opportunity pool by radar tier, split by hiring signal."), ui.output_ui("yc_tier_distribution")),
            ui.div({"class": "panel"}, ui.div({"class": "panel-title"}, "Founder / School Signal View"), ui.div({"class": "panel-note"}, "Company-level founder, school, and prior-company coverage by radar tier."), ui.output_ui("yc_founder_school")),
        ),
        ui.div({"class": "panel"}, ui.div({"class": "panel-title"}, "Topic Projection Map"), ui.div({"class": "panel-note"}, "Maps company descriptions, tags, and industries into thematic clusters using the pipeline's TF-IDF/PCA projection."), ui.output_ui("yc_topic_map")),
        ui.div({"class": "panel"}, ui.div({"class": "panel-title"}, "Company Explorer"), ui.output_data_frame("yc_table")),
    )


def startup_page_layout() -> ui.Tag:
    metrics = STARTUP["metrics"] if STARTUP_READY else {"rows": 0, "success_exits": 0, "roc_auc": np.nan}
    return ui.div(
        {"class": "page-wrap"},
        page_head(
            "Startup Intelligence",
            "Historical startup outcome signals",
            "A YC-styled research workspace for exit proxy modeling, funding maturity, founder signals, sectors, and geography. SEC Form D is intentionally excluded.",
        ),
        startup_filter_bar(),
        ui.div(
            {"class": "metric-grid"},
            metric_card("Model Rows", f"{metrics['rows']:,}", "Companies with modeling signal"),
            metric_card("Success Exits", f"{metrics['success_exits']:,}", "Acquisition or IPO proxy"),
            metric_card("ROC-AUC", f"{metrics['roc_auc']:.3f}" if not pd.isna(metrics["roc_auc"]) else "n/a", "Held-out model score"),
            metric_card("Filtered", ui.output_text("si_kpi_filtered"), "Rows driving these charts"),
        ),
        ui.div(
            {"class": "grid-2"},
            ui.div({"class": "panel"}, ui.div({"class": "panel-title"}, "Model Signal Importance"), ui.div({"class": "panel-note"}, "Global permutation importance grouped into investor-readable signal families."), ui.output_ui("si_importance")),
            ui.div({"class": "panel"}, ui.div({"class": "panel-title"}, "Funding Maturity Ladder"), ui.div({"class": "panel-note"}, "Filtered exit rate by funding-round maturity, with funding intensity encoded by color."), ui.output_ui("si_funding_ladder")),
        ),
        ui.div(
            {"class": "grid-2"},
            ui.div({"class": "panel"}, ui.div({"class": "panel-title"}, "Founder Signal Summary"), ui.div({"class": "panel-note"}, "Filtered team structure, education coverage, and funding maturity on one outcome scale."), ui.output_ui("si_founder_summary")),
            ui.div({"class": "panel"}, ui.div({"class": "panel-title"}, "Sector Risk / Reward"), ui.div({"class": "panel-note"}, "Filtered sector scale vs exit rate, sized by median funding."), ui.output_ui("si_sector")),
        ),
        ui.div({"class": "panel"}, ui.div({"class": "panel-title"}, "Geography Ecosystem View"), ui.div({"class": "panel-note"}, "Filtered country scale vs exit rate, sized by median funding."), ui.output_ui("si_geo")),
    )


def ml_page_layout() -> ui.Tag:
    metrics = STARTUP["metrics"] if STARTUP_READY else {"rows": 0, "roc_auc": np.nan}
    return ui.div(
        {"class": "page-wrap"},
        page_head(
            "ML Exit Predictor",
            "Scenario-based exit probability",
            "A live view of the historical startup exit model. Adjust company, founder, and funding signals to see how the trained model scores the scenario.",
        ),
        ui.div({"class": "panel controls-panel"}, ui.div({"class": "panel-title"}, "Scenario inputs"), ml_input_bar()),
        ui.div(
            {"class": "metric-grid"},
            metric_card("Predicted Exit Probability", ui.output_text("ml_prediction"), "Acquisition or IPO proxy"),
            metric_card("Dataset Median", ui.output_text("ml_dataset_median"), "Existing company predictions"),
            metric_card("Upper Quartile", ui.output_text("ml_dataset_q75"), "Existing company predictions"),
            metric_card("ROC-AUC", f"{metrics['roc_auc']:.3f}" if not pd.isna(metrics["roc_auc"]) else "n/a", "Held-out model score"),
        ),
        ui.div(
            {"class": "grid-2"},
            ui.div({"class": "panel"}, ui.div({"class": "panel-title"}, "Prediction Distribution"), ui.div({"class": "panel-note"}, "Shows where the selected scenario lands among existing model scores."), ui.output_ui("ml_distribution")),
            ui.div({"class": "panel"}, ui.div({"class": "panel-title"}, "Top Model Signals"), ui.div({"class": "panel-note"}, "Global feature importance from the trained RandomForest model."), ui.output_ui("ml_importance")),
        ),
        ui.div({"class": "panel"}, ui.div({"class": "panel-title"}, "Scenario Feature Values"), ui.output_data_frame("ml_scenario_table")),
    )


def app_header() -> ui.Tag:
    return ui.div(
        {"class": "top-shell"},
        ui.div(
            {"class": "brand"},
            ui.div({"class": "brand-mark"}, "Y"),
            ui.div(ui.div({"class": "brand-title"}, "FounderRadar"), ui.div({"class": "brand-sub"}, "YC + startup intelligence")),
        ),
        ui.div(
            {"class": "status-row"},
            source_badge("YC Kaggle datasets ready" if YC_READY else "YC Kaggle datasets missing", YC_READY),
            source_badge("Startup Investments ready" if STARTUP_READY else "Startup Investments missing", STARTUP_READY),
        ),
    )


app_ui = ui.page_fluid(
    ui.head_content(
        ui.tags.title("FounderRadar"),
        ui.tags.link(
            rel="icon",
            href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='6' fill='%23ff6600'/%3E%3Ctext x='16' y='22' text-anchor='middle' font-family='Arial' font-size='18' font-weight='700' fill='white'%3EY%3C/text%3E%3C/svg%3E",
        ),
        ui.tags.script(src="https://cdn.plot.ly/plotly-3.6.0.min.js"),
        ui.tags.style(
            """
            :root {
              --yc-orange:#ff6600; --ink:#1f2933; --muted:#667085; --line:#e6e8eb;
              --bg:#f7f4ef; --panel:#ffffff; --teal:#14b8a6; --blue:#2563eb; --rose:#e11d48;
            }
            body { margin:0; background:var(--bg); color:var(--ink); font-family:Aptos, "Segoe UI", sans-serif; }
            .container-fluid { padding:0; }
            .top-shell { display:flex; align-items:center; justify-content:space-between; gap:18px; padding:18px 24px 12px; background:#191714; color:#f8fafc; border-bottom:4px solid var(--yc-orange); }
            .brand { display:flex; align-items:center; gap:12px; }
            .brand-mark { width:38px; height:38px; display:grid; place-items:center; border-radius:7px; background:var(--yc-orange); font-weight:900; }
            .brand-title { font-size:21px; font-weight:850; line-height:1.1; }
            .brand-sub { color:#cbd5e1; font-size:12px; margin-top:3px; }
            .status-row { display:flex; align-items:center; gap:8px; flex-wrap:wrap; justify-content:flex-end; }
            .source-badge { padding:8px 10px; border-radius:6px; font-size:12px; border:1px solid rgba(255,255,255,.14); white-space:nowrap; }
            .source-badge.ok { background:rgba(20,184,166,.16); color:#99f6e4; }
            .source-badge.missing { background:rgba(255,102,0,.15); color:#fed7aa; }
            .nav-tabs { padding:0 24px; background:#191714; border-bottom:1px solid rgba(255,255,255,.14); gap:4px; }
            .nav-tabs .nav-link { color:#cbd5e1; border:0; border-radius:6px 6px 0 0; font-weight:750; padding:11px 14px; }
            .nav-tabs .nav-link:hover { color:#fff; border:0; background:rgba(255,255,255,.07); }
            .nav-tabs .nav-link.active { color:#191714; background:var(--bg); border:0; }
            .tab-content { padding:0; }
            .page-wrap { padding:24px; }
            .page-head { display:flex; align-items:flex-end; justify-content:space-between; gap:18px; margin-bottom:18px; }
            .eyebrow { color:var(--muted); font-size:12px; font-weight:850; letter-spacing:.08em; text-transform:uppercase; }
            h1 { font-size:34px; line-height:1.08; margin:3px 0 4px; letter-spacing:0; }
            .lede { max-width:860px; color:#475467; font-size:14px; line-height:1.5; }
            .filter-bar { display:grid; grid-template-columns:repeat(7, minmax(130px, 1fr)); gap:10px; align-items:end; background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; margin-bottom:14px; box-shadow:0 1px 2px rgba(16,24,40,.04); }
            .startup-filter { grid-template-columns:1fr 1fr 1.2fr 1.5fr; }
            .filter-bar label, .predictor-grid label { color:var(--muted); font-size:11px; font-weight:850; letter-spacing:.04em; text-transform:uppercase; }
            .filter-bar .form-select, .filter-bar .form-control, .predictor-grid .form-select, .predictor-grid .form-control { border-radius:6px; border:1px solid var(--line); min-height:36px; }
            .filter-bar .checkbox { padding-bottom:6px; }
            .metric-grid { display:grid; grid-template-columns:repeat(4, minmax(150px, 1fr)); gap:12px; margin-bottom:14px; }
            .metric { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; box-shadow:0 1px 2px rgba(16,24,40,.04); }
            .metric-title { color:var(--muted); font-size:11px; font-weight:850; letter-spacing:.04em; text-transform:uppercase; }
            .metric-value { font-size:27px; font-weight:850; line-height:1.1; margin-top:5px; }
            .metric-note { color:var(--muted); font-size:12px; margin-top:5px; }
            .grid-2 { display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:14px; }
            .panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; margin-bottom:14px; box-shadow:0 1px 2px rgba(16,24,40,.04); }
            .panel-title { font-size:15px; font-weight:850; margin-bottom:3px; }
            .panel-note { color:var(--muted); font-size:12px; margin-bottom:8px; line-height:1.4; }
            .predictor-grid { display:grid; grid-template-columns:repeat(4, minmax(160px, 1fr)); gap:10px; align-items:end; }
            .controls-panel { padding-bottom:6px; }
            .notice { max-width:860px; background:#fff; border:1px solid #fed7aa; border-left:5px solid var(--yc-orange); border-radius:8px; padding:22px; box-shadow:0 1px 2px rgba(16,24,40,.04); margin:24px; }
            .notice h2 { margin-top:0; }
            .notice pre { background:#111827; color:#e5e7eb; border-radius:6px; padding:12px; overflow:auto; }
            @media (max-width: 1180px) {
              .yc-filter, .predictor-grid { grid-template-columns:repeat(2, minmax(0, 1fr)); }
            }
            @media (max-width: 980px) {
              .top-shell, .page-head { display:block; }
              .status-row { justify-content:flex-start; margin-top:12px; }
              .metric-grid, .grid-2, .filter-bar, .startup-filter, .yc-filter, .predictor-grid { grid-template-columns:1fr; }
              .nav-tabs { padding:0 12px; }
              .page-wrap { padding:16px; }
            }
            """
        ),
    ),
    app_header(),
    ui.navset_tab(
        ui.nav_panel("YC Ecosystem", readiness_notice("yc") if not YC_READY else yc_page_layout(), value="yc"),
        ui.nav_panel("Startup Intelligence", readiness_notice("startup") if not STARTUP_READY else startup_page_layout(), value="startup"),
        ui.nav_panel("ML Exit Predictor", readiness_notice("startup") if not STARTUP_READY else ml_page_layout(), value="ml"),
        id="dashboard_tabs",
        selected="yc",
    ),
)


def startup_signal_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["segment", "dimension", "companies", "exit_rate", "median_funding_usd", "median_founders", "exit_rate_pct", "median_funding_m"])
    signal_df = df.copy()
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


def grouped_startup_summary(df: pd.DataFrame, group_col: str, min_rows: int = 3) -> pd.DataFrame:
    cols = [group_col, "companies", "exit_rate", "median_funding_usd", "median_founders", "exit_rate_pct", "median_funding_m"]
    if df.empty or group_col not in df:
        return pd.DataFrame(columns=cols)
    clean = df[df[group_col].notna()].copy()
    clean[group_col] = clean[group_col].astype(str).str.strip()
    clean = clean[~clean[group_col].str.lower().isin(MISSING_LABELS)]
    if clean.empty:
        return pd.DataFrame(columns=cols)
    threshold = max(min_rows, min(200, int(round(len(clean) * 0.01))))
    out = clean.groupby(group_col).agg(
        companies=("id", "size"),
        exit_rate=("success_exit", "mean"),
        median_funding_usd=("funding_total_usd", "median"),
        median_founders=("founder_count", "median"),
    ).reset_index()
    out = out[out["companies"] >= threshold].sort_values("exit_rate", ascending=False)
    out["exit_rate_pct"] = out["exit_rate"] * 100
    out["median_funding_m"] = out["median_funding_usd"] / 1_000_000
    return out


def yc_founder_signals(companies: pd.DataFrame) -> pd.DataFrame:
    founders = YC.get("founders", pd.DataFrame()) if YC_READY else pd.DataFrame()
    if founders.empty or "company_slug" not in founders:
        return pd.DataFrame()

    founder_signals = founders.copy()
    founder_signals["company_slug"] = founder_signals["company_slug"].fillna("").astype(str)
    for col in ["founder", "school", "prior_company"]:
        if col not in founder_signals:
            founder_signals[col] = ""
        founder_signals[col] = founder_signals[col].fillna("").astype(str).str.strip()

    company_signals = founder_signals.groupby("company_slug").agg(
        has_founder_record=("founder", lambda s: s.ne("").any()),
        has_school_record=("school", lambda s: s.ne("").any()),
        has_prior_company_record=("prior_company", lambda s: s.ne("").any()),
    )

    out = companies[["slug", "radar_tier"]].merge(company_signals, left_on="slug", right_index=True, how="left")
    for col in ["has_founder_record", "has_school_record", "has_prior_company_record"]:
        out[col] = out[col].fillna(False).astype(bool)
    return out


def server(input, output, session):
    @reactive.calc
    def yc_filtered() -> pd.DataFrame:
        if not YC_READY:
            return pd.DataFrame()
        df = YC_COMPANIES.copy()
        if input.yc_batch_era() != "All":
            df = df[df["batch_era"] == input.yc_batch_era()]
        if input.yc_industry() != "All":
            df = df[df["industry"] == input.yc_industry()]
        if input.yc_region() != "All":
            df = df[df["region"] == input.yc_region()]
        if input.yc_status() != "All":
            df = df[df["status"] == input.yc_status()]
        if input.yc_tier() != "All":
            df = df[df["radar_tier"] == input.yc_tier()]
        if input.yc_hiring():
            df = df[df["is_hiring"]]
        query = (input.yc_search() or "").strip().lower()
        if query:
            haystack = (
                df["name"].fillna("")
                + " "
                + df["industry"].fillna("")
                + " "
                + df["region"].fillna("")
                + " "
                + df["tags_text"].fillna("")
                + " "
                + df["one_liner"].fillna("")
            ).str.lower()
            df = df[haystack.str.contains(query, regex=False)]
        return df

    @reactive.calc
    def startup_filtered() -> pd.DataFrame:
        if not STARTUP_READY:
            return pd.DataFrame()
        df = STARTUP_ML.copy()
        if input.si_sector() != "All":
            df = df[df["category_code"].astype(str) == input.si_sector()]
        if input.si_country() != "All":
            df = df[df["country_code"].astype(str) == input.si_country()]
        df = df[df["predicted_exit_probability"].fillna(0) >= input.si_probability() / 100]
        query = (input.si_search() or "").strip().lower()
        if query:
            haystack = (df["name"].fillna("") + " " + df["category_code"].fillna("") + " " + df["country_code"].fillna("")).str.lower()
            df = df[haystack.str.contains(query, regex=False)]
        return df

    @reactive.calc
    def ml_scenario() -> pd.DataFrame:
        if not STARTUP_READY:
            return pd.DataFrame(columns=STARTUP_MODEL_FEATURES)
        values = {
            "category_code": input.ml_category(),
            "country_code": input.ml_country(),
            "founded_year": input.ml_founded_year(),
            "funding_total_usd": input.ml_funding_total_usd(),
            "funding_rounds": input.ml_funding_rounds(),
            "milestones": input.ml_milestones(),
            "relationships": input.ml_relationships(),
            "funding_span_years": input.ml_funding_span_years(),
            "founder_count": input.ml_founder_count(),
            "founder_degree_records": input.ml_founder_degree_records(),
            "founders_with_degree": input.ml_founders_with_degree(),
            "founders_with_mba": input.ml_founders_with_mba(),
            "founders_with_phd": input.ml_founders_with_phd(),
            "founders_with_elite_school": input.ml_founders_with_elite_school(),
            "founder_prior_company_links": input.ml_founder_prior_company_links(),
            "observed_rounds": input.ml_observed_rounds(),
            "total_raised_rounds": input.ml_total_raised_rounds(),
            "median_round_usd": input.ml_median_round_usd(),
            "avg_participants": input.ml_avg_participants(),
            "max_participants": input.ml_max_participants(),
            "observed_funding_span": input.ml_observed_funding_span(),
        }
        return pd.DataFrame([{feature: values.get(feature, np.nan) for feature in STARTUP_MODEL_FEATURES}])

    @reactive.calc
    def ml_probability() -> float:
        if STARTUP_MODEL is None or not STARTUP_MODEL_FEATURES:
            return np.nan
        try:
            return float(STARTUP_MODEL.predict_proba(ml_scenario()[STARTUP_MODEL_FEATURES])[:, 1][0])
        except Exception:
            return np.nan

    @output
    @render.text
    def yc_kpi_companies():
        return f"{len(yc_filtered()):,}"

    @output
    @render.text
    def yc_kpi_high():
        df = yc_filtered()
        return "0" if df.empty else f"{int((df['radar_score'] >= 75).sum()):,}"

    @output
    @render.text
    def yc_kpi_hiring():
        df = yc_filtered()
        return "0" if df.empty else f"{int(df['is_hiring'].sum()):,}"

    @output
    @render.text
    def yc_kpi_team():
        df = yc_filtered()
        return "0" if df.empty else f"{df['team_size'].median():.0f}"

    @output
    @render.text
    def si_kpi_filtered():
        return f"{len(startup_filtered()):,}"

    @output
    @render.text
    def ml_prediction():
        return pct(ml_probability())

    @output
    @render.text
    def ml_dataset_median():
        return pct(STARTUP_ML["predicted_exit_probability"].median()) if STARTUP_READY else "n/a"

    @output
    @render.text
    def ml_dataset_q75():
        return pct(STARTUP_ML["predicted_exit_probability"].quantile(0.75)) if STARTUP_READY else "n/a"

    @output
    @render.ui
    def yc_opportunity():
        df = yc_filtered().head(1400)
        if df.empty:
            return ui.p("No YC companies match the current filters.")
        fig = px.scatter(
            df,
            x="recognition_score",
            y="momentum_score",
            size=df["team_size"].clip(lower=2),
            color="industry",
            hover_name="name",
            hover_data=["batch", "region", "status", "radar_tier", "radar_score", "one_liner"],
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        recognition_cutoff = float(df["recognition_score"].quantile(0.50))
        momentum_cutoff = float(df["momentum_score"].quantile(0.75))
        x_max = max(float(df["recognition_score"].max()) * 1.08, recognition_cutoff * 1.35, 0.1)
        y_max = max(float(df["momentum_score"].max()) * 1.08, momentum_cutoff * 1.35, 0.1)
        fig.add_vrect(x0=0, x1=recognition_cutoff, fillcolor="#fff1e8", opacity=0.65, line_width=0)
        fig.add_hrect(y0=momentum_cutoff, y1=y_max, fillcolor="#ecfeff", opacity=0.45, line_width=0)
        fig.add_annotation(
            x=recognition_cutoff * 0.55,
            y=momentum_cutoff + ((y_max - momentum_cutoff) * 0.72),
            text="Lower recognition<br>higher momentum",
            showarrow=False,
            font=dict(color=YC_ORANGE, size=13),
        )
        fig.update_xaxes(range=[0, x_max])
        fig.update_yaxes(range=[0, y_max])
        fig.update_layout(xaxis_title="Recognition score", yaxis_title="Momentum score")
        return fig_html(fig, 460)

    @output
    @render.ui
    def yc_stream():
        df = yc_filtered()
        valid = df[(df["batch_year"] > 0) & ~df["industry"].map(_is_missing_label)].copy()
        if valid.empty:
            return ui.p("No batch-year data matches the filters.")
        top = valid["industry"].value_counts().head(8).index
        yearly = valid[valid["industry"].isin(top)].groupby(["batch_year", "industry"]).size().reset_index(name="companies")
        fig = go.Figure()
        palette = [YC_ORANGE, TEAL, BLUE, ROSE, "#a855f7", "#84cc16", "#f59e0b", "#64748b"]
        for i, industry in enumerate(top):
            part = yearly[yearly["industry"] == industry]
            fig.add_trace(go.Scatter(x=part["batch_year"], y=part["companies"], mode="lines", stackgroup="one", name=industry, line=dict(width=0.6, color=palette[i % len(palette)])))
        fig.update_layout(xaxis_title="Batch year", yaxis_title="Companies")
        return fig_html(fig, 380)

    @output
    @render.ui
    def yc_matrix():
        df = yc_filtered()
        clean = df[~df["industry"].map(_is_missing_label) & ~df["region"].map(_is_missing_label)].copy()
        if clean.empty:
            return ui.p("No YC companies match the current filters.")
        top_ind = clean["industry"].value_counts().head(14).index
        top_reg = clean["region"].value_counts().head(10).index
        heat = clean[clean["industry"].isin(top_ind) & clean["region"].isin(top_reg)].groupby(["industry", "region"]).agg(
            companies=("name", "count"),
            avg_radar=("radar_score", "mean"),
        ).reset_index()
        fig = px.density_heatmap(
            heat,
            x="region",
            y="industry",
            z="avg_radar",
            histfunc="avg",
            color_continuous_scale=["#f8fafc", "#fed7aa", YC_ORANGE],
            hover_data=["companies"],
        )
        fig.update_layout(xaxis_title="", yaxis_title="", coloraxis_colorbar_title="Avg radar")
        return fig_html(fig, 380)

    @output
    @render.ui
    def yc_tier_distribution():
        df = yc_filtered()
        if df.empty:
            return ui.p("No YC companies match the current filters.")
        chart = df.copy()
        chart["radar_tier"] = pd.Categorical(chart["radar_tier"], categories=RADAR_TIER_ORDER, ordered=True)
        chart["hiring_status"] = np.where(chart["is_hiring"], "Hiring", "Not hiring")
        grouped = chart.groupby(["radar_tier", "hiring_status"], observed=False).size().reset_index(name="companies")
        grouped = grouped[grouped["companies"] > 0]
        fig = px.bar(
            grouped,
            x="radar_tier",
            y="companies",
            color="hiring_status",
            text="companies",
            color_discrete_map={"Hiring": YC_ORANGE, "Not hiring": "#cbd5e1"},
            category_orders={"radar_tier": RADAR_TIER_ORDER, "hiring_status": ["Hiring", "Not hiring"]},
        )
        fig.update_layout(xaxis_title="", yaxis_title="Companies", barmode="stack")
        return fig_html(fig, 360)

    @output
    @render.ui
    def yc_founder_school():
        df = yc_filtered()
        if df.empty:
            return ui.p("No YC companies match the current filters.")
        signals = yc_founder_signals(df)
        if signals.empty:
            return ui.p("Founder and school records are unavailable for the current data.")
        chart = signals.melt(
            id_vars=["slug", "radar_tier"],
            value_vars=["has_founder_record", "has_school_record", "has_prior_company_record"],
            var_name="signal",
            value_name="has_signal",
        )
        chart = chart[chart["has_signal"]].copy()
        if chart.empty:
            return ui.p("No founder, school, or prior-company signals match the current filters.")
        chart["signal"] = chart["signal"].map(
            {
                "has_founder_record": "Founder record",
                "has_school_record": "School record",
                "has_prior_company_record": "Prior-company record",
            }
        )
        chart["radar_tier"] = pd.Categorical(chart["radar_tier"], categories=RADAR_TIER_ORDER, ordered=True)
        grouped = chart.groupby(["radar_tier", "signal"], observed=False)["slug"].nunique().reset_index(name="companies")
        grouped = grouped[grouped["companies"] > 0]
        fig = px.bar(
            grouped,
            x="radar_tier",
            y="companies",
            color="signal",
            barmode="group",
            text="companies",
            color_discrete_map={"Founder record": BLUE, "School record": TEAL, "Prior-company record": ROSE},
            category_orders={"radar_tier": RADAR_TIER_ORDER, "signal": ["Founder record", "School record", "Prior-company record"]},
        )
        fig.update_layout(xaxis_title="", yaxis_title="Companies")
        return fig_html(fig, 360)

    @output
    @render.ui
    def yc_topic_map():
        df = yc_filtered().copy()
        if df.empty:
            return ui.p("No YC companies match the current filters.")
        topic = df[(df["topic_x"].fillna(0) != 0) | (df["topic_y"].fillna(0) != 0)].head(1600)
        if topic.empty:
            return ui.p("Topic projection is unavailable for the current data.")
        fig = px.scatter(
            topic,
            x="topic_x",
            y="topic_y",
            size=topic["team_size"].clip(lower=2),
            color="radar_tier",
            hover_name="name",
            hover_data=["industry", "batch", "region", "radar_score", "one_liner"],
            color_discrete_map=RADAR_TIER_COLORS,
            category_orders={"radar_tier": RADAR_TIER_ORDER},
        )
        fig.update_layout(xaxis_title="Topic component 1", yaxis_title="Topic component 2")
        return fig_html(fig, 440)

    @output
    @render.data_frame
    def yc_table():
        cols = ["name", "batch", "industry", "region", "status", "team_size", "is_hiring", "radar_tier", "radar_score", "website"]
        return render.DataGrid(yc_filtered()[cols].head(700), filters=True, height="520px")

    @output
    @render.ui
    def si_importance():
        imp = STARTUP["importance"].head(12).sort_values("importance")
        if imp.empty:
            return ui.p("Model importance is unavailable.")
        fig = px.bar(
            imp,
            x="importance",
            y="feature",
            color="signal_group",
            orientation="h",
            text=imp["importance"].map(lambda x: f"{x:.3f}"),
            color_discrete_map={"Founder/team": TEAL, "Funding maturity": YC_ORANGE, "Market/geography": BLUE, "Company metadata": ROSE},
        )
        fig.update_layout(xaxis_title="ROC-AUC drop when shuffled", yaxis_title="")
        return fig_html(fig, 420)

    @output
    @render.ui
    def si_funding_ladder():
        summary = startup_signal_summary(startup_filtered())
        ladder = summary[summary["dimension"] == "Funding maturity"].copy()
        if ladder.empty:
            return ui.p("Funding maturity summary is unavailable for the current filters.")
        ladder = ladder.sort_values("exit_rate_pct")
        fig = px.bar(
            ladder,
            x="exit_rate_pct",
            y="segment",
            orientation="h",
            color="median_funding_m",
            text=ladder["exit_rate_pct"].map(lambda x: f"{x:.1f}%"),
            color_continuous_scale=["#2563eb", "#14b8a6", "#ff6600"],
            hover_data={"companies": ":,", "median_funding_m": ":.2f"},
        )
        fig.update_layout(xaxis_title="Exit rate", yaxis_title="", coloraxis_colorbar_title="Median $M")
        fig.update_xaxes(ticksuffix="%")
        return fig_html(fig, 420)

    @output
    @render.ui
    def si_founder_summary():
        summary = startup_signal_summary(startup_filtered())
        if summary.empty:
            return ui.p("Founder signal summary is unavailable for the current filters.")
        fig = px.scatter(
            summary,
            x="exit_rate_pct",
            y="segment",
            color="dimension",
            size="companies",
            hover_data={"companies": ":,", "median_funding_m": ":.2f", "median_founders": ":.1f"},
            color_discrete_sequence=[TEAL, YC_ORANGE, BLUE],
        )
        fig.update_layout(xaxis_title="Exit rate", yaxis_title="")
        fig.update_xaxes(ticksuffix="%")
        return fig_html(fig, 440)

    @output
    @render.ui
    def si_sector():
        sector = grouped_startup_summary(startup_filtered(), "category_code").head(24)
        if sector.empty:
            return ui.p("Sector summary is unavailable for the current filters.")
        fig = px.scatter(
            sector,
            x="companies",
            y="exit_rate_pct",
            size="median_funding_m",
            color="category_code",
            hover_data={"companies": ":,", "exit_rate_pct": ":.1f", "median_funding_m": ":.2f"},
            size_max=48,
        )
        fig.update_layout(xaxis_title="Companies", yaxis_title="Exit rate")
        fig.update_yaxes(ticksuffix="%")
        return fig_html(fig, 440)

    @output
    @render.ui
    def si_geo():
        country = grouped_startup_summary(startup_filtered(), "country_code").head(28)
        if country.empty:
            return ui.p("Country summary is unavailable for the current filters.")
        fig = px.scatter(
            country,
            x="companies",
            y="exit_rate_pct",
            size="median_funding_m",
            color="country_code",
            text="country_code",
            hover_data={"companies": ":,", "exit_rate_pct": ":.1f", "median_funding_m": ":.2f"},
            size_max=50,
        )
        fig.update_layout(xaxis_title="Companies in filtered set", yaxis_title="Exit rate")
        fig.update_xaxes(type="log")
        fig.update_yaxes(ticksuffix="%")
        return fig_html(fig, 440)

    @output
    @render.ui
    def ml_distribution():
        if not STARTUP_READY:
            return ui.p("Startup model data is unavailable.")
        current = ml_probability()
        scores = STARTUP_ML["predicted_exit_probability"].dropna() * 100
        fig = px.histogram(
            scores,
            nbins=40,
            labels={"value": "Predicted exit probability", "count": "Companies"},
            color_discrete_sequence=[YC_ORANGE],
        )
        if not pd.isna(current):
            fig.add_vline(x=current * 100, line_width=3, line_color=BLUE, annotation_text="Scenario", annotation_position="top right")
        fig.update_layout(showlegend=False, xaxis_title="Predicted exit probability", yaxis_title="Companies")
        fig.update_xaxes(ticksuffix="%")
        return fig_html(fig, 420)

    @output
    @render.ui
    def ml_importance():
        imp = STARTUP["importance"].head(10).sort_values("importance")
        if imp.empty:
            return ui.p("Model importance is unavailable.")
        fig = px.bar(
            imp,
            x="importance",
            y="feature",
            color="signal_group",
            orientation="h",
            color_discrete_map={"Founder/team": TEAL, "Funding maturity": YC_ORANGE, "Market/geography": BLUE, "Company metadata": ROSE},
        )
        fig.update_layout(xaxis_title="ROC-AUC drop when shuffled", yaxis_title="")
        return fig_html(fig, 420)

    @output
    @render.data_frame
    def ml_scenario_table():
        scenario = ml_scenario().copy()
        if scenario.empty:
            return render.DataGrid(pd.DataFrame(), height="260px")
        display = scenario.T.reset_index()
        display.columns = ["feature", "value"]
        return render.DataGrid(display, filters=False, height="360px")


async def _redirect_root(request):
    return RedirectResponse(url="/founder-radar/")


async def _favicon(request):
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        '<rect width="32" height="32" rx="6" fill="#ff6600"/>'
        '<text x="16" y="22" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700" fill="white">Y</text>'
        "</svg>"
    )
    return Response(svg, media_type="image/svg+xml")


shiny_app = App(app_ui, server)
asgi_app = Starlette(routes=[Route("/", _redirect_root), Route("/favicon.ico", _favicon), Mount("/founder-radar", app=shiny_app)])
app = shiny_app
