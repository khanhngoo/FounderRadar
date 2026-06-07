from __future__ import annotations

import shutil
import sys
from pathlib import Path

import kagglehub


DATASETS = {
    "y-combinator-directory": "miguelcorraljr/y-combinator-directory",
    "ycombinator-all-funded-companies-dataset": "sashakorovkina/ycombinator-all-funded-companies-dataset",
    "y-combinator-jobs-enriched": "lazarun/y-combinator-jobs-enriched",
    "startup-investments": "justinas/startup-investments",
}

STARTUP_REQUIRED_FILES = {
    "objects.csv",
    "acquisitions.csv",
    "ipos.csv",
    "relationships.csv",
    "degrees.csv",
    "funding_rounds.csv",
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    raw = root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for folder, dataset in DATASETS.items():
        target = raw / folder
        print(f"Downloading latest {dataset}")
        try:
            cache_path = Path(kagglehub.dataset_download(dataset))
            print(f"Path to dataset files: {cache_path}")
            sync_dataset(cache_path, target)
            print(f"Synced {dataset} -> {target}")
        except Exception as exc:
            print(f"Failed to download {dataset}: {exc}")
            failures.append(dataset)

    if failures:
        raise SystemExit("KaggleHub download failed for: " + ", ".join(failures))

    missing = validate_downloads(raw)
    if missing:
        print("Downloaded files, but validation found missing data:")
        for item in missing:
            print(f"- {item}")
        raise SystemExit(1)

    print("All required Kaggle datasets are present.")


def sync_dataset(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)

    for old_file in target.rglob("*"):
        if old_file.is_file() and old_file.name != ".gitkeep":
            old_file.unlink()

    for source_file in source.rglob("*"):
        if source_file.is_file():
            relative = source_file.relative_to(source)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, destination)


def validate_downloads(raw: Path) -> list[str]:
    missing: list[str] = []

    for folder, dataset in DATASETS.items():
        path = raw / folder
        csvs = sorted(path.rglob("*.csv"))
        if not csvs:
            missing.append(f"{dataset}: expected at least one CSV under {path}")

    startup_path = raw / "startup-investments"
    startup_files = {p.name for p in startup_path.rglob("*.csv")} if startup_path.exists() else set()
    for filename in sorted(STARTUP_REQUIRED_FILES - startup_files):
        missing.append(f"startup-investments: missing {startup_path / filename}")

    return missing


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted.")
