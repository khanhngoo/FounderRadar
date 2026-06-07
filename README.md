# FounderRadar

Python Shiny dashboard for exploring YC companies and surfacing under-recognized opportunities.

## Data

The redesigned dashboard requires Kaggle datasets. It no longer silently falls back to `yc-oss` data.

```sh
./.venv/bin/python scripts/fetch_data.py
```

The fetch script uses `kagglehub.dataset_download(...)` and copies the downloaded files into predictable folders under `data/raw/`.

The fetch script downloads:

- `miguelcorraljr/y-combinator-directory`
- `sashakorovkina/ycombinator-all-funded-companies-dataset`
- `lazarun/y-combinator-jobs-enriched`
- `justinas/startup-investments`

If these files are missing, the app shows a data-readiness notice with the setup command.

## Local Run

```sh
./.venv/bin/python -m py_compile app.py data_pipeline.py scripts/fetch_data.py
./.venv/bin/python scripts/fetch_data.py
./.venv/bin/python -m uvicorn app:asgi_app --host 127.0.0.1 --port 3838
```

Open:

```text
http://127.0.0.1:3838/founder-radar/
```

## Docker Deploy

```sh
docker build -t founderradar-shiny:python .
docker rm -f founderradar-shiny 2>/dev/null || true
docker run -d --name founderradar-shiny --restart unless-stopped -p 3838:3838 founderradar-shiny:python
```
