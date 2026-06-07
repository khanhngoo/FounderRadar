# FounderRadar Project Context

## Project Purpose

FounderRadar is a Python Shiny data visualization project for investors. It helps users explore startup ecosystems, especially Y Combinator companies, and surface potentially under-recognized opportunities through interactive visual analytics.

The app has three investor-facing dashboard pages:

- **YC Ecosystem**: focuses on YC company discovery, momentum, hiring activity, geography, industry concentration, and a company explorer.
- **Startup Intelligence**: focuses on historical startup outcome signals using a startup-investments dataset, including exit-proxy modeling, funding maturity, founder/team signals, sectors, and geography.
- **ML Exit Predictor**: exposes the trained historical exit model through scenario inputs and model result visualizations.

The project should be treated as an exploratory investment-research tool, not as investment advice. Scores and model outputs are ranking and explanation aids.

## Runtime Stack

- Python Shiny (`shiny`)
- Starlette/ASGI wrapper for routing
- Uvicorn for local serving
- Pandas and NumPy for data preparation
- Plotly for charts
- scikit-learn for feature engineering and the historical exit model
- KaggleHub for dataset download/sync

The app is served at:

```sh
./.venv/bin/python -m uvicorn app:asgi_app --host 127.0.0.1 --port 3838
```

Then open:

```text
http://127.0.0.1:3838/founder-radar/
```

## Data Sources

The dashboard intentionally requires local Kaggle datasets and does not silently fall back to cached or sample data. If required files are missing, the app shows a data-readiness notice with the fetch command.

Fetch data with:

```sh
./.venv/bin/python scripts/fetch_data.py
```

Required datasets:

- `miguelcorraljr/y-combinator-directory`
- `sashakorovkina/ycombinator-all-funded-companies-dataset`
- `lazarun/y-combinator-jobs-enriched`
- `justinas/startup-investments`

Expected local layout:

- `data/raw/y-combinator-directory/`
- `data/raw/ycombinator-all-funded-companies-dataset/`
- `data/raw/y-combinator-jobs-enriched/`
- `data/raw/startup-investments/`

`startup-investments` must include at least:

- `objects.csv`
- `acquisitions.csv`
- `ipos.csv`
- `relationships.csv`
- `degrees.csv`
- `funding_rounds.csv`

## Main Files

- `app.py`: Shiny UI, filters, chart rendering, tables, styling, and ASGI routes.
- `data_pipeline.py`: dataset readiness checks, data loading, normalization, feature engineering, YC radar scoring, startup exit model, and aggregate summaries.
- `scripts/fetch_data.py`: downloads Kaggle datasets with `kagglehub.dataset_download`, syncs them to `data/raw`, and validates required files.
- `README.md`: setup, run, and Docker instructions.
- `YC_DASHBOARD_DESIGN.md`: rationale and visualization choices for the YC dashboard.
- `STARTUP_INTELLIGENCE_DASHBOARD_DESIGN.md`: rationale and visualization choices for the historical startup intelligence dashboard.
- `ML_Project2_DataVisualization.ipynb`: source notebook/context for the Startup Intelligence analysis layer.

## Application Structure

`app.py` loads all dashboard data once at import time:

```python
DATA = load_dashboard_data(BASE_DIR)
READINESS = DATA["readiness"]
YC = DATA["yc"]
STARTUP = DATA["startup"]
```

Readiness flags control whether each page renders the dashboard or a missing-data notice:

- `YC_READY`
- `STARTUP_READY`

The UI is a single Shiny app with:

- top tab navigation
- data status badges
- page-local filter bars for each dashboard
- separate YC Ecosystem, Startup Intelligence, and ML Exit Predictor pages

The ASGI wrapper redirects `/` to `/founder-radar/` and mounts the Shiny app at `/founder-radar`.

## YC Ecosystem Dashboard

Goal: answer **"Where is YC momentum hiding?"**

Core normalized company fields:

- `name`
- `slug`
- `batch`
- `status`
- `industry`
- `subindustry`
- `region`
- `one_liner`
- `website`
- `team_size`
- `is_hiring`
- `top_company`
- `tags_text`
- `source_table`

Derived YC features:

- `batch_year`
- `batch_era`
- `sector_momentum`
- `geo_novelty`
- `industry_density`
- `hiring_signal`
- `recognition_score`
- `momentum_score`
- `underrecognition_signal`
- `raw_opportunity_score`
- `radar_score`
- `radar_tier`
- `topic_x`
- `topic_y`

Radar score is a heuristic, not a trained investment model. It starts with a raw opportunity score that combines:

- momentum
- under-recognition
- normalized team size

The displayed `radar_score` is the percentile rank of that raw opportunity score across the loaded YC company universe, scaled from 0 to 100. This keeps dashboard tiers useful even when raw score values are compressed.

Current YC views:

- KPI cards for filtered company count, high radar count, hiring count, and median team size
- Opportunity Quadrant Map
- YC Batch Pulse Streamgraph
- Industry x Region Matrix
- Company Explorer data grid

Key YC filters:

- batch era
- industry
- region
- status
- radar tier
- hiring only
- text search over company, industry, region, tags, and one-liner

## Startup Intelligence Dashboard

Goal: answer **"Which historical signals are associated with observable startup exits?"**

This page uses only the historical Kaggle Startup Investments dataset. SEC EDGAR/Form D data is intentionally excluded by current project scope.

The modeling table joins:

- company metadata and funding totals from `objects.csv`
- acquisition/IPO exit proxy from `acquisitions.csv` and `ipos.csv`
- founder-role and prior-company exposure from `relationships.csv`
- founder education signals from `degrees.csv`
- funding round maturity and participant signals from `funding_rounds.csv`

`success_exit` is a proxy based on acquisition or IPO evidence. It does not capture all forms of startup success.

Startup model behavior:

- Uses a RandomForestClassifier inside an sklearn Pipeline.
- Numeric funding-like values are imputed and log-transformed.
- Other numeric values are median-imputed.
- Categorical fields are imputed and one-hot encoded.
- Model rows are capped/sampled for performance.
- Permutation importance is computed on a capped held-out sample and grouped into investor-readable signal families.

Startup Intelligence views:

- KPI cards for model rows, success exits, ROC-AUC, and filtered chart row count
- Model Signal Importance
- Funding Maturity Ladder
- Founder Signal Summary
- Sector Risk / Reward
- Geography Ecosystem View
- Filtered aggregate charts that respond to Startup Intelligence filters

Key Startup Intelligence filters:

- sector
- country
- minimum predicted exit probability
- text search over company, sector, and country

## ML Exit Predictor Dashboard

Goal: answer **"How does the historical exit model score a user-defined scenario?"**

The page uses the trained RandomForest pipeline from the startup-investments modeling table. Users can adjust sector, country, founding year, funding, founder, education, participant, and round-history values. The page shows the live predicted exit probability, dataset comparison benchmarks, prediction distribution, model signal importance, and a scenario feature table.

## Design Direction

The product should feel like an analytical investor workspace rather than a marketing page.

Current design choices:

- All dashboard pages use an off-white workspace, graphite text, compact metric cards, and YC orange as the main accent.
- Secondary colors include teal, blue, and rose to avoid a one-note orange theme.
- Charts should remain interpretable and auditable; company-level tables are important because investors need to inspect examples behind aggregate patterns.

## Engineering Notes

- Data loading happens at app import time, so large data/model work affects server startup.
- `data/raw` can be large and should be treated as local data, not application logic.
- The app currently loads Plotly from a CDN in `app.py`.
- `data/cache/yc_oss_companies.json` exists but is not part of the redesigned required-data path.
- `.git` is not usable as a normal Git repository in the current workspace, so do not assume git commands will work.
- Keep future changes aligned with the existing compact Shiny/Plotly architecture unless there is a strong reason to introduce a new framework.

## Verification Commands

Useful checks:

```sh
./.venv/bin/python -m py_compile app.py data_pipeline.py scripts/fetch_data.py
./.venv/bin/python -m uvicorn app:asgi_app --host 127.0.0.1 --port 3838
```

Data fetch, if required:

```sh
./.venv/bin/python scripts/fetch_data.py
```
