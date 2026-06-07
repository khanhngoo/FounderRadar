# FounderRadar

FounderRadar is a Python Shiny dashboard for startup ecosystem research. It helps users explore Y Combinator companies, identify under-recognized YC opportunities, and inspect historical startup outcome signals from the Kaggle Startup Investments dataset.

The app has three dashboard tabs:

- **YC Ecosystem**: YC company discovery, radar scoring, hiring momentum, geography, industry concentration, founder/school signals, and topic clusters.
- **Startup Intelligence**: historical exit-proxy analysis using funding, founder, sector, and geography signals.
- **ML Exit Predictor**: an interactive scenario tool for the historical startup exit model.

The dashboard is an exploratory research tool, not investment advice.

## Requirements

- Python 3.10 or newer
- Internet access for first-time dependency and data downloads
- Kaggle access through `kagglehub`

The app is built with:

- Python Shiny
- Pandas and NumPy
- Plotly
- scikit-learn
- Starlette and Uvicorn
- KaggleHub

## Fresh Setup

Run these commands from the project root.

### 1. Create a Virtual Environment

```sh
python3 -m venv .venv
```

Activate it:

```sh
source .venv/bin/activate
```

On Windows PowerShell, use:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```sh
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Download Required Data

```sh
python scripts/fetch_data.py
```

The script downloads these Kaggle datasets and copies them into `data/raw/`:

- `miguelcorraljr/y-combinator-directory`
- `sashakorovkina/ycombinator-all-funded-companies-dataset`
- `lazarun/y-combinator-jobs-enriched`
- `justinas/startup-investments`

Expected folders after a successful download:

```text
data/raw/y-combinator-directory/
data/raw/ycombinator-all-funded-companies-dataset/
data/raw/y-combinator-jobs-enriched/
data/raw/startup-investments/
```

If KaggleHub asks for authentication, follow Kaggle's normal API-token setup for your machine, then rerun the fetch command.

### 4. Validate the Python Files

```sh
python -m py_compile app.py data_pipeline.py scripts/fetch_data.py
```

### 5. Run the Dashboard

```sh
python -m uvicorn app:asgi_app --host 127.0.0.1 --port 3838
```

Open this URL in your browser:

```text
http://127.0.0.1:3838/founder-radar/
```

If port `3838` is already in use, choose another port:

```sh
python -m uvicorn app:asgi_app --host 127.0.0.1 --port 3839
```

Then open:

```text
http://127.0.0.1:3839/founder-radar/
```

## Data Notes

FounderRadar intentionally requires local Kaggle data. It does not silently fall back to sample data, because the charts and model should reflect the real project datasets.

If required data is missing, the app shows a data-readiness notice with the command to fetch data.

The `startup-investments` dataset must include at least:

- `objects.csv`
- `acquisitions.csv`
- `ipos.csv`
- `relationships.csv`
- `degrees.csv`
- `funding_rounds.csv`

## Common Issues

### `ModuleNotFoundError`

Make sure the virtual environment is active and dependencies are installed:

```sh
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### KaggleHub Download Fails

Check that you have internet access and Kaggle access configured. Then rerun:

```sh
python scripts/fetch_data.py
```

### App Loads but Shows a Data-Readiness Notice

The CSV files are missing or incomplete. Run:

```sh
python scripts/fetch_data.py
```

### Server Starts but Browser Cannot Open the App

Confirm the URL includes `/founder-radar/`:

```text
http://127.0.0.1:3838/founder-radar/
```

If another process is using the port, start Uvicorn on another port.

## Project Files

- `app.py`: Shiny UI, chart rendering, filters, tables, styling, and ASGI routes.
- `data_pipeline.py`: data loading, normalization, YC radar scoring, startup feature engineering, and ML model training.
- `scripts/fetch_data.py`: downloads and validates Kaggle datasets.
- `CHARTS.md`: chart inventory, rationale, caveats, and suggested additions.
- `CONTEXT.md`: project context and engineering notes.
- `YC_DASHBOARD_DESIGN.md`: YC dashboard design rationale.
- `STARTUP_INTELLIGENCE_DASHBOARD_DESIGN.md`: Startup Intelligence design rationale.

## Docker Run

Build the image:

```sh
docker build -t founderradar-shiny:python .
```

Run the container:

```sh
docker rm -f founderradar-shiny 2>/dev/null || true
docker run -d --name founderradar-shiny --restart unless-stopped -p 3838:3838 founderradar-shiny:python
```

Open:

```text
http://127.0.0.1:3838/founder-radar/
```

Note: the container still needs the required data available inside the image or mounted into the container, depending on how you deploy it.
