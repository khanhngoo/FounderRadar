# FounderRadar Project Report

## 1. Vision & Style

FounderRadar is an investor-oriented data visualization dashboard for exploring startup ecosystems and identifying companies that deserve deeper research. The core vision is to give venture capital analysts, startup investors, and student researchers a compact analytical workspace where company-level opportunity signals can be compared against broader market, geography, founder, and funding patterns.

The project is built around three connected views:

- **YC Ecosystem** asks: where is Y Combinator momentum hiding?
- **Startup Intelligence** asks: which historical startup signals are associated with observable acquisition or IPO outcomes?
- **ML Exit Predictor** asks: how would a hypothetical company scenario be scored by the historical exit model?

The dashboard is intentionally framed as an exploratory research tool, not as investment advice. This distinction is important because both major scoring systems are derived signals. The YC `radar_score` is a heuristic ranking score, while the Startup Intelligence `predicted_exit_probability` is a machine learning output based on historical data. Both are useful for triage and pattern discovery, but they should support human review rather than replace investor judgment.

The visual style follows the idea of an investor research desk rather than a marketing website. All dashboard pages use an off-white workspace, graphite text, compact cards, and YC orange as the primary accent. This creates a recognizable connection to Y Combinator while keeping the interface analytical and restrained. Teal, blue, and rose provide secondary signal colors across ecosystem, historical outcome, and model views.

The interface style is deliberately dense but organized. Investors need to scan, compare, filter, and inspect details quickly. For that reason, the project avoids large decorative layouts and focuses on dashboards, filters, chart panels, KPI cards, and company tables. The tables are especially important because aggregate charts alone are not enough for investor workflows; users need to inspect the companies behind a pattern before deciding whether the pattern is meaningful.

**Recommendation:** Add a short methodology drawer or collapsible note inside each dashboard tab. The YC page should explain `radar_score`, `momentum_score`, and `recognition_score`; the Startup Intelligence page should explain `success_exit` and model limitations; the ML page should explain that predicted probabilities are historical ranking aids, not direct forecasts.

## 2. Data Architecture

FounderRadar uses a local Kaggle-based data architecture. The project intentionally requires the real datasets under `data/raw/` and does not silently fall back to sample data. This is a correct design choice for a data visualization project because it protects the integrity of the dashboard: if required files are missing, the user sees a data-readiness notice instead of a misleading partial dashboard.

The data-fetch workflow is implemented in `scripts/fetch_data.py`. It uses `kagglehub.dataset_download(...)` to download the required Kaggle datasets and then copies them into predictable folders under `data/raw/`. The app checks readiness through `dataset_readiness()` in `data_pipeline.py`.

The YC Ecosystem dashboard uses three YC-focused datasets:

- `miguelcorraljr/y-combinator-directory`
- `sashakorovkina/ycombinator-all-funded-companies-dataset`
- `lazarun/y-combinator-jobs-enriched`

These datasets are normalized into a company-level table with fields such as company name, slug, batch, status, industry, subindustry, region, one-liner, website, team size, hiring status, top-company flag, tags, and source table. The pipeline then engineers YC-specific features:

- `batch_year` and `batch_era`
- `sector_momentum`
- `geo_novelty`
- `industry_density`
- `hiring_signal`
- `recognition_score`
- `momentum_score`
- `underrecognition_signal`
- `radar_score`
- `radar_tier`
- `topic_x` and `topic_y` from TF-IDF plus PCA text projection

The current local load contains **6,389 YC companies**. The YC scoring architecture is correct because it separates interpretable intermediate signals before producing a final percentile-ranked `radar_score`. A percentile score is also appropriate because it keeps the tiers useful even when raw opportunity scores are compressed.

The Startup Intelligence dashboard uses the Kaggle Startup Investments dataset:

- `justinas/startup-investments`

The startup architecture is relational. The project joins multiple CSV files into a company-level modeling table:

- `objects.csv` provides company metadata, funding totals, status, geography, founding dates, milestones, and relationship counts.
- `acquisitions.csv` and `ipos.csv` define the `success_exit` proxy.
- `relationships.csv` identifies founder-role relationships and prior company exposure.
- `degrees.csv` provides founder education signals.
- `funding_rounds.csv` provides funding maturity, round amounts, participants, and funding span features.

The target variable, `success_exit`, is defined as evidence of acquisition or IPO. This is a defensible proxy for investor-observable outcomes, but it is not a complete definition of startup success. It excludes revenue growth, profitability, private valuation growth, survival, and strategic importance. The report and dashboard should continue to state this limitation clearly.

The current local load contains **42,751 startup ML rows**, **3,250 success exits**, and a held-out ROC-AUC of approximately **0.832**. The model uses a scikit-learn `Pipeline` with imputation, log transformation for funding-like values, one-hot encoding for categorical sector and country variables, and a balanced `RandomForestClassifier`. Permutation importance is computed and grouped into investor-readable families such as Founder/team, Funding maturity, Market/geography, and Company metadata.

This architecture is appropriate because the dashboard has two different analytical jobs. YC discovery uses current ecosystem and operating-momentum heuristics, while Startup Intelligence uses historical data and supervised learning. Combining both into one data model would blur their meanings. Keeping them separate makes the visualization story more honest and easier to audit.

**Recommendation:** Keep validating YC founder-context ingestion against the local Kaggle files. The UI now includes Top Founder Schools and Top Prior Companies panels, and those panels should remain explicit that founder, school, and prior-company records may be sparse depending on source-data coverage.

## 3. Design Elements

FounderRadar uses color, shape, and size to support the dashboard narrative rather than simply decorate the page.

Color is used to encode meaning and separate analytical modes. YC orange (`#ff6600`) acts as the main brand and opportunity accent. Graphite and muted gray create a professional analytical base. Teal, blue, and rose provide secondary signal families so the dashboard does not become a one-note orange interface. In Startup Intelligence, teal often relates to founder/team signals, orange to funding maturity, blue to market/geography, and rose to company metadata or alternate signal categories.

Shape is used through familiar dashboard structures: metric cards, chart panels, tab navigation, filter bars, scatterplot bubbles, heatmap cells, bar charts, and tables. These shapes match the user workflow. KPI cards give immediate context. Scatterplots support tradeoff analysis. Heatmaps support concentration scanning. Bar charts support ranked comparison. Data grids support detailed inspection.

Size is used as an additional data channel. In the YC Opportunity Quadrant Map, bubble size represents team size, allowing users to see whether a company has operational scale without making team size the only ranking factor. In Startup Intelligence charts, bubble size represents company count or median funding, which helps users avoid overreading small segments. This is especially important for investor dashboards because a high exit rate from a tiny group can look impressive but may not be reliable.

The major visualization choices support specific parts of the project story:

- **Opportunity Quadrant Map:** plots recognition against momentum, highlighting lower-recognition and higher-momentum companies.
- **YC Batch Pulse Streamgraph:** shows how YC industry concentration changes across batch years.
- **Industry x Region Matrix:** shows where strong radar scores cluster by sector and geography.
- **Radar Tier Distribution:** summarizes the filtered YC opportunity pool by radar tier and hiring signal.
- **Topic Projection Map:** uses company descriptions, tags, and industries to reveal thematic clusters.
- **Model Signal Importance:** explains which features move the historical exit model.
- **Funding Maturity Ladder:** shows exit-rate differences across funding-stage segments.
- **Founder Signal Summary:** compares team structure, education coverage, and funding maturity on a shared outcome scale.
- **Sector Risk / Reward:** separates sector scale from observed exit outcomes and funding intensity.
- **Geography Ecosystem View:** compares country-level company count, exit rate, and capital intensity.
- **Prediction Distribution:** shows where a user-created ML scenario sits among existing model scores.

The design is strongest when it pairs aggregate charts with inspectable company-level outputs. The YC Company Explorer and ML Scenario Feature Values table make the dashboard more rigorous because users can verify what is behind a score or chart pattern.

**Recommendation:** Add visible sample-size labels to charts that show rates or averages. For example, the Sector Risk / Reward, Geography Ecosystem View, Funding Maturity Ladder, and Industry x Region Matrix should make sample size more visible so users do not overinterpret unstable small groups.

## 4. Technical Rigor

The project uses a pragmatic Python data-application stack:

- **Python Shiny** for the dashboard UI and reactivity.
- **Starlette/ASGI** for routing and mounting the app at `/founder-radar/`.
- **Uvicorn** for local serving.
- **Pandas and NumPy** for loading, cleaning, joining, and aggregating data.
- **Plotly** for interactive charts.
- **scikit-learn** for preprocessing, modeling, and permutation importance.
- **KaggleHub** for reproducible dataset download and local synchronization.

These choices are technically appropriate for the project. Python Shiny keeps the application close to the data science workflow while still supporting interactive controls and reactive charts. Pandas is the right tool for the relational CSV joins and feature engineering. Plotly is appropriate because the dashboard depends heavily on hover details and interactive inspection. scikit-learn is appropriate because the model is a classical tabular classification problem that benefits from transparent preprocessing and standard metrics.

The source code is organized around clear responsibilities:

- `app.py` defines the user interface, filters, chart rendering, tables, visual styling, and ASGI routes.
- `data_pipeline.py` handles dataset readiness, data loading, normalization, YC scoring, startup feature engineering, model training, predictions, and aggregate summaries.
- `scripts/fetch_data.py` downloads and validates the required Kaggle datasets.
- The markdown design documents explain the dashboard rationale and chart choices.

The technical rigor is strongest in the data-readiness and modeling pipeline. The app checks whether required datasets exist before rendering the analysis pages. The startup model uses train/test splitting, stratification, imputation, log transformation, one-hot encoding, class balancing, ROC-AUC evaluation, and permutation importance. These are appropriate choices for a student data visualization project with an ML component.

The YC `radar_score` is also documented as a heuristic instead of a trained investment model. This is important technical honesty. The formula combines momentum, under-recognition, and normalized team size, then converts the raw opportunity score into a percentile rank. This makes the result easier to use for visual tiers while avoiding a false claim that the score predicts investment returns.

There are some technical limitations. Data loading and model fitting happen at app import time, which can slow server startup. Plotly is loaded from a CDN, so the dashboard depends on network access for full chart rendering. The YC Company Explorer data grid caps displayed rows for performance. The RandomForest model is sampled to 12,000 rows for performance, and permutation importance is computed on a capped sample. These are reasonable tradeoffs, but they should be documented so users understand performance and completeness constraints.

**Recommendation:** Add concise docstrings to the most important pipeline functions, especially `_prepare_yc_features()`, `_fit_startup_exit_model()`, `_load_startup_founder_features()`, and `_add_text_projection()`. The current code is readable, but formal documentation would improve grading against a technical-rigor rubric.

## 5. Accessibility & Interactivity

Interactivity is central to FounderRadar. The dashboard is designed around user-controlled exploration rather than static reporting.

The YC Ecosystem page supports filters for batch era, industry, region, status, radar tier, and free-text search across company names, industries, regions, tags, and one-liners. These controls let investors move from broad ecosystem patterns to narrow opportunity lists. For example, a user can filter for recent YC companies in a specific industry and then inspect high-radar companies in the table.

The Startup Intelligence page supports filters for sector, country, minimum predicted exit probability, and free-text search. The probability slider is especially important because it turns the ML output into an interactive thresholding tool. Users can see how the filtered population changes as they raise or lower the score cutoff.

The ML Exit Predictor page provides scenario controls for sector, country, founding year, funding totals, funding rounds, milestones, relationship count, founder count, education records, prior-company links, participant counts, and funding span. This makes the model more concrete: instead of only viewing global model outputs, users can test how a hypothetical company profile is scored and compare it against the dataset distribution.

Plotly hover interactions support accessibility in the analytical sense because they expose details without overcrowding charts. Users can inspect company names, one-liners, radar tiers, exit rates, median funding, and company counts directly from the charts. The Shiny `DataGrid` tables add another layer of interactivity through filtering and row-level inspection.

The project also includes responsive CSS. At narrower widths, filter bars, metric grids, and chart grids collapse into single-column layouts. This improves usability on smaller screens and prevents the dashboard from relying only on wide desktop monitors.

Accessibility could be improved further. The current dashboard relies heavily on color to distinguish categories and signal groups. It uses clear labels and hover text, but color-blind users may still have difficulty distinguishing some categories. Some interactive chart content may also be less accessible to keyboard-only or screen-reader users. Since Plotly charts are visually rich, the app should provide text summaries or table equivalents for key chart findings.

**Recommendation:** Add accessibility-focused improvements before final submission:

- Add keyboard-visible focus states for form inputs, tabs, and table controls.
- Add short text summaries below complex charts.
- Avoid relying only on color by adding labels, annotations, or pattern-independent grouping where practical.
- Add clearer chart titles or captions that explain what a user should compare.
- Keep tables as accessible alternatives to chart-only insights.
- Consider adding downloadable CSV outputs for filtered company and model results.

## Recommended Improvements Summary

The project is already coherent as a VC and startup-investor dashboard, but the following improvements would make it stronger:

1. Add methodology notes inside the app for `radar_score`, `success_exit`, and `predicted_exit_probability`.
2. Keep YC founder-context loading auditable so the founder school and prior-company views populate when source data supports them.
3. Add CSV export for filtered company tables and model results.
4. Add a prediction calibration chart to compare predicted probabilities with observed exit rates.
5. Add a confusion matrix at the current ML probability threshold.
6. Add sample-size labels or warnings to rate-based charts.
7. Add accessibility improvements such as focus states, text summaries, and non-color encodings.
8. Add docstrings to important pipeline and modeling functions.
9. Consider adding the notebook's SEC Form D live-data layer as a future optional panel, clearly labeled as U.S.-centric, incomplete, and weakly supervised.
10. Move expensive model training out of app import time if startup performance becomes a problem.

## Conclusion

FounderRadar succeeds as an analytical dashboard because its design, data architecture, and technical implementation all support the same visualization story: investors need to discover patterns, inspect examples, and understand uncertainty. The YC Ecosystem page helps users find under-recognized operating momentum. The Startup Intelligence page connects historical exits to funding, geography, sector, and founder/team signals. The ML Exit Predictor makes the model interactive and auditable through scenario testing.

The strongest aspect of the project is that it does not present a single opaque score as truth. Instead, it combines scoring, charts, filters, and company-level tables so users can move between overview and inspection. With stronger in-app methodology notes, accessibility additions, model calibration views, and improved founder-context ingestion, FounderRadar would become a more rigorous and evaluator-ready startup-investor dashboard.
