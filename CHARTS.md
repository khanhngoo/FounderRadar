# FounderRadar Chart Inventory

This document summarizes the charts currently implemented in FounderRadar, why they were chosen, and suggested chart additions for each dashboard.

The app has three dashboard pages:

- **YC Ecosystem**: an opportunity-discovery workspace for Y Combinator companies.
- **Startup Intelligence**: a historical outcome-signal workspace based on the Startup Investments dataset.
- **ML Exit Predictor**: an interactive scenario workspace for the trained historical exit model.

## YC Ecosystem Dashboard

The YC page is designed to answer: **Where is YC momentum hiding?**

It uses the YC-focused Kaggle datasets loaded through `data_pipeline.py` and applies filters for batch era, industry, region, status, radar tier, hiring status, and text search.

### KPI Cards

Current cards:

- **Companies**: count of filtered YC companies.
- **High Radar**: count of companies with `radar_score >= 75`.
- **Hiring**: count of companies with an active hiring signal.
- **Median Team**: median team size in the filtered set.

Why these are chosen:

- They provide a fast read on the active filter context before the user interprets the charts.
- They connect directly to the dashboard thesis: company supply, strong radar candidates, hiring momentum, and team scale.
- They make filter changes immediately auditable.

### Opportunity Quadrant Map

Implementation:

- Plot type: Plotly scatter plot.
- X-axis: `recognition_score`.
- Y-axis: `momentum_score`.
- Bubble size: clipped `team_size`.
- Color: `industry`.
- Hover fields: company name, batch, region, status, radar tier, radar score, and one-liner.
- Visual benchmark: median recognition and upper-quartile momentum, with a highlighted area for lower-recognition and higher-momentum companies.

Why it is chosen:

- It turns the core FounderRadar idea into a visual search task: find companies that appear active but not already highly recognized.
- Recognition and momentum are interpretable dimensions, so the user can reason about the score rather than only consuming a black-box rank.
- Bubble size adds useful scale without making team size the only ranking factor.

Caveats:

- The current implementation caps displayed rows with `head(1400)`, so very broad filter states may not show every matching company.
- Industry coloring can become visually busy when many industries are present.
- The underlying `recognition_score` and `momentum_score` are heuristics, not trained investment predictions.

### YC Batch Pulse Streamgraph

Implementation:

- Plot type: stacked line/area streamgraph using Plotly scatter traces with `stackgroup`.
- X-axis: `batch_year`.
- Y-axis: company count.
- Series: top 8 industries in the filtered data.

Why it is chosen:

- It shows how YC's sector mix changes over time.
- It is better than a single yearly bar chart for seeing changing industry composition.
- It helps identify sector momentum across cohorts, not just current sector size.

Caveats:

- Only the top 8 industries are shown.
- The chart depends on parsed batch years; companies without valid batch-year data are excluded.
- Unspecified and unknown industry labels are excluded before selecting top industries.
- A stacked shape can make individual category comparisons harder when many series overlap.

### Industry x Region Matrix

Implementation:

- Plot type: Plotly density heatmap.
- X-axis: top 10 regions.
- Y-axis: top 14 industries.
- Color value: average `radar_score`.
- Hover field: company count.

Why it is chosen:

- It makes sector-by-geography concentration easier to scan than a long table.
- Average radar score helps users look for pockets of strong candidates rather than simply the largest categories.
- Hovered company counts help prevent over-reading high averages from small cells.

Caveats:

- The color encodes average radar score, not company count.
- Small cells can produce unstable averages.
- The current chart only includes top industries and top regions after filtering.
- Unspecified and unknown industry or region labels are excluded before selecting top categories.

### Radar Tier Distribution

Implementation:

- Plot type: stacked bar chart.
- X-axis: `radar_tier`.
- Y-axis: company count.
- Stack color: hiring status.
- Tier order: Watch, Emerging, High Potential, Breakout.

Why it is chosen:

- It gives users an immediate read on the shape of the filtered opportunity pool.
- Splitting by hiring status shows whether strong radar tiers are also showing current operating momentum.
- It is a compact complement to the Opportunity Quadrant Map because it summarizes the same filtered population by tier.

Caveats:

- The chart is count-based and does not explain why companies land in a tier.
- Hiring is a binary signal in the current normalized data, so it does not show job count or hiring velocity.

### Founder / School Signal View

Implementation:

- Plot type: grouped bar chart.
- X-axis: `radar_tier`.
- Y-axis: company count.
- Bar groups: founder record, school record, and prior-company record.
- Founder records are aggregated to company-level flags to avoid overcounting companies with multiple founder rows.

Why it is chosen:

- It surfaces the founder context that is already normalized in the YC data pipeline.
- It helps users see whether higher radar tiers have stronger founder metadata coverage.
- It makes school and prior-company signals visible without making the main Company Explorer table too wide.

Caveats:

- Founder, school, and prior-company fields depend on dataset coverage and may be sparse.
- School and prior-company signals are coverage indicators, not direct quality measures.
- The view counts companies with available records; it does not count individual founders.

### Topic Projection Map

Implementation:

- Plot type: Plotly scatter plot.
- X-axis: `topic_x`.
- Y-axis: `topic_y`.
- Bubble size: clipped `team_size`.
- Color: `radar_tier`.
- Hover fields: company name, industry, batch, region, radar score, and one-liner.
- Coordinates come from the pipeline's TF-IDF/PCA projection over industry, tags, and one-liner text.

Why it is chosen:

- It reveals thematic clusters that may cut across explicit industry labels.
- It uses already-computed pipeline fields, so it adds discovery value without a new data dependency.
- Coloring by radar tier helps users spot clusters with many stronger candidates.

Caveats:

- PCA axes are abstract components, not human-labeled categories.
- The chart is unavailable if there is too little usable text to compute a projection.
- Broad filters can still create dense clusters, so hover inspection remains important.

### Company Explorer

Implementation:

- Plot type: Shiny `DataGrid`.
- Columns: name, batch, industry, region, status, team size, hiring flag, radar tier, radar score, and website.
- Table-level filters enabled.
- Limited to the first 700 filtered rows.

Why it is chosen:

- It makes aggregate chart findings actionable.
- Investors need to inspect companies behind any pattern before forming a view.
- It exposes enough raw fields to audit why a company appears in a tier or visual cluster.

Caveats:

- It is a table rather than a chart, but it is essential to the dashboard's workflow.
- Founder context is loaded in the pipeline but is not currently exposed in this table.

### Suggested YC Additions

1. **Batch Era x Radar Tier Heatmap**

   Show counts or average radar score by `batch_era` and `radar_tier`.

   Why: this would quickly answer whether the strongest opportunities are concentrated in recent, mid-age, or older cohorts.

2. **Industry-Level Radar Leaderboard**

   Show top industries by median radar score, high-radar count, and hiring share.

   Why: this would summarize which sectors deserve deeper inspection before users move into company-level exploration.

## Startup Intelligence Dashboard

The Startup Intelligence page is designed to answer: **Which historical signals are associated with observable startup exits?**

It uses the `justinas/startup-investments` Kaggle dataset. The exit label is a proxy based on acquisition or IPO evidence, not a complete measure of startup success.

### KPI Cards

Current cards:

- **Model Rows**: count of companies with enough signal to enter the modeling table.
- **Success Exits**: acquisition or IPO proxy count.
- **ROC-AUC**: held-out model score.
- **Filtered**: count of rows matching current Startup Intelligence filters.

Why these are chosen:

- They establish model scope and basic outcome prevalence.
- ROC-AUC gives a quick model-quality signal before users inspect importance or predictions.
- The filtered count keeps the explorer and charts grounded in the active filter context.

### Model Signal Importance

Implementation:

- Plot type: horizontal bar chart.
- X-axis: permutation importance measured as ROC-AUC drop when shuffled.
- Y-axis: model feature.
- Color: signal group: founder/team, funding maturity, market/geography, or company metadata.
- Shows top 12 features.

Why it is chosen:

- It makes the RandomForest model more auditable.
- Grouped colors translate raw model fields into investor-readable signal families.
- Horizontal bars are effective for ranked feature names.

Caveats:

- Permutation importance is computed on a capped held-out sample for performance.
- Importance is model-dependent and exploratory; it should not be read as causal evidence.
- One-hot encoded categorical effects are summarized at the source-feature level, not by individual category value.

### Funding Maturity Ladder

Implementation:

- Plot type: horizontal bar chart.
- X-axis: exit rate percentage.
- Y-axis: funding maturity segment.
- Color: median funding in millions.
- Text label: exit rate percentage.
- Segments are derived from funding-round count.

Why it is chosen:

- It presents funding stage as a maturity progression.
- It combines outcome rate with funding intensity, which helps separate maturity from capital volume.
- It is simpler and more defensible than a complex funding flow diagram for this dataset.

Caveats:

- Funding rounds are broad historical records and may be incomplete.
- Exit rate is based on the acquisition/IPO proxy only.
- Median funding can be skewed by sparse or stale records.

### Founder Signal Summary

Implementation:

- Plot type: bubble scatter plot.
- X-axis: exit rate percentage.
- Y-axis: signal segment.
- Color: dimension: team structure, education coverage, or funding maturity.
- Bubble size: company count.
- Hover fields: company count, median funding, and median founders.

Why it is chosen:

- It compares several interpretable startup signal families on the same outcome axis.
- Bubble size prevents users from treating tiny segments and large segments as equally reliable.
- It is compact enough to support side-by-side reading with sector and funding charts.

Caveats:

- Founder and education data coverage is incomplete.
- Education signals can introduce dataset and prestige bias.
- The chart compares observed associations, not causal drivers.

### Sector Risk / Reward

Implementation:

- Plot type: bubble scatter plot.
- X-axis: company count.
- Y-axis: exit rate percentage.
- Bubble size: median funding in millions.
- Color: `category_code`.
- Recomputes from the active Startup Intelligence filters and displays the top 24 sufficiently populated sectors after sorting by exit rate.

Why it is chosen:

- It separates sector popularity from sector outcome rate.
- Bubble size adds capital intensity as a third dimension.
- It helps distinguish crowded sectors from sectors with stronger observed exit outcomes.

Caveats:

- "Risk / reward" here is a descriptive proxy, not a full risk model.
- Sectors below the filtered company-count threshold are excluded.
- Historical sector codes may not map cleanly to modern startup categories.

### Geography Ecosystem View

Implementation:

- Plot type: bubble scatter plot.
- X-axis: company count, log-scaled.
- Y-axis: exit rate percentage.
- Bubble size: median funding in millions.
- Color and text label: `country_code`.
- Recomputes from the active Startup Intelligence filters and displays the top 28 sufficiently populated countries after sorting by exit rate.

Why it is chosen:

- It compares startup ecosystems by scale, observed outcome rate, and capital intensity.
- Log scaling helps large and smaller ecosystems fit in the same view.
- Country labels make the chart scan-friendly without requiring every point to be hovered.

Caveats:

- Country-level aggregation can hide city-level and sector-level differences.
- The chart should not imply geography alone causes outcomes.
- Countries below the filtered company-count threshold are excluded.

## ML Exit Predictor Dashboard

The ML Exit Predictor page is designed to answer: **How does the historical exit model score a user-defined scenario?**

It uses the trained RandomForest pipeline from `data_pipeline.py` and a page-local control grid for sector, country, founding year, funding, founder, education, participant, and round-history values.

### Scenario Prediction

Current cards and charts:

- **Predicted Exit Probability**: live `predict_proba` output for the selected feature values.
- **Dataset Median** and **Upper Quartile**: benchmark the scenario against existing company predictions.
- **Prediction Distribution**: histogram of existing model scores with the scenario marked.
- **Top Model Signals**: global permutation importance for the trained model.
- **Scenario Feature Values**: table showing the exact values sent into the model.

Why these are chosen:

- They separate scenario scoring from the aggregate Startup Intelligence page.
- They keep the model auditable by showing both global signal importance and the exact input row.
- The distribution chart prevents users from over-reading a probability without seeing its dataset context.

### Suggested Startup Intelligence Additions

1. **Calibration / Reliability Chart**

   Bucket predictions into deciles and compare average predicted probability with observed exit rate.

   Why: ROC-AUC shows ranking ability, but calibration shows whether predicted probabilities are numerically trustworthy.

2. **Confusion Matrix at Threshold**

   Show true positives, false positives, true negatives, and false negatives at the current probability threshold.

   Why: this would make the minimum probability slider more meaningful and reveal tradeoffs between recall and precision.

3. **Funding Timeline / Age-to-Exit View**

   Compare founding year, first funding year, last funding year, and exit status by cohort or sector.

   Why: the model has time and funding-span features, but the current charts do not directly show timing dynamics.

4. **Sector x Country Outcome Matrix**

   Show exit rate or company count by sector and country for sufficiently large cells.

   Why: sector and geography are currently separate. A combined matrix would reveal ecosystem specializations, such as sectors that perform differently by country.

## Cross-Dashboard Suggestions

1. **Methodology Notes Inside the UI**

   Add collapsible notes explaining `radar_score`, `success_exit`, and `predicted_exit_probability`.

   Why: both dashboards use derived signals. Short in-app methodology notes would reduce the chance that users overinterpret scores.

2. **Exportable Chart/Table State**

   Add CSV export for filtered tables and possibly chart data.

   Why: the dashboard is an investor research tool, and users will likely want to take filtered opportunity lists into follow-up workflows.

3. **Small Multiples for Filtered Comparison**

   Add compact views that compare selected industries, regions, or sectors across the same metric.

   Why: many current charts are global summaries. Small multiples would help compare categories without changing filters repeatedly.

4. **Sample Size Indicators**

   Add visible sample counts on charts where rates are shown.

   Why: charts using exit rates or average radar scores can be misleading when category sample sizes are small.
