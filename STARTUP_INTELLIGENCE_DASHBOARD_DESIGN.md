# Startup Intelligence Dashboard Design

## Dataset Source

This page uses the historical Kaggle Startup Investments dataset:

- `justinas/startup-investments`

It reproduces the notebook's historical analysis layer only. SEC EDGAR Form D data is intentionally excluded from this dashboard based on project scope.

## Notebook-Derived Signals

The page builds a company-level modeling table from:

- `objects.csv` for company metadata, funding totals, status, geography, and milestones
- `acquisitions.csv` and `ipos.csv` for the `success_exit` proxy
- `relationships.csv` for founder-role counts and prior-company exposure
- `degrees.csv` for founder education signals
- `funding_rounds.csv` for funding maturity and round-participant signals

`success_exit` is an acquisition/IPO proxy. It does not represent every kind of startup success, so model outputs should be read as exploratory ranking signals rather than investment advice.

## Theme Rationale

The Startup Intelligence page now shares the YC ecosystem dashboard's off-white workspace, graphite text, compact metric cards, and YC-orange accent. Teal, blue, rose, and orange remain signal colors for founder/team, market/geography, company metadata, and funding maturity.

## Visualization Choices

- **Model Signal Importance Bar** explains which feature groups move the exit model. It makes the ML model less opaque by grouping raw features into investor-readable themes.
- **Funding Maturity Ladder** shows exit rate by funding-round stage, with median funding encoded by color. This communicates maturity as a progression rather than isolated categories.
- **Founder Signal Summary** compares team structure, education coverage, and funding maturity on the same outcome axis. This is more compact and defensible than a complex founder flow diagram.
- **Sector Risk / Reward Bubble Plot** separates sector scale from exit outcomes and median funding. It helps users distinguish popular sectors from high-outcome sectors.
- **Geography Ecosystem View** compares country-level company count, exit rate, and median funding. It supports ecosystem comparison without implying geography alone causes outcomes.

The aggregate Startup Intelligence charts respond to the page-local sector, country, probability, and search filters. Company-level prediction exploration has moved to the separate ML Exit Predictor page.

## ML Exit Predictor Page

The ML Exit Predictor page exposes the trained RandomForest exit model through scenario inputs. Users choose company, founder, funding, round, and geography feature values, then inspect:

- live predicted exit probability
- dataset median and upper-quartile prediction benchmarks
- prediction distribution with the scenario marked
- top model signal importance
- a table of the active scenario feature values

## Design Intention

This page is meant to answer: "Which historical signals are associated with observable startup exits?" It focuses on interpretable filtered aggregates and model transparency. Scenario-level model inspection is handled by the ML Exit Predictor page.
