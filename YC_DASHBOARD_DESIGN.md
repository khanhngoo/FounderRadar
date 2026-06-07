# YC Ecosystem Dashboard Design

## Dataset Source

This page uses only the three YC-focused Kaggle datasets downloaded by `scripts/fetch_data.py`:

- `miguelcorraljr/y-combinator-directory`
- `sashakorovkina/ycombinator-all-funded-companies-dataset`
- `lazarun/y-combinator-jobs-enriched`

The dashboard intentionally blocks when these datasets are missing. The previous `yc-oss` fallback is useful for quick smoke tests, but it does not satisfy the product requirement for a three-dataset YC page and can make the dashboard look more complete than the local data really is.

## Theme Rationale

The page uses an off-white workspace, graphite text, compact metric cards, and YC orange as the main accent. This keeps the page close to YC's visual identity while still feeling like an analytical product rather than a marketing page.

Secondary colors are teal, blue, and rose. They prevent the interface from becoming a one-note orange dashboard and make category differences easier to scan.

## Visualization Choices

- **YC Batch Pulse Streamgraph** shows company count over batch years by top industries. This is better than a simple yearly bar chart because it reveals how YC's sector mix shifts over time.
- **Opportunity Quadrant Map** plots recognition against momentum, sized by team size. The intention is to surface companies that look active but less recognized, which matches FounderRadar's opportunity-discovery thesis.
- **Industry x Region Matrix** compares average radar score across sector and geography. A matrix is used because it makes concentration patterns easier to scan than a long table.
- **Company Explorer Table** keeps the dashboard actionable. After spotting a pattern, users need to inspect individual companies quickly.

The Batch Pulse Streamgraph and Industry x Region Matrix exclude unspecified or unknown category buckets so missing labels do not dominate the visual story.

## Design Intention

This page is meant to answer: "Where is YC momentum hiding?" The charts prioritize batch dynamics, sector geography, hiring activity, and an inspectable company list over generic counts.
