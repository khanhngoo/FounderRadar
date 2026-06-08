from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget


APP_DIR = Path(__file__).parent

SECTOR_COLORS = {
    "AI / ML": "#2563EB",
    "Fintech": "#14B8A6",
    "Healthcare": "#7C3AED",
    "Developer Tools": "#F59E0B",
    "Climate": "#16A34A",
    "Consumer": "#DC2626",
}

REGION_POINTS = {
    "United States": (-98, 39),
    "Europe": (12, 50),
    "India": (78, 22),
    "Canada": (-106, 56),
    "Latin America": (-60, -15),
    "Southeast Asia": (105, 12),
}

SECTOR_THEMES = {
    "AI / ML": ["Agentic AI", "Computer Vision", "LLM Ops", "Data Infrastructure"],
    "Fintech": ["Payments", "Risk Scoring", "B2B Finance", "Crypto Infrastructure"],
    "Healthcare": ["Clinical AI", "Care Navigation", "Diagnostics", "Digital Therapeutics"],
    "Developer Tools": ["DevOps", "Security", "API Tooling", "Data Engineering"],
    "Climate": ["Carbon Accounting", "Grid Software", "Climate Risk", "Clean Materials"],
    "Consumer": ["Creator Tools", "Marketplaces", "Consumer Social", "Productivity"],
}


def weighted_pick(items: list[str], weights: list[int], index: int) -> str:
    total = sum(weights)
    seed = ((index * 9301 + 49297) % 233280) / 233280
    cursor = seed * total

    for item, weight in zip(items, weights):
        cursor -= weight
        if cursor <= 0:
            return item

    return items[-1]


def make_sample_data() -> pd.DataFrame:
    sectors = ["AI / ML", "Fintech", "Healthcare", "Developer Tools", "Climate", "Consumer"]
    regions = ["United States", "Europe", "India", "Canada", "Latin America", "Southeast Asia"]
    schools = ["Stanford", "MIT", "UC Berkeley", "Harvard", "CMU", "Oxford", "Waterloo", "IIT Bombay"]
    companies = ["Google", "Meta", "Amazon", "Microsoft", "Stripe", "Goldman Sachs", "Apple", "McKinsey"]
    statuses = ["Active", "Acquired", "IPO", "Closed"]
    batches = ["W20", "S20", "W21", "S21", "W22", "S22", "W23", "S23", "W24", "S24"]

    founder_names = [
        "Alex Chen",
        "Priya Shah",
        "David Kim",
        "Sara Liu",
        "Miguel Santos",
        "Nina Patel",
        "John Smith",
        "Minh Nguyen",
        "Elena Rossi",
        "Arjun Mehta",
        "Maya Johnson",
        "Hannah Lee",
        "Tom Becker",
        "Aisha Khan",
        "Kenji Tanaka",
        "Linh Tran",
        "Ravi Menon",
        "Julia Evans",
    ]

    startup_prefixes = ["Apex", "Fin", "Infra", "Medi", "Clima", "Lumen", "Rev", "Nexa", "Trust", "Flow"]
    startup_suffixes = ["AI", "lytics", "Base", "Bridge", "Grid", "Stack", "Ops", "Loop", "Cloud", "Labs"]

    rows = []

    for i in range(180):
        sector = weighted_pick(sectors, [28, 20, 17, 18, 10, 12], i + 4)
        region = weighted_pick(regions, [48, 18, 12, 9, 6, 7], i + 9)
        status = weighted_pick(statuses, [70, 17, 5, 8], i + 15)
        school = weighted_pick(schools, [23, 18, 17, 12, 10, 8, 6, 6], i + 22)
        prior_company = weighted_pick(companies, [22, 16, 14, 14, 10, 9, 8, 7], i + 31)

        theme = SECTOR_THEMES[sector][(i + len(sector)) % len(SECTOR_THEMES[sector])]

        lon, lat = REGION_POINTS[region]
        lon = lon + ((i % 9) - 4) * 1.8
        lat = lat + (((i * 3) % 7) - 3) * 1.2

        funding = round(
            1.8 + ((i * 17) % 140) * 0.85 + (8 if status in ["Acquired", "IPO"] else 0),
            1,
        )

        rows.append(
            {
                "Startup": f"{startup_prefixes[i % len(startup_prefixes)]}{startup_suffixes[(i * 3) % len(startup_suffixes)]}",
                "Founder": founder_names[i % len(founder_names)],
                "Sector": sector,
                "Theme": theme,
                "Region": region,
                "Batch": batches[i % len(batches)],
                "Founded Year": 2020 + (i % 6),
                "School": school,
                "Prior Company": prior_company,
                "Outcome": status,
                "Funding M": funding,
                "Longitude": lon,
                "Latitude": lat,
                "Founder Count": 1 + ((i * 7) % 4),
            }
        )

    return pd.DataFrame(rows)


DATA = make_sample_data()


def polish_fig(fig: go.Figure, height: int | None = None, legend: bool = False) -> go.Figure:
    fig.update_layout(
        margin=dict(l=10, r=10, t=8, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color="#172033"),
        autosize=True,
        showlegend=legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0)",
        ),
    )
    if height is not None:
        fig.update_layout(height=height)
    return fig


KPI_ICONS = {
    "startups": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <rect x="4" y="4" width="16" height="16" rx="2" fill="none" stroke="currentColor" stroke-width="2"/>
          <path d="M8 4v16M12 4v16M16 4v16M4 8h16M4 12h16M4 16h16" stroke="currentColor" stroke-width="1.5"/>
        </svg>
    """,
    "founders": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <circle cx="12" cy="12" r="8" fill="none" stroke="currentColor" stroke-width="2.4"/>
          <circle cx="12" cy="12" r="3" fill="none" stroke="currentColor" stroke-width="2.2"/>
        </svg>
    """,
    "schools": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M12 3l9 7-9 7-9-7 9-7z" fill="none" stroke="currentColor" stroke-width="2"/>
          <path d="M12 8l4 3-4 3-4-3 4-3z" fill="none" stroke="currentColor" stroke-width="1.8"/>
        </svg>
    """,
    "companies": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <rect x="5" y="6" width="14" height="14" rx="2" fill="none" stroke="currentColor" stroke-width="2"/>
          <rect x="9" y="10" width="6" height="6" fill="none" stroke="currentColor" stroke-width="2"/>
        </svg>
    """,
    "regions": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <circle cx="12" cy="12" r="8" fill="none" stroke="currentColor" stroke-width="2"/>
          <path d="M12 4c2.2 2.2 3.2 5 3.2 8s-1 5.8-3.2 8M12 4c-2.2 2.2-3.2 5-3.2 8s1 5.8 3.2 8M4 12h16" fill="none" stroke="currentColor" stroke-width="1.6"/>
        </svg>
    """,
}


def kpi_card(label: str, output_id: str, caption: str, icon_key: str) -> ui.Tag:
    return ui.tags.article(
        {"class": "kpi-card"},
        ui.tags.div(ui.HTML(KPI_ICONS[icon_key]), class_="kpi-symbol"),
        ui.tags.div(
            ui.tags.span(label),
            ui.tags.strong(ui.output_text(output_id)),
            ui.tags.small(caption),
        ),
    )


def chart_panel(
    title: str,
    subtitle: str,
    output_id: str,
    wide: bool = False,
    compact: bool = False,
    section_id: str | None = None,
) -> ui.Tag:
    classes = "panel"

    if wide:
        classes += " wide-panel"

    if compact:
        classes += " compact-panel"

    return ui.tags.article(
        {"class": classes, **({"id": section_id} if section_id else {})},
        ui.tags.div(
            ui.tags.div(
                ui.tags.h2(title),
                ui.tags.p(subtitle),
            ),
            class_="panel-header",
        ),
        ui.tags.div(output_widget(output_id), class_="chart-widget"),
    )


def page_header(title: str, subtitle: str) -> ui.Tag:
    return ui.tags.header(
        ui.tags.div(
            ui.tags.h1(title),
            ui.tags.p(subtitle),
        ),
        class_="topbar",
    )


def kpi_section() -> ui.Tag:
    return ui.tags.section(
        kpi_card("Total startups", "kpi_startups", "Filtered companies", "startups"),
        kpi_card("Total founders", "kpi_founders", "Estimated people", "founders"),
        kpi_card("Schools", "kpi_schools", "Contributing schools", "schools"),
        kpi_card("Prior companies", "kpi_companies", "Previous employers", "companies"),
        kpi_card("Regions", "kpi_regions", "Startup locations", "regions"),
        class_="kpi-grid",
    )


def insights_panel() -> ui.Tag:
    return ui.tags.article(
        {"class": "panel insight-row-panel"},
        ui.tags.div(
            ui.tags.div(
                ui.tags.h2("Ecosystem Insights"),
                ui.tags.p("Auto-updated summary."),
            ),
            class_="panel-header",
        ),
        ui.output_ui("insights"),
    )


def dashboard_page() -> ui.Tag:
    return ui.tags.div(
        page_header(
            "Startup Ecosystem Overview",
            "High-level founder and sector signals for the filtered YC-backed startup sample.",
        ),
        kpi_section(),
        ui.tags.section(
            chart_panel("Top Founder Schools", "Ranked by founder count.", "school_chart"),
            chart_panel("Sector Trends Across Batches", "Share of startups by YC batch.", "trend_chart", wide=True),
            chart_panel("Prior Company Experience", "Previous employers feeding into YC startups.", "company_chart"),
            insights_panel(),
            class_="dashboard-grid",
        ),
    )


def map_page() -> ui.Tag:
    return ui.tags.div(
        page_header(
            "Global Ecosystem Map",
            "A dedicated spatial view with standard Plotly map interaction.",
        ),
        ui.tags.section(
            chart_panel(
                "Global Founder / Startup Concentration",
                "Bubble size shows filtered startup density by region. Drag and zoom to explore.",
                "map_chart",
                wide=True,
            ),
            class_="dashboard-grid map-page-grid",
        ),
    )


def pipeline_page() -> ui.Tag:
    return ui.tags.div(
        page_header(
            "Founder Pipeline",
            "Trace founder movement from school and prior employer into startup sector and outcome.",
        ),
        ui.tags.section(
            chart_panel(
                "Founder Pipeline",
                "School -> prior company -> sector -> outcome.",
                "pipeline_chart",
                wide=True,
            ),
            chart_panel(
                "Founder Network",
                "Schools, prior companies, founders, and startups as linked nodes.",
                "network_chart",
                wide=True,
            ),
            class_="dashboard-grid pipeline-page-grid",
        ),
    )


def compare_page() -> ui.Tag:
    return ui.tags.div(
        page_header(
            "Compare Sectors & Batches",
            "Compare sector landscapes, funding movement, and batch-level trends.",
        ),
        ui.tags.section(
            chart_panel(
                "Market Map / Sector Landscape",
                "Treemap grouped by sector and startup theme.",
                "market_map_chart",
                wide=True,
            ),
            chart_panel(
                "Animated Batch Timeline",
                "Funding and sector movement across YC batches.",
                "timeline_chart",
                wide=True,
            ),
            class_="dashboard-grid compare-page-grid",
        ),
    )


def explorer_page() -> ui.Tag:
    return ui.tags.div(
        page_header(
            "Startup / Founder Explorer",
            "Query, inspect, and export individual startup and founder records.",
        ),
        ui.tags.section(
            ui.input_text("table_search", "Table search", placeholder="Founder, startup, theme..."),
            ui.input_select("school_query", "School", ["All"] + sorted(DATA["School"].unique().tolist())),
            ui.input_select("company_query", "Prior company", ["All"] + sorted(DATA["Prior Company"].unique().tolist())),
            ui.input_numeric("min_funding", "Min funding ($M)", value=0, min=0, step=5),
            class_="query-grid",
        ),
        ui.tags.section(
            ui.tags.div(
                ui.tags.div(
                    ui.tags.h2("Startup / Founder Explorer"),
                    ui.tags.p(ui.output_text("result_count")),
                ),
                ui.download_button("download_csv", "Export CSV", class_="secondary-button"),
                class_="explorer-header",
            ),
            ui.output_data_frame("records_table"),
            class_="explorer-panel",
        ),
    )


app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.title("FounderRadar Shiny Dashboard"),
        ui.tags.link(rel="stylesheet", href="styles.css"),
        ui.tags.script(
            """
            document.addEventListener('click', function(event) {
              const item = event.target.closest('.nav-item');
              if (!item) return;
              document.querySelectorAll('.nav-item').forEach((el) => el.classList.remove('active'));
              item.classList.add('active');
              window.scrollTo({ top: 0, behavior: 'smooth' });
            });

            """
        ),
    ),
    ui.tags.div(
        {"class": "app-shell"},
        ui.tags.aside(
            {"class": "sidebar"},
            ui.tags.div(
                ui.tags.div("FounderRadar", class_="brand"),
                ui.tags.div("VC intelligence dashboard", class_="brand-subtitle"),
                class_="brand-block",
            ),
            ui.tags.nav(
                ui.input_action_button("nav_dashboard", "Dashboard", class_="nav-item active"),
                ui.input_action_button("nav_map", "Global Map", class_="nav-item"),
                ui.input_action_button("nav_explorer", "Explorer", class_="nav-item"),
                ui.input_action_button("nav_pipeline", "Pipeline", class_="nav-item"),
                ui.input_action_button("nav_compare", "Compare", class_="nav-item"),
                class_="nav-list",
            ),
            ui.tags.section(
                ui.tags.div("Filters", class_="filter-title"),
                ui.input_select("batch", "Batch", ["All"] + sorted(DATA["Batch"].unique().tolist())),
                ui.input_select("sector", "Sector", ["All"] + list(SECTOR_COLORS.keys())),
                ui.input_select("region", "Region", ["All"] + sorted(DATA["Region"].unique().tolist())),
                ui.input_select("outcome", "Outcome", ["All"] + sorted(DATA["Outcome"].unique().tolist())),
                ui.input_text("search", "Search", placeholder="Founder, startup, school..."),
                ui.input_action_button("reset", "Reset filters", class_="primary-button"),
                class_="filter-block",
            ),
        ),
        ui.tags.main(
            {"class": "main"},
            ui.navset_hidden(
                ui.nav_panel("Dashboard", dashboard_page(), value="dashboard"),
                ui.nav_panel("Global Map", map_page(), value="map"),
                ui.nav_panel("Explorer", explorer_page(), value="explorer"),
                ui.nav_panel("Pipeline", pipeline_page(), value="pipeline"),
                ui.nav_panel("Compare", compare_page(), value="compare"),
                id="main_nav",
                selected="dashboard",
            ),
        ),
    ),
)

def server(input, output, session):
    def switch_page(page: str):
        ui.update_navs("main_nav", selected=page)

    @reactive.effect
    @reactive.event(input.nav_dashboard)
    def _nav_dashboard():
        switch_page("dashboard")

    @reactive.effect
    @reactive.event(input.nav_map)
    def _nav_map():
        switch_page("map")

    @reactive.effect
    @reactive.event(input.nav_explorer)
    def _nav_explorer():
        switch_page("explorer")

    @reactive.effect
    @reactive.event(input.nav_pipeline)
    def _nav_pipeline():
        switch_page("pipeline")

    @reactive.effect
    @reactive.event(input.nav_compare)
    def _nav_compare():
        switch_page("compare")

    @reactive.effect
    @reactive.event(input.reset)
    def _reset_filters():
        ui.update_select("batch", selected="All")
        ui.update_select("sector", selected="All")
        ui.update_select("region", selected="All")
        ui.update_select("outcome", selected="All")
        ui.update_text("search", value="")
        ui.update_text("table_search", value="")
        ui.update_select("school_query", selected="All")
        ui.update_select("company_query", selected="All")
        ui.update_numeric("min_funding", value=0)

    @reactive.calc
    def filtered_data() -> pd.DataFrame:
        df = DATA.copy()

        if input.batch() != "All":
            df = df[df["Batch"] == input.batch()]

        if input.sector() != "All":
            df = df[df["Sector"] == input.sector()]

        if input.region() != "All":
            df = df[df["Region"] == input.region()]

        if input.outcome() != "All":
            df = df[df["Outcome"] == input.outcome()]

        query = input.search().strip().lower()

        if query:
            haystack = (
                df[["Startup", "Founder", "School", "Prior Company", "Sector", "Region"]]
                .astype(str)
                .agg(" ".join, axis=1)
                .str.lower()
            )
            df = df[haystack.str.contains(query, regex=False)]

        return df

    @reactive.calc
    def explorer_data() -> pd.DataFrame:
        df = filtered_data().copy()

        table_query = input.table_search().strip().lower()
        if table_query:
            haystack = (
                df[["Startup", "Founder", "Theme", "School", "Prior Company", "Sector", "Region"]]
                .astype(str)
                .agg(" ".join, axis=1)
                .str.lower()
            )
            df = df[haystack.str.contains(table_query, regex=False)]

        if input.school_query() != "All":
            df = df[df["School"] == input.school_query()]

        if input.company_query() != "All":
            df = df[df["Prior Company"] == input.company_query()]

        if input.min_funding() is not None:
            df = df[df["Funding M"] >= float(input.min_funding())]

        return df

    @output
    @render.text
    def kpi_startups():
        return f"{filtered_data()['Startup'].nunique():,}"

    @output
    @render.text
    def kpi_founders():
        return f"{int(filtered_data()['Founder Count'].sum()):,}"

    @output
    @render.text
    def kpi_schools():
        return f"{filtered_data()['School'].nunique():,}"

    @output
    @render.text
    def kpi_companies():
        return f"{filtered_data()['Prior Company'].nunique():,}"

    @output
    @render.text
    def kpi_regions():
        return f"{filtered_data()['Region'].nunique():,}"

    @output
    @render.text
    def result_count():
        return f"Showing {len(explorer_data()):,} queried records"

    @output
    @render_widget
    def school_chart():
        df = filtered_data()

        if df.empty:
            return polish_fig(go.Figure())

        counts = df["School"].value_counts().head(8).sort_values()

        fig = go.Figure(
            go.Bar(
                x=counts.values,
                y=counts.index,
                orientation="h",
                marker_color="#2563EB",
            )
        )

        fig.update_xaxes(showgrid=True, gridcolor="#E6EAF2", title="")
        fig.update_yaxes(title="")

        return polish_fig(fig)

    @output
    @render_widget
    def company_chart():
        df = filtered_data()

        if df.empty:
            return polish_fig(go.Figure())

        counts = df["Prior Company"].value_counts().head(8).sort_values()

        fig = go.Figure(
            go.Bar(
                x=counts.values,
                y=counts.index,
                orientation="h",
                marker_color="#14B8A6",
            )
        )

        fig.update_xaxes(showgrid=True, gridcolor="#E6EAF2", title="")
        fig.update_yaxes(title="")

        return polish_fig(fig)

    @output
    @render_widget
    def trend_chart():
        df = filtered_data()

        if df.empty:
            return polish_fig(go.Figure())

        grouped = df.groupby(["Batch", "Sector"]).size().reset_index(name="Count")
        totals = df.groupby("Batch").size().reset_index(name="Total")
        chart = grouped.merge(totals, on="Batch", how="left")
        chart["Share"] = chart["Count"] / chart["Total"]

        fig = go.Figure()

        for sector, color in SECTOR_COLORS.items():
            sector_df = chart[chart["Sector"] == sector]

            fig.add_trace(
                go.Scatter(
                    x=sector_df["Batch"],
                    y=sector_df["Share"],
                    mode="lines",
                    stackgroup="one",
                    name=sector,
                    line=dict(width=0.5, color=color),
                    hovertemplate="%{x}<br>%{y:.0%}<extra>" + sector + "</extra>",
                )
            )

        fig.update_yaxes(tickformat=".0%", range=[0, 1], showgrid=True, gridcolor="#E6EAF2")
        fig.update_xaxes(categoryorder="array", categoryarray=sorted(DATA["Batch"].unique()))

        return polish_fig(fig, legend=True)

    @output
    @render_widget
    def map_chart():
        df = filtered_data()

        if df.empty:
            return polish_fig(go.Figure())

        counts = df["Region"].value_counts().reset_index()
        counts.columns = ["Region", "Count"]
        counts["lon"] = counts["Region"].map(lambda r: REGION_POINTS[r][0])
        counts["lat"] = counts["Region"].map(lambda r: REGION_POINTS[r][1])

        max_count = counts["Count"].max()

        fig = go.Figure(
            go.Scattergeo(
                lon=counts["lon"],
                lat=counts["lat"],
                text=counts["Region"] + ": " + counts["Count"].astype(str),
                mode="markers+text",
                textposition="top center",
                marker=dict(
                    size=counts["Count"] / max_count * 34 + 8,
                    color="#2563EB",
                    opacity=0.72,
                    line=dict(color="#FFFFFF", width=1),
                ),
            )
        )

        fig.update_geos(
            projection_type="natural earth",
            lataxis=dict(range=[-58, 82]),
            lonaxis=dict(range=[-180, 180]),
            projection_rotation=dict(lon=0, lat=0, roll=0),
            center=dict(lat=15, lon=0),
            showcountries=True,
            countrycolor="#D9DEE8",
            showland=True,
            landcolor="#E6EAF2",
            showocean=True,
            oceancolor="#F5F7FB",
            bgcolor="rgba(0,0,0,0)",
        )
        fig.update_layout(
            dragmode="pan",
            uirevision="interactive-global-map",
        )

        return polish_fig(fig)

    @output
    @render_widget
    def pipeline_chart():
        df = filtered_data()

        if df.empty:
            return polish_fig(go.Figure())

        top_schools = df["School"].value_counts().head(4).index.tolist()
        top_companies = df["Prior Company"].value_counts().head(4).index.tolist()
        top_sectors = df["Sector"].value_counts().head(4).index.tolist()
        top_outcomes = df["Outcome"].value_counts().head(3).index.tolist()

        reduced = df.assign(
            School=lambda x: x["School"].where(x["School"].isin(top_schools), "Other schools"),
            **{
                "Prior Company": lambda x: x["Prior Company"].where(
                    x["Prior Company"].isin(top_companies),
                    "Other companies",
                )
            },
            Sector=lambda x: x["Sector"].where(x["Sector"].isin(top_sectors), "Other sectors"),
            Outcome=lambda x: x["Outcome"].where(x["Outcome"].isin(top_outcomes), "Other outcomes"),
        )

        stages = ["School", "Prior Company", "Sector", "Outcome"]

        labels = []
        for stage in stages:
            labels.extend(reduced[stage].drop_duplicates().tolist())

        label_index = {label: i for i, label in enumerate(labels)}

        sources = []
        targets = []
        values = []

        for left, right in zip(stages, stages[1:]):
            flows = reduced.groupby([left, right]).size().reset_index(name="Count")

            for _, row in flows.iterrows():
                sources.append(label_index[row[left]])
                targets.append(label_index[row[right]])
                values.append(int(row["Count"]))

        fig = go.Figure(
            go.Sankey(
                node=dict(
                    label=labels,
                    pad=14,
                    thickness=14,
                    color="#EAF2FF",
                    line=dict(color="#D9DEE8", width=1),
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color="rgba(37,99,235,0.18)",
                ),
            )
        )

        return polish_fig(fig)

    @output
    @render_widget
    def market_map_chart():
        df = filtered_data()

        if df.empty:
            return polish_fig(go.Figure())

        chart = (
            df.groupby(["Sector", "Theme"])
            .agg(
                Startups=("Startup", "count"),
                Funding=("Funding M", "sum"),
            )
            .reset_index()
        )
        sector_totals = chart.groupby("Sector")["Funding"].sum().reset_index()
        labels = sector_totals["Sector"].tolist() + chart["Theme"].tolist()
        parents = [""] * len(sector_totals) + chart["Sector"].tolist()
        values = sector_totals["Funding"].tolist() + chart["Funding"].tolist()
        colors = [SECTOR_COLORS.get(s, "#2563EB") for s in sector_totals["Sector"]] + [
            SECTOR_COLORS.get(s, "#2563EB") for s in chart["Sector"]
        ]

        fig = go.Figure(
            go.Treemap(
                labels=labels,
                parents=parents,
                values=values,
                branchvalues="total",
                marker=dict(colors=colors),
                texttemplate="<b>%{label}</b><br>$%{value:.0f}M",
                hovertemplate="%{parent}<br>%{label}<br>Funding: $%{value:.1f}M<extra></extra>",
            )
        )

        return polish_fig(fig)

    @output
    @render_widget
    def network_chart():
        df = filtered_data().head(26)

        if df.empty:
            return polish_fig(go.Figure())

        node_order = []
        edges = []

        for _, row in df.iterrows():
            chain = [row["School"], row["Prior Company"], row["Founder"], row["Startup"]]

            for node in chain:
                if node not in node_order:
                    node_order.append(node)

            edges.extend(
                [
                    (chain[0], chain[1]),
                    (chain[1], chain[2]),
                    (chain[2], chain[3]),
                ]
            )

        node_type = {}

        for node in node_order:
            if node in DATA["School"].unique():
                node_type[node] = "School"
            elif node in DATA["Prior Company"].unique():
                node_type[node] = "Company"
            elif node in DATA["Founder"].unique():
                node_type[node] = "Founder"
            else:
                node_type[node] = "Startup"

        levels = {
            "School": 0,
            "Company": 1,
            "Founder": 2,
            "Startup": 3,
        }

        level_counts = {level: 0 for level in levels}
        positions = {}

        for node in node_order:
            level = node_type[node]
            idx = level_counts[level]
            level_counts[level] += 1
            positions[node] = (levels[level], idx)

        coords = {}

        for node, (level, idx) in positions.items():
            x = level
            y = 1 - ((idx + 1) / (level_counts[node_type[node]] + 1))
            coords[node] = (x, y)

        edge_x = []
        edge_y = []

        for left, right in edges:
            x0, y0 = coords[left]
            x1, y1 = coords[right]

            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        color_by_type = {
            "School": "#2563EB",
            "Company": "#14B8A6",
            "Founder": "#F59E0B",
            "Startup": "#7C3AED",
        }

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=1, color="rgba(93,102,122,0.28)"),
                hoverinfo="skip",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[coords[n][0] for n in node_order],
                y=[coords[n][1] for n in node_order],
                mode="markers+text",
                text=[n if len(n) <= 13 else n[:12] + "..." for n in node_order],
                textposition="top center",
                marker=dict(
                    size=[16 if node_type[n] in ["Founder", "Startup"] else 13 for n in node_order],
                    color=[color_by_type[node_type[n]] for n in node_order],
                    line=dict(color="#FFFFFF", width=1),
                ),
                hovertext=[f"{node_type[n]}: {n}" for n in node_order],
                hoverinfo="text",
            )
        )

        fig.update_xaxes(
            visible=False,
            range=[-0.2, 3.2],
        )
        fig.update_yaxes(visible=False, range=[-0.1, 1.1])

        return polish_fig(fig)

    @output
    @render_widget
    def globe_chart():
        df = filtered_data()

        if df.empty:
            return polish_fig(go.Figure())

        counts = (
            df.groupby(["Region", "Longitude", "Latitude"])
            .size()
            .reset_index(name="Count")
        )

        counts = counts.sample(min(len(counts), 80), random_state=4)

        fig = go.Figure(
            go.Scattergeo(
                lon=counts["Longitude"],
                lat=counts["Latitude"],
                text=counts["Region"] + "<br>Records: " + counts["Count"].astype(str),
                mode="markers",
                marker=dict(
                    size=counts["Count"] / counts["Count"].max() * 18 + 5,
                    color="#14B8A6",
                    opacity=0.72,
                    line=dict(color="#FFFFFF", width=0.6),
                ),
                hovertemplate="%{text}<extra></extra>",
            )
        )

        fig.update_geos(
            projection_type="orthographic",
            projection_rotation=dict(lon=-25, lat=18),
            showcountries=True,
            countrycolor="#D9DEE8",
            showland=True,
            landcolor="#E6EAF2",
            showocean=True,
            oceancolor="#F5F7FB",
            bgcolor="rgba(0,0,0,0)",
        )

        return polish_fig(fig)

    @output
    @render_widget
    def timeline_chart():
        df = filtered_data()

        if df.empty:
            return polish_fig(go.Figure())

        chart = (
            df.groupby(["Batch", "Sector"])
            .agg(
                Startups=("Startup", "count"),
                Funding=("Funding M", "sum"),
                Founder_Count=("Founder Count", "sum"),
            )
            .reset_index()
        )

        batch_order = sorted(DATA["Batch"].unique())
        first_batch = batch_order[0]

        frames = []

        for batch in batch_order:
            batch_df = chart[chart["Batch"] == batch]

            frames.append(
                go.Frame(
                    name=batch,
                    data=[
                        go.Scatter(
                            x=batch_df["Startups"],
                            y=batch_df["Funding"],
                            mode="markers+text",
                            text=batch_df["Sector"],
                            textposition="top center",
                            marker=dict(
                                size=batch_df["Founder_Count"] / max(chart["Founder_Count"].max(), 1) * 36 + 10,
                                color=[SECTOR_COLORS.get(s, "#2563EB") for s in batch_df["Sector"]],
                                opacity=0.78,
                                line=dict(color="#FFFFFF", width=1),
                            ),
                            hovertemplate="%{text}<br>Startups: %{x}<br>Funding: $%{y:.1f}M<extra></extra>",
                        )
                    ],
                )
            )

        start_df = chart[chart["Batch"] == first_batch]

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=start_df["Startups"],
                    y=start_df["Funding"],
                    mode="markers+text",
                    text=start_df["Sector"],
                    textposition="top center",
                    marker=dict(
                        size=start_df["Founder_Count"] / max(chart["Founder_Count"].max(), 1) * 36 + 10,
                        color=[SECTOR_COLORS.get(s, "#2563EB") for s in start_df["Sector"]],
                        opacity=0.78,
                        line=dict(color="#FFFFFF", width=1),
                    ),
                    hovertemplate="%{text}<br>Startups: %{x}<br>Funding: $%{y:.1f}M<extra></extra>",
                )
            ],
            frames=frames,
        )

        fig.update_layout(
            xaxis_title="Startup count",
            yaxis_title="Mock funding ($M)",
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "x": 0,
                    "y": 1.18,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 700, "redraw": True},
                                    "fromcurrent": True,
                                },
                            ],
                        }
                    ],
                }
            ],
            sliders=[
                {
                    "steps": [
                        {
                            "args": [
                                [batch],
                                {
                                    "frame": {"duration": 250, "redraw": True},
                                    "mode": "immediate",
                                },
                            ],
                            "label": batch,
                            "method": "animate",
                        }
                        for batch in batch_order
                    ],
                    "x": 0,
                    "y": -0.02,
                    "len": 0.95,
                }
            ],
        )

        fig.update_xaxes(showgrid=True, gridcolor="#E6EAF2")
        fig.update_yaxes(showgrid=True, gridcolor="#E6EAF2")

        return polish_fig(fig)

    @output
    @render.ui
    def insights():
        df = filtered_data()

        if df.empty:
            return ui.tags.ul(
                ui.tags.li("No records match the current filters."),
                class_="insight-list",
            )

        top_sector = df["Sector"].value_counts().idxmax()
        top_school = df["School"].value_counts().idxmax()
        top_region = df["Region"].value_counts().idxmax()
        exit_rate = round(df["Outcome"].isin(["Acquired", "IPO"]).mean() * 100)

        return ui.tags.ul(
            ui.tags.li(
                ui.tags.strong(top_sector),
                ui.tags.span(" is the largest filtered sector."),
            ),
            ui.tags.li(
                ui.tags.strong(top_school),
                ui.tags.span(" is the most common school signal."),
            ),
            ui.tags.li(
                ui.tags.strong(top_region),
                ui.tags.span(" has the highest concentration."),
            ),
            ui.tags.li(
                ui.tags.strong(f"{exit_rate}% exit signal"),
                ui.tags.span(" from acquired or IPO outcomes."),
            ),
            class_="insight-list",
        )

    @output
    @render.data_frame
    def records_table():
        table_df = explorer_data().drop(
            columns=["Founder Count", "Longitude", "Latitude"],
            errors="ignore",
        )

        return render.DataGrid(
            table_df,
            width="100%",
            filters=True,
            height=360,
        )

    @render.download(filename="founderradar_filtered_records.csv")
    def download_csv():
        yield explorer_data().to_csv(index=False)


app = App(app_ui, server, static_assets=APP_DIR)

