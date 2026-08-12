import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.geo import haversine_km

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Rough centroid of Düsseldorf's Altstadt / Heinrich-Heine-Allee — used as the
# reference point for the "location quality" proxy in performance mode.
CITY_CENTER = (51.2254, 6.7768)
NEIGHBOR_RADIUS_KM = 0.5

OPERATOR_COLORS = {
    "Q-Park": "#3366CC",
    "APCOA": "#8FB8F0",
    "Contipark": "#E0574A",
    "B+B Parkhaus": "#F2A7A0",
}
UNSELECTED_COLOR = "#6E6E6E"

NEUTRAL_BAND_PCT = 5.0   # +/- this % vs the 500m-neighbour average reads as grey ("no real difference")
FULL_COLOR_PCT = 15.0    # by +/- this %, colour is already fully saturated red/green — not stretched
                          # out to whatever the most extreme garage in view happens to be, so a
                          # garage doesn't look washed-out grey just because one outlier exists elsewhere.

PERF_RED = (176, 42, 42)      # #B02A2A
PERF_GREY = (181, 181, 181)   # #B5B5B5
PERF_GREEN = (34, 139, 79)    # #228B4F


def _lerp_rgb(c1, c2, t):
    return tuple(round(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def _rgb_to_hex(rgb):
    return "#%02X%02X%02X" % rgb


def value_to_hex(v, band_pct=NEUTRAL_BAND_PCT, full_pct=FULL_COLOR_PCT):
    """Grey within +/- band_pct, solid red/green beyond +/- full_pct, smooth gradient
    between the two. Fixed thresholds (not scaled to the current dataset's extremes),
    so a garage 11% above its neighbours always reads as clearly green."""
    if -band_pct <= v <= band_pct:
        return _rgb_to_hex(PERF_GREY)
    if v <= -full_pct:
        return _rgb_to_hex(PERF_RED)
    if v >= full_pct:
        return _rgb_to_hex(PERF_GREEN)
    if v < 0:
        t = (-v - band_pct) / (full_pct - band_pct)
        return _rgb_to_hex(_lerp_rgb(PERF_GREY, PERF_RED, t))
    t = (v - band_pct) / (full_pct - band_pct)
    return _rgb_to_hex(_lerp_rgb(PERF_GREY, PERF_GREEN, t))


def build_performance_colorscale(max_abs_pct, band_pct=NEUTRAL_BAND_PCT, full_pct=FULL_COLOR_PCT):
    """Colorbar legend matching value_to_hex exactly. Used only for the SVG colorbar —
    Plotly's Scattermapbox marker layer mis-renders unevenly-spaced continuous
    colorscales on WebGL, so the actual markers are coloured manually via value_to_hex."""
    max_abs_pct = max(max_abs_pct, full_pct)
    band_edge = band_pct / (2 * max_abs_pct)
    full_edge = full_pct / (2 * max_abs_pct)
    lo_band, hi_band = 0.5 - band_edge, 0.5 + band_edge
    lo_full, hi_full = 0.5 - full_edge, 0.5 + full_edge
    stops = [[0.0, _rgb_to_hex(PERF_RED)]]
    if lo_full > 1e-6:
        stops.append([lo_full, _rgb_to_hex(PERF_RED)])
    stops += [[lo_band, _rgb_to_hex(PERF_GREY)], [hi_band, _rgb_to_hex(PERF_GREY)]]
    if hi_full < 1 - 1e-6:
        stops.append([hi_full, _rgb_to_hex(PERF_GREEN)])
    stops.append([1.0, _rgb_to_hex(PERF_GREEN)])
    return stops

st.set_page_config(page_title="Q-Park Parking Optimizer", layout="wide")


@st.cache_data
def load_data():
    kpi_df = pd.read_csv(os.path.join(OUTPUTS_DIR, "kpi_comparison.csv"))
    with open(os.path.join(OUTPUTS_DIR, "summary.json")) as f:
        summary = json.load(f)
    timeseries = {}
    garage_kpis = {}
    for scenario in kpi_df["scenario"]:
        ts_path = os.path.join(OUTPUTS_DIR, f"timeseries_{scenario}.csv")
        if os.path.exists(ts_path):
            timeseries[scenario] = pd.read_csv(ts_path)
        gk_path = os.path.join(OUTPUTS_DIR, f"garage_kpis_{scenario}.csv")
        if os.path.exists(gk_path):
            garage_kpis[scenario] = pd.read_csv(gk_path)
    return kpi_df, summary, timeseries, garage_kpis


@st.cache_data
def load_garages():
    return pd.read_csv(os.path.join(DATA_DIR, "assets.csv"))


def bubble_sizes(capacities, min_px=7, max_px=34):
    cmin, cmax = capacities.min(), capacities.max()
    if cmax == cmin:
        return pd.Series([(min_px + max_px) / 2] * len(capacities), index=capacities.index)
    return min_px + (capacities - cmin) / (cmax - cmin) * (max_px - min_px)


def neighbors_within_radius(garage_row, all_garages, radius_km):
    others = all_garages[all_garages["id"] != garage_row["id"]]
    dists = others.apply(
        lambda r: haversine_km(garage_row["lat"], garage_row["lon"], r["lat"], r["lon"]), axis=1
    )
    return others[dists <= radius_km]


kpi_df, summary, timeseries, garage_kpis = load_data()
garages_df = load_garages()
garages_df["capacity"] = garages_df["capacity"].astype(int)
garages_df["_size"] = bubble_sizes(garages_df["capacity"])

st.title("Q-Park Parking Optimizer — Düsseldorf Demonstrator")
st.caption(
    f"{summary['garages']} garages · {summary['total_capacity']:,} spaces · "
    f"{summary['day_type'].title()} demand profile"
)

scenario_names = kpi_df["scenario"].tolist()
selected = st.selectbox("Scenario", scenario_names, index=0)
row = kpi_df[kpi_df["scenario"] == selected].iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Revenue", f"€{row['revenue']:,.0f}")
col2.metric("Avg Occupancy", f"{row['avg_occupancy_pct']:.1f}%")
col3.metric("Sessions", f"{int(row['sessions']):,}")
col4.metric("Revenue / Space", f"€{row['revenue_per_space']:.2f}")

st.divider()

st.subheader("Garage map — all operators")
st.caption(
    "Bubble size = capacity (spaces). Names, addresses, capacities, and hourly/daily pricing are real, "
    "sourced from each operator's own site or live pricing API (Q-Park/Park One, APCOA, Contipark, "
    "B+B Parkhaus) and geocoded via OpenStreetMap. Unpublished digital-feature flags (app/ANPR) and "
    "member-discount tiers remain modeling assumptions, not confirmed figures."
)

color_mode = st.radio(
    "Colour mode",
    ["By operator", "Relative performance vs. nearby garages (500m)"],
    horizontal=True,
)

fig_map = go.Figure()

if color_mode == "By operator":
    operator_options = sorted(garages_df["operator"].unique())
    selected_operators = st.multiselect("Highlight operators", operator_options, default=operator_options)

    background = garages_df[~garages_df["operator"].isin(selected_operators)]
    if len(background):
        fig_map.add_trace(go.Scattermapbox(
            lat=background["lat"], lon=background["lon"],
            mode="markers",
            marker=dict(size=background["_size"], color=UNSELECTED_COLOR, opacity=0.8),
            text=background["name"] + " (" + background["operator"] + ") — " + background["capacity"].astype(str) + " spaces",
            hoverinfo="text",
            name="Not highlighted",
            showlegend=False,
        ))

    for op in selected_operators:
        sub = garages_df[garages_df["operator"] == op]
        fig_map.add_trace(go.Scattermapbox(
            lat=sub["lat"], lon=sub["lon"],
            mode="markers",
            marker=dict(size=sub["_size"], color=OPERATOR_COLORS.get(op, "#888888"), opacity=0.9),
            text=sub["name"] + " (" + sub["operator"] + ") — " + sub["capacity"].astype(str) + " spaces · €" + sub["hourly_rate"].round(2).astype(str) + "/h",
            hoverinfo="text",
            name=op,
        ))

    shown_df = garages_df[garages_df["operator"].isin(selected_operators)]
    map_col1, map_col2, map_col3 = st.columns(3)
    map_col1.metric("Garages highlighted", len(shown_df))
    map_col2.metric("Spaces highlighted", f"{int(shown_df['capacity'].sum()):,}")
    map_col3.metric("Operators highlighted", len(selected_operators))

else:
    # (column, invert): invert=True means LOWER raw values are better (e.g. distance to
    # city centre), so the sign is flipped before colouring — green always means "better".
    metric_options = {
        "Price (€/hour)": ("hourly_rate", False),
        "Asset quality (0-1 score)": ("quality_score", False),
        "Location quality (proximity to city centre)": ("_distance_km", True),
        "Occupancy (%)": ("_avg_occupancy_pct", False),
    }
    pf_col1, pf_col2 = st.columns([1, 2])
    with pf_col1:
        single_operator = st.selectbox("Operator to analyse", sorted(garages_df["operator"].unique()))
    with pf_col2:
        metric_label = st.selectbox("Metric — coloured vs. average of garages within 500m", list(metric_options.keys()))
    metric_col, metric_invert = metric_options[metric_label]

    def format_metric_value(v, col=metric_col):
        if col == "hourly_rate":
            return f"€{v:.2f}/h"
        if col == "quality_score":
            return f"{v:.2f} quality score"
        if col == "_distance_km":
            return f"{v:.2f} km from centre"
        if col == "_avg_occupancy_pct":
            return f"{v:.1f}% occupancy"
        return str(v)

    work_df = garages_df.copy()
    work_df["_distance_km"] = work_df.apply(
        lambda r: haversine_km(r["lat"], r["lon"], CITY_CENTER[0], CITY_CENTER[1]), axis=1
    )
    gk = garage_kpis.get(selected)
    if gk is not None:
        occ_map = dict(zip(gk["garage_id"], gk["avg_occupancy_pct"]))
        work_df["_avg_occupancy_pct"] = work_df["id"].map(occ_map).fillna(0.0)
    else:
        work_df["_avg_occupancy_pct"] = 0.0

    target = work_df[work_df["operator"] == single_operator].copy()
    deltas, neighbor_counts = [], []
    for _, g in target.iterrows():
        nb = neighbors_within_radius(g, work_df, NEIGHBOR_RADIUS_KM)
        neighbor_counts.append(len(nb))
        neighbor_avg = nb[metric_col].mean() if len(nb) else None
        if not neighbor_avg:  # no neighbours, or neighbour average is exactly 0 (can't take a %)
            deltas.append(None)
        else:
            pct = (g[metric_col] - neighbor_avg) / neighbor_avg * 100
            deltas.append(round(-pct if metric_invert else pct, 1))
    target["_delta"] = deltas
    target["_neighbor_count"] = neighbor_counts

    background = work_df[work_df["operator"] != single_operator]
    fig_map.add_trace(go.Scattermapbox(
        lat=background["lat"], lon=background["lon"],
        mode="markers",
        marker=dict(size=background["_size"], color=UNSELECTED_COLOR, opacity=0.8),
        text=(
            background["name"] + " (" + background["operator"] + ")<br>"
            + background[metric_col].map(format_metric_value)
        ),
        hoverinfo="text",
        showlegend=False,
    ))

    no_data = target[target["_delta"].isna()]
    has_data = target[target["_delta"].notna()]

    if len(no_data):
        fig_map.add_trace(go.Scattermapbox(
            lat=no_data["lat"], lon=no_data["lon"],
            mode="markers",
            marker=dict(size=no_data["_size"], color="#F5A623", opacity=0.9),
            text=(
                no_data["name"] + " (" + no_data["operator"] + ")<br>"
                + no_data[metric_col].map(format_metric_value)
                + "<br>No competitors within 500m"
            ),
            hoverinfo="text",
            name=f"{single_operator} (no 500m neighbours)",
        ))

    if len(has_data):
        max_abs = max(abs(has_data["_delta"].min()), abs(has_data["_delta"].max())) or FULL_COLOR_PCT
        marker_colors = has_data["_delta"].map(value_to_hex).tolist()

        # Scattermapbox on WebGL mis-renders continuous marker colouring for
        # unevenly-spaced custom colorscales, so colours are precomputed per-point
        # above (value_to_hex) and passed as literal hex strings here. This dummy,
        # invisible trace exists purely to draw a matching colorbar legend.
        fig_map.add_trace(go.Scattermapbox(
            lat=[has_data["lat"].iloc[0]], lon=[has_data["lon"].iloc[0]],
            mode="markers",
            marker=dict(
                size=0.01,
                color=[0],
                colorscale=build_performance_colorscale(max_abs),
                cmin=-max_abs,
                cmax=max_abs,
                showscale=True,
                colorbar=dict(
                    title=dict(text=metric_label + "<br>% vs 500m avg", side="right"),
                    ticksuffix="%",
                ),
                opacity=0,
            ),
            hoverinfo="skip",
            showlegend=False,
        ))

        fig_map.add_trace(go.Scattermapbox(
            lat=has_data["lat"], lon=has_data["lon"],
            mode="markers",
            marker=dict(size=has_data["_size"], color=marker_colors, opacity=0.95),
            text=(
                has_data["name"] + " (" + has_data["operator"] + ")<br>"
                + has_data[metric_col].map(format_metric_value)
                + "<br>" + metric_label + " vs neighbours: "
                + has_data["_delta"].map(lambda v: f"{v:+.1f}%")
                + "<br>Neighbours within 500m: " + has_data["_neighbor_count"].astype(str)
            ),
            hoverinfo="text",
            name=f"{single_operator} (relative performance)",
        ))

    price_note = (
        " For price, green just means \"priced above the local average\" — that isn't automatically "
        "good or bad for the business, judge it in context." if metric_col == "hourly_rate" else ""
    )
    st.caption(
        f"All garages are always shown. Grey = every operator except {single_operator}. "
        f"{single_operator}'s own garages are coloured by % difference from the average of all other "
        f"garages within 500m: grey = within ±{NEUTRAL_BAND_PCT:.0f}% (no real difference), "
        "green = more than that above average, red = more than that below. Amber = no competitor "
        "within 500m to compare against." + price_note + " \"Location quality\" compares distance to "
        "the Altstadt/city-centre (closer = green), a proxy rather than a published metric."
    )
    if metric_col == "_avg_occupancy_pct" and gk is None:
        st.warning("No occupancy data found for this scenario — re-run the simulation to populate it.")

fig_map.update_layout(
    mapbox_style="open-street-map",
    mapbox=dict(center=dict(lat=51.222, lon=6.783), zoom=12.6),
    height=560,
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
)
st.plotly_chart(fig_map, use_container_width=True)

st.divider()

st.subheader("Scenario comparison")
comp_col1, comp_col2 = st.columns(2)
with comp_col1:
    fig_rev = px.bar(kpi_df, x="scenario", y="revenue", title="Revenue by scenario", text_auto=".2s")
    st.plotly_chart(fig_rev, use_container_width=True)
with comp_col2:
    fig_occ = px.bar(kpi_df, x="scenario", y="avg_occupancy_pct", title="Average occupancy by scenario", text_auto=".1f")
    st.plotly_chart(fig_occ, use_container_width=True)

st.subheader(f"Hourly detail — {selected}")
ts = timeseries.get(selected)
if ts is not None:
    hcol1, hcol2 = st.columns(2)
    with hcol1:
        fig_occ_hour = px.line(ts, x="hour", y="occupancy_pct", title="Occupancy by hour", markers=True)
        st.plotly_chart(fig_occ_hour, use_container_width=True)
    with hcol2:
        fig_arr_hour = px.bar(ts, x="hour", y="arrivals", title="Arrivals by hour")
        st.plotly_chart(fig_arr_hour, use_container_width=True)

st.subheader("Micro-market revenue breakdown")
zone_data = summary["zone_breakdown"].get(selected, {})
if zone_data:
    zone_df = pd.DataFrame([
        {"zone": zone, "revenue": v["revenue"], "sessions": v["sessions"]}
        for zone, v in zone_data.items()
    ])
    fig_zone = px.bar(zone_df, x="zone", y="revenue", color="zone", title="Revenue by micro-market", text_auto=".2s")
    st.plotly_chart(fig_zone, use_container_width=True)
    st.dataframe(zone_df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("All scenarios — KPI table")
st.dataframe(kpi_df, use_container_width=True, hide_index=True)
