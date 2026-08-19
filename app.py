import hashlib
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.geo import haversine_km

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CITIES_DIR = os.path.join(DATA_DIR, "cities")

CITIES = {
    "Düsseldorf": os.path.join(DATA_DIR, "assets.csv"),
    "Berlin": os.path.join(CITIES_DIR, "berlin.csv"),
    "Frankfurt": os.path.join(CITIES_DIR, "frankfurt.csv"),
    "Hamburg": os.path.join(CITIES_DIR, "hamburg.csv"),
    "Hannover": os.path.join(CITIES_DIR, "hannover.csv"),
    "Köln": os.path.join(CITIES_DIR, "koeln.csv"),
    "München": os.path.join(CITIES_DIR, "muenchen.csv"),
    "Nürnberg": os.path.join(CITIES_DIR, "nuernberg.csv"),
    "Stuttgart": os.path.join(CITIES_DIR, "stuttgart.csv"),
}

NEIGHBOR_RADIUS_KM = 0.5

BASE_OPERATOR_COLORS = {
    "Q-Park": "#3366CC",
    "APCOA": "#8FB8F0",
    "Contipark": "#E0574A",
    "B+B Parkhaus": "#F2A7A0",
}
FALLBACK_PALETTE = [
    "#2CA02C", "#9467BD", "#8C564B", "#17BECF", "#BCBD22", "#FF7F0E", "#E377C2",
    "#1A9850", "#6A3D9A", "#B15928", "#A6CEE3", "#FDBF6F",
]
UNSELECTED_COLOR = "#6E6E6E"

NEUTRAL_BAND_PCT = 5.0   # +/- this % vs the 500m-neighbour average reads as grey ("no real difference")
FULL_COLOR_PCT = 15.0    # by +/- this %, colour is already fully saturated red/green — not stretched
                          # out to whatever the most extreme garage in view happens to be, so a
                          # garage doesn't look washed-out grey just because one outlier exists elsewhere.

PERF_RED = (176, 42, 42)      # #B02A2A
PERF_GREY = (181, 181, 181)   # #B5B5B5
PERF_GREEN = (34, 139, 79)    # #228B4F


def _stable_palette_index(name, n):
    """Deterministic across runs/processes (unlike Python's built-in hash(), which is
    randomized per-process) so the same operator name always lands on the same colour,
    regardless of which city's operator list it's being assigned within."""
    return int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16) % n


def operator_color_map(operators):
    colors = {}
    for op in operators:
        if op in BASE_OPERATOR_COLORS:
            colors[op] = BASE_OPERATOR_COLORS[op]
        else:
            colors[op] = FALLBACK_PALETTE[_stable_palette_index(op, len(FALLBACK_PALETTE))]
    return colors


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


st.set_page_config(page_title="Parking Competitive Analysis", layout="wide")


@st.cache_data
def load_garages(csv_path):
    df = pd.read_csv(csv_path)
    df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce")
    df["hourly_rate"] = pd.to_numeric(df["hourly_rate"], errors="coerce")
    if "daily_cap" in df.columns:
        df["daily_cap"] = pd.to_numeric(df["daily_cap"], errors="coerce")
    return df


def bubble_sizes(capacities, min_px=7, max_px=34):
    cmin, cmax = capacities.min(), capacities.max()
    mid = (min_px + max_px) / 2
    if pd.isna(cmin) or pd.isna(cmax) or cmax == cmin:
        return pd.Series([mid] * len(capacities), index=capacities.index)
    return (min_px + (capacities - cmin) / (cmax - cmin) * (max_px - min_px)).fillna(mid)


def neighbors_within_radius(garage_row, all_garages, radius_km):
    others = all_garages[all_garages["id"] != garage_row["id"]]
    dists = others.apply(
        lambda r: haversine_km(garage_row["lat"], garage_row["lon"], r["lat"], r["lon"]), axis=1
    )
    return others[dists <= radius_km]


def format_capacity(v):
    return f"{int(v)} spaces" if pd.notna(v) else "capacity n/a"


def format_price(v):
    return f"€{v:.2f}/h" if pd.notna(v) else "price n/a"


st.title("Parking Competitive Analysis")

city = st.selectbox("City", list(CITIES.keys()), index=0)
garages_df = load_garages(CITIES[city])
garages_df["_size"] = bubble_sizes(garages_df["capacity"])

total_capacity = garages_df["capacity"].sum()
n_operators = garages_df["operator"].nunique()
n_priced = garages_df["hourly_rate"].notna().sum()
st.caption(
    f"{len(garages_df)} garages · {n_operators} operators · {int(total_capacity):,} spaces "
    f"(where published) · {n_priced}/{len(garages_df)} with published hourly pricing"
)
st.caption(
    "Names, addresses, capacities, and hourly/daily pricing are real, sourced from each operator's own "
    "site or live pricing API and geocoded via OpenStreetMap/embedded page coordinates. Where an operator "
    "doesn't publish a rate for a garage, price is left blank rather than estimated."
)

color_mode = st.radio(
    "Colour mode",
    ["By operator", "Relative performance vs. nearby garages (500m)"],
    horizontal=True,
)

fig_map = go.Figure()

if color_mode == "By operator":
    operator_options = sorted(garages_df["operator"].unique())
    operator_colors = operator_color_map(operator_options)
    selected_operators = st.multiselect("Highlight operators", operator_options, default=operator_options)

    background = garages_df[~garages_df["operator"].isin(selected_operators)]
    if len(background):
        fig_map.add_trace(go.Scattermapbox(
            lat=background["lat"], lon=background["lon"],
            mode="markers",
            marker=dict(size=background["_size"], color=UNSELECTED_COLOR, opacity=0.8),
            text=(
                background["name"] + " (" + background["operator"] + ") — "
                + background["capacity"].map(format_capacity)
            ),
            hoverinfo="text",
            name="Not highlighted",
            showlegend=False,
        ))

    for op in selected_operators:
        sub = garages_df[garages_df["operator"] == op]
        fig_map.add_trace(go.Scattermapbox(
            lat=sub["lat"], lon=sub["lon"],
            mode="markers",
            marker=dict(size=sub["_size"], color=operator_colors.get(op, "#888888"), opacity=0.9),
            text=(
                sub["name"] + " (" + sub["operator"] + ") — " + sub["capacity"].map(format_capacity)
                + " · " + sub["hourly_rate"].map(format_price)
            ),
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
        "Location quality (proximity to city centre)": ("_distance_km", True),
    }
    pf_col1, pf_col2 = st.columns([1, 2])
    with pf_col1:
        single_operator = st.selectbox("Operator to analyse", sorted(garages_df["operator"].unique()))
    with pf_col2:
        metric_label = st.selectbox("Metric — coloured vs. average of garages within 500m", list(metric_options.keys()))
    metric_col, metric_invert = metric_options[metric_label]

    def format_metric_value(v, col=metric_col):
        if pd.isna(v):
            return "price n/a" if col == "hourly_rate" else "n/a"
        if col == "hourly_rate":
            return f"€{v:.2f}/h"
        if col == "_distance_km":
            return f"{v:.2f} km from centre"
        return str(v)

    work_df = garages_df.copy()
    city_center = (work_df["lat"].mean(), work_df["lon"].mean())
    work_df["_distance_km"] = work_df.apply(
        lambda r: haversine_km(r["lat"], r["lon"], city_center[0], city_center[1]), axis=1
    )

    target = work_df[work_df["operator"] == single_operator].copy()
    deltas, neighbor_counts = [], []
    for _, g in target.iterrows():
        nb = neighbors_within_radius(g, work_df, NEIGHBOR_RADIUS_KM)
        neighbor_counts.append(len(nb))
        own_value = g[metric_col]
        neighbor_avg = nb[metric_col].mean() if len(nb) else None
        if pd.isna(own_value) or not neighbor_avg or pd.isna(neighbor_avg):
            deltas.append(None)
        else:
            pct = (own_value - neighbor_avg) / neighbor_avg * 100
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
                + "<br>No comparable data within 500m"
            ),
            hoverinfo="text",
            name=f"{single_operator} (no comparison data)",
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
        "green = more than that above average, red = more than that below. Amber = no published data "
        "for this garage or no comparable garage within 500m." + price_note + " \"Location quality\" "
        "compares distance to this city's garage-density centroid (closer = green), a proxy rather than "
        "a published metric."
    )

fig_map.update_layout(
    mapbox_style="open-street-map",
    mapbox=dict(center=dict(lat=garages_df["lat"].mean(), lon=garages_df["lon"].mean()), zoom=11.5),
    height=560,
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
)
st.plotly_chart(fig_map, use_container_width=True)

st.divider()
st.subheader(f"All garages — {city}")
display_cols = ["id", "name", "operator", "capacity", "hourly_rate", "daily_cap", "has_ev", "address"]
display_cols = [c for c in display_cols if c in garages_df.columns]
st.dataframe(garages_df[display_cols], use_container_width=True, hide_index=True)
