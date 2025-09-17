#app.py
import math
import pandas as pd
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(page_title="MovieLens Explorer", layout="wide")

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("data/movie_ratings.csv")
    if "genres" in df.columns:
        df["genres_list"] = df["genres"].fillna("").apply(
            lambda s: [g.strip() for g in str(s).split("|") if g.strip()]
        )
    else:
        df["genres_list"] = [[] for _ in range(len(df))]

    #rating_year from timestamp if missing
    if "rating_year" not in df.columns and "timestamp" in df.columns:
        try:
            ts = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        except Exception:
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
        df["rating_year"] = ts.dt.year
    return df

df = load_data()
exp = df.explode("genres_list", ignore_index=True)

st.title("🎬 MovieLens — Ratings Explorer")
st.caption("Week 3 · Tabs per question. Popularity (Q1) ≠ Satisfaction (Q2).")

tab1, tab2, tab3, tab4 = st.tabs([
    "Q1 · Genres (Popularity)",
    "Q2 · Genres (Satisfaction)",
    "Q3 · Release Year Trend",
    "Q4 · Top Movies",
])

#Genre Breakdown
with tab1:
    st.subheader("Q1 · What's the breakdown of genres for the movies that were rated?")
    left, right = st.columns([1, 3], vertical_alignment="top")
    genre_series = exp["genres_list"].dropna().astype(str).str.strip()
    total_unique = int(genre_series[genre_series != ""].nunique())

    with left:
        max_k = max(1, total_unique)
        top_k = st.slider("Show top K genres", 1, max_k, min(15, max_k), step=1)
        include_other = st.toggle("Group remainder into “Other”", value=True)
        hide_unknown = st.toggle("Hide “Unknown” genre", value=True)

    with right:
        clean = genre_series.copy()
        if hide_unknown:
            clean = clean[clean.str.lower() != "unknown"]

        counts = (clean.value_counts()
                        .sort_values(ascending=True)
                        .rename_axis("genres_list")
                        .reset_index(name="n"))

        counts_sorted_desc = counts.sort_values("n", ascending=False)
        top = counts_sorted_desc.head(top_k).copy().sort_values("n", ascending=True)

        if include_other and len(counts_sorted_desc) > top_k:
            other_n = counts_sorted_desc["n"].iloc[top_k:].sum()
            top = pd.concat(
                [top, pd.DataFrame({"genres_list": ["Other"], "n": [other_n]})],
                ignore_index=True
            )

        fig1 = px.bar(
            top, x="n", y="genres_list", orientation="h",
            title="Ratings per Genre (Popularity — Top K)",
            labels={"n": "Number of Ratings", "genres_list": "Genre"}
        )
        st.plotly_chart(fig1, use_container_width=True)

    with st.expander("Notes"):
        st.write("- This is a popularity view (how often genres are rated). "
                 "“Other” appears only when Top-K doesn’t cover all genres. "
                 "Toggle hides the literal 'Unknown' genre if present.")

#Viewer Satisfaction
with tab2:
    st.subheader("Q2 · Which genres have the highest viewer satisfaction (highest ratings)?")
    left, right = st.columns([1, 3], vertical_alignment="top")

    # build stats once (before filtering)
    stats_all = (
        exp[exp["genres_list"].notna()]
        .groupby("genres_list")["rating"]
        .agg(mean="mean", n="size", std="std")
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    #pick a slider max that will actually hide something for this dataset
    #use the 95th percentile of counts (so moving the slider matters), cap at 2000
    if not stats_all.empty:
        practical_max = int(min(2000, max(50, stats_all["n"].quantile(0.95))))
    else:
        practical_max = 50

    with left:
        min_count = st.number_input("Minimum ratings per genre",
                                    min_value=0, max_value=practical_max,
                                    value=min(50, practical_max), step=10)
        show_error = st.toggle("Show error bars (std dev)", value=True)

    #apply the filter
    stats = stats_all[stats_all["n"] >= min_count].copy()
    removed_df = stats_all[stats_all["n"] < min_count].copy()
    removed_cnt = int(len(removed_df))

    # quick feedback so changes feel tangible
    if removed_cnt > 0:
        st.info(f"{removed_cnt} genre(s) excluded for having fewer than {min_count} ratings.")

    with right:
        if stats.empty:
            st.warning("No genres meet the current minimum rating threshold.")
        else:
            fig2 = px.bar(
                stats, x="genres_list", y="mean",
                color="mean", color_continuous_scale="Viridis",
                hover_data={"n": True, "mean": ":.2f", "std": ":.2f"},
                title="Average Rating by Genre (Satisfaction)",
                labels={"mean": "Mean Rating (1–5)", "genres_list": "Genre", "n": "# Ratings"}
            )
            if show_error and "std" in stats.columns:
                fig2.update_traces(error_y=dict(type="data", array=stats["std"].values))
            st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Filtered-out genres (low sample)"):
        if removed_cnt == 0:
            st.caption("None filtered out at this threshold.")
        else:
            st.dataframe(
                removed_df.sort_values("n", ascending=True)[["genres_list", "n"]],
                use_container_width=True
            )

    with st.expander("Notes"):
        st.write("- The minimum ratings threshold removes tiny-sample genres so outliers don't dominate.")

#Rating changes across the release years
with tab3:
    st.subheader("Q3 · How does mean rating change across movie release years?")

    VOL_COLOR  = "rgba(130,130,130,0.55)"   #bars (left axis)
    LINE_COLOR = "#1f77b4"                  #line (right axis)

    if "year" not in df.columns:
        st.error("The dataset is missing a 'year' column required for Q3.")
    else:
        left, right = st.columns([1, 3], vertical_alignment="top")

        with left:
            min_year, max_year = int(df["year"].min()), int(df["year"].max())
            year_range = st.slider(
                "Year range", min_value=min_year, max_value=max_year,
                value=(min_year, max_year), step=1
            )
            min_year_n = st.number_input("Minimum ratings per year", 0, 100_000, 50, step=10)
            window = st.slider("Smoothing window (years)", 1, 9, 1, step=1)

        with right:
            yr = (
                df.groupby("year")["rating"]
                  .agg(mean="mean", n="size")
                  .reset_index()
                  .query("@year_range[0] <= year <= @year_range[1] and n >= @min_year_n")
                  .sort_values("year")
            )

            if yr.empty:
                st.info("No data in this year range with the chosen minimum count.")
            else:
                ycol = "mean"
                if window > 1:
                    yr["mean_smooth"] = yr["mean"].rolling(window, center=True).mean()
                    ycol = "mean_smooth"

                fig3 = make_subplots(specs=[[{"secondary_y": True}]])

                # Bars: ratings volume (LEFT axis, gray)
                fig3.add_trace(
                    go.Bar(
                        x=yr["year"], y=yr["n"],
                        name="# Ratings / Year",
                        marker_color=VOL_COLOR
                    ),
                    secondary_y=False
                )

                #Line: mean rating (RIGHT axis, blue)
                fig3.add_trace(
                    go.Scatter(
                        x=yr["year"], y=yr[ycol],
                        name="Mean Rating",
                        mode="lines+markers",
                        line=dict(color=LINE_COLOR, width=2),
                        marker=dict(size=6)
                    ),
                    secondary_y=True
                )

                #Titles
                fig3.update_layout(
                    title_text="Release Year vs Mean Rating (+ ratings volume)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
                )

                #X axis
                fig3.update_xaxes(title_text="Release Year", showgrid=False)

                #Y axes: color-match titles & ticks to their series
                fig3.update_yaxes(
                    title_text="# Ratings (volume)",
                    secondary_y=False,
                    color="rgb(120,120,120)",
                    gridcolor="rgba(200,200,200,0.15)"
                )
                fig3.update_yaxes(
                    title_text="Mean Rating (1–5)",
                    secondary_y=True,
                    color=LINE_COLOR,
                    range=[1, 5], 
                    gridcolor="rgba(0,0,0,0)"
                )

                st.plotly_chart(fig3, use_container_width=True)

    with st.expander("Notes"):
        st.write(
            "- **Gray bars** (left axis) = how many ratings we have for that release year (volume/reliability).  \n"
            "- **Blue line** (right axis) = average rating (1–5) of movies released that year."
        )
#Top rated with 50/150 movies considered

with tab4:
    st.subheader("Q4 · What are the best-rated movies with minimum rating counts?")
    left, right = st.columns([1, 3], vertical_alignment="top")

    with left:
        thr_a = st.number_input("Min ratings (List A)", 0, 10_000, 50, step=10)
        thr_b = st.number_input("Min ratings (List B)", 0, 10_000, 150, step=10)
        chart_a = st.toggle("Also visualize List A as bars", value=False)

    with right:
        ms = (df.groupby(["movie_id", "title"], dropna=False)["rating"]
              .agg(mean="mean", n="size")
              .reset_index())

        a = ms[ms["n"] >= thr_a].sort_values(["mean", "n"], ascending=[False, False]).head(5).copy()
        b = ms[ms["n"] >= thr_b].sort_values(["mean", "n"], ascending=[False, False]).head(5).copy()
        a_disp = a.rename(columns={"mean": "Average Rating", "n": "# Ratings"})[["title", "Average Rating", "# Ratings"]]
        b_disp = b.rename(columns={"mean": "Average Rating", "n": "# Ratings"})[["title", "Average Rating", "# Ratings"]]

        st.markdown("**Top Movies (List A)**")
        st.dataframe(a_disp, use_container_width=True)

        if chart_a and not a.empty:
            #change tik size 
            lo = max(1.0, math.floor((a["mean"].min() - 0.05) * 20) / 20.0)
            hi = min(5.0, math.ceil((a["mean"].max() + 0.05) * 20) / 20.0)

            fig4 = px.bar(
                a.sort_values("mean"),
                x="mean", y="title", orientation="h",
                title="Top Movies (List A)",
                labels={"mean": "Average Rating", "title": "Movie"},
                hover_data={"n": True, "mean": ":.3f"}
            )
            fig4.update_traces(
                text=a.sort_values("mean")["mean"].round(2).astype(str),
                textposition="outside",
                cliponaxis=False
            )
            fig4.update_xaxes(
                range=[lo, hi],
                dtick=0.05,
                tickformat=".2f",
                title="Average Rating (1–5)"
            )
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown("**Top Movies (List B)**")
        st.dataframe(b_disp, use_container_width=True)

st.caption("Notes: Genres exploded for preference profiling. Thresholds reduce small-sample noise.")
