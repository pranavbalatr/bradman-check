"""Streamlit front end:  uv run streamlit run bradman/app.py"""

from __future__ import annotations

import altair as alt
import polars as pl
import streamlit as st

from bradman import data, engine, verdicts

st.set_page_config(page_title="Bradman Check", layout="wide")


@st.cache_data
def _load():
    table, cal = verdicts.load_tables()
    return data.load(), table, cal


df, table, cal = _load()
pool_names = verdicts.pool(table).sort("raw_average", descending=True)["player"].to_list()
all_names = sorted(table["player"].to_list())

st.title("Bradman Check")
st.caption(
    "Every men's Test innings from Statsguru. Not-outs are treated as censored innings, small "
    "samples are shrunk toward the era, and thresholds come from the percentiles of "
    f"{cal['pool_size']} specialist batters with {verdicts.POOL_MIN_INNINGS}+ innings."
)

default = [n for n in ("SPD Smith", "V Kohli", "SR Tendulkar", "JE Root") if n in all_names]
picked = st.multiselect("Compare against DG Bradman", all_names, default=default)
players = ["DG Bradman", *[p for p in picked if p != "DG Bradman"]]
rows = [table.filter(pl.col("player") == p).row(0, named=True) for p in players]

# ---- verdict table
st.subheader("The five tests")
records = []
for r in rows:
    for t in verdicts.grade(r, cal):
        records.append(
            {"player": r["player"], "test": f"{t.number}. {t.name}", "value": t.value,
             "percentile": t.percentile, "verdict": str(t.verdict), "detail": t.detail}
        )  # fmt: skip
rec = pl.DataFrame(records)
pivot = rec.pivot(on="player", index="test", values="verdict")
st.dataframe(pivot, hide_index=True, width="stretch")
with st.expander("Values and details"):
    st.dataframe(rec, hide_index=True, width="stretch")

# ---- Kaplan-Meier curves
st.subheader("Survival: P(score ≥ r), not-outs censored")
curves = []
for p in players:
    inn = data.player_innings(df, p)
    curves.append(engine.km_curve(inn, 250).with_columns(player=pl.lit(p)))
km = pl.concat(curves)
chart = (
    alt.Chart(km.to_pandas())
    .mark_line(interpolate="step-after")
    .encode(
        x=alt.X("runs:Q", title="runs r"),
        y=alt.Y("survival:Q", title="S(r)", axis=alt.Axis(format="%")),
        color="player:N",
        tooltip=["player", "runs", alt.Tooltip("survival:Q", format=".1%")],
    )
    .properties(height=380)
)
st.altair_chart(chart, width="stretch")

# ---- WPA breakdown
st.subheader("Win Probability Added (WinProbNet)")
st.caption(
    "Per innings: P(team wins | actual runs) − P(team wins | era-average runs), rest of team fixed."
)
wpa = pl.DataFrame(
    [
        {"player": r["player"], "WPA per innings": r["wpa_mean"] * 100,
         f"clutch WPA (rest < {verdicts.CLUTCH_REST_BELOW})": (r["clutch_wpa"] or 0) * 100,
         "clutch innings": r["clutch_innings"], "career WPA (wins)": r["wpa_total"]}
        for r in rows
    ]
)  # fmt: skip
st.dataframe(wpa, hide_index=True, width="stretch")

# ---- distribution context
st.subheader("Where they sit in history")
metric = st.selectbox("Metric", [m for _, _, m in verdicts.TESTS])
pool = verdicts.pool(table).select("player", metric).drop_nulls()
base = (
    alt.Chart(pool.to_pandas())
    .mark_bar(opacity=0.6)
    .encode(x=alt.X(f"{metric}:Q", bin=alt.Bin(maxbins=40)), y="count()")
)
marks = (
    alt.Chart(
        pl.DataFrame({"player": [r["player"] for r in rows], metric: [r[metric] for r in rows]})
        .drop_nulls()
        .to_pandas()
    )
    .mark_rule(strokeWidth=2)
    .encode(x=f"{metric}:Q", color="player:N", tooltip=["player", metric])
)
st.altair_chart(base + marks, width="stretch")
