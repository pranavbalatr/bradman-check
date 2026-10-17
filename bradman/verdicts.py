"""The five Bradman tests, graded against the historical distribution.

Thresholds are not hand-picked: each metric is computed for every qualified specialist
batter in Test history, and a player PASSes at the 90th percentile of that pool and gets
CAUTION at the 70th. Bradman's own value is shown next to every metric as the benchmark.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum

import numpy as np
import polars as pl

from bradman import engine, models
from bradman.data import DATA_DIR

PLAYERS_PATH = DATA_DIR / "players.parquet"
CALIBRATION_PATH = DATA_DIR / "calibration.json"
WPA_PATH = DATA_DIR / "innings_wpa.parquet"

MIN_TABLE_INNINGS = 10  # players listed in the table
POOL_MIN_INNINGS = 40  # calibration pool: established specialist batters
POOL_MAX_MODAL_POSITION = 6
PASS_PCT, CAUTION_PCT = 90, 70
CLUTCH_REST_BELOW = 200  # "collapse": rest of the team made fewer than this


class Verdict(StrEnum):
    PASS = "PASS"
    CAUTION = "CAUTION"
    FAIL = "FAIL"
    OPEN = "OPEN"


@dataclass(frozen=True)
class TestResult:
    number: int
    name: str
    metric: str
    value: float | None
    bradman: float
    percentile: float | None
    verdict: Verdict
    detail: str


TESTS = (
    (1, "Era-adjusted dominance", "z_eb"),
    (2, "Conversion once set (Kaplan-Meier)", "km_conversion"),
    (3, "Match-winning impact (WinProbNet WPA)", "wpa_mean"),
    (4, "Hostile-conditions resilience", "hostile_ratio"),
    (5, "Counterfactual 1928-48 average", "cf_1930s"),
)


# --------------------------------------------------------------------------- build


def build_tables(df: pl.DataFrame, em: models.EraModel, net: models.WinProbNet) -> pl.DataFrame:
    df = df.with_columns(wpa=pl.Series(models.wpa(net, df)))
    df.select("player_id", "match_key", "innings_num", "wpa").write_parquet(WPA_PATH)
    ctx = models.bradman_contexts(df)

    counts = df.group_by("player_id").len().filter(pl.col("len") >= MIN_TABLE_INNINGS)
    rows = []
    for pid in counts["player_id"].to_list():
        inn = df.filter(pl.col("player_id") == pid).sort("match_date", "innings_num")
        s = engine.summarise(inn)
        modal_pos = int(inn["batting_position"].mode().sort()[0])
        clutch = inn.filter(pl.col("rest_of_team_runs") < CLUTCH_REST_BELOW)
        own = models._design_frame(inn.filter(pl.col("batting_position") <= models.MAX_POSITION))
        own = own.with_columns(player_key=pl.lit(str(pid)))
        rows.append(
            asdict(s)
            | {
                "player_id": pid,
                "modal_position": modal_pos,
                "wpa_mean": float(inn["wpa"].mean()),
                "wpa_total": float(inn["wpa"].sum()),
                "clutch_wpa": float(clutch["wpa"].mean()) if clutch.height else None,
                "clutch_innings": clutch.height,
                "cf_1930s": models.counterfactual_average(em, ctx, pid, modal_pos),
                "model_own_avg": em.expected_average(own) if own.height else None,
            }
        )
    table = pl.DataFrame(rows, infer_schema_length=None)
    table.write_parquet(PLAYERS_PATH)
    return table


def pool(table: pl.DataFrame) -> pl.DataFrame:
    return table.filter(
        (pl.col("innings") >= POOL_MIN_INNINGS)
        & (pl.col("modal_position") <= POOL_MAX_MODAL_POSITION)
    )


def calibrate(table: pl.DataFrame) -> dict:
    p = pool(table)
    bradman = table.filter(pl.col("player") == "DG Bradman").row(0, named=True)
    cal = {"pool_size": p.height, "tests": {}}
    for _, _, metric in TESTS:
        vals = p[metric].drop_nulls().drop_nans().to_numpy()
        cal["tests"][metric] = {
            "pass": float(np.percentile(vals, PASS_PCT)),
            "caution": float(np.percentile(vals, CAUTION_PCT)),
            "median": float(np.median(vals)),
            "bradman": bradman[metric],
            "values": sorted(vals.tolist()),
        }
    CALIBRATION_PATH.write_text(json.dumps(cal))
    return cal


def load_tables() -> tuple[pl.DataFrame, dict]:
    if not PLAYERS_PATH.exists():
        raise FileNotFoundError("run `uv run python -m bradman train` first")
    return pl.read_parquet(PLAYERS_PATH), json.loads(CALIBRATION_PATH.read_text())


# --------------------------------------------------------------------------- grade


def _percentile(values: list[float], v: float) -> float:
    return 100.0 * np.searchsorted(values, v, side="right") / len(values)


def _grade(v: float | None, t: dict) -> Verdict:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return Verdict.OPEN
    if v >= t["pass"]:
        return Verdict.PASS
    if v >= t["caution"]:
        return Verdict.CAUTION
    return Verdict.FAIL


def _down(v: Verdict) -> Verdict:
    return {Verdict.PASS: Verdict.CAUTION, Verdict.CAUTION: Verdict.FAIL}.get(v, v)


def grade(row: dict, cal: dict) -> list[TestResult]:
    out = []
    for number, name, metric in TESTS:
        t = cal["tests"][metric]
        v = row[metric]
        verdict = _grade(v, t)
        pct = _percentile(t["values"], v) if v is not None else None
        if metric == "z_eb":
            detail = (
                f"EB avg {row['eb_average']:.1f} vs era peers {row['era_mean']:.1f} "
                f"(sd {row['era_std']:.1f}); raw avg {_f(row['raw_average'])}"
            )
        elif metric == "km_conversion":
            rho = row["weibull_rho"]
            detail = (
                f"P(100|50)={_pct(v)} (naive {_pct(row['naive_conversion'])}); "
                f"Weibull rho={_f(rho, 2)}; hazard/run early {_f(row['early_hazard'], 4)}"
                f" -> settled {_f(row['settled_hazard'], 4)}"
            )
            if rho is not None and rho >= 1:
                verdict = _down(verdict)
                detail += " [downgraded: hazard rises as innings goes on]"
        elif metric == "wpa_mean":
            detail = (
                f"{v * 100:+.2f} win-% per innings vs era-average batter; "
                f"clutch (rest of team <{CLUTCH_REST_BELOW}) {_signed_pct(row['clutch_wpa'])}"
                f" over {row['clutch_innings']} inns"
            )
        elif metric == "hostile_ratio":
            detail = (
                f"away or 3rd/4th inns EB {_f(row['hostile_eb'])} / "
                f"home 1st/2nd inns EB {_f(row['friendly_eb'])}"
            )
        else:
            detail = (
                f"model-predicted avg in Bradman's 80-innings environment "
                f"(own environment: {_f(row['model_own_avg'])})"
            )
        out.append(TestResult(number, name, metric, v, t["bradman"], pct, verdict, detail))
    return out


def _f(v, nd=1):
    return "n/a" if v is None else f"{v:.{nd}f}"


def _pct(v):
    return "n/a" if v is None else f"{v:.1%}"


def _signed_pct(v):
    return "n/a" if v is None else f"{v * 100:+.2f}"
