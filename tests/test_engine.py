"""Unit tests for the statistical engine on small hand-built innings tables."""

import math

import numpy as np
import polars as pl
import pytest

from bradman import engine


def innings(
    scores: list[str], era_mean: float = 40.0, era_std: float = 10.0, **cols
) -> pl.DataFrame:
    """'75*' = not out. Minimal columns the engine needs."""
    n = len(scores)
    base = {
        "player": ["X"] * n,
        "country": ["Testland"] * n,
        "year": [2000] * n,
        "runs": [int(s.rstrip("*")) for s in scores],
        "not_out": [s.endswith("*") for s in scores],
        "team_runs": [300] * n,
        "is_away": [False] * n,
        "innings_num": [1] * n,
        "era_mean": [era_mean] * n,
        "era_std": [era_std] * n,
    }
    return pl.DataFrame(base | cols)


def test_eb_matches_formula_and_shrinks_flukes():
    # 80*, 15 in a 35-average era: raw average 95 looks Bradman-like
    s = engine.summarise(innings(["80*", "15"], era_mean=35.0))
    assert s.raw_average == 95.0
    assert s.eb_average == pytest.approx((95 + 10 * 35) / (1 + 10))  # 40.45
    assert s.z_eb < 1.0 < s.z_raw


def test_eb_never_divides_by_zero():
    s = engine.summarise(innings(["40*", "35*", "50*"]))
    assert s.raw_average is None
    assert s.eb_average == pytest.approx((125 + 400) / 10)


def test_eb_converges_to_raw_average_with_data():
    many = innings(["50"] * 2000, era_mean=30.0)
    assert engine.summarise(many).eb_average == pytest.approx(50.0, rel=0.01)


def test_km_treats_unbeaten_fifty_as_censored_not_failed():
    # 100, 60 out, 70 out, plus a 75 that is either out or not out.
    out_version = engine.summarise(innings(["100", "60", "70", "75"]))
    censored = engine.summarise(innings(["100", "60", "70", "75*"]))
    # Dismissed at 75: 1 of 4 innings past 50 reached 100.
    assert out_version.km_conversion == pytest.approx(0.25)
    # Unbeaten 75: it leaves the risk set at 75 without failing. Deaths at 60 (4 at risk) and
    # 70 (3 at risk) give S(100) = 3/4 * 2/3 = 1/2 - the censored innings is credited with the
    # same chance of reaching 100 as the batter still in.
    assert censored.km_conversion == pytest.approx(0.5)
    # The naive scorecard ratio cannot tell the difference.
    assert out_version.naive_conversion == censored.naive_conversion == 0.25


def test_km_survival_at_integer_thresholds():
    runs = np.array([0, 49, 50, 100], float)
    out = np.array([True, True, True, True])
    assert engine.km_survival(runs, out, 50) == pytest.approx(0.5)  # 50 counts as reaching 50
    assert engine.km_survival(runs, out, 100) == pytest.approx(0.25)


def test_piecewise_hazard_counts_exposure_correctly():
    # 10 (out), 40 (out), 40* : early exposure = 10+1 + 30 + 30, deaths 1
    early, late = engine.piecewise_hazard(np.array([10, 40, 40.0]), np.array([True, True, False]))
    assert early == pytest.approx(1 / 71)
    # late: 40 out -> 10 runs + 1 failing trial; 40* -> 10 runs
    assert late == pytest.approx(1 / 21)


def test_weibull_shape_detects_falling_hazard():
    # Many ducks and a long tail = hazard falls once in: rho < 1
    falling = [0] * 30 + [1, 2, 3] * 5 + [80, 120, 150, 200, 250] * 3
    s = engine.summarise(innings([str(r) for r in falling]))
    assert s.weibull_rho < 1
    # Scores bunched around 40 = hazard rises with time: rho > 1
    rising = [35, 38, 40, 42, 45, 37, 41, 39, 44, 36] * 3
    assert engine.summarise(innings([str(r) for r in rising])).weibull_rho > 1


def test_hostile_ratio_splits_away_and_late_innings():
    inn = innings(
        ["100", "100", "20", "20"],
        is_away=[False, False, True, False],
        innings_num=[1, 2, 1, 4],
    )
    s = engine.summarise(inn, k=0.0001)
    assert s.friendly_eb == pytest.approx(100, rel=1e-3)
    assert s.hostile_eb == pytest.approx(20, rel=1e-3)
    assert s.hostile_ratio == pytest.approx(0.2, rel=1e-3)


def test_km_curve_starts_at_one_and_is_monotone():
    c = engine.km_curve(innings(["0", "12", "50*", "101", "7"]), 120)
    s = c["survival"].to_numpy()
    assert s[0] == 1.0 and np.all(np.diff(s) <= 1e-12) and not math.isnan(s[-1])
