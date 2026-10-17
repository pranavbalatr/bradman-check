"""Model behaviour on the trained artifacts."""

import json

import numpy as np
import polars as pl
import pytest

from bradman import models, verdicts


def test_winprobnet_beats_baseline_and_is_competitive_with_gbm(trained):
    report = json.loads(models.REPORT_PATH.read_text())["winprob"]
    assert report["brier_winprobnet"] < report["brier_baseline"] - 0.01
    assert report["brier_winprobnet"] < report["brier_hgb"] + 0.01


def test_win_probability_increases_with_runs(trained):
    _, net, _, _ = trained
    runs = np.arange(0, 301, 5)
    for inns in (1, 2, 3, 4):
        for rest in (100, 250, 400):
            x = models.win_features(runs, np.full_like(runs, rest), np.full_like(runs, inns),
                                    np.zeros_like(runs), np.full_like(runs, 35.0))  # fmt: skip
            p = net.prob(x)
            assert p[-1] > p[0], (inns, rest)
            # allow tiny numerical wiggles, but no real decrease anywhere on the grid
            assert np.all(np.diff(p) > -0.005), (inns, rest)


def test_wpa_zero_at_era_average_and_signed(trained, innings):
    _, net, _, _ = trained
    sample = innings.head(500)
    at_mean = sample.with_columns(runs=pl.col("era_mean").round().cast(pl.Int64))
    w = models.wpa(net, sample.with_columns(runs=pl.col("era_mean")))
    assert np.allclose(w, 0, atol=1e-6)
    big = models.wpa(net, at_mean.with_columns(runs=pl.lit(200)))
    duck = models.wpa(net, at_mean.with_columns(runs=pl.lit(0)))
    assert (big > 0).mean() > 0.95 and (duck < 0).mean() > 0.95


def test_counterfactual_is_consistent_for_bradman(trained, innings):
    em, _, table, _ = trained
    b = table.filter(pl.col("player") == "DG Bradman").row(0, named=True)
    # Bradman placed in his own environment = his own fitted environment
    assert b["cf_1930s"] == pytest.approx(b["model_own_avg"], rel=0.05)
    # Shrinkage pulls him below 99.94 but he stays far above everyone else
    assert 60 < b["cf_1930s"] < 99.94
    top = verdicts.pool(table).sort("cf_1930s", descending=True)
    assert top["player"][0] == "DG Bradman"


def test_bradman_passes_everything_but_its_own_benchmark_is_rare(trained):
    _, _, table, cal = trained
    b = table.filter(pl.col("player") == "DG Bradman").row(0, named=True)
    results = verdicts.grade(b, cal)
    assert [t.verdict for t in results if t.number in (1, 2, 5)] == ["PASS"] * 3
    assert results[0].percentile == 100.0
