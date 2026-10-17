"""The scraped table must reproduce Bradman's scorecard record exactly, and era baselines
must come out of the data."""

import polars as pl
import pytest

from bradman import data, engine


def test_bradman_exact_record(bradman):
    assert bradman.height == 80
    assert bradman["runs"].sum() == 6996
    assert bradman["not_out"].sum() == 10
    assert (~bradman["not_out"]).sum() == 70
    assert round(6996 / 70, 2) == 99.94
    assert (bradman["runs"] >= 100).sum() == 29
    assert bradman["runs"].is_between(50, 99).sum() == 13
    assert bradman["runs"].max() == 334
    assert bradman["year"].min() == 1928 and bradman["year"].max() == 1948


def test_bradman_matches_and_teams(bradman):
    # 52 Tests, but he batted in 50: absent hurt at The Oval 1938 and in Australia's only
    # innings at Melbourne in Feb 1932 (Statsguru lists both as 'absent').
    assert bradman["match_key"].n_unique() == 50
    assert set(bradman["country"].unique()) == {"Australia"}
    assert set(bradman["opposition"].unique()) == {
        "England",
        "South Africa",
        "West Indies",
        "India",
    }
    # His last innings: bowled second ball for 0 at The Oval, 1948
    last = bradman.sort("match_date").row(-1, named=True)
    assert last["runs"] == 0 and not last["not_out"] and last["ground"] == "The Oval"


def test_join_keeps_nearly_every_innings(innings):
    # every batter innings found its team innings (total, result, venue)
    assert innings["team_runs"].null_count() == 0
    assert (innings["team_runs"] >= innings["runs"]).all()
    assert innings.height > 90_000


def test_team_totals_and_results_for_a_known_match(innings):
    # The Oval 1938: England 903/7d, Hutton 364 - won by an innings
    hutton = innings.filter((pl.col("player") == "L Hutton") & (pl.col("runs") == 364))
    row = hutton.row(0, named=True)
    assert row["team_runs"] == 903 and row["declared"] and row["team_won"]
    assert row["venue"] == "home" and not row["is_away"]


def test_era_baselines_are_computed_not_constant(innings):
    eras = innings.group_by("year").agg(pl.col("era_mean").first(), pl.col("era_std").first())
    assert eras["era_mean"].null_count() == 0
    assert eras["era_mean"].n_unique() > 50  # varies year to year
    by_year = dict(zip(eras["year"], eras["era_mean"], strict=True))
    # uncovered 1880s pitches were far harder than the 2010s
    assert by_year[1885] + 8 < by_year[2015]
    assert 15 < eras["era_mean"].min() < eras["era_mean"].max() < 65


def test_bradman_z_scores(innings, bradman):
    s = engine.summarise(bradman)
    pid = bradman["player_id"][0]
    alltime = engine.career_z_alltime(innings, pid, min_innings=20)
    print(f"\nBradman: era z raw {s.z_raw:.2f}, era z EB {s.z_eb:.2f}, all-time z {alltime:.2f}")
    # Charles Davis reports ~4.4 sd against his peers; our leave-one-out era baseline gives
    # the same order of magnitude without any tuning.
    assert 4.0 < s.z_raw < 5.0
    assert s.z_eb < s.z_raw  # shrinkage only ever pulls toward the era
    # Pooling every era and every batting position (tailenders included) inflates it.
    assert alltime > s.z_raw


def test_era_baseline_excludes_the_player_himself(innings, bradman):
    # Bradman's peers in his own years, measured on someone who was not in the 1930s pool
    others = innings.filter(
        pl.col("year").is_in(bradman["year"].unique().implode())
        & (pl.col("player_id") != bradman["player_id"][0])
        & (pl.col("batting_position") > 6)
    )
    with_him = others.group_by("year").agg(pl.col("era_mean").first())
    without_him = bradman.group_by("year").agg(pl.col("era_mean").first())
    both = with_him.join(without_him, on="year", suffix="_loo")
    # Removing a 99.94 average can only lower the peer mean
    assert (both["era_mean_loo"] < both["era_mean"]).all()


def test_player_lookup_is_strict(innings):
    with pytest.raises(KeyError):
        data.player_innings(innings, "zzzz no such player")
    assert data.player_innings(innings, "SPD Smith")["player"][0] == "SPD Smith"
