"""Statistical engine: per-player summaries that treat a not-out as what it is, an
unfinished (right-censored) innings, and that shrink small samples toward the era.

Everything here is a pure function of one player's innings table (a polars frame from
data.load()), so it is easy to test and has no model dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl
from lifelines import KaplanMeierFitter, WeibullFitter

PRIOR_DISMISSALS = 10  # k: pseudo-dismissals of era-average batting added to every player
SETTLED_AT = 30  # runs; boundary between the "new batter" and "set batter" hazard regimes


@dataclass(frozen=True)
class Summary:
    player: str
    country: str
    first_year: int
    last_year: int
    innings: int
    not_outs: int
    dismissals: int
    runs: int
    hundreds: int
    fifties: int  # 50-99, the scorecard convention
    raw_average: float | None
    eb_average: float  # Empirical Bayes posterior mean (Gamma prior, exponential likelihood)
    era_mean: float  # innings-weighted era baseline across the player's career
    era_std: float
    z_raw: float | None
    z_eb: float
    km_s50: float  # Kaplan-Meier P(R >= 50)
    km_s100: float  # Kaplan-Meier P(R >= 100)
    km_conversion: float | None  # S(100) / S(50)
    naive_conversion: float | None  # hundreds / (hundreds + fifties), the censoring-blind way
    weibull_rho: float | None  # < 1: hazard falls as the innings goes on
    weibull_lambda: float | None
    early_hazard: float | None  # dismissals per run at risk in [0, SETTLED_AT)
    settled_hazard: float | None  # dismissals per run at risk in [SETTLED_AT, inf)
    team_share: float | None  # share of team runs in innings he batted
    hostile_eb: float | None  # EB avg: away OR 3rd/4th innings
    friendly_eb: float | None  # EB avg: home AND 1st/2nd innings
    hostile_ratio: float | None


def eb_average(runs: float, dismissals: float, prior_mean: float, k: float = PRIOR_DISMISSALS):
    """Posterior mean runs-per-dismissal.

    Model: each run is an independent 'trial' with dismissal hazard h, so runs-to-dismissal is
    exponential with mean mu = 1/h. Total runs is the exposure (not-outs included) and
    dismissals are the events, so the likelihood is h^D exp(-h R). With a Gamma(k, k*mu0) prior
    on h, the posterior is Gamma(k + D, k*mu0 + R) and E[1/h] = (R + k*mu0) / (D + k - 1).
    We use the conventional (R + k*mu0) / (D + k), i.e. k pseudo-dismissals at the era mean;
    it never divides by zero and equals the raw average as D grows."""
    return (runs + k * prior_mean) / (dismissals + k)


def _km(runs: np.ndarray, out: np.ndarray) -> KaplanMeierFitter:
    return KaplanMeierFitter().fit(durations=runs, event_observed=out)


def km_survival(runs: np.ndarray, out: np.ndarray, at: float) -> float:
    """P(R >= at) with not-outs censored. KM gives S(t) = P(R > t); runs are integers, so
    P(R >= 50) = S(49.5)."""
    if len(runs) == 0:
        return float("nan")
    return float(_km(runs, out).survival_function_at_times(at - 0.5).iloc[0])


def piecewise_hazard(runs: np.ndarray, out: np.ndarray, cut: int = SETTLED_AT):
    """Dismissals per run at risk, before and after `cut`. An innings of r runs is at risk for
    r 'run-trials' (it survived each) plus the trial it fell on, if dismissed."""
    exposure_early = np.minimum(runs, cut).sum()
    deaths_early = int(((runs < cut) & out).sum())
    exposure_early += deaths_early
    settled = runs >= cut
    exposure_late = (runs[settled] - cut).sum()
    deaths_late = int((settled & out).sum())
    exposure_late += deaths_late
    early = deaths_early / exposure_early if exposure_early else None
    late = deaths_late / exposure_late if exposure_late else None
    return early, late


def weibull(runs: np.ndarray, out: np.ndarray) -> tuple[float | None, float | None]:
    """Weibull fit to runs-to-dismissal. Durations shifted by 1 because a duck has zero runs
    and the Weibull needs positive durations. Shape rho < 1 means the hazard decreases as the
    batter gets in; rho > 1 means it increases."""
    if out.sum() < 5:
        return None, None
    wf = WeibullFitter().fit(runs + 1.0, event_observed=out)
    return float(wf.rho_), float(wf.lambda_)


def summarise(inn: pl.DataFrame, k: float = PRIOR_DISMISSALS) -> Summary:
    if inn.height == 0:
        raise ValueError("no innings")
    runs = inn["runs"].to_numpy().astype(float)
    out = ~inn["not_out"].to_numpy()
    total, dismissals = int(runs.sum()), int(out.sum())
    era_mean = float(inn["era_mean"].mean())
    era_std = float(inn["era_std"].mean())
    raw = total / dismissals if dismissals else None
    eb = eb_average(total, dismissals, era_mean, k)

    hundreds = int((runs >= 100).sum())
    fifties = int(((runs >= 50) & (runs < 100)).sum())
    s50, s100 = km_survival(runs, out, 50), km_survival(runs, out, 100)
    rho, lam = weibull(runs, out)
    early, late = piecewise_hazard(runs, out)

    team = inn.filter(pl.col("team_runs") > 0)
    share = float(team["runs"].sum() / team["team_runs"].sum()) if team.height else None

    hostile_mask = pl.col("is_away") | (pl.col("innings_num") >= 3)
    hostile, friendly = inn.filter(hostile_mask), inn.filter(~hostile_mask)

    def _eb(part: pl.DataFrame) -> float | None:
        if part.height == 0:
            return None
        d = int((~part["not_out"]).sum())
        return eb_average(float(part["runs"].sum()), d, float(part["era_mean"].mean()), k)

    h_eb, f_eb = _eb(hostile), _eb(friendly)
    return Summary(
        player=inn["player"][0],
        country=inn["country"].mode()[0],
        first_year=int(inn["year"].min()),
        last_year=int(inn["year"].max()),
        innings=inn.height,
        not_outs=int(inn["not_out"].sum()),
        dismissals=dismissals,
        runs=total,
        hundreds=hundreds,
        fifties=fifties,
        raw_average=raw,
        eb_average=eb,
        era_mean=era_mean,
        era_std=era_std,
        z_raw=(raw - era_mean) / era_std if raw is not None else None,
        z_eb=(eb - era_mean) / era_std,
        km_s50=s50,
        km_s100=s100,
        km_conversion=s100 / s50 if s50 > 0 else None,
        naive_conversion=hundreds / (hundreds + fifties) if hundreds + fifties else None,
        weibull_rho=rho,
        weibull_lambda=lam,
        early_hazard=early,
        settled_hazard=late,
        team_share=share,
        hostile_eb=h_eb,
        friendly_eb=f_eb,
        hostile_ratio=h_eb / f_eb if h_eb and f_eb else None,
    )


def km_curve(inn: pl.DataFrame, max_runs: int = 300) -> pl.DataFrame:
    """S(r) = P(R >= r) on an integer grid, for plotting."""
    runs = inn["runs"].to_numpy().astype(float)
    out = ~inn["not_out"].to_numpy()
    grid = np.arange(0, max_runs + 1)
    s = np.nan_to_num(_km(runs, out).survival_function_at_times(grid - 0.5).to_numpy(), nan=1.0)
    return pl.DataFrame({"runs": grid, "survival": s})


def career_z_alltime(df: pl.DataFrame, player_id: int, min_innings: int = 20) -> float:
    """Charles Davis-style reference: z of a career average against the career averages of
    every batter with at least `min_innings` innings (any position, any era)."""
    careers = (
        df.group_by("player_id")
        .agg(runs=pl.col("runs").sum(), outs=(~pl.col("not_out")).sum(), inns=pl.len())
        .filter((pl.col("inns") >= min_innings) & (pl.col("outs") > 0))
        .with_columns(avg=pl.col("runs") / pl.col("outs"))
    )
    me = careers.filter(pl.col("player_id") == player_id)["avg"]
    if me.len() == 0 or math.isnan(float(careers["avg"].std())):
        raise ValueError("player not in qualified pool")
    return float((me[0] - careers["avg"].mean()) / careers["avg"].std())
