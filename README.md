# Bradman Check

Measures Test batters against Don Bradman using every men's Test innings (1877–2026, 94,046
batting innings, 2,637 matches) from ESPNcricinfo Statsguru. Not-outs are treated as censored
innings, small samples are shrunk toward the era, and pass/caution/fail thresholds come from
the historical distribution rather than hand-picked numbers.

```
uv sync
uv run python -m bradman fetch          # scrape + cache Statsguru (slow: ~4h first time)
uv run python -m bradman train          # fit models, player table, calibration (~1 min)
uv run python -m bradman show "DG Bradman"
uv run python -m bradman compare "DG Bradman" "SPD Smith" "V Kohli" "SR Tendulkar"
uv run streamlit run bradman/app.py
uv run pytest
```

## Pipeline

| Stage | File | What it does |
|---|---|---|
| Data | `bradman/data.py` | Scrapes batting innings (by batting position) and team innings (by home/away/neutral), joins them on opposition + ground + date + innings (100% join rate), computes **leave-one-out** rolling era baselines. |
| Engine | `bradman/engine.py` | Empirical Bayes average, Kaplan–Meier S(50), S(100) and conversion, Weibull shape ρ, piecewise hazard, hostile-conditions ratio. |
| Models | `bradman/models.py` | Censored Poisson era model (counterfactual), WinProbNet (PyTorch) vs gradient boosting, WPA. |
| Verdicts | `bradman/verdicts.py` | Five tests graded at the 90th / 70th percentile of 425 specialist batters (40+ innings, usual position ≤ 6). |
| UI | `bradman/cli.py`, `bradman/app.py` | Terminal tables; Streamlit with KM curves, verdicts, WPA, distributions. |

## Methods

**Era baseline.** For year Y, peers are top-6 batters with ≥20 innings in [Y−5, Y+5]; the
baseline is the mean and sd of their averages. The player being measured is removed from his
own peer group. Without this, Bradman inflates the 1930s baseline (only 13–24 peers) and his z
falls from 4.64 to 3.38.

**Empirical Bayes.** Runs-to-dismissal is modelled as exponential; total runs are the exposure
(not-outs included) and dismissals are the events. A Gamma prior worth k = 10 dismissals at the
era mean gives `(R + k·μ_era) / (D + k)`. That handles zero dismissals and shrinks flukes:
`80*, 15` goes from 95.0 to 40.5.

**Survival.** Kaplan–Meier with `event = dismissed`, so an unbeaten 75 leaves the risk set
instead of counting as a failed conversion. Conversion = S(100)/S(50). Weibull ρ < 1 means the
hazard falls once a batter is in. Bradman's ρ (0.79) is lower than Smith's, Kohli's and
Tendulkar's. Some greats are above 1: Sutcliffe is 1.16.

**Era model (Test 5).** `log h = player + era5 + opposition×decade + innings + position + away`,
fitted as a Poisson regression of the dismissal indicator with exposure runs + 1. This is the
censored exponential likelihood. Only the player effects are L2-penalised (context columns are
rescaled so they are effectively unpenalised), which is the MAP form of a Gaussian random-effects
model. α is picked by held-out log-likelihood (1e-5, interior to the grid). Counterfactual:
keep the player's effect and usual position, then average 1/h − 1 over the era, opposition, venue
and innings mix of Bradman's 80 innings. As a baseline, a statsmodels MixedLM on log(runs+1)
(which ignores censoring) finds player differences explain only ~3% of per-innings variance
(ICC 0.034). Single innings are mostly noise.

**WinProbNet.** MLP (8→32→32→1, SiLU, sigmoid, AdamW with weight decay, early stopping)
predicting P(team won | runs, rest-of-team runs, innings number, away, era mean). Whole matches
are held out for testing:

| | Brier | AUC |
|---|---|---|
| base rate | 0.2098 | — |
| HistGradientBoosting (monotone in runs) | 0.1997 | 0.654 |
| **WinProbNet** | **0.1947** | **0.674** |

WPA = P(win | actual runs) − P(win | era-mean runs), with the rest of the team held fixed.
Clutch WPA = mean WPA when the rest of the team made < 200.

## Results

```
                                      DG Bradman     SPD Smith       V Kohli  SR Tendulkar
raw average                                99.94         55.70         46.85         53.79
EB average (k=10)                          92.91         54.77         46.36         53.25
era peer mean ± sd                     43.6±12.1      36.5±8.6      36.7±8.8      37.3±9.9
z raw                                       4.64          2.25          1.15          1.67
1. Era-adjusted dominance            +4.06σ PASS   +2.14σ PASS   +1.10σ PASS   +1.62σ PASS
2. Conversion once set (KM)           72.6% PASS    49.5% CAUT    51.2% PASS    44.5% CAUT
3. Match-winning impact (WPA)        +3.00% PASS   +0.48% PASS   +0.29% CAUT   +0.45% PASS
4. Hostile-conditions resilience       1.13 CAUT     0.77 FAIL     0.73 FAIL     0.99 CAUT
5. Counterfactual 1928-48 average      82.7 PASS     64.0 PASS     55.5 PASS     61.3 PASS
```

Bradman's era z of **+4.64σ** (raw) is close to Charles Davis's published ~4.4σ, with nothing
tuned to match it. Pooling all eras and positions instead gives 5.6σ, because tailenders
widen the pool.

## Known limitations (worth discussing)

- **The Gaussian prior over-shrinks a true outlier.** The cross-validated random-effects
  prior pulls Bradman from 99.9 to ~81 in his own environment, while the Gamma EB keeps him at
  92.9. A heavy-tailed prior (Student-t random effects) would fix this.
- **WPA replaces innings with a constant.** A "par" batter scores era-mean runs in every
  innings, but real scores are skewed, so the median specialist has slightly negative WPA
  (Jensen's inequality). Percentile grading is unaffected; a distributional replacement
  (averaging over the era's empirical innings) would be cleaner.
- **The win model sees only the batting side's innings**, not the opponent's scores or the
  match state, so its AUC is modest (0.67).
- **Opposition strength is a decade-level category**, not a bowling-attack rating.
- **Data comes from Statsguru**, scraped politely with caching for personal research. It is
  not included in this repo; run `fetch` then `train`. Until then, the data and model tests skip.
