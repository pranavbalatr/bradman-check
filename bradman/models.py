"""Two models trained on the full Test innings table.

1. Era model (counterfactual translation). Batting is a hazard process: every run is a trial
   the batter survives with hazard h. We fit

       log h_i = b_player + b_era5 + b_opposition x decade + b_away + b_innings + b_position

   as a Poisson regression of the dismissal indicator on log(exposure) - which is exactly the
   censored exponential survival likelihood, so not-outs are handled correctly. L2 shrinkage on
   all coefficients is the MAP version of a Gaussian random-effects model (players with few
   innings are pulled toward zero = average). Translating a player to 1928-48 means: keep his
   b_player and position, swap in the era/opposition/venue/innings mix of Bradman's 80 innings.

   For contrast we also fit the textbook Gaussian mixed model on log(runs + 1)
   (statsmodels MixedLM), which ignores censoring - see report["era_model"]["gaussian_mixed"].

2. WinProbNet (PyTorch). P(team won | batter runs, rest-of-team runs, innings number, away,
   era mean). Win Probability Added = P(actual runs) - P(era-average runs), everything else
   held fixed. Benchmarked on Brier score against HistGradientBoostingClassifier.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass

import numpy as np
import polars as pl
import scipy.sparse as sp
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import brier_score_loss, roc_auc_score
from torch import nn

from bradman.data import DATA_DIR

MODEL_DIR = DATA_DIR / "models"
REPORT_PATH = MODEL_DIR / "report.json"
BRADMAN_YEARS = (1928, 1948)
SEED = 7
MAX_POSITION = 7  # era model trains on batters and keeper-batters, not the tail
MIN_PLAYER_INNINGS = 10  # below this a player shares the "other" effect

# --------------------------------------------------------------------------- era model


def _design_frame(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        era5=(pl.col("year") // 5 * 5).cast(pl.Utf8),
        opp_decade=pl.col("opposition") + "-" + (pl.col("year") // 10 * 10).cast(pl.Utf8),
        inns=pl.col("innings_num").cast(pl.Utf8),
        pos=pl.col("batting_position").cast(pl.Utf8),
        away=pl.col("is_away").cast(pl.Float64),
        exposure=pl.col("runs").cast(pl.Float64) + 1.0,
        out=(~pl.col("not_out")).cast(pl.Float64),
    )


@dataclass
class EraModel:
    vocab: dict[str, dict[str, int]]  # per categorical column: level -> column index
    n_features: int
    reg: PoissonRegressor
    alpha: float

    CATS = ("player_key", "era5", "opp_decade", "inns", "pos")
    # sklearn applies one L2 penalty to every coefficient. Scaling a column by c divides its
    # effective penalty by c^2, so context columns are scaled up to be (almost) unpenalised
    # and only the player effects are shrunk - a random-effects prior on players alone.
    CONTEXT_SCALE = 100.0

    def matrix(self, d: pl.DataFrame) -> sp.csr_matrix:
        rows, cols = [], []
        n = d.height
        vals = []
        for c in self.CATS:
            idx = np.array([self.vocab[c].get(v, -1) for v in d[c].to_list()])
            ok = idx >= 0  # unseen level -> all zeros -> baseline
            rows.append(np.arange(n)[ok])
            cols.append(idx[ok])
            vals.append(np.full(ok.sum(), 1.0 if c == "player_key" else self.CONTEXT_SCALE))
        r, c, v = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
        m = sp.csr_matrix((v, (r, c)), shape=(n, self.n_features))
        away = sp.csr_matrix(d["away"].to_numpy().reshape(-1, 1) * self.CONTEXT_SCALE)
        return sp.hstack([m, away]).tocsr()

    def hazard(self, d: pl.DataFrame) -> np.ndarray:
        return self.reg.predict(self.matrix(d))

    def expected_average(self, d: pl.DataFrame) -> float:
        """Runs per dismissal if every context in `d` ends in a dismissal. Exposure was runs+1,
        so expected runs per innings is 1/h - 1."""
        return float(np.mean(1.0 / self.hazard(d) - 1.0))


def _player_keys(df: pl.DataFrame) -> pl.DataFrame:
    counts = df.group_by("player_id").len()
    keep = counts.filter(pl.col("len") >= MIN_PLAYER_INNINGS)["player_id"]
    return df.with_columns(
        player_key=pl.when(pl.col("player_id").is_in(keep.implode()))
        .then(pl.col("player_id").cast(pl.Utf8))
        .otherwise(pl.lit("other"))
    )


def fit_era_model(
    df: pl.DataFrame, alphas=(1e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4)
) -> tuple[EraModel, dict]:
    d = _player_keys(_design_frame(df.filter(pl.col("batting_position") <= MAX_POSITION)))
    vocab, offset = {}, 0
    for c in EraModel.CATS:
        levels = sorted(d[c].unique().to_list())
        vocab[c] = {v: offset + i for i, v in enumerate(levels)}
        offset += len(levels)
    rng = np.random.default_rng(SEED)
    holdout = rng.random(d.height) < 0.2
    y = (d["out"] / d["exposure"]).to_numpy()
    w = d["exposure"].to_numpy()

    def _fit(alpha, mask):
        em = EraModel(
            vocab,
            offset,
            PoissonRegressor(alpha=alpha, solver="newton-cholesky", tol=1e-9, max_iter=1000),
            alpha,
        )
        X = em.matrix(d)
        em.reg.fit(X[mask], y[mask], sample_weight=w[mask])
        return em, X

    scores = {}
    for a in alphas:
        em, X = _fit(a, ~holdout)
        mu = em.reg.predict(X[holdout])
        # Held-out censored-exponential log likelihood per innings: d log h - h * exposure
        ll = d["out"].to_numpy()[holdout] * np.log(mu) - mu * w[holdout]
        scores[a] = float(ll.mean())
    best = max(scores, key=scores.get)
    em, _ = _fit(best, np.ones(d.height, bool))
    report = {
        "rows": d.height,
        "players": len(vocab["player_key"]),
        "alpha_heldout_loglik": {str(k): v for k, v in scores.items()},
        "alpha": best,
    }
    return em, report


def fit_gaussian_mixed(df: pl.DataFrame) -> dict:
    """Baseline for discussion: log(runs+1) with a player random intercept. It treats 99* as
    a completed 99, so it under-rates batters who are often left not out."""
    import statsmodels.formula.api as smf

    d = _design_frame(df.filter(pl.col("batting_position") <= 6)).with_columns(
        lr=(pl.col("runs") + 1).log()
    )
    pdf = d.select("lr", "era5", "inns", "away", "player_id").to_pandas()
    res = smf.mixedlm("lr ~ C(era5) + C(inns) + away", pdf, groups=pdf["player_id"]).fit(
        method="lbfgs"
    )
    re_var = float(res.cov_re.iloc[0, 0])
    return {
        "rows": len(pdf),
        "player_re_sd": re_var**0.5,
        "residual_sd": float(res.scale) ** 0.5,
        "icc": re_var / (re_var + float(res.scale)),
        "away_coef": float(res.params["away"]),
    }


def bradman_contexts(df: pl.DataFrame) -> pl.DataFrame:
    """The environment of Bradman's 80 innings: era, opposition-decade, venue, match innings."""
    bradman = df.filter(pl.col("player") == "DG Bradman")
    assert bradman.height == 80, f"expected Bradman's 80 innings, found {bradman.height}"
    return _design_frame(bradman)


def counterfactual_average(em: EraModel, contexts: pl.DataFrame, player_id: int, pos: int):
    ctx = contexts.with_columns(
        player_key=pl.lit(str(player_id)), pos=pl.lit(str(min(pos, MAX_POSITION)))
    )
    return em.expected_average(ctx)


# --------------------------------------------------------------------------- win prob


WIN_FEATURES = ("runs", "rest_of_team_runs", "innings_num", "is_away", "era_mean")


def win_features(runs, rest, innings_num, is_away, era_mean) -> np.ndarray:
    """Scaled inputs: runs and rest-of-team in hundreds, innings one-hot, era in 40s."""
    runs, rest = np.asarray(runs, float), np.asarray(rest, float)
    inns = np.asarray(innings_num, int)
    onehot = np.eye(4)[np.clip(inns, 1, 4) - 1]
    return np.column_stack(
        [runs / 100, rest / 100, onehot, np.asarray(is_away, float), np.asarray(era_mean) / 40]
    ).astype(np.float32)


def _frame_features(d: pl.DataFrame, runs=None) -> np.ndarray:
    return win_features(
        d["runs"].to_numpy() if runs is None else runs,
        d["rest_of_team_runs"].to_numpy(),
        d["innings_num"].to_numpy(),
        d["is_away"].to_numpy(),
        d["era_mean"].to_numpy(),
    )


class WinProbNet(nn.Module):
    def __init__(self, n_in: int = 8, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )  # fmt: skip

    def forward(self, x):
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def prob(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        return self(torch.from_numpy(x)).numpy()


def _match_split(df: pl.DataFrame, frac: float = 0.2) -> np.ndarray:
    """Hold out whole matches so the same result never appears on both sides."""
    keys = df["match_key"].unique().sort().to_numpy()
    rng = np.random.default_rng(SEED)
    test_keys = set(rng.choice(keys, int(len(keys) * frac), replace=False))
    return np.array([k in test_keys for k in df["match_key"].to_list()])


def train_winprob(df: pl.DataFrame, epochs: int = 40) -> tuple[WinProbNet, dict]:
    torch.manual_seed(SEED)
    test = _match_split(df)
    X = _frame_features(df)
    y = df["team_won"].cast(pl.Float32).to_numpy()
    Xtr, ytr, Xte, yte = X[~test], y[~test], X[test], y[test]

    # carve a validation set out of train for early stopping
    rng = np.random.default_rng(SEED)
    val = rng.random(len(Xtr)) < 0.1
    tX, ty = torch.from_numpy(Xtr[~val]), torch.from_numpy(ytr[~val])
    vX, vy = torch.from_numpy(Xtr[val]), torch.from_numpy(ytr[val])

    net = WinProbNet(X.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-3)
    loss_fn = nn.BCELoss()
    best, best_state, patience = float("inf"), None, 0
    for _epoch in range(epochs):
        net.train()
        perm = torch.randperm(len(tX))
        for i in range(0, len(tX), 1024):
            b = perm[i : i + 1024]
            opt.zero_grad()
            loss_fn(net(tX[b]), ty[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = loss_fn(net(vX), vy).item()
        if vl < best - 1e-5:
            best, best_state, patience = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 5:
                break
    net.load_state_dict(best_state)

    # monotone in runs and rest-of-team runs: more runs can never hurt the win chance
    hgb = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, monotonic_cst=[1, 1, 0, 0, 0, 0, 0, 0], random_state=SEED
    ).fit(Xtr, ytr)

    p_net, p_hgb = net.prob(Xte), hgb.predict_proba(Xte)[:, 1]
    report = {
        "train_rows": int((~test).sum()),
        "test_rows": int(test.sum()),
        "base_rate": float(ytr.mean()),
        "brier_baseline": float(brier_score_loss(yte, np.full_like(yte, ytr.mean()))),
        "brier_winprobnet": float(brier_score_loss(yte, p_net)),
        "brier_hgb": float(brier_score_loss(yte, p_hgb)),
        "auc_winprobnet": float(roc_auc_score(yte, p_net)),
        "auc_hgb": float(roc_auc_score(yte, p_hgb)),
        "epochs_run": _epoch + 1,
    }
    return net, report


def wpa(net: WinProbNet, df: pl.DataFrame) -> np.ndarray:
    """Per innings: P(win | actual runs) - P(win | era-average runs), rest of team fixed."""
    actual = net.prob(_frame_features(df))
    replacement = net.prob(_frame_features(df, runs=df["era_mean"].to_numpy()))
    return actual - replacement


# --------------------------------------------------------------------------- persistence


def save(em: EraModel, net: WinProbNet, report: dict) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "era_model.pkl", "wb") as f:
        pickle.dump(em, f)
    torch.save(net.state_dict(), MODEL_DIR / "winprobnet.pt")
    REPORT_PATH.write_text(json.dumps(report, indent=2))


def load_models() -> tuple[EraModel, WinProbNet]:
    with open(MODEL_DIR / "era_model.pkl", "rb") as f:
        em = pickle.load(f)
    net = WinProbNet()
    net.load_state_dict(torch.load(MODEL_DIR / "winprobnet.pt"))
    return em, net
