"""Command line: fetch data, train models, inspect players.

uv run python -m bradman fetch
uv run python -m bradman train
uv run python -m bradman show "DG Bradman"
uv run python -m bradman compare "DG Bradman" "SPD Smith" "V Kohli" "SR Tendulkar"
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import polars as pl

from bradman import data, models, verdicts

MARK = {"PASS": "PASS", "CAUTION": "CAUT", "FAIL": "FAIL", "OPEN": "OPEN"}


def find(table: pl.DataFrame, name: str) -> dict:
    hit = table.filter(pl.col("player") == name)
    if hit.height == 0:
        hit = table.filter(
            pl.col("player").str.to_lowercase().str.contains(name.lower(), literal=True)
        )
    if hit.height == 0:
        raise SystemExit(f"no player with {verdicts.MIN_TABLE_INNINGS}+ innings matching {name!r}")
    if hit.height > 1:
        opts = ", ".join(f"{r['player']} ({r['country']})" for r in hit.iter_rows(named=True))
        raise SystemExit(f"{name!r} is ambiguous: {opts}")
    return hit.row(0, named=True)


def cmd_train(_args) -> None:
    df = data.load()
    t0 = time.time()
    print(f"innings: {df.height:,}  players: {df['player_id'].n_unique():,}")
    em, era_report = models.fit_era_model(df)
    print(f"era model: alpha={era_report['alpha']}  ({time.time() - t0:.0f}s)")
    try:
        era_report["gaussian_mixed"] = models.fit_gaussian_mixed(df)
        print(f"gaussian mixed baseline: {era_report['gaussian_mixed']}")
    except Exception as exc:  # the baseline is for discussion only; never block training
        era_report["gaussian_mixed"] = {"error": repr(exc)}
    net, win_report = models.train_winprob(df)
    print("winprob:", json.dumps(win_report, indent=1))
    models.save(em, net, {"era_model": era_report, "winprob": win_report})
    table = verdicts.build_tables(df, em, net)
    cal = verdicts.calibrate(table)
    print(f"player table: {table.height} players; calibration pool {cal['pool_size']}")
    print(f"done in {time.time() - t0:.0f}s")


def _header(r: dict) -> str:
    return (
        f"{r['player']} ({r['country']}, {r['first_year']}-{r['last_year']})  "
        f"{r['innings']} inns, {r['not_outs']} NO, {r['runs']:,} runs, "
        f"avg {verdicts._f(r['raw_average'], 2)}, "
        f"{r['hundreds']}x100, {r['fifties']}x50"
    )


def _fmt(metric: str, v) -> str:
    if v is None:
        return "n/a"
    if metric == "z_eb":
        return f"{v:+.2f}σ"
    if metric == "km_conversion":
        return f"{v:.1%}"
    if metric == "wpa_mean":
        return f"{v * 100:+.2f}%"
    if metric == "hostile_ratio":
        return f"{v:.2f}"
    return f"{v:.1f}"


def cmd_show(args) -> None:
    table, cal = verdicts.load_tables()
    r = find(table, args.player)
    print(_header(r))
    print(
        f"  EB avg {r['eb_average']:.2f} | era peers {r['era_mean']:.1f} ± {r['era_std']:.1f} | "
        f"z raw {verdicts._f(r['z_raw'], 2)} | "
        f"KM S(50) {r['km_s50']:.1%} S(100) {r['km_s100']:.1%} | "
        f"team share {verdicts._pct(r['team_share'])}"
    )
    print()
    print(f"  {'#':<2} {'test':<40} {'value':>9} {'Bradman':>9} {'pctile':>7}  verdict")
    for t in verdicts.grade(r, cal):
        pct = "n/a" if t.percentile is None else f"{t.percentile:.0f}"
        print(
            f"  {t.number:<2} {t.name:<40} {_fmt(t.metric, t.value):>9} "
            f"{_fmt(t.metric, t.bradman):>9} {pct:>7}  {t.verdict}"
        )
        print(f"     {t.detail}")
    print(
        f"\n  thresholds: PASS ≥ {verdicts.PASS_PCT}th, CAUTION ≥ {verdicts.CAUTION_PCT}th "
        f"percentile of {cal['pool_size']} specialist batters ({verdicts.POOL_MIN_INNINGS}+ inns)"
    )


def cmd_compare(args) -> None:
    table, cal = verdicts.load_tables()
    rows = [find(table, n) for n in args.players]
    w = max(12, *(len(r["player"]) for r in rows)) + 2
    print(f"{'':<34}" + "".join(f"{r['player']:>{w}}" for r in rows))
    lines = [
        ("years", lambda r: f"{r['first_year']}-{r['last_year']}"),
        ("innings", lambda r: str(r["innings"])),
        ("runs", lambda r: f"{r['runs']:,}"),
        ("raw average", lambda r: verdicts._f(r["raw_average"], 2)),
        ("EB average (k=10)", lambda r: f"{r['eb_average']:.2f}"),
        ("era peer mean ± sd", lambda r: f"{r['era_mean']:.1f}±{r['era_std']:.1f}"),
        ("z raw", lambda r: verdicts._f(r["z_raw"], 2)),
        ("KM S(50)", lambda r: f"{r['km_s50']:.1%}"),
        ("Weibull rho", lambda r: verdicts._f(r["weibull_rho"], 3)),
        ("naive 50->100", lambda r: verdicts._pct(r["naive_conversion"])),
        ("clutch WPA (rest<200)", lambda r: verdicts._signed_pct(r["clutch_wpa"]) + "%"),
        ("team run share", lambda r: verdicts._pct(r["team_share"])),
    ]
    for label, fn in lines:
        print(f"{label:<34}" + "".join(f"{fn(r):>{w}}" for r in rows))
    print()
    graded = [verdicts.grade(r, cal) for r in rows]
    for i, (num, name, metric) in enumerate(verdicts.TESTS):
        cells = "".join(
            f"{_fmt(metric, g[i].value) + ' ' + MARK[g[i].verdict]:>{w}}" for g in graded
        )
        print(f"{num}. {name[:31]:<31}" + cells)
    passes = "".join(f"{sum(t.verdict == 'PASS' for t in g)}/5 PASS".rjust(w) for g in graded)
    print(f"{'':<34}" + passes)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="bradman", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="scrape Statsguru into data/test_innings.parquet")
    sub.add_parser("train", help="fit models, build player table and calibration")
    s = sub.add_parser("show", help="five Bradman tests for one player")
    s.add_argument("player")
    c = sub.add_parser("compare", help="side-by-side comparison")
    c.add_argument("players", nargs="+")
    args = p.parse_args(argv)
    {
        "fetch": lambda a: data.build(),
        "train": cmd_train,
        "show": cmd_show,
        "compare": cmd_compare,
    }[args.cmd](args)


if __name__ == "__main__":
    main(sys.argv[1:])
