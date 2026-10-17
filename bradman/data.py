"""Data layer: every men's Test innings from ESPNcricinfo Statsguru, joined to team totals
and results, plus era baselines computed from the data itself.

Pipeline:
    Statsguru HTML pages (cached gzip, fetched politely)
      -> batting innings (player, runs, not out, position, ...)   one row per batter innings
      -> team innings    (team total, result, home/away)          one row per team innings
      -> joined innings table  -> data/test_innings.parquet
      -> era baselines   (rolling top-6 mean/std per year)        -> era_mean, era_std per innings
"""

from __future__ import annotations

import gzip
import hashlib
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import httpx
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INNINGS_PATH = DATA_DIR / "test_innings.parquet"

BASE = "https://stats.espncricinfo.com/ci/engine/stats/index.html?class=1;template=results;size=200"
USER_AGENT = "Mozilla/5.0 (Macintosh) bradman-check research scraper (personal, cached)"
DELAY_SECONDS = 1.0  # per worker, between live requests; pages are cached so this runs once
WORKERS = 4

# Era baselines: peers are top-order (positions 1-6) innings within +/- ERA_HALF_WINDOW years,
# and only batters with at least MIN_PEER_INNINGS innings in that window count as a "peer".
ERA_HALF_WINDOW = 5
MIN_PEER_INNINGS = 20
TOP_ORDER = 6


# --------------------------------------------------------------------------- fetching


def _cache_path(url: str) -> Path:
    return RAW_DIR / f"{hashlib.sha1(url.encode()).hexdigest()}.html.gz"


def fetch(url: str, client: httpx.Client) -> str:
    """GET a Statsguru page, from disk cache if we have it. Retries with backoff because
    Statsguru answers bursts with a 503 'Site under maintenance' page."""
    path = _cache_path(url)
    if path.exists():
        return gzip.decompress(path.read_bytes()).decode()
    for attempt in range(8):
        time.sleep(DELAY_SECONDS)
        try:
            resp = client.get(url)
            if resp.status_code == 200 and "engineTable" in resp.text:
                RAW_DIR.mkdir(parents=True, exist_ok=True)
                path.write_bytes(gzip.compress(resp.text.encode()))
                return resp.text
            reason = f"HTTP {resp.status_code}"
        except httpx.HTTPError as exc:
            reason = repr(exc)
        wait = min(30 * 2**attempt, 600)
        print(f"  {reason}; backing off {wait}s", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"giving up on {url}")


def _page_count(html: str) -> int:
    m = re.search(r"Page <b>\d+</b> of <b>(\d+)</b>", html)
    return int(m.group(1)) if m else 1


def _rows(html: str) -> list[str]:
    """The results table is the last engineTable containing data rows."""
    tables = [t for t in html.split('<table class="engineTable">') if 'class="data1"' in t]
    return re.findall(r'<tr class="data1">(.*?)</tr>', tables[-1], re.S) if tables else []


def _cells(row: str) -> list[str]:
    return re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)


def _text(cell: str) -> str:
    return re.sub(r"<[^>]+>", "", cell).strip()


def _link_id(cell: str, kind: str) -> int | None:
    m = re.search(rf"/ci/content/{kind}/(\d+)\.html", cell)
    return int(m.group(1)) if m else None


def _crawl(queries: dict[str, str], client: httpx.Client) -> dict[str, list[str]]:
    """Fetch every page of every query. Page 1 first (to learn page counts), then the rest
    with WORKERS concurrent requests - Statsguru takes ~0.2s per row server side."""
    with ThreadPoolExecutor(WORKERS) as pool:
        firsts = dict(
            zip(
                queries,
                pool.map(lambda q: fetch(f"{BASE};{q};page=1", client), queries.values()),
                strict=True,
            )
        )
        urls = [
            (label, page, f"{BASE};{q};page={page}")
            for label, q in queries.items()
            for page in range(2, _page_count(firsts[label]) + 1)
        ]
        print(f"  {len(queries)} queries, {len(urls) + len(queries)} pages", flush=True)
        done = 0

        def _get(item):
            nonlocal done
            html = fetch(item[2], client)
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(urls)} pages", flush=True)
            return html

        pages = list(pool.map(_get, urls))
    out = {label: _rows(html) for label, html in firsts.items()}
    for (label, _, _), html in zip(urls, pages, strict=True):
        out[label] += _rows(html)
    for label, rows in out.items():
        print(f"  {label}: {len(rows)} rows", flush=True)
    return out


# --------------------------------------------------------------------------- parsing


def _date(s: str) -> datetime:
    return datetime.strptime(s, "%d %b %Y")


def parse_batting_row(row: str, position: int) -> dict | None:
    c = _cells(row)
    if len(c) < 12:
        return None
    score = _text(c[1])
    m = re.fullmatch(r"(\d+)(\*?)", score)
    if not m:  # DNB, TDNB, absent, sub
        return None
    name_country = _text(c[0])
    nm = re.fullmatch(r"(.*?) \(([^)]*)\)", name_country)
    balls = _text(c[3])
    return {
        "player_id": _link_id(c[0], "player"),
        "player": nm.group(1) if nm else name_country,
        "country_code": nm.group(2) if nm else None,
        "runs": int(m.group(1)),
        "not_out": m.group(2) == "*",
        "balls": int(balls) if balls.isdigit() else None,
        "batting_position": position,
        "innings_num": int(_text(c[7])),
        "opposition": _text(c[9]).removeprefix("v ").strip(),
        "opposition_id": _link_id(c[9], "team"),
        "ground": _text(c[10]),
        "ground_id": _link_id(c[10], "ground"),
        "match_date": _date(_text(c[11])),
    }


def parse_team_row(row: str, venue: str) -> dict | None:
    c = _cells(row)
    if len(c) < 11:
        return None
    score = _text(c[1])
    m = re.match(r"(\d+)", score)
    if not m:  # DNB / forfeited
        return None
    return {
        "team": _text(c[0]),
        "team_id": _link_id(c[0], "team"),
        "team_runs": int(m.group(1)),
        "team_wickets": int(w.group(1)) if (w := re.search(r"/(\d+)", score)) else 10,
        "declared": score.endswith("d"),
        "innings_num": int(_text(c[5])),
        "result": _text(c[6]),
        "opposition_id": _link_id(c[8], "team"),
        "ground_id": _link_id(c[9], "ground"),
        "match_date": _date(_text(c[10])),
        "venue": venue,
    }


# --------------------------------------------------------------------------- build


def scrape() -> tuple[pl.DataFrame, pl.DataFrame]:
    queries = {
        f"team/{venue}": f"type=team;view=innings;home_or_away={code}"
        for venue, code in (("home", 1), ("away", 2), ("neutral", 3))
    } | {
        f"batting/{pos}": (
            f"type=batting;view=innings;batting_positionmin1={pos};"
            f"batting_positionmax1={pos};batting_positionval1=batting_position"
        )
        for pos in range(1, 12)
    }
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=180, follow_redirects=True) as cl:
        rows = _crawl(queries, cl)
    teams = [
        rec
        for label, rs in rows.items() if label.startswith("team/")
        for r in rs if (rec := parse_team_row(r, label.split("/")[1])) is not None
    ]  # fmt: skip
    batting = [
        rec
        for label, rs in rows.items() if label.startswith("batting/")
        for r in rs if (rec := parse_batting_row(r, int(label.split("/")[1]))) is not None
    ]  # fmt: skip
    return pl.DataFrame(batting), pl.DataFrame(teams)


def join_innings(batting: pl.DataFrame, teams: pl.DataFrame) -> pl.DataFrame:
    """Attach the batting side's team total, result and venue to every batter innings.
    A batter's 'opposition' is the team innings row's 'opposition' for the same match innings."""
    key = ["opposition_id", "ground_id", "match_date", "innings_num"]
    teams = teams.unique(subset=key, keep="first")
    df = batting.join(teams, on=key, how="inner")
    # Each match: identify by both teams + ground + start date.
    lo = pl.min_horizontal("team_id", "opposition_id")
    hi = pl.max_horizontal("team_id", "opposition_id")
    return (
        df.with_columns(
            match_key=pl.concat_str(
                [lo, hi, pl.col("ground_id"), pl.col("match_date").dt.strftime("%Y%m%d")],
                separator="-",
            ),
            year=pl.col("match_date").dt.year(),
            is_away=pl.col("venue") != "home",
            team_won=pl.col("result") == "won",
            country=pl.col("team"),
        )
        .with_columns(rest_of_team_runs=pl.col("team_runs") - pl.col("runs"))
        .sort("match_date", "innings_num", "batting_position")
    )


def era_baselines(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per calendar year Y, the peer group is every established top-order batter over
    [Y - w, Y + w]: at least MIN_PEER_INNINGS top-6 innings in the window. Returns
      years:   Y -> n, sum and sum of squares of peer averages, pooled runs and outs
      members: (Y, player) -> that player's own window average, runs, outs
    so attach_eras() can take each player *out* of his own baseline. Computed from the data,
    never hardcoded."""
    top = df.filter(pl.col("batting_position") <= TOP_ORDER).select(
        "player_id", "year", "runs", pl.col("not_out").not_().cast(pl.Int32).alias("out")
    )
    per_year = top.group_by("player_id", "year").agg(
        runs=pl.col("runs").sum(), outs=pl.col("out").sum(), inns=pl.len()
    )
    years, members = [], []
    for y in range(int(df["year"].min()), int(df["year"].max()) + 1):
        w = (
            per_year.filter(pl.col("year").is_between(y - ERA_HALF_WINDOW, y + ERA_HALF_WINDOW))
            .group_by("player_id")
            .agg(pl.col("runs").sum(), pl.col("outs").sum(), pl.col("inns").sum())
            .filter((pl.col("inns") >= MIN_PEER_INNINGS) & (pl.col("outs") > 0))
            .with_columns(avg=pl.col("runs") / pl.col("outs"))
        )
        if w.height >= 6:
            years.append(
                {
                    "year": y,
                    "n": w.height,
                    "s1": float(w["avg"].sum()),
                    "s2": float((w["avg"] ** 2).sum()),
                    "pooled_runs": float(w["runs"].sum()),
                    "pooled_outs": float(w["outs"].sum()),
                }
            )
            members.append(
                w.select("player_id", "avg", "runs", "outs").with_columns(year=pl.lit(y))
            )
    return pl.DataFrame(years), pl.concat(members).rename(
        {"avg": "self_avg", "runs": "self_runs", "outs": "self_outs"}
    )


def attach_eras(df: pl.DataFrame) -> pl.DataFrame:
    """era_mean / era_std / era_pooled for every innings, leave-one-out: a great batter is
    never part of the yardstick he is measured against (Bradman alone moves the 1930s peer
    mean by ~2 runs and its std by ~4). Years without enough peers borrow the nearest year."""
    years, members = era_baselines(df)
    all_years = pl.DataFrame({"year": range(int(df["year"].min()), int(df["year"].max()) + 1)})
    years = (
        all_years.join(years, on="year", how="left")
        .sort("year")
        .with_columns(
            pl.exclude("year").fill_null(strategy="forward").fill_null(strategy="backward")
        )
    )
    j = (
        df.drop("era_mean", "era_std", "era_pooled", "era_peers", strict=False)
        .join(years, on="year", how="left")
        .join(members, on=["player_id", "year"], how="left")
    )
    inside = pl.col("self_avg").is_not_null()
    n = pl.when(inside).then(pl.col("n") - 1).otherwise(pl.col("n"))
    s1 = pl.when(inside).then(pl.col("s1") - pl.col("self_avg")).otherwise(pl.col("s1"))
    s2 = pl.when(inside).then(pl.col("s2") - pl.col("self_avg") ** 2).otherwise(pl.col("s2"))
    runs = pl.col("pooled_runs") - pl.col("self_runs").fill_null(0)
    outs = pl.col("pooled_outs") - pl.col("self_outs").fill_null(0)
    return j.with_columns(
        era_mean=s1 / n,
        era_std=((s2 - s1**2 / n) / (n - 1)).sqrt(),
        era_pooled=runs / outs,
        era_peers=n,
    ).drop("n", "s1", "s2", "pooled_runs", "pooled_outs", "self_avg", "self_runs", "self_outs")


def build() -> pl.DataFrame:
    batting, teams = scrape()
    df = join_innings(batting, teams)
    df = attach_eras(df)
    cols = [
        "player_id", "player", "country", "country_code", "opposition", "ground", "venue",
        "is_away", "match_key", "match_date", "year", "innings_num", "batting_position", "runs",
        "balls", "not_out", "team_runs", "team_wickets", "declared", "rest_of_team_runs",
        "result", "team_won", "era_mean", "era_std", "era_pooled", "era_peers",
    ]  # fmt: skip
    df = df.select(cols)
    DATA_DIR.mkdir(exist_ok=True)
    df.write_parquet(INNINGS_PATH)
    print(
        f"batting rows {batting.height}, team rows {teams.height}, joined {df.height} "
        f"({df.height / batting.height:.2%}) -> {INNINGS_PATH}"
    )
    return df


def load() -> pl.DataFrame:
    if not INNINGS_PATH.exists():
        raise FileNotFoundError(f"{INNINGS_PATH} missing; run `uv run python -m bradman fetch`")
    return pl.read_parquet(INNINGS_PATH)


def player_innings(df: pl.DataFrame, name: str) -> pl.DataFrame:
    """Exact name match first (Statsguru style, e.g. 'SPD Smith'), else case-insensitive
    substring. Raises if ambiguous so we never silently mix two players."""
    exact = df.filter(pl.col("player") == name)
    if exact.height:
        ids = exact["player_id"].unique()
    else:
        ids = df.filter(
            pl.col("player").str.to_lowercase().str.contains(name.lower(), literal=True)
        )["player_id"].unique()
    if len(ids) == 0:
        raise KeyError(f"no player matching {name!r}")
    if len(ids) > 1:
        names = df.filter(pl.col("player_id").is_in(ids))["player"].unique().to_list()
        raise KeyError(f"{name!r} is ambiguous: {sorted(names)[:10]}")
    return df.filter(pl.col("player_id") == ids[0]).sort("match_date", "innings_num")


if __name__ == "__main__":
    build()
