"""Build the static stats site: Parquet data plus one page that queries it in the browser.

The page (site/index.html) runs DuckDB-WASM, which reads only the parts of each Parquet
file a query needs (HTTP range requests), so the million-row every-hand tables can be
looked up without downloading them. Output goes to site/_build/ (index.html and data/):

  presets.parquet    one row per house-rule preset (bids, points, a one-line summary, notes)
  charts.parquet     per preset, players, trump honors and small trumps: hands, average
                     points, highest bid with expected points >= 0 (charts/*.csv)
  chart_ev.parquet   the same groups, one row per bid: expected points of that bid (ev) and
                     net of the opponents' points on the hand (net)
  condensed.parquet  4-point Setback, from the every-hand tables (condensed_*.csv)
  hand_groups.parquet.zip, group_stats.parquet.zip
                     every distinct 6-card hand as bidder, for each preset with reduced
                     every-hand tables (allhands-r_<preset>_side_<P>p_d<deals>_s<S>.csv.gz,
                     allhands.py --reduced): hand_groups maps each hand (preset, players, key)
                     to its trump and group; group_stats has each group's numbers (preset,
                     players, search, group). Copied as is if DIR already has both .parquet
                     files (as a release would carry them).
  allhands.parquet.zip
                     the older single table: every distinct 6-card hand, 4-point Setback
                     (release tables allhands_*.csv.gz, or a prebuilt allhands.parquet), used
                     when there are no reduced tables

    uv run --extra site python site/build.py [--tables DIR]
"""
import argparse
import glob
import html
import json
import os
import re
import shutil
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from cardsim.pitch.notes import COMMON, NOTES  # noqa: E402
from cardsim.pitch.rules import PRESETS, describe  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_build")
SUITS = "SHDC"


# GitHub Pages gzips most responses on the fly and then serves byte ranges of the gzipped
# copy, which breaks range reads. It leaves .zip (and .gz, .png) alone, so the big table,
# the one file the page reads by range, is published under a .zip name. It is plain
# Parquet inside; the release carries it as allhands.parquet.
ALLHANDS = "allhands.parquet.zip"


def write(df, name, row_group_size=None):
    """Parquet with zstd; row groups small enough that a lookup reads little."""
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), os.path.join(OUT, "data", name),
                   compression="zstd", row_group_size=row_group_size)
    return len(df)


def presets():
    rows = []
    for name, r in PRESETS.items():
        from cardsim.pitch.lines import LINES
        rows.append(dict(preset=name, summary=describe(r), notes=NOTES[name], min_bid=r.min_bid, max_bid=r.max_bid,
                         line4=LINES.get((name, 4)), line6=LINES.get((name, 6)),
                         moon_bid=r.moon_bid, moon_value=r.moon_value, points_total=r.points_total,
                         target=r.target, deal_size=r.deal_size, hand_size=r.hand_size, jokers=r.jokers,
                         off_jack=r.off_jack, may_trump=r.may_trump, low_to_captor=r.low_to_captor))
    return pd.DataFrame(rows)


def charts():
    """charts/<preset>_<n>p.csv -> one wide table (without the ev columns) and one long one."""
    wide, long = [], []
    for fn in sorted(glob.glob(os.path.join(ROOT, "charts", "*_[46]p.csv"))):
        preset, n = re.match(r"(.+)_(\d)p\.csv$", os.path.basename(fn)).groups()
        df = pd.read_csv(fn, keep_default_na=False)  # "" is the no-honors label
        for c in df.columns:  # numbers left blank (inferred cells) -> NaN
            if c not in ("honors", "source"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df.insert(0, "players", int(n))
        df.insert(0, "preset", preset)
        evs = [c for c in df.columns if re.fullmatch(r"ev\d+", c)]
        keep = ["preset", "players", "honors", "small", "hands", "avg", "opp", "p_moon", "max_bid"]
        w = df[keep].copy()
        w["bid_line"] = df["bid_line"] if "bid_line" in df else df["max_bid"]
        w["inferred"] = df["inferred"] if "inferred" in df else 0
        w["source"] = df["source"] if "source" in df else "survey"
        w["ev_moon"] = df["ev_moon"] if "ev_moon" in df else None
        w["net_moon"] = df["net_moon"] if "net_moon" in df else None
        wide.append(w)
        ids = ["preset", "players", "honors", "small", "hands"]
        lg = df.melt(id_vars=ids, value_vars=evs, var_name="bid", value_name="ev")
        lg["net"] = df.melt(id_vars=ids, value_vars=["net" + c[2:] for c in evs])["value"]
        lg["bid"] = lg["bid"].str[2:].astype(int)
        long.append(lg)
    wide = pd.concat(wide, ignore_index=True)
    wide["n_honors"] = wide["honors"].str.split().str.len().fillna(0).astype(int)
    return wide, pd.concat(long, ignore_index=True).sort_values(["preset", "players", "honors", "small", "bid"])


def condensed():
    out = []
    for n in (4, 6):
        df = pd.read_csv(os.path.join(ROOT, f"condensed_{n}p.csv"), keep_default_na=False)
        df = df.rename(columns={"hon": "honors", "nx": "small", "w": "hands_weight"})
        df["honors"] = df["honors"].map(lambda h: " ".join("10" if c == "T" else c for c in h))
        df.insert(0, "players", n)
        out.append(df)
    return pd.concat(out, ignore_index=True)


def hand_key(hand):
    """Integer key of a table hand string (suit masks packed 13 bits apart, as
    setback.canonical_key and the page's JavaScript compute it)."""
    k = 0
    for c in hand.split():
        k |= 1 << (SUITS.index(c[1]) * 13 + "23456789TJQKA".index(c[0]))
    return k


def allhands(tables):
    out = []
    for fn in sorted(glob.glob(os.path.join(tables, "allhands_[46]p_d*_s*.csv.gz"))):
        n, deals, search = map(int, re.search(r"allhands_(\d)p_d(\d+)_s(\d+)", fn).groups())
        df = pd.read_csv(fn)
        df.insert(0, "search", search)
        df.insert(0, "players", n)
        df.insert(2, "key", [hand_key(h) for h in df["hand"]])
        out.append(df)
        print(f"  {os.path.basename(fn)}: {len(df):,} hands", flush=True)
    if not out:
        return None
    df = pd.concat(out, ignore_index=True).sort_values(["players", "search", "key"])
    for c in ("players", "search"):
        df[c] = df[c].astype("int8")
    return df


REDUCED_ALIASES = {"roadhouse": "setback4_must_follow"}  # same single-hand play: one table serves both


def reduced_tables(tables):
    """(hand_groups, group_stats, presets covered) from allhands-r_*_side_*.csv.gz, or None."""
    files = sorted(glob.glob(os.path.join(tables, "allhands-r_*_side_[46]p_d*_s*.csv.gz")))
    if not files:
        return None
    from cardsim.hands import hand_classes  # needs numba: only when there are tables to build
    keys, _ = hand_classes()
    hg, gs, seen = [], [], set()
    for fn in files:
        preset, n, deals, search = re.search(r"allhands-r_(.+)_side_(\d)p_d(\d+)_s(\d+)\.csv\.gz$",
                                             os.path.basename(fn)).groups()
        n, search = int(n), int(search)
        if preset in REDUCED_ALIASES:  # served by the table it shares (the page maps it)
            continue
        df = pd.read_csv(fn)
        assert len(df) == len(keys)
        if (preset, n) not in seen:  # the grouping depends on preset and players, not on search
            seen.add((preset, n))
            hg.append(pd.DataFrame({"preset": preset, "players": n, "key": keys, "trump": df["trump"],
                                    "weight": df["weight"].astype("int32"), "group": df["group"].astype("int32")}))
        g = df.drop(columns=["hand", "trump", "weight"]).drop_duplicates("group").sort_values("group")
        g.insert(0, "search", search)
        g.insert(0, "players", n)
        g.insert(0, "preset", preset)
        gs.append(g)
        print(f"  {os.path.basename(fn)}: {len(g):,} groups", flush=True)
    hg = pd.concat(hg, ignore_index=True).sort_values(["preset", "players", "key"])
    gs = pd.concat(gs, ignore_index=True).sort_values(["preset", "players", "search", "group"])
    for d in (hg, gs):
        d["players"] = d["players"].astype("int8")
    gs["search"] = gs["search"].astype("int8")
    gs["deals"] = gs["deals"].astype("int32")
    return hg, gs, sorted({p for p, _ in seen})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", default=ROOT, help="folder with the allhands_*.csv.gz release tables")
    a = ap.parse_args()
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(os.path.join(OUT, "data"))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html"), encoding="utf-8") as f:
        page = f.read().replace("<!--COMMON-->", html.escape(COMMON))
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8", newline="\n") as f:
        f.write(page)
    open(os.path.join(OUT, ".nojekyll"), "w").close()
    wide, long = charts()
    sizes = {"presets": write(presets(), "presets.parquet"),
             "charts": write(wide, "charts.parquet"),
             "chart_ev": write(long, "chart_ev.parquet"),
             "condensed": write(condensed(), "condensed.parquet")}
    pre_hg, pre_gs = (os.path.join(a.tables, f) for f in ("hand_groups.parquet", "group_stats.parquet"))
    red = None if os.path.exists(pre_hg) and os.path.exists(pre_gs) else reduced_tables(a.tables)
    pre = os.path.join(a.tables, "allhands.parquet")
    if os.path.exists(pre_hg) and os.path.exists(pre_gs):
        shutil.copy(pre_hg, os.path.join(OUT, "data", "hand_groups.parquet.zip"))
        shutil.copy(pre_gs, os.path.join(OUT, "data", "group_stats.parquet.zip"))
        sizes["hand_groups"] = pq.ParquetFile(pre_hg).metadata.num_rows
        sizes["group_stats"] = pq.ParquetFile(pre_gs).metadata.num_rows
        sizes["allhands_presets"] = sorted(set(pq.read_table(pre_hg, columns=["preset"]).column(0).to_pylist()))
    elif red is not None:
        hg, gs, covered = red
        sizes["hand_groups"] = write(hg, "hand_groups.parquet.zip", row_group_size=16_384)
        sizes["group_stats"] = write(gs, "group_stats.parquet.zip", row_group_size=4_096)
        sizes["allhands_presets"] = covered
    elif os.path.exists(pre):
        shutil.copy(pre, os.path.join(OUT, "data", ALLHANDS))
        sizes["allhands"] = pq.ParquetFile(pre).metadata.num_rows
    else:
        ah = allhands(a.tables)
        if ah is not None:
            sizes["allhands"] = write(ah, ALLHANDS, row_group_size=16_384)
    if "allhands_presets" in sizes:
        sizes["allhands_aliases"] = {k: v for k, v in REDUCED_ALIASES.items() if v in sizes["allhands_presets"]}
    with open(os.path.join(OUT, "data", "tables.json"), "w") as f:
        json.dump(sizes, f)
    files = {"allhands": ALLHANDS, "hand_groups": "hand_groups.parquet.zip", "group_stats": "group_stats.parquet.zip"}
    for name, rows in sizes.items():
        if not isinstance(rows, int):
            continue
        mb = os.path.getsize(os.path.join(OUT, "data", files.get(name, f"{name}.parquet"))) / 1e6
        print(f"{name}: {rows:,} rows, {mb:.1f} MB")


if __name__ == "__main__":
    main()
