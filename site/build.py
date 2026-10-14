"""Build the static stats site: Parquet data plus one page that queries it in the browser.

The page (site/index.html) runs DuckDB-WASM, which reads only the parts of each Parquet
file a query needs (HTTP range requests), so the million-row every-hand tables can be
looked up without downloading them. Output goes to site/_build/ (index.html and data/):

  presets.parquet    one row per house-rule preset (bids, points, a one-line summary)
  charts.parquet     per preset, players, trump honors and small trumps: hands, average
                     points, highest bid with expected points >= 0 (charts/*.csv)
  chart_ev.parquet   the same groups, one row per bid: expected points of that bid
  condensed.parquet  4-point Setback, from the every-hand tables (condensed_*.csv)
  allhands.parquet.zip
                     every distinct 6-card hand, 4-point Setback, as bidder. Built from the
                     release tables allhands_*.csv.gz, or copied as is if DIR already has an
                     allhands.parquet (the release carries one, so the site build skips the
                     slow part); skipped if neither is present

    uv run --extra site python site/build.py [--tables DIR]
"""
import argparse
import glob
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
        rows.append(dict(preset=name, summary=describe(r), min_bid=r.min_bid, max_bid=r.max_bid,
                         moon_bid=r.moon_bid, moon_value=r.moon_value, points_total=r.points_total,
                         target=r.target, deal_size=r.deal_size, hand_size=r.hand_size, jokers=r.jokers,
                         off_jack=r.off_jack, may_trump=r.may_trump, low_to_captor=r.low_to_captor))
    return pd.DataFrame(rows)


def charts():
    """charts/<preset>_<n>p.csv -> one wide table (without the ev columns) and one long one."""
    wide, long = [], []
    for fn in sorted(glob.glob(os.path.join(ROOT, "charts", "*_[46]p.csv"))):
        preset, n = re.match(r"(.+)_(\d)p\.csv$", os.path.basename(fn)).groups()
        df = pd.read_csv(fn, keep_default_na=False)
        df.insert(0, "players", int(n))
        df.insert(0, "preset", preset)
        evs = [c for c in df.columns if re.fullmatch(r"ev\d+", c)]
        keep = ["preset", "players", "honors", "small", "hands", "avg", "p_moon", "max_bid"]
        w = df[keep].copy()
        w["ev_moon"] = df["ev_moon"] if "ev_moon" in df else None
        wide.append(w)
        lg = df.melt(id_vars=["preset", "players", "honors", "small", "hands"], value_vars=evs,
                     var_name="bid", value_name="ev")
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", default=ROOT, help="folder with the allhands_*.csv.gz release tables")
    a = ap.parse_args()
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(os.path.join(OUT, "data"))
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html"), OUT)
    open(os.path.join(OUT, ".nojekyll"), "w").close()
    wide, long = charts()
    sizes = {"presets": write(presets(), "presets.parquet"),
             "charts": write(wide, "charts.parquet"),
             "chart_ev": write(long, "chart_ev.parquet"),
             "condensed": write(condensed(), "condensed.parquet")}
    pre = os.path.join(a.tables, "allhands.parquet")
    if os.path.exists(pre):
        shutil.copy(pre, os.path.join(OUT, "data", ALLHANDS))
        sizes["allhands"] = pq.ParquetFile(pre).metadata.num_rows
    else:
        ah = allhands(a.tables)
        if ah is not None:
            sizes["allhands"] = write(ah, ALLHANDS, row_group_size=16_384)
    with open(os.path.join(OUT, "data", "tables.json"), "w") as f:
        json.dump(sizes, f)
    for name, rows in sizes.items():
        mb = os.path.getsize(os.path.join(OUT, "data", ALLHANDS if name == "allhands" else f"{name}.parquet")) / 1e6
        print(f"{name}.parquet: {rows:,} rows, {mb:.1f} MB")


if __name__ == "__main__":
    main()
