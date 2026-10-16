# setback-sim

How high should you bid in **Setback** (also called **Pitch**)? This project answers that by
playing out **every possible 6-card hand** with computer players that look ahead, on CPU or
GPU, and turning the results into bidding charts.

**Browse the results:** [john-schreiber.github.io/setback-sim](https://john-schreiber.github.io/setback-sim/)
has every chart (with an explanation of each set of house rules), a lookup for any 6-card
hand, and a SQL console over all the data, running in your browser (Parquet files queried by
DuckDB-WASM).

**Just want the charts?** [`bidchart_condensed.md`](bidchart_condensed.md) has a compact
bidding chart for 4-handed and 6-handed play: for each trump holding, how many small trumps you
need to bid 2, 3 or 4. [`charts/`](charts/README.md) has a chart for every house-rule preset
(6-point, 7-point, 10-point, Roadhouse, Draw Pitch, Nine-Five and more), each built from 4
million deals (2 million 6-handed). A chart cell is the *highest* bid worth making by the line that wins
the most whole games (below): how far to go in an auction, not necessarily the best opening bid.
Each chart's `.csv` has the expected points of every bid, their **net** expected points (your
score change minus what the opponents score on the hand), the line's bid (`bid_line`) and the
older line's (`max_bid`, expected points >= 0).

## What it found (4-handed unless noted)

- **Almost every hand is worth a 2-bid**: only about 1.5% of hands lose points at 2. The real
  decision is 2 or 3. In 6-handed, about 28% of hands should pass.
- **A 4-bid needs the Jack of trump.** A-K-Q without the Jack never bids 4.
- **The 10 or the 2 of trump is worth about one small trump**, and an off-suit Ace about half
  to one.
- **In the auction, raise only as far as you need.** Jumping straight to your highest
  profitable bid gets you set more often and loses about 3.5% of games.
- **Take the bid from your partner** only with a hand worth a full bid more than theirs.
- **11-bids** (all 4 points and every trick) pay off at about a 50% chance of sweeping or
  better: essentially six trumps with the Jack, or five trumps with the A, K and J plus a side
  Ace.
- **A higher bid is never worth more on the same hand.** The play doesn't depend on the bid, so
  a higher bid scores the same when made and only adds ways to be set; its only use is winning
  the auction.
- **Bid less than "expected points >= 0" says, and count the opponents.** In whole games
  (`cardsim/pitch/passline.py`), the line that wins most is a threshold on *net* expected points:
  setback4 4-handed, bid only when net >= -0.625 (6-handed: -1.75). It is stable (with every team
  on it, no other line does better), and against it the older line wins 48.8% of 4-handed games.
  In ten-point, draw pitch and nine-five the older line was far too aggressive: against the new
  lines it wins only 38-42% of 4-handed games with look-ahead players. Every variant's line is
  in `passline.LINES`, checked with look-ahead players through a per-deal outcome cache
  (`cardsim/pitch/gamecache.py`).

The rules assumed: High, Low (scored by whoever captures it), Jack and Game; bids 2–4; the bidder
leads first and that card sets trump; you may always trump in. 4-handed is two teams of two,
6-handed three teams of two.

## How it works

- **Engines.** `cardsim/pitch` plays any of the house-rule presets in `cardsim/pitch/rules.py`
  from one set of rules (`core.py`) compiled twice: for the CPU with Numba (`fast.py`) and for
  NVIDIA GPUs with CUDA (`gpu.py`). Cards are bits in a 64-bit mask, and per-trump lookup tables
  let the off-jack and jokers count as trumps on the same code path. The original 4-point
  engines (`setback.py`, a pure-Python reference; `setback_fast.py`; `setback_gpu.py`) remain as
  the reference. The tests check that rule-based play matches the reference card for card, that
  an independent rules oracle accepts every game under every preset, and that CPU and GPU play
  bit-identical games, search included.
- **Players.** Rule-based players follow hand-written rules, tuned by finding where search
  disagreed with them and testing each fix head-to-head. Search players try each legal card
  against many guesses of the hidden cards and keep the best. With 16 guesses per decision,
  4-handed, they play about 400,000 games per second on an RTX 3070 and 4,500 on a Ryzen
  5600X's CPU. The play-outs that take 85-98% of GPU time run in a CUDA C++ kernel
  (`cardsim/pitch/rollout.cu`), the same play as the Numba code with 32-bit arithmetic and
  fewer branches, 2.4-3x faster and checked to give identical games; each GPU picks the
  faster kernel and its settings for itself (`cardsim/pitch/gpu_tune.py`).
- **Every hand.** Up to suit relabeling, the 20,358,520 possible 6-card hands reduce to
  962,988. `allhands.py` plays each one as bidder against hundreds of random deals and saves
  the per-hand results: the chance of making and expected points of each bid and, for tables
  built on the cluster (`--cluster`), the opponents' points, each bid's net expected points and
  the chance of taking all four points and every trick. `condense.py` turns those into the
  condensed charts.
- **Fewer hands to play.** The tables also show which cards barely matter to the bidder: a
  side-suit 2 to 9 changes the average by under 0.002 points, a trump 4 to 9 by under 0.015.
  So `allhands.py --reduced side` plays hands that differ only in side-suit 2s-9s as one group
  (68,677 groups 4-handed instead of 962,988; each deal plays a member drawn by how common it
  is), checked against the full tables to differ by nothing beyond sampling noise. At a third
  of the cost a group gets four times the deals, so these tables are also more precise; they
  cover setback4, must-follow and low-dealt (Roadhouse plays a hand exactly as must-follow).

## Running it

With [uv](https://docs.astral.sh/uv/):

```
uv sync --extra gpu                 # gpu needs an NVIDIA card; leave it off for CPU only
uv run python setback.py hand "AS JS 2S KH 9D 4C" -p 4 --search 16 --engine gpu
uv run python setback.py lookup "AS JS 2S KH 9D 4C" -p 6     # needs the every-hand tables
uv run python -m cardsim.pitch.bidchart --preset ten_point -p 4    # one preset's bid chart
uv run python -m cardsim.pitch.game --preset roadhouse --pairs cheap:table   # whole games
uv run python tests/test_pitch_core.py && uv run python tests/test_pitch_variants.py
uv run python tests/test_pitch_game.py && uv run python tests/test_pitch_gpu.py
uv run python tests/test_scoring.py && uv run python tests/test_dist.py
uv run python tests/test_update.py            # --offline to skip the GitHub part
```

To spread big studies over several GPU machines, see [docs/distributed.md](docs/distributed.md);
for AWS spot GPUs, [docs/aws.md](docs/aws.md).

| Script | What it does |
|---|---|
| `setback.py` | command line: simulate one hand, bid charts, table lookup; Python reference engine |
| `allhands.py` | the every-hand tables, on this GPU or the cluster (`--cluster URL`, which adds the opponents' points, net expected points and sweep chances) |
| `auction.py` | whole games with an auction: bidding strategies, stealing, dealer's privilege, 11-bids |
| `experiments.py`, `fit_trump.py`, `sweep.py` | targeted studies: weakest profitable bids, bare Ace vs two trumps, trump-choice weights, sweep chances |
| `condense.py` | condensed bid charts from the tables |
| `cardsim/pitch/bidchart.py` | bid charts for any preset, on this GPU or the cluster (`--cluster URL`) |
| `cardsim/pitch/game.py` | whole games with an auction under any preset |
| `cardsim/pitch/fit_trump.py` | per-preset weights for choosing trump |
| `cardsim/dist/` | coordinator and workers for spreading studies over several machines |
| `site/` | the stats site: `build.py` writes the Parquet data, a GitHub Actions workflow publishes it |

The every-hand tables (about 25 MB each), the site's `allhands.parquet` and the raw per-deal
results are published as [release assets](../../releases) rather than kept in the repo; v0.2
has the current tables. The site rebuilds on every push to its files; after publishing a
release, run it by hand on main (`gh workflow run pages.yml --ref main`) so it picks up the new
table.

## Roadmap

House rules, the coordinator/worker setup and the stats site are done. Next: Spades and
Hearts. See [ROADMAP.md](ROADMAP.md).

## License

MIT. See [LICENSE](LICENSE).
