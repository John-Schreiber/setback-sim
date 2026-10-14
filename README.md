# setback-sim

How high should you bid in **Setback** (also called **Pitch**)? This project answers that by
playing out **every possible 6-card hand** with computer players that look ahead, on CPU or
GPU, and turning the results into bidding charts.

**Just want the charts?** See [`setback_bidding_charts.pdf`](setback_bidding_charts.pdf): a
4-page guide with one chart for 4-handed and one for 6-handed play. For every exact trump
holding, [`setback_ev_by_trumps_and_aces.xlsx`](setback_ev_by_trumps_and_aces.xlsx) has the
expected points of each bid.

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

The rules assumed: High, Low (scored by whoever captures it), Jack and Game; bids 2–4; the bidder
leads first and that card sets trump; you may always trump in. 4-handed is two teams of two,
6-handed three teams of two.

## How it works

- **Engines.** Three engines play identical games: a pure-Python reference engine
  (`setback.py`), a compiled CPU engine (`setback_fast.py`, Numba) and a CUDA GPU engine
  (`setback_gpu.py`). Both compiled engines share the rules in `engine_core.py`. The tests check
  that rule-based play matches the reference engine card for card, and that the CPU and GPU
  engines match each other.
- **Players.** Rule-based players follow hand-written rules, tuned by finding where search
  disagreed with them and testing each fix head-to-head. Search players try each legal card
  against many guesses of the hidden cards and keep the best; on an RTX 3070 they play about
  29,000 4-handed games per second.
- **Every hand.** Up to suit relabeling, the 20,358,520 possible 6-card hands reduce to
  962,988. `allhands.py` plays each one as bidder against hundreds of random deals and saves
  the per-hand results. `condense.py`, `make_pdf.py` and `make_excel.py` turn those into the
  charts and the spreadsheet.

## Running it

```
uv venv -p 3.14 .venv
uv pip install --python .venv -e ".[gpu,reports]"      # gpu needs an NVIDIA card
.venv/Scripts/python setback.py hand "AS JS 2S KH 9D 4C" -p 4 --search 16 --engine gpu
.venv/Scripts/python setback.py lookup "AS JS 2S KH 9D 4C" -p 6     # needs the every-hand tables
.venv/Scripts/python test_fast.py && .venv/Scripts/python test_gpu.py
```

On Linux or macOS, use `.venv/bin/python` instead.

| Script | What it does |
|---|---|
| `setback.py` | command line: simulate one hand, bid charts, table lookup; Python reference engine |
| `allhands.py` | the every-hand tables (hours with search players; minutes with rule-based ones) |
| `auction.py` | whole games with an auction: bidding strategies, stealing, dealer's privilege, 11-bids |
| `experiments.py`, `fit_trump.py`, `sweep.py` | targeted studies: weakest profitable bids, bare Ace vs two trumps, trump-choice weights, sweep chances |
| `condense.py`, `make_pdf.py`, `make_excel.py` | charts, PDF and spreadsheet from the tables |

The every-hand tables (about 15 MB each) are published as
[release assets](../../releases) rather than kept in the repo.

## Roadmap

More house rules (jokers, off-jack, 10-point and Nine-Five), Spades and Hearts, a
coordinator/worker setup for running across several GPU machines, and a browsable stats
site. See [ROADMAP.md](ROADMAP.md).

## License

MIT. See [LICENSE](LICENSE).
