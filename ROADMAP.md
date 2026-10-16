# Roadmap

Four parts, in order. Each lands with tests showing it matches the reference engine, as
the current engines do.

## 1. House rules (done)

Built: `cardsim/pitch`, where rules are a config (`rules.py`) and every named variant is a
preset. Cards use a per-trump lookup table (each card's effective suit and rank), so the
off-jack and jokers count as trumps while one compiled code path runs on CPU and GPU. The
setback4 preset reproduces the original engine card for card, and every rule is checked
between the reference and compiled engines.

| Group | Rules supported |
|---|---|
| Deck | 52 cards; one or two jokers (ranking below the off-jack) |
| Trumps | off-jack (same-colour Jack) counts as trump |
| Points | High, Low, Jack, Game, off-jack, jokers, 10 of trump, 3 of trump, 5 and 9 of trump, last trick; Game card values; a tie for Game scores nobody |
| Low | scored by whoever captures it, or by whoever was dealt it |
| Deal | any number of cards (6, 7, 9, ...); discard non-trumps and draw back to 6 |
| Play | may trump in, or must follow suit; first lead must be trump, may be anything, or must not be trump; trump must be broken |
| Bidding | minimum and maximum bid; stuck dealer or redeal; dealer's privilege (at which levels); moon / smudge / 11 (scores N, or wins the game) |
| Scoring | target score; bidder goes out first; win by 2; must bid to go out; only the bidder scores ("no junk points"); exact-bid bonus; backdoor loss at a negative score |

**Presets (nine):** setback4, setback4_lowdealt, setback4_must_follow, six_point, seven_point,
ten_point, roadhouse, draw_pitch and nine_five, each explained in plain language in
`cardsim/pitch/notes.py` (shown on the site and in [`charts/`](charts/README.md)). Each has
its own trump-choice weights, a bid chart with expected and net expected points, and whole
games with an auction (`cardsim/pitch/game.py`).

Not yet: a Maryland Jack preset (its exact-bid bonus is supported), widow/kitty deals,
blind smudge, and the rare variants: Two-Trump, Reverse, Blind/Abacos/Man-o-war, call for a
partner, and the 9-card widow games (Campbell, High Five, Bundrick).

### Next for Setback

- **Bid with the opponents counted: done.** Whole-game sweeps (`passline.py`) found, per variant,
  the line on net expected points that wins the most games and is stable when every team uses
  it; checked with look-ahead players through `gamecache.py`. The charts now show that line
  (`bid_line`); the older one (expected points >= 0) stays in the `.csv` as `max_bid`.
- **Every-hand tables for more variants: done as reduced tables** (`allhands.py --reduced`):
  setback4, setback4_must_follow and setback4_lowdealt, 4- and 6-handed, both kinds of
  players; roadhouse plays a single hand exactly as must-follow and shares its table. Built
  locally and in the site's lookup (a house-rules selector), not yet published as a release.
- **Maryland Jack preset**, then the "not yet" rules above.

## 2. Other trick-taking games

Split out what every trick-taking game shares (dealing, legal moves, trick winner, and the
search player) from what is game-specific (bidding, scoring, rule-based players).

- **Spades:** bidding a number of tricks, nil and blind nil, bags, spades must be broken.
- **Hearts:** passing three cards, avoiding hearts and the Q♠, shooting the moon.

**Done when:** each game has its own reference engine, compiled engine and tests, plus a first
study (Spades: how many tricks to bid; Hearts: which cards to pass).

## 3. Coordinator and workers (done)

Built: `cardsim/dist` and `docker/`, described in [docs/distributed.md](docs/distributed.md).

- **Coordinator:** a small HTTP service with a job queue in SQLite, in Docker on the Windows
  machine, listening on the LAN. It splits a study into slices sized to take about 7.5
  minutes on each worker, re-issues slices whose worker goes quiet, and merges the results.
- **Workers:** in Docker (CUDA 13, or CUDA 12 for older cards such as the GTX 1060) or
  natively, which on Windows gives about 50% more CPU throughput than Docker Desktop. The
  Linux box runs a GPU and a CPU worker in Docker; the Windows machine runs both natively.
  With `--auto-update`, a worker fetches the exact code a study was submitted with from
  GitHub and restarts on it. Workers run slices in 20-second pieces and, when stopped (or
  warned of a spot interruption with `--spot`), hand back the unfinished part; GPU workers
  pick the faster play-out kernel (Numba or CUDA C++, identical games) and its settings for
  their card. [docs/aws.md](docs/aws.md) covers AWS spot workers.

A study split across machines, devices and CPU/GPU matches a single run exactly
(`tests/test_dist.py`), and the every-hand tables have been built across the cluster.

## 4. Browsable stats site (done)

Built with Parquet and DuckDB-WASM instead of the planned Datasette Lite: the page queries the
data in the browser with HTTP range requests, so even the 3.85-million-row every-hand table is
searchable (a hand lookup reads under 1 MB). It has every bid chart (with each variant
explained, and expected and net points per bid), a lookup for any 6-card hand, and a SQL
console. `site/build.py` writes the data, taking the every-hand table from the latest
release; `.github/workflows/pages.yml` rebuilds and publishes the site on each change, and is
run by hand on main after a release.
