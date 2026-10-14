# Roadmap

Four parts, in order. Each lands with tests showing it matches the reference engine, as
the current engines do.

## 1. House rules (done)

Done: `cardsim/pitch` with nine presets (setback4, setback4_lowdealt, setback4_must_follow,
six_point, seven_point, ten_point, roadhouse, draw_pitch, nine_five), per-preset trump-choice
weights, bid charts for each in [`charts/`](charts/README.md), and whole games with an auction.
Not yet: a Maryland Jack preset (its exact-bid bonus is supported), widow/kitty deals, blind
smudge, and the rare variants below.

Rules become a config instead of being fixed in code, and every named variant is a preset.
**Done when:** the default preset reproduces today's results exactly (same deals, same
cards played), and each new rule is checked card for card between the reference and compiled
engines.

Cards will use a per-trump lookup table (each card's effective suit and rank), so the
off-jack and jokers can count as trumps while one compiled code path still runs on CPU and GPU.

| Group | Rules |
|---|---|
| Deck | 52 cards; one or two jokers (ranking below the Jack) |
| Trumps | off-jack (same-colour Jack) counts as trump |
| Points | High, Low, Jack, Game, off-jack, jokers, 10 of trump, 3 of trump (3 points), 5 and 9 of trump (Nine-Five), last trick; Game card values and tie rule |
| Low | scored by whoever captures it, or by whoever was dealt it |
| Deal | 6, 7 or 9 cards; discard non-trumps and refill to 6; widow/kitty to the bidder |
| Play | may trump in, or must follow suit; first lead must be trump, may be anything, or must not be trump; trump must be broken |
| Bidding | minimum and maximum bid; forced dealer or redeal; dealer's privilege (at which levels); shoot the moon / smudge / 11 (scores N, or wins the game); blind smudge |
| Scoring | target score; bidder goes out first; win by 2; only the bidder scores ("no junk points"); exact-bid bonus; backdoor win at a negative score |

**Presets:** 4-point Setback (current), 6-point (off-jack and joker), 7-point (Cell), 10-point,
Roadhouse, Maryland Jack, Partnership Draw Pitch, and Nine-Five Setback (9 cards dealt, discard
to 6, the 9 and 5 of trump worth 9 and 5, 18 points, bids 9–18, smudge 19 and blind smudge 20,
game to 100).

Rare variants come later: Two-Trump, Reverse, Blind/Abacos/Man-o-war, call for a partner, and
the 9-card widow games (Campbell, High Five, Bundrick).

## 2. Other trick-taking games

Split out what every trick-taking game shares (dealing, legal moves, trick winner, and the
search player) from what is game-specific (bidding, scoring, rule-based players).

- **Spades:** bidding a number of tricks, nil and blind nil, bags, spades must be broken.
- **Hearts:** passing three cards, avoiding hearts and the Q♠, shooting the moon.

**Done when:** each game has its own reference engine, compiled engine and tests, plus a first
study (Spades: how many tricks to bid; Hearts: which cards to pass).

## 3. Coordinator and worker images (done)

Done: `cardsim/dist` and `docker/`, described in [docs/distributed.md](docs/distributed.md).
Workers run in Docker (CUDA 13, or CUDA 12 for older drivers) or natively on Windows; a study
split across CPU and GPU workers matches a single run exactly.

- **Coordinator:** a small HTTP service with a job queue (SQLite). It splits a study into jobs of
  **5–10 minutes** each, hands them out, stores results, and re-issues jobs whose worker
  goes quiet. It runs in Docker on the Windows machine and listens on the LAN.
- **Worker:** a CUDA image (`docker run --gpus all`) for the Linux GPU box. It pulls a job,
  runs it on the GPU, and pushes the results. This machine can run a worker too, through
  Docker Desktop's WSL2 GPU support.

Results are checkpointed per job, so a restarted worker or coordinator loses at most one job.
**Done when:** an every-hand table built across both machines matches one built on a single
machine.

## 4. Browsable stats site

Static hosting on GitHub Pages (the repo is public):

- **Datasette Lite** (Datasette running in the browser) over a small SQLite file with the
  condensed tables, so you can browse, filter and query them in SQL with no server.
- **Full per-hand tables** as downloads in GitHub Releases. If browsing them in place proves
  useful, Parquet files queried in the browser by DuckDB-WASM.

**Done when:** a GitHub Actions workflow rebuilds the site from the published data.
