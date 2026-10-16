"""The value of passing, measured: at every auction decision in whole games (look-ahead or
rule-based players, gamecache), both choices are scored on the same deal: bid at the cheapest
winning level, or pass and let the auction run on. The difference swing(bid) - swing(pass)
(swing: your score change minus the opponents' average) is accumulated per holding (trump
honors and small trumps, as in the charts) and level. The grounded rule is to bid when that is
at least 0; the script compares it with the fitted line (lines.LINES) and plays it in whole games.

    .venv/Scripts/python -m cardsim.pitch.passvalue --preset setback4 -p 4 --games 200000 --search 16
"""
import argparse
import time

import numpy as np

from cardsim.pitch import PitchRules, fast
from cardsim.pitch.bidchart import honor_list
from cardsim.pitch.game import MAX_SMALL, STRATS, Brain
from cardsim.pitch.gamecache import Cache, pass_values
from cardsim.pitch.lines import LINES


def label(key, hon):
    m, s = divmod(key, MAX_SMALL)
    return (" ".join(h for i, h in enumerate(hon) if m >> i & 1) or "(none)") + f" +{s}"


def measure(preset, players, games, seed, search, log=print):
    rules = PitchRules.preset(preset)
    g = fast.Game(rules, players)
    line = LINES[(preset, players)]
    brain = Brain(rules, players, threshold=line, measure="net")
    take = np.zeros(64, np.int64)
    for lv in rules.dealer_takes:
        take[lv] = 1
    st = np.full(g.nteams, STRATS["cheap"], np.int64)
    ns = np.zeros(g.nteams, np.int64)
    mb, mo = np.stack([brain.maxbid] * g.nteams), np.stack([brain.moon] * g.nteams)
    cache = Cache(g, games, seed, np.full(g.nteams, search, np.int64), max_deals=256)
    t0 = time.time()
    base = cache.run(st, ns, take, mb, mo, brain.honor_cards)
    log(f"{preset} {players}-handed, {games:,} games ({'look-ahead' if search else 'rule-based'} players), every team "
        f"on the fitted line net >= {line:+g}: cache {cache.played:,} hands in {time.time() - t0:.0f}s")
    cnt, s1, s2, took, sp, need = pass_values(games, players, g.team, st, ns, cache.rp, take, mb, mo, brain.honor_cards,
                                              g.tab, g.gf, g.tw, g.tl, g.tx, seed, cache.pts, cache.swept, cache.have,
                                              mb.shape[1])
    assert (need < 0).all()
    return rules, g, brain, take, st, ns, mb, mo, cache, base, (cnt, s1, s2, took, sp), line


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="setback4")
    ap.add_argument("-p", "--players", type=int, default=4)
    ap.add_argument("--games", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--search", type=int, default=16)
    a = ap.parse_args()
    rules, g, brain, take, st, ns, mb, mo, cache, base, (cnt, s1, s2, took, sp), line = measure(
        a.preset, a.players, a.games, a.seed, a.search)
    hon = [h for h, _ in honor_list(rules)]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / cnt
        se = np.sqrt(np.maximum(s2 / cnt - mean ** 2, 0) / np.maximum(cnt - 1, 1))
        pass_v = sp / cnt
        took_share = took / cnt
    seen = cnt > 0
    n_dec = cnt.sum()
    measured = mean >= 0
    fitted = took_share > 0.5
    agree = (cnt * (measured == fitted))[seen].sum() / n_dec
    print(f"{int(n_dec):,} decisions; the measured rule (bid when bid - pass >= 0) and the fitted line agree on "
          f"{agree:.1%} of them")
    for lv in range(rules.min_bid, rules.max_bid + 1):
        c = cnt[:, lv]
        if c.sum() == 0:
            continue
        print(f"  level {lv}: {int(c.sum()):9,} decisions; value of passing {np.nansum(sp[:, lv]) / c.sum():+.3f} "
              f"(net points of the hand); bid - pass {np.nansum(s1[:, lv]) / c.sum():+.3f}")
    # where they disagree, the most frequent cases
    dis = np.argwhere(seen & (measured != fitted))
    dis = sorted(dis, key=lambda kl: -cnt[kl[0], kl[1]])[:15]
    print("most frequent disagreements (holding, level: bid - pass +/- se; fitted line bids?):")
    for k, lv in dis:
        print(f"  {label(k, hon):18} {lv}: {mean[k, lv]:+.3f} +/- {se[k, lv]:.3f}  ({int(cnt[k, lv]):,} decisions)"
              f"  fitted line {'bids' if fitted[k, lv] else 'passes'}")
    # whole games: team A bids by the measured rule, the others by the fitted line
    mbA = brain.maxbid.copy()
    for k in np.flatnonzero(cnt.sum(axis=1) > 0):
        levels = [lv for lv in range(rules.min_bid, rules.max_bid + 1) if cnt[k, lv] >= 30]
        if levels:  # highest level with a measured edge for bidding
            ok = [lv for lv in levels if mean[k, lv] >= 0]
            mbA[k] = max(ok) if ok else 0
    o = cache.run(st, ns, take, np.stack([mbA] + [brain.maxbid] * (g.nteams - 1)), mo, brain.honor_cards)
    win = (o[:, 0] == 0).mean()
    print(f"whole games: team A bidding by the measured rule wins {win:.2%} +/- {np.sqrt(win * (1 - win) / len(o)):.2%} "
          f"against the fitted line (fair {1 / g.nteams:.1%})")


if __name__ == "__main__":
    main()
