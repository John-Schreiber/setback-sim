"""Where should the bidding line be? Whole games (game.py) in which one team bids up to the
highest bid whose chart value is at least a threshold, against teams using the charts' own
line (expected points >= 0). The value is either the expected points of the bid (ev: the
bidding team's score change) or its net expected points (net: minus what the opponents
score on the hand). Every threshold plays the same games (same seed), so differences in the
win rate are measured on paired games. With --search N the hands are played by look-ahead
players (N worlds per decision) through gamecache.Cache: each deal played once per possible
bidder on the GPU, shared by every line.

    .venv/Scripts/python -m cardsim.pitch.passline --preset setback4 -p 4 --games 400000
"""
import argparse
import time

import numpy as np

from cardsim.pitch import PitchRules, fast
from cardsim.pitch.game import (C_AUC0, C_SET0, C_WIN, STRATS, Brain, play_games, rule_params)


# The stable bidding line per (preset, players): bid up to the highest bid whose net expected
# points (score change minus the opponents' points on the hand, from the charts) are at least
# this. Found 2026-10-16 with rule-based players, 200,000 games per line: the best line against
# a field on today's line (ev >= 0), then iterated until team A's best response, with every
# other team on the line, stays on it (within one grid step / noise). Against a field on these
# lines, today's line wins 0.0-2.4 points less in the setback family and 7-23 points less in
# ten-point, draw pitch and nine-five. Every line was then checked with look-ahead players
# (gamecache, 16 worlds, 200,000 games): stable for all but two, which moved to the look-ahead
# fixed point (draw_pitch 6-handed, nine_five). With look-ahead players today's line wins 41%
# (ten-point), 42% (draw pitch) and 38% (nine-five) of 4-handed games against these lines.
LINES = {
    ("setback4", 4): -0.625, ("setback4", 6): -1.75,
    ("setback4_lowdealt", 4): -0.625, ("setback4_lowdealt", 6): -1.625,
    ("setback4_must_follow", 4): -0.75, ("setback4_must_follow", 6): -1.75,
    ("six_point", 4): -0.75, ("six_point", 6): -2.0625,
    ("seven_point", 4): -0.4375, ("seven_point", 6): -2.1875,
    ("ten_point", 4): -0.625, ("ten_point", 6): -2.5,
    ("roadhouse", 4): -0.75, ("roadhouse", 6): -1.375,
    ("draw_pitch", 4): -0.25, ("draw_pitch", 6): -0.875,  # 6p: look-ahead; rule-based -0.625
    ("nine_five", 4): -1.6875,  # look-ahead players; rule-based play settles at -3.375
}

_CACHES = {}


def sweep(preset, players, games, seed, measure, thresholds, strat="cheap", field=("ev", 0.0), search=0):
    """field: the (measure, threshold) every other team bids by. search > 0: look-ahead players
    (a shared gamecache.Cache per preset, players, games, seed and search)."""
    rules = PitchRules.preset(preset)
    g = fast.Game(rules, players)
    cache = None
    if search:
        from cardsim.pitch.gamecache import Cache
        ck = (preset, players, games, seed, search)
        if ck not in _CACHES:
            _CACHES[ck] = Cache(g, games, seed, np.full(g.nteams, search, np.int64), max_deals=256)
        cache = _CACHES[ck]
    base = Brain(rules, players, threshold=field[1], measure=field[0])
    take = np.zeros(64, np.int64)
    for lv in rules.dealer_takes:
        take[lv] = 1
    rp = rule_params(rules)
    st = np.full(g.nteams, STRATS[strat], np.int64)
    ns = np.zeros(g.nteams, np.int64)
    rows = []
    for th in thresholds:
        a = Brain(rules, players, threshold=th, measure=measure)
        mb = np.stack([a.maxbid] + [base.maxbid] * (g.nteams - 1))
        mo = np.stack([a.moon] + [base.moon] * (g.nteams - 1))
        if cache is not None:
            o = cache.run(st, ns, take, mb, mo, base.honor_cards)
        else:
            o = play_games(games, players, g.team, st, ns, rp, take, mb, mo, base.honor_cards, g.rules_bt, g.tab,
                           g.gf, g.tw, g.tl, g.tx, seed)
        win = (o[:, C_WIN] == 0).mean()
        tot = o.sum(axis=0)
        hands = tot[1]
        rows.append((th, win, np.sqrt(win * (1 - win) / games), tot[C_AUC0] / hands, tot[C_SET0] / max(1, tot[C_AUC0])))
    return rows, 1 / g.nteams


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="setback4")
    ap.add_argument("-p", "--players", type=int, default=4)
    ap.add_argument("--games", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--strategy", default="cheap", choices=list(STRATS))
    ap.add_argument("--field", default="ev:0", help="the other teams' line, MEASURE:THRESHOLD (default ev:0)")
    ap.add_argument("--ev", default="-1,-0.75,-0.5,-0.25,-0.1,0,0.1,0.25,0.5,0.75,1", help="ev thresholds ('' to skip)")
    ap.add_argument("--net", default="-2,-1.75,-1.5,-1.25,-1,-0.75,-0.5,-0.25,0", help="net thresholds ('' to skip)")
    ap.add_argument("--search", type=int, default=0, help="look-ahead players with N worlds (0: rule-based)")
    a = ap.parse_args()
    fm, ft = a.field.split(":")
    field = (fm, float(ft))
    for measure, spec in (("ev", a.ev), ("net", a.net)):
        if not spec:
            continue
        ths = [float(x) for x in spec.split(",")]
        t0 = time.time()
        rows, fair = sweep(a.preset, a.players, a.games, a.seed, measure, ths, a.strategy, field, a.search)
        print(f"{a.preset} {a.players}-handed, {a.games:,} games per line, team A bids up to the highest bid with "
              f"{measure} >= threshold; the others bid by {fm} >= {float(ft):+.2f}; "
              f"{f'look-ahead players ({a.search} worlds)' if a.search else 'rule-based players'}; fair share {fair:.1%} "
              f"({time.time() - t0:.0f}s)")
        print("  threshold   A wins           A takes the bid   A set when bidding")
        for th, win, se, auc, sets in rows:
            print(f"  {th:+9.2f}   {win:6.2%} +/-{se:.2%}   {auc:6.1%}            {sets:6.1%}")


if __name__ == "__main__":
    main()
