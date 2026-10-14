"""Every house-rule preset, checked by an independent rules oracle.

The oracle below works from the rules directly (no tables.py): which cards are
trumps, their order, legal plays, trick winners and scoring. Each game the
compiled engine plays is replayed card by card; every play must be legal, and the
final points must match the oracle's.

    .venv/Scripts/python tests/test_pitch_variants.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cardsim.cards import BIG_JOKER, COLOR_MATE, LITTLE_JOKER, cards_in, name, rank_of, suit_of  # noqa: E402
from cardsim.pitch import PRESETS, PitchRules  # noqa: E402
from cardsim.pitch import fast  # noqa: E402

TRUMP_ORDER = ["A", "K", "Q", "J", "offJ", "BJ", "LJ", "10", "9", "8", "7", "6", "5", "4", "3", "2"]
COUNT = {10: 10, 14: 4, 13: 3, 12: 2, 11: 1}


class Oracle:
    def __init__(self, rules, trump):
        self.r, self.t = rules, trump
        self.off_jack = 13 * COLOR_MATE[trump] + 9 if rules.off_jack else None

    def label(self, c):
        """Name of a trump card in TRUMP_ORDER terms."""
        if c == BIG_JOKER:
            return "BJ"
        if c == LITTLE_JOKER:
            return "LJ"
        if c == self.off_jack:
            return "offJ"
        return {14: "A", 13: "K", 12: "Q", 11: "J"}.get(rank_of(c), str(rank_of(c)))

    def is_trump(self, c):
        return c in (BIG_JOKER, LITTLE_JOKER) or c == self.off_jack or suit_of(c) == self.t

    def suit(self, c):
        return self.t if self.is_trump(c) else suit_of(c)

    def strength(self, c):
        return 100 - TRUMP_ORDER.index(self.label(c)) if self.is_trump(c) else rank_of(c)

    def legal(self, hand, trick, played):
        if not trick:
            nt = [c for c in hand if not self.is_trump(c)]
            if not played:
                if self.r.first_lead == "trump":
                    return [c for c in hand if self.is_trump(c)] or hand
                if self.r.first_lead == "not_trump" and nt:
                    return nt
            if self.r.trump_broken and nt and not any(self.is_trump(c) for c in played):
                return nt
            return hand
        lead = self.suit(trick[0][1])
        follow = [c for c in hand if self.suit(c) == lead]
        if not follow:
            return hand
        if lead == self.t or not self.r.may_trump:
            return follow
        return follow + [c for c in hand if self.is_trump(c) and c not in follow]

    def winner(self, trick):
        tr = [(p, c) for p, c in trick if self.is_trump(c)]
        if tr:
            return max(tr, key=lambda pc: self.strength(pc[1]))[0]
        lead = self.suit(trick[0][1])
        return max(((p, c) for p, c in trick if self.suit(c) == lead), key=lambda pc: rank_of(pc[1]))[0]

    def score(self, dealt, captured, team, last_team):
        r, t = self.r, self.t
        sc = [0] * (max(team) + 1)
        owner = {c: team[p] for p, h in enumerate(dealt) for c in h}
        taker = {c: k for k, cs in enumerate(captured) for c in cs}
        trumps = [c for c in owner if self.is_trump(c)]
        if trumps:
            sc[taker[max(trumps, key=self.strength)]] += r.high
            lo = min(trumps, key=self.strength)
            sc[taker[lo] if r.low_to_captor else owner[lo]] += r.low
        for c, k in taker.items():
            if not self.is_trump(c):
                continue
            if c == 13 * t + 9:
                sc[k] += r.jack
            elif c == self.off_jack:
                sc[k] += r.off_jack_pts
            elif c in (BIG_JOKER, LITTLE_JOKER):
                sc[k] += r.joker_pts
            elif suit_of(c) == t:
                sc[k] += {10: r.ten_pts, 3: r.three_pts, 5: r.five_pts, 9: r.nine_pts}.get(rank_of(c), 0)
        pips = [sum(COUNT.get(rank_of(c), 0) for c in cs if c < 52) for cs in captured]
        if pips.count(max(pips)) == 1:
            sc[pips.index(max(pips))] += r.game
        sc[last_team] += r.last_trick
        return sc


def replay(rules, g, hands, t, log_p, log_c, sc):
    o = Oracle(rules, t)
    n = g.nplayers
    team = list(g.team)
    held = [set(cards_in(int(h))) for h in hands]
    dealt = [set(h) for h in held]
    captured = [set() for _ in range(g.nteams)]
    played, trick, last_team = [], [], -1
    leader = 0
    for p, c in zip(log_p, log_c):
        p, c = int(p), int(c)
        assert p == (leader + len(trick)) % n, "wrong player to play"
        legal = o.legal(sorted(held[p]), trick, played)
        assert c in legal, f"illegal: {name(c)} from {[name(x) for x in sorted(held[p])]}, trick {[name(x) for _, x in trick]}"
        held[p].discard(c)
        trick.append((p, c))
        if len(trick) == n:
            w = o.winner(trick)
            captured[team[w]] |= {x for _, x in trick}
            played += [x for _, x in trick]
            trick, leader, last_team = [], w, team[w]
    assert not any(held), "cards left unplayed"
    want = o.score(dealt, captured, team, last_team)
    assert list(sc) == want, f"score {list(sc)} vs oracle {want}"


def check(preset, nplayers, deals=1500, search_deals=40):
    rules = PitchRules.preset(preset)
    g = fast.Game(rules, nplayers)
    if nplayers * rules.deal_size > 52 + rules.jokers:
        print(f"  {preset:20} {nplayers}-handed: skipped (deal needs {nplayers * rules.deal_size} cards)")
        return
    zero = np.zeros(g.nteams, np.int64)
    srch = np.full(g.nteams, 4, np.int64)
    pts = []
    for i in range(deals + search_deals):
        hands, t, _ = fast.deal(i, nplayers, rules.deal_size, rules.hand_size, rules.refill, 0,
                             g.tab, g.gf, g.tw, g.tl, g.tx)
        assert all(bin(int(h)).count("1") == rules.hand_size for h in hands), "hand sizes"
        assert np.bitwise_and.reduce(hands) == 0 or nplayers == 1
        for a in range(nplayers):
            for b in range(a + 1, nplayers):
                assert int(hands[a]) & int(hands[b]) == 0, "a card dealt twice"
        s = srch if i >= deals else zero
        sc, lp, lc = fast.play_game_log(hands, 0, t, g.team, g.nteams, True, g.rules_bt[g.team[0]], s,
                                        g.tab, g.gf)
        replay(rules, g, hands, t, lp, lc, sc)
        pts.append(sc.sum())
    print(f"  {preset:20} {nplayers}-handed: {deals} rule-based + {search_deals} search games legal and "
          f"scored right; points per hand {np.mean(pts):.2f} of {rules.points_total}")


if __name__ == "__main__":
    print("every preset, replayed by the rules oracle:")
    for preset in PRESETS:
        for n in (4, 6):
            check(preset, n)
