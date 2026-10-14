"""Setback (Pitch) bid simulator.

Monte Carlo: deal hands, the bidder picks trump and pitches (leads) first, all
tricks are played out by heuristic players, and we record the points the
bidding team takes.

Modes:
  hand    -- evaluate one specific hand:
               .venv/Scripts/python setback.py hand "AS JS 2S KH 9D 4C" -p 4
  lookup  -- instant answer from the every-hand table (allhands.py):
               .venv/Scripts/python setback.py lookup "AS JS 2S KH 9D 4C" -p 4
  survey  -- bid chart over random hands:
               .venv/Scripts/python setback.py survey -p 6 --search 16

The game engine runs compiled (setback_fast.py, Numba) by default, or on an
NVIDIA GPU with --engine gpu (setback_gpu.py, ~10x faster with --search); this file
keeps the pure-Python reference engine (--engine python) that test_fast.py
checks it against.

Points: High, Low, Jack, Game (4 total). Bids 2-4.
"""
import argparse
import os
import random
from collections import defaultdict
from multiprocessing import Pool, cpu_count

SUITS = "SHDC"
RNAME = {10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"}
RPARSE = {"A": 14, "K": 13, "Q": 12, "J": 11, "T": 10}
GAME_VAL = {10: 10, 14: 4, 13: 3, 12: 2, 11: 1}
DECK = [(r, s) for s in range(4) for r in range(2, 15)]
BIDS = (2, 3, 4)


def card_str(c):
    return RNAME.get(c[0], str(c[0])) + SUITS[c[1]]


def parse_card(tok):
    tok = tok.upper().replace("10", "T")
    r = RPARSE.get(tok[0]) or int(tok[0])
    return (r, SUITS.index(tok[1]))


def make_teams(nplayers, team_size):
    nteams = nplayers // team_size
    return [i % nteams for i in range(nplayers)]  # partners alternate around the table


# ---------------------------------------------------------------- play logic

def danger(c, trump, low_captured):
    """How costly it is to hand this card to the other side."""
    d = GAME_VAL.get(c[0], 0)
    if c[1] == trump:
        if c[0] == 11:
            d += 15
        if low_captured and c[0] <= 3:
            d += 8
    return d


def legal(hand, lead_suit, trump):
    if lead_suit is None:
        return hand
    follow = [c for c in hand if c[1] == lead_suit]
    if not follow:
        return hand
    if lead_suit == trump:
        return follow
    return follow + [c for c in hand if c[1] == trump]  # may always trump in


def beats(a, b, lead_suit, trump):
    """Does card a beat card b (b currently winning)?"""
    if a[1] == trump:
        return b[1] != trump or a[0] > b[0]
    if b[1] == trump:
        return False
    return a[1] == lead_suit and a[0] > b[0]


def is_boss_trump(c, trump, hand, gone):
    """c is the highest trump still unaccounted for (from this player's view)."""
    if c[1] != trump:
        return False
    for r in range(c[0] + 1, 15):
        t = (r, trump)
        if t not in gone and t not in hand:
            return False
    return True


def trick_winner(trick, trump):
    lead_suit = trick[0][1][1]
    wp, wc = trick[0]
    for p, c in trick[1:]:
        if beats(c, wc, lead_suit, trump):
            wp, wc = p, c
    return wp, wc


def unseen_higher(c, hand, gone, trick_cards):
    """Is any card of c's suit higher than c still unseen by this player?"""
    for r in range(c[0] + 1, 15):
        t = (r, c[1])
        if t not in gone and t not in hand and t not in trick_cards:
            return True
    return False


def could_beat(p, c, lead_suit, trump, hand, gone, trick_cards, voids):
    """Could player p (not me) still beat card c, given what I've seen?

    Unseen cards may be undealt, so this is "possibly", not "probably".
    """
    can_trump = trump not in voids[p] and any(
        (r, trump) not in gone and (r, trump) not in hand and (r, trump) not in trick_cards
        for r in range(2, 15))
    if c[1] == trump:
        return can_trump and unseen_higher(c, hand, gone, trick_cards)
    if can_trump:
        return True
    return lead_suit not in voids[p] and c[1] == lead_suit and unseen_higher(c, hand, gone, trick_cards)


def p_holds(c, lead_suit, trump, hand, gone, trick_cards, voids, opps_after, cards_left):
    """Probability no opponent still to play can beat c.

    Every unseen card (including undealt ones) is treated as equally likely to be
    in any given opponent's hand; suits an opponent has shown out of are excluded.
    """
    seen = gone.union(hand, trick_cards)
    unseen = [d for d in DECK if d not in seen]
    prob = 1.0
    for p in opps_after:
        pool = [d for d in unseen if d[1] not in voids[p]]
        b = sum(1 for d in pool if beats(d, c, lead_suit, trump))
        u = len(pool)
        for i in range(cards_left):  # P(none of the b beaters among p's cards)
            if u - i <= 0:
                break
            prob *= max(0, u - b - i) / (u - i)
    return prob


# Rule-based player settings. OLD_RULES reproduces the original players.
OLD_RULES = {
    "pitch": "top",     # first lead: "top" trump; "smart" (Ace, else a middle trump);
                        # "smart2" (smart, but never the Jack if another trump will do);
                        # "keepjack" (A/K/Q on top: lead it; else a middle trump, never the Jack)
    "ruff_min": 0,      # trump in on an off-suit trick only if it holds this much value
    "pull": False,      # keep leading middle trumps while opponents may hold trump
    "lead_aces": True,  # lead off-suit Aces when nothing better
    "schoolboy": False,  # holding A-K of trump, pitch the King (hides the Ace from partner)
    "bank_low": 0,      # trump an opponent's off-suit trick with your 2/3 of trump (banks Low):
                        # 1 = when it surely wins, 2 = also when the trick has points, 3 = always,
                        # 4 = also when the trick holds at least ruff_min
    "bank_jack": 0,     # a Jack that isn't boss: play it the moment it's safe (your side surely
                        # takes the trick). 1 = whenever a higher trump is unseen;
                        # 2 = only if you hold fewer higher trumps than are unseen (no backup)
    "lead10": 0,        # 1 = with no other plain card to lead, lead an off-suit 10 before a trump;
                        # k >= 2 = also lead a 10 over low cards when at most k-1 higher cards
                        # of its suit are unseen
    "pull_t2": False,   # apply "pull" only on the lead to trick 2 (right after the pitch)
    "presmear": 0,      # opponent winning, partner still to play and could beat it: throw an
                        # off-suit 10 for them. 1 = partner plays last, 2 = partner anywhere after
}
RULES = dict(OLD_RULES)


def default_rules(nplayers, role="bidder"):
    """Best settings found head-to-head (and from where search disagreed with the rules),
    for the bidding team (role="bidder") or the other teams (role="defense"):
    don't trump small tricks, but do trump with the 2/3 of trump to bank Low; throw a 10
    to a partner who may still win the trick; don't pitch the Jack; in 6-handed, pitch a
    middle trump when you lack the Ace."""
    bidder = role == "bidder"
    pull = dict(pull=2 if bidder else 1, pull_t2=True)  # keep drawing trump on trick 2
    if nplayers == 4:
        return dict(OLD_RULES, ruff_min=5, pitch="keepjack", bank_low=2, presmear=2,
                    lead10=int(bidder), **pull)
    return dict(OLD_RULES, ruff_min=8, pitch="smart2", presmear=2, lead10=4,
                bank_low=1 if bidder else 2, bank_jack=0 if bidder else 2, **pull)


def team_rules(nplayers, team, bidder=0):
    """default_rules for each team: the bidder's team plays the bidder settings."""
    return [default_rules(nplayers, "bidder" if t == team[bidder] else "defense")
            for t in range(max(team) + 1)]


def choose(me, hand, trick, trump, gone, team, nplayers, low_captured, voids, smear_p=None,
           rules=None):
    """smear_p: None = only count a trick as safe when it's certain; a number =
    count it safe when the estimated chance it holds is at least that.
    rules: see RULES."""
    rules = rules or RULES
    shed = lambda c: (danger(c, trump, low_captured), c[1] == trump, c[0])
    cheap = lambda c: (c[1] == trump, danger(c, trump, low_captured), c[0])
    opps = [p for p in range(nplayers) if team[p] != team[me]]

    if not trick:  # leading
        safe_lead = lambda c: not any(could_beat(p, c, c[1], trump, hand, gone, (), voids) for p in opps)
        opp_trump = any(could_beat(p, (1, trump), trump, trump, hand, gone, (), voids) for p in opps)
        bosses = [c for c in hand if is_boss_trump(c, trump, hand, gone)]
        if bosses and opp_trump:  # pull trump while opponents may still hold some
            return max(bosses)
        # opponents can't beat it (out of trump / suit is boss): cash Aces and 10s
        cash = [c for c in hand if c[1] != trump and safe_lead(c)]
        if cash:
            return max(cash, key=lambda c: (GAME_VAL.get(c[0], 0), c[0]))
        if bosses:
            return max(bosses)
        if rules["pull"] and opp_trump and (not rules.get("pull_t2") or len(gone) == nplayers):
            plain = [c for c in hand if c[1] == trump and danger(c, trump, low_captured) == 0]
            if len(plain) >= rules["pull"]:
                return min(plain)
        off = [c for c in hand if c[1] != trump]
        aces = [c for c in off if c[0] == 14]
        if aces and rules["lead_aces"]:
            return aces[0]
        if rules.get("lead10", 0) >= 2:
            seen = gone.union(hand)
            tens = [c for c in off if c[0] == 10 and
                    sum((r, c[1]) not in seen for r in (11, 12, 13, 14)) <= rules["lead10"] - 1]
            if tens:
                return min(tens)
        safe = [c for c in off if c[0] != 10]
        if safe:
            return min(safe)
        tens = [c for c in off if c[0] == 10]
        if rules.get("lead10", 0) and tens:
            return min(tens)
        return min(hand, key=shed)

    lead_suit = trick[0][1][1]
    opts = legal(hand, lead_suit, trump)
    wp, wc = trick_winner(trick, trump)
    last = len(trick) == nplayers - 1
    pts = sum(danger(c, trump, low_captured) for _, c in trick)
    trick_cards = [c for _, c in trick]
    after_me = [(me + i) % nplayers for i in range(1, nplayers - len(trick))]
    opps_after = [p for p in after_me if team[p] != team[me]]

    def holds(c, odds=False):
        """Card c wins the trick if played now, against everyone still to play.

        odds: also accept "probably holds" (estimated chance >= smear_p)."""
        if not any(could_beat(p, c, lead_suit, trump, hand, gone, trick_cards + [c], voids)
                   for p in opps_after):
            return True
        if smear_p is None or not odds:
            return False
        cards_left = len(hand)  # opponents after me hold as many cards as I do
        return p_holds(c, lead_suit, trump, hand, gone, trick_cards + [c], voids,
                       opps_after, cards_left) >= smear_p

    jack = (11, trump)
    bj = rules.get("bank_jack", 0)
    if bj and jack in opts:
        seen = gone.union(hand, trick_cards)
        unseen_hi = [r for r in (12, 13, 14) if (r, trump) not in seen]
        mine_hi = [r for r in (12, 13, 14) if (r, trump) in hand]
        if unseen_hi and (bj == 1 or len(mine_hi) < len(unseen_hi)):  # not boss, no backup
            if team[wp] == team[me]:
                if last or holds(wc):
                    return jack
            elif beats(jack, wc, lead_suit, trump) and (last or holds(jack)):
                return jack

    if team[wp] == team[me]:
        secure = last or holds(wc, odds=True)
        losers = [c for c in opts if not beats(c, wc, lead_suit, trump)] or opts
        if secure:  # smear points onto partner's trick
            return max(losers, key=lambda c: (danger(c, trump, low_captured), -c[0]))
        return min(losers, key=shed)

    if rules.get("bank_low", 0) and lead_suit != trump:
        low = [c for c in opts if c[1] == trump and c[0] <= 3 and beats(c, wc, lead_suit, trump)]
        if low:
            c = min(low)
            m = rules["bank_low"]
            if m == 3 or last or holds(c) or (m == 2 and pts > 0) or \
                    (m == 4 and pts >= rules["ruff_min"]):
                return c
    winners = [c for c in opts if beats(c, wc, lead_suit, trump)]
    if lead_suit != trump and pts < rules["ruff_min"]:  # not worth a trump
        winners = [c for c in winners if c[1] != trump]
    if winners:
        if last:
            return min(winners, key=cheap)
        sure = [c for c in winners if holds(c)]
        if sure and (pts > 0 or any(c[1] != trump for c in sure)):
            return min(sure, key=cheap)
        nontrump = [c for c in winners if c[1] != trump]
        if nontrump:
            return min(nontrump, key=cheap)
        if pts > 0:
            return min(winners, key=cheap)
    ps = rules.get("presmear", 0)
    if ps:
        pards = [q for q in after_me if team[q] == team[me] and (ps == 2 or q == after_me[-1])]
        if any(could_beat(q, wc, lead_suit, trump, hand, gone, trick_cards, voids) for q in pards):
            tens = [c for c in opts if c[0] == 10 and c[1] != trump and not beats(c, wc, lead_suit, trump)]
            if tens:
                return min(tens)
    return min(opts, key=shed)


def score_won(won, trump, low_captured=True, lo_owner_team=None):
    """Points per team from the cards each team captured."""
    nteams = len(won)
    score = [0] * nteams
    trumps = [c for w in won for c in w if c[1] == trump]
    hi, lo = max(trumps), min(trumps)
    for tm in range(nteams):
        if hi in won[tm]:
            score[tm] += 1
        if (11, trump) in won[tm]:
            score[tm] += 1
        if low_captured and lo in won[tm]:
            score[tm] += 1
    if not low_captured:
        score[lo_owner_team] += 1
    gp = [sum(GAME_VAL.get(c[0], 0) for c in w) for w in won]
    best = max(gp)
    if gp.count(best) == 1:  # tie for game: nobody scores it
        score[gp.index(best)] += 1
    return score


def note_void(voids, p, trick, c, trump):
    if not trick:
        return
    lead_suit = trick[0][1][1]
    if lead_suit == trump and c[1] != trump:
        voids[p].add(trump)
    elif lead_suit != trump and c[1] not in (lead_suit, trump):
        voids[p].add(lead_suit)  # may still hold trump (chose not to ruff)


def finish(hands, trick, leader, gone, won, voids, trump, team, track, pick):
    """Play the hand out from any point. Mutates its arguments.

    pick(p, hands, trick, gone, won, voids) -> card.
    """
    n = len(hands)
    while True:
        if len(trick) == n:
            wp, _ = trick_winner(trick, trump)
            for _, c in trick:
                gone.add(c)
                won[team[wp]].append(c)
            trick = []
            leader = wp
            if not hands[leader]:
                return won
        p = (leader + len(trick)) % n
        c = pick(p, hands, trick, gone, won, voids)
        hands[p].remove(c)
        if track:
            note_void(voids, p, trick, c, trump)
        trick.append((p, c))


def pitch_card(hand, trump, how, schoolboy=False):
    trumps = [c for c in hand if c[1] == trump]
    top = max(trumps)
    if schoolboy and top[0] == 14 and (13, trump) in trumps:
        return (13, trump)
    if how == "top":
        return top
    not_jack = [c for c in trumps if c[0] != 11]
    if how == "keepjack":
        if top[0] >= 12:
            return top
        mid = [c for c in trumps if 4 <= c[0] <= 9]
        return min(mid) if mid else (min(not_jack) if not_jack else top)
    if top[0] == 14:
        return top
    # no Ace: lead a middle trump, keeping the Jack, 10 and low trump to score later
    mid = [c for c in trumps if 4 <= c[0] <= 9 or c[0] in (12, 13)]
    if mid:
        return min(mid)
    if how == "smart2" and not_jack:
        return min(not_jack)
    return top


def heuristic_picker(trump, team, low_captured, smear_p, rules=None, bidder=0):
    """rules: one RULES dict for everyone, a list indexed by team, or None for
    team_rules (bidder and defense defaults)."""
    n = len(team)
    if rules is None:
        rules = team_rules(n, team, bidder)
    elif not isinstance(rules, (list, tuple)):
        rules = [rules] * (max(team) + 1)

    def pick(p, hands, trick, gone, won, voids):
        r = rules[team[p]]
        if not gone and not trick:  # the pitch
            return pitch_card(hands[p], trump, r["pitch"], r.get("schoolboy", False))
        return choose(p, hands[p], trick, trump, gone, team, n, low_captured, voids,
                      smear_p[team[p]], r)
    return pick


def sample_world(me, hands, unseen, voids, rng):
    """Deal the unseen cards to the other players, honoring known voids.

    Leftover unseen cards are the undealt stock. Returns hands (mine unchanged)."""
    n = len(hands)
    others = sorted((q for q in range(n) if q != me), key=lambda q: -len(voids[q]))
    for _ in range(30):
        pool = unseen[:]
        rng.shuffle(pool)
        out = [None] * n
        out[me] = list(hands[me])
        ok = True
        for q in others:
            need = len(hands[q])
            take, rest = [], []
            for d in pool:
                if len(take) < need and d[1] not in voids[q]:
                    take.append(d)
                else:
                    rest.append(d)
            if len(take) < need:
                ok = False
                break
            out[q] = take
            pool = rest
        if ok:
            return out
    # constraints couldn't be met (shouldn't happen): ignore voids
    pool = unseen[:]
    rng.shuffle(pool)
    out = [None] * n
    out[me] = list(hands[me])
    for q in range(n):
        if q != me:
            out[q], pool = pool[:len(hands[q])], pool[len(hands[q]):]
    return out


def search_picker(trump, team, low_captured, track, samples, rng, fallback):
    """Monte Carlo search: try each legal card against `samples` guessed deals of the
    hidden cards, play out the rest with the heuristic players, and keep the card
    with the best average (own team points - everyone else's)."""

    def pick(p, hands, trick, gone, won, voids):
        hand = hands[p]
        if not gone and not trick:
            opts = [c for c in hand if c[1] == trump]  # pitch: any trump
        else:
            lead_suit = trick[0][1][1] if trick else None
            opts = legal(hand, lead_suit, trump)
        if len(opts) == 1:
            return opts[0]
        seen = gone.union(hand, (c for _, c in trick))
        unseen = [d for d in DECK if d not in seen]
        totals = dict.fromkeys(opts, 0.0)
        leader = trick[0][0] if trick else p
        for _ in range(samples):
            world = sample_world(p, hands, unseen, voids, rng)
            for c in opts:
                hs = [list(h) for h in world]
                hs[p].remove(c)
                vs = [set(v) for v in voids]
                if track:
                    note_void(vs, p, trick, c, trump)
                w = finish(hs, trick + [(p, c)], leader, set(gone),
                           [list(x) for x in won], vs, trump, team, track, fallback)
                sc = score_won(w, trump)
                own = sc[team[p]]
                totals[c] += own - (sum(sc) - own)
        return max(opts, key=lambda c: (totals[c], -c[0]))
    return pick


def play(hands, bidder, trump, team, low_captured=True, track=True, smear_p=None,
         search=0, rng=None, rules=None):
    """Play out all tricks. Returns points per team.

    track: players note who has shown out of a suit (public information).
    smear_p: see choose(); a single value for everyone, or a list indexed by team.
    search: deals sampled per decision by the search player (0 = rule-based
      heuristic only); a single value for everyone, or a list indexed by team.
    rules: rule-based player settings (see RULES); one dict or a list by team.
    """
    nteams = max(team) + 1
    if not isinstance(smear_p, (list, tuple)):
        smear_p = [smear_p] * nteams
    if not isinstance(search, (list, tuple)):
        search = [search] * nteams
    n = len(hands)
    dealt_trumps = [(c, p) for p in range(n) for c in hands[p] if c[1] == trump]
    heur = heuristic_picker(trump, team, low_captured, smear_p, rules, bidder)
    rng = rng or random.Random()
    pickers = [search_picker(trump, team, low_captured, track, s, rng, heur) if s else heur
               for s in search]
    won = finish([list(h) for h in hands], [], bidder, set(), [[] for _ in range(nteams)],
                 [set() for _ in range(n)], trump, team, track,
                 lambda p, *a: pickers[team[p]](p, *a))
    lo_card, lo_owner = min(dealt_trumps)
    return score_won(won, trump, low_captured, team[lo_owner])


# ---------------------------------------------------------------- bidding

def suit_strength(hand, s, nplayers=4):
    """The bidder's score for naming suit s (see trump_weights.py); None if not held."""
    from trump_weights import LEN_W, RANK_W
    ranks = [r for r, su in hand if su == s]
    if not ranks:
        return None
    return sum(RANK_W[nplayers][r - 2] for r in ranks) + LEN_W[nplayers][len(ranks)]


def pick_trump(hand, nplayers=4):
    held = [s for s in range(4) if suit_strength(hand, s, nplayers) is not None]
    return max(held, key=lambda s: suit_strength(hand, s, nplayers))


def bid_ev(own, k):
    """Points the bidding team scores on this hand if they bid k (set = -k)."""
    return own if own >= k else -k


def summarize(results):
    """results: list of (own, opp). Returns dict of stats."""
    n = len(results)
    out = {"n": n, "mean": sum(o for o, _ in results) / n}
    for k in BIDS:
        out[f"p{k}"] = sum(o >= k for o, _ in results) / n
        out[f"ev{k}"] = sum(bid_ev(o, k) for o, _ in results) / n
    return out


def recommend(st, threshold=None):
    """Highest bid whose expected points are >= 0 (your max in the auction).

    With `threshold`, instead the highest bid you make at least that often.
    """
    if threshold is None:
        ok = [k for k in BIDS if st[f"ev{k}"] >= 0]
    else:
        ok = [k for k in BIDS if st[f"p{k}"] >= threshold]
    return str(max(ok)) if ok else "pass"


# ---------------------------------------------------------------- simulation

def deal_rest(my_hand, nplayers, rng):
    rest = [c for c in DECK if c not in my_hand]
    rng.shuffle(rest)
    hands = [list(my_hand)]
    for i in range(1, nplayers):
        hands.append(rest[(i - 1) * 6:i * 6])
    return hands


def sim_hand(hand, trump, nplayers, team, n, low_captured, seed, track=True, search=0):
    rng = random.Random(seed)
    res = []
    for _ in range(n):
        hands = deal_rest(hand, nplayers, rng)
        sc = play(hands, 0, trump, team, low_captured, track, search=search, rng=rng)
        own = sc[team[0]]
        res.append((own, sum(sc) - own))
    return res


def features(hand, trump):
    tr = sorted((r for r, s in hand if s == trump), reverse=True)
    top = {14: "A", 13: "K", 12: "Q"}.get(tr[0], "J-")
    low = any(r <= 3 for r in tr)
    jack = 11 in tr
    aces = min(2, sum(1 for r, s in hand if s != trump and r == 14))
    return top, len(tr), aces, jack, low


def survey_worker(args):
    nplayers, team, n, low_captured, seed, track, search = args
    rng = random.Random(seed)
    agg = defaultdict(list)
    for _ in range(n):
        deck = DECK[:]
        rng.shuffle(deck)
        hands = [deck[i * 6:(i + 1) * 6] for i in range(nplayers)]
        trump = pick_trump(hands[0], nplayers)
        sc = play(hands, 0, trump, team, low_captured, track, search=search, rng=rng)
        own = sc[team[0]]
        agg[features(hands[0], trump)].append((own, sum(sc) - own))
    return agg


# ---------------------------------------------------------------- every-hand table (allhands.py)

def canonical_key(hand):
    """Suit-relabeled key of a hand, as allhands.py stores it: the suit holding the
    biggest 13-bit rank mask becomes suit 0, and so on. Returns (key, order) where
    order[i] is the real suit that table suit i stands for."""
    masks = [0] * 4
    for r, su in hand:
        masks[su] |= 1 << (r - 2)
    order = sorted(range(4), key=lambda su: -masks[su])
    return sum(masks[order[i]] << (13 * i) for i in range(4)), order


def table_hand_str(key):
    return " ".join("23456789TJQKA"[c % 13] + "SHDC"[c // 13] for c in range(52) if (key >> c) & 1)


def lookup(hand, nplayers, threshold=None, table=None):
    """Print the every-hand table's row for this hand."""
    import glob
    import pandas as pd
    here = os.path.dirname(os.path.abspath(__file__))
    if table is None:
        found = sorted(glob.glob(os.path.join(here, f"allhands_{nplayers}p_d*_s*.csv.gz")))
        found = [f for f in found if "_first" not in f]
        if not found:
            raise SystemExit(f"no every-hand table for {nplayers}-handed yet (run allhands.py)")
        table = found[-1]
    df = pd.read_csv(table, index_col="hand")
    key, order = canonical_key(hand)
    row = df.loc[table_hand_str(key)]
    trump = order["SHDC".index(row["trump"])]
    st = {k: row[k] for k in ("mean", "p2", "p3", "p4", "ev2", "ev3", "ev4")}
    print(f"{nplayers}-handed   hand: {' '.join(card_str(c) for c in hand)}   "
          f"(table: {os.path.basename(table)}, {row['deals']} deals for this hand)\n")
    print("trump  avg pts  P(>=2)  P(>=3)  P(4)   EV2    EV3    EV4   bid")
    print(f"  {SUITS[trump]}    {st['mean']:5.2f}    {st['p2']:.2f}    {st['p3']:.2f}   "
          f"{st['p4']:.2f}  {st['ev2']:+.2f}  {st['ev3']:+.2f}  {st['ev4']:+.2f}  "
          f"{recommend(st, threshold)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["hand", "survey", "lookup"],
                    help="hand: simulate one hand; survey: bid chart; lookup: read the every-hand "
                         "table built by allhands.py (instant)")
    ap.add_argument("cards", nargs="?", help='6 cards, e.g. "AS JS 2S KH 9D 4C"')
    ap.add_argument("-p", "--players", type=int, default=4, choices=[4, 6])
    ap.add_argument("--team-size", type=int, default=2, choices=[2, 3],
                    help="6-handed: 2 = three teams of two, 3 = two teams of three")
    ap.add_argument("-n", "--sims", type=int, default=None)
    ap.add_argument("--low-dealt", action="store_true",
                    help="Low point goes to whoever was dealt it (default: whoever captures it)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Instead of EV >= 0, bid up to the highest k you make at least this often")
    ap.add_argument("--search", type=int, default=0, metavar="S",
                    help="All players use Monte Carlo search with S sampled deals per decision "
                         "(stronger, much slower; 16-20 is plenty)")
    ap.add_argument("--no-track", action="store_true",
                    help="Players don't note who has shown out of a suit (for comparison)")
    ap.add_argument("--engine", choices=["fast", "gpu", "python"], default="fast",
                    help="fast = compiled CPU engine (needs the .venv); gpu = CUDA engine, "
                         "about 10x faster with --search; python = reference engine")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    if a.players == 4:
        a.team_size = 2
    team = make_teams(a.players, a.team_size)
    low_captured = not a.low_dealt
    label = f"{a.players}-handed, {a.players // a.team_size} teams of {a.team_size}"

    nteams = max(team) + 1
    fast = None
    gpu = None
    if a.engine in ("fast", "gpu"):
        import numpy as np
        import setback_fast as fast
        if a.engine == "gpu":
            import setback_gpu as gpu
        team_np = np.array(team, np.int64)
        rules_np = fast.rules_array(team_rules(a.players, team))
        search_np = np.full(nteams, a.search, np.int64)

    if a.mode == "lookup":
        lookup([parse_card(t) for t in a.cards.split()], a.players, a.threshold)
        return

    if a.mode == "hand":
        hand = [parse_card(t) for t in a.cards.split()]
        assert len(hand) == 6 and len(set(hand)) == 6, "need 6 distinct cards"
        n = a.sims or (4000 if a.search else 100000)
        if gpu:
            n = a.sims or (20000 if a.search else 400000)
        if not fast:
            n = a.sims or (800 if a.search else 20000)
        print(f"{label}   hand: {' '.join(card_str(c) for c in hand)}   ({n} deals per suit)\n")
        print("trump  avg pts  P(>=2)  P(>=3)  P(4)   EV2    EV3    EV4   bid")
        for s in sorted({c[1] for c in hand}, key=lambda s: -suit_strength(hand, s, a.players)):
            if fast:
                mask = np.int64(sum(1 << (su * 13 + r - 2) for r, su in hand))
                hand_seed = a.seed * 1_000_003 + s * 7919
                if gpu:
                    res = gpu.sim_hand(mask, s, n, a.players, team_np, rules_np, search_np,
                                       low_captured, not a.no_track, hand_seed)
                else:
                    res = fast.sim_hand(mask, s, n, a.players, team_np, nteams, low_captured,
                                        not a.no_track, rules_np, search_np, hand_seed)
                st = summarize_df(pd_frame(res[:, 0]))
            else:
                with Pool(cpu_count()) as pool:
                    parts = pool.starmap(sim_hand, [
                        (hand, s, a.players, team, n // cpu_count(), low_captured,
                         a.seed * 1000 + s * 100 + i, not a.no_track, a.search)
                        for i in range(cpu_count())])
                st = summarize([x for part in parts for x in part])
            print(f"  {SUITS[s]}    {st['mean']:5.2f}    {st['p2']:.2f}    {st['p3']:.2f}   "
                  f"{st['p4']:.2f}  {st['ev2']:+.2f}  {st['ev3']:+.2f}  {st['ev4']:+.2f}  "
                  f"{recommend(st, a.threshold)}")
        return

    if gpu:
        n = a.sims or (1000000 if a.search else 10000000)
        out = gpu.survey(n, a.players, team_np, rules_np, search_np, low_captured,
                         not a.no_track, a.seed * 1_000_003)
        groups = survey_groups(out)
    elif fast:
        n = a.sims or (200000 if a.search else 2000000)
        out = fast.survey(n, a.players, team_np, nteams, low_captured, not a.no_track,
                          rules_np, search_np, a.seed * 1_000_003)
        groups = survey_groups(out)
    else:
        n = a.sims or (32000 if a.search else 200000)
        workers = cpu_count()
        with Pool(workers) as pool:
            parts = pool.map(survey_worker, [(a.players, team, n // workers, low_captured,
                                              a.seed * 1000 + i, not a.no_track, a.search)
                                             for i in range(workers)])
        agg = defaultdict(list)
        for part in parts:
            for k, v in part.items():
                agg[k].extend(v)
        groups = {k: summarize(v) for k, v in agg.items()}
        n = sum(st["n"] for st in groups.values())

    players = f"search players ({a.search} deals/decision)" if a.search else "rule-based players"
    print(f"{label}   {n} random hands, bidder names best suit, {players}\n")
    rule = ("max bid with expected points >= 0" if a.threshold is None
            else f"max bid made >= {a.threshold:.0%} of the time")
    print(f"Cell = {rule}  (EV bid 2 / EV bid 3 / EV bid 4).  "
          "EV = points if made, -bid if set.  '.' = too few samples\n")
    cols = [(False, False, "no J, no low"), (True, False, "J"), (False, True, "low (2/3)"),
            (True, True, "J + low")]
    print("off A = Aces held in non-trump suits\n")
    print(f"{'top':>4} {'#tr':>3} {'off A':>5} | " + " | ".join(f"{c[2]:^24}" for c in cols))
    print("-" * 120)
    csv = ["top,ntrump,offsuit_aces,jack,low,n,mean,p2,p3,p4,ev2,ev3,ev4,bid"]
    for top in ["A", "K", "Q", "J-"]:
        for nt in range(1, 7):
            for aces in range(3):
                cells = []
                for jack, low, _ in cols:
                    st = groups.get((top, nt, aces, jack, low))
                    if not st or st["n"] < 100:
                        cells.append(f"{'.':^24}")
                        continue
                    bid = recommend(st, a.threshold)
                    cells.append(f"{bid:>4} ({st['ev2']:+.2f}/{st['ev3']:+.2f}/{st['ev4']:+.2f})")
                    csv.append(f"{top},{nt},{aces},{int(jack)},{int(low)},{st['n']},{st['mean']:.3f},"
                               f"{st['p2']:.3f},{st['p3']:.3f},{st['p4']:.3f},"
                               f"{st['ev2']:.3f},{st['ev3']:.3f},{st['ev4']:.3f},{bid}")
                if any(c.strip() != "." for c in cells):
                    print(f"{top:>4} {nt:>3} {aces if aces < 2 else '2+':>5} | " + " | ".join(cells))
    fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      f"bidchart_{a.players}p_t{a.team_size}{'_lowdealt' if a.low_dealt else ''}"
                      f"{'_notrack' if a.no_track else ''}{f'_search{a.search}' if a.search else ''}.csv")
    with open(fn, "w") as f:
        f.write("\n".join(csv) + "\n")
    print(f"\nfull table -> {fn}")


# ---------------------------------------------------------------- pandas helpers (fast engine)

def pd_frame(own):
    import pandas as pd
    return pd.DataFrame({"own": own})


def summarize_df(df):
    """Same stats as summarize(), from a DataFrame with an `own` column."""
    out = {"n": len(df), "mean": df["own"].mean()}
    for k in BIDS:
        made = df["own"] >= k
        out[f"p{k}"] = made.mean()
        out[f"ev{k}"] = df["own"].where(made, -k).mean()
    return out


def survey_groups(out):
    """Fast-engine survey rows -> {(top, ntrump, aces, jack, low): stats}."""
    import pandas as pd
    import setback_fast as fast
    df = pd.DataFrame({
        "top": pd.Series(out[:, fast.F_TOP]).map({14: "A", 13: "K", 12: "Q"}).fillna("J-"),
        "ntr": out[:, fast.F_NTR], "aces": out[:, fast.F_ACES],
        "jack": out[:, fast.F_JACK].astype(bool), "low": out[:, fast.F_LOW].astype(bool),
        "own": out[:, fast.F_OWN],
    })
    for k in BIDS:
        df[f"made{k}"] = df["own"] >= k
        df[f"ev{k}"] = df["own"].where(df[f"made{k}"], -k)
    g = df.groupby(["top", "ntr", "aces", "jack", "low"]).agg(
        n=("own", "size"), mean=("own", "mean"),
        **{f"p{k}": (f"made{k}", "mean") for k in BIDS},
        **{f"ev{k}": (f"ev{k}", "mean") for k in BIDS})
    return {key: row.to_dict() for key, row in g.iterrows()}

if __name__ == "__main__":
    main()
