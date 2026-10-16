// Play-out kernel for search players, in CUDA C++: the same rule-based play as core.py
// (rollout, heur_pick, choose, pitch_card, scoring), so the GPU search plays exactly the
// games the Numba kernel gpu.k_rollout plays. Compiled at run time with NVRTC by
// gpu_cpp.py, which prepends the table, flag and rule constants from tables.py and
// core.py (F_*, M_*, G_*, R_*, MAXP, MAXT, MAXOPT, N_FIELDS, N_CARDS).
//
// Why C++: Numba widens 32-bit integer arithmetic to 64 bits, which on a GPU doubles the
// instructions and registers for every index, power and key. Here only the card masks are
// 64-bit. Function by function this mirrors core.py; keep the two in step (the tests
// require identical results).
//
// Divergence (Nsight Compute: about half the lanes idle) is cut where that costs no extra
// work: small decisions (beats, could_beat, voids, the running trick winner) are selects,
// not branches; teams sit in a register; seat arithmetic avoids the integer modulo. Loops
// over a hand's cards keep their early exits: running every thread to the worst case
// (all hand slots, every opponent) measured 15-23% slower than the divergence it saved.

typedef unsigned long long u64;
typedef long long i64;

struct Ctx {
    const i64* __restrict__ tab;  // tab[t][field][i] (tables.py), this game's trump t
    const i64* __restrict__ gf;   // game flags
    const i64* __restrict__ rules;  // rules[team][R_*]
    int t, n, hsize;   // trump, players, cards per hand (bounds every hand-mask loop)
    unsigned teams;    // team of seat p in bits 2p..2p+1
};

// ------------------------------------------------------------------ bits and tables

__device__ __forceinline__ u64 bit(int c) { return 1ull << c; }
__device__ __forceinline__ int lowest(u64 m) { return __ffsll((long long)m) - 1; }  // -1 if none
__device__ __forceinline__ int highest(u64 m) { return 63 - __clzll((long long)m); }  // -1 if none
__device__ __forceinline__ int popcount(u64 m) { return __popcll(m); }

__device__ __forceinline__ i64 T(const Ctx& x, int f, int i) { return __ldg(&x.tab[(x.t * N_FIELDS + f) * N_CARDS + i]); }
__device__ __forceinline__ u64 TM(const Ctx& x, int f, int i) { return (u64)T(x, f, i); }
__device__ __forceinline__ int TI(const Ctx& x, int f, int i) { return (int)T(x, f, i); }
__device__ __forceinline__ int ES(const Ctx& x, int c) { return TI(x, F_ESUIT, c); }
__device__ __forceinline__ int POW(const Ctx& x, int c) { return TI(x, F_POWER, c); }
__device__ __forceinline__ u64 ESM(const Ctx& x, int s) { return TM(x, F_ESM, s); }
__device__ __forceinline__ u64 ABOVE(const Ctx& x, int c) { return TM(x, F_ABOVE, c); }
__device__ __forceinline__ i64 MSK(const Ctx& x, int m) { return T(x, F_MASKS, m); }
__device__ __forceinline__ int G(const Ctx& x, int g) { return (int)__ldg(&x.gf[g]); }
__device__ __forceinline__ int R(const Ctx& x, int tm, int r) { return (int)__ldg(&x.rules[tm * N_RULES + r]); }
__device__ __forceinline__ int TEAM(const Ctx& x, int p) { return (x.teams >> (2 * p)) & 3; }
// (a + b) % n for 0 <= a, b < n, as a select instead of an integer division
__device__ __forceinline__ int seat(int a, int b, int n) { int s = a + b; return s >= n ? s - n : s; }
__device__ __forceinline__ int natural_card(int rank, int suit) { return suit * 13 + rank - 2; }

__device__ int top_trump(const Ctx& x, u64 m) {
    u64 hi = m & (u64)MSK(x, M_TR_HIGH);
    if (hi) return highest(hi);
    int c = (int)MSK(x, M_OFFJ);
    if (c >= 0 && ((m >> c) & 1)) return c;
    c = (int)MSK(x, M_BJ);
    if (c >= 0 && ((m >> c) & 1)) return c;
    c = (int)MSK(x, M_LJ);
    if (c >= 0 && ((m >> c) & 1)) return c;
    return highest(m & (u64)MSK(x, M_TR_LOW));
}

__device__ int low_trump(const Ctx& x, u64 m) {
    u64 lo = m & (u64)MSK(x, M_TR_LOW);
    if (lo) return lowest(lo);
    int c = (int)MSK(x, M_LJ);
    if (c >= 0 && ((m >> c) & 1)) return c;
    c = (int)MSK(x, M_BJ);
    if (c >= 0 && ((m >> c) & 1)) return c;
    c = (int)MSK(x, M_OFFJ);
    if (c >= 0 && ((m >> c) & 1)) return c;
    return lowest(m & (u64)MSK(x, M_TR_HIGH));
}

// ------------------------------------------------------------------ rules of play

__device__ __forceinline__ u64 legal(const Ctx& x, u64 hand, int lead) {
    if (lead < 0) return hand;
    u64 follow = hand & ESM(x, lead);
    if (!follow) return hand;
    if (lead == x.t || !G(x, G_MAY_TRUMP)) return follow;
    return follow | (hand & ESM(x, x.t));  // may always trump in
}

__device__ __forceinline__ u64 lead_mask(const Ctx& x, u64 hand, u64 gone) {
    u64 tm = ESM(x, x.t);
    u64 nt = hand & ~tm;
    if (nt) {
        if (gone == 0 && G(x, G_FIRST_LEAD) == 2) return nt;
        if (G(x, G_TRUMP_BROKEN) && (gone & tm) == 0) return nt;
    }
    return hand;
}

// (bitwise & | on bools below are deliberate: both sides are cheap and evaluating them
// avoids the branch a short-circuit && would need)
__device__ __forceinline__ bool beats(const Ctx& x, int a, int b, int lead) {
    const int sa = ES(x, a), sb = ES(x, b);
    const bool at = sa == x.t, bt = sb == x.t, higher = POW(x, a) > POW(x, b);
    return (at & (!bt | higher)) | (!at & !bt & (sa == lead) & higher);
}

__device__ __forceinline__ void trick_winner(const Ctx& x, const int* tp, const int* tc, int tlen, int& wp, int& wc) {
    const int lead = ES(x, tc[0]);
    wp = tp[0];
    wc = tc[0];
    for (int i = 1; i < tlen; i++) {
        const bool b = beats(x, tc[i], wc, lead);
        wp = b ? tp[i] : wp;
        wc = b ? tc[i] : wc;
    }
}

__device__ __forceinline__ bool could_beat(const Ctx& x, int p, int c, int lead, u64 seen, const int* voids) {
    const bool can_trump = (((voids[p] >> x.t) & 1) == 0) & ((ESM(x, x.t) & ~seen) != 0);
    const bool above = (ABOVE(x, c) & ~seen) != 0;
    const int sc = ES(x, c);
    const bool follows = (((voids[p] >> lead) & 1) == 0) & (sc == lead) & above;
    return sc == x.t ? (can_trump & above) : (can_trump | follows);
}

__device__ __forceinline__ void note_void(const Ctx& x, int* voids, int p, const int* tc, int tlen, int c) {
    if (tlen == 0) return;
    const int lead = ES(x, tc[0]), s = ES(x, c), t = x.t;
    const bool trump_led = lead == t;
    // trump led, not followed: void in trump; else off-suit, not followed: void in the led suit
    // (may still hold trump: chose not to ruff); else trumped in where one must follow: a void
    const bool v_trump = trump_led & (s != t);
    const bool v_lead = !trump_led & (((s != lead) & (s != t)) | ((s == t) & !G(x, G_MAY_TRUMP)));
    voids[p] |= (v_trump ? 1 << t : 0) | (v_lead ? 1 << lead : 0);
}

// lowest key wins; ties go to the lower card (ascending scan, strict compare)
__device__ __forceinline__ int min_key(const Ctx& x, u64 m, int field) {
    int best = -1;
    i64 bk = 0x7FFFFFFFFFFFFFFFll;
    while (m) {
        const int c = lowest(m);
        m &= m - 1;
        const i64 k = T(x, field, c);
        best = k < bk ? c : best;
        bk = k < bk ? k : bk;
    }
    return best;
}

// card c wins the trick against every opponent still to play (certain)
__device__ __forceinline__ bool holds(const Ctx& x, int c, int me, int tlen, int lead, u64 seen, const int* voids) {
    const int n = x.n, tme = TEAM(x, me);
    for (int i = 1; i < n - tlen; i++) {
        const int p = seat(me, i, n);
        if (TEAM(x, p) != tme && could_beat(x, p, c, lead, seen, voids)) return false;
    }
    return true;
}

__device__ int choose(const Ctx& x, int me, u64 hand, u64 leads, const int* tp, const int* tc, int tlen,
                      u64 gone, const int* voids, int rt) {
    const int t = x.t, n = x.n, tme = TEAM(x, me);
    const u64 TMK = ESM(x, t);
    const int ruff_min = R(x, rt, R_RUFF_MIN), pull = R(x, rt, R_PULL), lead_aces = R(x, rt, R_LEAD_ACES);

    if (tlen == 0) {  // leading
        u64 seen = gone | hand;
        u64 unseen_tr = TMK & ~seen;
        bool opp_trump = false;
        for (int p = 0; p < n; p++)
            if (TEAM(x, p) != tme && ((voids[p] >> t) & 1) == 0 && unseen_tr != 0) opp_trump = true;
        int top = top_trump(x, hand & TMK);
        bool boss = top >= 0 && (ABOVE(x, top) & ~seen) == 0 && ((leads >> top) & 1) == 1;
        if (boss && opp_trump) return top;  // pull trump while opponents may still hold some
        // opponents can't beat it: cash Aces and 10s
        int best = -1;
        i64 bk = -1;
        u64 m = leads & ~TMK;
        while (m) {
            const int c = lowest(m);
            m &= m - 1;
            bool safe = true;
            for (int p = 0; p < n; p++) {
                if (TEAM(x, p) == tme) continue;
                if (could_beat(x, p, c, ES(x, c), seen, voids)) { safe = false; break; }
            }
            const i64 k = safe ? T(x, F_CASH, c) : bk;
            best = k > bk ? c : best;
            bk = k > bk ? k : bk;
        }
        if (best >= 0) return best;
        if (boss) return top;
        if (pull > 0 && opp_trump && (!R(x, rt, R_PULL_T2) || popcount(gone) == n)) {
            u64 plain = leads & (u64)MSK(x, M_SAFE_TR);
            if (popcount(plain) >= pull) return low_trump(x, plain);
        }
        u64 off = leads & ~TMK;
        u64 aces = off & (u64)MSK(x, M_ACES), tens = off & (u64)MSK(x, M_TENS);
        if (lead_aces && aces) return lowest(aces);
        int lead10 = R(x, rt, R_LEAD10);
        if (lead10 >= 2) {
            u64 m2 = tens;
            while (m2) {
                int c = lowest(m2);
                m2 &= m2 - 1;
                if (popcount(ABOVE(x, c) & ~seen) <= lead10 - 1) return c;
            }
        }
        best = min_key(x, off & ~(u64)MSK(x, M_TENS), F_SAFE);
        if (best >= 0) return best;
        if (lead10 && tens) return lowest(tens);
        return min_key(x, leads, F_SHED);
    }

    const int lead = ES(x, tc[0]);
    const u64 opts = legal(x, hand, lead);
    int wp, wc;
    trick_winner(x, tp, tc, tlen, wp, wc);
    const bool last = tlen == n - 1;
    int pts = 0;
    u64 tmask = 0;
    for (int i = 0; i < tlen; i++) {
        pts += TI(x, F_DANGER, tc[i]);
        tmask |= bit(tc[i]);
    }
    const u64 seen = gone | hand | tmask;
    u64 beaters = ABOVE(x, wc);
    if (ES(x, wc) != t) beaters |= TMK;
    beaters &= opts;
    const bool partner_wins = TEAM(x, wp) == tme;

    const int bj = R(x, rt, R_BANK_JACK);
    const int jack = natural_card(11, t);
    if (bj && ((opts >> jack) & 1)) {
        u64 unseen_hi = ABOVE(x, jack) & ~seen;
        if (unseen_hi && (bj == 1 || popcount(ABOVE(x, jack) & hand) < popcount(unseen_hi))) {
            // a Jack that isn't boss and isn't backed up: play it the moment it's safe
            if (partner_wins) {
                if (last || holds(x, wc, me, tlen, lead, seen, voids)) return jack;
            } else if (beats(x, jack, wc, lead) && (last || holds(x, jack, me, tlen, lead, seen | bit(jack), voids))) {
                return jack;
            }
        }
    }

    if (partner_wins) {
        bool secure = last || holds(x, wc, me, tlen, lead, seen, voids);
        u64 losers = opts & ~beaters;
        if (losers == 0) losers = opts;
        if (secure) {  // smear points onto partner's trick; on ties a card of the led suit first
            int best = -1;
            i64 bk = (i64)0x8000000000000000ull;
            const u64 led = ESM(x, lead);
            for (int part = 0; part < 2; part++) {
                u64 m = part == 0 ? (losers & led) : (losers & ~led);
                while (m) {
                    const int c = lowest(m);
                    m &= m - 1;
                    const i64 k = T(x, F_SMEAR, c);
                    best = k > bk ? c : best;
                    bk = k > bk ? k : bk;
                }
            }
            return best;
        }
        return min_key(x, losers, F_SHED);
    }

    const int bank_low = R(x, rt, R_BANK_LOW);
    if (bank_low && lead != t) {
        // trump with the 2/3 of trump: takes the trick and banks the Low point
        u64 low = beaters & TMK & (bit(natural_card(2, t)) | bit(natural_card(3, t)));
        if (low) {
            int c = low_trump(x, low);
            if (bank_low == 3 || last || (bank_low == 2 && pts > 0) || (bank_low == 4 && pts >= ruff_min) ||
                holds(x, c, me, tlen, lead, seen | bit(c), voids))
                return c;
        }
    }
    u64 winners = beaters;
    if (lead != t && pts < ruff_min) winners &= ~TMK;  // not worth a trump
    if (winners) {
        if (last) return min_key(x, winners, F_CHEAP);
        u64 sure = 0, m = winners;
        while (m) {
            const int c = lowest(m);
            m &= m - 1;
            if (holds(x, c, me, tlen, lead, seen | bit(c), voids)) sure |= bit(c);
        }
        if (sure && (pts > 0 || (sure & ~TMK) != 0)) return min_key(x, sure, F_CHEAP);
        if (winners & ~TMK) return min_key(x, winners & ~TMK, F_CHEAP);
        if (pts > 0) return min_key(x, winners, F_CHEAP);
    }
    const int ps = R(x, rt, R_PRESMEAR);
    if (ps) {
        // partner still to play could beat the winner: throw them an off-suit 10
        bool pard_can = false;
        for (int i = 1; i < n - tlen; i++) {
            int q = seat(me, i, n);
            if (TEAM(x, q) == tme && (ps == 2 || i == n - tlen - 1) && could_beat(x, q, wc, lead, seen, voids))
                pard_can = true;
        }
        if (pard_can) {
            u64 m = opts & ~TMK & ~beaters;
            while (m) {
                int c = lowest(m);
                m &= m - 1;
                if (TI(x, F_NRANK, c) == 10) return c;
            }
        }
    }
    return min_key(x, opts, F_SHED);
}

__device__ int pitch_card(const Ctx& x, u64 hand, int mode, int schoolboy) {
    const int t = x.t;
    u64 tm = hand & ESM(x, t);
    int top = top_trump(x, tm);
    if (top < 0) return lowest(hand);  // no trump to pitch: lead anything
    int ace = natural_card(14, t), king = natural_card(13, t), jack = natural_card(11, t);
    if (schoolboy && top == ace && ((tm >> king) & 1)) return king;  // schoolboy pitch
    if (mode == 0) return top;
    u64 not_jack = tm & ~bit(jack);
    u64 natural = tm & ESM(x, t) & (0x1FFFull << (13 * t));
    int top_rank = ((natural >> top) & 1) ? TI(x, F_NRANK, top) : 11;
    u64 mid = natural & (u64)MSK(x, M_MID), mid2 = natural & (u64)MSK(x, M_MID2);
    if (mode == 2) {  // keepjack
        if (top_rank >= 12) return top;
        if (mid) return lowest(mid);
        return not_jack ? low_trump(x, not_jack) : top;
    }
    if (top == ace) return top;
    if (mid2) return lowest(mid2);
    if (mode == 3 && not_jack) return low_trump(x, not_jack);
    return top;
}

__device__ __forceinline__ int heur_pick(const Ctx& x, int p, const u64* hands, const int* tp, const int* tc, int tlen,
                                         u64 gone, const int* voids) {
    const int rt = TEAM(x, p);
    if (gone == 0 && tlen == 0 && G(x, G_FIRST_LEAD) == 0)  // the pitch
        return pitch_card(x, hands[p], R(x, rt, R_PITCH), R(x, rt, R_SCHOOLBOY));
    u64 leads = tlen == 0 ? lead_mask(x, hands[p], gone) : hands[p];
    return choose(x, p, hands[p], leads, tp, tc, tlen, gone, voids, rt);
}

__device__ __forceinline__ u64 options(const Ctx& x, u64 hand, u64 gone, const int* tc, int tlen) {
    if (tlen) return legal(x, hand, ES(x, tc[0]));
    if (gone == 0 && G(x, G_FIRST_LEAD) == 0) return hand & ESM(x, x.t);  // the pitch: any trump
    return lead_mask(x, hand, gone);
}

// ------------------------------------------------------------------ scoring

__device__ __forceinline__ int game_points(const Ctx& x, u64 m) {
    return 10 * popcount(m & (u64)MSK(x, M_TENS)) + 4 * popcount(m & (u64)MSK(x, M_ACES)) +
           3 * popcount(m & (u64)MSK(x, M_KINGS)) + 2 * popcount(m & (u64)MSK(x, M_QUEENS)) +
           popcount(m & (u64)MSK(x, M_JACKS));
}

__device__ __forceinline__ int card_points(const Ctx& x, u64 m) {
    int g = 0;
    m &= (u64)MSK(x, M_POINTS);
    while (m) {
        int c = lowest(m);
        m &= m - 1;
        g += TI(x, F_PTS, c);
    }
    return g;
}

// what search maximizes: my team's points minus everyone else's (Low to the captor)
__device__ int search_value(const Ctx& x, const u64* won, int nteams, int my_team, int last_team) {
    u64 alltr = 0;
    for (int k = 0; k < nteams; k++) alltr |= won[k] & ESM(x, x.t);
    int hi = top_trump(x, alltr), lo = low_trump(x, alltr);
    int own = 0, total = 0, best = -1, nbest = 0, bt = -1;
    for (int k = 0; k < nteams; k++) {
        int pts = card_points(x, won[k]);
        if (hi >= 0 && ((won[k] >> hi) & 1)) pts += G(x, G_HIGH);
        if (lo >= 0 && ((won[k] >> lo) & 1)) pts += G(x, G_LOW);
        if (k == last_team) pts += G(x, G_LAST);
        total += pts;
        if (k == my_team) own += pts;
        int g = game_points(x, won[k]);
        if (g > best) { best = g; nbest = 1; bt = k; }
        else if (g == best) nbest++;
    }
    if (nbest == 1) {
        total += G(x, G_GAME);
        if (bt == my_team) own += G(x, G_GAME);
    }
    return 2 * own - total;
}

// play out from any point with rule-based players; returns the team that took the last trick
__device__ int rollout(const Ctx& x, u64* hands, int* tp, int* tc, int tlen, int leader, u64 gone, u64* won,
                       int* voids, int track) {
    const int n = x.n;
    int last_team = -1;
    while (true) {
        if (tlen == n) {
            int wp, wc;
            trick_winner(x, tp, tc, tlen, wp, wc);
            int wt = TEAM(x, wp);
            for (int i = 0; i < n; i++) {
                gone |= bit(tc[i]);
                won[wt] |= bit(tc[i]);
            }
            tlen = 0;
            leader = wp;
            last_team = wt;
            if (hands[leader] == 0) return last_team;
        }
        int p = seat(leader, tlen, n);
        int c = heur_pick(x, p, hands, tp, tc, tlen, gone, voids);
        hands[p] &= ~bit(c);
        if (track) note_void(x, voids, p, tc, tlen, c);
        tp[tlen] = p;
        tc[tlen] = c;
        tlen++;
    }
}

__device__ __forceinline__ int kth_card(u64 m, int k) {
    for (int i = 0; i < k; i++) m &= m - 1;
    return lowest(m);
}

// One thread per (work item, sample), grid-stride, as gpu.k_rollout: work item = (game, k-th legal
// card); finish the hand from the sampled world and store the search value in vals[g, s, k].
extern "C" __global__ void k_rollout_cpp(
    const i64* __restrict__ hands, const i64* __restrict__ gone, const i64* __restrict__ won,
    const i64* __restrict__ voids, const i64* __restrict__ tp, const i64* __restrict__ tc, int tlen,
    const i64* __restrict__ leader, const i64* __restrict__ trumps, const i64* __restrict__ team, int n,
    int nteams, const i64* __restrict__ search, int nsamp, const i64* __restrict__ rules, int track,
    const i64* __restrict__ worlds, double* __restrict__ vals, const int* __restrict__ work_g,
    const int* __restrict__ work_k, const int* __restrict__ nwork, const i64* __restrict__ tab,
    const i64* __restrict__ gf) {
    const long long total = (long long)nwork[0] * nsamp;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x; idx < total;
         idx += (long long)gridDim.x * blockDim.x) {
        const int wi = (int)(idx / nsamp), s = (int)(idx % nsamp);
        const int g = work_g[wi], k = work_k[wi];
        unsigned teams = 0;
        for (int q = 0; q < n; q++) teams |= (unsigned)team[q] << (2 * q);
        const int p = seat((int)leader[g], tlen, n);
        if (s >= (int)search[team[p]]) continue;
        Ctx x{tab, gf, rules, (int)trumps[g], n, (int)gf[G_HAND], teams};
        const u64 hand = (u64)hands[g * n + p];
        int tcg[MAXP];
        for (int q = 0; q < MAXP; q++) tcg[q] = q < n ? (int)tc[g * n + q] : 0;
        const int c = kth_card(options(x, hand, (u64)gone[g], tcg, tlen), k);
        u64 hs[MAXP], w[MAXT];
        int vs[MAXP], tp2[MAXP], tc2[MAXP];
        for (int q = 0; q < MAXP; q++) {
            hs[q] = q < n ? (u64)worlds[((long long)g * nsamp + s) * n + q] : 0;
            vs[q] = q < n ? (int)voids[g * n + q] : 0;
            tp2[q] = q < n ? (int)tp[g * n + q] : 0;
            tc2[q] = tcg[q];
        }
        for (int q = 0; q < MAXT; q++) w[q] = q < nteams ? (u64)won[g * nteams + q] : 0;
        hs[p] = hand & ~bit(c);
        if (track) note_void(x, vs, p, tc2, tlen, c);
        tp2[tlen] = p;
        tc2[tlen] = c;
        const int lead_p = tlen ? tp2[0] : p;
        const int last = rollout(x, hs, tp2, tc2, tlen + 1, lead_p, (u64)gone[g], w, vs, track);
        vals[((long long)g * nsamp + s) * MAXOPT + k] = (double)search_value(x, w, nteams, TEAM(x, p), last);
    }
}
