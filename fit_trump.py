"""Fit the bidder's trump-choice weights to search results (experiments.py trumpdata).

The bidder names the suit with the highest score: the sum of a weight for each
trump card's rank plus a bonus for the number of trumps. Weights are fitted by
coordinate search to minimize the points lost by naming the wrong suit
(regret), on half the hands; the other half checks the fit. Only hands worth
bidding on (best suit's EV at 2 > -0.5) count.

    .venv/Scripts/python fit_trump.py -p 4
"""
import argparse

import numpy as np

OLD_W = np.array([1.5, 1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1.5, 2.5, 2.2, 3.0, 4.0])


def features(hands):
    """(hands, 4 suits, 13 rank flags + 6 length flags)."""
    x = np.zeros((len(hands), 4, 19))
    for s in range(4):
        m = (hands >> (13 * s)) & 0x1FFF
        for r in range(13):
            x[:, s, r] = (m >> r) & 1
        n = x[:, s, :13].sum(axis=1).astype(int)
        for k in range(1, 7):
            x[:, s, 12 + k] = n == k
    return x


def regret(theta, x, val, held):
    score = x @ theta
    score[~held] = -np.inf
    pick = np.argmax(score, axis=1)  # ties -> lowest suit, as in the engine
    return np.mean(np.nanmax(val, axis=1) - val[np.arange(len(val)), pick])


def fit(x, val, held, theta):
    best = regret(theta, x, val, held)
    for step in (1.0, 0.5, 0.25, 0.1, 0.05):
        improved = True
        while improved:
            improved = False
            for i in range(len(theta)):
                for d in (-step, step):
                    t = theta.copy()
                    t[i] += d
                    r = regret(t, x, val, held)
                    if r < best - 1e-6:
                        theta, best, improved = t, r, True
    return theta, best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--players", type=int, default=4, choices=[4, 6])
    a = ap.parse_args()
    z = np.load(f"trumpdata_{a.players}p.npz")
    hands = z["hands"]
    held = ~np.isnan(z["ev2"])
    keep = (held.sum(axis=1) >= 2) & (np.nanmax(z["ev2"], axis=1) > -0.5)
    x, held = features(hands[keep]), held[keep]
    v2, v3 = z["ev2"][keep], z["ev3"][keep]
    half = len(x) // 2
    tr, te = slice(0, half), slice(half, None)
    old = np.concatenate([OLD_W, np.zeros(6)])
    theta, _ = fit(x[tr], v2[tr], held[tr], old.copy())
    print(f"{a.players}-handed: {keep.sum():,} hands with a real choice of trump")
    for name, t in (("old weights", old), ("fitted", theta)):
        print(f"  {name:12} regret at bid 2: train {regret(t, x[tr], v2[tr], held[tr]):.4f}  "
              f"test {regret(t, x[te], v2[te], held[te]):.4f}   at bid 3 (test) {regret(t, x[te], v3[te], held[te]):.4f}  pts/hand")
    print("  rank weights  " + "  ".join(f"{'23456789TJQKA'[r]}:{theta[r]:.2f}" for r in range(13)))
    print("  length bonus  " + "  ".join(f"{k}:{theta[12 + k]:+.2f}" for k in range(1, 7)))
    np.save(f"trump_theta_{a.players}p.npy", theta)


if __name__ == "__main__":
    main()
