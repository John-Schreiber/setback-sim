"""How the bidder picks trump: the suit with the highest score, where a suit's score
is the sum of its cards' rank weights plus a bonus for how many cards it has.
Ties go to the lower suit (S, H, D, C). Weights are in hundredths, so every
engine adds them up exactly the same way.

Fitted by fit_trump.py to search results (experiments.py trumpdata): the
weights that lose the fewest points to naming the wrong suit. The original
hand-set weights were 2:150 3:100 4-9:80 10:150 J:250 Q:220 K:300 A:400, no bonus.

Shared by setback.py (reference engine) and engine_core.py (CPU and GPU).
"""
import numpy as np

#                 2    3    4    5    6    7    8    9   10    J    Q    K    A
RANK_W = {4: [125, 100, 60, 65, 75, 85, 75, 100, 125, 310, 205, 290, 450],
          6: [125, 75, 85, 90, 80, 75, 85, 90, 100, 240, 175, 275, 565]}
#            0 cards (unused), 1 .. 6 cards
LEN_W = {4: [0, 0, -20, 0, 5, -75, 0],
         6: [0, 0, -5, -10, 20, -50, 0]}

TW = np.array([RANK_W[4], RANK_W[6]], np.int64)
TL = np.array([LEN_W[4], LEN_W[6]], np.int64)
