## setback4_lowdealt, 6-handed

Highest bid that beats passing: expected points against passing (vs_pass: the score change minus the opponents' points on the hand, minus what passing costs, 1.625 here; cardsim/pitch/lines.py) >= 0. The .csv's max_bid is the older line, expected points >= 0 (bids 2-4; 'pass' = none), by trump honors held (rows) and number of other small trumps (columns). From every 6-card hand (reduced every-hand tables, search players); (n) = too few hands seen (200), so taken from the weaker holdings next to it.

| trump honors | +0 | +1 | +2 | +3 | +4 | +5 | +6 |
|---|---|---|---|---|---|---|---|
| A K Q J 10 2 | 4 |  |  |  |  |  |  |
| A K Q J 10 | 3 | 3 |  |  |  |  |  |
| A K Q J 2 | 4 | 4 |  |  |  |  |  |
| A K Q J | 3 | 3 | 3 |  |  |  |  |
| A K Q 10 2 | 4 | 4 |  |  |  |  |  |
| A K Q 10 | 3 | 3 | 3 |  |  |  |  |
| A K Q 2 | 4 | 4 | 4 |  |  |  |  |
| A K Q | 3 | 3 | 3 | 3 |  |  |  |
| A K J 10 2 | 4 | 4 |  |  |  |  |  |
| A K J 10 | 3 | 3 | 3 |  |  |  |  |
| A K J 2 | 4 | 4 | 4 |  |  |  |  |
| A K J | 3 | 3 | 3 | 3 |  |  |  |
| A K 10 2 | 4 | 4 | 4 |  |  |  |  |
| A K 10 | 3 | 3 | 3 | 3 |  |  |  |
| A K 2 | 4 | 4 | 4 | 4 |  |  |  |
| A K | 2 | 3 | 3 | 3 | 3 |  |  |
| A Q J 10 2 | 4 | 4 |  |  |  |  |  |
| A Q J 10 | 3 | 3 | 3 |  |  |  |  |
| A Q J 2 | 4 | 4 | 4 |  |  |  |  |
| A Q J | 3 | 3 | 3 | 3 |  |  |  |
| A Q 10 2 | 4 | 4 | 4 |  |  |  |  |
| A Q 10 | 3 | 3 | 3 | 3 |  |  |  |
| A Q 2 | 3 | 3 | 4 | 4 |  |  |  |
| A Q | 2 | 3 | 3 | 3 | 3 |  |  |
| A J 10 2 | 4 | 4 | 4 |  |  |  |  |
| A J 10 | 3 | 3 | 3 | 3 |  |  |  |
| A J 2 | 4 | 4 | 4 | 4 |  |  |  |
| A J | 2 | 3 | 3 | 3 | 4 |  |  |
| A 10 2 | 3 | 3 | 3 | 3 |  |  |  |
| A 10 | 2 | 2 | 3 | 3 | 3 |  |  |
| A 2 | 3 | 3 | 3 | 3 | 3 |  |  |
| A | 2 | 2 | 2 | 3 | 3 | 3 |  |
| K Q J 10 2 | 4 | 4 |  |  |  |  |  |
| K Q J 10 | 3 | 3 | 3 |  |  |  |  |
| K Q J 2 | 4 | 4 | 4 |  |  |  |  |
| K Q J | 3 | 3 | 3 | 3 |  |  |  |
| K Q 10 2 | 3 | 3 | 3 |  |  |  |  |
| K Q 10 | 2 | 2 | 2 | 2 |  |  |  |
| K Q 2 | 3 | 3 | 3 | 3 |  |  |  |
| K Q | pass | 2 | 2 | 2 | 2 |  |  |
| K J 10 2 | 4 | 4 | 4 |  |  |  |  |
| K J 10 | 2 | 3 | 3 | 3 |  |  |  |
| K J 2 | 3 | 4 | 4 | 4 |  |  |  |
| K J | 2 | 2 | 3 | 3 | 3 |  |  |
| K 10 2 | 2 | 3 | 3 | 3 |  |  |  |
| K 10 | pass | pass | 2 | 2 | 2 |  |  |
| K 2 | 2 | 2 | 3 | 3 | 3 |  |  |
| K | pass | pass | pass | 2 | 2 | 2 |  |
| Q J 10 2 | 3 | 3 | 3 |  |  |  |  |
| Q J 10 | 2 | 2 | 3 | 3 |  |  |  |
| Q J 2 | 3 | 3 | 3 | 3 |  |  |  |
| Q J | pass | 2 | 2 | 3 | 3 |  |  |
| Q 10 2 | 2 | 2 | 2 | 2 |  |  |  |
| Q 10 | pass | pass | pass | 2 | 2 |  |  |
| Q 2 | pass | 2 | 2 | 2 | 2 |  |  |
| Q | pass | pass | pass | pass | 2 | 2 |  |
| J 10 2 | 3 | 3 | 3 | 3 |  |  |  |
| J 10 | pass | 2 | 2 | 3 | 3 |  |  |
| J 2 | 2 | 3 | 3 | 3 | 3 |  |  |
| J | pass | pass | 2 | 2 | 2 | 3 |  |
| 10 2 | pass | 2 | 2 | 2 | 2 |  |  |
| 10 |  | pass | pass | pass | pass | 2 |  |
| 2 |  | pass | pass | 2 | 2 | 2 |  |
| (none) |  |  | pass | pass | pass | pass | pass |
