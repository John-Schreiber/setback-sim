## setback4_must_follow, 6-handed

Highest bid worth making by the line that wins the most whole games: net expected points (score change minus the opponents' points on the hand) >= -1.75 (cardsim/pitch/passline.py; the .csv's max_bid is the older line, expected points >= 0) (bids 2-4; 'pass' = none), by trump honors held (rows) and number of other small trumps (columns). From every 6-card hand (reduced every-hand tables, search players); (n) = too few hands seen (200), so taken from the weaker holdings next to it.

| trump honors | +0 | +1 | +2 | +3 | +4 | +5 | +6 |
|---|---|---|---|---|---|---|---|
| A K Q J 10 2 | 4 |  |  |  |  |  |  |
| A K Q J 10 | 4 | 4 |  |  |  |  |  |
| A K Q J 2 | 4 | 4 |  |  |  |  |  |
| A K Q J | 4 | 4 | 4 |  |  |  |  |
| A K Q 10 2 | 4 | 4 |  |  |  |  |  |
| A K Q 10 | 4 | 4 | 4 |  |  |  |  |
| A K Q 2 | 4 | 4 | 4 |  |  |  |  |
| A K Q | 4 | 4 | 4 | 4 |  |  |  |
| A K J 10 2 | 4 | 4 |  |  |  |  |  |
| A K J 10 | 4 | 4 | 4 |  |  |  |  |
| A K J 2 | 4 | 4 | 4 |  |  |  |  |
| A K J | 4 | 4 | 4 | 4 |  |  |  |
| A K 10 2 | 4 | 4 | 4 |  |  |  |  |
| A K 10 | 4 | 4 | 4 | 4 |  |  |  |
| A K 2 | 4 | 4 | 4 | 4 |  |  |  |
| A K | 3 | 4 | 4 | 4 | 4 |  |  |
| A Q J 10 2 | 4 | 4 |  |  |  |  |  |
| A Q J 10 | 4 | 4 | 4 |  |  |  |  |
| A Q J 2 | 4 | 4 | 4 |  |  |  |  |
| A Q J | 4 | 4 | 4 | 4 |  |  |  |
| A Q 10 2 | 4 | 4 | 4 |  |  |  |  |
| A Q 10 | 3 | 4 | 4 | 4 |  |  |  |
| A Q 2 | 3 | 4 | 4 | 4 |  |  |  |
| A Q | 3 | 3 | 4 | 4 | 4 |  |  |
| A J 10 2 | 4 | 4 | 4 |  |  |  |  |
| A J 10 | 4 | 4 | 4 | 4 |  |  |  |
| A J 2 | 3 | 4 | 4 | 4 |  |  |  |
| A J | 3 | 4 | 4 | 4 | 4 |  |  |
| A 10 2 | 3 | 3 | 3 | 4 |  |  |  |
| A 10 | 3 | 3 | 3 | 3 | 4 |  |  |
| A 2 | 2 | 3 | 3 | 3 | 4 |  |  |
| A | 2 | 2 | 3 | 3 | 3 | 4 |  |
| K Q J 10 2 | 4 | 4 |  |  |  |  |  |
| K Q J 10 | 4 | 4 | 4 |  |  |  |  |
| K Q J 2 | 4 | 4 | 4 |  |  |  |  |
| K Q J | 3 | 3 | 4 | 4 |  |  |  |
| K Q 10 2 | 3 | 3 | 3 |  |  |  |  |
| K Q 10 | 3 | 3 | 3 | 3 |  |  |  |
| K Q 2 | 2 | 3 | 3 | 3 |  |  |  |
| K Q | 2 | 2 | 3 | 3 | 3 |  |  |
| K J 10 2 | 3 | 4 | 4 |  |  |  |  |
| K J 10 | 3 | 3 | 3 | 4 |  |  |  |
| K J 2 | 3 | 3 | 4 | 4 |  |  |  |
| K J | 2 | 3 | 3 | 3 | 4 |  |  |
| K 10 2 | 2 | 3 | 3 | 3 |  |  |  |
| K 10 | pass | 2 | 2 | 3 | 3 |  |  |
| K 2 | pass | 2 | 3 | 3 | 3 |  |  |
| K | pass | pass | 2 | 2 | 2 | 3 |  |
| Q J 10 2 | 3 | 3 | 3 |  |  |  |  |
| Q J 10 | 2 | 3 | 3 | 3 |  |  |  |
| Q J 2 | 2 | 3 | 3 | 3 |  |  |  |
| Q J | pass | 2 | 3 | 3 | 3 |  |  |
| Q 10 2 | 2 | 2 | 2 | 3 |  |  |  |
| Q 10 | pass | pass | 2 | 2 | 2 |  |  |
| Q 2 | pass | pass | 2 | 2 | 3 |  |  |
| Q | pass | pass | pass | 2 | 2 | 2 |  |
| J 10 2 | 2 | 3 | 3 | 3 |  |  |  |
| J 10 | pass | 2 | 2 | 3 | 3 |  |  |
| J 2 | pass | 2 | 3 | 3 | 3 |  |  |
| J | pass | pass | 2 | 2 | 3 | 3 |  |
| 10 2 | pass | pass | 2 | 2 | 2 |  |  |
| 10 |  | pass | pass | pass | 2 | 2 |  |
| 2 |  | pass | pass | 2 | 2 | 2 |  |
| (none) |  |  | pass | pass | pass | pass | 2 |
