## draw_pitch, 4-handed

Highest bid worth making by the line that wins the most whole games: net expected points (score change minus the opponents' points on the hand) >= -0.25 (cardsim/pitch/passline.py; the .csv's max_bid is the older line, expected points >= 0) (bids 2-4; 'pass' = none), by trump honors held (rows) and number of other small trumps (columns). From 4,000,000 deals with search players; blank = fewer than 200 hands. A moon bid (5, worth 15) is marked * where it is also worth bidding.

| trump honors | +0 | +1 | +2 | +3 | +4 | +5 | +6 |
|---|---|---|---|---|---|---|---|
| A K Q J | 4 | 4 |  |  |  |  |  |
| A K Q 10 | 4 | 4 |  |  |  |  |  |
| A K Q 2 | 4 | 4 |  |  |  |  |  |
| A K Q | 3 | 4 | 4 |  |  |  |  |
| A K J 10 | 4 | 4 |  |  |  |  |  |
| A K J 2 | 4 | 4 |  |  |  |  |  |
| A K J | 4 | 4 | 4 |  |  |  |  |
| A K 10 2 | 3 | 3 |  |  |  |  |  |
| A K 10 | 3 | 3 | 3 |  |  |  |  |
| A K 2 | 3 | 3 | 3 |  |  |  |  |
| A K | 3 | 3 | 3 | 3 |  |  |  |
| A Q J 10 | 4 | 4 |  |  |  |  |  |
| A Q J 2 | 4 | 4 |  |  |  |  |  |
| A Q J | 4 | 4 | 4 |  |  |  |  |
| A Q 10 2 | 3 | 3 |  |  |  |  |  |
| A Q 10 | 3 | 3 | 3 |  |  |  |  |
| A Q 2 | 3 | 3 | 3 |  |  |  |  |
| A Q | 3 | 3 | 3 | 3 |  |  |  |
| A J 10 2 | 4 |  |  |  |  |  |  |
| A J 10 | 3 | 4 | 4 |  |  |  |  |
| A J 2 | 3 | 4 | 4 |  |  |  |  |
| A J | 3 | 3 | 4 | 4 |  |  |  |
| A 10 2 | 3 | 3 | 3 |  |  |  |  |
| A 10 | 2 | 3 | 3 | 3 |  |  |  |
| A 2 | 2 | 3 | 3 | 3 |  |  |  |
| A | 2 | 2 | 3 | 3 | 3 |  |  |
| K Q J 10 | 3 | 4 |  |  |  |  |  |
| K Q J 2 | 3 | 3 |  |  |  |  |  |
| K Q J | 3 | 3 | 3 |  |  |  |  |
| K Q 10 2 | 2 | 3 |  |  |  |  |  |
| K Q 10 | 2 | 2 | 2 |  |  |  |  |
| K Q 2 | 2 | 2 | 3 |  |  |  |  |
| K Q | 2 | 2 | 2 | 2 |  |  |  |
| K J 10 2 | 3 | 3 |  |  |  |  |  |
| K J 10 | 3 | 3 | 3 |  |  |  |  |
| K J 2 | 2 | 3 | 3 |  |  |  |  |
| K J | 2 | 2 | 3 | 3 |  |  |  |
| K 10 2 | 2 | 2 | 3 |  |  |  |  |
| K 10 | pass | 2 | 2 | 2 |  |  |  |
| K 2 | pass | 2 | 2 | 2 |  |  |  |
| K | pass | pass | pass | 2 | 2 |  |  |
| Q J 10 2 | 3 |  |  |  |  |  |  |
| Q J 10 | 2 | 3 | 3 |  |  |  |  |
| Q J 2 | 2 | 3 | 3 |  |  |  |  |
| Q J | 2 | 2 | 2 | 2 |  |  |  |
| Q 10 2 | pass | 2 | 2 |  |  |  |  |
| Q 10 | pass | pass | pass | 2 |  |  |  |
| Q 2 | pass | pass | 2 | 2 |  |  |  |
| Q | pass | pass | pass | pass | pass |  |  |
| J 10 2 | 2 | 2 | 3 |  |  |  |  |
| J 10 | pass | 2 | 2 | 3 |  |  |  |
| J 2 | pass | 2 | 2 | 3 |  |  |  |
| J | pass | pass | 2 | 2 | 2 |  |  |
| 10 2 | pass | pass | 2 | 2 |  |  |  |
| 10 | pass | pass | pass | pass | pass |  |  |
| 2 | pass | pass | pass | pass | pass |  |  |
| (none) |  |  | pass | pass | pass | pass |  |
