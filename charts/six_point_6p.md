## six_point, 6-handed

Highest bid worth making by the line that wins the most whole games: net expected points (score change minus the opponents' points on the hand) >= -2.0625 (cardsim/pitch/passline.py; the .csv's max_bid is the older line, expected points >= 0) (bids 2-6; 'pass' = none), by trump honors held (rows) and number of other small trumps (columns). From 2,000,000 deals with search players; blank = fewer than 200 hands.

| trump honors | +0 | +1 | +2 | +3 | +4 | +5 |
|---|---|---|---|---|---|---|
| A K Q J | 5 |  |  |  |  |  |
| A K Q offJ | 5 |  |  |  |  |  |
| A K Q BJ | 5 |  |  |  |  |  |
| A K Q 10 | 5 |  |  |  |  |  |
| A K Q 2 | 5 |  |  |  |  |  |
| A K Q | 5 | 5 | 5 |  |  |  |
| A K J offJ | 6 |  |  |  |  |  |
| A K J BJ | 6 |  |  |  |  |  |
| A K J 10 | 5 |  |  |  |  |  |
| A K J 2 | 5 |  |  |  |  |  |
| A K J | 5 | 5 | 5 |  |  |  |
| A K offJ BJ | 6 |  |  |  |  |  |
| A K offJ 10 | 5 |  |  |  |  |  |
| A K offJ 2 | 5 |  |  |  |  |  |
| A K offJ | 5 | 5 | 5 |  |  |  |
| A K BJ 10 | 5 |  |  |  |  |  |
| A K BJ 2 | 5 |  |  |  |  |  |
| A K BJ | 5 | 5 | 5 |  |  |  |
| A K 10 2 | 5 |  |  |  |  |  |
| A K 10 | 4 | 5 | 5 |  |  |  |
| A K 2 | 4 | 5 | 5 |  |  |  |
| A K | 4 | 4 | 5 | 5 |  |  |
| A Q J offJ | 5 |  |  |  |  |  |
| A Q J BJ | 5 |  |  |  |  |  |
| A Q J 10 | 5 |  |  |  |  |  |
| A Q J 2 | 5 |  |  |  |  |  |
| A Q J | 5 | 5 | 5 |  |  |  |
| A Q offJ BJ | 5 |  |  |  |  |  |
| A Q offJ 10 | 5 |  |  |  |  |  |
| A Q offJ 2 | 5 |  |  |  |  |  |
| A Q offJ | 5 | 5 | 5 |  |  |  |
| A Q BJ 10 | 5 |  |  |  |  |  |
| A Q BJ 2 | 5 |  |  |  |  |  |
| A Q BJ | 4 | 5 | 5 |  |  |  |
| A Q 10 2 | 4 |  |  |  |  |  |
| A Q 10 | 4 | 4 | 5 |  |  |  |
| A Q 2 | 4 | 4 | 4 |  |  |  |
| A Q | 4 | 4 | 4 | 5 |  |  |
| A J offJ BJ | 5 |  |  |  |  |  |
| A J offJ 10 | 5 |  |  |  |  |  |
| A J offJ 2 | 5 |  |  |  |  |  |
| A J offJ | 5 | 5 | 5 |  |  |  |
| A J BJ 10 | 5 |  |  |  |  |  |
| A J BJ 2 | 5 |  |  |  |  |  |
| A J BJ | 4 | 5 | 5 |  |  |  |
| A J 10 2 | 4 |  |  |  |  |  |
| A J 10 | 4 | 4 | 5 |  |  |  |
| A J 2 | 4 | 4 | 5 |  |  |  |
| A J | 4 | 4 | 4 | 5 |  |  |
| A offJ BJ 10 | 5 |  |  |  |  |  |
| A offJ BJ 2 | 4 |  |  |  |  |  |
| A offJ BJ | 4 | 5 | 5 |  |  |  |
| A offJ 10 2 | 4 |  |  |  |  |  |
| A offJ 10 | 4 | 4 | 5 |  |  |  |
| A offJ 2 | 4 | 4 | 5 |  |  |  |
| A offJ | 3 | 4 | 4 | 5 |  |  |
| A BJ 10 2 | 4 |  |  |  |  |  |
| A BJ 10 | 4 | 4 | 5 |  |  |  |
| A BJ 2 | 3 | 4 | 5 |  |  |  |
| A BJ | 3 | 4 | 4 | 5 |  |  |
| A 10 2 | 3 | 4 | 4 |  |  |  |
| A 10 | 3 | 3 | 4 | 4 |  |  |
| A 2 | 2 | 3 | 4 | 4 |  |  |
| A | 3 | 3 | 3 | 4 | 4 |  |
| K Q J offJ | 5 |  |  |  |  |  |
| K Q J BJ | 5 |  |  |  |  |  |
| K Q J 10 | 4 |  |  |  |  |  |
| K Q J 2 | 4 |  |  |  |  |  |
| K Q J | 4 | 4 | 4 |  |  |  |
| K Q offJ BJ | 5 |  |  |  |  |  |
| K Q offJ 10 | 4 |  |  |  |  |  |
| K Q offJ 2 | 4 |  |  |  |  |  |
| K Q offJ | 4 | 4 | 4 |  |  |  |
| K Q BJ 10 | 4 |  |  |  |  |  |
| K Q BJ 2 | 4 |  |  |  |  |  |
| K Q BJ | 3 | 4 | 4 |  |  |  |
| K Q 10 2 | 3 |  |  |  |  |  |
| K Q 10 | 3 | 3 | 3 |  |  |  |
| K Q 2 | 3 | 3 | 3 |  |  |  |
| K Q | 2 | 2 | 3 | 3 |  |  |
| K J offJ BJ | 5 |  |  |  |  |  |
| K J offJ 10 | 4 |  |  |  |  |  |
| K J offJ 2 | 4 |  |  |  |  |  |
| K J offJ | 4 | 4 | 4 |  |  |  |
| K J BJ 10 | 4 |  |  |  |  |  |
| K J BJ 2 | 4 |  |  |  |  |  |
| K J BJ | 3 | 4 | 4 |  |  |  |
| K J 10 2 | 3 |  |  |  |  |  |
| K J 10 | 3 | 3 | 4 |  |  |  |
| K J 2 | 3 | 4 | 4 |  |  |  |
| K J | 2 | 3 | 3 | 4 |  |  |
| K offJ BJ 10 | 4 |  |  |  |  |  |
| K offJ BJ 2 | 4 |  |  |  |  |  |
| K offJ BJ | 3 | 4 | 4 |  |  |  |
| K offJ 10 2 | 4 |  |  |  |  |  |
| K offJ 10 | 3 | 3 | 4 |  |  |  |
| K offJ 2 | 3 | 3 | 4 |  |  |  |
| K offJ | 2 | 3 | 3 | 3 |  |  |
| K BJ 10 2 | 3 |  |  |  |  |  |
| K BJ 10 | 2 | 3 | 4 |  |  |  |
| K BJ 2 | 2 | 3 | 4 |  |  |  |
| K BJ | 2 | 2 | 3 | 3 |  |  |
| K 10 2 | 2 | 3 | 3 |  |  |  |
| K 10 | pass | 2 | 2 | 2 |  |  |
| K 2 | pass | 2 | 3 | 3 |  |  |
| K | pass | pass | 2 | 2 | 2 |  |
| Q J offJ BJ | 4 |  |  |  |  |  |
| Q J offJ 10 | 4 |  |  |  |  |  |
| Q J offJ 2 | 4 |  |  |  |  |  |
| Q J offJ | 3 | 4 | 4 |  |  |  |
| Q J BJ 2 | 4 |  |  |  |  |  |
| Q J BJ | 3 | 4 | 4 |  |  |  |
| Q J 10 2 | 3 |  |  |  |  |  |
| Q J 10 | 3 | 3 | 3 |  |  |  |
| Q J 2 | 2 | 3 | 3 |  |  |  |
| Q J | 2 | 2 | 3 | 3 |  |  |
| Q offJ BJ 2 | 3 |  |  |  |  |  |
| Q offJ BJ | 3 | 3 | 4 |  |  |  |
| Q offJ 10 2 | 3 |  |  |  |  |  |
| Q offJ 10 | 2 | 3 | 3 |  |  |  |
| Q offJ 2 | 2 | 3 | 4 |  |  |  |
| Q offJ | pass | 2 | 2 | 3 |  |  |
| Q BJ 10 | 2 | 2 | 3 |  |  |  |
| Q BJ 2 | 2 | 3 | 3 |  |  |  |
| Q BJ | pass | 2 | 2 | 3 |  |  |
| Q 10 2 | pass | 2 | 2 |  |  |  |
| Q 10 | pass | pass | 2 | 2 |  |  |
| Q 2 | pass | pass | 2 | 3 |  |  |
| Q | pass | pass | pass | pass | 2 |  |
| J offJ BJ | 3 | 4 | 4 |  |  |  |
| J offJ 10 2 | 3 |  |  |  |  |  |
| J offJ 10 | 2 | 3 | 3 |  |  |  |
| J offJ 2 | 2 | 3 | 4 |  |  |  |
| J offJ | pass | 2 | 3 | 3 |  |  |
| J BJ 10 | 2 | 3 | 3 |  |  |  |
| J BJ 2 | 2 | 3 | 4 |  |  |  |
| J BJ | pass | 2 | 3 | 3 |  |  |
| J 10 2 | 2 | 3 | 3 |  |  |  |
| J 10 | pass | 2 | 2 | 3 |  |  |
| J 2 | pass | 2 | 3 | 3 |  |  |
| J | pass | pass | 2 | 2 | 3 |  |
| offJ BJ 10 | 2 | 3 | 3 |  |  |  |
| offJ BJ 2 | 2 | 3 | 4 |  |  |  |
| offJ BJ |  | 2 | 3 | 3 |  |  |
| offJ 10 2 | pass | 2 | 3 |  |  |  |
| offJ 10 | pass | pass | 2 | 2 |  |  |
| offJ 2 | pass | pass | 2 | 3 |  |  |
| offJ |  | pass | pass | 2 | 2 |  |
| BJ 10 2 | pass | 2 | 3 |  |  |  |
| BJ 10 |  | pass | 2 | 2 |  |  |
| BJ 2 |  | pass | 2 | 3 |  |  |
| BJ |  |  | pass | 2 | 2 |  |
| 10 2 | pass | pass | 2 | 2 |  |  |
| 10 |  | pass | pass | pass | pass |  |
| 2 |  | pass | pass | pass | 2 |  |
| (none) |  |  | pass | pass | pass | pass |
