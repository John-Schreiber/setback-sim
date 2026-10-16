## seven_point, 6-handed

Highest bid worth making by the line that wins the most whole games: net expected points (score change minus the opponents' points on the hand) >= -2.1875 (cardsim/pitch/passline.py; the .csv's max_bid is the older line, expected points >= 0) (bids 3-7; 'pass' = none), by trump honors held (rows) and number of other small trumps (columns). From 2,000,000 deals with search players; (n) = too few hands seen (200), so taken from the weaker holdings next to it.

| trump honors | +0 | +1 | +2 | +3 | +4 | +5 | +6 |
|---|---|---|---|---|---|---|---|
| A K Q J | 6 | (6) | (6) |  |  |  |  |
| A K Q offJ | 6 | (6) | (6) |  |  |  |  |
| A K Q BJ | 6 | (6) | (6) |  |  |  |  |
| A K Q LJ | 6 | (6) | (6) |  |  |  |  |
| A K Q 2 | 6 | (6) | (6) |  |  |  |  |
| A K Q | 5 | 5 | 6 | (6) |  |  |  |
| A K J offJ | 6 | (6) | (6) |  |  |  |  |
| A K J BJ | 6 | (6) | (6) |  |  |  |  |
| A K J LJ | 6 | (6) | (6) |  |  |  |  |
| A K J 10 | 6 | (6) | (6) |  |  |  |  |
| A K J 2 | 6 | (6) | (6) |  |  |  |  |
| A K J | 5 | 6 | 6 | (6) |  |  |  |
| A K offJ LJ | 6 | (6) | (6) |  |  |  |  |
| A K offJ 10 | 6 | (6) | (6) |  |  |  |  |
| A K offJ 2 | 5 | (6) | (6) |  |  |  |  |
| A K offJ | 5 | 6 | 6 | (6) |  |  |  |
| A K BJ LJ | 6 | (6) | (6) |  |  |  |  |
| A K BJ 10 | 6 | (6) | (6) |  |  |  |  |
| A K BJ 2 | 5 | (5) | (6) |  |  |  |  |
| A K BJ | 5 | 5 | 6 | (6) |  |  |  |
| A K LJ 10 | 6 | (6) | (6) |  |  |  |  |
| A K LJ 2 | 5 | (5) | (6) |  |  |  |  |
| A K LJ | 5 | 5 | 6 | (6) |  |  |  |
| A K 10 2 | 5 | (5) | (5) |  |  |  |  |
| A K 10 | 5 | 5 | 5 | (5) |  |  |  |
| A K 2 | 5 | 5 | 5 | (5) |  |  |  |
| A K | 4 | 5 | 5 | 5 | (5) |  |  |
| A Q J offJ | 6 | (6) | (6) |  |  |  |  |
| A Q J BJ | 6 | (6) | (6) |  |  |  |  |
| A Q J LJ | 6 | (6) | (6) |  |  |  |  |
| A Q J 10 | 6 | (6) | (6) |  |  |  |  |
| A Q J | 5 | 5 | 6 | (6) |  |  |  |
| A Q offJ BJ | 6 | (6) | (6) |  |  |  |  |
| A Q offJ LJ | 6 | (6) | (6) |  |  |  |  |
| A Q offJ 10 | 5 | (5) | (5) |  |  |  |  |
| A Q offJ 2 | 5 | (5) | (5) |  |  |  |  |
| A Q offJ | 5 | 5 | 5 | (5) |  |  |  |
| A Q BJ LJ | 6 | (6) | (6) |  |  |  |  |
| A Q BJ 10 | 5 | (5) | (5) |  |  |  |  |
| A Q BJ 2 | 5 | (5) | (5) |  |  |  |  |
| A Q BJ | 5 | 5 | 5 | (5) |  |  |  |
| A Q LJ 10 | 5 | (5) | (5) |  |  |  |  |
| A Q LJ | 5 | 5 | 5 | (5) |  |  |  |
| A Q 10 2 | 4 | (5) | (5) |  |  |  |  |
| A Q 10 | 4 | 5 | 5 | (5) |  |  |  |
| A Q 2 | 4 | 4 | 5 | (5) |  |  |  |
| A Q | 4 | 4 | 5 | 5 | (5) |  |  |
| A J offJ BJ | 6 | (6) | (6) |  |  |  |  |
| A J offJ LJ | 6 | (6) | (6) |  |  |  |  |
| A J offJ 10 | 5 | (5) | (6) |  |  |  |  |
| A J offJ 2 | 5 | (5) | (6) |  |  |  |  |
| A J offJ | 5 | 5 | 6 | (6) |  |  |  |
| A J BJ 2 | 5 | (5) | (6) |  |  |  |  |
| A J BJ | 5 | 5 | 6 | (6) |  |  |  |
| A J LJ 10 | 5 | (5) | (5) |  |  |  |  |
| A J LJ 2 | 5 | (5) | (5) |  |  |  |  |
| A J LJ | 5 | 5 | 5 | (5) |  |  |  |
| A J 10 2 | 5 | (5) | (5) |  |  |  |  |
| A J 10 | 4 | 5 | 5 | (5) |  |  |  |
| A J 2 | 4 | 5 | 5 | (5) |  |  |  |
| A J | 4 | 4 | 5 | 5 | (5) |  |  |
| A offJ BJ 10 | 5 | (5) | (5) |  |  |  |  |
| A offJ BJ 2 | 5 | (5) | (5) |  |  |  |  |
| A offJ BJ | 4 | 5 | 5 | (5) |  |  |  |
| A offJ LJ 2 | 4 | (5) | (5) |  |  |  |  |
| A offJ LJ | 4 | 5 | 5 | (5) |  |  |  |
| A offJ 10 2 | 4 | (5) | (5) |  |  |  |  |
| A offJ 10 | 4 | 5 | 5 | (5) |  |  |  |
| A offJ 2 | 4 | 5 | 5 | (5) |  |  |  |
| A offJ | 4 | 4 | 5 | 5 | (5) |  |  |
| A BJ LJ 10 | 5 | (5) | (5) |  |  |  |  |
| A BJ LJ 2 | 4 | (5) | (5) |  |  |  |  |
| A BJ LJ | 4 | 5 | 5 | (5) |  |  |  |
| A BJ 10 2 | 4 | (4) | (5) |  |  |  |  |
| A BJ 10 | 4 | 4 | 5 | (5) |  |  |  |
| A BJ 2 | 3 | 4 | 5 | (5) |  |  |  |
| A BJ | 3 | 4 | 4 | 5 | (5) |  |  |
| A LJ 10 2 | 4 | (4) | (5) |  |  |  |  |
| A LJ 10 | 4 | 4 | 5 | (5) |  |  |  |
| A LJ 2 | 3 | 4 | 5 | (5) |  |  |  |
| A LJ | 3 | 4 | 4 | 5 | (5) |  |  |
| A 10 2 | 3 | 4 | 4 | (4) |  |  |  |
| A 10 | 3 | 3 | 4 | 4 | (4) |  |  |
| A 2 | pass | 3 | 4 | 4 | (4) |  |  |
| A | 3 | 3 | 3 | 4 | 4 | (4) |  |
| K Q J offJ | 5 | (5) | (5) |  |  |  |  |
| K Q J LJ | 5 | (5) | (5) |  |  |  |  |
| K Q J 10 | 5 | (5) | (5) |  |  |  |  |
| K Q J 2 | 5 | (5) | (5) |  |  |  |  |
| K Q J | 4 | 4 | 4 | (4) |  |  |  |
| K Q offJ BJ | 5 | (5) | (5) |  |  |  |  |
| K Q offJ LJ | 5 | (5) | (5) |  |  |  |  |
| K Q offJ 2 | 5 | (5) | (5) |  |  |  |  |
| K Q offJ | 4 | 4 | 4 | (4) |  |  |  |
| K Q BJ LJ | 5 | (5) | (5) |  |  |  |  |
| K Q BJ 10 | 4 | (4) | (4) |  |  |  |  |
| K Q BJ 2 | 4 | (4) | (4) |  |  |  |  |
| K Q BJ | 4 | 4 | 4 | (4) |  |  |  |
| K Q LJ 2 | 4 | (4) | (4) |  |  |  |  |
| K Q LJ | 4 | 4 | 4 | (4) |  |  |  |
| K Q 10 2 | 3 | (3) | (4) |  |  |  |  |
| K Q 10 | 3 | 3 | 3 | (3) |  |  |  |
| K Q 2 | 3 | 3 | 4 | (4) |  |  |  |
| K Q | 3 | 3 | 3 | 3 | (3) |  |  |
| K J offJ LJ | 5 | (5) | (5) |  |  |  |  |
| K J offJ 2 | 4 | (4) | (5) |  |  |  |  |
| K J offJ | 4 | 4 | 5 | (5) |  |  |  |
| K J BJ | 4 | 4 | 4 | (4) |  |  |  |
| K J LJ 10 | 4 | (4) | (5) |  |  |  |  |
| K J LJ 2 | 4 | (4) | (5) |  |  |  |  |
| K J LJ | 4 | 4 | 5 | (5) |  |  |  |
| K J 10 2 | 4 | (4) | (4) |  |  |  |  |
| K J 10 | 3 | 4 | 4 | (4) |  |  |  |
| K J 2 | 3 | 4 | 4 | (4) |  |  |  |
| K J | 3 | 3 | 3 | 4 | (4) |  |  |
| K offJ BJ 2 | 4 | (4) | (5) |  |  |  |  |
| K offJ BJ | 3 | 4 | 5 | (5) |  |  |  |
| K offJ LJ 2 | 4 | (4) | (4) |  |  |  |  |
| K offJ LJ | 3 | 4 | 4 | (4) |  |  |  |
| K offJ 10 | 3 | 3 | 4 | (4) |  |  |  |
| K offJ 2 | 3 | 4 | 4 | (4) |  |  |  |
| K offJ | pass | 3 | 3 | 4 | (4) |  |  |
| K BJ LJ | 3 | 4 | 4 | (4) |  |  |  |
| K BJ 10 2 | 3 | (3) | (4) |  |  |  |  |
| K BJ 10 | 3 | 3 | 4 | (4) |  |  |  |
| K BJ 2 | 3 | 3 | 4 | (4) |  |  |  |
| K BJ | pass | pass | 3 | 3 | (3) |  |  |
| K LJ 10 | pass | 3 | 4 | (4) |  |  |  |
| K LJ 2 | pass | 3 | 4 | (4) |  |  |  |
| K LJ | pass | pass | 3 | 3 | (3) |  |  |
| K 10 2 | pass | 3 | 3 | (3) |  |  |  |
| K 10 | pass | pass | pass | 3 | (3) |  |  |
| K 2 | pass | pass | 3 | 3 | (3) |  |  |
| K | pass | pass | pass | pass | pass | (pass) |  |
| Q J offJ | 3 | 4 | 4 | (4) |  |  |  |
| Q J BJ | 3 | 4 | 4 | (4) |  |  |  |
| Q J LJ 10 | 3 | (4) | (4) |  |  |  |  |
| Q J LJ 2 | 4 | (4) | (4) |  |  |  |  |
| Q J LJ | 3 | 4 | 4 | (4) |  |  |  |
| Q J 10 2 | 3 | (3) | (4) |  |  |  |  |
| Q J 10 | 3 | 3 | 3 | (3) |  |  |  |
| Q J 2 | 3 | 3 | 4 | (4) |  |  |  |
| Q J | pass | pass | 3 | 3 | (3) |  |  |
| Q offJ BJ 10 | 3 | (3) | (4) |  |  |  |  |
| Q offJ BJ | 3 | 3 | 4 | (4) |  |  |  |
| Q offJ LJ | 3 | 3 | 4 | (4) |  |  |  |
| Q offJ 10 2 | 3 | (3) | (4) |  |  |  |  |
| Q offJ 10 | pass | 3 | 3 | (3) |  |  |  |
| Q offJ 2 | pass | 3 | 4 | (4) |  |  |  |
| Q offJ | pass | pass | 3 | 3 | (3) |  |  |
| Q BJ LJ | pass | 3 | 3 | (3) |  |  |  |
| Q BJ 10 | pass | pass | 3 | (3) |  |  |  |
| Q BJ 2 | pass | 3 | 3 | (3) |  |  |  |
| Q BJ | pass | pass | pass | 3 | (3) |  |  |
| Q LJ 10 | pass | 3 | 3 | (3) |  |  |  |
| Q LJ 2 | pass | 3 | 3 | (3) |  |  |  |
| Q LJ | pass | pass | pass | 3 | (3) |  |  |
| Q 10 2 | pass | pass | 3 | (3) |  |  |  |
| Q 10 | pass | pass | pass | pass | (pass) |  |  |
| Q 2 | pass | pass | pass | 3 | (3) |  |  |
| Q | pass | pass | pass | pass | pass | (pass) |  |
| J offJ BJ | 3 | 4 | 4 | (4) |  |  |  |
| J offJ LJ | 3 | 4 | 4 | (4) |  |  |  |
| J offJ 10 2 | 3 | (3) | (4) |  |  |  |  |
| J offJ 10 | 3 | 3 | 4 | (4) |  |  |  |
| J offJ 2 | pass | 3 | 4 | (4) |  |  |  |
| J offJ | pass | pass | 3 | 3 | (3) |  |  |
| J BJ LJ | pass | 3 | 4 | (4) |  |  |  |
| J BJ 10 | pass | 3 | 4 | (4) |  |  |  |
| J BJ 2 | pass | 3 | 4 | (4) |  |  |  |
| J BJ | pass | pass | 3 | 3 | (3) |  |  |
| J LJ 10 | pass | 3 | 4 | (4) |  |  |  |
| J LJ 2 | pass | 3 | 4 | (4) |  |  |  |
| J LJ | pass | pass | 3 | 3 | (3) |  |  |
| J 10 2 | pass | 3 | 3 | (3) |  |  |  |
| J 10 | pass | pass | pass | 3 | (3) |  |  |
| J 2 | pass | pass | 3 | 3 | (3) |  |  |
| J | pass | pass | pass | pass | pass | (pass) |  |
| offJ BJ LJ |  | 3 | 4 | (4) |  |  |  |
| offJ BJ 10 | pass | pass | 3 | (3) |  |  |  |
| offJ BJ 2 | pass | 3 | 4 | (4) |  |  |  |
| offJ BJ |  | pass | 3 | 3 | (3) |  |  |
| offJ LJ 10 | pass | 3 | 3 | (3) |  |  |  |
| offJ LJ 2 | pass | 3 | 4 | (4) |  |  |  |
| offJ LJ |  | pass | 3 | 3 | (3) |  |  |
| offJ 10 2 | pass | pass | 3 | (3) |  |  |  |
| offJ 10 | pass | pass | pass | pass | (pass) |  |  |
| offJ 2 | pass | pass | pass | 3 | (3) |  |  |
| offJ |  | pass | pass | pass | pass | (pass) |  |
| BJ LJ 10 |  | pass | 3 | (3) |  |  |  |
| BJ LJ 2 | pass | pass | 4 | (4) |  |  |  |
| BJ LJ |  | pass | pass | 3 | (3) |  |  |
| BJ 10 2 | pass | pass | 3 | (3) |  |  |  |
| BJ 10 |  | pass | pass | pass | (pass) |  |  |
| BJ 2 |  | pass | pass | 3 | (3) |  |  |
| BJ |  |  | pass | pass | pass | (pass) |  |
| LJ 10 2 | pass | pass | 3 | (3) |  |  |  |
| LJ 10 |  | pass | pass | pass | (pass) |  |  |
| LJ 2 |  | pass | pass | 3 | (3) |  |  |
| LJ |  |  | pass | pass | pass | (pass) |  |
| 10 2 | pass | pass | pass | pass | (pass) |  |  |
| 10 |  | pass | pass | pass | pass | (pass) |  |
| 2 |  | pass | pass | pass | pass | (pass) |  |
| (none) |  |  | pass | pass | pass | pass |  |
