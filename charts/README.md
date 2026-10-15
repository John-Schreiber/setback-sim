# Bidding charts by house rules

Each chart gives the highest bid worth making (expected points >= 0) by the honors you hold in your trump suit and how many other small trumps you have. Built by `python -m cardsim.pitch.bidchart` from large GPU surveys with search players; the `.csv` next to each chart has the expected points of every bid for every group, and the net expected points (score change minus the opponents' points on the hand).

| Rules | 4-handed | 6-handed | What changes |
|---|---|---|---|
| **setback4** | [chart](setback4_4p.md) | [chart](setback4_6p.md) | 6 cards; may trump in; High 1, Low 1, Jack 1, Game 1 (4 total); bids 2-4; game to 11 |
| **setback4_lowdealt** | [chart](setback4_lowdealt_4p.md) | [chart](setback4_lowdealt_6p.md) | 6 cards; may trump in; High 1, Low 1 (to whoever was dealt it), Jack 1, Game 1 (4 total); bids 2-4; game to 11 |
| **setback4_must_follow** | [chart](setback4_must_follow_4p.md) | [chart](setback4_must_follow_6p.md) | 6 cards; must follow suit; High 1, Low 1, Jack 1, Game 1 (4 total); bids 2-4; game to 11 |
| **six_point** | [chart](six_point_4p.md) | [chart](six_point_6p.md) | 6 cards; may trump in; High 1, Low 1, Jack 1, off-jack 1, 1 joker 1 each, Game 1 (6 total); bids 2-6; game to 11 |
| **seven_point** | [chart](seven_point_4p.md) | [chart](seven_point_6p.md) | 6 cards; may trump in; High 1, Low 1, Jack 1, off-jack 1, 2 jokers 1 each, Game 1 (7 total); bids 3-7; game to 21 |
| **ten_point** | [chart](ten_point_4p.md) | [chart](ten_point_6p.md) | 6 cards; may trump in; High 1, Low 1, Jack 1, off-jack 1, 2 jokers 1 each, 3 of trump 3, Game 1 (10 total); bids 3-10; game to 52 |
| **roadhouse** | [chart](roadhouse_4p.md) | [chart](roadhouse_6p.md) | 6 cards; must follow suit; High 1, Low 1, Jack 1, Game 1 (4 total); bids 2-4, moon 5 = 15; game to 15 |
| **draw_pitch** | [chart](draw_pitch_4p.md) | [chart](draw_pitch_6p.md) | 6 cards, discard non-trumps and draw back to 6; may trump in; High 1, Low 1, Jack 1, Game 1 (4 total); bids 2-4, moon 5 = 15; game to 15 |
| **nine_five** | [chart](nine_five_4p.md) | - | 9 cards, discard non-trumps and draw back to 6; may trump in; High 1, Low 1, Jack 1, 5 of trump 5, 9 of trump 9, Game 1 (18 total); bids 9-18, moon 19 = 19; game to 100 |
