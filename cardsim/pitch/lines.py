"""The bidding line per variant (see passline.py for how it was found). No dependencies, so the
site build (pandas and pyarrow only) can read it."""

# The stable bidding line per (preset, players): bid up to the highest bid whose net expected
# points (score change minus the opponents' points on the hand, from the charts) are at least
# this. Found 2026-10-16 with rule-based players, 200,000 games per line: the best line against
# a field on today's line (ev >= 0), then iterated until team A's best response, with every
# other team on the line, stays on it (within one grid step / noise). Against a field on these
# lines, today's line wins 0.0-2.4 points less in the setback family and 7-23 points less in
# ten-point, draw pitch and nine-five. Every line was then checked with look-ahead players
# (gamecache, 16 worlds, 200,000 games): stable for all but two, which moved to the look-ahead
# fixed point (draw_pitch 6-handed, nine_five). With look-ahead players today's line wins 41%
# (ten-point), 42% (draw pitch) and 38% (nine-five) of 4-handed games against these lines.
#
# Why the threshold is below zero: passing isn't worth zero. Measured directly (passvalue.py:
# at every auction decision in 200,000 look-ahead games of setback4 4-handed, both choices
# scored on the same deal), passing costs about 0.2-0.3 net points per hand, because an
# opponent then takes the contract and gains on you. The grounded rule is "bid when bidding
# beats passing" (bid - pass >= 0); it agrees with the line on 97% of decisions and plays
# even with it in whole games (49.7% +/- 0.1%). The line is that rule in the charts' units
# (net measured against zero rather than against passing).
LINES = {
    ("setback4", 4): -0.625, ("setback4", 6): -1.75,
    ("setback4_lowdealt", 4): -0.625, ("setback4_lowdealt", 6): -1.625,
    ("setback4_must_follow", 4): -0.75, ("setback4_must_follow", 6): -1.75,
    ("six_point", 4): -0.75, ("six_point", 6): -2.0625,
    ("seven_point", 4): -0.4375, ("seven_point", 6): -2.1875,
    ("ten_point", 4): -0.625, ("ten_point", 6): -2.5,
    ("roadhouse", 4): -0.75, ("roadhouse", 6): -1.375,
    ("draw_pitch", 4): -0.25, ("draw_pitch", 6): -0.875,  # 6p: look-ahead; rule-based -0.625
    ("nine_five", 4): -1.6875,  # look-ahead players; rule-based play settles at -3.375
}
