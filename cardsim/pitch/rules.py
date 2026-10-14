"""Pitch / Setback house rules as data, with the named variants as presets.

`PitchRules.preset("setback4")` is the game the original engine plays (and must
reproduce card for card). Every field below is one house-rule switch.
"""
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class PitchRules:
    name: str = "setback4"

    # ---- deck and trumps
    jokers: int = 0              # 0, 1 (big joker only) or 2; jokers are trumps ranking below the Jack
    off_jack: bool = False       # the same-colour Jack is a trump, ranking just below the Jack

    # ---- deal
    hand_size: int = 6           # cards held during play
    deal_size: int = 6           # cards dealt; more than hand_size means discard down after trump is named
    refill: bool = False         # after trump is named, everyone discards non-trumps and draws back up

    # ---- play
    may_trump: bool = True       # may trump instead of following suit (False = must follow if able)
    first_lead: str = "trump"    # the pitch: "trump", "any", or "not_trump" (Barn pitch)
    trump_broken: bool = False   # trump may not be led until someone has trumped in

    # ---- points per hand (value of each; 0 = not scored)
    high: int = 1                # highest trump in play
    low: int = 1                 # lowest trump in play
    low_to_captor: bool = True   # Low scores for whoever captures it (False: whoever was dealt it)
    jack: int = 1                # Jack of trump (to whoever captures it)
    off_jack_pts: int = 0        # off-jack
    joker_pts: int = 0           # each joker
    ten_pts: int = 0             # 10 of trump
    three_pts: int = 0           # 3 of trump (ten-point pitch: 3)
    five_pts: int = 0            # 5 of trump (Nine-Five: 5)
    nine_pts: int = 0            # 9 of trump (Nine-Five: 9)
    game: int = 1                # most counting cards (10=10, A=4, K=3, Q=2, J=1); a tie scores nobody
    last_trick: int = 0          # last trick

    # ---- bidding
    min_bid: int = 2
    max_bid: int = 4             # highest ordinary bid
    moon_bid: int = 0            # extra bid needing every point and every trick (0 = none), e.g. 11
    moon_value: int = 11         # what a made moon bid scores (set: loses the same)
    moon_wins_game: bool = False
    dealer_stuck: bool = True    # everyone passes: dealer must take the minimum bid (False: redeal)
    dealer_takes: tuple = ()     # bid levels the dealer may take at the current high, e.g. (4,)

    # ---- scoring
    target: int = 11
    bidder_out_first: bool = True
    win_by: int = 1
    no_junk: bool = False        # only the bidding side scores
    exact_bonus: int = 0         # Maryland Jack: +1 for making exactly the bid
    backdoor: int = 0            # e.g. -11: a side at or below this loses (0 = off)

    @property
    def points_total(self):
        """Points available in a hand (if every point card is in play)."""
        return (self.high + self.low + self.jack + self.off_jack_pts * self.off_jack
                + self.joker_pts * self.jokers + self.ten_pts + self.three_pts + self.five_pts
                + self.nine_pts + self.game + self.last_trick)

    def with_(self, **kw):
        return replace(self, **kw)

    @staticmethod
    def preset(name):
        return replace(PRESETS[name])


PRESETS = {
    # the original engine's game
    "setback4": PitchRules(),
    "setback4_lowdealt": PitchRules(name="setback4_lowdealt", low_to_captor=False),
    "setback4_must_follow": PitchRules(name="setback4_must_follow", may_trump=False),
    # Contway 6-point: one joker and the off-jack; high, low, jack, off-jack, joker, game
    "six_point": PitchRules(name="six_point", jokers=1, off_jack=True, off_jack_pts=1, joker_pts=1,
                            max_bid=6),
    # Cell pitch (Kentucky seven-point): both jokers and the off-jack; bids 3-7, game to 21
    "seven_point": PitchRules(name="seven_point", jokers=2, off_jack=True, off_jack_pts=1, joker_pts=1,
                              min_bid=3, max_bid=7, target=21),
    # Ten-point pitch: high, low, jack, off-jack, two jokers, game, and the 3 of trump worth 3
    "ten_point": PitchRules(name="ten_point", jokers=2, off_jack=True, off_jack_pts=1, joker_pts=1,
                            three_pts=3, min_bid=3, max_bid=10, target=52),
    # Roadhouse: 4 points, must follow suit, moon worth 15, game to 15 win by 2
    "roadhouse": PitchRules(name="roadhouse", may_trump=False, moon_bid=5, moon_value=15,
                            target=15, win_by=2),
    # Partnership draw pitch (New Yorker): discard non-trumps and draw, bids 2-4, moon, to 15
    "draw_pitch": PitchRules(name="draw_pitch", refill=True, moon_bid=5, moon_value=15, target=15),
    # Nine-Five setback: 9 cards, discard to 6; high, low, jack, game + 9 (9) and 5 (5) of trump = 18
    "nine_five": PitchRules(name="nine_five", deal_size=9, refill=True, nine_pts=9, five_pts=5,
                            min_bid=9, max_bid=18, moon_bid=19, moon_value=19, target=100,
                            bidder_out_first=True, backdoor=-100),
}
