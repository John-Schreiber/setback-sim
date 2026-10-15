"""Plain-language explanations of the house-rule presets (rules.py), for the charts'
README and the stats site. Kept out of rules.py so editing them doesn't change the
code version the cluster checks.

COMMON covers what every preset shares; NOTES says what each one is and how it
differs. Keep them in step with PRESETS.
"""

COMMON = (
    "In every variant the high bidder names trump by leading it, partners sit across from each other "
    "(4-handed: two teams of two; 6-handed: three teams of two), and each hand is scored the same "
    "way: if the bidding team takes at least its bid it scores every point it took, and if not it "
    "loses the bid (is set). The other teams always score the points they took, set or not. If "
    "everyone passes, the dealer is stuck with the lowest bid; if two teams reach the target on the "
    "same hand, the bidder goes out first. Unless noted, you may trump in even when you could follow "
    "suit. "
    "High is the top trump played, Low the lowest, and Game goes to the team with the most counting "
    "cards in its tricks (each 10 counts 10, Ace 4, King 3, Queen 2, Jack 1); if two teams tie for "
    "Game, nobody scores it. With jokers or the off-jack, trumps rank A K Q J, off-jack, big joker, "
    "little joker, 10 down to 2."
)

NOTES = {
    "setback4": (
        "Four-point Setback, the classic game. Six cards each from a 52-card deck; four points a hand: "
        "High, Low (to whoever captures it), Jack of trump and Game. Bids 2 to 4; game to 11."),
    "setback4_lowdealt": (
        "Four-point Setback, but Low scores for the team that was dealt it, not the team that captures "
        "it. Holding the lowest trump is a sure point, and nobody can steal it."),
    "setback4_must_follow": (
        "Four-point Setback where you must follow suit if you can, and may trump only when you are "
        "void in the suit led."),
    "six_point": (
        "Six-point Pitch with one joker and the off-jack (the other Jack of trump's colour), both "
        "trumps worth a point each: High, Low, Jack, off-jack, joker and Game. Six cards each from 53 "
        "cards; bids 2 to 6; game to 11."),
    "seven_point": (
        "Seven-point Pitch with both jokers and the off-jack, each worth a point: High, Low, Jack, "
        "off-jack, big joker, little joker and Game. Six cards each from 54 cards; bids 3 to 7; game "
        "to 21."),
    "ten_point": (
        "Ten-point Pitch: the seven-point deck and points, plus the 3 of trump worth 3 points (ten in "
        "all). Bids 3 to 10; game to 52."),
    "roadhouse": (
        "Four-point Setback where you must follow suit if you can (trump only when void), with a moon "
        "bid: bid 5 to take all four points and every trick, scoring 15 if you do and losing 15 if "
        "you don't. Game to 15, and you must win by 2."),
    "draw_pitch": (
        "Partnership draw Pitch: after trump is named, everyone throws away their non-trumps and "
        "draws back to six from the rest of the deck, starting left of the dealer, so hands are "
        "mostly trump. Four points, bids 2 to 4, a moon bid of 5 worth 15 (all points and every "
        "trick); game to 15. The chart is for the six cards you are dealt, before the draw."),
    "nine_five": (
        "Nine-Five Setback: nine cards dealt; after trump is named everyone discards their non-trumps "
        "and draws back to six (keeping point cards, then the highest trumps, if they hold more than "
        "six). Points: High, Low, Jack and Game, plus the 9 of trump worth 9 and the 5 of trump worth "
        "5, for 18 in all. Bids 9 to 18, or 19 to take every point and every trick (worth 19). Game "
        "to 100; a team that falls to -100 loses. The chart is for the nine cards you are dealt."),
}
