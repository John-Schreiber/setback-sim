"""Build setback_bidding_charts.pdf from the condensed charts (condense.py).

    .venv/Scripts/python make_pdf.py
"""
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

import condense as cd

pdfmetrics.registerFont(TTFont("Arial", r"C:\Windows\Fonts\arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", r"C:\Windows\Fonts\arialbd.ttf"))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold")

INK = colors.HexColor("#1f2933")
MUTED = colors.HexColor("#52606d")
RULE = colors.HexColor("#cbd2d9")
BID_COLORS = {0: colors.HexColor("#f6d5d1"), 2: colors.HexColor("#fdf1c7"),
              3: colors.HexColor("#d7ebce"), 4: colors.HexColor("#9fd08a")}
BID_NAMES = {0: "pass", 2: "bid 2", 3: "bid 3", 4: "bid 4"}

base = ParagraphStyle("base", fontName="Arial", fontSize=10, leading=14, textColor=INK)
small = ParagraphStyle("small", parent=base, fontSize=8.5, leading=11.5, textColor=MUTED)
title = ParagraphStyle("title", parent=base, fontName="Arial-Bold", fontSize=22, leading=26,
                       spaceAfter=4)
subtitle = ParagraphStyle("subtitle", parent=base, fontSize=12, leading=16, textColor=MUTED,
                          spaceAfter=14)
h1 = ParagraphStyle("h1", parent=base, fontName="Arial-Bold", fontSize=15, leading=19,
                    spaceBefore=4, spaceAfter=6)
h2 = ParagraphStyle("h2", parent=base, fontName="Arial-Bold", fontSize=11.5, leading=15,
                    spaceBefore=10, spaceAfter=4)
bullet = ParagraphStyle("bullet", parent=base, leftIndent=14, bulletIndent=3, spaceAfter=2)
cell = ParagraphStyle("cell", parent=base, fontSize=11, leading=13, alignment=TA_CENTER)
cellhead = ParagraphStyle("cellhead", parent=base, fontName="Arial-Bold", fontSize=9.5,
                          leading=11.5, alignment=TA_CENTER)

ROWS = ["AKQJ", "AKQ", "AKJ", "AQJ", "KQJ", "AK", "AQ", "AJ", "KQ", "KJ", "QJ",
        "A", "K", "Q", "J", ""]
COLS = [("", "no 10 or 2"), ("T", "+ 10"), ("2", "+ 2"), ("T2", "+ 10 and 2")]


def red(s):
    """Hearts and diamonds in red."""
    return s.replace("♥", '<font color="#c0392b">♥</font>').replace("♦", '<font color="#c0392b">♦</font>')


def P(text, style=base):
    return Paragraph(red(text), style)


def bullets(items, style=bullet):
    return [Paragraph(red(t), style, bulletText="•") for t in items]


def need_table(p, d):
    g = cd.grouped(cd.load(p, d), ["hon", "nx"])
    need, nonmono = cd.thresholds(g)
    assert not nonmono
    return need


def chart(p, need):
    """The 16 x 4 grid; cell = small trumps needed for 2 · 3 · 4, coloured by the
    bid you can make with no small trumps."""
    f = lambda v: "–" if v is None else str(v)
    head = [P("<b>Your trump honors</b>", cellhead)] + [P(f"<b>{name}</b>", cellhead) for _, name in COLS]
    data = [head]
    style = [("GRID", (0, 0), (-1, -1), 0.6, colors.white),
             ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4e7eb")),
             ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#f5f7fa")),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 4.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 4.2)]
    for r, top in enumerate(ROWS, start=1):
        label = " ".join(top) if top else "none of A K Q J"
        row = [P(f"<b>{label}</b>", cell)]
        for c, (extra, _) in enumerate(COLS, start=1):
            hon = "".join(h for h in cd.HONORS if h in top + extra)
            n = need[hon]
            row.append(P(f"{f(n[2])}  ·  {f(n[3])}  ·  {f(n[4])}", cell))
            bid0 = max([b for b in (2, 3, 4) if n[b] == 0], default=0)
            style.append(("BACKGROUND", (c, r), (c, r), BID_COLORS[bid0]))
        data.append(row)
    t = Table(data, colWidths=[1.55 * inch] + [1.35 * inch] * 4, repeatRows=1)
    t.setStyle(TableStyle(style))
    return t


def legend():
    cells = [[P(f"<b>{BID_NAMES[b]}</b>", cellhead) for b in (0, 2, 3, 4)]]
    t = Table(cells, colWidths=[0.95 * inch] * 4)
    t.setStyle(TableStyle([("BACKGROUND", (i, 0), (i, 0), BID_COLORS[b]) for i, b in enumerate((0, 2, 3, 4))]
                          + [("GRID", (0, 0), (-1, -1), 0.6, colors.white),
                             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Arial", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.75 * inch, 0.5 * inch, "Setback bidding charts  ·  from simulating every 6-card hand")
    canvas.drawRightString(letter[0] - 0.75 * inch, 0.5 * inch, f"page {doc.page}")
    canvas.restoreState()


def build(fn="setback_bidding_charts.pdf"):
    need4, need6 = need_table(4, 224), need_table(6, 112)
    s = []

    # ---------------------------------------------------------------- page 1: how to read
    s += [P("Setback Bidding Charts", title),
          P("How high to bid, from your trump holding: 4-handed and 6-handed", subtitle),
          P("How to read a chart", h1)]
    s += bullets([
        "<b>Pick your trump suit</b>: the suit you'd name if you win the bid, usually your best.",
        "<b>Find your row</b>: which of the <b>A, K, Q, J</b> of that suit you hold.",
        "<b>Find your column</b>: whether you also hold the <b>10</b> and/or the <b>2</b> of that "
        "suit. \"No 10 or 2\" means you hold neither.",
        "<b>Read the cell</b>: three numbers, for <b>bid 2 · bid 3 · bid 4</b>. Each is how many "
        "<b>small trumps</b> (3 through 9 of your trump suit) you need for that bid. <b>0</b> means "
        "none needed; <b>–</b> means never, however many you hold.",
        "<b>Count your small trumps and bid the highest level you have enough for.</b> If you "
        "don't have enough for 2, pass.",
    ])
    s += [Spacer(1, 4),
          P("The colour of a cell shows what you can bid with <b>no</b> small trumps. Every small "
            "trump you hold can only move you up from there, never down.", base),
          Spacer(1, 6), legend()]

    s += [P("Examples", h1)]
    ex = [
        ["4-handed", "A♠ 7♠ 5♠ + three other cards", "row A, column \"no 10 or 2\": 0 · 1 · –",
         "2 small trumps: enough for 3 → <b>bid 3</b>"],
        ["4-handed", "K♠ J♠ 4♠ 3♠ + two others", "row K J, \"no 10 or 2\": 0 · 0 · 2",
         "2 small trumps: enough for 4 → <b>bid 4</b>"],
        ["4-handed", "Q♥ 8♥ + four others", "row Q, \"no 10 or 2\": 0 · 4 · –",
         "1 small trump: 2 yes, 3 needs 4 → <b>bid 2</b>"],
        ["6-handed", "K♦ 6♦ + four others", "row K, \"no 10 or 2\": 1 · – · –",
         "1 small trump: enough for 2 → <b>bid 2</b>"],
        ["6-handed", "K♦ alone in diamonds", "same cell: 1 · – · –",
         "no small trumps → <b>pass</b>"],
        ["6-handed", "J♣ 5♣ 2♣ + three others", "row J, column \"+ 2\": 1 · 2 · –",
         "1 small trump → <b>bid 2</b> (one more for 3)"],
    ]
    t = Table([[P(f"<b>{a}</b>", small), P(b), P(c, small), P(d)] for a, b, c, d in ex],
              colWidths=[0.8 * inch, 1.95 * inch, 2.15 * inch, 2.1 * inch])
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    s += [t]

    s += [P("What \"should bid\" means", h1),
          P("A bid is on the chart when, on average, it <b>gains your team points</b>: the points you "
            "take when you make it outweigh losing the bid when you're set. The chart gives the "
            "<b>highest</b> such bid, which is the most you should go to in the auction. If a lower "
            "bid wins the auction, bid that: in whole-game simulations, raising only as far as you "
            "need beat jumping straight to your highest bid.", base),
          Spacer(1, 6),
          P("Hands right at a boundary are close calls, and either bid is fine. Off-suit Aces can tip "
            "a close call up a level (page 4).", base),
          PageBreak()]

    # ---------------------------------------------------------------- page 2: 4-handed
    s += [P("4-handed (two teams of two)", h1),
          P("Each cell: small trumps (3–9) needed to <b>bid 2 · bid 3 · bid 4</b>. "
            "Colour = what you can bid with no small trumps.", small),
          Spacer(1, 6), chart(4, need4), Spacer(1, 8), legend(),
          P("Reading the 4-handed chart", h2)]
    s += bullets([
        "<b>Almost every hand is worth a 2-bid.</b> The real decision is usually 2 or 3. Only about "
        "2% of hands should pass: two small trumps and nothing else, a bare 10, or a bare 2.",
        "<b>A 4-bid needs the Jack.</b> Ace-King-Queen without the Jack never bids 4. With the "
        "Jack and the Ace or King, you can bid 4 with one or two small trumps, or even with none.",
        "<b>The 10 or the 2 of trump is worth about one small trump</b>: the 10 for Game, the 2 for Low.",
        "<b>Any one honor is a 2-bid by itself.</b> For 3, a bare Ace needs one small trump, a King or "
        "Jack two, and a Queen four.",
    ])
    s += [PageBreak()]

    # ---------------------------------------------------------------- page 3: 6-handed
    s += [P("6-handed (three teams of two)", h1),
          P("Each cell: small trumps (3–9) needed to <b>bid 2 · bid 3 · bid 4</b>. "
            "Colour = what you can bid with no small trumps.", small),
          Spacer(1, 6), chart(6, need6), Spacer(1, 8), legend(),
          P("Reading the 6-handed chart", h2)]
    s += bullets([
        "<b>About one bid tighter than 4-handed.</b> With two other teams against you, roughly 28% "
        "of hands should pass.",
        "<b>The Ace is the key card.</b> Any hand with the Ace of trump is a 2-bid. Without it, a bare "
        "King, Queen or Jack needs small trumps even to bid 2.",
        "<b>A 4-bid needs the Ace plus the Jack or King</b>, usually with some length.",
        "<b>A bare Ace beats two non-Ace trumps.</b> Holding a lone Ace in one suit and, say, K-Q in "
        "another, name the Ace's suit.",
    ])
    s += [PageBreak()]

    # ---------------------------------------------------------------- page 4: background
    s += [P("Off-suit Aces", h1),
          P("An Ace in a side suit is worth about <b>half to one small trump</b>. <b>One</b> off-suit Ace "
            "can tip a close call up a level: 4-handed, K J plus one small trump with an Ace on the side "
            "is a 4-bid instead of a 3. <b>Two</b> off-suit Aces count as one extra small trump. They're "
            "left off the charts because they change the bid for under 1% of hands. Off-suit 10s don't "
            "change it at all.", base)]
    s += [P("Rules these charts assume", h1)]
    s += bullets([
        "6 cards each, from a 52-card deck. 4-handed is two teams of two; 6-handed is three teams of two.",
        "4 points per hand: <b>High</b> (highest trump played), <b>Low</b> (lowest trump played, scored "
        "by whoever <i>captures</i> it), <b>Jack</b> of trump, and <b>Game</b> (most counting cards: "
        "10 = 10, A = 4, K = 3, Q = 2, J = 1; a tie scores nobody).",
        "Bids are 2, 3 or 4. The bidder leads first, and that card sets trump.",
        "Follow suit if you can, or trump instead. When trump is led, you must follow with trump if you can.",
        "Making your bid scores the points you take; failing loses the bid. The other team scores what it takes.",
    ])
    s += [P("How the charts were made", h1),
          P("Every possible 6-card hand was played out by computer: all 20,358,520 of them, which reduce "
            "to 962,988 distinct hands once suits are relabeled. The bidder names their best suit and "
            "plays against 224 (4-handed) or 112 (6-handed) random deals of the other cards. Every "
            "player looks ahead at each decision: it tries each card it could play against many guesses "
            "of the hidden cards and keeps the best. Each chart cell averages every hand with that trump "
            "holding, weighted by how often it's dealt. The charts give the same bid as the full per-hand "
            "results for 91% of hands (4-handed) and 88% (6-handed); most of the rest are close calls.",
            base)]
    s += [P("What the charts don't cover", h1)]
    s += bullets([
        "<b>The auction.</b> The charts say what a hand is worth if you play it at that bid. Taking the "
        "bid also keeps the other team from getting it cheaply. In 4-handed, opening 2 on any hand "
        "comes out even, because nearly every hand is worth 2 anyway.",
        "<b>The score.</b> Near the end of a game you may want to bid more or less aggressively.",
        "<b>House rules.</b> Variants such as a Joker, off-Jack, low to whoever was dealt it, or a "
        "4-bid that wins the game would change the numbers.",
    ])
    s += [P("Key takeaways", h1)]
    s += bullets([
        "<b>4-handed:</b> bid 2 on almost anything; the question is whether you have a 3. Save 4 for "
        "hands with the Jack.",
        "<b>6-handed:</b> without the Ace, you need real length; pass weak hands.",
        "<b>The 10 or 2 of trump ≈ one small trump; an off-suit Ace ≈ half to one.</b>",
        "<b>In the auction, raise only as far as you need</b> to win the bid. Overbidding to your maximum "
        "gets you set more often.",
    ])

    doc = SimpleDocTemplate(fn, pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                            topMargin=0.7 * inch, bottomMargin=0.8 * inch,
                            title="Setback Bidding Charts", author="Setback simulator")
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    return fn


if __name__ == "__main__":
    print("wrote", build())
