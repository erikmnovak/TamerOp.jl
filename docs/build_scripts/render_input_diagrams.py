#!/usr/bin/env python3
"""Render authored schematics for ingestion and exact rhomboid guides.

Run with --check to compare the tracked SVGs with these drawing definitions
without creating directories or changing files.
"""

from argparse import ArgumentParser
from html import escape
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "assets"
INK = "#243c49"
MUTED = "#526473"
TEAL = "#287271"
ORANGE = "#a34f2f"
BACKGROUND = "#fffdf8"


class Drawing:
    def __init__(self, height, title, description):
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="520" height="{height}" '
            f'viewBox="0 0 520 {height}" role="img" aria-labelledby="title description">',
            f'<title id="title">{escape(title)}</title>',
            f'<desc id="description">{escape(description)}</desc>',
            '<!-- Authored schematic; regenerate with docs/build_scripts/render_input_diagrams.py. -->',
            '<defs><marker id="arrow" markerWidth="7" markerHeight="7" '
            'refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7" '
            f'fill="{MUTED}"/></marker></defs>',
            f'<rect width="520" height="{height}" fill="{BACKGROUND}"/>',
            '<g font-family="DejaVu Sans, sans-serif" '
            f'fill="{INK}" stroke-linecap="round" stroke-linejoin="round">',
        ]

    def text(self, x, y, value, size=24, anchor="middle", color=INK):
        self.parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
            f'fill="{color}">{escape(value)}</text>'
        )

    def line(self, x1, y1, x2, y2, color=MUTED, width=2, arrow=False):
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{color}" stroke-width="{width}"{marker}/>'
        )

    def dot(self, x, y, filled=True):
        self.parts.append(
            f'<circle cx="{x}" cy="{y}" r="8" '
            f'fill="{TEAL if filled else BACKGROUND}" stroke="{TEAL}" stroke-width="3"/>'
        )

    def finish(self):
        return "\n".join(self.parts + ["</g></svg>", ""])


def floor_placement():
    drawing = Drawing(
        710,
        "Restriction and floor placement give different sampled filtrations",
        "A single cell is born at grade one. Restricting to grid zero and two keeps it "
        "absent at zero. Floor placement moves its birth to zero. Filled circles "
        "mean the cell is present; the open circle means it is absent.",
    )
    drawing.text(260, 38, "A cell born at grade 1", 26)
    drawing.text(260, 79, "Supplied grid: {0, 2}", 23)
    for y, title in ((181, "Original filtration"), (355, "Restriction to the grid"),
                     (538, "Floor placement")):
        drawing.text(260, y - 53, title)
        drawing.line(82, y, 465, y, arrow=True)
        for x, grade in ((90, "0"), (265, "1"), (440, "2")):
            drawing.line(x, y - 6, x, y + 6)
            drawing.text(x, y + 35, grade, 23)
    drawing.line(265, 181, 440, 181, TEAL, 7)
    drawing.dot(265, 181)
    drawing.text(265, 165, "birth", 22)
    drawing.dot(90, 355, filled=False)
    drawing.dot(440, 355)
    drawing.text(90, 426, "absent", 22)
    drawing.text(440, 426, "present", 22)
    drawing.line(90, 538, 440, 538, TEAL, 7)
    drawing.dot(90, 538)
    drawing.dot(440, 538)
    drawing.text(260, 610, "The birth has moved to 0.", 23)
    drawing.text(260, 677, "Parameter increases to the right.", 23)
    return drawing.finish()


def module_diamond():
    drawing = Drawing(
        646,
        "A finite module needs equal composites around the diamond",
        "Four rational spaces labelled one through four. Both paths from one to four "
        "consist of two maps. The left path gives one; the right gives two. Changing "
        "the map from three to four to one makes the two composites agree.",
    )
    drawing.text(260, 38, "Two paths through the diamond", 25)
    drawing.text(260, 82, "Every space is ℚ.", 23)
    drawing.text(260, 116, "Arrow labels are scalar matrices.", 23)
    arrows = (
        (244, 398, 120, 305, "[1]", 160, 369, MUTED),
        (276, 398, 400, 305, "[1]", 360, 369, MUTED),
        (120, 276, 244, 190, "[1]", 160, 232, MUTED),
        (400, 276, 276, 190, "[2]", 360, 232, ORANGE),
    )
    for x1, y1, x2, y2, label, tx, ty, color in arrows:
        drawing.line(x1, y1, x2, y2, color, 3, arrow=True)
        drawing.text(tx, ty, label, 25, color=color)
    for x, y, label in ((260, 424, "1: ℚ"), (88, 298, "2: ℚ"),
                        (432, 298, "3: ℚ"), (260, 179, "4: ℚ")):
        drawing.text(x, y, label, 26)
    drawing.text(260, 477, "via 2: 1 × 1 = 1", 24)
    drawing.text(260, 519, "via 3: 2 × 1 = 2", 24, color=ORANGE)
    drawing.text(260, 582, "Set the map 3 → 4 to [1].", 24)
    drawing.text(260, 620, "Both composites then equal [1].", 23)
    return drawing.finish()


def terminal_cap():
    drawing = Drawing(
        911,
        "The terminal cap joins the two components at radius one",
        "The rhomboid for two sites at zero and two is cut at depth one. Its horizontal "
        "section is an edge between the two singleton vertices. At radius below one "
        "those vertices are separate, and at radius one the cap joins them. The "
        "vertical axis is positive depth and the horizontal axis sums site coordinates.",
    )
    drawing.text(260, 38, "Why a depth window needs its cap", 25)
    drawing.text(260, 82, "Two sites: u = 0 and v = 2.", 23)
    drawing.text(260, 121, "Schematic at radius r = 1", 23)
    drawing.line(75, 495, 75, 177, arrow=True)
    drawing.line(75, 495, 451, 495, arrow=True)
    drawing.text(75, 160, "depth", 23)
    drawing.parts.append(
        '<polygon points="135,465 135,340 365,215 365,340" '
        'fill="#dceae6" stroke="#78958f" stroke-width="2"/>'
    )
    drawing.line(115, 340, 390, 340, TEAL, 5)
    vertices = (
        (135, 465, "∅", 155, 472, "start"),
        (135, 340, "{u}", 135, 316, "middle"),
        (365, 340, "{v}", 395, 348, "start"),
        (365, 215, "{u,v}", 365, 194, "middle"),
    )
    for x, y, label, tx, ty, anchor in vertices:
        drawing.dot(x, y)
        drawing.text(tx, ty, label, 23, anchor=anchor)
    for y, depth in ((465, "0"), (340, "1"), (215, "2")):
        drawing.text(61, y + 7, depth, 23, anchor="end")
    for x, coordinate in ((135, "0"), (365, "2")):
        drawing.text(x, 527, coordinate, 23)
    drawing.text(260, 380, "cap at depth 1", 23)
    drawing.text(260, 566, "sum of site coordinates", 23)
    drawing.text(260, 627, "At fixed depth k = 1", 25)
    drawing.text(260, 672, "r < 1: two components", 24)
    drawing.dot(150, 704)
    drawing.dot(370, 704)
    drawing.text(260, 760, "r = 1: one component", 24)
    drawing.line(150, 792, 370, 792, TEAL, 5)
    drawing.dot(150, 792)
    drawing.dot(370, 792)
    drawing.text(260, 857, "The horizontal section keeps", 23)
    drawing.text(260, 890, "the joining edge at depth 1.", 23)
    return drawing.finish()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    drawings = {
        "ingestion/floor_placement.svg": floor_placement(),
        "ingestion/module_diamond.svg": module_diamond(),
        "rhomboid/terminal_cap.svg": terminal_cap(),
    }
    for name, content in drawings.items():
        target = ASSETS / name
        if args.check:
            if not target.exists() or target.read_text() != content:
                raise SystemExit(f"Figure differs from its source: {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
    print("Input schematics match their source." if args.check else "Rendered input schematics.")


if __name__ == "__main__":
    main()
