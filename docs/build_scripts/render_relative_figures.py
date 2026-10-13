#!/usr/bin/env python3
"""Draw the relative pair's segment and delayed-entry schematics.

These are labelled mathematical schematics, not captured package output.
Run with --check to verify that the tracked SVGs agree with this source.
The executable recipe supplies the matching interval and connecting-map check.
"""

from argparse import ArgumentParser
from pathlib import Path
from xml.sax.saxutils import escape

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "relative_persistence"
INK = "#172b42"
BLUE = "#245c99"
TEAL = "#0f766e"
ORANGE = "#b45309"
MUTED = "#475569"


class Drawing:
    def __init__(self, height, title, description):
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="520" height="{height}" '
            f'viewBox="0 0 520 {height}" role="img" aria-labelledby="title description">',
            f'<title id="title">{escape(title)}</title>',
            f'<desc id="description">{escape(description)}</desc>',
            '<!-- Authored schematic; regenerate with docs/build_scripts/render_relative_figures.py. -->',
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="7" markerHeight="7" orient="auto"><path d="M 1 1 L 9 5 L 1 9" '
            f'fill="none" stroke="{BLUE}" stroke-width="1.5"/></marker></defs>',
            f'<rect width="520" height="{height}" fill="white"/>',
            f'<g font-family="DejaVu Sans, sans-serif" fill="{INK}" text-anchor="middle">',
        ]

    def text(self, x, y, value, size=24, color=INK, anchor="middle"):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" '
                          f'fill="{color}" text-anchor="{anchor}">{escape(value)}</text>')

    def line(self, x1, y1, x2, y2, color=BLUE, width=2.5, arrow=False, dash=""):
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        self.parts.append(f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" '
                          f'stroke="{color}" stroke-width="{width}" '
                          f'stroke-dasharray="{dash}"{marker}/>')

    def circle(self, x, y, fill, color=BLUE, radius=6):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{radius}" '
                          f'fill="{fill}" stroke="{color}" stroke-width="2.5"/>')

    def finish(self):
        return "\n".join(self.parts + ["</g></svg>", ""])


def segment_boundary():
    d = Drawing(610, "The relative edge and its connecting map",
                "At t at least one, K is an oriented edge e from v0 to v1, and A is its endpoints. "
                "Its ambient boundary is v1 minus v0. Both vertices vanish in quotient chains, "
                "making the edge a relative cycle. The connecting map takes the edge class "
                "in H1(K,A) to the endpoint difference in H0(A). This is a chain schematic, "
                "not a geometric realization of the quotient.")
    d.text(260, 36, "An edge relative to its endpoints", 25)
    d.text(260, 88, "K at t ≥ 1", 25)
    d.line(140, 143, 368, 143, width=4, arrow=True)
    d.circle(135, 143, ORANGE, ORANGE)
    d.circle(385, 143, ORANGE, ORANGE)
    d.text(260, 128, "e", 25, BLUE)
    d.text(135, 182, "v₀ (10)", 23)
    d.text(385, 182, "v₁ (20)", 23)
    d.text(260, 218, "A = {v₀, v₁}", 23, ORANGE)
    d.text(260, 256, "∂ₖe = v₁ − v₀", 27)
    d.line(35, 278, 485, 278, "#cbd5e1", 1)
    d.text(260, 319, "Q = C(K) / C(A)", 26)
    d.text(260, 358, "Both vertex coordinates become zero.", 22, MUTED)
    d.text(260, 402, "∂e = 0, so e is a relative cycle.", 23, TEAL)
    d.line(35, 427, 485, 427, "#cbd5e1", 1)
    d.text(118, 469, "H₁(K, A)", 25)
    d.text(398, 469, "H₀(A)", 25)
    d.text(115, 520, "[e]", 27, BLUE)
    d.line(162, 512, 278, 512, arrow=True)
    d.text(220, 498, "δ", 25, BLUE)
    d.text(390, 520, "[v₁] − [v₀]", 25, ORANGE)
    d.text(260, 574, "δ recovers the original endpoint boundary.", 21)
    return d.finish()


def delayed_entry():
    d = Drawing(644, "Cell entry and relative homology in a changing pair",
                "All grades are exact integers. The quotient coordinate v0 exists on [0,2), "
                "v1 on [0,3), and e on [1,4). A filled endpoint includes the grade and an open "
                "endpoint excludes it. Relative H0 has intervals [0,1) and [0,2); relative H1 "
                "has [3,4). The edge becomes a cycle only when both vertices have entered A.")
    d.text(260, 36, "Letting cells enter A later", 25)
    d.text(260, 82, "Coordinates in the quotient chains", 24)
    xs = [170 + 65 * t for t in range(5)]
    for t, x in enumerate(xs):
        d.text(x, 124, str(t), 23)
        d.line(x, 137, x, 277, "#e2e8f0", 1, dash="3 4")
    d.text(62, 124, "t", 23, MUTED)
    for label, start, end, y in (("v₀", 0, 2, 161), ("v₁", 0, 3, 210), ("e", 1, 4, 259)):
        d.text(76, y + 8, label, 25)
        d.line(xs[start], y, xs[end], y, BLUE, 5)
        d.circle(xs[start], y, BLUE)
        d.circle(xs[end], y, "white")
    d.circle(98, 311, BLUE)
    d.text(114, 318, "enters K", 21, anchor="start")
    d.circle(300, 311, "white")
    d.text(316, 318, "enters A", 21, anchor="start")
    d.line(35, 344, 485, 344, "#cbd5e1", 1)
    d.text(260, 385, "Intervals in relative homology", 24)
    for t, x in enumerate(xs):
        d.text(x, 426, str(t), 23)
        d.line(x, 439, x, 575, "#e2e8f0", 1, dash="3 4")
    d.text(62, 426, "t", 23, MUTED)
    d.text(76, 492, "H₀", 25)
    d.text(76, 568, "H₁", 25)
    for start, end, y, color in ((0, 1, 464, TEAL), (0, 2, 508, TEAL), (3, 4, 560, ORANGE)):
        d.line(xs[start], y, xs[end], y, color, 5)
        d.circle(xs[start], y, color, color)
        d.circle(xs[end], y, "white", color)
    d.text(260, 615, "Filled ends are included; open ends are excluded.", 20)
    return d.finish()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    drawings = {"segment_boundary.svg": segment_boundary(), "delayed_entry.svg": delayed_entry()}
    if not args.check:
        ASSETS.mkdir(parents=True, exist_ok=True)
    for name, content in drawings.items():
        target = ASSETS / name
        if args.check:
            if not target.exists() or target.read_text() != content:
                raise SystemExit(f"Figure differs from its source: {target}")
        else:
            target.write_text(content)
    print("Relative pair schematics match their source." if args.check else "Rendered relative pair schematics.")


if __name__ == "__main__":
    main()
