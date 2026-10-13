#!/usr/bin/env python3
"""Draw the retained chain correspondence as a labelled mathematical schematic.

Run with --check to verify the published SVG against this source. No package
computation is represented by the arrows beyond the displayed map equations.
"""

from argparse import ArgumentParser
from pathlib import Path
from xml.sax.saxutils import escape

ASSET = Path(__file__).resolve().parents[1] / "assets" / "implementation" / "source_representatives.svg"
INK = "#172b42"
BLUE = "#245c99"
TEAL = "#0f766e"
MUTED = "#475569"


def draw():
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="520" height="838" '
        'viewBox="0 0 520 838" role="img" aria-labelledby="title description">',
        '<title id="title">Chains and cocycles on source cells</title>',
        '<desc id="description">A chain map i sends computed chains C to source chains S. '
        'A retained retraction p goes from S to C and recovers a computed chain after lifting. '
        'A filling f with boundary z maps to a filling i f whose boundary is i z. '
        'A computed cocycle alpha evaluates a source chain s by applying p first: '
        '(p transpose alpha)(s) equals alpha(p s). The source cochain is restricted to active cells. '
        'This is a schematic of the map equations.</desc>',
        '<!-- Authored schematic; regenerate with docs/build_scripts/render_source_figures.py. -->',
        '<defs>',
    ]
    for name, color in (("lift", BLUE), ("retraction", TEAL), ("boundary", MUTED)):
        parts.append(f'<marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" '
                     'markerWidth="7" markerHeight="7" orient="auto">'
                     f'<path d="M 1 1 L 9 5 L 1 9" fill="none" stroke="{color}" '
                     'stroke-width="1.5"/></marker>')
    parts.extend(['</defs>', '<rect width="520" height="838" fill="white"/>',
                  f'<g font-family="DejaVu Sans, sans-serif" fill="{INK}" text-anchor="middle">'])

    def text(x, y, value, size=24, color=INK):
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}">{escape(value)}</text>')

    def arrow(x1, y1, x2, y2, kind="lift"):
        color = {"lift": BLUE, "retraction": TEAL, "boundary": MUTED}[kind]
        parts.append(f'<path d="M {x1} {y1} L {x2} {y2}" stroke="{color}" '
                     f'stroke-width="2.5" fill="none" marker-end="url(#{kind})"/>')

    def rule(y):
        parts.append(f'<path d="M 26 {y} H 494" stroke="#cbd5e1" stroke-width="1.2"/>')

    text(260, 37, "Chains and cocycles on source cells", 25)
    for x, title, symbol in ((22, "Computed", "C"), (306, "Source", "S")):
        parts.append(f'<rect x="{x}" y="75" width="192" height="112" rx="10" '
                     'fill="#eef5fb" stroke="#93b3cf" stroke-width="1.5"/>')
        text(x + 96, 113, title, 25)
        text(x + 96, 157, f"chains {symbol}", 27)
    arrow(220, 126, 300, 126)
    text(260, 111, "i", 26, BLUE)
    arrow(300, 162, 220, 162, "retraction")
    text(260, 193, "p", 26, TEAL)
    text(260, 229, "When a retraction p is retained,", 23)
    text(260, 260, "p(i(c)) = c for each computed chain c.", 23)
    rule(283)

    text(260, 324, "A filling and its boundary", 25)
    text(118, 374, "f", 31, BLUE)
    text(402, 374, "i f", 31, BLUE)
    arrow(155, 364, 365, 364)
    text(260, 352, "i", 24, BLUE)
    arrow(118, 387, 118, 434, "boundary")
    arrow(402, 387, 402, 434, "boundary")
    text(88, 419, "∂", 27, MUTED)
    text(372, 419, "∂", 27, MUTED)
    text(118, 476, "z", 31, BLUE)
    text(402, 476, "i z", 31, BLUE)
    arrow(155, 466, 365, 466)
    text(260, 453, "i", 24, BLUE)
    text(260, 523, "∂f = z   ⇒   ∂(i f) = i z", 26)
    text(260, 561, "The cycle z is filled by f at death.", 23)
    rule(586)

    text(260, 629, "A cocycle evaluates source chains", 25)
    text(118, 682, "α", 31, TEAL)
    text(402, 682, "pᵀα", 31, TEAL)
    arrow(155, 672, 355, 672, "retraction")
    text(260, 660, "pᵀ", 24, TEAL)
    text(260, 727, "(pᵀα)(s) = α(p s)", 27)
    text(260, 767, "Send s back by p, then evaluate α.", 23)
    text(260, 798, "Use only active source cells.", 23)
    return "\n".join(parts + ["</g></svg>", ""])


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = draw()
    if args.check:
        if not ASSET.exists() or ASSET.read_text() != content:
            raise SystemExit(f"Figure differs from its source: {ASSET}")
        print("Source correspondence schematic matches its source.")
    else:
        ASSET.parent.mkdir(parents=True, exist_ok=True)
        ASSET.write_text(content)
        print("Rendered source correspondence schematic.")


if __name__ == "__main__":
    main()
