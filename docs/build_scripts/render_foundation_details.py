#!/usr/bin/env python3
"""Render authored foundation schematics and check their finite examples.

Run from any directory. ``--check`` checks the independent calculations and
that the tracked SVGs agree with this source, without writing files. These
diagrams illustrate the mathematics in the chapters; they are not package
output or claims about a renderer or an interactive frontend.
"""

from argparse import ArgumentParser
from fractions import Fraction
from itertools import product
from pathlib import Path
from xml.sax.saxutils import escape


ASSETS = Path(__file__).resolve().parents[1] / "assets"
INK = "#172b42"
BLUE = "#245c99"
TEAL = "#0f766e"
ORANGE = "#b45309"
MUTED = "#475569"


class Drawing:
    def __init__(self, height, title, description):
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="500" height="{height}" '
            f'viewBox="0 0 500 {height}" role="img" aria-labelledby="title description">',
            f'<title id="title">{escape(title)}</title>',
            f'<desc id="description">{escape(description)}</desc>',
            '<!-- Authored schematic; regenerate with docs/build_scripts/render_foundation_details.py. -->',
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="7" markerHeight="7" orient="auto"><path d="M 1 1 L 9 5 L 1 9" '
            f'fill="none" stroke="{TEAL}" stroke-width="1.5"/></marker></defs>',
            f'<rect width="500" height="{height}" fill="white"/>',
            f'<g font-family="DejaVu Sans, sans-serif" fill="{INK}" text-anchor="middle">',
        ]

    def text(self, x, y, value, size=24, color=INK, anchor="middle"):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" '
                          f'fill="{color}" text-anchor="{anchor}">{escape(value)}</text>')

    def rect(self, x, y, w, h, fill="white", stroke=BLUE, width=2, radius=0, dash=""):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
                          f'fill="{fill}" stroke="{stroke}" stroke-width="{width}" '
                          f'rx="{radius}" stroke-dasharray="{dash}"/>')

    def line(self, x1, y1, x2, y2, color=TEAL, width=2.5, arrow=False, dash=""):
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        self.parts.append(f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" '
                          f'stroke="{color}" stroke-width="{width}" '
                          f'stroke-dasharray="{dash}"{marker}/>')

    def circle(self, x, y, radius, fill, stroke=BLUE, width=2):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{fill}" '
                          f'stroke="{stroke}" stroke-width="{width}"/>')

    def column(self, x, y, entries, color=INK):
        for i, entry in enumerate(entries):
            self.text(x, y + i * 28, str(entry), color=color)
        top, bottom = y - 21, y + (len(entries) - 1) * 28 + 6
        for side in (-1, 1):
            edge, inward = x + side * 23, x + side * 16
            self.parts.append(f'<path d="M {inward} {top} H {edge} V {bottom} H {inward}" '
                              f'fill="none" stroke="{color}" stroke-width="2"/>')

    def finish(self):
        return "\n".join(self.parts + ["</g></svg>", ""])


def ring_class():
    d = Drawing(550, "A loop remains while its homology class becomes zero",
                "An authored schematic over F two. At parameter one, eight squares form a ring. "
                "At parameter five the center square enters. The orange loop is the boundary "
                "of the center: nonzero in homology before five and zero at five. The interval "
                "below is closed at zero and open at five; one and four lie inside it.")
    d.text(250, 36, "Following the loop through the filling", 25)
    for center, scale in ((125, 1), (375, 5)):
        d.text(center, 83, f"t = {scale}", 26)
        for i, j in product(range(3), repeat=2):
            if (i, j) == (1, 1) and scale < 5:
                continue
            fill = "#b7e2dd" if (i, j) == (1, 1) else "#dbeafe"
            d.rect(center - 48 + 32 * i, 115 + 32 * j, 32, 32, fill, width=1.5)
        d.rect(center - 16, 147, 32, 32, "none", ORANGE, 4)
    d.line(191, 163, 308, 163, arrow=True)
    d.text(250, 139, "include", 22, TEAL)
    d.text(125, 251, "M(1) = F₂")
    d.text(375, 251, "M(5) = 0")
    d.text(125, 287, "loop class", 23, BLUE)
    d.text(375, 287, "zero class", 23, MUTED)
    d.line(202, 278, 298, 278, arrow=True)
    d.text(250, 338, "The loop now bounds the center cell.", 23)
    d.text(250, 393, "Interval in first homology", 24)
    d.line(70, 434, 420, 434, BLUE, 6)
    d.circle(70, 434, 7, BLUE)
    d.circle(420, 434, 7, "white", BLUE, 3)
    for scale in (0, 1, 4, 5):
        x = 70 + 70 * scale
        if scale in (1, 4):
            d.line(x, 421, x, 448, MUTED, 1.5)
        d.text(x, 477, str(scale), 24)
    d.text(456, 442, "t", 25)
    d.text(250, 525, "0 ≤ t < 5: the class is nonzero", 23)
    return d.finish()


def commuting_square():
    d = Drawing(590, "Two routes give the same linear map",
                "An authored module over any field k on p less than q and r less than s. "
                "The spaces at p and s are k, and those at q and r are k squared. "
                "The maps from p are columns (1,0) and (0,1); each map to s is row (1,1). "
                "The vector one reaches one along either route. There is no comparison between q and r.")
    d.text(250, 36, "Two routes through the same module", 25)
    d.text(250, 70, "Matrices act on the selected vectors", 22, MUTED)
    for x, y, name, space, vector in ((115, 172, "p", "k", "1"),
                                    (375, 172, "q", "k²", "(1, 0)"),
                                    (115, 424, "r", "k²", "(0, 1)"),
                                    (375, 424, "s", "k", "1")):
        d.rect(x - 75, y - 48, 150, 96, "#eff6ff", radius=7)
        d.text(x, y - 12, f"M({name}) = {space}", 23)
        d.text(x, y + 28, vector, 27, BLUE)
    d.line(196, 172, 294, 172, arrow=True)
    d.column(245, 110, (1, 0))
    d.line(115, 226, 115, 370, arrow=True)
    d.column(65, 288, (0, 1))
    d.line(375, 226, 375, 370, arrow=True)
    d.text(430, 307, "[1  1]", 23)
    d.line(196, 424, 294, 424, arrow=True)
    d.text(245, 468, "[1  1]", 23)
    d.text(250, 532, "Both composites send a ↦ a.", 24)
    d.text(250, 566, "Thus M(p ≤ s) = [1].", 23, MUTED)
    return d.finish()


def rho(bands):
    return (int("L" not in bands), int("H" in bands))


def compare_encodings():
    d = Drawing(858, "Nine band labels reduce to four signatures",
                "An authored schematic of the closed square on the whole real plane. "
                "A cropped three by three grid labels each region by its band pair and signature. "
                "L means less than zero, C means zero through two inclusive, and H means greater than two. "
                "The disconnected upper left and lower right regions share signature (0,1). "
                "Below, pi nine sends (1,1) to (C,C), rho sends it to (1,0), and pi four gives (1,0) directly.")
    d.text(250, 36, "Nine labels, four signatures", 25)
    d.text(250, 72, "Cropped parameter plane", 23, MUTED)
    colors = {(0, 0): "#e2e8f0", (1, 0): "#dbeafe",
              (0, 1): "#ffedd5", (1, 1): "#d1fae5"}
    for i, bx in enumerate("LCH"):
        for j, by in enumerate("HCL"):
            x, y = 85 + 110 * i, 104 + 110 * j
            signature = rho((bx, by))
            d.rect(x, y, 110, 110, colors[signature], "#94a3b8", 1)
            d.text(x + 55, y + 43, f"({bx}, {by})", 23)
            d.text(x + 55, y + 80, f"({signature[0]}, {signature[1]})", 23)
    d.rect(85, 104, 330, 330, "none", "#64748b", 1.5, dash="5 5")
    d.rect(195, 214, 110, 110, "none", BLUE, 4)
    d.line(65, 454, 441, 454, arrow=True)
    d.line(65, 454, 65, 91, arrow=True)
    for value, x, y in (("0", 195, 324), ("2", 305, 214)):
        d.line(x, 449, x, 459, MUTED, 1.5)
        d.line(60, y, 70, y, MUTED, 1.5)
        d.text(x, 486, value, 23)
        d.text(44, y + 8, value, 23)
    d.text(460, 462, "x", 24)
    d.text(65, 77, "y", 24)
    d.text(250, 524, "0 and 2 belong to band C.", 23)
    d.text(250, 565, "One parameter, two lookups", 25)
    d.rect(163, 586, 174, 56, "#f8fafc", radius=7)
    d.text(250, 622, "(1, 1)", 26)
    d.line(211, 648, 125, 709, arrow=True)
    d.text(130, 672, "π₉", 28, TEAL)
    d.line(289, 648, 375, 709, arrow=True)
    d.text(370, 672, "π₄", 28, TEAL)
    d.rect(22, 716, 174, 66, "#dbeafe", radius=7)
    d.rect(304, 716, 174, 66, "#dbeafe", radius=7)
    d.text(109, 758, "(C, C)", 27)
    d.text(391, 758, "(1, 0)", 27)
    d.line(202, 749, 298, 749, arrow=True)
    d.text(250, 730, "ρ", 28, TEAL)
    d.text(250, 829, "Both recover M(1, 1) = ℚ.", 24)
    return d.finish()


def chosen_vector():
    d = Drawing(714, "A vector enters the overlap and then maps to zero",
                "An authored schematic over k in the summand bases of the two closed squares. "
                "At p=(one half, one half), the selected vector is one. The column (1,0) sends it "
                "to (1,0) at q=(three halves, three halves). The row (0,1) sends it to zero at "
                "r=(five halves, five halves). The target space at r is k, belonging to the second square.")
    d.text(250, 36, "Follow the first square's vector", 25)
    rows = ((85, "p = (½, ½): M(p) = k", "chosen vector: 1", BLUE),
            (305, "q = (³⁄₂, ³⁄₂): M(q) = k²", "image: (1, 0)", BLUE),
            (525, "r = (⁵⁄₂, ⁵⁄₂): M(r) = k", "image: 0", MUTED))
    for y, label, vector, color in rows:
        d.rect(35, y, 430, 108, "#f8fafc", radius=7)
        d.text(250, y + 35, label, 24)
        d.text(250, y + 80, vector, 27, color)
    d.line(140, 199, 140, 299, arrow=True)
    d.column(250, 234, (1, 0))
    d.text(366, 253, "inclusion", 23, TEAL)
    d.line(140, 419, 140, 519, arrow=True)
    d.text(250, 478, "[0  1]", 25)
    d.text(372, 478, "projection", 23, TEAL)
    d.text(250, 678, "M(r) contains the second square's vectors.", 22)
    return d.finish()


def nine_region_encoding():
    d = Drawing(1040, "Nine parameter regions and their finite poset",
                "An authored schematic in two stacked panels. The upper panel assigns each "
                "parameter its band pair, using L below zero, C from zero through two inclusive, "
                "and H above two. The lower panel is the nine-element product poset, with "
                "arrows rightward and upward. Only the central label carries Q; the others carry zero. "
                "The outer parameter regions continue beyond the drawing window.")
    d.text(250, 36, "Regions and their finite labels", 25)
    d.text(250, 74, "Parameter plane (x, y)", 24)
    for i, bx in enumerate("LCH"):
        for j, by in enumerate("HCL"):
            x, y = 85 + 110 * i, 108 + 110 * j
            middle = (bx, by) == ("C", "C")
            d.rect(x, y, 110, 110, "#dbeafe" if middle else "#f8fafc", "#94a3b8", 1)
            d.text(x + 55, y + 48, f"({bx}, {by})", 23)
            if middle:
                d.text(x + 55, y + 84, "ℚ", 27, BLUE)
    d.rect(85, 108, 330, 330, "none", "#64748b", 1.5, dash="5 5")
    d.rect(195, 218, 110, 110, "none", BLUE, 3)
    d.line(65, 457, 441, 457, arrow=True)
    d.line(65, 457, 65, 94, arrow=True)
    for value, x, y in (("0", 195, 328), ("2", 305, 218)):
        d.line(x, 452, x, 462, MUTED, 1.5)
        d.line(60, y, 70, y, MUTED, 1.5)
        d.text(x, 489, value, 23)
        d.text(44, y + 8, value, 23)
    d.text(460, 465, "x", 24)
    d.text(65, 80, "y", 24)
    d.line(250, 510, 250, 567, arrow=True)
    d.text(286, 547, "π₉", 28, TEAL)
    d.text(250, 608, "Finite poset P₉", 25)
    for row in range(3):
        for col in range(2):
            y = 684 + 135 * row
            d.line(152 + 150 * col, y, 198 + 150 * col, y, arrow=True)
    for col in range(3):
        for row in range(2):
            x = 100 + 150 * col
            d.line(x, 770 + 135 * row, x, 733 + 135 * row, arrow=True)
    for i, bx in enumerate("LCH"):
        for j, by in enumerate("HCL"):
            x, y = 100 + 150 * i, 684 + 135 * j
            middle = (bx, by) == ("C", "C")
            d.rect(x - 47, y - 43, 94, 86, "#dbeafe" if middle else "white", radius=6)
            d.text(x, y - 8, f"({bx}, {by})", 23)
            d.text(x, y + 29, "ℚ" if middle else "0", 27, BLUE if middle else INK)
    d.text(250, 1022, "Arrows show immediate comparisons.", 22, MUTED)
    return d.finish()


def signature_encoding():
    d = Drawing(640, "Four signatures and their vector spaces",
                "An authored schematic of the Boolean poset on signatures (u,c). "
                "The bottom label (0,0) and top label (1,1) carry zero. "
                "The incomparable middle labels (1,0) and (0,1) carry Q and zero respectively. "
                "Arrows point upward and increase one bit. These positions are finite-poset layout, "
                "not coordinates in the original plane.")
    d.text(250, 36, "Four signatures in P₄", 25)
    d.text(250, 75, "Each label records (u, c).", 23, MUTED)
    for source, target in (((220, 417), (140, 347)), ((280, 417), (360, 347)),
                           ((140, 253), (220, 183)), ((360, 253), (280, 183))):
        d.line(*source, *target, arrow=True)
    for x, y, label, space in ((250, 138, "(1, 1)", "0"),
                                (110, 300, "(1, 0)", "ℚ"),
                                (390, 300, "(0, 1)", "0"),
                                (250, 462, "(0, 0)", "0")):
        d.rect(x - 65, y - 43, 130, 86, "#dbeafe" if space == "ℚ" else "white", radius=7)
        d.text(x, y - 8, label, 26)
        d.text(x, y + 29, space, 28, BLUE if space == "ℚ" else INK)
    d.text(250, 554, "u = 1 when x ≥ 0 and y ≥ 0", 23)
    d.text(250, 589, "c = 1 when x > 2 or y > 2", 23)
    d.text(250, 626, "Each bit is zero otherwise.", 22, MUTED)
    return d.finish()


def square_indicators():
    d = Drawing(1420, "An upset and a downset meet in a closed square",
                "Three stacked authored schematics use the same coordinate scale. "
                "The upset U includes x and y at least zero and continues upward and rightward. "
                "The downset D includes x and y at most two and continues downward and leftward. "
                "Their intersection is the closed square from zero to two. "
                "Solid colored boundaries are included. Dashed frames crop unbounded regions.")
    d.text(250, 36, "A square from two regions", 25)
    for index, (label, color) in enumerate((("U: x ≥ 0, y ≥ 0", BLUE),
                                           ("D: x ≤ 2, y ≤ 2", TEAL),
                                           ("U ∩ D = [0, 2]²", INK))):
        top = 112 + 450 * index
        d.text(250, top - 31, label, 25, color)
        d.rect(110, top, 280, 280, "#f8fafc", "none")
        if index == 0:
            d.rect(180, top, 210, 210, "#dbeafe", "none")
            d.line(180, top, 180, top + 210, BLUE, 3)
            d.line(180, top + 210, 390, top + 210, BLUE, 3)
            d.text(280, top + 115, "U", 31, BLUE)
        elif index == 1:
            d.rect(110, top + 70, 210, 210, "#d5eee9", "none")
            d.line(110, top + 70, 320, top + 70, TEAL, 3)
            d.line(320, top + 70, 320, top + 280, TEAL, 3)
            d.text(210, top + 185, "D", 31, TEAL)
        else:
            d.rect(180, top + 70, 140, 140, "#b7d6dd", BLUE, 3)
            d.text(250, top + 150, "U ∩ D", 28)
        d.rect(110, top, 280, 280, "none", "#94a3b8", 1.5, dash="5 5")
        d.line(90, top + 298, 420, top + 298, arrow=True)
        d.line(90, top + 298, 90, top - 12, arrow=True)
        for value, x, y in (("0", 180, top + 210), ("2", 320, top + 70)):
            d.line(x, top + 293, x, top + 303, MUTED, 1.5)
            d.line(85, y, 95, y, MUTED, 1.5)
            d.text(x, top + 330, value, 23)
            d.text(69, y + 8, value, 23)
        d.text(439, top + 306, "x", 24)
        d.text(90, top - 25, "y", 24)
        note = ("Continues upward and rightward", "Continues downward and leftward", "All four edges and corners are included")[index]
        d.text(250, top + 373, note, 22, color)
    return d.finish()


def overlapping_summands():
    d = Drawing(670, "Two closed square supports overlap",
                "An authored parameter plane showing S one, the closed square from zero to two, "
                "and S two, the closed square from one to three. The overlap has dimension two. "
                "Points p=(one half,one half), q=(three halves,three halves), and "
                "r=(five halves,five halves) have spaces k, k squared, and k. "
                "The arrows follow an increasing sequence of parameters, not a chosen vector.")
    d.text(250, 36, "Two square supports overlap", 25)
    d.text(250, 76, "Parameter plane (x, y)", 24)
    d.rect(100, 270, 180, 180, "#dbeafe", "none")
    d.rect(190, 180, 180, 180, "#d5eee9", "none")
    d.rect(190, 270, 90, 90, "#b7d6dd", "none")
    d.rect(100, 270, 180, 180, "none", BLUE, 3)
    d.rect(190, 180, 180, 180, "none", TEAL, 3)
    d.line(80, 450, 425, 450, arrow=True)
    d.line(100, 470, 100, 137, arrow=True)
    for n in range(4):
        x, y = 100 + 90 * n, 450 - 90 * n
        d.line(x, 445, x, 455, MUTED, 1.5)
        if n:
            d.line(95, y, 105, y, MUTED, 1.5)
            d.text(76, y + 8, str(n), 23)
        d.text(x, 487, str(n), 23)
    d.text(444, 458, "x", 24)
    d.text(100, 124, "y", 24)
    d.line(155, 395, 224, 326, arrow=True)
    d.line(246, 304, 314, 236, arrow=True)
    for x, y, label in ((145, 405, "p"), (235, 315, "q"), (325, 225, "r")):
        d.circle(x, y, 6, INK, "white", 2)
        d.text(x + 22, y + 23, label, 26)
    d.text(143, 304, "S₁", 26, BLUE)
    d.text(330, 344, "S₂", 26, TEAL)
    d.text(235, 291, "dim 2", 22)
    d.text(140, 540, "S₁ = [0, 2]²", 24, BLUE)
    d.text(360, 540, "S₂ = [1, 3]²", 24, TEAL)
    d.text(250, 591, "p: k       q: k²       r: k", 25)
    d.text(250, 633, "At q, use the basis order (S₁, S₂).", 23)
    return d.finish()


def f2_rank(columns):
    basis = {}
    for col in columns:
        while col:
            pivot = col.bit_length() - 1
            if pivot in basis:
                col ^= basis[pivot]
            else:
                basis[pivot] = col
                break
    return len(basis)


def check_mathematics():
    # Boundary matrices of the actual cubical ring, independently of TamerOp.
    squares = list(product(range(3), repeat=2))
    vertices = list(product(range(4), repeat=2))
    edges = [("h", i, j) for i in range(3) for j in range(4)]
    edges += [("v", i, j) for i in range(4) for j in range(3)]
    vertex_id, edge_id = {v: i for i, v in enumerate(vertices)}, {e: i for i, e in enumerate(edges)}
    d1 = []
    for direction, i, j in edges:
        end = (i + 1, j) if direction == "h" else (i, j + 1)
        d1.append((1 << vertex_id[(i, j)]) | (1 << vertex_id[end]))
    d2 = {}
    for i, j in squares:
        boundary = (("h", i, j), ("h", i, j + 1), ("v", i, j), ("v", i + 1, j))
        d2[(i, j)] = sum(1 << edge_id[e] for e in boundary)
    loop = d2[(1, 1)]
    loop_boundary = 0
    for i, column in enumerate(d1):
        if loop & (1 << i):
            loop_boundary ^= column
    assert loop_boundary == 0
    for t, expected_h1 in ((1, 1), (4, 1), (5, 0)):
        columns = [d2[s] for s in squares if s != (1, 1) or t >= 5]
        assert len(edges) - f2_rank(d1) - f2_rank(columns) == expected_h1
        assert f2_rank(columns + [loop]) - f2_rank(columns) == expected_h1
    # The chosen four-point module commutes over every field: both products are 1.
    assert sum(a * b for a, b in zip((1, 1), (1, 0))) == 1
    assert sum(a * b for a, b in zip((1, 1), (0, 1))) == 1
    # Check rho on its whole finite poset, then the distinct boundary classes.
    bands = list(product("LCH", repeat=2))
    leq = lambda a, b: all(x <= y for x, y in zip(a, b))
    order = {c: i for i, c in enumerate("LCH")}
    for a, b in product(bands, repeat=2):
        if leq(tuple(order[x] for x in a), tuple(order[x] for x in b)):
            assert leq(rho(a), rho(b))
    probes = list(product((Fraction(-1), Fraction(0), Fraction(1, 2), Fraction(2), Fraction(5, 2), Fraction(3)), repeat=2))
    band = lambda x: "L" if x < 0 else "C" if x <= 2 else "H"
    pi9 = lambda p: tuple(band(x) for x in p)
    pi4 = lambda p: (int(all(x >= 0 for x in p)), int(any(x > 2 for x in p)))
    dim = lambda p: int(all(0 <= x <= 2 for x in p))
    for p in probes:
        assert rho(pi9(p)) == pi4(p)
        assert dim(p) == int(pi9(p) == ("C", "C")) == int(pi4(p) == (1, 0))
    for p, q in product(probes, repeat=2):
        if leq(p, q):
            expected_rank = dim(p) * dim(q)
            assert expected_rank == int(pi9(p) == pi9(q) == ("C", "C"))
            assert expected_rank == int(pi4(p) == pi4(q) == (1, 0))
    # The presentation's active identity block has one pivot per common square.
    def active(point, lower, upper):
        return all(lower <= x <= upper for x in point)
    path = [(Fraction(i, 2), Fraction(i, 2)) for i in (1, 3, 5)]
    assert [int(active(p, 0, 2)) + int(active(p, 1, 3)) for p in path] == [1, 2, 1]
    assert [int(active(p, 0, 2)) for p in path] == [1, 1, 0]
    assert sum(a * b for a, b in zip((0, 1), (1, 0))) == 0
    assert all(active(p, 0, 2) and active(p, 1, 3) for p in ((1, 1), (2, 2)))
    assert not active((Fraction(1, 2), Fraction(5, 2)), 0, 2)
    assert not active((Fraction(1, 2), Fraction(5, 2)), 1, 3)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    check_mathematics()
    drawings = {
        "persistence_modules/ring_class.svg": ring_class(),
        "persistence_modules/commuting_square.svg": commuting_square(),
        "finite_encodings/compare_encodings.svg": compare_encodings(),
        "indicator_presentations/chosen_vector.svg": chosen_vector(),
        "finite_encodings/nine_region_encoding.svg": nine_region_encoding(),
        "finite_encodings/signature_encoding.svg": signature_encoding(),
        "indicator_presentations/square_indicators.svg": square_indicators(),
        "indicator_presentations/overlapping_summands.svg": overlapping_summands(),
    }
    for name, source in drawings.items():
        path = ASSETS / name
        if args.check:
            assert path.read_text(encoding="utf-8") == source, f"Regenerate {path}"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
    print("Checked ring boundaries, commuting paths, encoding recovery and vector images.")
    print("SVG sources match." if args.check else "Wrote the foundation schematics.")


if __name__ == "__main__":
    main()
