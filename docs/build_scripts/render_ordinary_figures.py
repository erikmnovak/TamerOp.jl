#!/usr/bin/env python3
"""Render the weighted clique example in the ordinary persistence guide."""
from argparse import ArgumentParser
from pathlib import Path
from render_foundation_details import Drawing, BLUE, TEAL, MUTED

TARGET = Path(__file__).resolve().parents[1] / 'assets/ordinary_persistence/rips_filling.svg'


def rips_filling():
    d = Drawing(580, 'The diagonal fills a four-edge cycle',
        'A schematic of the supplied sparse weighted graph, over F two. Four edges '
        'form a cycle at parameter one. At two the diagonal from vertex one to '
        'vertex three enters together with triangles 123 and 134. Their common '
        'diagonal cancels in the boundary, leaving the original cycle. '
        'First homology has interval [1,2). Positions do not encode distances.')
    d.text(250, 36, 'When triangles fill the loop', 25)
    for center, stage in ((125, 1), (375, 2)):
        d.text(center, 84, f't = {stage}', 26)
        points = [(center-56,226),(center+56,226),(center+56,114),(center-56,114)]
        if stage == 2:
            for triangle, color in (((0,1,2),'#b7e2dd'),((0,2,3),'#dbeafe')):
                vertices=' '.join(f'{points[i][0]},{points[i][1]}' for i in triangle)
                d.parts.append(f'<polygon points="{vertices}" fill="{color}"/>')
            d.line(*points[0], *points[2], TEAL, 3)
        for i, p in enumerate(points):
            d.line(*p, *points[(i+1)%4], BLUE, 4)
            d.circle(*p, 5, BLUE)
            d.text(p[0]+(-18 if i in (0,3) else 18), p[1]+(25 if i<2 else -13), str(i+1), 22)
        d.text(center, 299, 'H₁ = F₂' if stage==1 else 'H₁ = 0', 25)
    d.line(195, 170, 305, 170, arrow=True)
    d.text(250, 362, '∂(123 + 134) = 12 + 23 + 34 + 14', 22)
    d.text(250, 399, 'The shared diagonal cancels over F₂.', 22, MUTED)
    d.text(250, 449, 'Interval in first homology', 24)
    d.line(120, 487, 380, 487, BLUE, 6)
    d.circle(120, 487, 7, BLUE)
    d.circle(380, 487, 7, 'white', BLUE, 3)
    d.text(120, 528, '1', 24)
    d.text(380, 528, '2', 24)
    d.text(424, 495, 't', 25)
    d.text(250, 567, 'Filled end included; open end excluded.', 21)
    return d.finish().replace('render_foundation_details.py','render_ordinary_figures.py')


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    content=rips_filling()
    if args.check:
        if not TARGET.exists() or TARGET.read_text()!=content:
            raise SystemExit(f'Figure differs from its source: {TARGET}')
    else:
        TARGET.parent.mkdir(parents=True,exist_ok=True)
        TARGET.write_text(content)
    print('Ordinary persistence schematic matches its source.' if args.check else 'Rendered ordinary persistence schematic.')


if __name__ == '__main__':
    main()
