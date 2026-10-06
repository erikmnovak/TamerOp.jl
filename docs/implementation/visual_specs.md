# From visual specifications to renderers and exports

A figure of a finite encoding should preserve the distinction that makes its
mathematics useful. Two zero matrix entries can mean different things: one is
forced by the parameter order, while the other is a coefficient that could have
been nonzero. A missing resolution degree is different again. Without a
termination certificate, its absence does not establish a vanishing term.

TamerOp prepares those distinctions before asking a rendering backend to draw.
The input object and selected question produce a `VisualizationSpec`: literal
coefficient text, semantic graphical layers, axes, annotations and retained
mathematical metadata. CairoMakie and WGLMakie consume the same specification.
Their job is layout and drawing; they do not construct a resolution, choose a
basis or infer a mathematical certificate from a picture.

## Separate the mathematical request from its presentation

The public path first checks that the object's recipe accepts the requested
selection and options. Each object family declares its kinds and meaningful
keywords. Building a specification may perform explicit mathematical work;
rendering an existing specification reuses that prepared answer.

```text
stored object + mathematical selection
                  |
           validate the request
                  |
       prepare coefficients and claims
                  |
           VisualizationSpec
             /           \
       Cairo figure     WGL scene
             \           /
         selected backend export
```

The specification has either layers or child panels. Its layers distinguish
points, paths, support polygons, interval data, text and literal matrices.
Semantic roles such as source, target and selected are resolved by the style at
render time. This permits color or grayscale changes without changing a matrix,
its labels or the membership of a support. Matrix coefficients stay textual;
finite-field coefficients are not converted to magnitudes on a real color scale.

Panel metadata records selected indices, field, category, degree convention
and computation scope. Matrix panels retain the exact selected matrix alongside
the displayed block. Their target-row/source-column labels identify coordinates;
a display limit is not a mathematical truncation. Empty matrices keep both
original dimensions, so a map into the zero space remains distinguishable from
one out of it.

## Resolve storage conventions before drawing an arrow

The resolution guide's diamond module has projective multiplicities one, two
and one in degrees zero, one and two. Those numbers describe the resolution
terms; they do not decompose the resolved module. A support sheet therefore
labels a selected principal summand of a term, not a purported persistent class.

Projective differentials run from degree `k` to `k-1`, whereas injective
differentials run from `k-1` to `k`. The selected degree consequently marks a
source column in one view and a target row in the other. A one-step upset
presentation stores its matrix transposed relative to that display convention.
The preparation step normalizes these cases; the renderer sees only target rows
and source columns. Injective coefficients are recovered at the recorded target
socles, where they determine the maps between principal downsets.

For source label `u` and target label `v`, the principal-summand coefficient is
allowed only when `v ≤ u`, on either side. A forbidden nonzero is rejected.
The displayed forbidden zeros receive a dagger and a separate semantic role;
permitted zeros retain ordinary coefficient text. At a selected finite vertex,
preparation restricts the matrix to active summands, preserving their global IDs.
An upset presentation produces a cokernel, a downset copresentation a kernel,
and a fringe map an image. The visible construction and selected dimension follow
that distinction.

The grade-plane view accepts a supplied planar order embedding. It compares
order using the supplied coordinates before converting the drawing to floating
point, and rejects distinct grades that collapse to one displayed point. It
never uses Hasse-layout positions as grades. These remain finite-category
multiplicities: [RIVET's Betti convention](https://rivet.readthedocs.io/en/latest/preliminaries.html#invariants-of-a-bipersistence-module)
concerns an ambient minimal free resolution and supplies mathematical context,
not an automatic identification or an algorithm dependency.

## A certificate is data, not a visual impression

The default table counts stored generators and declares its minimality and
terminal status unchecked. Optional verification freshly checks the whole
stored prefix. Besides the augmented equations, it checks vertexwise exactness:
once consecutive composites vanish, adjacent ranks must sum to the dimension
of the intermediate stalk. The projective augmentation must be surjective; the
injective coaugmentation must be injective. Stored term maps must agree with the
canonical principal-summand coordinates, and stored differential maps must agree
with their extracted coefficient matrices.

Minimality uses each map's actual kernel or image. For a projective term at
vertex `v`, let `D` be its map to the preceding object and let `D_old` retain
columns born strictly below `v`. If `b` summands are born at `v`, the cover is
minimal exactly when

$$\operatorname{rank}(D)-\operatorname{rank}(D_{\rm old})=b.$$

This says its kernel lies in the radical: no new generator can be removed using
older generators. Dually, an injective hull is essential when its image contains
the socle of the injective term. If `E_soc` consists of the standard basis
columns for summands ending at `v`, the check is

$$\operatorname{rank}[D\ E_{\rm soc}]=\operatorname{rank}(D).$$

These checks work at the last retained term too. Absence of a next differential
alone is insufficient to certify minimality. A nonzero terminal kernel or
cokernel marks a truncated prefix; a zero one certifies termination. The view
never pads uncomputed degrees with zero rows. Exact coefficients use field-aware
rank and equality; numerical fields retain their declared tolerance semantics.

A supplied basis change undergoes a separate check. Its source and target
matrices must be invertible and obey the grading, as must their inverses. With
`B = inverse(T)*A*S`, the equation `A*S = T*B` certifies the selected incidence map
in new coordinates. This does not find a minimal presentation or transform a
whole complex. Adjacent differentials and the augmentation still use their
original coordinates unless an independently constructed object supplies them.

## Retained work, drawing costs and export

A table allocates a degree-by-vertex count array without elimination. Incidence
selects one differential and optionally a stalk block. Explicit verification
traverses all retained degrees and finite vertices and performs ranks; displaying
only a few rows does not reduce that verification scope. Dense active stalks and
basis-change inversions can dominate a view of a large sparse resolution.
Matrices in specification metadata are copied so that later display edits cannot
mutate the resolution. The graph and separate support panels add their own
layout and scene costs; support-panel limits state how much is shown.

Neither resolution construction nor retained-result caching is hidden in these
recipes. A caller can retain a specification and render it again with another
style or backend. Rebuilding with verification recomputes the checks, avoiding
stale claims after mutation. A different mathematical selection requires a new
specification. Live inspection sessions have their own selection/lifecycle
contract; loading WGLMakie does not add callbacks to static resolution views.

Backends register render and save functions through optional package extensions.
An explicit backend must be active. Automatic inline rendering prefers WGL in
a notebook when available, then Cairo; saving prefers an available static
backend. Explicit filename exports use their declared format, while the simple
stem-based export returns the actual path, backend and format. These choices
change presentation and file production, not which mathematical object was
prepared. Shared matrix/layout code keeps literal coefficients and semantic
annotations consistent between Cairo and WGL.

The source boundaries are [specification types](../../src/visualization/types.jl),
[request/specification validation](../../src/visualization/validation.jl),
[resolution preparation](../../src/visualization/builders_resolutions.jl),
[render/export dispatch](../../src/visualization/rendering.jl), and
[shared Makie rendering](../../ext/visualization_makie_common.jl).
The [executable guide](../tutorials/resolutions.ipynb) develops the mathematical
example; the [reference](../src/reference/visualization.md) owns keyword contracts.

See the [bibliography](references.md).
