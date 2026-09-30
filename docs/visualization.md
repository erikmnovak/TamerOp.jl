# Seeing an encoding, its spaces, and its maps

An encoding assigns each parameter to a label in a finite poset. A picture of
that assignment should let you answer a concrete question: **which region
contains this point?** A dimension picture then attaches the dimension of the
space at that label. To understand how a vector continues from one parameter
to another, inspect the corresponding structure map. The
[executable square notebook](tutorials/inspect_encoding.ipynb) follows that
whole path, starting from the module in [finite encodings](finite_encodings.md).

Start by inspecting the available views of your particular object:

```julia
import TamerOp as OP
import TamerOp.Advanced as OA

OP.available_visuals(enc)
OA.check_visual_request(enc; kind=:query_overlay, point=[1//1, 1//1],
                        box=([-1, -1], [3, 3]))
spec = OA.visual_spec(enc; kind=:query_overlay, point=[1//1, 1//1],
                      box=([-1, -1], [3, 3]))
OA.visual_summary(spec)
```

Here `enc` is a two-parameter encoding result, for example the square from the
finite-encoding lesson. Building a specification needs no plotting package.
The request report lists supported recipe keywords, the qualitative work
required to construct the view, and activated renderers. It does not estimate
elapsed time. Unknown keywords and options that would have no effect on the
chosen recipe are errors.

## Reading the parameter plane

The planar region recipes use the actual classifier. Grid regions begin at
filtration thresholds, respect each axis's orientation, and include their
unbounded final slabs after clipping to the requested viewing box. They are
not midpoint bins around sampled grades. Polyhedral encodings use rational
halfspace clipping, so a slanted region remains slanted. General polyhedral
views follow the rational-coordinate contract of that encoder; grid views
also preserve supported exact algebraic grades, such as square roots. A region with several
pieces keeps the same ID and color on every piece. Filtering the viewport does
not renumber the surviving labels.

Solid edges are included in the indicated region; dashed edges are excluded.
Dotted edges mark a cut made by the viewing window. An open circle excludes a
vertex; a filled marker includes it. Adjacent regions can have coincident
outlines: use an exact query to resolve ownership at that location. A
lower-dimensional region is drawn as a segment or point. Clipping an unbounded
region does not make it bounded mathematically.

Region `0`, where supplied by the classifier, means **outside the represented
encoding**. It is shown separately from a represented region whose vector
space has dimension zero. For integer encodings, the picture uses tiles for
the nearest lattice point, with round-to-even ties; the subtitle states this
drawing convention.

Queries are classified using the coordinates you supplied, before conversion
for drawing. This matters even for the square: `(2 + 2^-53, 1)` represented as
an exact rational lies beyond its closed right edge, although its horizontal
coordinate rounds to `2.0` in Float64. Use
`[2 + 1//(2^53), 1//1]` to express that exact point in Julia.

```julia
spec = OA.visual_spec(enc; kind=:query_overlay,
    points=[[2//1, 1//1], [2 + 1//(2^53), 1//1]],
    box=([3//2, 1//2], [5//2, 3//2]))
OA.visual_metadata(spec).query_readout
```

The readout retains original coordinates, the actual region ID, drawing
coordinates, and whether the point is inside the viewing window. Distinct
exact coordinates may occupy the same screen position. In that case the
subtitle flags drawing precision, and metadata retain the distinct points.
The picture is an approximation of coordinates, not a replacement classifier.
`metadata.geometry.components` retain exact vertices, dimensions, and edge
and vertex inclusion. A rank-query overlay similarly retains exact pairs,
region IDs, and computed ranks in `metadata.query_results`.

## Choosing a view

| Object and question | Recipe | Effective selections | Scope and work |
| --- | --- | --- | --- |
| Finite poset, module, or encoding: what is its order? | `:hasse` | `vertex` or `pair` in finite labels | Actual cover relations in a schematic layout; module inputs add dimensions without querying structure maps |
| Module or encoding: what space or map did I construct? | `:module_inspector` | `vertex` or `pair`; planar encodings also accept `point`, `parameter_pair`, and `box`; `matrix_limit` | Finite-poset and readout panels, plus actual planar regions when available; a defined pair requests its matrix and rank |
| Retained finite fringe: how does its matrix produce a space or map? | `:presentation_inspector` | `vertex` or `pair`; supported planar encodings also accept `point`, `parameter_pair`, and `box`; `upset`, `downset`, `matrix_limit`; single-stalk `basis=true` | Support membership, full coefficients and active blocks; optional embedded image basis, or endpoint bases and induced map for a defined pair |
| Encoding: which region contains a parameter? | `:regions`, `:region_labels`, `:query_overlay` | `box`; `point` or `points` for queries | Two-parameter grid, box, polyhedral, and integer encodings; materializes clipped geometry |
| Restricted Hilbert result: how large is each space? | `:hilbert_heatmap` | `box` | Same planar geometry; color records dimension |
| Cohomology dimensions: where is a degree supported? | `:cohomology_support`, `:cohomology_support_plane` | `box` | Same planar geometry; degree retained in the result |
| Rank result with geometric provenance: what survives from x to y? | `:rank_query_overlay` | `pair` or `pairs`, `box` | Exact comparable-point validation and stored rank queries |
| Rank table: which finite-poset pairs have a given rank? | `:rank_heatmap`, `:rank_rectangles` | None | Materializes a pair table; incomparable pairs are missing, not zero |
| Rectangle signed barcode: what do its coefficients reconstruct? | `:density_image` | None | Accumulates weights at actual axis coordinates, including irregular or negative coordinates |
| Sampled multiparameter image | `:mpp_image` | None | Displays the supplied image; distinct from rectangle reconstruction |
| Image or volume: which array entries are displayed? | `:image`, `:slice_viewer` | `view_dims=(x_axis,y_axis)`, `slice_indices`, `colormap` | Heatmap rows are y and columns are x; fixed-axis defaults are described below |
| Channel image | `:channels` | `view_dims`, `colormap` | One panel per channel; channel selection is not silently substituted for slicing |

Unspecified fixed image axes use their middle index, rounded down when the
length is even. The exception is a three-dimensional array with at most four
entries along axis 3: when that axis is fixed, its default is index 1, treating
it as a channel axis. Supply `slice_indices` to choose another slice explicitly.

This table covers the corrected shared paths. `available_visuals(obj)` and
`OA.check_visual_request(obj; kind=...)` give the full contract for a particular
object, including existing point-cloud, graph, barcode, and slice-family views.
Higher-dimensional encoding maps are not advertised as planar region views.
Geometry construction does not materialize module bases or cycle
representatives. Some other recipes, such as slice barcode queries, perform
additional mathematics; their request reports identify that work.

## From a parameter to a space and a map

The module inspector shows how the finite representation answers a question
about the original parameters. For the square encoding, choose two parameters
inside the support:

```julia
spec = OA.visual_spec(enc; kind=:module_inspector,
    parameter_pair=((1//2, 1//2), (3//2, 3//2)),
    box=([-1, -1], [3, 3]))
inspection = OA.visual_metadata(spec).inspection
inspection.matrix       # a 1 x 1 matrix [1] over QQ
inspection.rank         # 1
inspection.kernel_dimension # 0
```

The left panel shows the original parameter plane. Colors and IDs agree with
the middle panel, which draws the finite poset actually returned by the
encoder. Its arrows are **cover relations**: an arrow from `u` to `v` says
`u < v` with no finite label strictly between them. Vertical position indicates
order; horizontal and vertical plot coordinates are schematic, not original
parameters. The right panel shows the selected map. Its **columns are source
coordinates and its rows are target coordinates**. The field, matrix size,
rank, kernel dimension, and image dimension accompany the entries.

The chapter's nine-region grid and the package's signature poset are different
finite descriptions of the square. The inspector draws the returned poset and
looks up its labels; it never treats either label count or numeric ordering as
the definition of the square. A color can therefore occur on disconnected
pieces of the parameter plane.

To inspect one space, use `point=(1//1,1//1)`. To ask directly about the finite
model, use `vertex=q` or `pair=(u,v)`, with actual IDs from that model. Choose
one selection form at a time. For example:

```julia
classifier = OP.encoding_map(enc)
q = OA.locate(classifier, (1//1, 1//1))
stalk = OA.visual_spec(enc; kind=:module_inspector, vertex=q)
order = OA.visual_spec(OP.encoding_poset(enc); kind=:hasse)
```

The stalk readout describes coordinates in the module's stored basis. It does
not identify those basis vectors with input cycles or with an embedding into
the target of the indicator presentation. Use the presentation inspector below
to examine that embedding when a finite fringe is retained.

For a comparable non-cover pair, the poset view adds a dashed, bent arrow for
the selection while retaining the solid cover arrows. Equal labels give the
identity of their space, including the `0 x 0` identity of a zero space. An
incomparable pair, or a pair ordered only in reverse, has **no structure map
in the requested direction**; the readout does not substitute a zero matrix.

This distinction also applies before assigning finite labels. In the square,
`(1//4,3//2)` and `(3//2,1//4)` share a label but are incomparable in the original
coordinatewise order. Selecting them with `parameter_pair` correctly reports
that there is no ambient structure map. Selecting `pair=(q,q)` instead asks
for a map of the finite model and returns its identity. For grid encodings,
ambient comparisons honor the classifier's axis orientation.

Arbitrary finite posets and `PModule` objects need no geometric coordinates:
their inspector has the poset and readout panels. An encoding adds a parameter
panel only when its classifier supports the planar region recipes above.

### Cost and what the matrix means

`:hasse` and a stalk-only or unselected inspector use dimensions and leave lazy
structure maps uncomputed. Selecting a defined pair obtains the encoded
module and its structure map. For a lazy result, this can first materialize
the module's cover maps. No recipe requests the full table of all comparable
pairs. Cover extraction and region clipping still have costs of their own.

`matrix_limit=(12,12)` limits the number of displayed rows and columns. The
readout identifies truncation and retains the full selected matrix in
`OA.visual_metadata(spec).inspection.matrix`; rank and kernel dimension use
that full matrix. Coefficients are printed as field elements, including exact
rational or finite-field entries. With `RealField`, rank follows the stated
numerical tolerances. A matrix is a snapshot in the stored module's coordinate
bases; it is not a canonical representative under arbitrary basis changes.

The notebook adds two overlapping square summands to make the maps matter.
Along an increasing path, the spaces have dimensions `1 -> 2 -> 1`. Both
successive maps have rank one, yet their composite is zero: the vector coming
from the first square disappears before the endpoint in the second square.
The inspector can show these matrices; dimensions alone cannot explain this
behavior.

## From a presentation matrix to its image

The presentation inspector answers the preceding lesson's next question:
**how did the input produce this space?** At a parameter, its active upsets
select source columns and its active downsets select target rows. The image
of that restricted matrix is the stalk. Support membership alone does not
determine its dimension: an active matrix can be zero.

For the square encoding above, inspect the retained finite presentation and
then select an interior stalk:

```julia
H = OP.encoding_presentation(enc)
q = OA.locate(OP.encoding_map(enc), (1//1, 1//1))
s = OA.presentation_stalk(enc; vertex=q)
OA.active_rows(s), OA.active_columns(s)
OA.presentation_matrix(s)
OA.presentation_summary(s).dimension

with_basis = OA.presentation_stalk(enc; vertex=q, basis=true)
OA.image_basis(with_basis)

presentation_spec = OA.visual_spec(enc; kind=:presentation_inspector,
    point=(1//1, 1//1), basis=true, upset=1, downset=1,
    box=([-1, -1], [3, 3]))
```

The default stalk query computes the active block and its rank, leaving the
image basis uncomputed. With `basis=true`, each basis column is a vector in
the active downset coordinates; its row labels identify those coordinates.
Thus a zero-dimensional image in a one-dimensional target has a `1 x 0`
basis matrix, whereas a zero-dimensional target has no coordinate rows.

The figure shows the selected upset and downset, identifies active rows and
columns of the full coefficient matrix, and displays the active block and,
when requested, its image basis. `upset` and `downset` choose support panels;
they do not restrict the algebraic calculation to those two indicators.
On supported planar encodings, membership is colored on the actual
classifier geometry. Boundary styles describe those encoding regions,
including their viewing-window cuts; they are not newly inferred boundaries
of a merged support. With no supported ambient geometry, the figure reports
membership on finite labels. It does not invent spatial coordinates.

`OP.encoding_presentation(enc)` returns the retained finite fringe or
`nothing` when none is available. It does not reconstruct a presentation from
a module. Original input data or historical presentation metadata are not a
substitute: the finite witness must belong to the current poset and field.
Presentation queries also accept a finite `FringeModule` directly. They inspect
its image, whose chosen bases need not match the stored module's bases in an
arbitrary hand-built encoding result.

### Inspect the induced map

For a comparable pair, the target indicator coordinates project onto the
downsets still active at the later label. Let `R` denote that projection,
`Bu` and `Bv` the endpoint image bases, and `C` the induced map. They satisfy
`Bv * C == R * Bu`: project in downset coordinates, then express the result in
the target image basis.

```julia
u = OA.locate(OP.encoding_map(enc), (1//2, 1//2))
v = OA.locate(OP.encoding_map(enc), (3//2, 3//2))
m = OA.presentation_map(enc; source=u, target=v)
Bu = OA.image_basis(OA.source_stalk(m))
Bv = OA.image_basis(OA.target_stalk(m))
R, C = OA.ambient_projection(m), OA.induced_map(m)
@assert Bv * C == R * Bu   # exact rational coefficients in this example

map_spec = OA.visual_spec(enc; kind=:presentation_inspector,
    parameter_pair=((1//2, 1//2), (3//2, 3//2)),
    upset=1, downset=1, box=([-1, -1], [3, 3]))
```

A pair query explicitly computes the endpoint bases required for this map;
`basis=true` is only a single-stalk option. The map view shows endpoint blocks,
image bases, the ambient projection, and the induced matrix. Finite-label
selection asks about the finite presentation; `parameter_pair` also checks
order in the original parameters. An unordered pair has no forward map.

An unselected overview does not compute image bases. `matrix_limit` limits
displayed entries, not matrix construction, rank, or basis computation.
Coefficients remain field-valued text, and numerical-field ranks and bases
follow the supplied tolerances. The equality above is exact over `QQ`;
numerical comparisons require those tolerances.

The [notebook](tutorials/inspect_encoding.ipynb) applies this calculation to
the two squares. At `(1//2,5//2)`, row `D2` and column `U1` are active but
their block is `[0]`; the image-basis shape is `1 x 0`. It also checks
`Bv * C == R * Bu` and the zero composite along the three-point path.

## Rendering and saving

```julia
using CairoMakie
OP.visualize(spec; size=(1500, 650))
OP.save_visual("square.svg", spec; size=(1500, 650))
```

CairoMakie produces static figures. WGLMakie produces a browser scene; volume
slice sliders require a live Julia session. Exporting `:slice_viewer` to HTML
raises an error because offline Julia callbacks are unavailable; use `:image`
to export the selected slice instead. General hover inspection and linked region
selection are not implemented, and the interaction summary says so. Current
query labels are drawn annotations. Both inspectors use explicit
API selections: changing `point`, `vertex`, or a pair builds a new view.
Renderer controls are `figure` and `size`;
recipe options belong to specification construction. See
[optional integrations](optional_integrations.md) for installation and export.

The [notebook](tutorials/inspect_encoding.ipynb) constructs the square, checks
selected spaces and maps, explains the two-square presentation through active
blocks and image bases, and exports PNG and SVG figures through the public
API. These static panels provide the finite-encoding inspection lesson.
Clicking across linked panels and following a user-selected vector remain
future work.
