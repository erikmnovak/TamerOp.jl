# Seeing an encoding and querying its regions

An encoding assigns each parameter to a label in a finite poset. A picture of
that assignment should let you answer a concrete question: **which region
contains this point?** A dimension picture then attaches the dimension of the
space at that label. Neither picture, by itself, determines the structure maps.
See [finite encodings](finite_encodings.md) for the distinction.

Start by inspecting the available views of your particular object:

```julia
using TamerOp
import TamerOp.Advanced as OA

available_visuals(enc)
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

## Rendering and saving

```julia
using CairoMakie
visualize(spec; size=(900, 650))
save_visual("square.svg", spec; size=(900, 650))
```

CairoMakie produces static figures. WGLMakie produces a browser scene; volume
slice sliders require a live Julia session. Exporting `:slice_viewer` to HTML
raises an error because offline Julia callbacks are unavailable; use `:image`
to export the selected slice instead. General hover inspection and linked region
selection are not implemented, and the interaction summary says so. Current
query labels are drawn annotations. Renderer controls are `figure` and `size`;
recipe options belong to specification construction. See
[optional integrations](optional_integrations.md) for installation and export.

The next teaching view will link these regions to the finite poset, the space
at a selected label, and its structure maps. The current exact region and
query specifications supply its reusable geometric foundation.
