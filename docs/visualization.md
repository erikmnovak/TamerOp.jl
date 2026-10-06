# Choosing views and inspecting intervals

A view should answer a question about the object you have computed: where
its spaces live, which vectors continue, or what an interval's endpoints
establish. This guide explains how to choose that view, read its conventions,
explore slices and existing barcodes, and save the result.

For the workflow from an encoding to selected spaces, maps and presentation
bases, use [Exploring spaces and maps](spaces_and_maps.md). It also introduces
the linked inspector's selection controls and lifecycle. The
[square notebook](tutorials/inspect_encoding.ipynb) develops the mathematical
example with saved figures; the [ring notebook](tutorials/ring.ipynb) begins
with ordinary intervals.

The examples use the following imports. For parameter-plane and styling
examples, `enc` is the square constructed in the
[spaces-and-maps guide](spaces_and_maps.md#start-from-a-recognizable-object).
The slice and interval examples below supply their own inputs.

```julia
import TamerOp as OP
import TamerOp.Advanced as OA
```

`OP.available_visuals(object)` lists the views for a particular result.
`OA.check_visual_request(object; kind=...)` reports accepted keywords and
qualitative computation costs, without estimating elapsed time. Unknown
keywords and options that would have no effect on a recipe are errors.
Building a mathematical specification with `OA.visual_spec` needs no plotting
package; load CairoMakie to render a static figure, or WGLMakie for the live
sessions below.

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

## Start with the question

For an ordinary result `diagram`, a first barcode needs only its view and
degree after loading CairoMakie:

```julia
import CairoMakie
OP.visualize(diagram; kind=:barcode, dim=1)
```

The default figure includes a title, parameter labels, endpoint conventions
and readable margins. Small barcodes use a compact canvas; persistence diagrams
keep equal coordinate units. Symbols for infinity, censoring or clipping are
explained when present. Omitted groups and display-precision limitations remain
visible; the underlying exact records and full metadata are retained.

Use `window=(low, high)` on each call when comparing intervals on the same
scale. Cosmetic options such as `style=OP.VisualStyle(fontsize=18)` and
`size=(760, 360)` are useful for sharing a figure and need not appear in every
exploratory call. A style applies only to calls where it is supplied.
The [ring lesson](tutorials/ring.ipynb) develops this progression and finishes
with an optional preview/export section.

## Which vectors survive from here?

Stalk dimensions describe how much is present at each parameter. They do not
say how much survives a move to another parameter. Fixing a source point `p`
lets us ask that second question over the whole parameter plane:

```julia
OP.visualize(enc; kind=:rank_section, source=(1, 1))
```

At each comparable target `q`, the figure shows the rank of `M(p ≤ q)`:
how many independent vectors from the source remain independent in the target.
Both coordinates of `p` are fixed; both coordinates of `q` vary along the axes.
Labels give ranks and varying stalk dimensions in nonzero stalk regions;
repeated geometric cells share one label. For the closed square module,
the rank is one while both points remain in its support and zero after the
class has died. The source itself is marked in the plane.

The white region has rank zero. The gray order mask contains no forward map;
it is not evidence that a class has died. An unrepresented region is unknown
and has its own pale-gray appearance. This distinction is made in the original
parameter order, before replacing points with finite labels. Two incomparable
points in the same classifier region still have no structure map. Reversed
grid axes retain their declared order, which is stated on the axes.

To understand one value, select its target:

```julia
OP.visualize(enc; kind=:rank_section, source=(1, 1), point=(2, 2))
```

The adjacent panel displays the actual matrix with source columns and target
rows, both endpoint dimensions, and its rank. For the dual question, use
`target=(2, 2)`: the target stays fixed and the source varies. These are
sections of the ordinary rank invariant on comparable pairs. They are not
generalized ranks over intervals, nor do they identify or track individual
homology classes.

A few anchors can reveal what one view misses. They share a numerical scale:

```julia
OP.visualize(enc; kind=:rank_section, source=[(0, 0), (1, 1), (2, 2)])
```

For a finite module `M`, `source=1` or `target=3` selects a finite vertex and
shows a schematic Hasse diagram. Use `source=[1, 2, 3]` for finite small
multiples and `vertex=3` to select the other endpoint. An integer anchor on an
encoding also selects its finite model; it does not choose a representative
parameter. Use `OP.encoding_module(enc)` when comparing several finite anchors.

The linked inspector offers the same questions without rebuilding the session:

```julia
session = OP.inspection_session(enc; view=:rank_from)
OP.visualize(session)  # load WGLMakie for the live browser view
```

Choose **Rank from source** or **Rank to target** in the View control. A single
point or vertex sets the anchor. A source/target pair also selects the map;
its source is the anchor in the first view, its target in the second. Select
points in the navigation panels or use the exact coordinate fields to settle
boundary questions. Clicking the rank section itself selects the varying
endpoint while retaining its anchor. The section keeps both coordinates of the chosen endpoint
fixed. Reset clears the anchor; the usual module and presentation views remain
available in the same session.

Each distinct anchor label requests one rank row or column, without constructing
the full pair table. Live sessions retain these rows in their bounded cache.
Moving an anchor inside one fiber reuses the ranks but recomputes its exact
order mask. Only a selected map is copied into the readout; querying ranks may
materialize a lazy encoded module. Numerical fields use their declared rank
tolerances. A `box` clips the drawing, without changing the mathematical ranks.

## Maps between modules

A structure map moves between two spaces inside one module. A module morphism
`f : M → N` instead supplies a map `f(q) : M(q) → N(q)` at every finite label,
compatible with the structure maps. When you have constructed such a morphism,
start with its two modules and one component:

```julia
OP.visualize(f; vertex=2)
```

The source and target use the same poset layout. The selected matrix has source
coordinates in its columns and target coordinates in its rows. Its rank,
kernel dimension and cokernel dimension describe that component. The diagrams
are schematic: their horizontal and vertical positions are not filtration
parameters. The overview checks compatibility on the cover relations; it does
not infer a morphism from overlapping supports.
As with the algebra routines, the input modules are assumed to satisfy their
own composition laws.

To inspect compatibility along a comparable pair, select a naturality square:

```julia
OP.visualize(f; kind=:naturality, pair=(1, 4))
```

The four displayed maps satisfy

```math
N(p\leq q)\,f_p=f_q\,M(p\leq q).
```

The two products appear in the same source and target bases, so their comparison
has a precise meaning. Incomparable or reversed labels have no forward square.
With rational or finite-field coefficients the equality is exact; a numerical
field reports the residual and tolerance. A selected square establishes only
that equation, while the morphism overview checks all cover squares.

### Inspect what the morphism kills and reaches

```julia
OP.visualize(f; kind=:kernel_image_cokernel, vertex=2)
```

This request constructs the kernel, image and cokernel modules. It shows their
spaces over the common poset and the actual maps
`ker(f) → M`, `im(f) → N`, and `N → coker(f)` at the selected label. These matrices
use the bases chosen by the algebra routines. The computed modules retain their
own structure maps; their existence does not assert a direct-sum decomposition.
Use `OA.visual_spec` when you want to retain the inspectable result before
rendering or saving it. Matrix display limits cap the figure, not the underlying
computation.

For a supplied short exact sequence, the analogous view is:

```julia
ses = OA.short_exact_sequence(inclusion, projection)
OP.visualize(ses; vertex=2)
```

The sequence view checks the actual maps afresh, including naturality,
injectivity, surjectivity, zero composition and equality of the middle image
and kernel.
Matching dimensions alone does not prove exactness. An unchecked sequence
container that fails these conditions is displayed with its failed checks.
Numerical-field results carry the owner's tolerance-dependent meaning.

### Relate a supplied map to geometric supports

If `enc` retains a supported planar classifier on the **same finite poset
object** as `f`, use it to interpret both modules on that parameter domain:

```julia
OP.visualize(f; kind=:morphism_support, encoding=enc, box=([0, 0], [2, 2]))
```

The panels show the source support, target support, their overlay, and support
of the image. A zero morphism between nonzero modules has an empty image even where
the first two supports overlap. Both modules are pulled back along the supplied
classifier; similar vertex numbers in separately constructed encodings are not
sufficient. Relate their finite bases explicitly before making this comparison.
The window clips the drawing, while the classifier retains boundary and domain
semantics. Unrepresented regions remain distinct from represented zero spaces.

### Choose a Hom map or compare lifts

A computed ordinary Hom space supplies actual morphisms. Select a basis element
and then inspect it with the same component or square controls:

```julia
H = OP.hom(M, N)
OP.visualize(H; basis_index=1, vertex=2)
```

This choice materializes the Hom basis and uses the coordinate choices made by
the computation. A basis element is not canonical, and this is ordinary Hom in
the stated finite-poset category.

For a supplied cochain map `f`, the view follows its differential square in one
degree. Request the induced cohomology map explicitly when that is your question:

```julia
OP.visualize(f; kind=:chain_map, degree=0, vertex=2, induced=true)
```

For a supplied `ModuleCochainHomotopy`, compare its two maps with
`kind=:homotopy_comparison`. The witness equation is `f-g = d h+h d`, with
cohomological differentials increasing degree. `induced=true` also computes the
two cohomology maps in compatible quotient bases. Supplied `induced_maps=(a,b)`
are checked against that computation rather than accepted on appearance.

A projective resolution has the separate `:resolution_lift` view. Supply its
`target_resolution`, resolved `morphism`, and coefficient-matrix `lift`, then
choose `degree` and `vertex`. An optional `comparison_lift` is checked against
the same module map; `homotopy` supplies the actual homological witness between
them. Here degree `k` means `P_k`, and `homotopy[k+1]` goes from `P_k` to
`Q_(k+1)`. The view checks the supplied augmentation and chain equations, displays
the permitted generator-labelled coefficients and selected stalk maps, and
states the verified degree range. Generator labels name actual finite-poset
vertices, not invented geometric grades. A dagger marks a zero forced by the
order relation; an ordinary `0` is an allowed coefficient that happens to vanish.
A truncated lift does not certify an uncomputed tail. Different chain-level
coefficients can induce the same map;
no uniqueness of lifts is assumed.

All of these views use selections supplied in Julia. Static exports and WGL
browser figures retain those selections; they do not add live selection
controls. Their matrices, diagrams and numerical checks can be examined before
rendering through `OA.visual_spec(...)` and saved with `OP.save_visual(...)`.

## Inspect resolution terms and their maps

A resolution already stores finite algebraic data that can be inspected without
returning to an input filtration. Its default view is a degree-by-vertex table:
`visualize(resolution)` counts stored projective or injective summands.
Use `verify=true` to check exactness, minimality and completion of that prefix.
For a selected differential, use `kind=:resolution`, `degree`, `summand` and
`vertex`; its coefficient rows/columns and support panels share summand IDs.

The [resolution guide](tutorials/resolutions.ipynb) develops a diamond example,
its injective dual, truncation, supplied grades and graded basis changes.
The [reference](src/reference/visualization.md#resolutions-and-presentation-incidence)
gives the precise contracts. Grade-plane views require supplied coordinates;
a finite-poset resolution does not become an ambient multigraded free resolution
just because it can be drawn in a plane.

## Choosing a view

| Object and question | Recipe | Effective selections | Scope and work |
| --- | --- | --- | --- |
| Finite poset, module, or encoding: what is its order? | `:hasse` | `vertex` or `pair` in finite labels | Actual cover relations in a schematic layout; module inputs add dimensions without querying structure maps |
| Module or encoding: what space or map did I construct? | `:module_inspector` | `vertex` or `pair`; planar encodings also accept `point`, `parameter_pair`, and `box`; `matrix_limit` | Finite-poset and readout panels, plus actual planar regions when available; a defined pair requests its matrix and rank |
| Module morphism: how do its components fit together? | `:morphism_inspector`, `:naturality` | `vertex` or `pair`; `matrix_limit` | Shared source/target layout, component matrix, and actual naturality composites |
| Module morphism: what does it kill or reach? | `:kernel_image_cokernel`, `:morphism_support` | `vertex` for subquotient matrices; `encoding` and `box` for supports | Constructs actual subquotients or pulls supports back through an explicit common classifier |
| Short exact sequence: do these maps make it exact? | `:exact_sequence` | Optional `vertex`; `matrix_limit` | Fresh checks from the maps, plus selected inclusion, projection and composite |
| Ordinary Hom space: what does a basis map do? | `:hom_basis` | `basis_index`, `vertex` or `pair`; `matrix_limit` | Materializes the Hom basis and inspects the selected morphism |
| Supplied cochain map or homotopy: what descends to cohomology? | `:chain_map`, `:homotopy_comparison` | `degree`, `vertex`; opt-in `induced`; `matrix_limit` | Verifies the supplied equations; induced-map computation uses explicit quotient bases |
| Projective resolution: how does a supplied lift represent a module map? | `:resolution_lift` | `target_resolution`, `morphism`, `lift`; optional comparison/witness; `degree`, `vertex` | Verifies supplied augmented-chain equations and displays actual generator coefficients |
| Retained finite fringe: how does its matrix produce a space or map? | `:presentation_inspector` | `vertex` or `pair`; supported planar encodings also accept `point`, `parameter_pair`, and `box`; `upset`, `downset`, `matrix_limit`; single-stalk `basis=true` | Support membership, full coefficients and active blocks; optional embedded image basis, or endpoint bases and induced map for a defined pair |
| Live inspection session: how do these panels describe the same selection? | `:linked_inspector` | Change selections with `select_inspection!` or the live controls | WGLMakie with live Julia; module and retained-presentation views share a selection; use `inspection_snapshot` for static export |
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

For comparable image snapshots, pass the same `colorrange=(low, high)` to
each request. The limits must be finite and strictly increasing; `nothing`
keeps automatic scaling. This matters for constant masks: an all-zero image
and an all-one image must not both be rescaled to the same colour. `title`
and `colorbar_label` describe the quantity actually shown. These controls
also propagate to channel panels and live slice updates. Image axes include
the full half-index border of the outer pixels and use equal coordinate units.
The [ring notebook](tutorials/ring.ipynb) uses these controls for its empty,
ring and filled masks, all with `colorrange=(0, 1)`.
Two-dimensional images label their row and column indices; small image axes
show integer cell indices. A bare image call chooses the image recipe and
its canvas without needing `kind`, `view_dims` or `size` overrides.

`available_visuals(obj)` and
`OA.check_visual_request(obj; kind=...)` give the full contract for a particular
object, including point-cloud, graph, barcode, and slice-family views.
Higher-dimensional encoding maps are not advertised as planar region views.
Geometry construction does not materialize module bases or cycle
representatives. Some other recipes, such as slice barcode queries, perform
additional mathematics; their request reports identify that work.

## Follow a selected space or map

Use the [spaces-and-maps guide](spaces_and_maps.md#connect-the-answer-to-a-figure)
to connect parameters, finite labels and matrix readouts. Its
[presentation branch](spaces_and_maps.md#look-inside-a-retained-presentation)
explains active blocks, image bases and induced maps. Its
[live session](spaces_and_maps.md#explore-nearby-selections-in-a-live-session)
section covers exact entry, linked viewers, snapshots, reset and close.
Those same session operations apply to the slice and interval inspectors
below. A finite encoding alone need not retain cycles in its original data;
ordinary representative inspection requires the explicit retention described
in the barcode section.

## Move a line and read its intervals

How do classes continue when both parameters increase along a chosen line?
Restricting the encoded module to that line gives a one-parameter module.
Its intervals describe which classes persist along this particular path; they
do not determine every map of the original two-parameter module.

Return to the notebook's two-square example, constructed here as `enc2`. Its
summands have supports `[0,2]^2` and `[1,3]^2`. On the diagonal
`q(t) = (0,0) + t(1,1)`, their restrictions are the **closed** intervals
`[0,2]` and `[1,3]`. At `t=2`, both are still present. This differs from a
half-open barcode convention that would discard the first class at that point.

```julia
import WGLMakie

opts = OA.EncodingOptions(; backend=:pl_backend, poset_kind=:signature,
                          field=OP.CoreModules.QQField())
enc2 = OP.encode([OA.BoxUpset([0,0]), OA.BoxUpset([1,1])],
                 [OA.BoxDownset([2,2]), OA.BoxDownset([3,3])],
                 Rational{BigInt}[1 0; 0 1], opts)
slice_session = OP.inspection_session(enc2; box=([-1,-1], [4,4]),
    slice=(basepoint=(0,0), direction=(1,1)))
OP.visualize(slice_session; backend=:wglmakie)

# Select a multiplicity group in both the barcode and decorated diagram.
OA.select_inspection!(slice_session; interval=1)

# Translate the line upward: the intervals become [0,1] and [1,2].
OA.select_inspection!(slice_session;
    slice=(basepoint=(0,1), direction=(1,1)))
```

The **Exact basepoint** and **Exact direction** fields specify `q(t)=a+t*d`.
Directions must be coordinatewise nonnegative and nonzero; horizontal and
vertical lines are allowed. The direction is not normalized, so replacing
`d` by `2d` rescales the interval parameters. Rational input retains its exact
meaning. The **Draft angle** and **Draft offset** sliders propose a line using
decimal approximations; they populate the fields without computing persistence.
They start from the viewport center and shift perpendicularly to the chosen
direction. This can change the origin and scale of `t` even when the geometric
line is unchanged; read the committed equation when comparing endpoints.
Press **Apply slice** to commit the line and update both charts. Until then,
the figures describe the previously committed line.

Click a barcode interval or diagram point to select its group in both charts
and highlight its segment in the parameter plane. The **Selected interval
group** dropdown provides a keyboard alternative and an exact endpoint and
multiplicity readout. Coincident diagram points share a multiline label listing
their distinct groups; repeated clicks cycle through those groups. Identical
decorated intervals form one group with a multiplicity. Group IDs
apply only to the current line: changing the line clears its interval selection,
without claiming to track an individual class between slices. The independent
stalk or map selection remains in place.

Filled endpoint circles mean inclusion, and hollow circles mean exclusion.
The diagram's bracket labels retain the same distinction. An interval supported
at a single parameter remains visible on the diagonal. For example,
`basepoint=(0,2), direction=(1,1)` meets each square only at a corner: the answer
is `[0,0]` together with `[1,1]`, each of multiplicity one. A midpoint-only
sampling of the line would miss both.

The default `slice_scope=:window` computes the **finite-window restriction**.
An end at the viewing boundary is marked `?`, meaning that it is censored:
this computation does not establish whether the class continues beyond the
box. For instance, the diagonal window from `(3/2,3/2)` to `(7/4,7/4)` gives
two copies of `[3/2,7/4]`, with both ends censored. That answer describes the
restricted module, but it cannot recover the two different ambient intervals.

To ask for those actual intervals, compute the restriction to the whole line:

```julia
global_session = OP.inspection_session(enc2;
    box=([3//2,3//2], [7//4,7//4]),
    slice=(basepoint=(0,0), direction=(1,1)), slice_scope=:global)
OP.visualize(global_session; backend=:wglmakie)
OA.inspection_snapshot(global_session).metadata.slice_result.intervals
# Two groups: [0,2] and [1,3], each with multiplicity one.

# Keep the same line, but ask only what the small window establishes.
OA.select_inspection!(global_session; slice_scope=:window)
# One group: two copies of [3/2,7/4], with both ends censored.
```

The browser's **Slice scope** selector offers the same choice; press **Apply
slice** to commit it. Changing scope clears the selected interval because the
two computations can have different groups. In global mode the parameter
plane still uses the chosen box, while exact readouts retain the full interval.
An interval wholly outside the box remains in the result and can be selected
from the dropdown, even though there is no segment to highlight in that box.

The endpoint display distinguishes the evidence available:

| Display | What the computation establishes |
| --- | --- |
| Filled or hollow endpoint | A known finite endpoint, included or excluded respectively |
| Finite continuation arrow / `outside` label | A known finite endpoint lies beyond the drawing window; its exact value remains in the readout |
| `?` | A restricted computation has reached its boundary without establishing the ambient endpoint |
| Labelled `Inf` lane or arrow | The result certifies an infinite endpoint; this is an essential direction of the interval |

Infinite endpoints themselves are never included. A class born at a finite
parameter and persisting for all larger parameters has an interval such as
`[1,Inf)`. A class present for arbitrarily small parameters can instead have a
left endpoint `-Inf`; a constant class on the whole line has `(-Inf,Inf)`.
These are different from a finite interval whose death is merely outside the
picture. Displayed/total group and multiplicity counts explain omissions from
the window or rendering budget without changing the underlying interval data.
Diagram labels can move to remain readable. Their connector lines point to the
plotted intervals; moving a label changes neither its endpoints nor its selection.
Barcode endpoint labels grow inward from their anchors and sit above the bar,
so the selection highlight does not cover their text. These are drawing choices;
the exact interval endpoints and their inclusion remain in the readout.

For a window restriction, the computation uses the prepared exact planar
geometry. For a global restriction, it enumerates all classifier changes along
the line, including those outside the picture. Each boundary point is
evaluated separately from adjacent open intervals. Beyond the first and last
change the classifier is constant, so those outer intervals certify any
infinite tails. The finite chain of spaces and maps determines the barcode,
preserving singleton intervals and endpoint inclusion. Coefficients use the
encoding's field; `RealField` retains its usual numerical rank semantics.
Reversed grid axes are excluded.
General polyhedral classifiers require rational line and viewing-box coordinates;
irrational algebraic inputs are rejected explicitly.

Box classifiers, general polyhedral classifiers, and the nearest-lattice
extension used for integer drawings can certify a whole line when all its
strata are represented. A positively oriented grid usually leaves parameters
below its first thresholds unrepresented. Such a global request is rejected;
it does not extend the unknown part by zero. Use a fully represented window
for that grid. The same rejection applies to an incomplete polyhedral
partition. A line missing the box has an empty window restriction, but can
still have nonzero intervals in global mode.

Computing a slice can materialize a lazy module and perform quadratic rank
work in the number of event strata. `slice_limit=512` bounds that count before
the rank calculation; use a smaller box for window mode or explicitly raise
the limit if needed. A smaller picture does not reduce global event work.
Interval selection reuses the current result, and revisiting a cached line
reuses its restriction. This is a count limit, not a time or memory guarantee.

```julia
OA.select_inspection!(slice_session; interval=0)  # Clear only the interval selection.
OP.save_visual("selected-slice.svg", OA.inspection_snapshot(slice_session);
               backend=:cairomakie)
OA.select_inspection!(slice_session; slice=false)  # Hide the slice, retaining the stalk/map.
OA.close_inspection!(slice_session)
```

The saved snapshot retains its decorated barcode and diagram without live
callbacks. These interval groups do not track classes between changing lines,
and a finite encoding alone need not retain source-cycle representatives.

## Inspect an existing barcode and its members

Sometimes the question begins after the restriction has already been
computed: which interval is this, and what does its multiplicity mean? The
same endpoint conventions and linked selection apply to ordinary persistence
diagrams, raw interval dictionaries or vectors, packed barcodes, selected
`SliceBarcodesResult` and `ProjectedBarcodesResult` entries, and
`FiberedSliceResult` restrictions. Pass `index` when a family has more than
one barcode. A raw barcode supplies its own endpoints; displaying an explicit
infinity does not add a proof about an unrecorded source computation.

```julia
import WGLMakie

bars = Dict((0,2) => 2, (1,Inf) => 1, (8,9) => 1)
interval_session = OP.inspection_session(bars;
    window=(-1,4), max_intervals=200)
OP.visualize(interval_session; backend=:wglmakie)
OA.select_inspection!(interval_session; interval=1)
OA.inspection_summary(interval_session)
```

This input has three distinct interval groups and total multiplicity four.
The displayed window intersects two groups. The finite interval `[8,9)` is
still available in the exact selector; the interval `[1,Inf)` has a labelled
infinity lane. The two copies of `[0,2)` share one bar and a multiplicity label.
They carry no source-cell information because none was supplied.

Click either chart or use **Selected interval group**. Both charts and the
readout share the same group ID. Coincident points cycle through their groups
on repeated clicks. `max_intervals` limits displayed groups, retaining the
full group list and counts. Selecting an omitted group brings it into the
display if it intersects the window; selecting an offscreen group gives its
exact readout. This operation reads retained intervals and does not recompute
persistence. Ordinary diagrams use `dim` to choose homological degree and
respect sublevel or superlevel orientation. In superlevel order, classes
continue toward smaller parameter values and essential intervals end at
`-Inf`.

An ordinary computation can retain more than endpoints: it can keep a cycle
representing each original interval. This must be requested when computing
the diagram, because endpoints alone cannot recover the cycle. Return to the
ring from [ordinary persistence](ordinary_persistence.md):

```julia
values = zeros(Int, 3, 3)
values[2,2] = 5
diagram = OP.cubical_persistence(values; representatives=true)
cycle_session = OP.inspection_session(diagram; dim=1, window=(-1,6))
OP.visualize(cycle_session; backend=:wglmakie)
OA.select_inspection!(cycle_session; interval=1, representative=true)

cycle = OA.persistence_representative(diagram; dim=1, kind=:finite, index=1)
cycle.cycle.cell_ids
cycle.bounding_chain.cell_ids
```

The hole has interval `[0,5)`. Its retained cycle is nonzero in homology at
birth and remains nonzero before five. At five it becomes the boundary of the
returned two-dimensional bounding chain. This example uses the default `F2`.
With another prime field, the readout names that field and retains its actual
coefficients. It shows literal source-cell IDs, dimension-local indices, grades and coefficients.
It shows at most twelve cells per chain with displayed/total counts; the
accessor and snapshot metadata retain the full chains. These are deterministic
choices made by the reduction, rather than canonical or geometrically
optimized cycles. Cell IDs alone do not assert an embedding in a point cloud
or image.

If several original intervals have the same decorated endpoints, choose
**Original member number** and press **Select member** before checking
**Show retained representative**. Equivalently, use
`OA.select_inspection!(cycle_session; interval=i, member=j, representative=true)`
for an existing group `i` and its original member `j`. Different members can
have different cycles even though their endpoints agree. A group with a
single original member selects that member automatically. Changing the group
or member clears the representative opt-in unless explicitly requested again.
An ordinary diagram computed without retention reports the representative
as unavailable; raw and slice barcodes without source correspondence do the
same. Neither case invents a cycle from the plotted bar.

The snapshot, reset, linked-view and close operations work for these
interval sessions. Export `OA.inspection_snapshot(cycle_session)` to keep the
selection and its literal readout. Ordinary representative retention does not
by itself supply source-cell correspondences for arbitrary multiparameter slices.

## Rendering and saving

Use the same `VisualStyle` for an interactive view and its saved figure. A
style changes how the picture is drawn; its spaces, matrices, region labels,
interval endpoints and selected query stay the same. Create a fresh session
and select a map in the square:

```julia
import WGLMakie
import CairoMakie

session = OP.inspection_session(enc; box=([-1,-1], [3,3]))
OA.select_inspection!(session; parameter_pair=((0,0), (1,1)))
style = OP.VisualStyle(fontsize=18, linewidth_scale=1.2)
OP.visualize(session; backend=:wglmakie, style)
snapshot = OA.inspection_snapshot(session)
OP.save_visual("selected-map.svg", snapshot; backend=:cairomakie, style)

print_style = OP.VisualStyle(palette=:grayscale, fontsize=18)
OP.save_visual("selected-map-print.pdf", snapshot;
               backend=:cairomakie, style=print_style)
```

The default accessible palette uses consistent source and target colors across
the parameter plane, finite poset and matrix headings. Their labels and marker
shapes also identify their roles. A source stays a source when its color changes:
`OP.VisualStyle(colors=(source=:darkblue, target=:darkorange))` changes the
appearance without changing the meaning. Region colors repeat for large posets;
the actual region IDs identify the fibers. Included, excluded and viewing-window
edges retain their solid, dashed and dotted distinctions in grayscale.

`fontsize` sets the base text size; headings and existing text sizes scale with
it. `font` and `mono_font` select the main and coefficient fonts. Browsers use
those fonts when installed and otherwise fall back to sans-serif and monospace.
`gap` and `padding` control spacing in pixels. `linewidth_scale` and
`markersize_scale` multiply visual strokes and markers; a marker whose size
represents a ball in parameter coordinates keeps that mathematical size.
Matrices retain literal field coefficients, including fractions and empty
matrix shapes. Browser matrices can scroll when their contents exceed the panel.

Numerical heatmaps keep their recipe's colormap unless `colormap` is supplied;
the grayscale palette defaults to `:grays`. Missing cells have a separate
appearance and never become zero-valued cells. A style does not change the
numeric range or synchronize scales between different plots.

Style settings apply to one render or export, without changing global Makie
themes. If you supply an existing `figure`, its identity and size are retained;
the style updates its background, outer padding and panel spacing as well as
the newly drawn content. Layer-specific literal colors remain literal with
the accessible palette; grayscale rendering converts them to luminance. Semantic role overrides
in `colors` take precedence over the palette, and an explicit `colormap` takes
precedence over its numerical-map default. Reuse the style when exporting a
snapshot: the mathematical specification does not remember a viewer's style.
`save_visuals(...; style)` supplies a batch default; a request's own `style`
overrides it. Unknown settings and invalid sizes are rejected. Pass `style` to
`visualize`, `render`, or an export call, rather than to `visual_spec`.

```julia
import CairoMakie
OP.visualize(spec; size=(1500, 650))
OP.save_visual("square.svg", spec; size=(1500, 650))
```

CairoMakie produces static figures. WGLMakie produces a browser scene; volume
slice sliders require a live Julia session. Exporting `:slice_viewer` to HTML
raises an error because offline Julia callbacks are unavailable; use `:image`
to export the selected slice instead. The linked inspector also requires live
Julia; export its `inspection_snapshot` to keep a static selection. Static query
labels are drawn annotations: changing `point`, `vertex`, or a pair in `visual_spec`
builds a new view, whereas an inspection session updates its linked panels.
Renderer controls are `figure`, `size`, and `style`;
recipe options belong to specification construction. See
[optional integrations](optional_integrations.md) for installation and export.

The [notebook](tutorials/inspect_encoding.ipynb) constructs the square, checks
selected spaces and maps, explains the two-square presentation through active
blocks and image bases, and exports PNG and SVG figures through the public
API. An optional section provides commands for exploring the same example in
a live inspection session, with exact point and map selections and a static
snapshot. The published lesson displays the saved static figures without
requiring a live Julia session.


## Inspect a distance witness

`visualize(witness)` accepts the result of `bottleneck_matching(a, b)` and shows
its actual pair assignments, aligned barcodes and pair costs. A live
`inspection_session(a, b)` links those views; ordinary diagrams require an
explicit `dim`. Repeated intervals retain separate member IDs and essential
endpoints retain their infinite costs. Start with the
[distance-witness recipe](tutorials/distance_witness.ipynb) for a small checked
calculation. The [finite-window matching guide](exact_matching.md#retain-and-inspect-the-witness)
owns comparison of slices, sampled cost maps and exact optimizer witnesses.
