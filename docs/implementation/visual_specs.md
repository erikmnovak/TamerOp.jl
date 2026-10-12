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

## Reuse source preparation before drawing

A second view of the same point cloud should not need a second neighbor graph.
The preparation owner shared with sparse ingestion computes Euclidean kNN or
closed-radius edges in the full coordinate space, using ingestion's
floating-point metric. An activated exact spatial
tree can accelerate that computation; the fallback uses ingestion's bounded
nearest-neighbor selection rather than sorting every distance row. Rendering
receives the resulting segments and never performs this search.

For example, selecting axes `(2,1)` swaps the drawing coordinates. It leaves
the graph computed from all source coordinates unchanged. Selecting fewer
vertices then keeps only edges with both endpoints displayed. These operations
occur after preparation: the original edge index and endpoints remain attached
to each surviving segment. Supplied graph polylines follow the same edge IDs.
Weighted edges retain the full graph's bin limits, so an omitted extreme edge
cannot silently change every displayed color.

An explicit `SessionCache` gives ingestion and previews the same geometry
bucket. Neighbor entries have structural coordinate snapshots, graph settings
and the resolved backend as their keys. Comparing snapshots detects in-place
coordinate changes and avoids treating a hash collision as identity. The bucket
retains four graphs, evicting the oldest entry before inserting another.
Projection preparation has its own four-entry bound. A lock covers lookup and
construction, so concurrent readers share one prepared result. Source mutation
must happen outside those calls. Display arrays are copied out; editing returned
metadata does not alter a later graph query.

Content checks still require a linear pass over coordinates. Retained graphs
cost space proportional to their edges and neighbor-distance data, and projection
storage grows with the source coordinates. This reuse removes repeated neighbor
construction, not the cost of inspecting the input or selecting displayed edges. When every
source vertex is displayed, a computed neighbor graph already certifies its
endpoints: its eligible count is known, and selecting the first bounded edge
IDs does not scan the full graph. Supplied graphs still validate their endpoints.
`cache=:auto` on a preview retains no data after the call; reuse across requests
is explicit. Rendering a previously returned specification is cheaper again
because it needs no source validation or graph preparation.

Preparation and display bounds have different meanings. Before brute force,
the requested distance-pair bound is checked. kNN also checks its directed
candidate bound before allocation. Radius construction stops with an error
before appending an edge beyond its output cap, both in the fallback and spatial
tree routes. A failed construction is not cached as a partial graph. Display
budgets subsequently select source indices and report omissions without changing
the mathematical input. Point-density bins still count the full point cloud.

The shared [graph preparation](../../src/data_ingestion/preview_graphs.jl) owns
neighbor reuse; [source preparation](../../src/visualization/source_previews.jl)
owns projection reuse and display identities. The
[usage section](../visualization.md#previewing-source-data) develops the workflow,
and the [reference](../src/reference/visualization.md#source-previews-and-preparation-budgets)
owns exact option contracts.

## Keep collection identities attached to their numbers

A collection arrives as feature, distance or kernel tables with one declared
sample order. Preparation first checks the sample/cell budgets, then shapes,
labels and value contracts, and copies the small supplied tables. It performs
no pairwise algorithm. FeatureSet inputs keep their original metadata; feature
definitions are attached to columns so selecting a number can reveal the
mathematical query it answers. Missing values and positive infinite distances
have distinct retained values even though both need a masked heatmap cell.

The heatmap uses rows for samples and columns for features or target samples.
Its floating drawing matrix is separate from the original selected value.
Pair tables are checked for exact symmetry; a nonnegative zero-diagonal table
still carries no triangle-inequality certificate, and symmetry alone does not
certify a positive semidefinite kernel. A supplied embedding retains its method,
settings and seed in sample order. Its screen geometry does not alter the input
matrix or establish that its distances are preserved.

Selection rebuilds a bounded specification from retained data. The same sample
index highlights an embedding point and reveals its supplied invariant/source
panels. Composite sample views are flattened into their prepared leaf panels;
no new invariant or source reconstruction runs. Session updates validate first,
prepare the next snapshot, and only then commit and notify listeners. Failed
updates preserve the previous selection. Closing releases collection inputs and
witness references while leaving an exportable final snapshot.

A pair witness is addressed by ordered sample IDs because reversing a matching
requires reversing its assignments. Opening a retained witness creates an
independent matching inspector. Computing a new one is an explicit selected-pair
operation delegated to the existing matching owner; it never fills or corrects
the supplied matrix. The browser owns the lifetime of an embedded pair view,
while a caller-created matching session remains independent. These boundaries
keep a click on a collection from starting hidden all-pairs work.

The [collection builder](../../src/visualization/builders_collections.jl) owns
value semantics and preparation; the [session](../../src/visualization/inspection_collections.jl)
owns selection and lifetime. The [guide](../tutorials/comparisons.ipynb) develops
the example, and the [reference](../src/reference/visualization.md#collections-of-features-distances-and-kernels)
spells out shapes, budgets and matching options.

## Align queries before subtracting observables

A comparison receives named results and a declared alignment of their queries.
Finite vertex lists are supplied separately for every result. Original parameter
points instead pass through each encoding's classifier. This avoids a common
false identification: a vertex number is local to its finite description.
Dimensions are prepared first, without materializing any structure maps.

A rank request fixes one source probe and evaluates its row. For encoded modules,
preparation checks comparability in the original parameter order as well as the
finite poset. Two incomparable parameters can land in one coarse fiber; their
equal labels do not create a forward map. Unrepresented points and unordered
pairs remain distinct masked values through subtraction and rendering. The live
session retains one row per result, so selecting another target or baseline
reuses the computation. A changed anchor replaces that bounded retained row.
For a lazy encoding, the first nontrivial map calls `encoding_module`, which
materializes the finite representation. The viewer bounds rank-table work,
not the owner's module-construction cost.

Ordinary diagrams supply another implementation of the same observable contract.
Dimension counts intervals containing a parameter, and rank counts intervals
containing both ordered endpoints. These sums use all grouped records with their
multiplicities and endpoint conventions, even when the drawing budget omits
some groups. A cutoff-surviving bar supplies no information beyond the recorded
query window. The drawing clips that record to known coverage and marks a censored endpoint;
the stored interval remains separate from this graphical restriction.

The difference is the case's integer value minus the baseline's value. It is
meaningful as a numerical comparison without claiming that the spaces share a
basis, that fields are canonically interchangeable, or that bars are matched.
Selected matrices therefore remain separate literal panels. For floating fields,
the owner rank routine is reevaluated at half and double the declared tolerances.
This modest sensitivity diagnostic is printed with its limitations; it is not
an exactness certificate. Authored construction and approximation descriptions
remain separate from provenance recorded by the computational owner.

## Compose semantic views before resolving their positions

Composition takes already prepared leaf views. Each layer and panel keeps a
name, its original metadata and its plotting recipe. An overlay compares the
supplied coordinate context and display windows before concatenating drawing
layers. Rendering delegates each child's layers back to its own drawing path,
so interval endpoint decorations, source labels and scalar ranges retain their
meaning. A shared legend deduplicates equal encodings and rejects a repeated
label with a conflicting encoding.

The context consists of domain and provenance identifiers, units, parameter
orientations, field and degree. Known object metadata constrains the caller's
declaration. Equal identifiers express a declared correspondence, not a
recomputed theorem; an intentional comparison records its mismatches and prints
its explanation. Independent panels need no common domain. Explicitly linked
axes do, while linked scalar colors additionally require the same quantity and
colormap. Preparation scans finite samples for a common range without altering
the stored values.

An annotation stores a named endpoint selector rather than a screen position.
A source-cell selector resolves through retained source geometry; an interval
selector resolves through original member IDs, even when equal bars share a
row. A matrix selector refers to the actual coefficient label. These selectors
are checked against the displayed subset: omitted data do not receive invented
positions. An authored coordinate or callout remains visibly distinct from a
computed correspondence.

The ordinary cycle connector makes a narrower, checkable claim. It requeries
the selected interval representative and checks its field, source identity,
parameter and coefficient vector against the supplied cycle. This computation
uses the source-lifting contract described in the
[source-representative account](source_representatives.md). It does not solve a
new representative optimization or identify a canonical geometric hole.
Composition and render validation repeat this check, so mutated mathematical
storage cannot silently retain a stale solid arrow.

At render time, a data endpoint is projected through its axis into the figure's
pixel coordinates. Panel and figure endpoints use normalized coordinates, and
matrix endpoints use the coefficient label's measured box. Reactive viewport
and camera-projection dependencies update the endpoints after layout, pan, zoom
or resize; observing limits alone can read the preceding camera transformation.
A transparent child scene, appended after the panel scenes, draws annotations
above their backgrounds in WGL as well as Cairo. Text is drawn over a separate
light halo so small signed weights remain legible against a scalar background.
The curved path is a bounded quadratic arc; brackets and highlights
are constructed from the same resolved endpoints. An endpoint outside the
visible data window hides the annotation. It is never clamped to an unrelated
boundary feature. These choices use Makie's
[coordinate spaces](https://docs.makie.org/stable/reference/generic/space.html)
and [layout observables](https://docs.makie.org/stable/explanations/layouting.html)
as rendering mechanisms, not as mathematical evidence.

Static HTML follows an explicit alternate export route. Cairo renders the
complete selected composition to SVG; the export embeds that vector scene and
its textual interpretation in one responsive HTML file. A labelled scroll region
preserves the SVG's intrinsic size on narrow screens, so scientific labels remain
readable and panels and connectors retain one coordinate frame. No runtime,
saved Julia closure or remote resource is needed. The exported picture is fixed; live WGL axis interaction
is a separate route, and mathematical serialization remains with the IO owner.

Preparation is linear in the retained layer and annotation counts, apart from
requested source-certificate checks and scalar scans. Rendering has one reactive
endpoint per anchor and a fixed-size sampled arc per arrow. No all-pairs source
search, decomposition or new homology computation is introduced by layout.

## Prepare signed contributions before drawing

A signed rectangle measure stores endpoint pairs and integer weights. To inspect
a rank query, preparation compares both endpoints against the query in their
exact coordinate types, then accumulates positive and negative contributions
using arbitrary-precision integers. A cumulative point measure instead tests
membership in the query's lower orthant. A declared interval-family summary
uses inclusion of finite label sets. Sharing a plotting layer does not make
these inversion domains interchangeable.

The query sum is formed before selecting terms for drawing. Consequently a
small display budget or clipped window cannot change the reconstructed value.
Input IDs and full endpoint pairs remain attached to the records; a selected
term can replace the final budgeted entry without losing its identity. Infinite
deaths remain infinite in those records, while the drawing marks their clipped
positions explicitly. A supplied Hilbert/Euler background must declare its
source and share the input's axes and units before its sampled layer is composed
with signed glyphs.

Rectangle smoothing subtracts a nonnegative vector from the upper endpoint and
tests whether the resulting rectangle is empty under its actual boundary
convention. The selected sum then uses the original pair query with its target
shifted by that vector. Signed prominence is the sign times the remaining side
lengths. These constructions follow
[Botnan, Oppermann and Oudot, §6.2](https://link.springer.com/article/10.1007/s10208-024-09672-9);
they do not perform density convolution or introduce an arbitrary importance score.

One optional callback evaluates the original observable independently. Its
answer and the reconstructed answer are retained separately, including when
they disagree. Term and view changes in a live session reuse that answer;
changed query or smoothing requests a fresh one. Updates build a candidate
snapshot before committing it. Failed validation preserves the old state.
The contribution scan visits each retained term once and stores one record per
term; coordinate precision and finite-label containment determine the cost of
each test. Input validation also checks the supplied axes or interval family.
Drawing is bounded separately. Preparation requires neither a full rank tensor
nor a new finite encoding. Inversion algorithms belong to the signed-measure
owner; the renderer only receives the resulting semantics.

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

### Batch drawing without changing the question

A layer can contain thousands of edges while sharing one color, line width and
meaning. The renderer stores their endpoints in a single segment plot. Each
pair remains independent: batching does not join the end of one edge to the
start of another. A barcode similarly expands the supplied multiplicities into
their original rows, then draws those strokes together. The canonical decorated
barcode batches colored strokes, foreground endpoint markers and endpoint labels
separately. Open marker centers and selected marks retain their order within the
marker buffer. Diagram markers likewise share a scatter plot, while annotations
retain their measured layout and exact interval identities. Connected paths retain
their internal joins; separators between paths prevent spurious connecting edges.
Closure edges remain explicit.

Diagram annotations are arranged after the axis limits and tick orientation are
established. Arranging them against temporary limits would repeat the collision
layout when those limits change. Subsequent resizing still updates their measured
positions, with their mathematical anchors unchanged.
The renderer measures each string's actual glyph rectangle, then places labels
in axis-local pixels. A bounded, deterministic search prefers positions that
avoid earlier labels and data markers. Existing offsets are reused when they
remain clear after a viewport change. The two coordinate bounds are applied
independently, including at corners. Greedy placement can leave overlapping
labels or crossing leaders in dense views, even when a more expensive layout
could separate them. Every displayed interval keeps its label.

One text plot holds all labels and one segment plot holds their leaders. Leaders
end at the measured label rectangle and leave a gap around the data marker.
Neither layout nor leader clipping changes the interval's data coordinates.
Projection and viewport changes update these two plots directly, without a
second hidden annotation scene or a connector subplot for each label. Glyph
and coordinate buffers still grow with the number of displayed labels.


Planar polygons use Makie's vector polygon recipe when their outlines can be
drawn together. Outlined polygons with overlapping bounding boxes retain the
supplied painting order, because drawing every fill before every outline could
expose an earlier boundary. This conservative choice can leave some concave,
disjoint polygons unbatched. Spatial faces are triangulated in the plane of
greatest projected area and collected into one mesh with separate vertices per
face. A vertical face is therefore not lost by projection onto the horizontal
plane, and adjacent faces do not acquire artificially smoothed normals.

Text is batched with a position, alignment and pixel offset for each label.
Halo text retains label order, including when labels overlap. WGL draws its
native outside stroke with a uniform width; Cairo interleaves a background
stroke and foreground glyph for each string. WGL does not serialize a vector
of text stroke widths. An exact coefficient table uses
one text plot and one layout anchor. Measured glyph extents determine the maximum
width of each column and height of each row. Matrix entries keep the monospace
font and literal strings such as `1/3`; row and column labels keep their basis
roles. A composition connector follows the selected coefficient's position as
the layout resizes, without requiring a separate plot or layout block for it.

These mechanisms use the dependency's [segment](https://docs.makie.org/stable/reference/plots/linesegments.html),
[polygon](https://docs.makie.org/stable/reference/plots/poly.html) and
[text](https://docs.makie.org/stable/reference/plots/text.html) recipes. Batching
reduces scene objects and their reactive bookkeeping. The coordinate and glyph
buffers still grow with the displayed data; triangulation, font layout and the
exported geometry still have to be computed. The renderer adds no persistent
cross-figure geometry cache. The caller owns a retained specification or figure,
and closing a linked inspector releases its session resources.

Display budgets act before rendering. For example, `matrix_limit` selects a
displayed block while retaining the complete inspected matrix, and
`max_intervals` limits visible interval groups while retaining their identities
and exact endpoints. Reducing scene objects never silently removes entries or
changes a mathematical query. A prepared specification draws all its supplied
layers; it is not an instruction to recompute a cheaper mathematical answer.

The source boundaries are [specification types](../../src/visualization/types.jl),
[request/specification validation](../../src/visualization/validation.jl),
[resolution preparation](../../src/visualization/builders_resolutions.jl),
[render/export dispatch](../../src/visualization/rendering.jl), and
[shared Makie rendering](../../ext/visualization_makie_common.jl).
The [executable guide](../tutorials/resolutions.ipynb) develops the mathematical
example; the [reference](../src/reference/visualization.md) owns keyword contracts.

## Coordinate sections and shared scales

A `ScalarField` has one array dimension per declared coordinate, followed by
an optional layer dimension. A plane `(3,1)` varies coordinate 3 horizontally
and coordinate 1 vertically. The builder indexes all other coordinates at the
shared selected point, then writes the result with rows following the vertical
coordinate and columns following the horizontal coordinate. The common renderer
transposes that matrix exactly once for Makie's coordinate-first convention.
Asymmetric grids therefore preserve their meaning under an exchange of axes.

A comparison accepts fields only when coordinates, labels, units and ordered
layer identities agree. That input declares a pointwise correspondence; it does
not establish a module isomorphism. A baseline difference is formed only where
both samples are finite. The color range is scanned over all supplied samples
and layers, then reused across selections and planes. Signed data require a
symmetric range around zero and a diverging colormap. A manual range follows the
same signed contract. Field IDs and layer labels remain in each panel, while the
fixed-coordinate values and selected value are visible without browser controls.

Contours are opt-in levels of a particular interpolant: each grid rectangle is
split from southwest to northeast into affine triangles. A half-open edge rule
finds each level crossing; unavailable triangles contribute no segments. These
curves describe interpolated scalar values, not certified support boundaries.
A triangle constant at the requested level has no crossing curve to draw.
Contour colors and line styles are derived from their numeric levels, so a
level retains its appearance when other levels are added or removed.
Panel axes scale independently to keep strongly unequal coordinate ranges
readable; their ticks retain the declared coordinates.
The segment layers reuse the normal renderer and retain their numeric levels.
No three-dimensional surface or isosurface is inferred from a non-coordinate
finite poset. Such a poset continues to use its graph view.

Preparation costs one global scan for the shared scale and work proportional
to the selected plane samples for each redraw, multiplied by the number of
requested contour levels. A session caches the scale unless the baseline or
normalization changes. It limits the total panels to twelve and charges contour
levels against the displayed-cell budget. Input copying is explicit; no rank,
barcode or generalized-rank computation occurs during a field selection.
The [landscape adapter account](feature_transforms.md) explains how a sampled
line family obtains valid ambient coordinates before entering this pipeline.

See the [bibliography](references.md).
