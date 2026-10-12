# Visualization reference

`visualize(object; kind=:auto, ...)` builds a mathematical specification and
renders it through an explicitly loaded backend. `visual_spec(object; ...)`
returns that inspectable specification without rendering. The latter and its
accessors are available through `TamerOp.Advanced`. Use `available_visuals` to
list the supported kinds for the actual object, and `check_visual_request`
to inspect its accepted keyword and cost contract.

Selection, capture, closing and reopening are specified in the
[live and saved session reference](visual_sessions.md). A portable saved view
retains named answers and their exact available data; new mathematical queries
require a supported source and Julia.

## Source previews and preparation budgets

Point clouds support `:points_2d`, `:points_3d`, `:point_density`,
`:knn_graph` and `:radius_graph` when their coordinate dimension permits it.
`GraphData` displays its supplied edges; `EmbeddedPlanarGraph2D` preserves the
supplied polyline for each edge. Its `max_path_points=100_000` limits the
total vertices in selected polylines before copying them; it raises an error
instead of simplifying their geometry. A `FilteredComplex` displays actual cells at
its selected parameter, with the separate `max_cells` contract below.

| Option | Default | Meaning |
| :--- | :--- | :--- |
| `max_points` | `5000` | Maximum displayed source vertices, spread deterministically through original index order |
| `max_edges` | `10000` | Maximum displayed edges whose endpoints are both displayed; original edge order breaks ties |
| `max_labels` | `50` | Maximum labels among displayed vertices, in source order |
| `nn_backend` | `:auto` | For metric graphs, activated exact NearestNeighbors or brute force; explicit `:nearestneighbors` and `:bruteforce` are accepted |
| `max_distance_pairs` | `10_000_000` | Maximum brute-force distance evaluations allowed before graph construction |
| `max_graph_edges` | `1_000_000` | Radius-output cap; for kNN, cap on the `n * min(k,n-1)` directed candidate upper bound |

All budgets are nonnegative `Int`s. A preparation limit raises an error; a
display limit reports omissions. Point-only views do not accept edge or graph
preparation options. Three-dimensional views do not accept labels.
`k` is a positive `Int`; it is capped at the number of available neighbors.
`radius` is finite and nonnegative, with a closed distance boundary; zero
connects coincident points. Its omitted default is half the greatest distance
from the first point. Distances use all source coordinates, independently of
`dims`; projection changes only the drawing. Search uses ingestion's `Float64`
Euclidean metric: “exact” describes the neighbor search, not exact arithmetic
on arbitrary source coordinates. Nearest-neighbor ties follow the
selected ingestion backend's tie choice, so tied edges need not agree across
backends. No approximate backend is chosen by a preview. Drawing coordinates
use `Float64`; a loss from higher-precision input coordinates sets
`drawing_rounding=true` and appears in the caption. Source IDs remain unchanged.

`visual_metadata(spec)` retains `point_ids`, `edge_ids`, `label_ids`,
`source_edges`, totals, displayed counts and omissions. A metric preview is
labelled `construction=:reconstructed_metric_preview`; a supplied graph is
`:supplied_graph`. The former does not identify the graph actually used by an
arbitrary filtration. Weight-bin limits use the full supplied graph, and axis
limits use the full source coordinates. A point-density heatmap counts all
source points even when its point overlay is limited. Codensity ball snapshots
accept `max_points`, retain original displayed/filtered IDs and report the full
retained population separately.

`cache=sc::SessionCache` shares up to four prepared neighbor graphs and four
coordinate projections. Compatible sparse Rips-family ingestion (including its
flag expansion) and preview calls use the same graph preparation. Reuse requires
matching coordinate contents, graph kind, parameter, resolved backend and
approximation settings. A radius cutoff applied after kNN construction, a
collapse, or a different filtration can change the actual construction; use its
retained complex when that distinction matters.

In-place coordinate edits cause a cache miss. Changing axes, labels or display
budgets can reuse the graph. `graph_reused` and `geometry_reused` report reuse;
`cache=:auto` or `cache=nothing` keeps no preparation between preview calls.
A cached graph still has to fit `max_graph_edges`; distance work already done
is not charged again. Do not mutate a source concurrently with preparation.
Dropping the session releases its retained preparation when no other owner holds
it. Rendering an existing specification performs no new neighbor computation.

See [previewing source data](../../visualization.md#previewing-source-data)
for the workflow and [preparation mechanisms](../../implementation/visual_specs.md#reuse-source-preparation-before-drawing)
for cache identity, work and storage costs.

## Geometric constructions and multicover

`filtered_complex(points, filtration; cache=:auto)` retains the actual geometric
model for `RhomboidFiltration`, `CoreFiltration` and `CoreDelaunayFiltration`.
An explicit `SessionCache` reuses compatible construction data; `cache=nothing`
disables that reuse. Native rhomboids retain exact lifted vertices, signed
cellular boundaries, inside/on labels and constrained squared radii. Bounded
depth windows include horizontal terminal caps. The subdivision–Čech backend
retains labelled subset flags and a schematic barycentre/cardinality layout,
including distinct labels at coincident coordinates. A flag layout's third
coordinate is subset cardinality, not a capped coverage grade; subsets above
the upper depth cap remain necessary intersections. Core constructions retain
their original simplices and numerical grades. A subset of `k_values` defines
a step filtration that agrees with the full core construction on the selected
depth slices; an unlisted depth does not recompute its neighbour threshold. Sampling or reduced models do
not gain a geometric comparison guarantee through this adapter.

The [multicover guide](../../tutorials/multicover.ipynb) develops these choices;
the [implementation account](../../implementation/rhomboid_construction.md)
explains how retained identities relate to the boundary matrices.

```julia
cells = filtered_complex(PointCloud([0 0; 2 0]),
    RhomboidFiltration(depth_range=(1,1)))
spec = TamerOp.Advanced.visual_spec(cells;
    kind=:geometric_filtration, parameter=(1,1), invariant=:barcode)
visualize(spec)
```

`kind=:geometric_filtration` is an explicit view; the default remains
`:filtered_complex`. The geometric view links constructed cells, sampled
source membership, appearance events in the radius–depth plane and selected
homology. `spec.metadata.betti` contains dimensions for all chain degrees of
the active retained model; `selected_betti` selects `degree`. These are computed
from boundary ranks over `field`, never from the source raster.

| Keyword | Contract |
| --- | --- |
| `parameter` | `(physical_radius, integer_depth)`, radius nonnegative, inside the inclusive radius cap and closed depth window. Default radius zero and depth one, clamped to the window. Fractional depth and nonfinite coordinates are rejected. |
| `degree=0`, `field=CoreModules.F2()` | Degree of the selected dimension/barcode and coefficient field. Dimensions accept the supported exact or numerical fields; barcodes require a prime field. A degree above the constructed dimension has zero homology in that model. |
| `invariant=:dimensions` | Compute active Betti dimensions. `:none` skips algebra; `:barcode` additionally reduces the fixed-depth radius filtration. |
| `max_algebra_cells=10000` | Maximum active cells for dimensions; maximum constructed cells for the conservative barcode preflight. Exceeding it is an error. |
| `max_matrix_entries=1_000_000` | Sum of active adjacent boundary matrix areas for ranks; conservative square of total cell count for barcode reduction. These are preflight bounds, not a process-memory guarantee. |
| `source_dims=(1,2)` | Two source spatial coordinates. Source dimension must be at least two. Higher-dimensional data use a spatial section through `source_anchor`; they do not display projected balls as sections. |
| `source_anchor=nothing` | Full rational-coordinate source tuple fixing omitted coordinates; finite floating values convert to their exact binary rationals. Default is the site centroid. Selected coordinates vary across the panel. Site markers are projected; sphere outlines and membership use the declared section. |
| `sample_resolution=25` | Integer from 2 through 128 on each source axis. Exact membership is evaluated at those finite drawing sample points; the raster is an illustration, not a topology certificate. |
| `compare=nothing` | Optional second retained construction with identical indexed sites, family and geometric coordinate representation. Core parameters `beta` and `k_values` must match. Both windows must contain the parameter. Schematic flag overlays are rejected. |

The cell controls `cell`, `dimensions`, `activity`, `boundary`, `view_dims`,
`max_cells`, `labels`, `max_labels`, `alpha` and `field_axis` retain the
[filtered-complex contracts](#filtered-complexes-and-source-cells).
`view_dims` concerns **cell geometry**, while `source_dims` concerns source
space. A planar native rhomboid has three lifted coordinates; larger ambient
models require an explicit choice of two or three cell coordinates. A projected
drawing does not slice a cell or identify coincident vertices. Selection also
highlights the involved source labels. Higher cells remain in algebra when
their two-skeleton is displayed.

`max_cells` bounds the cell and event drawings, not the complex used in algebra;
exact event records remain in the parameter panel metadata. A construction
`max_dim` really truncates the complex and is reported beside the resulting
homology. With a radius cutoff, surviving barcode endpoints are drawn as closed,
censored endpoints. They do not certify survival beyond the constructed window.
The returned `diagram` belongs to the truncated finite complex; its unpaired
columns must be interpreted with the retained geometric window metadata.

Comparison metadata records its hypotheses. Complete native rhomboid models
on identical indexed sites describe the same multicover on their common
window. Other permitted overlays are labelled `:geometric_comparison_only`;
they assert neither a cell correspondence nor an equivalence or approximation
bound. The source panel and computed invariant always belong to the primary
object. Approximate geometric models require their own guarantees and are not
accepted as certified multicover overlays.

Three `Advanced` accessors provide the underlying queries:

- `geometric_data(cells)` returns indexed sites, representation, executed
  backend, axes, radius/depth window, grade precision and construction dimension
  limit. Retained arrays are read-only while consumers use them. It rejects
  complexes without construction-owned geometric provenance.
- `coverage_at(cells, point; parameter)` returns `count`, `covered_sites`,
  `eligible_sites`, `contributing_sites`, `member` and exact `radius_squared`.
  Points have one finite rational coordinate per source dimension; floating
  coordinates convert to their exact binary rationals. The radius can be an
  exact algebraic value. Multicover counts
  every indexed closed ball. Core membership uses the retained vertex grades;
  core–Delaunay additionally restricts contributions to full-cloud nearest-site
  Voronoi cells, including ties. Numerical core grades remain numerical despite
  exact arithmetic in this membership query.
- `radius_slice(cells; depth)` returns a one-parameter `GradedComplex` in
  physical radius, preserving dimension-local cell IDs and signed boundaries.
  Each multicritical cell receives the least radius among its occurrences at
  depth at least the request. Cells with no allowed occurrence are removed.
  The original radius cap and construction skeleton still limit interpretation.
  Empty windows return a typed empty complex. No homology or grid snapping occurs.

`inspection_session(cells; kind=:geometric_filtration, ...)` accepts the same
options and returns a `ComplexInspectionSession`. Its existing parameter/cell
controls update every panel. Browser text inputs accept integers, fractions and
decimals; supply exact algebraic radii through `select_inspection!` in Julia. Algebra and the source raster are rebuilt only
when a selection is applied; hover reads retained cell records. Invalid updates
leave the previous state and snapshot intact. Close the session with
`Advanced.close_inspection!`; its final static snapshot remains available.

## Collections of features, distances and kernels

`visual_collection` prepares supplied tables in a common sample order. The
[comparison guide](../../tutorials/comparisons.ipynb) works through interval
features, bottleneck distances and a supplied embedding. No feature extraction,
normalization, clustering, pairwise distances or dimensionality reduction runs
when the collection is constructed or selected.

```julia
collection = visual_collection(features; definitions)
visualize(collection)
```

Here `features` is a `Featurizers.FeatureSet`; its IDs, column names and mathematical interchange metadata
are retained without its runtime cache or reference-bank objects. `definitions` supplies one mathematical description per column.
For plain matrices use `visual_collection(; sample_ids, features, feature_names,
definitions, ...)`. Sample IDs and feature names are nonempty unique strings.
Rows always follow `sample_ids`; neither sorting nor clustering is implicit.

| Input | Contract |
|---|---|
| `features` | Nonempty sample-by-feature real matrix; `missing` is allowed. Each feature has a name and definition. |
| `distances`, `distance_semantics` | Sample-by-sample, exactly symmetric, nonnegative, zero diagonal. Positive infinity and symmetric missing entries are allowed. Supply a nonempty description of the distance, units and conventions. Triangle inequalities are not certified. |
| `kernel`, `kernel_semantics` | Exactly symmetric sample-by-sample real matrix. Negative values and symmetric missing entries are allowed. The viewer does not certify positive semidefiniteness or convert similarities to distances. |
| `normalization="not declared; viewer applies none"` | Describes a transform already applied, including fitted parameters or their source; it never changes the input. |
| `invariant_settings=(;)`, `degree=nothing`, `field="not declared"` | Declared mathematical context. The FeatureSet overload stores its metadata under `invariant_settings.feature_set`; additional authored settings can accompany it. The overload reads a recorded `normalization` when present unless explicitly overridden. Per-sample metadata can record heterogeneous choices. |
| `embedding` | Optional `(coordinates=Y, method="...", settings=(;...), seed=...)`, with finite n-by-2 coordinates in sample order. Seed is an integer or `nothing`. The display need not preserve input distances. |
| `samples` | Optional vector of NamedTuples in sample order. Accepted fields are `object`, static `invariant` and `source` specifications, and `metadata=(;)`. Sample selection shows its supplied views without computing them. |
| `witnesses=Dict()` | Existing matching witnesses keyed by **ordered** pairs of distinct sample-ID strings. A reverse assignment is not inferred. |
| `max_samples=200`, `max_cells=40_000` | Reject excess samples or total supplied table cells before copying/querying. Subset explicitly; budgets never silently truncate. |

At least one table is required. `NaN`, booleans, negative distances, nonsymmetric
pair tables and finite values outside the Float64 display range are rejected.
Only distance tables may contain positive infinity. Exact original table values
remain in specification metadata; heatmaps use Float64 coordinates and values.
Small tables label missing/infinite cells; in larger tables their masks and
hover readout preserve that distinction. Tick labels may be thinned for legibility;
selection indices and IDs still refer to every supplied row and column.

`visualize(collection; view=:features, sample=1, target=2, feature=1)` selects a
supplied table, sample and feature/pair. The default view is the first available
of features, distances and kernel; target defaults to the second sample or the
only sample. With no feature table, feature selection is `nothing`. Sample and
feature selections accept their string IDs/names or one-based indices. Color
ranges are computed within the selected table; they do not standardize columns
or promise comparability between heterogeneous features.

`describe(collection)` records the semantic declarations and non-certification
of metric, kernel and embedding claims. `collection_sample(collection; sample)`
returns `(id, material)`, with the retained sample object, views and context.
`visual_spec(collection).metadata.selected` contains the original selected
value, feature definition and sample/target metadata. Sample views and objects
are caller-supplied associations, not certified derivations of the table. Treat
all retained inputs as read-only and rebuild after mutation.

`inspection_session(collection; ...)` creates an Advanced
`CollectionInspectionSession`. `select_inspection!` accepts `view`, `sample`,
`target` and `feature`. A matrix click selects its row and column; an embedding
click selects a sample while retaining the target. Coincident embedding points
can be selected separately through the indexed controls. Hover reads prepared
values without changing selection. Invalid updates retain the previous state.
Independent viewers share the session's selection; closing one browser view
releases that view's callbacks. Closing the session releases its collection and
listeners while retaining the last static `inspection_snapshot`.

`collection_matching(session; compute=false, kwargs...)` opens an independent
matching inspector for the ordered selected pair. The collection overload takes
`sample` and `target` explicitly. By default it requires a retained witness.
With `compute=true`, it invokes A85 only on those two retained `object`s and
forwards mathematical options: `dim` for ordinary diagrams, or finite-window
`opts` for compatible encoded pairs. Missing objects or unsupported pairs raise
an error. No table entry is overwritten. A matching may have different semantics
from a supplied distance or kernel, so inspect its context before comparing
numbers. The returned inspector has its own lifecycle.

The browser's **Open supplied matching** control uses retained witnesses only;
it closes that embedded child when the parent selection changes or the view
closes. Separately created matching sessions remain independent. Static exports
use `inspection_snapshot(session)` and do not embed live callbacks.

## Comparisons through declared probes

`ComparisonCase(object; degree=nothing, filtration="not declared",
construction="not declared", approximation="not declared", units="parameter")`
accepts a finite `PModule`, `EncodingResult`, or ordinary `PersistenceDiagram`.
The field and recorded provenance are read from the result; authored descriptions
are retained separately. `degree` selects an ordinary homology degree (default
0), or documents a module's degree without contradicting recorded provenance.
No coefficient conversion, basis identification or approximation certification
is performed. The [comparison guide](../../tutorials/comparisons.ipynb) develops
field, basis, degree and discretization examples.

`compare_visuals(cases::NamedTuple; correspondence, vertices=nothing,
points=nothing, parameters=nothing, window=nothing, max_queries=4096,
max_intervals=200)` returns a `VisualComparison`. Supply two to eight cases and
exactly one probe declaration:

| Declaration | Meaning |
|---|---|
| `vertices=(left=[...], right=[...])` | Explicitly aligned finite vertex labels, with exactly the case names and order. A single label list is rejected. |
| `points=[(x,y), ...]` | Shared original coordinates, located independently in supported planar encodings. A named tuple supplies case-specific coordinates for an explicitly described mapping. |
| `parameters=[t,...]` | Shared original-unit real parameters for ordinary diagrams. |

Probe collections are finite vectors or tuples with equal nonzero lengths.
The case-by-probe budget is checked before materializing them. Shared parameters
require equal declared units; the viewer does not convert units. The required
`correspondence` is a nonempty explanation, recorded as `:declared_not_certified`.
Inputs and prepared storage must remain read-only; reconstruct after mutation.

`visualize(comparison)` / `visual_spec(comparison)` use `kind=:comparison`.
Their mathematical options are:

| Option | Contract |
|---|---|
| `view=:hilbert` | Prepared stalk dimensions or interval-membership counts at the probes; no structure-map queries. |
| `view=:rank` | Ranks from `source=1` to every probe, per case; no all-pairs table. |
| `view=:barcode` | Ordinary intervals with a shared finite display window, plus dimension differences at the probes. |
| `source=1`, `probe=1` | Aligned query indices, not finite vertex IDs. The probe is highlighted in every panel. |
| `baseline=first(keys(cases))` | Subtract that case's integer observable from each case. |
| `matrix=false`, `matrix_limit=(8,8)` | Explicitly request selected module maps in their stored bases; crop their display while retaining full matrices in metadata. Available only for module rank views. |

A mask distinguishes unavailable queries from an unordered pair. Neither becomes
zero. For ambient point pairs, both the original parameter order and finite
order must permit a map; a coarse classifier cannot manufacture comparability.
Signed differences are differences of integers, never of matrices or bars.
Each value column shares one range; each difference column shares a symmetric
range about zero. The horizontal strip axis is the probe index, with no spatial
interpolation. The readout retains exact selected parameters and authored and
recorded construction, field, degree, orientation, window and approximation data.

For `RealField`, rank uses the field's `atol` and `rtol`. The readout also reports
ranks with both tolerances multiplied by 1/2 and 2. This is a sensitivity check,
not a condition estimate, error bound, or exact-rank certificate. Stalk dimensions
may themselves depend on earlier numerical choices. No cross-characteristic
invariance is assumed.

Ordinary intervals preserve multiplicity and sublevel/superlevel endpoint
conventions. The common display `window` defaults to the finite interval endpoints
and probes (expanded if degenerate); it is distinct from query coverage. Diagrams
restricted by a recorded radius/threshold cutoff are unknown outside that
coverage. Surviving bars stop at the coverage boundary with a censored `?`;
their original records remain in panel metadata as `stored_records`.
`max_intervals` limits drawn groups without changing sums over all
records. Neutral bar colors and local member IDs imply no class correspondence;
use a matching-witness view for computed assignments.

`inspection_session(comparison; view=:hilbert, source=1, probe=1, baseline=...,
matrix=false, matrix_limit=(8,8))` creates an Advanced
`ComparisonInspectionSession`. `select_inspection!` accepts only `view`, `source`,
`probe`, `baseline` and `matrix`; leaving rank mode clears matrix inspection.
Invalid updates leave both selection and retained row unchanged. One anchor row
per case is cached; changing the target or baseline reuses it. The first
nontrivial map of a lazy encoding materializes its finite module through
`encoding_module`; this owner cost is separate from the bounded rank row. A live WGL view
links strip selections, axis navigation and the barcode query marker. Hover
reads retained values. Closing releases cases, row caches and listeners while
keeping the final static snapshot. These APIs describe bounded observations;
a zero difference does not certify module isomorphism or an approximation bound.

## Resolutions and presentation incidence

These overloads accept `DerivedFunctors.ProjectiveResolution`,
`DerivedFunctors.InjectiveResolution`,
`IndicatorResolutions.UpsetResolutionResult`,
`IndicatorResolutions.DownsetResolutionResult`, and the workflow
`Results.ResolutionResult` wrapping either derived resolution. They consume
stored terms, not just a bare matrix of multiplicities, and never construct or
extend a resolution. The [resolution guide](../../tutorials/resolutions.ipynb)
works through a diamond example with known Betti and Bass tables.

| Kind | Objects and meaning | Keywords |
|---|---|---|
| `:betti_table` (projective default) | Stored principal-upset counts, degree rows and finite-vertex columns | `verify=false`, `matrix_limit=(12,12)` |
| `:bass_table` (injective default) | Stored principal-downset counts, same table convention | `verify=false`, `matrix_limit=(12,12)` |
| `:resolution` | Table, selected support, differential incidence and optional stalk | `degree=0`, `summand=nothing`, `vertex=nothing`, `grades=nothing`, `verify=false`, `matrix_limit=(12,12)`, `support_sheets=false`, `basis_change=nothing` |
| `:betti_degrees`, `:bass_degrees` | Multiplicities at supplied planar grades in one stored degree | `grades` required, `degree=0`, `verify=false` |
| `:resolution_lift` | Supplied projective-resolution lift and its equations | Explicit `kind=:resolution_lift`; see the [map guide](../../visualization.md#maps-between-modules) |

Degree means homological degree for projectives and cohomological degree for
injectives. Degree zero selects the (co)augmentation when `vertex` is given.
A positive degree `k` selects `P_k → P_(k-1)` or `I^(k-1) → I^k` respectively.
`summand` is a one-based index in the selected term's stored summand order;
it highlights the projective source column or injective target row. An empty
term has no selectable summand. `vertex` is a finite-poset ID, not an ambient
parameter. Its active rows and columns retain their global summand IDs.
If `matrix_limit` crops the coefficient panel, the selected row or column is
retained within that limit. Metadata records the displayed global indices.

A default table reports **unverified multiplicities**. `verify=true` checks
stored principal bases, agreement of differential coefficients with stored
maps, augmentation naturality, augmented equations, and vertexwise exactness
through the computed prefix. It tests each cover/hull for minimality and the
terminal kernel/cokernel for completion. A valid nonminimal resolution is
viewable and labelled `:nonminimal`; malformed or inexact prefixes are rejected.
A nonzero terminal kernel/cokernel gives `:truncated`. Missing higher degrees
are never padded with zero rows. Exact fields use exact coefficients and rank;
`RealField` uses its declared equality tolerances and numerical-rank backend.
Verification is fresh and does not trust cached minimality summaries.

`grades` supplies one finite real coordinate pair per finite vertex. Its
coordinatewise order must agree with the finite order. No coordinates are
inferred from a Hasse diagram or an encoding classifier's representative points.
Exact supplied grades remain in metadata; plotting converts them to floating
coordinates. Distinct grades that collide in those display coordinates are
rejected; use the exact table instead. A grade plot makes no claim about an ambient multigraded free
resolution. A multiplicity label counts coincident summands at a vertex;
the combined resolution view highlights the selected summand's vertex.

`support_sheets=true` shows separately labelled principal supports in the
selected term. `matrix_limit[1]` bounds the number of these panels; an explicitly
selected summand is also retained. Panel separation is schematic and does not
supply an extra parameter or a decomposition of the resolved module.

### Cokernel presentations and kernel copresentations

`IndicatorTypes.UpsetPresentation` and `DownsetCopresentation` support
`:presentation_incidence` as their default. They accept the resolution selection
keywords except `verify`. Degree 0 selects generators/cogenerators; degree 1
selects relations/corelations. Both show the same single presentation map.
Supports must be principal on the supplied finite poset. A projective
presentation is displayed as a **cokernel**; the downset copresentation is a
**kernel**. Neither is the fringe **image** shown by `:presentation_inspector`.
No minimality or subsequent resolution is inferred.

All displayed coefficient matrices use target rows and source columns. This
transposes the stored `UpsetPresentation.delta` convention. Labels identify the
actual summands and, when supplied, their grades. A dagger marks an
order-forced zero; an ordinary zero is allowed by order. At a selected vertex,
the active matrix determines the reported cokernel/kernel dimension.

### Supplied graded basis changes

On a selected differential or presentation, use
`basis_change=(; source=S, target=T)`. The square matrices must use the same
coefficient field and stored source/target summand counts. Their coefficients
must obey the principal-support order; both must be invertible. The viewer
computes `B = inverse(T)*A*S` and checks `A*S = T*B`. It retains both matrices,
the changes and their inverses, and the equation result in metadata.

This certifies the selected map in different graded coordinates, not a
minimalization or decomposition algorithm. It does not transform adjacent maps
or mutate the original object. The augmentation's target is not generally a
principal sum, so resolution basis changes require a positive selected degree.

### Specification data and costs

`visual_metadata(spec)` retains counts, stored degrees, computed length, category,
field, direction, verification status, selection and supplied grades. Incidence
views additionally retain `coefficient_matrix`, `stalk_matrix`, `incidence`,
`basis_change`, and displayed support-sheet indices/counts. A presentation stalk
also supplies `represented_dimension`. Matrix panels retain the full selected
matrix, its original shape and displayed row/column ranges. These arrays are
copied; editing them does not change the input resolution.

Default tables count generators without elimination. A selected injective
coefficient matrix is recovered from its recorded stalk maps. Explicit
verification traverses the whole stored prefix and performs field-aware ranks;
a basis change performs inversions and equation checks. Matrix display limits
do not limit those mathematical checks or discard the exact selected matrix.
Rendering receives only prepared data and performs no resolution algebra.
These views use argument-based selections with CairoMakie or WGLMakie; they do
not supply live resolution inspector controls.

## Signed observables and cancellation

The [slices and signed-contributions guide](../../tutorials/slices_diagrams.ipynb)
uses one finite encoding to compare a diagonal slice, pair ranks, Hilbert values
and declared-family generalized ranks.

| Input | Views | Reconstruction query and rule |
|---|---|---|
| `RectSignedBarcode{2}` | `:signed_cancellation`, `:signed_diagonals`, `:signed_prominence`; existing `:rectangles` remains the default | `query=(p=(x,y), q=(x,y))`; sum weights with `lower ≤ p ≤ q ≤ upper` on the declared grid |
| `PointSignedMeasure{4}` | The three signed views; cancellation is the default | Same pair query; four coordinates mean `(birth_x,birth_y,death_x,death_y)`. Default `boundary=:halfopen` uses `birth ≤ p ≤ q < death` |
| `PointSignedMeasure{2}` | `:signed_cancellation`; existing `:signed_atoms` remains the default | `query=(x,y)`; inclusive coordinatewise lower-orthant sum. Declare `observable=:hilbert`, `:euler` or the default `:grid_function` |
| `IntervalRankSummary` | `:signed_cancellation` | `query=(vertices=[...],)`; sum coefficients of containing intervals in the declared family. Undeclared queries are rejected |

Each signed view links geometry or finite-label membership, exact contributions
and a readout. Rank views also show signed prominence. `:signed_diagonals` changes
the rectangle glyphs; `:signed_prominence` places prominence first. Positive and
negative terms have distinct semantic colors and marker shapes. For a
`SignedMeasureDecomposition`, first select its intended component with the
owner's semantic accessor; one combined container can hold different observables.

### Boundaries, checks and smoothing

`RectSignedBarcode` has inclusive integer coordinates. When constructed from a
`GridEncodingMap`, these may be grid indices rather than physical parameter
values: consult its stored axes. The workflow `rank_signed_measure(encoding)`
provides physical half-open birth/death endpoints. A hand-supplied four-coordinate
measure may explicitly use `boundary=:closed`; this changes its mathematics.
Births must be finite, deaths finite or positive infinity, and rectangles
nonempty. Cumulative point atoms require finite coordinates. All these inspectors
require integer weights and preserve them as `BigInt` in sums.

`smoothing=(εx,εy)` defaults to zero and applies only to rank rectangles. Both
entries must be nonnegative. It replaces the upper endpoint by `upper−ε` and
removes empty rectangles according to the boundary convention. A pair query then
reconstructs the original rank at `(p,q+ε)`. For inclusive grid barcodes, `p` and
`q+ε` must belong to the declared axes. No continuous module extension is inferred
from that finite grid. Point-atom convolution and signed transport are separate
computations, not meanings of this option.

`direct` is optional. It receives `(p,q+ε)` for a rank query, the point for a
cumulative query, or a copy of the finite-label vector for a family query. It
must return an integer. The view reports `agreement=:equal`, `:mismatch` or
`:not_checked` beside the independently retained reconstructed and direct values.
A mismatch remains inspectable. Equality at one query does not certify the whole
measure, its source provenance, an unsampled domain or a module decomposition.
Request preflight does not execute `direct`; constructing the specification does.

### Selection, display limits and backgrounds

`term` selects a one-based input term ID; `nothing` or `0` clears it.
`max_terms=40` bounds displayed terms, retaining a selected term within the budget.
Every retained coefficient still enters the exact sum. IDs preserve input order
and duplicate terms; they do not identify corresponding bars across different
modules. The finite-family table shows at most the first 40 labels and records
its displayed rows and columns. Exact full label sets remain in metadata.

`window=((xmin,ymin),(xmax,ymax))` clips only the geometric drawing. Infinite
endpoints remain infinite in metadata and receive explicit labels in the figure.
`units=("parameter 1","parameter 2")` names the parameter coordinates; prominence
uses the same units for widths. Geometry converts to floating display coordinates;
exact endpoints and contributions remain in the specification.

An optional background is a named tuple with `axes`, `values`, `observable`,
`source` and `units`. Its observable is `:hilbert`, `:euler` or `:rank_diagonal`;
`source` names the supplied mathematical object. Axes and units must agree with
the signed input. `values[j,i]` corresponds to `(axes[1][i],axes[2][j])`; values
must be finite and at most 10,000 samples are accepted. Colored sample points
show the background without assuming interpolation between grid coordinates.
This explicit shared-grid composition does not establish a correspondence
between unrelated encodings.
Finite-label interval summaries cannot accept a parameter-plane background.

Building a signed specification scans the retained terms and allocates their
contribution records. It does not enumerate all rank queries or reconstruct a
module. The optional direct query has its own owner's cost. Rendering and export
consume the prepared specification without repeating that query.

### Live signed inspection

`inspection_session(measure; ...)` accepts the signed-cancellation keywords above
and returns `SignedInspectionSession`. `select_inspection!` changes `term`,
`query`, `smoothing` or `view`; `reset_inspection!` restores the initial query and
smoothing and clears the term. Invalid updates leave the state and snapshot
unchanged. A direct answer is retained while query and smoothing remain fixed,
so term selection and view changes do not call the mathematical owner again.
Treat retained inputs, callback behavior and snapshot data as read-only.

With WGLMakie, `visualize(session)` opens query fields, smoothing controls and
linked term picking. Clicking coincident targets cycles their input IDs. Integer,
fraction and decimal coordinate input is parsed without evaluating Julia code.
The controls require live Julia. `inspection_snapshot` returns the same static
specification for Cairo/WGL rendering or export. `close_inspection!` releases
inputs and listeners while retaining that last snapshot; a closed live widget
cannot be reopened. `inspection_summary` and `check_inspection_session` expose
query counts, agreement, state and lifecycle validation.

## Spectral pages and differentials

`ChainComplexes.SpectralSequence` and `DerivedFunctors.TorSpectralSequence`
support `:spectral_page` (the default) and `:spectral_differential`. They consume
computed sequences; the renderer does not construct a complex or derive page
algebra. The [complexes guide](../../tutorials/complexes.ipynb) follows a rank-one
second differential through its two next-page quotients.

| Keyword | Contract |
|---|---|
| `page=2` | Positive `Int`, or `:inf` for the owner's associated graded total result |
| `source=(p,q)` | Required only for `:spectral_differential`; a bidegree in the supplied and displayed windows |
| `compute=true` | Permit requested page/map computation; `false` uses stored results and marks missing data unknown |
| `window=nothing` | Optional `((pmin,pmax),(qmin,qmax))` integer display bounds inside the supplied rectangle |
| `truncated=false` | Declare with `true` that the supplied complex is itself a truncation of a larger intended input |
| `max_terms=400` | Maximum number of displayed grid sites, including zero/unknown sites; exceeding it is an error |
| `matrix_limit=(12,12)` | Maximum displayed rows and columns, retaining complete selected matrices in metadata |
| `structure_maps=()` | Supplied labelled matrices between terms of the selected page |

The default E2 dimensions are already stored by the sequence owner. Requesting
an uncached later page with `compute=true` can perform elimination and populate
its cache. Selecting a differential can materialize the owner's whole page of
maps, then computes the selected kernel/image and next-page contributions.
`compute=false` never requests missing owner pages or differentials; selected
local kernel/image calculations still occur when all their inputs are retained.
Call `visual_spec` to prepare this work separately from rendering.

Every displayed bidegree is labelled: a dot and integer denote a positive
term dimension, `0` denotes a computed zero, and `?` denotes an uncomputed term.
The display window changes no differential or page calculation. A selected
differential whose target is offscreen retains its matrix and target quotient
in the readout; no shortened arrow pretends to end at the crop boundary.

Vertical-first cohomological differentials have displacement `(r,1-r)`;
horizontal-first ones have `(1-r,r)`. Tor reverses both bidegrees, so the
corresponding homological displacements are `(-r,r-1)` and `(r-1,-r)`.
All matrices use target rows and source columns in the owner's chosen page
coordinates. Kernel columns lie in the source term; image columns lie in the
target. At each selected endpoint the viewer checks consecutive composition
and compares `dim ker(outgoing) - rank(incoming)` with the next page. The
quotient representatives it retains are not canonical identifications with
the next page's independently chosen basis.

The finite-filtration bound and equality with the owner's computed limiting
dimensions concern only the supplied complex. Neither the last cached page nor
a single zero differential certifies convergence. `truncated=true` does not
extend such a statement to a larger, missing complex. The limiting page supplies
associated graded pieces; these views neither choose a canonical splitting nor
solve multiplicative extension problems. A finite-page differential cannot be
selected on `page=:inf`.

A structure record has the form
`(source=(p,q), target=(u,v), matrix=A, label="supplied u")`.
The endpoints must be distinct terms of a computed page. `A` must have the
coefficient element type and target-by-source dimensions of those terms;
numerical entries must be finite. Dashed arrows and their labels report the
actual supplied map and rank; zero-rank maps receive labels without a nonzero
arrow. Metadata retains the matrices. Shape and rank checks do not certify
multiplication, a Leibniz rule, compatibility with other pages or an extension.

The metadata records `dimensions` (or `nothing` for an unknown page),
`page_status`, `differential_status`, `full_window`, `display_cropped`,
`input_truncated`, `convention`, `first`, `convergence_bound`,
`limiting_dimensions`, and `extensions=:not_resolved`. A selected differential
adds its matrix, kernel, image, rank, incoming matrix, local quotient
representatives and both endpoints' next dimensions under `selected`.
Matrices/dimensions in a specification are snapshots, not a mutation interface
for the source sequence. Numerical fields retain their declared tolerances;
exact-field equations and ranks use that field's arithmetic.

### Live page navigation

`inspection_session(ss; page=2, compute=true, window=nothing, truncated=false,
max_terms=400, matrix_limit=(12,12))` returns a `SpectralInspectionSession`.
Render it with WGLMakie while Julia is running. The page controls, bidegree
inputs and actual chart clicks all use the same `select_inspection!` methods:

```julia
OA.select_inspection!(session; page=2, source=(0,1))
OA.select_inspection!(session; page=3, compute=false)
OA.select_inspection!(session; compute=true)
OA.select_inspection!(session; source=nothing)
```

Here `OA = TamerOp.Advanced`. A page change clears the previous source unless a
new one is supplied. Hover reads retained dimensions only. Invalid updates leave
the prior selection and snapshot intact, although an owner computation that
preceded an error may have populated its own cache. Treat retained source
sequences as read-only while viewers use them. Use independent calls to
`visualize(session)` for linked viewers; an individual widget has one browser
client. `reset_inspection!` restores the starting choices and
`close_inspection!` releases input references and callbacks. The last
`inspection_snapshot` remains available for static export. Live canvases are
owned by the inspector; compose static snapshots into a supplied figure.

## Filtered complexes and source cells

`filtered_complex` retains the finite cells, their grades and geometric source
identities so a picture can answer what is present at a chosen parameter.
The [source-cell guide](../../tutorials/source_cells.ipynb) develops the workflow
with static figures. Construction does not compute homology or representatives.

```julia
cells = filtered_complex([0 0 0; 0 5 0; 0 0 0])
visualize(cells; parameter=0)
visualize(cells; parameter=5, cell=45)
```

The array form uses the same ordinary cubical construction as
`cubical_persistence`. Both `input=:top_cells` and `input=:vertices` support
any positive array dimension and preserve exact input values. One-dimensional
geometry is placed on a line in the plane; higher dimensions retain their
coordinates for explicit projection. `construction=ConstructionOptions(...)`
passes construction budgets through the same cubical entrypoints.
`order=:sublevel` or `:superlevel` sets the activity direction. `periodic` accepts
a Boolean or one Boolean per spatial axis. In the top-cell model, faces inherit
the minimum incident value for sublevels and maximum for superlevels. In the
vertex model, cells inherit the maximum for sublevels and minimum for superlevels.

`filtered_complex(data, filtration)` constructs supported point-cloud simplicial,
embedded graph and vertex-valued cubical image models through the ingestion
owner. These retain the numerical grade contract of that ingestion route;
the `ImageNd` route uses Float64 grades. Construction metadata records the
requested and effective family, grade scale, sparsification and collapse choices.
For vertex-valued images, `CubicalFiltration(orientation=(-1,))` constructs
superlevel stars. Image filtrations accept one `+1` or `-1` per channel;
mixed directions take maxima or minima independently in those channels.
A collapsed model is labelled reduced; source vertex identities do not supply
a chain lift through that collapse. Specialized cell models requiring their own
geometry adapters can use the explicit cellular form below.

For a supplied simplex tree, use `filtered_complex(tree; coordinates=matrix)`;
rows are source vertex coordinates. For a supplied graded or multicritical
cellular complex, use `filtered_complex(G; geometry=vertices_by_cell,
orientation=(1,1))`. Geometry has one vertex list per global cell in
dimension-major order: a point, an ordered edge/polyline, a cyclic planar polygon,
or vertices retained for a higher cell. `source_ids` can supply explicit source
identities; its default is `(dimension, dimension_local_cell_id)`. Higher cells
remain in the complex when a skeleton is drawn. Supplied geometry describes
attaching cells; integer boundary and grade checks do not certify an embedding.

For a known cubical enumeration, `cubical_shape` supplies vertex-grid counts
instead of `geometry`; cell counts and absolute incidence matrices must match
the cubical model. Periodic cells are drawn in an unwrapped fundamental domain:
opposite boundary faces are identified, and a wrap edge is not a long interior
segment. A cancelled boundary incidence is still zero in the algebra.

The `Advanced` accessors `graded_complex`, `grade_orientation`, `cell_geometry`
and `cell_source` expose the retained data. `describe` gives construction and
size information. `check_filtered_complex(x; throw=false)` checks finite
geometry/grades, identities, matrix shapes, integer boundary-square zero and
boundary filtration compatibility, including multicritical unions. Wrap this
report with `Advanced.data_validation_summary` for a compact REPL summary. Treat the
retained arrays as read-only while an inspector is using them.

| Visualization option | Meaning and default |
| --- | --- |
| `parameter` | Exact grade tuple, or a scalar for one parameter. Defaults to the first corner in the declared orientations. A cell is active when any appearance grade precedes this point; comparison occurs before Float64 drawing conversion. |
| `cell=nothing` | Global cell index to inspect. Distinct from a dimension-local ID, source-image index or persistence interval ID. |
| `dimensions=0:2` | Which dimensions to draw. Always a display filter; active higher cells remain counted. |
| `activity=:active` | Draw active cells. `:all` adds faint inactive context without changing membership. |
| `boundary=true` | Highlight the selected cell's incident boundary cells and retain their signed integer coefficients. This is not a selected persistence cycle. |
| `view_dims=nothing` | Two or three source-coordinate axes. Required explicitly above ambient dimension three; projections retain original coordinates and do not change cell identities. |
| `max_cells=10000` | Maximum displayed cells. The selected cell and its boundary have priority when eligible. Activity counts and selected algebra use the full complex; omitted counts remain visible. |
| `labels=false`, `max_labels=50` | Bounded 2D index labels. 3D selection uses the pointer or index control and readout. |
| `alpha=0.35` | Opacity of active faces. |
| `field_axis=nothing` | Color samples by an actual source-grade coordinate. Top-cell array inputs show face-center samples; other default inputs show vertex samples. No interpolation, binning or inferred density estimator. Multicritical samples without a single scalar value are rejected. |

`visual_spec(cells)` returns `kind=:filtered_complex`. Metadata includes exact
parameter, full activity mask/counts, displayed indices/counts, omitted cells,
selected source identity, appearance grades, original geometric coordinates
and boundary coefficients. A cell outside the display dimensions or active
subset can still be inspected. Source grades and coordinates retain their input
values; rendering is approximate. Projection/drawing collisions are identified
where detected. Arbitrary polygon embedding, self-intersection and geometric
attaching-map correctness remain properties of the supplied geometry.

`inspection_session(cells; ...)` creates a `ComplexInspectionSession` for WGLMakie.
`select_inspection!` changes `parameter`, `cell`, `dimensions`, `activity` or
`boundary`; invalid edits preserve the last valid selection and snapshot.
Hover reads retained cells without recomputing topology. Separate calls to
`visualize(session)` make independent linked viewers. Disconnect disposes that
view's callbacks; `close_inspection!` closes the shared session and releases its
input. The last `inspection_snapshot` remains available for static export.
Static CairoMakie exports support PNG, SVG and PDF; live widgets need Julia.

Construction validates the supplied chain/filtration data and can be more
expensive than plotting. Static queries scan retained cells and their grade
sets; they do not run persistence. Geometry, source identities and grades consume
storage in addition to the abstract complex. Display budgets limit rendered
cells, not the mathematical object or its construction cost.

## Source-chain overlays

A retained `SourceChain` draws selected coefficients on its actual source cells:
`visualize(chain)` supports degrees zero through two, with exact source IDs,
field, parameter and signed boundary readout. `view_dims`, `max_cells`,
`max_labels` and `labels` control the drawing; metadata records omissions.
Periodic cells use an unwrapped fundamental domain. Rendering computes neither
representatives nor an optimized cycle. The ordinary interval inspector adds
available source cycles and death fillings only after selecting an original
member and explicitly requesting its representative.

See [source representatives](source_representatives.md) for retention,
verification, cocycle variance, scale and optimization contracts, and the
[source-cell guide](../../tutorials/source_cells.ipynb) for executed examples.

## Components and merge histories

`visualize(component_history(cells); scale)` links H₀ bars, actual source
components and the merge forest for one filtration or specified oriented slice.
`inspection_session(history; scale)` adds exact event navigation and linked bar,
source-vertex and merge-node selection. The [component reference](source_representatives.md#components-and-merge-histories)
owns parameter conventions, ties, source correspondence, display budgets and
lifecycle; the [source-cell guide](../../tutorials/source_cells.ipynb) develops the
workflow with two components that join across a bridge.

## Chain and cochain inspection

A complex carries more information than its homology dimensions. The chain
inspector exposes its actual incoming/outgoing matrices, a selected vector,
and the equations that identify a cycle or its quotient class. The exploratory
walkthrough is [Working with complexes and spectral pages](../../tutorials/complexes.ipynb).

```julia
visualize(C; degree=1)
visualize(C; degree=1, vector=z)
visualize(C; degree=1, compute=true)
session = inspection_session(C; degree=1)
```

`CochainComplex` and `ModuleCochainComplex` support `:chain_inspector`. A module
complex additionally takes `vertex` (default `1`). `CochainMap` supports the same
view directly; `ModuleCochainMap` retains its default `:chain_map` and accepts
`kind=:chain_inspector` for vector/class inspection. Its `homotopy=H` option
requires a supplied `ModuleCochainHomotopy` whose first map is the inspected map;
the checked identity is `f-g = d h + h d`. The existing `:homotopy_comparison`
and resolution-lift views provide whole-map comparisons.

`FilteredComplex` and `SourceChain` retain their geometry defaults and also
accept `kind=:chain_inspector`. For a filtered complex, specify `parameter`,
`degree` and optionally `field` (default `QQField()`). Activity uses the exact
filtration comparisons, including multicritical unions and mixed orientations.
Only active cells enter the matrices. Coordinates index the active cells in
source order, not all cells of the unfiltered object. Full active index lists
remain in the specification metadata. `convention=:homological` is the default:
its differential lowers the displayed cell degree. `:cohomological` transposes
the active boundaries and increases degree. A `SourceChain` fixes its retained
field, parameter, degree and chain/cochain convention, and supplies its own
vector when no replacement is requested. Verified source geometry is reused;
vector-space and module-stalk inputs do not acquire geometry from their labels.
Source overlays use exact rational or prime coefficients in cell degrees zero
through two. Higher-degree and numerical-field queries retain the algebraic
view. Inspecting a supplied filtered complex constructs its active finite chain
matrices; no chain complex or forgotten source correspondence is inferred from
an unrelated module encoding.

| Option | Meaning |
| :--- | :--- |
| `degree` | Selected stored degree; defaults to the first degree, or the retained source chain's degree. |
| `vector=nothing` | No vector query. Supplying coefficients requests their cycle and quotient analysis. |
| `coordinates=:ambient` | Coefficients in the selected chain/cochain group. `:class` instead reconstructs a cycle from quotient coordinates. |
| `compute=false` | No basis query by default. `true` computes and displays cycle, boundary and quotient bases. Vector selection computes the quotient data it needs regardless of this flag. |
| `matrix_limit=(12,12)` | Maximum displayed rows/columns; exact full matrices remain in metadata. |
| `max_terms=12` | Bound the degree overview around the selection without truncating its complex. At least three terms must be allowed. |
| `view_dims`, `max_cells=10000` | Projection and geometry budgets for retained source overlays; these do not alter algebra. |

An ambient vector with nonzero differential is displayed as a non-cycle and has
no quotient coordinates. For a cycle `z`, the record contains chosen class
coordinates, a representative `r` and an explicit preimage `w` satisfying
`z = r + d(w)`. A zero class is identified as a boundary, including the zero
vector. Coordinates and witnesses are noncanonical and have no minimum-support
guarantee. Empty spaces retain their zero-dimensional shapes. Tables state the
coefficient field, and `RealField` inputs retain their tolerance policy; their
membership/equality results are numerical checks.

The selected chain map is checked before its quotient map is formed. The image
vector and its target class use actual component matrices. Module-valued
queries use **local stalk quotient bases**, which need not coincide with the
bases of a separately constructed cohomology module. A supplied homotopy gives
an actual boundary witness for equal image classes on cycles.

`inspection_session(cells; kind=:chain_inspector, ...)` selects the algebra
inspector rather than the filtered-geometry controls. `select_inspection!`
changes degree, vertex, parameter, vector, coordinate mode or basis computation.
`inspection_session(source_chain)` starts at its retained vector in the actual
source complex, so subsequent selections can explore that complex.
Changing a location or coordinate mode clears the previous vector unless a
replacement is supplied. Invalid updates preserve the previous snapshot.
Independent browser views share the selection, hover reads prepared data only,
and closing a viewer disposes its callbacks. Closing the session releases the
input and listeners while preserving its final static snapshot. Treat inputs
as read-only throughout a live session. Linked views require WGLMakie and live
Julia; export the snapshot for static use.

## Connecting-map inspection

A retained `LongExactSequence` or `SnakeLemmaResult` supports
`kind=:connecting_map`. These views consume actual supplied results; they do
not construct a cone or a snake diagram merely to draw it.

```julia
sequence = Advanced.long_exact_sequence(f; output=:full)
visualize(sequence; degree=-1, map=:delta, vector=[1])
visualize(snake; vertex=1, map=:delta)
```

For a cone sequence, `degree` lies in its stored range and `map` is `:f`, `:i`
or `:delta` (default). The diagram displays
`H^t(C) -> H^t(D) -> H^t(Cone) -> H^(t+1)(C)`. The cone convention is
`Cone(f)^t = D^t ⊕ C^(t+1)` with block differential `[d_D f; 0 -d_C]`;
`delta` comes from the projection to the shifted complex. Retained matrices are
checked against their inducing chain maps. Selected source quotient coordinates
are lifted to a cocycle, mapped, and compared with a target representative
through an explicit boundary adjustment. Consecutive-map and image/kernel
checks cover the displayed interior terms, without asserting missing endpoints
of a cropped sequence.

For a snake result, select `vertex` and one of `:k1`, `:k2`, `:delta`, `:c1`,
`:c2`. Vectors use the selected kernel/cokernel coordinates. The result retains
its six-term sequence, not the original two-row diagram; the view does not
invent missing diagram-chase lifts. `matrix_limit` bounds displayed tables.
Both result families support `inspection_session`, arrow selection, vector
queries, linked viewers and static snapshots. These results already contain
quotient data; constructing them is an explicit prior computation.

## Refinement and change of poset

Finite maps, joint encodings, common-refinement results and finite module
translations support static and linked comparison views. The
[change-of-poset reference](change_of_posets.md) owns source/witness/comparison
contracts, selection, matrix and graph budgets, and finite-category guarantees.
The [usage notebook](../../tutorials/compare_encodings.ipynb) follows the actual
regions, projections and commuting squares of two encodings of a square.

Selected-vector results use `kind=:vector_persistence`, with an optional target
label for actual map and image matrices. See the authoritative
[selected-vector contracts](modules.md#inspect-the-result) and the
[generated-submodule guide](../../tutorials/generated_submodules.ipynb).

## Derived groups and multiplication

Ext/Tor spaces and algebras support `:graded_dimensions` and
`:derived_class`; algebras add `:product_table`, and projective Ext models
add `:yoneda_product`. Hom spaces also accept `:graded_dimensions` while
retaining their default basis-map view. Homogeneous algebra elements select
their own degree and coordinates. The
[derived reference](derived.md#visualization-requests) owns their selection,
cost, budget and algebraic contracts; the
[derived guide](../../tutorials/derived.ipynb) develops the worked examples.

## Composing views and annotations

A composition keeps named panels and their mathematical metadata together.
Start with `visual_panel(object; ...)` to select one prepared view, then use
`overlay_visuals` for a shared coordinate plane or `compose_visuals` for a figure.
The [composition guide](../../tutorials/compose_figures.ipynb) shows the actual
results; the [focused recipe](../../recipes/compose_figure.md) begins with a scalar
table and ends with a signed overlay.

**Compatibility.** Overlays require a `VisualContext` for every input and equal
coordinate-domain identity, units, orientation, coefficient field, degree,
provenance identifier and explicit display window. Known field/degree metadata
is checked against declarations. Context identifiers are supplied declarations,
not automatic proofs of common origin. Incompatible inputs require a nonempty
`comparison` explanation, printed on the figure with the mismatches. This
performs no unit conversion, reparameterization or mathematical identification.

Independent panels can have different coordinate domains. `link_axes` selects
explicit groups with the same compatibility checks. `link_colors` additionally
requires exactly one scalar heatmap per panel, with the same quantity label
and colormap. Its common range spans all finite samples in the group. Groups
must be disjoint; combine overlapping groups into one. Named overlays preserve
their children's separate scalar ranges; link scalar panels before adding a
more specialized overlay. Shared legend labels must have identical encodings;
a conflict is an error, not a silent recoloring.

**Coordinates and identities.** `VisualAnchor` uses a named panel, a data point,
a normalized panel/figure point, or a retained object selector. Supported
selectors are `(:layer, layer, element)`, `(:overlay, layer_name, layer, element)`,
`(:matrix, row, column)`, `(:source_cell, index)`, and
`(:interval, degree, kind, index)`. Source-cell indices are local to the chain's
dimension. Matrix indices refer to the displayed coefficient block. Interval
indices retain the original ordinary member, including in a group of equal bars.
An object omitted by a display budget is an error. Points outside the current
live axis window hide the annotation; they are never clamped onto another feature.
`visual_panel(...; limits=...)` changes the prepared view's camera. It does not
recompute a recipe's clipped glyphs or recover omitted objects; request a larger
recipe window or display budget when those data are needed.
Data anchors require planar views. A 3D view may use a panel-space anchor, or an
explicit planar projection for a source-cell anchor.

**Claims.** `VisualAnnotation` creates authored arrows, callouts, brackets and
rectangular highlights. A nonempty interpretation label and dashed style identify
these explanations. `correspondence_annotation(diagram, cycle)` instead requeries
and checks a retained ordinary representative, including source object, field,
degree, scale and coefficient vector. It accepts the actual `source_cycle` from
`persistence_representative`; it does not accept an arbitrary homologous cycle as
that retained basis member. The source and interval panels must come from the
checked objects. Solid strokes identify this restricted computed relationship.
Other scientific relationships can be explained with authored annotations; no
unimplemented matching or decomposition is inferred from an arrow.

**Rendering and reopening.** CairoMakie writes PNG, SVG and PDF. Composed HTML
also requires CairoMakie: it embeds a responsive SVG and an accessible list of
annotation meanings and context. The file needs no Julia runtime or external
assets; it preserves the selected fixed view. WGLMakie renders a live composition
whose annotations follow axis limits and layout while Julia is connected. This
is different from an offline interactive inspector or a saved mathematical
artifact. Use `save_visual(dir, stem, figure; format=:html)` for the fixed HTML
and `visualize(figure; backend=:wglmakie)` for the live scene. Specifications and
retained mathematical inputs are read-only; construct a new spec when changing
the selected data.

```@docs
TamerOp.VisualContext
TamerOp.VisualAnchor
TamerOp.Visualization.VisualAnnotation
TamerOp.visual_panel
TamerOp.overlay_visuals
TamerOp.compose_visuals
TamerOp.correspondence_annotation
```

## Backend and frontend contract

The [display support matrix](../../visualization_support.md) distinguishes native
rendering, executed standalone-browser interaction and notebook embedding.
CairoMakie returns static figures. WGLMakie returns a Bonito application for
linked inspectors and `ImageNd` volume slice viewers; ordinary static
specifications return figures. Volume sliders select array indices on every
non-displayed dimension and independent renders do not share a selection.
Closing a viewer or disconnecting its browser releases its controls and retained
volume. Export a selected `:image` plane for a static volume figure.

Live Julia callbacks cannot be preserved by static WGL HTML export. Saved visual
sessions instead provide the explicitly captured, finite offline state set;
resuming computation requires a supported source and Julia.

## Generalized rank, boundary barcodes and GRIL

`visualize(M; kind=:generalized_rank, vertices)` highlights the selected
connected convex finite-label region and its exact rank. An `EncodingResult`
uses its finite module. `witnesses=true` adds compatible-section, quotient and
comparison matrices. `boundary=true` requires a positive 2D grid encoding and
adds its boundary path and barcode, with repeated labels kept as distinct
occurrences. `occurrence` and `bar` select retained boundary entries.

`visualize(enc; kind=:generalized_rank, center, width=0, length=1,
levels=(1,2,3), lengths=(1,2,3), level=1)` draws a continuous closed worm
and its exact generalized rank. Its GRIL panel compares exact suprema across
the declared rank levels at the selected length. The dashed line marks the
queried width. Grid lines are actual cell boundaries; labels show finite
label and stalk dimension. The geometric contract is a positive 2D grid with
half-open cells, zero below either first axis, and constant upper tails.
No radius mesh or boundary-zigzag substitution is used for this query.

Both accept `budget`, `max_vertices=120` and `matrix_limit=(8,8)`. Mathematical
queries precede conversion to floating drawing coordinates; exact domains
and answers remain in metadata. The finite diagram is schematic, while the
worm axes are original parameter coordinates.

`visualize(boundary_result; occurrence=0, bar=0)` uses
`kind=:boundary_zigzag`. `visualize(gril_result; center=1,
level=first(levels), length=first(lengths))` uses `kind=:gril`, displaying
retained probe centers and the selected center's layers. Here `center` is a
one-based result index, while `level` and `length` are actual requested values.
There is no interpolation between the supplied centers. Empty GRIL results
have no probe to display and are rejected.

For live queries, use [`generalized_rank_session`](visual_sessions.md#query-generalized-rank-regions).
The [region-query guide](../../tutorials/generalized_rank_queries.ipynb) compares
the finite and continuous interpretations; the
[algorithm account](../../implementation/generalized_rank.md) explains the
exact computations and their costs.


## Coordinate sections of scalar fields

`ScalarField(axes, values; id="field", axis_labels=nothing, units="value",
layers=nothing, provenance=(;))` adapts sampled real values on declared Cartesian
coordinates. Axes must be finite, strictly increasing, nonempty and distinct
in floating display coordinates. There must be at least two axes. Without
`layers`, array dimensions match the coordinate axes. With explicit distinct
layer IDs, the final array dimension follows their order. `NaN` denotes an
unavailable sample; infinity is rejected. Coordinates and scalar values are
stored as Float64 numerical samples. Retain the original encoding for exact
queries; this adapter is not an exact arithmetic representation. Inputs are copied.

`visualize(field; kind=:coordinate_sections, ...)` and
`visualize([field1, field2]; ...)` prepare static small multiples. Multiple
fields must have distinct IDs and identical coordinate axes, axis labels,
units and ordered layer IDs. The caller thereby declares their coordinate
correspondence; use `compare_visuals` for other supplied correspondences.

| Option | Meaning |
| --- | --- |
| `planes` | Ordered `(horizontal_axis, vertical_axis)` pairs. Default: `(1,2)` in two dimensions; `(1,2),(1,3),(2,3)` otherwise. At most 12 total panels. |
| `indices` | One stored sample index on every coordinate axis, defaulting to its middle sample. The other coordinates of each plane are fixed here. |
| `layer` | A retained layer ID, defaulting to the first. No summation or averaging is implicit. |
| `normalization` | `:auto`, `:sequential` or `:symmetric`. Signed values require a symmetric range; `:auto` chooses it for negative values or differences. |
| `colorrange` | Optional finite increasing pair; signed ranges must be symmetric about zero. Otherwise use all supplied samples and layers. |
| `contours` | Up to 12 distinct increasing finite levels of the affine SW–NE-triangulated interpolant. Omit unavailable triangles. |
| `baseline` | A supplied field ID for pointwise differences, or `nothing`. Missing values remain missing. |
| `max_cells` | Positive limit on displayed plane samples times contour levels, default 250,000. |

Every panel retains field and layer IDs, the ordered plane, actual fixed values,
selected value, availability count and common scale. [The features guide](../../tutorials/features.ipynb)
provides executable examples.

### Ambient landscape adapter

`ambient_landscape(L; axes, basepoints, source_window, direction_index=1,
sample_error=nothing, line_tolerance=1e-10, max_points=250_000)` converts a
selected stored `MPLandscape` direction into `Advanced.AmbientLandscapeGrid`.
There must be one actual parameter-zero basepoint per stored offset. The
positive stored direction defines `x=basepoint+t*direction`; it is not normalized
again. `source_window` is a finite increasing parameter pair or one pair per
offset. A query must also lie in the stored `tgrid`. There is no extrapolation
and no interpolation between lines. Duplicate or tolerance-overlapping lines
are rejected.

Heights are multiplied by the largest direction component to use ambient
infinity-norm units. Equal components yield the diagonal interpretation;
unequal components are explicitly directional. Retained slice weights do not
alter pointwise heights. `Advanced.landscape_values` gives the coordinate-first
array with layers last; `landscape_grid`, `landscape_layers`, `ambient_coverage`,
`ambient_errors`, `ambient_landscape_summary` and `describe` expose its domain
and interpretation. `check_ambient_landscape` validates hand-built results.

A supplied nonnegative `sample_error` is a uniform bound in original line-parameter
units. Conditional error bounds add the one-Lipschitz interpolation term before
rescaling; otherwise errors are unknown (`NaN`). Under geometric tolerance,
bounds concern the projected line point. No earlier approximation or confidence
bound is inferred. `ScalarField(grid; id, axis_labels)` adapts this result for
sections. The [implementation account](../../implementation/feature_transforms.md)
derives the parameter conversion and error term.

### Live field inspection

`field_session(field_or_fields; ...)` accepts the same section options and copies
the supplied fields. Use `Advanced.select_inspection!`, `inspection_selection`,
`inspection_summary`, `inspection_snapshot`, `reset_inspection!` and
`close_inspection!` as for other sessions. Invalid updates are transactional.
Changing indices, planes, contours or layer reuses the global scale; baseline
and normalization changes rescan retained samples. Selections do not rerun the
underlying invariant.

WGL controls provide coordinate sliders, ordered-plane and layer entries,
baseline and contour entries, reset and close. A chart click selects the nearest
stored point on that plane. Independent viewers of the same session synchronize;
disposing one does not close the session. Cairo and saved HTML use its static
snapshot. The new session does not support portable saved-source reconstruction.

## Ensembles of repeated results

Prepared `EnsembleSummary` objects support sampled bands, physical-coordinate
frequency maps, retained ordinary diagrams and linked source selections. The
[ensemble reference](ensembles.md) owns these options and statistical contracts;
[the guide](../../tutorials/uncertainty.ipynb) develops their interpretation.
