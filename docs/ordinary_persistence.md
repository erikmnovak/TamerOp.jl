# Ordinary persistent homology

When a shape grows, connected components can merge and holes can appear or
fill in. Ordinary persistence records how long these features survive as one
parameter changes. In the example below, squares appear at their array values:
the boundary squares form a ring at zero, and the center fills its hole at five.

`persistence_diagram` computes the persistent homology of a finite
one-parameter chain complex: cells with boundary maps and a single birth
parameter for each cell. Its reduction works over a prime field, with
`TamerOp.CoreModules.F2()` as the default. Integer boundary coefficients are
read modulo the selected prime, including their signs. Use a field object;
symbolic aliases, rational fields and floating-point fields are not accepted
by this direct ordinary-persistence interface.

```julia
import TamerOp as OP

# A square ring appears at 0 and is filled at 5.
values = zeros(Int, 3, 3)
values[2, 2] = 5
D = OP.cubical_persistence(values)
OP.finite_intervals(D; dim=1)       # [(0, 5)]
OP.essential_births(D; dim=0)       # [0]
OP.persistence_intervals(D; dim=0)  # [(0, Inf)]
OP.describe(D)
OP.provenance(D)
```

The result records connected components in degree `H_0`, holes bounded by
loops in degree `H_1`, and higher-dimensional features in subsequent degrees.
Here the single finite degree-one interval records the hole from zero to five;
the essential degree-zero birth records the connected component that remains.
Degree arguments are named keywords. A nonnegative degree above the stored
range has no intervals, except when an explicit `max_homology_dim` request
excluded that degree. Negative degrees are invalid. The two exact-data
accessors return copies, so editing their output does not modify the diagram.

## Choosing coefficients

A coefficient field specifies how chains add and cancel. The default F₂ has
only zero and one, so adding an edge to itself removes it. Over F₃, coefficients
are zero, one and two, and reversing an orientation multiplies by two, the
residue of minus one. The existing signed simplicial and cubical boundaries
therefore matter when we change fields.

```julia
import TamerOp.CoreModules: F3, Fp

D3 = OP.cubical_persistence(values; field=F3())
OP.finite_intervals(D3; dim=1)       # [(0, 5)]
D101 = OP.cubical_persistence(values; field=Fp(101))
OP.finite_intervals(D101; dim=1)     # [(0, 5)]
```

The ring has the same interval in these fields, but that need not hold for
other spaces. For example, a cellular description of the real projective
plane has one cell in each of dimensions zero, one and two, with the
boundary of the two-cell equal to twice the one-cell. That boundary vanishes
over F₂. Over an odd prime it is nonzero: if the one-cell appears at one and
the two-cell at two, the degree-one class dies at two. Over F₂ it survives,
and an essential degree-two class appears as well. Changing the field requires
recomputing the diagram; changing a stored field label cannot give this result.

All prime moduli accepted by `Fp(p)` are supported, including primes up to
`typemax(Int)`. Modular arithmetic avoids machine-integer overflow. The field
contract applies to supplied one-parameter complexes, typed ingestion, both
cubical input conventions, sublevels, superlevels and retained representatives.
Grade ordering and endpoint types remain independent of the coefficient field.

## Interval conventions

A sublevel filtration includes cells whose grade is at most the parameter.
Its finite pair `(birth, death)` represents `[birth, death)`: the class exists
at its birth and is absent at its death. Essential classes persist to `Inf`.

A superlevel filtration includes cells whose grade is at least the parameter,
with maps toward decreasing parameters. Its pair `(birth, death)` has
`birth > death` and represents `(death, birth]` in the original numerical
coordinates. Essential classes persist to `-Inf`.

Cells at equal grades are inserted together mathematically. Internal reduction
orders faces before cofaces to break ties, but its zero-length pairs represent
the zero persistence module and are omitted from the returned barcode. There
is no diagnostic option that changes this barcode convention.

```julia
upper = OP.cubical_persistence(5 .- values; order=:superlevel)
OP.finite_intervals(upper; dim=1)       # [(5, 0)]
OP.persistence_intervals(upper; dim=0)  # [(5, -Inf)]
```

Finite endpoints and essential births retain the supplied grade type, including
integers, rationals, arbitrary precision numbers and exact real algebraic
values. Essential births are stored separately from finite bars.
`persistence_intervals` combines them with an explicit floating infinity marker
without converting any birth or finite death. Use `finite_intervals` and
`essential_births` when exact homogeneous element types matter.

## Cubical inputs and periodicity

`cubical_persistence(values; input=:top_cells)` interprets a two-dimensional
array as grades on squares. A face receives the minimum value of its incident
squares for sublevels, or their maximum for superlevels. This gives the union
of closed squares selected at each parameter.

With `input=:vertices`, entries grade vertices. Each cell receives the maximum
of its vertices for sublevels (the lower star), or their minimum for superlevels
(the upper star). Vertex input supports arrays of any positive dimension.
These input conventions describe different filtrations and must be matched
when comparing software.

`periodic=true` identifies both ends of every axis. A tuple of Booleans selects
axes individually. One periodic axis gives a circle factor, including an axis
with a single cell; two periodic axes give a torus. Nonperiodic axes retain their
boundary. Inputs must be nonempty and have finite values.

```julia
torus = OP.cubical_persistence(fill(2//3, 1, 1); periodic=(true, true))
OP.essential_births(torus; dim=0)   # [2//3]
OP.essential_births(torus; dim=1)   # [2//3, 2//3]
OP.essential_births(torus; dim=2)   # [2//3]
OP.check_torus_persistence(torus; throw=true)
```

`check_torus_persistence` checks the diagram and its expected essential counts
`(1,2,1)`. Counts alone do not prove that an unknown source is homeomorphic to
a torus. It returns a report by default; `throw=true` rejects failures. Use
`check_h1=false` or `check_h2=false` only when explicitly checking fewer degrees.

The equivalent typed vertex workflow is
`persistence_diagram(ImageNd(values), CubicalFiltration(periodic=true))`.
It preserves construction budgets and an explicitly supplied ingestion cache.
Both vertex entrypoints build cubical topology through the public ingestion
builder using order ranks, then restore original grades exactly. Thus the
builder's floating grade storage never rounds the actual vertex values; upper
stars also avoid negating integers at the limits of their range.

## Rips: edges, holes and filling simplices

Suppose four points form a loop. The four edges establish the loop, but they
cannot tell us when it fills in: that requires triangles. A Rips filtration
adds a simplex when all of its edges have appeared, at the largest of their
distances. To compute homology through degree `q`, construct simplices through
`q + 1`. For connected components this means edges; for holes it means
triangles; for two-dimensional cavities it means tetrahedra.

`RipsFiltration(max_dim=2)` therefore controls **simplex dimension**. The
ordinary request `max_homology_dim=1` selects **homology degrees zero and one**
and checks that the construction includes the triangles needed for deaths.
An insufficient `max_dim` raises an explanatory error. Queries in degrees above
an explicitly requested maximum also raise an error. Without the homology
keyword, the diagram describes the entire constructed skeleton; its top-degree
classes can be artifacts of omitting the next simplex dimension.

A sparse distance matrix lets us supply the edges we actually have. Here four
edges enter at one, and a diagonal enters at two. The diagonal completes two
triangles, which fill the hole:

```julia
using SparseArrays

S = sparse([1, 2, 3, 1, 1], [2, 3, 4, 4, 3], [1.0, 1, 1, 1, 2], 4, 4)
filtration = OP.RipsFiltration(max_dim=2)
D = OP.persistence_diagram(S, filtration; max_homology_dim=1)
OP.finite_intervals(D; dim=1)  # [(1.0, 2.0)]
```

The matrix dimensions retain all four vertices, including any isolated ones.
A missing stored entry means an absent edge, **not distance zero**. A stored
zero is a genuine edge born at zero; dropping stored zeros changes the input
filtration. Either triangle of the matrix is accepted. If both directions are
stored, their values must agree within the symmetry tolerance; the upper
entry determines the distance when they differ within that tolerance. Sparse transpose
and adjoint inputs follow the same rule. Dense matrices require both directions;
positive infinity can denote an absent edge in either representation. Distances
must be nonnegative, diagonal entries zero and all values free of NaNs, with a
`1e-10` tolerance for symmetry and small rounding errors near zero. Construction
uses Float64 distances. These checks do not certify the triangle inequality.
Thus an arbitrary supplied graph defines its weighted clique filtration;
identifying it with a metric Rips filtration requires that interpretation of
the supplied distances. The builder does not fill in missing distances by
shortest paths.

The radius is an inclusive **edge-distance cutoff**. It is not a ball radius
that gets doubled. If we stop the example at one, the filling triangles have
not appeared:

```julia
limited = OP.persistence_diagram(S, OP.RipsFiltration(max_dim=2, radius=1);
                                 max_homology_dim=1)
OP.essential_births(limited; dim=1)  # [1.0]
OP.provenance(limited).window       # (0.0, 1.0)
```

Here the surviving class is essential in the constructed filtration. It is not
proved to survive beyond the cutoff. Barcode and diagram views display this
radius restriction alongside surviving bars. `provenance(limited).rips` records the
input convention, cutoff and simplex/homology coverage. A sparse graph gives
the exact restriction of a larger metric input through a radius only when it
contains every edge at or below that radius; missing distances cannot establish
that completeness by themselves.

`ConstructionOptions(sparsify=:radius)` uses the supplied radius to select
edges. `sparsify=:knn` selects the undirected union of each vertex's nearest
neighbors among available edges, and also respects a supplied radius. Neighbor
selection generally changes the filtration; it is not a certificate that all
Rips bars are preserved. The same graph selection and optional certified
edge-collapse machinery serve sparse and dense distance input. Set limits with
`ConstructionOptions(budget=(max_edges=..., max_simplices=...))`. Exceeding a
limit raises an error; construction never drops cells merely to fit a budget.
The existing `memory_budget_bytes` option bounds an estimated dense boundary
footprint, not total process memory. Advanced users can inspect
`OP.DataIngestion.estimate_ingestion(S, filtration)` before construction;
higher-dimensional counts are conservative bounds, without enumerating cliques.

## Choosing landmarks and measuring what was omitted

A smaller set of points can make a Rips calculation affordable. We also want to
know how well those points cover the original data. **Landmarks** are selected
original points; their **covering radius** is the largest distance from any
original point to its nearest selected point.

For five equally spaced points on a line, select three landmarks:

```julia
cloud = OP.PointCloud([[0.0], [1.0], [2.0], [3.0], [4.0]])
selection = OP.select_landmarks(cloud; count=3)
OP.landmark_indices(selection)  # [1, 5, 3]
OP.covering_radius(selection)  # 1.0
```

The routine starts at point 1, then repeatedly chooses a point farthest from
those already selected. It keeps this selection order, with ties resolved by
the original index. Distinct input indices remain distinct even if their points
coincide. Here the endpoints come first, followed by the midpoint; every omitted
point lies at distance one from a landmark.

The same subset can now serve the ordinary barcode or the encoding workflow:

```julia
landmarks = OP.LandmarkRipsFiltration(
    landmarks=OP.landmark_indices(selection), max_dim=2)
D_landmarks = OP.persistence_diagram(cloud, landmarks; max_homology_dim=1)
OP.covering_radius(OP.landmark_selection(D_landmarks))  # 1.0
G_landmarks = OP.encode(cloud, landmarks; stage=:graded_complex)
```

`G_landmarks` retains the selected filtered complex for further questions.
Continue `encode` to a module or encoding when those are needed; the direct
barcode call answers the one-parameter question without constructing that
additional object. The returned diagram records the selected original indices
and their coverage. `landmark_selection(D)` returns `nothing` if no subset was
selected. Automatic `RipsFiltration(n_landmarks=..., construction=
ConstructionOptions(sparsify=:greedy_perm))` records the same information.
A complete dense distance matrix can replace the point cloud in either route.
Sparse omissions do not supply enough information to measure coverage.

For a finite **metric** input and covering radius $r$, the bottleneck distance
between the full input and landmark Rips diagrams is at most $2r$, using edge
lengths as the parameter. To see where this scale comes from, send each point
to its nearest landmark. If two original points are at distance at most $t$,
the triangle inequality places their chosen landmarks at distance at most
$t+2r$. The resulting simplicial maps, together with inclusion of the landmarks,
give the persistence comparison. This is the covering-radius bound also
explained in [Ripser.py's landmark tutorial](https://ripser.scikit-tda.org/en/latest/notebooks/Greedy%20Subsampling%20for%20Fast%20Approximate%20Computation.html).

`OP.describe(selection)` includes this **conditional** bound, ordered insertion
radii, and each original point's nearest landmark index and distance. The first
insertion radius is `Inf`, since no landmark precedes it. Matrix validation does
not certify the triangle inequality. This bound concerns full metric Rips
filtrations with the filling dimensions present; it does not certify additional
nearest-neighbor pruning, top-degree artifacts of an incomplete skeleton, or
essential bars censored at a finite cutoff. Distances use Float64; numerical
rounding is separate from the mathematical bound.

Coverage needs only linear storage in the input and subset sizes. If a later
calculation needs every landmark-to-original-point distance, retain that larger
table explicitly:

```julia
with_distances = OP.select_landmarks(cloud; count=3, retain_distances=true)
OP.landmark_distances(with_distances)  # 3 rows (landmarks), 5 columns (input points)
```

Rows follow selection order and columns retain original point order. Supplying
`indices=[5, 1, 3]` instead of `count=3` inspects that particular ordered subset.
The coverage result owns its arrays; changing the input later does not update
an already computed selection.

## Graphs whose vertices appear at different times

Distances give the usual Rips convention: all vertices exist at zero. A filtered
graph may instead introduce vertices at different times. An edge must then
appear no earlier than either endpoint. `EdgeWeightedFiltration` accepts these
birth times directly and can fill cliques: a clique is a set of vertices with
all pairwise edges present, and its simplex appears at the latest face birth.

Consider three vertices born at zero, one, and two. The edges arrive at three,
four, and four:

```julia
graph = OP.GraphData(3, [(1, 2), (2, 3), (1, 3)]; weights=[3.0, 4.0, 4.0])
weighted = OP.EdgeWeightedFiltration(vertex_births=[0.0, 1.0, 2.0], max_dim=2)
D_weighted = OP.persistence_diagram(graph, weighted; max_homology_dim=1)
OP.finite_intervals(D_weighted; dim=0)  # [(1.0, 3.0), (2.0, 4.0)]
OP.essential_births(D_weighted; dim=0)  # [0.0]
```

At three, the components born at zero and one merge; the older component
survives. At four the remaining component joins it. The triangle also appears
at four, so no positive-length loop interval remains. The default `max_dim=1`
keeps only the graph; choose `max_dim=q+1` when requesting homology through
`max_homology_dim=q` so that possible filling simplices are present.

A weighted adjacency matrix describes the same input using its diagonal for
vertex births and off-diagonal entries for edge births:

```julia
A = [0.0 3.0 4.0; 3.0 1.0 4.0; 4.0 4.0 2.0]
D_matrix = OP.persistence_diagram(A, OP.EdgeWeightedFiltration(max_dim=2);
                                 max_homology_dim=1)
```

Dense matrices must be symmetric off the diagonal; `Inf` denotes an absent
edge. Sparse matrices accept either triangle, require matching values when
both directions are stored, and treat omitted edges as absent. Stored zeros
are real birth times; omitted diagonal entries mean vertex birth zero. Negative
births are allowed. Every edge must satisfy the endpoint condition, including
edges outside a requested cutoff; malformed input raises an error. Birth times
must be finite and representable as Float64. The `RipsFiltration` distance
contract remains distinct: nonnegative distances and a zero diagonal.

For `GraphData`, omitted vertex births default to zero. The filtration's optional
`edge_weights` vector overrides the graph's weights. With matrix input, provide
births in the matrix alone. `threshold=t` restricts the filtration inclusively:
vertices born later are omitted, and surviving bars are known only through that
threshold. `OP.provenance(D_weighted).weighted_flag` records the original vertex
indices and conventions.

The same input works with `encode(...; stage=:graded_complex)` or
`stage=:simplex_tree`. Barcode and cocycle queries use implicit computation
through H₂; higher degrees and homology representatives use explicit
construction. `cocycles=true` retains cochains, and `representatives=true`
retains homology cycles and death fillings. Edge, simplex and memory-estimate
budgets retain their existing meanings. Weighted-vertex input requires
`sparsify=:none` and `collapse=:none`; use `threshold` to restrict its range.

## Rips barcodes without building every boundary

The four-edge loop above needs triangles to decide when its hole dies. That
does not mean we must store every triangle and its boundary in advance. For a
barcode request, TamerOp can generate the needed containing simplices as the
calculation reaches them, then discard that temporary work. This is the
**implicit Rips** route.

The default `method=:auto` chooses it for point clouds, dense or sparse distance
matrices, landmark Rips requests, and weighted-vertex flag inputs through H₂, over any supported prime
field. It computes the same finite and essential intervals as explicit
construction, including finite deaths. Distances and their ordering use the
existing Float64 geometry contract; exact field arithmetic does not make the
input geometry exact.

```julia
fast = OP.persistence_diagram(S, filtration; max_homology_dim=1)
explicit = OP.persistence_diagram(S, filtration; max_homology_dim=1,
                                  method=:explicit)
OP.finite_intervals(fast; dim=1) == OP.finite_intervals(explicit; dim=1)  # true
OP.provenance(fast).backend              # :implicit_rips_cohomology
OP.provenance(fast).computation.boundary_matrix_materialized  # false
```

For connected components, joining two previously separate components records
a death. For higher degrees, the calculation works with **cofaces**: simplices
containing a given simplex with one additional vertex. Their signed incidence
coefficients support odd prime fields as well as F₂. Reverse-order cohomology
reduction, clearing of already paired simplices, and reconstruction from
stored change vectors avoid retaining the original or reduced coboundary
matrix. These methods follow [Ulrich Bauer's description of
Ripser](https://arxiv.org/abs/1908.02518). Over a field, this dual computation
gives the ordinary homology barcode. Add `cocycles=true` to retain representatives
for the [scale-specific cohomology workflow](#measuring-a-hole-with-a-cocycle).

Some pairs can be recognized from the local order alone: a simplex is the
last face of its first containing simplex, and both appear at the same grade.
These are **apparent pairs**. Their zero-length intervals do not appear in the
barcode. The calculation can reconstruct their signed coefficients when needed,
instead of storing a general reduction vector for each pair. Other one-term
vectors use compact storage; vectors with several terms retain all coefficients.

For a bounded barcode request on a complete distance graph, TamerOp can also
stop once a vertex connects to every other vertex. If the smallest such radius
is `r`, then every simplex present at `r` can be joined to that vertex: the flag
complex is a cone. Higher homology has vanished, and only one connected component
survives. This is a certificate for the complete requested barcode, not a new
approximation. Inspect `provenance(fast).computation.terminal_radius` for the
radius and selected-graph vertex, or `nothing` when the certificate is unused.
The current shortcut applies to barcode-only requests with an explicit
`max_homology_dim`; retained cocycles, weighted vertices and requests that include the top homology
of a truncated skeleton keep their requested graph and degree range. User budget
checks still precede this shortcut. A supplied radius below the certified
terminal radius keeps its original cutoff interpretation.

This removes a major source of storage, but not all size limits. The selected
neighbor graph is stored; a dense input still has quadratically many edges.
The current degree's remaining simplices and its reduction change vectors also
take space, and difficult inputs can generate substantial temporary work.
Sparse input remains a sparse graph rather than becoming a dense distance
table. Reduction statistics in `provenance(fast).reduction_stats` count work
and stored terms; they are not byte or process-memory measurements.

The degree and radius rules above still apply. A bounded request through H₁
uses edges and triangles even if `max_dim` permits more dimensions. Without
`max_homology_dim`, the complete requested skeleton is respected, including
top-degree classes that further simplices might kill. `method=:auto` uses the
explicit route for higher degrees or `representatives=true`, retaining the
existing cycle and death-filling output. Set `method=:implicit` to require the
implicit route and receive an error if the request is unsupported.

Edge budgets are checked before optional certified collapse. Simplex and
memory-estimate budgets apply to the retained expansion through the dimensions
needed for this request. Supplying either of those budgets triggers a streamed
clique count, which can itself be costly on a dense graph; it does not build
boundaries. `memory_budget_bytes` retains its construction meaning: the maximum
estimated dense boundary size, not a limit on actual reduction memory or RSS.
No cells are silently discarded to satisfy a budget.

Landmark selection and nearest-neighbor graph selection still change the input
filtration. Their use, retained point indices, edge counts and certified collapse
option appear in `provenance(fast).input_selection`. A radius cutoff instead
restricts the parameter range. These choices are separate from the implicit
reduction itself, which preserves the barcode of the selected graph filtration.
All graph and reduction state belongs to the current call; this route does not
retain mathematical results in an `EncodingCache` for later queries.

## General filtered complexes

Pass a one-parameter `GradedComplex` directly to `persistence_diagram`, or pass
data and a typed filtration to construct the complex first:

```julia
D = OP.persistence_diagram([0.0 2.0; 2.0 0.0], OP.RipsFiltration(max_dim=1))
OP.finite_intervals(D; dim=0)  # [(0.0, 2.0)]
```

A direct complex is checked before reduction: packed cell dimensions and sparse
storage must be valid, boundaries must have the correct shapes, double
boundaries must vanish in the selected field, and every coefficient nonzero in
that field must respect the selected filtration order. Empty typed complexes are accepted.
This is an algebraic filtration contract in the selected field: an integer
boundary coefficient divisible by its characteristic vanishes. A boundary
squared that is zero over F₂ need not be zero over an odd prime.

Changing `order` on an arbitrary ingestion request does not turn a lower-star
builder into an upper-star builder. The resulting grades must satisfy the
chosen order or the call fails. The dedicated cubical vertex route handles the
upper-star construction explicitly. Other builders keep their own geometry and
grade-precision contracts; exact reduction does not certify their geometry.

`provenance(D)` records the field, homological convention, filtration direction,
interval convention, grade type and reduction backend. Cubical routes also
record array shape, periodicity and input convention. A generic ingestion build
does not carry execution provenance, so its effective construction
and any substitution are reported as `:not_recorded`, rather than inferred from
the requested filtration name. A hand-built `PersistenceDiagram` likewise
reports its computation history as `:not_recorded` unless metadata was supplied;
constructing stored intervals does not claim that the reducer ran.

## Computing a barcode without retaining chains

A barcode records births and deaths. Computing it does not require retaining
all the chains used to establish those events. For a supplied complex or an
explicit construction over F₂, the default request
uses a reduction organized by homological degree. Once a birth is paired with
a death in the next degree, its column is known to reduce to zero and can be
skipped. This is called **clearing**. A reusable binary workspace handles column
additions. Stored columns use individual entries when sparse and groups of
binary coefficients when several entries share a machine word. Adding a group
at once performs the same arithmetic over F₂ with less bookkeeping. These
choices preserve the barcode; they do not approximate the grades or boundaries.
Completed columns are extracted in one traversal, and saved columns share
storage within the current computation. This avoids repeated bookkeeping and
individual column-array objects; it does not cache a barcode for a later call.

Connected-component persistence depends only on vertices and edges, even when
the complex also contains squares, cubes, or other higher-dimensional cells.
When each edge has zero or two boundary coefficients that are nonzero modulo
two, the computation tracks components directly: joining two components kills
the younger one. In a graph, closing a cycle creates an essential degree-one
class. With higher-dimensional cells present, the pairings already found in
higher degrees determine which of those cycles eventually die.

The highest boundary can sometimes use the same idea in reverse. If each of
its rows has at most two nonzero coefficients modulo two, regard the
highest-dimensional cells as vertices and their shared faces as edges, then
process this graph in reverse filtration order. A bookkeeping vertex handles
faces with only one incident cell; it contributes no reported class. Real
unpaired top-dimensional classes remain in the barcode, including those on
periodic grids. Eligibility is checked from the supplied boundary matrix;
arbitrary algebraic boundaries retain general reduction when the condition
fails. These shortcuts preserve all finite and essential intervals.

The usual input checks remain enabled. In particular, for the binary route
the double boundary must vanish exactly modulo two. Dense, sufficiently reused boundary columns can use
packed binary products for this check; sparse inputs use direct parity checks.
Neither route performs floating-point arithmetic on the boundary coefficients.

Requesting `representatives=true` retains the column reduction that constructs
cycles and filling chains. Its selected representatives are preserved. You can
inspect the executed route with `provenance(D).backend`: barcode-only requests
report `:f2_clearing` for the degree-organized route, which may use the component
shortcuts above, or `:f2_graph_union_find` for a graph-only complex. Requests
retaining chains report
`:f2_column_reduction`. The public call and interval conventions are the same.

For odd primes, sparse column elimination adds scaled columns and normalizes
pivot coefficients using field inverses. It preserves signs and checks double
boundaries in that field. With representatives requested, the same operations
track source chains so that each finite death filling has precisely the
returned cycle as its boundary. Barcode-only requests omit that tracking.
This route reports `:prime_column_reduction`; the binary graph, dual-graph and
packed-XOR shortcuts remain specific to F₂.

## Which cells represent an interval?

The ring's interval `[0,5)` tells us when its hole exists. To inspect a chain of
edges representing that hole, retain the reduction's choices during computation.
A **cycle** is a chain whose boundary is zero. Here its homology class remains
nonzero from parameter zero until parameter five, when it becomes the boundary
of a chain of squares. That latter chain explains the interval's death.

Representative inspection is an advanced operation, so import the `Advanced`
namespace alongside the ordinary workflow:

```julia
import TamerOp.Advanced as OA

retained = OP.cubical_persistence(values; representatives=true)
OP.persistence_diagram_summary(retained).representatives_available  # true
selected = OA.persistence_representative(retained; dim=1, kind=:finite, index=1)
selected.available              # true
selected.interval               # (0, 5)
selected.cycle.cell_indices     # Positions among the one-dimensional cells
selected.cycle.cell_ids         # Their original labels in the constructed complex
selected.cycle.coefficients     # Coefficients modulo two
selected.bounding_chain         # A two-dimensional chain bounding this cycle at 5
```

Both chain records include their dimension, exact cell grades, and coefficients
as nonzero integer residues `1:p-1` in the recorded field. For F₂ these
coefficients are all one; for odd primes they can differ. Cycle and filling
equations are interpreted modulo that prime. Indices refer
to positions within that chain dimension; IDs are the labels supplied by the
complex. IDs can repeat in different dimensions and do not establish pixel,
point-cloud, or other source geometry. For cubical inputs these records refer
to the constructed cubical complex. Drawing its cells on the original image
requires an additional geometric correspondence.

The returned cycle is a noncanonical choice made by the field-specific reduction: changing
the ordering of tied cells can change it. There is no shortest-cycle or preferred
geometric-shape guarantee. A finite interval's `bounding_chain` has this cycle
as its boundary at death. An essential interval has `bounding_chain=nothing`,
because it never becomes a boundary in the supplied finite complex. For
superlevels the same statements hold as the parameter decreases; birth remains
included and finite death excluded.

Retention is opt-in because tracking change-of-basis columns can substantially
increase memory during reduction. The default result keeps interval data:

```julia
intervals_only = OP.cubical_persistence(values)
OA.persistence_representative(intervals_only; dim=1).reason  # :not_retained
```

An unavailable response has `available=false` and no cycle. A hand-built
diagram containing only endpoints has no retained representative either;
adding descriptive provenance does not supply the missing chains. Invalid
dimensions, interval kinds, and member indices are errors. Pass
`representatives=true` to `persistence_diagram` as well when starting from a
graded complex or a data-and-filtration pair.

Equal intervals need an additional choice. The one-cell torus has two essential
degree-one classes born at `2//3`; their common endpoints do not identify one
of the two cycles:

```julia
retained_torus = OP.cubical_persistence(fill(2//3, 1, 1);
    periodic=true, representatives=true)
first_cycle = OA.persistence_representative(retained_torus;
    dim=1, kind=:essential, index=1)
second_cycle = OA.persistence_representative(retained_torus;
    dim=1, kind=:essential, index=2)
```

`index` refers to the original member in `finite_intervals` or `essential_births`,
according to `kind`. It is local to that diagram. In a grouped visualization,
first select the interval group, then choose an original member before requesting
its cycle. These choices do not identify classes between different slices or
separate computations. The [visualization guide](visualization.md) explains
how the barcode, diagram and representative readout share this selection.

## Measuring a hole with a cocycle

A cycle describes a hole by combining edges into a closed loop. Another way to
study the same hole is to assign values to edges and add those values around a
loop. To obtain a measurement that depends only on its homology class, the sum
must vanish around the boundary of every active two-dimensional cell. Such an
assignment is a **cocycle**. More generally, a degree-`d` cochain assigns a field
coefficient to each oriented `d`-cell; it is a cocycle when its coboundary is zero.
Two cocycles differing by a coboundary represent the same cohomology class.

The ring example supplies both perspectives. We can keep a cocycle while
computing its barcode, then ask for its values at a particular scale:

```julia
import TamerOp as OP
import TamerOp.Advanced as OA
import TamerOp.CoreModules: F3

values = zeros(Int, 3, 3)
values[2, 2] = 5
cohomology = OP.cubical_persistence(values; field=F3(), cocycles=true)
OP.finite_intervals(cohomology; dim=1)  # [(0, 5)]
measurement = OA.persistence_cocycle(cohomology; dim=1, scale=2)
measurement.cochain.cell_indices
measurement.cochain.coefficients      # Nonzero residues modulo three
OP.describe(cohomology).cocycles_available  # true
```

Unlisted cells have coefficient zero. At scale two, these coefficients satisfy
the cocycle equations on all active squares and represent a nonzero class.
Scale five is excluded: the hole has filled, and this class cannot extend
through that stage. Requesting it there raises an error. `kind=:essential`
selects an essential interval; `index` distinguishes separate members with
equal endpoints. The requested scale must be finite and inside both the
selected interval and any recorded radius window.

Why specify a scale? More cells become available as a sublevel grows. A
cocycle on the larger complex gives one on the smaller complex by forgetting
its values on cells that are absent there. This is **restriction**. It goes
from `H^d(K_t)` to `H^d(K_s)` when `s ≤ t`, opposite to the homology map from
`H_d(K_s)` to `H_d(K_t)`. For two scales within the same retained member's
interval, TamerOp returns exactly these restrictions of a fixed cochain.
Before birth that cochain restricts to zero. For superlevels, smaller complexes
occur at larger numerical parameters, so the numerical inequalities reverse.

For finite chain complexes over a field, cohomology is naturally dual to
homology and has the same interval multiset. This does not identify their
chosen representatives. `representatives=true` still asks for homology cycles
and death fillings. `cocycles=true` asks for cohomology representatives. Both
can be requested together, at the cost of both reductions; matching interval
indices do not assert that these independently chosen bases are dual.

### Following the cochain back to its cells

For a supplied `GradedComplex`, the output uses its original cell labels,
dimension-local positions, exact grades and boundary orientations. Cubical and
other explicitly constructed complexes report their constructed cells; those
labels alone do not define a geometric overlay on the original data.

The implicit Rips route also supports `cocycles=true` through degree two,
including degree-zero functions on connected components. Its sparse cochain
records contain `source_vertices`: an ordered tuple of original point indices
or distance-matrix row indices for each simplex. The tuple's order specifies
orientation, including when landmarks were listed out of numerical order.
`cell_indices` and `cell_ids` on this route are combinatorial simplex indices
within the selected vertex set, not positions in an explicitly stored complex.

```julia
distances = [0.0 1 2 1; 1 0 1 2; 2 1 0 1; 1 2 1 0]
rips = OP.persistence_diagram(distances, OP.RipsFiltration(max_dim=2);
    field=F3(), max_homology_dim=1, cocycles=true)
edge_values = OA.persistence_cocycle(rips; dim=1, scale=1)
edge_values.source_vertices           # Oriented edges in the input vertex labels
edge_values.cochain.coefficients
```

Landmark or nearest-neighbor selection still means we are studying the selected
complex. It does not produce cocycles on discarded points or edges. Cocycle
retention requires `collapse=:none`: returning to the original complex after
an edge collapse needs a verified cochain lifting map, which this interface
does not provide.

Retention is optional because representative supports can be large. The default
barcode route keeps its existing reductions and stores no public cocycles.
The implicit route with retention keeps selected cochains, including the work
needed for degree zero, while continuing to generate cofaces as needed.
`describe` reports availability and `provenance` records the field, filtration
and variance. Returned cochain records are immutable copies.

Cocycles provide input for further questions that endpoints alone cannot
answer. Circular coordinates additionally need an integer lift and a coordinate
solver; cup products need a compatible multiplication on cochains. Retaining
cocycles supplies their starting data, not those additional constructions.

## Compare diagrams and make features

Choose the same homology degree explicitly in each analysis call. The
[worked notebook](tutorials/ordinary_analysis.ipynb) follows three points through
two predictable merges, diagram comparison, finite features and a saved result.

```julia
other_values = copy(values)
other_values[2, 2] = 6
other = OP.cubical_persistence(other_values)
OP.bottleneck_distance(D, other; dim=1)       # 1.0
OP.wasserstein_distance(D, other; dim=1, p=2) # 1.0

landscape = OP.persistence_landscape(D; dim=1, tgrid=0:0.5:6, kmax=2)
image = OP.persistence_image(D; dim=1,
    xgrid=0:0.5:2, ygrid=0:0.5:6, sigma=0.5)
OP.persistence_silhouette(D; dim=1, tgrid=0:0.5:6)
OP.barcode_entropy(D; dim=1, normalize=false) # 0.0: only one finite bar
OP.barcode_summary(D; dim=1)                 # Total persistence 5.0
```

These methods reuse the barcode matching and feature algorithms used elsewhere
in the library. Repeated intervals remain separate members. Fields may differ
between diagrams: the distance compares their interval multisets, so it does
not compare chosen chains, certify a source correspondence or construct a map.
`bottleneck_matching` returns member indices in stored finite-member order,
followed by included essential members; zero denotes the diagonal.

### Decide how to treat essential intervals

Distances use `essential=:keep` by default. Essential bars can only match other
essential bars. Different essential counts give infinite distance; equal counts
use birth differences. Finite features default to `essential=:error` and reject
an essential bar instead of silently dropping it.

| Choice | Meaning |
| --- | --- |
| `essential=:drop` | Describe only finite bars; the input diagram remains intact. |
| `essential=:cap, essential_cap=c` | Replace essential deaths with the chosen grade `c`, strictly after their births in filtration order. Finite bars are not clipped. |
| `essential=:keep` | Keep infinite deaths for distances or explicit interval inspection; finite features reject them. |
| `essential=:error` | Require the selected degree to have no essential bars. |

For example, `OP.barcode_summary(D; dim=0, essential=:drop)` omits the connected
component that survives forever. Capping that component is an observation
choice, not its computed death and not extended persistence. The cap is always
in original grade units, including for superlevels.

### Use comparable coordinates and grids

`analysis_barcode(D; dim, ...)` exposes an independent vector of exact analysis
intervals. Sublevel analysis time is `(grade-origin)/scale`; superlevel analysis
time is `(origin-grade)/scale`, where `scale` must be positive. Superlevel time
is reflected, retaining the direction of the filtration. A comparison requires
the same order on both diagrams. Integer, rational and floating grades become
exact rationals (including the exact binary value of a float); algebraic grades
remain algebraic. The original result retains its grade types and provenance.

Every numerical grid is in these analysis coordinates. Supply the same `origin`,
`scale`, grid and feature settings when comparing results. Grids must be finite
and strictly increasing after Float64 conversion; landscape and silhouette
grids require at least two points. An automatic landscape grid that collapses
large exact endpoints is rejected: translate by a common `origin` and, when
needed, choose a common `scale` before sampling.

Bottleneck matching compares exact costs for these adapted intervals, then
returns a Float64 distance. Wasserstein forms endpoint differences before
conversion and uses numerical assignment; its `p` is at least one and its
ground norm `q` is `1`, `2` or `Inf`. `p=Inf` selects bottleneck and requires
`q=Inf`. Powered costs that overflow or underflow are rejected with a request
to rescale. The auction backend uses a relative numerical tolerance; choose
`backend=:hungarian` for direct assignment on the Float64 cost matrix.

Landscapes sample ordered tents. Silhouettes average weighted tents. Images
sample Gaussian values at grid centers, **not pixel integrals**; their rows are
y-coordinates and columns are x-coordinates. Entropy uses lifetime weights by
default and `normalize=true` divides by the logarithm of the number of weighted
members. These arrays and summaries use Float64 arithmetic and lose information;
a distance between feature arrays is not automatically a diagram distance.

## Save a diagram with retained data

```julia
OP.save_persistence_diagram_json("ring.json", D)
restored = OP.load_persistence_diagram_json("ring.json")
OP.inspect_json("ring.json")
OA.check_persistence_diagram_json("ring.json")
OP.bottleneck_distance(D, restored; dim=1) # 0.0
```

The versioned owned format preserves all degrees, exact endpoint values and
types, coefficient field, filtration order, provenance and any retained cycles,
death fillings and cocycles. Available landmark selections and oriented source
vertex records survive too. Saving does not compute missing representatives or
include an unretained source complex or its geometry. Reloaded cocycles support
the same scale-specific queries as before saving.

The default profile is compact; `profile=:debug` requests indented JSON.
`inspect_json` reads a cheap header, while
`Advanced.check_persistence_diagram_json` validates the complete saved result.
The loader checks schema and retained storage, but independently verifying chain
equations still requires the source complex. Supported metadata includes scalar,
array, tuple and dictionary data, built-in source type labels and landmark
selections. Unsupported custom objects are rejected before writing; they are
not converted silently to strings. No executable source is evaluated on loading.

## Results and figures

`result_summary(D)` and `describe(D)` show the same cheap mathematical summary.
The advanced `field(D)` and `filtration_order(D)` accessors return the coefficient
field and order. `check_persistence_diagram(D)` validates a hand-built diagram;
wrap its report with `persistence_validation_summary` for compact notebook
presentation. It checks barcode storage, not an unrecorded source computation.

`visualize(D; kind=:persistence_diagram, dim=1)` and
`visualize(D; kind=:barcode, dim=1)` use the shared visualization interface.
Plotting makes explicit Float64 display copies while retaining exact intervals
in the visualization metadata. Values that overflow or distinct endpoints
that collapse at display precision are rejected. Essential bars have a separate
labelled infinity lane or continuation arrows, including `-Inf` for superlevels.
Rendering requires an explicitly loaded plotting extension; constructing and
inspecting a diagram does not require a plotting package.

This is a direct route from a one-parameter filtered complex to its barcode;
it does not require constructing an `EncodingResult`. With several parameters,
we need to retain vector spaces and their maps to support a wider range of
questions. Continue with [why two parameters change the problem](two_parameters.md)
to see why a barcode no longer gives the same kind of complete description.
