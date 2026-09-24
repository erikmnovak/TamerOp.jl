# Exact multicover computation

For a nonempty finite point cloud, `RhomboidFiltration` models the region
covered by at least `k` indexed closed balls of radius `r`. Radius increases
and coverage decreases: physical grades
are `(r,k)` and the encoding uses oriented coordinates `(r,-k)`.
Source coordinates must admit exact rational conversion, as integers,
rationals and finite floating-point values do. Irrational algebraic input
coordinates are not supported. All backends keep exact predicates and
algebraic physical radius grades for those coordinates. Float inputs mean
their represented binary values.

```julia
using TamerOp
points = PointCloud([0 0; 2 0; 1 2])
filtration = RhomboidFiltration(depth_range=(1,2))
result = encode(points, filtration; degree=1)
provenance(result) # model, depth window, field and executed backend
```

`depth_range=(lo,hi)` includes both endpoints and must lie in `0:npoints`.
Depth zero is contractible ambient space. A positive-depth window can be
empty at small radii; its module is zero there. For native backends, omitting
`depth_range` retains the full unsliced construction. Native bounded
construction prunes carriers during enumeration and builds integer slices and closed slabs,
including the horizontal cap at `hi`. This cap is essential: two sites at
distance two merge at radius one even in the window `(1,1)`.
See the [sliced-cell construction and signs](rhomboid_depth.md).

Choose `backend=:exhaustive` for the dimension-generic support enumerator,
or `:incremental` for exact weighted-Delaunay level discovery in affine
dimensions zero through three. An input embedded in higher ambient dimension
retains its original Euclidean metric. Both native routes require general
position in the region of geometry they compute. The incremental route also
computes prerequisite levels below `lo`; degeneracy there can prevent that
route. `:auto` uses incremental enumeration for affine dimension two with at
least 32 sites and an explicit window ending at depth two; other requests use
exhaustive enumeration. This conservative choice follows the measured
workload regimes. The executed backend is visible in provenance, and neither
choice changes the input.

The incremental implementation uses the subset-label recovery of
[Edelsbrunner and Osang, Algorithm 1](https://arxiv.org/abs/2011.03617).
It retains entire exact lower facets rather than choosing diagonals in
higher-order cells. CDD recomputes the rational weighted hull at each level;
the implementation does not claim a fully dynamic hull or output-linear
runtime. Exact cells, constrained radii and incidence are shared with the
exhaustive route.

## Reusing exact geometry

Pass one `SessionCache` through repeated calls to reuse the computed cells and
exact squared radii, including calls for different homology degrees:

```julia
session = SessionCache()
h0 = encode(points, filtration; degree=0, cache=session)
h1 = encode(points, filtration; degree=1, cache=session)
tree = encode(points, filtration; stage=:simplex_tree, cache=session)
```

Native geometry is shared when exact coordinate contents and shape, requested
backend and depth window agree. Radius cutoffs, skeleton dimensions, coefficient
fields, axes and output stages may differ: their output is built from the same
unfiltered carriers. The subdivision-Cech backend similarly reuses its exact
subset minimum-ball radii for the same coordinates and lower depth bound;
changing its upper depth bound only changes the emitted grades. Neither cache
mixes the native and subdivision-Cech constructions.

Coordinate edits between calls produce a different entry. Equal coordinate
contents in a second point cloud can reuse an entry. Returned complexes own
their grades and boundaries, so editing them does not edit the geometry cache.
Do not modify a point cloud while another task is reading it. Budget checks
still apply on hits, including the recorded peak estimates for incremental
enumeration and the output-specific slicing or triangulation costs.

Concurrent first requests publish one complete artifact under the session's
encoding-cache lock; clearing uses the same lock. CDD computation uses a
separate execution boundary shared with PL geometry; see the
[concurrency contract](cdd_concurrency.md) for external-call and configuration
limits. The session lock protects artifact publication, and the CDD boundary
protects backend execution.
`cache=:auto` creates a cache for that call; cross-call reuse requires keeping
the explicit session. `cache=nothing` disables this geometry cache. A session
retains each distinct coordinate/window/backend entry until it is cleared or
discarded; it does not automatically evict old coordinate snapshots. Retained
memory therefore grows when repeatedly editing clouds in one long session.
Construction budgets apply to each requested construction; they are not an
aggregate memory limit for all retained session entries.
For a new batch of datasets, `session = SessionCache()` starts with empty
storage; the old storage can be reclaimed once nothing else references it.

## Degenerate and repeated sites

```julia
square = PointCloud([-1 -1; 1 -1; 1 1; -1 1])
filtration = RhomboidFiltration(
    backend=:subdivision_cech, depth_range=(1,4),
    construction=ConstructionOptions(
        budget=ConstructionBudget(max_simplices=20_000)))
result = encode(square, filtration; degree=1)
```

This explicit backend handles cospherical inputs and repeated labeled sites
without perturbation. Two labels at the same coordinate count as two balls.
A finite `max_simplices` budget is required because subset and flag enumeration
can grow exponentially. `max_edges` and memory estimates can add limits.
Memory budgets are preallocation estimates, not a hard operating-system cap
on exact-integer or native hull storage.

For a nonempty subset `S`, let `rho(S)` be its exact minimum enclosing radius.
The complex has a vertex for every retained subset and a simplex for every
strict inclusion flag `S0 < ... < Sq`. Its birth is
`(rho(Sq), min(|S0|,hi))`. Retain subsets of size at least `lo`, including
subsets above `hi`: deleting them would remove needed intersections. When
`lo=0`, the empty subset has radius zero and supplies a cone vertex. Thus at
each requested positive depth `k`, the complex is the order complex of Cech
simplices with at least `k` vertices. Radius and depth maps are literal
inclusions of these same labeled flags.

Minimum enclosing balls are computed by enumerating affinely independent
supports of at most affine-dimension-plus-one sites. The optimum occurs among
these supports. Each candidate accepted for a subset is a containing ball,
so taking the least candidate radius cannot underestimate the optimum.
The square roots remain exact, including equality at birth and death values.

The natural equivalence to multicover follows from the diagrammatic
subdivision-Cech argument of
[Blumberg and Lesnick, Theorem 4.13 and Remark 4.14](https://arxiv.org/html/2010.09628v3).
The closed-cover hypotheses hold for finite intersections of closed Euclidean
balls, including singleton intersections at exact birth radii: nested closed
convex sets have the homotopy extension property. See
[Bauer, Kerber, Roll and Rolle, v6, Theorem 5.9(1b), Corollary 5.16 and Proposition 5.19](https://arxiv.org/pdf/2203.03571v6).
For repeated sites, the cover and its intersections remain **indexed by label
subsets**, even when their geometric sets coincide. The same union-of-labels
comparison then applies; deduplicating those indices would change coverage.
At depth zero the empty subset supplies the cone and the comparison with the
ambient space. The resulting equivalence respects maps in both parameters.
The native rhomboid comparison is developed by
[Corbet, Kerber, Lesnick and Osang](https://arxiv.org/html/2103.07823v3); the
[bounded-depth guide](rhomboid_depth.md) gives the explicit terminal attachment.

Native degenerate polyhedral cells remain a separate research extension.
For example, opposite-pair labels in a square have the same geometric
barycenter; treating them as independent cube vertices would give the wrong
incidence. The implemented exact fallback retains the labels and uses
simplicial flags, whose boundaries are well defined over every supported
coefficient field.

## Output size and interpretation

The default `max_dim=nothing` retains the complete chosen model. A requested
`max_dim` constructs its skeleton; degree `h` usually needs at least `h+1`
to retain boundaries. The flag model can have much higher dimension than the
native model. A `radius` cutoff is inclusive and retains its endpoint in the
computed grid. Custom axes must lie within the requested radius and oriented
depth window. As in other ingestion workflows, custom axes floor-snap cell
births; include every critical coordinate to preserve the uncoarsened
filtration. Provenance identifies floor snapping and explicit quantization.
The generic encoding's constant extension beyond its upper endpoint is not a
claim about multicover outside that window. Coefficient choices retain their
ordinary ingestion contracts and are recorded separately.

Native `stage=:graded_complex` and module stages use cellular boundaries.
`stage=:simplex_tree` explicitly requests a compatible subdivision and may be
substantially larger. The degenerate backend is already simplicial. Use
`estimate_ingestion`, inspect the graded complex, and set budgets before
larger runs. Estimates ignore radius exclusions and shared-cell reductions
where those are not cheaply known.

An explicit session reuses both compatible exact geometry and downstream
algebra; see [reusing exact geometry](#reusing-exact-geometry). Each requested
stage still checks its own output budget, including subdivision costs.
