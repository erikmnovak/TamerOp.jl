# Inspection and explicit computation

After constructing an encoding, first ask what is already known and which
calculation your next question requires. Its finite poset and parameter map
may be available before all vector spaces and structure maps have been computed.
Such a result is *lazy*: it keeps enough data to carry out the deferred work
when requested. The [finite-encoding introduction](finite_encodings.md)
describes these mathematical pieces; this guide explains how to inspect them
and deliberately request more.

Displaying a mathematical object should not silently solve another mathematical
problem. `show`, `describe`, and owner summary functions inspect stored data.
They may count or summarize that data, but do not construct missing modules,
fiber ranks, spectral pages, homology representatives, or geometry caches.
Inspection is not necessarily constant time: summarizing an existing dimension
vector or sparse support still scans its entries.

For an ingestion encoding `enc`, choose the computation you need:

```julia
describe(enc)              # stored metadata, provenance, materialized status
dimensions(enc)            # dimension vector only; cached for subsequent calls
pmodule(enc)               # full module, including structure maps; cached
```

Before a dimension query, `describe(enc).module_dims` is `nothing` and the
plain-text display says `not computed`. Afterward, the summary reports the
cached vector without constructing the full module. `materialized` distinguishes
these states: having all dimensions does not mean that structure maps exist.
A full module request also makes its dimensions available without repeating
the dimension computation.

`encoding_poset`, `encoding_map`, `encoding_axes`, and provenance accessors do
not materialize the module. An encoded complex's `encoding_complex` accessor
returns its stored complex, which can itself be lazy. Its summary reports the
number of stored terms and differentials; printing it does not fill them.
`encoding_module` has the same explicit full-module behavior as `pmodule`.

Repeated ingestion calls with `cache=SessionCache()` identify a computed module
by its actual constructed cells, grades, boundaries, coefficient field and
encoding request. Changing an input boundary or birth must not reuse the old
module even when the input object, grid and dimension vector are unchanged.
The cache snapshots sparse boundary storage without densifying it. This costs
a scan of the constructed chain data on cached calls; `cache=nothing` avoids
that scan. Lazy ingestion results also own snapshots of their sparse boundaries
and classifier axes, so later source edits cannot change a pending computation.
Treat returned modules and lazy results as mathematical values:
after editing source data, call `encode` again rather than mutating an already
computed result or expecting it to update itself.

Point-cloud geometry caches also distinguish coordinate contents: moving a
site invalidates the old Delaunay geometry or landmark radius graph. Landmark
graph keys inspect the selected landmarks; unchanged unselected sites do not
force a new graph. Equality checks retain coordinates, rather than relying on
their hashes alone. Input mutation is supported between calls, not during a
running computation. Use `PointCloud(coords; copy=true)` when the dataset
should own coordinates independently of the supplied matrix.

## Encoding metadata

`compile_encoding(P, pi)` attaches a classifier to its poset. It does not
generate axes or representative points. The summary's `has_axes` and
`has_representatives` flags report accessor capabilities or supplied metadata,
not whether generated arrays have been cached.

Request `encoding_axes` or `encoding_representatives` when those data are
needed. Grid representatives enumerate the Cartesian product of the axes and
can be large. Generated metadata is returned on demand, not implicitly
memoized; callers who want retained metadata can supply `axes=...` and
`reps=...` when compiling. Adding a workflow session cache preserves supplied
metadata and leaves absent metadata lazy.

## Algebra, geometry, and features

The same distinction applies after encoding: a summary describes what is stored,
while a mathematical query may construct a new answer. Choose the query from
the information you need, such as a dimension, a representative vector, or a
geometric measurement.

- A fringe display reports existing fiber dimensions. Use `fiber_dimension`
  or `dimensions` explicitly to compute ranks.
- A spectral summary reports `convergence_page=nothing` when stored dimension
  tables do not yet certify the first stable page. `convergence_bound` is the
  finite-filtration width bound. Request `convergence_page(ss)` explicitly to
  compute missing pages. An isolated cached later page does not certify the
  earliest stable page if earlier pages remain unknown.
- Ext comparison maps, homology representatives, algebra products, and
  common-refinement bases remain lazy under inspection. Their explicit
  mathematical accessors may compute and cache them.
- Geometry summaries leave exact region geometry and structured-poset relation
  caches untouched. Area, membership, and other geometry queries retain their
  own computation contracts.
- `nfeatures(spec)` counts the output slots without constructing feature names.
  `feature_names(spec)` deliberately constructs those labels. A rank-grid vector
  always has `nvertices^2` slots, even when its intermediate rank table omits
  zero entries.

For example, after constructing a finite encoded module, you may first want
only the dimensions of its Ext spaces. Over an exact field, that query retains
the cycles and boundaries but leaves the choice of quotient representatives
and their coordinates until you request them. You can still ask the same
result for representatives, coordinates, induced maps, and products. The first
such request does the additional work; later requests reuse the stored data.
These computations remain in the finite-poset category described in
[Mathematical categories](math_categories.md).

Deferring coordinates does not defer checking that boundaries are cycles.
Cohomology results from exact complexes check that condition when built.
Numerical fields keep
their checked boundary solve at construction, with the supplied tolerances.
Cached basis matrices are shared data and should be treated as read-only.

For a unified Ext result, `basis(E, t)` returns copies of its chosen
representatives. Repeating that call reuses the stored basis. If you request
the other resolution model with `model=:projective` or `model=:injective`,
the comparison map carries those same classes into that model; the order
still agrees with the result's canonical coordinates. Thus changing models
does not change which Ext class a coordinate vector describes.
Comparisons check inverse identities exactly over exact fields and with the
supplied field tolerances for numerical coefficients.
Public coordinate queries continue to check that the supplied vectors are
cycles, even after a coordinate calculation has been cached. Over rational
coefficients, the retained calculation uses a set of independent ambient
coordinates to recover the class and checks that every remaining coordinate
agrees with a cycle. This avoids solving for all boundary coordinates on each
query while keeping the check exact. Requesting these coordinates does not
construct the representative matrices: those remain deferred until a basis or
representative is requested. A factorization already needed to check boundaries
is reused by the same result. A vector outside the cycle space is rejected even
when the quotient has dimension zero; numerical coefficients retain their
supplied tolerance rules.

Over a prime field with characteristic greater than three, native coordinate
solves now retain their factors on the homology, cohomology or subquotient
result. A factor records work needed to solve the same left-hand matrix
again. Later vectors are still checked against the full represented subspace.
A new result does its own preparation, and discarding the result releases its local
factors. The specialized F2/F3 engines and the existing Nemo backend selection
continue to apply.

Validators and explicit mathematical queries are separate from passive
inspection: they can perform the work required by their documented contracts.

## Asking for several extension products together

Retaining the finite module also lets us ask how its extension classes compose.
Suppose `E_LM`, `E_MN` and `E_LN` are projective Ext results for modules L, M
and N on the same finite poset. A class from Ext^q(L,M) can be composed with
one from Ext^p(M,N), giving a class in Ext^(p+q)(L,N). These computations use
[the finite-poset category](math_categories.md).

For one pair, use coordinate vectors with `yoneda_product`. For a table, put
each collection of coordinate vectors into the columns of a matrix:

```julia
import TamerOp as OP
using LinearAlgebra
DF = OP.DerivedFunctors

# Here K is the coefficient type of the already constructed Ext results.
# E_LM and E_LN must include degree p+q; E_MN must include degree p.
B = Matrix{K}(I, DF.dim(E_MN, p), DF.dim(E_MN, p))
A = Matrix{K}(I, DF.dim(E_LM, q), DF.dim(E_LM, q))
target, products = DF.yoneda_product(E_MN, p, B, E_LM, q, A; ELN=E_LN)
```

`products[:, j, i]` is the coordinate vector for column `B[:, j]` composed
with column `A[:, i]`. Identity matrices request every pair of basis classes;
other columns request the combinations you supply. Over exact fields, the
product has exactly the same coordinates as the corresponding scalar call.
Numerical fields retain their tolerance contract. If the supplied target uses
another projective resolution, the result is transported into that model.

To compose extension classes, the computation first expresses a class as
compatible maps between resolution terms, called a lift. A table request builds
each right-hand class's lift once and reuses it across the left-hand classes.
Those lifts are local to the call. Each product
still undergoes its checked coordinate calculation. Set `return_cocycle=true`
only if you also need the explicit cocycles: the third return value has the
same two column indices, with its first index running over cochain coordinates
in the returned target model.

For repeated multiplication in Ext^*(M,M), use `ExtAlgebra` and its homogeneous
elements. Its first product in a pair of degrees now builds the basis table
with the same preparation; later products reuse that completed table. This
retained-answer workflow is distinct from benchmarking a new computation after
clearing mathematical results, as explained in the [benchmarking guide](benchmarking.md).
