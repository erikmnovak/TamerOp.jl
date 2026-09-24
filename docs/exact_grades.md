# Exact rhomboid grades

`RhomboidFiltration()` stores physical radii as `AlgebraicReal` values. The
construction computes rational squared radii and takes exact square roots.
Sorting, membership, cell attachments and grid indices use these exact values;
two radii with the same `Float64` display remain different computational grades.
Exactness refers to the supplied coordinates: a floating input means its
represented binary value, not an unrecorded intended decimal value.

```julia
using TamerOp

A = AlgebraicReal
r = sqrt(A(2))
r^2 == 2                       # true exactly
Float64(r)                     # explicit approximation for display

points = PointCloud(reshape(QQ[0, 2, 4 + QQ(1, big(2)^70)], :, 1))
enc = encode(points, RhomboidFiltration(); degree=0)
encoding_axes(enc)             # distinct exact critical values
provenance(enc).approximation.grade_arithmetic  # :exact_real_algebraic
```

An exact cutoff such as `RhomboidFiltration(radius=sqrt(A(2)))` includes cells
born at that radius. Finite values smaller or larger than the `Float64` range
remain valid grades. `AlgebraicReal` supports exact arithmetic and comparisons;
converting an irrational value to `QQ` or `Int` raises `InexactError`.

## Coordinates and queries

The physical parameters are `(radius, depth)`. Their order increases radius
and decreases depth. The encoded axes are therefore `(r, -k)`, both increasing.
`locate` receives physical `(r, k)` queries; grid signed-measure and Euler axes
are the stored oriented `(r, -k)` coordinates. This distinction matters when
supplying your own axes. The default critical grid retains every birth. An
explicit coarse grid or quantization deliberately changes grade placement;
`provenance(enc).discretization` records that choice.

An exact radius is not a squared-radius coordinate. Replacing `r` by `r^2`
preserves the filtration order, but changes straight slices and metric values.
The slicing and matching paths preserve the radius coordinate. The exact 2D
matching optimizer retains its existing finite-window and positively oriented
classifier contract. The default rhomboid classifier has orientation `(1,-1)`
and is rejected by that optimizer. To compare in increasing coordinates
`(r,-k)`, use a classifier with orientation `(1,1)` on the same stored axes and
finite poset, and express the matching window in those coordinates. Its
geometric arithmetic is exact, and its final scalar output is `Float64`.

Exact 2D slice queries retain algebraic endpoints. Sampled queries retain the
specified sample points and their exact classifications, but still only
represent those samples. Compiled sampled plans preserve exact parameter
values and can be reused across modules. Ingestion, slice-plan and feature
caches compare exact request contents, including filtration parameters,
bounds, directions, offsets and sample values. The ingestion keys also retain
the final encoding axes, so changing a grid cannot reuse a module computed on
another grid. Request arrays are snapshotted into immutable keys; later edits
cannot change an already-stored key. Different exact values remain distinct
even when their hashes coincide.
For a hand-built grid, use `AlgebraicReal.(axis)` on each axis to select the
same exact slice-arrangement arithmetic. This guide does not change the
geometry-arithmetic contract of every other classifier family.

Featurizer constructors retain exact query directions, offsets, slice bounds
`tmin`/`tmax`, barcode-clipping `window` bounds and supplied invariant-query
axes. Saving and loading feature metadata preserves these coordinates.
Feature caches and composite feature
groups share computations only when the corresponding exact requests agree.
Barcode summaries and top-k features compare exact endpoints and subtract them
before converting persistence lengths to numerical outputs.
Images, silhouettes and entropy also form exact endpoint differences before
evaluating numerical powers, exponentials and logarithms.
Euler-surface tables also retain exact axis labels, including when materialized
as columns through Tables.jl.

Feature evaluation grids such as landscape `tgrid` and image `xgrid`/`ygrid`,
weights, kernel parameters and final feature vectors remain numerical. A
numerical sampling grid can miss a narrow feature; it does not replace or
merge the underlying exact grades.

Projection-based invariants also retain algebraic levels when constructing
their finite target chains and pushforward maps. Their cached barcodes keep
exact endpoints, and distance evaluation subtracts those endpoints before
rounding. A projection pushforward and restriction to a line are different
mathematical constructions; preserving grades does not make their invariants
equal. Numerical quadrature and feature outputs still have their stated
floating-point precision.

## Saving and inspecting

Owned dataset and encoding JSON stores rational coordinates as canonical
rational strings, and real algebraic coordinates by a primitive integer
minimal polynomial and the index of its real root. Loading validates these
records and preserves exact values, orientations and maps. Ordinary floating
coordinate arrays keep their compact numeric storage.

Saved ingestion encodings retain their recorded coordinate meaning. After
loading, `provenance(enc).coordinate_semantics` reports the persisted grade
scale, orientation and grade arithmetic. It does not recreate execution
history that the artifact did not record. A raw graded complex has generic
coordinates; save its filtration specification when you also need to preserve
the construction's geometric interpretation.

Exact algebraic arithmetic costs more than floating arithmetic. It supplies a
correctness guarantee rather than a speed claim. The [multicover guide](multicover.md)
describes bounded-depth models, incremental enumeration and the explicit exact
backend for degenerate inputs.
