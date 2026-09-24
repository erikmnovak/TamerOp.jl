# Exact coverage-depth windows

`RhomboidFiltration(depth_range=(lo, hi))` represents the inclusive integer
coverage-depth window `lo:hi`. Radius increases and depth decreases. The
construction retains the horizontal cap at depth `hi`, including when
`lo == hi`.

```julia
using TamerOp

points = PointCloud(reshape(QQ[0, 2], :, 1))
filtration = RhomboidFiltration(depth_range=(1, 1), backend=:exhaustive)
enc = encode(points, filtration; degree=0)
```

Here the degree-zero dimension is two below radius one and one at radius one.
The cap contains the joining edge. Removing the original depth-two vertex and
all its incident cells would lose that edge and give the wrong answer.

## The represented bifiltration

Write an original rhomboid as `(I,O)`, where `I` and `O` are its disjoint inside
and on-sphere site sets. With ordered coordinates `t in [0,1]^q`, where
`q = |O|`, its depth is `|I| + sum(t)`. The implementation intersects rhomboids
with integer horizontal planes and closed unit slabs between `lo` and `hi`.
It identifies shared intersections by their smallest original carrier and
integer level.

Each resulting cell receives its carrier's constrained minimum radius and its
minimum depth. Consequently its faces are present whenever it is present.
At `(r,k)` the active complex is the part of this sliced tiling in
`k <= depth <= hi` whose cells have radius at most `r`. Structure maps are the
literal inclusions of these complexes.

The proof uses a capped mapping telescope and the sliced-rhomboid nerve
comparison of Corbet, Kerber, Lesnick and Osang, Theorem 12. The telescope
retains the depth-`hi` space and attaches the cylinders of the inclusions down
to depth `k`; projection onto the spatial coordinate is a natural homotopy
equivalence to the depth-`k` multicover. Its nerve contains the terminal
order-`hi` complex as well as every intervening mixed-order complex. Compatible
convex-cell covers give the same nerve for the restricted sliced model.
This proves compatibility of persistence maps as well as individual spaces.
The terminal component is essential even when the union of intervening
cylinders is empty. See
[Computing the multicover bifiltration](https://arxiv.org/pdf/2103.07823).

This application requires the native general-position hypotheses. The
separate `backend=:subdivision_cech` construction supports degenerate and
repeated labeled sites. A depth window also restricts the represented query
domain; it does not certify answers outside that window.

## Radius-preserving pruning

Native geometry retains only carriers whose depth interval meets the window,
with positive-dimensional carriers anchored strictly below `hi`. This happens
while constructing the geometry. It does not build the full tiling and filter
its final module.

For a positive-dimensional retained carrier, every original coface has an
anchor no larger than its anchor and a depth interval containing its interval.
All cofaces that can improve its constrained radius therefore survive the
same pruning rule. The exceptional case is a vertex at depth `hi`. A minimum
ball enclosing its nonempty inside set must touch an inside site; otherwise
its radius could decrease. Moving that site onto the boundary supplies an
incident minimizing carrier anchored below `hi`. The empty vertex has radius
zero. These observations justify descending radius propagation on the retained
carriers without replacing constrained minima by unconstrained enclosing-ball
radii.

The exhaustive backend still examines candidate supports outside the window.
The incremental backend also avoids discovering unnecessary later depth
levels. Storage and time savings depend on the requested window and input;
integer slicing itself adds cells.

## Integral boundaries and the simplex stage

Slab cells use the ordered cube orientation `e_1,...,e_q`. Horizontal cells use
`e_1-e_q,...,e_(q-1)-e_q`. For coordinate number `i`, the lower and upper
coordinate facets have incidence `(-1)^i` and `(-1)^(i-1)`. Slab lower and upper
horizontal caps have incidence `(-1)^q` and `(-1)^(q-1)`. Only actual
codimension-one faces contribute. Coincident endpoint descriptions in a sliced
square or interval are counted once. The constructor verifies `boundary² = 0`
over the integers before changing coefficient field.

`stage=:simplex_tree` explicitly constructs the barycentric subdivision: one
vertex per sliced cell, one simplex per strict face flag, and the grade of the
largest cell on that flag. Shared faces acquire identical subdivisions, and
subdivision restricts at every filtration parameter. Flag counts and workspace
estimates are checked against construction budgets before allocation. This
stage can be much larger than the native cellular complex.

`max_dim` applies after slicing. A `q`-dimensional original rhomboid can supply
an essential `(q-1)`-dimensional cap. As for other skeleton requests, degree
`h` homology generally needs cells through dimension `h+1` to retain its
boundaries. Radius grades retain the [exact arithmetic contract](exact_grades.md).

## Incremental enumeration

`backend=:incremental` computes native rhomboids in affine dimensions zero
through three. `backend=:exhaustive` retains the dimension-generic support
enumeration route. These choices feed the same carrier-radius and boundary
construction; they do not define different multicover approximations. The
result provenance records both the requested and executed backend.

The measured default `backend=:auto` selects incremental enumeration only for
affine dimension two, at least 32 sites, and an explicit depth window ending
at two. Other requests use exhaustive enumeration. Local paired measurements
found both wins and regressions; small and full-depth weighted hulls did not
show a consistent improvement.

For a subset `Q` of `k` sites, form the exact lifted point

```text
(mean(a for a in Q), mean(dot(a,a) for a in Q)).
```

Its lower convex hull gives a weighted Delaunay mosaic. Start with singleton
labels at depth one. Each recovered rhomboid supplies subset vertices for
later depths, so the algorithm does not enumerate all subsets of the input.
The intersection and union of a first-generation cell's labels recover its
inside and on-sphere sets. Completeness follows from Edelsbrunner and Osang's
earlier-level vertex discovery theorem; see Theorem 5, Lemma 2 and Algorithm 1
in [A Simple Algorithm for Higher-order Delaunay Mosaics and Alpha
Shapes](https://arxiv.org/pdf/2011.03617v1).

The implementation uses CDD's exact rational convex-hull conversion. It adds
an upward vertical ray to the lifted hull and selects facets with a negative
final normal coefficient. This also handles a simplex whose lifted vertices
all lie in one affine plane. CDD returns whole facets, so nonsimplicial
higher-generation cells need no arbitrary triangulation. Every recovered top
rhomboid is checked against its original exact sphere classification.

For a cloud in a proper affine subspace, an injective coordinate projection
reduces only the horizontal coordinates of the lifted hull. The final lifted
coordinate still uses the original full squared norm. This preserves the
Euclidean metric of the input. Lifted means, facet incidence and sphere
classification never pass through floating-point arithmetic.

The algorithm stops after the requested upper depth, or earlier when no new
top rhomboids are possible. It still processes prerequisite depths below
`lo`; degeneracy there can prevent incremental enumeration even if exhaustive
enumeration could construct the narrower window. Native construction rejects
encountered degeneracy and never perturbs the sites or silently selects
subdivision-Cech.

Here *incremental* means discovery across depth levels. CDD recomputes each
level's weighted hull. Runtime depends on that hull computation, exact integer
sizes, and constrained-radius calculations; no output-linear bound is claimed.
Convex-hull computation is an allowed weighted-Delaunay implementation in
[Section 4.5 of Computing the multicover
bifiltration](https://arxiv.org/html/2103.07823v3).

When benchmarking exhaustive and incremental execution, use identical data
and depth windows. Track retained original carriers, emitted cells and
exact-critical-grid H0 ingestion separately. Full and capped models have
different cell structures, so their storage comparisons are separate from
equality checks between enumeration backends. Label first-operation and warm
timings separately, and include ordinary package loading when measuring a
fresh process. Julia allocation counters exclude CDD/GMP's native working
memory. Compatible requests can reuse compiled geometry through the
[session cache](multicover.md#reusing-exact-geometry).

A depth window and radius cutoff can produce an empty complex. Empty graded
and simplex-tree stages retain their two parameter coordinates and exact
algebraic radius type; they do not acquire an artificial vertex. Dataset JSON
round trips preserve this shape through a closed `empty_grade_type` record.
To encode an empty dataset as a module, supply explicit axes: there are no
cell grades from which to infer a parameter grid. The resulting stalks and
structure maps are zero on that grid.
