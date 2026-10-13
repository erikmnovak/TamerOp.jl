# Exact coverage-depth windows

A multicover keeps the locations covered by at least a chosen number of balls.
When only a few coverage depths matter, we would like to construct just that
part of the model. The difficulty is at the ends of the requested range:
cells that cross an endpoint can leave a boundary cell there, and that cell
may be needed to join components or fill a cycle. We will see this already
with two sites. The [multicover guide](tutorials/multicover.ipynb) develops the
radius and depth interpretation through computed figures.

`RhomboidFiltration(depth_range=(lo, hi))` represents the inclusive integer
window `lo:hi`. Increasing radius or decreasing depth enlarges the represented
space. At the upper depth `hi`, the construction retains a **terminal cap**,
the horizontal section made by cutting the cells at that depth. It is needed
even when `lo == hi`, so the window contains just one depth.

```julia
import TamerOp as OP

points = OP.PointCloud(reshape(OP.QQ[0, 2], :, 1))
filtration = OP.RhomboidFiltration(depth_range=(1, 1), backend=:exhaustive)
enc = OP.encode(points, filtration; degree=0)
```

Here degree zero homology, over the default rational field, counts connected
components. The two balls are disjoint below radius one and touch at radius
one. The dimension is therefore two before contact and one at contact. In
the cell model, the joining edge lies in the cap. Deleting the original
vertex at depth two and all its incident cells would remove this edge and
leave the wrong number of components.

![The rhomboid for sites 0 and 2 has subset vertices at depths 0, 1 and 2. Its horizontal section at depth 1 joins the two singleton vertices when radius reaches 1.](assets/rhomboid/terminal_cap.svg)

*The lifted coordinates are the sum of the selected sites and their number.
The horizontal cut retains an edge between the singleton vertices. The small
complexes below show its effect on connected components at depth one; they
are schematic views of the same two-site calculation.*

## The represented bifiltration

A native rhomboid is labelled by two disjoint sets of sites $(I,O)$: those
inside a separating sphere and those on it. Write $q=|O|$ and give its cube
coordinates $t=(t_1,\ldots,t_q)\in[0,1]^q$. Its depth is

$$d(t)=|I|+t_1+\cdots+t_q.$$

Intersecting these cubes with the integer horizontal planes and the closed
unit slabs between `lo` and `hi` gives the sliced model. A resulting cell can
belong to several original rhomboids. Its **carrier** is the smallest original
face containing it; the carrier and integer level identify shared sliced
cells without duplication.

Each sliced cell inherits its carrier's constrained minimum radius and its
own minimum depth. These grades ensure that every face is present whenever
its cell is present. At physical parameter $(r,k)$ we take the cells with
radius at most $r$ in the portion $k\leq d\leq\mathrm{hi}$ of the sliced
tiling. Increasing $r$ or decreasing $k$ includes one of these complexes in
another, and the inclusions induce the structure maps after taking homology.

The comparison with multicover needs to preserve these maps. One can form a
**mapping telescope** by retaining the space at depth `hi` and attaching
cylinders for the successive inclusions down to depth $k$. Projection onto
the spatial coordinate gives a natural homotopy equivalence with the
multicover at depth $k$. Here naturality means that the comparisons commute
with the inclusions as radius or depth changes.

Covering this telescope by compatible convex cells gives a **nerve**, the
simplicial complex that records which cover sets have a common intersection.
It contains the terminal complex at depth `hi` and the complexes connecting
successive depths. The restricted sliced model has a cover with the same
nerve. This is the capped version of the sliced rhomboid comparison in
[Corbet, Kerber, Lesnick and Osang, Theorem 12](https://arxiv.org/html/2103.07823v3).
When `lo == hi`, there are no intervening cylinders, but the terminal space
and its nerve are still present. This explains why the cap cannot be omitted.

The argument uses the native general position hypotheses. Degenerate or
repeated labelled sites require the separate `backend=:subdivision_cech`
construction. The window also bounds the query domain whose geometry is
represented; it gives no conclusion outside those depths.

## Radius-preserving pruning

The constructor can discard irrelevant original rhomboids before slicing.
It keeps carriers whose depth interval meets the window; a carrier of positive
dimension must have **anchor depth** $|I|<\mathrm{hi}$. We need to know that
this early pruning still finds the right birth radius for the cells it keeps.

A cell's constrained minimum radius can be attained on a larger cell that
contains it, called a **coface**. For a retained carrier of positive dimension,
every original coface has anchor depth no larger than its own and a depth
interval containing its interval. Such a coface meets the same pruning rule,
so all cofaces that could lower its radius remain available.

The remaining case is a vertex at depth `hi`. If its inside set is nonempty,
a smallest ball enclosing that set must touch an inside site; otherwise the
radius could be decreased. Moving that site from the inside set to the
on-sphere set gives an incident minimizing carrier with anchor below `hi`.
The empty vertex has radius zero. Thus descending radius propagation still
finds the constrained minima, without replacing them by radii of unconstrained
enclosing balls.

The exhaustive backend still examines candidate supports outside the window.
The incremental backend can avoid discovering later depth levels that are
unneeded. How much time and storage this saves depends on the cloud and
window, and the slicing itself can add cells.

## Integral boundaries and the simplex stage

To compute homology we also need the signs of the boundary maps. A slab cell
uses the ordered cube orientation $e_1,\ldots,e_q$, while a horizontal cell
uses $e_1-e_q,\ldots,e_{q-1}-e_q$. For coordinate $i$, its lower and upper
coordinate facets have incidence coefficients $(-1)^i$ and $(-1)^{i-1}$.
The lower and upper horizontal caps of a slab have coefficients $(-1)^q$
and $(-1)^{q-1}$.

Only actual faces of codimension one contribute. When two endpoint
descriptions coincide in a sliced square or interval, they describe one face
and are counted once. The constructor checks $\partial^2=0$ over the integers
before changing coefficient field, so the signed boundaries can then be used
in the requested homology computation.

Requesting `stage=:simplex_tree` subdivides this cellular complex. Each
sliced cell becomes a vertex, and each **flag**, a strict chain of faces,
becomes a simplex with the grade of its largest cell. This is the barycentric
subdivision. Shared faces get identical subdivisions, and restricting to any
filtration parameter commutes with the subdivision. It can contain many more
simplices than the cellular model; flag counts and workspace estimates are
checked against construction budgets before allocation.

`max_dim` is applied after slicing. A rhomboid of dimension $q$ can supply a
cap of dimension $q-1$, so discarding the original rhomboid too early would
lose an admissible cell. As elsewhere, degree $h$ homology generally needs
cells through dimension $h+1$, whose boundaries may kill classes in degree
$h$. Radius grades retain the [exact arithmetic contract](exact_grades.md).

## Incremental enumeration

`backend=:incremental` discovers native rhomboids across depth levels in
affine dimensions zero through three. `backend=:exhaustive` enumerates
supports in any affine dimension. Both feed the same radius and boundary
construction, so they describe the same multicover under their hypotheses.
The requested and executed choices are recorded in provenance.

The default `backend=:auto` selects incremental enumeration for affine
dimension two, at least 32 sites and an explicit depth window ending at two.
Other requests use exhaustive enumeration. Incremental construction still
processes prerequisite depths below `lo`; degeneracy there can prevent that
route even if exhaustive enumeration can construct the narrower window.
Encountered native degeneracy causes an error, without perturbing the sites
or silently switching to subdivision–Čech.

The [construction account](implementation/rhomboid_construction.md#discover-cells-across-depth-levels)
follows the lifted subset coordinates, exact lower hulls and stopping rule.
Compatible requests can reuse that geometry through an explicit
[session cache](tutorials/multicover.ipynb#reusing-exact-geometry).

## When the selected complex is empty

A depth window and radius cutoff may leave no cells. The graded complex and
simplex tree still retain their two parameter coordinates and exact algebraic
radius type, without adding an artificial vertex. Dataset JSON preserves this
shape through its `empty_grade_type` record.

To encode an empty dataset, supply explicit axes: there are no cell grades
from which to infer a grid. Every stalk and structure map on that grid is
zero. As for a nonempty window, this records the module on the domain you
specified.
