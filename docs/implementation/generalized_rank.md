# Computing generalized rank and GRIL

A finite encoding retains how vectors move through the parameter order.
Generalized rank uses those maps to ask which classes persist consistently
across a whole connected region. This can reveal information that no list of
pairwise map ranks retains. TamerOp computes the canonical limit-to-colimit
map, then uses that operation for signed summaries on selected interval
families and for generalized-rank invariant landscapes (GRIL).

The distinction between the finite base and the ambient parameter space matters.
`generalized_rank(M; vertices=I)` restricts the finite module to the selected
labels. `gril(M, grid; centers=..., levels=..., lengths=...)` instead restricts
a specified continuous extension to geometric regions in the plane. Its finite
diagram must be justified by that geometry. Both routes reuse the encoded
spaces and maps; neither reconstructs an input filtration.

The [executable examples](../examples/generalized_rank.jl) build the branching,
signed, and landscape examples used below.

## A branching question that endpoint ranks cannot answer

Take two incomparable sources, each carrying a line, and a common target
carrying a plane. If both source maps land on the same line in the plane,
one class can persist consistently through both branches. If their images
are different lines, no nonzero class can do so. Both modules have stalk
dimensions `(1,1,2)` and both arrow ranks are one. Their generalized ranks on
the three-point region are one and zero.

In the [generalized-rank construction of Kim and Mémoli](https://doi.org/10.1007/s41468-021-00075-1),
the limit records compatible choices of vectors at every vertex. The colimit
identifies vectors related by a structure map. Passing from a compatible
choice to its common colimit class gives the map whose rank we want.

Write the selected stalk direct sum as

$$V=\bigoplus_{a\in I}M_a.$$

For each cover $a\prec b$, let $A_{ab}:M_a\to M_b$ be the structure map.
The constraint matrix $C$ stacks equations $A_{ab}x_a-x_b=0$.
The relation matrix $R$ has columns $\iota_a v-\iota_b A_{ab}v$, where
$\iota_a$ includes one stalk into $V$. Consequently,

$$L=\ker C,\qquad Q=V/\operatorname{im}R.$$

Covers suffice because the input is a functor: all other equations and
relations follow by composition. Query validation checks nonemptiness,
connectivity and order convexity in the original finite poset. It does not
replace the module's functoriality contract for hand-built inputs.

The implementation obtains a basis matrix $B$ for $\ker C$ and a full-row-rank
matrix $q$ whose kernel is $\operatorname{im}R$. The latter is the transpose
of a nullspace basis of $R^T$. Choose any one vertex $a$ and let $E_a$ retain
only its block in $V$. The returned integer is

$$\operatorname{rank}(qE_aB).$$

Connectivity makes this map independent of the anchor: across an arrow the
limit equation and colimit relation identify the two choices, and an
undirected path connects any two vertices. Summing over anchors would multiply
the map by $|I|$; in characteristic dividing $|I|$, that would incorrectly turn
a nonzero comparison into zero. TamerOp uses one anchor.

```text
compatible vectors       one anchor stalk        common quotient class
       ker C  ----------->  M_a  ------------------>  V / im R
                project           include in V, then take the quotient
```

The matrices are assembled sparsely. Exact elimination belongs to
`FieldLinAlg`, using its rational or prime-field machinery. No floating-rank
threshold is introduced here; `RealField` inputs are rejected. The default
return value is the rank. `witnesses=true` retains the chosen limit basis,
colimit projection and comparison matrix, accessible through `limit_basis`,
`colimit_projection` and `comparison_map`. They certify the computation in
the stated coordinates, without claiming canonical bases. A zero stalk in
a connected diagram forces the rank to zero, so scalar queries can stop there.

## Signed reconstruction on a declared family

Suppose the caller supplies a finite family $\mathcal F$ of connected convex
label sets. The total compression contract uses each whole restricted diagram.
Once its ranks $r(I)$ are known, descending inclusion order gives

$$d(I)=r(I)-\sum_{J\in\mathcal F,\ I\subsetneq J}d(J).$$

This is Möbius inversion on the **declared family**, ordered by inclusion,
and guarantees $r(I)=\sum_{J\supseteq I}d(J)$ for $I\in\mathcal F$.
The coefficients use arbitrary-precision integers because repeated subtraction
can exceed the range suggested by individual stalk dimensions.

For three different lines entering a plane, use the family consisting of the
central vertex and the three two-vertex branches. The central rank is two,
each branch rank is one, and the coefficients are `[-1,1,1,1]`. The negative
coefficient removes an overlap in this reconstruction. It does not mean that
the module has a negative summand, or that an interval decomposition was found.

`reconstruct_rank(summary; vertices=I)` rejects undeclared queries. Other
compression systems, such as source/sink compression, are also rejected:
they need their own preservation hypotheses. The implemented contract is
`:total`. In particular, the operation does not enumerate every interval;
[generalized-diagram output sizes can be superpolynomial](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.SoCG.2025.64).

## From a grid encoding to a continuous worm

The [GRIL construction of Xin, Mukherjee, Samaga and Dey](https://proceedings.mlr.press/v221/xin23a.html)
uses expanding closed regions around a center $p$. A worm of length $\ell$
and width $\delta$ is the union of all radius-$\delta$ squares whose centers
run continuously along the anti-diagonal segment of half-length
$(\ell-1)\delta$. Equivalently, its points satisfy

$$|x-p_x|\leq\ell\delta,\quad |y-p_y|\leq\ell\delta,\quad
|x+y-p_x-p_y|\leq2\delta.$$

For $\ell=1$ this is a square. For larger $\ell$ its diagonal sides matter:
a union of only $2\ell-1$ squares is a different region. TamerOp uses the
continuous definition, with exact rational arithmetic for its boundaries.
Floating coordinates are converted to their exact binary rational values;
they are not silently rounded to a decimal grid.

The supported encoding has two strictly increasing axes and the positive
product order. The module is constant on each half-open grid cell, using
its lower-left label. It is zero below either first coordinate and constant
along the upper tails. These extension conventions are part of the input
interpretation, not an inferred boundary condition for every encoding backend.
A compiled wrapper around this grid has the same meaning. General polyhedral
encodings and reversed-axis grids are outside this operation's contract.

### Why a finite restriction computes the continuous rank

Intersect each grid cell with the closed worm, retaining every nonempty
intersection. A nonempty intersection is a rectangle, with possibly open
upper sides, cut by two parallel diagonal half-planes. When $\delta>0$ it
is connected by comparability paths: within its interior, sufficiently small
horizontal and vertical steps connect points, and boundary points connect
to the interior by comparable steps. Degenerate nonempty intersections are
points or horizontal/vertical segments and have the same property. On each
intersection the module and its maps are constant identities.

All limit coordinates in one such fiber must therefore agree, and all its
colimit copies are identified. Contract it to one stalk. Between different
fibers, retain the original grid map whenever some point in the first is
coordinatewise below some point in the second. This finite diagram gives
exactly the same compatibility equations and quotient relations as the
continuous restriction. Composing those relations is harmless even when
a single comparable pair does not realize the composite.

The implementation decides those incidences by interval arithmetic. For a
cell rectangle $[a,b]\times[c,d]$, with the actual upper-endpoint flags, and
worm band $s_-\leq x+y\leq s_+$, its $x$ projection is

$$[a,b]\cap[s_- - d,\ s_+ - c].$$

An open upper endpoint $d$ makes the second interval's lower endpoint open.
The corresponding formula gives the $y$ projection. If two cells differ
strictly in both grid indices, every point in the first precedes every point
in the second. If an index is shared, the projected intervals determine
whether an ordered pair exists. Endpoint flags distinguish an actual contact
from a contact at a grid boundary excluded from that cell.

The contracted diagram enters the same constraint/relation solver as a
selected finite-poset query. Width zero is simply the center stalk. A worm
that enters the zero extension has rank zero.

### Exact critical widths instead of a radius mesh

For fixed $p$ and $\ell$, GRIL returns

$$\lambda_M(p,k,\ell)=\sup\{\delta\geq0:
\operatorname{grank}(M|_{W(p,\delta,\ell)})\geq k\},$$

with value zero when no width is admissible. Worms are nested, and generalized
rank is nonincreasing under inclusion of connected intervals.

The zero-below convention bounds the answer by
$\min(p_x-x_1,p_y-y_1)/\ell$ when the center is above both first coordinates.
Within that bound, cell membership and incidence can change only when
projection endpoints change order. Each endpoint is a minimum or maximum
of affine functions of $\delta$. For the $x$ coordinate these functions are:
axis coordinates; $p_x\pm\ell\delta$; $p_x\pm(\ell+2)\delta$; and
$p_x+p_y-y_j\pm2\delta$. The $y$ list is symmetric.

TamerOp enumerates their positive pairwise crossings within the bound. On
each intervening open interval all membership and incidence tests have fixed
truth values, so the contracted diagram and its rank are constant. A query
at the rational midpoint therefore determines the whole open interval.
Binary search finds the last interval with rank at least $k$; its right
endpoint is the exact supremum. Whether that endpoint itself attains the
rank does not change the supremum. This also handles a positive rank only
at width zero. Queried ranks are reused across levels at the same center
and length; the output retains widths, not elimination workspaces.

For the one-dimensional stalk encoding the quadrant $[0,\infty)^2$, centered
at $(4,6)$, the level-one widths for lengths `1,2,4` are exactly `4,2,1`.
Level two is zero. There is no artificial upper cutoff at the grid's last label.

## Costs and boundaries

For a selected region with $k$ vertices in an $n$-vertex base, the current
validation and cover construction use at most $O(nk+k^3)$ order tests and
$O(k^2)$ order storage. If $D$ is the sum of stalk dimensions, constraints
have $D$ columns and relations have $D$ rows. Exact elimination may densify
and rational coefficients may grow. Sparse assembly does not promise sparse
elimination or a byte-level bound on arbitrary-precision integers.

A declared family requires one rank query per member and quadratic family
inclusion comparisons. GRIL validates the grid order, builds only cells
meeting each worm's bounding box, and may compare quadratically many cells.
Its event construction is quadratic in the number of affine endpoint
expressions. Binary search uses logarithmically many interval probes per
requested level, sharing already computed probes.

`GeneralizedRankBudget` bounds vertices, stalk dimension, potential matrix
entries, order work, output/query counts, and total critical events. A bound
failure raises an error; no partial landscape is presented as complete.
These are bounded exact algorithms for selected questions. There is no claim
of the specialized zigzag performance used by the original GRIL software,
or of arbitrary-encoding support, learned probe selection, or differentiation.

The owner implementations are
[`generalized_rank.jl`](../../src/invariants/generalized_rank.jl) and
[`gril.jl`](../../src/invariants/gril.jl). `Workflow` supplies the encoding
wrappers; both routes preserve the coefficient field and explicit budgets.

[Bibliography](references.md).
