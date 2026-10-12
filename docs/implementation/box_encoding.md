# Axis-aligned box encoding

How can a grid encode a module supported on a line or a single point? Its cells
must retain closed boundaries as well as the open regions between them. The
finite encoding then stores the same spaces and maps on a finite poset, even
when a nonzero space occupies no ambient volume.

This account follows the box backend from generator thresholds to that finite
description. The [indicator-presentation lesson](../indicator_presentations.md)
explains why a matrix between upset and downset indicators describes a module;
the [finite-encoding introduction](../finite_encodings.md) explains recovery
through a classifier.

The finite-encoding viewpoint comes from Ezra Miller's theory of modules over
posets, discussed in that introduction. The event grid below is the package's
specialized construction for axis-aligned indicator presentations.

## A closed point needs three cells

Consider one parameter, the upset $U=[0,\infty)$, the downset
$D=(-\infty,0]$, and the one-by-one coefficient matrix $[1]$. At a parameter
$x$, the represented vector space is the image of the active matrix block.
That space is one-dimensional only at $x=0$.

The classifier records $y=1$ when $x\in U$ and $z=1$ when $x\notin D$.
Both bits increase with the parameter:

| Parameter cell | $(y,z)$ | Fiber dimension |
| --- | --- | --- |
| $x<0$ | $(0,0)$ | $0$ |
| $x=0$ | $(1,0)$ | $1$ |
| $x>0$ | $(1,1)$ | $0$ |

Crossing zero therefore has two events. First the birth becomes active; then
the death ceases to be active. Keeping the intermediate state retains the
point module. Merging these events would lose it.

Internally, the sorted axis wall sequence is `[0.0, 0.0]`. The cell between
equal walls is their singleton. At a birth-only wall, equality belongs to the
cell on the right; at a death-only wall it belongs to the cell on the left.
Those walls need only one event because the boundary has the neighboring
cell's signature. The repeated-wall representation thus adds cells precisely
where the closed birth and death conditions require them.

Products of these axis cells retain faces and corners. For example,
$U=[0,\infty)^2$ and $D=(-\infty,0]\times(-\infty,2]$ with coefficient $[1]$
give a module supported on the segment $\{0\}\times[0,2]$. Along the segment
the nonzero structure maps are identities; maps entering from or leaving to a
zero space are zero. Ambient area alone cannot recover that information.

## Traverse events, then identify equal signatures

The encoder visits the cell product with its first axis varying fastest. For
each generator it maintains the number of satisfied coordinate inequalities.
Moving through one axis event updates only generators with that threshold.
The reverse updates restore the state when the traversal carries to the next
axis. It does not test every generator against every new cell from scratch.

The current $(y,z)$ bits are packed into machine words and used as a dictionary
key. Equal signatures share a region label, and the cell table retains the
label for each visited cell. The finite order is componentwise inclusion of
the two bit families. The resulting fringe presentation carries the original
coefficients wherever its indicator intersection is nonempty. Its stalks are
images of active matrix blocks; induced maps use their actual image bases.
This preserves the coefficient field and the maps, as well as dimensions.

`poset_kind=:signature` retains the signature order representation.
`poset_kind=:dense` materializes its relation. Neither choice removes boundary
strata. The field controls linear algebra, while the geometric thresholds
remain those stored by `BoxUpset` and `BoxDownset`.

## Classify exact coordinates and keep certified witnesses

Point lookup searches the wall sequence on each axis, corrects equality for a
death event, and reads one cell-table entry. At a repeated wall, the last equal
entry is the death event: the correction selects the singleton immediately
before it. Tuple, vector and batched queries use this same rule. A uniform-axis
arithmetic estimate is checked against the stored walls before it is used;
rational queries retain their exact side of a wall.

Representatives are exact rational witnesses relative to the stored Float64
thresholds. Bounded cells use the rational midpoint of their two walls;
unbounded cells use an exterior rational point. Equal walls give the singleton
itself. These witnesses are prepared once per axis and reused during traversal.
This also handles consecutive floating-point thresholds: their open interval
has rational interior points even though it has no Float64 interior point.

`check_box_encoding_map` checks axis events, table shape, lookup metadata and
the signatures of stored representatives. JSON serialization records rational
representatives exactly. Converting a witness to Float64 for drawing can move
it to a different boundary, so drawing coordinates must not replace query
coordinates.

## Geometry retains zero-volume regions

Volume integrates full-dimensional cells, so a point or a face has zero ambient
volume. Bounding boxes and extent queries include the lower-dimensional cells
and distinguish them from an empty intersection. Endpoint ownership decides
whether a query window actually meets a cell or merely its excluded endpoint.

Boundary measure counts each geometric face patch once, using its left, right
and boundary labels. Thus a segment in the plane has boundary measure equal
to its length, and an isolated point on the line contributes one boundary
point. Face-only patches have no unique outward side: the boundary breakdown
marks them `:stratum` and returns a zero normal. Adjacency records contacts of
region closures with positive codimension-one measure, including contacts
through a separately labeled face. Higher-codimension contacts contribute zero
to that measure.

For principal directions of a zero-volume region, the backend normalizes
measure on its cells of greatest intrinsic dimension: length on segments, area
on faces, and counting measure on isolated points. Geometry summaries return
floating-point coordinates and moments; the exact classifier and witnesses
remain available for mathematical membership queries.

Mean width measures a different quantity: the average length of a projection.
For a planar segment of length $L$, it is $2L/\pi$, although the segment's
geometric boundary measure is $L$. The convex planar shortcut accounts for
this distinction; the general cell method computes directional extents.

## Costs and the general polyhedral route

If axis $i$ has $b_i$ distinct birth thresholds and $d_i$ distinct death
thresholds, its cell count is $b_i+d_i+1$. A shared value counts in both
families. The direct table therefore has
$C=\prod_i(b_i+d_i+1)$ entries, although equal signatures may give far fewer
regions. `max_regions` bounds this grid-cell work before allocation; it does
not merely bound the number of distinct output labels. Growth with the number
of parameters remains a real limitation.

`backend=:auto` can select this construction for finite closed axis-aligned
generators, including shared thresholds. General polyhedral presentations use
the polyhedral route when they have sloped boundaries or unions, or when
converting their exact thresholds to Float64 would lose information. Input
generator pieces must be closed; strict inequalities arise inside the
polyhedral subdivision, but strict input pieces are rejected. A rational
representative certifies a cell of the stored box
input; it does not restore precision already lost when a caller constructed
that input from rounded thresholds.

[Implementation bibliography](references.md)
