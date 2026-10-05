# Exact matching distance in a finite window

Once two modules have compatible finite encodings, we can compare what they
show along increasing lines through parameter space. Each line uses the retained
spaces and maps to give a one-parameter barcode. The bottleneck distance to
another barcode minimizes the largest cost of matching endpoints, allowing a finite
interval to be discarded at half its length. Matching distance combines these
comparisons over all positive-slope lines with a direction-dependent weight.
This summarizes differences between the two retained modules; see the
[finite-encoding introduction](finite_encodings.md) for the object being compared.

`matching_distance_exact_2d` computes the supremum of weighted bottleneck
distances over all positive-slope lines, with both slice barcodes clipped to
the same finite rectangle. The optimization uses exact arithmetic; the public
answer is its conversion to `Float64`.

```julia
opts = InvariantOptions(box=([xmin, ymin], [xmax, ymax]), threads=false)
d = matching_distance_exact_2d(encM, encN; opts)
```

The two encodings must use the same classifier and finite poset. An omitted
window is inferred by `encoding_box`; choose it explicitly when it is part of
the mathematical question. `:L1` with `:lesnick_l1` and `:Linf` with
`:lesnick_linf` give the same weighted quantity. The API rejects other
normalization/weight pairings.

## Scope and hypotheses

The classifier must be constant on the open cells of its declared axis grid
and order preserving, including at coordinate boundaries. Every open cell
met by the window must be represented. The two modules must be functors on
the same poset over the same field. These are mathematical input hypotheses;
the optimizer checks encountered region labels and comparability, but does
not prove global functoriality or a custom classifier's correctness.

Native box encodings and positively oriented `GridEncodingMap` classifiers
are supported. A custom classifier must faithfully locate exact interior
query points: rational points for rational coordinates, real algebraic points
when algebraic coordinates or window endpoints are supplied. A locator that
silently rounds those points to `Float64` does not meet this contract.
Polyhedral classifiers and rounded lattice classifiers are outside this
optimizer's scope and are rejected. The coordinate orientations must be
positive; see [exact grades](exact_grades.md) for expressing rhomboid
coordinates as `(radius, -depth)`.

The window has finite, ordered endpoints. A zero-width or zero-height window
has value zero. For the rational arrangement route, the window also has to
fit its floating query storage; the algebraic coordinate route retains exact
window storage. A finite floating coordinate means its represented binary
value, not an intended decimal value. Geometry uses `Rational{BigInt}` or
`AlgebraicReal`, including predicates, intersections and cost comparisons.
Coefficient-field linear algebra retains its separate contract: exact fields
give exact barcode ranks; `RealField` uses its numerical rank policy.

Every essential bar is cut at the window exit. Thus a finite answer is not a
claim that the unrestricted matching distance is finite. The windowed and
unrestricted quantities agree if the window contains both modules' support.
For example, the constant rank-one module versus zero has windowed distance
`min(width, height)/2`, although its unrestricted slices have essential bars.
Enlarging the window changes the question.

The final `Float64` conversion is a numerical output boundary. A positive
exact result below its representable range can round to zero, and a large
finite result can round to infinity. Within that range the result is rounded;
the API does not expose an exact scalar or a certified interval. Exactness
here concerns exhaustive optimization before that conversion.

## Proof of coverage

There are infinitely many lines through the window, so checking a chosen sample
would not establish the maximum. The proof below divides the line parameters
into finitely many pieces on which both barcode events and matching costs have
fixed formulas. It then explains why checking the vertices of those pieces
suffices under the preceding hypotheses.

Write the window as `[xmin,xmax] × [ymin,ymax]`, and initially assume positive
width and height. In the shallow chart parameterize a line by

```text
y = q*x + h,       0 < q <= 1,
u = q*x,           x = u/q, y = u+h.
```

The parameter `u` includes the weight for either supported normalization.
Translating its origin translates both barcodes equally and changes no cost.
A vertical crossing at `x=a` has `u=q*a`; a horizontal crossing at `y=b` has
`u=b-h`. These are affine functions of `(q,h)`.

**Geometric cells.** Lines meeting the window form the polygon

```text
0 <= q <= 1,
ymin - q*xmax <= h <= ymax - q*xmin.
```

Include the window sides among the coordinate cuts. Vertical event order is
fixed for `q>0`, as is horizontal event order. Consequently every possible
event-order change lies on one of the mixed equalities `q*a = b-h`. Cutting
the polygon by all these equalities fixes the entry event, exit event, and
sequence of open cells on each full-dimensional piece. Window/grid
intersections are necessary even when no window corner is a grid vertex.
An exact interior point identifies the chain. Its finite index barcode is
constant throughout the piece; replacing index endpoints by the corresponding
events gives affine birth and death functions. Repeated adjacent region
labels may be compressed because the corresponding maps are identities.

**Cost switches.** For two finite bars, the cross-match cost is

```text
max(b - b', b' - b, d - d', d' - d).
```

A bar's diagonal cost is `(d-b)/2`. Collect all these affine expressions for
both barcodes, including zero. Multiplicities give distinct matching vertices
even when their endpoint expressions coincide. The augmented bipartite graph
has one diagonal copy per bar and a complete zero-cost diagonal-to-diagonal
block, so every allowed partial matching with diagonal deletions is included.

Refine the geometric polygon by every pairwise equality of these affine
expressions that crosses its interior. Their ordering is fixed in each
resulting open polygon. The largest cost in any fixed perfect matching is
therefore one fixed expression there; the smallest such maximum over the
finitely many matchings is also one fixed expression. Hence the bottleneck
distance is affine on each refined polygon. This accounts for switches both
between cross matches and between cross matches and diagonal deletion.

**Vertices suffice.** An affine function on a compact polygon attains a
maximum at a vertex. Those vertices are geometric vertices, switch/border
intersections or switch/switch intersections. The implementation includes all
three. Parallel lines need no intersection; a switch merely touching a
vertex or coinciding with a border introduces no new vertex. Exact tests
discard intersections outside the polygon. Zero and duplicate costs require
no perturbation or tolerance.

**Boundary slices.** On a geometric wall, some successive crossing times
coalesce. Remove the intervening zero-length segments from a neighboring
chain. Functoriality identifies the composite through those segments with
the map between the retained endpoints. Thus the rank invariant, and hence
the nonzero-length barcode, is that of this coarsened chain. Approaching the
wall from either neighboring polygon gives this same barcode after deleting
zero-length bars. A value supported only at the crossing time is ephemeral
and has zero bottleneck cost. This explains why evaluating neighboring
affine formulas on their closures gives the correct boundary value; it is
not an assumption about a floating locator's boundary tie-breaking. It also
explains why order preservation and module functoriality are essential
hypotheses. At a line tangent to the window, all clipped bars have zero
length.

**Axes and steep slopes.** Every bar in the shallow chart has weighted
lifespan at most `q*(xmax-xmin)`. Deleting all bars bounds the distance by half
that number, so the `q=0` limit is zero uniformly in offset. Swap the axes and
repeat to cover slopes greater than or equal to one and the other axis
limit. The diagonal occurs in both charts. The two compactified charts and
their vertices therefore cover the entire stated supremum.

This argument applies over the rational and real algebraic ordered fields:
all constructed coordinates, affine intersections and comparisons stay
exact. The finite algorithm terminates if its work budget permits it.

## A module pair whose maximum needs a cost switch

Let `I(R)` denote the rectangle interval module, with identity maps within
the rectangle and zero maps outside it. Endpoint inclusion does not affect
the bottleneck values below.
In the window `[0,4] × [0,4]`, take

```text
M = I([0,3] × [1,4]) ⊕ I([1,4] × [0,4]),
N = I([1,3] × [1,3]).
```

Their exact weighted matching distance is `5/4`.
To see the upper bound, parameterize any positive line in weighted units as
`x=a*t`, `y=b*t+h`, with `a,b>=1` and `min(a,b)=1`. The rectangle of `N`
lies in both rectangles of `M`. Each endpoint moves by at most one weighted
unit on passing from either `M` interval to the `N` interval: the relevant
coordinate differences are at most one and the divisors `a,b` are at least
one. Whenever the `N` interval is present, matching it to either `M` interval
therefore costs at most one.

If the `M` intervals have lengths `l1,l2`, their endpoint formulas give

```text
l1 <= 3/a - (1-h)/b,
l2 <= (4-h)/b - 1/a,
l1 + l2 <= 2/a + 3/b <= 5.
```

Match `N` to the longer interval and delete the shorter one, at cost at most
`max(1, (l1+l2)/4) <= 5/4`. If `N` is absent, each existing `M` interval
has length at most two by the same endpoint-displacement bounds, so deleting
them costs at most one.

On the diagonal line `y=x+1/2`, the diagrams are

```text
M: [1/2,3], [1,7/2],
N: [1,5/2].
```

At least one `M` interval must be deleted, costing `5/4`; matching the other
to `N` costs at most one. This attains the bound. More generally, on
`y=x+h`, `0<=h<=1`, the exact distance is
`min(1+h/2, 3/2-h/2)`. Its maximum at `h=1/2` switches which `M` interval
is deleted. No integer grid event coincides on this line: it is inside a
barcode-combinatorics cell. The diagonal is a seam between the implementation's
two slope charts, not a boundary in the space of positive lines. Thus a
geometric-event-only optimizer misses the actual source of the maximum.

## Literature and limits

Critical barcode events alone do not determine the maximum: optimal matching
switches also matter, as explained by
[Brooks et al., *Switch Points of Bi-Persistence Matching Distance*](https://arxiv.org/abs/2312.02955).
[Bjerkevik and Kerber's exact algorithm](https://jocg.org/index.php/jocg/article/view/3341)
uses arrangements and a decision procedure with different complexity bounds.
The implementation here is an exhaustive affine refinement for the finite
window and coordinate-cell hypotheses proved above. It does not implement
either paper's complete algorithm or claim their asymptotic performance.

`max_candidates` conservatively charges geometric splits, barcode expansion,
cost pairs and intersection pairs, including rejected intersections. Exceeding
it throws rather than returning a partial or sampled maximum. The arrangement
has a separate `max_cells` budget. These are combinatorial work guards, not
wall-clock or peak-memory bounds. Pass `cache=session` with a retained
`SessionCache()` to reuse the arrangement and module index-barcode caches
across compatible encoding queries. The optimizer's exact geometric cells
and pair-dependent cost switches are rebuilt for each exact call. Use the
explicit sampled API for exploratory queries when this exhaustive optimizer
is too costly.
