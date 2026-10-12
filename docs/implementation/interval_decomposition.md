# Interval multiplicities and certified decomposition

A finite encoding retains enough algebra to ask whether a particular interval
occurs as a direct summand. Restricting to that interval loses boundary maps,
so its generalized rank cannot answer this question. We instead compute maps
both into and out of the interval module on the **whole finite base**.
Their compositions detect the part that admits a retraction.

The [mathematical lesson](../decomposition.md) explains the distinction;
the [usage guide](../tutorials/decomposition.ipynb) constructs a splitting.
This account follows the computation from a single interval to an exhaustive,
bounded search and its optional simultaneous maps.

## A small pairing measures the multiplicity

Fix a nonempty connected convex interval $I$, and let $L=k_I$.
Because $\operatorname{End}(L)=k$, composition gives a bilinear pairing

$$
\operatorname{Hom}(M,L)\times\operatorname{Hom}(L,M)\longrightarrow k,
\qquad (g,f)\longmapsto gf.
$$

Choose bases $f_1,\ldots,f_a$ and $g_1,\ldots,g_b$. The matrix $C$ is defined
by $g_jf_i=C_{ji}1_L$. At any one vertex $q\in I$ this is just the product
of a row and a column. The composition is scalar on all of $I$, so the anchor
is sufficient even when the interval branches.

The rank of $C$ equals the multiplicity. A nonzero pairing yields maps
$f:L\to M$ and $g:M\to L$ whose composite can be rescaled to $1_L$,
splitting off one copy. Relative to this splitting, the pairing has a
one-dimensional nondegenerate block plus the pairing of the complement.
Induction gives exactly $\operatorname{rank}C$ copies. If the residual pairing
is zero, another split copy would provide a nonzero composite, a contradiction.
This argument works over the supported exact rational and prime fields; it
requires no algebraic closure.

TamerOp obtains the two Hom bases from its existing naturality solver. The
unknowns are the component entries on $I$, but equations use every ambient
cover touching those entries. For $M=(k\xrightarrow{1}k)$ and $I=\{2\}$,
the incoming edge forces every map $M\to k_I$ to be zero. Omitting this edge
would incorrectly report a copy. Exact Hom equations are streamed; bases are
materialized for the composition and discarded after a scalar query.

[Asashiba–Liu](https://arxiv.org/abs/2411.11594) develops the multiplicity
problem and structure-map/presentation formulas. The local engine uses the
composition-pairing argument above, rather than those specialized rank formulas
or filtration-level essential covers. It shares the degree-zero endpoint
pruning described next. No external decomposition software is invoked.

## Use endpoints to eliminate impossible summands

At vertex $v$, the top is the quotient by the sum of incoming images, and the
socle is the intersection of outgoing kernels. It is enough to use covers:
longer incoming images and outgoing conditions factor through them. Their
dimensions are therefore

$$
t_v=\dim M_v-\operatorname{rank}[M_{u,v}]_{u\lessdot v},\qquad
s_v=\dim M_v-\operatorname{rank}\begin{bmatrix}M_{v,w}\end{bmatrix}_{v\lessdot w}.
$$

A copy of $k_I$ contributes one top dimension at each minimum of $I$ and one
socle dimension at each maximum. Thus the multiplicity is bounded by the
smallest corresponding $t_v$, $s_v$, and stalk dimension on $I$.
A zero bound bypasses both Hom computations. This is the degree-zero critical
set criterion of Asashiba–Liu, Lemma 3.39; higher resolution-level critical
sets are not computed here.

For a complete search, enumerate nonempty antichains $A$ in the top support
and $B$ in the socle support. Every candidate interval has the form

$$
I=\uparrow A\cap\downarrow B.
$$

Keep only connected sets with actual minima $A$, maxima $B$, and nonzero
stalks throughout. Such intersections are convex. The endpoint sets determine
the interval uniquely, and every interval summand passes these tests. This
proves completeness without enumerating every subset of the ambient base.
A constant module on an 80-vertex chain has one endpoint pair. A broad
antichain can still have exponentially many endpoint subsets; explicit
antichain and endpoint-pair budgets bound that work.

## Recover simultaneous maps only when requested

Select independent columns $J$ of $C$, then independent rows $K$ of $C[:,J]$.
The square minor $D=C[K,J]$ is invertible. The selected incoming maps form
$i:L^{\oplus r}\to M$; multiplying the selected outgoing maps by $D^{-1}$
gives $p:M\to L^{\oplus r}$ with $pi=1$.

Let $j:R\to M$ be the kernel inclusion of $p$. The complement projection is
the unique solution of

$$
jq=1-ip.
$$

The exact full-column solver checks this equation in the original coordinates;
see [exact rational coordinates](qq_coordinates.md) for its rational route.
It gives $qj=1$, $pj=0$, $qi=0$, and $ip+jq=1$. These equations certify a
module splitting, not only a dimension count.

Different interval supports can have nonzero Hom spaces between them.
Computing independent inclusions into the original module and concatenating
them would not ensure orthogonality. Instead, the witness route extracts
each interval from the current complement and composes its maps through all
previous complement maps. The final summand uses support order, with one block
per positive multiplicity. Its complement may remain decomposable into
non-interval modules.

The count-only route evaluates every candidate against the original module
and subtracts its dimension contribution. Krull–Schmidt uniqueness ensures
that the reported interval copies coexist as a direct summand. It avoids all
kernel modules and retained splitting maps. Both routes report identical
multiplicities and residual dimensions.

## Scope, validation and cost

Inputs are checked for path independence on their complete finite base.
A chosen family certifies multiplicities only for those intervals. A zero
residual already proves interval-decomposability; a nonzero residual proves
its failure only after the complete pruned search. Budget exhaustion raises
an error without returning a partial negative answer.

With $d=\sum_{v\in I}\dim M_v$, each Hom basis and the composition matrix has
at most $d^2$ logical entries. The bound includes potentially dense output
from sparse elimination. Path validation is bounded by the largest squared stalk dimension.
Top/socle matrices and the finite order table have
separate preallocation checks against the same entry limit. Witnesses also
check the sum of squared stalk dimensions before retaining simultaneous maps.
These are per-workspace limits, not a byte-accurate peak-memory guarantee;
exact rational coefficient growth can dominate arithmetic cost.

The finite-module workflow uses the encoding session cache when materializing
an encoded input. Top/socle data are shared within one decomposition call.
There is no persistent cache of multiplicities or mutable Hom results.
Treat source modules and retained maps as read-only during computation;
validation recomputes the multiplicities and checks all retained naturality,
splitting and reconstruction equations against the current source.

The owner is `IntervalDecompositions`, with task forwarding in `Workflow`.
The result's provenance identifies the finite category and exact coefficient
field. Pullback carries a supplied splitting along a classifier; completeness
in an ambient category is a separate mathematical question.

[Bibliography](references.md)
