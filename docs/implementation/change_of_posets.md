# Change of poset, refinement and Kan extensions

A finite encoding separates a parameter classifier from a module on its finite
target. Changing the target therefore requires two kinds of data: a map between
finite posets and linear maps relating the represented modules. Keeping only
the number of regions or their stalk dimensions loses the information needed
to verify a comparison.

This account follows those data through restriction, a realized refinement and
Kan transport. The [usage notebook](../tutorials/compare_encodings.ipynb) explores
the resulting diagrams; the [reference](../src/reference/change_of_posets.md)
specifies inputs, returns and inspection options. The mathematical distinction
between an ambient module and a chosen finite representation belongs to
[the category lesson](../math_categories.md).

## The finite base is retained, not inferred from a drawing

An `EncodingMap` stores `Q`, `P` and a vector of target labels, one for each
source vertex. Monotonicity requires $\pi(q)\leq\pi(q')$ whenever $q\leq q'$.
The implementation checks the labelled order rather than presuming that label
numbers themselves are ordered. A Hasse layout is a schematic drawing of this
order; it is not a parameter embedding.

A `PModule` stores its field, stalk dimensions and cover matrices. Every matrix
has source columns and target rows. Relations ensure that composites through
different cover paths agree. Restriction uses the original stalk at $\pi(q)$
and the original structure map at $\pi(q)\leq\pi(q')$. A source cover may map
to an equality or to a longer target chain, so copying only target cover keys
would be incorrect. Equal labels use an identity, including its empty shape
at a zero-dimensional stalk.

The comparison viewer normalizes each branch to the same base. For a refinement
it uses the two projections; for a single map $Q\to P$ it uses the identity of
$Q$ and $\pi$. Thus the two inspected modules are always restrictions on a
declared common finite poset. It never needs a geometric embedding to obtain
these maps.

## Realized pairs and the product answer different questions

For finite classifiers $a:S\to P$ and $b:S\to R$, `joint_encoding` forms

$$J=\{(a(s),b(s)):s\in S\}\subseteq P\times R.$$

It deduplicates and lexicographically labels these pairs, constructs their
induced product relation, and returns the surjection $S\to J$ and both
projections. Checking the result includes surjectivity, distinct projected
pairs, monotonicity and the induced order. The construction avoids allocating
the full product: pair collection is linear in the finite source before sorting,
while its explicit relation needs quadratic space/time in the number of
realized pairs, plus underlying order-query costs.

Without that witness, common refinement on different finite bases constructs
an abstract product and restricts both modules to it. Product-specific kernels
can reuse repeated blocks, but the output still means the entire product.
A realized joint result instead constructs the two restrictions on $J$.
Neither route silently intersects continuous classifier fibers. A finite
sample of an infinite parameter space is not a completeness certificate.

The pair panel retains this distinction. Outlines mark pairs included in the
actual comparison base; finite-source witnesses determine realized/unrealized
colors. If the Cartesian background exceeds its budget, it is omitted with
an explicit notice while the actual common pairs remain. No missing drawing
cell is interpreted as a zero stalk.

## Kan extensions retain chosen limit and colimit coordinates

The categorical background is [Riehl's account of Kan extensions and
adjunctions](references.md#kan-extensions-and-adjunctions). The finite matrices
and inspection certificates described here are constructed within TamerOp.

For $M:Q\to\mathrm{Vect}$ and a monotone $\pi:Q\to P$, the stalk of the left
Kan extension at $p$ is the colimit over $\pi(q)\leq p$. The finite construction
starts with a direct sum of contributing stalks and quotients by relations
identifying each vector with its image along the diagram maps. The right Kan
stalk is the limit over $p\leq\pi(q)$: a kernel of compatibility equations in
the corresponding product of stalks. The field's exact or numerical linear
algebra supplies the quotient/kernel coordinates.

Target structure maps come from inclusions between these indexing diagrams,
then descend to quotient coordinates or restrict to kernel coordinates.
Consequently a change in basis is expected even when dimensions agree.
The unit and counit use the universal structural maps into/out of these chosen
coordinates. Their component equations must be checked as morphisms; a matrix
of the right shape is insufficient.

Kan kernels have finite combinatorial and linear-algebra costs depending on
fiber sizes, stalk dimensions and relation sparsity. They are computed explicitly
by the algebra API, before visualization. The viewer accepts a supplied unit,
counit or another natural identification; it never recomputes one merely to
make two diagrams line up.

## What a commuting-square certificate actually checks

Suppose the two restrictions are $L,R$ and the supplied components are
$f_q:L(q)\to R(q)$. On a selected ordered pair the viewer retains

$$R(q\leq q')f_q=f_{q'}L(q\leq q').$$

It displays the four actual matrices and both products. Verification on every
cover establishes naturality because the modules satisfy their path relations.
The viewer also computes component ranks: a natural morphism is a finite
natural isomorphism precisely when every component is invertible. An arbitrary
coordinate identification is never inserted because the dimensions agree.
For reverse-direction comparisons the roles of $L$ and $R$ are exchanged.

An invalid but well-shaped morphism remains drawable with its failing equations;
incompatible endpoints or matrix shapes are rejected. Incomparable or reversed
pairs have no forward square. For real coefficients the readout reports residual
and tolerance, and ranks follow the selected numerical field policy. Exact-field
matrices and all selected products are copied into the snapshot so later cache
activity does not alter a saved certificate.

The four-element height-one poset with every minimum below every maximum gives
a useful boundary to this certificate. Its constant module has a left Kan unit
to the pullback from one point that is an isomorphism on that module. Yet its
self-Ext in degree one has dimension one, while the one-point answer is zero.
The order complex is a cycle. A natural isomorphism inside one representation
category does not establish an equivalence between two different categories.
The viewer therefore retains `finite_poset_representations` and an explicit
absence of an ambient/derived invariance assertion even after all squares pass.

## Preparation, navigation and optimization boundaries

Preparation checks the order maps, supplied modules, pullback agreement on covers,
and any comparison. It prepares deterministic Hasse layouts and supported planar
classifier geometry. Supplying a lazy encoding for those checks materializes its
finite module. This is a deliberate algebra-inspection cost, exposed in the
reference, rather than a promise of a dimensions-only preview.

A session reuses that preparation. Selection computes only the requested maps
and products and constructs an immutable-by-convention snapshot. Side labels
retain every preimage; parameter points are classified independently by the two
actual classifiers. Neither operation chooses arbitrary representatives or
asserts ambient order from finite labels. Hover reads retained geometry and IDs.
Updates build and validate a candidate before committing it; failure leaves the
old snapshot and revision unchanged. Closing releases prepared inputs/listeners
and preserves the final snapshot.

Future optimizations can reuse restriction blocks, batched field solves,
prepared geometry or verified comparison certificates. They must preserve the
same bases, coordinate maps, field policy, realized-source scope and visible
budget behavior. In particular, replacing equality of cover maps by equality
of dimensions, sampling naturality without changing its certificate scope, or
relabeling an abstract product as a realized refinement would change the answer.
