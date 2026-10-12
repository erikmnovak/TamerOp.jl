# Yoneda products and validated Tor algebra

Knowing the dimension of an Ext group tells us how many independent classes
exist. Multiplication asks a different question: what class results when two
compatible classes compose? The answer requires representatives, comparison
maps and quotient coordinates. This account follows that calculation, then
explains why Tor needs additional multiplication data and how those data are
checked before a product table is drawn.

The objects are representations of one finite poset over a coefficient field.
The [usage guide](../tutorials/derived.ipynb) starts from the diamond directly;
an encoding input supplies the same finite setting through its stored module.
The calculation does not prove that the resulting Ext groups are unchanged
under arbitrary refinement, or equal to Ext in the original parameter
category. [Category and variance](../math_categories.md) remain part of the
input contract.

## Lift before composing

Suppose we have three modules $L,M,N$ and classes

$$
\alpha\in\operatorname{Ext}^q(L,M),\qquad
\beta\in\operatorname{Ext}^p(M,N).
$$

Their Yoneda product is a class in $\operatorname{Ext}^{p+q}(L,N)$.
The relation between this composition and splicing actual extensions is
described in [the Stacks Project's Ext treatment](https://stacks.math.columbia.edu/tag/06XP).
TamerOp implements the calculation using projective resolutions. It does not
construct a sequence of extension modules merely to draw the result.

A projective resolution is stored using generator labels in the finite
poset and coefficient matrices for its differentials. A cochain
$P_q(L)\to M$ is a concatenated vector of target-stalk coordinates, one
block for each projective generator. Quotient coordinates specify a linear
combination of chosen cocycle representatives. This storage lets the
computation use finite maps and small blocks without assembling every
stalk component of an expanded morphism.

Let $a:P_q(L)\to M$ and $b:P_p(M)\to N$ be the chosen cocycles.
The lift routine first lifts $a$ through the augmentation of the resolution
of $M$. It then solves successive compatibility equations to construct
maps $F_i:P_{q+i}(L)\to P_i(M)$. The product cocycle is $bF_p$.
The diagram records the two operations; its top horizontal arrow is the
computed lift, not an inferred map between unrelated bases.

[![The right class is lifted to a map between projective-resolution terms, composed with the left cocycle, and then expressed modulo boundaries.](../assets/implementation/yoneda_lift.svg)](../assets/implementation/yoneda_lift.svg)

[Open the diagram at full size.](../assets/implementation/yoneda_lift.svg)

The right-hand resolution must therefore reach degree $p+q$, even when the
right class itself has degree $q$. The left model needs degree $p$. Before
lifting, the owner verifies that the middle modules agree in field, order,
stalk coordinates and cover maps. Matching dimension vectors alone would
leave the composition undefined.

The composition kernel splits $b$ into its generator blocks. A nonzero
coefficient of $F_p$ requires a structure map of $N$ between the corresponding
labels. The kernel either accumulates these coefficient-scaled map-vector
products directly or prepares the needed map blocks, depending on its
workload gate. It then concatenates the output blocks and converts the
cocycle into quotient coordinates. The lift and quotient routines use the
shared field-aware linear algebra; the
[rational-coordinate account](qq_coordinates.md) explains exact coordinate
recovery and its membership certificates.

## Fix the target coordinates

A displayed coordinate is meaningful only in a declared target model.
By default, the product uses the right factor's resolution of $L$ and
computes Hom into $N$. A supplied `ELN` can fix another resolution of the
same pair. If the models differ, the owner lifts the identity of $L$ from
the requested target resolution to the resolution used for composition.
Precomposition transfers the cocycle into the requested complex before
quotient coordinates are recovered.

Consequently, the optional returned cocycle and the coordinate vector both
belong to the returned target space. This is stronger than comparing group
dimensions, and necessary when comparing two products. No extra Koszul
factor is inserted by this unshifted comparison convention. Signs already
occur in the differentials and the resulting coefficients.

On the diamond, the two degree-one products from the bottom simple to the
top simple sum to zero in a one-dimensional degree-two group. They are both
nonzero. Over an odd-characteristic field they have opposite coordinates;
over characteristic two the same relation is written with equal coefficients.
This small example exposes a sign that a dimension table cannot show.

The inspection builder also checks the displayed representative explicitly.
For the returned cocycle $z$ with coordinates $c$, let $h(c)$ be the owner's
chosen representative. It solves for an incoming cochain $w$ and retains

$$
d z=0,\qquad [z]=c,\qquad z=h(c)+d w.
$$

The last equation explains why the actual composite need not equal the
chosen representative as a vector. The adjustment $w$ is a particular
solution, not a canonical choice. For a zero product, it witnesses that
the composite is a boundary. Numerical fields use the recorded tolerances
rather than presenting a floating residual as a symbolic proof.

## Reuse one table at a time

`ExtAlgebra` retains one projective self-Ext model and caches structure
constants by degree pair. For dimensions $d_p,d_q,d_{p+q}$, its table has
$d_{p+q}$ rows and $d_p d_q$ columns. Column $(i-1)d_q+j$ is the product
of the left basis vector $i$ and right basis vector $j$, so the table acts
on `kron(left,right)`.

The batched Yoneda route lifts a right basis class once and composes it
with all left basis classes. Cache lookup and publication are synchronized;
the expensive construction runs outside the publication lock. The retained
matrix should be treated as read-only. The dimension view does not populate
this product cache or force the lazy quotient bases.

Storage for a completed table scales with $d_{p+q}d_p d_q$, in addition to
the resolutions and lifts. The visual request checks input-pair and output-entry
budgets before requesting multiplication. Matrix cropping happens afterward
and reduces only drawing size. A selected `:yoneda_product` request avoids
constructing a complete self-algebra table when only one composition is needed.

## Why Tor needs a supplied product

An incidence-algebra Tor computation pairs a right module on the opposite
poset with a left module. Its output is a graded vector space; these two
arguments do not supply a canonical algebra multiplication. A `TorAlgebra`
therefore receives actual chain matrices

$$
\mu_{p,q}:C_p\otimes C_q\longrightarrow C_{p+q}.
$$

Their columns use the same right-index-fastest tensor convention as the
Ext table. A generator can provide the matrices lazily. Missing maps are
errors unless their source or target chain space is zero. A supplied unit
is a homology class, not an assumed strict unit on all chains.

For a matrix to induce multiplication, it must send cycles to cycles and
send products involving a boundary to boundaries. The implementation writes
the cycle space as a sum of a boundary basis $B$ and chosen homology
representatives $H$. Before publishing a multiplication matrix, it checks
products of $B_p$ with all cycles in degree $q$, products of $H_p$ with
$B_q$, and products of $H_p$ with $H_q$. The first two groups must have
zero target homology coordinates; the last must admit homology coordinates.
Bilinearity then gives independence from both choices of representative.

Checking only the chosen homology representatives would be insufficient.
For example, if a boundary $b$ multiplied by a cycle $z$ represented a
nonzero class, replacing a representative $x$ by $x+b$ would change the
answer. The boundary checks exclude exactly this defect. Incoming boundaries
from the extra degree above the displayed range are included.

## Descent and algebra laws are separate checks

Descent is required for every new homology multiplication table. The stronger
`check_tor_algebra(A; algebraic=true)` request additionally checks the
chain Leibniz identity, associativity on homology and a supplied unit.
The [differential graded algebra definition](https://stacks.math.columbia.edu/tag/061U)
supplies the compatibility law. In homological notation it reads

$$
\partial(xy)=(\partial x)y+(-1)^p x(\partial y),\qquad x\in C_p.
$$

The implementation evaluates its matrix form, including the incoming-boundary
guard degree. Associativity is checked by comparing the two composites of
homology multiplication matrices for every computed degree triple. It does
not demand strict associativity on chains, and its report does not certify
uncomputed degrees. Both unit identities are checked if a unit was supplied.

The broader validator works on a consistent snapshot and does not populate
the original algebra's cache. Ordinary table computation publishes its
result with a generation check, preventing a product replaced through a
setter from being reinserted by an older computation. Lazy generators must
remain stable until explicitly replaced and permit concurrent calls.

## Prepare mathematical data before drawing

The visualization owner consumes these checked outputs. Ext retains
cohomological labels, while Tor keeps homological labels and reindexes its
representative complex by $C^{-s}=C_s$ for shared inspection. Selected
vectors always use the underlying owner's quotient basis; recomputing a
new quotient solely for display could choose different coordinates.

Prepared specifications copy coefficient matrices, preserve full matrix
sizes when drawing is cropped, and identify the finite base, field, model,
degree and sign convention. A renderer places those records without
performing algebra. The [visual specification account](visual_specs.md)
owns rendering and export mechanics. Actual supplied exact sequences can
be inspected separately; a product class alone is displayed as the
representative that was computed.

See the [implementation bibliography](references.md).
