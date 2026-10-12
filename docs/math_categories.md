# Categories, encodings, and derived computations

Suppose we have encoded a module and now want to compare it with another
one. We can look for compatible maps between them, take kernels and
quotients, or ask how they fit into exact sequences. The finite poset makes
these calculations accessible, but it also becomes part of the question:
a resolution on one poset need not answer the same question as a resolution
on another, even when both encodings recover the same original module.

We will follow that distinction through a small example where the answer
can be computed by hand. The [finite encoding introduction](finite_encodings.md)
explains how the original spaces and maps are recovered, and
[why finite computations stay tame](practical_tameness.md) explains why
kernels, images and homology respect that recovery. Here we keep those
facts in view while examining the additional requirements for Ext and Tor.

Write $\operatorname{Rep}_{\mathbb{k}}(P)$ for the category of modules
on a finite poset $P$, with finite-dimensional spaces over a fixed field
$\mathbb{k}$. A module assigns a space to each label and a compatible
linear map to each comparison. A morphism between two such modules is a
family of linear maps commuting with their structure maps, also called a
**natural transformation**. These are the objects and morphisms used by
TamerOp's Hom, Ext, resolution and Yoneda product operations on $P$.
The poset is the actual input to the calculation.

An encoding includes a further map $q:Q\to P$ from the original parameter
poset, preserving its order, and an identification $M\cong q^*H$ for a
module $H$ on $P$. At an original parameter $a$, the pullback $q^*H$ uses
$H(q(a))$, with maps supplied by the corresponding comparisons in $P$.
Thus we recover $M$, including its structure maps. To compare derived
calculations on $P$ and $Q$, we will also have to follow what happens to
their resolutions.

## What each API computes

Hom is the vector space of module morphisms. To introduce Ext and Tor,
recall that a sequence is **exact** when the image of each map is the
kernel of the next. A **resolution** expresses a module by an exact
sequence of terms on which the desired operation is easier to compute.
For Ext we can use projective terms, whose maps lift through surjections,
or injective terms, into which maps extend across injections. Applying
Hom to such a resolution and taking cohomology gives Ext. In degree one,
Ext describes short exact sequences with the two chosen modules as their
end terms, up to the usual equivalence of extensions.

Tor is obtained by tensoring a projective resolution and taking homology.
Here the tensor product pairs a module on $P$ with one on $P^{\mathrm{op}}$,
the same labels with their order reversed. It is the tensor product over
$\mathbb{k}[P]$, the **incidence algebra**, whose basis elements represent
comparable pairs and whose multiplication composes matching pairs. Reversing
the order supplies the right module needed for this pairing.

The table records which version of these constructions a call returns.
It also identifies the cases where a stored part of a resolution needs
additional terms before its answer can be interpreted as Ext or Tor.

| Family | Mathematical category and output | Scope of the guarantee |
| --- | --- | --- |
| `Hom`, `hom`, `hom_dimension` | Natural transformations in `Rep_k(P)` | Source and target must share the finite base. This is a vector space of natural transformations, not a persistence module called internal Hom. |
| `Ext`, `ExtInjective`, `ext` | `Ext^t_Rep_k(P)(M,N)` | Projective, injective, and unified models compute the same derived functor on this fixed category. Canonical coordinates are a choice of model and basis. |
| `projective_resolution`, `injective_resolution`, `resolve` | Resolutions in `Rep_k(P)` | Principal upsets/downsets are the projective/injective building blocks. Betti/Bass counts record the multiplicities of principal terms on this base poset. A finite prefix is not necessarily a complete resolution. |
| `IndicatorResolutions.upset_resolution`, `downset_resolution` | Principal indicator resolutions of the finite module | The `PModule` constructors use projective covers/injective hulls. Arbitrary indicator presentations supplied separately need not have this property. |
| `Tor`, `tor` | `Tor_s^{k[P]}(R,L)`, with `R` represented on `P^op` | Resolving either argument computes the same pairing on this fixed base. Both arguments are covariant in module morphisms. |
| Ext/Tor functorial maps and long exact sequences | Maps for these functors on the same finite base | Lifted resolutions must use compatible chain coordinates. Comparing different encodings requires a separate comparison of categories. |
| `ExtAlgebra`, Yoneda products, Ext action on Tor | Composition of extensions, and the induced action on Tor, over the finite incidence algebra | Products are defined in the same category; transport to another category needs a compatible functor and comparison. |
| `TorAlgebra` | Additional multiplication supplied on a computed Tor complex | There is no generic canonical product on Tor of two arbitrary modules. Descent is checked before producing homology products; `check_tor_algebra(...; algebraic=true)` additionally checks its stated algebraic contracts. |
| Module cochain complexes, maps, homotopies, cones, triangles | Complexes in `Rep_k(P)` | Cohomology modules and the maps inducing isomorphisms on cohomology (quasi-isomorphisms) refer to this base. The convention is `C[k]^t = C^(t+k)`. |
| `RHomComplex`, `DerivedTensorComplex` | Total complexes formed from a stored resolution prefix | Their raw boundary cohomology can differ from the derived functor when the prefix is incomplete. Containers retain the base; raw scalar complexes do not. |
| `hyperExt`, `hyperTor` | Ext or Tor with a complex as input, over the finite base | `degree_range` records the degrees justified by the stored resolutions. Hyper-Tor degree `n` corresponds to total cochain degree `-n`; a source term in degree `p` contributes ordinary `Tor_s` at `n=s-p`. |
| Named `ExtSpectralSequence`, `TorSpectralSequence` | Successive pages approaching the total (co)homology of completed resolutions on this base | Incomplete explicit resolution budgets are rejected. Ext uses cohomological indexing; the Tor wrapper reverses bidegrees. |
| `ExtDoubleComplex`, `TorDoubleComplex`, raw `spectral_sequence` | The supplied complex with two compatible differential directions, possibly truncated | Its total cohomology is always the target of that bicomplex's spectral sequence. An Ext/Tor interpretation requires the resolution and acyclicity hypotheses. |
| `ExtZn`, `ExtRn`, and geometric resolution/spectral wrappers | Derived operations on the selected finite encoding or finite box | A geometric input type does not turn the computation into ambient Ext/Tor. Retain the classifier and window separately when interpreting the result geometrically. |

For `ExtDoubleComplex(F,dF,E,dE)` with user-supplied indicator terms, complete
projective first terms and injective second terms suffice for the Ext
interpretation. Completeness alone is insufficient for arbitrary upset and
downset indicators. The constructor taking a finite module builds the principal
indicator resolutions required for its advertised interpretation.

## Geometric encoding and boundary classes

A closed boundary can carry a nonzero space even when the neighboring
open regions carry zero. For example, take the upset $[0,\infty)$, the
downset $(-\infty,0]$, and the coefficient matrix $[1]$. Their intersection
is the single point $0$, so the represented module has space $\mathbb{k}$
there and zero elsewhere. Any encoding of this presentation has to retain
that point and its maps; sampling only the open intervals on either side
would miss the module.

With `backend=:auto`, TamerOp selects an encoder that preserves these
closed boundary classes. The general polyhedral encoder, `backend=:pl`,
can represent the point and intersections of boundary faces in higher
dimensions. The selected method is recorded in `encoding_result.backend`.
Its name tells us how the encoding was constructed; the spaces and maps
at the boundary tell us whether it represents the intended module.

The general `:pl` encoder decides whether a region contains a point by
testing its inequalities over rational coordinates. Some inequalities are
strict because the region excludes a generator boundary. A common slack
variable, bounded between zero and one, tests whether all those strict
inequalities can hold simultaneously. By default, the test imposes no
fixed positive margin, so it can retain arbitrarily narrow rational
regions and regions lying entirely on a boundary.

A floating point input is interpreted using the binary value it actually
stores; a decimal-looking number is not rounded to a simpler rational.
A witness is a point satisfying the region's inequalities, and it remains
rational if conversion to `Float64` would move it outside the region.
Choosing an explicitly positive `strict_eps` instead requires a fixed
feasibility margin and can remove regions narrower than that margin.

For a union of closed monotone polyhedra, the encoder records membership
in the constituent facet inequalities as well as in the original
generators. This finer subdivision gives convex cells while retaining
which original generators contain them. It allows separate pieces with
the same generator membership to remain represented. In the input
inequalities $Ax\leq b$, the rows of $A$ must have nonpositive entries
for upsets and nonnegative entries for downsets. The input generator
pieces must be closed; strict inequalities are used internally to
separate their membership regions, but strict input pieces are rejected.
Exceeding a region budget raises an error before a partial encoding can
be returned.

Automatic conversion of rational polyhedral generators to the box
backend requires closed principal orthants, meaning regions specified by
one lower or upper bound on every coordinate, with endpoints exactly
representable as `Float64`. General unions, missing axis bounds and exact
endpoints that would be rounded need the general `:pl` route. Strict input
pieces remain outside the supported input contract. This geometry choice does not change the
coefficient field requested in `EncodingOptions(field=...)`, and all supported
routes retain the supplied coefficient matrix. The selected backend remains
visible as `encoding_result.backend`. `provenance(encoding_result).approximation`
records the feasibility method (`:exact_rational`, `:fixed_margin`, or
`:not_applicable`), requested `strict_eps`, and effective margin; the default
exact method has no margin. The stored options retain the original request.

## Resolution independence and encoding independence

Within one abelian category, different projective or injective
resolutions compute the same derived functor. We compare two resolutions
by a **chain map**, a family of maps commuting with their differentials,
that extends the identity or the chosen module morphism. Two choices of
this lift differ by a chain homotopy: their difference can be expressed
using the differentials and auxiliary maps of adjacent degrees. This
makes their induced maps on homology agree after applying Hom or tensor. In the unified Ext
model, `comparison_isomorphism` and `comparison_isomorphisms` express these
isomorphisms in the chosen bases. Under the usual resolution hypotheses, this agrees with Ext in the
derived category, where maps inducing isomorphisms on cohomology are made
invertible, and with Yoneda Ext, defined by equivalence classes of exact
sequences of the appropriate length. See the
[Stacks Project, Ext groups](https://stacks.math.columbia.edu/tag/06XP).

Ezra Miller's finite encoding and syzygy theorems provide finite
representations and indicator resolutions for tame modules. Comparing
Ext in two categories requires additional information about those
resolutions, as the following example shows. The category of tame modules in that
theory uses tame morphisms, whose components admit compatible finite
descriptions as in the preceding chapter. See [Ezra Miller, Sections 4.1, 4.5, and 6.2](https://arxiv.org/html/2008.00063).

### Why exact pullback is insufficient

The restriction, or pullback, functor $q^*$ evaluates a diagram and its
maps at $q(a)$. Since kernels and cokernels are computed at each label,
this functor is exact. A short exact sequence therefore pulls back to a
short exact sequence, and an exact resolution remains exact. The terms
of that resolution, however, may lose the lifting or extension property
that made them projective or injective.

Take the poset $Q$ with minima $1,2$, maxima $3,4$, and all four relations
from a minimum to a maximum. Let $C$ have space $\mathbb{k}$ at every
label and identity maps on all four arrows. We can encode $C$ using the
identity map of $Q$, or using the map $q:Q\to\{*\}$ that sends all four
labels to one point. Pulling the space $\mathbb{k}$ back from that point
recovers exactly $C$.

![The poset with four labels has minima 1 and 2, maxima 3 and 4, and identity maps from each minimum to each maximum. Collapsing all labels to one point still recovers the constant module. The Hom differential on the poset with four labels has rank one and cokernel k; Ext in degree one over the point is zero.](assets/math_categories/encoding_ext.svg)

*All four spaces on the left are $\mathbb{k}$ and every arrow is the
identity. The crossing of the diagonal arrows is not another vertex.
The collapse retains this particular module, while changing the category
in which we choose a resolution. The resulting Ext groups differ:*

$$
\operatorname{Ext}^1_{\operatorname{Rep}_k(Q)}(C,C)=k,
\qquad
\operatorname{Ext}^1_k(k,k)=0.
$$

We can derive the first answer explicitly. Let $P(i)$ be the principal
projective at label $i$: it has a copy of $\mathbb{k}$ at every label
above $i$ and zero elsewhere, with identity maps within its support.
The map $P(1)\oplus P(2)\to C$ is the identity at each minimum and
addition at each maximum. Its kernel is a line spanned by $(1,-1)$ at
labels $3$ and $4$, with zero spaces at the minima. Thus the kernel is
$P(3)\oplus P(4)$, and we have a projective resolution

$$
0\longrightarrow P(3)\oplus P(4)
\longrightarrow P(1)\oplus P(2)\longrightarrow C\longrightarrow0.
$$

Applying $\operatorname{Hom}(-,C)$ leaves the map

$$
\mathbb{k}^2\xrightarrow{\delta}\mathbb{k}^2,
\qquad \delta(a,b)=(a-b,a-b).
$$

Its matrix has two equal, nonzero rows, so it has rank one over every
field. Its cokernel is therefore a line, giving
$\operatorname{Ext}^1(C,C)\cong\mathbb{k}$. Over the poset with one point,
we are working with vector spaces, whose short exact sequences split;
there $\operatorname{Ext}^1(\mathbb{k},\mathbb{k})=0$.

Tensoring the same projective resolution with the constant right module
gives a map $\mathbb{k}^2\to\mathbb{k}^2$ of rank one. Its kernel is a
line, so $\operatorname{Tor}_1$ is also $\mathbb{k}$ over $Q$, while
it vanishes over the point.

This pullback is even **fully faithful**, meaning that it gives a
bijection on each space of morphisms: because the diagram is connected,
a natural map between two constant modules must use the same linear map
at every vertex. We have therefore preserved the objects under discussion
and all maps between constant modules, yet their Ext groups differ.
Comparing derived functors asks for more than exactness and full faithfulness.

### Sufficient comparison hypotheses

An order isomorphism `q: Q -> P` gives an exact equivalence of representation
categories with inverse restriction along `q^-1`. It takes principal
projectives and injectives to the relabeled principal objects. Relabeling a
resolution therefore compares Ext in every computed degree;
relabeling both tensor arguments gives the corresponding Tor comparisons.
The identification includes the module maps, so it supplies the
commuting diagrams needed to compare resolutions.

A more general sufficient **Ext** criterion is an exact fully faithful functor
`F` that takes the projectives in a valid source resolution to projectives in
the target category. Then `F(P_*)` is a projective resolution of `F(M)`, and
full faithfulness gives a natural isomorphism of Hom complexes
`Hom(P_*,N) ≅ Hom(F(P_*),F(N))`. Its cohomology yields the Ext comparison.
There is a dual injective criterion. This gives a sufficient condition whose hypotheses must be established
for the particular encoding map.
For ambient categories one must specify the allowed objects and morphisms and
verify that these resolutions and hypotheses apply there.

For **Tor**, an arbitrary fully faithful functor does not suffice: one also
needs compatible right/left transport and a tensor comparison that is an
isomorphism on suitable resolving objects. For a finite order isomorphism those identifications are available.
A more general ambient tensor comparison needs its own justification.

## Transport and the realized joint encoding

Given a map $q:Q\to P$ that preserves order, restriction reads a module
on $P$ at the labels $q(a)$ to obtain one on $Q$. **Kan extension** goes
in the other direction. Left Kan extension combines the relevant spaces
by quotienting their direct sum by the diagram relations; right Kan
extension selects tuples of vectors compatible with those relations.
`ChangeOfPosets` provides both operations and their derived functors.

Their defining properties are **adjunctions**, correspondences between
morphisms on the two bases. A map $q_!M\to N$ corresponds to a map
$M\to q^*N$, and a map $q^*N\to M$ corresponds to a map $N\to q_*M$.
Here $q_!$ denotes left Kan extension, $q_*$ denotes right Kan extension,
and $q^*$ denotes restriction. We summarize the two correspondences by

$$
q_!\dashv q^*\dashv q_*.
$$

Each correspondence supplies canonical comparison maps, called its
**unit** and **counit**, by applying it to identity maps. For example,
the left unit is $M\to q^*q_!M$ and the left counit is
$q_!q^*N\to N$. The calls `kan_unit(...; side=:left|:right)` and
`kan_counit(...)` return the corresponding comparisons.
These maps are natural: they commute with module morphisms. Their
triangle identities say that passing from one side of an adjunction and
back in the prescribed way gives the identity. They become isomorphisms
when $q$ is an order isomorphism. For a general map, they need not be
invertible, and Kan extension need not preserve exact sequences; its
derived functors measure that failure.

To follow a map through left Kan extension, we need coordinates on the
resulting quotient. A **colimit** combines the spaces in a diagram while
identifying vectors related by its arrows. For a left Kan fiber,
write it as `S / im(Rel)`, where `S` is the direct sum of its module
spaces and the columns of `Rel` record the diagram relations. Let the
columns of `N` form a basis of `ker(transpose(Rel))`. Then
`Q = transpose(N)` gives coordinates on the quotient: we have `Q*Rel=0`,
and comparing dimensions gives `ker(Q)=im(Rel)`. Given a left inverse
`J*N=I`, put `W=transpose(J)`. Since `Q*W=I`, this is a **section**,
choosing a representative in `S` for each quotient vector. A diagram map `F` induces
`Q_target * F * W_source`; since `F` preserves relations, the result does not
depend on the chosen section. The implementation can obtain `J` by selecting the identity block in
the free rows of its nullspace basis, then transposing as above. Right Kan extension instead uses a kernel inclusion and its
left inverse, so that construction has the opposite roles.

For example, collapse the V-shaped poset `1<2, 1<3` to a point. The diagram
`(0,k,k)` has colimit `k⊕k`, while the constant diagram `(k,k,k)` has colimit
`k`. The map between them that is the identity at vertices `2,3` induces the
addition map `(x,y) ↦ x+y`. The dimensions tell us only that the source is a plane and the target
a line. The quotient relations explain why the map adds the two coordinates.

The abstract Cartesian product `P1 × P2` provides projection maps to two finite
bases, even when no shared ambient source is known. If actual maps
`q1: Q -> P1` and `q2: Q -> P2` are supplied, their realized joint encoding is
the image `J = {(q1(a),q2(a)): a in Q}`, with the order induced from the product.
`joint_encoding(q1,q2)` constructs this finite image and the factorization
through `Q -> J`. Pullback along the projections and then along `Q -> J`
recovers the original pullbacks on both objects and morphisms, by composition.
This remains true even when the induced order on `J` contains comparable pairs
not witnessed by comparable source points.

For already constructed finite classifiers and encoding results, the public
workflow is:

```julia
joint = TamerOp.Advanced.joint_encoding(q1, q2)
translated = TamerOp.common_refinement(enc1, enc2, joint)
TamerOp.provenance(translated).refinement  # :realized_joint_image

# q: Q -> P and M a module on Q:
eta = TamerOp.Advanced.kan_unit(q, M; side=:left)  # M -> q* q_! M
TamerOp.Modules.check_morphism(eta)
```

The overload without `joint` uses the target posets alone. It cannot infer
which pairs of classifier values are realized on a shared ambient source.

For example, if both classifiers are the identity on a chain with two points, the
abstract product has four vertices but the realized image is its diagonal with two vertices. Independent classifiers over a continuous ambient domain require
geometric intersection/feasibility information to construct the realized image;
the abstract product alone does not supply it. To compare Ext or Tor on either resulting base, we still need the
resolution hypotheses discussed above.

## Inspectable provenance and coefficient changes

A computed result carries **provenance**, a record of the mathematical
input and conventions behind it. For an Ext result `E`, `provenance(E)`
includes `category`, the actual `base_poset`, `field`, `degree`,
`degree_convention`, and `model`. The value
`ambient_identification=:not_asserted` records that no identification with
an ambient derived functor has been established. Tor records `orientation=(right=:opposite,left=:forward)`
and the fact that a morphism in either module argument induces a map in
the same direction (covariance). The result summaries expose this same
record as `.provenance` without constructing comparison maps or bases.

Raw `CochainComplex`, `DoubleComplex`, and `SpectralSequence` objects retain
their coefficient field, including custom `RealField` tolerances; their
provenance reports `field_source=:stored_field`. Constructors from raw matrices
accept `field=...`, with an inferred default only when no field is supplied.
The matrices alone specify no finite base, so these objects report
`base_poset=nothing`. Retain the result carrying the original module
when you need that information.
An explicitly reindexed Tor spectral sequence cannot reconstruct the
discarded base from its matrices. See [numerical algebra](numerical_algebra.md)
for the role of tolerances in quotient coordinates and products.

Changing the coefficient field changes the mathematical input. Extending
scalars along a field extension preserves exact sequences, a property
called **flatness**. Interpreting integral or rational matrix entries
modulo a prime is a different operation.
There is no field homomorphism `QQ -> Fp`. For instance the integral differential
`[2]` has rank one over `QQ` and rank zero over `F2`. Reusing a previously
computed dimension table or resolution and relabeling its field is therefore
invalid. Derived results must be recomputed after changing coefficients.
Accordingly, `change_field` rejects relabeling between fields of computed derived
answers. On an `EncodingResult`, it converts the stored module matrices as a
new algebraic input; it does not recompute the homology of the original filtered
complex. It drops witnesses for the presentation and its image, along with the original
homology degree and reconstruction claims, because taking the image of a presentation can fail
to commute with reduction modulo a prime. Lazy inputs follow the same operation
after explicit materialization. Re-encode the original filtration over the new
field when that is the intended mathematical computation.
Entrywise conversion can also break path independence, morphism naturality, or
the equation `d^2=0`: for example, cancellation in characteristic two need not
remain cancellation after lifting entries to `QQ`. Converted modules, maps,
and module complexes are checked in the target field and rejected when these
relations fail. Successful conversion gives a new algebraic input satisfying those
relations; it does not identify its invariants with the old ones. Lazy and materialized encoded complexes use this same
validated operation.
Conversions to floating point also require numerical tolerances, so
their interpretation must include those tolerances.

## Encoding serialization and comparison maps

Saving an encoding should let a later computation recover its spaces and maps,
even if the chosen coordinates change. A natural isomorphism means that the
changes of basis at individual labels commute with every structure map. This
is the comparison needed when checking a saved and reloaded module.

`save_encoding_json(path, enc)` stores a finite fringe presentation. Loading
reconstructs its image as a module on the stored finite poset. Preservation is
up to natural isomorphism: the reconstructed stalk bases can differ from the
original module's bases. Thus compare structure maps using stalk
identifications `J_q`, checking `J_v * restored(u,v) = original(u,v) * J_u`.
This equation checks that the recovered vectors continue in the same
way through the poset. Matching dimensions would leave that question open. A restored poset is also a new
Julia object; use an explicit `EncodingMap` sending each label to itself when a later
restriction must target the original poset object.

A saved encoding retains the recorded mathematical degree, category,
window, orientation, exact coordinate values, construction, discretization,
approximation and method used to construct it. The loaded object's actual field
(including numerical tolerances) and finite base take precedence over metadata.
Its encoding backend is `:serialization`; the retained construction record describes
the construction that originally produced the stored module. Runtime caches and
arbitrary Julia objects are not serialized. Unsupported provenance values fail
explicitly. The strict and trusted loaders both validate stored classifiers and
mathematical metadata; `validation=:trusted` skips only the documented mask
checks. `check_encoding_json` validates these payloads even when the requested
computational result would only be a fringe.

The optional `load_encoding_json(...; field=...)` override reinterprets the
**stored fringe presentation** and recomputes its image. This differs from
`change_field(enc, field)`, which reinterprets the stored module matrices. For
example, the rational fringe matrix `[2]` presents a one-dimensional constant
module, but its image after reduction modulo two is zero. Loading with a changed
field records `:reinterpret_stored_fringe_presentation`, clears the original
homology degree claim, and retains the original construction contract as `source`.
To compute homology over the new field, re-encode the original filtered complex.
