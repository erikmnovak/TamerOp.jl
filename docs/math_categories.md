# Categories, encodings, and derived computations

TamerOp computes Hom, Ext, resolutions, and Yoneda products in the category
`Rep_k(P)` of finite-dimensional covariant representations of the **actual finite
input poset** `P`. Its Tor operation pairs a representation of `P^op` with one of
`P`, using the finite incidence-algebra tensor product. These are the categories
reported by `provenance(result)` and by existing result summaries.

An ambient encoding is additional data: an order-preserving map `q: Q -> P`
together with a specified identification `M ≅ q*H`. Keeping that identification
retains the original module, including its structure maps. It does not by itself
identify Ext or Tor computed on `P` with the corresponding groups on `Q`.

## What each API computes

| Family | Mathematical category and output | Scope of the guarantee |
| --- | --- | --- |
| `Hom`, `hom`, `hom_dimension` | Natural transformations in `Rep_k(P)` | Source and target must share the finite base. This is a vector space of natural transformations, not a persistence module called internal Hom. |
| `Ext`, `ExtInjective`, `ext` | `Ext^t_Rep_k(P)(M,N)` | Projective, injective, and unified models compute the same derived functor on this fixed category. Canonical coordinates are a choice of model and basis. |
| `projective_resolution`, `injective_resolution`, `resolve` | Resolutions in `Rep_k(P)` | Principal upsets/downsets are the projective/injective building blocks. Betti/Bass counts belong to this base poset. A finite prefix is not necessarily a complete resolution. |
| `IndicatorResolutions.upset_resolution`, `downset_resolution` | Principal indicator resolutions of the finite module | The `PModule` constructors use projective covers/injective hulls. Arbitrary indicator presentations supplied separately need not have this property. |
| `Tor`, `tor` | `Tor_s^{k[P]}(R,L)`, with `R` represented on `P^op` | Resolving either argument computes the same pairing on this fixed base. Both arguments are covariant in module morphisms. |
| Ext/Tor functorial maps and long exact sequences | Maps for those same finite-category functors | Lifted resolutions must use compatible chain coordinates. These APIs are not arbitrary change-of-encoding comparison maps. |
| `ExtAlgebra`, Yoneda products, Ext action on Tor | Yoneda composition and cap action over the finite incidence algebra | Products are defined in the same category; transport to another category needs a compatible functor and comparison. |
| `TorAlgebra` | Additional multiplication supplied on a computed Tor complex | There is no generic canonical product on Tor of two arbitrary modules. Descent is checked before producing homology products; `check_tor_algebra(...; algebraic=true)` additionally checks its stated algebraic contracts. |
| Module cochain complexes, maps, homotopies, cones, triangles | Complexes in `Rep_k(P)` | Cohomology modules, quasi-isomorphisms, and commuting diagrams refer to this base. The convention is `C[k]^t = C^(t+k)`. |
| `RHomComplex`, `DerivedTensorComplex` | Total complexes formed from a stored resolution prefix | Their raw boundary cohomology can differ from the derived functor when the prefix is incomplete. Containers retain the base; raw scalar complexes do not. |
| `hyperExt`, `hyperTor` | Certified hyper-derived groups over the finite base | `degree_range` is the certified range. Hyper-Tor degree `n` corresponds to total cochain degree `-n`; a source term in degree `p` contributes ordinary `Tor_s` at `n=s-p`. |
| Named `ExtSpectralSequence`, `TorSpectralSequence` | Spectral sequences of completed finite-category resolutions | Incomplete explicit resolution budgets are rejected. Ext uses cohomological indexing; the Tor wrapper reverses bidegrees. |
| `ExtDoubleComplex`, `TorDoubleComplex`, raw `spectral_sequence` | The supplied or explicitly truncated bicomplex | Its total cohomology is always the target of that bicomplex's spectral sequence. An Ext/Tor interpretation requires the resolution and acyclicity hypotheses. |
| `ExtZn`, `ExtRn`, and geometric resolution/spectral wrappers | Derived operations on the selected finite encoding or finite box | A geometric input type does not turn the computation into ambient Ext/Tor. Retain the classifier and window separately when interpreting the result geometrically. |

For `ExtDoubleComplex(F,dF,E,dE)` with user-supplied indicator terms, complete
projective first terms and injective second terms suffice for the Ext
interpretation. Completeness alone is insufficient for arbitrary upset and
downset indicators. The native finite-module constructor builds the principal
indicator resolutions required for its advertised interpretation.

## Resolution independence and encoding independence

Projective/injective resolution independence is a theorem **inside one abelian
category**. A comparison lift is a chain map extending the identity or the
specified module morphism; two such lifts are homotopic. Applying Hom or tensor
and passing to cohomology/homology gives the induced map. In the unified Ext
model, `comparison_isomorphism` and `comparison_isomorphisms` express these
isomorphisms in the chosen bases. Ext defined by resolutions agrees with
derived-category and Yoneda Ext under the usual resolution hypotheses; see the
[Stacks Project, Ext groups](https://stacks.math.columbia.edu/tag/06XP).

Miller's finite-encoding and syzygy theorems establish finite representations
and indicator resolutions for tame modules; they do not state that every
finite encoding preserves ambient Ext. The category of tame modules in that
theory also specifies tame morphisms, rather than silently taking every
ambient morphism. See [Miller, Sections 4.1, 4.5, and 6.2](https://arxiv.org/html/2008.00063).

### Why exact pullback is insufficient

Restriction `q*: Rep_k(P) -> Rep_k(Q)` is exact because it evaluates diagrams
and their maps at `q(a)`. Kernels and cokernels are computed pointwise. Thus a
short exact sequence, a complex, or an exact resolution pulls back to an exact
sequence or complex. Its terms need not remain projective or injective.

A concrete counterexample uses the four-point poset `Q` with minima `1,2`,
maxima `3,4`, and all four relations from a minimum to a maximum. Let `C` be
the constant diagram `k` with identity structure maps. Both the identity
encoding of `Q` and the collapse `q: Q -> {pt}` encode the same ambient module
`C = q*k`. Nevertheless,

\[
\operatorname{Ext}^1_{\operatorname{Rep}_k(Q)}(C,C)=k,
\qquad
\operatorname{Ext}^1_k(k,k)=0.
\]

An independent calculation uses a projective resolution: the cover is
`P(1) ⊕ P(2)`, whose kernel is `P(3) ⊕ P(4)`. Applying `Hom(-,C)` gives a map
`k^2 -> k^2` with both rows proportional to `(1,-1)`. It has rank one over
every field, so its cokernel is one-dimensional. Tensoring the same resolution
with the constant right diagram gives a rank-one differential `k^2 -> k^2`,
so the analogous `Tor_1` is also `k`, while it vanishes over the point.
These calculations concern the same recovered ambient diagrams; the derived
categories used in the two computations differ.
Indeed this constant-diagram pullback is even fully faithful: connectedness of
the four-edge diagram forces every natural map between two constant diagrams
to use the same linear map at every vertex. Thus exactness plus full
faithfulness still does not establish derived full faithfulness.

### Sufficient comparison hypotheses

An order isomorphism `q: Q -> P` gives an exact equivalence of representation
categories with inverse restriction along `q^-1`. It takes principal
projectives and injectives to the relabeled principal objects. Relabeling a
resolution therefore yields genuine Ext comparisons in every computed degree;
relabeling both tensor arguments gives the corresponding Tor comparisons.
Identifications must include the module maps, not merely matched dimensions.

A more general sufficient **Ext** criterion is an exact fully faithful functor
`F` that takes the projectives in a valid source resolution to projectives in
the target category. Then `F(P_*)` is a projective resolution of `F(M)`, and
full faithfulness gives a natural isomorphism of Hom complexes
`Hom(P_*,N) ≅ Hom(F(P_*),F(N))`. Its cohomology yields the Ext comparison.
There is a dual injective criterion. This is a proof of a sufficient condition,
not an automatic property or a runtime certificate for arbitrary encoding maps.
For ambient categories one must specify the allowed objects and morphisms and
verify that these resolutions and hypotheses apply there.

For **Tor**, an arbitrary fully faithful functor does not suffice: one also
needs compatible right/left transport and a tensor comparison that is an
isomorphism on suitable resolving objects. This library makes the safe finite
order-isomorphism claim, not a blanket ambient tensor-comparison claim.

## Transport and the realized joint encoding

For a finite monotone map `q: Q -> P`, `ChangeOfPosets` provides restriction,
left/right Kan extension, and derived Kan extension computations. The unit and
counit maps express the actual adjunctions

\[
q_!\dashv q^*\dashv q_*.
\]

`kan_unit(...; side=:left|:right)` and `kan_counit(...)` return those maps.
Their naturality and triangle identities are meaningful comparison statements;
they are not assertions that all four maps are isomorphisms. For an order
isomorphism they are isomorphisms. General Kan extension is not exact, which is
why its derived functors are separate operations.

The quotient maps matter independently of the dimensions. For a left-Kan
fiber write its colimit as `S / im(Rel)`, where `S` is the direct sum of its
module fibers and `Rel` records the diagram relations. If `N` is a full-column
basis of `ker(transpose(Rel))`, then the quotient-coordinate map is
`Q = transpose(N)`. Indeed `Q*Rel=0` and dimensions give
`ker(Q)=im(Rel)`. Given any left inverse `J*N=I`, the matrix
`W=transpose(J)` is a section because `Q*W=I`. A diagram map `F` induces
`Q_target * F * W_source`; since `F` preserves relations, the result does not
depend on the chosen section. The sparse selector optimization uses the
identity block in the nullspace's free rows to obtain `J`, then dualizes it in
exactly this way. Right Kan extension instead uses a kernel inclusion and its
left inverse, so that construction has the opposite roles.

This audit repaired an existing left-Kan defect: using `N` itself as a section
and an arbitrary left inverse as the quotient did not ensure that relations
were killed. On the V-shaped poset `1<2, 1<3` collapsed to a point, the map
from `(0,k,k)` into the constant diagram `(k,k,k)`, identity at vertices `2,3`,
must induce the fold `k⊕k -> k`. The old construction could return the zero map
despite correct colimit dimensions. The independent oracle now checks this
nonzero fold through the canonical unit and counit maps, together with
`Q*Rel=0` and `Q*W=I` over every supported field.

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

For example, if both classifiers are the identity on a two-point chain, the
abstract product has four vertices but the realized image is its two-vertex
diagonal. Independent classifiers over a continuous ambient domain require
geometric intersection/feasibility information to construct the realized image;
the abstract product alone does not supply it. Neither construction gives
unconditional invariance of finite-base Ext/Tor.

## Inspectable provenance and coefficient changes

For a native Ext result `E`, `provenance(E)` includes `category`, the actual
`base_poset`, `field`, `degree`, `degree_convention`, and `model`. Its
`ambient_identification=:not_asserted` prevents an encoding from silently
upgrading the claim. Tor records `orientation=(right=:opposite,left=:forward)`
and covariance in both module arguments. Existing owner summaries expose this
same record as `.provenance` without constructing comparison maps or bases.

Raw `CochainComplex`, `DoubleComplex`, and `SpectralSequence` objects retain
their coefficient field, including custom `RealField` tolerances; their
provenance reports `field_source=:stored_field`. Raw-matrix constructors
accept `field=...`, with an inferred default only when no field is supplied.
They still have no finite-module origin and report `base_poset=nothing`.
Retain the module-aware container or workflow result for that information.
An explicitly reindexed Tor spectral sequence cannot reconstruct the
discarded base from its matrices. See [numerical algebra](numerical_algebra.md)
for the role of tolerances in quotient coordinates and products.

Coefficient conversion is another change of mathematical input. An actual
field extension preserves exactness of finite complexes by flatness; this is
different from interpreting integral or rational matrices modulo a prime.
There is no field homomorphism `QQ -> Fp`. For instance the integral differential
`[2]` has rank one over `QQ` and rank zero over `F2`. Reusing a previously
computed dimension table or resolution and relabeling its field is therefore
invalid. Derived results must be recomputed after changing coefficients.
Accordingly, `change_field` rejects cross-field relabeling of computed derived
answers. On an `EncodingResult`, it converts the stored module matrices as a
new algebraic input; it does not recompute the homology of the original filtered
complex. It drops presentation/image witnesses and the original homology-degree
and reconstruction claims, because taking the image of a presentation can fail
to commute with reduction modulo a prime. Lazy inputs follow the same operation
after explicit materialization. Re-encode the original filtration over the new
field when that is the intended mathematical computation.
Entrywise conversion can also break path independence, morphism naturality, or
the equation `d^2=0`: for example, cancellation in characteristic two need not
remain cancellation after lifting entries to `QQ`. Converted modules, maps,
and module complexes are checked in the target field and rejected when these
relations fail. Successful conversion produces a new algebraic input, not an
invariance certificate. Lazy and materialized encoded complexes use this same
validated operation.
Floating-point conversions additionally impose a numerical tolerance contract;
they are not exact base-change certificates.

## Encoding serialization and comparison maps

`save_encoding_json(path, enc)` stores a finite fringe presentation. Loading
reconstructs its image as a module on the stored finite poset. Preservation is
up to natural isomorphism: the reconstructed stalk bases can differ from the
original module's bases. Thus compare structure maps using stalk
identifications `J_q`, checking `J_v * restored(u,v) = original(u,v) * J_u`.
Dimensions alone do not verify this contract. A restored poset is also a new
Julia object; use an explicit identity-on-labels `EncodingMap` when a later
restriction must target the original poset object.

Workflow encoding artifacts retain the recorded mathematical degree, category,
window, orientation, exact coordinate values, construction, discretization,
approximation, and producer-backend evidence. The loaded object's actual field
(including numerical tolerances) and finite base take precedence over metadata.
Its encoding backend is `:serialization`; retained producer evidence describes
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
homology-degree claim, and retains the original producer contract as `source`.
To compute homology over the new field, re-encode the original filtered complex.

The A75 encoding tests use unimodular integral changes of basis, so their
independently specified natural maps exist over every tested field. They check
all comparable structure maps and noncommuting endomorphism compositions through
serialization, explicit base identifications, realized joint refinement, and
fresh/reused translation caches. This is a shared integral oracle, not an
assertion that arbitrary homology is independent of characteristic.

The finite comparison and counterexample oracles live in
`test/test_encoding.jl`; owner provenance and encoded spectral-budget
contracts are tested in `test/test_derived_functors.jl`. These tests establish
the stated finite comparisons and reject overbroad interpretations. They do
not certify arbitrary ambient categories or uncomputed degrees.
