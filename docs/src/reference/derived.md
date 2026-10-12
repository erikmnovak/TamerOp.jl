# Derived spaces, representatives and products

These operations compute in the representation category of the **recorded
finite poset**. An encoding's recovery map does not automatically identify
these groups with Ext or Tor in an ambient persistence category. The
[category guide](../../math_categories.md) states the distinction; the
[worked guide](../../tutorials/derived.ipynb) develops a nonzero diamond example.

## Compute and inspect a graded space

```julia
E = OP.ext(M, N; maxdeg=2, model=:auto, canon=:auto, cache=:auto)
T = OP.tor(Rop, L; maxdeg=2, model=:auto, cache=:auto)
OP.Advanced.ext_summary(E)
OP.Advanced.tor_summary(T)
OP.Advanced.dim(E, 2)
OP.visualize(E)
```

`M,N` are `PModule`s on the identical finite base, or compatible
`EncodingResult`s. Ext is contravariant in the first argument and covariant
in the second. `Rop` is a right module represented on the opposite poset of
the left module `L`; Tor is covariant in each argument. This Tor is over the
incidence algebra, not the pointwise tensor of two forward modules.
Coefficient fields must agree.

`maxdeg` specifies computed degrees `0:maxdeg`; the construction retains
the next differential needed for the last quotient. Degrees above the range
are uncomputed, even when the last displayed group is zero. Ext models are
`:projective`, `:injective`, or `:unified`; `:auto` selects projective.
Unified Ext chooses projective or injective canonical coordinates with
`canon`. Tor's `:first` and `:second` models resolve the respective argument;
`:auto` selects `:first`. `cache=sc::OP.SessionCache` requests compatible
cross-call reuse. Resolution choices can change representatives and bases.

`degree_range`, `degree_dimensions`, `source_module`, `target_module`,
and `provenance` on `OP.Advanced` provide semantic access to the result.
`OP.provenance(E)` records the finite base, field, variance and model.
`basis`, `representative` and `coordinates` are explicit heavier requests.
`representative(E,t,c)` maps quotient coordinates to a cocycle; for Tor it
returns a cycle. `coordinates(E,t,z)` maps a cycle back to its quotient
coordinates. Arbitrary noncycles are not valid inputs to that operation.

```julia
C = OP.Advanced.representative_complex(E)
z = OP.Advanced.representative(E, 2, OP.QQ[1])
```

`representative_complex` returns the stored Hom cochain complex for native
Ext models, or the canonical model for unified Ext. For Tor it copies the
boundaries into cochain indexing `C^(-s)=C_s`, retaining the extra incoming
term. It does not compute quotient bases. Treat the returned complex and
derived models as read-only; Ext returns its retained object.

## Compose Ext classes

```julia
E_LN, c, z = OP.Advanced.yoneda_product(E_MN, p, beta, E_LM, q, alpha;
    ELN=E_LN, return_cocycle=true)
```

All three models are `ExtSpaceProjective`s. Coordinates `beta` and `alpha`
belong to their specified quotient bases and have lengths `dim(E_MN,p)` and
`dim(E_LM,q)`. This computes left after right:
`Ext^p(M,N) × Ext^q(L,M) → Ext^(p+q)(L,N)`. Degrees are nonnegative;
the right resolution must extend through `p+q`. Middle modules must agree in
field, finite order, stalk coordinates and structure maps. Equal dimensions
alone do not identify them. A supplied `ELN` fixes target coordinates and must
represent the same source `L` and target `N` through the output degree.

The ordinary result is `(target, coordinates)`. With `return_cocycle=true`,
the third result is the computed composite cocycle in the returned target
model. A different target resolution is handled by an actual comparison
lift. No extra Koszul sign is applied in this composition convention.
The matrix-input overload returns a tensor indexed by target coordinate,
left input column, right input column; a right class is lifted once for
all supplied left classes. No extension sequence is constructed by this call.

## Multiplication tables and Tor structures

```julia
A = OP.ext_algebra(enc; maxdeg=2)
mu = OP.Advanced.multiplication_matrix(A, 1, 1)
```

`ExtAlgebra` retains one projective self-Ext model. The matrix has shape
`dim(A,p+q) × (dim(A,p)*dim(A,q))`. Column `(i-1)*dim(A,q)+j`
is the product of left basis vector `i` with right basis vector `j`.
It therefore acts on `kron(left,right)`. Only nonnegative degree pairs within
the computed range are accepted. Tables are computed lazily and cached by
degree pair; `cached_product_degrees` reads existing cache keys without work.
Returned cached matrices are read-only: copy before editing.

`OP.Advanced.TorAlgebra(T; mu_chain, mu_chain_gen, unit_coords)` equips an
existing Tor space with **supplied** chain multiplication. A pair `(p,q)`
requires a matrix from `C_p ⊗ C_q` to `C_(p+q)`, with the same tensor ordering.
Missing products with nonzero source and target are errors. An optional
generator supplies sparse matrices lazily. Use the owner setters
`OP.DerivedFunctors.set_chain_product!` and `set_chain_product_generator!`
to replace data and invalidate caches; do not mutate stored arrays or
captured generator state.

Every newly computed Tor multiplication table verifies that products of
cycles remain cycles and products involving boundaries give zero homology
coordinates. `check_tor_algebra(A; algebraic=true)` additionally checks the
Leibniz rule, associativity **on homology**, and both unit identities when
`unit_coords` was supplied. Its report states the verified degree range.
Leibniz checks need the extra incoming-boundary degree; it cannot be silently
omitted. This is not a claim of strict chain associativity or of a unit where
none was supplied. Exact fields use exact equations; `RealField` retains
numerical tolerances and rejects nonfinite data.

## Visualization requests

| Input and kind | Selection | Work and meaning |
| :--- | :--- | :--- |
| Hom, Ext, Tor or algebra: `:graded_dimensions` | `matrix_limit=(12,12)` | Read computed dimensions only; no new products or quotient bases. Ext/Tor/algebras use this default. Hom retains its default `:hom_basis` view. |
| Ext/Tor or algebra: `:derived_class` | `degree=0`, `vector=nothing` | Select quotient coordinates; omitted vector selects the first basis class, or zero in a zero group. Compute/check its representative and adjacent differentials. |
| `ExtElement` or `TorElement` | No degree/vector override | Use the element's retained algebra, degree and coordinates. |
| Algebra: `:product_table` | `p=0`, `q=0`; optional `left` and `right` coordinate vectors | Compute/reuse one complete pair of degrees. Both selection vectors must be supplied together. |
| Projective Ext: `:yoneda_product` | `right`, `p`, `q`, `left_vector`, `right_vector`, optional `target` | Compute the selected left-after-right composition and its cocycle. The right resolution reaches `p+q`. |

All views accept `matrix_limit=(rows,columns)` with positive integer limits.
This crops drawing only; full matrices and mathematical selections are
retained in metadata. Exact coefficient strings distinguish modular and
rational answers. Empty groups and matrices retain their actual shapes.

Tables have `max_products=4096` and `max_entries=1_000_000`. They bound input
basis pairs and output structure constants respectively, before multiplication
is requested. They do not bound all intermediate resolution/lift storage.
`validate_algebra=false` is the default; setting it to `true` for a Tor
algebra requests the broader algebraic validation and can generate other
degree pairs beyond the selected table's budget. Use this consciously on
bounded problems. It is rejected for Ext algebras.

`OP.Advanced.check_visual_request` validates request shapes, degrees and
budgets without computing products. Model compatibility and actual algebraic
equations are checked during specification construction. Invalid requests
raise `ArgumentError`; incompatible mathematical models or invalid chain
products are rejected by their owning computation. Finite-field coordinates
accept exact integers/rationals, not silent conversion of decimal inputs.

`OP.Advanced.visual_spec` returns a `VisualizationSpec`. Its `metadata`
retains provenance; class views add `selection`, table views add `matrix`
and `basis_pairs`, and Yoneda views add the actual `target_space` and the
boundary adjustment from the composite to the chosen representative.
Numerical zero-class and equation checks use the field tolerances.
These are static views with native Cairo/WGL rendering where installed.
They do not create linked algebra sessions or infer extension modules.

For products drawn as maps between terms of an actual supplied exact sequence,
use its [exact-sequence view](visualization.md). The
[implementation account](../../implementation/yoneda_products.md) explains
the product and representative checks. Spectral sequences and complex maps
have their own [complex guide](../../tutorials/complexes.ipynb).
