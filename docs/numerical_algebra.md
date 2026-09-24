# Numerical derived algebra

`RealField` computes numerical kernels, images and quotient coordinates using
the tolerances stored in the field. It does not supply the exact-rank guarantee
of `QQField` or a prime field. On well-conditioned data whose nonzero singular
values are separated from the rank cutoff, Ext and Tor should nevertheless
have the mathematically expected dimensions and satisfy their map and product
identities within the declared tolerance. A wrong dimension on such a fixture
is a defect, not an acceptable consequence of using floating point.

```julia
using TamerOp

field = RealField(Float64; atol=1e-12, rtol=1e-10)
```

Use the same field, including its tolerances, throughout a derived computation.
Changing tolerances can change a numerical quotient and its coordinates.
Complexes, homology/cohomology data, and spectral subquotients retain this
field through construction, lazy representatives, shifts, cones, and induced
maps. Raw matrix constructors accept `field=...`; omitting it infers the
default field from the coefficient type. Summaries and provenance expose the
stored field so a custom tolerance remains inspectable.

For numerical cancellation, choose a positive absolute tolerance appropriate
to the units of the data. The default `RealField` has `atol=0`: a mathematically
zero product represented by tiny rounding residuals can then fail a purely
relative consistency check. Retaining a user-supplied positive tolerance is
essential; it does not justify silently loosening that tolerance.

Independent computations can choose different bases even when they represent
the same homology group. Compare them using a coordinate transport or a
basis-independent mathematical quantity, rather than assuming their matrices
must be entrywise equal.

## Rank and solve decisions

The numerical rank cutoff for a coefficient matrix `A` is

```text
tau(A) = atol + rtol * opnorm(A, 1).
```

The absolute term matters for small matrices or small units of measurement;
the relative term scales with the matrix. Tolerances and matrix entries must
be finite, and tolerances must be nonnegative. Rescale data if forming the
cutoff or a factorization would overflow.

The field-aware linear-algebra owner exposes operation-specific backends:

| Operation | Explicit real backends |
| --- | --- |
| Rank and nullspace | `:float_dense_qr`, `:float_sparse_qr`, `:float_dense_svd` |
| Column space and full-column solve | `:float_dense_qr`, `:float_sparse_qr` |
| Reduced row echelon form | `:float_dense_rref`, `:float_sparse_rref` |

`backend=:auto` selects the operation's normal backend. RREF uses left-to-right
columns and partial row pivoting; QR selects independent columns; dense SVD
compares singular values with the cutoff. These rank statistics need not agree
near a cutoff. An explicit backend is useful for checking that a computation
lies away from that ambiguous regime.

Sparse QR receives the field cutoff during factorization. Its reported rank
and column permutation are used together when constructing the image and
kernel. Factoring with a different cutoff and then merely counting accepted
diagonal entries is invalid: the accepted columns need not form the prefix
assumed by the subsequent triangular solve.

A solve has a separate consistency test. For each right-hand side `y_j` and
computed solution `x_j`, the full-column solver checks a backward-error bound
of the form

```text
norm(B*x_j - y_j) <= atol + rtol * (norm(B)*norm(x_j) + norm(y_j)).
```

Each column must pass; a large, accurate right-hand side cannot hide an
incompatible smaller one. Scaling a compatible right-hand side should not
cause rejection merely because the coefficient matrix stayed small. This
residual criterion checks compatibility with the selected numerical image; it
does not restore a direction discarded by the rank decision. The existing
explicit `check_rhs=false` option bypasses this consistency check.

## Derived-algebra oracles

The focused A74 tests first enforce the exact hand-derived answers on the
older zero/identity fixtures: A2 Ext dimensions, the diamond's minimal Betti
table, and nonzero connecting maps for a nonsplit short exact sequence. Real
coefficients also run the Ext/Tor functoriality, long exact sequence, hyperTor
and Ext-action tests that previously skipped them. Matrix identities use the
field tolerance; dimensions and ranks still have exact expected integer values.
Numerical `Hom` forms sparse naturality constraints and computes their
field-aware nullspace. Streaming exact-zero elimination is reserved for exact
fields: rounding residuals in dependent naturality equations must not become
additional pivots that remove actual morphisms.

Two further fixtures start from rational data and convert those same matrices
to `Float64`:

- On the diamond, the direct sum of all four simples and the constant
  projective-injective module has Ext dimensions `(7,4,1)`. Rational changes of
  basis mix its stalk coordinates. The first two Hom differentials have ranks
  `(3,1)` and their product vanishes. The tests check these facts, projective
  and injective dimensions, representative round trips, independently known
  endomorphism spectra, units, strict Yoneda associativity and compatibility
  of products with induced maps in all three variables. Direct `Hom` has
  dimension seven; its basis satisfies naturality and reconstructs both a
  prescribed endomorphism and the complete rational Hom basis after transport.
- A supplied dual-number differential graded algebra has `d(h)=v`, `d(z)=0`,
  `v*h=h*v=z` and zero products between degree-one elements. Rational changes
  of basis produce a dense nonintegral differential with Tor dimensions
  `(1,1,0)`. Explicit homology-coordinate transports compare rational and
  floating products and induced maps. Nonzero terms verify the odd-degree
  Leibniz cancellation; independent boundary changes verify descent, and
  actual module maps preserve products and the supplied unit.

These fixtures use `atol=1e-12`, `rtol=1e-10`. For an identity `X=Y`, the test
requires

```text
norm(X-Y) <= atol + rtol * max(norm(X), norm(Y)).
```

It records the maximum residual divided by
`max(1, norm(X), norm(Y))`. It also records change-of-basis condition numbers,
the nonzero singular values or their condition ratio, and the tolerance.
Rank-deficient differentials have infinite ordinary condition number; the
reported ratio uses their independently known nonzero singular values.
This records the numerical regime being tested instead of treating a passing
dimension alone as a certificate.

On the recorded Julia 1.12.1 run, the maximum scaled residual was
`3.34e-15` for the Ext/Hom fixture and `4.44e-16` for Tor. Both fixtures had
change-of-basis condition numbers below `1.83`; the two Ext differential
condition ratios were `1.57` and `1.00`, and the Tor differential's unique
nonzero singular value was `1.11`. These are observations from the focused
run, not guarantees for arbitrary inputs.

The nonintegral fixtures are not a universal numerical stability proof. In
particular, they do not certify arbitrarily ill-conditioned resolutions,
arbitrary coefficient scales or every floating scalar type. Tor multiplication
still requires explicit compatible product data; numerical tolerance does not
create a canonical product on arbitrary additive Tor groups.

Cached full-column solves require an unchanged coefficient matrix. A change
to the field's effective rank cutoff triggers refactorization. Use
`cache=false` throughout workflows that reuse a matrix object while changing
its entries.

## Near-singular inputs

A singular value or elimination pivot close to `tau(A)` makes numerical rank
sensitive to perturbations, scaling and backend choice. Different operations
can then select different ranks, so the library cannot promise one coherent
numerical quotient across an entire derived computation. A downstream solve,
chain-map check or quotient construction may reject incompatible numerical
decisions. An ambiguous rank is not mathematically exact, and computation
does not silently switch to rational algebra.
When that distinction matters, inspect the singular-value gap, compare
backends or tolerances, and retain the original rational input for an exact
computation.

An approximate kernel vector must still satisfy its residual contract.
Likewise, an image basis must span the selected numerical image. A large
residual caused by mismatched pivot ordering is an implementation error even
if another direction of the matrix is nearly singular. The sparse-QR repair
addresses precisely that error; it is distinct from legitimate disagreement
over whether a tiny singular direction should be retained.

The A58/A59 and A70 contracts remain relevant: RREF must actually return
reduced rows, solves must verify right-hand sides, and products must descend
to quotient classes. Numerical tests complement these requirements; they do
not replace them with dimension inequalities or equality up to sign.
