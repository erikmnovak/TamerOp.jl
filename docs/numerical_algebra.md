# Numerical derived algebra

The algebra of a finite module depends on which vectors are independent, which
are sent to zero, and which represent the same class after taking a quotient.
With floating coefficients, these questions require a numerical rank decision.
This guide explains how tolerances affect derived computations on the retained
finite module and how to assess their results. The
[category guide](math_categories.md) specifies what its Hom, Ext, and Tor
calculations mean; the numerical choices here do not change that category.

`RealField` computes numerical kernels, images and quotient coordinates using
the tolerances stored in the field. It does not supply the exact-rank guarantee
of `QQField` or a prime field. On well-conditioned data whose nonzero singular
values are separated from the rank cutoff, Ext and Tor should nevertheless
have the mathematically expected dimensions and satisfy their map and product
identities within the declared tolerance.

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
does not restore a direction discarded by the rank decision. The
explicit `check_rhs=false` option bypasses this consistency check.

## Checking numerical results

Dimensions alone do not establish that a numerical derived computation has
preserved the expected maps. Check the relevant matrix identities too: for
example, two successive differentials should compose to zero, and a proposed
module morphism should commute with the structure maps. For an identity
`X=Y`, a tolerance-based comparison can use

```text
norm(X-Y) <= atol + rtol * max(norm(X), norm(Y)).
```

Here `atol` and `rtol` are the tolerances chosen for the comparison. Inspect
the residual relative to `max(1, norm(X), norm(Y))` together with the
condition numbers of any changes of basis. For a rank-deficient differential,
the ordinary condition number is infinite; inspect its nonzero singular values
and their separation from the rank cutoff instead. A small residual describes
that computation, not numerical stability for arbitrary input scales.

Numerical `Hom` forms naturality constraints and computes their field-aware
nullspace. The tolerance prevents rounding residuals in dependent equations
from being treated automatically as new independent constraints. Tor
multiplication requires explicit compatible product data; numerical tolerance
does not create a canonical product on arbitrary additive Tor groups.

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

An approximate kernel vector must still satisfy its residual contract, and an
image basis must span the selected numerical image. Likewise, a solve checks
its right-hand sides, and a product on a quotient must be independent of the
chosen representatives. Inspect these properties alongside the numerical rank;
a dimension alone does not determine the resulting algebra.
