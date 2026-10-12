# Interval decomposition contracts

These operations compute actual interval summands in representations of a
specified finite poset. Begin with the [lesson](../../decomposition.md) for
meaning, the [guide](../../tutorials/decomposition.ipynb) for exploration, or the
[implementation account](../../implementation/interval_decomposition.md) for
algorithms and costs.

## Inputs and calls

```julia
interval_multiplicity(M; vertices=[2,4])
r = interval_decomposition(M; family=[[2,4]], witnesses=true)
is_interval_decomposable(r)
```

`M` is a `PModule` over `QQField()` or a `PrimeField`. `RealField` is rejected.
The complete module must define a functor; structure and path independence are
checked, including maps outside the queried interval. Vertex IDs need not be
in topological order. An interval is a nonempty, connected, order-convex set
of distinct finite labels. Accept vectors, tuples, ranges or sets; reject
Boolean/noninteger labels, duplicates, out-of-range labels and invalid supports.

The root task methods also accept `EncodingResult` with `cache=:auto` or a
`SessionCache`. They use the represented finite module. Query labels name its
poset vertices, not points or an unspecified region in the original parameter
domain. No completeness or invariance under change of encoding is asserted.

| Call | Result and scope |
| :--- | :--- |
| `interval_multiplicity(M; vertices, budget=...)` | Integer multiplicity of that interval in the whole module. |
| `interval_decomposition(M; family=:all, witnesses=false, budget=...)` | `Advanced.IntervalDecompositionResult` with positive counts, residual dimensions, and complete search scope. |
| `interval_decomposition(M; family=[I,J], ...)` | Counts for precisely the supplied distinct intervals; empty families are allowed. |
| `is_interval_decomposable(M; kwargs...)` | Computes a decomposition and returns the three-valued answer below. |
| `Advanced.interval_candidates(M; budget=...)` | Sorted connected convex supports passing degree-zero top/socle and nonzero-stalk pruning; survivors need not occur. Finite module input only. |

For a result, `is_interval_decomposable(r)` returns `true` when all residual
stalks are zero, `false` for a nonzero residual after `family=:all`, and
`missing` for a nonzero residual with a supplied family. Even a supplied
family listing every interval is conservatively marked as supplied; use
`:all` to certify an interval-free residual. Exhausting a budget throws and
is never returned as `false` or `missing`.

## Inspecting the result

Use `interval_decomposition_summary(r)` or `describe(r)` for cheap metadata.
`Advanced.interval_family(r)`, `Advanced.interval_multiplicities(r)` and
`Advanced.residual_dimensions(r)` return copies. Supports are sorted
lexicographically by finite labels, and counts have the same order. Zero
multiplicities are omitted. `Advanced.source_module(r)` and
`Advanced.source_poset(r)` return the retained input objects by reference.
`provenance(r)` states the category, base, field, pairing algorithm, pruning
level and completeness scope.

`Advanced.decomposition_maps(r)` requires `witnesses=true`; otherwise it throws.
It returns a named tuple with the following natural maps and modules:

| Member | Meaning |
| :--- | :--- |
| `summand`, `inclusion`, `projection` | $S$, $i:S\to M$, $p:M\to S$. |
| `residual`, `residual_inclusion`, `residual_projection` | $R$, $j:R\to M$, $q:M\to R$. |

The equations are $pi=1_S$, $qj=1_R$, $pj=0$, $qi=0$, and $ip+jq=1_M$.
At each vertex, the basis of $S$ concatenates the active support blocks in
result order, with one coordinate per copy. Individual block maps can be
selected from the corresponding columns of $i$ and rows of $p$. These bases
and splittings are not canonical. The residual has no further copy of the
queried intervals; only complete search rules out every interval summand.
A residual is not claimed to be indecomposable.

## Bounds and checking

Every task accepts `budget=Advanced.IntervalDecompositionBudget()`:

| Keyword | Default | Quantity bounded |
| :--- | ---: | :--- |
| `max_vertices` | 512 | Size of the entire finite base. |
| `max_dimension` | 2048 | Sum of all stalk dimensions. |
| `max_matrix_entries` | 8,000,000 | Logical entries in each path-validation/order/top/socle/Hom/pairing workspace; also total squared stalk dimensions for retained splittings. |
| `max_antichains` | 10,000 | Nonempty antichains per endpoint support. |
| `max_candidates` | 100,000 | Endpoint pairs before filtering, or supplied intervals. |

Limits are positive non-Boolean machine integers. They limit work and logical
storage, not exact byte usage or the growth of rational coefficients.
Enumeration can be exponential. Budget violations and invalid query families
raise `ArgumentError`; malformed module input also raises validation errors.
No partial result is returned. A supplied family bypasses endpoint enumeration.

`Advanced.check_interval_decomposition(r; budget=..., throw=false)` recomputes
multiplicities, residual dimensions and scope, then checks every retained
splitting and reconstruction equation. Its typed
`IntervalDecompositionValidationSummary` exposes `valid`, `issues`,
`multiplicities_checked` and `splitting_checked`. With no retained maps,
`splitting_checked` is false. Validation budget exhaustion is an invalid report;
`throw=true` raises `ArgumentError`. This is an expensive exact check, unlike
`describe`. Treat retained source data and witnesses as read-only.
