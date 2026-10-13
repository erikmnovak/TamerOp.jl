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

## Checked summand collections and visual inspection

`summand_collection(r)` adapts an interval result with `witnesses=true` into an
`Advanced.SummandCollection`. One block represents each interval type, retaining
its multiplicity and its actual inclusion and projection. The adapter rechecks
the result before extracting the blocks. `summand_summary(c)` and `describe(c)`
report the field, local block IDs, multiplicities and residual dimensions.
`Advanced.summand_module(c; id)` and `Advanced.summand_maps(c; id)` return
the retained block and its inclusion/projection pair. Use `id=:residual` for
the original residual and `Advanced.source_module(c)` for the full module.
These accessors return read-only references, not copies.

For supplied general pieces, use

```julia
c = summand_collection(M, pieces; inclusions, projections,
    residual=R, residual_inclusion=j, residual_projection=q)
```

The modules must share the identical finite poset and exact coefficient field.
There is one inclusion and projection per piece. If no residual is supplied,
it is the zero module and the pieces must reconstruct all of `M`. Every piece
has multiplicity one unless it comes from the interval-result adapter; no
isomorphism grouping or indecomposability test is inferred for supplied pieces.
`labels` may supply distinct nonempty names. Optional `grades` records have
`generators` and `relations` lists of planar coordinates, one record per block.
These are supplied annotations with counts, not a checked presentation.

`Advanced.check_summand_collection(c; throw=false)` returns a
`SummandValidationSummary` with `valid` and `issues`. It checks functoriality,
cover naturality, every retraction and mixed zero composite, and reconstruction
of the identity at every stalk. `throw=true` raises `ArgumentError` for invalid
storage or equations. Construction accepts `budget=IntervalDecompositionBudget()`;
the source dimensions, finite order, pair count and matrix work are bounded.
Retained modules and maps are references and must remain read-only during use.

`visualize(c)` and `visualize(r)` draw the static `:summand_gallery` recipe.
`summand_session(c; kwargs...)` prepares an `Advanced.SummandInspectionSession`
with the same options. The [worked guide](../../tutorials/decomposition.ipynb)
explains the panels; shared lifecycle and saved-state operations are described
in [visual sessions](visual_sessions.md).

| Option | Meaning |
| :--- | :--- |
| `encoding=nothing`, `box=nothing` | An identical-base planar classifier supplies exact fibers in the viewing window. Otherwise the finite order has schematic coordinates. |
| `drawing=nothing` | Optional `Advanced.SummandSupportDrawing`, mutually exclusive with `encoding`. Its geometry remains unverified. |
| `selected=nothing` | Candidate block IDs; `nothing` means all, `[]` clears selection. |
| `focus=nothing`, `vertex=nothing`, `pair=nothing` | Detail block (first displayed by default), stalk (first by default), and optional structure-map source/target labels. `focus=0` clears detail. |
| `layout=:facets` | `:facets`, `:isolate`, `:overlay`, or `:outline`; the last two require geometric supports. Isolate changes the gallery detail, not the aggregate selected dimensions. |
| `sort_by=:index`, `min_value=0`, `limit=8` | Rank by `:index`, `:multiplicity`, `:support_vertices`, `:total_dimension`, `:max_dimension`, `:generators` or `:relations`; filter by that statistic, then keep up to 16 blocks. Ties use original IDs. |
| `type_filter=:all` | `:interval` selects certified interval blocks; `:thin` means maximum stalk dimension at most one, `:thick` means greater than one. Thinness alone is not an interval proof. |
| `scale=:linear`, `shared_scale=true` | Linear or logarithmic dimension bins. Zero is white; shared scales keep the full module's maximum through selection changes. Independent scales are labelled. |
| `hom=false`, `hom_pair=nothing`, `basis_index=1` | Ordinary Hom table and optional basis component from the selected source/target blocks. Matrix rows are targets, columns sources. |
| `matrix_limit=(8,8)` | Displayed row/column limit; retained matrices remain exact. |
| `max_vertices=120`, `max_summands=128` | Finite diagram and supplied block limits, checked before preparing views. |
| `max_hom_pairs=64`, `max_hom_entries=1_000_000` | Maximum selected pair count and squared unknown count per Hom solve, accounting for potentially dense bases. |

Ranking, type filtering and the display limit apply in that order. The
complement panel contains omitted blocks plus the retained residual, so its
dimensions always add to the selected dimensions to recover the full module.
The residual alone is reported separately. Finite support counts and total
stalk dimensions are sensitive to the finite model; neither is ambient area.
Block IDs are local to the supplied splitting and have no stability or matching
guarantee across different inputs.

`SummandSupportDrawing(boxes; provenance, error_statement=...)` accepts one
list of pairs of planar corners per block. Each box is lower closed and upper
open; overlapping boxes describe a union. At most 10,000 boxes are accepted.
The supplied error statement is displayed as provenance, not verified as a
bound. An exact split can therefore have an unverified support drawing. The
viewer does not provide the separate support-approximation computation.

Live selection uses `Advanced.select_inspection!`, `reset_inspection!` and
`close_inspection!`. Prepared geometry and the checked split are reused; only
explicit Hom requests perform additional algebra. Failed updates preserve the
previous snapshot and selection. Hover reads retained data. Close releases
inputs, Hom spaces and listeners while retaining `inspection_snapshot`.
Static PNG/SVG/PDF and saved finite states are supported; arbitrary live
queries require Julia/WGLMakie. Saved views do not serialize this mathematical
source for reconstruction, and standalone browser checks do not qualify a
notebook host.
