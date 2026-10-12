# Filtered pairs and relative persistence

A filtered pair consists of compatible inclusions Aₜ ⊆ Kₜ. Its relative chain
complex is C(Kₜ)/C(Aₜ), and its persistence maps come from the inclusions of
pairs. The same quotient cell can later become zero when it enters A.

Start with [persistence relative to a boundary](../../recipes/relative_pair.md)
for an executable example. The methods below belong to
`TamerOp.OrdinaryPersistence`. `FilteredPair`, `filtered_pair` and
`persistence_diagram` are available at the package root. The detailed pair and
map methods are available through `TamerOp.Advanced`; shared `basis`,
`dimensions`, `describe`, `field` and `provenance` keep their existing owners.
Field constructors such as `F2`, `F3` and `Fp` belong to `TamerOp.CoreModules`.

## Construct and validate a pair

```julia
pair = filtered_pair(K, cells; grades=nothing, field=F2(), order=:sublevel)
```

K must be a finite one-parameter `GradedComplex` with finite real grades and
integer sparse boundary matrices. Boundary coefficients are interpreted in
the selected supported prime field. The constructor checks shapes, d² = 0,
filtration compatibility and the selected subcomplex. It copies its inputs.
Treat stored arrays as read-only.

| Input | Contract |
| :--- | :--- |
| `cells[d+1]` | Distinct positions in K's degree-d cell vector. Supply one vector per ambient chain degree. IDs are not positions and need not be unique. |
| `grades=nothing` | A inherits the grades of its selected K-cells. |
| Explicit `grades[d+1]` | A-entry grades in the same order as `cells[d+1]`. Conversion to K's scalar type must preserve each value. Sorting positions also sorts their grades. |
| `order=:sublevel` | Ambient entry ≤ subcomplex entry; every nonzero boundary term in A enters no later than its coface. |
| `order=:superlevel` | Both inequalities reverse; grade values stay in their original units. |
| `field` | A supported prime field. Closure and d² use this field; a based chain subcomplex does not certify a geometric realization. |

`check_filtered_pair(pair; throw=false)` returns a report with `valid` and
`issues`; `throw=true` raises an `ArgumentError` for invalid data.
`FilteredPairValidationSummary(pair)` provides readable display of that
report. `filtered_pair_summary(pair)` and `describe(pair)` give cheap cell
counts, field and order. `provenance(pair)` returns copied immutable cell,
index and grade records; it does not retain a geometric source model.

## Compute intervals

```julia
diagram = persistence_diagram(pair)
```

The result is an ordinary `PersistenceDiagram` of the **relative** homology
module, with pair provenance. Degrees are homological and start at zero.
Sublevels use [birth, death); superlevels use (death, birth] in the original
parameter. Zero-length intervals are omitted. Essential deaths remain
separate from exact scalar grades.

Inherited A-grades select direct quotient reduction. Delayed A-entry selects
a filtered mapping cone with a natural quasi-isomorphism to the quotient;
`provenance(diagram).construction.effective` records `:quotient` or
`:mapping_cone`. The field and order come from the pair and are not overridden
on the diagram request.

Ordinary interval accessors, distances, features and owned diagram JSON apply
to this result. Saving it preserves the declared pair provenance, not the
pair's boundary matrices or a reconstructible geometric source. Retained
ordinary cycle/cocycle options are not accepted for this overload: artificial
cone cells must not appear as original source representatives. Use the explicit
quotient-homology queries below for bases and maps. They do not assign those
bases to individual barcode intervals.

## Inspect chains at a scale

`relative_complex(pair; scale)` constructs a snapshot of all three complexes.
The finite scale is compared directly to stored grades, without rounding.
The snapshot copies its pair. All matrices have target rows and source columns.

| Query | Result |
| :--- | :--- |
| `dimensions(c; part=:relative)` | Chain dimensions in degrees 0, 1, … . |
| `relative_cells(c; dim, part=:relative)` | Matrix-order ambient positions, original IDs and part-specific cell-entry grades. |
| `relative_boundary(c; dim, part=:relative)` | A copied sparse differential d_dim. Degree zero has zero rows; the differential one degree above the top has zero columns. |
| `relative_maps(c; dim)` | `inclusion`: C_dim(A) → C_dim(K); `projection`: C_dim(K) → C_dim(K,A). |
| `relative_structure_map(cₛ, cₜ; dim, part=:relative)` | The induced chain map between snapshots of the same pair. |

`part` is exactly `:ambient`, `:subcomplex` or `:relative`. A relative cell
that enters A between scales maps to zero. Sublevel source scales must be
≤ target scales; superlevels require ≥. Different pair contents, fields or
orders are rejected. Above the ambient top degree, only the zero differential
is directly queryable; other dimension queries must name a stored degree.

## Inspect homology and the exact sequence

`relative_homology(pair; scale)` or `relative_homology(c)` computes explicit
homology spaces over the pair's field. This is heavier than a barcode query:
it computes all three complexes' homology in all stored degrees and can use
dense linear algebra. `relative_complex(h)` accesses its retained snapshot.

| Query | Result |
| :--- | :--- |
| `dimensions(h; part=:relative)` | Homology dimensions by degree. |
| `basis(h; dim, part=:relative)` | Copied cycle representatives as columns, ordered by `relative_cells(relative_complex(h); dim, part)`. |
| `relative_maps(h; dim)` | Homology inclusion, projection and connecting maps in the chosen bases. |
| `relative_structure_map(hₛ, hₜ; dim, part=:relative)` | The induced persistence map in the source and target homology bases. |

The maps at degree d have these degrees and signs:

```math
\begin{gathered}
H_d(A_t)\xrightarrow{i_*}H_d(K_t)
\xrightarrow{q_*}H_d(K_t,A_t),\\
H_d(K_t,A_t)\xrightarrow{\delta}H_{d-1}(A_t),\\
\delta[z]=[\partial_K\widetilde z].
\end{gathered}
```

Here z is a quotient cycle and its lift uses the complementary ambient cell
coordinates. The connecting map has degree −1. All other displayed maps have
degree zero. Unreduced homology is used, so the connecting map at degree zero
has zero rows. These maps form the long exact sequence and commute with the
filtration maps; ranks alone do not specify their matrices.

`describe`, `field` and `provenance` apply to both snapshot types. Returned
query matrices are copies. Stored arrays and the retained chain snapshot are
read-only data, as are the original pair's internals.
