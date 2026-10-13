# Filtered pairs and relative persistence

A filtered pair consists of inclusions $A_t\subseteq K_t$ that respect the
filtration: as $K_t$ grows, so does its chosen subcomplex $A_t$. In degree
$d$, its relative chain space is the quotient $C_d(K_t)/C_d(A_t)$, where
$C_d$ denotes chains on the $d$-dimensional cells over the chosen field.
Chains supported in $A_t$ count as zero. A cell can therefore enter the
quotient when it enters $K$ and later become zero when it enters $A$.

The [boundary recipe](../../recipes/relative_pair.md) follows this process
on a segment and computes its relative interval and connecting map. This
page specifies the inputs, returned objects and matrix conventions. Use
these imports for the qualified calls below:

```julia
import TamerOp as OP
import TamerOp.Advanced as OA
using TamerOp.CoreModules: F2
```

`FilteredPair`, `filtered_pair` and `persistence_diagram` are available at the
package root. Detailed chain and homology queries are in `TamerOp.Advanced`.
The pair implementation belongs to `TamerOp.OrdinaryPersistence`; shared
accessors such as `basis`, `dimensions` and `describe` also apply to other
mathematical objects. Prime field constructors, including `F3` and `Fp`,
come from `TamerOp.CoreModules`.

## Construct and validate a pair

```julia
pair = OP.filtered_pair(K, cells; grades=nothing, field=F2(), order=:sublevel)
```

`K` must be a finite one-parameter `GradedComplex` with finite real grades
and integer sparse boundary matrices. Boundary coefficients are interpreted
in the selected prime field. The constructor checks matrix shapes, the
identity $\partial^2=0$, filtration compatibility and the selected
subcomplex. It copies its inputs; treat its stored arrays as read-only.

| Input | Contract |
| :--- | :--- |
| `cells[d+1]` | Distinct positions in `K`'s degree-`d` cell vector. Supply one vector per ambient chain degree, including empty vectors. Cell IDs are labels, not positions, and need not be unique. |
| `grades=nothing` | $A$ inherits the grades of its selected $K$-cells. |
| Explicit `grades[d+1]` | Entry grades in $A$, in the same order as `cells[d+1]`. Conversion to `K`'s scalar type must preserve each value. Sorting positions also sorts their grades. |
| `order=:sublevel` | Each cell enters $K$ no later than $A$. Every boundary term with a nonzero coefficient must enter $A$ no later than the cell whose boundary contains it. |
| `order=:superlevel` | Both inequalities reverse; grade values stay in their original units. |
| `field` | A supported prime field. Boundary closure and $\partial^2=0$ are checked in that field. Selecting cell coordinates that form a chain subcomplex does not certify a geometric realization. |

`OA.check_filtered_pair(pair; throw=false)` returns a report with `valid`
and `issues`; `throw=true` raises an `ArgumentError` for invalid data.
`OA.FilteredPairValidationSummary(pair)` provides readable display of this
report. `OA.filtered_pair_summary(pair)` and `OP.describe(pair)` give cell
counts, field and order without computing homology. `OP.provenance(pair)`
returns copied immutable cell, index and grade records; it does not retain
a geometric source model.

## Compute intervals

```julia
diagram = OP.persistence_diagram(pair)
```

The result is an ordinary `PersistenceDiagram` for the relative homology
module $H_d(K_t,A_t)$, recording the pair in its provenance. Homological
degrees start at zero. Sublevels use intervals $[\mathrm{birth},\mathrm{death})$;
superlevels use $(\mathrm{death},\mathrm{birth}]$ in the original parameter.
Intervals of zero length are omitted. Essential classes retain their exact
birth grades separately from finite intervals; the combined interval accessor
supplies the appropriate infinite death.

If every selected cell inherits its grade from $K$, the computation reduces
a graded quotient directly. If any cell enters $A$ later, it uses a filtered
mapping cone, an auxiliary complex with the same relative persistence
module. The [implementation account](../../implementation/relative_persistence.md)
explains the construction and the isomorphism. The field and order come from
the pair and cannot be overridden in the diagram request.
`OP.provenance(diagram).construction.effective` reports `:quotient` or
`:mapping_cone`.

The usual [interval accessors, distances and features](../../ordinary_persistence.md)
apply to this diagram, as does the owned diagram JSON format. Saving it
preserves the declared pair provenance, without the pair's boundary matrices
or a reconstructible geometric source. This overload does not accept ordinary
cycle or cocycle retention options, because auxiliary cone cells are not
original source representatives. The quotient homology queries below return
bases and maps at a chosen scale; they do not assign those bases to individual
barcode intervals.

## Inspect chains at a scale

`OA.relative_complex(pair; scale)` constructs a `RelativeComplex` containing
all three chain complexes, $C(A_t)$, $C(K_t)$ and $C(K_t)/C(A_t)$, at the
specified finite scale. It compares the scale directly to the stored grades,
without rounding, and copies its source pair. All matrices have target rows
and source columns.

In the table, `c`, `cₛ` and `cₜ` are these chain snapshots. Query names are
qualified by `OA` when called from the imports above.

| Query | Result |
| :--- | :--- |
| `dimensions(c; part=:relative)` | Chain dimensions in degrees 0, 1, … . |
| `relative_cells(c; dim, part=:relative)` | Ambient positions, original IDs and entry grades in matrix order. `part=:subcomplex` reports grades in $A$; the other parts report grades in $K$. |
| `relative_boundary(c; dim, part=:relative)` | A copied sparse differential $\partial_{\mathrm{dim}}$. Degree zero has zero rows; the differential one degree above the top has zero columns. |
| `relative_maps(c; dim)` | `inclusion`: $C_d(A_t)\to C_d(K_t)$; `projection`: $C_d(K_t)\to C_d(K_t,A_t)$, where $d$ is `dim`. |
| `relative_structure_map(cₛ, cₜ; dim, part=:relative)` | The induced chain map between snapshots of the same pair. |

`part` is exactly `:ambient`, `:subcomplex` or `:relative`. A cell that
survives in the quotient at both scales maps to the same ambient cell
coordinate; one that enters $A$ between them maps to zero. For sublevels,
the source scale must be at most the target scale. Superlevels require the
reverse inequality. Different pair contents, fields or orders are rejected.
Dimension queries name a stored degree, except that `relative_boundary`
also accepts the zero differential one degree above the ambient top degree.

## Inspect homology and the exact sequence

`OA.relative_homology(pair; scale)` or `OA.relative_homology(c)` computes a
`RelativeHomology` containing homology spaces of all three complexes in all
stored degrees. This is more work than a barcode query and can use dense
linear algebra. `OA.relative_complex(h)` returns its retained chain snapshot.

In the table, `h`, `hₛ` and `hₜ` are homology snapshots; the query names
are again available through `OA`.

| Query | Result |
| :--- | :--- |
| `dimensions(h; part=:relative)` | Homology dimensions by degree. |
| `basis(h; dim, part=:relative)` | Copied cycle representatives as columns. Rows follow `relative_cells(relative_complex(h); dim, part)`. |
| `relative_maps(h; dim)` | Homology inclusion, projection and connecting maps in the chosen bases. |
| `relative_structure_map(hₛ, hₜ; dim, part=:relative)` | The induced persistence map in the source and target homology bases. |

For each degree $d$, `relative_maps` returns the maps $i_*$, $q_*$ and
$\delta$ in the long exact sequence

```math
\cdots\longrightarrow H_d(A_t)\xrightarrow{i_*}H_d(K_t)
\xrightarrow{q_*}H_d(K_t,A_t)
\xrightarrow{\delta}H_{d-1}(A_t)\longrightarrow\cdots.
```

Exactness means that at each space the image of the incoming map equals
the kernel of the outgoing map. Inclusion and projection preserve degree.
The connecting map lowers degree by one: a relative cycle has boundary zero
modulo $A$, so lifting it to $K$ produces a boundary supported in $A$. The
sign convention is

```math
\delta[z]=[\partial_K\widetilde z],
```

where $\widetilde z$ is the lift using the complementary ambient cell
coordinates. Unreduced homology is used, so the connecting map in degree
zero has zero rows. These maps commute with the filtration maps. Their
matrices use the chosen homology bases; knowing their ranks alone does not
determine those matrices.

`describe`, `field` and `provenance` apply to both snapshot types. Query
matrices are returned as copies. Treat stored arrays and retained chain
snapshots as read-only, including the original pair's internal arrays.
