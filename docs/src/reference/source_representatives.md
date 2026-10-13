# Source representatives

A source representative expresses a chain or cochain on the cells from which
a computation was built. Obtaining it requires three kinds of retained data:
the source geometry, the computed coefficients and a map relating the two
cell bases. The [source cells guide](../../tutorials/source_cells.ipynb)
develops examples of cycles, fillings, cocycles and component memberships;
the [implementation account](../../implementation/source_representatives.md)
explains the equations behind their checks.

## Retention and ordinary queries

`persistence_diagram(data, filtration; source=true)` and
`cubical_persistence(values; source=true)` retain source geometry and checked
maps by constructing explicit complexes. Choose `representatives=true` to
retain cycle and filling data, or `cocycles=true` to retain cochain data. These
are independent choices: source geometry alone does not supply a selected
class representative.

You may also pass a `FilteredComplex` or `SourceCorrespondence` directly to
`persistence_diagram`, which retains that supplied source data. A correspondence
for ordinary persistence must have one parameter and agree with the requested
`order`; its own coefficient field determines the calculation's field. Ordinary
source queries require a splitting retraction: if $i$ sends computed chains to
source chains and $p$ sends source chains back, then $pi$ must be the identity.
A general map without this retraction remains usable explicitly through
`lift_source_chain`.

Automatic geometry follows the
[filtered complex input contract](visualization.md#filtered-complexes-and-source-cells).
A distance matrix or a supplied cellular boundary matrix does not determine
coordinates. Supply a `FilteredComplex` with geometry for those inputs.
Cubical arrays retain exact grades, sublevel or superlevel order and periodic
identifications. A one dimensional vertex array is drawn in a plane with its
second coordinate zero. Sampling and graph sparsification determine the
retained model. A collapse request additionally retains that same model before
collapse and a checked correspondence to it.

### A cycle and its finite death filling

`Advanced.persistence_representative(diagram; dim, kind=:finite, index=1,
scale=nothing)` selects an original member of the interval list. The scale
defaults to birth and must lie both inside the selected interval, with finite
death excluded, and inside the recorded construction window.

The result's `cycle` and `bounding_chain` fields contain the computed cycle
and, for finite death, its filling. When source data were retained,
`source_cycle` is a checked `SourceChain` at the query scale and
`source_bounding_chain` is checked at death. The boundary of that source filling
must equal the source cycle. Without source retention these source fields are
`nothing`; without representative retention the query reports `available=false`.
Descriptive metadata cannot recover coefficients that were not retained.

### A cocycle at a chosen scale

A cochain assigns coefficients to cells and evaluates chains by linearity.
It is a cocycle when it vanishes on boundaries. The query
`Advanced.persistence_cocycle(diagram; dim, scale, kind=:finite, index=1)`
returns `source_cochain` when a checked source retraction was retained.
The cochain is restricted to active source cells and checked against their
boundaries.

This query uses the transpose of the retraction $p$, because evaluation on a
source chain $s$ is evaluation on $ps$ in the computed complex. The map $i$
alone has the wrong direction. Homology and cohomology reductions choose
representatives independently, so matching their interval member indices does
not establish a dual pairing, including when intervals have equal endpoints.

## Correspondences and chain queries

These operations and types belong to `TamerOp.Advanced` and are implemented
by `TamerOp.DataIngestion`.

| Operation | Contract |
| --- | --- |
| `source_correspondence(computed, source=computed; field=PrimeField(2), lift=nothing, retraction=nothing)` | Accepts two `FilteredComplex` objects. By default it matches exact source identities and contracts complementary cells at equal appearance grade sets. Explicit maps are converted to sparse matrices over the field and checked. |
| `source_correspondence(data, filtration; field=PrimeField(2), cache=nothing)` | Retains the constructed geometry and, when collapse is requested, the same model before collapse. |
| `lift_source_chain(c; dim, coefficients, parameter, kind=:cycle)` | Accepts a coefficient vector indexed by cells in degree `dim`. Checks activity and the appropriate boundary equations before and after applying the correspondence. `:chain` permits arbitrary chains; `:cocycle` requires a retraction. |
| `source_chain_data(chain)` | Returns support indices, source identities, oriented geometry, grade sets, coefficients (integer residues over prime fields), field, parameter and metadata. |
| `check_source_correspondence(c; throw=false)` | Checks the complexes, the filtered chain map equations and any splitting equation. Returns `SourceValidationSummary`. |
| `check_source_chain(chain; throw=false)` | Checks the source complex and geometry, support activity and the declared cycle or cocycle equation. Returns `SourceValidationSummary`. |
| `optimize_representative(chain; weights=nothing, max_candidates=100000)` | Finds minimum weighted support within the nonzero cycle's homology class at its parameter. Refuses to enumerate a boundary image exceeding the candidate budget. |

`SourceCorrespondence`, `SourceChain` and `SourceValidationSummary` provide
compact `show` output and structured `describe` results. Treat retained arrays
as read only. Correspondences support `QQField()` and prime fields; ordinary
persistence and finite representative optimization require prime fields.

Supply a map for every retained chain degree, including zero. Rows index the
destination cells and columns index the domain cells, each within that degree.
Pad absent top degrees explicitly. Coefficients must belong to the chosen
exact field or be integers that coerce to it; `QQField()` also accepts rational
coefficients. Automatic source identities must be unique within each dimension.
Geometry describes the cells' support and need not give an embedding.

The map and chain operations accept several parameters and multicritical
cells, meaning cells with several possible appearance grades. A cell is active
when any one of those grades precedes the query parameter in every declared
coordinate order. The correspondence relates the supplied cell bases. It
does not identify an unrelated module basis, a class on another slice or a
point discarded before construction by proximity in a drawing.

### Minimizing weighted support

The optimizer considers cycles obtained from the input by adding boundaries
in the active finite complex. Weights are finite nonnegative integer,
rational or floating values, one for every source cell in the chain degree;
the default is one. The objective sums weights only where a coefficient is
nonzero. Equal objective values keep the first enumerated candidate.

The result's metadata records the parameter, exact objective, weights,
candidate count, original coefficients and a `boundary_adjustment` witness.
The witness certifies that the new cycle differs from the original by a
boundary. A zero homology class, inactive support, a noncycle, invalid weights
or an exceeded budget raises `ArgumentError`. The minimum concerns the given
weights and parameter; it need not minimize geometric length or persist as a
minimum after the active complex changes.

## Drawing and saving source results

`visualize(chain; view_dims=nothing, max_cells=10000, max_labels=50, labels=true)`
draws support in degrees zero, one and two over the retained source complex.
With more than three source coordinates, supply an explicit projection onto
two or three coordinates. Higher chain degrees remain available algebraically
but are rejected by this renderer.

The background and selected support each obey `max_cells`. Metadata retains
exact data and reports how many cells were displayed or omitted. The compact
source readout lists at most eight records and reports additional omissions.
Coefficient labels use the oriented source basis, and the readout gives signed
boundaries. Periodic geometry is drawn in a fundamental domain whose boundary
identifications remain part of the source complex.

For an ordinary `inspection_session(diagram; dim)`, select an interval and its
original member, then request `representative=true`. Available source cycles
are drawn at birth and fillings at death. Static snapshots and live browser
views use the same records. Drawing a barcode or selecting a cell does not
compute missing representatives or optimize their support.

`persistence_diagram_data` and the persistence JSON format preserve retained
source complexes and maps. Loading checks their equations; automatic
contractions are recomputed and compared. The data codec accepts supported
metadata values and rejects arbitrary callbacks. Saving intervals alone cannot
restore omitted source cells or maps later.

```@docs
TamerOp.DataIngestion.SourceCorrespondence
TamerOp.DataIngestion.SourceChain
TamerOp.Advanced.SourceValidationSummary
TamerOp.Advanced.source_correspondence
TamerOp.Advanced.lift_source_chain
TamerOp.Advanced.source_chain_data
TamerOp.Advanced.check_source_correspondence
TamerOp.Advanced.check_source_chain
TamerOp.Advanced.optimize_representative
```

## Components and merge histories

`component_history(cells)` constructs a `ComponentHistory` from the vertices
and edges of a `FilteredComplex`, its *one-skeleton*. The history retains
original vertex and edge identities and exact births and merges. Each edge
with nonzero boundary must have two opposite unit coefficients, giving its
endpoints. A loop with zero boundary cannot join two components. Higher cells
do not change degree zero homology, so under this incidence condition the
component history is independent of the coefficient field.

Start with `component_count(history; scale)` if you only need a count.
`components_at(history; scale)` returns the partition in rows containing
`component`, `node`, `vertices` and `sources`. Vertex indices are positions in
the source vertex list; `sources` contains the original source identities.
`component_members(history; scale, component)` follows the chosen birth vertex.
Its status is `:not_born`, `:alive` or `:merged`. After the selected bar dies,
its vertices belong to the returned descendant component, while the bar
itself remains ended.

For a source with one parameter, `scale` uses the original parameter units,
including superlevel thresholds. For several parameters, construct the history
with `slice=(base=(...), direction=(...))`. Queries then use the scalar $t$
along `base + t * direction`. Each direction coordinate must advance strictly
in its declared source order; a superlevel coordinate requires a negative
direction, and zero direction coordinates are rejected. A multicritical cell
enters at the earliest slice parameter reaching any of its appearance grades.
No direction normalization or sampling is applied. The resulting history
belongs to this line, rather than assigning a universal merge tree to the
multiparameter filtration.

All events at the query scale are included. The older birth survives a merge,
with the smaller source vertex index breaking equal birth ties. Edges at the
same grade are processed in source order. This gives a reproducible binary
forest for a simultaneous event without assigning different times to its
merges. Births that die at the same parameter remain in the forest.
`component_intervals(h; include_ephemeral=true)` includes their empty intervals
in its returned rows; an ordinary barcode omits them.

The detailed operations belong to `Advanced`, with implementation in
`OrdinaryPersistence`:

| Operation | Result |
| --- | --- |
| `component_history_summary(h)` | Counts and conventions without listing memberships; also available through `describe`. |
| `component_intervals(h; include_ephemeral=false)` | Rows containing a source birth vertex label, birth, death, immediate absorbing label and source identity. Essential death is `nothing`, relative to the retained source. Superlevel endpoints keep original units. |
| `component_merges(h)` | The merge node, exact scale, joining edge and source identity, both children and the survivor and absorbed labels. Leaf nodes are vertex indices; merge nodes follow them. |
| `component_parameter(h, scale)` | The corresponding original parameter tuple. |
| `check_component_history(h; throw=false)` | Checks the source and forest, including reconstruction of exact events. This performs a full validation. |

`persistence_diagram(h; field=PrimeField(2))` produces the ordinary degree zero
diagram. Metadata retains birth vertex labels for finite and essential
intervals. Those labels belong to the history and need not be the representatives
chosen by an independent reduction. Keep the history for membership queries:
the generated diagram contains intervals and labels without copying the
source correspondence.

Conversely, `component_history(diagram)` requires retained source data and
checks equality of degree zero interval multisets. Equal intervals do not
identify individual basis representatives. A `SourceCorrespondence` input
requires a checked retraction and complexes satisfying the graph incidence
condition on both sides. It compares their degree zero interval multisets
and obtains memberships from the original source. These correspondence and
diagram methods accept one parameter; for a slice of a multiparameter source,
pass the retained original `FilteredComplex` directly.

A reduced `FilteredComplex` supplied alone describes its retained cells and
cannot recover removed vertices. An unrelated encoding or unsupported reduced
basis likewise cannot provide source membership solely through equal endpoints
or positions in a picture.

### Static and linked component views

`visualize(h; scale)` shows the degree zero intervals, source components and
merge forest. `component` selects a source birth vertex. `event` selects a
merge node, moves to its exact scale and by default selects the absorbed
birth label. Source geometry uses two or three coordinates; give an explicit
`view_dims` projection for more coordinates.

Superlevel charts show increasing progress as `-parameter`, while controls
and exact records keep the original parameter units. Connectivity comes from
source incidences, including periodic identifications. If distinct exact
values round to the same drawing coordinate, the readout reports this while
retaining the exact selection and membership.

`max_cells=10000` bounds the source drawing. `max_branches=100` separately bounds
interval rows and displayed forest nodes, prioritizing selected records and
reporting omissions. No extra merge replaces an omitted node.
`max_labels=30` limits labels and the compact membership readout.
`show_ephemeral=false` hides empty bars by default. These drawing limits leave
the full partition and exact selection in snapshot metadata unchanged.

`inspection_session(h; scale, ...)` creates a `ComponentInspectionSession`;
render it with `visualize(session; backend=:wglmakie)`. The browser links bars,
source vertices in a planar view and forest nodes to exact scale, component
and event controls. Three dimensional source geometry remains rotatable;
selection is available through bars, the forest and controls. Previous and
next buttons visit source event scales. Hover reads already prepared identities.

`Advanced.select_inspection!` changes `scale`, `component` or `event`. Changing
the scale or component clears the selected event unless one is explicitly
supplied, and invalid changes preserve the previous snapshot. Independent
rendered views share selection and release their own listeners when disconnected.
Closing the session releases its input and listeners and retains the final
static snapshot. The [live inspection guide](../../tutorials/live_inspection.ipynb)
explains this shared session behavior.

Construction sorts edge events and stores a forest of linear size. Counting
components traverses the forest without listing their vertices; full membership
queries and visualization snapshots construct the source vertex partition.
Drawing limits do not reduce that algebraic query. Treat retained source and
history arrays as read only.

```@docs
TamerOp.component_history
TamerOp.component_count
TamerOp.components_at
TamerOp.component_members
TamerOp.OrdinaryPersistence.ComponentHistory
TamerOp.Advanced.component_history_summary
TamerOp.Advanced.component_intervals
TamerOp.Advanced.component_merges
TamerOp.Advanced.component_parameter
TamerOp.Advanced.check_component_history
TamerOp.Advanced.ComponentInspectionSession
```

To express a selected image in a finite homology module on source cells, use
`propagation_source_chain`. Supply the ordered target homology basis explicitly.
The routine checks that basis but relies on the declared correspondence with
module coordinates. See the
[selected vector source link](modules.md#link-to-retained-source-cycles).
