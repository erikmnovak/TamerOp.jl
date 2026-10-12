# Source representatives

Source geometry, reduction coordinates and a correspondence between cell bases
are separate retained objects. The [source-cell guide](../../tutorials/source_cells.ipynb)
develops the workflow; [the implementation account](../../implementation/source_representatives.md)
explains verification and cost.

## Retention and ordinary queries

`persistence_diagram(data, filtration; source=true)` and
`cubical_persistence(values; source=true)` retain source geometry and verified
maps. This is opt-in and uses explicit complexes. It does not imply
`representatives=true` or `cocycles=true`. With neither, the diagram still has
only interval answers. A supplied `FilteredComplex` or `SourceCorrespondence`
can be passed directly to `persistence_diagram`; this explicitly retains that
source data. An ordinary correspondence must have one parameter and agree
with the requested `order`; its field is authoritative. Ordinary source-class
retention requires a verified splitting retraction. General maps without this
certificate can still be applied explicitly with `lift_source_chain`.

Supported automatic geometry comes from the filtered-complex input contract.
Source retention does not invent coordinates for a distance matrix or a
cellular complex. Supply a `FilteredComplex` with actual geometry in those
cases. Cubical arrays preserve exact grades, sublevel/superlevel order and
periodic identifications. One-dimensional vertex arrays are drawn in a plane
with second coordinate zero. Sampling and graph sparsification remain part of
the retained model. A collapse request additionally retains the uncollapsed
version of that same model and a verified correspondence.

`Advanced.persistence_representative(diagram; dim, kind=:finite, index=1,
scale=nothing)` selects an original interval member. A scale defaults to birth
and must lie in the selected interval and recorded construction window. The
computed cycle and death-filling records remain available as `cycle` and
`bounding_chain`. With source retention, `source_cycle` is a checked
`SourceChain` at that scale; `source_bounding_chain` is checked at finite death.
Their boundary equation is verified on original cells. Without source
retention those additional fields are `nothing`. Without representative
retention the query reports `available=false`; provenance text cannot enable it.

`Advanced.persistence_cocycle(diagram; dim, scale, kind=:finite, index=1)`
additionally returns `source_cochain` when a verified source retraction was
retained. This cochain is restricted to active source cells and checked against
their coboundary equation. A source lift alone is insufficient: cohomology is
contravariant. Cycles and cocycles are chosen independently, including among
equal intervals. Their member indices do not establish duality.

## Correspondences and chain queries

The following operations and types live in `TamerOp.Advanced`, with ownership
in `TamerOp.DataIngestion`.

| Operation | Contract |
| --- | --- |
| `source_correspondence(computed, source=computed; field=PrimeField(2), lift=nothing, retraction=nothing)` | Accepts two `FilteredComplex` objects. Default maps match exact source identities and contract the complement at equal grade sets. Explicit maps are sparse-converted over the field and checked. |
| `source_correspondence(data, filtration; field=PrimeField(2), cache=nothing)` | Retains actual constructed geometry and, for collapse, the same model before collapse. |
| `lift_source_chain(c; dim, coefficients, parameter, kind=:cycle)` | Accepts one dimension-local coefficient vector. Checks input and source activity and cycle/cocycle equations. `kind=:chain` permits arbitrary chains; `:cocycle` requires a retraction. |
| `source_chain_data(chain)` | Copies support indices, source IDs, oriented geometry, grade sets, residues, field, parameter and metadata. |
| `check_source_correspondence(c; throw=false)` | Checks complex validity, filtered chain-map equations and any splitting equation. Returns `SourceValidationSummary`. |
| `check_source_chain(chain; throw=false)` | Checks geometry/complex validity, support activity and the declared boundary/coboundary equation. Returns `SourceValidationSummary`. |
| `optimize_representative(chain; weights=nothing, max_candidates=100000)` | Exact minimum weighted support in the nonzero cycle's homology class at its parameter; refuses if the finite boundary-image search exceeds the budget. |

`SourceCorrespondence`, `SourceChain` and `SourceValidationSummary` have compact
`show` and structured `describe` methods. Treat their arrays as read-only.
Source correspondences support `QQField()` and prime fields; the ordinary
persistence and finite enumeration entrypoints require prime fields.
Maps cover every retained chain degree, including zero: rows index the
destination, columns the domain. Pad absent top degrees explicitly. Map
coefficients must belong to the chosen exact field or coerce from integers;
rational coefficients are accepted over `QQField()`.
Automatic source identities must be unique within each dimension. Supplied
geometry describes cell support and need not certify an embedding.

General multiparameter and multicritical complexes are accepted by the map and
chain operations. A cell is active when any one appearance grade precedes the
parameter under all declared coordinate orientations. A source correspondence
does not identify an unrelated finite-module basis, a class on another slice,
or a discarded point solely because their displayed positions are close.

The optimizer's weights are finite nonnegative integer, rational or floating
values, one per source cell in the chain degree. Their default is one. It
minimizes the sum of weights on nonzero coefficients, with deterministic
enumeration ties. Metadata records the parameter, exact objective, weights,
candidate count, original coefficients and a `boundary_adjustment` witness.
The input must be a nonzero homology class. Invalid weights, an exceeded budget,
inactive support, a noncycle or a zero class raise `ArgumentError`. The guarantee
concerns the active finite complex, not geometric length or later parameters.

## Drawing and persistence

`visualize(chain; view_dims=nothing, max_cells=10000, max_labels=50, labels=true)`
draws degree-zero, -one or -two support on the retained source complex. Source
coordinates above dimension three require an explicit two- or three-coordinate
projection. Metadata retains exact data and displayed/omitted counts. The background and
selected-support passes each obey `max_cells`; the compact source readout lists
at most eight records and reports omissions. Coefficient
labels use the oriented source basis; the readout gives signed boundaries.
Periodic seams are a fundamental-domain drawing. Higher chain degrees remain
available algebraically and are rejected by this overlay renderer.

For an ordinary `inspection_session(diagram; dim)`, select an interval and its
original member, then request `representative=true`. Available source cycles
are drawn at birth and fillings at death. Static snapshots and the live browser
view consume the same records. Ordinary barcode plotting, cell selection and
rendering never compute missing representatives or optimize a cycle.

`persistence_diagram_data` and the owned persistence JSON format preserve
retained source complexes and maps. Restoring checks their equations; automatic
contractions are recomputed and compared. The safe codec accepts supported
data-valued metadata and rejects arbitrary callbacks. Saving an endpoint-only
diagram does not recover omitted source data.

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

`component_history(cells)` constructs a `ComponentHistory` from the actual
one-skeleton of a `FilteredComplex`. It retains original vertex and edge
identities and exact births and merges. Each nonzero edge boundary must have
two opposite unit coefficients; a zero-boundary loop cannot merge components.
Higher cells do not change H₀. Under this incidence contract, the answer is
independent of the coefficient field.

Start with `component_count(history; scale)` when only the count is needed.
`components_at(history; scale)` materializes the partition, with rows containing
`component`, `node`, `vertices` and `sources`. Vertex indices are dimension-local
positions in the retained source; `sources` contains its original identities.
`component_members(history; scale, component)` follows the chosen birth vertex.
Its status is `:not_born`, `:alive` or `:merged`. After a bar dies, the returned
vertices belong to its current descendant component. They do not constitute a
claim that the selected bar remains alive.

For a one-parameter source, `scale` uses the original parameter, including
superlevel filtrations. For a multiparameter source, supply
`slice=(base=(...), direction=(...))` when constructing the history. Queries
then use the affine parameter *t* of `base + t * direction`. Every direction
coordinate must be strictly positive in the source's declared oriented order;
zero-direction slices are rejected. With mixed orientations, a decreasing
source coordinate therefore requires a negative direction. A multicritical
cell enters at the earliest slice parameter that reaches any of its appearance
grades. No sampling or direction normalization is applied. This produces a
history for that slice, not a universal bifiltration dendrogram or a ToMATo
clustering tree for a different filtration.

All events at the query scale are included. The older birth survives a merge;
equal births use the smaller source vertex index. Edges at the same grade are
processed in source order. The resulting binary order resolves a simultaneous
event reproducibly, without asserting that those merges occurred at distinct
times. Zero-length births are retained in the forest, omitted from ordinary
intervals by default, and available with `include_ephemeral=true`.

The detailed operations belong to `Advanced` (owner `OrdinaryPersistence`):

| Operation | Result |
| --- | --- |
| `component_history_summary(h)` | Counts and conventions without materializing memberships; also available through `describe`. |
| `component_intervals(h; include_ephemeral=false)` | Rows with source birth-vertex label, birth, death, immediate absorbing label and source identity. Essential death is `nothing`, relative to the retained finite source complex. Superlevel endpoints retain original units. |
| `component_merges(h)` | Merge-node index, exact scale, actual edge and source ID, both child nodes and survivor/absorbed labels. Leaf nodes are vertex indices; merge nodes follow them. |
| `component_parameter(h, scale)` | The corresponding original parameter tuple. |
| `check_component_history(h; throw=false)` | Source and forest validation, including reconstruction of the exact events. This is an explicit full check. |

`persistence_diagram(h; field=PrimeField(2))` produces its ordinary H₀ diagram.
Metadata retains the birth-vertex labels for finite and essential intervals;
these labels refer to the history, rather than a reduction-basis representative.
Keep the history for membership queries: this generated diagram contains
intervals and labels, without copying a source correspondence. Conversely, `component_history(diagram)` requires retained source data and
checks equality of its H₀ interval multiset. Equal intervals do not identify
individual basis representatives. A `SourceCorrespondence` input requires a
verified retraction and graph-incidence complexes on both sides; it checks
original/computed H₀ interval multisets and uses the original source for all
memberships. Passing a reduced `FilteredComplex` alone describes that retained
model; it cannot restore removed vertices without the original source. The
correspondence/diagram entrypoints compare one-parameter
inputs; a multiparameter slice history takes the retained original
`FilteredComplex` directly. An unrelated encoding or unsupported reduced basis cannot supply
source membership by proximity or by equality of endpoints alone.

### Static and linked component views

`visualize(h; scale)` shows the H₀ intervals, actual source components and merge
forest. `component` selects a source birth vertex. `event` selects a merge node,
moves to its exact scale, and by default selects the absorbed birth label.
Source geometry uses two or three coordinates; above three, give an explicit
`view_dims` projection. Superlevel charts use increasing oriented progress
`-parameter`, while controls and exact records retain the original parameter.
The source drawing records periodic identifications rather than inferring
connectivity from visible distances. Distinct exact event or query values can
round to the same drawing coordinate; the readout flags this while retaining
the exact query and membership.

`max_cells=10000` bounds the source drawing. `max_branches=100` separately bounds
the number of interval rows and displayed forest nodes, with selected records
prioritized. Omitted nodes and branches are reported; the renderer does not
invent a replacement merge. `max_labels=30` limits labels and the compact
membership readout. `show_ephemeral=false` hides empty bars by default. These
budgets do not alter the full partition or exact selection in snapshot metadata.

`inspection_session(h; scale, ...)` creates a `ComponentInspectionSession`.
Render it through `visualize(session; backend=:wglmakie)`. The browser links bar,
2D source-vertex and forest-node clicks to exact scale, component and event
controls. Three-dimensional source geometry remains rotatable, with selection
available through bars, the forest and controls. Previous/next moves between
actual source event scales. Hover reads prepared identities only.
`Advanced.select_inspection!` changes `scale`, `component` or `event`;
changing scale or component clears an event unless one is explicitly supplied.
Invalid changes preserve the prior snapshot. Independent rendered views share
selection and release their own listeners on disconnection. Closing the session
releases its input and listeners while retaining the final static snapshot.

Construction stores a linear-size forest and sorts edge events. Counts traverse
that forest without enumerating members. Full memberships and visualization
snapshots materialize source vertex partitions; display budgets limit drawing,
not that algebraic query. Treat retained source and history arrays as read-only.

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

A selected image in a finite homology module can be expressed on retained
source cells with `propagation_source_chain`. Its ordered target homology basis
is supplied explicitly; the routine checks that basis but does not infer its
identification with module coordinates. See the
[selected-vector source link](modules.md#link-to-retained-source-cycles).
