# Live and saved inspection sessions

A session associates a mathematical selection with its prepared answer.
Live sessions can query a retained object in Julia. Saved sessions preserve
one or more named answers, their exact selections and their drawing choices.
An included or checked referenced mathematical source permits a supported
live inspector to be reconstructed separately.

The [live-inspection guide](../../tutorials/live_inspection.ipynb) develops this
workflow with a finite encoding. The [reopen-session recipe](../../recipes/reopen_session.md)
saves an ordinary interval for offline use. View kinds, semantic layers,
styles and ordinary figure exports belong to the
[visualization reference](visualization.md).

## Choose the execution contract

| Operation | Result | Execution requirement |
| --- | --- | --- |
| `inspection_session(object; ...)` | A live selected-query session | Julia and a supported mathematical object |
| `Advanced.inspection_snapshot(session)` | Its prepared static specification | No new selection query |
| `visual_state(session)` | A detached record of the selected view | Julia at capture time; no live callback in the result |
| `save_visual_session("view.json", states; ...)` | A versioned declarative session artifact | No plotting backend needed to retain prepared states |
| `save_visual_session("view.html", states; ...)` | A self-contained collection of prepared SVG views and data | CairoMakie at export; browser only after export |
| `load_visual_session(path)` | A `SavedVisualSession` | Julia, with no plotting backend needed merely to load |
| `visualize(saved; state=...)` | A rendering of one stored answer | A supported rendering backend |
| `inspection_session(saved; state=...)` | A newly reconstructed live inspector | Julia and an included or valid referenced supported object |

The HTML state selector works without Julia or network access. It switches
among prepared states; it does not implement arbitrary live selections.
Embedding a mathematical object does not add a Julia runtime to the browser.

## Author a finite module or presentation

`module_editor` constructs an `Advanced.ModuleEditorSession` containing an
independent input draft. The root entrypoints are:

```julia
module_editor(M; history_limit=32, max_vertices=32,
    max_dimension=32, max_entries=20000)
module_editor(; order, dimensions, maps, field=CoreModules.QQField(), ...)
module_editor(; order, upsets, downsets, matrix, field=CoreModules.QQField(), ...)
edit_module!(editor; map=((source, target), coefficient_matrix))
editor_object(editor)
editor_code(editor)
```

`M` may be a finite `PModule` or `FringeModule`. Raw `order[i,j]` records
$i\leq j$, including the diagonal and all transitive relations. Module
`dimensions` has one nonnegative entry per vertex; `maps` assigns exactly one
target-by-source matrix to each cover. A fringe draft instead gives lists of
vertex IDs for `upsets` and `downsets`, with coefficient `matrix` indexed by
downsets in rows and upsets in columns. Support closure, matrix shape and
nonzero coefficients on empty intersections are checked by the fringe owner.
Supports and order relations are never completed automatically.

The [construction guide](../../tutorials/presentations.ipynb) explains these
choices; the [recipe](../../recipes/build_module.md) builds a checked diamond.
No ambient classifier is inferred from a finite diagram or its drawing.

### Draft validity and edit history

Unlike an invalid inspection *query*, an invalid mathematical *draft* commits
and remains visible. `Advanced.check_module_editor(editor; throw=false)`
returns `valid`, `issues`, `witness`, `field` and `input_kind`.
A path witness contains both actual paths and their composite matrices;
see [`module_path_witness`](modules.md#diagnose-a-failed-commuting-relation).
Wrap the report in `Advanced.VisualizationValidationSummary` for compact
notebook display. `throw=true` rejects an invalid draft with `ArgumentError`.
`check_inspection_session` separately checks the session's storage and lifecycle;
a well-formed editing session can contain mathematically invalid input.

`edit_module!` replaces supplied `order`, `dimensions`, `maps`, `field`, or
presentation `upsets`, `downsets`, `matrix` in one undoable transaction. The
single-map form `map=(edge, nothing)` removes an entry. Do not supply `map`
and `maps` together. Changing a dimension does not resize its adjacent maps.
Malformed input types, literal syntax and budget violations leave the draft,
selection and history unchanged. Rational and prime fields use exact equality;
real fields use the owner's multiplication-scaled numerical tolerance.
A coefficient-field edit reinterprets the entered numbers and validates again;
it is not a base-change theorem. For example, `1/3` cannot be entered in F3.

| Advanced operation | Meaning |
| --- | --- |
| `editor_draft(editor)` | Independent copy of entered data, including invalid data |
| `select_inspection!(editor; vertex, edge)` | Select a stalk or entered cover matrix; no history entry |
| `undo_module_edit!(editor)` / `redo_module_edit!(editor)` | Move one step through retained draft edits |
| `reset_inspection!(editor)` | Restore initial input as an undoable edit |
| `inspection_summary(editor)` | Validity, selection, revision, history lengths, budgets and listener diagnostics |
| `inspection_snapshot(editor)` | Static panels, including failed paths and coefficients when present |
| `close_inspection!(editor)` | Release editable data, history and listeners; retain the final snapshot |

History is bounded by `history_limit`; zero disables undo. A new edit clears
redo. Matrix axes and stalk dimensions/support counts are bounded by
`max_dimension`; order axes and support-list lengths by `max_vertices`;
`max_entries` bounds all cover matrices together or the presentation matrix.
These are authoring budgets, distinct from the 12-by-12 displayed matrix crop.

### Export and live controls

`editor_object` constructs a fresh, checked ordinary mathematical object.
`editor_code` generates a self-contained Julia construction from checked
numeric values and qualified APIs. Both require a valid open draft. Code
export supports QQ, prime fields and Float16/32/64 real fields. No entered
text or loaded artifact is evaluated as Julia code. Save returned source with
`write(path, editor_code(editor))` and review it before running it.

`visualize(editor; backend=:wglmakie)` provides keyboard-accessible table
controls for order, dimensions, coefficients, fields and indicator supports,
as well as undo, redo and downloads. Matrix text uses semicolon-separated
rows and exact fractions such as `1/3`; rows and columns explicitly preserve
empty-axis shapes. The field control accepts QQ, F followed by a prime, or
Real(Float16), Real(Float32) or Real(Float64). The live editor requires Julia. For a static figure, render
`inspection_snapshot(editor)` with CairoMakie or WGLMakie instead.

`save_visual_session(path, editor)` retains the draft in the snapshot,
including invalid input and its diagnostics. `module_editor(load_visual_session(path))`
revalidates that draft in Julia and begins a fresh history. Neither callbacks
nor undo history are serialized. Saved HTML shows the retained answer offline;
it cannot edit coefficients. Closing an editor leaves its last snapshot
saveable, while new object/code export requires an open editor.

## Create and update a live session

The root entrypoint `inspection_session` dispatches on the input object.
Object-specific view and query options are documented in the visualization
reference. The shared operations are on `TamerOp.Advanced`:

```julia
A = TamerOp.Advanced
A.inspection_selection(session)
A.inspection_snapshot(session)
A.inspection_summary(session)
A.select_inspection!(session; ...)
A.reset_inspection!(session)
A.close_inspection!(session)
```

`inspection_selection` returns the current exact selected data;
`inspection_snapshot` returns the current static `VisualizationSpec`.
`inspection_summary` describes retained state, capabilities and lifecycle
without requesting a new mathematical answer.

Updates are synchronous and validate a candidate before committing it.
An invalid request preserves the previous session selection and snapshot.
Reentrant updates during notification are rejected. A rendering listener
failure is recorded after the valid state commits. Inputs and returned
mathematical storage must be treated as read-only; a session does not monitor
arbitrary mutation of its underlying object.

Reset follows the selected object's contract. For a module inspector it clears
the selected endpoints and basis opt-in while retaining its view, viewport,
supports and slice line. Other families can restore their own initial query.
Closing is idempotent, releases live resources, and leaves the last static
snapshot inspectable. New queries require a new session.

### Finite modules and encodings

```julia
inspection_session(object; view=:module, box=nothing,
    matrix_limit=(12, 12), cache_limit=16,
    upset=nothing, downset=nothing,
    slice=nothing, slice_scope=:window, slice_limit=512)
```

`object` is a finite `PModule` or an `EncodingResult`. Geometry and original
parameter selections require a supported planar classifier. `box` fixes the
viewing window; it does not change the module. `view=:presentation` requires
a retained fringe witness. `:rank_from` and `:rank_to` fix one endpoint of an
ordinary rank section.

Use one selector at a time: `vertex`, `pair`, `point` or `parameter_pair`.
Parameter selections retain both exact original coordinates and finite labels.
Original-parameter comparability is checked even if two points share a label.
An unrepresented classifier result is distinct from a represented zero stalk.
`input=:pointer` identifies approximate pointer coordinates; supplied values
use `:provided`.

`matrix_limit` restricts displayed entries, not construction of the map.
`cache_limit` bounds selected-algebra and slice caches; zero disables those
caches. Slice requests retain exact line parameters and their scope. A
finite-window endpoint can be censored; a whole-line request must justify
its outer behavior. Interval IDs are local to the current line and are cleared
when that line changes.

### Ordinary intervals

```julia
inspection_session(diagram; dim=0, window=nothing, max_intervals=200)
A.select_inspection!(session; interval=1, member=1, representative=true)
```

`dim` chooses homological degree. A displayed interval group collects equal
decorated intervals; `member` identifies an original interval within that
group. Representatives are read only after explicit request and only when
the original computation retained them. Window and display limits do not
change the retained endpoints. Selection does not rerun persistence.

The same interval-view machinery accepts other barcode families, but a
portable snapshot of such a family does not imply that it supports the
ordinary-diagram live-restoration route.

## Capture a selected view

```julia
frame = visual_state(session)
```

A frame freezes the current static specification, exact selection and
declarative request together with its axes, scales and layout. Later selections
in the original session do not alter it. Supported visual specifications can
also be retained as prepared views; they supply no automatic live inspector
when no reconstruction request exists.

The result type is `Advanced.VisualSessionState`.
`describe(frame)` reports its visual kind, selection, resumable family and
omitted metadata. Style and figure size are supplied separately when saving.
Capture retains the specification's declared windows and axes. It does not
read a renderer's transient camera, pan or zoom state.

For a supported resumable inspector, capture also records the complete source
content identity. This can materialize a lazy encoded module and validate its
maps. A source change away from the selected readout still changes that identity.
For a static frame without a live-restoration request or full-source
fingerprint, capture the specification and supply its selected data explicitly:

```julia
static_frame = visual_state(Advanced.inspection_snapshot(session);
    selection=Advanced.inspection_selection(session))
```

The `selection` keyword defaults to an empty named tuple for a specification.

The declarative record supports exact scalars, coefficient-field data,
supported containers and visual layers. Callbacks, locks, caches, renderer
handles and arbitrary runtime objects are not saved. Opaque metadata that
cannot be retained is recorded as omitted. This does not manufacture an
owned mathematical source from its display string.

A closed session can still supply a static capture. Its released mathematical
object and live callbacks are not recovered by capturing that snapshot.

## Save one state or a named collection

```julia
save_visual_session(path, input; object=nothing, object_path=nothing,
    selected=nothing, caption="", title="Saved mathematical views",
    style=VisualStyle(), size=nothing, provenance=(;), max_bytes=268435456)
```

```julia
save_visual_session("view.json", session;
    object=enc, caption="The selected structure map.")

save_visual_session("views.html", (overview=frame1, selected=frame2);
    selected=:selected, caption="Two prepared views of the same result.")
```

`path` selects JSON or HTML by its `.json` or `.html` extension. Create its
containing directory first. `input` can be a session, a captured state, a
supported visual specification, or a named tuple of those values.
A single input receives the state ID `:view`; a collection keeps its supplied
names. `selected=nothing` chooses the first state. An explicit `selected`
must identify a retained state. State names contain 1 to 256 UTF-8 bytes,
and a collection contains between 1 and 128 states.

`caption` and `provenance` are explanatory data, not executable instructions
or mathematical certificates. `provenance` is a named tuple. The saved
collection uses the supplied `style` and optional `(width, height)` `size`.
Pass the same style explicitly when preserving a nondefault rendering;
capture does not inspect a renderer's current palette.

`max_bytes` defaults to 256 MiB and limits the written artifact. Encoding or
rendering failure leaves an existing destination intact; replacement occurs
only after a complete temporary artifact is prepared.

By default the artifact retains prepared states and readouts. `object=...`
explicitly includes a supported mathematical source for live reconstruction:
a finite module, a supported encoding or an owned ordinary
`PersistenceDiagram`. Mathematical result serialization remains a separate
contract. In particular, an ordinary-diagram JSON artifact alone does not
specify the selected member, view window or display choices.

An attached or referenced source must match every resumable frame's captured
identity. Saving also reconstructs the supported requests and compares their
selections and prepared answers with the retained frames. These checks can
repeat mathematical query work. A different source is rejected even when one
selected readout happens to be equal.

Encoding artifacts normally reconstruct a module through its fringe image,
which need not preserve its original bases. A saved visual source also retains
the original finite module's coordinates so the recorded matrix and a resumed
supported query use the same basis convention. The supported finite source
codec has a default 10,000-vertex limit and checks its module data and
functoriality.

`object_path` provides a checked relative reference to a supported owned
artifact within the saved bundle's directory. The reference records its
content identity. Preserve that resource and its relative location when
moving the bundle. Opening a stored static state does not need the resource;
resuming queries rejects a missing or changed source. An external resource
does not become embedded merely because a view refers to it.

`object_path` may identify an owned visual-source record, encoding artifact
or ordinary-diagram artifact. When both `object` and `object_path` are
supplied, their source content identities must agree, including the stored
module coordinates. A plain encoding artifact can reconstruct different
bases; the owned visual-source representation preserves those coordinates.
An absolute input path is accepted only when its resolved resource lies
inside the bundle; the stored reference is relative.

### JSON and HTML preserve different presentations of the same record

JSON stores the versioned session data for later inspection or rendering in
Julia. HTML embeds the session data and SVG for every retained state, along
with its own state-selection controls. The browser provides the selected
figure, exact selection, mathematical context, plotted-value download and
complete session-record download.

HTML does not fetch a renderer, mathematical source or script from the
network. Its built-in controls require JavaScript; the initial figure remains
visible when those controls cannot run. New mathematical queries still require
Julia. A fragment such as `#state=selected` identifies a named prepared view.

Previously checked composition connectors retain their frozen coordinates and
recorded evidence status. Authored connectors remain authored. Reopening a
picture does not rerun a certificate or relate it to a changed source object.

## Load, inspect and resume

```julia
saved = load_visual_session("views.html")
ids = Advanced.visual_session_states(saved)
selection = Advanced.inspection_selection(saved; state=:selected)
snapshot = Advanced.inspection_snapshot(saved; state=:selected)
visualize(saved; state=:selected)
```

The signature is `load_visual_session(path; max_bytes=268435456)`.
The loader accepts either saved-session representation and returns an
`Advanced.SavedVisualSession`. It validates schema version 1, the content
digest and supported declarative data;
loading HTML reads its embedded record without executing the page's program.
No plotting backend is required until rendering. Omitting `state` uses the
artifact's initial selected state. An unknown state identity is an error in
the Julia API.

`describe(saved)` and `Advanced.inspection_summary(saved)` report state IDs,
initial selection, source mode and identity, resumable families and omissions.
`Advanced.check_visual_session(saved; throw=false)` validates state IDs,
requests, snapshots and reference structure and returns a report with `valid`
and `issues`. `throw=true` rejects an invalid record. This check does not
resolve an external source or run a mathematical query; source-content checks
occur when resuming. The digest detects altered content and is not an
authenticated signature.

```julia
resumed = inspection_session(saved; state=:selected, object=nothing)
```

This constructs a new supported live session from the retained request and
source, then restores its selection through the normal validation path.
Caches and callbacks start afresh. It does not unclose the original session.
An explicitly supplied `object` can replace an unavailable source resource
only when it matches the recorded content identity. It cannot add a new
source identity to an artifact saved without one.

| Saved material | Static restoration | Live restoration |
| --- | --- | --- |
| Prepared specification and selection only | Supported | No mathematical source to query |
| Finite module with a supported recorded request | Supported | Supported in Julia |
| Encoding with supported classifier and recorded request | Supported | Supported in Julia |
| Owned ordinary diagram with recorded interval request | Supported | Supported in Julia |
| Other inspector's prepared snapshot | Supported for supported declarative layers | Requires a separately supported reconstruction route |
| Closed-session capture | Supported | No live request recovered from released runtime state |

Missing source data does not justify inventing a zero answer or silently
recomputing a different object. Invalid selection IDs, incompatible requests,
unsupported schema data and changed referenced identities are reported as
errors at their relevant validation boundary.

## Export a saved figure with its data record

```julia
save_visual("figures", "selected", saved; state=:selected, format=:svg)
save_visual("figures/selected.pdf", saved; state=:selected)
```

These saved-session overloads use the stored style and size unless overridden.
SVG, PNG and PDF exports default to `sidecar=true`: alongside the image at
`path`, they write `path * ".json"`. Thus `selected.svg` has a companion
`selected.svg.json` that can be opened with `load_visual_session`.

The companion is a reopenable static snapshot. It retains the exact available
plotted values, selection, caption, provenance, original source identity and
the image's filename and SHA-256 digest. It contains no full mathematical
source and offers no live-restoration request. The digest records which image
was exported; loading the companion recovers its data without reading that
image. Use `sidecar=false` when only the image is needed.

Both outputs are prepared before either destination is replaced. Each file
replacement is atomic, but the pair is not a filesystem transaction.

`save_visual(...; format=:html)` exports the chosen state as a self-contained
saved-view HTML file. It is likewise a static selected answer with the original
source identity in its provenance, even if the input saved session includes a
mathematical object. Use `save_visual_session` to preserve a collection and
explicitly attach a supported source for new queries.

## Costs and preservation limits

Capturing and saving copies prepared data. A resumable capture also identifies
its complete source, potentially materializing a lazy module and checking its
maps. Saving with that source replays supported requests to check the selected
answers. Multiple states retain multiple
snapshots, and HTML additionally retains every rendered SVG. Including a
mathematical object increases storage and may require materializing the data
needed to serialize it. Runtime caches are excluded.

Restoring a recorded snapshot avoids its original mathematical query; rendering
still has a cost. Resuming an inspector can recompute construction and selected
queries. Display budgets do not establish bounds on that mathematical work.
Exact stored values do not imply exact screen coordinates or pixel-identical
layout across rendering environments.

For the computation behind these choices, see
[linked inspection sessions](../../implementation/inspection_sessions.md) and
[serialization and provenance](../../implementation/serialization.md).

## Query generalized-rank regions

`generalized_rank_session(M; vertices, witnesses=false, boundary=false)` creates
an `Advanced.GeneralizedRankInspectionSession`. `M` may be a finite `PModule`
or an `EncodingResult`. Labels must be distinct, nonempty, connected and
order-convex in its finite poset. QQ and prime fields are supported.
The default retains a scalar rank; `witnesses=true` retains limit, quotient
and comparison matrices. Matrix drawing is bounded by `matrix_limit=(8,8)`.

`boundary=true` additionally requires a positively oriented two-dimensional
grid encoding with the actual product order and column-major labels. It
computes the index-lattice boundary barcode described below. `occurrence=0`
and `bar=0` mean no selection; positive values index the retained path and
barcode entries. Multiplicities are not expanded into indistinguishable rows.

`generalized_rank_session(enc; center, width=0, length=1, levels=(1,2,3),
lengths=(1,2,3), level=1)` instead selects a continuous closed worm on that
supported grid encoding. The center is a planar tuple or vector; integer,
rational and finite floating coordinates retain their exact represented
values. Width must be nonnegative. The selected positive integer level and
length must occur in the declared `levels` and `lengths` families. Grid cells
are right-continuous, with zero below either first axis coordinate and constant
upper tails. Finite-label witnesses and boundary barcodes are rejected here.

Both constructors accept `budget=Advanced.GeneralizedRankBudget()` and
`max_vertices=120` for the finite drawing. Budget failures raise an error
without committing a partial answer. Input objects are treated as read-only;
construct a new session after changing their maps or encoding. An encoding
input goes through `encoding_module`; a lazy input therefore materializes and
caches its full finite module before query/drawing budget validation. These
budgets do not bound initial materialization.

| Selection through `Advanced.select_inspection!` | Computation |
| --- | --- |
| Finite `vertices` | New region rank; clears old occurrence and bar selections |
| `witnesses`, `boundary` | Explicit additional finite-region computations as needed |
| `occurrence`, `bar` | Retained boundary selection; no rank computation |
| Worm `center` | New fixed-probe landscape and worm rank |
| `width`, `length` | New worm rank; reuses the landscape at this center |
| `level` | Displays a retained layer; reuses both landscape and worm rank |

`inspection_selection`, `inspection_summary`, `inspection_snapshot`,
`check_inspection_selection`, `check_inspection_session`, `reset_inspection!`
and `close_inspection!` use the shared Advanced lifecycle. Invalid updates
preserve the old selection, snapshot and counters. Closing releases the input
and listeners but retains the final snapshot. Summary counters distinguish
region/worm queries, landscape calls and boundary calls; they are not a count
of every elimination inside those compound operations.

The WGLMakie viewer has keyboard-accessible exact entry, finite-node toggling,
boundary occurrence/bar picking, worm-center picking, and layer selection.
Pointer coordinates are approximate. Hover only reads retained values.
`visualize(Advanced.inspection_snapshot(session))` and `save_visual` produce
static views; exporting a linked inspector itself is rejected. Live notebook
hosts have separate qualification requirements from the standalone browser.
The [usage guide](../../tutorials/generalized_rank_queries.ipynb) develops these
choices through hand-checkable examples.

### Boundary barcode data

`Advanced.boundary_zigzag(M, grid; vertices, budget)` accepts a finite module
and a supported grid or its compiled wrapper. It returns a
`BoundaryZigzagResult`, with exact inclusive occurrence bars and repeated
finite labels retained. It does not accept a continuous worm or arbitrary path.

- `Advanced.boundary_path(result)` returns copies of `vertices`, `indices`,
  `arrows` and `dimensions`. Indices are lattice positions. Arrow sign `+1`
  means left-to-right; `-1` means right-to-left. Repeated labels are distinct
  positions joined by the indicated maps, including an identity when required.
- `Advanced.boundary_bars(result)` returns `(first, last, multiplicity)` tuples.
  Endpoints are one-based, inclusive path occurrences. Multiplicities are positive.
- `generalized_rank(result)` returns the full-bar multiplicity, checked against
  the original region rank. `describe(result)` reports scope and query count.
- `Advanced.check_boundary_zigzag(result; throw=false)` checks retained shapes,
  barcode dimension reconstruction and full-bar agreement. It does not rerun
  the original module maps.

The algorithm uses all contiguous path ranks and finite differences; it is a
bounded reference barcode computation. For `t` occurrences it requests
`t*(t+1)/2 + 1` ranks including the region check. Algebra budgets count repeated
stalks. No ambient interval decomposition or fast zigzag performance is claimed.
