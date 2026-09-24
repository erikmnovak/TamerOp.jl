# Inspection and explicit computation

Displaying a mathematical object should not silently solve another mathematical
problem. `show`, `describe`, and owner summary functions inspect stored data.
They may count or summarize that data, but do not construct missing modules,
fiber ranks, spectral pages, homology representatives, or geometry caches.
Inspection is not necessarily constant time: summarizing an existing dimension
vector or sparse support still scans its entries.

For an ingestion encoding `enc`, choose the computation you need:

```julia
describe(enc)              # stored metadata, provenance, materialized status
dimensions(enc)            # dimension vector only; cached for subsequent calls
pmodule(enc)               # full module, including structure maps; cached
```

Before a dimension query, `describe(enc).module_dims` is `nothing` and the
plain-text display says `not computed`. Afterward, the summary reports the
cached vector without constructing the full module. `materialized` distinguishes
these states: having all dimensions does not mean that structure maps exist.
A full module request also makes its dimensions available without repeating
the dimension computation.

`encoding_poset`, `encoding_map`, `encoding_axes`, and provenance accessors do
not materialize the module. An encoded complex's `encoding_complex` accessor
returns its stored complex, which can itself be lazy. Its summary reports the
number of stored terms and differentials; printing it does not fill them.
`encoding_module` has the same explicit full-module behavior as `pmodule`.

Repeated ingestion calls with `cache=SessionCache()` identify a computed module
by its actual constructed cells, grades, boundaries, coefficient field and
encoding request. Changing an input boundary or birth must not reuse the old
module even when the input object, grid and dimension vector are unchanged.
The cache snapshots sparse boundary storage without densifying it. This costs
a scan of the constructed chain data on cached calls; `cache=nothing` avoids
that scan. Lazy ingestion results also own snapshots of their sparse boundaries
and classifier axes, so later source edits cannot change a pending computation.
Treat returned modules and lazy results as mathematical values:
after editing source data, call `encode` again rather than mutating an already
computed result or expecting it to update itself.

Point-cloud geometry caches also distinguish coordinate contents: moving a
site invalidates the old Delaunay geometry or landmark radius graph. Landmark
graph keys inspect the selected landmarks; unchanged unselected sites do not
force a new graph. Equality checks retain coordinates, rather than relying on
their hashes alone. Input mutation is supported between calls, not during a
running computation. Use `PointCloud(coords; copy=true)` when the dataset
should own coordinates independently of the supplied matrix.

## Encoding metadata

`compile_encoding(P, pi)` attaches a classifier to its poset. It does not
generate axes or representative points. The summary's `has_axes` and
`has_representatives` flags report accessor capabilities or supplied metadata,
not whether generated arrays have been cached.

Request `encoding_axes` or `encoding_representatives` when those data are
needed. Grid representatives enumerate the Cartesian product of the axes and
can be large. Generated metadata is returned on demand, not implicitly
memoized; callers who want retained metadata can supply `axes=...` and
`reps=...` when compiling. Adding a workflow session cache preserves supplied
metadata and leaves absent metadata lazy.

## Algebra, geometry, and features

- A fringe display reports existing fiber dimensions. Use `fiber_dimension`
  or `dimensions` explicitly to compute ranks.
- A spectral summary reports `convergence_page=nothing` when stored dimension
  tables do not yet certify the first stable page. `convergence_bound` is the
  finite-filtration width bound. Request `convergence_page(ss)` explicitly to
  compute missing pages. An isolated cached later page does not certify the
  earliest stable page if earlier pages remain unknown.
- Ext comparison maps, homology representatives, algebra products, and
  common-refinement bases remain lazy under inspection. Their explicit
  mathematical accessors may compute and cache them.
- Geometry summaries leave exact region geometry and structured-poset relation
  caches untouched. Area, membership, and other geometry queries retain their
  own computation contracts.
- `nfeatures(spec)` counts the output slots without constructing feature names.
  `feature_names(spec)` deliberately constructs those labels. A rank-grid vector
  always has `nvertices^2` slots, even when its intermediate rank table omits
  zero entries.

Validators and explicit mathematical queries are separate from passive
inspection: they can perform the work required by their documented contracts.
