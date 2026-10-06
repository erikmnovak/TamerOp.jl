# Changelog

## Unreleased

Changes prepared for the next release. Once a finite encoded module has been
constructed, its retained algebra should support further questions without
repeating work. These updates make dimensions, bases and coordinates cheaper
to obtain while preserving the existing basis conventions and the
[finite-poset interpretation of Ext and Tor](docs/math_categories.md).

### Distance and matching inspection

- Add reusable matching diagrams, aligned barcodes and pair costs with diagonal
  assignments, essential endpoints, duplicate-member IDs and linked selection.
- Compare common slices through two encoded modules, inspect a bounded sampled
  cost map, and explicitly request the exact finite-window optimum.
- Retain the exact value, line and numerical-owner assignment with
  `matching_distance_exact_2d(...; witness=true)` while keeping the scalar default.
- Add a focused matching recipe and extend the finite-window comparison guide.

### Resolution inspection

- Inspect finite-poset Betti/Bass multiplicities, selected differentials and
  principal supports, with explicit minimality, truncation and category scope.
- Add supplied grade-plane views and certified graded basis changes for
  resolution and presentation incidence; distinguish cokernels, kernels and
  fringe images. Rendering consumes prepared mathematical data.
- Add a resolution usage notebook and method reference in their planned homes.
  Projective lift views now use explicit `kind=:resolution_lift`; the default
  resolution view is the cheap multiplicity table.

### Benchmarks

- Added the final Ripser.py comparison: ordinary barcodes, cocycles and landmark workflows across 37 requests, with separate reserved evaluation, machine details and downloadable results. Independent checks now certify nontriviality of the sampled larger H2 cocycles.

### Generalized rank and GRIL

- Add exact generalized rank on selected connected convex finite subposets,
  with optional limit/colimit witnesses over rational and prime fields.
- Add signed summaries on declared finite interval families, with total-diagram
  compression and reconstruction restricted to that family.
- Compute continuous-worm GRIL at fixed probes on positive 2D grid encodings,
  using exact critical widths and explicit extension and resource contracts.
  Include independent mathematical oracles, executable examples, and an
  implementation account; plan separate mathematical and library-usage lessons.

### Ordinary persistence

- Analyze `PersistenceDiagram` directly by degree with bottleneck/Wasserstein
  distances, landscapes, images, silhouettes, entropy and scalar summaries.
  Make essential-bar handling, superlevel reflection and numerical grids explicit.
- Save and load owned diagram results with exact grades, field, provenance and
  retained cycle/cocycle data; add a checked novice analysis notebook.
- Repair shared Wasserstein auction refinement and exact-rational bottleneck
  comparisons involving essential intervals, with independent matching oracles.

- Fuse farthest-landmark selection with distance updates and accelerate native
  floating-point distance checks while preserving selection ties and errors.
- Reuse certified-radius neighbor lists for Rips coface searches while retaining
  the original catalog rows and exact filtration grades.
- Reduce F2 queue work with sorted batches, in-place sorting and buffer reuse
  within a query. Keep compact term heaps when a terminal radius removes neighbors,
  and signed coefficient heaps over odd prime fields.

- Find immediate equal-grade coface cancellations before filling Rips queues.
  Build complete graph rows directly, reuse sorted sparse rows, and avoid
  boxing the H0 clearing set. Preserve exact grades, signs and cocycle outputs.
- Check landmark finiteness in the existing distance-validation traversal,
  retaining complete input checks without a second full scan.

- Store one-term Rips reduction columns compactly, share edge ordering between
  connected components and H1, and reconstruct apparent pairs when needed.
  Preserve signed prime-field computation, signed-zero tie ordering and opt-in
  cocycle representatives.
- Stop eligible bounded Rips barcode requests at a certified cone radius, with
  the certificate recorded in provenance. Keep requested budgets, user cutoffs,
  retained cocycles and truncated-skeleton semantics intact.
- Pack triangle catalogs without narrowing Float64 grades or changing the
  simplex tie order, reducing catalog storage and sorting work.

- Reduce implicit Rips work with direct complete-graph distance lookup, direct
  triangle enumeration and preallocated dense catalogs. Compact F2 queue records
  omit redundant coefficients while preserving parity cancellation, signed
  odd-prime reduction and opt-in cocycle representatives.
- Validate dense distance matrices in cache-local tiles, improving landmark
  selection while retaining symmetry, diagonal, sign and conversion checks.

- Retain landmark selection order, insertion radii, nearest-point assignments
  and covering radius, with an optional full distance table and a conditional
  metric Rips approximation bound. Preserve distinct coincident input indices.
- Extend `EdgeWeightedFiltration` with vertex births, clique dimensions and an
  inclusive threshold. Accept graph inputs and dense/sparse matrices with
  diagonal vertex births, using the implicit barcode/cocycle route through H2
  and explicit construction for higher degrees or homology representatives.

- Add opt-in persistent cocycles over prime fields, with a scale-specific
  accessor, restriction conventions, exact stored grades and source-cell
  coefficients. Implicit Rips also retains oriented input vertex identities.
  Homology cycles and death fillings remain a separate option; barcode-only
  calls retain their existing reduction paths.

### Visualization

- Add anchored rank sections with a fixed source or target, exact parameter-order
  masks, selected-map readouts and small multiples sharing a scale. Linked
  inspectors reuse bounded rank rows and columns while preserving exact anchor
  geometry, reversed axes and the distinction between zero and undefined maps.

- Add algebra inspection for module morphisms, naturality squares, computed
  kernels/images/cokernels, and exact sequences. Display selected Hom basis
  maps, supplied chain maps and homotopies, and projective-resolution lifts
  with their actual equations and induced-map comparisons. Support overlays
  require a shared classifier; numerical identities show residuals and
  tolerances. All views share static Cairo/WGL rendering and export.

- Make default image and interval figures easier to read with compact layouts,
  simple axes, integer pixel labels for small images, and explanations of the
  endpoint symbols actually shown. Keep relevant display limitations visible.
- Keep image comparisons on a chosen colour range, label their displayed
  quantity, and show complete outer pixels. Small interval plots now label
  well-separated births and deaths inside the viewing window.

- Keep barcode infinity, censoring and continuation labels inside the plotting
  area and clear of selection strokes at larger text sizes, preserving their
  exact interval anchors.
- Rotate crowded persistence-diagram x-axis tick labels when needed, using
  measured text widths and restoring horizontal labels when space permits.
  Tick values and interval coordinates stay unchanged.
- Add maintained Playwright checks for live interval and encoding inspectors,
  including the earlier two-square manual cases, keyboard scrolling and actual
  browser zoom. The optional harness starts and stops its own Julia server.

### Documentation

- Publish validated documentation from `main` through GitHub Pages, with
  separate deployment permissions and browser checks under the project URL.

- Publish the expanded PHAT ordinary-barcode comparison across 48 medium/large
  inputs and sixteen structural variants, with timing tables, scaling curves,
  machine details and verified downloadable results. Replace the earlier public
  PHAT result set while retaining its raw evidence locally. Present practical
  timing differences and actual savings before win/loss counts; include PHAT
  and QPA reports and data in the documentation site.

- Give the reading map equally visible example-first and definitions-first
  starts, labeled optional detours, and installation alongside the ring.
  Keep reference material off the main diagram and both starts visible on
  narrow screens.
- Align lesson endings, the reading map and website footer around explicit
  reading routes. Remove development history and validation status from
  teaching pages, retaining mathematical requirements and clearly labeled
  placeholders for future figures.
- Explain why finite constructions supply tameness and preserve it through
  compatible algebraic operations, with a worked square morphism, a homology
  diagram, and precise links to the closure results of Ezra Miller and Lukas
  Waas. Extend the reading map and clarify each foundation chapter's role.
- Add a linked reading map with practical and mathematical routes, an accessible
  text outline, and grouped links to further guides. Both map views share one
  source, with published pages and repository treatments clearly labeled.
- Connect the ring lesson to finite encodings with a short two-parameter
  explanation, reusing the parameter-order diagram and preserving the existing
  mathematical chapter sequence as the full development.
- Teach plotting options as they become useful, keeping comparison scales
  explicit and moving styling/export to an optional section. Codify this in
  the writing guide; the website folds optional notebook sections while the
  executed download retains their cells and figures.
- Add a ring teaching notebook with saved filtration snapshots, interval
  figures and a prediction exercise. A separate documentation environment
  builds a Documenter lesson and executed download from the same source,
  with mathematical checks, figure/link validation and CI review artifacts.

### Algebra and performance

- Compute Rips barcodes through H₂ directly from point clouds and dense/sparse
  distance graphs, generating signed cofaces as needed instead of materializing
  boundary matrices. Support all prime fields, radius limits, landmarks and
  existing construction budgets; retain explicit construction for cycle
  representatives and higher degrees. Record the executed route and graph
  selection in provenance.

- Give sparse Rips distance matrices explicit missing-edge and zero-distance
  semantics, reusing budgeted clique construction. Check requested homology
  degrees against available filling dimensions, record radius restrictions,
  and apply radius cutoffs to nearest-neighbor graphs consistently.

- Extend direct ordinary persistence to prime fields, including F₃ and F₁₀₁,
  with signed modular boundaries, exact stored grades and optional coefficient-bearing
  cycles and death fillings. Preserve the F₂ fast paths and basis conventions;
  validate chain and filtration conditions in the selected field.

- Use component merging for ordinary F2 connected-component persistence inside
  higher-dimensional complexes, and a reversed dual graph for eligible top
  boundaries. Preserve periodic essential classes, exact grades, input checks,
  algebraic fallbacks and the existing representative-producing reduction.
- Extract finished binary columns in bulk and share storage across saved pivot
  columns within each ordinary F2 barcode computation. Preserve exact supports,
  filtration order, input validation and retained representatives.
- Add densely occupied binary pivot columns by whole words during ordinary F2
  barcode reduction, and reuse the previous pivot's word when locating its
  successor. Keep sparse columns sparse and preserve retained representatives.
- Use stable integer-grade ordering for ordinary persistence, preserving ties,
  unsigned and extreme grades, and the existing conventions for other real grades.
- Compute ordinary F2 barcodes with clearing and a reusable binary-column
  workspace indexed by filtration order. Graph-shaped degree-one complexes use
  component merging and cycle births. Keep the existing representative-producing
  reduction and report the executed route in result provenance.
- Speed up exact double-boundary validation with direct parity accumulation and
  packed binary products selected by column density and reuse. Preserve all
  storage, filtration and chain checks, including signed/even coefficients.
- Speed up graded-complex construction by keeping grade conversion specialized
  on its dimension and scalar type. Preserve exact grades, existing `BigFloat`
  precision, empty-input checks and independent grade storage.
- Reuse scratch buffers during ordinary F2 persistence reduction, including
  representative tracking. Preserve the selected cycles, bounding chains and
  boundary validation while reducing temporary column allocations.
- Select Nemo for sufficiently dense rational factors and matrix products,
  including coordinate setup. Preserve exact selected rows, bases, membership
  checks and explicit backend choices. Small, sparse and structured work retains
  its native paths; factor storage and existing sharing remain in place.
- Speed up complete membership checks for eligible dense rational batches with
  the same exact product backend. Explicit Julia-only solves keep Julia checks;
  scalar, sparse and structured inputs retain their existing paths.
- Factor eligible dense rational matrices directly from an invertible leading
  square block. Preserve the canonical selected rows and use general row
  selection when that block is singular, including full-rank matrices with
  nonconsecutive selected rows.
- Select rational pivot columns by forward elimination, preserving the same
  ordered basis without computing a full reduced matrix just to discard it.
- Remove copied row slices and redundant input copies from exact particular
  solves. Detect inconsistent right-hand sides from augmented pivot columns;
  preserve the free-variable convention and numerical-field behavior.
- Share packed F2 factor application between dense and sparse solves, reusing
  one scratch vector within each call while retaining full RHS checks.
- Compute rational Hom constraints with sparse integer echelon rows and final
  back substitution. Clearing denominators avoids repeated fraction reduction
  while preserving the ordered basis of actual module maps.
- Reuse result-owned factors for native prime-field quotient coordinates above
  characteristic three. Every checked query still verifies its full input;
  new results start with their own factors. F2/F3 and Nemo routing are retained.
- Construct Ext and Tor representatives with the shared exact rational product
  path, skipping structural zeros without changing their coordinates.
- Add matrix-coordinate Yoneda products for whole tables, reusing each lift
  within the request. `ExtAlgebra` uses this preparation when building its
  multiplication tables. Scalar products, signs and comparison maps between
  target resolutions retain their contracts.
- Use the same rational products for module-map composition, induced homology
  and cohomology maps, and exact relation checks. Keep sparse output where both
  factors are sparse, preserve native diagonal/banded/triangular algorithms,
  and retain the numerical-field validation tolerances.
- Defer boundary coordinates and quotient representatives for exact cohomology
  dimension queries, while still checking at construction that boundaries are
  cycles. Later queries reuse the same result.
- Build homology, cohomology and subquotient representatives by selecting the
  required cycle-basis columns directly, avoiding unnecessary multiplication.
- Retrieve canonical Ext bases from retained representatives. When another
  resolution model is requested, transport the whole basis together. Preserve
  its coordinate order and return independent representative vectors.
- Reduce unnecessary rational arithmetic in matrix products, solution checks
  and row reduction. Avoid a second elimination when factoring an invertible
  square matrix; retain singular-input checks and the selected exact backend.
- Reuse a checked coordinate calculation for rational homology and cohomology.
  Every public query still verifies cycle membership in all ambient coordinates,
  including when the quotient is zero. Coordinate requests leave representative
  matrices deferred until explicitly requested.
- Reuse factors already computed during construction, including sharing across
  equal matrices while their weak-cache entries remain alive. Build quotient
  projections without retaining an unnecessary completed-basis inverse.
- Expand precompilation to explicit Hom maps and positive-degree projective Ext
  on a branching poset, covering work missed by the earlier two-vertex example.

See [lazy inspection and reuse](docs/lazy_inspection.md) for the resulting query
behavior. No new public cache option or query-count threshold is required.

### Correctness and validation

- Reject nonfinite rational coefficients before skipping zero products, so a
  malformed differential cannot pass a relation check just because the adjacent
  map is zero. Check before modifying a supplied product destination.
- Check scalar and batched Yoneda products over QQ, F2, F3, F101 and Real64,
  including units, ordering, empty batches, invalid inputs and transport into
  another target resolution. Preserve strict associativity checks.
- Fix projective/injective Ext comparison checks over `RealField` to use the
  configured numerical tolerances in both inverse directions. Exact fields
  continue to require exact equality.
- Reject nonfinite entries in Julia's rational row-reduction path before
  applying arithmetic shortcuts.
- Add independent exact elimination and quotient-coordinate examples, including
  nonconsecutive pivots, boundary shifts, invalid first queries, sparse inputs,
  array views, lazy basis construction, shared factors and concurrent queries.
- Extend tests of Ext dimensions, representatives and induced maps to the
  Boolean three-cube and a 4-by-4 grid, with exact and numerical coefficients.
  Retain checks of products, associativity and comparisons between resolutions.

An earlier targeted algebra integration run passed 12,466 assertions with
Julia 1.12.1, four threads, and QQ, F2, F3, F5 and Real64 selected. Independent exact checks
also verified complete Hom kernels and unchanged ordered Ext representatives
on 16 before/after fixture exports. This was a focused run, not a rerun of the
full repository suite; see [testing](docs/testing.md) for validation scope.

For the shared-product and Yoneda-table implementation, focused product and
integration runs passed 24,525 assertions. After the finite-coefficient and
structured-storage review, the final kernel and complex-validation run passed
another 1,380 assertions (some repeat earlier checks). These maintained-runner
checks include randomized module-map oracles, exact/numerical fields, threaded
parity and backendized storage; they do not replace the full suite.

The rational Hom and native prime-coordinate follow-up passed 7,863 focused
assertions, including independent rational kernels, checked finite-field solves,
threaded quotient coordinates, backend routing and strict Yoneda identities.
It uses the maintained test runner; the full repository suite was not rerun.

The dense certificate and leading-block follow-up passes 11,133 focused
assertions, including hand-derived exact inverses, singular leading blocks,
invalid first and last entries, explicit backend choices and threaded checks.
Existing homology, cohomology, Ext/Tor and strict Yoneda checks are included;
this is not a full-suite rerun.

### Measured performance scope

Selective dense rational routing passes 10,565 focused assertions and independent
checks of 17 development workflows and 16 dense controls. Two process pairs yield
492 accepted samples with verified resets and zero measured compilation.
Complete size-16 coordinate requests improve by 2.07–4.06×; retained checked
queries improve by 1.45–2.06×. These controls start from supplied cycle and
boundary bases, with result construction and the requested answer timed together.

The largest observed size-4 addition is 0.026 ms. Other module workflows are
mixed, with a largest observed addition of 0.517 ms in one pair. Reachable
size-16 result/cache storage is smaller, but Julia allocation counters omit
FLINT allocations; no process-memory reduction is claimed. These are TamerOp
before/after diagnostics, not a new QPA comparison or a full-suite test run.

The rational Hom and native prime-coordinate follow-up verified 520 fresh-result
samples and ten identical, independently checked before/after answers. The
large-coefficient rational Hom request improves by 1.39–2.08× with 64% fewer
allocated bytes; F101 Yoneda tables improve by 1.72–2.90× with 40% fewer allocated
bytes. The full F101 tensor request ranges from near parity to 1.62× faster.
These are two shared-host TamerOp process pairs, not a new QPA comparison.

Keep the tradeoffs visible: a size-16 rational cohomology scalar control adds
2.23–2.27 ms in both pairs, and several other controls are mixed. A separate
240-sample prime-coordinate diagnostic finds 4.60–6.36× faster retained scalar
queries at size 16, but first requests add 23–126 μs and retain 4,592 extra bytes.
Retained 32-column queries are 4–19% slower. These changes target repeated
coordinate work within complete computations; they do not make every query
faster or close the broader optimization study.

The shared-product and Yoneda-table changes improved a rational cube product
request by 23.8–27.3× and a rational grid request by 2.83–4.20× in two shared-host
passes with compiled code and verified mathematical resets. The cube request's
allocation traffic fell from about 164 MiB to 4.1 MiB. All nineteen selected
before/after answers were identical and independently checked. These are TamerOp
before/after results, not new QPA comparison results.

The initial study recorded dense and small-request losses; its final baseline
reached the time cap before completing four dense controls' second passes.
A completed follow-up with symmetric timing-only workers verified 380 samples.
The cube gain remains 22.9–24.4× and complete grid Hom/Ext improves by 1.11–1.16×.
The small F3 product adds 0.18–0.65 ms. Two larger dense cells remain slower in
standalone timings, while interleaved rollback and CPU-time checks do not
consistently reproduce a major extra multiplication cost. Retained dense storage
is unchanged. Keep these shared-host losses and uncertainties visible; the
follow-up introduces no speculative gate or further production change. The
broader optimization study remains open.

For the final factor-reuse and projection changes, one dense rational
cohomology fixture improved from 8.4–9.2 ms to 3.2–4.5 ms for its first coordinate
query. Allocations fell from 9.66 MB to 2.99 MB, and retained result-plus-factor
storage fell from about 161 KB to 103 KB. These are two warm benchmark passes
on a shared machine; first query means a fresh result, not Julia startup, and
retained object size is not peak process memory.

Performance depends on the workload. Some repeated queries and complete
computations on the grid fixtures took longer. In the measured dense fixture,
a conventional checked solve was cheaper for one isolated scalar query, while
the retained plan reduced repeated-query times relative to those solves. These
measurements do not establish a speedup for every operation or input.

### Visualization

- Share exact interval records across barcode and persistence-diagram recipes
  for ordinary, packed, sliced and projected results. Keep multiplicity groups,
  original member references, display counts and finite clipping separate from
  essential infinity and unknown window endpoints.
- Certify whole-line endpoints for supported planar encodings with
  `slice_scope=:global`. The drawing window no longer limits the mathematical
  restriction; incomplete domain coverage is rejected explicitly.
- Add linked interval-only inspection with selection from either chart,
  display budgets, exact readouts and explicit choice among repeated intervals.
  Ordinary F2 persistence can retain noncanonical cycles and bounding chains
  with `representatives=true`; `Advanced.persistence_representative` retrieves
  an original interval's representative. Retention is off by default.
- Link a movable planar slice to decorated barcode and persistence-diagram
  views in the inspector. Preserve open/closed endpoints, singleton intervals
  and multiplicities, and mark finite-window cuts without inferring essential
  classes. Exact line controls, bounded reuse and shared interval selection
  also work with static snapshots.
- Add `VisualStyle` for shared typography, spacing, colors and marker emphasis
  across static figures, the live inspector and saved exports. Styles apply to
  each call without changing the mathematical specification or global themes.
- Keep source and target roles recognizable through labels and marker shapes
  in accessible and grayscale palettes. Preserve exact matrix coefficients,
  empty spaces, boundary conventions and data-space marker extents.
- Apply the shared style to existing module/presentation inspection panels,
  barcode and persistence-diagram views, and numerical heatmaps. Batch exports
  accept a default style and a separate override for each requested figure.

See [visualization](docs/visualization.md#rendering-and-saving) for examples.

### Documentation

- Add a [contributor guide](CONTRIBUTING.md) covering issue reports, development
  setup, focused checks, documentation, and pull requests.
- Use the expanded name, Toolkit for Algebraic Module Encodings over R^n
  and Other Posets, in the software citation. Explain the connection to
  tameness and the project's origin as an implementation of Ezra Miller's theory.
- Center the README and Julia help on the finite encoded object and the
  computations it supports.
- Add a [finite-encoding introduction](docs/finite_encodings.md) and a
  [writing guide](docs/writing.md) for connected, approachable exposition.
- Give existing guides question-driven introductions while retaining their
  mathematical hypotheses and detailed operation requirements.

## 0.1.0 — unreleased candidate

This section describes the first planned package release. It does not announce
a published tag or General registration. The release procedure and validation
requirements are in [the maintainer guide](docs/releasing.md).

### Mathematical capabilities

- Finite-poset encodings of persistence modules from point clouds, images,
  graphs and hand-built presentations, with field-aware algebra and explicit
  construction budgets.
- Homology, invariants, resolutions, Ext/Tor and their maps, with category and
  encoding hypotheses documented in [the mathematical-category guide](docs/math_categories.md).
- Exact algebraic grades for supported geometric constructions, with
  [multicover backend and degeneracy contracts](docs/multicover.md) and
  [exact-grade semantics](docs/exact_grades.md).
- One-parameter persistent homology over F₂, including cubical sublevel and
  superlevel filtrations, exact finite endpoints and periodic axes. Other
  coefficient fields are rejected by this reducer; see
  [ordinary persistence](docs/ordinary_persistence.md).
- A finite-window exact two-dimensional matching-distance implementation and
  separately identified sampled computations. See the
  [exact-matching scope](docs/exact_matching.md).

### Installation and use

- An installable Julia package with a beginner walkthrough, isolated analysis
  environments, `import TamerOp as OP`, selective imports and advanced access.
- Optional plotting, table/file and ecosystem integrations loaded through
  Julia's native package extensions. Core installation does not install a
  renderer; the [integration guide](docs/optional_integrations.md) lists the
  additional packages and imports.
- Persistence diagrams and barcodes, with CairoMakie PNG/SVG output and WGLMakie
  HTML export. Browser interaction after Julia exits is a separate validation
  concern and is not certified by file-generation tests.
- Result summaries, mathematical provenance, lazy inspection and explicit
  numerical contracts. Installed package loading does not benchmark or write
  a tuning profile into package storage.
- Maintained package-mode tests and installed-package verification, release
  instructions, and [software citation metadata](CITATION.cff).

### Correctness checks before registration

- Iteration cardinality and checked indexing for finite upsets/downsets, packed
  barcodes, backend matrices, polyline views and injective-generator views.
- Field-aware dictionary and set keys. Finite-field scalars are `Number`s;
  modular integer arithmetic remains available, while key identity retains the
  characteristic. Coerce dictionary keys explicitly into the coefficient field.
- Encoding/result validators check stored dimensions, coefficient fields and
  base-poset agreement without forcing lazy computations. Exceptions raised
  inside custom invariant callbacks retain their original meaning.
- Invariant entrypoints accept their documented keyword options and preserve
  positional owner calls, query strictness and automatic thread defaults. Rank
  invariants accept both finite-poset modules and fringe presentations.
- Batched polyhedral lookup skips bucket indexing for one or two small
  inequality systems while retaining indexing for larger or facet-heavy cells.
- Polyhedral encodings retain coefficients, closed boundary classes and every
  piece of a nonconvex support. General PL strict feasibility is rational by
  default, with explicit fixed-margin approximation available as an option.
- Installed-source verification compares authenticated Git file contents on
  Linux, macOS and Windows; Windows filesystem modes are treated separately.

### Release scope

Julia compatibility starts at 1.12. A completed test run certifies only its
recorded Julia version, platform, dependencies, fields and execution settings;
see [testing](docs/testing.md). This candidate's validation must pass before it
is registered or tagged. No comparative speedup or universal correctness claim
is attached to this release summary. Version 0.1.0 is pre-1.0 software and its
public API can change in later minor versions.
