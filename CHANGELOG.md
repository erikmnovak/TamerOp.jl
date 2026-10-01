# Changelog

## Unreleased

Changes prepared for the next release. Once a finite encoded module has been
constructed, its retained algebra should support further questions without
repeating work. These updates make dimensions, bases and coordinates cheaper
to obtain while preserving the existing basis conventions and the
[finite-poset interpretation of Ext and Tor](docs/math_categories.md).

### Algebra and performance

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

### Measured performance scope

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
