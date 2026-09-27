# Changelog

## Unreleased

Changes prepared for the next release. Once a finite encoded module has been
constructed, its retained algebra should support further questions without
repeating work. These updates make dimensions, bases and coordinates cheaper
to obtain while preserving the existing basis conventions and the
[finite-poset interpretation of Ext and Tor](docs/math_categories.md).

### Algebra and performance

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

A targeted algebra integration run passed 12,466 assertions with Julia 1.12.1,
four threads, and QQ, F2, F3, F5 and Real64 selected. Independent exact checks
also verified complete Hom kernels and unchanged ordered Ext representatives
on 16 before/after fixture exports. This was a focused run, not a rerun of the
full repository suite; see [testing](docs/testing.md) for validation scope.

### Measured performance scope

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
