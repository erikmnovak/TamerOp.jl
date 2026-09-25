# Changelog

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
