# Visualization support and acceptance

Use this matrix to choose a display and to review a visualization change. Support is a declared behavior; verification is evidence for a specified path. A native figure or serialized HTML is not evidence that a browser or notebook executed it.

The maintained source is [visualization_support.toml](visualization_support.toml). Regenerate this page with `python test/visualization_contracts.py --write`; CI rejects unassigned source files, unassigned native/browser tests and stale generated tables. Reviewers must also identify new recipes within existing source files.

## Reading the matrix

- **N**: family-specific native mathematical/renderer checks; no browser execution implied.
- **B**: maintained Chromium scenario exercises this family through real browser events.
- **S**: supported through shared rendering/export code, covered by shared checks; not a separate browser acceptance claim for every recipe.
- **U**: integration exists but that named frontend has not been qualified.
- **—**: no such callback/widget is offered. Static selection remains a Julia argument.

Cairo means static PNG/SVG/PDF figures. WGL native means native scene construction; HTML means exported scenes or fixed SVG compositions. Live Chromium means a standalone Bonito application with Julia running. Saved-view HTML instead selects among explicitly precomputed states after Julia has exited.

The notebook column applies to each of IJulia/Jupyter, Pluto and VS Code: none inherits standalone-browser acceptance. Static notebook publication and executed Julia cells also do not certify a live notebook frontend. Firefox and WebKit have no declared acceptance baseline yet.

The maintained standalone-browser environment requires Bonito 5.2 within major version 5. Browser evidence applies to the resolved harness stack; it does not qualify older nested-application integrations.

## Families and display paths

| Family | Cairo | WGL native | Exported HTML | Standalone Chromium | Live notebooks |
| --- | --- | --- | --- | --- | --- |
| [Encoding regions and exact query overlays](#encoding-regions-and-exact-query-overlays) | N | N | S | B | U |
| [Point clouds, graphs, images and construction counts](#point-clouds-graphs-images-and-construction-counts) | N | N | B | — | U |
| [Image volume slice widget](#image-volume-slice-widget) | N | N | — | B | U |
| [Hilbert/rank, signed density, MPPI and sampled landscapes](#hilbertrank-signed-density-mppi-and-sampled-landscapes) | N | N | S | — | U |
| [Finite spaces, maps and presentations](#finite-spaces-maps-and-presentations) | N | N | S | B | U |
| [Anchored rank sections](#anchored-rank-sections) | N | N | S | B | U |
| [Restrictions to lines and slice barcodes](#restrictions-to-lines-and-slice-barcodes) | N | N | S | B | U |
| [Barcodes and persistence diagrams](#barcodes-and-persistence-diagrams) | N | N | S | B | U |
| [Matching witnesses and distance comparisons](#matching-witnesses-and-distance-comparisons) | N | N | S | B | U |
| [Signed decompositions and cancellation](#signed-decompositions-and-cancellation) | N | N | S | B | U |
| [Morphisms, Hom and exact sequences](#morphisms-hom-and-exact-sequences) | N | N | S | — | U |
| [Betti/Bass and resolution incidence](#bettibass-and-resolution-incidence) | N | N | S | — | U |
| [Spectral pages and differentials](#spectral-pages-and-differentials) | N | N | S | B | U |
| [Selected vectors and generated submodules](#selected-vectors-and-generated-submodules) | N | S | S | — | U |
| [Refinement and change of poset](#refinement-and-change-of-poset) | N | N | S | B | U |
| [Source representatives and fillings](#source-representatives-and-fillings) | N | S | S | B | U |
| [Filtered source complexes](#filtered-source-complexes) | N | N | S | B | U |
| [Chains, cohomology classes and connecting maps](#chains-cohomology-classes-and-connecting-maps) | N | N | S | B | U |
| [Ext, Tor and products](#ext-tor-and-products) | N | N | S | — | U |
| [H0 components and merge histories](#h0-components-and-merge-histories) | N | N | S | B | U |
| [Multicover and geometric constructions](#multicover-and-geometric-constructions) | N | N | S | B | U |
| [Field, degree and construction comparisons](#field-degree-and-construction-comparisons) | N | N | S | B | U |
| [Supplied feature, distance and kernel collections](#supplied-feature-distance-and-kernel-collections) | N | N | S | B | U |
| [Semantic figure composition](#semantic-figure-composition) | N | N | B | B | U |
| [Saved mathematical views](#saved-mathematical-views) | N | S | B | B (precomputed) | U |
| [Validated finite-module and fringe editor](#validated-finite-module-and-fringe-editor) | N | N | S | B | U |
| [Generalized rank regions, boundary zigzags and GRIL](#generalized-rank-regions-boundary-zigzags-and-gril) | N | N | S | B | U |
| [Coordinate sections and ambient landscapes](#coordinate-sections-and-ambient-landscapes) | N | N | S | B | U |
| [Repeated invariants and mean uncertainty](#repeated-invariants-and-mean-uncertainty) | N | N | S | B | U |

These entries identify maintained checks, not a claim that a test file proves its latest execution passed. CI supplies a run result and retained artifacts; local acceptance records preserve their environment, failures, reruns and scope. A green shared check does not qualify every possible input, recipe or frontend.

## Recorded verification

Acceptance recorded on 2026-10-10. Julia 1.12.1, Makie 0.24.15, CairoMakie 0.15.15, WGLMakie 0.13.15, Bonito 5.2.0, Observables 0.5.5; Node 22.23.3, Playwright 1.63.0 and Chromium 153.0.8010.12 (revision 1243). One Julia/BLAS thread and one browser worker.

- Native: The earlier 29-owner qualification exercised: 23,860 assertions passed and 12 nested-app stylesheet assertions failed. After repair, a fresh focused run passed 251/251 embedding, style, coordinate, volume and WGL-export assertions. The later saved-HTML readability repair passed 103/103 export assertions in another fresh process. A99 subsequently passed 3,200 focused owner/API assertions with graphics loaded; its final field-label/export-control rendering run passed 14/14. A54 subsequently passed 545 focused checks, including 19 actual Cairo/WGL renderer/export assertions, and an extended 462-assertion boundary-oracle run. A44 subsequently passed 876 focused checks, including 43 native Cairo/WGL assertions. Its separate fresh mathematical regression run passed 1,158 and API guards passed 2,936. These overlapping counts are separate runs, not an aggregate total.
- Live browser: The earlier 27 standalone Chromium scenarios passed. Both additional A99 editor scenarios passed, including numeric edits, invalid relation recovery, downloads, linked-client cleanup and narrow keyboard scrolling. The strengthened composition-host keyboard scenario also passed against a fresh Julia server, including resize, pointer pan and unchanged correspondence geometry. Both A54 scenarios passed on the final browser run: finite selection, boundary occurrences/bars, matrix toggles, exact rational worm entry, verified pointer centers, layer reuse, linked-client cleanup and narrow keyboard scrolling. The A44 standalone Chromium scenario passed real sliders, native chart selection, plane/layer/baseline changes, invalid-input rollback, two-viewer synchronization with visible slider positions after reset, stale-error clearing, disconnect cleanup, narrow keyboard scrolling and close/release.
- Offline exports: All 3 scenarios passed with Julia stopped and networking disabled: fixed SVG compositions, saved finite states, and an actually rendered WGL scene with reload. Narrow SVG views preserve their intrinsic size and support keyboard and pointer scrolling.

This dated record describes the tested revision and environment; current CI results remain the evidence for subsequent changes. The matrix validator checks the record's format, not the truth of an execution claim.

## Standing A36 and A37 contracts

**A36 — mathematical coordinates.** Use asymmetric and nonuniform fixtures wherever a transpose or midpoint convention could look plausible. Assert actual plotted cell boundaries and values in both backends, selected axis order, fixed coordinates, masks, orientation and exact-query behavior. Exact classification precedes floating drawing conversion. Native slider callbacks and real browser slider transport are separate checks.

**A37 — advertised interaction.** Controls need accessible names and keyboard operation without requiring a canvas click. Tab and Shift+Tab must leave plots and reach later controls, including after a figure rebuild. Test pointer events when picking is advertised, read-only hover, invalid-input rollback, shared selection across clients, disconnection and close/disposal. Verify visible updates and mathematical readouts. Declare live Julia, static scene navigation and precomputed offline selection distinctly. At narrow widths, preserve readable scientific labels and keyboard/pointer access to wide charts; fitting the surrounding page alone does not establish readability.

## Required checks for a change

1. Update the family entry when adding a recipe, a backend or a new interaction. Provide an independent mathematical oracle and real backend assertions appropriate to it.
2. For an advertised live control, add a browser scenario with real user input, readout/drawing assertions and lifecycle coverage. Preserve narrow-layout and keyboard access.
3. Run the matrix validator and affected native/browser owners locally. The [Visualization CI workflow](../.github/workflows/Visualization.yml) requires both rendering extensions and runs all registered visualization owners, live Chromium, then no-server exported-HTML scenarios. Missing offline fixtures fail rather than skip.
4. Review representative figures. Tests of coordinates, transport or DOM layout do not establish readability or a clear mathematical explanation.

Commands and environment setup are in [testing](testing.md) and the [browser harness](../test/browser/README.md). These are contributor acceptance requirements; development status and test counts do not belong in mathematical lessons.

## Evidence by family

### Encoding regions and exact query overlays

Linked parameter selection uses the finite-module inspector; planar geometry only.

Native owners: [test_visualization.jl](../test/test_visualization.jl).

Browser scenarios: [manual-inspectors.spec.mjs](../test/browser/manual-inspectors.spec.mjs).

### Point clouds, graphs, images and construction counts

Static source previews with original IDs, explicit display/preparation budgets and bounded content-checked SessionCache reuse. Metric graphs are labelled reconstructions; supplied graphs retain actual edges. Exported source-preview HTML has its own offline Chromium check. ImageNd live volume controls have their own row.

Native owners: [test_visualization.jl](../test/test_visualization.jl), [test_visualization_a39.jl](../test/test_visualization_a39.jl).

Browser scenarios: [source-preview-offline.spec.mjs](../test/browser/source-preview-offline.spec.mjs).

### Image volume slice widget

Displayed axes and fixed indices are explicit. Cairo renders the selected plane; live sliders require Julia.

Native owners: [test_visualization.jl](../test/test_visualization.jl).

Browser scenarios: [volume.spec.mjs](../test/browser/volume.spec.mjs).

### Hilbert/rank, signed density, MPPI and sampled landscapes

Static recipes; anchored ranks, signed queries and interval inspectors have separate rows. Sampling coordinates are not ambient landscape certificates.

Native owners: [test_visualization.jl](../test/test_visualization.jl), [test_visualization_a41_intervals.jl](../test/test_visualization_a41_intervals.jl).

### Finite spaces, maps and presentations

Actual finite labels, exact entry, fields, selected image bases and parameter-order masks.

Native owners: [test_visualization.jl](../test/test_visualization.jl).

Browser scenarios: [manual-inspectors.spec.mjs](../test/browser/manual-inspectors.spec.mjs), [style-zoom.spec.mjs](../test/browser/style-zoom.spec.mjs).

### Anchored rank sections

Uses the module session; one selected anchor row and explicit comparable-pair domain.

Native owners: [test_visualization_a84.jl](../test/test_visualization_a84.jl).

Browser scenarios: [rank-sections.spec.mjs](../test/browser/rank-sections.spec.mjs).

### Restrictions to lines and slice barcodes

Whole-line certification and window censoring remain different contracts; no between-line class continuation.

Native owners: [test_visualization.jl](../test/test_visualization.jl), [test_visualization_a41_slices.jl](../test/test_visualization_a41_slices.jl), [test_visualization_a41_sessions.jl](../test/test_visualization_a41_sessions.jl), [test_visualization_a41_live.jl](../test/test_visualization_a41_live.jl).

Browser scenarios: [inspectors.spec.mjs](../test/browser/inspectors.spec.mjs), [manual-inspectors.spec.mjs](../test/browser/manual-inspectors.spec.mjs).

### Barcodes and persistence diagrams

Multiplicity, source-member identity, endpoints, infinity and clipping; retained representatives are opt-in.

Native owners: [test_visualization_a41_intervals.jl](../test/test_visualization_a41_intervals.jl), [test_visualization_a41_sessions.jl](../test/test_visualization_a41_sessions.jl), [test_visualization_a41_live.jl](../test/test_visualization_a41_live.jl).

Browser scenarios: [inspectors.spec.mjs](../test/browser/inspectors.spec.mjs), [style-zoom.spec.mjs](../test/browser/style-zoom.spec.mjs).

### Matching witnesses and distance comparisons

Selected supplied/computed witnesses and finite-window optimizer scope; not a general interleaving certificate.

Native owners: [test_visualization_a85.jl](../test/test_visualization_a85.jl).

Browser scenarios: [matching.spec.mjs](../test/browser/matching.spec.mjs).

### Signed decompositions and cancellation

Named observables and declared reconstruction domains, exact terms, endpoint glyphs and smoothing.

Native owners: [test_visualization_a86.jl](../test/test_visualization_a86.jl).

Browser scenarios: [signed.spec.mjs](../test/browser/signed.spec.mjs).

### Morphisms, Hom and exact sequences

Selections are Julia arguments. These static recipes do not advertise a bespoke live inspector.

Native owners: [test_visualization_a42b.jl](../test/test_visualization_a42b.jl), [test_visualization_a42b_comparisons.jl](../test/test_visualization_a42b_comparisons.jl).

### Betti/Bass and resolution incidence

Static selected summands/maps in the declared category; no ambient free-resolution inference.

Native owners: [test_visualization_a42c.jl](../test/test_visualization_a42c.jl).

### Spectral pages and differentials

Selected page/degree, actual differentials and quotient coordinates; associated-graded scope.

Native owners: [test_visualization_a42d.jl](../test/test_visualization_a42d.jl).

Browser scenarios: [spectral.spec.mjs](../test/browser/spectral.spec.mjs).

### Selected vectors and generated submodules

Static arguments; checked module-map propagation is not a splitting or class-tracking claim.

Native owners: [test_propagation.jl](../test/test_propagation.jl).

### Refinement and change of poset

Verified order maps and actual correspondences, not equal-index identification.

Native owners: [test_visualization_a88.jl](../test/test_visualization_a88.jl).

Browser scenarios: [transport.spec.mjs](../test/browser/transport.spec.mjs).

### Source representatives and fillings

Uses retained interval/source correspondence; finite-module bases alone do not identify source cycles.

Native owners: [test_source_representatives.jl](../test/test_source_representatives.jl), [test_visualization_a41_live.jl](../test/test_visualization_a41_live.jl), [test_visual_session_exports.jl](../test/test_visual_session_exports.jl).

Browser scenarios: [source.spec.mjs](../test/browser/source.spec.mjs).

### Filtered source complexes

Actual cells, grades and source IDs, periodic/3D geometry and explicit skeleton/display budgets.

Native owners: [test_visualization_a92.jl](../test/test_visualization_a92.jl).

Browser scenarios: [complex.spec.mjs](../test/browser/complex.spec.mjs).

### Chains, cohomology classes and connecting maps

Chosen representatives, original-basis witnesses and field-aware quotient coordinates.

Native owners: [test_visualization_a89.jl](../test/test_visualization_a89.jl).

Browser scenarios: [chains.spec.mjs](../test/browser/chains.spec.mjs).

### Ext, Tor and products

Static selected representatives and validated products; no bespoke live inspector.

Native owners: [test_visualization_a90.jl](../test/test_visualization_a90.jl).

### H0 components and merge histories

Original memberships, ties and merge edges; component identity is not an arbitrary barcode basis.

Native owners: [test_visualization_a91.jl](../test/test_visualization_a91.jl).

Browser scenarios: [components.spec.mjs](../test/browser/components.spec.mjs).

### Multicover and geometric constructions

Uses the complex session; retained construction cells, radius/depth and explicit lifted versus schematic geometry.

Native owners: [test_visualization_a93.jl](../test/test_visualization_a93.jl).

Browser scenarios: [geometric.spec.mjs](../test/browser/geometric.spec.mjs).

### Field, degree and construction comparisons

Explicit alignment, scale and provenance; no inferred basis or class identity.

Native owners: [test_visualization_a94.jl](../test/test_visualization_a94.jl).

Browser scenarios: [comparison.spec.mjs](../test/browser/comparison.spec.mjs).

### Supplied feature, distance and kernel collections

Stable sample identities, declared features and supplied embeddings; no hidden all-pairs computation.

Native owners: [test_visualization_a98.jl](../test/test_visualization_a98.jl).

Browser scenarios: [collection.spec.mjs](../test/browser/collection.spec.mjs).

### Semantic figure composition

Authored versus checked anchors; live axis transforms and fixed self-contained SVG HTML are different products. Browser acceptance covers the maintained Bonito host; embedding a raw Figure in another host requires that host's keyboard integration.

Native owners: [test_visualization_a102.jl](../test/test_visualization_a102.jl), [test_visual_session_composition.jl](../test/test_visual_session_composition.jl).

Browser scenarios: [composition.spec.mjs](../test/browser/composition.spec.mjs), [composition-offline.spec.mjs](../test/browser/composition-offline.spec.mjs).

### Saved mathematical views

Offline selection is among explicitly precomputed states. Arbitrary Julia computations and source replay remain live operations. Narrow views scroll complete figures at their intrinsic size.

Native owners: [test_visualization_a100.jl](../test/test_visualization_a100.jl), [test_visual_session_codec.jl](../test/test_visual_session_codec.jl), [test_visual_session_boundaries.jl](../test/test_visual_session_boundaries.jl), [test_visual_session_exports.jl](../test/test_visual_session_exports.jl).

Browser scenarios: [portable-session.spec.mjs](../test/browser/portable-session.spec.mjs).

### Validated finite-module and fringe editor

Cairo/WGL render retained snapshots; editing needs the live WGL table controls. Exact failed-path witnesses, invalid drafts, finite fringe support closure, bounded history and checked object/code export. Saved HTML is a prepared view, not an offline editor. Live notebook hosts require separate qualification.

Native owners: [test_visualization_a99.jl](../test/test_visualization_a99.jl).

Browser scenarios: [editor.spec.mjs](../test/browser/editor.spec.mjs).

### Generalized rank regions, boundary zigzags and GRIL

Exact finite-region and continuous-worm queries; optional matrices and grid-index boundary barcodes retain repeated occurrences. GRIL layers use exact suprema at declared probes. Live query selection requires Julia; snapshots are static. No learned probes, derivatives or ambient decomposition claim.

Native owners: [test_visualization_a54.jl](../test/test_visualization_a54.jl).

Browser scenarios: [generalized-rank.spec.mjs](../test/browser/generalized-rank.spec.mjs).

### Coordinate sections and ambient landscapes

Declared Cartesian scalar samples and compatible comparisons, ordered planes, fixed values, layer IDs, global signed scales and affine contours. Ambient landscape queries preserve actual line parameters and source windows, with unavailable cells and conditional errors. Live selection requires Julia; static snapshots do not add saved-source reconstruction. No exact isosurface or interpolation between stored landscape lines is inferred.

Native owners: [test_visualization_a44.jl](../test/test_visualization_a44.jl).

Browser scenarios: [fields.spec.mjs](../test/browser/fields.spec.mjs).

### Repeated invariants and mean uncertainty

Aligned supplied grids and ordinary diagrams; empirical spread/frequency, conditional bounded-mean and approximate row-bootstrap bands at fixed probes. Linked replicate/source and probe selection requires Julia; no continuous-domain or whole-module confidence is implied.

Native owners: [test_visualization_a95.jl](../test/test_visualization_a95.jl).

Browser scenarios: [ensemble.spec.mjs](../test/browser/ensemble.spec.mjs).
