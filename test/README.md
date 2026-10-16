# Testing TamerOp

Start with the files that exercise the computation you changed. This guide
explains the test runner, optional integrations, mathematical checks and full
release verification. The later sections retain the specialist fixtures and
acceptance requirements for contributors working on those features.

- [Choose files, testsets and fields](#choose-files-testsets-and-fields)
- [Visualization contracts](#visualization-acceptance-contracts) and [browser setup](browser/README.md)
- [Optional dependencies](#optional-dependencies)
- [Public onboarding checks](#public-onboarding-checks-and-local-tutorials)
- [Minimal environments](#core-only-and-ordinary-persistence-checks)
- [Release candidate verification](#release-candidate-verification)

The website has its own [build and notebook checks](https://github.com/erikmnovak/TamerOp-site/blob/main/docs/README.md).
For timings, follow the [benchmark protocol](https://github.com/erikmnovak/TamerOp-benchmarks/blob/main/PROTOCOL.md).
A performance comparison needs equivalent mathematical outputs and explicit
measurement of compilation, preparation and reused results.

## Choose files, testsets and fields

Run the terminal commands below from the repository root with Julia 1.12 and an
instantiated project. An *owner file* groups tests for a particular subsystem. The maintained entrypoint is `test/runtests.jl`; it loads
this checkout using `using TamerOp`. A package or extension loading failure
fails the run.

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. test/runtests.jl --list
julia --project=. test/runtests.jl --file=test_data_pipeline.jl --prefix=A14
julia --project=. --threads=4 test/runtests.jl --file=test_derived_functors.jl --prefix=A14 --fields=QQ,F3,Real64
```

The generalized-rank and GRIL checks use independent canonical-map elimination,
closed-form support examples, an independent polyhedral distance oracle, and
redundant-grid refinement. They also execute the maintained example script:

```sh
julia --project=. test/runtests.jl --file=test_invariants.jl --prefix=A54
julia --project=. test/fixtures/generalized_rank_examples.jl
```

The region explorer adds independent interval-summand barcode oracles over QQ,
F2, F3 and F5, transactional selection tests, nonuniform-coordinate native
rendering, and real browser entry, pointer, linked-client and teardown checks:

```sh
julia --project=. test/runtests.jl --file=test_visualization_a54.jl --prefix="A54 visual"
cd test/browser
npx playwright test generalized-rank.spec.mjs
```

The native renderer testset needs CairoMakie and WGLMakie loaded; use the
extension lane described below. Standalone browser checks do not qualify a
notebook frontend.

Repeat `--file` to select several owner files and `--prefix` to select several
families. Names are case sensitive. Unknown files, arguments, fields and unmatched
prefixes fail. Prefixes match the runtime testset description, including field
names produced in loops. Selecting a parent includes its descendants; selecting
a descendant retains its ancestor's setup and assertions. Code outside testsets
still runs, so use both file and prefix selection to bound work.

`--fields=QQ,F2,F3,F5,Real64` selects the coefficient fields used by shared
test loops: rational numbers (`QQ`), the prime fields with two, three or five
elements, and floating point arithmetic (`Real64`). An *oracle* is an
independently calculated expected answer. Oracles written for a particular
field and comparisons across characteristics retain those fields. The runner prints the selected fields and Julia thread counts.

`test/prelude.jl` owns shared imports, field definitions, aliases and fixtures.
`test/test_contracts.jl` owns source/API guards. All tests use the same prelude;
do not copy it or extract it by parsing `runtests.jl`.

## Visualization acceptance contracts

The [support matrix](visualization_support.toml) separates supported display paths
from the evidence for them. Mathematical coordinates and advertised
interaction are standing requirements for visualization changes. Run the affected independent oracles and actual
renderers; live controls additionally need real browser input and cleanup checks.
Use asymmetric, nonuniform fixtures wherever a transpose or midpoint convention
could give a plausible picture. Check actual cell boundaries, values, axis
order, fixed coordinates, masks and orientation in both backends. Classify exact
parameters before converting coordinates for drawing. Live controls need
accessible names, visible changes and matching mathematical readouts; test real
pointer input, keyboard alternatives, read-only hover, invalid-input recovery,
linked clients, disconnection and closure.
Shared renderer changes also run `test_visualization_a38.jl` through this lane.
Its structural budgets include recipe children and check that adding primitives
does not add one plot per primitive, while independent coordinates, path breaks, literal coefficients,
matrix spacing and painting-order cases protect the represented data. Diagram
annotation checks include actual glyph bounds and bounded connector plot trees.
Retained inspector checks assert plot and canvas identity during compatible
updates, correct structural fallback, exact selected values and observer cleanup. Native scene
construction alone cannot certify a backend's serialization: keep halo labels
and other nondefault attributes in the actual export/browser fixtures. Timing
thresholds are kept out of CI: performance studies measure construction, native
updates and export separately, with allocations and retained objects distinct
from the memory used by the whole process. Keep recipe display limits explicit; a batching optimization
must not silently discard labels, multiplicities, source identities or queries.
Check that Tab and Shift+Tab can leave plots and reach later controls, including
after a figure rebuild. A canvas must not trap focus or require a mouse click
to escape it.
At narrow widths, scientific labels must remain readable and wide charts must
remain reachable by keyboard and pointer. Check the figure region separately
from page overflow; shrinking a whole figure can hide a readability failure.

```sh
python test/visualization_contracts.py
python -m unittest discover -s test -p 'test_visualization_contracts.py'
python test/visualization_contracts.py --run-native
```

The last command uses the prepared `test/browser` graphics environment and
requires CairoMakie and WGLMakie. It selects every native owner recorded in the
matrix, including the split interval and newer inspector files. The dedicated
Visualization CI workflow runs this selection and the live/no-server browser
lanes on each push and pull request. Dependency absence, missing export fixtures
and unassigned source files/tests fail this acceptance lane. Reviewers must also
identify new recipes inside existing source files. The optional local
core-only suite may still skip renderer tests when graphics are not installed.
Making this job a branch-protection merge requirement is a repository setting;
the workflow itself cannot enforce that setting.
Both browser lanes require every collected scenario to expect and achieve a
pass. Runtime skips, missing results and successful retries after a failure do
not count as acceptance; the shared reporter makes them fail the run.

Update the family entry when adding a recipe, backend or interaction, including
its independent oracle and backend checks. Add a browser scenario for a new live
control and review representative exported figures and captured browser views.

### Reading the support manifest

[visualization_support.toml](visualization_support.toml) names each family, its
source files and the tests that exercise it. Its values describe the kind of
support and evidence:

| Field and value | Meaning |
| --- | --- |
| `cairo` or `wgl` = `native` | The family has its own mathematical and native renderer checks. Cairo produces static figures; WGL constructs browser scenes. |
| `cairo`, `wgl` or `html` = `shared` | Shared rendering or export code supplies the behavior and its checks. |
| `html` = `browser` | A browser scenario opens the exported scene or fixed composition. |
| `live` = `supported` | A standalone Bonito application uses Julia to answer requests from the browser. |
| `live` = `precomputed` | Browser controls choose among states computed before Julia exits. |
| `live` or `html` = `unsupported` | That interaction or export is not offered for the family. Static selections can still be Julia arguments. |
| `notebooks` = `unverified` | The named notebook frontend has no recorded qualification for these live controls. |

Native callbacks, serialized markup, standalone Chromium, and the individual
IJulia/Jupyter, Pluto and VS Code hosts have separate checks. Testing a notebook's
Julia cells and published figures covers their static output. The browser
[environment](browser/README.md#set-up-the-dependencies) and the manifest's dated
verification record identify the stack used for interaction checks; use fresh
run results when evaluating later changes. The validator checks the record's
format and source assignments, while logs and artifacts establish its results.

## Optional dependencies

```sh
julia --project=. -e 'using Pkg; Pkg.test(test_args=["--file=test_extensions.jl", "--require-extension=TamerOpTablesExt"])'
```

`Pkg.test(test_args=[...])` accepts the same arguments and installs declared test
extras, including Tables. A direct runner invocation uses the active environment.
`--require-extension=NAME` imports the dependencies declared in `Project.toml`
and requires Julia's extension loader to activate that extension. Missing
required dependencies fail; optional-dependency tests skip only when dependencies
are unavailable. Extension files are never manually included by the tests.

CI has a separate environment containing every declared extension's dependencies,
constrained by the project's compatibility bounds. Activation is checked for
every declared extension; the featurizer owner additionally exercises table/IO,
kernel and distance behavior. The visualization owner exports CairoMakie SVG/PNG
figures, checks ordinary persistence specifications, and verifies that WGLMakie
HTML exports contain a serialized scene and canvas. It also checks static export
after switching from WGLMakie to CairoMakie. Browser execution of that HTML is a
separate integration check. Ordinary persistence has an independent
cubical chain and homology-map oracle. These checks do not replace the full
release matrix.

The `A35` visualization testsets check shared styles against the two-square
example: the two rank-one maps still compose to zero, exact matrix entries
remain unchanged, and equal endpoints remain distinct from different parameters
that round to the same drawing coordinates. Native renderer checks cover fonts,
marker shapes and sizes, grayscale, missing heatmap cells, SVG/batch exports
and live-control serialization. Supplied figures must retain their identity and
size while adopting the style; text offsets must leave exact query coordinates
and coefficients unchanged. Default text-role contrast is checked separately
from custom palette choices. Run them with both rendering extensions available:

```sh
julia --project=. test/runtests.jl --file=test_visualization.jl --prefix=A35 --require-extension=TamerOpCairoMakieExt --require-extension=TamerOpWGLMakieExt
```

Inspect representative exported figures for clipping and readable labels.
Recheck the live inspector in a browser at narrow widths and larger text sizes;
native serialization assertions do not certify CSS layout or keyboard focus.

The `A40 A41` visualization testsets check linked finite-window slices using
independent intersections of lines with closed squares. They retain closed
deaths, tangent singleton intervals, multiplicities and censored window ends;
dimension and map-rank checks connect the intervals to the encoded module.
Session tests check rollback, bounded reuse, independent stalk/map selection
and interval IDs shared by the barcode and diagram. Native controls and
serialization remain separate from manual browser acceptance:

```sh
julia --project=. test/runtests.jl --file=test_visualization.jl --prefix="A40 A41" --require-extension=TamerOpCairoMakieExt --require-extension=TamerOpWGLMakieExt
```

In the browser, move both sliders with continuous updates enabled and verify the
committed line and intervals. Then disable continuous updates and verify that
slider drafts change results only after **Apply slice**. Select intervals from each chart and from the dropdown;
the same group must highlight in both charts and the parameter plane. Check
singleton and empty slices, invalid input recovery, linked tabs and closure.
Read exact endpoints when drawing coordinates coincide. The interval and representative fixtures below extend these checks to whole
lines and retained source data.

### A41 interval semantics and retained representatives

The additional A41 files separate mathematical interval behavior from browser
controls. They cover complete-line restrictions with certified tails, shared
barcode/diagram adapters, selection under display budgets, and original interval
members. The ordinary persistence owner checks retained cycles directly against
source boundary matrices: cycles have zero boundary, finite bounding chains have
the selected cycle as their boundary, and the class is nonzero exactly during
the reported lifetime. These checks include duplicates, both parameter orders,
exact grades and unavailable representatives.

Native plot checks also measure diagram-label bounds and separation in Cairo
and WGL at two text sizes, before and after resizing. They verify that annotations
retain the same interval anchors and exact records.

Run the mathematical and session checks first, then the native live controls
in an environment containing WGLMakie:

```sh
julia --project=. test/runtests.jl \
  --file=test_ordinary_persistence.jl \
  --file=test_visualization_a41_intervals.jl \
  --file=test_visualization_a41_slices.jl \
  --file=test_visualization_a41_sessions.jl --prefix=A41

julia --project=. test/runtests.jl \
  --file=test_visualization_a41_live.jl --prefix=A41 \
  --require-extension=TamerOpWGLMakieExt
```

The live checks dispatch actual native mouse events and serialize Bonito controls;
they exercise native events and serialization. Test the resulting controls
through an actual browser using the [browser harness](browser/README.md).
That guide describes the fixture servers, screenshots, keyboard access,
scrolling, page zoom, session cleanup and offline exports. The support matrix
identifies the current required scenarios; its recorded verification has a
source and date.

The manual fixtures below cover periodic intervals, superlevels, coincident
picks, display budgets and incomplete parameter domains. Record the frontend
and source revision when checking them.

For a manual check, display a fresh `visualize(session; backend=:wglmakie)` widget
in the live frontend or local Bonito server being tested. Keep Julia running;
static exported HTML is not a substitute for testing these callbacks. Each
independent browser page must create its own widget, sharing the same session
when testing linked views. Start with two equal essential intervals:

```julia
import TamerOp as OP
import TamerOp.Advanced as OA
using WGLMakie

torus = OP.cubical_persistence(fill(2//3, 1, 1);
    periodic=true, representatives=true)
session = OP.inspection_session(torus; dim=1)
viewer = OP.visualize(session; backend=:wglmakie)
display(viewer)
```

The following manual sequence extends the automated fixture coverage:

1. Select the interval group in either chart and in **Selected interval group**.
   Both charts should highlight the same group, with birth `2/3`, essential death
   `+Inf`, and multiplicity two. The infinity lane is a display position, not a
   finite death value.
2. Request **Show retained representative** before choosing a member. The error
   should explain the missing member, leave the mathematical selection unchanged,
   and return the checkbox to its previous state. Enter `1`, press **Select
   member**, then request the representative. Repeat with member `2`; the two
   cycles have different cell indices. Selecting a new member clears the previous
   representative request. Invalid text and member `3` should recover without
   replacing the last valid state.
3. Recompute the same torus without `representatives=true` and open a new session.
   Selecting a member and requesting its cycle should explain that it was not
   retained. It must not invent a cycle from the interval endpoints.
4. Repeat with `order=:superlevel`. The essential lane and exact interval must
   point toward `-Inf`. For a finite representative, use the square ring from
   [ordinary persistence](https://tamerop.com/ordinary_persistence.html#which-cells-represent-an-interval)
   in degree one: its `[0,5)` class has a cycle and a bounding chain at five.

Next open an interval-only fixture to separate clipping from infinity:

```julia
clipped = OP.inspection_session([(0,2), (0,3), (10,11)];
    window=(0,1), max_intervals=2)
display(OP.visualize(clipped; backend=:wglmakie))
```

The first two diagram points coincide at drawing coordinates `(0,1)`. Their exact
deaths remain two and three, with finite continuation marks. Repeated clicks
should cycle between the two groups, and the selector should distinguish them.
Select the third group: its exact `[10,11)` readout should remain available while
no bar or point is highlighted in the window. All three IDs remain selectable.
With `max_intervals=1`, selecting the other visible group should bring it into
the single displayed slot without changing the total count. Requesting a
representative of a raw interval must explain the absent source correspondence.

Finally, check **Slice scope** in the linked encoding inspector. For the existing
two-square example and diagonal line, use the narrow viewing box
`([3//2,3//2], [7//4,7//4])`. **Whole line (certified endpoints)** should retain
`[0,2]` and `[1,3]`, with finite endpoints outside the drawing window. **Viewing-window
restriction** should instead show one multiplicity-two interval from `3/2` to
`7/4`, censored at both ends. Changing the scope must clear the old interval
selection and preserve the independent stalk/map selection. A whole-line request
on an incompletely represented domain must report that limitation and preserve
the previous valid state.

For every fixture, test keyboard access, narrow widths, 150--200% zoom, readable
exact labels and error recovery. Diagram labels should remain inside the panel;
their connector lines should end at the unchanged interval points. Check both
superlevel points near the right edge and nearby points in the infinity lanes.
Two fresh widgets for one session should share
selections while retaining independent browser clients. Closing one client must
leave its sibling usable; **Close inspector** closes the shared session. Hover
must not request a representative or change the selection. Compare an exported
static snapshot with the selected browser state, and record the browser/frontend
and source revision alongside the result.

## Public onboarding checks and local tutorials

The image comparison tests use fixed colour ranges for empty and filled masks,
complete pixel borders, labels and actual CairoMakie values. After preparing
the [graphics environment](browser/README.md#set-up-the-dependencies), run:

```sh
julia --startup-file=no --threads=1 --project=test/browser test/runtests.jl --file=test_visualization.jl --prefix='A44 image comparison' --prefix='A81 ingestion axes' --fields=F2 --require-extension=TamerOpCairoMakieExt
```

The `A45 informative defaults` testset checks the short public barcode call
against the ring's exact endpoints, clipping and truncation notices, compact
layout, image indices, and styling without global side effects. Select
`--prefix=A45` in the same file and environment to exercise those checks.

The published package does not contain the local `examples/`, `audit/`, or
`benchmark/` directories. Its required tests are self-contained.
`test_examples.jl` always checks the public first-computation API against the
known square-ring interval `[0,5)` and its essential connected component. When
CairoMakie is available, its public export check writes and verifies real PNG/SVG
files in a temporary directory without loading any tutorial files.

```sh
julia --project=. test/runtests.jl --file=test_examples.jl --prefix='Package onboarding'
julia --project=. test/runtests.jl --file=test_field_linalg.jl --file=test_featurizers.jl --prefix=A15
```

Use `--require-extension=TamerOpCairoMakieExt` in an environment containing the
renderer to require its activation. Only the export check skips when that
optional dependency is unavailable. The initialization/provenance checks cover
read-only loading, profile rollback and operation without Git.

Developers who retain the ignored local tutorials can also run:

```sh
julia --project=. test/runtests.jl --file=test_examples.jl
```

That selection additionally executes the available tutorial scripts and notebook
checks. Each tutorial check reports a skip if its local file is absent; these
skips do not imply that the unpublished tutorials have been verified by CI.
The independent public onboarding checks still execute. Tutorial outputs use
temporary directories; direct script users can select an output directory with
`TAMEROP_EXAMPLE_OUTPUT_ROOT`.

Ordinary CI covers package/API contracts, the runner, linear algebra, ingestion,
invariants and public onboarding on Julia 1.12 on Linux, macOS and Windows.
Linux runs with one and four Julia threads. These are selected owner suites,
not the complete release matrix. The extension job also exercises the package
test target through `Pkg.test` and requires all declared extensions.

The full suite is available when explicitly desired:

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
# Equivalent direct invocation, using only the active environment:
julia --project=. test/runtests.jl
```

The CI `workflow_dispatch` input `full_suite=true` enables it across the configured
platform/thread matrix. Record the tested source snapshot,
resolved versions, raw results and exclusions; configuring these jobs does not
establish that they have passed. Historical audit drivers that extracted the old
bootstrap now fail explicitly on current source; reproduce them with their
recorded source archives, or use the maintained runner for current tests.

## Core-only and ordinary persistence checks

The checked-in environment excludes plotting and IO adapter dependencies.
Use `JULIA_LOAD_PATH=@:@stdlib` to avoid accidentally finding optional packages
in a personal default environment (on Windows, use `@;@stdlib`). The minimal
checks also reject explicit missing Folds/progress requests, including empty
batches. Before importing any adapters:

```sh
JULIA_LOAD_PATH=@:@stdlib TAMEROP_TEST_MINIMAL=true julia --project=. test/runtests.jl --file=test_extensions.jl --file=test_visualization.jl --file=test_ordinary_persistence.jl --file=test_featurizers.jl --prefix=A16 --prefix=A21
```

For an extension environment, install the desired dependencies there, and use
`--require-extension=TamerOpCairoMakieExt` (or another declared extension) to
make its absence a failure. Run the same checks in a second fresh process after
precompilation: runtime registrations must survive native cached loading.
The ordinary persistence owner needs no optional packages. Its binary-column tests
check complete supports, workspace reuse, and mixed sparse/packed columns after
shared storage grows, using independent set arithmetic. It checks all
persistence-map ranks against a separately constructed cubical chain oracle,
including periodic axes of length one and two, upper stars, essential classes,
and exact rational grades that have the same Float64 approximation. Additional random chain complexes check
persistent-map ranks for the component and dual-boundary shortcuts, including
one-ended algebraic boundaries that need general reduction, rows with more than
two cofacets, and exact large grades. Closed periodic examples retain their
essential top-dimensional classes.

The related cache-ownership regression checks left and right Kan structure
maps and identity morphisms on equal, separately stored posets. Run its five
fields with multiple worker threads:

```sh
julia --project=. --threads=4 test/runtests.jl --file=test_encoding.jl --prefix='A16 Kan maps respect module-owned caches on equal posets' --fields=QQ,F2,F3,F5,Real64
```

## Release candidate verification

`Release.yml` runs every maintained owner suite against one Git commit on Linux
with Julia 1.12.0 and the current 1.12 patch, Linux with four worker threads and
one interactive thread, and current Julia 1.12 on macOS and Windows. Seven groups
partition the complete owner list in `release_groups.toml`; both ordinary CI
and the release runner reject missing or duplicated owners. Each owner runs in
a fresh Julia process to bound compiler memory and avoid state leaking from
one owner into another. All shared QQ/F2/F3/F5/Real64
loops and the long randomized tests execute. Individual mathematical fixtures
retain their explicit field, backend and random-seed choices.

Separate Linux jobs require all declared extensions and run their complete
geometry and interface owner suites, including native optional geometry backends. This is a declared test matrix, not a claim that every
combination of optional packages, backends and dependency versions has been
tested. Linear-algebra owner tests explicitly exercise supported backend routes;
other owners retain their automatic or fixture-specific backend selection.

The workflow runs on `release/**` branches or by manual dispatch. Verification
workflows use [per-workflow, per-branch concurrency groups](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#concurrency)
to cancel superseded candidate runs when a new commit is pushed. Each group
uploads its source commit/tree, resolved Project/Manifest, Julia/platform/thread
settings, initial seed, threshold-profile hash, commands, per-owner exit codes
and raw logs, including assertion totals and skipped tests. A passing release
requires every group and the separate installed-package workflow to pass on the
same candidate. A later source change requires revalidating affected checks and
a final unchanged candidate; a configured workflow alone is not evidence.

Some owner suites include timing guards. The integer-grid box-cache and
polyhedral batch-cache comparisons warm both variants, alternate their order,
collect unused objects before each paired sample, and record the raw samples. Collections within a batch remain
measured. Interpret these bounds within the recorded machine, Julia version
and thread configuration; retain and investigate timing failures alongside
mathematical assertions.

To reproduce a complete core run locally, use a clean candidate checkout and
fresh directories **outside** that checkout:

```sh
julia --startup-file=no test/release_environment.jl /tmp/tamerop-release-env core
JULIA_NUM_THREADS=1 julia --startup-file=no --project=/tmp/tamerop-release-env test/release.jl --group=all --output=/tmp/tamerop-release-results
```

Use a new environment with `extensions` instead of `core`, then run both
`--extensions=all --group=geometry` and `--extensions=all --group=interfaces`,
with a separate new output directory for each. This reproduces the two optional
integration jobs. Existing package downloads can be reused; the resolved
environment is always recorded.

The distinct [installed-package check](../RELEASING.md#2-verify-before-registration)
uses `Pkg.add` on the exact commit. It records `candidate_files.toml` from Git
and verifies installed paths, raw file bytes and symlink kinds against that
inventory before and after computation. POSIX executable bits are checked on
POSIX systems; Windows retains those Git modes in the inventory without
requiring its filesystem permissions to reproduce them. A mismatch produces
`source_mismatch.toml`. The repository's `.gitattributes` keeps text line endings
as LF across platforms. Windows checkouts can leave `.gitattributes` itself with
CRLF despite that policy. Only that metadata file may be normalized for
comparison, only when it reproduces the expected Git blob; the report records
this conversion. Code/data bytes remain exact, and an unchanged raw filesystem
fingerprint is required after computation and export.

The [release guide](../RELEASING.md) also covers registration and tagging. The
release suite does not need the ignored local tutorials or audit drivers; their
absent tutorial checks remain explicit skips, while self-contained public
mathematical checks still run.

## Persistent cocycle verification

Run the direct cohomology checks with:

```sh
julia --project=. test/runtests.jl --file=test_ordinary_persistence.jl --prefix=A117
```

These compare cohomology barcodes with homology and use independent dense
BigInt modular algebra to check cocycle equations, non-coboundary classes,
simultaneous bases, restriction maps and failure to extend a dying class.
Fixtures include signed basis changes, characteristic-dependent cellular
complexes, implicit flag complexes, exact grades, periodic cubical inputs and
landmarks whose original vertex indices are not sorted. The tests distinguish
cochains on selected input simplices from constructed-cell labels and reject
requests for unsupported cochain lifting through collapse.

The [ordinary persistence guide](https://tamerop.com/ordinary_persistence.html#measuring-a-hole-with-a-cocycle)
explains the public workflow. A cocycle is not an integer lift, a circular
coordinate or a cup product; those require separate verification.

### Landmark coverage and weighted-vertex input

Run the focused input checks with:

```sh
julia --project=. test/runtests.jl --file=test_ordinary_persistence.jl --prefix='Rips inputs:'
```

These tests check landmark order, ties, coincident points, insertion radii,
nearest assignments, optional distance retention and the nearest-point distance
inequality behind the conditional metric bound. Implicit/explicit and encoding
routes share the same selected input and report its coverage. Weighted-vertex
examples use independently enumerated simplices and signed boundary matrices;
independent BigInt linear algebra checks persistence-map ranks over F2, F3 and
F101. Cochain equations, non-coboundaries and restrictions are checked as well
as intervals. Negative births, cutoffs, isolated vertices, sparse stored zeros,
conflicting edges and construction budgets have explicit cases. These checks
are separate from a comparative Ripser timing study.

## Algebra visualization checks

Run the morphism and comparison owners through the package-mode runner:

```sh
julia --project=. test/runtests.jl --file=test_visualization_a42b.jl --file=test_visualization_a42b_comparisons.jl
```

The core fixtures check component and naturality matrices over rationals,
prime fields and a numerical field; nonzero maps with zero composites; empty
spaces; and the actual kernel/image/cokernel inclusions and projections.
Two candidate exact sequences have identical dimensions and component ranks,
but a nonzero composite makes one fail. Rechecking after mutation must not
reuse a stale exactness claim. Numerical equations straddle declared tolerances.
Support tests distinguish overlap from the image of a supplied map and reject
an unrelated classifier with coincidentally matching label numbers.

The comparison fixtures select actual Hom basis morphisms and inspect supplied
cochain maps, homotopies and projective-resolution lifts. Their oracles compare
augmentation, chain and homotopy equations and independently known induced
maps. Malformed witnesses, forbidden generator coefficients, incompatible
coordinates and unrepresented degrees must fail explicitly.

The native rendering test exercises PNG, SVG, PDF and WGL HTML exports when
those extensions are available. It checks coefficient strings, zero-space
matrices and the shared spanned-panel layout. These recipes use static API
selections; they introduce no new live browser callbacks. Numerical tests and
successful file creation are supplemented by visual review of the exported
figures, including grayscale and larger type.

### Resolution tables and incidence

Run the resolution owner and the supplied-lift regression together:

```sh
julia --project=. test/runtests.jl --file=test_visualization_a42c.jl --file=test_visualization_a42b_comparisons.jl
```

The diamond's bottom simple has independently predicted projective
multiplicities `(1,2,1)`; the top simple gives the injective dual. The fixtures
check their actual degree-by-vertex tables, indicator-resolution parity,
nonzero differentials and augmented equations over QQ, F2, F3, F5 and a real
field. One-vertex resolutions with a contractible extra summand distinguish
exactness from minimality. Zero modules, truncated prefixes, malformed stored
matrices, nonprincipal supports, order-forbidden coefficients, singular or
ungraded basis changes, numerical tolerance boundaries and grade-coordinate
collisions have separate assertions. Cropped incidence must retain a selected
summand's global label and preserve the full mathematical matrix in metadata.

For native rendering, use an environment with CairoMakie and WGLMakie installed
and append both `--require-extension=TamerOpCairoMakieExt` and
`--require-extension=TamerOpWGLMakieExt`. A run that skips optional renderers
does not establish backend coverage. The owner checks actual axis viewports
and PNG/SVG/PDF production. Review those figures for layout and readable labels.
These resolution selections are static Julia arguments; live notebook controls
and spectral-page views require their own tests.

## Anchored rank sections

`test/test_visualization_a84.jl` checks ordinary pair ranks over QQ, F2, F3,
F5 and Real64. Its independent oracles include two maps of rank one whose
composite is zero, a second module with the same stalk dimensions and a
nonzero composite, and a matrix whose rank changes in characteristic two.
The square fixtures check actual endpoint dimensions and boundary ranks.
Exact parameter masks are checked separately from finite-label rank rows:
incomparable points in one fiber, reversed axes, unrepresented parameters,
and distinct exact anchors that collapse to the same drawing coordinate.
Pointer anchors use the exact represented values of their floating coordinates
for clipping; their original approximate-input status remains visible.

The same file checks bounded live-cache reuse, transactional failures and
reset/close behavior. With CairoMakie and WGLMakie available it also renders
static SVGs in both palettes and exercises native Bonito controls. Run that
optional lane with both extensions required so missing dependencies cannot
silently substitute a skip for rendering evidence:

```sh
julia --project=. test/runtests.jl --file=test_visualization_a84.jl \
  --require-extension=TamerOpCairoMakieExt --require-extension=TamerOpWGLMakieExt
```

Use an environment containing both backends and the current checkout (the
browser environment alone may need CairoMakie added). The dependency-free
mathematical lane can use `--project=.` without extension requirements.

`test/browser/rank-sections.spec.mjs` checks the live square fixture through
Chromium: both rank directions, rank-one adjacent maps and their zero
composite, same-fiber incomparability, invalid-input recovery, real pointer
selection, hover without rank work, narrow layout and closure. Its screenshots
retain the displayed sections. Run from `test/browser` with the setup and
environment controls in its [README](browser/README.md):

```sh
npx playwright test rank-sections.spec.mjs
```

When reviewing a figure, check that its axes name the varying endpoint and
its heading identifies both fixed coordinates. The order mask must pass
through a classifier fiber when the anchor lies inside that fiber. It must
not inherit the whole finite label's support. White rank-zero regions, gray
unordered regions and unrepresented regions must remain distinguishable.
The selected matrix must agree with the rank and both endpoint dimensions;
a schematic finite-poset drawing must make no ambient-coordinate claim.
Check that repeated cells do not crowd the rank labels and that selection
markers leave those labels readable. The native live-view test also checks
that headings and the legend leave a usable plot viewport. The grayscale
check keeps every positive-rank shade darker than the no-map mask; review
small multiples as well as single-anchor figures for this distinction.

## Ordinary diagram analysis and saved results (A116)

Run the diagram adapter checks with:

```sh
julia --project=. test/runtests.jl --file=test_ordinary_persistence.jl --prefix=A116
```

The distance oracle enumerates partial bijections independently, including all
unused-point diagonal costs. It checks each ground norm, Wasserstein exponents
and assignment backend, repeated bars, reflected superlevels and separately
matched essential births. Moderate repeated fixtures exercise the automatic
auction route. Direct tent, Gaussian, weighted-average and Shannon formulas
check the features. Extreme exact grades test subtraction before conversion,
explicit rescaling and rejection of collapsed numerical grids.

Owned-result checks preserve scalar grade types, prime fields, orientation,
provenance and retained homology/cohomology data. After loading, the independent
boundary/rank oracles check cycles, death fillings and cocycle restrictions
against the supplied complexes. Implicit Rips and landmark source indices are
checked separately. Rejection tests cover schema/version/header disagreement,
unknown tags, malformed data and unsupported metadata without overwriting a
pre-existing file. These tests do not claim that loading alone proves source
chain equations.

### Matching witnesses and comparisons

Run the focused mathematical/session checks with
`julia --project=. test/runtests.jl --file=test_visualization_a85.jl`.
They cover independently calculated diagonal and essential costs, duplicate
members, exact versus sampled slice values, the interior `5/4` optimum,
normalization and field parity, rejected-update recovery and lifecycle. Require
`TamerOpCairoMakieExt` and `TamerOpWGLMakieExt` in a rendering environment to
include native figures and PNG/SVG/PDF exports. A skipped rendering test is not
rendering evidence. Use two Julia threads to exercise optimizer parity.

`test/browser/matching.spec.mjs` uses the maintained browser harness. Its two
scenarios operate the actual controls and canvases: coincident-member cycling,
diagonal/essential readouts, independent linked viewers and disconnect cleanup,
invalid input recovery, narrow-screen keyboard scrolling, a sampled value of
one versus the exact interior optimum `5/4`, reset and close.

### Spectral-page inspection

`test/runtests.jl --file=test_visualization_a42d.jl --prefix=A42d` exercises
staircase bicomplexes with known higher differentials: earlier pages agree but
`d_2` or `d_3` is nonzero. It checks full- and rank-one maps, kernel/image
quotients, zero and uncomputed terms, cropped targets, horizontal-first and Tor
indexing, field parity, supplied structure matrices and session recovery.
Require both Makie extensions to exercise native rendering and PNG/SVG/PDF
exports. `test/browser/spectral.spec.mjs` covers the `/spectral` fixture through
actual page controls, unknown-to-computed transitions, pointer selection,
linked viewers, hover without computation, narrow keyboard scrolling and close.

## Signed observables and cancellation

`test/test_visualization_a86.jl` checks all comparable pairs of a small module
against its actual structure-map ranks, cumulative point sums, and declared
interval-family generalized ranks. It includes inclusive and half-open endpoint
oracles, irregular rational axes, infinite deaths, smoothing at equality,
negative coefficients, large exact cancellations, direct-query mismatches,
display budgets and transactional session updates. Optional Cairo/WGL checks
exercise exports and native controls; require both extensions for renderer
acceptance. The existing A81 signed-density test remains the regression for
older recipes and unsupported-dimensional discovery.

```sh
julia --project=. test/runtests.jl --file=test_visualization_a86.jl \
  --require-extension=TamerOpCairoMakieExt --require-extension=TamerOpWGLMakieExt
```

`test/browser/signed.spec.mjs` separately checks real Chromium input and linked
view lifecycle. Mathematical tests, native callback tests, reviewed figures
and browser interaction are distinct pieces of evidence.

### Filtered-complex geometry

Run `test/runtests.jl --file=test_visualization_a92.jl --prefix=A92` with the
CairoMakie and WGLMakie extension requirements when validating native rendering.
The owner checks exact before/after activity, signed triangle boundaries,
multicritical unions, source identities, image input conventions, superlevel
and periodic cells, active-volume versus displayed-skeleton counts, budgets,
transactional selection and disposal. It compares ordinary cubical diagrams
with the geometry-bearing complex to prevent a plausible picture of a different
cell model. The maintained `test/browser/complex.spec.mjs` scenarios exercise
actual parameter edits, picking, boundary controls, linked viewers, invalid
edits, narrow keyboard scrolling, 3D display and cleanup.

### Verified source representatives

Run `test/runtests.jl --file=test_source_representatives.jl` for original
boundary equations, filtered contractions, contravariant lifts, prime fields,
multicritical activity, periodic cubical geometry, owned source round-trips and
bounded weighted-support optimization against independent enumeration. Ordinary
persistence and A41 interval-session tests cover the unchanged default path;
A92 tests cover the shared geometry owner. `test/browser/source.spec.mjs` checks
actual pointer/keyboard selection and source-overlay lifecycle. 

### Chains, classes and connecting maps

`test/runtests.jl --file=test_visualization_a89.jl --prefix=A89` checks selected
vectors against explicit kernels/images, original-basis boundary witnesses,
field-dependent quotients, coordinate changes, source activity and cone/snake
connecting maps. Module-stalk and supplied-homotopy fixtures check induced
classes rather than only matching dimensions. Native lanes require CairoMakie
and WGLMakie explicitly; the maintained `test/browser/chains.spec.mjs` scenarios
exercise actual degree/arrow selection, vector input, recovery and lifecycle.

## Refinement and transport inspection

Run `test/runtests.jl --file=test_visualization_a88.jl` for finite-source
realization, exact/numerical comparison equations, incompatible endpoints,
zero maps, category counterexamples, transactional selection and renderer/export
checks. Optional CairoMakie/WGLMakie imports enable native figure and control
checks. `test/browser/transport.spec.mjs` checks actual canvas selection,
multiple preimages, exact parameter controls, square matrices, linked viewers,
failed-update rollback and cleanup in Chromium. 

The A104 selected-vector suite is `test/test_propagation.jl`. It checks exact
image columns, independent square-support oracles, fields, coordinate changes,
submodule and quotient equations, explicit source-cycle bases and native
exports. 

### H0 component histories

Run `test/runtests.jl --file=test_visualization_a91.jl` for independent graph
connectivity and persistence-reduction oracles, exact event endpoints, ties,
zero-length bars, superlevels, mixed orientations, multicritical slices and
original memberships through persistence reduction. Invalid incidence and
non-equivalent split maps must fail. The same file covers bounded static views,
transactional sessions and optional Cairo/WGL rendering and exports.
`test/browser/components.spec.mjs` exercises actual bars, source vertices and
merge nodes, rational input, rollback, linked views, narrow keyboard scrolling
and cleanup. 

### Derived classes and products

The focused `test_visualization_a90.jl` suite checks finite-category Ext/Tor
views against the signed diamond relation and noncommutative matrix products,
including zero groups, canonical model choices, representative equations,
budgets, changes of basis and invalid Tor descent. With CairoMakie installed
it also exports PNG/SVG/PDF; WGLMakie, when available, renders the same four
static kinds. Run the owner from the repository root:

```sh
julia --project=. test/runtests.jl --file=test_visualization_a90.jl
```

### Figure composition

Run `test/runtests.jl --file=test_visualization_a102.jl` through the normal
package environment, with CairoMakie available for native attachment and
PNG/SVG/PDF/HTML coverage. The focused oracles cover exact member/source links,
coordinate incompatibility, named target retention, independent scalar ranges,
shared axes, literal matrix anchors and malformed/omitted targets. The maintained `test/browser/composition.spec.mjs` checks real WGL axis changes,
pan/resize, a narrow view and self-contained offline HTML. Set
`TAMEROP_COMPOSITION_HTML` to a generated composed HTML figure for the offline
scenario; an omitted path is reported as a skip, not acceptance evidence.

### Field, degree and construction comparisons

Run the comparison owner through the canonical runner:

```sh
julia --project=. test/runtests.jl --file=test_visualization_a94.jl
```

Its independent fixtures use a determinant of minus two across characteristics,
explicit invertible basis changes satisfying a commuting equation, exact grid
queries (including points closer than display precision), reversed parameter
order, a cellular differential of two, ordinary open/closed endpoints and cutoff
coverage. A numerical diagonal distinguishes exact rank from tolerance-sensitive
rank. Session checks cover transactional rejection, bounded row reuse and release;
a lazy encoding separates dimensions from the first materialized map.

The two Chromium scenarios in
`test/browser/comparison.spec.mjs` cover live pointer and linked-view behavior;
they do not certify every notebook frontend or browser engine.

## Geometric construction inspection

`test/test_visualization_a93.jl` compares geometric inspection with independently
known source membership and homology events: two-site contact and terminal
caps, a square loop ending at exact physical radius `sqrt(2)`, repeated indexed
sites, empty windows, lifted coordinates and spatial sections. Native and flag
barcodes agree on a small triangle across all depths. Core constructions retain
their numerical grades and selected-neighbour-depth semantics. Display budgets,
coefficient fields, censored radius caps and failed-selection rollback are
checked separately.

Review the figure's actual
cells, skeleton/projection labels, source sampling, exact-versus-drawing radius
semantics and barcode endpoints. `test/browser/geometric.spec.mjs` exercises
parameter controls and real cell picks against the local Julia server, including
linked viewers, invalid-depth recovery, narrow screens and close/disconnect.
This standalone browser check is separate from notebook frontend acceptance.

## Collections of supplied features and distances

`test/test_visualization_a98.jl` checks sample/column identities against
asymmetric feature fixtures, exact rational readouts, symmetric masks and
infinite distances, indefinite kernels, budgets, FeatureSet metadata and
transactional selection. A direct one-interval bottleneck oracle checks the
three-sample guide's distance table. Pair opening, explicitly computed pairs,
listener failures, nested sample views and closing are tested independently.

`test/browser/collection.spec.mjs` uses real pointer events and controls to
check matrix and embedding selection, feature definitions, retained sample
views, supplied A85 witnesses, linked viewers, invalid-input recovery and
narrow-screen/close/disconnect behavior. This standalone Chromium check does
not certify every notebook frontend or browser engine.

### Saved mathematical views

Use the focused owners `test_visual_session_codec.jl`,
`test_visualization_a100.jl`, `test_visual_session_composition.jl` and
`test_visual_session_boundaries.jl` for exact values, owned source coordinates,
identity changes, relative resources, malformed records, hidden matrix entries,
cleared selections and archived correspondences. They need no plotting backend.
`test_visual_session_exports.jl` exercises real Cairo SVG/PNG/PDF/HTML and the
reopenable image records; require `TamerOpCairoMakieExt` for that lane.

The separate offline browser lane in `test/browser/portable-session.config.mjs`
runs after its Julia producer exits. It verifies file relocation, selected
states, downloads, keyboard access, narrow layout and absence of network calls.
The browser README gives the producer and test commands. Save/load in another
fresh Julia process additionally checks that owned source identity and exact
coordinate data survive process-local caches and object identities.

### Finite authoring and path witnesses

Run `test/test_visualization_a99.jl` with `test/test_contracts.jl` for exact
path enumeration, arbitrary DAG labels, unequal-length routes, fields,
transactional edits, zero-dimensional matrices, fringe support conditions,
history and executable-code/saved-draft round trips. Load CairoMakie and
WGLMakie explicitly for the native graphics lane; a dependency skip does not
qualify editor rendering. `test/browser/editor.spec.mjs` exercises actual
keyboard input, failed-path readouts, invalid-input recovery, downloads,
linked viewers, narrow layout and teardown for both input families. 

## Relative persistence of filtered pairs

The A118 blocks in `test_ordinary_persistence.jl` construct quotient complexes
independently at each scale and compare every queried persistent-map rank to
BigInt modular elimination. They cover the segment/endpoints and disk/boundary
examples, delayed subcomplex entry, empty/equal pairs, randomized skeletons
with changed chain bases, both parameter orders, exact grades, odd-prime signs
and overflow-sensitive coefficients. They also check actual quotient
differentials, chain maps, homology bases, all three exact-sequence maps,
exactness and naturality; matching interval counts alone is insufficient.

```sh
julia --startup-file=no --project=. test/runtests.jl \
  --file=test_ordinary_persistence.jl --prefix=A118
```

The related A112, A116 and A117 blocks retain prime-field reduction, saved
result and cohomology coverage. A118's result carries the ordinary diagram
contract but no retained artificial cone representatives. Its explicit
quotient bases are scale-specific and must not be interpreted as a canonical
barcode-member matching. Future extensions must preserve this distinction
and test their actual induced maps, including the connecting sign over an
odd prime.

## Exact interval summands

The interval-decomposition owner has independent interval-sum and chain-rank
oracles, exhaustive tiny prime-field splitting checks, non-interval branching
residuals, basis/label changes and thin-cycle field-dependent obstructions.
Retained maps are checked for naturality, both retractions, orthogonal mixed
composites and complete identity reconstruction. Empty modules/families,
invalid intervals, malformed results and exhausted search bounds have separate
negative and boundary cases.

```sh
julia --project=. test/runtests.jl --file=test_interval_decompositions.jl --file=test_contracts.jl
```

For summand inspection, run `test_visualization_a87.jl` in the graphics
environment and the browser scenario `summands.spec.mjs`. The algebra fixtures
compare selected/complement dimensions with explicit interval sums, check the
actual projection columns after a change of basis, and reject maps that match
dimensions but fail the splitting equations. Geometry fixtures use nonuniform,
reversed and non-square classifiers. The Hom oracle distinguishes maps into an
interval from maps out of it. Browser checks exercise real pointer and keyboard
selection, invalid-input recovery, shared clients and cleanup.

## Continuous filtration and slice controls

`continuous.spec.mjs` exercises actual range inputs and native canvas drags,
including physical radius and integer depth, stored degree thresholds, exact event entry, linked viewers,
rapid requests and closure. The small image has eight active faces below 5
and nine at 5. The larger fixture has a 25 by 25 top-cell image and 2,601 total
cells. Native A123 cases in `test_visualization_a92.jl` check independent
triangle activity, superlevel events, rejection of obsolete transactions,
invalid-query rollback and camera retention. Keep these cases in the required
visualization lane when changing request scheduling or controls.

Interaction timing starts at the browser pointer range-input event and ends after the matching
committed readout, an advanced renderer frame and two animation frames. The fixture reports first opening,
first interaction and repeated interactions separately. This measures the visible
browser response on a declared workload; it is not a proof of a universal frame
rate. Keep compilation, warmed queries and cache reuse distinct, and report
concurrent workloads alongside timings. Notebook hosts need their own qualification.
