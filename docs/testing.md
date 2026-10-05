# Running correctness tests

Use this guide to choose correctness checks for a change and collect evidence
for a release. Start with the test files for the affected subsystem; the later
sections explain optional integrations, public examples, and verification of a
complete release candidate. For documentation changes, also follow the
[writing and review guidance](writing.md): a passing example does not establish
that its explanation is clear or its mathematical claims are justified.

For performance studies, also read the [benchmarking manual](benchmarking.md).
Passing correctness checks makes a timing eligible for comparison; it does not
establish that compilation, caching or setup costs were measured equivalently.

Run the terminal commands below from the repository root with Julia 1.12 and an
instantiated project. The maintained entrypoint is `test/runtests.jl`; it loads
this checkout using `using TamerOp`. A package or extension loading failure
fails the run.

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. test/runtests.jl --list
julia --project=. test/runtests.jl --file=test_data_pipeline.jl --prefix=A14
julia --project=. --threads=4 test/runtests.jl --file=test_derived_functors.jl --prefix=A14 --fields=QQ,F3,Real64
```

Repeat `--file` to select several owner files and `--prefix` to select several
families. Names are case sensitive. Unknown files, arguments, fields and unmatched
prefixes fail. Prefixes match the runtime testset description, including field
names produced in loops. Selecting a parent includes its descendants; selecting
a descendant retains its ancestor's setup and assertions. Code outside testsets
still runs, so use both file and prefix selection to bound work.

`--fields=QQ,F2,F3,F5,Real64` controls shared parameterized loops. Fixed-field
oracles and intentional comparisons across characteristics retain their own
fields. The runner prints the selected fields and Julia thread counts.

`test/prelude.jl` owns shared imports, field definitions, aliases and fixtures.
`test/test_contracts.jl` owns source/API guards. All tests use the same prelude;
do not copy it or extract it by parsing `runtests.jl`.

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

In the browser, move both draft sliders and verify that results change only
after **Apply slice**. Select intervals from each chart and from the dropdown;
the same group must highlight in both charts and the parameter plane. Check
singleton and empty slices, invalid input recovery, linked tabs and closure.
Read exact endpoints when drawing coordinates coincide. The author reports
that the separate A40/A41 finite-window live-browser checklist passed on
30 September 2026. That acceptance covers the earlier controls, not the new
whole-line scope, interval-family inspector or representative selection below.

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
they do not execute a browser's JavaScript or certify its layout. As of
1 October 2026, all 1,173 dedicated A41 assertions pass. The focused runs cover
12,086 distinct passing assertions, including affected ordinary-persistence and
visualization regressions, API guards and runner contracts. The final renderer
run passes 806 checks, including 344 label-bound, separation and resize checks
in Cairo and WGL. Eight static examples were exported in PNG, SVG and PDF;
the PNG figures passed visual review. These are focused checks, not a full
package-suite run.

**Browser acceptance, 3 October 2026:** all five maintained scenarios passed
in one fresh-server Chromium 153 run (24.4 minutes). The runner exited with
status zero, without a termination signal. Julia completed normal shutdown;
the fixture process was gone, ports 8848/8849 were closed, and the stop marker
was removed. This supersedes the scope of the 2 October evidence, which first
covered two scenarios and later covered four main-run cases plus a separate
zoom rerun.

The interval-diagram tick correction passed **958/958 focused native
assertions** in Cairo and WGL at font sizes 18 and 24, including narrow widths
and resizing. Crowded x-axis labels rotate vertically and return to horizontal
when space permits; every tick value, exact interval endpoint and diagram
point is retained. Review of the actual 24px band browser canvas and its
narrow, horizontally scrolled view confirms separated finite/infinity labels
and readable headings. Very small standalone figures may still require more
height for wrapped headings; the native tick checks do not certify an entire
figure's layout at arbitrary dimensions.

The earlier barcode correction places endpoint text inward and above selection
strokes. Its 2 October owner-file run passed 1,228/1,228 native assertions. These
native results overlap earlier checks and must not be summed into a new total;
they are also separate from the five browser scenarios.

The maintained [browser suite](../test/browser/README.md) now contains five
scenarios. The three additional scenarios reproduce the two-square examples
from the earlier manual acceptance, using the finite encoding of
`k_[0,2]^2` plus `k_[1,3]^2` over the rational numbers:

| Scenario | Mathematical and browser checks |
| --- | --- |
| Ordinary intervals | Triangle intervals `[0,1)` with multiplicity two and `[0,Inf)`; original members and retained chains; actual barcode/diagram picks, errors and linked-view lifecycle |
| Whole-line band slices | `(-Inf,1)`, `(-Inf,Inf)` and `[0,Inf)`, with finite censored window cuts; draft/apply behavior, scope changes, linked charts and lifecycle |
| Square spaces, maps and presentations | Dimensions `1,2,1` in the first square, overlap and second square; exact closed-boundary membership; two rank-one maps with zero composite; incomparable parameters sharing a label; active `[0]` block with a `1 x 0` image basis; pointer/hover, view/support changes and lifecycle |
| Finite-window square slices | Diagonal `[0,2]`, `[1,3]`; translated `[0,1]`, `[1,2]`; tangent singleton intervals; empty and missed-window lines; invalid-input rollback, independent stalk/map selection, reset/hide and linked tabs |
| Large grayscale square inspector | The same zero-map and incomparable-parameter answers; source/target labels and marker shapes, visible keyboard focus, scroll access and chart picks at 150% and 200% browser zoom |

The two historical square scenarios use accessible styling at 18px; the
grayscale scenario uses 24px. The ordinary and band fixtures use the configured
font size. A hidden, read-only fixture record exposes committed Julia state,
matrix shapes and values, hover text and rendered target coordinates. Tests
change selections through actual controls and mouse events. Hover checks
compare the mathematical selection and computation counters before and after
pointer motion, so merely displaying a tooltip must not trigger new algebra.

At 640px viewport width, keyboard arrow keys scroll each visible plot container
to both ends. The suite then picks the right-hand diagram and the barcode to
verify that both remain reachable. It also checks page overflow and preserves
the independent mathematical query. The grayscale scenario loads a local
extension into full Chromium with a temporary profile and calls Chrome's
`tabs.setZoom` at factors `1.5` and `2`. It reads the zoom factor back, checks the
device-pixel-ratio change, and verifies that the visual viewport has not been
pinch-zoomed. This exercises page zoom separately from changing plot text size
or viewport width; setup and implementation references are in the browser guide.

The five-case run independently reproduces the earlier manual two-square
results and verifies keyboard scroll access with actual chart picks. Screenshot
acceptance is recorded separately from interaction assertions: the actual band
canvas, its narrow-right view, and all six zoom viewport captures passed visual
review. The zoom captures show controls and both chart ends at 150% and 200%.

The broader checklist below remains open for the torus, superlevel and raw
interval fixtures, coincident picks, display budgets, incompletely represented
domains, and the remaining visual checks. Notebook frontends and other browsers
remain separate acceptance work. The new page-zoom scenario is limited to the
grayscale square fixture; it does not establish zoom behavior for every recipe.

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
   [ordinary persistence](ordinary_persistence.md#which-cells-represent-an-interval)
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

The tracked [ring notebook](tutorials/ring.ipynb) and
[first encoding inspection](tutorials/inspect_encoding.ipynb) are selected in
`publication.toml` for the [documentation publication workflow](README.md).
The workflow executes each notebook once through IJulia, verifies its
mathematical assertions and required static figures, then generates Documenter
HTML and an executed download. Conversion checks cover
code/math separation, output order, portable links, optional-section folding
and missing/error outputs;
site checks inspect local links, anchors, figures and source/download hashes.
Reading-map checks cover converging and separate routes, cycle detection,
link resolution, escaping, and invalid layouts. Review its native card links,
arrow placement, text outline, keyboard focus, and narrow-screen scrolling
in the browser; the map needs no live Julia session or JavaScript renderer.
This workflow is required in documentation CI and does not treat missing
plotting dependencies as a passing skip.

The independent [spaces-and-maps guide](spaces_and_maps.md) has a focused
example renderer, `docs/build_scripts/render_spaces_maps.jl`. It executes the
guide's 15 static Julia blocks directly from their canonical Markdown, checks
the square's dimensions and maps, distinguishes an incomparable parameter
pair, and verifies a redundant presentation's rank, deferred basis and induced
map equation. It exports the two figures used on the page. This is separate
from notebook execution and from live WGLMakie interaction checks.

The focused `A44 image comparison` testset checks fixed colour ranges for empty
and filled masks, complete pixel borders, labels and native CairoMakie values.
Run it alongside the image-axis regression in the documentation environment:

```sh
julia --startup-file=no --threads=1 --project=docs test/runtests.jl --file=test_visualization.jl --prefix='A44 image comparison' --prefix='A81 ingestion axes' --fields=F2
```

The `A45 informative defaults` testset checks the short public barcode call
against the ring's exact endpoints, visible clipping/censoring/truncation
notices, compact native layout, image indices, and per-call styling without
global side effects. Run `--prefix=A45` in the same owner/environment for these
checks and the small-interval tick checks. Review the optional website section
both collapsed and expanded, including keyboard access, and execute the main
notebook path with optional cells omitted to check its independence.

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
platform/thread matrix. A79 still requires recording a stable source snapshot,
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
partition the complete owner list; the runner rejects missing or duplicated
owners. Each owner runs in a fresh Julia process to bound compiler memory and
avoid state leaking from one owner into another. All shared QQ/F2/F3/F5/Real64
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

The distinct [installed-package check](releasing.md#2-verify-before-registration)
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

The [release guide](releasing.md) also covers registration and tagging. The
release suite does not need the ignored local tutorials or audit drivers; their
absent tutorial checks remain explicit skips, while self-contained public
mathematical checks still run.
