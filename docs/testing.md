# Running correctness tests

Run tests from the repository root with Julia 1.12 and an instantiated project.
The maintained entrypoint is `test/runtests.jl`; it loads this checkout using
`using TamerOp`. A package or extension loading failure fails the run.

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

## Public onboarding checks and local tutorials

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
The ordinary persistence owner needs no optional packages. It checks all
persistence-map ranks against a separately constructed cubical chain oracle,
including periodic axes of length one and two, upper stars, essential classes,
and exact rational grades that have the same Float64 approximation.

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

The workflow runs on `release/**` branches or by manual dispatch. Each group
uploads its source commit/tree, resolved Project/Manifest, Julia/platform/thread
settings, initial seed, threshold-profile hash, commands, per-owner exit codes
and raw logs, including assertion totals and skipped tests. A passing release
requires every group and the separate installed-package workflow to pass on the
same candidate. A later source change requires revalidating affected checks and
a final unchanged candidate; a configured workflow alone is not evidence.

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
as LF across platforms, so byte comparisons do not hide newline changes.

The [release guide](releasing.md) also covers registration and tagging. The
release suite does not need the ignored local tutorials or audit drivers; their
absent tutorial checks remain explicit skips, while self-contained public
mathematical checks still run.
