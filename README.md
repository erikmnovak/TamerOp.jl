# TamerOp

TamerOp is a Julia library for computing with persistence modules and finite
posets. It supports point clouds, images, graphs and hand-built algebraic
presentations, with tools for homology, invariants, derived functors and figures.

**Requirements:** Julia **1.12 or a compatible later 1.x version** and an internet
connection for the initial installation. Julia 1.12 is the version currently
used by the package's tests. You do not need to clone this repository or install
its mathematical dependencies individually.

**Distribution status:** install from GitHub using the instructions below.
Name-only installation with `Pkg.add("TamerOp")` is the goal after registration
in Julia's General registry; this README does not claim that registration has
already happened. GitHub installation retrieves the committed, published source.
Local changes on the author's computer become available only after they are
committed and pushed.

- [Install TamerOp](#install-tamerop)
- [Your first computation](#your-first-computation)
- [Save and run a script](#save-and-run-a-script)
- [Make your first figure](#make-your-first-figure)
- [Reopen your project](#reopen-your-project)
- [Choose an import style](#choose-an-import-style)
- [Update or reproduce your environment](#update-or-reproduce-your-environment)
- [Troubleshooting](#troubleshooting)
- [Advanced users and performance settings](#advanced-users-and-performance-settings)
- [Mathematical scope and provenance](#mathematical-scope-and-provenance)
- [Releases and citation](#releases-and-citation)
- [Developing TamerOp](#developing-tamerop)

## Install TamerOp

### 1. Open Julia

If Julia is not installed, follow the
[official installation instructions](https://docs.julialang.org/en/v1/manual/installation/).
Open the Julia application, or type `julia` in your operating system's terminal.
When Julia is ready, you will see a prompt resembling `julia>`.

The blocks labelled **Julia** below belong at that prompt. Copy the code only;
do not type the prompt itself. Blocks labelled **terminal** belong in your
operating system's terminal, outside Julia.

Check your Julia version by entering:

```julia
VERSION
```

Use Julia 1.12 or a compatible newer 1.x version. If your version is older,
update Julia first. If you installed Julia through Juliaup, these commands in
your **terminal** install and start the tested 1.12 series:

```sh
juliaup add 1.12
julia +1.12
```

Juliaup is a Julia version manager, not a prerequisite for using TamerOp.

### 2. Create a folder for your work

A Julia *environment* records which packages a project uses. Giving your work
its own environment makes it easier to reopen, share and reproduce without
interfering with other projects. This setup creates a folder named `tamerop-work`
inside your home folder. Run these lines in **Julia**:

```julia
import Pkg

workdir = joinpath(homedir(), "tamerop-work")
mkpath(workdir)
cd(workdir)
Pkg.activate(workdir)
Pkg.add(url="https://github.com/erikmnovak/TamerOp.jl.git")
```

`Pkg` is Julia's built-in package manager. It downloads TamerOp and its required
dependencies and records them in this folder's `Project.toml` and `Manifest.toml`.
Keep those two files. You normally do not need to edit them yourself.

The first installation may take several minutes while Julia downloads binary
dependencies and **precompiles** code for later use. Plotting packages take
additional time if you choose to install them. Let the operation finish and
wait for the `julia>` prompt to return. Installation messages and precompilation
progress are normal; an `ERROR:` message needs attention.

You install a package once per environment. You do **not** put `Pkg.add(...)`
at the top of every analysis script. Package storage is shared between
environments, so using the same package version in another project normally
reuses its downloaded files. See Julia's
[environment guide](https://pkgdocs.julialang.org/v1/environments/).

If you already know which environment you want, the installation itself is just:

```julia
import Pkg
Pkg.add(url="https://github.com/erikmnovak/TamerOp.jl.git")
```

### 3. Load the library

In the same Julia session:

```julia
import TamerOp as OP
```

`OP` is a short name for the library in your code. Import it again in each new
Julia session or standalone script. There is no need to import internal source
files, change `LOAD_PATH`, or assemble a long setup block.

## Your first computation

Run this in **Julia**, after installing the package:

```julia
import TamerOp as OP

values = [0 0 0;
          0 5 0;
          0 0 0]

diagram = OP.cubical_persistence(values)
println(OP.finite_intervals(diagram; dim=1))
```

The expected answer is:

```text
[(0, 5)]
```

Each matrix entry gives the appearance value of a square. At parameter `0`,
the eight outer squares form a ring. At parameter `5`, the central square
fills its hole. The interval `[0,5)` records that one-dimensional homology class.
Here `dim=1` asks about holes; `dim=0` asks about connected components. Ordinary
persistence currently computes over the two-element field, F₂.

There is also one connected component that never dies:

```julia
OP.essential_births(diagram; dim=0)  # [0]
OP.describe(diagram)
OP.provenance(diagram)
```

`describe` gives a short summary. `provenance` records the mathematical
conventions used. The [ordinary persistence guide](docs/ordinary_persistence.md)
explains interval endpoints, superlevels and periodic images.

## Save and run a script

Save the following text as **`first_analysis.jl`** inside `tamerop-work` using a
plain-text editor. Ensure it is not accidentally named `first_analysis.jl.txt`.

```julia
import TamerOp as OP

values = [0 0 0; 0 5 0; 0 0 0]
diagram = OP.cubical_persistence(values)
println(OP.finite_intervals(diagram; dim=1))
```

From the Julia session where you activated that folder and used `cd(workdir)`:

```julia
include("first_analysis.jl")
```

This `include` runs **your own script**; it is not how TamerOp itself is loaded.
Alternatively, open a **terminal** in `tamerop-work` and run:

```sh
julia --project=. first_analysis.jl
```

The dot in `--project=.` means “use the environment in the current folder.”
Scripts need `println` or `display` to show their results; the Julia prompt
automatically displays the value of an expression you enter interactively.
Lines beginning with `#` are comments. A semicolon inside a function call
introduces named options, as in `dim=1`.

In VS Code, open your work folder and select its Julia environment before
running code. In Jupyter or another notebook interface, activate that same
environment in the Julia kernel before importing TamerOp. Editors and notebooks
are optional; the Julia prompt and a plain `.jl` file are sufficient.

## Make your first figure

Plotting is optional. With your work environment active, install the static
renderer once in **Julia**:

```julia
import Pkg
Pkg.add("CairoMakie")
```

Then put this in a script or run it in Julia:

```julia
import TamerOp as OP
import CairoMakie

diagram = OP.cubical_persistence([0 0 0; 0 5 0; 0 0 0])
figure = OP.visualize(diagram; kind=:barcode, dim=1, backend=:cairomakie)
display(figure)

OP.save_visual("first_barcode.png", diagram;
               kind=:barcode, dim=1, backend=:cairomakie)
OP.save_visual("first_barcode.svg", diagram;
               kind=:barcode, dim=1, backend=:cairomakie)
```

Look in your current working folder for the two files. `pwd()` tells you where
that folder is. PNG is convenient for viewing and slides; SVG is a vector
format suitable for resizing. A terminal may not show a graphical preview;
saving the files still works without an interactive plotting window.

Save the plotting block above as `first_plot.jl` in your work folder to rerun
it. The saved figures are written to the current working folder.

Use `WGLMakie` for interactive HTML figures and install other adapters only when
needed. See the [optional integration guide](docs/optional_integrations.md).
Importing a renderer activates its integration with TamerOp in that session.
Installing it without importing it is not enough.

## Reopen your project

The next time you start Julia, run:

```julia
import Pkg
workdir = joinpath(homedir(), "tamerop-work")
cd(workdir)
Pkg.activate(workdir)
import TamerOp as OP
```

Your existing installation is reused; do not repeat `Pkg.add` each time.
If you copied the project to a different machine, add `Pkg.instantiate()` after
activation to install the versions recorded by its environment files.

From a **terminal** already in your work folder, `julia --project=.` starts Julia
with that environment selected. Your analysis files can then start with only
`import TamerOp as OP` and any optional packages they actually use.

`Pkg.activate(...)` selects packages; `cd(...)` selects where relative file
paths are read and written. Neither operation substitutes for the other.

## Choose an import style

The examples above use one short module name:

```julia
import TamerOp as OP
OP.cubical_persistence([0 1; 1 0])
```

You may instead import only the functions you need:

```julia
using TamerOp: cubical_persistence, finite_intervals

diagram = cubical_persistence([0 0 0; 0 5 0; 0 0 0])
finite_intervals(diagram; dim=1)
```

`using TamerOp` makes the exported convenience names available without a prefix.
The `OP.` style is useful when several libraries have functions with the same
name. The package is named `TamerOp`, with that capitalization; `.jl` is the
file extension, not part of the name you import. Julia uses
`import TamerOp as OP`, not `using TamerOp as OP`.

For function help, type `?` at an empty Julia prompt, then `OP.encode`, or use:

```julia
@doc OP.encode
```

Start multiparameter workflows with `OP.encode`, then choose an invariant or
algebraic operation. The [ingestion options guide](docs/ingestion_options.md)
explains coefficient fields, grids and output stages; the
[multicover guide](docs/multicover.md) includes point-cloud examples.

## Update or reproduce your environment

With your work environment active:

```julia
import Pkg
Pkg.status()
Pkg.update("TamerOp")
```

The GitHub installation tracks a branch, so updating may bring in changes from
that branch and adjust compatible dependencies. Restart Julia after updating
before importing the new code. This project is pre-release; save your environment
files before updating an analysis that must remain reproducible.

Keep your analysis code, input data, `Project.toml` and `Manifest.toml` together.
The manifest records the resolved dependency versions and the installed source
revision. On another machine, activate that folder and run `Pkg.instantiate()`.
Preserving the manifest is particularly important while TamerOp is installed
from an unregistered repository. See
[Julia's package-management guide](https://pkgdocs.julialang.org/v1/managing-packages/).

For a deliberately fixed revision, use `Pkg.add(url=..., rev=...)` with an actual
commit identifier or published tag from the repository. No unreleased tag is
assumed by these instructions. After General registration, new users will be
able to use `Pkg.add("TamerOp")`; existing URL installations can then use
`Pkg.free("TamerOp")` to return to registry-managed versions.

## Troubleshooting

| What you see | What to check |
| --- | --- |
| `julia` is not recognized in your terminal | Open the Julia application, or finish the PATH setup described by the official installer. Restart the terminal after installing. |
| `Package TamerOp not found in current path` | Activate the environment where you installed TamerOp. Inspect `Base.active_project()` and `Pkg.status()`. |
| `TamerOp` cannot be found by `Pkg.add("TamerOp")` | Use the GitHub URL above until registration is complete. |
| A Julia-version compatibility error | Check `VERSION`. This package requires Julia 1.12 or a compatible newer 1.x version. |
| `UndefVarError: OP not defined` | Run `import TamerOp as OP` in this session or at the start of the script. |
| A plotting backend is unavailable | Install the requested renderer in the active environment, then import it in this session. |
| The terminal does not open a plot window | Use `OP.save_visual` and open the saved PNG or SVG. |
| A file is missing or appears in an unexpected place | Check `pwd()`. Relative filenames use that folder, not necessarily your script's folder. |
| `Unsatisfiable requirements` | Try the dedicated project environment above. Existing packages in a shared environment may impose conflicting versions. |
| Download, proxy or certificate errors | Check your connection and your institution's proxy settings. Preserve the complete error message; do not disable certificate verification. |
| Julia still shows old behavior after an update | Restart Julia and reactivate the intended environment. A loaded module is not replaced by updating files on disk. |

For a failed precompile, read the first underlying error, restart Julia, activate
the correct environment, and run `Pkg.instantiate()` followed by `Pkg.precompile()`.
If it still fails, open an [issue](https://github.com/erikmnovak/TamerOp.jl/issues)
with the complete error and a small example. Include `VERSION`,
`Base.active_project()`, and the output of `Pkg.status()`; remove private paths
or data before posting.

## Advanced users and performance settings

The concise import does not remove advanced functionality. The curated advanced
surface is `OP.Advanced`; subsystem modules remain available for specialist use:

```julia
import TamerOp as OP
import TamerOp.DerivedFunctors as DF

# Inspect an advanced API or an owner-specific operation through Julia's help.
@doc OP.Advanced.FinitePoset
@doc DF.Ext
```

Use the public task-oriented functions for ordinary analysis. Import modules or
named functions rather than individual implementation files. The library's
internal underscore-prefixed helpers are not a user API.

Required exact algebra and geometry dependencies are installed by `Pkg`.
Optional renderers and adapters remain optional. Ordinary import does not run
autotuning or write files into the package directory. A compatible repository
`linalg_thresholds.toml` is loaded when present; otherwise the built-in defaults
are used. A profile generated on another machine is not a prerequisite for use.

Advanced users may explicitly tune linear algebra for their current session:

```julia
OP.FieldLinAlg.autotune_linalg_thresholds!(; save=false)
```

This runs benchmarks and can take time; it is not an installation step. For
explicit persistence, the same function accepts a writable `path` with
`save=true`. The canonical developer profile remains `linalg_thresholds.toml`
at the repository root. Saving to another path does not automatically select
that file on a future import; normal users can simply use the defaults.

## Mathematical background

TamerOp is a Julia library for multiparameter persistence built from an encoding-first viewpoint. The central idea is that a multiparameter persistence module should be represented first by a finite, computable encoding on a finite poset, and then analyzed through that encoding. This perspective is guided by the theory of tame modules, especially the viewpoint developed by Ezra Miller: a multiparameter module is understood through finite combinatorial data that captures its essential structure while remaining mathematically faithful.

In practice, this means TamerOp treats encoding not as an implementation detail, but as the central mathematical bridge between raw data and downstream computation. Rather than committing early to one invariant or one storage format, the library emphasizes building finite encoded models that can support many later tasks. This makes it possible to organize multiparameter persistence workflows around a common discrete object instead of a collection of unrelated ad hoc pipelines.

TamerOp starts from several kinds of inputs. These include raw data such as point clouds, graphs, and images, as well as more algebraic inputs such as fringes, flanges, and other presentation-style objects. From those inputs, the library constructs finite-poset encodings that serve as the canonical computational model. Once that encoded model is available, TamerOp supports a broad range of outputs: invariant computations, signed measures, sliced and fibered constructions, homological algebra, derived-functor calculations, visualization, serialization, and related workflows.

This organization is meant to make the library useful both for computation and for mathematical experimentation. A user can begin from concrete data, move to an encoding, and then ask many different questions of the same encoded object. The same encoded perspective also supports more algebraic workflows, where the starting point is already a module presentation rather than a dataset. In both cases, the finite-poset encoding is the common language connecting input, computation, and output.


## Optional plotting and integrations

The mathematical core loads with `using TamerOp`. Plotting and ecosystem
adapters are optional packages activated by explicit imports, for example
`using TamerOp, CairoMakie` after installing CairoMakie in your environment.
See [optional integrations](docs/optional_integrations.md) for installation,
backend selection and export instructions, and
[ordinary persistence](docs/ordinary_persistence.md) for exact F₂ barcodes,
sublevel/superlevel intervals and periodic cubical examples.

## Mathematical scope and provenance

`hom`, `ext`, resolutions and Yoneda products compute in the representation
category of the reported finite poset. `tor` computes over its incidence
algebra. Different encodings of the same ambient module can have different
finite-category Ext/Tor groups. Resolution independence does not establish
encoding independence; see the [category and comparison guide](docs/math_categories.md)
for precise hypotheses, explicit comparison maps and a counterexample.

The [exact matching guide](docs/exact_matching.md) states the finite-window
contract and explains why geometric cells and barcode-cost switches suffice.
The [numerical algebra guide](docs/numerical_algebra.md) explains `RealField`
rank decisions, solve residuals and the limits of near-singular computations.

Use `provenance(result)` or `describe(result).provenance` to inspect the actual
field, finite base, degree convention and recorded construction. Ingestion
results also report their window, orientation, executed backends, grade
arithmetic, sparsification and whether a chosen grid floor-snapped births. This
inspection does not materialize a lazy module. Information not retained by a
raw object is explicitly marked as unknown.

Inspection preserves lazy computation. `show(result)` and `describe(result)`
report stored information; an uncomputed dimension is shown as `not computed`.
For an ingestion encoding, `dimensions(result)` computes and caches only the
dimension vector, while `pmodule(result)` explicitly constructs the module and
its structure maps. Axes queries do not enumerate grid representatives.
See [inspection and explicit computation](docs/lazy_inspection.md) for spectral
pages, representative data, and cache behavior.

High-dimensional alpha and Delaunay lower-star inputs reject unsupported
dimensions by default. `highdim_policy=:rips` explicitly requests a different
construction; provenance identifies that substitution and its grade scale.
To compute homology or an invariant over a different field, recompute from the
original data or complex. Relabelling a stored answer is not a coefficient
change.

## Geometric bifiltrations

For finite point clouds in one or two ambient dimensions:

- `FunctionDelaunayFiltration(vertex_values=...)` implements the incremental
  Delaunay-Cech model of function-sublevel offsets, including insertion cofaces.
  Its grades use minimum enclosing-ball radii. Exact cocircular insertion
  degeneracies are currently rejected explicitly; collinear clouds are supported.
- `CoreFiltration(beta=1.0, k_values=nothing)` constructs the Cech nerve of
  nearest-neighbor core balls. `CoreDelaunayFiltration(...)` restricts those
  balls to full-cloud Voronoi cells, including full cocircular intersections.
  Both count the center as neighbor one and use increasing radius/decreasing
  `k`. Omitting `k_values` uses every density level `1:n`; a supplied subset
  gives exact selected slices with a step extension between them.
- `GraphCoreFiltration(...)` is the separate graph k-core construction.

The constructions accept distinct finite points and use floating-point radii
with robust geometric predicates. `max_dim` truncates simplex dimension;
computing degree `h` homology requires at least `max_dim=h+1`.
Use `estimate_ingestion` and `ConstructionBudget` before larger builds,
especially for the combinatorial Cech construction. Start with
`encode(data, filtration; stage=:graded_complex)` to inspect the output.

`RhomboidFiltration()` constructs the actual unsliced rhomboid multicover
bifiltration in any ambient dimension. At `(r, k)`, it models points covered by
at least `k` radius-`r` balls; `k` is the physical coverage count, with orientation
`(1, -1)`. Exact rational sphere predicates determine the tiling and squared
birth radii; physical radii are stored as exact `AlgebraicReal` values.
Source coordinates must admit exact rational conversion (integers, rationals
or finite floating-point values); irrational algebraic input coordinates are
not supported.
Distinct critical grades survive encoding, queries and JSON round trips, even
when their floating displays coincide. See the [exact-grade guide](docs/exact_grades.md)
for radius semantics, oriented axes and sliced computations. Native backends
require distinct sites in general position; `backend=:subdivision_cech`
supports degenerate and repeated sites with an explicit construction budget.
`depth_range=(lo,hi)` constructs a capped model of the requested coverage
window. See the [multicover guide](docs/multicover.md) for backend choices,
mathematical comparisons and examples.

The native default `max_dim=nothing` retains all cellular dimensions through
intrinsic dimension plus one. Unsliced native cells use cubical boundaries;
capped cells use signed polyhedral boundaries. `stage=:simplex_tree` requests
a coherent simplicial subdivision. The subdivision-Cech fallback is already
simplicial and can have much higher dimension. Use construction budgets:
intermediate constructions can grow combinatorially even when `max_dim` or
`radius` restricts the output.

```julia
using TamerOp

points = PointCloud([[0.0], [2.0]])
filtration = RhomboidFiltration(construction=ConstructionOptions(
    budget=ConstructionBudget(max_simplices=10_000)))
complex = encode(points, filtration; stage=:graded_complex)
components = encode(points, filtration; degree=0)
```

Ingestion `degree=k` computes covariant homology `H_k`. If you explicitly
request `stage=:cochain`, the cellular chains are reindexed as `C^(-k)=C_k`,
so `cohomology_module(C, -k)` gives the same homology module. This convention
keeps inclusion maps compatible with cellular boundaries.

See the [function-Delaunay paper](https://arxiv.org/abs/2310.15902),
[core bifiltration paper](https://arxiv.org/abs/2405.01214), and
[multicover paper](https://arxiv.org/abs/2103.07823).

## Releases and citation

[The changelog](CHANGELOG.md) describes the planned 0.1.0 release and its scope.
An unreleased candidate is not a registered release: continue using the GitHub
installation instructions until this README announces otherwise. The
[maintainer release guide](docs/releasing.md) explains the checks and General
registration needed to enable `Pkg.add("TamerOp")`.

If TamerOp contributes to your research, cite **Erik Novak, TamerOp.jl:
Computing with persistence modules and finite posets**, with the
[repository URL](https://github.com/erikmnovak/TamerOp.jl), and record the version
or Git commit used. [CITATION.cff](CITATION.cff) contains the software citation
metadata. GitHub can use this file to provide a **Cite this repository** button
with APA and BibTeX formats; see [GitHub's citation guide](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files).
No release date or DOI is claimed for the candidate.

To see the package installed in your current Julia analysis environment:

```julia
import Pkg
Pkg.status("TamerOp")
```

Keep that environment's `Project.toml` and `Manifest.toml` with your analysis.
The manifest records the source revision for a GitHub installation as well as
dependency versions. A citation credits the software; those environment files
help someone reproduce the computation.

## Developing TamerOp

This section is for contributors editing the library. Users running analyses
can follow the installation instructions above without cloning the repository.

From a clone, instantiate its development environment, then run selected tests:

```sh
julia --project=. -e 'import Pkg; Pkg.instantiate()'
```

Focused correctness checks use the maintained [test runner](docs/testing.md):

```sh
julia --project=. test/runtests.jl --file=test_data_pipeline.jl --prefix=A14
```

The runner also supports field selection, required extensions and threaded runs.

See [option contracts](docs/option_contracts.md) and
[ingestion options](docs/ingestion_options.md) for representation, coefficient-field,
stage and validation choices.
