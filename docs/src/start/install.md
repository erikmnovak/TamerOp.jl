# Install and open a lesson notebook

The lessons include saved figures, so you can read them before installing
anything. To run their computations, you need Julia 1.12 or a compatible newer
Julia 1.x release; follow the
[official Julia installation instructions](https://julialang.org/install/)
if needed. The blocks labelled `julia` below belong at the Julia prompt,
while terminal commands are labelled `sh`.

We will first install the registered TamerOp package and check it with the
ring example. The notebook setup then adds the plotting and notebook packages
and selects the development version used by the website. Julia's package
manager obtains TamerOp and its dependencies from
[General](https://github.com/JuliaRegistries/General/tree/master/T/TamerOp),
its package registry, so you can install by name without cloning the repository.

## Create an environment

A Julia environment records the packages used by your work, allowing you to
reopen it later with the same versions. Create one in a folder of your choice;
this example uses `tamerop-work` under your home directory:

```julia
import Pkg
workdir = joinpath(homedir(), "tamerop-work")
mkpath(workdir)
cd(workdir)
Pkg.activate(workdir)
Pkg.add("TamerOp")
```

`Pkg` selects a compatible registered release and records the environment in
`Project.toml` and `Manifest.toml`. Keep those files with your work. Installation
can take several minutes while Julia downloads dependencies and precompiles
code for later use. Once it finishes, later sessions can load the package
without repeating `Pkg.add`.

Check the installation with a small computation:

```julia
import TamerOp as OP
diagram = OP.cubical_persistence([0 0 0; 0 5 0; 0 0 0])
OP.finite_intervals(diagram; dim=1)
```

The result is `[(0, 5)]`: the outer squares surround a hole at zero, and the
central square fills it at five. The [ring lesson](../../tutorials/ring.ipynb)
develops the interpretation with figures.

## Open and run a notebook

Download the [executed ring notebook](../downloads/ring.ipynb), the
[square inspection notebook](../downloads/inspect_encoding.ipynb), or the
[guide from inputs to computed objects](../downloads/inputs_to_objects.ipynb). Each already contains
its static figures, which you can read without running the code.

To rerun these website notebooks, use the development version: the registered
0.1.0 release does not include their `VisualStyle` and module-inspector APIs.
Keep that choice in a separate lesson environment, leaving the registered
package in your work environment:

```julia
import Pkg
lessondir = joinpath(homedir(), "tamerop-lessons")
mkpath(lessondir)
cd(lessondir)
Pkg.activate(lessondir)
Pkg.add(name="TamerOp", rev="main")
Pkg.add(["CairoMakie", "IJulia"])
```

The lesson environment now follows the repository's development branch, so
later updates will obtain changes from `main`. If TamerOp was already loaded,
restart Julia before activating this environment and starting Jupyter:

```julia
import Pkg
lessondir = joinpath(homedir(), "tamerop-lessons")
Pkg.activate(lessondir)
import IJulia
IJulia.notebook(dir=lessondir)
```

Save the downloaded notebook in `tamerop-lessons`. IJulia connects Julia to the
Jupyter notebook interface and can offer to install Jupyter if it is missing.
In the browser, open the notebook, select a Julia kernel (the Julia process
that runs its code), and run the cells from the top. The first package load
and figure may take longer while Julia compiles them.

CairoMakie draws the static figures used in the main lesson. The square
lesson has an optional section explaining how to add WGLMakie and inspect
the example in a live Julia session. You can follow the main computation
and read its saved figures with the packages installed above.

If the kernel cannot find a package, activate your lesson folder in a cell before
the lesson's imports:

```julia
import Pkg
Pkg.activate("/absolute/path/to/tamerop-lessons")
```

Use your actual folder path. Keep `Project.toml` and `Manifest.toml` with the
notebook so you can reopen the environment. From a terminal in that folder,
`julia --project=.` also starts Julia with those packages selected.

## Update or return to a registered release

With the intended environment active, `Pkg.status("TamerOp")` shows which
version or source branch it uses. `Pkg.update("TamerOp")` updates registered
packages to compatible releases; an environment tracking `main` instead follows
that branch. Restart Julia before loading updated code.

If an existing environment tracks a GitHub URL or development checkout and
you want to return it to registered releases, run:

```julia
import Pkg
Pkg.free("TamerOp")
Pkg.update("TamerOp")
```

If name-only installation cannot find TamerOp, check the capitalization and run
`Pkg.Registry.update()` before retrying. If General is missing from
`Pkg.Registry.status()`, add it with `Pkg.Registry.add("General")`.

## Read before running

In the [ring lesson](../../tutorials/ring.ipynb), nine squares appear at two
different thresholds. Before running the cells, use their assigned values
to predict when the hole forms and fills. The figures and returned interval
let you check that prediction as you read.
