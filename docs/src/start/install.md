# Install and open a lesson notebook

You need Julia 1.12 or a compatible newer Julia 1.x release. If necessary,
follow the [official Julia installation instructions](https://julialang.org/install/).
The commands below labelled `julia` belong in Julia; the `sh` command belongs
in a terminal.

Install TamerOp by name from [Julia's General registry](https://github.com/JuliaRegistries/General/tree/master/T/TamerOp).
You do not need to clone its repository or install its dependencies individually.
Plotting and notebooks are optional additions.

## Create an environment

A Julia environment records the packages used by your work. In Julia, create
one in a folder of your choice. This example uses `tamerop-work` under your
home directory:

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
and initial precompilation can take several minutes; you do not need to repeat
`Pkg.add` each time you start Julia.

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

Download the [executed ring notebook](../downloads/ring.ipynb) or the
[square inspection notebook](../downloads/inspect_encoding.ipynb). Both already
contain their static figures, which you can read without running the code.

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

This explicitly tracks the repository's development branch rather than a
registered release. Restart Julia if TamerOp was already loaded, then activate
the lesson environment and start Jupyter:

```julia
import Pkg
lessondir = joinpath(homedir(), "tamerop-lessons")
Pkg.activate(lessondir)
import IJulia
IJulia.notebook(dir=lessondir)
```

Save the downloaded notebook in `tamerop-lessons`. IJulia can offer to install
Jupyter if it is missing. In the browser, open
the downloaded notebook, select a Julia kernel, and run the cells from the top.
The first package load and figure may take longer while Julia compiles them. The notebook
uses CairoMakie for static figures. The square lesson's optional live inspector
explains how to add WGLMakie and run its examples in a live Julia session;
the saved figures and main computation need only the packages installed above.

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

The [ring lesson](../../tutorials/ring.ipynb) and downloaded notebook both show
the computed figures without Julia. Begin by predicting which squares are
present and when their hole fills. Running the code then checks your prediction.
