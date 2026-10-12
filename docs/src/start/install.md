# Install and open a lesson notebook

The lessons include saved figures, so you can read them before installing
anything. To run their computations, you need Julia 1.12 or a compatible newer
Julia 1.x release; follow the
[official Julia installation instructions](https://julialang.org/install/)
if needed. The blocks labelled `julia` below belong at the Julia prompt,
while terminal commands are labelled `sh`.

## Create an environment

A Julia environment records the packages used by your work, allowing you to
reopen it later with the same versions. Create one in a folder of your choice
and install the latest version of TamerOp. This example uses `tamerop-work`
under your home directory:

```julia
import Pkg
workdir = joinpath(homedir(), "tamerop-work")
mkpath(workdir)
cd(workdir)
Pkg.activate(workdir)
Pkg.add(url="https://github.com/erikmnovak/TamerOp.jl", rev="main")
```

`Pkg`, Julia's package manager, downloads TamerOp and its dependencies and
records them in `Project.toml` and `Manifest.toml`. Keep those files with your
work. Installation can take several minutes while Julia downloads dependencies
and precompiles code for later use. Once it finishes, later sessions can load
the package without repeating `Pkg.add`.

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
[guide from inputs to computed objects](../downloads/inputs_to_objects.ipynb).
Each already contains its static figures, which you can read without running
the code. Save the notebook in your `tamerop-work` folder.

Add the plotting and notebook packages to that environment:

```julia
import Pkg
workdir = joinpath(homedir(), "tamerop-work")
Pkg.activate(workdir)
Pkg.add(["CairoMakie", "IJulia"])
import IJulia
IJulia.notebook(dir=workdir)
```

IJulia connects Julia to the Jupyter notebook interface and can offer to
install Jupyter if it is missing. In the browser, open the notebook, select a
Julia kernel (the Julia process that runs its code), and run the cells from
the top. The first package load and figure may take longer while Julia
compiles them.

CairoMakie draws the static figures used in the main lesson. The square
lesson has an optional section explaining how to add WGLMakie and inspect
the example in a live Julia session. You can follow the main computation
and read its saved figures with the packages installed above.

If the kernel cannot find a package, activate your work folder in a cell
before the lesson's imports:

```julia
import Pkg
Pkg.activate("/absolute/path/to/tamerop-work")
```

Use your actual folder path. Keep `Project.toml` and `Manifest.toml` with the
notebook so you can reopen the environment. From a terminal in that folder,
`julia --project=.` also starts Julia with those packages selected.

## Update your installation

With your work environment active, `Pkg.status("TamerOp")` shows the installed
source revision. To obtain the latest version, run:

```julia
import Pkg
Pkg.update("TamerOp")
```

Restart Julia before loading the updated code. For an analysis you need to
reproduce, keep its environment files and use `Pkg.instantiate()` to recover
the recorded dependencies instead of updating them.

## Read before running

In the [ring lesson](../../tutorials/ring.ipynb), nine squares appear at two
different thresholds. Before running the cells, use their assigned values
to predict when the hole forms and fills. The figures and returned interval
let you check that prediction as you read.
