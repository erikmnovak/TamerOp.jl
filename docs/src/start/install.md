# Install and open the ring notebook

You need Julia 1.12 or a compatible newer Julia 1.x release. If necessary,
follow the [official Julia installation instructions](https://julialang.org/install/).
The commands below labelled `julia` belong in Julia; the `sh` command belongs
in a terminal.

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
Pkg.add(url="https://github.com/erikmnovak/TamerOp.jl.git")
Pkg.add(["CairoMakie", "IJulia"])
```

TamerOp is installed from its GitHub source. The notebook's publication record
identifies the checkout used to produce the displayed results; installing the
current repository can select a different revision.

## Open and run the notebook

Download the [executed ring notebook](../downloads/ring.ipynb) into your work
folder. It already contains its static figures. To edit and rerun it, start
Jupyter from the environment you just created:

```julia
import IJulia
IJulia.notebook(dir=workdir)
```

IJulia can offer to install Jupyter if it is missing. In the browser, open
`ring.ipynb`, select a Julia kernel, and run the cells from the top. The first
package load and figure may take longer while Julia compiles them. The notebook
uses CairoMakie for static figures; it does not require a live visualization
server or WGLMakie.

If the kernel cannot find a package, activate your work folder in a cell before
the lesson's imports:

```julia
import Pkg
Pkg.activate("/absolute/path/to/tamerop-work")
```

Use your actual folder path. Keep `Project.toml` and `Manifest.toml` with the
notebook so you can reopen the environment. From a terminal in that folder,
`julia --project=.` also starts Julia with those packages selected.

## Read before running

The [ring lesson](../../tutorials/ring.ipynb) and downloaded notebook both show
the computed figures without Julia. Begin by predicting which squares are
present and when their hole fills. Running the code then checks your prediction.
