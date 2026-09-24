# Optional integrations

`using TamerOp` loads the mathematical core. Rendering, ecosystem adapters and
accelerated geometric backends activate through Julia's native package
extensions when you explicitly import their dependency packages.

From an activated environment containing TamerOp, install a dependency once,
then import it in each Julia session:

```julia
import Pkg
Pkg.add("CairoMakie")
using TamerOp, CairoMakie

diagram = cubical_persistence([0 0 0; 0 2 0; 0 0 0])
spec = TamerOp.Advanced.visual_spec(diagram; kind=:barcode, dim=1)
visualize(diagram; kind=:barcode, dim=1)
save_visual("barcode.svg", diagram; kind=:barcode, dim=1)
```

`TamerOp.Advanced.visual_spec`, diagram computation and result inspection need no renderer.
CairoMakie exports static figures; `Pkg.add("WGLMakie"); using WGLMakie`
activates interactive rendering and HTML figure export. A missing backend or a
failed export raises an error. An HTML summary of a specification is not a
rendered figure and is never substituted for one.

| Capability | Install and import |
| --- | --- |
| Static plots and PNG/SVG/PDF export | `CairoMakie` |
| Interactive plots and HTML figure export | `WGLMakie` |
| Feature CSV import and CSV-specific export options | `CSV` |
| Table interface | `Tables` |
| Data frame conversion | `DataFrames`, `Tables` |
| Arrow feature storage | `Arrow`, `Tables` |
| Parquet feature storage | `Parquet2`, `Tables` |
| NumPy storage | `NPZ` |
| Nearest-neighbor acceleration | `NearestNeighbors` |
| Planar Delaunay acceleration | `DelaunayTriangulation` |
| Distance objects | `Distances` |
| Kernel objects | `KernelFunctions` |
| Folds batch execution | `Folds` |
| Progress reporting | `ProgressLogging` |

For example, `using TamerOp, DelaunayTriangulation` activates the fast Delaunay
backend. `delaunay_backend=:auto` uses an already activated backend or the native
naive implementation. Explicit `:fast` requests require the dependency to be
loaded. Nearest-neighbor `:auto` similarly retains its deterministic brute-force
fallback. No query imports packages behind the scenes.

Basic feature CSV export and native delimited dataset ingestion remain available
in the core. Reading feature CSV artifacts and using CSV-specific writer
keywords require the CSV extension.

`BatchOptions(backend=:folds)` requires `using Folds` when threading is enabled;
`threaded=false` explicitly selects serial execution. `BatchOptions(progress=true)`
requires `using ProgressLogging`. Missing integrations are rejected before a
batch computation, including empty batches.

Native extensions require normal package loading with `using TamerOp`.
Directly including `src/TamerOp.jl` is not an installation method. Loading errors
from a dependency or its extension are reported without manual extension-file
inclusion or catch-all recovery.

Nemo, Polyhedra and CDDLib remain required dependencies of the exact mathematical
core. This separation removes renderer and adapter installation requirements;
it does not replace the core's exact algebra or polyhedral geometry backends.

The root `Manifest.toml` describes the core environment. Installing an optional
package into that environment will intentionally update its project and
manifest; use a separate Julia environment for a distinct selection of adapters.
See [testing](testing.md) for minimal and extension-enabled verification.
