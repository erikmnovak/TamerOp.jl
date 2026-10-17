# Inspect a result and save its figure

Use this workflow to turn a computed result into a figure while keeping its
mathematical values available for further work. We use a small image whose
outer squares appear at `0` and whose central square appears at `5`. Its hole
therefore has interval `[0,5)`: present at `0`, gone at `5`.

This workflow uses the optional CairoMakie package to render an SVG figure.
Install it in the same Julia environment as TamerOp:

```julia
import Pkg
Pkg.add("CairoMakie")
```

Run [visualization.jl](../examples/visualization.jl), choosing where to save the
files:

```julia
import TamerOp
include(joinpath(pkgdir(TamerOp), "agent", "examples", "visualization.jl"))
result = run_visualization_example(; output_dir="figures")
```

The script imports CairoMakie, computes `cubical_persistence`, and checks the
interval through `finite_intervals(diagram; dim=1)`. Here `dim=1` asks about
loops around holes, and the coefficients are in F₂, the field with two elements.
Each matrix entry describes a square's appearance value.

## Inspect before drawing

`TamerOp.Advanced.visual_spec` prepares the contents, labels and axes of a
figure. The example requests a barcode, the horizontal interval drawing of a
persistence result, and selects its first interval. `window=(-1,6)` leaves
room around the endpoints; it changes the displayed range, not the interval.
`check_visual_spec` checks that the prepared figure is consistent enough to
render.

Read mathematical values through the result's accessors, as the script does
with `finite_intervals`. A rounded axis label or a screenshot should not replace
the computed endpoint when reporting a result.

## Save and reopen

The script writes two files with different uses:

| File | Contents |
| --- | --- |
| `ring.svg` | A labelled barcode figure for a browser, article or presentation. |
| `ring.json` | The persistence diagram, which `load_persistence_diagram_json` restores for further queries. |

The saved diagram is reopened and its interval is checked against the original.
The example's return value contains both paths and the in-memory diagram and
figure specification. To display the figure during an interactive Julia session,
use `TamerOp.visualize(result.specification; backend=:cairomakie)`.

For another result, first inspect `available_visuals(result_object)` to choose
a supported view, then check the installed `visual_spec` documentation for its
options. Keep labels, homology degree and the displayed parameter range tied to
the question being answered. If rendering reports that a backend is unavailable,
confirm that CairoMakie is installed in the active environment and has been
imported in this session.

Continue with the [visualization reference](https://tamerop.com/guides/visualization.html)
and [saving visual sessions](https://tamerop.com/recipes/reopen_session.html),
or return to the [agent guide](../README.md).
