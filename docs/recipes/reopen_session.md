# Save a selection and reopen a supported offline view

This recipe saves an ordinary persistence view that someone can open with
Julia closed. The HTML file offers two prepared states: the complete
degree-one barcode and a selected interval with its retained representative.
It contains its own drawings and controls, so copying the file is enough to
share those choices.

Start with a small image ring. The boundary enters at zero and the central
square at five, giving one degree-one interval `[0,5)`. Request representatives
when computing persistence; selecting an interval later cannot recover a
cycle that was not retained.

```julia
import TamerOp as OP
using CairoMakie
A = OP.Advanced

values = zeros(Int, 3, 3)
values[2, 2] = 5
diagram = OP.cubical_persistence(values; representatives=true)
session = OP.inspection_session(diagram; dim=1, window=(-1, 6))
overview = OP.visual_state(session)
```

For this supported inspector, capture also fingerprints the complete diagram
so a later attached source must be the same retained result. To keep only a
static specification without a live-restoration request, use
`visual_state(A.inspection_snapshot(session); selection=A.inspection_selection(session))`
instead.

The window clips the drawing, not the mathematical endpoints. Select the
interval and its original member, then freeze a second state. Equal decorated
intervals share a group; the member identifies which retained cycle to read.

```julia
A.select_inspection!(session; interval=1, member=1, representative=true)
selected = OP.visual_state(session)

mkpath("shared")
path = joinpath("shared", "ring.html")
OP.save_visual_session(path, (overview=overview, selected=selected);
    selected=:selected,
    caption="The image-ring interval [0,5): a cycle appears at zero and fills at five.")
A.close_inspection!(session)
@assert isfile(path)
abspath(path)
```

Open the printed path in a browser, then close Julia. The state selector still
switches between **overview** and **selected**. In the selected state, the
barcode and diagram identify `[0,5)` and the readout reports the chosen member's
cycle and filling. The data downloads preserve the recorded exact values;
the drawing uses numerical coordinates.

You can also inspect the saved record from a fresh Julia session. This needs
TamerOp but no plotting package until you request a rendered figure:

```julia
import TamerOp as OP
A = OP.Advanced

saved = OP.load_visual_session(joinpath("shared", "ring.html"))
@assert Set(A.visual_session_states(saved)) == Set((:overview, :selected))
selection = A.inspection_selection(saved; state=:selected)
@assert selection.interval == 1 && selection.member == 1
@assert selection.representative
selection
```

This file preserves two prepared views. It does not run new persistence
computations or accept new live selections. To resume this supported interval
inspector in Julia, include `object=diagram` when saving, then call
`inspection_session(saved; state=:selected)` after loading. That explicitly
retains the mathematical result as well as the views and increases the
artifact's size. A diagram JSON file by itself does not preserve the selected
view.

The [live-inspection guide](../tutorials/live_inspection.ipynb) develops the
same distinction with an encoding, exact parameter queries and new selections
after restoration. The [session reference](../src/reference/visual_sessions.md)
specifies supported objects, identity checks and resource requirements.
