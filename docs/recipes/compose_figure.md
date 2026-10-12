# Compose a scalar background and signed changes

Start with a scalar table on two declared axes. The result will be one figure
whose grayscale background shows the values and whose signed atoms reconstruct
them by cumulative summation. This is a view of supplied samples; a shaded cell
does not certify the boundary geometry of a persistence module.

```julia
import TamerOp as OP
using CairoMakie

xs = [-1, 0, 1, 2, 3]
ys = [-1, 0, 1, 2, 3]
values = [Int(0 <= x < 2 && 0 <= y < 2) for x in xs, y in ys]
atoms = OP.SignedMeasures.point_signed_measure(values, (xs, ys))

context = OP.VisualContext(domain="square parameters",
    units=("x", "y"), provenance="square support formula")
background = OP.visual_panel(permutedims(values); x=xs, y=ys,
    quantity="dimension", context)
changes = OP.visual_panel(atoms; kind=:signed_atoms, context,
    limits=(x=(-1, 3), y=(-1, 3)))
overlay = OP.overlay_visuals((background=background, changes=changes);
    title="A square and its signed changes")
OP.visualize(overlay)
```

The table construction indexes `x` first; the background expects rows along
`y`, hence the transpose. Positive and negative atoms alternate at the four
corners. Summing the weights below a sampled point in both coordinates gives
one inside the square and zero outside it.

To add a caption attached to data, name the panel and give two anchors:

```julia
note = OP.VisualAnnotation(
    OP.VisualAnchor(:square; point=(0, 0)),
    OP.VisualAnchor(:square; point=(2, 2));
    kind=:highlight, label="The square support")
figure = OP.compose_visuals((square=overlay,); annotations=(note,),
    title="Read the changes in their original coordinates")
OP.visualize(figure)
```

The dashed highlight is an authored explanation. A solid computed link requires
a checked correspondence; the [composition guide](../tutorials/compose_figures.ipynb)
shows a retained source cycle connected to its actual barcode member, with
rendered examples. Matching colors alone never establish that relationship.

```julia
OP.save_visual("figures", "square", figure; format=:svg)
OP.save_visual("figures", "square", figure; format=:html)
```

The HTML file embeds the selected SVG and can reopen without Julia or network
access. It scales panels and annotations together; it does not offer a live
inspector. Consult the [API contract](../src/reference/visualization.md#composing-views-and-annotations)
for compatibility checks, object anchors and intentional comparisons.


For a field with more than two coordinates, first choose its planes and fixed
values with `ScalarField` and `field_session`. The [features guide](../tutorials/features.ipynb)
shows the checked section workflow and shared-scale comparisons before composition.
