# Change of posets and encoding comparisons

A transport operation changes a finite representation along a declared order
map. The [usage notebook](../../tutorials/compare_encodings.ipynb) follows two
encodings of one square and a Kan comparison. [The implementation account](../../implementation/change_of_posets.md)
explains the actual restrictions, limits, colimits and certificates.

## Computation and retained results

Let `pi : Q -> P` be an `Advanced.EncodingMap`. `Advanced.check_monotone_map(pi)`
checks labels and order. In particular, matching vertex counts does not identify
posets or provide a comparison map.

| Operation | Input and output contract |
| --- | --- |
| `restriction(pi, enc)` | Pulls a module on `P` back to `Q`; returns `ModuleTranslationResult`, retaining `enc` and `pi`. No ambient classifier on `Q` is invented. |
| `pushforward_left(pi, enc)` | Left Kan extension from `Q` to `P`; retains the original result and the postcomposed classifier when one exists. |
| `pushforward_right(pi, enc)` | Right Kan extension from `Q` to `P`, with the same result wrapper convention. |
| `Advanced.joint_encoding(left, right)` | Requires monotone finite classifiers with the same labelled source; returns the image of their label pairs with the induced product order. No full product is constructed. |
| `common_refinement(A, B)` | Returns translated modules and projections on a common finite base. With different bases this is an abstract product, not a computed intersection of continuous fibers. |
| `common_refinement(A, B, joint)` | Uses the supplied realized finite joint image. The projection targets must be the original module bases. |
| `Advanced.kan_unit(pi, M; side=:left)` | For a module on `Q`, gives `M -> pi* Lan_pi M`. With `side=:right`, accepts a module on `P` and gives `M -> Ran_pi pi* M`. |
| `Advanced.kan_counit(pi, M; side=:left)` | For a module on `P`, gives `Lan_pi pi* M -> M`. With `side=:right`, accepts a module on `Q` and gives `pi* Ran_pi M -> M`. |

The owner also accepts `PModule`/`PMorphism` inputs for restriction and Kan
operations, returning owner-native modules/morphisms. Workflow wrappers preserve
these methods. Unit and counit return actual `PMorphism` components in the
constructed coordinates; they need not be isomorphisms. Their default `check=true`
validates the result. See method docstrings for computation options such as
threading and session-cache reuse.

Use `Advanced.common_poset`, `Advanced.projection_maps` and
`Advanced.translated_modules` on a common-refinement result. A joint result also
supports `encoding_map` and `Advanced.check_joint_encoding`. A translation
result supports `Advanced.translation_map`, `Advanced.translation_kind`,
`Advanced.translated_module` and `Advanced.source_result`. `describe` and
`provenance` retain construction/category information.

## Static comparison views

`visualize`, `Advanced.visual_spec` and `save_visual` accept these objects:

| Object | Default view | Meaning of left and right |
| --- | --- | --- |
| `EncodingMap` | `:order_map` | Source `Q`, target `P`, common base `Q`. |
| `JointEncodingResult` | `:joint_encoding` | Two classifier targets and their realized joint base. |
| `CommonRefinementTranslationResult` | `:common_refinement` | Original projection targets and the retained common base. |
| `ModuleTranslationResult` with finite map | `:change_of_poset` | Modules on `Q` and `P`. For restriction the left module is the new restriction; for pushforward the right module is the new Kan output. |

Coordinate-only `:regions`, `:region_labels` and `:pushforward_overlay` remain
available for finite maps with the required coordinate layout. Those historical
views are vertex occupancy/layout displays, not continuous parameter fibers.
The new default Hasse views work on arbitrary finite posets.

- `sources=(left,right)` supplies original `PModule` or `EncodingResult` objects
  to an order-map/refinement view. Refinement results do not retain those
  originals themselves. Translation results already retain their source and
  reject this keyword. Actual maps, fields and poset identity are checked;
  incompatible sources raise `ArgumentError`.
- `joint=witness` certifies realized pairs on a supplied **finite** source.
  A `JointEncodingResult` is its own witness. Without a witness, a construction
  tag is reported but absent Cartesian pairs are not asserted unrealized.
  Witness targets must be the displayed posets. A realized result must have
  exactly the witness's projected pairs.
- `comparison=f` supplies a `PMorphism` between the two modules pulled to the
  common base. `direction=:left_to_right` is the default;
  `:right_to_left` supports, for example, a right Kan counit. Endpoints must agree
  with actual modules and cover maps, not only dimensions. Bad endpoints or
  component shapes raise; a well-shaped but nonnatural comparison remains
  inspectable with a failed certificate. All cover squares and component ranks
  determine the reported natural-isomorphism status.
- Select one of `vertex=q`, `pair=(p,q)`, `point=(x,y)`, or
  `side=:left/:right/:base, label=q`. A side label highlights **all** common
  preimages. A parameter point requires two retained supported planar classifiers;
  it is evaluated exactly in both. Missing and multiple common labels are
  reported without selecting an arbitrary candidate. Label 0 is unrepresented,
  not a zero-dimensional stalk.
- `view=:overview` links the three finite posets, label pairs and available
  parameter geometries. `view=:square` displays the selected comparison's four
  arrows and two composites; without a morphism it displays available selected
  pullback maps. A reversed/incomparable pair has no forward square.
- `box=((xmin,ymin),(xmax,ymax))` bounds actual parameter geometry.
  `max_vertices=200` refuses a larger displayed finite poset rather than changing
  its order. Above 20 vertices, Hasse text labels are limited to the selection;
  all vertices and IDs remain in metadata and live hover. `max_pairs=2500` bounds the optional Cartesian background; actual
  common pairs remain drawn. `matrix_limit=(12,12)` crops displayed entries only.

Matrices have source columns and target rows in stored module coordinates.
Rational/prime-field comparisons are exact. `RealField` equations use
`atol + rtol * max(maxabs(left), maxabs(right))`; numerical ranks use the field's
rank policy. A tolerance-based agreement is not a symbolic certificate.

Preparation validates modules and cover maps and materializes supplied lazy
encodings. Session navigation reuses this preparation; a pair query may compose
its selected transitive maps. No viewer computes a missing Kan extension,
unit/counit, all-pairs rank table, or derived functor. `check_visual_request`
checks recipe/selection syntax; full source/comparison checks occur when the
specification or session is built.

## Live inspection and scope

`inspection_session(result; kwargs...)` creates an
`Advanced.TransportInspectionSession` with the same options. Use
`Advanced.select_inspection!` for selection and view, `inspection_selection`,
`inspection_summary`, `inspection_snapshot`, `check_inspection_selection`,
`check_inspection_session`, `reset_inspection!` and `close_inspection!` in
`Advanced`. Failed updates leave the previous snapshot and revision intact.
View-only updates retain selection. A new selector replaces the previous one.
Treat inputs as read-only while the session exists.

Explicitly import WGLMakie to render the live session. Hasse clicks select a
side label; hover reads retained labels without algebra. Controls also accept
exact rational parameter coordinates and directed common-base pairs. Multiple
viewers share session state. Closing the session releases the source objects,
prepared data and listeners, preserving the final static snapshot for export.
A viewer disconnect releases that viewer's callbacks without closing the shared
session. Static exports are snapshots, not offline live widgets.

Every certificate is a statement in the displayed finite representation
category. Equal dimensions do not prove a natural identification. Even an
invertible Kan unit on one module does not identify the two finite derived
categories. A finite witness does not prove complete realization for an
unenumerated ambient parameter space. Ambient and derived comparisons require
their own hypotheses and supplied maps.
