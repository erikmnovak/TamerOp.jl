# Visualization reference

`visualize(object; kind=:auto, ...)` builds a mathematical specification and
renders it through an explicitly loaded backend. `visual_spec(object; ...)`
returns that inspectable specification without rendering. The latter and its
accessors are available through `TamerOp.Advanced`. Use `available_visuals` to
list the supported kinds for the actual object, and `check_visual_request`
to inspect its accepted keyword and cost contract.

## Resolutions and presentation incidence

These overloads accept `DerivedFunctors.ProjectiveResolution`,
`DerivedFunctors.InjectiveResolution`,
`IndicatorResolutions.UpsetResolutionResult`,
`IndicatorResolutions.DownsetResolutionResult`, and the workflow
`Results.ResolutionResult` wrapping either derived resolution. They consume
stored terms, not just a bare matrix of multiplicities, and never construct or
extend a resolution. The [resolution guide](../../tutorials/resolutions.ipynb)
works through a diamond example with known Betti and Bass tables.

| Kind | Objects and meaning | Keywords |
|---|---|---|
| `:betti_table` (projective default) | Stored principal-upset counts, degree rows and finite-vertex columns | `verify=false`, `matrix_limit=(12,12)` |
| `:bass_table` (injective default) | Stored principal-downset counts, same table convention | `verify=false`, `matrix_limit=(12,12)` |
| `:resolution` | Table, selected support, differential incidence and optional stalk | `degree=0`, `summand=nothing`, `vertex=nothing`, `grades=nothing`, `verify=false`, `matrix_limit=(12,12)`, `support_sheets=false`, `basis_change=nothing` |
| `:betti_degrees`, `:bass_degrees` | Multiplicities at supplied planar grades in one stored degree | `grades` required, `degree=0`, `verify=false` |
| `:resolution_lift` | Supplied projective-resolution lift and its equations | Explicit `kind=:resolution_lift`; see the [map guide](../../visualization.md#maps-between-modules) |

Degree means homological degree for projectives and cohomological degree for
injectives. Degree zero selects the (co)augmentation when `vertex` is given.
A positive degree `k` selects `P_k → P_(k-1)` or `I^(k-1) → I^k` respectively.
`summand` is a one-based index in the selected term's stored summand order;
it highlights the projective source column or injective target row. An empty
term has no selectable summand. `vertex` is a finite-poset ID, not an ambient
parameter. Its active rows and columns retain their global summand IDs.
If `matrix_limit` crops the coefficient panel, the selected row or column is
retained within that limit. Metadata records the displayed global indices.

A default table reports **unverified multiplicities**. `verify=true` checks
stored principal bases, agreement of differential coefficients with stored
maps, augmentation naturality, augmented equations, and vertexwise exactness
through the computed prefix. It tests each cover/hull for minimality and the
terminal kernel/cokernel for completion. A valid nonminimal resolution is
viewable and labelled `:nonminimal`; malformed or inexact prefixes are rejected.
A nonzero terminal kernel/cokernel gives `:truncated`. Missing higher degrees
are never padded with zero rows. Exact fields use exact coefficients and rank;
`RealField` uses its declared equality tolerances and numerical-rank backend.
Verification is fresh and does not trust cached minimality summaries.

`grades` supplies one finite real coordinate pair per finite vertex. Its
coordinatewise order must agree with the finite order. No coordinates are
inferred from a Hasse diagram or an encoding classifier's representative points.
Exact supplied grades remain in metadata; plotting converts them to floating
coordinates. Distinct grades that collide in those display coordinates are
rejected; use the exact table instead. A grade plot makes no claim about an ambient multigraded free
resolution. A multiplicity label counts coincident summands at a vertex;
the combined resolution view highlights the selected summand's vertex.

`support_sheets=true` shows separately labelled principal supports in the
selected term. `matrix_limit[1]` bounds the number of these panels; an explicitly
selected summand is also retained. Panel separation is schematic and does not
supply an extra parameter or a decomposition of the resolved module.

### Cokernel presentations and kernel copresentations

`IndicatorTypes.UpsetPresentation` and `DownsetCopresentation` support
`:presentation_incidence` as their default. They accept the resolution selection
keywords except `verify`. Degree 0 selects generators/cogenerators; degree 1
selects relations/corelations. Both show the same single presentation map.
Supports must be principal on the supplied finite poset. A projective
presentation is displayed as a **cokernel**; the downset copresentation is a
**kernel**. Neither is the fringe **image** shown by `:presentation_inspector`.
No minimality or subsequent resolution is inferred.

All displayed coefficient matrices use target rows and source columns. This
transposes the stored `UpsetPresentation.delta` convention. Labels identify the
actual summands and, when supplied, their grades. A dagger marks an
order-forced zero; an ordinary zero is allowed by order. At a selected vertex,
the active matrix determines the reported cokernel/kernel dimension.

### Supplied graded basis changes

On a selected differential or presentation, use
`basis_change=(; source=S, target=T)`. The square matrices must use the same
coefficient field and stored source/target summand counts. Their coefficients
must obey the principal-support order; both must be invertible. The viewer
computes `B = inverse(T)*A*S` and checks `A*S = T*B`. It retains both matrices,
the changes and their inverses, and the equation result in metadata.

This certifies the selected map in different graded coordinates, not a
minimalization or decomposition algorithm. It does not transform adjacent maps
or mutate the original object. The augmentation's target is not generally a
principal sum, so resolution basis changes require a positive selected degree.

### Specification data and costs

`visual_metadata(spec)` retains counts, stored degrees, computed length, category,
field, direction, verification status, selection and supplied grades. Incidence
views additionally retain `coefficient_matrix`, `stalk_matrix`, `incidence`,
`basis_change`, and displayed support-sheet indices/counts. A presentation stalk
also supplies `represented_dimension`. Matrix panels retain the full selected
matrix, its original shape and displayed row/column ranges. These arrays are
copied; editing them does not change the input resolution.

Default tables count generators without elimination. A selected injective
coefficient matrix is recovered from its recorded stalk maps. Explicit
verification traverses the whole stored prefix and performs field-aware ranks;
a basis change performs inversions and equation checks. Matrix display limits
do not limit those mathematical checks or discard the exact selected matrix.
Rendering receives only prepared data and performs no resolution algebra.
These views use argument-based selections with CairoMakie or WGLMakie; they do
not supply live resolution inspector controls.
