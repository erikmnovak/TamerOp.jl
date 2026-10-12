# Finite modules, selected vectors and submodules

A finite module stores a vector space at each poset vertex and compatible
linear maps along its order. Coordinates in these spaces have meaning only
with respect to the stored bases. A finite encoding additionally retains a
classifier relating original parameters to those vertices.

The [selected-vector guide](../../tutorials/generated_submodules.ipynb) explores
these operations on two overlapping squares. The [implementation account](../../implementation/submodules_and_quotients.md)
explains how the selected images acquire their own structure maps.

To enter and repair a small diagram directly, use the
[construction guide](../../tutorials/presentations.ipynb) and its
[editor contract](visual_sessions.md#author-a-finite-module-or-presentation).

## Select and propagate

`Advanced.check_module(M)` validates stored dimensions and cover-map shapes.
`Advanced.check_module_functoriality(M)` additionally checks independence of
all cover-path composites, using exact equality or the declared `RealField`
tolerances. It uses fresh composition caches, so stale cached maps cannot
conceal changed input. Both accept `throw=true`; the stronger check can require
substantial work on a large poset and is used when retaining a complete saved
visual source.

```julia
r = OP.propagate(M; source=p, vectors=v)
r = OP.propagate(enc; source=p, vectors=B)
summary = OP.propagation_summary(r)
A = OP.Advanced.propagation_image(r; target=q)
```

`M` is a `PModule`; `enc` is an `EncodingResult`. `p` and `q` are finite vertex
IDs in `1:nvertices(P)`, excluding booleans. They are not parameter coordinates.
`v` is a vector of length `dim_at(M,p)`. `B` is a matrix with that many rows;
each column is a selected vector. Dependent columns, zero columns and a matrix
with no columns are accepted. Coefficients are converted to the module's field;
incompatible elements or inexact conversions fail. Nonfinite numerical inputs
are rejected. Storage is validated before propagation, and all applicable cover routes are checked on the selected columns. The input is required to be a valid module; this is not a full path-independence test on unselected vectors.

The result is a `VectorPropagation`. At `q ≥ p`, its image is the full matrix
`M(p ≤ q)B`, preserving the supplied column count, including zeros. Outside the
upper ideal the image and its dimension are `nothing`. A zero-row matrix with
the correct column count is a defined map into a zero stalk. Invalid IDs throw
an `ArgumentError`; an incorrect coordinate row count throws `DimensionMismatch`.

`propagation_summary` and `describe` report source, field, basis convention,
column count, image dimensions, applicable vertices, nonzero support and the
retained encoding. `provenance(r)` additionally records the finite base and a
copy of the chosen source coordinates, with the encoding provenance when present.
`dimensions(r)` copies the image-dimension vector.
`propagation_image` returns a copy. The summary does not compute a decomposition
or certify a direct summand. Exact fields give exact answers; `RealField`
uses the field's numerical rank tolerances. Changing those tolerances can change
reported dimensions and support.

An encoding input materializes a lazy module and retains the encoding. A
`PModule` input retains no classifier. Treat the module, encoding and result
arrays as read-only while using the result. There is no hidden cross-call result
cache. A new selection makes a new computation. `Advanced.check_propagation(r)`
returns the existing abelian `ValidationSummary` wrapper; `.report.valid` and
`.report.issues` expose its result. `throw=true` raises on failure. The check
verifies shapes, applicability, ranks, source columns and all relevant cover
equations against the retained module.

## Construct the subobject and quotient

```julia
S = OP.generated_submodule(r)
N = OP.Advanced.submodule_object(S)
i = OP.Advanced.inclusion_map(S)
Q, projection = OP.Advanced.quotient_with_projection(S)
```

`generated_submodule(r; cache=:auto)` returns the existing `Submodule` wrapper
for the smallest finite-poset submodule containing the selected columns at `p`.
At vertices outside the upper ideal its stalk is zero. Its inclusion is a
`PMorphism` into the original module, with injective components and commuting
cover squares. Its local bases may differ from the supplied spanning columns.

The advanced `cache=CoverCache` option reuses the categorical cover cache on the
identical base. `ambient_module(S)` returns the ambient object;
`check_submodule(S; throw=false)` validates the module and inclusion.
`quotient(S)` returns only the quotient module; `quotient_with_projection(S)`
returns it together with the morphism from the ambient module. Stalkwise,
`projection(q) * inclusion(q) == 0`, and both morphisms respect structure maps.
Floating-field versions use numerical coordinate/rank contracts.

These operations construct subobjects and quotients in the declared finite
representation category. They do not return complementary summands, HN factors,
indecomposable decompositions or an encoding-independent derived invariant.

## Inspect the result

```julia
OP.visualize(r)
OP.visualize(r; target=q)
OP.Advanced.visual_spec(r; target=q, matrix_limit=(12,12), max_vertices=200)
```

The sole result kind is `:vector_persistence`. The finite-order panel shows
selected-image dimensions, distinguishing zero from no map. With a retained
supported planar classifier, a second panel shows its actual fibers. An optional
`box` supplies a finite viewing window in the classifier's usual format.
Unrepresented regions remain unknown. These colors pull back the **finite
submodule**: no original source point has been selected, so they are not a claim
about the upper cone of an arbitrary point inside the source fiber.

`target` requests the full structure map and propagated columns in stored
coordinates. An inapplicable target gets an explicit no-map readout.
`matrix_limit` contains two positive integers and crops only the displayed
matrix; metadata retains the complete matrices. `max_vertices` is a positive
integer graph budget, default 200, and exceeding it raises instead of silently
omitting vertices. Building a view validates retained propagation, may construct
planar geometry, and computes the selected full map only when requested. It
never computes a generated module or quotient. The existing morphism views can
inspect the optional inclusion or projection and their naturality squares.

## Link to retained source cycles

```julia
z = OP.Advanced.propagation_source_chain(r; target=q, basis=cycles, column=1)
OP.visualize(z)
```

This source-cycle route accepts QQ and prime fields. `cycles` is a nonempty vector of `SourceChain` cycles, one per coordinate of the
target stalk, in that coordinate order. They must share the identical source
complex, field, degree and parameter. The function checks activity, cycle
equations and that their classes form a complete independent homology basis.
It then combines them with the propagated coefficients in the selected column.

The caller supplies the identification between this homology basis and the
module's target coordinates. The routine records
`basis_identification=:supplied_by_caller`; equality of dimensions cannot prove
that identification. This local check does not certify homological provenance
of every module map. It does not recover discarded representatives or transport
cocycles covariantly. A missing forward map, empty basis, invalid column or
incompatible cycle data raises an error. The returned `SourceChain` retains
original cell identities and geometry and works with the existing source-cell
viewers. See [source representative contracts](source_representatives.md).

## Diagnose a failed commuting relation

`Advanced.module_path_witness(M::PModule)` returns `nothing` when all cover
paths agree, or a record with `source`, `target`, `first_path`, `second_path`,
`first_composite`, `second_composite` and `field`. Structural errors raise
`ArgumentError` before multiplication. Exact fields compare exactly; a
`RealField` uses the multiplication-scaled tolerance of
`check_module_functoriality`. The returned matrices use the original stalk bases.

The diagnostic follows cover paths directly without reading or changing the
module's cached composites. It checks arbitrary path merges, including two
routes of different lengths, rather than only four-vertex diamonds. It retains
one canonical prefix at each reachable vertex per source and compares each
alternative extension. This polynomial traversal avoids enumerating all paths;
it still performs potentially substantial matrix multiplication and is intended
for small authored diagrams. A failure proves these supplied maps do not define
a representation in the stated field/tolerance contract.

The [construction guide](../../tutorials/presentations.ipynb) uses this witness
through the editor's validation report. The [implementation account](../../implementation/inspection_sessions.md#keep-a-draft-distinct-from-a-mathematical-object)
explains the traversal and transaction boundary.
