# Options must affect the requested computation

Use the options object belonging to the task. The [ingestion guide](ingestion_options.md)
describes filtration construction, grid representation, field selection and JSON
replay. The following contracts apply to the other option owners.
These option types are available from `TamerOp.Options` and the curated
`TamerOp.Advanced` surface.

## Finite encoding

`EncodingOptions(poset_kind=:dense)` now selects a dense finite poset through
both direct owner calls and workflow calls. Omitting the explicit `poset_kind`
keyword uses the value in the options object; an explicit keyword overrides it.
Geometric poset-only operations do not use a coefficient field. Operations that
produce a fringe or module use `opts.field` as the target field.

A Zn flange call without explicit options preserves the flange's field. Passing
an `EncodingOptions` object explicitly selects its field, whose default is QQ.
Changing characteristic can change ranks and homology; it is not an invariance
claim. Rational coefficients whose denominators vanish in the target field are
rejected by the existing field conversion contract.

`strict_eps` belongs to general PL geometry. The default `nothing` certifies
strict feasibility over the rationals, including arbitrarily narrow cells. An
explicit positive value selects a fixed feasibility margin and can omit cells;
point membership still uses the original strict boundaries. Result provenance
records `feasibility=:exact_rational` or `:fixed_margin`. Zn and axis-aligned box
encoders reject a nondefault value instead of accepting an ineffective setting. Derived
Zn box computations honor their encoding field and region budget, and reject
controls that do not apply to that construction.

## Derived computations

Use `DerivedFunctorOptions` to select the actual computational model. A fixed
projective or injective computation rejects an incompatible model or canonical
coordinate choice. `canon=:auto` resolves to the model actually used;
`:none` requests native coordinates. Unified Ext retains its projective/injective
coordinate comparison capability. Tor has its own `:first`/`:second` models and
rejects inapplicable canonical choices.

For example, use `ExtLongExactSequenceFirst(...; opts=DerivedFunctorOptions(...))`
for the canonical keyword surface. Long exact sequences and derived Kan
computations validate their fixed models before building or consulting caches.
Negative requested degrees are rejected.

`ResolutionOptions(check=true)` validates structural resolution data, including
cached results, whether or not `minimal=true`. The structural checks cover term
alignment, generator data and differential compositions. They are not independent
proofs of exactness of every returned resolution. The finite-poset builders
construct minimal resolutions; `minimal=true` additionally requests the
minimality assertion when checking is enabled. `check=false` explicitly skips
those diagnostics, and negative length bounds are rejected.

Use `ExtDoubleComplex(M, N; maxlen=...)` and
`ExtSpectralSequence(M, N; maxlen=...)` directly. Their former
`ResolutionOptions` overloads were removed because they forwarded only `maxlen`
and ignored the remaining controls. The first constructs the specified bounded
bicomplex; the second still requires complete resolutions for its Ext abutment.

## Module construction and queries

`ModuleOptions(cache=cover_cache)` supplies cover information to module
construction and structure-map queries. The cache must belong to the represented
poset. Supplied caches and conflicting cache arguments are validated even for
identity maps and empty batches.

`check_sizes=false` is a trusted constructor-only opt-out. Structure-map queries
always validate indices and comparability and reject that opt-out. Default
constructor validation also covers reused map storage.

The unused `FiniteFringeOptions` container was removed. Finite-fringe
constructors retain their actual scalar, checking and storage arguments; use
those directly.
