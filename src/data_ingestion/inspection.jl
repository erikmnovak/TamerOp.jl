# Passive inspection of lazy ingestion storage. Never populates caches.

_module_materialized(M::_LazyEncodedModule) = M.cached_module !== nothing
_complex_materialized(L::LazyModuleCochainComplex) =
    all(!isnothing, L.terms) && all(!isnothing, L.diffs)

"""
    describe(L::DataIngestion.LazyModuleCochainComplex)

Inspect stored cell counts, the cochain degree range, and the population of
term/differential caches. Inspection does not construct modules, matrices,
active-cell lists, or grid representatives. A partial complex remains lazy.
"""
function describe(L::LazyModuleCochainComplex)
    return (kind=:lazy_module_cochain_complex, field=L.field,
        degree_range=L.tmin:L.tmax, nvertices=nvertices(L.P),
        cell_counts=Tuple(length(g) for g in L.grades_by_dim),
        nterms=length(L.terms), ndifferentials=length(L.diffs),
        materialized=_complex_materialized(L),
        materialized_terms=count(!isnothing, L.terms),
        materialized_differentials=count(!isnothing, L.diffs),
        active_degrees=count(!isnothing, L.active_by_dim),
        vertex_indices_cached=L.vertex_idxs !== nothing)
end

ModuleComplexes.module_complex_summary(L::LazyModuleCochainComplex) = describe(L)

function Base.show(io::IO, L::LazyModuleCochainComplex)
    d = describe(L)
    print(io, "LazyModuleCochainComplex(degrees=", d.degree_range,
        ", vertices=", d.nvertices, ", materialized=", d.materialized, ")")
end

function Base.show(io::IO, ::MIME"text/plain", L::LazyModuleCochainComplex)
    d = describe(L)
    print(io, "LazyModuleCochainComplex",
        "\n  field: ", d.field, "\n  degree_range: ", d.degree_range,
        "\n  vertices: ", d.nvertices, "\n  cells by dimension: ", d.cell_counts,
        "\n  materialized terms: ", d.materialized_terms, "/", d.nterms,
        "\n  materialized differentials: ", d.materialized_differentials, "/", d.ndifferentials)
end

function describe(M::_LazyEncodedModule)
    return (kind=:lazy_encoded_module, field=M.lazy.field,
        degree=M.degree, degree_convention=:homological,
        nvertices=nvertices(M.lazy.P), module_dims=M.dims,
        materialized=_module_materialized(M))
end

function Base.show(io::IO, M::_LazyEncodedModule)
    d = describe(M)
    print(io, "LazyEncodedModule(degree=", d.degree, ", vertices=", d.nvertices,
        ", materialized=", d.materialized, ")")
end

function Base.show(io::IO, ::MIME"text/plain", M::_LazyEncodedModule)
    d = describe(M)
    print(io, "LazyEncodedModule", "\n  field: ", d.field,
        "\n  homology degree: ", d.degree, "\n  vertices: ", d.nvertices,
        "\n  materialized: ", d.materialized, "\n  module_dims: ")
    _show_stored_dimensions(io, d.module_dims)
end
