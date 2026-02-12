module Modules

using SparseArrays, LinearAlgebra
using ..CoreModules: QQ, ModuleOptions, AbstractCoeffField, QQField, PrimeField,
    BackendMatrix, coeff_type, field_from_eltype, eye, coerce
import ..CoreModules: change_field
import ..FiniteFringe
import ..FiniteFringe: AbstractPoset, FinitePoset, cover_edges, leq, nvertices,
       CoverCache, PosetCache, get_cover_cache, warm_cache!, cover_cache,
       clear_cover_cache!, _pairkey, _chosen_predecessor
import ..FieldLinAlg
import Base.Threads


export PModule, PMorphism,
       ModuleOptions,
       change_field,
       zero_pmodule, zero_morphism, direct_sum, direct_sum_with_maps, map_leq

"""
    CoverEdgeMapStore{K,MatT}

Internal storage for the structure maps of a `PModule` on the *cover* relations
of a finite poset.

For each vertex `v` we store:

  * `preds[v]`  : cover predecessors of `v` (sorted)
  * `maps_from_pred[v]` : matrices for the maps `u -> v` in the same order

Dually we store:

  * `succs[u]`  : cover successors of `u` (sorted)
  * `maps_to_succ[u]` : the same matrices, indexed by successors

This eliminates hot-path dictionary lookups when traversing the cover graph.

Canonical access is `store[u,v]` (with `haskey(store, u, v)`).
"""
struct CoverEdgeMapStore{K,MatT<:AbstractMatrix{K}}
    preds::Vector{Vector{Int}}
    succs::Vector{Vector{Int}}
    maps_from_pred::Vector{Vector{MatT}}
    maps_to_succ::Vector{Vector{MatT}}
    nedges::Int
end

Base.length(store::CoverEdgeMapStore) = store.nedges

# Structural equality and hashing for CoverEdgeMapStore.
# Without this, `==` falls back to object identity, so two stores with the same
# content compare unequal.
function Base.:(==)(A::CoverEdgeMapStore, B::CoverEdgeMapStore)
    return A.nedges == B.nedges &&
           A.preds == B.preds &&
           A.succs == B.succs &&
           A.maps_from_pred == B.maps_from_pred &&
           A.maps_to_succ == B.maps_to_succ
end

function Base.isequal(A::CoverEdgeMapStore, B::CoverEdgeMapStore)
    return A.nedges == B.nedges &&
           isequal(A.preds, B.preds) &&
           isequal(A.succs, B.succs) &&
           isequal(A.maps_from_pred, B.maps_from_pred) &&
           isequal(A.maps_to_succ, B.maps_to_succ)
end

function Base.hash(A::CoverEdgeMapStore, h::UInt)
    h = hash(A.nedges, h)
    h = hash(A.preds, h)
    h = hash(A.succs, h)
    h = hash(A.maps_from_pred, h)
    h = hash(A.maps_to_succ, h)
    return h
end

# Iterator: yield ((u, v), map) for each cover edge u -> v, ordered by v then preds[v].
function Base.iterate(store::CoverEdgeMapStore)
    v = 1
    while v <= length(store.preds) && isempty(store.preds[v])
        v += 1
    end
    if v > length(store.preds)
        return nothing
    end
    k = 1
    return ((store.preds[v][k], v), store.maps_from_pred[v][k]), (v, k)
end

function Base.iterate(store::CoverEdgeMapStore, state::Tuple{Int,Int})
    v, k = state
    k += 1
    if k <= length(store.preds[v])
        return ((store.preds[v][k], v), store.maps_from_pred[v][k]), (v, k)
    end
    v += 1
    while v <= length(store.preds) && isempty(store.preds[v])
        v += 1
    end
    if v > length(store.preds)
        return nothing
    end
    k = 1
    return ((store.preds[v][k], v), store.maps_from_pred[v][k]), (v, k)
end

# ----- helper: binary search in sorted Int vectors -----
@inline function _find_sorted_index(xs::Vector{Int}, x::Int)::Int
    i = searchsortedfirst(xs, x)
    if i <= length(xs) && @inbounds xs[i] == x
        return i
    end
    return 0
end

# ----- type-stable zero map creation (dense/sparse) -----
@inline function _zero_edge_map(::Type{Matrix{K}}, ::Type{K}, m::Int, n::Int) where {K}
    return zeros(K, m, n)
end

@inline function _zero_edge_map(::Type{SparseMatrixCSC{K,Int}}, ::Type{K}, m::Int, n::Int) where {K}
    return spzeros(K, m, n)
end

@inline function _zero_edge_map(::Type{MatT}, ::Type{K}, m::Int, n::Int) where {K,MatT<:AbstractMatrix{K}}
    return convert(MatT, zeros(K, m, n))
end

# ----- dictionary-like API (KeyError if not a cover edge) -----
#
# Canonical access:
#   store[u,v]  (KeyError if (u,v) is not a cover edge)
#   haskey(store, u, v)
#
# This avoids tuple allocations and keeps hot loops allocation-free.
# For bulk traversal, prefer store-aligned iteration via succs/maps_to_succ.

@inline function Base.haskey(store::CoverEdgeMapStore, u::Int, v::Int)::Bool
    return _find_sorted_index(store.preds[v], u) != 0
end

@inline function Base.getindex(store::CoverEdgeMapStore{K,MatT}, u::Int, v::Int) where {K,MatT}
    i = _find_sorted_index(store.preds[v], u)
    i == 0 && throw(KeyError((u, v)))
    return @inbounds store.maps_from_pred[v][i]
end

# NOTE: We intentionally avoid generic adapter helpers for edge map access.
# We instead dispatch explicitly on the supported representations at construction time.

"""
    CoverEdgeMapStore{K,MatT}(Q, dims, edge_maps; cache=nothing, check_sizes=true)

Build a store aligned with the cover graph of `Q`.

This is the canonical internal representation used by `PModule` for cover-edge maps.
The constructor supports two input representations:

- `edge_maps::AbstractDict{Tuple{Int,Int},...}`: a tuple-keyed mapping where `(u,v)`
  stores the map for the cover relation `u \\lessdot v`.
- `edge_maps::CoverEdgeMapStore{K,MatT}`: an already-built store.

Missing cover-edge maps are filled with the appropriate zero map.

We keep this as an explicit dispatch point (Dict vs CoverEdgeMapStore) to avoid
runtime "two-world" adapter helpers.
"""
function CoverEdgeMapStore{K,MatT}(
    Q::AbstractPoset,
    dims::Vector{Int},
    edge_maps::AbstractDict{Tuple{Int,Int},<:Any};
    cache::Union{Nothing,CoverCache}=nothing,
    check_sizes::Bool=true,
) where {K,MatT<:AbstractMatrix{K}}

    cc = cache === nothing ? cover_cache(Q) : cache
    preds = cc.preds
    succs = cc.succs
    n = nvertices(Q)

    # Incoming storage (indexed by v, then by sorted preds[v]).
    maps_from_pred = [Vector{MatT}(undef, length(preds[v])) for v in 1:n]

    for v in 1:n
        pv = preds[v]
        mv = maps_from_pred[v]
        dv = dims[v]
        @inbounds for i in eachindex(pv)
            u = pv[i]
            # Dict representation uses tuple keys (u,v).
            if haskey(edge_maps, (u, v))
                A = convert(MatT, edge_maps[(u, v)])
            else
                A = _zero_edge_map(MatT, K, dv, dims[u])
            end

            if check_sizes
                if size(A, 1) != dv || size(A, 2) != dims[u]
                    error("edge map ($u,$v) has size $(size(A)), expected ($(dv),$(dims[u]))")
                end
            end
            mv[i] = A
        end
    end

    # Outgoing storage holds references to the same matrices as maps_from_pred.
    maps_to_succ = [Vector{MatT}(undef, length(succs[u])) for u in 1:n]
    @inbounds for u in 1:n
        su = succs[u]
        mu = maps_to_succ[u]
        for j in eachindex(su)
            v = su[j]
            i = _find_sorted_index(preds[v], u)
            i == 0 && error("internal error: missing cover edge ($u,$v)")
            mu[j] = maps_from_pred[v][i]
        end
    end

    nedges = sum(length, succs)
    return CoverEdgeMapStore{K,MatT}(preds, succs, maps_from_pred, maps_to_succ, nedges)
end

function CoverEdgeMapStore{K,MatT}(
    Q::AbstractPoset,
    dims::Vector{Int},
    edge_maps::CoverEdgeMapStore{K,MatT};
    cache::Union{Nothing,CoverCache}=nothing,
    check_sizes::Bool=true,
) where {K,MatT<:AbstractMatrix{K}}

    # If the caller already has a store, return it (after optional validation).
    if check_sizes
        cc = cache === nothing ? cover_cache(Q) : cache
        if edge_maps.preds != cc.preds || edge_maps.succs != cc.succs
            error("edge_maps store does not match the cover graph of Q")
        end

        preds = cc.preds
        n = nvertices(Q)
        if length(edge_maps.maps_from_pred) != n
            error("edge_maps store has wrong size (expected $n vertices)")
        end

        for v in 1:n
            pv = preds[v]
            mv = edge_maps.maps_from_pred[v]
            dv = dims[v]
            if length(mv) != length(pv)
                error("edge_maps store mismatch at vertex $v")
            end
            @inbounds for i in eachindex(pv)
                u = pv[i]
                A = mv[i]
                if size(A, 1) != dv || size(A, 2) != dims[u]
                    error("edge map ($u,$v) has size $(size(A)), expected ($(dv),$(dims[u]))")
                end
            end
        end
    end

    return edge_maps
end

# ------------------------------ tiny internal model ------------------------------

"""
A minimal module over a finite poset `Q`.

A `PModule` is a functor `Q -> Vec_K` specified by:

  * `dims[i] = dim_K M(i)`
  * maps on cover relations `u leq with dot v`

For performance, cover-edge maps are stored in a `CoverEdgeMapStore`,
aligned with the cover graph, rather than in a dictionary keyed by `(u,v)`.
"""
struct PModule{K,F<:AbstractCoeffField,MatT<:AbstractMatrix{K}}
    field::F
    Q::AbstractPoset
    dims::Vector{Int}
    edge_maps::CoverEdgeMapStore{K,MatT}
end

# choose a storage matrix type from a user-provided mapping
@inline _is_nemo_field(field::AbstractCoeffField) =
    field isa QQField || (field isa PrimeField && field.p > 3)

@inline _is_backend_dense_candidate(A::AbstractMatrix) = !(A isa SparseMatrixCSC)

@inline function _should_backendize_map(field::AbstractCoeffField, A::AbstractMatrix)
    _is_nemo_field(field) || return false
    FieldLinAlg.have_nemo() || return false
    _is_backend_dense_candidate(A) || return false
    m, n = size(A)
    return m * n >= FieldLinAlg.NEMO_THRESHOLD[]
end

function _should_backendize_maps(field::AbstractCoeffField, edge_maps)::Bool
    _is_nemo_field(field) || return false
    FieldLinAlg.have_nemo() || return false
    for (_, A) in edge_maps
        if A isa AbstractMatrix && _should_backendize_map(field, A)
            return true
        end
    end
    return false
end

@inline function _pmodule_mat_type(::Type{K}, edge_maps, field::AbstractCoeffField) where {K}
    if _should_backendize_maps(field, edge_maps)
        return BackendMatrix{K}
    end
    V = Base.valtype(edge_maps)
    if V <: AbstractMatrix{K} && isconcretetype(V)
        return V
    end
    return Matrix{K}
end

"""
    PModule{K}(Q, dims, edge_maps; check_sizes=true)

Construct a `PModule` over coefficient type `K`. `edge_maps` may be a dict or
any mapping supporting `(u,v)` keys. Missing cover maps become zero maps.
"""
function PModule{K}(Q::AbstractPoset, dims::Vector{Int}, edge_maps;
                    check_sizes::Bool=true,
                    opts::ModuleOptions=ModuleOptions(),
                    field::AbstractCoeffField=field_from_eltype(K)) where {K}
    if opts != ModuleOptions()
        check_sizes == true || error("PModule: pass either check_sizes or opts, not both.")
        check_sizes = opts.check_sizes
    end
    coeff_type(field) == K || error("PModule: coeff_type(field) != K")
    MatT = _pmodule_mat_type(K, edge_maps, field)
    store = CoverEdgeMapStore{K,MatT}(Q, dims, edge_maps; check_sizes=check_sizes)
    return PModule{K, typeof(field), MatT}(field, Q, dims, store)
end

# rebase existing store to this poset (important for ChangeOfPosets)
function PModule{K}(Q::AbstractPoset, dims::Vector{Int}, store::CoverEdgeMapStore{K,MatT};
                    check_sizes::Bool=true,
                    opts::ModuleOptions=ModuleOptions(),
                    field::AbstractCoeffField=field_from_eltype(K)) where {K,MatT<:AbstractMatrix{K}}
    if opts != ModuleOptions()
        check_sizes == true || error("PModule: pass either check_sizes or opts, not both.")
        check_sizes = opts.check_sizes
    end
    cc = get_cover_cache(Q)
    if MatT == Matrix{K} && _should_backendize_maps(field, store)
        dense_edge = Dict{Tuple{Int,Int}, Matrix{K}}()
        sizehint!(dense_edge, length(store))
        for ((u, v), A) in store
            dense_edge[(u, v)] = Matrix{K}(A)
        end
        return PModule{K}(Q, dims, dense_edge; check_sizes=check_sizes, field=field)
    end
    if store.preds === cc.preds && store.succs === cc.succs
        coeff_type(field) == K || error("PModule: coeff_type(field) != K")
        return PModule{K, typeof(field), MatT}(field, Q, dims, store)
    end
    new_store = CoverEdgeMapStore{K,MatT}(Q, dims, store; cache=cc, check_sizes=check_sizes)
    coeff_type(field) == K || error("PModule: coeff_type(field) != K")
    return PModule{K, typeof(field), MatT}(field, Q, dims, new_store)
end

# infer coefficient type from first map
function PModule(Q::AbstractPoset, dims::Vector{Int}, edge_maps;
                 check_sizes::Bool=true,
                 opts::ModuleOptions=ModuleOptions(),
                 field::Union{AbstractCoeffField,Nothing}=nothing)
    if opts != ModuleOptions()
        check_sizes == true || error("PModule: pass either check_sizes or opts, not both.")
        check_sizes = opts.check_sizes
    end
    for (_, A) in edge_maps
        return PModule{eltype(A)}(Q, dims, edge_maps; check_sizes=check_sizes,
                                  field=field === nothing ? field_from_eltype(eltype(A)) : field)
    end
    error("Cannot infer coefficient type K from empty edge_maps; use PModule{K}(...)")
end


"""
    dim_at(M::PModule, q::Integer) -> Int

Return the dimension of the stalk (fiber) of the P-module `M` at the vertex `q`.

Mathematically: `dim_at(M, q) = dim_k M(q)`.

This mirrors the existing `dim_at` query used in the Zn/Flange layer, but for
internal `PModule`s used by IndicatorResolutions / DerivedFunctors.

Notes
-----
- Vertices are encoded as integers `1:Q.n`.
- This is a pure convenience method: it simply returns `M.dims[q]`.
"""
function dim_at(M::PModule{K}, q::Integer) where {K}
    return M.dims[Int(q)]
end

# ----------------------------
# Field coercion helpers
# ----------------------------

function _coerce_matrix(field::AbstractCoeffField, A::AbstractMatrix{K}) where {K}
    K2 = coeff_type(field)
    if A isa SparseMatrixCSC{K,Int}
        S = spzeros(K2, size(A, 1), size(A, 2))
        @inbounds for j in 1:size(A, 2)
            for idx in A.colptr[j]:(A.colptr[j + 1] - 1)
                i = A.rowval[idx]
                S[i, j] = coerce(field, A.nzval[idx])
            end
        end
        return S
    end

    M = Matrix{K2}(undef, size(A, 1), size(A, 2))
    @inbounds for j in 1:size(A, 2), i in 1:size(A, 1)
        M[i, j] = coerce(field, A[i, j])
    end
    return M
end

"""
    change_field(M, field)

Return a P-module obtained by coercing all structure maps into `field`.
"""
function change_field(M::PModule{K}, field::AbstractCoeffField) where {K}
    K2 = coeff_type(field)
    edge = Dict{Tuple{Int,Int}, AbstractMatrix{K2}}()
    sizehint!(edge, length(M.edge_maps))
    for ((u, v), A) in M.edge_maps
        edge[(u, v)] = _coerce_matrix(field, A)
    end
    return PModule{K2}(M.Q, M.dims, edge; field=field)
end

"Vertexwise morphism of P-modules (components are M_i \to N_i)."
struct PMorphism{K, F<:AbstractCoeffField, MatT<:AbstractMatrix{K}}
    dom::PModule{K,F}
    cod::PModule{K,F}
    comps::Vector{MatT}   # comps[i] has size cod.dims[i] x dom.dims[i]
    function PMorphism{K,F,MatT}(dom::PModule{K,F}, cod::PModule{K,F},
                                 comps::Vector{MatT}) where {K,F<:AbstractCoeffField,MatT<:AbstractMatrix{K}}
        return new{K,F,MatT}(dom, cod, comps)
    end
end

@inline function _morphism_mat_type(::Type{K}, comps, field::AbstractCoeffField) where {K}
    if _is_nemo_field(field) && FieldLinAlg.have_nemo()
        for A in comps
            if A isa AbstractMatrix{K} && _should_backendize_map(field, A)
                return BackendMatrix{K}
            end
        end
    end
    T = Base.eltype(comps)
    if T <: AbstractMatrix{K} && isconcretetype(T)
        return T
    end
    return Matrix{K}
end

function _build_p_morphism(dom::PModule{K,F}, cod::PModule{K,F},
                           comps::AbstractVector{<:AbstractMatrix{K}}) where {K,F}
    dom.field == cod.field || error("PMorphism: field mismatch")
    dom.Q === cod.Q || error("PMorphism: domain/codomain posets differ")
    n = nvertices(dom.Q)
    length(comps) == n || error("PMorphism: expected $n vertex components, got $(length(comps))")

    MatT = _morphism_mat_type(K, comps, dom.field)
    cvec = Vector{MatT}(undef, n)
    @inbounds for i in 1:n
        A = comps[i]
        if size(A, 1) != cod.dims[i] || size(A, 2) != dom.dims[i]
            error("PMorphism: component $i has size $(size(A)), expected ($(cod.dims[i]),$(dom.dims[i]))")
        end
        cvec[i] = convert(MatT, A)
    end
    return PMorphism{K,F,MatT}(dom, cod, cvec)
end

function PMorphism{K}(dom::PModule{K,F}, cod::PModule{K,F},
                      comps::AbstractVector{<:AbstractMatrix{K}}) where {K,F}
    return _build_p_morphism(dom, cod, comps)
end

function PMorphism{K,F}(dom::PModule{K,F}, cod::PModule{K,F},
                        comps::AbstractVector{<:AbstractMatrix{K}}) where {K,F}
    return _build_p_morphism(dom, cod, comps)
end

function PMorphism{K}(dom::PModule{K,F}, cod::PModule{K,F},
                      comps::Vector{MatT}) where {K,F,MatT<:AbstractMatrix{K}}
    return _build_p_morphism(dom, cod, comps)
end

function PMorphism{K,F}(dom::PModule{K,F}, cod::PModule{K,F},
                        comps::Vector{MatT}) where {K,F,MatT<:AbstractMatrix{K}}
    return _build_p_morphism(dom, cod, comps)
end

PMorphism(dom::PModule{K,F}, cod::PModule{K,F},
          comps::AbstractVector{<:AbstractMatrix{K}}) where {K,F} =
    _build_p_morphism(dom, cod, comps)

PMorphism(dom::PModule{K,F}, cod::PModule{K,F},
          comps::Vector{MatT}) where {K,F,MatT<:AbstractMatrix{K}} =
    _build_p_morphism(dom, cod, comps)

"""
    change_field(f, field)

Return a PMorphism obtained by coercing domain, codomain, and components into `field`.
"""
function change_field(f::PMorphism{K}, field::AbstractCoeffField) where {K}
    dom2 = change_field(f.dom, field)
    cod2 = change_field(f.cod, field)
    K2 = coeff_type(field)
    comps2 = Vector{Matrix{K2}}(undef, length(f.comps))
    @inbounds for i in 1:length(f.comps)
        comps2[i] = _coerce_matrix(field, f.comps[i])
    end
    return PMorphism{K2, typeof(field)}(dom2, cod2, comps2)
end

"Identity morphism."
id_morphism(M::PModule{K,F}) where {K,F} =
    PMorphism{K,F}(M, M, [eye(M.field, M.dims[i]) for i in 1:length(M.dims)])


    
function _predecessors(Q::AbstractPoset)
    return get_cover_cache(Q).preds
end


# ----------------------------
# Zero objects + direct sums (PModules)
# ----------------------------

"""
    zero_pmodule(Q::AbstractPoset; field=QQField())

The zero P-module on a finite poset Q (all stalks 0 and all structure maps 0).
"""
function zero_pmodule(Q::AbstractPoset; field::AbstractCoeffField=QQField())
    K = coeff_type(field)
    edge = Dict{Tuple{Int,Int}, Matrix{K}}()
    for (u,v) in cover_edges(Q)
        edge[(u,v)] = zeros(K, 0, 0)
    end
    return PModule{K}(Q, zeros(Int, nvertices(Q)), edge; field=field)
end

"""
    zero_morphism(M::PModule{K}, N::PModule{K}) -> PMorphism{K}

Zero morphism M -> N.
"""
function zero_morphism(M::PModule{K}, N::PModule{K}) where {K}
    Q = M.Q
    @assert N.Q === Q
    M.field == N.field || error("zero_morphism: field mismatch")
    n = nvertices(Q)
    comps = Vector{Matrix{K}}(undef, n)
    for i in 1:n
        comps[i] = zeros(K, N.dims[i], M.dims[i])
    end
    return PMorphism{K, typeof(M.field)}(M, N, comps)
end

"""
    direct_sum(A::PModule{K}, B::PModule{K}) -> PModule{K}

Binary direct sum A oplus B as a P-module.
"""
function direct_sum(A::PModule{K}, B::PModule{K}) where {K}
    Q = A.Q
    n = nvertices(Q)
    @assert B.Q === Q
    A.field == B.field || error("direct_sum: field mismatch")

    dims = [A.dims[i] + B.dims[i] for i in 1:n]

    # Fast path: traverse cover edges via store-aligned succ lists and grab maps
    # by index (no tuple allocation, no search).
    cc = cover_cache(Q)
    if (A.edge_maps.succs == cc.succs && A.edge_maps.preds == cc.preds &&
        B.edge_maps.succs == cc.succs && B.edge_maps.preds == cc.preds)

        preds = cc.preds
        succs = cc.succs

        maps_from_pred = [Vector{Matrix{K}}(undef, length(preds[v])) for v in 1:n]
        maps_to_succ   = [Vector{Matrix{K}}(undef, length(succs[u])) for u in 1:n]

        @inbounds for u in 1:n
            su = succs[u]
            Au = A.edge_maps.maps_to_succ[u]
            Bu = B.edge_maps.maps_to_succ[u]
            outu = maps_to_succ[u]

            aU, bU = A.dims[u], B.dims[u]

            for j in eachindex(su)
                v = su[j]
                aV, bV = A.dims[v], B.dims[v]

                Muv = zeros(K, aV + bV, aU + bU)

                if aV != 0 && aU != 0
                    copyto!(view(Muv, 1:aV, 1:aU), Au[j])
                end
                if bV != 0 && bU != 0
                    copyto!(view(Muv, aV+1:aV+bV, aU+1:aU+bU), Bu[j])
                end

                outu[j] = Muv
                ip = _find_sorted_index(preds[v], u)
                @inbounds maps_from_pred[v][ip] = Muv
            end
        end

        store = CoverEdgeMapStore{K,Matrix{K}}(preds, succs, maps_from_pred, maps_to_succ, cc.nedges)
        return PModule{K}(Q, dims, store; field=A.field)
    end

    # Fallback (rare): use keyed access.
    edge = Dict{Tuple{Int,Int}, Matrix{K}}()
    sizehint!(edge, length(A.edge_maps))
    for (u, v) in cover_edges(Q)
        Au = A.edge_maps[u, v]
        Bu = B.edge_maps[u, v]
        aU, bU = A.dims[u], B.dims[u]
        aV, bV = A.dims[v], B.dims[v]
        Muv = zeros(K, aV + bV, aU + bU)
        if aV != 0 && aU != 0
            copyto!(view(Muv, 1:aV, 1:aU), Au)
        end
        if bV != 0 && bU != 0
            copyto!(view(Muv, aV+1:aV+bV, aU+1:aU+bU), Bu)
        end
        edge[(u, v)] = Muv
    end
    return PModule{K}(Q, dims, edge; field=A.field)
end


"""
    direct_sum_with_maps(A,B) -> (S, iA, iB, pA, pB)

Direct sum together with canonical injections/projections.
"""
function direct_sum_with_maps(A::PModule{K}, B::PModule{K}) where {K}
    S = direct_sum(A, B)
    Q = A.Q
    n = nvertices(Q)
    @assert B.Q === Q

    iA_comps = Vector{Matrix{K}}(undef, n)
    iB_comps = Vector{Matrix{K}}(undef, n)
    pA_comps = Vector{Matrix{K}}(undef, n)
    pB_comps = Vector{Matrix{K}}(undef, n)

    @inbounds for u in 1:n
        a = A.dims[u]
        b = B.dims[u]

        iA = zeros(K, a + b, a)
        iB = zeros(K, a + b, b)
        pA = zeros(K, a, a + b)
        pB = zeros(K, b, a + b)

        for t in 1:a
            iA[t, t] = one(K)
            pA[t, t] = one(K)
        end
        for t in 1:b
            iB[a + t, t] = one(K)
            pB[t, a + t] = one(K)
        end

        iA_comps[u] = iA
        iB_comps[u] = iB
        pA_comps[u] = pA
        pB_comps[u] = pB
    end

    F = typeof(S.field)
    iA = PMorphism{K,F}(A, S, iA_comps)
    iB = PMorphism{K,F}(B, S, iB_comps)
    pA = PMorphism{K,F}(S, A, pA_comps)
    pB = PMorphism{K,F}(S, B, pB_comps)
    return S, iA, iB, pA, pB
end




@inline function _map_leq_cover_chain(M::PModule{K}, u::Int, v::Int, cc::CoverCache) where {K}
    # Compute M(u<=v) by composing cover-edge maps along the chosen chain.
    # Assumes u < v and u <= v.
    if u == v
        return eye(M.field, M.dims[v])
    end
    @inbounds if cc.C !== nothing && cc.C[u, v]
        return M.edge_maps[u, v]
    end
    if cc.C === nothing && haskey(M.edge_maps, u, v)
        return M.edge_maps[u, v]
    end
    w = _chosen_predecessor(cc, u, v)
    return FieldLinAlg._matmul(M.edge_maps[w, v], _map_leq_cover_chain(M, u, w, cc))
end

"""
    map_leq(M::PModule, u, v; cache=nothing) -> Matrix

Return the structure map `M(u <= v)` for a comparable pair `u <= v`.

The internal `PModule` stores only the maps on *cover* edges of the Hasse
diagram.  This function composes those cover maps along a (poset-dependent)
chosen cover chain.

For a functorial module, the resulting map is independent of the chosen chain;
the chain is only used as a witness that `u <= v`.

Performance notes:
  * If `cache` is omitted, the per-poset lazy `CoverCache` for `M.Q` is used.
  * The chosen cover chain is cached inside the `CoverCache` via parent pointers,
    so repeated calls avoid rescanning predecessor lists.

Warning:
  The returned matrix may alias internal storage when `u < v` is a *cover edge*.
  Treat it as read-only.
"""
function map_leq(M::PModule{K}, u::Int, v::Int;
                 cache::Union{Nothing,CoverCache}=nothing,
                 opts::ModuleOptions=ModuleOptions()) where {K}
    Q = M.Q
    n = nvertices(Q)
    (1 <= u <= n && 1 <= v <= n) || error("map_leq: indices out of range")

    u == v && return eye(M.field, M.dims[v])
    leq(Q, u, v) || error("map_leq: need u <= v in the poset (got u=$u, v=$v)")

    if opts != ModuleOptions()
        cache === nothing || error("map_leq: pass either cache or opts, not both.")
        cache = opts.cache
    end
    cc = cache === nothing ? cover_cache(Q) : cache
    return _map_leq_cover_chain(M, u, v, cc)
end

end
