module Modules

using SparseArrays, LinearAlgebra
using ..CoreModules: QQ, ModuleOptions, AbstractCoeffField, QQField, PrimeField, RealField,
    BackendMatrix, coeff_type, field_from_eltype, eye, coerce
import ..CoreModules: change_field
import ..FiniteFringe
import ..FiniteFringe: AbstractPoset, FinitePoset, cover_edges, leq, nvertices,
       CoverCache, PosetCache, _get_cover_cache,
       _clear_cover_cache!, _pairkey, _chosen_predecessor
import ..FieldLinAlg
import Base.Threads

const MAP_LEQ_MEMO_MAX_PER_THREAD = Ref(200_000)
const MAP_LEQ_MANY_PLAN_MIN_LEN = Ref(128)
const MAP_LEQ_MANY_PLAN_MAX_PER_THREAD = Ref(128)
const DIRECT_SUM_SPARSE_MIN_TOTAL_ENTRIES = Ref(16_384)
const DIRECT_SUM_SPARSE_MAX_DENSITY = Ref(0.55)
const DIRECT_SUM_SPARSE_MAX_DENSITY_PRIME = Ref(0.28)
const DIRECT_SUM_SPARSE_MAX_DENSITY_REAL = Ref(0.20)

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

    cc = cache === nothing ? _get_cover_cache(Q) : cache
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
        slot = cc.pred_slot_of_succ[u]
        for j in eachindex(su)
            v = su[j]
            i = slot[j]
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
        cc = cache === nothing ? _get_cover_cache(Q) : cache
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

mutable struct MapLeqScratch{K}
    chain::Vector{Int}
    tmp1::Matrix{K}
    tmp2::Matrix{K}
end

struct _MapLeqManyPlan
    npairs::Int
    first_pair::Tuple{Int,Int}
    last_pair::Tuple{Int,Int}
    kinds::Vector{UInt8}          # 0: u==v, 1: cover edge, 2: two-hop, 3: long chain
    us::Vector{Int}
    vs::Vector{Int}
    mids::Vector{Int}             # only used for two-hop entries
    chains::Vector{Vector{Int}}   # only populated for kind==3
end

"""
A minimal module over a finite poset `Q`.

A `PModule` is a functor `Q -> Vec_K` specified by:

  * `dims[i] = dim_K M(i)`
  * maps on cover relations `u leq with dot v`

For performance, cover-edge maps are stored in a `CoverEdgeMapStore`,
aligned with the cover graph, rather than in a dictionary keyed by `(u,v)`.
"""
struct PModule{K,F<:AbstractCoeffField,MatT<:AbstractMatrix{K},QT<:AbstractPoset}
    field::F
    Q::QT
    dims::Vector{Int}
    edge_maps::CoverEdgeMapStore{K,MatT}
    map_compose::Vector{Dict{UInt64, MatT}}
    map_many_plan::Vector{Dict{UInt64,_MapLeqManyPlan}}
    map_scratch::Vector{MapLeqScratch{K}}
    identity_compose::Vector{Dict{Int, MatT}}
end

# choose a storage matrix type from a user-provided mapping
@inline _is_nemo_field(field::AbstractCoeffField) =
    field isa QQField || (field isa PrimeField && field.p > 3)

@inline _is_backend_dense_candidate(A::AbstractMatrix) = !(A isa SparseMatrixCSC)

@inline function _should_backendize_map(field::AbstractCoeffField, A::AbstractMatrix)
    _is_nemo_field(field) || return false
    FieldLinAlg._have_nemo() || return false
    _is_backend_dense_candidate(A) || return false
    m, n = size(A)
    return m * n >= FieldLinAlg.NEMO_THRESHOLD[]
end

function _should_backendize_maps(field::AbstractCoeffField, edge_maps)::Bool
    _is_nemo_field(field) || return false
    FieldLinAlg._have_nemo() || return false
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

@inline function _new_map_leq_memo(::Type{MatT}) where {MatT}
    nt = max(1, Threads.maxthreadid())
    return [Dict{UInt64, MatT}() for _ in 1:nt]
end

@inline function _new_map_leq_many_plan_cache()
    nt = max(1, Threads.maxthreadid())
    return [Dict{UInt64,_MapLeqManyPlan}() for _ in 1:nt]
end

@inline function _new_map_leq_scratch(::Type{K}) where {K}
    nt = max(1, Threads.maxthreadid())
    return [MapLeqScratch{K}(Int[], Matrix{K}(undef, 0, 0), Matrix{K}(undef, 0, 0)) for _ in 1:nt]
end

@inline function _new_identity_compose(::Type{MatT}) where {MatT}
    nt = max(1, Threads.maxthreadid())
    return [Dict{Int, MatT}() for _ in 1:nt]
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
    QT = typeof(Q)
    return PModule{K, typeof(field), MatT, QT}(field, Q, dims, store,
                                               _new_map_leq_memo(MatT),
                                               _new_map_leq_many_plan_cache(),
                                               _new_map_leq_scratch(K),
                                               _new_identity_compose(MatT))
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
    cc = _get_cover_cache(Q)
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
        QT = typeof(Q)
        return PModule{K, typeof(field), MatT, QT}(field, Q, dims, store,
                                                   _new_map_leq_memo(MatT),
                                                   _new_map_leq_many_plan_cache(),
                                                   _new_map_leq_scratch(K),
                                                   _new_identity_compose(MatT))
    end
    new_store = CoverEdgeMapStore{K,MatT}(Q, dims, store; cache=cc, check_sizes=check_sizes)
    coeff_type(field) == K || error("PModule: coeff_type(field) != K")
    QT = typeof(Q)
    return PModule{K, typeof(field), MatT, QT}(field, Q, dims, new_store,
                                               _new_map_leq_memo(MatT),
                                               _new_map_leq_many_plan_cache(),
                                               _new_map_leq_scratch(K),
                                               _new_identity_compose(MatT))
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
    if _is_nemo_field(field) && FieldLinAlg._have_nemo()
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

@inline _map_leq_key(u::Int, v::Int)::UInt64 = _pairkey(u, v)

@inline function _thread_slot(n::Int)::Int
    return min(n, max(1, Threads.threadid()))
end

@inline function _map_leq_memo_dict(M::PModule{K,F,MatT}) where {K,F,MatT}
    return M.map_compose[_thread_slot(length(M.map_compose))]
end

@inline function _map_leq_many_plan_dict(M::PModule)
    return M.map_many_plan[_thread_slot(length(M.map_many_plan))]
end

@inline function _map_leq_scratch(M::PModule{K}) where {K}
    return M.map_scratch[_thread_slot(length(M.map_scratch))]
end

@inline function _identity_compose_dict(M::PModule{K,F,MatT}) where {K,F,MatT}
    return M.identity_compose[_thread_slot(length(M.identity_compose))]
end

@inline function _map_leq_memo_get(M::PModule{K,F,MatT}, u::Int, v::Int) where {K,F,MatT}
    memo = _map_leq_memo_dict(M)
    A = get(memo, _map_leq_key(u, v), nothing)
    A === nothing && return nothing
    return A::MatT
end

@inline function _map_leq_memo_set!(M::PModule{K,F,MatT}, u::Int, v::Int, A) where {K,F,MatT}
    memo = _map_leq_memo_dict(M)
    if length(memo) >= MAP_LEQ_MEMO_MAX_PER_THREAD[]
        empty!(memo)
    end
    Atyped = A isa MatT ? A : convert(MatT, A)
    memo[_map_leq_key(u, v)] = Atyped
    return Atyped
end

@inline function _identity_map(M::PModule{K,F,MatT}, v::Int) where {K,F,MatT}
    d = M.dims[v]
    memo = _identity_compose_dict(M)
    A = get(memo, d, nothing)
    if A === nothing
        Aeye = eye(M.field, d)
        Atyped = Aeye isa MatT ? Aeye : convert(MatT, Aeye)
        memo[d] = Atyped
        return Atyped
    end
    return A::MatT
end

@inline _map_leq_many_plan_key(pairs) = UInt(objectid(pairs))

@inline function _is_cover_pair(cc::CoverCache, u::Int, v::Int)::Bool
    if cc.C !== nothing
        @inbounds return cc.C[u, v]
    end
    return _find_sorted_index(cc.succs[u], v) != 0
end

@inline function _plan_signature_matches(plan::_MapLeqManyPlan, pairs)::Bool
    length(pairs) == plan.npairs || return false
    isempty(pairs) && return true
    @inbounds return pairs[firstindex(pairs)] == plan.first_pair &&
                     pairs[lastindex(pairs)] == plan.last_pair
end

function _build_map_leq_many_plan(pairs::AbstractVector{<:Tuple{Int,Int}},
                                  Q::AbstractPoset,
                                  cc::CoverCache):: _MapLeqManyPlan
    m = length(pairs)
    kinds = Vector{UInt8}(undef, m)
    us = Vector{Int}(undef, m)
    vs = Vector{Int}(undef, m)
    mids = zeros(Int, m)
    chains = [Int[] for _ in 1:m]
    n = nvertices(Q)
    tmp_chain = Int[]

    @inbounds for i in eachindex(pairs)
        u, v = pairs[i]
        (1 <= u <= n && 1 <= v <= n) || error("map_leq_many!: indices out of range at i=$i (u=$u, v=$v)")
        us[i] = u
        vs[i] = v

        if u == v
            kinds[i] = 0x00
            continue
        end
        leq(Q, u, v) || error("map_leq_many!: need u <= v at i=$i (u=$u, v=$v)")

        if _is_cover_pair(cc, u, v)
            kinds[i] = 0x01
            continue
        end

        p = _chosen_predecessor(cc, u, v)
        if p == u
            kinds[i] = 0x01
            continue
        end

        p2 = _chosen_predecessor(cc, u, p)
        if p2 == u
            kinds[i] = 0x02
            mids[i] = p
            continue
        end

        kinds[i] = 0x03
        _build_cover_chain!(tmp_chain, u, v, cc)
        chains[i] = copy(tmp_chain)
    end

    first_pair = m == 0 ? (0, 0) : pairs[firstindex(pairs)]
    last_pair = m == 0 ? (0, 0) : pairs[lastindex(pairs)]
    return _MapLeqManyPlan(m, first_pair, last_pair, kinds, us, vs, mids, chains)
end

function _get_or_build_map_leq_many_plan!(M::PModule,
                                          pairs::AbstractVector{<:Tuple{Int,Int}},
                                          Q::AbstractPoset,
                                          cc::CoverCache)
    length(pairs) >= MAP_LEQ_MANY_PLAN_MIN_LEN[] || return nothing
    cache = _map_leq_many_plan_dict(M)
    k = _map_leq_many_plan_key(pairs)
    plan = get(cache, k, nothing)
    if plan !== nothing && _plan_signature_matches(plan, pairs)
        return plan
    end

    new_plan = _build_map_leq_many_plan(pairs, Q, cc)
    if length(cache) >= MAP_LEQ_MANY_PLAN_MAX_PER_THREAD[]
        empty!(cache)
    end
    cache[k] = new_plan
    return new_plan
end

function _map_leq_many_with_plan!(dest::AbstractVector,
                                  M::PModule{K,F,MatT},
                                  pairs::AbstractVector{<:Tuple{Int,Int}},
                                  plan::_MapLeqManyPlan) where {K,F,MatT}
    _plan_signature_matches(plan, pairs) || return false

    @inbounds for i in eachindex(pairs)
        kind = plan.kinds[i]
        u = plan.us[i]
        v = plan.vs[i]

        if kind == 0x00
            dest[i] = _identity_map(M, v)
            continue
        elseif kind == 0x01
            dest[i] = M.edge_maps[u, v]
            continue
        end

        if M.dims[u] == 1 && M.dims[v] == 1
            coeff = one(K)
            if kind == 0x02
                p = plan.mids[i]
                coeff *= M.edge_maps[p, v][1, 1]
                coeff *= M.edge_maps[u, p][1, 1]
            else
                chain = plan.chains[i]
                @inbounds for t in 1:(length(chain) - 1)
                    coeff *= M.edge_maps[chain[t], chain[t + 1]][1, 1]
                end
            end
            A = Matrix{K}(undef, 1, 1)
            A[1, 1] = coeff
            dest[i] = _as_mattype(MatT, A)
            continue
        end

        memoA = _map_leq_memo_get(M, u, v)
        if memoA !== nothing
            dest[i] = memoA
            continue
        end

        A = if kind == 0x02
            p = plan.mids[i]
            if MatT <: Matrix{K}
                s = _map_leq_scratch(M)
                E2 = M.edge_maps[p, v]
                E1 = M.edge_maps[u, p]
                out = _scratch_mat!(s, true, size(E2, 1), size(E1, 2))
                mul!(out, E2, E1)
                copy(out)
            else
                FieldLinAlg._matmul(M.edge_maps[p, v], M.edge_maps[u, p])
            end
        else
            chain = plan.chains[i]
            s = _map_leq_scratch(M)
            MatT <: Matrix{K} ? _compose_chain_dense!(M, chain, s) : _compose_chain_generic(M, chain)
        end

        dest[i] = _map_leq_memo_set!(M, u, v, A)
    end
    return true
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

@inline _is_sparse_map_type(::Type{MatT}) where {MatT} = MatT <: SparseMatrixCSC

@inline function _direct_sum_sparse_density_cap(field::AbstractCoeffField)
    if field isa RealField
        return DIRECT_SUM_SPARSE_MAX_DENSITY_REAL[]
    elseif field isa PrimeField
        return DIRECT_SUM_SPARSE_MAX_DENSITY_PRIME[]
    end
    return DIRECT_SUM_SPARSE_MAX_DENSITY[]
end

function _direct_sum_sparse_preferred(A::PModule{K,FA,MatA},
                                      B::PModule{K,FB,MatB}) where {K,FA,FB,MatA,MatB}
    # Sparse output only pays off reliably when both inputs are already sparse.
    (_is_sparse_map_type(MatA) && _is_sparse_map_type(MatB)) || return false
    total_entries = 0
    total_nnz = 0
    for ((u, v), Au) in A.edge_maps
        Bu = B.edge_maps[u, v]
        total_entries += (A.dims[v] + B.dims[v]) * (A.dims[u] + B.dims[u])
        total_nnz += nnz(Au) + nnz(Bu)
    end
    total_entries == 0 && return false
    total_entries < DIRECT_SUM_SPARSE_MIN_TOTAL_ENTRIES[] && return false
    return total_nnz <= _direct_sum_sparse_density_cap(A.field) * total_entries
end

@inline function _direct_sum_mat_type(::Type{K},
                                      A::PModule{K,FA,MatA},
                                      B::PModule{K,FB,MatB}) where {K,FA,FB,MatA,MatB}
    return _direct_sum_sparse_preferred(A, B) ? SparseMatrixCSC{K,Int} : Matrix{K}
end

@inline _col_nnz(A::SparseMatrixCSC, j::Int) = @inbounds A.colptr[j + 1] - A.colptr[j]
@inline function _col_nnz(A::AbstractMatrix, j::Int)
    c = 0
    @inbounds for i in 1:size(A, 1)
        iszero(A[i, j]) || (c += 1)
    end
    return c
end

@inline function _write_col_nonzeros!(rowval::Vector{Int},
                                      nzval::Vector{K},
                                      ptr::Int,
                                      A::SparseMatrixCSC{K,Int},
                                      j::Int,
                                      row_shift::Int) where {K}
    @inbounds for idx in A.colptr[j]:(A.colptr[j + 1] - 1)
        rowval[ptr] = A.rowval[idx] + row_shift
        nzval[ptr] = A.nzval[idx]
        ptr += 1
    end
    return ptr
end

@inline function _write_col_nonzeros!(rowval::Vector{Int},
                                      nzval::Vector{K},
                                      ptr::Int,
                                      A::AbstractMatrix{K},
                                      j::Int,
                                      row_shift::Int) where {K}
    @inbounds for i in 1:size(A, 1)
        v = A[i, j]
        iszero(v) && continue
        rowval[ptr] = i + row_shift
        nzval[ptr] = v
        ptr += 1
    end
    return ptr
end

@inline function _direct_sum_edge_map(::Type{Matrix{K}},
                                      Au::AbstractMatrix{K},
                                      Bu::AbstractMatrix{K},
                                      aU::Int, bU::Int, aV::Int, bV::Int) where {K}
    Muv = zeros(K, aV + bV, aU + bU)
    if aV != 0 && aU != 0
        copyto!(view(Muv, 1:aV, 1:aU), Au)
    end
    if bV != 0 && bU != 0
        copyto!(view(Muv, aV+1:aV+bV, aU+1:aU+bU), Bu)
    end
    return Muv
end

@inline function _direct_sum_edge_map(::Type{SparseMatrixCSC{K,Int}},
                                      Au::AbstractMatrix{K},
                                      Bu::AbstractMatrix{K},
                                      aU::Int, bU::Int, aV::Int, bV::Int) where {K}
    out_m = aV + bV
    out_n = aU + bU
    out_n == 0 && return spzeros(K, out_m, out_n)

    colptr = Vector{Int}(undef, out_n + 1)
    colptr[1] = 1
    nextptr = 1

    @inbounds for j in 1:aU
        nextptr += _col_nnz(Au, j)
        colptr[j + 1] = nextptr
    end
    @inbounds for j in 1:bU
        nextptr += _col_nnz(Bu, j)
        colptr[aU + j + 1] = nextptr
    end

    nnz_total = nextptr - 1
    rowval = Vector{Int}(undef, nnz_total)
    nzval = Vector{K}(undef, nnz_total)

    @inbounds for j in 1:aU
        ptr = colptr[j]
        _write_col_nonzeros!(rowval, nzval, ptr, Au, j, 0)
    end
    @inbounds for j in 1:bU
        ptr = colptr[aU + j]
        _write_col_nonzeros!(rowval, nzval, ptr, Bu, j, aV)
    end

    return SparseMatrixCSC(out_m, out_n, colptr, rowval, nzval)
end

"""
    direct_sum(A::PModule{K}, B::PModule{K}) -> PModule{K}

Binary direct sum A oplus B as a P-module.
"""
function direct_sum(A::PModule{K,FA,MatA}, B::PModule{K,FB,MatB}) where {K,FA,FB,MatA,MatB}
    Q = A.Q
    n = nvertices(Q)
    @assert B.Q === Q
    A.field == B.field || error("direct_sum: field mismatch")

    dims = [A.dims[i] + B.dims[i] for i in 1:n]
    OutMatT = _direct_sum_mat_type(K, A, B)

    # Fast path: traverse cover edges via store-aligned succ lists and grab maps
    # by index (no tuple allocation, no search).
    cc = _get_cover_cache(Q)
    if (A.edge_maps.succs == cc.succs && A.edge_maps.preds == cc.preds &&
        B.edge_maps.succs == cc.succs && B.edge_maps.preds == cc.preds)

        preds = cc.preds
        succs = cc.succs

        maps_from_pred = [Vector{OutMatT}(undef, length(preds[v])) for v in 1:n]
        maps_to_succ   = [Vector{OutMatT}(undef, length(succs[u])) for u in 1:n]

        @inbounds for u in 1:n
            su = succs[u]
            pred_slots = cc.pred_slot_of_succ[u]
            Au = A.edge_maps.maps_to_succ[u]
            Bu = B.edge_maps.maps_to_succ[u]
            outu = maps_to_succ[u]

            aU, bU = A.dims[u], B.dims[u]

            for j in eachindex(su)
                v = su[j]
                aV, bV = A.dims[v], B.dims[v]

                Muv = _direct_sum_edge_map(OutMatT, Au[j], Bu[j], aU, bU, aV, bV)

                outu[j] = Muv
                ip = pred_slots[j]
                @inbounds maps_from_pred[v][ip] = Muv
            end
        end

        store = CoverEdgeMapStore{K,OutMatT}(preds, succs, maps_from_pred, maps_to_succ, cc.nedges)
        return PModule{K}(Q, dims, store; field=A.field)
    end

    # Fallback (rare): use keyed access.
    edge = Dict{Tuple{Int,Int}, OutMatT}()
    sizehint!(edge, length(A.edge_maps))
    for (u, v) in cover_edges(Q)
        Au = A.edge_maps[u, v]
        Bu = B.edge_maps[u, v]
        aU, bU = A.dims[u], B.dims[u]
        aV, bV = A.dims[v], B.dims[v]
        edge[(u, v)] = _direct_sum_edge_map(OutMatT, Au, Bu, aU, bU, aV, bV)
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




@inline function _scratch_mat!(s::MapLeqScratch{K}, which::Bool, m::Int, n::Int)::Matrix{K} where {K}
    if which
        if size(s.tmp1, 1) != m || size(s.tmp1, 2) != n
            s.tmp1 = Matrix{K}(undef, m, n)
        end
        return s.tmp1
    else
        if size(s.tmp2, 1) != m || size(s.tmp2, 2) != n
            s.tmp2 = Matrix{K}(undef, m, n)
        end
        return s.tmp2
    end
end

@inline function _build_cover_chain!(chain::Vector{Int}, u::Int, v::Int, cc::CoverCache)
    empty!(chain)
    push!(chain, v)
    d = v
    while d != u
        d = _chosen_predecessor(cc, u, d)
        push!(chain, d)
    end
    reverse!(chain)
    return chain
end

@inline function _compose_chain_dense!(M::PModule{K}, chain::Vector{Int}, s::MapLeqScratch{K}) where {K}
    m = length(chain)
    first_map = M.edge_maps[chain[1], chain[2]]
    acc = _scratch_mat!(s, true, size(first_map, 1), size(first_map, 2))
    copyto!(acc, first_map)
    acc_is_tmp1 = true

    @inbounds for i in 2:(m - 1)
        E = M.edge_maps[chain[i], chain[i + 1]]
        out = _scratch_mat!(s, !acc_is_tmp1, size(E, 1), size(acc, 2))
        mul!(out, E, acc)
        acc = out
        acc_is_tmp1 = !acc_is_tmp1
    end
    return copy(acc)
end

@inline function _compose_chain_generic(M::PModule, chain::Vector{Int})
    m = length(chain)
    A = M.edge_maps[chain[1], chain[2]]
    @inbounds for i in 2:(m - 1)
        E = M.edge_maps[chain[i], chain[i + 1]]
        A = FieldLinAlg._matmul(E, A)
    end
    return A
end

@inline _as_mattype(::Type{MatT}, A) where {MatT} = A isa MatT ? A : convert(MatT, A)

@inline function _map_leq_short_chain_no_memo(M::PModule{K,F,MatT}, u::Int, v::Int, cc::CoverCache) where {K,F,MatT}
    # Try a no-memo fast path when the chosen chain has length <= 2 edges.
    p = _chosen_predecessor(cc, u, v)
    if p == u
        return M.edge_maps[u, v]
    end
    p2 = _chosen_predecessor(cc, u, p)
    p2 == u || return nothing
    if MatT <: Matrix{K}
        s = _map_leq_scratch(M)
        E2 = M.edge_maps[p, v]
        E1 = M.edge_maps[u, p]
        out = _scratch_mat!(s, true, size(E2, 1), size(E1, 2))
        mul!(out, E2, E1)
        A = copy(out)
    else
        A = FieldLinAlg._matmul(M.edge_maps[p, v], M.edge_maps[u, p])
    end
    return _as_mattype(MatT, A)
end

@inline function _map_leq_scalar_chain_no_memo(M::PModule{K,F,MatT}, u::Int, v::Int, cc::CoverCache) where {K,F,MatT}
    # For 1x1 fibers, avoid matrix multiplications and compose scalars directly.
    (M.dims[u] == 1 && M.dims[v] == 1) || return nothing
    d = v
    coeff = one(K)
    while d != u
        p = _chosen_predecessor(cc, u, d)
        (M.dims[p] == 1 && M.dims[d] == 1) || return nothing
        coeff *= M.edge_maps[p, d][1, 1]
        d = p
    end
    A = Matrix{K}(undef, 1, 1)
    A[1, 1] = coeff
    return _as_mattype(MatT, A)
end

@inline function _map_leq_cover_chain(M::PModule{K,F,MatT}, u::Int, v::Int, cc::CoverCache) where {K,F,MatT}
    # Compute M(u<=v) by composing cover-edge maps along a cached cover chain.
    # Assumes u < v and u <= v.
    @inbounds if cc.C !== nothing && cc.C[u, v]
        return M.edge_maps[u, v]
    end
    if cc.C === nothing && haskey(M.edge_maps, u, v)
        return M.edge_maps[u, v]
    end
    memoA = _map_leq_memo_get(M, u, v)
    memoA === nothing || return memoA

    # Fast path: skip memo/scratch machinery for tiny queries.
    # - scalar-chain composition for 1x1 fibers
    # - short chosen chains (<=2 edges)
    Afast = _map_leq_scalar_chain_no_memo(M, u, v, cc)
    Afast === nothing || return Afast
    Afast = _map_leq_short_chain_no_memo(M, u, v, cc)
    Afast === nothing || return Afast

    s = _map_leq_scratch(M)
    chain = _build_cover_chain!(s.chain, u, v, cc)
    A = MatT <: Matrix{K} ? _compose_chain_dense!(M, chain, s) : _compose_chain_generic(M, chain)
    return _map_leq_memo_set!(M, u, v, A)
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
  The returned matrix may alias internal storage or memoized composed maps.
  Treat it as read-only.
"""
function map_leq(M::PModule{K}, u::Int, v::Int;
                 cache::Union{Nothing,CoverCache}=nothing,
                 opts::ModuleOptions=ModuleOptions()) where {K}
    Q = M.Q
    n = nvertices(Q)
    (1 <= u <= n && 1 <= v <= n) || error("map_leq: indices out of range")

    u == v && return _identity_map(M, v)
    leq(Q, u, v) || error("map_leq: need u <= v in the poset (got u=$u, v=$v)")

    if opts != ModuleOptions()
        cache === nothing || error("map_leq: pass either cache or opts, not both.")
        cache = opts.cache
    end
    cc = cache === nothing ? _get_cover_cache(Q) : cache
    return _map_leq_cover_chain(M, u, v, cc)
end

"""
    map_leq_many!(dest, M, pairs; cache=nothing, opts=ModuleOptions())

Batch form of `map_leq` for comparable pairs `(u,v)`.
Writes one map per pair into `dest` and returns `dest`.
"""
function map_leq_many!(dest::AbstractVector,
                       M::PModule{K,F,MatT},
                       pairs::AbstractVector{<:Tuple{Int,Int}};
                       cache::Union{Nothing,CoverCache}=nothing,
                       opts::ModuleOptions=ModuleOptions()) where {K,F,MatT}
    if opts != ModuleOptions()
        cache === nothing || error("map_leq_many!: pass either cache or opts, not both.")
        cache = opts.cache
    end
    length(dest) == length(pairs) ||
        error("map_leq_many!: destination length $(length(dest)) does not match pair count $(length(pairs)).")

    Q = M.Q
    n = nvertices(Q)
    cc = cache === nothing ? _get_cover_cache(Q) : cache
    plan = _get_or_build_map_leq_many_plan!(M, pairs, Q, cc)
    if plan !== nothing && _map_leq_many_with_plan!(dest, M, pairs, plan)
        return dest
    end
    @inbounds for i in eachindex(pairs)
        u, v = pairs[i]
        (1 <= u <= n && 1 <= v <= n) || error("map_leq_many!: indices out of range at i=$i (u=$u, v=$v)")
        if u == v
            dest[i] = _identity_map(M, v)
        else
            leq(Q, u, v) || error("map_leq_many!: need u <= v at i=$i (u=$u, v=$v)")
            dest[i] = _map_leq_cover_chain(M, u, v, cc)
        end
    end
    return dest
end

"""
    map_leq_many(M, pairs; cache=nothing, opts=ModuleOptions()) -> Vector

Allocate and return a vector of maps `M(u<=v)` for each pair `(u,v)`.
"""
function map_leq_many(M::PModule{K,F,MatT},
                      pairs::AbstractVector{<:Tuple{Int,Int}};
                      cache::Union{Nothing,CoverCache}=nothing,
                      opts::ModuleOptions=ModuleOptions()) where {K,F,MatT}
    out = Vector{MatT}(undef, length(pairs))
    return map_leq_many!(out, M, pairs; cache=cache, opts=opts)
end

end
