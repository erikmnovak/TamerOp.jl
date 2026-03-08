module IndicatorResolutions

# Indicator resolutions: build (F,dF) and (E,dE) data (upset/downset) and provide
# utilities for converting between presentations and PModule objects.
#
# Responsibilities:
#   - turn a finite fringe module into indicator presentations (upset/downset)
#   - build longer indicator resolutions (F,dF) and (E,dE)
#   - verification routines and small caching helpers
#
# Design notes:
#   - This module does not compute Ext itself. It supplies resolutions and minimal
#     presentation helpers consumed by `DerivedFunctors.HomExtEngine`.
#   - Derived-functor drivers that interpret resolutions (e.g. first-page diagnostics
#     and Ext dimension computations) live in `DerivedFunctors.ExtTorSpaces`.

using SparseArrays, LinearAlgebra
using ..FiniteFringe
using ..IndicatorTypes: UpsetPresentation, DownsetCopresentation
using ..CoreModules: AbstractCoeffField, RealField, ResolutionCache, ResolutionKey3, _resolution_key3,
                     IndicatorResolutionPayload, coeff_type, eye, field_from_eltype
using ..FieldLinAlg

using ..Modules: CoverCache, _get_cover_cache, _clear_cover_cache!,
                 CoverEdgeMapStore, _find_sorted_index,
                 PModule, PMorphism, dim_at,
                 zero_pmodule, zero_morphism, map_leq, map_leq_many, map_leq_many!,
                 MapLeqQueryBatch, prepare_map_leq_batch, id_morphism
import ..Modules: id_morphism

import ..AbelianCategories
using ..AbelianCategories: kernel_with_inclusion, _cokernel_module

import ..FiniteFringe: AbstractPoset, FinitePoset, Upset, Downset,
                       principal_upset, principal_downset,
                       cover_edges, nvertices, leq, _succs, _preds,
                       _pred_slots_of_succ,
                       upset_indices, downset_indices
import ..FiniteFringe: build_cache!
import Base.Threads

const INDICATOR_MAP_MEMO_THRESHOLD = Ref(1_000_000)
const INDICATOR_MAP_BATCH_THRESHOLD = Ref(4)
const INDICATOR_THREADS_MIN_VERTICES = Ref(24)
const INDICATOR_THREADS_MIN_TOTAL_DIMS = Ref(192)
const INDICATOR_THREADS_MIN_WORK = Ref(12_000)
const INDICATOR_PREFIX_CACHE_ENABLED = Ref(true)
const INDICATOR_INCREMENTAL_LINALG = Ref(true)
const INDICATOR_MAP_PLAN_MIN_VERTICES = Ref(32)
const INDICATOR_MAP_PLAN_MIN_PAIRS = Ref(256)
const INDICATOR_PREFIX_CACHE_SHARDS = Ref(8)
const INDICATOR_BIRTH_PLAN_CACHE_SHARDS = Ref(8)
const INDICATOR_PMODULE_DIRECT_STORE_MIN_EDGES = Ref(128)
const INDICATOR_PMODULE_DIRECT_STORE_MIN_WORK = Ref(8_192)
const INDICATOR_INCREMENTAL_LINALG_MIN_MAPS = Ref(4)
const INDICATOR_INCREMENTAL_LINALG_MIN_ENTRIES = Ref(4_096)
const INDICATOR_INCREMENTAL_VERTEX_CACHE = Ref(true)
const INDICATOR_INCREMENTAL_VERTEX_CACHE_MIN_VERTICES = Ref(96)
const INDICATOR_INCREMENTAL_VERTEX_CACHE_MIN_TOTAL_DIMS = Ref(256)

const _IncrementalVertexCacheEntry{K} = AbelianCategories._VertexIncrementalCacheEntry{K}

struct _BirthPlans
    upset::Vector{Vector{Int}}
    downset::Vector{Vector{Int}}
end

const _INDICATOR_PREFIX_FAMILY_LOCK = Base.ReentrantLock()
const _INDICATOR_PREFIX_FAMILIES = IdDict{DataType,Any}()
const _INDICATOR_BIRTH_PLAN_LOCKS = [Base.ReentrantLock() for _ in 1:INDICATOR_BIRTH_PLAN_CACHE_SHARDS[]]
const _INDICATOR_BIRTH_PLAN_SHARDS = [IdDict{AbstractPoset,_BirthPlans}() for _ in 1:INDICATOR_BIRTH_PLAN_CACHE_SHARDS[]]

@inline _indicator_shard_index(x, nshards::Int) = Int(mod(UInt(objectid(x)), UInt(nshards))) + 1
@inline _indicator_use_direct_pmodule_store(nedges::Int, total_work::Int) =
    nedges >= INDICATOR_PMODULE_DIRECT_STORE_MIN_EDGES[] &&
    total_work >= INDICATOR_PMODULE_DIRECT_STORE_MIN_WORK[]
@inline _indicator_use_incremental_union(nrows::Int, total_cols::Int, nmats::Int) =
    INDICATOR_INCREMENTAL_LINALG[] &&
    nmats >= INDICATOR_INCREMENTAL_LINALG_MIN_MAPS[] &&
    nrows * total_cols >= INDICATOR_INCREMENTAL_LINALG_MIN_ENTRIES[]
@inline _indicator_use_incremental_vertex_cache(n::Int, total_dims::Int) =
    INDICATOR_INCREMENTAL_VERTEX_CACHE[] &&
    n >= INDICATOR_INCREMENTAL_VERTEX_CACHE_MIN_VERTICES[] &&
    total_dims >= INDICATOR_INCREMENTAL_VERTEX_CACHE_MIN_TOTAL_DIMS[]

@inline _indicator_use_array_memo(n::Int) = n * n <= INDICATOR_MAP_MEMO_THRESHOLD[]
@inline _indicator_memo_index(n::Int, u::Int, v::Int) = (u - 1) * n + v

@inline function _indicator_new_array_memo(::Type{K}, n::Int) where {K}
    memo = Vector{Union{Nothing,Matrix{K}}}(undef, n * n)
    fill!(memo, nothing)
    return memo
end

@inline function _indicator_memo_get(memo::AbstractVector{Union{Nothing,Matrix{K}}}, n::Int, u::Int, v::Int) where {K}
    return memo[_indicator_memo_index(n, u, v)]
end

@inline function _indicator_memo_set!(memo::AbstractVector{Union{Nothing,Matrix{K}}}, n::Int, u::Int, v::Int, val::Matrix{K}) where {K}
    memo[_indicator_memo_index(n, u, v)] = val
    return val
end

@inline function _indicator_memo_for_module!(
    pool::IdDict{Any,Vector{Union{Nothing,Matrix{K}}}},
    M::PModule{K},
) where {K}
    return get!(pool, M) do
        _indicator_new_array_memo(K, nvertices(M.Q))
    end
end

@inline function _map_leq_cached_many_indicator(
    M::PModule{K},
    pairs::Vector{Tuple{Int,Int}},
    cc::CoverCache,
    memo::AbstractVector{Union{Nothing,Matrix{K}}},
) where {K}
    out = Vector{Matrix{K}}(undef, length(pairs))
    missing_pairs = Tuple{Int,Int}[]
    missing_idx = Int[]
    n = nvertices(M.Q)

    @inbounds for i in eachindex(pairs)
        u, v = pairs[i]
        X = _indicator_memo_get(memo, n, u, v)
        if X === nothing
            push!(missing_pairs, (u, v))
            push!(missing_idx, i)
        else
            out[i] = X
        end
    end

    if !isempty(missing_pairs)
        if length(missing_pairs) <= INDICATOR_MAP_BATCH_THRESHOLD[]
            @inbounds for t in eachindex(missing_idx)
                i = missing_idx[t]
                u, v = missing_pairs[t]
                Xraw = map_leq(M, u, v; cache=cc)
                Xmat = Xraw isa Matrix{K} ? Xraw : Matrix{K}(Xraw)
                out[i] = _indicator_memo_set!(memo, n, u, v, Xmat)
            end
        else
            fetched = Vector{AbstractMatrix{K}}(undef, length(missing_pairs))
            batch = prepare_map_leq_batch(missing_pairs)
            map_leq_many!(fetched, M, batch; cache=cc)
            @inbounds for t in eachindex(missing_idx)
                i = missing_idx[t]
                u, v = missing_pairs[t]
                Xraw = fetched[t]
                Xmat = Xraw isa Matrix{K} ? Xraw : Matrix{K}(Xraw)
                out[i] = _indicator_memo_set!(memo, n, u, v, Xmat)
            end
        end
    end

    return out
end

@inline function _map_leq_cached_many_indicator(
    M::PModule{K},
    batch::MapLeqQueryBatch,
    cc::CoverCache,
    memo::AbstractVector{Union{Nothing,Matrix{K}}},
) where {K}
    pairs = batch.pairs
    out = Vector{Matrix{K}}(undef, length(pairs))
    missing_pairs = Tuple{Int,Int}[]
    missing_idx = Int[]
    n = nvertices(M.Q)

    @inbounds for i in eachindex(pairs)
        u, v = pairs[i]
        X = _indicator_memo_get(memo, n, u, v)
        if X === nothing
            push!(missing_pairs, (u, v))
            push!(missing_idx, i)
        else
            out[i] = X
        end
    end

    if !isempty(missing_pairs)
        if length(missing_pairs) <= INDICATOR_MAP_BATCH_THRESHOLD[]
            @inbounds for t in eachindex(missing_idx)
                i = missing_idx[t]
                u, v = missing_pairs[t]
                Xraw = map_leq(M, u, v; cache=cc)
                Xmat = Xraw isa Matrix{K} ? Xraw : Matrix{K}(Xraw)
                out[i] = _indicator_memo_set!(memo, n, u, v, Xmat)
            end
        else
            fetched = Vector{AbstractMatrix{K}}(undef, length(missing_pairs))
            miss_batch = prepare_map_leq_batch(missing_pairs)
            map_leq_many!(fetched, M, miss_batch; cache=cc)
            @inbounds for t in eachindex(missing_idx)
                i = missing_idx[t]
                u, v = missing_pairs[t]
                Xraw = fetched[t]
                Xmat = Xraw isa Matrix{K} ? Xraw : Matrix{K}(Xraw)
                out[i] = _indicator_memo_set!(memo, n, u, v, Xmat)
            end
        end
    end

    return out
end

@inline function _map_leq_fill_memo_indicator!(
    M::PModule{K},
    pairs::Vector{Tuple{Int,Int}},
    cc::CoverCache,
    memo::AbstractVector{Union{Nothing,Matrix{K}}},
    ws,
) where {K}
    n = nvertices(M.Q)
    empty!(ws.missing_pairs_buf)

    @inbounds for i in eachindex(pairs)
        u, v = pairs[i]
        _indicator_memo_get(memo, n, u, v) === nothing || continue
        push!(ws.missing_pairs_buf, (u, v))
    end

    isempty(ws.missing_pairs_buf) && return nothing

    if length(ws.missing_pairs_buf) <= INDICATOR_MAP_BATCH_THRESHOLD[]
        @inbounds for i in eachindex(ws.missing_pairs_buf)
            u, v = ws.missing_pairs_buf[i]
            Xraw = map_leq(M, u, v; cache=cc)
            Xmat = Xraw isa Matrix{K} ? Xraw : Matrix{K}(Xraw)
            _indicator_memo_set!(memo, n, u, v, Xmat)
        end
        return nothing
    end

    resize!(ws.fetched_buf, length(ws.missing_pairs_buf))
    batch = _workspace_get_or_prepare_batch!(ws, ws.missing_pairs_buf)
    map_leq_many!(ws.fetched_buf, M, batch; cache=cc)
    @inbounds for i in eachindex(ws.missing_pairs_buf)
        u, v = ws.missing_pairs_buf[i]
        Xraw = ws.fetched_buf[i]
        Xmat = Xraw isa Matrix{K} ? Xraw : Matrix{K}(Xraw)
        _indicator_memo_set!(memo, n, u, v, Xmat)
    end
    return nothing
end

@inline function _pairs_signature(pairs::Vector{Tuple{Int,Int}})
    h = hash(length(pairs))
    @inbounds for p in pairs
        h = hash(p, h)
    end
    return UInt64(h)
end

@inline function _pairs_equal(a::Vector{Tuple{Int,Int}}, b::Vector{Tuple{Int,Int}})
    length(a) == length(b) || return false
    @inbounds for i in eachindex(a)
        a[i] == b[i] || return false
    end
    return true
end

@inline function _workspace_get_or_prepare_batch!(ws, pairs::Vector{Tuple{Int,Int}})
    key = _pairs_signature(pairs)
    batch = get(ws.map_batch_cache, key, nothing)
    if batch !== nothing && _pairs_equal(batch.pairs, pairs)
        return batch
    end
    b = prepare_map_leq_batch(pairs)
    ws.map_batch_cache[key] = b
    return b
end

@inline function _map_leq_fill_memo_indicator!(
    M::PModule{K},
    batch::MapLeqQueryBatch,
    cc::CoverCache,
    memo::AbstractVector{Union{Nothing,Matrix{K}}},
    ws,
) where {K}
    return _map_leq_fill_memo_indicator!(M, batch.pairs, cc, memo, ws)
end

@inline function _sparse_from_workspace!(
    ws,
    m::Int,
    n::Int,
)
    nnz_trip = length(ws.I)
    resize!(ws.klasttouch, n)
    resize!(ws.csrrowptr, m + 1)
    resize!(ws.csrcolval, nnz_trip)
    resize!(ws.csrnzval, nnz_trip)

    # Ping-pong owned CSC buffers. Each call hands a slot's arrays to the
    # returned matrix and rotates to the other slot for the next call.
    slot = ws.csc_slot
    colptr = ws.csccolptr_slots[slot]
    rowval = ws.cscrowval_slots[slot]
    nzval = ws.cscnzval_slots[slot]
    resize!(colptr, n + 1)
    resize!(rowval, nnz_trip)
    resize!(nzval, nnz_trip)

    A = SparseArrays.sparse!(
        ws.I,
        ws.J,
        ws.V,
        m,
        n,
        +,
        ws.klasttouch,
        ws.csrrowptr,
        ws.csrcolval,
        ws.csrnzval,
        colptr,
        rowval,
        nzval,
    )

    # Transfer ownership of CSC arrays to `A`; install fresh arrays in the slot.
    ws.csccolptr_slots[slot] = Int[]
    ws.cscrowval_slots[slot] = Int[]
    ws.cscnzval_slots[slot] = Vector{eltype(ws.V)}()
    ws.csc_slot = slot == 1 ? 2 : 1

    return A
end

@inline function _accumulate_product_entries_upset!(
    I::Vector{Int},
    J::Vector{Int},
    V::Vector{K},
    A::AbstractMatrix{K},
    B::AbstractMatrix{K},
    row0::Int,
    col0::Int,
    nrows::Int,
    ncols::Int,
    theta_gid0::Int,
    lambda_gid0::Int,
) where {K}
    kdim = size(A, 2)
    @inbounds for rr in 0:(nrows - 1)
        row = row0 + rr
        for cc in 0:(ncols - 1)
            col = col0 + cc
            s = zero(K)
            for k in 1:kdim
                s += A[row, k] * B[k, col]
            end
            if s != 0
                push!(I, theta_gid0 + cc)
                push!(J, lambda_gid0 + rr)
                push!(V, s)
            end
        end
    end
    return nothing
end

@inline function _accumulate_product_entries_downset!(
    I::Vector{Int},
    J::Vector{Int},
    V::Vector{K},
    A::AbstractMatrix{K},
    B::AbstractMatrix{K},
    row0::Int,
    col0::Int,
    nrows::Int,
    ncols::Int,
    theta_gid0::Int,
    lambda_gid0::Int,
) where {K}
    kdim = size(A, 2)
    @inbounds for rr in 0:(nrows - 1)
        row = row0 + rr
        for cc in 0:(ncols - 1)
            col = col0 + cc
            s = zero(K)
            for k in 1:kdim
                s += A[row, k] * B[k, col]
            end
            if s != 0
                push!(I, theta_gid0 + rr)
                push!(J, lambda_gid0 + cc)
                push!(V, s)
            end
        end
    end
    return nothing
end

@inline function _generator_id_ranges(counts::Vector{Int})
    ranges = Vector{UnitRange{Int}}(undef, length(counts))
    off = 0
    @inbounds for i in eachindex(counts)
        c = counts[i]
        if c > 0
            ranges[i] = (off + 1):(off + c)
            off += c
        else
            ranges[i] = 1:0
        end
    end
    return ranges, off
end

@inline function _subsequence_identity_sparse(
    ::Type{K},
    tgt_ids::Vector{Int},
    src_ids::Vector{Int},
) where {K}
    nt = length(tgt_ids)
    ns = length(src_ids)
    I = Int[]
    J = Int[]
    V = K[]
    sizehint!(I, min(nt, ns))
    sizehint!(J, min(nt, ns))
    sizehint!(V, min(nt, ns))

    i = 1
    j = 1
    oneK = one(K)
    @inbounds while i <= nt && j <= ns
        ti = tgt_ids[i]
        sj = src_ids[j]
        if ti == sj
            push!(I, i)
            push!(J, j)
            push!(V, oneK)
            i += 1
            j += 1
        elseif ti < sj
            i += 1
        else
            error("_subsequence_identity_sparse: source generator id missing from target active set")
        end
    end
    j > ns || error("_subsequence_identity_sparse: source generator id missing from target active set")
    return sparse(I, J, V, nt, ns)
end

@inline function _projection_identity_sparse(
    ::Type{K},
    tgt_ids::Vector{Int},
    src_ids::Vector{Int},
) where {K}
    nt = length(tgt_ids)
    ns = length(src_ids)
    I = Int[]
    J = Int[]
    V = K[]
    sizehint!(I, min(nt, ns))
    sizehint!(J, min(nt, ns))
    sizehint!(V, min(nt, ns))

    i = 1
    j = 1
    oneK = one(K)
    @inbounds while i <= nt && j <= ns
        ti = tgt_ids[i]
        sj = src_ids[j]
        if ti == sj
            push!(I, i)
            push!(J, j)
            push!(V, oneK)
            i += 1
            j += 1
        elseif sj < ti
            j += 1
        else
            error("_projection_identity_sparse: target generator id missing from source active set")
        end
    end
    i > nt || error("_projection_identity_sparse: target generator id missing from source active set")
    return sparse(I, J, V, nt, ns)
end

mutable struct _ResolutionWorkspace{K}
    I_chunks::Vector{Vector{Int}}
    J_chunks::Vector{Vector{Int}}
    V_chunks::Vector{Vector{K}}
    I::Vector{Int}
    J::Vector{Int}
    V::Vector{K}
    self_start::Vector{Int}
    # SparseArrays.sparse! scratch storage.
    klasttouch::Vector{Int}
    csrrowptr::Vector{Int}
    csrcolval::Vector{Int}
    csrnzval::Vector{K}
    csccolptr_slots::Vector{Vector{Int}}
    cscrowval_slots::Vector{Vector{Int}}
    cscnzval_slots::Vector{Vector{K}}
    csc_slot::Int
    # Batch map_leq scratch storage.
    pairs_buf::Vector{Tuple{Int,Int}}
    missing_pairs_buf::Vector{Tuple{Int,Int}}
    fetched_buf::Vector{AbstractMatrix{K}}
    map_batch_cache::Dict{UInt64,MapLeqQueryBatch}
    kernel_vertex_cache::Vector{_IncrementalVertexCacheEntry{K}}
    cokernel_vertex_cache::Vector{_IncrementalVertexCacheEntry{K}}
end

@inline function _new_resolution_workspace(::Type{K}, _n::Int) where {K}
    nt = max(1, Threads.maxthreadid())
    return _ResolutionWorkspace{K}(
        [Int[] for _ in 1:nt],
        [Int[] for _ in 1:nt],
        [K[] for _ in 1:nt],
        Int[],
        Int[],
        K[],
        zeros(Int, _n),
        Int[],
        Int[],
        Int[],
        K[],
        [Int[], Int[]],
        [Int[], Int[]],
        [K[], K[]],
        1,
        Tuple{Int,Int}[],
        Tuple{Int,Int}[],
        AbstractMatrix{K}[],
        Dict{UInt64,MapLeqQueryBatch}(),
        _IncrementalVertexCacheEntry{K}[],
        _IncrementalVertexCacheEntry{K}[],
    )
end

@inline function _workspace_prepare!(ws::_ResolutionWorkspace{K}, _n::Int) where {K}
    if length(ws.self_start) != _n
        resize!(ws.self_start, _n)
    end
    empty!(ws.I)
    empty!(ws.J)
    empty!(ws.V)
    for t in 1:length(ws.I_chunks)
        empty!(ws.I_chunks[t])
        empty!(ws.J_chunks[t])
        empty!(ws.V_chunks[t])
    end
    empty!(ws.pairs_buf)
    empty!(ws.missing_pairs_buf)
    empty!(ws.fetched_buf)
    fill!(ws.kernel_vertex_cache, nothing)
    fill!(ws.cokernel_vertex_cache, nothing)
    ws.csc_slot = 1
    return ws
end

mutable struct _UpsetPrefixState{K}
    gens_by_a::Vector{Vector{Vector{Tuple{Int,Int}}}}
    dF::Vector{SparseMatrixCSC{K,Int}}
    curr_pi::PMorphism{K}
    curr_gens::Vector{Vector{Tuple{Int,Int}}}
    done::Bool
end

mutable struct _DownsetPrefixState{K}
    gens_by_b::Vector{Vector{Vector{Tuple{Int,Int}}}}
    dE::Vector{SparseMatrixCSC{K,Int}}
    C::PModule{K}
    prev_q::PMorphism{K}
    prev_gens::Vector{Vector{Tuple{Int,Int}}}
    done::Bool
end

mutable struct _IndicatorPrefixFamily{K}
    upset_locks::Vector{Base.ReentrantLock}
    upset_shards::Vector{IdDict{PModule{K},_UpsetPrefixState{K}}}
    downset_locks::Vector{Base.ReentrantLock}
    downset_shards::Vector{IdDict{PModule{K},_DownsetPrefixState{K}}}
end

@inline function _new_indicator_prefix_family(::Type{K}) where {K}
    nshards = INDICATOR_PREFIX_CACHE_SHARDS[]
    return _IndicatorPrefixFamily{K}(
        [Base.ReentrantLock() for _ in 1:nshards],
        [IdDict{PModule{K},_UpsetPrefixState{K}}() for _ in 1:nshards],
        [Base.ReentrantLock() for _ in 1:nshards],
        [IdDict{PModule{K},_DownsetPrefixState{K}}() for _ in 1:nshards],
    )
end

function _indicator_prefix_family(::Type{K}) where {K}
    Base.lock(_INDICATOR_PREFIX_FAMILY_LOCK)
    try
        fam = get(_INDICATOR_PREFIX_FAMILIES, K, nothing)
        if fam === nothing
            fam = _new_indicator_prefix_family(K)
            _INDICATOR_PREFIX_FAMILIES[K] = fam
        end
        return fam::_IndicatorPrefixFamily{K}
    finally
        Base.unlock(_INDICATOR_PREFIX_FAMILY_LOCK)
    end
end

function _clear_indicator_prefix_caches!()
    Base.lock(_INDICATOR_PREFIX_FAMILY_LOCK)
    try
        for fam_any in values(_INDICATOR_PREFIX_FAMILIES)
            fam = fam_any
            for i in eachindex(fam.upset_shards)
                Base.lock(fam.upset_locks[i])
                try
                    empty!(fam.upset_shards[i])
                finally
                    Base.unlock(fam.upset_locks[i])
                end
                Base.lock(fam.downset_locks[i])
                try
                    empty!(fam.downset_shards[i])
                finally
                    Base.unlock(fam.downset_locks[i])
                end
            end
        end
    finally
        Base.unlock(_INDICATOR_PREFIX_FAMILY_LOCK)
    end
    for i in eachindex(_INDICATOR_BIRTH_PLAN_SHARDS)
        Base.lock(_INDICATOR_BIRTH_PLAN_LOCKS[i])
        try
            empty!(_INDICATOR_BIRTH_PLAN_SHARDS[i])
        finally
            Base.unlock(_INDICATOR_BIRTH_PLAN_LOCKS[i])
        end
    end
    return nothing
end

@inline function _upset_prefix_steps(M::PModule{K}) where {K}
    fam = _indicator_prefix_family(K)
    idx = _indicator_shard_index(M, length(fam.upset_shards))
    Base.lock(fam.upset_locks[idx])
    try
        st = get(fam.upset_shards[idx], M, nothing)
        st === nothing && return 0
        return length(st.dF)
    finally
        Base.unlock(fam.upset_locks[idx])
    end
end

@inline function _downset_prefix_steps(M::PModule{K}) where {K}
    fam = _indicator_prefix_family(K)
    idx = _indicator_shard_index(M, length(fam.downset_shards))
    Base.lock(fam.downset_locks[idx])
    try
        st = get(fam.downset_shards[idx], M, nothing)
        st === nothing && return 0
        return length(st.dE)
    finally
        Base.unlock(fam.downset_locks[idx])
    end
end

@inline function _cached_birth_plans(P::AbstractPoset)
    idx = _indicator_shard_index(P, length(_INDICATOR_BIRTH_PLAN_SHARDS))
    Base.lock(_INDICATOR_BIRTH_PLAN_LOCKS[idx])
    try
        plans = get(_INDICATOR_BIRTH_PLAN_SHARDS[idx], P, nothing)
        if plans === nothing
            n = nvertices(P)
            upset = Vector{Vector{Int}}(undef, n)
            downset = Vector{Vector{Int}}(undef, n)
            @inbounds for i in 1:n
                downset[i] = collect(downset_indices(P, i))
                upset[i] = collect(upset_indices(P, i))
            end
            plans = _BirthPlans(upset, downset)
            _INDICATOR_BIRTH_PLAN_SHARDS[idx][P] = plans
        end
        return plans::_BirthPlans
    finally
        Base.unlock(_INDICATOR_BIRTH_PLAN_LOCKS[idx])
    end
end

@inline function _fill_birth_self_start_down!(
    starts::Vector{Int},
    P::AbstractPoset,
    counts::Vector{Int},
)
    n = nvertices(P)
    length(starts) == n || error("_fill_birth_self_start_down!: wrong length")
    @inbounds for i in 1:n
        pos = 1
        for p in downset_indices(P, i)
            p == i && break
            pos += counts[p]
        end
        starts[i] = pos
    end
    return starts
end

@inline function _fill_birth_self_start_up!(
    starts::Vector{Int},
    P::AbstractPoset,
    counts::Vector{Int},
)
    n = nvertices(P)
    length(starts) == n || error("_fill_birth_self_start_up!: wrong length")
    @inbounds for i in 1:n
        pos = 1
        for u in upset_indices(P, i)
            u == i && break
            pos += counts[u]
        end
        starts[i] = pos
    end
    return starts
end

@inline function _choose_projective_generators(
    field::AbstractCoeffField,
    Img::AbstractMatrix{K},
    d::Int,
) where {K}
    r = size(Img, 2)
    beta = d - r
    beta <= 0 && return Int[]

    A = Matrix{K}(undef, d, r + d)
    if r > 0
        @views A[:, 1:r] .= Img
    end
    @views fill!(A[:, (r + 1):(r + d)], zero(K))
    @inbounds for j in 1:d
        A[j, r + j] = one(K)
    end

    _, pivs = FieldLinAlg.rref(field, A; pivots=true)
    out = Int[]
    sizehint!(out, beta)
    @inbounds for p in pivs
        if p > r
            push!(out, p - r)
            length(out) == beta && break
        end
    end
    return out
end

function _map_leq_cached_indicator(
    M::PModule{K},
    u::Int,
    v::Int,
    cc::CoverCache,
    memo::AbstractVector{Union{Nothing,Matrix{K}}}
)::Matrix{K} where {K}
    n = nvertices(M.Q)
    X = _indicator_memo_get(memo, n, u, v)
    X === nothing || return X
    Xraw = map_leq(M, u, v; cache=cc)
    Xmat = Xraw isa Matrix{K} ? Xraw : Matrix{K}(Xraw)
    return _indicator_memo_set!(memo, n, u, v, Xmat)
end

id_morphism(H::FiniteFringe.FringeModule{K}) where {K} =
    id_morphism(pmodule_from_fringe(H))

@inline _is_exact_field(field::AbstractCoeffField) = !(field isa RealField)

@inline _resolution_cache_shard_index(dicts) =
    min(length(dicts), max(1, Threads.threadid()))
@inline _thread_local_index(arr) =
    min(length(arr), max(1, Threads.threadid()))

@inline function _indicator_use_threads(
    requested::Bool,
    nverts::Int,
    total_dims::Int,
    expected_work::Int,
)
    requested || return false
    Threads.nthreads() > 1 || return false
    nverts >= INDICATOR_THREADS_MIN_VERTICES[] || return false
    total_dims >= INDICATOR_THREADS_MIN_TOTAL_DIMS[] || return false
    expected_work >= INDICATOR_THREADS_MIN_WORK[] || return false
    return true
end

@inline function _indicator_use_map_plans(
    nverts::Int,
    total_pairs::Int,
)
    nverts >= INDICATOR_MAP_PLAN_MIN_VERTICES[] || return false
    total_pairs >= INDICATOR_MAP_PLAN_MIN_PAIRS[] || return false
    return true
end

@inline function _indicator_cache_admit_locked!(
    cache::ResolutionCache,
    key::ResolutionKey3,
    val,
)::Bool
    # Explicit resolution caches have strict reuse semantics: once the caller
    # asks for caching, indicator-resolution payloads are admitted immediately.
    _ = cache
    _ = key
    _ = val
    return true
end

@inline function _indicator_primary_dict(cache::ResolutionCache, ::Type{R}) where {R}
    cache.indicator_primary_type === R || return nothing
    return cache.indicator_primary::Dict{ResolutionKey3,R}
end

@inline function _indicator_primary_shard(cache::ResolutionCache, ::Type{R}) where {R}
    cache.indicator_primary_type === R || return nothing
    shard = cache.indicator_primary_shards[_resolution_cache_shard_index(cache.indicator_primary_shards)]
    shard === nothing && return nothing
    return shard::Dict{ResolutionKey3,R}
end

@inline function _ensure_indicator_primary_locked!(cache::ResolutionCache, ::Type{R}) where {R}
    if cache.indicator_primary_type === nothing && isempty(cache.indicator)
        cache.indicator_primary_type = R
        cache.indicator_primary = Dict{ResolutionKey3,R}()
        for i in eachindex(cache.indicator_primary_shards)
            cache.indicator_primary_shards[i] = Dict{ResolutionKey3,R}()
        end
    end
    return _indicator_primary_dict(cache, R)
end

@inline function _resolution_cache_indicator_get(cache::ResolutionCache, key::ResolutionKey3, ::Type{R}) where {R}
    primary = _indicator_primary_dict(cache, R)
    if primary !== nothing
        if length(cache.indicator_primary_shards) == 1
            return get(primary, key, nothing)
        end
        shard = _indicator_primary_shard(cache, R)
        shard === nothing || begin
            v = get(shard, key, nothing)
            v === nothing || return v
        end
        Base.lock(cache.lock)
        try
            v = get(primary, key, nothing)
            if v !== nothing && shard !== nothing
                shard[key] = v
            end
            return v
        finally
            Base.unlock(cache.lock)
        end
    end

    # Single-thread fast path: avoid lock/shard indirection on misses.
    if length(cache.indicator_shards) == 1
        v = get(cache.indicator, key, nothing)
        return v === nothing ? nothing : (v.value::R)
    end

    shard = cache.indicator_shards[_resolution_cache_shard_index(cache.indicator_shards)]
    v = get(shard, key, nothing)
    v === nothing || return (v.value::R)
    Base.lock(cache.lock)
    try
        v = get(cache.indicator, key, nothing)
    finally
        Base.unlock(cache.lock)
    end
    v === nothing || begin
        vv = v.value::R
        shard[key] = v
        return vv
    end
    return nothing
end

@inline function _resolution_cache_indicator_store!(cache::ResolutionCache, key::ResolutionKey3, val::R) where {R}
    primary = _indicator_primary_dict(cache, R)
    if primary === nothing
        Base.lock(cache.lock)
        try
            primary = _ensure_indicator_primary_locked!(cache, R)
            if primary !== nothing
                extant = get(primary, key, nothing)
                extant === nothing || return extant
                primary[key] = val
                shard = cache.indicator_primary_shards[_resolution_cache_shard_index(cache.indicator_primary_shards)]::Dict{ResolutionKey3,R}
                shard[key] = val
                return val
            end
        finally
            Base.unlock(cache.lock)
        end
    else
        if length(cache.indicator_primary_shards) == 1
            extant = get(primary, key, nothing)
            extant === nothing || return extant
            _indicator_cache_admit_locked!(cache, key, val) || return val
            primary[key] = val
            return val
        end
        shard = _indicator_primary_shard(cache, R)
        shard === nothing || begin
            extant = get(shard, key, nothing)
            extant === nothing || return extant
        end
        Base.lock(cache.lock)
        try
            extant = get(primary, key, nothing)
            extant === nothing || return extant
            _indicator_cache_admit_locked!(cache, key, val) || return val
            primary[key] = val
            shard === nothing || (shard[key] = val)
            return val
        finally
            Base.unlock(cache.lock)
        end
    end

    # Single-thread fast path: lock-free get!/insert.
    if length(cache.indicator_shards) == 1
        extant = get(cache.indicator, key, nothing)
        extant === nothing || return (extant.value::R)
        _indicator_cache_admit_locked!(cache, key, val) || return val
        out = get!(cache.indicator, key) do
            IndicatorResolutionPayload(val)
        end
        return out.value::R
    end

    shard = cache.indicator_shards[_resolution_cache_shard_index(cache.indicator_shards)]
    existing = get(shard, key, nothing)
    existing === nothing || return (existing.value::R)

    Base.lock(cache.lock)
    out = try
        extant = get(cache.indicator, key, nothing)
        if extant !== nothing
            extant
        elseif !_indicator_cache_admit_locked!(cache, key, val)
            nothing
        else
            get!(cache.indicator, key) do
                IndicatorResolutionPayload(val)
            end
        end
    finally
        Base.unlock(cache.lock)
    end
    out === nothing && return val
    outR = out.value::R
    shard[key] = out
    return outR
end

function _is_zero_matrix(field::AbstractCoeffField, M)
    if field isa RealField
        tol = field.atol + field.rtol * opnorm(M, 1)
        return norm(M) <= tol
    end
    return nnz(M) == 0
end

function _rank_restricted_field(field::AbstractCoeffField, A, rows, cols)
    return FieldLinAlg.rank_restricted(field, A, rows, cols)
end

# ----------------------- from FringeModule to PModule ----------------------------

@inline function _active_summands_per_vertex(summands, n::Int)
    out = [Int[] for _ in 1:n]
    @inbounds for idx in eachindex(summands)
        mask = summands[idx].mask
        for q in 1:n
            mask[q] && push!(out[q], idx)
        end
    end
    return out
end

@inline function _project_rows_to_vertex!(
    Im::Matrix{K},
    Bu::AbstractMatrix{K},
    rows_u::Vector{Int},
    rows_v::Vector{Int},
) where {K}
    rows_u === rows_v && return copyto!(Im, Bu)
    i = 1
    j = 1
    nv = length(rows_v)
    nu = length(rows_u)
    @inbounds while i <= nv && j <= nu
        rv = rows_v[i]
        ru = rows_u[j]
        if ru == rv
            copyto!(view(Im, i, :), view(Bu, j, :))
            i += 1
            j += 1
        elseif ru < rv
            j += 1
        else
            error("_project_rows_to_vertex!: projection rows mismatch")
        end
    end
    i > nv || error("_project_rows_to_vertex!: projection rows mismatch")
    return Im
end

"""
    pmodule_from_fringe(H::FiniteFringe.FringeModule{K})
Return an internal `PModule{K}` whose fibers and structure maps are induced by the
fringe presentation `phi : oplus k[U_i] to oplus k[D_j]` (Defs. 3.16-3.17).
Implementation: M_q = im(phi_q) inside E_q; along a cover u<v the map is the restriction
E_u to E_v followed by projection to M_v.
"""
function pmodule_from_fringe(H::FiniteFringe.FringeModule{K}) where {K}
    field = H.field
    Q = H.P
    n = nvertices(Q)
    build_cache!(Q; cover=true)
    cc = _get_cover_cache(Q)
    active_U = _active_summands_per_vertex(H.U, n)
    active_D = _active_summands_per_vertex(H.D, n)

    # Basis for each fiber M_q as columns of a K matrix B[q] spanning im(phi_q).
    B = Vector{Matrix{K}}(undef, n)
    dims = zeros(Int, n)
    for q in 1:n
        cols = active_U[q]
        rows = active_D[q]
        if isempty(cols) || isempty(rows)
            B[q] = zeros(K, length(rows), 0)
            dims[q] = 0
            continue
        end
        phi_q = @view H.phi[rows, cols]
        Bq = FieldLinAlg.colspace(field, phi_q)
        B[q] = Bq
        dims[q] = size(Bq, 2)
    end

    # Structure maps on cover edges, aligned with the cover cache store so later
    # kernels can traverse cover edges without dict lookups.
    total_work = 0
    @inbounds for u in 1:n
        du = dims[u]
        du == 0 && continue
        for v in _succs(cc, u)
            total_work += du * dims[v]
        end
    end

    if !_indicator_use_direct_pmodule_store(cc.nedges, total_work)
        edge_maps = Dict{Tuple{Int,Int},Matrix{K}}()
        sizehint!(edge_maps, cc.nedges)
        @inbounds for u in 1:n
            for v in _succs(cc, u)
                du = dims[u]
                dv = dims[v]
                X = if du == 0 || dv == 0
                    zeros(K, dv, du)
                else
                    rows_u = active_D[u]
                    rows_v = active_D[v]
                    Im = zeros(K, length(rows_v), du)
                    _project_rows_to_vertex!(Im, B[u], rows_u, rows_v)
                    FieldLinAlg.solve_fullcolumn(field, B[v], Im)
                end
                edge_maps[(u, v)] = X
            end
        end
        return PModule{K}(Q, dims, edge_maps; field=field)
    end

    preds = [_preds(cc, v) for v in 1:n]
    succs = [_succs(cc, u) for u in 1:n]
    maps_from_pred = [Vector{Matrix{K}}(undef, length(preds[v])) for v in 1:n]
    maps_to_succ = [Vector{Matrix{K}}(undef, length(succs[u])) for u in 1:n]

    @inbounds for u in 1:n
        su = succs[u]
        for j in eachindex(su)
            v = su[j]
            du = dims[u]
            dv = dims[v]
            X = if du == 0 || dv == 0
                zeros(K, dv, du)
            else
                rows_u = active_D[u]
                rows_v = active_D[v]
                Im = zeros(K, length(rows_v), du)
                _project_rows_to_vertex!(Im, B[u], rows_u, rows_v)
                FieldLinAlg.solve_fullcolumn(field, B[v], Im)
            end
            maps_to_succ[u][j] = X
            maps_from_pred[v][_pred_slots_of_succ(cc, u)[j]] = X
        end
    end
    store = CoverEdgeMapStore{K,Matrix{K}}(preds, succs, maps_from_pred, maps_to_succ, cc.nedges)
    return PModule{K}(Q, dims, store; field=field)
end

# -------------------------- projective cover (Def. 6.4.1) --------------------------

@inline function _colspace_union_incremental(
    field::AbstractCoeffField,
    nrows::Int,
    mats,
    ::Type{K},
) where {K}
    basis = zeros(K, nrows, 0)
    @inbounds for A in mats
        ncols = size(A, 2)
        ncols == 0 && continue
        if size(basis, 2) == 0
            basis = FieldLinAlg.colspace(field, A)
        else
            merged = Matrix{K}(undef, nrows, size(basis, 2) + ncols)
            @views merged[:, 1:size(basis, 2)] .= basis
            @views merged[:, size(basis, 2)+1:end] .= A
            basis = FieldLinAlg.colspace(field, merged)
        end
        size(basis, 2) == nrows && return basis
    end
    return basis
end

@inline function _colspace_union_dense(
    field::AbstractCoeffField,
    nrows::Int,
    total_cols::Int,
    mats,
    ::Type{K},
) where {K}
    total_cols == 0 && return zeros(K, nrows, 0)
    merged = Matrix{K}(undef, nrows, total_cols)
    offset = 0
    @inbounds for A in mats
        ncols = size(A, 2)
        ncols == 0 && continue
        @views merged[:, offset + 1:offset + ncols] .= A
        offset += ncols
    end
    offset == total_cols || error("_colspace_union_dense: total_cols mismatch")
    return FieldLinAlg.colspace(field, merged)
end

"Incoming image at v from immediate predecessors; basis matrix with K columns."
function _incoming_image_basis(M::PModule{K}, v::Int; cache::Union{Nothing,CoverCache}=nothing) where {K}
    field = M.field
    dv = M.dims[v]
    pv = M.edge_maps.preds[v]
    maps = M.edge_maps.maps_from_pred[v]

    if dv == 0 || isempty(pv)
        return zeros(K, dv, 0)
    end

    tot = 0
    @inbounds for u in pv
        tot += M.dims[u]
    end
    if tot == 0
        return zeros(K, dv, 0)
    end

    if length(pv) == 1
        return FieldLinAlg.colspace(field, maps[1])
    end

    if !_indicator_use_incremental_union(dv, tot, length(pv))
        return _colspace_union_dense(field, dv, tot, maps, K)
    end

    return _colspace_union_incremental(field, dv, maps, K)
end



"""
    projective_cover(M::PModule{K})
Return (F0, pi0, gens_at) where F0 is a direct sum of principal upsets covering M,
pi0 : F0 \to M is the natural surjection, and `gens_at[v]` lists the generators activated
at vertex v (each item is a pair (p, local_index_in_Mp)).
"""
function projective_cover(M::PModule{K};
                          cache::Union{Nothing,CoverCache}=nothing,
                          map_memo::Union{Nothing,AbstractVector{Union{Nothing,Matrix{K}}}}=nothing,
                          workspace::Union{Nothing,_ResolutionWorkspace{K}}=nothing,
                          threads::Bool = (Threads.nthreads() > 1)) where {K}
    field = M.field
    Q = M.Q; n = nvertices(Q)
    build_cache!(Q; cover=true, updown=true)
    cc = cache === nothing ? _get_cover_cache(Q) : cache
    total_dims = sum(M.dims)
    expected_work = max(1, cc.nedges) * max(1, total_dims)
    threaded = _indicator_use_threads(threads, n, total_dims, expected_work)
    map_memo_local = map_memo === nothing ? _indicator_new_array_memo(K, n) : map_memo
    ws = workspace
    if ws === nothing && !threaded
        ws = _new_resolution_workspace(K, n)
    end
    memos = threaded ?
        [_indicator_new_array_memo(K, n)
         for _ in 1:max(1, Threads.maxthreadid())] : Vector{Vector{Union{Nothing,Matrix{K}}}}()

    # number of generators at each vertex = dim(M_v) - rank(incoming_image)
    gens_at = Vector{Vector{Tuple{Int,Int}}}(undef, n)
    chosen_at = Vector{Vector{Int}}(undef, n)
    gen_of_p = fill(0, n)
    if threaded
        Threads.@threads for v in 1:n
            Img = _incoming_image_basis(M, v; cache=cc)
            chosen = _choose_projective_generators(field, Img, M.dims[v])
            chosen_at[v] = chosen
            gens_at[v] = [(v, j) for j in chosen]
            gen_of_p[v] = length(chosen)
        end
    else
        for v in 1:n
            Img = _incoming_image_basis(M, v; cache=cc)
            chosen = _choose_projective_generators(field, Img, M.dims[v])
            chosen_at[v] = chosen
            gens_at[v] = [(v, j) for j in chosen]
            gen_of_p[v] = length(chosen)
        end
    end

        # F0 as a direct sum of principal upsets.
    #
    # IMPORTANT CORRECTNESS NOTE
    # On a general finite poset, the cover-edge maps of a direct sum of principal
    # upsets are NOT given by a "rectangular identity" unless the chosen basis at
    # every vertex extends the basis at each predecessor as a prefix.
    #
    # We therefore build the cover-edge maps by matching generator labels (p,j)
    # across vertices. This agrees with the representable functor structure and is
    # independent of the arbitrary ordering of vertices.
    F0_dims = zeros(Int, n)
    if threaded
        Threads.@threads for i in 1:n
            s = 0
            for p in downset_indices(Q, i)
                s += gen_of_p[p]
            end
            F0_dims[i] = s
        end
    else
        for i in 1:n
            s = 0
            for p in downset_indices(Q, i)
                s += gen_of_p[p]
            end
            F0_dims[i] = s
        end
    end

    gid_ranges, _ = _generator_id_ranges(gen_of_p)
    # Active generator ids at each vertex i (sorted global dense ids).
    active_ids = Vector{Vector{Int}}(undef, n)
    if threaded
        Threads.@threads for i in 1:n
            lst = Vector{Int}(undef, F0_dims[i])
            pos = 1
            for p in downset_indices(Q, i)
                rng = gid_ranges[p]
                @inbounds for gid in rng
                    lst[pos] = gid
                    pos += 1
                end
            end
            active_ids[i] = lst
        end
    else
        for i in 1:n
            lst = Vector{Int}(undef, F0_dims[i])
            pos = 1
            for p in downset_indices(Q, i)
                rng = gid_ranges[p]
                @inbounds for gid in rng
                    lst[pos] = gid
                    pos += 1
                end
            end
            active_ids[i] = lst
        end
    end

    F0_edges = Dict{Tuple{Int,Int}, SparseMatrixCSC{K,Int}}()
    if threaded
        edges = cover_edges(Q)
        mats = Vector{SparseMatrixCSC{K,Int}}(undef, length(edges))
        Threads.@threads for idx in eachindex(edges)
            u, v = edges[idx]
            Muv = _subsequence_identity_sparse(K, active_ids[v], active_ids[u])
            mats[idx] = Muv
        end
        for idx in eachindex(edges)
            F0_edges[edges[idx]] = mats[idx]
        end
    else
        @inbounds for u in 1:n
            su = _succs(cc, u)
            for v in su
                Muv = _subsequence_identity_sparse(K, active_ids[v], active_ids[u])
                F0_edges[(u, v)] = Muv
            end
        end
    end
    F0 = PModule{K}(Q, F0_dims, F0_edges; field=field)


    # pi0 : F0 -> M
    #
    # Preallocate each component and fill blockwise.
    # Old code used repeated hcat (allocates many temporaries).
    # Cache the chosen basis indices in each M_p once.
    J_at = chosen_at
    total_pairs = 0
    @inbounds for i in 1:n
        for p in downset_indices(Q, i)
            gen_of_p[p] == 0 && continue
            total_pairs += 1
        end
    end
    use_plans = _indicator_use_map_plans(n, total_pairs)
    plist_by_i = Vector{Vector{Int}}()
    pairs_by_i = Vector{Vector{Tuple{Int,Int}}}()
    batch_by_i = Vector{Union{Nothing,MapLeqQueryBatch}}()
    if use_plans
        resize!(plist_by_i, n)
        resize!(pairs_by_i, n)
        resize!(batch_by_i, n)
        @inbounds for i in 1:n
            plist = Int[]
            pairs = Tuple{Int,Int}[]
            for p in downset_indices(Q, i)
                gen_of_p[p] == 0 && continue
                push!(plist, p)
                push!(pairs, (p, i))
            end
            plist_by_i[i] = plist
            pairs_by_i[i] = pairs
            batch_by_i[i] = isempty(pairs) ? nothing : prepare_map_leq_batch(pairs)
        end
    end

    comps = Vector{Matrix{K}}(undef, n)
    if threaded
        Threads.@threads for i in 1:n
            memo = memos[_thread_local_index(memos)]
            Mi = M.dims[i]
            Fi = F0_dims[i]
            cols = zeros(K, Mi, Fi)
            col = 1
            plist = if use_plans
                plist_by_i[i]
            else
                loc = Int[]
                for p in downset_indices(Q, i)
                    gen_of_p[p] == 0 && continue
                    push!(loc, p)
                end
                loc
            end
            batch = if use_plans
                batch_by_i[i]
            else
                pairs = Tuple{Int,Int}[(p, i) for p in plist]
                isempty(pairs) ? nothing : prepare_map_leq_batch(pairs)
            end
            maps = batch === nothing ? Matrix{K}[] : _map_leq_cached_many_indicator(M, batch, cc, memo)
            @inbounds for t in eachindex(plist)
                p = plist[t]
                k = gen_of_p[p]
                A = maps[t]
                Jp = J_at[p]
                for s in 1:k
                    j = Jp[s]
                    copyto!(view(cols, :, col), view(A, :, j))
                    col += 1
                end
            end
            comps[i] = cols
        end
    else
        for i in 1:n
            Mi = M.dims[i]
            Fi = F0_dims[i]
            cols = zeros(K, Mi, Fi)
            col = 1
            plist = Int[]
            pairs = Tuple{Int,Int}[]
            batch = nothing
            if use_plans
                plist = plist_by_i[i]
                pairs = pairs_by_i[i]
                batch = batch_by_i[i]
            elseif ws === nothing
                for p in downset_indices(Q, i)
                    gen_of_p[p] == 0 && continue
                    push!(plist, p)
                end
                pairs = Tuple{Int,Int}[(p, i) for p in plist]
                batch = isempty(pairs) ? nothing : prepare_map_leq_batch(pairs)
            else
                pairs = ws.pairs_buf
                empty!(pairs)
                for p in downset_indices(Q, i)
                    gen_of_p[p] == 0 && continue
                    push!(plist, p)
                    push!(pairs, (p, i))
                end
            end
            if ws === nothing
                maps = batch === nothing ? Matrix{K}[] : _map_leq_cached_many_indicator(M, batch, cc, map_memo_local)
                @inbounds for t in eachindex(plist)
                    p = plist[t]
                    k = gen_of_p[p]
                    A = maps[t]
                    Jp = J_at[p]
                    for s in 1:k
                        j = Jp[s]
                        copyto!(view(cols, :, col), view(A, :, j))
                        col += 1
                    end
                end
            else
                if length(pairs) <= INDICATOR_MAP_BATCH_THRESHOLD[]
                    _map_leq_fill_memo_indicator!(M, pairs, cc, map_memo_local, ws)
                else
                    if use_plans
                        _map_leq_fill_memo_indicator!(M, batch::MapLeqQueryBatch, cc, map_memo_local, ws)
                    else
                        _map_leq_fill_memo_indicator!(M, _workspace_get_or_prepare_batch!(ws, pairs), cc, map_memo_local, ws)
                    end
                end
                @inbounds for p in plist
                    k = gen_of_p[p]
                    A = _indicator_memo_get(map_memo_local, n, p, i)::Matrix{K}
                    Jp = J_at[p]
                    for s in 1:k
                        j = Jp[s]
                        copyto!(view(cols, :, col), view(A, :, j))
                        col += 1
                    end
                end
            end
            comps[i] = cols
        end
    end
    pi0 = PMorphism{K}(F0, M, comps)
    return F0, pi0, gens_at
end


# Basis of the "socle" at vertex u: kernel of the stacked outgoing map
# M_u to oplus_{u<v} M_v along cover edges u < v.
# Columns of the returned matrix span soc(M)_u subseteq M_u.
function _socle_basis(M::PModule{K}, u::Int; cache::Union{Nothing,CoverCache}=nothing) where {K}
    field = M.field
    cc = (cache === nothing ? _get_cover_cache(M.Q) : cache)
    su = _succs(cc, u)
    du = M.dims[u]

    if isempty(su) || du == 0
        return eye(field, du)
    end

    # Build the stacked outgoing map A : M_u -> (direct sum over cover successors).
    # A has size (sum_v dim(M_v)) x dim(M_u).
    tot = 0
    @inbounds for j in eachindex(su)
        tot += M.dims[su[j]]
    end
    if tot == 0
        # No nonzero target fibers; outgoing map is zero, so socle is all of M_u.
        return eye(field, du)
    end
    S = _outgoing_span_basis(M, u; cache=cc)
    size(S, 1) == 0 && return eye(field, du)
    return FieldLinAlg.nullspace(field, S)  # columns span the socle at u
end


# A canonical left-inverse for a full-column-rank matrix S: L*S = I.
# Implemented as L = (S^T S)^{-1} S^T using solve_fullcolumn on the Gram matrix.
function _left_inverse_full_column(field::AbstractCoeffField, S::AbstractMatrix{K}) where {K}
    s = size(S,2)
    if s == 0
        return zeros(K, 0, size(S,1))
    end
    G = transpose(S) * S                       # s*s Gram matrix, invertible if S has full column rank
    return FieldLinAlg.solve_fullcolumn(field, G, transpose(S)) # returns (S^T S)^{-1} S^T
end

# Build the injective (downset) hull:  iota : M into E  where
# E is a direct sum of principal downsets with multiplicities = socle dimensions.
# Also return the generator labels as (u, j) with u the vertex and j the column.
function _injective_hull(M::PModule{K};
                         cache::Union{Nothing,CoverCache}=nothing,
                         map_memo::Union{Nothing,AbstractVector{Union{Nothing,Matrix{K}}}}=nothing,
                         workspace::Union{Nothing,_ResolutionWorkspace{K}}=nothing,
                         threads::Bool = (Threads.nthreads() > 1)) where {K}
    field = M.field
    Q = M.Q; n = nvertices(Q)
    build_cache!(Q; cover=true, updown=true)
    cc = cache === nothing ? _get_cover_cache(Q) : cache
    total_dims = sum(M.dims)
    expected_work = max(1, cc.nedges) * max(1, total_dims)
    threaded = _indicator_use_threads(threads, n, total_dims, expected_work)
    map_memo_local = map_memo === nothing ? _indicator_new_array_memo(K, n) : map_memo
    ws = workspace
    if ws === nothing && !threaded
        ws = _new_resolution_workspace(K, n)
    end
    memos = threaded ?
        [_indicator_new_array_memo(K, n)
         for _ in 1:max(1, Threads.maxthreadid())] : Vector{Vector{Union{Nothing,Matrix{K}}}}()

    # socle bases at each vertex and their multiplicities
    Soc = Vector{Matrix{K}}(undef, n)
    mult = zeros(Int, n)
    if threaded
        Threads.@threads for u in 1:n
            Soc[u]  = _socle_basis(M, u; cache=cc)
            mult[u] = size(Soc[u], 2)
        end
    else
        for u in 1:n
            Soc[u]  = _socle_basis(M, u; cache=cc)
            mult[u] = size(Soc[u], 2)
        end
    end

        # generator labels for the chosen downset summands
    gens_at = Vector{Vector{Tuple{Int,Int}}}(undef, n)
    for u in 1:n
        gens_at[u] = [(u, j) for j in 1:mult[u]]
    end

    # fiber dimensions of E
    Edims = zeros(Int, n)
    if threaded
        Threads.@threads for i in 1:n
            s = 0
            for u in upset_indices(Q, i)
                s += mult[u]
            end
            Edims[i] = s
        end
    else
        for i in 1:n
            s = 0
            for u in upset_indices(Q, i)
                s += mult[u]
            end
            Edims[i] = s
        end
    end

    gid_ranges, _ = _generator_id_ranges(mult)
    # Active generator ids at each vertex i (sorted global dense ids).
    # This ordering matches the row-stacking order used in iota below.
    active_ids = Vector{Vector{Int}}(undef, n)
    if threaded
        Threads.@threads for i in 1:n
            lst = Vector{Int}(undef, Edims[i])
            pos = 1
            for u in upset_indices(Q, i)
                rng = gid_ranges[u]
                @inbounds for gid in rng
                    lst[pos] = gid
                    pos += 1
                end
            end
            active_ids[i] = lst
        end
    else
        for i in 1:n
            lst = Vector{Int}(undef, Edims[i])
            pos = 1
            for u in upset_indices(Q, i)
                rng = gid_ranges[u]
                @inbounds for gid in rng
                    lst[pos] = gid
                    pos += 1
                end
            end
            active_ids[i] = lst
        end
    end

    # E structure maps along cover edges u<v are coordinate projections:
    # keep exactly those generators that are still active at v.
    Eedges = Dict{Tuple{Int,Int}, SparseMatrixCSC{K,Int}}()
    if threaded
        edges = cover_edges(Q)
        mats = Vector{SparseMatrixCSC{K,Int}}(undef, length(edges))
        Threads.@threads for idx in eachindex(edges)
            u, v = edges[idx]
            Muv = _projection_identity_sparse(K, active_ids[v], active_ids[u])
            mats[idx] = Muv
        end
        for idx in eachindex(edges)
            Eedges[edges[idx]] = mats[idx]
        end
    else
        @inbounds for u in 1:n
            su = _succs(cc, u)
            for v in su
                Muv = _projection_identity_sparse(K, active_ids[v], active_ids[u])
                Eedges[(u, v)] = Muv
            end
        end
    end
    E = PModule{K}(Q, Edims, Eedges; field=field)

    # iota : M -> E
    Linv = [ _left_inverse_full_column(field, Soc[u]) for u in 1:n ]
    active_sources = Vector{Vector{Int}}(undef, n)
    if threaded
        Threads.@threads for i in 1:n
            count = 0
            for u in upset_indices(Q, i)
                mult[u] == 0 && continue
                count += 1
            end
            ulist = Vector{Int}(undef, count)
            pos = 1
            for u in upset_indices(Q, i)
                mult[u] == 0 && continue
                @inbounds ulist[pos] = u
                pos += 1
            end
            active_sources[i] = ulist
        end
    else
        for i in 1:n
            count = 0
            for u in upset_indices(Q, i)
                mult[u] == 0 && continue
                count += 1
            end
            ulist = Vector{Int}(undef, count)
            pos = 1
            for u in upset_indices(Q, i)
                mult[u] == 0 && continue
                @inbounds ulist[pos] = u
                pos += 1
            end
            active_sources[i] = ulist
        end
    end

    total_pairs = sum(length, active_sources)
    use_plans = _indicator_use_map_plans(n, total_pairs)
    ulist_by_i = active_sources
    pairs_by_i = Vector{Vector{Tuple{Int,Int}}}()
    batch_by_i = Vector{Union{Nothing,MapLeqQueryBatch}}()
    if use_plans
        resize!(pairs_by_i, n)
        resize!(batch_by_i, n)
        @inbounds for i in 1:n
            ulist = ulist_by_i[i]
            pairs = Vector{Tuple{Int,Int}}(undef, length(ulist))
            for t in eachindex(ulist)
                pairs[t] = (i, ulist[t])
            end
            pairs_by_i[i] = pairs
            batch_by_i[i] = isempty(pairs) ? nothing : prepare_map_leq_batch(pairs)
        end
    end

    # Only vertices with nontrivial socle contribute rows.
    comps = Vector{Matrix{K}}(undef, n)
    if threaded
        Threads.@threads for i in 1:n
            memo = memos[_thread_local_index(memos)]
            rows = zeros(K, Edims[i], M.dims[i])
            r = 1
            ulist = if use_plans
                ulist_by_i[i]
            else
                ulist_by_i[i]
            end
            batch = if use_plans
                batch_by_i[i]
            else
                pairs = Tuple{Int,Int}[(i, u) for u in ulist]
                isempty(pairs) ? nothing : prepare_map_leq_batch(pairs)
            end
            maps = batch === nothing ? Matrix{K}[] : _map_leq_cached_many_indicator(M, batch, cc, memo)
            @inbounds for t in eachindex(ulist)
                u = ulist[t]
                m = mult[u]
                Mi_to_Mu = maps[t]
                @views mul!(rows[r:r+m-1, :], Linv[u], Mi_to_Mu)
                r += m
            end
            @assert r == Edims[i] + 1
            comps[i] = rows
        end
    else
        for i in 1:n
            rows = zeros(K, Edims[i], M.dims[i])
            r = 1
            ulist = ulist_by_i[i]
            pairs = Tuple{Int,Int}[]
            batch = nothing
            if use_plans
                pairs = pairs_by_i[i]
                batch = batch_by_i[i]
            elseif ws === nothing
                pairs = Tuple{Int,Int}[(i, u) for u in ulist]
                batch = isempty(pairs) ? nothing : prepare_map_leq_batch(pairs)
            else
                pairs = ws.pairs_buf
                empty!(pairs)
                for u in ulist
                    push!(pairs, (i, u))
                end
            end
            if ws === nothing
                maps = batch === nothing ? Matrix{K}[] : _map_leq_cached_many_indicator(M, batch, cc, map_memo_local)
                @inbounds for t in eachindex(ulist)
                    u = ulist[t]
                    m = mult[u]
                    Mi_to_Mu = maps[t]
                    @views mul!(rows[r:r+m-1, :], Linv[u], Mi_to_Mu)
                    r += m
                end
            else
                if length(pairs) <= INDICATOR_MAP_BATCH_THRESHOLD[]
                    _map_leq_fill_memo_indicator!(M, pairs, cc, map_memo_local, ws)
                else
                    if use_plans
                        _map_leq_fill_memo_indicator!(M, batch::MapLeqQueryBatch, cc, map_memo_local, ws)
                    else
                        _map_leq_fill_memo_indicator!(M, _workspace_get_or_prepare_batch!(ws, pairs), cc, map_memo_local, ws)
                    end
                end
                @inbounds for u in ulist
                    m = mult[u]
                    Mi_to_Mu = _indicator_memo_get(map_memo_local, n, i, u)::Matrix{K}
                    @views mul!(rows[r:r+m-1, :], Linv[u], Mi_to_Mu)
                    r += m
                end
            end
            @assert r == Edims[i] + 1
            comps[i] = rows
        end
    end
    iota = PMorphism{K}(M, E, comps)

    return E, iota, gens_at
end

"""
    injective_hull(M::PModule{K}; cache=nothing, threads=(Threads.nthreads() > 1))

Build the one-step injective hull `M -> E` and return `(E, iota, gens_at_E)`.
"""
function injective_hull(
    M::PModule{K};
    cache::Union{Nothing,CoverCache}=nothing,
    map_memo::Union{Nothing,AbstractVector{Union{Nothing,Matrix{K}}}}=nothing,
    workspace::Union{Nothing,_ResolutionWorkspace{K}}=nothing,
    threads::Bool=(Threads.nthreads() > 1),
) where {K}
    return _injective_hull(M; cache=cache, map_memo=map_memo, workspace=workspace, threads=threads)
end

"""
    upset_presentation_one_step(Hfringe::FringeModule)
Compute the one-step upset presentation (Def. 6.4.1):
    F1 --d1--> F0 --pi0-->> M,
and return the lightweight wrapper `UpsetPresentation{K}(P, U0, U1, delta, H)`.
"""
function upset_presentation_one_step(H::FiniteFringe.FringeModule{K}) where {K}
    M = pmodule_from_fringe(H)          # internal PModule over K
    cc = _get_cover_cache(M.Q)
    # First step: projective cover of M
    F0, pi0, gens_at_F0 = projective_cover(M; cache=cc)
    # Kernel K1 with inclusion i1 : K1 \into F0
    K1, i1 = kernel_with_inclusion(pi0; cache=cc)
    # Second projective cover (of K1)
    F1, pi1, gens_at_F1 = projective_cover(K1; cache=cc)
    # Differential d1 = i1 \circ pi1 : F1 \to F0
    comps = [i1.comps[i] * pi1.comps[i] for i in 1:length(M.dims)]
    d1 = PMorphism{K}(F1, F0, comps)

    # Build indicator wrapper:
    # U0: list of principal upsets, one per generator in gens_at_F0[p]
    P = M.Q
    U0 = Upset[]
    for p in 1:nvertices(P)
        for _ in gens_at_F0[p]
            push!(U0, principal_upset(P, p))
        end
    end
    U1 = Upset[]
    for p in 1:nvertices(P)
        for _ in gens_at_F1[p]
            push!(U1, principal_upset(P, p))
        end
    end

    # scalar delta block: one entry for each pair (theta in U1, lambda in U0) if ptheta <= plambda.
    # Extract the scalar from d1 at the minimal vertex i = plambda.
    m1 = length(U1); m0 = length(U0)
    delta = spzeros(K, m1, m0)
    # local index maps: at vertex i, Fk_i basis is "all generators at p with p <= i" in (p increasing) order.
    # Build offsets to find local coordinates.
    function local_index_list(gens_at)
        # return vector of vectors L[i] listing global generator indices active at vertex i
        L = Vector{Vector{Tuple{Int,Int}}}(undef, nvertices(P))
        for i in 1:nvertices(P)
            L[i] = Tuple{Int,Int}[]
            for p in downset_indices(P, i)
                append!(L[i], gens_at[p])
            end
        end
        L
    end
    L0 = local_index_list(gens_at_F0)
    L1 = local_index_list(gens_at_F1)

    # Build a map from "global generator number in U0/U1" to its vertex p and position j in M_p
    globalU0 = Tuple{Int,Int}[]  # (p,j)
    for p in 1:nvertices(P); append!(globalU0, gens_at_F0[p]); end
    globalU1 = Tuple{Int,Int}[]
    for p in 1:nvertices(P); append!(globalU1, gens_at_F1[p]); end

    # helper: find local column index of global generator g=(p,j) at vertex i
    function local_col_of(L, g::Tuple{Int,Int}, i::Int)
        for (c, gg) in enumerate(L[i])
            if gg == g; return c; end
        end
        return 0
    end

    for (lambda, (plambda, jlambda)) in enumerate(globalU0)
        for (theta, (ptheta, jtheta)) in enumerate(globalU1)
            if leq(P, plambda, ptheta)   # containment for principal upsets: Up(ptheta) subseteq Up(plambda)
                i = ptheta              # read at the minimal vertex where the domain generator exists
                col = local_col_of(L1, (ptheta, jtheta), i)
                row = local_col_of(L0, (plambda, jlambda), i)
                if col > 0 && row > 0
                    val = d1.comps[i][row, col]
                    if val != 0
                        delta[theta, lambda] = val
                    end
                end
            end

        end
    end

    UpsetPresentation{K}(P, U0, U1, delta, H)
end

# ------------------------ downset copresentation (Def. 6.4.2) ------------------------

# The dual story: compute an injective hull E0 of M and the next step E1 with rho0 : E0 \to E1.
# For brevity we implement the duals by applying the above steps to M^op using
# down-closures/up-closures symmetry. Here we implement directly degreewise.

"Outgoing coimage at u to immediate successors; basis for the span of maps M_u to oplus_{u<v} M_v."
function _outgoing_span_basis(M::PModule{K}, u::Int; cache::Union{Nothing,CoverCache}=nothing) where {K}
    field = M.field
    cc = (cache === nothing ? _get_cover_cache(M.Q) : cache)
    su = _succs(cc, u)
    du = M.dims[u]

    if isempty(su) || du == 0
        return zeros(K, 0, du)
    end

    # Stack outgoing maps B : M_u -> (direct sum over cover successors).
    tot = 0
    @inbounds for j in eachindex(su)
        tot += M.dims[su[j]]
    end
    if tot == 0
        return zeros(K, 0, du)
    end
    maps_u = M.edge_maps.maps_to_succ[u]
    if length(su) == 1
        A = maps_u[1]
        return size(A, 1) == 0 ? zeros(K, 0, du) :
               transpose(FieldLinAlg.colspace(field, transpose(A)))
    end
    cols = (transpose(A) for A in maps_u)
    if !_indicator_use_incremental_union(du, tot, length(su))
        return transpose(_colspace_union_dense(field, du, tot, cols, K))
    end
    return transpose(_colspace_union_incremental(field, du, cols, K))
end


"Essential socle dimension at u = dim(M_u) - rank(outgoing span) (dual of generators)."
function _socle_count(M::PModule{K}, u::Int) where {K}
    field = M.field
    S = _outgoing_span_basis(M, u)
    M.dims[u] - FieldLinAlg.rank(field, S)
end

"""
    downset_copresentation_one_step(Hfringe::FringeModule)

Compute the one-step downset **copresentation** (Def. 6.4(2)):
    M = ker(rho : E^0 to E^1),
with E^0 and E^1 expressed as direct sums of principal downsets and rho assembled
from the actual vertexwise maps, not just from the partial order.  Steps:

1. Build the injective (downset) hull iota0 : M into E^0.
2. Form C = coker(iota0) as a P-module together with q : E^0 to C.
3. Build the injective (downset) hull j : C into E^1 and set rho0 = j circ q : E^0 -> E^1.
4. Read scalar entries of rho at minimal vertices, as for delta on the upset side.
"""
function downset_copresentation_one_step(H::FiniteFringe.FringeModule{K}) where {K}
    # Convert fringe to internal PModule over K
    M = pmodule_from_fringe(H)
    Q = M.Q; n = nvertices(Q)
    cc = _get_cover_cache(Q)

    # (1) Injective hull of M: E0 with inclusion iota0
    E0, iota0, gens_at_E0 = _injective_hull(M; cache=cc)

    # (2) Degreewise cokernel C and the quotient q : E0 \to C
    C, q = _cokernel_module(iota0; cache=cc)

    # (3) Injective hull of C: E1 with inclusion j : C \into E1
    E1, j, gens_at_E1 = _injective_hull(C; cache=cc)

    # Compose to get rho0 : E0 \to E1 (at each vertex i: rho0[i] = j[i] * q[i])
    comps_rho0 = [ j.comps[i] * q.comps[i] for i in 1:n ]
    rho0 = PMorphism{K}(E0, E1, comps_rho0)

    # (4) Assemble the indicator wrapper: labels D0, D1 and the scalar block rho.
    # D0/D1 each contain one principal downset for every generator (u,j) chosen above.
    D0 = Downset[]
    for u in 1:n, _ in gens_at_E0[u]
        push!(D0, principal_downset(Q, u))
    end
    D1 = Downset[]
    for u in 1:n, _ in gens_at_E1[u]
        push!(D1, principal_downset(Q, u))
    end

    # Local index lists: at vertex i, the active generators are those born at u with i <= u.
    function _local_index_list_D(gens_at)
        L = Vector{Vector{Tuple{Int,Int}}}(undef, n)
        for i in 1:n
            lst = Tuple{Int,Int}[]
            for u in upset_indices(Q, i)
                append!(lst, gens_at[u])
            end
            L[i] = lst
        end
        L
    end
    L0 = _local_index_list_D(gens_at_E0)
    L1 = _local_index_list_D(gens_at_E1)

    # Global enumerations (lambda in D0, theta in D1) with their birth vertices u_lambda, u_theta.
    globalD0 = Tuple{Int,Int}[]; for u in 1:n; append!(globalD0, gens_at_E0[u]); end
    globalD1 = Tuple{Int,Int}[]; for u in 1:n; append!(globalD1, gens_at_E1[u]); end

    # Helper to find the local column index of a global generator at vertex i
    function _local_col_of(L, g::Tuple{Int,Int}, i::Int)
        for (c, gg) in enumerate(L[i])
            if gg == g; return c; end
        end
        return 0
    end

    # Assemble the scalar monomial matrix rho by reading the minimal vertex i = u_theta.
    m1 = length(globalD1); m0 = length(globalD0)
    rho = spzeros(K, m1, m0)

    for (lambda, (ulambda, jlambda)) in enumerate(globalD0)
        for (theta, (utheta, jtheta)) in enumerate(globalD1)
            if leq(Q, utheta, ulambda) # D(utheta) subseteq D(ulambda)
                i   = utheta
                col = _local_col_of(L0, (ulambda, jlambda), i)
                row = _local_col_of(L1, (utheta, jtheta), i)
                if col > 0 && row > 0
                    val = rho0.comps[i][row, col]
                    if val != 0
                        rho[theta, lambda] = val
                    end
                end
            end
        end
    end

    return DownsetCopresentation{K}(Q, D0, D1, rho, H)
end



"""
    prune_zero_relations(F::UpsetPresentation{K}) -> UpsetPresentation{K}

Remove rows of `delta` that are identically zero (redundant relations) and drop the
corresponding entries of `U1`. The cokernel is unchanged.
"""
function prune_zero_relations(F::UpsetPresentation{K}) where {K}
    m1, m0 = size(F.delta)
    keep = trues(m1)
    # mark zero rows
    rows, _, _ = findnz(F.delta)
    seen = falses(m1); @inbounds for r in rows; seen[r] = true; end
    @inbounds for r in 1:m1
        if !seen[r]; keep[r] = false; end
    end
    new_U1 = [F.U1[i] for i in 1:m1 if keep[i]]
    new_delta = F.delta[keep, :]
    UpsetPresentation{K}(F.P, F.U0, new_U1, new_delta, F.H)
end

"""
    cancel_isolated_unit_pairs(F::UpsetPresentation{K}) -> UpsetPresentation{K}

Iteratively cancels isolated nonzero entries `delta[theta,lambda]` for which:
  * the theta-th row has exactly that one nonzero,
  * the lambda-th column has exactly that one nonzero, and
  * U1[theta] == U0[lambda] as Upsets (principal upsets match).

Each cancellation removes one generator in U0 and one relation in U1 without
changing the cokernel.
"""
function cancel_isolated_unit_pairs(F::UpsetPresentation{K}) where {K}
    P, U0, U1, Delta = F.P, F.U0, F.U1, F.delta
    while true
        m1, m0 = size(Delta)
        rows, cols, _ = findnz(Delta)
        # count nonzeros per row/col
        rcount = zeros(Int, m1)
        ccount = zeros(Int, m0)
        @inbounds for k in eachindex(rows)
            rcount[rows[k]] += 1; ccount[cols[k]] += 1
        end
        # search an isolated pair with matching principal upsets
        found = false
        theta = 0; lambda = 0
        @inbounds for k in eachindex(rows)
            r = rows[k]; c = cols[k]
            if rcount[r] == 1 && ccount[c] == 1
                # require identical principal upsets
                if U1[r].P === U0[c].P && U1[r].mask == U0[c].mask
                    theta, lambda = r, c; found = true; break
                end
            end
        end
        if !found; break; end
        # remove row theta and column lambda
        keep_rows = trues(m1); keep_rows[theta] = false
        keep_cols = trues(m0); keep_cols[lambda] = false
        U1 = [U1[i] for i in 1:m1 if keep_rows[i]]
        U0 = [U0[j] for j in 1:m0 if keep_cols[j]]
        Delta = Delta[keep_rows, keep_cols]
    end
    UpsetPresentation{K}(P, U0, U1, Delta, F.H)
end

"""
    minimal_upset_presentation_one_step(H::FiniteFringe.FringeModule)
        -> UpsetPresentation{K}

Build a one-step upset presentation and apply safe minimality passes:
1) drop zero relations; 2) cancel isolated isomorphism pairs.
"""
function minimal_upset_presentation_one_step(H::FiniteFringe.FringeModule{K}) where {K}
    F = upset_presentation_one_step(H)     # existing builder
    F = prune_zero_relations(F)
    F = cancel_isolated_unit_pairs(F)
    return F
end


"""
    prune_unused_targets(E::DownsetCopresentation{K}) -> DownsetCopresentation{K}

Drop rows of `rho` that are identically zero (unused target summands in E^1). The kernel is unchanged.
"""
function prune_unused_targets(E::DownsetCopresentation{K}) where {K}
    m1, m0 = size(E.rho)
    keep = trues(m1)
    rows, _, _ = findnz(E.rho)
    seen = falses(m1); @inbounds for r in rows; seen[r] = true; end
    @inbounds for r in 1:m1
        if !seen[r]; keep[r] = false; end
    end
    new_D1 = [E.D1[i] for i in 1:m1 if keep[i]]
    new_rho = E.rho[keep, :]
    DownsetCopresentation{K}(E.P, E.D0, new_D1, new_rho, E.H)
end

"""
    cancel_isolated_unit_pairs(E::DownsetCopresentation{K}) -> DownsetCopresentation{K}

Iteratively cancels isolated nonzero entries `rho[theta,lambda]` with matching principal downsets
(D1[theta] == D0[lambda]) and unique in their row/column.
"""
function cancel_isolated_unit_pairs(E::DownsetCopresentation{K}) where {K}
    P, D0, D1, R = E.P, E.D0, E.D1, E.rho
    while true
        m1, m0 = size(R)
        rows, cols, _ = findnz(R)
        rcount = zeros(Int, m1)
        ccount = zeros(Int, m0)
        @inbounds for k in eachindex(rows)
            rcount[rows[k]] += 1; ccount[cols[k]] += 1
        end
        found = false; theta = 0; lambda = 0
        @inbounds for k in eachindex(rows)
            r = rows[k]; c = cols[k]
            if rcount[r] == 1 && ccount[c] == 1
                if D1[r].P === D0[c].P && D1[r].mask == D0[c].mask
                    theta, lambda = r, c; found = true; break
                end
            end
        end
        if !found; break; end
        keep_rows = trues(m1); keep_rows[theta] = false
        keep_cols = trues(m0); keep_cols[lambda] = false
        D1 = [D1[i] for i in 1:m1 if keep_rows[i]]
        D0 = [D0[j] for j in 1:m0 if keep_cols[j]]
        R = R[keep_rows, keep_cols]
    end
    DownsetCopresentation{K}(P, D0, D1, R, E.H)
end

"""
    minimal_downset_copresentation_one_step(H::FiniteFringe.FringeModule)
        -> DownsetCopresentation{K}

Build a one-step downset copresentation and apply safe minimality passes:
1) drop zero target rows; 2) cancel isolated isomorphism pairs.
"""
function minimal_downset_copresentation_one_step(H::FiniteFringe.FringeModule)
    E = downset_copresentation_one_step(H)
    E = prune_unused_targets(E)
    E = cancel_isolated_unit_pairs(E)
    return E
end



# =============================================================================
# Longer indicator resolutions and high-level Ext driver
# =============================================================================
# We expose:
#   * upset_resolution(H; maxlen)     -> (F, dF)
#   * downset_resolution(H; maxlen)   -> (E, dE)
#   * indicator_resolutions(HM, HN; maxlen) -> (F, dF, E, dE)
#
# The outputs (F, dF, E, dE) are exactly the shapes expected by
# HomExt.build_hom_tot_complex / HomExt.ext_dims_via_resolutions:
#   - F is a Vector{UpsetPresentation{K}} with F[a+1].U0 = U_a
#   - dF[a] is the sparse delta_a : U_a <- U_{a+1}  (shape |U_{a+1}| x |U_a|)
#   - E is a Vector{DownsetCopresentation{K}} with E[b+1].D0 = D_b
#   - dE[b] is the sparse rho_b : D_b -> D_{b+1}    (shape |D_{b+1}| x |D_b|)
#
# Construction mirrors section 6.1 and the one-step routines already present.
# =============================================================================

# ------------------------------ small helpers --------------------------------

# Build the list of principal upsets from per-vertex generator labels returned by
# projective_cover: gens_at[v] is a vector of pairs (p, j).  Each pair contributes
# one principal upset at vertex p.
function _principal_upsets_from_gens(P::AbstractPoset,
                                     gens_at::Vector{Vector{Tuple{Int,Int}}})
    n = nvertices(P)
    U_at = Vector{Upset}(undef, n)
    @inbounds for p in 1:n
        U_at[p] = principal_upset(P, p)
    end
    U = Upset[]
    for p in 1:n
        for _ in gens_at[p]
            push!(U, U_at[p])
        end
    end
    return U
end

# Build the list of principal downsets from per-vertex labels returned by _injective_hull
function _principal_downsets_from_gens(P::AbstractPoset,
                                       gens_at::Vector{Vector{Tuple{Int,Int}}})
    n = nvertices(P)
    D_at = Vector{Downset}(undef, n)
    @inbounds for u in 1:n
        D_at[u] = principal_downset(P, u)
    end
    D = Downset[]
    for u in 1:n
        for _ in gens_at[u]
            push!(D, D_at[u])
        end
    end
    return D
end

# Local basis index lists used by Workflow presentation conversions.
function _local_index_list_up(
    P::AbstractPoset,
    gens_at::Vector{Vector{Tuple{Int,Int}}},
)
    n = nvertices(P)
    L = Vector{Vector{Tuple{Int,Int}}}(undef, n)
    @inbounds for i in 1:n
        lst = Tuple{Int,Int}[]
        for p in downset_indices(P, i)
            append!(lst, gens_at[p])
        end
        L[i] = lst
    end
    return L
end

function _local_index_list_down(
    P::AbstractPoset,
    gens_at::Vector{Vector{Tuple{Int,Int}}},
)
    n = nvertices(P)
    L = Vector{Vector{Tuple{Int,Int}}}(undef, n)
    @inbounds for i in 1:n
        lst = Tuple{Int,Int}[]
        for u in upset_indices(P, i)
            append!(lst, gens_at[u])
        end
        L[i] = lst
    end
    return L
end

"""
    fringe_presentation(M::PModule{K}) -> FringeModule{K}

Construct a canonical fringe presentation whose image recovers `M`.
"""
function fringe_presentation(M::PModule{K}) where {K}
    P = M.Q
    field = M.field
    F0, pi0, gens_at_F0 = projective_cover(M)
    E0, iota0, gens_at_E0 = _injective_hull(M)

    comps = Vector{Matrix{K}}(undef, nvertices(P))
    for i in 1:nvertices(P)
        comps[i] = iota0.comps[i] * pi0.comps[i]
    end
    f = PMorphism{K}(F0, E0, comps)

    U = _principal_upsets_from_gens(P, gens_at_F0)
    D = _principal_downsets_from_gens(P, gens_at_E0)

    L0 = _local_index_list_up(P, gens_at_F0)
    L1 = _local_index_list_down(P, gens_at_E0)

    globalU = Tuple{Int,Int}[]
    for p in 1:nvertices(P)
        append!(globalU, gens_at_F0[p])
    end
    globalD = Tuple{Int,Int}[]
    for u in 1:nvertices(P)
        append!(globalD, gens_at_E0[u])
    end

    function local_pos(L, g::Tuple{Int,Int}, i::Int)
        for (j, gg) in enumerate(L[i])
            gg == g && return j
        end
        return 0
    end

    phi = spzeros(K, length(D), length(U))
    for (lambda, (plambda, jlambda)) in enumerate(globalU)
        for (theta, (ptheta, jtheta)) in enumerate(globalD)
            if leq(P, plambda, ptheta)
                i = plambda
                col = local_pos(L0, (plambda, jlambda), i)
                row = local_pos(L1, (ptheta, jtheta), i)
                if row > 0 && col > 0
                    phi[theta, lambda] = f.comps[i][row, col]
                end
            end
        end
    end

    return FiniteFringe.FringeModule{K}(P, U, D, phi; field=field)
end

@inline function _truncate_steps(maxlen::Union{Int,Nothing}, available::Int)
    return maxlen === nothing ? available : min(available, maxlen)
end

@inline function _upset_birth_block_plan(P::AbstractPoset)
    return _cached_birth_plans(P).downset
end

@inline function _downset_birth_block_plan(P::AbstractPoset)
    return _cached_birth_plans(P).upset
end

@inline function _package_upset_from_state(
    ::Type{K},
    P::AbstractPoset,
    st::_UpsetPrefixState{K},
    maxlen::Union{Int,Nothing},
) where {K}
    nd = _truncate_steps(maxlen, length(st.dF))
    ng = nd + 1
    U_by_a = Vector{Vector{Upset}}(undef, ng)
    @inbounds for a in 1:ng
        U_by_a[a] = _principal_upsets_from_gens(P, st.gens_by_a[a])
    end
    F = Vector{UpsetPresentation{K}}(undef, ng)
    @inbounds for a in 1:ng
        U0 = U_by_a[a]
        if a < ng
            U1 = U_by_a[a + 1]
            delta = st.dF[a]
        else
            U1 = Upset[]
            delta = spzeros(K, 0, length(U0))
        end
        F[a] = UpsetPresentation{K}(P, U0, U1, delta, nothing)
    end
    return F, st.dF[1:nd]
end

@inline function _package_downset_from_state(
    ::Type{K},
    P::AbstractPoset,
    st::_DownsetPrefixState{K},
    maxlen::Union{Int,Nothing},
) where {K}
    nd = _truncate_steps(maxlen, length(st.dE))
    ng = nd + 1
    D_by_b = Vector{Vector{Downset}}(undef, ng)
    @inbounds for b in 1:ng
        D_by_b[b] = _principal_downsets_from_gens(P, st.gens_by_b[b])
    end
    E = Vector{DownsetCopresentation{K}}(undef, ng)
    @inbounds for b in 1:ng
        D0 = D_by_b[b]
        if b < ng
            D1 = D_by_b[b + 1]
            rho_b = st.dE[b]
        else
            D1 = Downset[]
            rho_b = spzeros(K, 0, length(D0))
        end
        E[b] = DownsetCopresentation{K}(P, D0, D1, rho_b, nothing)
    end
    return E, st.dE[1:nd]
end

# ------------------------------ upset resolution ------------------------------

"""
    upset_resolution(H::FiniteFringe.FringeModule{K}; maxlen=nothing)

Compute an upset (projective) indicator resolution of the fringe module `H`.
`maxlen` is a cutoff on the number of differentials computed (it does not pad output).
"""
function _extend_upset_prefix_state!(
    st::_UpsetPrefixState{K},
    P::AbstractPoset,
    birth_plan::Vector{Vector{Int}},
    cc::CoverCache,
    memo_pool::IdDict{Any,Vector{Union{Nothing,Matrix{K}}}},
    ws::_ResolutionWorkspace{K},
    target_steps::Int,
    threads::Bool,
) where {K}
    n = nvertices(P)
    use_vertex_cache = _indicator_use_incremental_vertex_cache(
        n,
        max(sum(st.curr_pi.dom.dims), sum(st.curr_pi.cod.dims)),
    )
    while !st.done && length(st.dF) < target_steps
        Kmod, iota = kernel_with_inclusion(
            st.curr_pi;
            cache=cc,
            incremental_cache=(use_vertex_cache ? ws.kernel_vertex_cache : nothing),
        )
        if sum(Kmod.dims) == 0
            st.done = true
            break
        end
        map_memo_next = _indicator_memo_for_module!(memo_pool, Kmod)
        Fnext, pinext, gens_at_next = projective_cover(
            Kmod;
            cache=cc,
            map_memo=map_memo_next,
            workspace=ws,
            threads=threads,
        )

        iota_comps = iota.comps
        pinext_comps = pinext.comps
        counts_prev = [length(st.curr_gens[p]) for p in 1:n]
        counts_next = [length(gens_at_next[p]) for p in 1:n]
        gid_prev, total_prev = _generator_id_ranges(counts_prev)
        gid_next, total_next = _generator_id_ranges(counts_next)
        starts_next = _fill_birth_self_start_down!(ws.self_start, P, counts_next)

        if threads && Threads.nthreads() > 1
            for tid in 1:length(ws.I_chunks)
                empty!(ws.I_chunks[tid])
                empty!(ws.J_chunks[tid])
                empty!(ws.V_chunks[tid])
            end
            Threads.@threads for ptheta in 1:n
                ctheta = counts_next[ptheta]
                ctheta == 0 && continue
                tid = _thread_local_index(ws.I_chunks)
                I_t = ws.I_chunks[tid]
                J_t = ws.J_chunks[tid]
                V_t = ws.V_chunks[tid]
                i = ptheta
                Ai = iota_comps[i]
                Bi = pinext_comps[i]
                theta_gid0 = first(gid_next[ptheta])
                plist = birth_plan[ptheta]
                col0 = starts_next[ptheta]
                row0 = 1
                @inbounds for plambda in plist
                    clambda = counts_prev[plambda]
                    clambda == 0 && continue
                    lambda_gid0 = first(gid_prev[plambda])
                    _accumulate_product_entries_upset!(
                        I_t, J_t, V_t,
                        Ai, Bi,
                        row0, col0, clambda, ctheta,
                        theta_gid0, lambda_gid0,
                    )
                    row0 += clambda
                end
            end
            empty!(ws.I)
            empty!(ws.J)
            empty!(ws.V)
            for tid in 1:length(ws.I_chunks)
                append!(ws.I, ws.I_chunks[tid])
                append!(ws.J, ws.J_chunks[tid])
                append!(ws.V, ws.V_chunks[tid])
            end
        else
            empty!(ws.I)
            empty!(ws.J)
            empty!(ws.V)
            for ptheta in 1:n
                ctheta = counts_next[ptheta]
                ctheta == 0 && continue
                i = ptheta
                Ai = iota_comps[i]
                Bi = pinext_comps[i]
                theta_gid0 = first(gid_next[ptheta])
                plist = birth_plan[ptheta]
                col0 = starts_next[ptheta]
                row0 = 1
                @inbounds for plambda in plist
                    clambda = counts_prev[plambda]
                    clambda == 0 && continue
                    lambda_gid0 = first(gid_prev[plambda])
                    _accumulate_product_entries_upset!(
                        ws.I, ws.J, ws.V,
                        Ai, Bi,
                        row0, col0, clambda, ctheta,
                        theta_gid0, lambda_gid0,
                    )
                    row0 += clambda
                end
            end
        end
        delta = _sparse_from_workspace!(ws, total_next, total_prev)
        push!(st.dF, delta)
        push!(st.gens_by_a, gens_at_next)
        st.curr_pi = pinext
        st.curr_gens = gens_at_next
    end
    return st
end

function _extend_downset_prefix_state!(
    st::_DownsetPrefixState{K},
    P::AbstractPoset,
    birth_plan::Vector{Vector{Int}},
    cc::CoverCache,
    memo_pool::IdDict{Any,Vector{Union{Nothing,Matrix{K}}}},
    ws::_ResolutionWorkspace{K},
    target_steps::Int,
    threads::Bool,
) where {K}
    n = nvertices(P)
    while !st.done && length(st.dE) < target_steps
        if sum(st.C.dims) == 0
            st.done = true
            break
        end

        map_memo_next = _indicator_memo_for_module!(memo_pool, st.C)
        E1, j, gens_at_E1 = _injective_hull(
            st.C;
            cache=cc,
            map_memo=map_memo_next,
            workspace=ws,
            threads=threads,
        )

        j_comps = j.comps
        q_comps = st.prev_q.comps
        counts_prev = [length(st.prev_gens[u]) for u in 1:n]
        counts_next = [length(gens_at_E1[u]) for u in 1:n]
        gid_prev, total_prev = _generator_id_ranges(counts_prev)
        gid_next, total_next = _generator_id_ranges(counts_next)
        starts_next = _fill_birth_self_start_up!(ws.self_start, P, counts_next)

        if threads && Threads.nthreads() > 1
            for tid in 1:length(ws.I_chunks)
                empty!(ws.I_chunks[tid])
                empty!(ws.J_chunks[tid])
                empty!(ws.V_chunks[tid])
            end
            Threads.@threads for utheta in 1:n
                ctheta = counts_next[utheta]
                ctheta == 0 && continue
                tid = _thread_local_index(ws.I_chunks)
                I_t = ws.I_chunks[tid]
                J_t = ws.J_chunks[tid]
                V_t = ws.V_chunks[tid]
                i = utheta
                Ai = j_comps[i]
                Bi = q_comps[i]
                theta_gid0 = first(gid_next[utheta])
                ulist = birth_plan[utheta]
                row0 = starts_next[utheta]
                col0 = 1
                @inbounds for ulambda in ulist
                    clambda = counts_prev[ulambda]
                    clambda == 0 && continue
                    lambda_gid0 = first(gid_prev[ulambda])
                    _accumulate_product_entries_downset!(
                        I_t, J_t, V_t,
                        Ai, Bi,
                        row0, col0, ctheta, clambda,
                        theta_gid0, lambda_gid0,
                    )
                    col0 += clambda
                end
            end
            empty!(ws.I)
            empty!(ws.J)
            empty!(ws.V)
            for tid in 1:length(ws.I_chunks)
                append!(ws.I, ws.I_chunks[tid])
                append!(ws.J, ws.J_chunks[tid])
                append!(ws.V, ws.V_chunks[tid])
            end
        else
            empty!(ws.I)
            empty!(ws.J)
            empty!(ws.V)
            for utheta in 1:n
                ctheta = counts_next[utheta]
                ctheta == 0 && continue
                i = utheta
                Ai = j_comps[i]
                Bi = q_comps[i]
                theta_gid0 = first(gid_next[utheta])
                ulist = birth_plan[utheta]
                row0 = starts_next[utheta]
                col0 = 1
                @inbounds for ulambda in ulist
                    clambda = counts_prev[ulambda]
                    clambda == 0 && continue
                    lambda_gid0 = first(gid_prev[ulambda])
                    _accumulate_product_entries_downset!(
                        ws.I, ws.J, ws.V,
                        Ai, Bi,
                        row0, col0, ctheta, clambda,
                        theta_gid0, lambda_gid0,
                    )
                    col0 += clambda
                end
            end
        end

        Rh = _sparse_from_workspace!(ws, total_next, total_prev)
        push!(st.dE, Rh)
        push!(st.gens_by_b, gens_at_E1)

        Cnext, q_next = _cokernel_module(
            j;
            cache=cc,
            incremental_cache=(INDICATOR_INCREMENTAL_LINALG[] ? ws.cokernel_vertex_cache : nothing),
        )
        st.C = Cnext
        st.prev_q = q_next
        st.prev_gens = gens_at_E1
        st.done = sum(Cnext.dims) == 0
    end
    return st
end

function upset_resolution(H::FiniteFringe.FringeModule{K};
                          maxlen::Union{Int,Nothing}=nothing,
                          cache::Union{Nothing,CoverCache}=nothing,
                          map_memo::Union{Nothing,AbstractVector{Union{Nothing,Matrix{K}}}}=nothing,
                          map_memo_pool::Union{Nothing,IdDict{Any,Vector{Union{Nothing,Matrix{K}}}}}=nothing,
                          workspace::Union{Nothing,_ResolutionWorkspace{K}}=nothing,
                          threads::Bool = (Threads.nthreads() > 1)) where {K}
    return upset_resolution(
        pmodule_from_fringe(H);
        maxlen=maxlen,
        cache=cache,
        map_memo=map_memo,
        map_memo_pool=map_memo_pool,
        workspace=workspace,
        threads=threads,
    )
end


"""
upset_resolution(M::PModule{K}; maxlen::Union{Int,Nothing}=nothing)
        -> (F::Vector{UpsetPresentation{K}}, dF::Vector{SparseMatrixCSC{K,Int}})

Overload of `upset_resolution` for an already-constructed finite-poset module `M`.

This is the same construction as `upset_resolution(::FiniteFringe.FringeModule{K})`,
but it skips the `pmodule_from_fringe` conversion step. This matters when callers
already have a `PModule{K}` (for example, after encoding a Z^n or R^n module to a
finite poset, or after explicitly calling `pmodule_from_fringe`).
"""
function upset_resolution(M::PModule{K};
                          maxlen::Union{Int,Nothing}=nothing,
                          cache::Union{Nothing,CoverCache}=nothing,
                          map_memo::Union{Nothing,AbstractVector{Union{Nothing,Matrix{K}}}}=nothing,
                          map_memo_pool::Union{Nothing,IdDict{Any,Vector{Union{Nothing,Matrix{K}}}}}=nothing,
                          workspace::Union{Nothing,_ResolutionWorkspace{K}}=nothing,
                          threads::Bool = (Threads.nthreads() > 1)) where {K}
    field = M.field
    P = M.Q
    n = nvertices(P)
    total_dims = sum(M.dims)
    build_cache!(P; cover=true, updown=true)
    cc = cache === nothing ? _get_cover_cache(P) : cache
    use_vertex_cache = _indicator_use_incremental_vertex_cache(n, total_dims)
    memo_pool = map_memo_pool === nothing ? IdDict{Any,Vector{Union{Nothing,Matrix{K}}}}() : map_memo_pool
    if map_memo !== nothing
        memo_pool[M] = map_memo
    end
    map_memo_local = _indicator_memo_for_module!(memo_pool, M)
    ws = workspace === nothing ? _new_resolution_workspace(K, n) : workspace
    _workspace_prepare!(ws, n)
    birth_plan = _upset_birth_block_plan(P)

    use_prefix_cache = INDICATOR_PREFIX_CACHE_ENABLED[] &&
                       maxlen !== nothing &&
                       cache === nothing &&
                       map_memo === nothing &&
                       map_memo_pool === nothing &&
                       workspace === nothing

    if use_prefix_cache
        target_steps = Int(maxlen)
        fam = _indicator_prefix_family(K)
        shard = _indicator_shard_index(M, length(fam.upset_shards))
        Base.lock(fam.upset_locks[shard])
        try
            st_any = get(fam.upset_shards[shard], M, nothing)
            st = if st_any === nothing || !(st_any isa _UpsetPrefixState{K})
                F0, pi0, gens_at_F0 = projective_cover(
                    M;
                    cache=cc,
                    map_memo=map_memo_local,
                    workspace=ws,
                    threads=threads,
                )
                _ = F0
                st_new = _UpsetPrefixState{K}(
                    [gens_at_F0],
                    SparseMatrixCSC{K,Int}[],
                    pi0,
                    gens_at_F0,
                    false,
                )
                fam.upset_shards[shard][M] = st_new
                st_new
            else
                st_any::_UpsetPrefixState{K}
            end

            _extend_upset_prefix_state!(st, P, birth_plan, cc, memo_pool, ws, target_steps, threads)
            return _package_upset_from_state(K, P, st, maxlen)
        finally
            Base.unlock(fam.upset_locks[shard])
        end
    end

    # First projective cover: F0 --pi0--> M, with labels gens_at_F0
    F0, pi0, gens_at_F0 = projective_cover(M; cache=cc, map_memo=map_memo_local, workspace=ws, threads=threads)

    # Keep generator labels through the loop and materialize principal upsets only once at packaging.
    gens_by_a = Vector{Vector{Vector{Tuple{Int,Int}}}}()
    push!(gens_by_a, gens_at_F0)

    dF = Vector{SparseMatrixCSC{K,Int}}()

    # Iteration state
    curr_dom  = F0
    curr_pi   = pi0
    curr_gens = gens_at_F0
    steps = 0

    while true
        if maxlen !== nothing && steps >= maxlen
            break
        end

        # Next kernel and projective cover
        Kmod, iota = kernel_with_inclusion(
            curr_pi;
            cache=cc,
            incremental_cache=(use_vertex_cache ? ws.kernel_vertex_cache : nothing),
        )   # ker(F_prev -> ...)
        if sum(Kmod.dims) == 0
            break
        end
        map_memo_next = _indicator_memo_for_module!(memo_pool, Kmod)
        Fnext, pinext, gens_at_next = projective_cover(Kmod; cache=cc, map_memo=map_memo_next, workspace=ws, threads=threads)

        # d = iota o pinext : Fnext -> curr_dom
        # Build sparse delta entries directly from required comparable blocks.
        iota_comps = iota.comps
        pinext_comps = pinext.comps

        # Dense local-index starts (born-at-vertex blocks).
        counts_prev = [length(curr_gens[p]) for p in 1:n]
        counts_next = [length(gens_at_next[p]) for p in 1:n]
        gid_prev, total_prev = _generator_id_ranges(counts_prev)
        gid_next, total_next = _generator_id_ranges(counts_next)
        starts_next = _fill_birth_self_start_down!(ws.self_start, P, counts_next)

        # Assemble sparse delta: rows index next, cols index prev
        if threads && Threads.nthreads() > 1
            for tid in 1:length(ws.I_chunks)
                empty!(ws.I_chunks[tid])
                empty!(ws.J_chunks[tid])
                empty!(ws.V_chunks[tid])
            end
            Threads.@threads for ptheta in 1:n
                ctheta = counts_next[ptheta]
                ctheta == 0 && continue
                tid = _thread_local_index(ws.I_chunks)
                I_t = ws.I_chunks[tid]
                J_t = ws.J_chunks[tid]
                V_t = ws.V_chunks[tid]
                i = ptheta
                Ai = iota_comps[i]
                Bi = pinext_comps[i]
                theta_gid0 = first(gid_next[ptheta])
                plist = birth_plan[ptheta]
                col0 = starts_next[ptheta]
                row0 = 1
                @inbounds for plambda in plist
                    clambda = counts_prev[plambda]
                    if clambda == 0
                        continue
                    end
                    lambda_gid0 = first(gid_prev[plambda])
                    _accumulate_product_entries_upset!(
                        I_t, J_t, V_t,
                        Ai, Bi,
                        row0, col0, clambda, ctheta,
                        theta_gid0, lambda_gid0,
                    )
                    row0 += clambda
                end
            end
            empty!(ws.I)
            empty!(ws.J)
            empty!(ws.V)
            for tid in 1:length(ws.I_chunks)
                append!(ws.I, ws.I_chunks[tid])
                append!(ws.J, ws.J_chunks[tid])
                append!(ws.V, ws.V_chunks[tid])
            end
            delta = _sparse_from_workspace!(ws, total_next, total_prev)
        else
            empty!(ws.I)
            empty!(ws.J)
            empty!(ws.V)
            for ptheta in 1:n
                ctheta = counts_next[ptheta]
                ctheta == 0 && continue
                i = ptheta
                Ai = iota_comps[i]
                Bi = pinext_comps[i]
                theta_gid0 = first(gid_next[ptheta])
                plist = birth_plan[ptheta]
                col0 = starts_next[ptheta]
                row0 = 1
                @inbounds for plambda in plist
                    clambda = counts_prev[plambda]
                    if clambda == 0
                        continue
                    end
                    lambda_gid0 = first(gid_prev[plambda])
                    _accumulate_product_entries_upset!(
                        ws.I, ws.J, ws.V,
                        Ai, Bi,
                        row0, col0, clambda, ctheta,
                        theta_gid0, lambda_gid0,
                    )
                    row0 += clambda
                end
            end
            delta = _sparse_from_workspace!(ws, total_next, total_prev)
        end

        push!(dF, delta)
        push!(gens_by_a, gens_at_next)

        # Advance
        curr_dom  = Fnext
        curr_pi   = pinext
        curr_gens = gens_at_next
        steps += 1
    end

    # Materialize principal upsets only once when packaging output.
    U_by_a = Vector{Vector{Upset}}(undef, length(gens_by_a))
    @inbounds for a in eachindex(gens_by_a)
        U_by_a[a] = _principal_upsets_from_gens(P, gens_by_a[a])
    end

    # Package as UpsetPresentation list.
    F = Vector{UpsetPresentation{K}}(undef, length(gens_by_a))
    for a in 1:length(gens_by_a)
        U0 = U_by_a[a]
        if a < length(gens_by_a)
            U1 = U_by_a[a + 1]
            delta = dF[a]
        else
            U1 = Upset[]
            delta = spzeros(K, 0, length(U0))
        end
        F[a] = UpsetPresentation{K}(P, U0, U1, delta, nothing)
    end

    return F, dF
end


# ---------------------------- downset resolution ------------------------------

"""
    downset_resolution(H::FiniteFringe.FringeModule{K}; maxlen=nothing)

Compute a downset (injective) indicator resolution of the fringe module `H`.
`maxlen` is a cutoff on the number of differentials computed (it does not pad output).
"""
function downset_resolution(H::FiniteFringe.FringeModule{K};
                            maxlen::Union{Int,Nothing}=nothing,
                            cache::Union{Nothing,CoverCache}=nothing,
                            map_memo::Union{Nothing,AbstractVector{Union{Nothing,Matrix{K}}}}=nothing,
                            map_memo_pool::Union{Nothing,IdDict{Any,Vector{Union{Nothing,Matrix{K}}}}}=nothing,
                            workspace::Union{Nothing,_ResolutionWorkspace{K}}=nothing,
                            threads::Bool = (Threads.nthreads() > 1)) where {K}
    return downset_resolution(
        pmodule_from_fringe(H);
        maxlen=maxlen,
        cache=cache,
        map_memo=map_memo,
        map_memo_pool=map_memo_pool,
        workspace=workspace,
        threads=threads,
    )
end


"""
downset_resolution(M::PModule{K}; maxlen::Union{Int,Nothing}=nothing)
        -> (E::Vector{DownsetCopresentation{K}}, dE::Vector{SparseMatrixCSC{K,Int}})

Overload of `downset_resolution` for an already-constructed finite-poset module `M`.

This is the same construction as `downset_resolution(::FiniteFringe.FringeModule{K})`,
but it skips the `pmodule_from_fringe` conversion step.
"""
function downset_resolution(M::PModule{K};
                            maxlen::Union{Int,Nothing}=nothing,
                            cache::Union{Nothing,CoverCache}=nothing,
                            map_memo::Union{Nothing,AbstractVector{Union{Nothing,Matrix{K}}}}=nothing,
                            map_memo_pool::Union{Nothing,IdDict{Any,Vector{Union{Nothing,Matrix{K}}}}}=nothing,
                            workspace::Union{Nothing,_ResolutionWorkspace{K}}=nothing,
                            threads::Bool = (Threads.nthreads() > 1)) where {K}
    field = M.field
    P = M.Q
    n = nvertices(P)
    build_cache!(P; cover=true, updown=true)
    cc = cache === nothing ? _get_cover_cache(P) : cache
    memo_pool = map_memo_pool === nothing ? IdDict{Any,Vector{Union{Nothing,Matrix{K}}}}() : map_memo_pool
    if map_memo !== nothing
        memo_pool[M] = map_memo
    end
    map_memo_local = _indicator_memo_for_module!(memo_pool, M)
    ws = workspace === nothing ? _new_resolution_workspace(K, n) : workspace
    _workspace_prepare!(ws, n)
    birth_plan = _downset_birth_block_plan(P)

    use_prefix_cache = INDICATOR_PREFIX_CACHE_ENABLED[] &&
                       maxlen !== nothing &&
                       cache === nothing &&
                       map_memo === nothing &&
                       map_memo_pool === nothing &&
                       workspace === nothing

    if use_prefix_cache
        target_steps = Int(maxlen)
        fam = _indicator_prefix_family(K)
        shard = _indicator_shard_index(M, length(fam.downset_shards))
        Base.lock(fam.downset_locks[shard])
        try
            st_any = get(fam.downset_shards[shard], M, nothing)
            st = if st_any === nothing || !(st_any isa _DownsetPrefixState{K})
                E0, iota0, gens_at_E0 = _injective_hull(
                    M;
                    cache=cc,
                    map_memo=map_memo_local,
                    workspace=ws,
                    threads=threads,
                )
                _ = E0
                C0, q0 = _cokernel_module(
                    iota0;
                    cache=cc,
                    incremental_cache=(INDICATOR_INCREMENTAL_LINALG[] ? ws.cokernel_vertex_cache : nothing),
                )
                st_new = _DownsetPrefixState{K}(
                    [gens_at_E0],
                    SparseMatrixCSC{K,Int}[],
                    C0,
                    q0,
                    gens_at_E0,
                    sum(C0.dims) == 0,
                )
                fam.downset_shards[shard][M] = st_new
                st_new
            else
                st_any::_DownsetPrefixState{K}
            end

            _extend_downset_prefix_state!(st, P, birth_plan, cc, memo_pool, ws, target_steps, threads)
            return _package_downset_from_state(K, P, st, maxlen)
        finally
            Base.unlock(fam.downset_locks[shard])
        end
    end

    # First injective hull: iota0 : M -> E0
    E0, iota0, gens_at_E0 = _injective_hull(M; cache=cc, map_memo=map_memo_local, workspace=ws, threads=threads)
    # Keep generator labels through the loop and materialize principal downsets only once at packaging.
    gens_by_b = Vector{Vector{Vector{Tuple{Int,Int}}}}()
    push!(gens_by_b, gens_at_E0)

    dE = Vector{SparseMatrixCSC{K,Int}}()

    # First cokernel
    C, q = _cokernel_module(
        iota0;
        cache=cc,
        incremental_cache=(INDICATOR_INCREMENTAL_LINALG[] ? ws.cokernel_vertex_cache : nothing),
    )

    # Iteration state
    prev_E    = E0
    prev_gens = gens_at_E0
    prev_q    = q
    steps = 0

    while true
        if sum(C.dims) == 0
            break
        end
        if maxlen !== nothing && steps >= maxlen
            break
        end

        # Injective hull of the current cokernel: j : C -> E1
        map_memo_next = _indicator_memo_for_module!(memo_pool, C)
        E1, j, gens_at_E1 = _injective_hull(C; cache=cc, map_memo=map_memo_next, workspace=ws, threads=threads)

        # rho_b = j o prev_q : prev_E -> E1
        # Build sparse rho entries directly from comparable birth blocks.
        j_comps = j.comps
        q_comps = prev_q.comps

        # Dense local-index starts (born-at-vertex blocks).
        counts_prev = [length(prev_gens[u]) for u in 1:n]
        counts_next = [length(gens_at_E1[u]) for u in 1:n]
        gid_prev, total_prev = _generator_id_ranges(counts_prev)
        gid_next, total_next = _generator_id_ranges(counts_next)
        starts_next = _fill_birth_self_start_up!(ws.self_start, P, counts_next)

        # Assemble sparse rho: rows index D1 (next), cols index D0 (prev)
        if threads && Threads.nthreads() > 1
            for tid in 1:length(ws.I_chunks)
                empty!(ws.I_chunks[tid])
                empty!(ws.J_chunks[tid])
                empty!(ws.V_chunks[tid])
            end
            Threads.@threads for utheta in 1:n
                ctheta = counts_next[utheta]
                ctheta == 0 && continue
                tid = _thread_local_index(ws.I_chunks)
                I_t = ws.I_chunks[tid]
                J_t = ws.J_chunks[tid]
                V_t = ws.V_chunks[tid]
                i = utheta
                Ai = j_comps[i]
                Bi = q_comps[i]
                theta_gid0 = first(gid_next[utheta])
                ulist = birth_plan[utheta]
                row0 = starts_next[utheta]
                col0 = 1
                @inbounds for ulambda in ulist
                    clambda = counts_prev[ulambda]
                    if clambda == 0
                        continue
                    end
                    lambda_gid0 = first(gid_prev[ulambda])
                    _accumulate_product_entries_downset!(
                        I_t, J_t, V_t,
                        Ai, Bi,
                        row0, col0, ctheta, clambda,
                        theta_gid0, lambda_gid0,
                    )
                    col0 += clambda
                end
            end
            empty!(ws.I)
            empty!(ws.J)
            empty!(ws.V)
            for tid in 1:length(ws.I_chunks)
                append!(ws.I, ws.I_chunks[tid])
                append!(ws.J, ws.J_chunks[tid])
                append!(ws.V, ws.V_chunks[tid])
            end
            Rh = _sparse_from_workspace!(ws, total_next, total_prev)
        else
            empty!(ws.I)
            empty!(ws.J)
            empty!(ws.V)
            for utheta in 1:n
                ctheta = counts_next[utheta]
                ctheta == 0 && continue
                i = utheta
                Ai = j_comps[i]
                Bi = q_comps[i]
                theta_gid0 = first(gid_next[utheta])
                ulist = birth_plan[utheta]
                row0 = starts_next[utheta]
                col0 = 1
                @inbounds for ulambda in ulist
                    clambda = counts_prev[ulambda]
                    if clambda == 0
                        continue
                    end
                    lambda_gid0 = first(gid_prev[ulambda])
                    _accumulate_product_entries_downset!(
                        ws.I, ws.J, ws.V,
                        Ai, Bi,
                        row0, col0, ctheta, clambda,
                        theta_gid0, lambda_gid0,
                    )
                    col0 += clambda
                end
            end
            Rh = _sparse_from_workspace!(ws, total_next, total_prev)
        end

        push!(dE, Rh)
        push!(gens_by_b, gens_at_E1)

        # Next cokernel and advance
        C, q_next = _cokernel_module(
            j;
            cache=cc,
            incremental_cache=(INDICATOR_INCREMENTAL_LINALG[] ? ws.cokernel_vertex_cache : nothing),
        )
        prev_E    = E1
        prev_gens = gens_at_E1
        prev_q    = q_next
        steps += 1
    end

    # Materialize principal downsets only once when packaging output.
    D_by_b = Vector{Vector{Downset}}(undef, length(gens_by_b))
    @inbounds for b in eachindex(gens_by_b)
        D_by_b[b] = _principal_downsets_from_gens(P, gens_by_b[b])
    end

    # Package as DownsetCopresentation list.
    E = Vector{DownsetCopresentation{K}}(undef, length(gens_by_b))
    for b in 1:length(gens_by_b)
        D0 = D_by_b[b]
        if b < length(gens_by_b)
            D1 = D_by_b[b + 1]
            rho_b = dE[b]
        else
            D1 = Downset[]
            rho_b = spzeros(K, 0, length(D0))
        end
        E[b] = DownsetCopresentation{K}(P, D0, D1, rho_b, nothing)
    end

    return E, dE
end


# --------------------- aggregator + high-level Ext driver ---------------------

"""
    indicator_resolutions(HM, HN; maxlen=nothing)
        -> (F, dF, E, dE)

Convenience wrapper: build an upset resolution for the source module (fringe HM)
and a downset resolution for the target module (fringe HN).  The `maxlen` keyword
cuts off each side after that many steps (useful for quick tests).
"""
@inline function _indicator_resolutions_from_pmodules(
    MM::PModule{K},
    NN::PModule{K};
    maxlen::Union{Int,Nothing}=nothing,
    threads::Bool = (Threads.nthreads() > 1),
) where {K}
    PM = MM.Q
    PN = NN.Q
    build_cache!(PM; cover=true, updown=true)
    ccM = _get_cover_cache(PM)
    if PN === PM
        ccN = ccM
    else
        build_cache!(PN; cover=true, updown=true)
        ccN = _get_cover_cache(PN)
    end

    ws_M = _new_resolution_workspace(K, nvertices(PM))
    map_memo_M = _indicator_new_array_memo(K, nvertices(PM))
    map_memo_N = NN === MM ? map_memo_M : _indicator_new_array_memo(K, nvertices(PN))
    memo_pool = IdDict{Any,Vector{Union{Nothing,Matrix{K}}}}()
    F, dF = upset_resolution(
        MM;
        maxlen=maxlen,
        cache=ccM,
        map_memo=map_memo_M,
        map_memo_pool=memo_pool,
        workspace=ws_M,
        threads=threads,
    )

    ws_N = PN === PM ? ws_M : _new_resolution_workspace(K, nvertices(PN))
    E, dE = downset_resolution(
        NN;
        maxlen=maxlen,
        cache=ccN,
        map_memo=map_memo_N,
        map_memo_pool=memo_pool,
        workspace=ws_N,
        threads=threads,
    )

    return (F, dF, E, dE)
end

function indicator_resolutions(HM::FiniteFringe.FringeModule{K},
                               HN::FiniteFringe.FringeModule{K};
                               maxlen::Union{Int,Nothing}=nothing,
                               threads::Bool = (Threads.nthreads() > 1),
                               cache::Union{Nothing,ResolutionCache}=nothing) where {K}
    key = cache === nothing ? nothing : _resolution_key3(HM, HN, maxlen === nothing ? -1 : Int(maxlen))

    if cache !== nothing
        cache_val_type = Tuple{
            Vector{UpsetPresentation{K}},
            Vector{SparseMatrixCSC{K,Int}},
            Vector{DownsetCopresentation{K}},
            Vector{SparseMatrixCSC{K,Int}},
        }

        cached = _resolution_cache_indicator_get(cache, key, cache_val_type)
        cached === nothing || return cached
    end

    MM = pmodule_from_fringe(HM)
    NN = HM === HN ? MM : pmodule_from_fringe(HN)
    out = _indicator_resolutions_from_pmodules(
        MM,
        NN;
        maxlen=maxlen,
        threads=threads,
    )

    if cache !== nothing
        return _resolution_cache_indicator_store!(cache, key::ResolutionKey3, out)
    end
    return out
end

# --------------------- resolution verification (structural checks) ---------------------

"""
    verify_upset_resolution(F, dF; vertices=:all,
                            check_d2=true,
                            check_exactness=true,
                            check_connected=true)

Structural checks for an upset (projective) indicator resolution:
  * d^2 = 0 (as coefficient matrices),
  * exactness at intermediate stages (vertexwise),
  * each nonzero entry corresponds to a connected homomorphism in the principal-upset case,
    i.e. support inclusion U_{a+1} subseteq U_a.

Throws an error if a check fails; returns true otherwise.
"""
function verify_upset_resolution(F::Vector{UpsetPresentation{K}},
                                dF::Vector{SparseMatrixCSC{K,Int}};
                                vertices = :all,
                                check_d2::Bool = true,
                                check_exactness::Bool = true,
                                check_connected::Bool = true) where {K}
    field = (F[1].H === nothing) ? field_from_eltype(K) : F[1].H.field

    @assert length(dF) == length(F) - 1 "verify_upset_resolution: expected length(dF) == length(F)-1"
    P = F[1].P
    for f in F
        @assert f.P === P "verify_upset_resolution: all presentations must use the same poset"
    end

    U_by_a = [f.U0 for f in F]  # U_0,...,U_A
    vs = (vertices === :all) ? (1:nvertices(P)) : vertices


    # Connectedness / valid monomial support (principal-upset case):
    # a nonzero entry for k[Udom] -> k[Ucod] is only allowed when Udom subseteq Ucod.
    if check_connected
        for a in 1:length(dF)
            delta = dF[a]                 # rows: U_{a+1}, cols: U_a
            Udom  = U_by_a[a+1]
            Ucod  = U_by_a[a]
            for col in 1:size(delta,2)
                for ptr in delta.colptr[col]:(delta.colptr[col+1]-1)
                    row = delta.rowval[ptr]
                    val = delta.nzval[ptr]
                    if !iszero(val) && !FiniteFringe.is_subset(Udom[row], Ucod[col])
                        error("Upset resolution: nonzero delta at (row=$row,col=$col) but U_{a+1}[row] not subset of U_a[col] (a=$(a-1))")
                    end
                end
            end
        end
    end

    # d^2 = 0
    if check_d2 && length(dF) >= 2
        for a in 1:(length(dF)-1)
            C = dF[a+1] * dF[a]
            dropzeros!(C)
            if !_is_zero_matrix(field, C)
                error("Upset resolution: dF[$(a+1)]*dF[$a] != 0 (nnz=$(nnz(C)))")
            end
        end
    end

    # Vertexwise exactness at intermediate stages:
    # For each q, check rank(d_{a}) + rank(d_{a+1}) = dim(F_a(q)).
    if check_exactness && length(dF) >= 2
        for q in vs
            for k in 2:(length(U_by_a)-1)  # check exactness at U_k (k=1..A-1)
                active_k   = findall(u -> u.mask[q], U_by_a[k])
                dim_mid = length(active_k)
                if dim_mid == 0
                    continue
                end
                active_km1 = findall(u -> u.mask[q], U_by_a[k-1])
                active_kp1 = findall(u -> u.mask[q], U_by_a[k+1])

                delta_prev = dF[k-1]  # U_k -> U_{k-1}
                delta_next = dF[k]    # U_{k+1} -> U_k

                r_prev = isempty(active_km1) ? 0 : _rank_restricted_field(field, delta_prev, active_k, active_km1)
                r_next = isempty(active_kp1) ? 0 : _rank_restricted_field(field, delta_next, active_kp1, active_k)

                if r_prev + r_next != dim_mid
                    error("Upset resolution: vertex q=$q fails exactness at degree k=$(k-1): rank(prev)=$r_prev, rank(next)=$r_next, dim=$dim_mid")
                end
            end
        end
    end

    return true
end


"""
    verify_downset_resolution(E, dE; vertices=:all,
                              check_d2=true,
                              check_exactness=true,
                              check_connected=true)

Structural checks for a downset (injective) indicator resolution:
  * d^2 = 0,
  * exactness at intermediate stages (vertexwise),
  * valid monomial support (principal-downset case): nonzero entry implies D_b subseteq D_{b+1}.

Throws an error if a check fails; returns true otherwise.
"""
function verify_downset_resolution(E::Vector{DownsetCopresentation{K}},
                                  dE::Vector{SparseMatrixCSC{K,Int}};
                                  vertices = :all,
                                  check_d2::Bool = true,
                                  check_exactness::Bool = true,
                                  check_connected::Bool = true) where {K}
    field = (E[1].H === nothing) ? field_from_eltype(K) : E[1].H.field

    @assert length(dE) == length(E) - 1 "verify_downset_resolution: expected length(dE) == length(E)-1"
    P = E[1].P
    for e in E
        @assert e.P === P "verify_downset_resolution: all copresentations must use the same poset"
    end

    D_by_b = [e.D0 for e in E]  # D_0,...,D_B
    vs = (vertices === :all) ? (1:nvertices(P)) : vertices


    # Valid monomial support (principal-downset case):
    # rho has rows in D_{b+1} and cols in D_b, so nonzero implies D_b subseteq D_{b+1}.
    if check_connected
        for b in 1:length(dE)
            rho = dE[b]                 # rows: D_{b+1}, cols: D_b
            Ddom = D_by_b[b]
            Dcod = D_by_b[b+1]
            for col in 1:size(rho,2)
                for ptr in rho.colptr[col]:(rho.colptr[col+1]-1)
                    row = rho.rowval[ptr]
                    val = rho.nzval[ptr]
                    if !iszero(val) && !FiniteFringe.is_subset(Dcod[row], Ddom[col])
                        error("Downset resolution: nonzero rho at (row=$row,col=$col) but D_{b+1}[row] not subset of D_b[col] (b=$(b-1))")
                    end
                end
            end
        end
    end

    # d^2 = 0
    if check_d2 && length(dE) >= 2
        for b in 1:(length(dE)-1)
            C = dE[b+1] * dE[b]
            dropzeros!(C)
            if !_is_zero_matrix(field, C)
                error("Downset resolution: dE[$(b+1)]*dE[$b] != 0 (nnz=$(nnz(C)))")
            end
        end
    end

    # Vertexwise exactness at intermediate stages:
    if check_exactness && length(dE) >= 2
        for q in vs
            for k in 2:(length(D_by_b)-1)  # check exactness at D_k (k=1..B-1)
                active_k   = findall(d -> d.mask[q], D_by_b[k])
                dim_mid = length(active_k)
                if dim_mid == 0
                    continue
                end
                active_km1 = findall(d -> d.mask[q], D_by_b[k-1])
                active_kp1 = findall(d -> d.mask[q], D_by_b[k+1])

                rho_prev = dE[k-1]  # D_{k-1} -> D_k
                rho_next = dE[k]    # D_k -> D_{k+1}

                r_prev = isempty(active_km1) ? 0 : _rank_restricted_field(field, rho_prev, active_k, active_km1)
                r_next = isempty(active_kp1) ? 0 : _rank_restricted_field(field, rho_next, active_kp1, active_k)

                if r_prev + r_next != dim_mid
                    error("Downset resolution: vertex q=$q fails exactness at degree k=$(k-1): rank(prev)=$r_prev, rank(next)=$r_next, dim=$dim_mid")
                end
            end
        end
    end

    return true
end

end # module
