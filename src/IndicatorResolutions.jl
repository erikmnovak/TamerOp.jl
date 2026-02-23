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
using ..CoreModules: AbstractCoeffField, RealField, ResolutionCache, ResolutionKey3, _resolution_key3, coeff_type, eye, field_from_eltype
using ..FieldLinAlg

using ..Modules: CoverCache, _get_cover_cache, _clear_cover_cache!,
                 CoverEdgeMapStore, _find_sorted_index,
                 PModule, PMorphism, dim_at,
                 zero_pmodule, zero_morphism, map_leq, id_morphism
import ..Modules: id_morphism

import ..AbelianCategories
using ..AbelianCategories: kernel_with_inclusion, _cokernel_module

import ..FiniteFringe: AbstractPoset, FinitePoset, Upset, Downset,
                       principal_upset, principal_downset,
                       cover_edges, nvertices, leq,
                       upset_indices, downset_indices
import ..FiniteFringe: build_cache!
import Base.Threads

const INDICATOR_MAP_MEMO_THRESHOLD = Ref(1_000_000)

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

@inline function _write_subsequence_identity!(
    Muv::Matrix{K},
    tgt_ids::Vector{Int},
    src_ids::Vector{Int},
) where {K}
    oneK = one(K)
    i = 1
    j = 1
    nt = length(tgt_ids)
    ns = length(src_ids)
    @inbounds while i <= nt && j <= ns
        ti = tgt_ids[i]
        sj = src_ids[j]
        if ti == sj
            Muv[i, j] = oneK
            i += 1
            j += 1
        elseif ti < sj
            i += 1
        else
            error("_write_subsequence_identity!: source generator id missing from target active set")
        end
    end
    j > ns || error("_write_subsequence_identity!: source generator id missing from target active set")
    return Muv
end

@inline function _write_projection_identity!(
    Muv::Matrix{K},
    tgt_ids::Vector{Int},
    src_ids::Vector{Int},
) where {K}
    oneK = one(K)
    i = 1
    j = 1
    nt = length(tgt_ids)
    ns = length(src_ids)
    @inbounds while i <= nt && j <= ns
        ti = tgt_ids[i]
        sj = src_ids[j]
        if ti == sj
            Muv[i, j] = oneK
            i += 1
            j += 1
        elseif sj < ti
            j += 1
        else
            error("_write_projection_identity!: target generator id missing from source active set")
        end
    end
    i > nt || error("_write_projection_identity!: target generator id missing from source active set")
    return Muv
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

@inline function _resolution_cache_indicator_get(cache::ResolutionCache, key::ResolutionKey3, ::Type{R}) where {R}
    shard = cache.indicator_shards[_resolution_cache_shard_index(cache.indicator_shards)]
    v = get(shard, key, nothing)
    v === nothing || return (v::R)
    Base.lock(cache.lock)
    try
        v = get(cache.indicator, key, nothing)
    finally
        Base.unlock(cache.lock)
    end
    v === nothing || begin
        vv = v::R
        shard[key] = vv
        return vv
    end
    return nothing
end

@inline function _resolution_cache_indicator_store!(cache::ResolutionCache, key::ResolutionKey3, val::R) where {R}
    shard = cache.indicator_shards[_resolution_cache_shard_index(cache.indicator_shards)]
    existing = get(shard, key, nothing)
    existing === nothing || return (existing::R)
    shard[key] = val
    Base.lock(cache.lock)
    out = get(cache.indicator, key, nothing)
    if out === nothing
        cache.indicator[key] = val
        out = val
    end
    Base.unlock(cache.lock)
    outR = out::R
    shard[key] = outR
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

    # Basis for each fiber M_q as columns of a K matrix B[q] spanning im(phi_q).
    B = Vector{Matrix{K}}(undef, n)
    dims = zeros(Int, n)
    for q in 1:n
        cols = findall(U -> U.mask[q], H.U)
        rows = findall(D -> D.mask[q], H.D)
        if isempty(cols) || isempty(rows)
            B[q] = zeros(K, length(rows), 0)
            dims[q] = 0
            continue
        end
        phi_q = @view H.phi[rows, cols]
        B[q] = FieldLinAlg.colspace(field, phi_q)
        dims[q] = size(B[q], 2)
    end

    # Death projection E_u \to E_v on a cover u<v: keep row indices j that remain active at v.
    function death_projection(u::Int, v::Int)
        rows_u = findall(D -> D.mask[u], H.D)
        rows_v = findall(D -> D.mask[v], H.D)
        pos_v = Dict{Int,Int}(rows_v[i] => i for i in 1:length(rows_v))
        P = zeros(K, length(rows_v), length(rows_u))
        for (jpos, jidx) in enumerate(rows_u)
            if haskey(pos_v, jidx)
                P[pos_v[jidx], jpos] = one(K)
            end
        end
        P
    end

    # Structure map on a cover u<v: M_u --incl--> E_u --proj--> E_v --coords--> M_v
    edge_maps = Dict{Tuple{Int,Int}, AbstractMatrix{K}}()
    C = cover_edges(Q)

    @inbounds for (u, v) in C
        du = dims[u]
        dv = dims[v]
        if du == 0 || dv == 0
            edge_maps[(u, v)] = zeros(K, dv, du)
        else
            Puv = death_projection(u, v)      # E_u -> E_v
            Im  = Puv * B[u]                  # in E_v coordinates
            X   = FieldLinAlg.solve_fullcolumn(field, B[v], Im)
            edge_maps[(u, v)] = X             # M_u -> M_v
        end
    end

    PModule{K}(Q, dims, edge_maps; field=field)
end

# -------------------------- projective cover (Def. 6.4.1) --------------------------

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

    # dense fast path
    if eltype(maps) <: Matrix{K}
        A = Matrix{K}(undef, dv, tot)
        col = 1
        @inbounds for i in eachindex(pv)
            u = pv[i]
            du = M.dims[u]
            if du > 0
                A[:, col:col+du-1] .= maps[i]
                col += du
            end
        end
        return FieldLinAlg.colspace(field, A)
    end

    # fallback: sparse/abstract matrix types
    return FieldLinAlg.colspace(field, hcat(maps...))
end



"""
    projective_cover(M::PModule{K})
Return (F0, pi0, gens_at) where F0 is a direct sum of principal upsets covering M,
pi0 : F0 \to M is the natural surjection, and `gens_at[v]` lists the generators activated
at vertex v (each item is a pair (p, local_index_in_Mp)).
"""
function projective_cover(M::PModule{K};
                          cache::Union{Nothing,CoverCache}=nothing,
                          threads::Bool = (Threads.nthreads() > 1)) where {K}
    field = M.field
    Q = M.Q; n = nvertices(Q)
    build_cache!(Q; cover=true, updown=true)
    cc = cache === nothing ? _get_cover_cache(Q) : cache
    map_memo = _indicator_new_array_memo(K, n)
    memos = threads && Threads.nthreads() > 1 ?
        [_indicator_new_array_memo(K, n)
         for _ in 1:max(1, Threads.maxthreadid())] : Vector{Vector{Union{Nothing,Matrix{K}}}}()

    # number of generators at each vertex = dim(M_v) - rank(incoming_image)
    gens_at = Vector{Vector{Tuple{Int,Int}}}(undef, n)
    chosen_at = Vector{Vector{Int}}(undef, n)
    gen_of_p = fill(0, n)
    if threads && Threads.nthreads() > 1
        Threads.@threads for v in 1:n
            Img = _incoming_image_basis(M, v; cache=cc)
            beta = M.dims[v] - size(Img, 2)
            chosen = Int[]
            if beta > 0 && M.dims[v] > 0
                S = Img
                Id = eye(field, M.dims[v])
                rS = size(S, 2)
                for j in 1:M.dims[v]
                    T = hcat(S, Id[:, j])
                    if FieldLinAlg.rank(field, T) > rS
                        push!(chosen, j); S = FieldLinAlg.colspace(field, T); rS += 1
                        length(chosen) == beta && break
                    end
                end
            end
            chosen_at[v] = chosen
            gens_at[v] = [(v, j) for j in chosen]
            gen_of_p[v] = length(chosen)
        end
    else
        for v in 1:n
            Img = _incoming_image_basis(M, v; cache=cc)
            beta = M.dims[v] - size(Img, 2)
            chosen = Int[]
            if beta > 0 && M.dims[v] > 0
                S = Img
                Id = eye(field, M.dims[v])
                rS = size(S, 2)
                for j in 1:M.dims[v]
                    T = hcat(S, Id[:, j])
                    if FieldLinAlg.rank(field, T) > rS
                        push!(chosen, j); S = FieldLinAlg.colspace(field, T); rS += 1
                        length(chosen) == beta && break
                    end
                end
            end
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
    if threads && Threads.nthreads() > 1
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
    if threads && Threads.nthreads() > 1
        Threads.@threads for i in 1:n
            lst = Int[]
            for p in downset_indices(Q, i)
                gen_of_p[p] > 0 || continue
                append!(lst, gid_ranges[p])
            end
            issorted(lst) || sort!(lst)
            active_ids[i] = lst
        end
    else
        for i in 1:n
            lst = Int[]
            for p in downset_indices(Q, i)
                gen_of_p[p] > 0 || continue
                append!(lst, gid_ranges[p])
            end
            issorted(lst) || sort!(lst)
            active_ids[i] = lst
        end
    end

    F0_edges = Dict{Tuple{Int,Int}, AbstractMatrix{K}}()
    if threads && Threads.nthreads() > 1
        edges = cover_edges(Q)
        mats = Vector{Matrix{K}}(undef, length(edges))
        Threads.@threads for idx in eachindex(edges)
            u, v = edges[idx]
            Muv = zeros(K, F0_dims[v], F0_dims[u])
            _write_subsequence_identity!(Muv, active_ids[v], active_ids[u])
            mats[idx] = Muv
        end
        for idx in eachindex(edges)
            F0_edges[edges[idx]] = mats[idx]
        end
    else
        @inbounds for u in 1:n
            su = cc.succs[u]
            for v in su
                Muv = zeros(K, F0_dims[v], F0_dims[u])
                _write_subsequence_identity!(Muv, active_ids[v], active_ids[u])
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

    comps = Vector{Matrix{K}}(undef, n)
    if threads && Threads.nthreads() > 1
        Threads.@threads for i in 1:n
            memo = memos[_thread_local_index(memos)]
            Mi = M.dims[i]
            Fi = F0_dims[i]
            cols = zeros(K, Mi, Fi)
            col = 1
            for p in downset_indices(Q, i)
                k = gen_of_p[p]
                k == 0 && continue
                A = _map_leq_cached_indicator(M, p, i, cc, memo)  # M_p -> M_i
                Jp = J_at[p]
                @inbounds for t in 1:k
                    j = Jp[t]
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
            for p in downset_indices(Q, i)
                k = gen_of_p[p]
                k == 0 && continue
                A = _map_leq_cached_indicator(M, p, i, cc, map_memo)  # M_p -> M_i
                Jp = J_at[p]
                @inbounds for t in 1:k
                    j = Jp[t]
                    copyto!(view(cols, :, col), view(A, :, j))
                    col += 1
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
    su = cc.succs[u]
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

    A = Matrix{K}(undef, tot, du)
    row = 1
    maps_u = M.edge_maps.maps_to_succ[u]

    @inbounds for j in eachindex(su)
        v = su[j]
        dv = M.dims[v]
        if dv > 0
            A[row:row+dv-1, :] .= maps_u[j]
            row += dv
        end
    end

    return FieldLinAlg.nullspace(field, A)  # columns span the socle at u
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
                         threads::Bool = (Threads.nthreads() > 1)) where {K}
    field = M.field
    Q = M.Q; n = nvertices(Q)
    build_cache!(Q; cover=true, updown=true)
    cc = cache === nothing ? _get_cover_cache(Q) : cache
    map_memo = _indicator_new_array_memo(K, n)
    memos = threads && Threads.nthreads() > 1 ?
        [_indicator_new_array_memo(K, n)
         for _ in 1:max(1, Threads.maxthreadid())] : Vector{Vector{Union{Nothing,Matrix{K}}}}()

    # socle bases at each vertex and their multiplicities
    Soc = Vector{Matrix{K}}(undef, n)
    mult = zeros(Int, n)
    if threads && Threads.nthreads() > 1
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
    if threads && Threads.nthreads() > 1
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
    if threads && Threads.nthreads() > 1
        Threads.@threads for i in 1:n
            lst = Int[]
            for u in upset_indices(Q, i)
                mult[u] > 0 || continue
                append!(lst, gid_ranges[u])
            end
            issorted(lst) || sort!(lst)
            active_ids[i] = lst
        end
    else
        for i in 1:n
            lst = Int[]
            for u in upset_indices(Q, i)
                mult[u] > 0 || continue
                append!(lst, gid_ranges[u])
            end
            issorted(lst) || sort!(lst)
            active_ids[i] = lst
        end
    end

    # E structure maps along cover edges u<v are coordinate projections:
    # keep exactly those generators that are still active at v.
    Eedges = Dict{Tuple{Int,Int}, AbstractMatrix{K}}()
    if threads && Threads.nthreads() > 1
        edges = cover_edges(Q)
        mats = Vector{Matrix{K}}(undef, length(edges))
        Threads.@threads for idx in eachindex(edges)
            u, v = edges[idx]
            Muv = zeros(K, Edims[v], Edims[u])
            _write_projection_identity!(Muv, active_ids[v], active_ids[u])
            mats[idx] = Muv
        end
        for idx in eachindex(edges)
            Eedges[edges[idx]] = mats[idx]
        end
    else
        @inbounds for u in 1:n
            su = cc.succs[u]
            for v in su
                Muv = zeros(K, Edims[v], Edims[u])
                _write_projection_identity!(Muv, active_ids[v], active_ids[u])
                Eedges[(u, v)] = Muv
            end
        end
    end
    E = PModule{K}(Q, Edims, Eedges; field=field)

    # iota : M -> E
    Linv = [ _left_inverse_full_column(field, Soc[u]) for u in 1:n ]

    # Only vertices with nontrivial socle contribute rows.
    comps = Vector{Matrix{K}}(undef, n)
    if threads && Threads.nthreads() > 1
        Threads.@threads for i in 1:n
            memo = memos[_thread_local_index(memos)]
            rows = zeros(K, Edims[i], M.dims[i])
            r = 1
            for u in upset_indices(Q, i)
                m = mult[u]
                m == 0 && continue
                Mi_to_Mu = _map_leq_cached_indicator(M, i, u, cc, memo)
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
            for u in upset_indices(Q, i)
                m = mult[u]
                m == 0 && continue
                Mi_to_Mu = _map_leq_cached_indicator(M, i, u, cc, map_memo)
                @views mul!(rows[r:r+m-1, :], Linv[u], Mi_to_Mu)
                r += m
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
    threads::Bool=(Threads.nthreads() > 1),
) where {K}
    return _injective_hull(M; cache=cache, threads=threads)
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
    su = cc.succs[u]
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

    B = Matrix{K}(undef, tot, du)
    row = 1
    maps_u = M.edge_maps.maps_to_succ[u]

    @inbounds for j in eachindex(su)
        v = su[j]
        dv = M.dims[v]
        if dv > 0
            B[row:row+dv-1, :] .= maps_u[j]
            row += dv
        end
    end

    # Return a basis for the row space of B as an r x du matrix with full row rank.
    return transpose(FieldLinAlg.colspace(field, transpose(B)))
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
    U = Upset[]
    for p in 1:nvertices(P)
        for _ in gens_at[p]
            push!(U, principal_upset(P, p))
        end
    end
    U
end

# Build the list of principal downsets from per-vertex labels returned by _injective_hull
function _principal_downsets_from_gens(P::AbstractPoset,
                                       gens_at::Vector{Vector{Tuple{Int,Int}}})
    D = Downset[]
    for u in 1:nvertices(P)
        for _ in gens_at[u]
            push!(D, principal_downset(P, u))
        end
    end
    D
end

# For upset side: at vertex i, which global generators are active (born at p <= i)?
# Returns L[i] = vector of global generator labels (p,j) visible at i.
function _local_index_list_up(P::AbstractPoset,
                              gens_at::Vector{Vector{Tuple{Int,Int}}})
    L = Vector{Vector{Tuple{Int,Int}}}(undef, nvertices(P))
    for i in 1:nvertices(P)
        lst = Tuple{Int,Int}[]
        for p in downset_indices(P, i)
            append!(lst, gens_at[p])
        end
        L[i] = lst
    end
    L
end

# For downset side: at vertex i, which global generators are active (born at u with i <= u)?
# Returns L[i] = vector of global generator labels (u,j) visible at i.
function _local_index_list_down(P::AbstractPoset,
                                gens_at::Vector{Vector{Tuple{Int,Int}}})
    L = Vector{Vector{Tuple{Int,Int}}}(undef, nvertices(P))
    for i in 1:nvertices(P)
        lst = Tuple{Int,Int}[]
        for u in upset_indices(P, i)
            append!(lst, gens_at[u])
        end
        L[i] = lst
    end
    L
end

# Precompute per-vertex lookup from generator label to local column index.
function _local_index_pos(L::Vector{Vector{Tuple{Int,Int}}})
    pos = Vector{Dict{Tuple{Int,Int},Int}}(undef, length(L))
    for i in 1:length(L)
        d = Dict{Tuple{Int,Int},Int}()
        for (c, g) in enumerate(L[i])
            d[g] = c
        end
        pos[i] = d
    end
    return pos
end

# Find the local column index (1-based) of a global generator g=(p,j) or (u,j) in L[i].
# Returns 0 if not present.
function _local_col_of(L::Vector{Vector{Tuple{Int,Int}}}, g::Tuple{Int,Int}, i::Int)
    for (c, gg) in enumerate(L[i])
        if gg == g; return c; end
    end
    return 0
end

# Dense -> sparse helper (build directly from triplets to avoid dense blocks).
function _empty_sparse(K, nr::Int, nc::Int)
    return spzeros(K, nr, nc)
end

# ------------------------------ upset resolution ------------------------------

"""
    upset_resolution(H::FiniteFringe.FringeModule{K}; maxlen=nothing)

Compute an upset (projective) indicator resolution of the fringe module `H`.
`maxlen` is a cutoff on the number of differentials computed (it does not pad output).
"""
function upset_resolution(H::FiniteFringe.FringeModule{K};
                          maxlen::Union{Int,Nothing}=nothing,
                          threads::Bool = (Threads.nthreads() > 1)) where {K}
    return upset_resolution(pmodule_from_fringe(H); maxlen=maxlen, threads=threads)
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
                          threads::Bool = (Threads.nthreads() > 1)) where {K}
    field = M.field
    P = M.Q

    # First projective cover: F0 --pi0--> M, with labels gens_at_F0
    F0, pi0, gens_at_F0 = projective_cover(M; threads=threads)

    U_by_a = Vector{Vector{Upset}}()
    push!(U_by_a, _principal_upsets_from_gens(P, gens_at_F0))

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
        Kmod, iota = kernel_with_inclusion(curr_pi)   # ker(F_prev -> ...)
        if sum(Kmod.dims) == 0
            break
        end
        Fnext, pinext, gens_at_next = projective_cover(Kmod; threads=threads)

        # d = iota o pinext : Fnext -> curr_dom
        comps = Vector{Matrix{K}}(undef, nvertices(P))
        if threads && Threads.nthreads() > 1
            Threads.@threads for i in 1:nvertices(P)
                comps[i] = iota.comps[i] * pinext.comps[i]
            end
        else
            for i in 1:nvertices(P)
                comps[i] = iota.comps[i] * pinext.comps[i]
            end
        end
        d = PMorphism{K}(Fnext, curr_dom, comps)

        # Labels and local index lists
        U_next = _principal_upsets_from_gens(P, gens_at_next)
        Lprev  = _local_index_list_up(P, curr_gens)
        Lnext  = _local_index_list_up(P, gens_at_next)
        pos_prev = _local_index_pos(Lprev)
        pos_next = _local_index_pos(Lnext)

        # Global enumerations of generators (record their birth vertices)
        global_prev = Tuple{Int,Int}[]
        for p in 1:nvertices(P)
            append!(global_prev, curr_gens[p])
        end
        global_next = Tuple{Int,Int}[]
        for p in 1:nvertices(P)
            append!(global_next, gens_at_next[p])
        end

        # Assemble sparse delta: rows index next, cols index prev
        if threads && Threads.nthreads() > 1
            I_chunks = [Int[] for _ in 1:max(1, Threads.maxthreadid())]
            J_chunks = [Int[] for _ in 1:max(1, Threads.maxthreadid())]
            V_chunks = [K[] for _ in 1:max(1, Threads.maxthreadid())]
            Threads.@threads for lambda in 1:length(global_prev)
                tid = _thread_local_index(I_chunks)
                plambda, jlambda = global_prev[lambda]
                for (theta, (ptheta, jtheta)) in enumerate(global_next)
                    # Containment for principal upsets: Up(ptheta) subseteq Up(plambda)
                    if leq(P, plambda, ptheta)
                        # Read at the minimal vertex where the domain generator exists
                        i = ptheta
                        col = get(pos_next[i], (ptheta, jtheta), 0)
                        row = get(pos_prev[i], (plambda, jlambda), 0)
                        if col > 0 && row > 0
                            val = d.comps[i][row, col]
                            if val != 0
                                push!(I_chunks[tid], theta)
                                push!(J_chunks[tid], lambda)
                                push!(V_chunks[tid], val)
                            end
                        end
                    end
                end
            end
            I = Int[]; J = Int[]; V = K[]
            for tid in 1:length(I_chunks)
                append!(I, I_chunks[tid])
                append!(J, J_chunks[tid])
                append!(V, V_chunks[tid])
            end
            delta = sparse(I, J, V, length(global_next), length(global_prev))
        else
            delta = _empty_sparse(K, length(global_next), length(global_prev))
            for (lambda, (plambda, jlambda)) in enumerate(global_prev)
                for (theta, (ptheta, jtheta)) in enumerate(global_next)
                    # Containment for principal upsets: Up(ptheta) subseteq Up(plambda)
                    if leq(P, plambda, ptheta)
                        # Read at the minimal vertex where the domain generator exists
                        i = ptheta
                        col = get(pos_next[i], (ptheta, jtheta), 0)
                        row = get(pos_prev[i], (plambda, jlambda), 0)
                        if col > 0 && row > 0
                            val = d.comps[i][row, col]
                            if val != 0
                                delta[theta, lambda] = val
                            end
                        end
                    end
                end
            end
        end

        push!(dF, delta)
        push!(U_by_a, U_next)

        # Advance
        curr_dom  = Fnext
        curr_pi   = pinext
        curr_gens = gens_at_next
        steps += 1
    end

    # Package as UpsetPresentation list
    F = Vector{UpsetPresentation{K}}(undef, length(U_by_a))
    for a in 1:length(U_by_a)
        U0 = U_by_a[a]
        if a < length(U_by_a)
            U1 = U_by_a[a+1]
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
                            threads::Bool = (Threads.nthreads() > 1)) where {K}
    return downset_resolution(pmodule_from_fringe(H); maxlen=maxlen, threads=threads)
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
                            threads::Bool = (Threads.nthreads() > 1)) where {K}
    field = M.field
    P = M.Q

    # First injective hull: iota0 : M -> E0
    E0, iota0, gens_at_E0 = _injective_hull(M; threads=threads)
    D_by_b = Vector{Vector{Downset}}()
    push!(D_by_b, _principal_downsets_from_gens(P, gens_at_E0))

    dE = Vector{SparseMatrixCSC{K,Int}}()

    # First cokernel
    C, q = _cokernel_module(iota0)

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
        E1, j, gens_at_E1 = _injective_hull(C; threads=threads)

        # rho_b = j o prev_q : prev_E -> E1
        comps = Vector{Matrix{K}}(undef, nvertices(P))
        if threads && Threads.nthreads() > 1
            Threads.@threads for i in 1:nvertices(P)
                comps[i] = j.comps[i] * prev_q.comps[i]
            end
        else
            for i in 1:nvertices(P)
                comps[i] = j.comps[i] * prev_q.comps[i]
            end
        end
        rho = PMorphism{K}(prev_E, E1, comps)

        # Labels and local index lists
        D_next = _principal_downsets_from_gens(P, gens_at_E1)
        L0     = _local_index_list_down(P, prev_gens)
        L1     = _local_index_list_down(P, gens_at_E1)
        pos0   = _local_index_pos(L0)
        pos1   = _local_index_pos(L1)

        globalD0 = Tuple{Int,Int}[]
        for u in 1:nvertices(P)
            append!(globalD0, prev_gens[u])
        end
        globalD1 = Tuple{Int,Int}[]
        for u in 1:nvertices(P)
            append!(globalD1, gens_at_E1[u])
        end

        # Assemble sparse rho: rows index D1 (next), cols index D0 (prev)
        if threads && Threads.nthreads() > 1
            I_chunks = [Int[] for _ in 1:max(1, Threads.maxthreadid())]
            J_chunks = [Int[] for _ in 1:max(1, Threads.maxthreadid())]
            V_chunks = [K[] for _ in 1:max(1, Threads.maxthreadid())]
            Threads.@threads for lambda in 1:length(globalD0)
                tid = _thread_local_index(I_chunks)
                ulambda, jlambda = globalD0[lambda]
                for (theta, (utheta, jtheta)) in enumerate(globalD1)
                    # Containment for principal downsets: D(utheta) subseteq D(ulambda)
                    if leq(P, utheta, ulambda)
                        i   = utheta
                        col = get(pos0[i], (ulambda, jlambda), 0)
                        row = get(pos1[i], (utheta,  jtheta),  0)
                        if col > 0 && row > 0
                            val = rho.comps[i][row, col]
                            if val != 0
                                push!(I_chunks[tid], theta)
                                push!(J_chunks[tid], lambda)
                                push!(V_chunks[tid], val)
                            end
                        end
                    end
                end
            end
            I = Int[]; J = Int[]; V = K[]
            for tid in 1:length(I_chunks)
                append!(I, I_chunks[tid])
                append!(J, J_chunks[tid])
                append!(V, V_chunks[tid])
            end
            Rh = sparse(I, J, V, length(globalD1), length(globalD0))
        else
            Rh = _empty_sparse(K, length(globalD1), length(globalD0))
            for (lambda, (ulambda, jlambda)) in enumerate(globalD0)      # col in D0
                for (theta,  (utheta,  jtheta))  in enumerate(globalD1)  # row in D1
                    # Containment for principal downsets: D(utheta) subseteq D(ulambda)
                    if leq(P, utheta, ulambda)
                        i   = utheta
                        col = get(pos0[i], (ulambda, jlambda), 0)
                        row = get(pos1[i], (utheta,  jtheta),  0)
                        if col > 0 && row > 0
                            val = rho.comps[i][row, col]
                            if val != 0
                                Rh[theta, lambda] = val
                            end
                        end
                    end
                end
            end
        end

        push!(dE, Rh)
        push!(D_by_b, D_next)

        # Next cokernel and advance
        C, q_next = _cokernel_module(j)
        prev_E    = E1
        prev_gens = gens_at_E1
        prev_q    = q_next
        steps += 1
    end

    # Package as DownsetCopresentation list
    E = Vector{DownsetCopresentation{K}}(undef, length(D_by_b))
    for b in 1:length(D_by_b)
        D0 = D_by_b[b]
        if b < length(D_by_b)
            D1 = D_by_b[b+1]
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
function indicator_resolutions(HM::FiniteFringe.FringeModule{K},
                               HN::FiniteFringe.FringeModule{K};
                               maxlen::Union{Int,Nothing}=nothing,
                               threads::Bool = (Threads.nthreads() > 1),
                               cache::Union{Nothing,ResolutionCache}=nothing) where {K}
    if cache !== nothing
        maxkey = maxlen === nothing ? -1 : Int(maxlen)
        key = _resolution_key3(HM, HN, maxkey)
        cache_val_type = Tuple{
            Vector{UpsetPresentation{K}},
            Vector{SparseMatrixCSC{K,Int}},
            Vector{DownsetCopresentation{K}},
            Vector{SparseMatrixCSC{K,Int}},
        }

        cached = _resolution_cache_indicator_get(cache, key, cache_val_type)
        cached === nothing || return cached

        out = begin
            F, dF = upset_resolution(HM; maxlen=maxlen, threads=threads)
            E, dE = downset_resolution(HN; maxlen=maxlen, threads=threads)
            (F, dF, E, dE)
        end

        return _resolution_cache_indicator_store!(cache, key, out)
    end

    F, dF = upset_resolution(HM; maxlen=maxlen, threads=threads)
    E, dE = downset_resolution(HN; maxlen=maxlen, threads=threads)
    return F, dF, E, dE
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
