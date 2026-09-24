# =============================================================================
# rank_cache.jl
#
# Shared memoization helpers for rank-style invariant queries.
#
# Owns:
# - map-leq memo helpers used by rank-style invariant computations
# - `RankQueryCache` and its locate/rank memoization utilities for Zn encodings
#
# Does not own:
# - rank invariant algorithms
# - rectangle signed barcode kernels
# - slice query orchestration
# =============================================================================

const RANK_INVARIANT_MEMO_THRESHOLD = Ref(1_000_000)
const RECTANGLE_LOC_LINEAR_CACHE_THRESHOLD = Ref(20_000_000)

@inline function _use_array_memo(n::Int)
    return n * n <= RANK_INVARIANT_MEMO_THRESHOLD[]
end

@inline function _new_array_memo(::Type{K}, n::Int) where {K}
    memo = Vector{Union{Nothing,AbstractMatrix{K}}}(undef, n * n)
    fill!(memo, nothing)
    return memo
end

@inline function _grid_cache_index(p::NTuple{N,Int}, dims::NTuple{N,Int}) where {N}
    idx = 1
    stride = 1
    @inbounds for i in 1:N
        pi = p[i]
        di = dims[i]
        if pi < 1 || pi > di
            return 0
        end
        idx += (pi - 1) * stride
        stride *= di
    end
    return idx
end

@inline _memo_index(n::Int, u::Int, v::Int) = (u - 1) * n + v

@inline function _memo_get(memo::AbstractVector{Union{Nothing,AbstractMatrix{K}}}, n::Int, u::Int, v::Int) where {K}
    return memo[_memo_index(n, u, v)]
end

@inline function _memo_set!(memo::AbstractVector{Union{Nothing,AbstractMatrix{K}}}, n::Int, u::Int, v::Int, val::AbstractMatrix{K}) where {K}
    memo[_memo_index(n, u, v)] = val
    return val
end

function _map_leq_cached(
    M::PModule{K},
    u::Int,
    v::Int,
    cc,
    memo
)::AbstractMatrix{K} where {K}
    if memo isa AbstractDict
        key = (u, v)
        if haskey(memo, key)
            return memo[key]
        end
    else
        n = nvertices(M.Q)
        X = _memo_get(memo, n, u, v)
        X === nothing || return X
    end

    if u == v
        X = _eye(K, M.dims[v])
        if memo isa AbstractDict
            memo[(u, v)] = X
        else
            _memo_set!(memo, nvertices(M.Q), u, v, X)
        end
        return X
    end

    if cc.C !== nothing
        if cc.C[u, v]
            X = M.edge_maps[u, v]
            if memo isa AbstractDict
                memo[(u, v)] = X
            else
                _memo_set!(memo, nvertices(M.Q), u, v, X)
            end
            return X
        end
    end

    cc_full = cc === nothing ? build_cache!(M) : cc
    pred = _chosen_predecessor(cc_full, u, v)
    if pred == 0
        if cc_full.C !== nothing
            preds = _preds(cc_full.cover_cache, v)
            for w in preds
                if !cc_full.R[u, w]
                    continue
                end
                Y = _map_leq_cached(M, w, v, cc_full, memo)
                X = M.edge_maps[w, v]
                if memo isa AbstractDict
                    memo[(u, v)] = FieldLinAlg._matmul(Y, X)
                    return memo[(u, v)]
                else
                    return _memo_set!(memo, nvertices(M.Q), u, v, FieldLinAlg._matmul(Y, X))
                end
            end
        end
        error("_map_leq_cached: could not find predecessor on a path from $u to $v")
    end

    X = _map_leq_cached(M, u, pred, cc_full, memo)
    Y = M.edge_maps[pred, v]
    if memo isa AbstractDict
        memo[(u, v)] = FieldLinAlg._matmul(Y, X)
        return memo[(u, v)]
    end
    return _memo_set!(memo, nvertices(M.Q), u, v, FieldLinAlg._matmul(Y, X))
end

"""
    RankQueryCache(pi::ZnEncodingMap)

Cache object used by `rectangle_signed_barcode(M, pi, ...)` and related routines.

Lifecycle and ownership:
- the cache is tied to one specific `ZnEncodingMap` `pi`,
- use it across repeated queries of one fixed module against that same encoding,
- discard it when the module or encoding changes.

Concurrent queries may share the cache while the module and encoding stay
unchanged. Lookup and publication are synchronized; expensive rank/locate work
runs outside the cache lock, so simultaneous misses may compute twice. Bulk
rectangle queries gather cached ranks together, compute each distinct missing
pair once per call, then publish together. Their tensor computations use private
arrays. Clearing its rank/locate entries fences off publication to those entries
by already-running queries; those queries may still finish with their own results.
Separate rectangle-result/tensor caches are not reset by that internal clear.
Clearing does not authorize
reuse with a different module: discard the cache when its inputs change.

Storage regime:
- `loc_cache` memoizes `locate(pi, g)` for lattice points `g`,
- the rank cache stores `rank_map(M, a, b)` results for region pairs,
- moderate region counts use a linear `nregions(pi)^2` array-backed layout,
- larger region counts fall back to a dictionary layout.

Warm/cold behavior:
- a fresh cache starts cold, with empty locate and rank memo state,
- repeated calls warm the locate cache and the rank cache independently,
- the same cache should be reused when you want repeated rectangle/Euler/MMA
  workflows on the same finite encoding to avoid rebuilding rank data.

Best practice:
- reuse one cache per encoding and module,
- do not share a cache across different `pi`,
- prefer session-level workflow caching for ordinary users and reserve direct
  `RankQueryCache` management for advanced repeated-query workflows.

The cache contains:
- `loc_cache`: memoizes `locate(pi, g)` for lattice points `g` (an `NTuple{N,Int}`),
- rank cache: memoizes `rank_map(M, a, b)` for region indices `a,b` in the encoded poset,
  using a linear-index array for moderate region counts and a dict fallback otherwise.

# Examples

```jldoctest
julia> using TamerOp

julia> const CM = TamerOp.CoreModules;

julia> const FZ = TamerOp.FlangeZn;

julia> const OPT = TamerOp.Options;

julia> const ZE = TamerOp.ZnEncoding;

julia> tau = FZ.face(1, Int[]);

julia> fg = FZ.Flange{CM.QQ}(1,
                            [FZ.IndFlat(tau, (0,)); FZ.IndFlat(tau, (1,))],
                            [FZ.IndInj(tau, (0,)); FZ.IndInj(tau, (1,))],
                            reshape(CM.QQ[1, 0, 0, 1], 2, 2);
                            field=CM.QQField());

julia> _, pi = ZE.encode_poset_from_flanges((fg,), OPT.EncodingOptions());

julia> cache = TamerOp.InvariantCore.RankQueryCache(pi);

julia> TamerOp.describe(cache).cache_layout
:linear

julia> TamerOp.InvariantCore._rank_query_locate!(cache, (0,));

julia> TamerOp.describe(cache).loc_cache_size
1
```
"""
mutable struct RankQueryCache{N}
    pi::ZnEncodingMap
    n::Int
    loc_cache::Dict{NTuple{N,Int},Int}
    n_regions::Int
    use_linear_rank_cache::Bool
    rank_cache_linear::Vector{Int}
    rank_cache_filled::BitVector
    rank_cache::Dict{Tuple{Int,Int},Int}
    lock::ReentrantLock
    generation::UInt
end

function RankQueryCache(pi::ZnEncodingMap)
    N = pi.n
    length(pi.coords) == N || error("RankQueryCache: expected length(pi.coords) == pi.n")
    n_regions = length(pi.sig_y)
    max_linear = 16_000_000
    nelems_big = big(n_regions) * big(n_regions)
    use_linear = nelems_big <= max_linear

    rank_cache_linear = use_linear ? zeros(Int, Int(nelems_big)) : Int[]
    rank_cache_filled = use_linear ? falses(Int(nelems_big)) : falses(0)

    return RankQueryCache{N}(pi, pi.n, Dict{NTuple{N,Int},Int}(),
        n_regions, use_linear, rank_cache_linear, rank_cache_filled,
        Dict{Tuple{Int,Int},Int}(), ReentrantLock(), zero(UInt),
    )
end

RankQueryCache(pi::CompiledEncoding{<:ZnEncodingMap}) = RankQueryCache(pi.pi)

function encoding end
function nregions end
function cache_layout end
function loc_cache_size end
function rank_cache_size end

@inline encoding(cache::RankQueryCache) = cache.pi
@inline nregions(cache::RankQueryCache) = cache.n_regions
@inline cache_layout(cache::RankQueryCache) = cache.use_linear_rank_cache ? :linear : :dict
@inline function loc_cache_size(cache::RankQueryCache)
    Base.lock(cache.lock)
    try
        return length(cache.loc_cache)
    finally
        Base.unlock(cache.lock)
    end
end
@inline function rank_cache_size(cache::RankQueryCache)
    Base.lock(cache.lock)
    try
        return cache.use_linear_rank_cache ? count(identity, cache.rank_cache_filled) : length(cache.rank_cache)
    finally
        Base.unlock(cache.lock)
    end
end

@inline function _rank_query_cache_describe(cache::RankQueryCache)
    Base.lock(cache.lock)
    try
        return (;
            kind=:rank_query_cache,
            ambient_dim=cache.n,
            nregions=nregions(cache),
            cache_layout=cache_layout(cache),
            loc_cache_size=loc_cache_size(cache),
            rank_cache_size=rank_cache_size(cache),
            warm=(loc_cache_size(cache) > 0 || rank_cache_size(cache) > 0),
        )
    finally
        Base.unlock(cache.lock)
    end
end

"""
    describe(cache::RankQueryCache) -> NamedTuple

Return a compact semantic summary of the current cache state.
"""
describe(cache::RankQueryCache) = _rank_query_cache_describe(cache)

"""
    check_rank_query_cache(cache; throw=false) -> NamedTuple

Validate the structural consistency of a [`RankQueryCache`](@ref).
"""
function check_rank_query_cache(cache::RankQueryCache; throw::Bool=false)
    Base.lock(cache.lock)
    try
        issues = String[]
        cache.pi.n == cache.n || push!(issues, "cache ambient dimension does not match the attached encoding.")
        length(cache.pi.coords) == cache.n || push!(issues, "encoding coordinate axis count does not match cache ambient dimension.")
        expected_regions = length(cache.pi.sig_y)
        cache.n_regions == expected_regions || push!(issues, "cache stores n_regions=$(cache.n_regions), expected $expected_regions from the encoding.")

        if cache.use_linear_rank_cache
            expected_linear = cache.n_regions * cache.n_regions
            length(cache.rank_cache_linear) == expected_linear ||
                push!(issues, "linear rank cache has length $(length(cache.rank_cache_linear)), expected $expected_linear.")
            length(cache.rank_cache_filled) == expected_linear ||
                push!(issues, "linear rank occupancy bitvector has length $(length(cache.rank_cache_filled)), expected $expected_linear.")
            isempty(cache.rank_cache) || push!(issues, "dict rank cache should stay empty in linear-cache mode.")
            if length(cache.rank_cache_linear) == length(cache.rank_cache_filled)
                any(v -> v < 0, cache.rank_cache_linear[cache.rank_cache_filled]) &&
                    push!(issues, "linear rank cache contains negative stored ranks.")
            end
        else
            isempty(cache.rank_cache_linear) || push!(issues, "linear rank storage should be empty in dict-cache mode.")
            isempty(cache.rank_cache_filled) || push!(issues, "linear rank occupancy storage should be empty in dict-cache mode.")
            for ((a, b), v) in cache.rank_cache
                (1 <= a <= cache.n_regions && 1 <= b <= cache.n_regions) ||
                    push!(issues, "dict rank cache contains an out-of-range key ($(a), $(b)).")
                v >= 0 || push!(issues, "dict rank cache contains a negative stored rank.")
            end
        end

        for (g, rid) in cache.loc_cache
            rid >= 0 || push!(issues, "locate cache contains a negative region id.")
            rid <= cache.n_regions || push!(issues, "locate cache contains region id $rid outside 0:$(cache.n_regions).")
            length(g) == cache.n || push!(issues, "locate cache contains a point with the wrong ambient dimension.")
        end

        report = (;
            kind=:rank_query_cache,
            valid=isempty(issues),
            ambient_dim=cache.n,
            nregions=nregions(cache),
            cache_layout=cache_layout(cache),
            loc_cache_size=loc_cache_size(cache),
            rank_cache_size=rank_cache_size(cache),
            issues=issues,
        )
        if throw && !report.valid
            Base.throw(ArgumentError("check_rank_query_cache: " * join(report.issues, " ")))
        end
        return report
    finally
        Base.unlock(cache.lock)
    end
end

function show(io::IO, cache::RankQueryCache)
    d = describe(cache)
    print(io, "RankQueryCache(n=", d.ambient_dim,
          ", nregions=", d.nregions,
          ", layout=", d.cache_layout,
          ", locate=", d.loc_cache_size,
          ", rank=", d.rank_cache_size, ")")
end

function show(io::IO, ::MIME"text/plain", cache::RankQueryCache)
    d = describe(cache)
    println(io, "RankQueryCache")
    println(io, "  ambient_dim: ", d.ambient_dim)
    println(io, "  nregions: ", d.nregions)
    println(io, "  cache_layout: ", d.cache_layout)
    println(io, "  loc_cache_size: ", d.loc_cache_size)
    println(io, "  rank_cache_size: ", d.rank_cache_size)
    println(io, "  warm: ", d.warm)
end

@inline _rank_cache_index(a::Int, b::Int, n_regions::Int) = (a - 1) * n_regions + b

@inline function _rank_cache_get!(rq_cache::RankQueryCache, a::Int, b::Int, builder::F) where {F<:Function}
    linear = rq_cache.use_linear_rank_cache && 1 <= a <= rq_cache.n_regions && 1 <= b <= rq_cache.n_regions
    idx = linear ? _rank_cache_index(a, b, rq_cache.n_regions) : 0
    local generation
    Base.lock(rq_cache.lock)
    try
        generation = rq_cache.generation
        if linear
            rq_cache.rank_cache_filled[idx] && return rq_cache.rank_cache_linear[idx]
        else
            found = get(rq_cache.rank_cache, (a, b), nothing)
            found === nothing || return found
        end
    finally
        Base.unlock(rq_cache.lock)
    end

    # Rank computation may yield or reenter this cache. Keep it outside the
    # publication lock; concurrent misses can compute the same value safely.
    value = builder()
    Base.lock(rq_cache.lock)
    try
        rq_cache.generation == generation || return value
        if linear
            if !rq_cache.rank_cache_filled[idx]
                rq_cache.rank_cache_linear[idx] = value
                rq_cache.rank_cache_filled[idx] = true
            end
            return rq_cache.rank_cache_linear[idx]
        end
        return get!(rq_cache.rank_cache, (a, b), value)
    finally
        Base.unlock(rq_cache.lock)
    end
end

# Fill a private rank tensor from a pure index -> region-pair map. Zero labels
# represent absent regions (or unused dense tensor entries) and have rank zero.
# Cache reads and publication are each one critical section. Expensive rank
# evaluation, miss deduplication, and subsequent tensor algebra hold no lock.
# No cache-wide snapshot is made: storage scales with this query's misses only.
function _rank_cache_batch!(output::Array{Int}, cache::RankQueryCache,
                            pair_at::P, builder::F; threads::Bool=false) where {P,F}
    missing = Int[]
    local generation
    linear = cache.use_linear_rank_cache
    n = cache.n_regions
    Base.lock(cache.lock)
    try
        generation = cache.generation
        @inbounds for i in eachindex(output)
            a, b = pair_at(i)
            if a == 0 || b == 0
                output[i] = 0
            elseif linear && 1 <= a <= n && 1 <= b <= n
                slot = _rank_cache_index(a, b, n)
                if cache.rank_cache_filled[slot]
                    output[i] = cache.rank_cache_linear[slot]
                else
                    push!(missing, i)
                end
            else
                found = get(cache.rank_cache, (a,b), nothing)
                if found === nothing
                    push!(missing, i)
                else
                    output[i] = found
                end
            end
        end
    finally
        Base.unlock(cache.lock)
    end
    isempty(missing) && return output

    # Repeated grid points can represent the same region pair. Compute each
    # missing pair once, even in a fresh cache, without locking a shared Dict.
    pairs = Tuple{Int,Int}[]
    pair_slots = Dict{Tuple{Int,Int},Int}()
    aliases = Vector{Int}(undef, length(missing))
    for (j, i) in enumerate(missing)
        pair = pair_at(i)
        slot = get(pair_slots, pair, 0)
        if slot == 0
            push!(pairs, pair)
            slot = length(pairs)
            pair_slots[pair] = slot
        end
        @inbounds aliases[j] = slot
    end
    values = Vector{Int}(undef, length(pairs))
    if threads && Threads.nthreads() > 1 && length(pairs) > 1
        Threads.@threads for j in eachindex(pairs)
            local a, b
            a, b = @inbounds pairs[j]
            @inbounds values[j] = builder(a, b)
        end
    else
        for j in eachindex(pairs)
            a, b = @inbounds pairs[j]
            @inbounds values[j] = builder(a, b)
        end
    end

    Base.lock(cache.lock)
    try
        # An in-flight query may finish after clear, but cannot populate a new
        # generation. Nor may it replace a value won by a concurrent publisher.
        if cache.generation == generation
            @inbounds for j in eachindex(pairs)
                a, b = pairs[j]
                if linear && 1 <= a <= n && 1 <= b <= n
                    slot = _rank_cache_index(a, b, n)
                    if cache.rank_cache_filled[slot]
                        values[j] = cache.rank_cache_linear[slot]
                    else
                        cache.rank_cache_linear[slot] = values[j]
                        cache.rank_cache_filled[slot] = true
                    end
                else
                    values[j] = get!(cache.rank_cache, (a,b), values[j])
                end
            end
        end
    finally
        Base.unlock(cache.lock)
    end
    @inbounds for j in eachindex(missing)
        output[missing[j]] = values[aliases[j]]
    end
    return output
end

function _clear_rank_query_cache!(cache::RankQueryCache)
    Base.lock(cache.lock)
    try
        cache.generation += one(UInt)
        empty!(cache.loc_cache)
        empty!(cache.rank_cache)
        fill!(cache.rank_cache_filled, false)
    finally
        Base.unlock(cache.lock)
    end
    return nothing
end

@inline _rank_cache_get!(builder::F, rq_cache::RankQueryCache, a::Int, b::Int) where {F<:Function} =
    _rank_cache_get!(rq_cache, a, b, builder)

@inline function _resolve_rank_query_cache(pi::ZnEncodingMap,
                                           rq_cache::Union{Nothing,RankQueryCache})
    if rq_cache === nothing
        return RankQueryCache(pi)
    end
    rq_cache.pi === pi || error("rank_query: rq_cache.pi must match pi")
    rq_cache.n == pi.n || error("rank_query: rq_cache has wrong ambient dimension")
    return rq_cache
end

@inline function _rank_query_point_tuple(x::NTuple{N,<:Integer}, n::Int) where {N}
    N == n || error("rank_query: point dimension mismatch (got $N, expected $n)")
    return ntuple(i -> Int(x[i]), n)
end

@inline function _rank_query_point_tuple(x::AbstractVector{<:Integer}, n::Int)
    length(x) == n || error("rank_query: point dimension mismatch (got $(length(x)), expected $n)")
    return ntuple(i -> Int(x[i]), n)
end

@inline function _rank_query_locate!(rq_cache::RankQueryCache{N}, x::NTuple{N,Int}) where {N}
    local generation
    Base.lock(rq_cache.lock)
    try
        generation = rq_cache.generation
        found = get(rq_cache.loc_cache, x, nothing)
        found === nothing || return found
    finally
        Base.unlock(rq_cache.lock)
    end
    value = locate(rq_cache.pi, x)
    Base.lock(rq_cache.lock)
    try
        rq_cache.generation == generation || return value
        return get!(rq_cache.loc_cache, x, value)
    finally
        Base.unlock(rq_cache.lock)
    end
end
