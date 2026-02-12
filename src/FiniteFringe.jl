module FiniteFringe

using SparseArrays, LinearAlgebra
using ..CoreModules: QQ, QQField, FiniteFringeOptions, AbstractCoeffField, coeff_type, field_from_eltype, coerce
import ..CoreModules: change_field
import ..FieldLinAlg


# =========================================
# Iterator helpers (tuple/iterator-first APIs)
# =========================================

struct IndicesView
    data::Vector{Int}
end

Base.IteratorSize(::Type{IndicesView}) = Base.HasLength()
Base.eltype(::Type{IndicesView}) = Int
Base.length(view::IndicesView) = length(view.data)
Base.size(view::IndicesView) = (length(view.data),)
Base.getindex(view::IndicesView, i::Int) = view.data[i]
Base.iterate(view::IndicesView, state::Int=1) =
    state > length(view.data) ? nothing : (view.data[state], state + 1)

struct PosetLeqIter{P}
    P::P
    i::Int
    is_upset::Bool
end

Base.IteratorSize(::Type{PosetLeqIter}) = Base.HasLength()
Base.eltype(::Type{PosetLeqIter}) = Int
function Base.length(it::PosetLeqIter)
    n = nvertices(it.P)
    cnt = 0
    if it.is_upset
        @inbounds for j in 1:n
            cnt += leq(it.P, it.i, j) ? 1 : 0
        end
    else
        @inbounds for j in 1:n
            cnt += leq(it.P, j, it.i) ? 1 : 0
        end
    end
    return cnt
end
function Base.iterate(it::PosetLeqIter, state::Int=1)
    n = nvertices(it.P)
    j = state
    if it.is_upset
        while j <= n
            if leq(it.P, it.i, j)
                return j, j + 1
            end
            j += 1
        end
    else
        while j <= n
            if leq(it.P, j, it.i)
                return j, j + 1
            end
            j += 1
        end
    end
    return nothing
end

struct BitRowIter{A<:AbstractVector{Bool}}
    row::A
end

Base.IteratorSize(::Type{BitRowIter}) = Base.HasLength()
Base.eltype(::Type{BitRowIter}) = Int
Base.length(it::BitRowIter) = count(it.row)
Base.iterate(it::BitRowIter, state::Int=1) = begin
    j = findnext(it.row, state)
    j === nothing ? nothing : (j, j + 1)
end

struct ProductIndexIter{N}
    cart::CartesianIndices{N,NTuple{N,UnitRange{Int}}}
    strides::NTuple{N,Int}
end

Base.IteratorSize(::Type{ProductIndexIter}) = Base.HasLength()
Base.eltype(::Type{ProductIndexIter}) = Int
Base.length(it::ProductIndexIter) = length(it.cart)
function Base.iterate(it::ProductIndexIter{N}, state...) where {N}
    nxt = iterate(it.cart, state...)
    nxt === nothing && return nothing
    I, st = nxt
    lin = 1
    @inbounds for k in 1:N
        lin += (I[k] - 1) * it.strides[k]
    end
    return lin, st
end

# =========================================
# Poset interface + finite poset and indicator sets
# =========================================

abstract type AbstractPoset end

const UPDOWN_CACHE_MODE = Ref(:auto)  # :auto | :always | :never
const UPDOWN_CACHE_THRESHOLD_FINITE = Ref(500_000)
const UPDOWN_CACHE_THRESHOLD_GENERIC = Ref(200_000)

@inline _updown_cache_skip_auto(::AbstractPoset) = false
@inline _updown_cache_threshold(::AbstractPoset) = UPDOWN_CACHE_THRESHOLD_GENERIC[]

"""
    set_updown_cache_policy!(; mode=:auto, finite_threshold=500_000, generic_threshold=200_000)

Tune lazy upset/downset cache construction:
- `mode=:auto`: cache only when below thresholds.
- `mode=:always`: always build cached upset/downset lists.
- `mode=:never`: never build cached upset/downset lists.
"""
function set_updown_cache_policy!(;
                                  mode::Symbol=UPDOWN_CACHE_MODE[],
                                  finite_threshold::Integer=UPDOWN_CACHE_THRESHOLD_FINITE[],
                                  generic_threshold::Integer=UPDOWN_CACHE_THRESHOLD_GENERIC[])
    mode in (:auto, :always, :never) ||
        error("set_updown_cache_policy!: mode must be :auto, :always, or :never (got $(repr(mode))).")
    finite_threshold >= 0 || error("set_updown_cache_policy!: finite_threshold must be >= 0.")
    generic_threshold >= 0 || error("set_updown_cache_policy!: generic_threshold must be >= 0.")
    UPDOWN_CACHE_MODE[] = mode
    UPDOWN_CACHE_THRESHOLD_FINITE[] = Int(finite_threshold)
    UPDOWN_CACHE_THRESHOLD_GENERIC[] = Int(generic_threshold)
    return nothing
end

updown_cache_policy() = (
    mode = UPDOWN_CACHE_MODE[],
    finite_threshold = UPDOWN_CACHE_THRESHOLD_FINITE[],
    generic_threshold = UPDOWN_CACHE_THRESHOLD_GENERIC[],
)

@inline function _should_cache_updown(P::AbstractPoset, n::Int)
    mode = UPDOWN_CACHE_MODE[]
    mode === :always && return true
    mode === :never && return false
    mode === :auto || error("UPDOWN cache mode must be one of :auto, :always, :never (got $(repr(mode))).")
    _updown_cache_skip_auto(P) && return false
    return n * n <= _updown_cache_threshold(P)
end

function _ensure_updown_cache!(P::AbstractPoset)
    hasproperty(P, :cache) || return nothing
    pc = getproperty(P, :cache)
    hasproperty(pc, :upsets) || return nothing
    hasproperty(pc, :downsets) || return nothing
    upsets = getproperty(pc, :upsets)
    downsets = getproperty(pc, :downsets)
    if upsets !== nothing && downsets !== nothing
        return (upsets, downsets)
    end
    n = nvertices(P)
    _should_cache_updown(P, n) || return nothing
    lock = hasproperty(pc, :lock) ? getproperty(pc, :lock) : nothing
    if lock !== nothing
        Base.lock(lock)
    end
    try
        upsets = getproperty(pc, :upsets)
        downsets = getproperty(pc, :downsets)
        if upsets === nothing || downsets === nothing
            up = Vector{Vector{Int}}(undef, n)
            down = Vector{Vector{Int}}(undef, n)
            @inbounds for i in 1:n
                up[i] = Int[]
                down[i] = Int[]
            end
            @inbounds for i in 1:n
                for j in 1:n
                    if leq(P, i, j)
                        push!(up[i], j)
                    end
                    if leq(P, j, i)
                        push!(down[i], j)
                    end
                end
            end
            setproperty!(pc, :upsets, up)
            setproperty!(pc, :downsets, down)
            upsets = up
            downsets = down
        end
        return (upsets, downsets)
    finally
        if lock !== nothing
            Base.unlock(lock)
        end
    end
end

function cached_upset_indices(P::AbstractPoset, i::Int)
    cache = _ensure_updown_cache!(P)
    cache === nothing && return nothing
    return cache[1][i]
end

function cached_downset_indices(P::AbstractPoset, i::Int)
    cache = _ensure_updown_cache!(P)
    cache === nothing && return nothing
    return cache[2][i]
end

"""
    CoverEdges

Lightweight cover-relation wrapper that supports:
- adjacency queries via `C[u,v]`,
- iteration over cover edges `(u,v)`,
- matrix recovery via `BitMatrix(C)` / `Matrix(C)`.
"""
struct CoverEdges
    mat::BitMatrix
    edges::Vector{Tuple{Int,Int}}
end

Base.size(C::CoverEdges) = size(C.mat)
Base.getindex(C::CoverEdges, i::Int) = C.edges[i]
Base.getindex(C::CoverEdges, i::Int, j::Int) = C.mat[i,j]
Base.length(C::CoverEdges) = length(C.edges)
Base.firstindex(C::CoverEdges) = firstindex(C.edges)
Base.lastindex(C::CoverEdges) = lastindex(C.edges)
Base.eachindex(C::CoverEdges) = eachindex(C.edges)
Base.eltype(::Type{CoverEdges}) = Tuple{Int,Int}
Base.IteratorSize(::Type{CoverEdges}) = Base.HasLength()
Base.iterate(C::CoverEdges, state::Int=1) =
    state > length(C.edges) ? nothing : (C.edges[state], state + 1)
Base.convert(::Type{BitMatrix}, C::CoverEdges) = C.mat
Base.convert(::Type{Matrix{Bool}}, C::CoverEdges) = Matrix(C.mat)
Base.BitArray{2}(C::CoverEdges) = C.mat
Base.Array{Bool,2}(C::CoverEdges) = Matrix(C.mat)
Base.BitMatrix(C::CoverEdges) = C.mat
Base.Matrix(C::CoverEdges) = Matrix(C.mat)
Base.findall(C::CoverEdges) = C.edges

"""
    CoverCache(Q)

An internal cache for poset cover data and a hot-path memo used by `map_leq`.
This cache is stored lazily on each poset object (via `PosetCache`).

Thread-safety:

- `succs`, `preds`, and `C` are read-only once constructed.
- `chain_parent` is a *vector of dicts*, one dict per Julia thread, so the hot-path
  memo writes are thread-local and do not require locks.
"""
struct CoverCache
    Q::AbstractPoset
    C::Union{BitMatrix,Nothing}
    succs::Vector{Vector{Int}}
    preds::Vector{Vector{Int}}

    # For each thread, chain_parent[tid][pairkey(a,d)] = chosen predecessor b in preds[d]
    # with a <= b (used to speed up witness chains in kernel/image constructions).
    chain_parent::Vector{Dict{UInt64, Int}}

    # Number of cover edges in Q (used to size edge-indexed stores).
    nedges::Int
end

mutable struct PosetCache
    cover_edges::Union{Nothing,CoverEdges}
    cover::Union{Nothing,CoverCache}
    lock::Base.ReentrantLock
    upsets::Union{Nothing,Vector{Vector{Int}}}
    downsets::Union{Nothing,Vector{Vector{Int}}}
    PosetCache() = new(nothing, nothing, Base.ReentrantLock(), nothing, nothing)
end

function get_cover_cache(Q::AbstractPoset)
    if hasproperty(Q, :cache)
        pc = getproperty(Q, :cache)
        if pc isa PosetCache
            if pc.cover === nothing
                Base.lock(pc.lock)  # lock only on miss path
                try
                    if pc.cover === nothing
                        pc.cover = _build_cover_cache(Q)
                    end
                finally
                    Base.unlock(pc.lock)
                end
            end
            return pc.cover
        end
    end
    error("get_cover_cache: poset type $(typeof(Q)) does not support caching")
end

warm_cache!(Q::AbstractPoset) = (get_cover_cache(Q); Q)

cover_cache(Q::AbstractPoset) = get_cover_cache(Q)

"""
    build_cache!(Q; cover=true, updown=true)

Sequential cache build step for a poset. Use this before entering threaded
read-only loops to avoid lock contention and duplicate lazy-initialization work.
"""
function build_cache!(Q::AbstractPoset; cover::Bool=true, updown::Bool=true)
    cover && get_cover_cache(Q)
    updown && _ensure_updown_cache!(Q)
    return Q
end

"""
    clear_cover_cache!(Q)

Clear the cached cover data stored on a poset object.
"""
function clear_cover_cache!(Q::AbstractPoset)
    if hasproperty(Q, :cache)
        pc = getproperty(Q, :cache)
        if pc isa PosetCache
            Base.lock(pc.lock)
            try
                pc.cover_edges = nothing
                pc.cover = nothing
                pc.upsets = nothing
                pc.downsets = nothing
            finally
                Base.unlock(pc.lock)
            end
            return nothing
        end
    end
    error("clear_cover_cache!: poset type $(typeof(Q)) does not support caching")
end

# Packs two Int32-ish values into a UInt64 for faster Dict keys.
# This is used both here and in the `CoverCache.chain_parent` hot-path memo.
@inline function _pairkey(u::Int, v::Int)::UInt64
    return (UInt64(u) << 32) | UInt64(v)
end

function _build_cover_cache(Q::AbstractPoset)
    # Cache enough to quickly traverse the cover graph and to build edge-indexed stores.
    Ce = cover_edges(Q)

    # BitMatrix adjacency for O(1) cover checks (only for FinitePoset).
    C = Q isa FinitePoset ? BitMatrix(Ce) : nothing

    nedges = length(Ce)
    n = nvertices(Q)
    outdeg = zeros(Int, n)
    indeg = zeros(Int, n)

    for (a, b) in Ce
        outdeg[a] += 1
        indeg[b] += 1
    end

    succs = [Vector{Int}(undef, outdeg[u]) for u in 1:n]
    preds = [Vector{Int}(undef, indeg[u]) for u in 1:n]
    outk = ones(Int, n)
    ink = ones(Int, n)

    for (a, b) in Ce
        succs[a][outk[a]] = b
        preds[b][ink[b]] = a
        outk[a] += 1
        ink[b] += 1
    end

    @inbounds for u in 1:n
        sort!(succs[u])
        sort!(preds[u])
    end

    chain_parent = [Dict{UInt64, Int}() for _ in 1:max(1, Base.Threads.maxthreadid())]

    return CoverCache(Q, C, succs, preds, chain_parent, nedges)
end

@inline function _chain_parent_dict(cc::CoverCache)::Dict{UInt64, Int}
    return cc.chain_parent[min(length(cc.chain_parent), max(1, Base.Threads.threadid()))]
end

function _chosen_predecessor(cc::CoverCache, a::Int, d::Int)
    k = _pairkey(a, d)
    chain_parent = _chain_parent_dict(cc)
    b = get(chain_parent, k, 0)
    if b == 0
        b = findfirst(x -> x != a && leq(cc.Q, a, x), cc.preds[d])
        b = (b === nothing) ? a : cc.preds[d][b]
        chain_parent[k] = b
    end
    return b
end

"""
    nvertices(P) -> Int

Return the number of vertices in the finite poset `P`.
"""
nvertices(::AbstractPoset) = error("nvertices(P) is not implemented for $(typeof(P)).")

"""
    leq(P, i, j) -> Bool

Return `true` iff `i <= j` in the poset `P`.
"""
leq(::AbstractPoset, ::Int, ::Int) =
    error("leq(P, i, j) is not implemented for $(typeof(P)).")

"""
    leq_matrix(P) -> BitMatrix

Materialize the order matrix for `P`. This is a fallback and should be avoided
for large structured posets.
"""
function leq_matrix(P::AbstractPoset)
    n = nvertices(P)
    L = falses(n, n)
    @inbounds for i in 1:n, j in 1:n
        L[i, j] = leq(P, i, j)
    end
    return L
end

"""
    upset_indices(P, i)
    downset_indices(P, i)
    upset_iter(P, i)
    downset_iter(P, i)

Return an iterable of indices in the principal upset/downset of `i`.
Fallback implementations scan the whole poset.
"""
function upset_indices(P::AbstractPoset, i::Int)
    cached = cached_upset_indices(P, i)
    cached === nothing || return IndicesView(cached)
    return PosetLeqIter(P, i, true)
end

function downset_indices(P::AbstractPoset, i::Int)
    cached = cached_downset_indices(P, i)
    cached === nothing || return IndicesView(cached)
    return PosetLeqIter(P, i, false)
end

"""
    upset_iter(P, i)
    downset_iter(P, i)

Iterate over indices in the principal upset/downset of `i`.
This avoids allocating a fresh vector on each call; when cached vectors exist,
those are returned directly (as iterables).
"""
function upset_iter(P::AbstractPoset, i::Int)
    return upset_indices(P, i)
end

function downset_iter(P::AbstractPoset, i::Int)
    return downset_indices(P, i)
end

"""
    leq_row(P, i)
    leq_col(P, j)

Return indices in the i-th row / j-th column of the order relation.
Defaults to upset/downset indices.
"""
leq_row(P::AbstractPoset, i::Int) = upset_indices(P, i)
leq_col(P::AbstractPoset, j::Int) = downset_indices(P, j)

"""
    poset_equal(P, Q) -> Bool

Structural equality of finite posets, defaulting to order-matrix comparison.
"""
function poset_equal(P::AbstractPoset, Q::AbstractPoset)
    nvertices(P) == nvertices(Q) || return false
    return leq_matrix(P) == leq_matrix(Q)
end

"""
    poset_equal_opposite(P, Q) -> Bool

Return true iff `P` equals the opposite of `Q`.
"""
function poset_equal_opposite(P::AbstractPoset, Q::AbstractPoset)
    nvertices(P) == nvertices(Q) || return false
    return leq_matrix(P) == transpose(leq_matrix(Q))
end

"""
    FinitePoset(leq; check=true)

A finite poset stored by its order matrix.

The input `leq` is an `n x n` Boolean matrix with the intended meaning

    leq[i,j] == true    iff    i <= j.

By default (`check=true`) we validate that `leq` is a partial order, i.e. that it is

  * reflexive,
  * antisymmetric,
  * transitive.

The transitivity check is (in the worst case) cubic in `n`. For *known-good*
programmatically constructed posets (grid posets, boolean lattices, products of
chains, ...), you can set `check=false` to skip validation.

This is a performance feature; if you pass `check=false` and `leq` is not a
partial order, downstream computations may give nonsense.
"""
struct FinitePoset <: AbstractPoset
    n::Int
    _leq::BitMatrix           # _leq[i,j] = true  iff i <= j
    cache::PosetCache
    function FinitePoset(leq::AbstractMatrix{Bool};
                         check::Bool=true)
        n1, n2 = size(leq)
        @assert n1 == n2 "leq must be square"

        # Normalize storage.
        L = leq isa BitMatrix ? leq : BitMatrix(leq)

        if check
            _validate_partial_order_matrix!(L)
        end

        new(n1, L, PosetCache())
    end
end

# Convenience constructor when you already know n.
# (This is also useful as a search/replace target in performance hot paths.)
FinitePoset(n::Int, leq::AbstractMatrix{Bool};
            check::Bool=true) = begin
    @assert size(leq, 1) == n && size(leq, 2) == n "leq must be an n x n Boolean matrix"
    FinitePoset(leq; check=check)
end

nvertices(P::FinitePoset) = P.n
leq(P::FinitePoset, i::Int, j::Int) = P._leq[i,j]
leq_matrix(P::FinitePoset) = P._leq
@inline _updown_cache_threshold(::FinitePoset) = UPDOWN_CACHE_THRESHOLD_FINITE[]

function upset_indices(P::FinitePoset, i::Int)
    cached = cached_upset_indices(P, i)
    cached === nothing || return IndicesView(cached)
    row = @view P._leq[i, :]
    return BitRowIter(row)
end

function downset_indices(P::FinitePoset, i::Int)
    cached = cached_downset_indices(P, i)
    cached === nothing || return IndicesView(cached)
    col = @view P._leq[:, i]
    return BitRowIter(col)
end

leq_row(P::FinitePoset, i::Int) = P._leq[i, :]
leq_col(P::FinitePoset, j::Int) = P._leq[:, j]

"""
    ProductOfChainsPoset(sizes)

Poset on a product of chains with sizes given by `sizes`.
Vertex indices use mixed radix ordering with the first axis varying fastest.
"""
struct ProductOfChainsPoset{N} <: AbstractPoset
    sizes::NTuple{N,Int}
    strides::NTuple{N,Int}
    cache::PosetCache
end

function _poset_strides(sizes::NTuple{N,Int}) where {N}
    strides = Vector{Int}(undef, N)
    strides[1] = 1
    for i in 2:N
        strides[i] = strides[i-1] * sizes[i-1]
    end
    return ntuple(i -> strides[i], N)
end

ProductOfChainsPoset(sizes::NTuple{N,Int}) where {N} =
    ProductOfChainsPoset{N}(sizes, _poset_strides(sizes), PosetCache())

ProductOfChainsPoset(sizes::NTuple{N,Int}, strides::NTuple{N,Int}) where {N} =
    ProductOfChainsPoset{N}(sizes, strides, PosetCache())

ProductOfChainsPoset(sizes::AbstractVector{<:Integer}) =
    ProductOfChainsPoset(ntuple(i -> Int(sizes[i]), length(sizes)))
@inline _updown_cache_skip_auto(::ProductOfChainsPoset) = true

function nvertices(P::ProductOfChainsPoset)
    n = 1
    @inbounds for s in P.sizes
        n *= s
    end
    return n
end

@inline _coord_at(idx::Int, size::Int, stride::Int) =
    (div(idx - 1, stride) % size) + 1

@inline function _index_to_coords!(coords::Vector{Int}, idx::Int,
                                  sizes::NTuple{N,Int}, strides::NTuple{N,Int}) where {N}
    x = idx - 1
    @inbounds for i in 1:N
        coords[i] = div(x, strides[i]) % sizes[i] + 1
    end
    return coords
end

@inline function leq(P::ProductOfChainsPoset{N}, i::Int, j::Int) where {N}
    @inbounds for k in 1:N
        ci = _coord_at(i, P.sizes[k], P.strides[k])
        cj = _coord_at(j, P.sizes[k], P.strides[k])
        if ci > cj
            return false
        end
    end
    return true
end

function upset_indices(P::ProductOfChainsPoset{N}, i::Int) where {N}
    cached = cached_upset_indices(P, i)
    cached === nothing || return IndicesView(cached)
    ranges = ntuple(k -> _coord_at(i, P.sizes[k], P.strides[k]):P.sizes[k], N)
    return ProductIndexIter(CartesianIndices(ranges), P.strides)
end

function downset_indices(P::ProductOfChainsPoset{N}, i::Int) where {N}
    cached = cached_downset_indices(P, i)
    cached === nothing || return IndicesView(cached)
    ranges = ntuple(k -> 1:_coord_at(i, P.sizes[k], P.strides[k]), N)
    return ProductIndexIter(CartesianIndices(ranges), P.strides)
end

"""
    GridPoset(coords)

Axis-aligned grid poset with coordinate vectors per axis. Indices follow the
same mixed radix ordering as `ProductOfChainsPoset`.
"""
struct GridPoset{N,T} <: AbstractPoset
    coords::NTuple{N,Vector{T}}
    sizes::NTuple{N,Int}
    strides::NTuple{N,Int}
    cache::PosetCache
end

function GridPoset(coords::NTuple{N,Vector{T}}) where {N,T}
    for i in 1:N
        axis = coords[i]
        for j in 2:length(axis)
            if axis[j] <= axis[j - 1]
                error("GridPoset: coords[$i] must be strictly increasing (no duplicates).")
            end
        end
    end
    sizes = ntuple(i -> length(coords[i]), N)
    return GridPoset{N,T}(coords, sizes, _poset_strides(sizes), PosetCache())
end

GridPoset(coords::AbstractVector{<:AbstractVector}) = GridPoset(ntuple(i -> Vector(coords[i]), length(coords)))
@inline _updown_cache_skip_auto(::GridPoset) = true

nvertices(P::GridPoset) = nvertices(ProductOfChainsPoset(P.sizes))
@inline function leq(P::GridPoset{N}, i::Int, j::Int) where {N}
    @inbounds for k in 1:N
        ci = _coord_at(i, P.sizes[k], P.strides[k])
        cj = _coord_at(j, P.sizes[k], P.strides[k])
        if ci > cj
            return false
        end
    end
    return true
end

upset_indices(P::GridPoset{N}, i::Int) where {N} =
    upset_indices(ProductOfChainsPoset(P.sizes, P.strides, P.cache), i)
downset_indices(P::GridPoset{N}, i::Int) where {N} =
    downset_indices(ProductOfChainsPoset(P.sizes, P.strides, P.cache), i)

"""
    ProductPoset(P1, P2)

Product of two finite posets with the first factor varying fastest.
"""
struct ProductPoset{P1<:AbstractPoset,P2<:AbstractPoset} <: AbstractPoset
    P1::P1
    P2::P2
    cache::PosetCache
end

ProductPoset(P1::AbstractPoset, P2::AbstractPoset) =
    ProductPoset{typeof(P1),typeof(P2)}(P1, P2, PosetCache())

nvertices(P::ProductPoset) = nvertices(P.P1) * nvertices(P.P2)

@inline function _prod_indices(n1::Int, idx::Int)
    i1 = ((idx - 1) % n1) + 1
    i2 = div(idx - 1, n1) + 1
    return i1, i2
end

@inline function leq(P::ProductPoset, i::Int, j::Int)
    n1 = nvertices(P.P1)
    i1, i2 = _prod_indices(n1, i)
    j1, j2 = _prod_indices(n1, j)
    return leq(P.P1, i1, j1) && leq(P.P2, i2, j2)
end

"""
    RegionsPoset(Q, regions)

Structured poset on regions of a finite poset `Q`, with order defined by:
`A <= B` iff there exist `a in A`, `b in B` with `a <= b` in `Q` (Prop. 4.15).
"""
struct RegionsPoset{P<:AbstractPoset} <: AbstractPoset
    Q::P
    regions::Vector{Vector{Int}}
    n::Int
    cache::PosetCache
end

function RegionsPoset(Q::AbstractPoset, regions::Vector{Vector{Int}})
    return RegionsPoset{typeof(Q)}(Q, regions, length(regions), PosetCache())
end

nvertices(P::RegionsPoset) = P.n

function leq(P::RegionsPoset, i::Int, j::Int)
    @inbounds for a in P.regions[i]
        for b in P.regions[j]
            if leq(P.Q, a, b)
                return true
            end
        end
    end
    return false
end


# ---- internal helpers ---------------------------------------------------------

# Return the smallest index j for which sub[j] is true and sup[j] is false.
# If sub is a subset of sup, return 0.
@inline function _subset_violation_index(sub::BitVector, sup::BitVector, n::Int)::Int
    @assert length(sub) == length(sup) == n
    sc = sub.chunks
    uc = sup.chunks
    nchunks = length(sc)

    # Mask off unused bits in the final word, to be robust to any garbage bits.
    r = n & 63
    lastmask = (r == 0) ? typemax(UInt64) : (UInt64(1) << r) - 1

    @inbounds for w in 1:nchunks
        diff = sc[w] & ~uc[w]
        if w == nchunks
            diff &= lastmask
        end
        if diff != 0
            tz = trailing_zeros(diff)
            j = (w - 1) * 64 + tz + 1
            return (j <= n) ? j : 0
        end
    end
    return 0
end

function _validate_partial_order_matrix!(L::BitMatrix)
    n1, n2 = size(L)
    @assert n1 == n2 "leq must be square"
    n = n1

    # Reflexive.
    @inbounds for i in 1:n
        if !L[i, i]
            error("FinitePoset: leq must be reflexive; missing leq[$i,$i] == true")
        end
    end

    # Antisymmetric: i<=j and j<=i implies i=j.
    @inbounds for i in 1:n
        for j in (i + 1):n
            if L[i, j] && L[j, i]
                error("FinitePoset: leq violates antisymmetry: leq[$i,$j] and leq[$j,$i] are both true")
            end
        end
    end

    # Transitive: i<=k and k<=j implies i<=j.
    #
    # Bitset formulation: whenever i<=k, the principal upset of k must be a subset
    # of the principal upset of i. We check subset failures via chunk-wise
    # operations to avoid scalar triple loops.
    rows = Vector{BitVector}(undef, n)
    @inbounds for i in 1:n
        rows[i] = L[i, :]
    end

    @inbounds for i in 1:n
        ri = rows[i]
        # NOTE: for BitVectors, the supported API is `findnext(bitvec, start)`.
        # The 3-argument form `findnext(bitvec, true, start)` is not defined for
        # BitArrays in Julia 1.12.
        k = findnext(ri, 1)
        while k !== nothing
            j = _subset_violation_index(rows[k], ri, n)
            if j != 0
                error("FinitePoset: leq violates transitivity at (i,k,j)=($i,$k,$j)")
            end
            k = findnext(ri, k + 1)
        end
    end

    return nothing
end

# Cover-edge wrappers and constructor specializations are defined near the
# poset-cache declarations so cache storage can be strongly typed.


# BitVector helpers (chunk-level) for allocation-free set operations.
@inline function _tailmask(nbits::Int)::UInt64
    r = nbits & 63
    return (r == 0) ? typemax(UInt64) : (UInt64(1) << r) - 1
end

@inline function _or_chunks!(dest::BitVector, src::BitVector)
    dc = dest.chunks
    sc = src.chunks
    nchunks = length(dc)
    @inbounds for w in 1:nchunks
        dc[w] |= sc[w]
    end
    # Ensure unused tail bits are always zeroed.
    dc[end] &= _tailmask(length(dest))
    return dest
end

@inline function _andnot_chunks!(dest::BitVector, mask::BitVector)
    dc = dest.chunks
    mc = mask.chunks
    nchunks = length(dc)
    @inbounds for w in 1:nchunks
        dc[w] &= ~mc[w]
    end
    # Ensure unused tail bits are always zeroed.
    dc[end] &= _tailmask(length(dest))
    return dest
end

function _compute_cover_edges_bitset(L::BitMatrix)::CoverEdges
    n = size(L, 1)

    # Copy the rows of the order matrix as BitVectors so we can do chunk-wise
    # OR/AND operations. This is the key speedup over scalar triple loops.
    rows = Vector{BitVector}(undef, n)
    @inbounds for i in 1:n
        rows[i] = L[i, :]
    end

    mat = falses(n, n)
    edges = Tuple{Int,Int}[]

    # Workspace reused across i.
    upper = falses(n)      # strict upper set of i, then the set of covers
    redundant = falses(n)  # elements proved non-minimal in upper

    @inbounds for i in 1:n
        copyto!(upper, rows[i])
        upper[i] = false
        fill!(redundant, false)

        # Mark all non-minimal elements of upper(i).
        #
        # For each k in upper(i), any element j > k is not minimal in upper(i),
        # hence i is not covered by j.
        k = findnext(upper, 1)
        while k !== nothing
            # We want strict_upper(k), i.e. exclude k itself. Since rows[k] is
            # reflexive, we OR it in and then restore redundant[k].
            prev = redundant[k]
            _or_chunks!(redundant, rows[k])
            redundant[k] = prev
            k = findnext(upper, k + 1)
        end

        # covers(i) = upper(i) \ redundant
        _andnot_chunks!(upper, redundant)

        j = findnext(upper, 1)
        while j !== nothing
            mat[i, j] = true
            push!(edges, (i, j))
            j = findnext(upper, j + 1)
        end
    end

    return CoverEdges(mat, edges)
end

function _set_cover_edges_cache!(P::AbstractPoset, C::CoverEdges)
    hasproperty(P, :cache) || return C
    pc = getproperty(P, :cache)
    pc isa PosetCache || return C
    Base.lock(pc.lock)
    try
        pc.cover_edges = C
    finally
        Base.unlock(pc.lock)
    end
    return C
end

function _cover_edges_cached_or_build!(builder::Function, P::AbstractPoset, cached::Bool)
    cached || return builder()
    hasproperty(P, :cache) || return builder()
    pc = getproperty(P, :cache)
    pc isa PosetCache || return builder()

    C = pc.cover_edges
    C === nothing || return C

    Base.lock(pc.lock)  # lock only on miss path
    try
        C = pc.cover_edges
        if C === nothing
            C = builder()
            pc.cover_edges = C
        end
        return C
    finally
        Base.unlock(pc.lock)
    end
end

"""
    cover_edges(P; cached=true)

Return the cover relation (Hasse diagram) of the finite poset `P`.

The return value is a `CoverEdges` object `C` supporting:

  * `C[i,j]`          (adjacency query),
  * iteration         (`for (i,j) in C`),
  * `findall(C)`      (edge list),
  * `BitMatrix(C)`    (adjacency matrix).

For performance, the result is cached per `FinitePoset` instance by default.
Pass `cached=false` to force recomputation.
"""
function cover_edges(P::FinitePoset;
                     cached::Bool=true)
    return _cover_edges_cached_or_build!(P, cached) do
        _compute_cover_edges_bitset(P._leq)
    end
end

function _cover_edges_from_edges(n::Int, edges::Vector{Tuple{Int,Int}})
    mat = falses(n, n)
    @inbounds for (i, j) in edges
        mat[i, j] = true
    end
    return CoverEdges(mat, edges)
end

function cover_edges(P::AbstractPoset;
                     cached::Bool=true)
    return _cover_edges_cached_or_build!(P, cached) do
        L = leq_matrix(P)
        _compute_cover_edges_bitset(L isa BitMatrix ? L : BitMatrix(L))
    end
end

function cover_edges(P::ProductOfChainsPoset{N};
                     cached::Bool=true) where {N}
    return _cover_edges_cached_or_build!(P, cached) do
        n = nvertices(P)
        edges = Tuple{Int,Int}[]
        coords = Vector{Int}(undef, N)
        @inbounds for idx in 1:n
            _index_to_coords!(coords, idx, P.sizes, P.strides)
            for k in 1:N
                if coords[k] < P.sizes[k]
                    coords[k] += 1
                    lin = 1
                    for t in 1:N
                        lin += (coords[t] - 1) * P.strides[t]
                    end
                    push!(edges, (idx, lin))
                    coords[k] -= 1
                end
            end
        end
        _cover_edges_from_edges(n, edges)
    end
end

function cover_edges(P::GridPoset{N};
                     cached::Bool=true) where {N}
    return _cover_edges_cached_or_build!(P, cached) do
        cover_edges(ProductOfChainsPoset(P.sizes, P.strides, P.cache); cached=true)
    end
end

function cover_edges(P::ProductPoset;
                     cached::Bool=true)
    return _cover_edges_cached_or_build!(P, cached) do
        n1 = nvertices(P.P1)
        n2 = nvertices(P.P2)
        n = n1 * n2
        edges = Tuple{Int,Int}[]
        C1 = cover_edges(P.P1; cached=cached)
        C2 = cover_edges(P.P2; cached=cached)
        @inbounds for (i1, j1) in C1
            for i2 in 1:n2
                src = i1 + (i2 - 1) * n1
                dst = j1 + (i2 - 1) * n1
                push!(edges, (src, dst))
            end
        end
        @inbounds for (i2, j2) in C2
            for i1 in 1:n1
                src = i1 + (i2 - 1) * n1
                dst = i1 + (j2 - 1) * n1
                push!(edges, (src, dst))
            end
        end
        _cover_edges_from_edges(n, edges)
    end
end

struct Upset
    P::AbstractPoset
    mask::BitVector
end
struct Downset
    P::AbstractPoset
    mask::BitVector
end

struct _WPairData
    rows_by_ucomp::Vector{Vector{Int}}   # for each component of U_M[i], rows supported in that component
    rows_by_dcomp::Vector{Vector{Int}}   # for each component of D_N[t], rows supported in that component
end


Base.length(U::Upset) = length(U.mask)
Base.length(D::Downset) = length(D.mask)
Base.eltype(::Type{Upset}) = Int
Base.eltype(::Type{Downset}) = Int
Base.IteratorSize(::Type{Upset}) = Base.HasLength()
Base.IteratorSize(::Type{Downset}) = Base.HasLength()

# Iterate over vertices contained in an upset/downset.
function Base.iterate(U::Upset, state::Int=1)
    n = length(U.mask)
    i = state
    @inbounds while i <= n
        if U.mask[i]
            return i, i + 1
        end
        i += 1
    end
    return nothing
end

function Base.iterate(D::Downset, state::Int=1)
    n = length(D.mask)
    i = state
    @inbounds while i <= n
        if D.mask[i]
            return i, i + 1
        end
        i += 1
    end
    return nothing
end

# Allocation-free bitset predicates (used heavily in matching / slicing code).
@inline function is_subset(a::BitVector, b::BitVector)
    @assert length(a) == length(b)
    ac = a.chunks
    bc = b.chunks
    nchunks = length(ac)
    lastmask = _tailmask(length(a))
    @inbounds for w in 1:nchunks
        diff = ac[w] & ~bc[w]
        if w == nchunks
            diff &= lastmask
        end
        if diff != 0
            return false
        end
    end
    return true
end

@inline function intersects(a::BitVector, b::BitVector)
    @assert length(a) == length(b)
    ac = a.chunks
    bc = b.chunks
    nchunks = length(ac)
    lastmask = _tailmask(length(a))
    @inbounds for w in 1:nchunks
        v = ac[w] & bc[w]
        if w == nchunks
            v &= lastmask
        end
        if v != 0
            return true
        end
    end
    return false
end

is_subset(U1::Upset, U2::Upset) = is_subset(U1.mask, U2.mask)
is_subset(D1::Downset, D2::Downset) = is_subset(D1.mask, D2.mask)
intersects(U::Upset, D::Downset) = intersects(U.mask, D.mask)

# Upset/downset closures and principal sets (used in resolutions)
function upset_closure(P::AbstractPoset, S::BitVector)
    U = copy(S)
    n = nvertices(P)
    for i in 1:n
        U[i] || continue
        for j in upset_indices(P, i)
            U[j] = true
        end
    end
    Upset(P, U)
end
function downset_closure(P::AbstractPoset, S::BitVector)
    D = copy(S)
    n = nvertices(P)
    for j in 1:n
        D[j] || continue
        for i in downset_indices(P, j)
            D[i] = true
        end
    end
    Downset(P, D)
end

upset_from_generators(P::AbstractPoset, gens::Vector{Int}) =
    upset_closure(P, BitVector([i in gens for i in 1:nvertices(P)]))
downset_from_generators(P::AbstractPoset, gens::Vector{Int}) =
    downset_closure(P, BitVector([i in gens for i in 1:nvertices(P)]))

# Principal upset \uparrow p and principal downset \downarrow p (representables/corepresentables).
function principal_upset(P::AbstractPoset, p::Int)
    n = nvertices(P)
    mask = falses(n)
    for q in upset_indices(P, p)
        mask[q] = true
    end
    return Upset(P, BitVector(mask))
end

function principal_downset(P::AbstractPoset, p::Int)
    n = nvertices(P)
    mask = falses(n)
    for q in downset_indices(P, p)
        mask[q] = true
    end
    return Downset(P, BitVector(mask))
end

#############################
# Structural equality + hashing
#############################

import Base: ==, isequal, hash

# ---- FinitePoset ----
# Adjust field names (:n, :leq) to match your struct definition.
# From your printout it looks like FinitePoset(3, Bool[...]) so probably:
#   n::Int
#   leq::AbstractMatrix{Bool}   (or BitMatrix / Matrix{Bool})
==(P::FinitePoset, Q::FinitePoset) =
    (P.n == Q.n) && (P._leq == Q._leq)

isequal(P::FinitePoset, Q::FinitePoset) = (P == Q)

hash(P::FinitePoset, h::UInt) = hash(P.n, hash(P._leq, h))


# ---- Upset ----
# Adjust field names to match your Upset struct.
# From your printout: Upset(FinitePoset(...), Bool[...]) so likely:
#   P::FinitePoset
#   mem::AbstractVector{Bool}   (or BitVector)
==(U::Upset, V::Upset) =
    (U.P == V.P) && (U.mask == V.mask)

isequal(U::Upset, V::Upset) = (U == V)

hash(U::Upset, h::UInt) = hash(U.P, hash(U.mask, h))


# ---- Downset ----
# Same idea as Upset. Adjust field names.
==(D::Downset, E::Downset) =
    (D.P == E.P) && (D.mask == E.mask)

isequal(D::Downset, E::Downset) = (D == E)

hash(D::Downset, h::UInt) = hash(D.P, hash(D.mask, h))


# =========================================
# Fringe presentations (Defs. 3.16 - 3.17)
# =========================================

struct FringeModule{K, F<:AbstractCoeffField, MAT<:AbstractMatrix{K}}
    field::F
    P::AbstractPoset
    U::Vector{Upset}                  # birth upsets (columns)
    D::Vector{Downset}                # death downsets (rows)
    phi::MAT                          # size |D| x |U|

    function FringeModule{K,F,MAT}(field::F,
                                   P::AbstractPoset,
                                   U::Vector{Upset},
                                   D::Vector{Downset},
                                   phi::MAT) where {K, F<:AbstractCoeffField, MAT<:AbstractMatrix{K}}
        @assert size(phi,1) == length(D) && size(phi,2) == length(U)
        coeff_type(field) == K || error("FringeModule: coeff_type(field) != K")

        M = new{K,F,MAT}(field, P, U, D, phi)
        _check_monomial_condition(M)
        return M
    end
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
    change_field(H, field)

Return a FringeModule obtained by coercing `phi` into `field`.
"""
function change_field(H::FringeModule{K}, field::AbstractCoeffField) where {K}
    K2 = coeff_type(field)
    Phi = _coerce_matrix(field, H.phi)
    return FringeModule{K2, typeof(field), typeof(Phi)}(field, H.P, H.U, H.D, Phi)
end

# Allow calls like FringeModule{K}(P, U, D, phi) by inferring MAT from phi.
function FringeModule{K}(P::AbstractPoset,
                         U::Vector{Upset},
                         D::Vector{Downset},
                         phi::AbstractMatrix{K};
                         field::AbstractCoeffField=field_from_eltype(K)) where {K}
    F = typeof(field)
    return FringeModule{K, F, typeof(phi)}(field, P, U, D, phi)
end

# Convenience constructor (default dense; set store_sparse=true to store CSC).
function FringeModule(P::AbstractPoset,
                      U::Vector{Upset},
                      D::Vector{Downset},
                      phi::AbstractMatrix{K};
                      store_sparse::Bool=false,
                      field::AbstractCoeffField=field_from_eltype(K)) where {K}
    phimat = store_sparse ? SparseArrays.sparse(phi) : Matrix{K}(phi)
    F = typeof(field)
    return FringeModule{K, F, typeof(phimat)}(field, P, U, D, phimat)
end

"""
    one_by_one_fringe(P, U, D[, scalar]) -> FringeModule

Convenience constructor for a 1x1 fringe presentation:

- one upset generator `U`,
- one downset cogenerator `D`,
- and a 1x1 structure matrix `phi = [scalar]`.

This is handy for tests, examples, and quickly building interval-like modules
on a finite poset without writing out the matrix by hand.

Notes:
- The 3-argument form defaults to `scalar = one(coeff_type(field))` with `field = QQField()`.
- The 4-argument form is fully generic in the scalar type `K`.
- A keyword form `; scalar=..., field=...` is also provided for ergonomics.
- In addition to `U::Upset` / `D::Downset`, you may also pass membership masks
  `U_mask::AbstractVector{Bool}` and `D_mask::AbstractVector{Bool}`.
"""
function one_by_one_fringe(P::AbstractPoset, U::Upset, D::Downset, scalar::K;
                           field::AbstractCoeffField=QQField()) where {K}
    s = coerce(field, scalar)
    phi = spzeros(typeof(s), 1, 1)
    phi[1, 1] = s
    return FringeModule{typeof(s)}(P, [U], [D], phi; field=field)
end

# Default scalar (by field).
one_by_one_fringe(P::AbstractPoset, U::Upset, D::Downset) =
    one_by_one_fringe(P, U, D; field=QQField())

# Keyword-friendly wrapper.
one_by_one_fringe(P::AbstractPoset, U::Upset, D::Downset;
                  scalar=nothing,
                  field::AbstractCoeffField=QQField()) =
begin
    scalar === nothing && (scalar = one(coeff_type(field)))
    one_by_one_fringe(P, U, D, coerce(field, scalar); field=field)
end

# ------------------ mask-based convenience overloads ------------------

function _coerce_bool_mask(P::FinitePoset, mask::AbstractVector{Bool}, name::AbstractString)::BitVector
    if length(mask) != P.n
        error(name * " mask must have length P.n=" * string(P.n) *
              "; got length " * string(length(mask)) * ".")
    end
    return (mask isa BitVector) ? mask : BitVector(mask)
end

function one_by_one_fringe(P::FinitePoset,
                           U_mask::AbstractVector{Bool},
                           D_mask::AbstractVector{Bool},
                           scalar::K;
                           field::AbstractCoeffField=QQField()) where {K}
    Um = _coerce_bool_mask(P, U_mask, "Upset")
    Dm = _coerce_bool_mask(P, D_mask, "Downset")
    return one_by_one_fringe(P, Upset(P, Um), Downset(P, Dm), coerce(field, scalar); field=field)
end

one_by_one_fringe(P::FinitePoset,
                  U_mask::AbstractVector{Bool},
                  D_mask::AbstractVector{Bool}) =
    one_by_one_fringe(P, U_mask, D_mask; field=QQField())

one_by_one_fringe(P::FinitePoset,
                  U_mask::AbstractVector{Bool},
                  D_mask::AbstractVector{Bool};
                  scalar=nothing,
                  field::AbstractCoeffField=QQField()) =
begin
    scalar === nothing && (scalar = one(coeff_type(field)))
    one_by_one_fringe(P, U_mask, D_mask, coerce(field, scalar); field=field)
end




# Prop. 3.18: Nonzero entry only if U_i \cap D_j \neq \emptyset.
function _check_monomial_condition(M::FringeModule{K}) where {K}
    m, n = size(M.phi)
    @assert m == length(M.D) && n == length(M.U) "Dimension mismatch"

    phi = M.phi
    if phi isa SparseMatrixCSC
        I, J, V = findnz(phi)
        for t in eachindex(V)
            v = V[t]
            if v != zero(K)
                j = I[t]; i = J[t]
                @assert intersects(M.U[i], M.D[j]) "Nonzero phi[j,i] requires U[i] cap D[j] neq emptyset (Prop. 3.18)"
            end
        end
    else
        for j in 1:m, i in 1:n
            v = phi[j,i]
            if v != zero(K)
                @assert intersects(M.U[i], M.D[j]) "Nonzero phi[j,i] requires U[i] cap D[j] neq emptyset (Prop. 3.18)"
            end
        end
    end
end


# ------------------ evaluation (degreewise image; after Def. 3.17) ------------------


"""
    fiber_dimension(M::FringeModule{K}, q::Int) -> Int

Compute dim_k M_q as rank of phi_q : F_q \to E_q (degreewise image).
"""
function fiber_dimension(M::FringeModule{K}, q::Int) where {K}
    cols = findall(U -> U.mask[q], M.U)
    rows = findall(D -> D.mask[q], M.D)
    if isempty(cols) || isempty(rows); return 0; end
    return FieldLinAlg.rank_restricted(M.field, M.phi, rows, cols)
end

# ------------------ Hom for fringe modules via commuting squares ------------------
#
# We build the linear system for commuting squares in the indicator-module category
# using the full Hom descriptions from Prop. 3.10 (componentwise bases).
#
# V1 = Hom(F_M, F_N)  basis = components of U_M[i] contained in U_N[j]
# V2 = Hom(E_M, E_N)  basis = components of D_N[t] contained in D_M[s]
# W  = Hom(F_M, E_N)  basis = components of U_M[i] cap D_N[t]
#
# Then Hom(M,N) = ker(d0) / (ker(T) + ker(S)) with d0 = [T  -S].

"Undirected adjacency of the Hasse cover graph."
function _cover_undirected_adjacency(P::FinitePoset)
    C = cover_edges(P)
    adj = [Int[] for _ in 1:P.n]
    for (i, j) in C
        push!(adj[i], j)
        push!(adj[j], i)
    end
    return adj
end

"Connected components of a subset mask in the undirected Hasse cover graph."
function _component_data(adj::Vector{Vector{Int}}, mask::BitVector)
    n = length(mask)
    comp = fill(0, n)
    reps = Int[]
    cid = 0
    for v in 1:n
        if mask[v] && comp[v] == 0
            cid += 1
            push!(reps, v)
            queue = [v]
            comp[v] = cid
            head = 1
            while head <= length(queue)
                x = queue[head]; head += 1
                for y in adj[x]
                    if mask[y] && comp[y] == 0
                        comp[y] = cid
                        push!(queue, y)
                    end
                end
            end
        end
    end

    comp_masks = [falses(n) for _ in 1:cid]
    for v in 1:n
        c = comp[v]
        if c != 0
            comp_masks[c][v] = true
        end
    end
    return comp, cid, comp_masks, reps
end

# Count connected components in mask_a intersect mask_b without materializing the intersection mask.
function _component_reps_intersection!(
    reps::Vector{Int},
    adj::Vector{Vector{Int}},
    mask_a::BitVector,
    mask_b::BitVector,
    marks::Vector{Int},
    mark::Int,
    queue::Vector{Int},
)
    empty!(reps)
    ncomp = 0
    n = length(mask_a)
    @inbounds for v in 1:n
        if marks[v] != mark && mask_a[v] && mask_b[v]
            ncomp += 1
            push!(reps, v)
            marks[v] = mark
            empty!(queue)
            push!(queue, v)
            head = 1
            while head <= length(queue)
                x = queue[head]
                head += 1
                for y in adj[x]
                    if marks[y] != mark && mask_a[y] && mask_b[y]
                        marks[y] = mark
                        push!(queue, y)
                    end
                end
            end
        end
    end
    return ncomp
end

"Dimension of Hom(M,N) over a field K using the commuting-square presentation."
function hom_dimension(M::FringeModule{K}, N::FringeModule{K}) where {K}
    @assert M.P === N.P "Posets must match"
    P = M.P

    # Precompute undirected cover adjacency once.
    adj = _cover_undirected_adjacency(P)

    nUM = length(M.U); nDM = length(M.D)
    nUN = length(N.U); nDN = length(N.D)

    # Component decompositions for all upsets in M and downsets in N.
    Ucomp_id_M    = Vector{Vector{Int}}(undef, nUM)
    Ucomp_masks_M = Vector{Vector{BitVector}}(undef, nUM)
    Ucomp_n_M     = Vector{Int}(undef, nUM)
    for i in 1:nUM
        comp_id, ncomp, comp_masks, _ = _component_data(adj, M.U[i].mask)
        Ucomp_id_M[i] = comp_id
        Ucomp_masks_M[i] = comp_masks
        Ucomp_n_M[i] = ncomp
    end

    Dcomp_id_N    = Vector{Vector{Int}}(undef, nDN)
    Dcomp_masks_N = Vector{Vector{BitVector}}(undef, nDN)
    Dcomp_n_N     = Vector{Int}(undef, nDN)
    for t in 1:nDN
        comp_id, ncomp, comp_masks, _ = _component_data(adj, N.D[t].mask)
        Dcomp_id_N[t] = comp_id
        Dcomp_masks_N[t] = comp_masks
        Dcomp_n_N[t] = ncomp
    end

    # Build W = oplus_{i,t} Hom(k[U_M[i]], k[D_N[t]]) with basis indexed by components of U_i cap D_t.
    w_index = zeros(Int, nUM, nDN)   # (iM,tN) -> index into w_data
    w_data  = _WPairData[]
    W_dim = 0
    marks = zeros(Int, P.n)
    queue = Int[]
    reps_int = Int[]
    mark = 1

    for iM in 1:nUM
        for tN in 1:nDN
            mark += 1
            if mark == typemax(Int)
                fill!(marks, 0)
                mark = 1
            end
            ncomp_int = _component_reps_intersection!(
                reps_int, adj, M.U[iM].mask, N.D[tN].mask, marks, mark, queue
            )
            if ncomp_int > 0
                base = W_dim
                W_dim += ncomp_int

                rows_by_u = [Int[] for _ in 1:Ucomp_n_M[iM]]
                rows_by_d = [Int[] for _ in 1:Dcomp_n_N[tN]]
                for c in 1:ncomp_int
                    row = base + c
                    v = reps_int[c]
                    cu = Ucomp_id_M[iM][v]
                    cd = Dcomp_id_N[tN][v]
                    push!(rows_by_u[cu], row)
                    push!(rows_by_d[cd], row)
                end

                push!(w_data, _WPairData(rows_by_u, rows_by_d))
                w_index[iM, tN] = length(w_data)
            end
        end
    end

    # V1 basis: components of U_M[i] contained in U_N[j].
    V1 = Tuple{Int,Int,Int}[]  # (iM, jN, compU)
    for iM in 1:nUM
        for jN in 1:nUN
            for cU in 1:Ucomp_n_M[iM]
                if is_subset(Ucomp_masks_M[iM][cU], N.U[jN].mask)
                    push!(V1, (iM, jN, cU))
                end
            end
        end
    end
    V1_dim = length(V1)

    # V2 basis: components of D_N[t] contained in D_M[s].
    V2 = Tuple{Int,Int,Int}[]  # (sM, tN, compD)
    for sM in 1:nDM
        for tN in 1:nDN
            for cD in 1:Dcomp_n_N[tN]
                if is_subset(Dcomp_masks_N[tN][cD], M.D[sM].mask)
                    push!(V2, (sM, tN, cD))
                end
            end
        end
    end
    V2_dim = length(V2)

    # Build T and S as dense matrices (exact arithmetic, small sizes expected).
    T = zeros(K, W_dim, V1_dim)
    for (col, (iM, jN, cU)) in enumerate(V1)
        for tN in 1:nDN
            val = N.phi[tN, jN]
            if val != zero(K)
                pid = w_index[iM, tN]
                if pid != 0
                    rows = w_data[pid].rows_by_ucomp[cU]
                    for r in rows
                        T[r, col] += val
                    end
                end
            end
        end
    end

    S = zeros(K, W_dim, V2_dim)
    for (col, (sM, tN, cD)) in enumerate(V2)
        for iM in 1:nUM
            val = M.phi[sM, iM]
            if val != zero(K)
                pid = w_index[iM, tN]
                if pid != 0
                    rows = w_data[pid].rows_by_dcomp[cD]
                    for r in rows
                        S[r, col] += val
                    end
                end
            end
        end
    end

    big = hcat(T, -S)

    rT   = FieldLinAlg.rank(M.field, T)
    rS   = FieldLinAlg.rank(M.field, S)
    rBig = FieldLinAlg.rank(M.field, big)

    dimKer_big = (V1_dim + V2_dim) - rBig
    dimKer_T   = V1_dim - rT
    dimKer_S   = V2_dim - rS
    return dimKer_big - (dimKer_T + dimKer_S)
end


# ---------- utility: dense\tosparse over K ----------

"""
    dense_to_sparse_K(A)

Convert a dense matrix `A` to a sparse matrix with the same element type.

For an explicit target coefficient type `K`, call `dense_to_sparse_K(A, K)`.
"""
function dense_to_sparse_K(A::AbstractMatrix{T}, ::Type{K}) where {T,K}
    m,n = size(A)
    S = spzeros(K, m, n)
    for j in 1:n, i in 1:m
        v = K(A[i,j])
        if v != zero(K); S[i,j] = v; end
    end
    S
end

dense_to_sparse_K(A::AbstractMatrix{T}) where {T} = dense_to_sparse_K(A, T)

export FiniteFringeOptions,
       AbstractPoset, FinitePoset, ProductOfChainsPoset, GridPoset, ProductPoset,
       RegionsPoset,
       set_updown_cache_policy!, updown_cache_policy,
       build_cache!,
       nvertices, leq, leq_matrix, upset_indices, downset_indices, upset_iter, downset_iter, leq_row, leq_col,
       poset_equal, poset_equal_opposite,
       CoverEdges, Upset, Downset, principal_upset, principal_downset,
       upset_from_generators, downset_from_generators, upset_closure, downset_closure, cover_edges,
       FringeModule, one_by_one_fringe, fiber_dimension, hom_dimension, dense_to_sparse_K,
       change_field

end # module


# -----------------------------------------------------------------------------
# IndicatorTypes (merged from the old src/IndicatorTypes.jl)
#
# This module contains small shared record types for one-step indicator
# presentations/copresentations (Def. 6.4 in Miller).
#
# We keep the module name `PosetModules.IndicatorTypes` and the type names
# unchanged for API stability. The only change is *file placement*:
# these types depend only on FiniteFringe + SparseArrays, so defining them here
# keeps the include-order simple (they are available early).
# -----------------------------------------------------------------------------

module IndicatorTypes
# Small shared types for indicator presentations/copresentations.
# These mirror the data in Def. 6.4 (Miller) and are used by Hom/Ext assembly.

using SparseArrays
using ..FiniteFringe: FinitePoset, Upset, Downset, FringeModule  # P, labeling sets
using ..CoreModules: coeff_type

"""
    UpsetPresentation{K}

A one-step **upset presentation** (Def. 6.4): F1 --delta--> F0 -> M with labels by upsets.
Fields:
- `P`: underlying finite poset
- `U0`: column labels for F0 (birth upsets)
- `U1`: row labels for F1
- `delta`: block monomial matrix delta: (#U1) x (#U0), entries in `K`
- `H`: source fringe module when available (used for exact derived-functor computations)
"""
struct UpsetPresentation{K}
    P::FinitePoset
    U0::Vector{Upset}
    U1::Vector{Upset}
    delta::SparseMatrixCSC{K,Int}
    H::Union{Nothing, FringeModule{K}}

    function UpsetPresentation{K}(P, U0, U1, delta, H; field = nothing) where {K}
        deltaK = SparseMatrixCSC{K,Int}(delta)
        return new{K}(P, U0, U1, deltaK, H)
    end
end

"""
    DownsetCopresentation{K}

A one-step **downset copresentation** (Def. 6.4 dual): M -> E^0 --rho--> E^1 with labels by downsets.
Fields:
- `P`: underlying finite poset
- `D0`: row labels for E^0 (death downsets)
- `D1`: column labels for E^1
- `rho`: block monomial matrix rho: (#D1) x (#D0), entries in `K`
- `H`: target fringe module when available (used for exact derived-functor computations)
"""
struct DownsetCopresentation{K}
    P::FinitePoset
    D0::Vector{Downset}
    D1::Vector{Downset}
    rho::SparseMatrixCSC{K,Int}
    H::Union{Nothing, FringeModule{K}}

    function DownsetCopresentation{K}(P, D0, D1, rho, H; field = nothing) where {K}
        rhoK = SparseMatrixCSC{K,Int}(rho)
        return new{K}(P, D0, D1, rhoK, H)
    end
end

function UpsetPresentation(
    P::FinitePoset,
    U0::Vector{Upset},
    U1::Vector{Upset},
    delta::SparseMatrixCSC,
    H;
    field = nothing,
)
    if field === nothing
        return IndicatorTypes.UpsetPresentation{eltype(delta)}(P, U0, U1, delta, H)
    end
    K = coeff_type(field)
    deltaK = SparseMatrixCSC{K,Int}(delta)
    return IndicatorTypes.UpsetPresentation{K}(P, U0, U1, deltaK, H)
end

function DownsetCopresentation(
    P::FinitePoset,
    D0::Vector{Downset},
    D1::Vector{Downset},
    rho::SparseMatrixCSC,
    H;
    field = nothing,
)
    if field === nothing
        return IndicatorTypes.DownsetCopresentation{eltype(rho)}(P, D0, D1, rho, H)
    end
    K = coeff_type(field)
    rhoK = SparseMatrixCSC{K,Int}(rho)
    return IndicatorTypes.DownsetCopresentation{K}(P, D0, D1, rhoK, H)
end

export UpsetPresentation, DownsetCopresentation
end # module IndicatorTypes


module Encoding
# =============================================================================
# Finite encodings ("uptight posets") from a finite family of constant upsets.
#
# References: Miller section 4 (Defs. 4.12 - 4.18 and Thm. 4.19 - 4.22).
# =============================================================================

using SparseArrays
using ..FiniteFringe
import ..FiniteFringe: nvertices, leq

# This submodule defines methods on CoreModules.AbstractPLikeEncodingMap and
# calls CoreModules.locate/dimension/etc., so we must import the sibling module
# binding into scope (not just individual names).
import ..CoreModules

# ----------------------------- Data structures -------------------------------

"""
    EncodingMap

A finite encoding `pi : Q to P`, where `Q` and `P` are finite posets.

Fields
- `Q` : source poset
- `P` : target poset (the uptight poset)
- `pi_of_q` : a vector of length `nvertices(Q)` with `pi_of_q[q] in 1:nvertices(P)`
"""
struct EncodingMap{Q<:AbstractPoset,P<:AbstractPoset}
    Q::Q
    P::P
    pi_of_q::Vector{Int}
end

"""
    UptightEncoding

Bundle that stores `pi` together with the family `Y` of constant upsets used
to construct the uptight poset. This is handy for inspection and debugging.
"""
struct UptightEncoding
    pi::EncodingMap
    Y::Vector{FiniteFringe.Upset}
end

# ------------------ Postcomposition with a finite encoding map ---------------

"""
    PostcomposedEncodingMap(pi0, pi)

Postcompose an *ambient* encoding map `pi0` (e.g. Z^n- or R^n-encoding) with a *finite*
encoding map `pi : Q -> P` (an `EncodingMap`). Conceptually:

    x mapsto q = locate(pi0, x) mapsto pi(q)

This is used by `Workflow.coarsen` to keep user-facing `encode(...)` semantics stable
while compressing the *finite* encoding poset.
"""
struct PostcomposedEncodingMap{PI<:CoreModules.AbstractPLikeEncodingMap} <: CoreModules.AbstractPLikeEncodingMap
    pi0::PI
    pi_of_q::Vector{Int}              # pi : Q -> P encoded as forward table
    Pn::Int                           # number of regions in P
    reps_cache::Base.RefValue{Union{Nothing,Vector{Tuple}}}    # lazy cache for representatives(::...)
end

@inline PostcomposedEncodingMap(pi0::CoreModules.AbstractPLikeEncodingMap, pi::EncodingMap) =
    PostcomposedEncodingMap(pi0, pi.pi_of_q, nvertices(pi.P), Ref{Union{Nothing,Vector{Tuple}}}(nothing))

@inline function CoreModules.locate(pi::PostcomposedEncodingMap, x::AbstractVector)
    q = CoreModules.locate(pi.pi0, x)
    q == 0 && return 0
    @inbounds return pi.pi_of_q[q]
end

# Disambiguation:
# There is a generic fallback
#   locate(::AbstractPLikeEncodingMap, ::NTuple{N,<:Real})
# in CoreModules. Since PostcomposedEncodingMap <: AbstractPLikeEncodingMap,
# we must ensure our method is MORE specific on the x-type too, otherwise
# calls like locate(pi, (q,)) become ambiguous.
@inline function CoreModules.locate(
    pi::PostcomposedEncodingMap,
    x::NTuple{N,T},
) where {N, T<:Real}
    q = CoreModules.locate(pi.pi0, x)
    q == 0 && return 0
    @inbounds return pi.pi_of_q[q]
end


@inline CoreModules.dimension(pi::PostcomposedEncodingMap) = CoreModules.dimension(pi.pi0)
@inline CoreModules.axes_from_encoding(pi::PostcomposedEncodingMap) = CoreModules.axes_from_encoding(pi.pi0)

function CoreModules.representatives(pi::PostcomposedEncodingMap)
    cached = pi.reps_cache[]
    cached !== nothing && return cached

    reps0 = CoreModules.representatives(pi.pi0)
    repsP = Vector{Tuple}(undef, pi.Pn)
    filled = falses(pi.Pn)

    @inbounds for q in eachindex(pi.pi_of_q)
        p = pi.pi_of_q[q]
        if !filled[p]
            repsP[p] = reps0[q] isa Tuple ? reps0[q] : Tuple(reps0[q])
            filled[p] = true
        end
    end

    all(filled) || error("PostcomposedEncodingMap: could not build representatives for all regions.")

    pi.reps_cache[] = repsP
    return repsP
end

# ----------------------- Uptight regions from a family Y ---------------------

# Partition Q into uptight regions: a ~ b iff they lie in exactly the same members of Y.
function _uptight_regions(Q::FiniteFringe.AbstractPoset, Y::Vector{FiniteFringe.Upset})
    m = length(Y)
    if m <= 64
        sigs = Dict{UInt64, Vector{Int}}()
        @inbounds for q in 1:nvertices(Q)
            key = zero(UInt64)
            for i in 1:m
                if Y[i].mask[q]
                    key |= (UInt64(1) << (i - 1))
                end
            end
            push!(get!(sigs, key, Int[]), q)
        end
        return collect(values(sigs))
    elseif m <= 128
        sigs = Dict{Tuple{UInt64,UInt64}, Vector{Int}}()
        @inbounds for q in 1:nvertices(Q)
            lo = zero(UInt64)
            hi = zero(UInt64)
            for i in 1:64
                if Y[i].mask[q]
                    lo |= (UInt64(1) << (i - 1))
                end
            end
            for i in 65:m
                if Y[i].mask[q]
                    hi |= (UInt64(1) << (i - 65))
                end
            end
            key = (lo, hi)
            push!(get!(sigs, key, Int[]), q)
        end
        return collect(values(sigs))
    else
        # Fallback for very large Y: keep content-hashable tuple signatures.
        sigs = Dict{Tuple{Vararg{Bool}}, Vector{Int}}()
        @inbounds for q in 1:nvertices(Q)
            key = ntuple(i -> Y[i].mask[q], m)
            push!(get!(sigs, key, Int[]), q)
        end
        return collect(values(sigs))
    end
end

# Build the partial order on regions: A <= B if exists a \in A, b \in B with a <= b in Q (Prop. 4.15),
# then take transitive closure (Def. 4.17).
function _uptight_poset(Q::FiniteFringe.AbstractPoset, regions::Vector{Vector{Int}};
                        poset_kind::Symbol = :regions)
    r = length(regions)
    if poset_kind == :regions
        return RegionsPoset(Q, regions)
    elseif poset_kind == :dense
        rel = falses(r, r)
        for A in 1:r, B in 1:r
            if A == B
                rel[A,B] = true
                continue
            end
            found = false
            for a in regions[A], b in regions[B]
                if FiniteFringe.leq(Q, a, b); found = true; break; end
            end
            rel[A,B] = found
        end
        # Transitive closure (Floyd-Warshall boolean)
        for k in 1:r, i in 1:r, j in 1:r
            rel[i,j] = rel[i,j] || (rel[i,k] && rel[k,j])
        end
        return FiniteFringe.FinitePoset(rel; check=false)
    else
        error("_uptight_poset: poset_kind must be :regions or :dense")
    end
end

# Build the encoding map pi : Q \to P_Y
function _encoding_map(Q::FiniteFringe.AbstractPoset,
                       P::FiniteFringe.AbstractPoset,
                       regions::Vector{Vector{Int}})
    pi_of_q = zeros(Int, nvertices(Q))
    for (idx, R) in enumerate(regions)
        for q in R
            pi_of_q[q] = idx
        end
    end
    EncodingMap(Q, P, pi_of_q)
end

# -------------------------- Image / preimage helpers -------------------------

"Image of a Q-upset under `pi` as a P-upset (Def. 4.12 / Remark 4.13)."
function image_upset(pi::EncodingMap, U::FiniteFringe.Upset)
    maskP = falses(nvertices(pi.P))
    for q in 1:nvertices(pi.Q)
        if U.mask[q]; maskP[pi.pi_of_q[q]] = true; end
    end
    FiniteFringe.upset_closure(pi.P, maskP)
end

"Image of a Q-downset under `pi` as a P-downset."
function image_downset(pi::EncodingMap, D::FiniteFringe.Downset)
    maskP = falses(nvertices(pi.P))
    for q in 1:nvertices(pi.Q)
        if D.mask[q]; maskP[pi.pi_of_q[q]] = true; end
    end
    FiniteFringe.downset_closure(pi.P, maskP)
end

"Preimage of a P-upset under `pi` as a Q-upset."
function preimage_upset(pi::EncodingMap, Uhat::FiniteFringe.Upset)
    maskQ = falses(nvertices(pi.Q))
    for q in 1:nvertices(pi.Q)
        if Uhat.mask[pi.pi_of_q[q]]; maskQ[q] = true; end
    end
    FiniteFringe.upset_closure(pi.Q, maskQ)
end

"Preimage of a P-downset under `pi` as a Q-downset."
function preimage_downset(pi::EncodingMap, Dhat::FiniteFringe.Downset)
    maskQ = falses(nvertices(pi.Q))
    for q in 1:nvertices(pi.Q)
        if Dhat.mask[pi.pi_of_q[q]]; maskQ[q] = true; end
    end
    FiniteFringe.downset_closure(pi.Q, maskQ)
end

# ---------------------------- Public constructors ----------------------------

"""
    build_uptight_encoding_from_fringe(M::FringeModule; poset_kind=:regions) -> UptightEncoding

Given a fringe presentation on `Q` with upsets `U_i` (births) and downsets `D_j` (deaths),
form the finite family `Y = { U_i } cup { complement(D_j) }` of constant upsets (Def. 4.18),
build the uptight regions (Defs. 4.12 - 4.17), and return the finite encoding `pi: Q -> P_Y`.
`poset_kind=:regions` returns a structured `RegionsPoset`, `:dense` materializes a `FinitePoset`.
"""
function build_uptight_encoding_from_fringe(M::FiniteFringe.FringeModule;
                                            poset_kind::Symbol = :regions)
    Q = M.P
    Y = FiniteFringe.Upset[]
    append!(Y, M.U)
    for Dj in M.D
        comp = BitVector(.!Dj.mask)                 # complement is also a Q-upset
        push!(Y, FiniteFringe.upset_closure(Q, comp))
    end
    regions = _uptight_regions(Q, Y)
    P = _uptight_poset(Q, regions; poset_kind = poset_kind)
    pi = _encoding_map(Q, P, regions)
    UptightEncoding(pi, Y)
end

"""
    pullback_fringe_along_encoding(H_hat::FringeModule_on_P, pi::EncodingMap) -> FringeModule_on_Q

Prop. 4.11 (used in the proof of Thm. 6.12): pull back a monomial matrix for a module on `P`
by replacing row labels `D_hat_j` with `pi^{-1}(D_hat_j)` and column labels `U_hat_i` with `pi^{-1}(U_hat_i)`.
The scalar matrix is unchanged.
"""
function pullback_fringe_along_encoding(Hhat::FiniteFringe.FringeModule, pi::EncodingMap)
    UQ = [preimage_upset(pi, Uhat) for Uhat in Hhat.U]
    DQ = [preimage_downset(pi, Dhat) for Dhat in Hhat.D]
    FiniteFringe.FringeModule{eltype(Hhat.phi)}(pi.Q, UQ, DQ, Hhat.phi)
end

"""
    pushforward_fringe_along_encoding(H::FringeModule_on_Q, pi::EncodingMap) -> FringeModule_on_P

Push a fringe presentation forward along a finite encoding map `pi : Q -> P`.

This sends each upset generator `U_i` of `H` to `image_upset(pi, U_i)` and each
downset generator `D_j` to `image_downset(pi, D_j)`, while keeping the scalar matrix
`phi` unchanged.
"""
function pushforward_fringe_along_encoding(H::FiniteFringe.FringeModule, pi::EncodingMap)
    Uhat = [image_upset(pi, U) for U in H.U]
    Dhat = [image_downset(pi, D) for D in H.D]
    FiniteFringe.FringeModule{eltype(H.phi)}(pi.P, Uhat, Dhat, H.phi)
end


export EncodingMap, UptightEncoding,
       build_uptight_encoding_from_fringe, pullback_fringe_along_encoding,
       image_upset, image_downset, preimage_upset, preimage_downset,
       pullback_fringe_along_encoding, pushforward_fringe_along_encoding       

end # module
