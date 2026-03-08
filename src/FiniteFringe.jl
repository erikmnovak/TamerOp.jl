module FiniteFringe

using SparseArrays, LinearAlgebra
using ..CoreModules: QQ, QQField, AbstractCoeffField, coeff_type, field_from_eltype, coerce
using ..Options: FiniteFringeOptions
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
const CHAIN_PARENT_DENSE_MIN_ENTRIES = Ref(4_096)
const CHAIN_PARENT_DENSE_MAX_ENTRIES_PER_THREAD = Ref(1_000_000)
const CHAIN_PARENT_DENSE_MAX_TOTAL_ENTRIES = Ref(8_000_000)

@inline _updown_cache_skip_auto(::AbstractPoset) = false
@inline _updown_cache_threshold(::AbstractPoset) = UPDOWN_CACHE_THRESHOLD_GENERIC[]

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

function _cached_upset_indices(P::AbstractPoset, i::Int)
    cache = _ensure_updown_cache!(P)
    cache === nothing && return nothing
    return cache[1][i]
end

function _cached_downset_indices(P::AbstractPoset, i::Int)
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

mutable struct _ChainParentDenseMemo
    seen::BitVector
    vals::Vector{Int}
    n::Int
end

struct _PackedAdjacency
    ptr::Vector{Int}
    idx::Vector{Int}
end

@inline function _adj_bounds(adj::_PackedAdjacency, v::Int)
    return adj.ptr[v], adj.ptr[v + 1] - 1
end

struct _PackedIntSlice <: AbstractVector{Int}
    data::Vector{Int}
    lo::Int
    hi::Int
end

Base.IndexStyle(::Type{_PackedIntSlice}) = IndexLinear()
Base.firstindex(::_PackedIntSlice) = 1
@inline Base.length(s::_PackedIntSlice) = max(0, s.hi - s.lo + 1)
Base.lastindex(s::_PackedIntSlice) = length(s)
Base.size(s::_PackedIntSlice) = (length(s),)
Base.axes(s::_PackedIntSlice) = (Base.OneTo(length(s)),)
Base.isempty(s::_PackedIntSlice) = length(s) == 0

@inline function Base.getindex(s::_PackedIntSlice, i::Int)
    @boundscheck checkbounds(s, i)
    return @inbounds s.data[s.lo + i - 1]
end

@inline function Base.iterate(s::_PackedIntSlice, state::Int=1)
    state > length(s) && return nothing
    return s[state], state + 1
end

"""
    CoverCache(Q)

An internal cache for poset cover data and a hot-path memo used by `map_leq`.
This cache is stored lazily on each poset object (via `PosetCache`).

Thread-safety:

- `succs`, `preds`, and `C` are read-only once constructed.
- `chain_parent` / `chain_parent_dense` are per-thread hot-path memos for witness
  predecessor lookups. Writes are thread-local and do not require locks.
"""
struct CoverCache
    Q::AbstractPoset
    C::Union{BitMatrix,Nothing}
    succ_ptr::Vector{Int}
    succ_idx::Vector{Int}
    succ_pred_slot::Vector{Int}
    pred_ptr::Vector{Int}
    pred_idx::Vector{Int}
    pred_succ_slot::Vector{Int}
    undir::Union{Nothing,_PackedAdjacency}

    # Sparse fallback memo: chain_parent[tid][pairkey(a,d)] = chosen predecessor b.
    chain_parent::Vector{Dict{UInt64, Int}}
    # Dense memo for finite posets when n^2 is moderate; optional for memory control.
    chain_parent_dense::Union{Nothing,Vector{_ChainParentDenseMemo}}

    # Number of cover edges in Q (used to size edge-indexed stores).
    nedges::Int
end

@inline function _succs(cc::CoverCache, u::Int)
    lo = cc.succ_ptr[u]
    hi = cc.succ_ptr[u + 1] - 1
    return _PackedIntSlice(cc.succ_idx, lo, hi)
end

@inline function _preds(cc::CoverCache, v::Int)
    lo = cc.pred_ptr[v]
    hi = cc.pred_ptr[v + 1] - 1
    return _PackedIntSlice(cc.pred_idx, lo, hi)
end

@inline function _pred_slots_of_succ(cc::CoverCache, u::Int)
    lo = cc.succ_ptr[u]
    hi = cc.succ_ptr[u + 1] - 1
    return _PackedIntSlice(cc.succ_pred_slot, lo, hi)
end

@inline function _succ_slots_of_pred(cc::CoverCache, v::Int)
    lo = cc.pred_ptr[v]
    hi = cc.pred_ptr[v + 1] - 1
    return _PackedIntSlice(cc.pred_succ_slot, lo, hi)
end

mutable struct PosetCache
    cover_edges::Union{Nothing,CoverEdges}
    cover::Union{Nothing,CoverCache}
    lock::Base.ReentrantLock
    upsets::Union{Nothing,Vector{Vector{Int}}}
    downsets::Union{Nothing,Vector{Vector{Int}}}
    hom_route_choice::Dict{UInt64,Symbol}
    PosetCache() = new(nothing, nothing, Base.ReentrantLock(), nothing, nothing, Dict{UInt64,Symbol}())
end

function Base.propertynames(::CoverCache, private::Bool=false)
    base = (:Q, :C, :succ_ptr, :succ_idx, :succ_pred_slot,
            :pred_ptr, :pred_idx, :pred_succ_slot,
            :undir, :chain_parent, :chain_parent_dense, :nedges)
    return private ? base : base
end

# Internal lazy accessor for per-poset cover cache.
function _get_cover_cache(Q::AbstractPoset)
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
    error("_get_cover_cache: poset type $(typeof(Q)) does not support caching")
end

"""
    build_cache!(Q; cover=true, updown=true)

Canonical public cache-build entrypoint for a poset. Use this before entering threaded
read-only loops to avoid lock contention and duplicate lazy-initialization work.
"""
function build_cache!(Q::AbstractPoset; cover::Bool=true, updown::Bool=true)
    cover && _get_cover_cache(Q)
    updown && _ensure_updown_cache!(Q)
    return Q
end

"""
    _clear_cover_cache!(Q)

Clear the cached cover data stored on a poset object.
"""
function _clear_cover_cache!(Q::AbstractPoset)
    if hasproperty(Q, :cache)
        pc = getproperty(Q, :cache)
        if pc isa PosetCache
            Base.lock(pc.lock)
            try
                pc.cover_edges = nothing
                pc.cover = nothing
                pc.upsets = nothing
                pc.downsets = nothing
                empty!(pc.hom_route_choice)
            finally
                Base.unlock(pc.lock)
            end
            return nothing
        end
    end
    error("_clear_cover_cache!: poset type $(typeof(Q)) does not support caching")
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

    succ_ptr = Vector{Int}(undef, n + 1)
    pred_ptr = Vector{Int}(undef, n + 1)
    succ_ptr[1] = 1
    pred_ptr[1] = 1
    @inbounds for u in 1:n
        succ_ptr[u + 1] = succ_ptr[u] + outdeg[u]
        pred_ptr[u + 1] = pred_ptr[u] + indeg[u]
    end

    succ_idx = Vector{Int}(undef, nedges)
    pred_idx = Vector{Int}(undef, nedges)
    outk = copy(succ_ptr)
    ink = copy(pred_ptr)

    for (a, b) in Ce
        succ_idx[outk[a]] = b
        pred_idx[ink[b]] = a
        outk[a] += 1
        ink[b] += 1
    end

    @inbounds for u in 1:n
        slo, shi = succ_ptr[u], succ_ptr[u + 1] - 1
        plo, phi = pred_ptr[u], pred_ptr[u + 1] - 1
        slo <= shi && sort!(@view succ_idx[slo:shi])
        plo <= phi && sort!(@view pred_idx[plo:phi])
    end

    succ_pred_slot = Vector{Int}(undef, nedges)
    pred_succ_slot = Vector{Int}(undef, nedges)
    @inbounds for u in 1:n
        slo, shi = succ_ptr[u], succ_ptr[u + 1] - 1
        for sp in slo:shi
            v = succ_idx[sp]
            plo, phi = pred_ptr[v], pred_ptr[v + 1] - 1
            i = searchsortedfirst(@view(pred_idx[plo:phi]), u)
            predp = plo + i - 1
            succ_pred_slot[sp] = i
            pred_succ_slot[predp] = sp - slo + 1
        end
    end

    nt = max(1, Base.Threads.maxthreadid())
    chain_parent = [Dict{UInt64, Int}() for _ in 1:max(1, Base.Threads.maxthreadid())]
    chain_parent_dense = nothing
    dense_entries = n * n
    use_dense_parent = (Q isa FinitePoset) &&
                       (dense_entries >= CHAIN_PARENT_DENSE_MIN_ENTRIES[]) &&
                       (dense_entries <= CHAIN_PARENT_DENSE_MAX_ENTRIES_PER_THREAD[]) &&
                       (dense_entries * nt <= CHAIN_PARENT_DENSE_MAX_TOTAL_ENTRIES[])
    if use_dense_parent
        chain_parent_dense = [
            _ChainParentDenseMemo(falses(dense_entries), zeros(Int, dense_entries), n)
            for _ in 1:nt
        ]
    end

    undir_ptr = Vector{Int}(undef, n + 1)
    undir_ptr[1] = 1
    @inbounds for u in 1:n
        undir_ptr[u + 1] = undir_ptr[u] + outdeg[u] + indeg[u]
    end
    undir_idx = Vector{Int}(undef, undir_ptr[end] - 1)
    undir_k = copy(undir_ptr)
    @inbounds for u in 1:n
        for p in succ_ptr[u]:(succ_ptr[u + 1] - 1)
            undir_idx[undir_k[u]] = succ_idx[p]
            undir_k[u] += 1
        end
        for p in pred_ptr[u]:(pred_ptr[u + 1] - 1)
            undir_idx[undir_k[u]] = pred_idx[p]
            undir_k[u] += 1
        end
    end

    return CoverCache(Q, C, succ_ptr, succ_idx, succ_pred_slot,
                      pred_ptr, pred_idx, pred_succ_slot,
                      _PackedAdjacency(undir_ptr, undir_idx),
                      chain_parent, chain_parent_dense, nedges)
end

@inline function _chain_parent_dict(cc::CoverCache)::Dict{UInt64, Int}
    return cc.chain_parent[min(length(cc.chain_parent), max(1, Base.Threads.threadid()))]
end

@inline function _chain_parent_dense(cc::CoverCache)
    dense = cc.chain_parent_dense
    dense === nothing && return nothing
    return dense[min(length(dense), max(1, Base.Threads.threadid()))]
end

function _clear_chain_parent_cache!(cc::CoverCache)
    for d in cc.chain_parent
        empty!(d)
    end
    if cc.chain_parent_dense !== nothing
        for m in cc.chain_parent_dense
            fill!(m.seen, false)
        end
    end
    return nothing
end

@inline function _chosen_predecessor_slow(cc::CoverCache, a::Int, d::Int)
    lo, hi = cc.pred_ptr[d], cc.pred_ptr[d + 1] - 1
    @inbounds for p in lo:hi
        b = cc.pred_idx[p]
        (b != a && leq(cc.Q, a, b)) && return b
    end
    return a
end

function _chosen_predecessor(cc::CoverCache, a::Int, d::Int)
    dense = _chain_parent_dense(cc)
    if dense !== nothing
        idx = (a - 1) * dense.n + d
        @inbounds if dense.seen[idx]
            return dense.vals[idx]
        end
        b = _chosen_predecessor_slow(cc, a, d)
        @inbounds begin
            dense.vals[idx] = b
            dense.seen[idx] = true
        end
        return b
    end

    k = _pairkey(a, d)
    chain_parent = _chain_parent_dict(cc)
    b = get(chain_parent, k, 0)
    if b == 0
        b = _chosen_predecessor_slow(cc, a, d)
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

Return an iterable of indices in the principal upset/downset of `i`.
Fallback implementations scan the whole poset.
"""
function upset_indices(P::AbstractPoset, i::Int)
    cached = _cached_upset_indices(P, i)
    cached === nothing || return IndicesView(cached)
    return PosetLeqIter(P, i, true)
end

function downset_indices(P::AbstractPoset, i::Int)
    cached = _cached_downset_indices(P, i)
    cached === nothing || return IndicesView(cached)
    return PosetLeqIter(P, i, false)
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
    cached = _cached_upset_indices(P, i)
    cached === nothing || return IndicesView(cached)
    row = @view P._leq[i, :]
    return BitRowIter(row)
end

function downset_indices(P::FinitePoset, i::Int)
    cached = _cached_downset_indices(P, i)
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
    cached = _cached_upset_indices(P, i)
    cached === nothing || return IndicesView(cached)
    ranges = ntuple(k -> _coord_at(i, P.sizes[k], P.strides[k]):P.sizes[k], N)
    return ProductIndexIter(CartesianIndices(ranges), P.strides)
end

function downset_indices(P::ProductOfChainsPoset{N}, i::Int) where {N}
    cached = _cached_downset_indices(P, i)
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

struct Upset{P<:AbstractPoset}
    P::P
    mask::BitVector
    function Upset(Pobj::PT, mask::BitVector) where {PT<:AbstractPoset}
        nvertices(Pobj) == length(mask) ||
            error("Upset: mask length $(length(mask)) must equal nvertices(P)=$(nvertices(Pobj)).")
        new{PT}(Pobj, mask)
    end
end
struct Downset{P<:AbstractPoset}
    P::P
    mask::BitVector
    function Downset(Pobj::PT, mask::BitVector) where {PT<:AbstractPoset}
        nvertices(Pobj) == length(mask) ||
            error("Downset: mask length $(length(mask)) must equal nvertices(P)=$(nvertices(Pobj)).")
        new{PT}(Pobj, mask)
    end
end

struct _WPairData
    # Packed row lists by source-upset component.
    u_ptr::Vector{Int}
    u_rows::Vector{Int}
    # Packed row lists by target-downset component.
    d_ptr::Vector{Int}
    d_rows::Vector{Int}
end

struct _FringeComponentDecomp
    comp_id::Vector{Vector{Int}}
    comp_masks::Vector{Vector{BitVector}}
    comp_n::Vector{Int}
end

mutable struct _FringeDenseIdxPlan{K}
    W_dim::Int
    V1_dim::Int
    V2_dim::Int
    t_rows::Vector{Int}
    t_cols::Vector{Int}
    t_tN::Vector{Int}
    t_jN::Vector{Int}
    s_rows::Vector{Int}
    s_cols::Vector{Int}
    s_sM::Vector{Int}
    s_iM::Vector{Int}
    Tbuf::Matrix{K}
    Sbuf::Matrix{K}
    bigbuf::Matrix{K}
end

mutable struct _FringeDensePathPlan{K}
    W_dim::Int
    V1::Vector{NTuple{3,Int}}
    V2::Vector{NTuple{3,Int}}
    w_index::Matrix{Int}
    w_data::Vector{_WPairData}
    nUM::Int
    nDN::Int
    Tbuf::Matrix{K}
    Sbuf::Matrix{K}
    bigbuf::Matrix{K}
end

mutable struct _FringeSparsePlan{K}
    W_dim::Int
    V1::Vector{NTuple{3,Int}}
    V2::Vector{NTuple{3,Int}}
    w_index::Matrix{Int}
    w_data::Vector{_WPairData}
    nUM::Int
    nDN::Int
    T::SparseMatrixCSC{K,Int}
    S::SparseMatrixCSC{K,Int}
    t_tN::Vector{Int}
    t_jN::Vector{Int}
    s_sM::Vector{Int}
    s_iM::Vector{Int}
    t_nzptr::Union{Nothing,Vector{Int}}
    s_nzptr::Union{Nothing,Vector{Int}}
    t_nzptr_max::Int
    s_nzptr_max::Int
    hcat_buf::SparseMatrixCSC{K,Int}
    hcat_buf_rev::SparseMatrixCSC{K,Int}
    nnzT::Int
    nnzS::Int
end

mutable struct _FringePairCache{K}
    partner_id::UInt64
    partner_phi_id::UInt64
    dense_idx_plan::Union{Nothing,_FringeDenseIdxPlan{K}}
    dense_path_plan::Union{Nothing,_FringeDensePathPlan{K}}
    sparse_plan::Union{Nothing,_FringeSparsePlan{K}}
    route_choice::Union{Nothing,Symbol}
end

struct _FringeRouteChoiceEntry
    fingerprint::UInt64
    choice::Symbol
end

mutable struct _FringeHomCache{K}
    adj::Union{Nothing,_PackedAdjacency}
    upset::Union{Nothing,_FringeComponentDecomp}
    downset::Union{Nothing,_FringeComponentDecomp}
    pair_cache::Vector{_FringePairCache{K}}
    route_fingerprint_choice::Vector{_FringeRouteChoiceEntry}
    route_timing_fallbacks::Int
    _FringeHomCache{K}() where {K} = new(nothing, nothing, nothing,
                                         _FringePairCache{K}[],
                                         _FringeRouteChoiceEntry[],
                                         0)
end

struct _FiberQueryIndex
    col_ptr::Vector{Int}
    col_idx::Vector{Int}
    row_ptr::Vector{Int}
    row_idx::Vector{Int}
end

const FIBER_DIM_CACHE_BUILD_AFTER = Ref(2)
const FIBER_DIM_EAGER_INDEX_MAX_CELLS = Ref(65_536)
const HOM_PAIR_CACHE_MAX_ENTRIES = Ref(8)


Base.length(U::Upset) = length(U.mask)
Base.length(D::Downset) = length(D.mask)
Base.eltype(::Type{<:Upset}) = Int
Base.eltype(::Type{<:Downset}) = Int
Base.IteratorSize(::Type{<:Upset}) = Base.HasLength()
Base.IteratorSize(::Type{<:Downset}) = Base.HasLength()

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

function upset_from_generators(P::AbstractPoset, gens::AbstractVector{<:Integer})
    n = nvertices(P)
    mask = falses(n)
    @inbounds for g in gens
        gi = Int(g)
        (1 <= gi <= n) || error("upset_from_generators: generator index $gi out of bounds for poset with $n vertices.")
        mask[gi] = true
    end
    return upset_closure(P, mask)
end

function downset_from_generators(P::AbstractPoset, gens::AbstractVector{<:Integer})
    n = nvertices(P)
    mask = falses(n)
    @inbounds for g in gens
        gi = Int(g)
        (1 <= gi <= n) || error("downset_from_generators: generator index $gi out of bounds for poset with $n vertices.")
        mask[gi] = true
    end
    return downset_closure(P, mask)
end

function upset_from_generators(P::AbstractPoset, gens_mask::AbstractVector{Bool})
    n = nvertices(P)
    length(gens_mask) == n ||
        error("upset_from_generators: mask length $(length(gens_mask)) must equal nvertices(P)=$n.")
    return upset_closure(P, BitVector(gens_mask))
end

function downset_from_generators(P::AbstractPoset, gens_mask::AbstractVector{Bool})
    n = nvertices(P)
    length(gens_mask) == n ||
        error("downset_from_generators: mask length $(length(gens_mask)) must equal nvertices(P)=$n.")
    return downset_closure(P, BitVector(gens_mask))
end

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

struct FringeModule{K, P<:AbstractPoset, F<:AbstractCoeffField, MAT<:AbstractMatrix{K}}
    field::F
    P::P
    U::Vector{Upset{P}}               # birth upsets (columns)
    D::Vector{Downset{P}}             # death downsets (rows)
    phi::MAT                          # size |D| x |U|
    phi_density::Float64
    fiber_index::Base.RefValue{Union{Nothing,_FiberQueryIndex}}
    fiber_queries::Base.RefValue{Int}
    fiber_dims::Base.RefValue{Union{Nothing,Vector{Int}}}
    hom_cache::Base.RefValue{_FringeHomCache{K}}

    function FringeModule{K,P,F,MAT}(field::F,
                                     Pobj::P,
                                     U::Vector{Upset{P}},
                                     D::Vector{Downset{P}},
                                     phi::MAT) where {K, P<:AbstractPoset, F<:AbstractCoeffField, MAT<:AbstractMatrix{K}}
        @assert size(phi,1) == length(D) && size(phi,2) == length(U)
        coeff_type(field) == K || error("FringeModule: coeff_type(field) != K")

        idx = _should_build_fiber_query_index(Pobj, U, D) ?
            _build_fiber_query_index(Pobj, U, D) : nothing

        M = new{K,P,F,MAT}(field, Pobj, U, D, phi, _matrix_density(phi),
                         Ref{Union{Nothing,_FiberQueryIndex}}(idx),
                         Ref{Int}(0),
                         Ref{Union{Nothing,Vector{Int}}}(nothing),
                         Ref(_FringeHomCache{K}()))
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
    return FringeModule{K2, typeof(H.P), typeof(field), typeof(Phi)}(field, H.P, H.U, H.D, Phi)
end

# Canonical public constructor: infer matrix storage from `phi`, require explicit `field`.
@inline function _coerce_upsets(P::PT, Uin::AbstractVector{<:Upset}) where {PT<:AbstractPoset}
    U = Vector{Upset{PT}}(undef, length(Uin))
    @inbounds for i in eachindex(Uin)
        Ui = Uin[i]
        Ui.P === P || error("FringeModule: all upsets must belong to the same poset object P.")
        U[i] = Ui::Upset{PT}
    end
    return U
end

@inline function _coerce_downsets(P::PT, Din::AbstractVector{<:Downset}) where {PT<:AbstractPoset}
    D = Vector{Downset{PT}}(undef, length(Din))
    @inbounds for i in eachindex(Din)
        Di = Din[i]
        Di.P === P || error("FringeModule: all downsets must belong to the same poset object P.")
        D[i] = Di::Downset{PT}
    end
    return D
end

function FringeModule{K}(P::PT,
                         Uin::AbstractVector{<:Upset},
                         Din::AbstractVector{<:Downset},
                         phi::AbstractMatrix{K};
                         field::AbstractCoeffField) where {K,PT<:AbstractPoset}
    F = typeof(field)
    U = _coerce_upsets(P, Uin)
    D = _coerce_downsets(P, Din)
    return FringeModule{K, PT, F, typeof(phi)}(field, P, U, D, phi)
end

"""
    one_by_one_fringe(P, U, D, scalar; field=QQField()) -> FringeModule

Convenience constructor for a 1x1 fringe presentation:

- one upset generator `U`,
- one downset cogenerator `D`,
- and a 1x1 structure matrix `phi = [scalar]`.

This is handy for tests, examples, and quickly building interval-like modules
on a finite poset without writing out the matrix by hand.

Notes:
- The scalar is required; pass `one(coeff_type(field))` explicitly when desired.
- This method is fully generic in the scalar type `K`.
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
@inline function _should_build_fiber_query_index(P::AbstractPoset,
                                                 U::AbstractVector,
                                                 D::AbstractVector)
    return nvertices(P) * (length(U) + length(D)) <= FIBER_DIM_EAGER_INDEX_MAX_CELLS[]
end

function _build_fiber_query_index(P::AbstractPoset,
                                  Usets::AbstractVector,
                                  Dsets::AbstractVector)
    n = nvertices(P)
    col_counts = zeros(Int, n)
    row_counts = zeros(Int, n)

    @inbounds for U in Usets
        q = findnext(U.mask, 1)
        while q !== nothing
            col_counts[q] += 1
            q = findnext(U.mask, q + 1)
        end
    end

    @inbounds for D in Dsets
        q = findnext(D.mask, 1)
        while q !== nothing
            row_counts[q] += 1
            q = findnext(D.mask, q + 1)
        end
    end

    col_ptr = Vector{Int}(undef, n + 1)
    row_ptr = Vector{Int}(undef, n + 1)
    col_ptr[1] = 1
    row_ptr[1] = 1
    @inbounds for q in 1:n
        col_ptr[q + 1] = col_ptr[q] + col_counts[q]
        row_ptr[q + 1] = row_ptr[q] + row_counts[q]
    end

    col_idx = Vector{Int}(undef, col_ptr[end] - 1)
    row_idx = Vector{Int}(undef, row_ptr[end] - 1)
    col_next = copy(col_ptr)
    row_next = copy(row_ptr)

    @inbounds for (i, U) in enumerate(Usets)
        q = findnext(U.mask, 1)
        while q !== nothing
            col_idx[col_next[q]] = i
            col_next[q] += 1
            q = findnext(U.mask, q + 1)
        end
    end

    @inbounds for (j, D) in enumerate(Dsets)
        q = findnext(D.mask, 1)
        while q !== nothing
            row_idx[row_next[q]] = j
            row_next[q] += 1
            q = findnext(D.mask, q + 1)
        end
    end

    return _FiberQueryIndex(col_ptr, col_idx, row_ptr, row_idx)
end

function _build_fiber_query_index(M::FringeModule)
    return _build_fiber_query_index(M.P, M.U, M.D)
end

@inline function _ensure_fiber_query_index!(M::FringeModule)
    idx = M.fiber_index[]
    idx !== nothing && return idx
    idx = _build_fiber_query_index(M)
    M.fiber_index[] = idx
    return idx
end

@inline function _ensure_fiber_dim_cache!(M::FringeModule)
    dims = M.fiber_dims[]
    if dims === nothing
        dims = fill(typemin(Int), nvertices(M.P))
        M.fiber_dims[] = dims
    end
    return dims
end

function fiber_dimension(M::FringeModule{K}, q::Int) where {K}
    dims = _ensure_fiber_dim_cache!(M)
    cached = dims[q]
    cached != typemin(Int) && return cached

    idx = M.fiber_index[]
    if idx === nothing
        M.fiber_queries[] += 1
        if _should_build_fiber_query_index(M.P, M.U, M.D) ||
           M.fiber_queries[] >= FIBER_DIM_CACHE_BUILD_AFTER[]
            idx = _ensure_fiber_query_index!(M)
        end
    end

    if idx === nothing
        cols = findall(U -> U.mask[q], M.U)
        rows = findall(D -> D.mask[q], M.D)
        if isempty(cols) || isempty(rows)
            dims[q] = 0
            return 0
        end
        d = FieldLinAlg.rank_restricted(M.field, M.phi, rows, cols)
        dims[q] = d
        return d
    end

    clo, chi = idx.col_ptr[q], idx.col_ptr[q + 1] - 1
    rlo, rhi = idx.row_ptr[q], idx.row_ptr[q + 1] - 1
    if clo > chi || rlo > rhi
        dims[q] = 0
        return 0
    end
    cols = @view idx.col_idx[clo:chi]
    rows = @view idx.row_idx[rlo:rhi]
    d = FieldLinAlg.rank_restricted(M.field, M.phi, rows, cols)
    dims[q] = d
    return d
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

const HOM_DIM_SPARSE_BUILD_MIN_ENTRIES = Ref(20_000)
const HOM_DIM_SPARSE_DENSITY_THRESHOLD = Ref(0.30)
const HOM_DIM_INTERNAL_TINY_WORK_THRESHOLD = Ref(350)
const HOM_DIM_INTERNAL_SPARSE_DENSITY_THRESHOLD = Ref(0.24)
const HOM_DIM_INTERNAL_SPARSE_WORK_THRESHOLD = Ref(80_000)
const HOM_DIM_INTERNAL_DENSE_WORK_THRESHOLD = Ref(8_000)
const HOM_DIM_INTERNAL_WORK_AMBIGUITY_BAND = Ref(500)
const HOM_DIM_INTERNAL_DENSITY_AMBIGUITY_BAND = Ref(0.020)
const HOM_DIM_DENSE_DENSITY_FULL_SCAN_ENTRIES = Ref(32_768)
const HOM_DIM_DENSE_DENSITY_SAMPLE_SIZE = Ref(4_096)

@inline function _matrix_density(A::SparseMatrixCSC)
    m, n = size(A)
    den = m * n
    den == 0 && return 0.0
    return nnz(A) / den
end

@inline function _estimate_dense_matrix_density(A::AbstractMatrix)
    den = length(A)
    den == 0 && return 0.0
    if den <= HOM_DIM_DENSE_DENSITY_FULL_SCAN_ENTRIES[]
        nz = 0
        @inbounds for v in A
            !iszero(v) && (nz += 1)
        end
        return nz / den
    end

    sample_n = min(den, HOM_DIM_DENSE_DENSITY_SAMPLE_SIZE[])
    step = max(1, fld(den, sample_n))
    nz = 0
    seen = 0
    idx = 0
    @inbounds for v in A
        idx += 1
        ((idx - 1) % step == 0) || continue
        !iszero(v) && (nz += 1)
        seen += 1
        seen >= sample_n && break
    end
    seen == 0 && return 0.0
    return nz / seen
end

_matrix_density(A::AbstractMatrix) = _estimate_dense_matrix_density(A)

@inline function _hom_work_estimate(M::FringeModule, N::FringeModule)
    return length(M.U) * length(N.D) +
           length(M.U) * length(N.U) +
           length(M.D) * length(N.D)
end

@inline function _heuristic_hom_internal_choice(M::FringeModule,
                                                N::FringeModule)
    dmin = min(M.phi_density, N.phi_density)
    dmax = max(M.phi_density, N.phi_density)
    work = _hom_work_estimate(M, N)
    tiny_work = HOM_DIM_INTERNAL_TINY_WORK_THRESHOLD[]
    sparse_dens = HOM_DIM_INTERNAL_SPARSE_DENSITY_THRESHOLD[]
    sparse_work = HOM_DIM_INTERNAL_SPARSE_WORK_THRESHOLD[]
    dense_work = HOM_DIM_INTERNAL_DENSE_WORK_THRESHOLD[]
    work_band = HOM_DIM_INTERNAL_WORK_AMBIGUITY_BAND[]
    dens_band = HOM_DIM_INTERNAL_DENSITY_AMBIGUITY_BAND[]

    work <= tiny_work && return :dense_idx_internal

    if dmin <= sparse_dens - dens_band && work <= sparse_work + work_band
        return :sparse_path
    end

    if dmin >= sparse_dens + dens_band
        if work <= dense_work - work_band
            return :dense_idx_internal
        elseif work >= dense_work + work_band
            return :dense_path
        end
        return nothing
    end

    if work <= tiny_work + work_band
        return :dense_idx_internal
    elseif work >= dense_work + work_band
        return :dense_path
    elseif dmax <= sparse_dens && work <= sparse_work + work_band
        return :sparse_path
    end
    return nothing
end

@inline function _resolve_hom_dimension_path(M::FringeModule,
                                             N::FringeModule)
    choice = _heuristic_hom_internal_choice(M, N)
    return choice === nothing ? :auto_internal : choice
end

@inline function _heuristic_dense_internal_choice(M::FringeModule,
                                                  N::FringeModule)
    choice = _heuristic_hom_internal_choice(M, N)
    choice === :sparse_path && return :dense_idx_internal
    return choice
end

@inline function _heuristic_sparse_internal_choice(M::FringeModule,
                                                   N::FringeModule)
    choice = _heuristic_hom_internal_choice(M, N)
    choice === :dense_path && return nothing
    return choice
end

@inline _hom_density_bin(d::Float64) = clamp(Int(floor(d * 16.0)), 0, 16)
@inline _hom_work_bin(w::Int) = w <= 0 ? 0 : (floor(Int, log2(float(w))) + 1)

@inline function _hom_route_fingerprint(M::FringeModule,
                                        N::FringeModule,
                                        route::Symbol)
    dM = M.phi_density
    dN = N.phi_density
    work = _hom_work_estimate(M, N)
    return UInt64(hash((route,
                        length(M.U), length(M.D), length(N.U), length(N.D),
                        _hom_density_bin(dM), _hom_density_bin(dN),
                        _hom_work_bin(work),
                        M.phi isa SparseMatrixCSC, N.phi isa SparseMatrixCSC,
                        typeof(M.field), typeof(M.phi), typeof(N.phi)), UInt(0)))
end

function _hcat_signed_sparse(T::SparseMatrixCSC{K,Int},
                             S::SparseMatrixCSC{K,Int}) where {K}
    mT, nT = size(T)
    mS, nS = size(S)
    mT == mS || error("_hcat_signed_sparse: row mismatch $mT vs $mS")

    nnzT = nnz(T)
    nnzS = nnz(S)
    colptr = Vector{Int}(undef, nT + nS + 1)
    rowval = Vector{Int}(undef, nnzT + nnzS)
    nzval = Vector{K}(undef, nnzT + nnzS)

    p = 1
    colptr[1] = 1
    @inbounds for j in 1:nT
        firstptr = T.colptr[j]
        lastptr = T.colptr[j + 1] - 1
        if firstptr <= lastptr
            len = lastptr - firstptr + 1
            copyto!(rowval, p, T.rowval, firstptr, len)
            copyto!(nzval, p, T.nzval, firstptr, len)
            p += len
        end
        colptr[j + 1] = p
    end
    @inbounds for j in 1:nS
        firstptr = S.colptr[j]
        lastptr = S.colptr[j + 1] - 1
        if firstptr <= lastptr
            len = lastptr - firstptr + 1
            copyto!(rowval, p, S.rowval, firstptr, len)
            for t in 0:(len - 1)
                nzval[p + t] = -S.nzval[firstptr + t]
            end
            p += len
        end
        colptr[nT + j + 1] = p
    end
    return SparseMatrixCSC(mT, nT + nS, colptr, rowval, nzval)
end

@inline function _rank_hcat_signed(field::AbstractCoeffField,
                                   T::SparseMatrixCSC,
                                   S::SparseMatrixCSC)
    return FieldLinAlg.rank(field, _hcat_signed_sparse(T, S))
end

@inline function _rank_hcat_signed(field::AbstractCoeffField,
                                   T::AbstractMatrix,
                                   S::AbstractMatrix)
    return FieldLinAlg.rank(field, hcat(T, -S))
end

@inline function _rank_hcat_signed_workspace!(field::AbstractCoeffField,
                                              out::AbstractMatrix{K},
                                              T::AbstractMatrix{K},
                                              S::AbstractMatrix{K}) where {K}
    m, nT = size(T)
    mS, nS = size(S)
    m == mS || error("_rank_hcat_signed_workspace!: row mismatch $m vs $mS")
    size(out, 1) == m && size(out, 2) == nT + nS ||
        error("_rank_hcat_signed_workspace!: workspace has wrong size")

    @inbounds for j in 1:nT
        for i in 1:m
            out[i, j] = T[i, j]
        end
    end
    @inbounds for j in 1:nS
        jj = nT + j
        for i in 1:m
            out[i, jj] = -S[i, j]
        end
    end
    return FieldLinAlg.rank(field, out)
end

function _build_sparse_hcat_workspace(T::SparseMatrixCSC{K,Int},
                                      S::SparseMatrixCSC{K,Int}) where {K}
    mT, nT = size(T)
    mS, nS = size(S)
    mT == mS || error("_build_sparse_hcat_workspace: row mismatch $mT vs $mS")

    nnzT = nnz(T)
    nnzS = nnz(S)
    colptr = Vector{Int}(undef, nT + nS + 1)
    rowval = Vector{Int}(undef, nnzT + nnzS)
    nzval = Vector{K}(undef, nnzT + nnzS)

    p = 1
    colptr[1] = 1
    @inbounds for j in 1:nT
        firstptr = T.colptr[j]
        lastptr = T.colptr[j + 1] - 1
        len = lastptr - firstptr + 1
        if len > 0
            copyto!(rowval, p, T.rowval, firstptr, len)
            p += len
        end
        colptr[j + 1] = p
    end
    @inbounds for j in 1:nS
        firstptr = S.colptr[j]
        lastptr = S.colptr[j + 1] - 1
        len = lastptr - firstptr + 1
        if len > 0
            copyto!(rowval, p, S.rowval, firstptr, len)
            p += len
        end
        colptr[nT + j + 1] = p
    end
    return SparseMatrixCSC(mT, nT + nS, colptr, rowval, nzval), nnzT
end

@inline function _rank_hcat_signed_sparse_workspace!(field::AbstractCoeffField,
                                                     out::SparseMatrixCSC{K,Int},
                                                     T::SparseMatrixCSC{K,Int},
                                                     S::SparseMatrixCSC{K,Int},
                                                     nnzT::Int) where {K}
    nnzT == nnz(T) || error("_rank_hcat_signed_sparse_workspace!: nnzT mismatch")
    nnzS = nnz(S)
    nz = out.nzval
    @inbounds begin
        copyto!(nz, 1, T.nzval, 1, nnzT)
        for i in 1:nnzS
            nz[nnzT + i] = -S.nzval[i]
        end
    end
    return FieldLinAlg.rank(field, out)
end

@inline function _rank_hcat_signed_sparse_workspace_with_prefix_rank!(
    field::AbstractCoeffField,
    out::SparseMatrixCSC{K,Int},
    L::SparseMatrixCSC{K,Int},
    R::SparseMatrixCSC{K,Int},
    nnzL::Int,
    left_cols::Int,
) where {K}
    nnzL == nnz(L) || error("_rank_hcat_signed_sparse_workspace_with_prefix_rank!: nnzL mismatch")
    nnzR = nnz(R)
    nz = out.nzval
    @inbounds begin
        copyto!(nz, 1, L.nzval, 1, nnzL)
        for i in 1:nnzR
            nz[nnzL + i] = -R.nzval[i]
        end
    end

    m, n = size(out)
    if m == 0 || n == 0
        return 0, 0
    end
    red = FieldLinAlg._SparseRREF{K}(n)
    rows = FieldLinAlg._sparse_rows(out)
    maxrank = min(m, n)
    rleft = 0
    @inbounds for i in 1:m
        if FieldLinAlg._sparse_rref_push_homogeneous!(red, rows[i])
            red.pivot_cols[end] <= left_cols && (rleft += 1)
            length(red.pivot_cols) == maxrank && break
        end
    end
    return length(red.pivot_cols), rleft
end

@inline function _sparse_col_row_ptr(A::SparseMatrixCSC{K,Int},
                                     row::Int,
                                     col::Int) where {K}
    lo = A.colptr[col]
    hi = A.colptr[col + 1] - 1
    while lo <= hi
        mid = (lo + hi) >>> 1
        rv = A.rowval[mid]
        if rv < row
            lo = mid + 1
        elseif rv > row
            hi = mid - 1
        else
            return mid
        end
    end
    return 0
end

@inline _hom_sparse_standalone_rank_backend(::AbstractCoeffField, ::SparseMatrixCSC) = :auto
# The sparse Hom path repeatedly ranks very low-density QQ matrices; forcing
# the Julia sparse engine avoids expensive QQ->Nemo conversion on this kernel.
@inline _hom_sparse_standalone_rank_backend(::QQField, ::SparseMatrixCSC) = :julia_sparse

@inline function _hom_intersection_dim(field::AbstractCoeffField,
                                       T::AbstractMatrix,
                                       S::AbstractMatrix,
                                       rT::Int,
                                       rS::Int)
    (rT == 0 || rS == 0) && return 0
    BT = FieldLinAlg.colspace(field, T)
    BS = FieldLinAlg.colspace(field, S)
    (size(BT, 2) == 0 || size(BS, 2) == 0) && return 0
    rUnion = FieldLinAlg.rank(field, hcat(BT, BS))
    return rT + rS - rUnion
end

"Undirected adjacency of the Hasse cover graph."
function _cover_undirected_adjacency(P::AbstractPoset)
    return _get_cover_cache(P).undir::_PackedAdjacency
end

"Connected components of a subset mask in the undirected Hasse cover graph."
function _component_data(adj::_PackedAdjacency, mask::BitVector)
    n = length(mask)
    comp = fill(0, n)
    reps = Int[]
    cid = 0
    queue = Int[]
    for v in 1:n
        if mask[v] && comp[v] == 0
            cid += 1
            push!(reps, v)
            empty!(queue)
            push!(queue, v)
            comp[v] = cid
            head = 1
            while head <= length(queue)
                x = queue[head]
                head += 1
                lo, hi = _adj_bounds(adj, x)
                @inbounds for p in lo:hi
                    y = adj.idx[p]
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

@inline function _ensure_hom_cache!(M::FringeModule{K}) where {K}
    hc = M.hom_cache[]
    if hc.adj === nothing
        hc.adj = _cover_undirected_adjacency(M.P)
    end
    return hc::_FringeHomCache{K}
end

@inline function _pair_cache_ids(N::FringeModule)
    return UInt64(objectid(N)), UInt64(objectid(N.phi))
end

function _lookup_pair_cache(hc::_FringeHomCache{K},
                            N::FringeModule) where {K}
    pid, pphi = _pair_cache_ids(N)
    entries = hc.pair_cache
    @inbounds for i in eachindex(entries)
        entry = entries[i]
        if entry.partner_id == pid && entry.partner_phi_id == pphi
            if i != 1
                entries[i], entries[1] = entries[1], entries[i]
            end
            return entries[1]
        end
    end
    return nothing
end

function _ensure_pair_cache!(hc::_FringeHomCache{K},
                             N::FringeModule) where {K}
    entry = _lookup_pair_cache(hc, N)
    entry !== nothing && return entry::_FringePairCache{K}

    pid, pphi = _pair_cache_ids(N)
    entry = _FringePairCache{K}(pid, pphi, nothing, nothing, nothing, nothing)
    pushfirst!(hc.pair_cache, entry)
    max_entries = HOM_PAIR_CACHE_MAX_ENTRIES[]
    if length(hc.pair_cache) > max_entries
        resize!(hc.pair_cache, max_entries)
    end
    return entry
end

@inline function _route_fingerprint_choice_get(hc::_FringeHomCache, fkey::UInt64)
    entries = hc.route_fingerprint_choice
    @inbounds for i in eachindex(entries)
        entry = entries[i]
        entry.fingerprint == fkey && return entry.choice
    end
    return nothing
end

function _route_fingerprint_choice_set!(hc::_FringeHomCache,
                                        fkey::UInt64,
                                        choice::Symbol)
    entries = hc.route_fingerprint_choice
    @inbounds for i in eachindex(entries)
        if entries[i].fingerprint == fkey
            entries[i] = _FringeRouteChoiceEntry(fkey, choice)
            return choice
        end
    end
    push!(entries, _FringeRouteChoiceEntry(fkey, choice))
    return choice
end

@inline function _poset_cache_or_nothing(P)
    if hasproperty(P, :cache)
        pc = getproperty(P, :cache)
        return pc isa PosetCache ? pc : nothing
    end
    return nothing
end

@inline function _hom_route_choice_get(P, fkey::UInt64)
    pc = _poset_cache_or_nothing(P)
    pc === nothing && return nothing
    Base.lock(pc.lock)
    try
        return get(pc.hom_route_choice, fkey, nothing)
    finally
        Base.unlock(pc.lock)
    end
end

@inline function _hom_route_choice_set!(P, fkey::UInt64, choice::Symbol)
    pc = _poset_cache_or_nothing(P)
    pc === nothing && return nothing
    Base.lock(pc.lock)
    try
        pc.hom_route_choice[fkey] = choice
    finally
        Base.unlock(pc.lock)
    end
    return nothing
end

function _build_component_decomp(adj::_PackedAdjacency,
                                 sets::AbstractVector)
    nsets = length(sets)
    comp_id = Vector{Vector{Int}}(undef, nsets)
    comp_masks = Vector{Vector{BitVector}}(undef, nsets)
    comp_n = Vector{Int}(undef, nsets)
    @inbounds for i in 1:nsets
        cid, ncomp, masks, _ = _component_data(adj, sets[i].mask)
        comp_id[i] = cid
        comp_masks[i] = masks
        comp_n[i] = ncomp
    end
    return _FringeComponentDecomp(comp_id, comp_masks, comp_n)
end

@inline function _ensure_upset_component_decomp!(M::FringeModule)
    hc = _ensure_hom_cache!(M)
    if hc.upset === nothing
        hc.upset = _build_component_decomp(hc.adj::_PackedAdjacency, M.U)
    end
    return hc.upset::_FringeComponentDecomp
end

@inline function _ensure_downset_component_decomp!(M::FringeModule)
    hc = _ensure_hom_cache!(M)
    if hc.downset === nothing
        hc.downset = _build_component_decomp(hc.adj::_PackedAdjacency, M.D)
    end
    return hc.downset::_FringeComponentDecomp
end

function _component_subset_targets(comp_masks::Vector{Vector{BitVector}},
                                   target_masks::Vector{BitVector})
    out = Vector{Vector{Vector{Int}}}(undef, length(comp_masks))
    @inbounds for i in eachindex(comp_masks)
        cmasks = comp_masks[i]
        targets_i = Vector{Vector{Int}}(undef, length(cmasks))
        for c in eachindex(cmasks)
            mask = cmasks[c]
            js = Int[]
            for j in eachindex(target_masks)
                if is_subset(mask, target_masks[j])
                    push!(js, j)
                end
            end
            targets_i[c] = js
        end
        out[i] = targets_i
    end
    return out
end

# Count connected components in mask_a intersect mask_b without materializing the intersection mask.
function _component_reps_intersection!(
    reps::Vector{Int},
    adj::_PackedAdjacency,
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
                lo, hi = _adj_bounds(adj, x)
                @inbounds for p in lo:hi
                    y = adj.idx[p]
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

@inline function _pack_row_buckets(rows_by_comp::Vector{Vector{Int}})
    ncomp = length(rows_by_comp)
    ptr = Vector{Int}(undef, ncomp + 1)
    ptr[1] = 1
    total = 0
    @inbounds for c in 1:ncomp
        total += length(rows_by_comp[c])
        ptr[c + 1] = total + 1
    end
    rows = Vector{Int}(undef, total)
    p = 1
    @inbounds for c in 1:ncomp
        rc = rows_by_comp[c]
        len = length(rc)
        if len > 0
            copyto!(rows, p, rc, 1, len)
            p += len
        end
    end
    return ptr, rows
end

@inline function _wpair_u_bounds(w::_WPairData, cU::Int)
    return w.u_ptr[cU], w.u_ptr[cU + 1] - 1
end

@inline function _wpair_d_bounds(w::_WPairData, cD::Int)
    return w.d_ptr[cD], w.d_ptr[cD + 1] - 1
end

function _build_wpair_layout(adj::_PackedAdjacency,
                             Usets::AbstractVector,
                             Dsets::AbstractVector,
                             Ucomp_id::Vector{Vector{Int}},
                             Ucomp_n::Vector{Int},
                             Dcomp_id::Vector{Vector{Int}},
                             Dcomp_n::Vector{Int},
                             nverts::Int)
    nU = length(Usets)
    nD = length(Dsets)
    w_index = zeros(Int, nU, nD)
    w_data = _WPairData[]
    W_dim = 0
    marks = zeros(Int, nverts)
    queue = Int[]
    reps_int = Int[]
    mark = 1

    for iU in 1:nU
        for jD in 1:nD
            mark += 1
            if mark == typemax(Int)
                fill!(marks, 0)
                mark = 1
            end
            ncomp_int = _component_reps_intersection!(
                reps_int, adj, Usets[iU].mask, Dsets[jD].mask, marks, mark, queue
            )
            if ncomp_int > 0
                base = W_dim
                W_dim += ncomp_int

                rows_by_u = [Int[] for _ in 1:Ucomp_n[iU]]
                rows_by_d = [Int[] for _ in 1:Dcomp_n[jD]]
                @inbounds for c in 1:ncomp_int
                    row = base + c
                    v = reps_int[c]
                    cu = Ucomp_id[iU][v]
                    cd = Dcomp_id[jD][v]
                    push!(rows_by_u[cu], row)
                    push!(rows_by_d[cd], row)
                end
                u_ptr, u_rows = _pack_row_buckets(rows_by_u)
                d_ptr, d_rows = _pack_row_buckets(rows_by_d)
                push!(w_data, _WPairData(u_ptr, u_rows, d_ptr, d_rows))
                w_index[iU, jD] = length(w_data)
            end
        end
    end

    return w_index, w_data, W_dim
end

function _build_dense_idx_hom_plan(M::FringeModule{K},
                                   N::FringeModule{K}) where {K}
    @assert M.P === N.P "Posets must match"
    adj = (_ensure_hom_cache!(M).adj)::_PackedAdjacency

    nUM = length(M.U); nDM = length(M.D)
    nUN = length(N.U); nDN = length(N.D)

    Udec_M = _ensure_upset_component_decomp!(M)
    Ddec_N = _ensure_downset_component_decomp!(N)
    Ucomp_id_M = Udec_M.comp_id
    Ucomp_masks_M = Udec_M.comp_masks
    Ucomp_n_M = Udec_M.comp_n
    Dcomp_id_N = Ddec_N.comp_id
    Dcomp_masks_N = Ddec_N.comp_masks
    Dcomp_n_N = Ddec_N.comp_n

    w_index, w_data, W_dim = _build_wpair_layout(
        adj, M.U, N.D, Ucomp_id_M, Ucomp_n_M, Dcomp_id_N, Dcomp_n_N, nvertices(M.P)
    )

    U_targets = _component_subset_targets(Ucomp_masks_M, [N.U[j].mask for j in 1:nUN])
    D_targets = _component_subset_targets(Dcomp_masks_N, [M.D[s].mask for s in 1:nDM])

    V1 = Tuple{Int,Int,Int}[]
    for iM in 1:nUM
        targets_i = U_targets[iM]
        for cU in 1:Ucomp_n_M[iM]
            for jN in targets_i[cU]
                push!(V1, (iM, jN, cU))
            end
        end
    end
    V1_dim = length(V1)

    V2 = Tuple{Int,Int,Int}[]
    for tN in 1:nDN
        targets_t = D_targets[tN]
        for cD in 1:Dcomp_n_N[tN]
            for sM in targets_t[cD]
                push!(V2, (sM, tN, cD))
            end
        end
    end
    V2_dim = length(V2)

    t_rows = Int[]
    t_cols = Int[]
    t_tN = Int[]
    t_jN = Int[]
    sizehint!(t_rows, max(16, W_dim * 4))
    sizehint!(t_cols, max(16, W_dim * 4))
    sizehint!(t_tN, max(16, W_dim * 4))
    sizehint!(t_jN, max(16, W_dim * 4))

    @inbounds for (col, (iM, jN, cU)) in enumerate(V1)
        for tN in 1:nDN
            pid = w_index[iM, tN]
            pid == 0 && continue
            w = w_data[pid]
            lo, hi = _wpair_u_bounds(w, cU)
            lo > hi && continue
            for k in lo:hi
                push!(t_rows, w.u_rows[k])
                push!(t_cols, col)
                push!(t_tN, tN)
                push!(t_jN, jN)
            end
        end
    end

    s_rows = Int[]
    s_cols = Int[]
    s_sM = Int[]
    s_iM = Int[]
    sizehint!(s_rows, max(16, W_dim * 4))
    sizehint!(s_cols, max(16, W_dim * 4))
    sizehint!(s_sM, max(16, W_dim * 4))
    sizehint!(s_iM, max(16, W_dim * 4))

    @inbounds for (col, (sM, tN, cD)) in enumerate(V2)
        for iM in 1:nUM
            pid = w_index[iM, tN]
            pid == 0 && continue
            w = w_data[pid]
            lo, hi = _wpair_d_bounds(w, cD)
            lo > hi && continue
            for k in lo:hi
                push!(s_rows, w.d_rows[k])
                push!(s_cols, col)
                push!(s_sM, sM)
                push!(s_iM, iM)
            end
        end
    end

    Tbuf = zeros(K, W_dim, V1_dim)
    Sbuf = zeros(K, W_dim, V2_dim)
    bigbuf = Matrix{K}(undef, W_dim, V1_dim + V2_dim)
    return _FringeDenseIdxPlan{K}(W_dim, V1_dim, V2_dim,
                                  t_rows, t_cols, t_tN, t_jN,
                                  s_rows, s_cols, s_sM, s_iM,
                                  Tbuf, Sbuf, bigbuf)
end

@inline function _ensure_dense_idx_hom_plan!(M::FringeModule{K},
                                             N::FringeModule{K}) where {K}
    hc = _ensure_hom_cache!(M)
    entry = _ensure_pair_cache!(hc, N)
    plan = entry.dense_idx_plan
    if plan === nothing
        plan = _build_dense_idx_hom_plan(M, N)
        entry.dense_idx_plan = plan
    end
    return plan::_FringeDenseIdxPlan{K}
end

function _build_dense_path_hom_plan(M::FringeModule{K},
                                    N::FringeModule{K}) where {K}
    @assert M.P === N.P "Posets must match"
    adj = (_ensure_hom_cache!(M).adj)::_PackedAdjacency

    nUM = length(M.U)
    nDM = length(M.D)
    nUN = length(N.U)
    nDN = length(N.D)

    Udec_M = _ensure_upset_component_decomp!(M)
    Ddec_N = _ensure_downset_component_decomp!(N)
    Ucomp_id_M = Udec_M.comp_id
    Ucomp_masks_M = Udec_M.comp_masks
    Ucomp_n_M = Udec_M.comp_n
    Dcomp_id_N = Ddec_N.comp_id
    Dcomp_masks_N = Ddec_N.comp_masks
    Dcomp_n_N = Ddec_N.comp_n

    w_index, w_data, W_dim = _build_wpair_layout(
        adj, M.U, N.D, Ucomp_id_M, Ucomp_n_M, Dcomp_id_N, Dcomp_n_N, nvertices(M.P)
    )

    U_targets = _component_subset_targets(Ucomp_masks_M, [N.U[j].mask for j in 1:nUN])
    D_targets = _component_subset_targets(Dcomp_masks_N, [M.D[s].mask for s in 1:nDM])

    V1 = NTuple{3,Int}[]
    for iM in 1:nUM
        targets_i = U_targets[iM]
        for cU in 1:Ucomp_n_M[iM]
            for jN in targets_i[cU]
                push!(V1, (iM, jN, cU))
            end
        end
    end

    V2 = NTuple{3,Int}[]
    for tN in 1:nDN
        targets_t = D_targets[tN]
        for cD in 1:Dcomp_n_N[tN]
            for sM in targets_t[cD]
                push!(V2, (sM, tN, cD))
            end
        end
    end

    V1_dim = length(V1)
    V2_dim = length(V2)
    Tbuf = zeros(K, W_dim, V1_dim)
    Sbuf = zeros(K, W_dim, V2_dim)
    bigbuf = Matrix{K}(undef, W_dim, V1_dim + V2_dim)
    return _FringeDensePathPlan{K}(W_dim, V1, V2, w_index, w_data,
                                   nUM, nDN, Tbuf, Sbuf, bigbuf)
end

@inline function _ensure_dense_path_hom_plan!(M::FringeModule{K},
                                              N::FringeModule{K}) where {K}
    hc = _ensure_hom_cache!(M)
    entry = _ensure_pair_cache!(hc, N)
    plan = entry.dense_path_plan
    if plan === nothing
        plan = _build_dense_path_hom_plan(M, N)
        entry.dense_path_plan = plan
    end
    return plan::_FringeDensePathPlan{K}
end

function _hom_dimension_dense_idx_path(M::FringeModule{K},
                                         N::FringeModule{K}) where {K}
    plan = _ensure_dense_idx_hom_plan!(M, N)
    T = plan.Tbuf
    S = plan.Sbuf
    z = zero(K)
    fill!(T, z)
    fill!(S, z)

    Nphi = N.phi
    Mphi = M.phi
    @inbounds for i in eachindex(plan.t_rows)
        v = Nphi[plan.t_tN[i], plan.t_jN[i]]
        v == z && continue
        T[plan.t_rows[i], plan.t_cols[i]] += v
    end
    @inbounds for i in eachindex(plan.s_rows)
        v = Mphi[plan.s_sM[i], plan.s_iM[i]]
        v == z && continue
        S[plan.s_rows[i], plan.s_cols[i]] += v
    end

    rT = FieldLinAlg.rank(M.field, T)
    rS = FieldLinAlg.rank(M.field, S)
    rBig = _rank_hcat_signed_workspace!(M.field, plan.bigbuf, T, S)
    dimKer_big = (plan.V1_dim + plan.V2_dim) - rBig
    dimKer_T = plan.V1_dim - rT
    dimKer_S = plan.V2_dim - rS
    return dimKer_big - (dimKer_T + dimKer_S)
end

function _hom_dimension_dense_path(M::FringeModule{K}, N::FringeModule{K}) where {K}
    plan = _ensure_dense_path_hom_plan!(M, N)
    T = plan.Tbuf
    S = plan.Sbuf
    z = zero(K)
    fill!(T, z)
    fill!(S, z)

    Nphi = N.phi
    Mphi = M.phi
    V1 = plan.V1
    V2 = plan.V2
    w_index = plan.w_index
    w_data = plan.w_data
    nDN = plan.nDN
    nUM = plan.nUM

    @inbounds for col in eachindex(V1)
        (iM, jN, cU) = V1[col]
        for tN in 1:nDN
            val = Nphi[tN, jN]
            val == z && continue
            pid = w_index[iM, tN]
            pid == 0 && continue
            w = w_data[pid]
            lo, hi = _wpair_u_bounds(w, cU)
            for k in lo:hi
                T[w.u_rows[k], col] += val
            end
        end
    end

    @inbounds for col in eachindex(V2)
        (sM, tN, cD) = V2[col]
        for iM in 1:nUM
            val = Mphi[sM, iM]
            val == z && continue
            pid = w_index[iM, tN]
            pid == 0 && continue
            w = w_data[pid]
            lo, hi = _wpair_d_bounds(w, cD)
            for k in lo:hi
                S[w.d_rows[k], col] += val
            end
        end
    end

    rT = FieldLinAlg.rank(M.field, T)
    rS = FieldLinAlg.rank(M.field, S)
    rBig = _rank_hcat_signed_workspace!(M.field, plan.bigbuf, T, S)
    V1_dim = length(V1)
    V2_dim = length(V2)
    dimKer_big = (V1_dim + V2_dim) - rBig
    dimKer_T = V1_dim - rT
    dimKer_S = V2_dim - rS
    return dimKer_big - (dimKer_T + dimKer_S)
end

function _build_sparse_hom_plan(M::FringeModule{K},
                                N::FringeModule{K}) where {K}
    @assert M.P === N.P "Posets must match"
    adj = (_ensure_hom_cache!(M).adj)::_PackedAdjacency

    nUM = length(M.U); nDM = length(M.D)
    nUN = length(N.U); nDN = length(N.D)

    Udec_M = _ensure_upset_component_decomp!(M)
    Ddec_N = _ensure_downset_component_decomp!(N)
    Ucomp_id_M = Udec_M.comp_id
    Ucomp_masks_M = Udec_M.comp_masks
    Ucomp_n_M = Udec_M.comp_n
    Dcomp_id_N = Ddec_N.comp_id
    Dcomp_masks_N = Ddec_N.comp_masks
    Dcomp_n_N = Ddec_N.comp_n

    w_index, w_data, W_dim = _build_wpair_layout(
        adj, M.U, N.D, Ucomp_id_M, Ucomp_n_M, Dcomp_id_N, Dcomp_n_N, nvertices(M.P)
    )

    U_targets = _component_subset_targets(Ucomp_masks_M, [N.U[j].mask for j in 1:nUN])
    D_targets = _component_subset_targets(Dcomp_masks_N, [M.D[s].mask for s in 1:nDM])

    V1 = NTuple{3,Int}[]
    for iM in 1:nUM
        targets_i = U_targets[iM]
        for cU in 1:Ucomp_n_M[iM]
            for jN in targets_i[cU]
                push!(V1, (iM, jN, cU))
            end
        end
    end

    V2 = NTuple{3,Int}[]
    for tN in 1:nDN
        targets_t = D_targets[tN]
        for cD in 1:Dcomp_n_N[tN]
            for sM in targets_t[cD]
                push!(V2, (sM, tN, cD))
            end
        end
    end

    z = zero(K)
    Nphi = N.phi
    t_colptr = Vector{Int}(undef, length(V1) + 1)
    t_rowval = Int[]
    t_tN = Int[]
    t_jN = Int[]
    t_colptr[1] = 1
    @inbounds for col in eachindex(V1)
        (iM, jN, cU) = V1[col]
        for tN in 1:nDN
            Nphi[tN, jN] == z && continue
            pid = w_index[iM, tN]
            pid == 0 && continue
            w = w_data[pid]
            lo, hi = _wpair_u_bounds(w, cU)
            for k in lo:hi
                push!(t_rowval, w.u_rows[k])
                push!(t_tN, tN)
                push!(t_jN, jN)
            end
        end
        t_colptr[col + 1] = length(t_rowval) + 1
    end
    t_nzval = Vector{K}(undef, length(t_rowval))
    @inbounds for i in eachindex(t_nzval)
        t_nzval[i] = Nphi[t_tN[i], t_jN[i]]
    end
    t_nzptr = if Nphi isa SparseMatrixCSC{K,Int}
        ptrs = Vector{Int}(undef, length(t_tN))
        @inbounds for i in eachindex(t_tN)
            p = _sparse_col_row_ptr(Nphi, t_tN[i], t_jN[i])
            p == 0 && error("_build_sparse_hom_plan: internal sparse pointer miss for T entry.")
            ptrs[i] = p
        end
        ptrs
    else
        nothing
    end
    t_nzptr_max = t_nzptr === nothing ? 0 : maximum(t_nzptr)
    T = SparseMatrixCSC(W_dim, length(V1), t_colptr, t_rowval, t_nzval)

    Mphi = M.phi
    s_colptr = Vector{Int}(undef, length(V2) + 1)
    s_rowval = Int[]
    s_sM = Int[]
    s_iM = Int[]
    s_colptr[1] = 1
    @inbounds for col in eachindex(V2)
        (sM, tN, cD) = V2[col]
        for iM in 1:nUM
            Mphi[sM, iM] == z && continue
            pid = w_index[iM, tN]
            pid == 0 && continue
            w = w_data[pid]
            lo, hi = _wpair_d_bounds(w, cD)
            for k in lo:hi
                push!(s_rowval, w.d_rows[k])
                push!(s_sM, sM)
                push!(s_iM, iM)
            end
        end
        s_colptr[col + 1] = length(s_rowval) + 1
    end
    s_nzval = Vector{K}(undef, length(s_rowval))
    @inbounds for i in eachindex(s_nzval)
        s_nzval[i] = Mphi[s_sM[i], s_iM[i]]
    end
    s_nzptr = if Mphi isa SparseMatrixCSC{K,Int}
        ptrs = Vector{Int}(undef, length(s_sM))
        @inbounds for i in eachindex(s_sM)
            p = _sparse_col_row_ptr(Mphi, s_sM[i], s_iM[i])
            p == 0 && error("_build_sparse_hom_plan: internal sparse pointer miss for S entry.")
            ptrs[i] = p
        end
        ptrs
    else
        nothing
    end
    s_nzptr_max = s_nzptr === nothing ? 0 : maximum(s_nzptr)
    S = SparseMatrixCSC(W_dim, length(V2), s_colptr, s_rowval, s_nzval)

    hcat_buf, nnzT = _build_sparse_hcat_workspace(T, S)
    hcat_buf_rev, nnzS = _build_sparse_hcat_workspace(S, T)
    return _FringeSparsePlan{K}(W_dim, V1, V2, w_index, w_data, nUM, nDN,
                                T, S, t_tN, t_jN, s_sM, s_iM,
                                t_nzptr, s_nzptr,
                                t_nzptr_max, s_nzptr_max,
                                hcat_buf, hcat_buf_rev,
                                nnzT, nnzS)
end

@inline function _ensure_sparse_hom_plan!(M::FringeModule{K},
                                          N::FringeModule{K}) where {K}
    hc = _ensure_hom_cache!(M)
    entry = _ensure_pair_cache!(hc, N)
    plan = entry.sparse_plan
    if plan === nothing
        plan = _build_sparse_hom_plan(M, N)
        entry.sparse_plan = plan
    end
    return plan::_FringeSparsePlan{K}
end

function _hom_dimension_sparse_path(M::FringeModule{K}, N::FringeModule{K}) where {K}
    plan = _ensure_sparse_hom_plan!(M, N)
    T = plan.T
    S = plan.S
    z = zero(K)
    Nphi = N.phi
    Mphi = M.phi

    t_nzptr = plan.t_nzptr
    if t_nzptr !== nothing && Nphi isa SparseMatrixCSC{K,Int} &&
       plan.t_nzptr_max <= length(Nphi.nzval)
        @inbounds for i in eachindex(T.nzval)
            v = Nphi.nzval[t_nzptr[i]]
            T.nzval[i] = v == z ? z : v
        end
    else
        @inbounds for i in eachindex(T.nzval)
            v = Nphi[plan.t_tN[i], plan.t_jN[i]]
            T.nzval[i] = v == z ? z : v
        end
    end

    s_nzptr = plan.s_nzptr
    if s_nzptr !== nothing && Mphi isa SparseMatrixCSC{K,Int} &&
       plan.s_nzptr_max <= length(Mphi.nzval)
        @inbounds for i in eachindex(S.nzval)
            v = Mphi.nzval[s_nzptr[i]]
            S.nzval[i] = v == z ? z : v
        end
    else
        @inbounds for i in eachindex(S.nzval)
            v = Mphi[plan.s_sM[i], plan.s_iM[i]]
            S.nzval[i] = v == z ? z : v
        end
    end

    nT = size(T, 2)
    nS = size(S, 2)
    if nS <= nT
        rS = FieldLinAlg.rank(M.field, S; backend=_hom_sparse_standalone_rank_backend(M.field, S))
        rUnion, rT = _rank_hcat_signed_sparse_workspace_with_prefix_rank!(
            M.field, plan.hcat_buf, T, S, plan.nnzT, nT
        )
        return rT + rS - rUnion
    else
        rT = FieldLinAlg.rank(M.field, T; backend=_hom_sparse_standalone_rank_backend(M.field, T))
        rUnion, rS = _rank_hcat_signed_sparse_workspace_with_prefix_rank!(
            M.field, plan.hcat_buf_rev, S, T, plan.nnzS, nS
        )
        return rT + rS - rUnion
    end
end

@inline function _hom_dimension_with_path(M::FringeModule{K},
                                          N::FringeModule{K},
                                          path::Symbol) where {K}
    path === :sparse_path && return _hom_dimension_sparse_path(M, N)
    path === :dense_path && return _hom_dimension_dense_path(M, N)
    path === :dense_idx_internal && return _hom_dimension_dense_idx_path(M, N)
    error("_hom_dimension_with_path: unknown path $(repr(path)).")
end

@inline function _store_hom_route_choice!(hc::_FringeHomCache,
                                          P,
                                          entry,
                                          fkey::UInt64,
                                          choice::Symbol)
    entry.route_choice = choice
    _route_fingerprint_choice_set!(hc, fkey, choice)
    _hom_route_choice_set!(P, fkey, choice)
    return choice
end

function _select_hom_internal_path_timed!(M::FringeModule{K},
                                          N::FringeModule{K},
                                          hc::_FringeHomCache{K},
                                          entry::_FringePairCache{K},
                                          fkey::UInt64) where {K}
    hc.route_timing_fallbacks += 1

    t0 = time_ns()
    v_sparse = _hom_dimension_sparse_path(M, N)
    t_sparse = time_ns() - t0

    t0 = time_ns()
    v_dense = _hom_dimension_dense_path(M, N)
    t_dense = time_ns() - t0

    t0 = time_ns()
    v_denseidx = _hom_dimension_dense_idx_path(M, N)
    t_denseidx = time_ns() - t0

    (v_sparse == v_dense && v_dense == v_denseidx) ||
        error("hom_dimension path mismatch during one-shot internal selection.")

    choice = :dense_idx_internal
    best = t_denseidx
    if t_sparse < best
        choice = :sparse_path
        best = t_sparse
    end
    if t_dense < best
        choice = :dense_path
    end
    return _store_hom_route_choice!(hc, M.P, entry, fkey, choice)
end

function _select_hom_internal_path!(M::FringeModule{K},
                                    N::FringeModule{K}) where {K}
    hc = _ensure_hom_cache!(M)
    entry = _ensure_pair_cache!(hc, N)
    choice = entry.route_choice
    choice !== nothing && return choice::Symbol

    fkey = _hom_route_fingerprint(M, N, :internal_choice)
    choice = _hom_route_choice_get(M.P, fkey)
    if choice !== nothing
        entry.route_choice = choice
        _route_fingerprint_choice_set!(hc, fkey, choice)
        return choice::Symbol
    end
    choice = _route_fingerprint_choice_get(hc, fkey)
    if choice !== nothing
        entry.route_choice = choice
        return choice::Symbol
    end

    choice = _heuristic_hom_internal_choice(M, N)
    if choice !== nothing
        return _store_hom_route_choice!(hc, M.P, entry, fkey, choice)
    end

    return _select_hom_internal_path_timed!(M, N, hc, entry, fkey)
end

function _clear_hom_route_choice!(M::FringeModule{K}) where {K}
    hc = _ensure_hom_cache!(M)
    for entry in hc.pair_cache
        entry.route_choice = nothing
    end
    empty!(hc.route_fingerprint_choice)
    hc.route_timing_fallbacks = 0
    pc = _poset_cache_or_nothing(M.P)
    if pc !== nothing
        Base.lock(pc.lock)
        try
            empty!(pc.hom_route_choice)
        finally
            Base.unlock(pc.lock)
        end
    end
    return nothing
end

"Dimension of Hom(M,N) over a field K using the production auto route selector."
function hom_dimension(M::FringeModule{K}, N::FringeModule{K}) where {K}
    @assert M.P === N.P "Posets must match"
    path = _select_hom_internal_path!(M, N)
    return _hom_dimension_with_path(M, N, path)
end


# ---------- utility: dense\tosparse over K ----------

"""
    _dense_to_sparse_K(A)

Convert a dense matrix `A` to a sparse matrix with the same element type.

For an explicit target coefficient type `K`, call `_dense_to_sparse_K(A, K)`.
"""
function _dense_to_sparse_K(A::AbstractMatrix{T}, ::Type{K}) where {T,K}
    m,n = size(A)
    S = spzeros(K, m, n)
    for j in 1:n, i in 1:m
        v = K(A[i,j])
        if v != zero(K); S[i,j] = v; end
    end
    S
end

_dense_to_sparse_K(A::AbstractMatrix{T}) where {T} = _dense_to_sparse_K(A, T)

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
using ..CoreModules: AbstractCoeffField, coeff_type

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
    U0::Vector{<:Upset}
    U1::Vector{<:Upset}
    delta::SparseMatrixCSC{K,Int}
    H::Union{Nothing, FringeModule{K}}

    function UpsetPresentation{K}(P, U0, U1, delta, H) where {K}
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
    D0::Vector{<:Downset}
    D1::Vector{<:Downset}
    rho::SparseMatrixCSC{K,Int}
    H::Union{Nothing, FringeModule{K}}

    function DownsetCopresentation{K}(P, D0, D1, rho, H) where {K}
        rhoK = SparseMatrixCSC{K,Int}(rho)
        return new{K}(P, D0, D1, rhoK, H)
    end
end

function UpsetPresentation(
    P::FinitePoset,
    U0::Vector{<:Upset},
    U1::Vector{<:Upset},
    delta::SparseMatrixCSC,
    H;
    field::AbstractCoeffField,
)
    K = coeff_type(field)
    deltaK = SparseMatrixCSC{K,Int}(delta)
    return IndicatorTypes.UpsetPresentation{K}(P, U0, U1, deltaK, H)
end

function UpsetPresentation(
    P::FinitePoset,
    U0::Vector{<:Upset},
    U1::Vector{<:Upset},
    delta::SparseMatrixCSC{K,Int},
    H,
) where {K}
    return IndicatorTypes.UpsetPresentation{K}(P, U0, U1, delta, H)
end

function DownsetCopresentation(
    P::FinitePoset,
    D0::Vector{<:Downset},
    D1::Vector{<:Downset},
    rho::SparseMatrixCSC,
    H;
    field::AbstractCoeffField,
)
    K = coeff_type(field)
    rhoK = SparseMatrixCSC{K,Int}(rho)
    return IndicatorTypes.DownsetCopresentation{K}(P, D0, D1, rhoK, H)
end

function DownsetCopresentation(
    P::FinitePoset,
    D0::Vector{<:Downset},
    D1::Vector{<:Downset},
    rho::SparseMatrixCSC{K,Int},
    H,
) where {K}
    return IndicatorTypes.DownsetCopresentation{K}(P, D0, D1, rho, H)
end

end # module IndicatorTypes


module Encoding
# =============================================================================
# Finite encodings ("uptight posets") from a finite family of constant upsets.
#
# References: Miller section 4 (Defs. 4.12 - 4.18 and Thm. 4.19 - 4.22).
# =============================================================================

using SparseArrays
import ..FiniteFringe
import ..FiniteFringe: AbstractPoset, FringeModule, RegionsPoset, nvertices, leq

# This submodule defines methods on EncodingCore.AbstractPLikeEncodingMap and
# calls EncodingCore.locate/dimension/etc., so we must import the sibling module
# binding into scope (not just individual names).
import ..EncodingCore

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
struct PostcomposedEncodingMap{PI<:EncodingCore.AbstractPLikeEncodingMap} <: EncodingCore.AbstractPLikeEncodingMap
    pi0::PI
    pi_of_q::Vector{Int}              # pi : Q -> P encoded as forward table
    Pn::Int                           # number of regions in P
    reps_cache::Base.RefValue{Union{Nothing,Vector{Tuple}}}    # lazy cache for representatives(::...)
end

@inline PostcomposedEncodingMap(pi0::EncodingCore.AbstractPLikeEncodingMap, pi::EncodingMap) =
    PostcomposedEncodingMap(pi0, pi.pi_of_q, nvertices(pi.P), Ref{Union{Nothing,Vector{Tuple}}}(nothing))

@inline function EncodingCore.locate(pi::PostcomposedEncodingMap, x::AbstractVector)
    q = EncodingCore.locate(pi.pi0, x)
    q == 0 && return 0
    @inbounds return pi.pi_of_q[q]
end

@inline function EncodingCore.locate(pi::PostcomposedEncodingMap, x::AbstractVector; kwargs...)
    q = EncodingCore.locate(pi.pi0, x; kwargs...)
    q == 0 && return 0
    @inbounds return pi.pi_of_q[q]
end

# Disambiguation:
# There is a generic fallback
#   locate(::AbstractPLikeEncodingMap, ::NTuple{N,<:Real})
# in EncodingCore. Since PostcomposedEncodingMap <: AbstractPLikeEncodingMap,
# we must ensure our method is MORE specific on the x-type too, otherwise
# calls like locate(pi, (q,)) become ambiguous.
@inline function EncodingCore.locate(
    pi::PostcomposedEncodingMap,
    x::NTuple{N,T},
) where {N, T<:Real}
    q = EncodingCore.locate(pi.pi0, x)
    q == 0 && return 0
    @inbounds return pi.pi_of_q[q]
end

@inline function EncodingCore.locate(
    pi::PostcomposedEncodingMap,
    x::NTuple{N,T};
    kwargs...,
) where {N, T<:Real}
    q = EncodingCore.locate(pi.pi0, x; kwargs...)
    q == 0 && return 0
    @inbounds return pi.pi_of_q[q]
end


@inline EncodingCore.dimension(pi::PostcomposedEncodingMap) = EncodingCore.dimension(pi.pi0)
@inline EncodingCore.axes_from_encoding(pi::PostcomposedEncodingMap) = EncodingCore.axes_from_encoding(pi.pi0)

function EncodingCore.representatives(pi::PostcomposedEncodingMap)
    cached = pi.reps_cache[]
    cached !== nothing && return cached

    reps0 = EncodingCore.representatives(pi.pi0)
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
            vec = get!(sigs, key) do
                Int[]
            end
            push!(vec, q)
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
            vec = get!(sigs, key) do
                Int[]
            end
            push!(vec, q)
        end
        return collect(values(sigs))
    else
        # Large-Y fast path:
        # use packed UInt64 signatures with a hash-chain table to avoid
        # constructing Tuple{Vararg{Bool}} keys per vertex.
        n = nvertices(Q)
        nchunks = cld(m, 64)
        scratch = zeros(UInt64, nchunks)

        heads = Dict{UInt, Int}()                # hash => head class index
        next_samehash = Int[]                    # linked list next pointers
        signatures = Vector{Vector{UInt64}}()    # one packed signature per class
        classes = Vector{Vector{Int}}()          # class members

        sizehint!(next_samehash, max(1, min(n, 4096)))
        sizehint!(signatures, max(1, min(n, 4096)))
        sizehint!(classes, max(1, min(n, 4096)))

        @inbounds for q in 1:n
            fill!(scratch, 0x0)
            for i in 1:m
                if Y[i].mask[q]
                    c = ((i - 1) >>> 6) + 1
                    bit = (i - 1) & 0x3f
                    scratch[c] |= UInt64(1) << bit
                end
            end

            h = UInt(0x9e3779b97f4a7c15)
            for c in 1:nchunks
                h = hash(scratch[c], h)
            end

            idx = get(heads, h, 0)
            found = 0
            while idx != 0
                sig = signatures[idx]
                same = true
                for c in 1:nchunks
                    if sig[c] != scratch[c]
                        same = false
                        break
                    end
                end
                if same
                    found = idx
                    break
                end
                idx = next_samehash[idx]
            end

            if found == 0
                push!(signatures, copy(scratch))
                push!(classes, Int[q])
                prev_head = get(heads, h, 0)
                push!(next_samehash, prev_head)
                heads[h] = length(classes)
            else
                push!(classes[found], q)
            end
        end
        return classes
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
function _image_upset(pi::EncodingMap, U::FiniteFringe.Upset)
    maskP = falses(nvertices(pi.P))
    for q in 1:nvertices(pi.Q)
        if U.mask[q]; maskP[pi.pi_of_q[q]] = true; end
    end
    FiniteFringe.upset_closure(pi.P, maskP)
end

"Image of a Q-downset under `pi` as a P-downset."
function _image_downset(pi::EncodingMap, D::FiniteFringe.Downset)
    maskP = falses(nvertices(pi.P))
    for q in 1:nvertices(pi.Q)
        if D.mask[q]; maskP[pi.pi_of_q[q]] = true; end
    end
    FiniteFringe.downset_closure(pi.P, maskP)
end

"Preimage of a P-upset under `pi` as a Q-upset."
function _preimage_upset(pi::EncodingMap, Uhat::FiniteFringe.Upset)
    maskQ = falses(nvertices(pi.Q))
    for q in 1:nvertices(pi.Q)
        if Uhat.mask[pi.pi_of_q[q]]; maskQ[q] = true; end
    end
    FiniteFringe.upset_closure(pi.Q, maskQ)
end

"Preimage of a P-downset under `pi` as a Q-downset."
function _preimage_downset(pi::EncodingMap, Dhat::FiniteFringe.Downset)
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
    UQ = [_preimage_upset(pi, Uhat) for Uhat in Hhat.U]
    DQ = [_preimage_downset(pi, Dhat) for Dhat in Hhat.D]
    FiniteFringe.FringeModule{eltype(Hhat.phi)}(pi.Q, UQ, DQ, Hhat.phi; field=Hhat.field)
end

"""
    pushforward_fringe_along_encoding(H::FringeModule_on_Q, pi::EncodingMap) -> FringeModule_on_P

Push a fringe presentation forward along a finite encoding map `pi : Q -> P`.

This sends each upset generator `U_i` of `H` to its image under `pi` and each
downset generator `D_j` to its image under `pi`, while keeping the scalar matrix
`phi` unchanged.
"""
function pushforward_fringe_along_encoding(H::FiniteFringe.FringeModule, pi::EncodingMap)
    Uhat = [_image_upset(pi, U) for U in H.U]
    Dhat = [_image_downset(pi, D) for D in H.D]
    FiniteFringe.FringeModule{eltype(H.phi)}(pi.P, Uhat, Dhat, H.phi; field=H.field)
end


end # module
