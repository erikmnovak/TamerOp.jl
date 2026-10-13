# Read-only selected homology queries: build only the requested stalks and
# inclusion maps, and only chain degrees t-1, t, t+1. Never populate the parent
# encoding's full-grid active lists, dimensions or module cache.
import ..DerivedFunctors: _query_poset, _query_module, _betti_may_be_nonzero
import ..InvariantCore: rank_map

_query_poset(M::_LazyEncodedModule) = M.lazy.P

function _query_module(M::_LazyEncodedModule, labels::Vector{Int})
    L = M.lazy
    P = DerivedFunctors._query_subposet(L.P, labels)
    sizes = length.(L.axes)
    strides = ntuple(i -> prod(sizes[1:i-1]; init=1), length(sizes))
    points = [ntuple(i -> div(v-1, strides[i]) % sizes[i] + 1, length(sizes)) for v in labels]
    nd = length(L.terms)
    active = Any[nothing for _ in 1:nd]
    for i in max(1,M.degree):min(nd,M.degree+2)
        active[i] = _active_lists(L.births_by_dim[i], points, L.orientation; multicritical=L.multicritical)
    end
    local_complex = LazyModuleCochainComplex(L.tmin, L.tmax, P, L.axes, L.orientation,
        L.field, L.multicritical, L.grades_by_dim, L.boundaries,
        Any[nothing for _ in L.boundaries], points, L.births_by_dim, active,
        Any[nothing for _ in 1:nd], Any[nothing for _ in 1:nd],
        Any[nothing for _ in L.diffs])
    return _cohomology_module_from_lazy(local_complex, M.degree), collect(eachindex(labels))
end

function _in_entrance_join_closure(births::Vector{NTuple{N,Int}}, point::NTuple{N,Int}) where {N}
    joined = ntuple(_ -> 0, N)
    found = false
    for b in births
        _grid_birth_leq(b, point) || continue
        joined = ntuple(i -> max(joined[i], b[i]), N)
        found = true
    end
    return found && joined == point
end

function _betti_may_be_nonzero(M::_LazyEncodedModule, v::Int)
    L = M.lazy; t = M.degree
    0 <= t < length(L.terms) || return false
    births = L.births_by_dim[t+1]
    births isa Vector{<:Tuple} || return true
    upper = t+2 <= length(L.terms) ? L.births_by_dim[t+2] : empty(births)
    upper isa Vector{<:Tuple} || return true
    sizes = length.(L.axes)
    point = ntuple(i -> div(v-1, prod(sizes[1:i-1]; init=1)) % sizes[i]+1, length(sizes))
    # Guidolin--Landi, Theorem 5.10 (arXiv:2303.05294v3): the union of
    # Betti supports lies in join(G(C_t)) union join(G(C_{t+1})). Membership
    # is tested by joining all entrance grades <= point, without enumerating
    # the closure. Native grid births define free one-critical chain modules
    # under zero-below/constant-above extension, including a coarsened grid.
    return _in_entrance_join_closure(births, point) || _in_entrance_join_closure(upper, point)
end

function DerivedFunctors.betti_table(M::_LazyEncodedModule; vertices=nothing,
        opts::ResolutionOptions=ResolutionOptions(), pad_to::Union{Nothing,Int}=nothing,
        threads::Bool=(Threads.nthreads()>1))
    M.cached_module === nothing || return DerivedFunctors.betti_table(M.cached_module;
        vertices, opts, pad_to, threads)
    if vertices !== nothing
        result = DerivedFunctors._selected_betti_table(M, vertices, opts, pad_to; threads)
        result === nothing || return result
    end
    return DerivedFunctors.betti_table(materialize_module(M); vertices, opts, pad_to, threads)
end

function rank_map(M::_LazyEncodedModule, a::Int, b::Int; cache=nothing, memo=nothing)::Int
    P = M.lazy.P
    1 <= a <= nvertices(P) || error("rank_map: a out of range")
    1 <= b <= nvertices(P) || error("rank_map: b out of range")
    leq(P,a,b) || error("rank_map: a and b are not comparable (need a <= b)")
    M.cached_module === nothing || return rank_map(M.cached_module,a,b; cache,memo)
    # The caller's global map memo cannot be reused for these local vertex IDs.
    labels = a == b ? [a] : [a,b]
    local_module, ids = _query_module(M, labels)
    return rank_map(local_module, first(ids), last(ids))
end
