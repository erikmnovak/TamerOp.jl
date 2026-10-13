# Dynamic minimum spanning forest for the second coordinate of a lexicographic
# graph sweep. Edge nodes carry weights; vertex nodes have weight zero.
# Splay link-cut trees implement the merge-time operation of Luo, Morozov and
# Scoccola, arXiv:2410.22242v3, Algorithm 4, in amortized logarithmic time.
mutable struct _H0Forest
    left::Vector{Int}
    right::Vector{Int}
    parent::Vector{Int}
    reverse::BitVector
    weight::Vector{Int}
    maximum::Vector{Int}
    stack::Vector{Int}
end
_H0Forest(weights::Vector{Int}) = _H0Forest(zeros(Int,length(weights)),
    zeros(Int,length(weights)),zeros(Int,length(weights)),falses(length(weights)),
    weights,collect(eachindex(weights)),Int[])
@inline _h0_root(T, x) = T.parent[x] == 0 ||
    (T.left[T.parent[x]] != x && T.right[T.parent[x]] != x)
@inline function _h0_pull!(T, x)
    best = x
    for y in (T.left[x],T.right[x])
        y == 0 && continue
        z = T.maximum[y]
        if (T.weight[z],z) > (T.weight[best],best)
            best = z
        end
    end
    T.maximum[x] = best
end
@inline function _h0_reverse!(T,x)
    x == 0 && return
    T.left[x],T.right[x] = T.right[x],T.left[x]
    T.reverse[x] = !T.reverse[x]
end
@inline function _h0_push!(T,x)
    T.reverse[x] || return
    _h0_reverse!(T,T.left[x]); _h0_reverse!(T,T.right[x])
    T.reverse[x] = false
end
function _h0_rotate!(T,x)
    p,g = T.parent[x],T.parent[T.parent[x]]
    if !_h0_root(T,p)
        T.left[g] == p ? (T.left[g]=x) : (T.right[g]=x)
    end
    T.parent[x] = g
    if T.left[p] == x
        b = T.right[x]; T.right[x] = p; T.left[p] = b
    else
        b = T.left[x]; T.left[x] = p; T.right[p] = b
    end
    b == 0 || (T.parent[b] = p)
    T.parent[p] = x
    _h0_pull!(T,p); _h0_pull!(T,x)
end
function _h0_splay!(T,x)
    empty!(T.stack); push!(T.stack,x)
    y=x
    while !_h0_root(T,y)
        y=T.parent[y]; push!(T.stack,y)
    end
    for y in Iterators.reverse(T.stack); _h0_push!(T,y); end
    while !_h0_root(T,x)
        p = T.parent[x]
        if !_h0_root(T,p)
            g = T.parent[p]
            _h0_rotate!(T, (T.left[p]==x)==(T.left[g]==p) ? p : x)
        end
        _h0_rotate!(T,x)
    end
end
function _h0_access!(T,x)
    last=0; y=x
    while y != 0
        _h0_splay!(T,y)
        T.right[y]=last
        last == 0 || (T.parent[last]=y)
        _h0_pull!(T,y)
        last=y; y=T.parent[y]
    end
    _h0_splay!(T,x)
end
function _h0_makeroot!(T,x)
    _h0_access!(T,x); _h0_reverse!(T,x)
end
function _h0_findroot!(T,x)
    _h0_access!(T,x); _h0_push!(T,x)
    while T.left[x] != 0
        x=T.left[x]; _h0_push!(T,x)
    end
    _h0_splay!(T,x)
    return x
end
function _h0_link!(T,u,v)
    _h0_makeroot!(T,u); T.parent[u]=v
end
function _h0_cut!(T,u,v)
    _h0_makeroot!(T,u); _h0_access!(T,v)
    (T.left[v]==u && T.right[u]==0) || error("H0 forest: vertices are not adjacent")
    T.left[v]=0; T.parent[u]=0; _h0_pull!(T,v)
end
# Called only after exposing a path. Reporting its nodes charges linear work
# in the returned witness, while Betti-only requests never traverse the path.
function _h0_path_nodes(T,v)
    nodes=Int[]; stack=Int[]; x=v
    while x != 0 || !isempty(stack)
        while x != 0
            _h0_push!(T,x); push!(stack,x); x=T.left[x]
        end
        x=pop!(stack); push!(nodes,x); x=T.right[x]
    end
    return nodes
end
