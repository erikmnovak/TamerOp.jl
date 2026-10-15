# Lossless elimination of superfluous vertices (Kerber--Russold, Definition 3.11),
# retaining nonbinary scalars on both external edges and internal continuations.
"""A compressed coefficient-labelled graphcode. Each member span retains its
original row/interval labels and every intermediate scalar, so expansion
recovers the exact supplied bases and matrices, including signs and weights.
Use `graphcode_edges` and `graphcode_continuations` to inspect those scalars."""
struct CompressedGraphcode{K,F,X,Y}
    field::F
    x::Vector{X}
    y::Vector{Y}
    supports::Vector{Tuple{Int,Int}}
    members::Vector{Vector{Tuple{Int,Int}}}
    edges::Vector{Tuple{Int,Int}}
    coefficients::Vector{K}
    continuations::Vector{Vector{K}}
end
field(c::CompressedGraphcode)=c.field
graphcode_axes(g::CompressedGraphcode)=(copy(g.x),copy(g.y))
"""External compressed edges, indexed by compressed vertex, with exact weights.
Each edge runs from the last member of its source span to the first member of
its target span. Internal span maps are returned by `graphcode_continuations`."""
graphcode_edges(c::CompressedGraphcode)=[(;source=u,target=v,coefficient=a) for ((u,v),a) in zip(c.edges,c.coefficients)]
"""Retained row/interval labels and consecutive map scalars for each compressed
vertex. The kth scalar maps member k to member k+1; all arrays are owned copies."""
graphcode_continuations(c::CompressedGraphcode)=[(;members=copy(m),coefficients=copy(a)) for (m,a) in zip(c.members,c.continuations)]
function graphcode_summary(c::CompressedGraphcode)
    (;kind=:compressed_graphcode,grid=(length(c.x),length(c.y)),vertices=length(c.supports),edges=length(c.edges),
        expanded_intervals=sum(length,c.members),field=field(c),basis=:noncanonical,compression=:superfluous_vertices)
end
describe(c::CompressedGraphcode)=graphcode_summary(c)
Base.show(io::IO,c::CompressedGraphcode)=print(io,"CompressedGraphcode(",length(c.supports)," vertices, ",length(c.edges)," edges)")
"""Losslessly remove every superfluous graph vertex. Terminal vertices remain;
this is graph compression in the supplied bases, not indecomposable decomposition.
The result owns no source matrices or expanded row coefficient tables.
"""
function compress_graphcode(g::Graphcode{K};budget::GraphcodeBudget=GraphcodeBudget()) where {K}
    check_graphcode(g;throw=true,budget)
    nodes=[(j,i) for j in eachindex(g.rows) for i in eachindex(g.rows[j])]
    lookup=Dict(v=>i for (i,v) in enumerate(nodes));supports=[g.rows[j][i] for (j,i) in nodes]
    members=[[v] for v in nodes];incoming=[Int[] for _ in nodes];outgoing=[Int[] for _ in nodes]
    weights=Dict{Tuple{Int,Int},K}();continuations=[K[] for _ in nodes]
    for e in graphcode_edges(g)
        a,b=lookup[(e.row,e.source)],lookup[(e.row+1,e.target)]
        push!(outgoing[a],b);push!(incoming[b],a)
        weights[a,b]=e.coefficient
    end
    alive=trues(length(nodes));changed=true
    while changed
        changed=false
        for w in eachindex(nodes)
            alive[w] && length(incoming[w])==1 && !isempty(outgoing[w]) || continue
            v=only(incoming[w])
            length(outgoing[v])==1 && supports[v]==supports[w] || continue
            push!(continuations[v],weights[v,w]);append!(continuations[v],continuations[w])
            delete!(weights,(v,w))
            append!(members[v],members[w]);outgoing[v]=copy(outgoing[w])
            for t in outgoing[w]
                replace!(incoming[t],w=>v)
                weights[v,t]=pop!(weights,(w,t))
            end
            alive[w]=false;empty!(incoming[w]);empty!(outgoing[w]);changed=true
        end
    end
    kept=findall(alive);renumber=Dict(v=>i for (i,v) in enumerate(kept))
    edges=[(v,w) for v in kept for w in outgoing[v]]
    c=CompressedGraphcode(field(g),copy(g.x),copy(g.y),supports[kept],members[kept],
        [(renumber[v],renumber[w]) for (v,w) in edges],K[weights[e] for e in edges],continuations[kept])
    check_graphcode(c;throw=true,budget);c
end
"""Expand a compressed graphcode, restoring the original row/interval ordering."""
function expand_graphcode(c::CompressedGraphcode{K};budget::GraphcodeBudget=GraphcodeBudget()) where {K}
    _field(field(c));coeff_type(field(c))===K || throw(ArgumentError("compressed coefficient storage disagrees with field"))
    m,n=length(c.x),length(c.y);_coordinates(c.x,"x");_coordinates(c.y,"y")
    _bound(big(m)*n,budget.max_vertices,"vertices");_bound(sum(big(length(v)) for v in c.members;init=big(0)),budget.max_intervals,"intervals")
    length(c.supports)==length(c.members)==length(c.continuations) || throw(ArgumentError("compressed supports, members and continuations disagree"))
    length(c.edges)==length(c.coefficients) || throw(ArgumentError("compressed edges and coefficients disagree"))
    all(!iszero,c.coefficients) || throw(ArgumentError("compressed edges must have nonzero coefficients"))
    counts=zeros(Int,n);seen=Set{Tuple{Int,Int}}()
    for (members,scalars) in zip(c.members,c.continuations)
        isempty(members) && throw(ArgumentError("compressed node must retain a nonempty row span"))
        length(scalars)==length(members)-1 && all(!iszero,scalars) ||
            throw(ArgumentError("each internal continuation needs a nonzero coefficient"))
        for (k,(r,i)) in enumerate(members)
            _index(r,n,"member row");i>0 || throw(ArgumentError("invalid member interval"))
            k>1 && r!=members[k-1][1]+1 && throw(ArgumentError("compressed members must span consecutive rows"))
            (r,i) in seen && throw(ArgumentError("duplicate compressed member"));push!(seen,(r,i));counts[r]=max(counts[r],i)
        end
    end
    sum(big,counts)==length(seen) || throw(ArgumentError("compressed row members have gaps"))
    _bound(sum(big(counts[j])*counts[j+1] for j in 1:n-1;init=big(0)),budget.max_entries,"expanded coefficients")
    rows=[Vector{Tuple{Int,Int}}(undef,k) for k in counts]
    maps=[zeros(K,counts[j+1],counts[j]) for j in 1:n-1]
    for (v,members) in enumerate(c.members)
        for (r,i) in members;rows[r][i]=c.supports[v];end
        for k in 1:length(members)-1
            r,s=members[k];t=members[k+1][2];maps[r][t,s]=c.continuations[v][k]
        end
    end
    length(unique(c.edges))==length(c.edges) || throw(ArgumentError("duplicate compressed edge"))
    for ((v,w),a) in zip(c.edges,c.coefficients)
        _index(v,length(c.members),"edge source");_index(w,length(c.members),"edge target")
        r,s=last(c.members[v]);q,t=first(c.members[w]);q==r+1 || throw(ArgumentError("edge must join adjacent span boundaries"))
        maps[r][t,s]=a
    end
    graphcode(rows,maps;x=c.x,y=c.y,field=field(c),budget)
end
_check_graphcode(c::CompressedGraphcode,budget)=(_check_graphcode(expand_graphcode(c;budget),budget);nothing)
graphcode_module(c::CompressedGraphcode)=graphcode_module(expand_graphcode(c))
