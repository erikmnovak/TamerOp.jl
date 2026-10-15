# Lossless elimination of superfluous vertices (Kerber--Russold, Definition 3.11).
# Original row/member identities make expansion deterministic, including duplicates.
struct CompressedGraphcode{X,Y}
    x::Vector{X}
    y::Vector{Y}
    supports::Vector{Tuple{Int,Int}}
    members::Vector{Vector{Tuple{Int,Int}}}
    edges::Vector{Tuple{Int,Int}}
end
field(::CompressedGraphcode)=F2()
graphcode_axes(g::CompressedGraphcode)=(copy(g.x),copy(g.y))
function graphcode_summary(c::CompressedGraphcode)
    (;kind=:compressed_graphcode,grid=(length(c.x),length(c.y)),vertices=length(c.supports),edges=length(c.edges),
        expanded_intervals=sum(length,c.members),field=F2(),basis=:noncanonical,compression=:superfluous_vertices)
end
describe(c::CompressedGraphcode)=graphcode_summary(c)
Base.show(io::IO,c::CompressedGraphcode)=print(io,"CompressedGraphcode(",length(c.supports)," vertices, ",length(c.edges)," edges)")
"""Losslessly remove every superfluous graph vertex. Terminal vertices remain;
this is graph compression in the supplied bases, not indecomposable decomposition.
The result owns no source matrices or expanded row coefficient tables.
"""
function compress_graphcode(g::Graphcode;budget::GraphcodeBudget=GraphcodeBudget())
    check_graphcode(g;throw=true,budget)
    nodes=[(j,i) for j in eachindex(g.rows) for i in eachindex(g.rows[j])]
    lookup=Dict(v=>i for (i,v) in enumerate(nodes));supports=[g.rows[j][i] for (j,i) in nodes]
    members=[[v] for v in nodes];incoming=[Int[] for _ in nodes];outgoing=[Int[] for _ in nodes]
    for e in graphcode_edges(g)
        a,b=lookup[(e.row,e.source)],lookup[(e.row+1,e.target)]
        push!(outgoing[a],b);push!(incoming[b],a)
    end
    alive=trues(length(nodes));changed=true
    while changed
        changed=false
        for w in eachindex(nodes)
            alive[w] && length(incoming[w])==1 && !isempty(outgoing[w]) || continue
            v=only(incoming[w])
            length(outgoing[v])==1 && supports[v]==supports[w] || continue
            append!(members[v],members[w]);outgoing[v]=copy(outgoing[w])
            for t in outgoing[w];replace!(incoming[t],w=>v);end
            alive[w]=false;empty!(incoming[w]);empty!(outgoing[w]);changed=true
        end
    end
    kept=findall(alive);renumber=Dict(v=>i for (i,v) in enumerate(kept))
    c=CompressedGraphcode(copy(g.x),copy(g.y),supports[kept],members[kept],
        [(renumber[v],renumber[w]) for v in kept for w in outgoing[v]])
    check_graphcode(c;throw=true,budget);c
end
"""Expand a compressed graphcode, restoring the original row/interval ordering."""
function expand_graphcode(c::CompressedGraphcode;budget::GraphcodeBudget=GraphcodeBudget())
    m,n=length(c.x),length(c.y);_coordinates(c.x,"x");_coordinates(c.y,"y")
    _bound(big(m)*n,budget.max_vertices,"vertices");_bound(sum(big(length(v)) for v in c.members;init=big(0)),budget.max_intervals,"intervals")
    length(c.supports)==length(c.members) || throw(ArgumentError("compressed supports and members disagree"))
    counts=zeros(Int,n);seen=Set{Tuple{Int,Int}}()
    for members in c.members
        isempty(members) && throw(ArgumentError("compressed node must retain a nonempty row span"))
        for (k,(r,i)) in enumerate(members)
            _index(r,n,"member row");i>0 || throw(ArgumentError("invalid member interval"))
            k>1 && r!=members[k-1][1]+1 && throw(ArgumentError("compressed members must span consecutive rows"))
            (r,i) in seen && throw(ArgumentError("duplicate compressed member"));push!(seen,(r,i));counts[r]=max(counts[r],i)
        end
    end
    sum(big,counts)==length(seen) || throw(ArgumentError("compressed row members have gaps"))
    _bound(sum(big(counts[j])*counts[j+1] for j in 1:n-1;init=big(0)),budget.max_entries,"expanded coefficients")
    rows=[Vector{Tuple{Int,Int}}(undef,k) for k in counts];K=coeff_type(F2())
    maps=[zeros(K,counts[j+1],counts[j]) for j in 1:n-1]
    for (v,members) in enumerate(c.members)
        for (r,i) in members;rows[r][i]=c.supports[v];end
        for k in 1:length(members)-1
            r,s=members[k];t=members[k+1][2];maps[r][t,s]=one(K)
        end
    end
    length(unique(c.edges))==length(c.edges) || throw(ArgumentError("duplicate compressed edge"))
    for (v,w) in c.edges
        _index(v,length(c.members),"edge source");_index(w,length(c.members),"edge target")
        r,s=last(c.members[v]);q,t=first(c.members[w]);q==r+1 || throw(ArgumentError("edge must join adjacent span boundaries"))
        maps[r][t,s]=one(K)
    end
    graphcode(rows,maps;x=c.x,y=c.y,budget)
end
_check_graphcode(c::CompressedGraphcode,budget)=(_check_graphcode(expand_graphcode(c;budget),budget);nothing)
graphcode_module(c::CompressedGraphcode)=graphcode_module(expand_graphcode(c))
