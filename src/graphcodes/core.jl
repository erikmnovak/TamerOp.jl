# Owned row supports, exact maps, and independently checkable basis witnesses.
struct GraphcodeBudget
    max_vertices::Int
    max_dimension::Int
    max_intervals::Int
    max_entries::Int
    function GraphcodeBudget(;max_vertices=10_000,max_dimension=256,max_intervals=10_000,max_entries=2_000_000)
        xs=(max_vertices,max_dimension,max_intervals,max_entries)
        all(x->x isa Integer && !(x isa Bool) && 0<=x<=typemax(Int),xs) ||
            throw(ArgumentError("graphcode budgets must be nonnegative integers fitting Int"))
        new(Int.(xs)...)
    end
end
_bound(n,limit,label)=n<=limit || throw(ArgumentError("graphcode $label budget exceeded ($n > $limit)"))
_active(row,t)=findall(s->s[1]<=t<=s[2],row)
_admissible(s,t)=t[1]<=s[1]<=t[2]<=s[2]
_index(i,n,label)=i isa Integer && !(i isa Bool) && 1<=i<=n ? Int(i) :
    throw(ArgumentError("$label must be an integer in 1:$n"))

"""A basis-dependent graphcode with inclusive finite-index supports.
`graphcode_rows` reports observed deaths and censored right boundaries;
`graphcode_map` gives coefficients in the retained row bases. Returned
accessor arrays are copies. Direct mutation of stored fields is unsupported.
"""
struct Graphcode{K,F,X,Y}
    field::F
    x::Vector{X}
    y::Vector{Y}
    rows::Vector{Vector{Tuple{Int,Int}}}
    maps::Vector{Matrix{K}}
    bases::Union{Nothing,Vector{Vector{Matrix{K}}}}
    source::Union{Nothing,PModule{K}}
    vertices::Matrix{Int}
    scope::Symbol
end
# Copy mathematical data without copying task-local caches or locks.
function Base.copy(g::Graphcode{K}) where {K}
    source=if g.source===nothing
        nothing
    else
        Q=ProductOfChainsPoset((length(g.x),length(g.y)))
        PModule{K}(Q,copy(g.source.dims),Dict((u,v)=>Matrix(map_leq(g.source,u,v)) for (u,v) in cover_edges(Q));field=field(g))
    end
    Graphcode(field(g),copy(g.x),copy(g.y),[copy(row) for row in g.rows],copy.(g.maps),
        g.bases===nothing ? nothing : [copy.(row) for row in g.bases],source,copy(g.vertices),g.scope)
end
field(g::Graphcode)=g.field
graphcode_axes(g::Graphcode)=(copy(g.x),copy(g.y))
graphcode_vertices(g::Graphcode)=copy(g.vertices)
function graphcode_rows(g::Graphcode)
    [[(;row=j,index=i,first=a,last=b,birth=g.x[a],death=b==length(g.x) ? nothing : g.x[b+1],
        left_closed=true,right_closed=b==length(g.x),terminal=b==length(g.x) ? :censored : :observed)
        for (i,(a,b)) in enumerate(row)] for (j,row) in enumerate(g.rows)]
end
function graphcode_map(g::Graphcode;row)
    copy(g.maps[_index(row,length(g.maps),"connecting row")])
end
"""Columns identify active row intervals with the retained input stalk basis."""
function graphcode_basis(g::Graphcode;row,column)
    j=_index(row,length(g.y),"row");i=_index(column,length(g.x),"column")
    g.bases===nothing ? Matrix{coeff_type(field(g))}(I,length(_active(g.rows[j],i)),length(_active(g.rows[j],i))) : copy(g.bases[j][i])
end
function graphcode_edges(g::Graphcode)
    [(;row=j,source=s,target=t,coefficient=A[t,s]) for (j,A) in enumerate(g.maps)
        for s in axes(A,2) for t in axes(A,1) if !iszero(A[t,s])]
end
function graphcode_summary(g::Graphcode)
    (;kind=:graphcode,grid=(length(g.x),length(g.y)),intervals=sum(length,g.rows),
        edges=sum(A->count(!iszero,A),g.maps;init=0),field=field(g),scope=g.scope,
        basis=:noncanonical,source_bases=g.bases!==nothing,compression=:none,right_boundary=:censored)
end
describe(g::Graphcode)=graphcode_summary(g)
Base.show(io::IO,g::Graphcode)=print(io,"Graphcode(",length(g.x)," x ",length(g.y),", ",sum(length,g.rows)," intervals, ",field(g),")")

"""Reconstruct the exact finite-grid module in the stored barcode bases."""
function graphcode_module(g::Graphcode)
    m,n=length(g.x),length(g.y);K=coeff_type(field(g));Q=ProductOfChainsPoset((m,n))
    active=[_active(g.rows[j],i) for i in 1:m,j in 1:n]
    maps=Dict{Tuple{Int,Int},Matrix{K}}()
    for j in 1:n,i in 1:m
        u=i+(j-1)*m
        i<m && (maps[u,u+1]=K[a==b for a in active[i+1,j],b in active[i,j]])
        j<n && (maps[u,u+m]=g.maps[j][active[i,j+1],active[i,j]])
    end
    PModule{K}(Q,vec(length.(active)),maps;field=field(g))
end

"""Connected graph components in the current bases; these need not be indecomposable."""
function graphcode_components(g::Graphcode)
    nodes=[(j,i) for j in eachindex(g.rows) for i in eachindex(g.rows[j])]
    lookup=Dict(v=>i for (i,v) in enumerate(nodes));adj=[Int[] for _ in nodes]
    for e in graphcode_edges(g)
        a,b=lookup[(e.row,e.source)],lookup[(e.row+1,e.target)]
        push!(adj[a],b);push!(adj[b],a)
    end
    seen=falses(length(nodes));parts=Vector{Tuple{Int,Int}}[]
    for i in eachindex(nodes)
        seen[i] && continue
        stack=[i];seen[i]=true;part=Tuple{Int,Int}[]
        while !isempty(stack)
            u=pop!(stack);push!(part,nodes[u])
            for v in adj[u];seen[v] || (seen[v]=true;push!(stack,v));end
        end
        push!(parts,sort!(part))
    end
    parts
end

"""Return a graph component's summand and exact inclusion/projection maps."""
function graphcode_summand(g::Graphcode;component)
    parts=graphcode_components(g);ids=Set(parts[_index(component,length(parts),"component")])
    ambient=graphcode_module(g);m,n=length(g.x),length(g.y);K=coeff_type(field(g))
    selected=[findall(k->(j,k) in ids,_active(g.rows[j],i)) for j in 1:n for i in 1:m]
    maps=Dict(e=>Matrix(map_leq(ambient,e[1],e[2])[selected[e[2]],selected[e[1]]]) for e in cover_edges(ambient.Q))
    N=PModule{K}(ambient.Q,length.(selected),maps;field=field(g))
    inc=Matrix{K}[];prj=Matrix{K}[]
    for (q,s) in enumerate(selected)
        B=zeros(K,ambient.dims[q],length(s))
        for (j,i) in enumerate(s);B[i,j]=one(K);end
        push!(inc,B);push!(prj,Matrix(transpose(B)))
    end
    (;module_object=N,ambient,inclusion=PMorphism(N,ambient,inc),projection=PMorphism(ambient,N,prj))
end

struct GraphcodeValidationSummary
    valid::Bool
    issues::Vector{String}
end
Base.show(io::IO,v::GraphcodeValidationSummary)=print(io,"GraphcodeValidationSummary(valid=",v.valid,", issues=",length(v.issues),")")
describe(v::GraphcodeValidationSummary)=(;valid=v.valid,issues=copy(v.issues))
Base.:(==)(a::GraphcodeValidationSummary,b::GraphcodeValidationSummary)=a.valid==b.valid && a.issues==b.issues
Base.isequal(a::GraphcodeValidationSummary,b::GraphcodeValidationSummary)=isequal(a.valid,b.valid) && isequal(a.issues,b.issues)
Base.hash(v::GraphcodeValidationSummary,h::UInt)=hash((v.valid,Tuple(v.issues)),h)

function _check_graphcode(g::Graphcode{K},budget) where {K}
    _field(field(g));coeff_type(field(g))===K || throw(ArgumentError("graphcode coefficient storage disagrees with its field"))
    _rows(g.rows,g.maps,g.x,g.y,field(g),budget)
    m,n=length(g.x),length(g.y)
    size(g.vertices)==(m,n) || throw(ArgumentError("source vertex grid has the wrong shape"))
    all(>(0),g.vertices) || throw(ArgumentError("source vertices must be positive indices"))
    g.scope in (:finite_grid,:grid_restriction,:supplied_rows) || throw(ArgumentError("invalid graphcode scope"))
    (g.source===nothing)==(g.bases===nothing) || throw(ArgumentError("source and basis witnesses must occur together"))
    if g.source!==nothing
        length(g.bases)==n || throw(ArgumentError("one basis family per row is required"))
        N=graphcode_module(g);S=g.source
        S.Q isa ProductOfChainsPoset{2} && S.Q.sizes==(m,n) && S.Q.orientation==(1,1) ||
            throw(ArgumentError("retained source must use the canonical ordered grid"))
        S.field==field(g) && S.dims==N.dims || throw(ArgumentError("source dimensions or field disagree"))
        for j in 1:n
            length(g.bases[j])==m || throw(ArgumentError("one basis per column is required"))
            for i in 1:m
                B=g.bases[j][i];q=i+(j-1)*m;d=N.dims[q]
                size(B)==(d,d) && FieldLinAlg.rank(field(g),B)==d || throw(ArgumentError("barcode basis is not invertible"))
            end
        end
        for (u,v) in cover_edges(N.Q)
            ui,uj=mod1(u,m),cld(u,m);vi,vj=mod1(v,m),cld(v,m)
            map_leq(S,u,v)*g.bases[uj][ui]==g.bases[vj][vi]*map_leq(N,u,v) ||
                throw(ArgumentError("barcode basis does not intertwine the source maps"))
        end
    end
    nothing
end
"""Check supports, coefficient admissibility, and every retained basis/map equation."""
function check_graphcode(g;throw=false,budget::GraphcodeBudget=GraphcodeBudget())
    issues=String[]
    try;_check_graphcode(g,budget)
    catch err;err isa InterruptException && rethrow();push!(issues,sprint(showerror,err));end
    throw && !isempty(issues) && Base.throw(ArgumentError(join(issues,"; ")))
    GraphcodeValidationSummary(isempty(issues),issues)
end
