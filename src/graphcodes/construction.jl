# Exact row decompositions and admissible coefficients of natural maps.
function _coordinates(values,label)
    values isa AbstractVector && !isempty(values) || throw(ArgumentError("$label must be a nonempty coordinate vector"))
    all(x->x isa Real && !(x isa Bool) && isfinite(x),values) || throw(ArgumentError("$label must contain finite real coordinates"))
    T=mapreduce(typeof,promote_type,values);isconcretetype(T) || throw(ArgumentError("$label needs a concrete coordinate type"))
    xs=T[x for x in values]
    all(xs[i]==values[i] for i in eachindex(xs)) || throw(ArgumentError("coordinate conversion loses precision"))
    (all(xs[i]<xs[i+1] for i in 1:length(xs)-1) || all(xs[i]>xs[i+1] for i in 1:length(xs)-1)) ||
        throw(ArgumentError("$label must be strictly monotone in module order"))
    xs
end
function _field(F)
    F isa Union{QQField,PrimeField} || throw(ArgumentError("graphcodes require QQField() or a prime field"))
    F
end
function _coefficient(F,x)
    K=coeff_type(F)
    if x isa FpElem
        typeof(x)===K || throw(ArgumentError("graphcode coefficient belongs to a different field"))
        return x
    end
    x isa Union{Integer,Rational} || throw(ArgumentError("graphcode coefficients must be exact elements of the specified field"))
    isfinite(x) || throw(ArgumentError("graphcode coefficients must be finite"))
    coerce(F,x)
end
function _rows(rows,maps,x,y,F,budget)
    m,n=length(x),length(y)
    _coordinates(x,"x");_coordinates(y,"y")
    _bound(big(m)*n,budget.max_vertices,"vertices")
    length(rows)==n && length(maps)==n-1 || throw(ArgumentError("one barcode per row and one map per adjacent row pair are required"))
    _bound(sum(big(length(r)) for r in rows;init=big(0)),budget.max_intervals,"intervals")
    for row in rows,s in row
        s isa Tuple && length(s)==2 && all(v->v isa Integer && !(v isa Bool),s) && 1<=s[1]<=s[2]<=m ||
            throw(ArgumentError("row intervals must be inclusive index pairs (first,last) inside 1:$m"))
    end
    for row in rows,i in 1:m
        _bound(count(s->s[1]<=i<=s[2],row),budget.max_dimension,"stalk dimension")
    end
    _bound(sum(big(length(rows[j]))*length(rows[j+1]) for j in 1:n-1;init=big(0)),budget.max_entries,"coefficient entries")
    for (j,A) in enumerate(maps)
        A isa AbstractMatrix && size(A)==(length(rows[j+1]),length(rows[j])) || throw(ArgumentError("connecting matrix $j has the wrong shape"))
        for s in axes(A,2),t in axes(A,1)
            a=_coefficient(F,A[t,s])
            iszero(a) || _admissible(rows[j][s],rows[j+1][t]) ||
                throw(ArgumentError("nonzero coefficient ($t,$s) in row $j violates the interval Hom condition"))
        end
    end
    nothing
end

"""`graphcode(rows, maps; x, y, field=F2())` builds a checked graphcode from inclusive
index supports `(first,last)` in each row and target-by-source coefficient
matrices. `x` and `y` are exact coordinate labels in module order (either
increasing or decreasing). Duplicate intervals remain separate basis members.
"""
function graphcode(rows::AbstractVector,maps::AbstractVector;x,y,field=F2(),budget::GraphcodeBudget=GraphcodeBudget())
    F=_field(field)
    xs=_coordinates(x,"x");ys=_coordinates(y,"y");_rows(rows,maps,xs,ys,F,budget)
    K=coeff_type(F);rr=[Tuple{Int,Int}[(Int(a),Int(b)) for (a,b) in row] for row in rows]
    aa=Matrix{K}[map(x->_coefficient(F,x),A) for A in maps]
    Graphcode(F,xs,ys,rr,aa,nothing,nothing,reshape(collect(1:length(xs)*length(ys)),length(xs),length(ys)),:supplied_rows)
end

function _grid(M,vertices,x,y,budget)
    Q=M.Q;scope=:grid_restriction
    if vertices===nothing
        Q isa Union{ProductOfChainsPoset{2},GridPoset{2}} || throw(ArgumentError("supply an order-preserving matrix of vertex indices for a non-grid module"))
        m,n=Q.sizes;_bound(big(m)*n,budget.max_vertices,"vertices")
        m>0 && n>0 || throw(ArgumentError("graphcodes require a nonempty index grid"))
        orientation=Q isa ProductOfChainsPoset ? Q.orientation : (1,1)
        ii=orientation[1]==1 ? collect(1:m) : collect(m:-1:1)
        jj=orientation[2]==1 ? collect(1:n) : collect(n:-1:1)
        ids=[i+(j-1)*m for i in ii,j in jj]
        x===nothing && (x=Q isa GridPoset ? Q.coords[1][ii] : ii)
        y===nothing && (y=Q isa GridPoset ? Q.coords[2][jj] : jj)
        scope=:finite_grid
    else
        vertices isa AbstractMatrix || throw(ArgumentError("vertices must be a matrix of finite-poset labels"))
        m,n=size(vertices);m>0 && n>0 || throw(ArgumentError("vertex grid must be nonempty"))
        _bound(big(m)*n,budget.max_vertices,"vertices")
        ids=map(v->_index(v,nvertices(Q),"vertex"),vertices)
        x===nothing && (x=collect(1:m));y===nothing && (y=collect(1:n))
    end
    xs=_coordinates(x,"x");ys=_coordinates(y,"y")
    size(ids)==(length(xs),length(ys)) || throw(ArgumentError("coordinates do not match the vertex grid"))
    m,n=size(ids)
    for j in 1:n,i in 1:m
        i<m && !leq(Q,ids[i,j],ids[i+1,j]) && throw(ArgumentError("vertex grid is not order preserving along x"))
        j<n && !leq(Q,ids[i,j],ids[i,j+1]) && throw(ArgumentError("vertex grid is not order preserving along y"))
    end
    ids,xs,ys,scope
end

"""`graphcode(M; vertices=nothing, x=nothing, y=nothing, budget=GraphcodeBudget())`
constructs row barcodes and their actual maps from a finite module over QQ or a prime field. Native
two-dimensional grid orders, including reversed axes, use all vertices.
An explicit order-preserving vertex matrix instead declares a grid restriction;
repeated labels are permitted. The result owns the restricted source and its
noncanonical basis identifications. It does not infer maps from bar matching.
"""
function graphcode(M::PModule;vertices=nothing,x=nothing,y=nothing,budget::GraphcodeBudget=GraphcodeBudget())
    F=_field(M.field)
    _bound(nvertices(M.Q),budget.max_vertices,"input vertices")
    ids,xs,ys,scope=_grid(M,vertices,x,y,budget);m,n=size(ids);K=coeff_type(F)
    dims=M.dims[vec(ids)];d=maximum(dims;init=0);_bound(d,budget.max_dimension,"stalk dimension")
    entries=sum(big(v)^2 for v in dims;init=big(0))
    Q=ProductOfChainsPoset((m,n));edges=collect(cover_edges(Q))
    entries+=sum(big(dims[u])*dims[v] for (u,v) in edges;init=big(0))
    _bound(3entries+6big(d)^2,budget.max_entries,"source and row bases")
    # Reject malformed input diagrams before trusting an arbitrary path query.
    check_module_functoriality(M;throw=true)
    source=PModule{K}(Q,dims,Dict((u,v)=>Matrix(map_leq(M,ids[u],ids[v])) for (u,v) in edges);field=F)
    Z=ZigzagPersistence;b=Z.ZigzagBudget(max_vertices=budget.max_vertices,max_dimension=budget.max_dimension,
        max_matrix_entries=budget.max_entries,max_representative_entries=budget.max_entries)
    rows=Vector{Tuple{Int,Int}}[];bases=Vector{Matrix{K}}[];total=0
    for j in 1:n
        rids=collect(1:m).+(j-1)*m
        z=Z.ZigzagModule(dims[rids],[Matrix(map_leq(source,rids[i],rids[i+1])) for i in 1:m-1];directions=fill(:forward,m-1),field=F,budget=b)
        result=Z.zigzag_persistence(z;representatives=true,budget=b);bars=Z.zigzag_intervals(result)
        order=sortperm(bars;by=t->(t.first,t.last));supports=[(bars[k].first,bars[k].last) for k in order]
        total+=length(supports);_bound(total,budget.max_intervals,"intervals")
        Bs=Matrix{K}[]
        for i in 1:m
            B=zeros(K,dims[rids[i]],dims[rids[i]])
            for (c,k) in enumerate(_active(supports,i));B[:,c]=Z.zigzag_representative(result;interval=order[k],vertex=i);end
            push!(Bs,B)
        end
        push!(rows,supports);push!(bases,Bs)
    end
    _bound(entries+sum(big(length(rows[j]))*length(rows[j+1]) for j in 1:n-1;init=big(0)),budget.max_entries,"retained entries")
    maps=Matrix{K}[]
    for j in 1:n-1
        C=zeros(K,length(rows[j+1]),length(rows[j]))
        for i in 1:m
            born=findall(s->s[1]==i,rows[j]);isempty(born) && continue
            active=_active(rows[j],i);target=_active(rows[j+1],i)
            columns=[findfirst(==(k),active) for k in born]
            C[target,born]=FieldLinAlg.solve_fullcolumn(F,bases[j+1][i],map_leq(source,i+(j-1)*m,i+j*m)*bases[j][i][:,columns])
        end
        push!(maps,C)
    end
    g=Graphcode(F,xs,ys,rows,maps,bases,source,ids,scope)
    check_graphcode(g;throw=true,budget);g
end
graphcode(e::Results.EncodingResult;kwargs...)=graphcode(Results.encoding_module(e);kwargs...)
