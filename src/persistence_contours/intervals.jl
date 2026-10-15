# Sorted antichains make both survival and the optimal interval covering sweep
# linear after presentation normalization. Chacholski--Corbet--Sattelberger,
# Algorithm 3.24 / Corollary 3.26, doi:10.1007/s41468-024-00169-6.
function _minimal(points)
    sorted=sort!(NTuple{2,QQ}[p for p in points]);result=NTuple{2,QQ}[]
    for p in sorted
        (isempty(result) || p[2]<last(result)[2]) && push!(result,p)
    end
    result
end
function _in_upset(g,p)
    i=searchsortedlast(g,p;by=first)
    i>0 && g[i][2]<=p[2]
end
function _intersect_upsets(g,d)
    result=NTuple{2,QQ}[];i=0;j=0
    while i<length(g) || j<length(d)
        x=i==length(g) ? d[j+1][1] : j==length(d) ? g[i+1][1] : min(g[i+1][1],d[j+1][1])
        while i<length(g) && g[i+1][1]==x;i+=1;end
        while j<length(d) && d[j+1][1]==x;j+=1;end
        if i>0 && j>0
            y=max(g[i][2],d[j][2])
            (isempty(result) || y<last(result)[2]) && push!(result,(x,y))
        end
    end
    result
end

"""Read a declared half-open, zero-extended Cartesian encoding as one interval.
Every finite stalk must have dimension zero or one, and every cover between
nonzero stalks must be nonzero. The complete support is verified against its
upset presentation. `axes` includes the final upper boundary on each axis.
"""
function BivariateInterval(M::PModule;axes,budget::ContourBudget=ContourBudget())
    _limit(nvertices(M.Q),budget.max_vertices,"finite vertices")
    check_module_functoriality(M;throw=true)
    xs=Tuple([_coordinate.(collect(a)) for a in axes])
    length(xs)==2 && all(a->length(a)>=2 && issorted(a) && allunique(a),xs) ||
        throw(ArgumentError("supply two strictly increasing arrays of cell boundaries"))
    shape=(length(xs[1])-1,length(xs[2])-1)
    prod(big.(shape))==length(M.dims) || throw(DimensionMismatch("axis cells do not match the finite module"))
    Q=ProductOfChainsPoset(shape)
    Set(cover_edges(Q))==Set(cover_edges(M.Q)) || throw(ArgumentError("the module must use the declared positive Cartesian order"))
    all(d->d in (0,1),M.dims) || throw(ArgumentError("this route requires a single thin interval; direct sums need a general solver"))
    for (u,v) in cover_edges(M.Q)
        M.dims[u]*M.dims[v]==0 || !iszero(M.edge_maps[u,v]) || throw(ArgumentError("a supported interval cover has a zero map"))
    end
    labels=LinearIndices(shape);active(i,j)=1<=i<=shape[1] && 1<=j<=shape[2] && M.dims[labels[i,j]]==1
    births=NTuple{2,QQ}[];deaths=NTuple{2,QQ}[]
    for j in 1:shape[2],i in 1:shape[1]
        active(i,j) || continue
        !active(i-1,j) && !active(i,j-1) && push!(births,(xs[1][i],xs[2][j]))
        active(i+1,j) || push!(deaths,(xs[1][i+1],xs[2][j]))
        active(i,j+1) || push!(deaths,(xs[1][i],xs[2][j+1]))
    end
    I=BivariateInterval(births,deaths;field=M.field,budget)
    for j in 1:shape[2],i in 1:shape[1]
        p=(xs[1][i],xs[2][j])
        (_in_upset(I.generators,p) && !_in_upset(I.relations,p))==active(i,j) ||
            throw(ArgumentError("the finite support is not an interval with these boundary conventions"))
    end
    I
end
function _survivors(I,v)
    g=I.generators;d=I.relations;alive=trues(length(g));j=0
    for i in eachindex(g)
        q=g[i].+v
        while j<length(d) && d[j+1][1]<=q[1];j+=1;end
        alive[i]=j==0 || d[j][2]>q[2]
    end
    alive
end
function _shift_sweep(I,v,b,witnesses)
    g=I.generators;n=length(g)
    _limit(big(n)+length(I.relations),b.max_generators,"presentation corners")
    _limit(3big(n)+length(I.relations),b.max_work,"sweep work")
    alive=_survivors(I,v);chosen=Int[];packing=Int[];i=1;j=1
    while i<=n
        if !alive[i];i+=1;continue;end
        demand=i;j=max(j,i)
        while j<n && g[j+1][1]<=g[i][1]+v[1];j+=1;end
        push!(chosen,j);push!(packing,demand)
        while i<=n && (!alive[i] || g[j][2]<=g[i][2]+v[2]);i+=1;end
    end
    retained=witnesses ? (;indices=chosen,generators=copy(g[chosen]),packing_indices=packing,
        packing_points=[g[i].+v for i in packing]) : nothing
    ShiftDimensionResult(I,v,length(chosen),retained,(;generators=n,killing_corners=length(I.relations),survivors=count(alive)))
end
"""Compute the minimum number of homogeneous generators whose submodule
contains the shifted image. The linear sweep is exact for this single
bivariate interval. Optional witnesses include generators and a matching
packing certificate proving that fewer generators cannot suffice.
"""
function shift_dimension(I::BivariateInterval;shift,witnesses::Bool=false,budget::ContourBudget=ContourBudget())
    v=_point(shift);all(>=(0),v) || throw(ArgumentError("shift coordinates must be nonnegative"))
    owned=BivariateInterval(I.generators,I.relations;field=I.field,budget)
    _shift_sweep(owned,v,budget,witnesses)
end
function check_shift_dimension(r::ShiftDimensionResult;budget::ContourBudget=ContourBudget(),throw=false)
    _validation(;throw) do
        I=BivariateInterval(r.interval.generators,r.interval.relations;field=r.interval.field,budget)
        all(>=(0),r.shift) && r.value>=0 || error("invalid shift or value")
        if r.witnesses===nothing
            _shift_sweep(I,r.shift,budget,false).value==r.value || error("incorrect shift dimension")
        else
            w=r.witnesses;g=I.generators;n=length(g);ids=w.indices;pack=w.packing_indices
            length(ids)==length(pack)==length(w.generators)==length(w.packing_points)==r.value || error("certificate size mismatch")
            all(i->1<=i<=n,ids) && all(i->1<=i<=n,pack) && allunique(ids) && allunique(pack) || error("invalid generator indices")
            w.generators==g[ids] && w.packing_points==[g[i].+r.shift for i in pack] || error("certificate coordinates mismatch")
            alive=_survivors(I,r.shift)
            _limit(big(n)*r.value+big(r.value)^2*n,budget.max_work,"certificate checks")
            all(i->!alive[i] || any(j->_precedes(g[j],g[i].+r.shift),ids),1:n) || error("selected generators do not contain the shifted image")
            all(i->alive[i],pack) || error("packing contains a killed demand")
            for a in 1:length(pack),b in a+1:length(pack)
                p=min.(g[pack[a]].+r.shift,g[pack[b]].+r.shift)
                any(x->_precedes(x,p),g) && error("two packed demands share a possible generator")
            end
        end
        true
    end
end
