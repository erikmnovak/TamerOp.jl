# Anchor-generated subspaces retain exact images, so slope queries do not build
# a module or a resolution for every candidate. Geometry supplies actual volume.
struct SkyscraperResult{M,K,C,W}
    source::M
    anchor::Int
    weights::Vector{QQ}
    slopes::Vector{QQ}
    source_bases::Vector{Matrix{K}}
    term_dimensions::Vector{Vector{Int}}
    witnesses::W
    context::C
    statistics::NamedTuple
end
"""A bounded, zero-extended half-open Cartesian encoding. Each axis supplies
all cell boundaries, including the terminal bound. The first axis varies
fastest. `density` is a positive scalar or one positive rational value per cell.
The default is Lebesgue volume; singleton strata are not silently given
positive volume. The module must use the matching positive product order.
"""
struct SkyscraperGrid{N,M}
    finite::M
    axes::NTuple{N,Vector{QQ}}
    shape::NTuple{N,Int}
    density::Vector{QQ}
    function SkyscraperGrid(M::PModule,axes;density=1,budget::HNBudget=HNBudget())
        _input(M,budget)
        xs=map(x->_rational.(collect(x)),Tuple(axes))
        !isempty(xs) && all(x->length(x)>=2 && issorted(x) && allunique(x),xs) ||
            throw(ArgumentError("supply strictly increasing finite cell boundaries on every axis"))
        shape=map(x->length(x)-1,xs)
        prod(big.(shape))==nvertices(M.Q) || throw(DimensionMismatch("cell counts do not match module vertices"))
        Q=ProductOfChainsPoset(shape)
        Set(cover_edges(Q))==Set(cover_edges(M.Q)) || throw(ArgumentError("the finite module must have the declared positive Cartesian order"))
        source=_snapshot(M)
        new{length(xs),typeof(source)}(source,xs,shape,_density(density,length(M.dims)))
    end
end
describe(g::SkyscraperGrid)=(;kind=:skyscraper_grid,shape=g.shape,axes=map(copy,g.axes),
    extension=:zero,cells=:left_closed_right_open,measure=_measure(g),density=copy(g.density),field=g.finite.field)
Base.show(io::IO,g::SkyscraperGrid)=print(io,"SkyscraperGrid(",g.shape,", half-open, zero extension)")

function _anchor_candidates(F,d,maps,b,w)
    d>2 && F isa PrimeField && return _subspaces(F,d,b,w)
    d<=2 || throw(ArgumentError("the complete kernel-family route requires anchor dimension at most two"))
    K=coeff_type(F);spaces=Matrix{K}[zeros(K,d,0),eye(F,d)]
    if d==2
        for A in maps
            B=Matrix{K}(FL.nullspace(F,A))
            size(B,2)==1 || continue
            p=findfirst(!iszero,B[:,1]);B ./= B[p,1]
            any(C->C==B,spaces) || push!(spaces,B)
        end
    end
    d==0 && pop!(spaces)
    _bound(w.subspaces+length(spaces),b.max_subspaces,"anchor subspaces")
    w.subspaces+=length(spaces)
    spaces
end

struct _AnchorData{K}
    maps::Vector{Matrix{K}}
    spaces::Vector{Matrix{K}}
    dimensions::Vector{Vector{Int}}
end
function _anchor_maps(M::PModule{K},anchor,b) where K
    n=length(M.dims);d=M.dims[anchor]
    _bound(sum(big(v)*d for v in M.dims;init=big(0)),b.max_entries,"anchor maps")
    [leq(M.Q,anchor,v) ? Matrix{K}(map_leq(M,anchor,v)) : zeros(K,M.dims[v],d) for v in 1:n]
end
function _anchor_data(M::PModule{K},anchor,b,w) where K
    n=length(M.dims);d=M.dims[anchor];maps=_anchor_maps(M,anchor,b)
    spaces=_anchor_candidates(M.field,d,maps,b,w)
    _bound(big(length(spaces))*n,b.max_entries,"candidate dimension table")
    dims=Vector{Int}[]
    for U in spaces
        push!(dims,[_rank(M.field,A*U) for A in maps])
    end
    _AnchorData(maps,spaces,dims)
end

# Anchors of dimension at most two have a complete kernel family over any
# field: a line outside every rank-one kernel cannot beat the full stalk.
# Small finite anchors
# have inexpensive complete subspace tables; larger ones use the same general
# optimizer as HN, without retaining unrequested full filtration witnesses.
function _small_anchor(F,d)
    d<=2 || (F isa PrimeField && d==3 && F.p<=3)
end
function _anchor_query_data(M::PModule{K},a,weights,b,w) where K
    if _small_anchor(M.field,M.dims[a])
        data=_anchor_data(M,a,b,w)
        return data,_anchor_flag(M.field,data,weights,a,b,w),:anchor_subspaces
    end
    data,inverse_slopes=_general_anchor_data(M,a,weights,b,w)
    all(>(0),inverse_slopes) || throw(ArgumentError("a nonzero generated quotient has zero geometric mass; choose a positive measure or an anchor with positive-volume persistence"))
    data,(collect(2:length(data.spaces)),inv.(inverse_slopes)),:certified_matrix_space
end
function _general_anchor_data(M::PModule{K},a,weights,b,w;maps=_anchor_maps(M,a,b)) where K
    B=[_basis(M.field,A) for A in maps]
    G,_=_subobject(M,B)
    # Inverse slopes also make sense at closed cell corners, where some
    # geometric masses vanish. No division by a zero geometric mass occurs.
    C=G;projection=eye(M.field,G.dims[a]);bases=Matrix{K}[]
    inverse_slopes=QQ[];dims=Vector{Int}[];retained=big(0)
    while C.dims[a]>0
        N=C;inclusions=[eye(M.field,d) for d in C.dims]
        while true
            mass=sum(weights.*N.dims);degree=N.dims[a]
            theta=-degree.*weights;theta[a]+=mass
            U,defect=_maximum_weight_submodule(N,theta,b,w)
            if iszero(defect)
                push!(inverse_slopes,mass/degree);break
            end
            T,_=_subobject(N,U)
            0<T.dims[a]<N.dims[a] || error("anchor optimizer did not decrease dimension")
            inclusions=[inclusions[v]*U[v] for v in eachindex(U)];N=T
        end
        T,i=_subobject(C,inclusions);D,q=AC.quotient_with_projection(i)
        current=G.dims-C.dims+T.dims
        retained+=big(G.dims[a])*current[a]+length(current)
        _bound(retained,b.max_entries,"anchor filtration entries")
        U=Matrix{K}(FL.nullspace(M.field,transpose(FL.nullspace(M.field,transpose(inclusions[a])))*projection))
        push!(bases,U);push!(dims,current)
        projection=q.comps[a]*projection;C=D
    end
    spaces=Matrix{K}[zeros(K,M.dims[a],0)]
    append!(spaces,[B[a]*U for U in bases])
    data=_AnchorData(maps,spaces,vcat([zeros(Int,length(weights))],dims))
    data,inverse_slopes
end

function _anchor_flag(F,data,weights,anchor,b,w)
    dims=data.dimensions;spaces=data.spaces
    current=findfirst(A->size(A,2)==0,spaces)
    slopes=QQ[];chosen=Int[];d=size(first(data.maps),2)
    while size(spaces[current],2)<d
        best=0;bs=zero(QQ);bestd=0
        for j in eachindex(spaces)
            _tick!(w,b);k=size(spaces[j],2)-size(spaces[current],2)
            k>0 && _contains(F,spaces[j],spaces[current]) || continue
            mass=sum((weights[v]*(dims[j][v]-dims[current][v]) for v in eachindex(weights));init=zero(QQ))
            mass>0 || throw(ArgumentError("a nonzero generated quotient has zero geometric mass; choose a positive measure or an anchor with positive-volume persistence"))
            s=k/mass
            if best==0 || s>bs || (s==bs && k>bestd)
                best=j;bs=s;bestd=k
            end
        end
        best>0 || error("incomplete anchor candidate family")
        push!(chosen,best);push!(slopes,bs);current=best
    end
    chosen,slopes
end

function _anchor_result(M::PModule{K},anchor,weights,data,b,w,context,witnesses;
        flag=_anchor_flag(M.field,data,weights,anchor,b,w)) where K
    ids,slopes=flag
    bases=copy.(data.spaces[ids]);dims=copy.(data.dimensions[ids])
    retained=sum(big(length(U)) for U in bases;init=big(0));lastdims=zeros(Int,length(M.dims))
    for d in dims
        retained+=_filtration_entries(M,d,lastdims;witnesses);lastdims=d
    end
    _bound(retained,b.max_entries,"retained filtration entries")
    terms=PModule{K}[];inclusions=PMorphism{K}[];factors=PModule{K}[];factor_projections=PMorphism{K}[]
    if witnesses
        previous=[zeros(K,d,0) for d in M.dims]
        for U in bases
            B=[_basis(M.field,A*U) for A in data.maps]
            N,j=_subobject(M,B)
            C=[Matrix{K}(FL.solve_fullcolumn(M.field,B[v],previous[v])) for v in eachindex(B)]
            S,i=_subobject(N,C);D,q=AC.quotient_with_projection(i)
            push!(terms,N);push!(inclusions,j);push!(factors,D);push!(factor_projections,q)
            previous=B
        end
    end
    maps=witnesses ? (;terms,inclusions,factors,factor_projections) : nothing
    SkyscraperResult(M,anchor,copy(weights),slopes,bases,dims,maps,context,
        _statistics(w))
end
function _vertex(M,v)
    v isa Integer && !(v isa Bool) && 1<=v<=length(M.dims) || throw(ArgumentError("expected a valid finite-poset vertex"))
    Int(v)
end

"""Prepare exact skyscraper threshold queries at one finite-poset anchor.
The default mass is counting measure. Explicit positive weights define another
central charge. Higher thresholds retain HIGHER slopes (smaller average mass).
Small anchors use complete subspace/kernel families. Arbitrary thickness over
QQ and prime fields uses certified matrix-space optimization, with explicit
workspace and search budgets. No finite-field result is relabelled as rational.
"""
function skyscraper(M::PModule;anchor,weights=ones(Int,length(M.dims)),
        witnesses::Bool=false,budget::HNBudget=HNBudget())
    _input(M,budget);a=_vertex(M,anchor)
    weights=_rational.(collect(weights))
    length(weights)==length(M.dims) && all(>(0),weights) || throw(ArgumentError("supply one positive mass weight per vertex"))
    M=_snapshot(M);w=_Work(0,0);data,flag,route=_anchor_query_data(M,a,weights,budget,w)
    context=(;category=:finite_poset,measure=:declared_stalk_weights,anchor=a,route)
    _anchor_result(M,a,weights,data,budget,w,context,witnesses;flag)
end

function _grid_label(g,x)
    x isa Union{Tuple,AbstractVector} && length(x)==length(g.shape) || throw(ArgumentError("wrong parameter dimension"))
    p=Tuple(_rational.(x));index=Int[]
    for j in eachindex(p)
        g.axes[j][1]<=p[j]<g.axes[j][end] || return 0,p
        push!(index,searchsortedlast(g.axes[j],p[j]))
    end
    LinearIndices(g.shape)[index...],p
end
function _grid_weights(g,p)
    QQ[g.density[k]*prod(max(zero(QQ),g.axes[j][I[j]+1]-max(g.axes[j][I[j]],p[j])) for j in eachindex(g.shape))
        for (k,I) in enumerate(CartesianIndices(g.shape))][:]
end
"""Compute at an exact ambient anchor in a declared half-open grid encoding.
Mass is integrated over the generated submodule's actual clipped cells, not
counted from finite labels. This preserves geometric weighting on refinement.
"""
function skyscraper(g::SkyscraperGrid;anchor,witnesses::Bool=false,budget::HNBudget=HNBudget())
    _input(g.finite,budget);a,p=_grid_label(g,anchor)
    a>0 || throw(ArgumentError("anchor is outside the declared half-open support box"))
    M=_snapshot(g.finite);w=_Work(0,0);weights=_grid_weights(g,p)
    data,flag,route=_anchor_query_data(M,a,weights,budget,w)
    context=(;category=:bounded_cartesian_encoding,measure=_measure(g),density=copy(g.density),anchor=p,
        axes=map(copy,g.axes),shape=g.shape,extension=:zero,cells=:left_closed_right_open,route)
    _anchor_result(M,a,weights,data,budget,w,context,witnesses;flag)
end
skyscraper_slopes(r::SkyscraperResult)=copy(r.slopes)
skyscraper_dimensions(r::SkyscraperResult)=copy.(r.term_dimensions)
function skyscraper_maps(r::SkyscraperResult)
    r.witnesses===nothing && throw(ArgumentError("request witnesses=true to retain the actual filtration maps"))
    r.witnesses
end
function skyscraper_rank(r::SkyscraperResult;target,threshold=0)
    v=_vertex(r.source,target);t=_rational(threshold)
    leq(r.source.Q,r.anchor,v) || throw(ArgumentError("target is incomparable with or precedes the anchor"))
    i=findlast(>=(t),r.slopes)
    i===nothing ? 0 : r.term_dimensions[i][v]
end
function skyscraper_rank(g::SkyscraperGrid;source,target,threshold=0,budget::HNBudget=HNBudget())
    a,p=_grid_label(g,source);b,q=_grid_label(g,target)
    all(p .<= q) || throw(ArgumentError("source must precede target coordinatewise"))
    _rational(threshold)
    (a==0 || b==0) && return 0
    skyscraper_rank(skyscraper(g;anchor=p,budget);target=b,threshold)
end
function skyscraper_summary(r::SkyscraperResult)
    (;kind=:skyscraper,field=r.source.field,anchor=r.context.anchor,finite_anchor=r.anchor,
      slopes=skyscraper_slopes(r),term_dimensions=skyscraper_dimensions(r),
      measure=r.context.measure,context=r.context,exact=true,witnesses=r.witnesses!==nothing,
      statistics=r.statistics,threshold_convention=:slope_greater_or_equal)
end
describe(r::SkyscraperResult)=skyscraper_summary(r)
field(r::SkyscraperResult)=r.source.field
provenance(r::SkyscraperResult)=(;r.context...,field=r.source.field,exact=true,
    weights=copy(r.weights),threshold_convention=:slope_greater_or_equal)
Base.show(io::IO,r::SkyscraperResult)=print(io,"SkyscraperResult(anchor=",r.context.anchor,", ",length(r.slopes)," factors)")

function check_skyscraper(r::SkyscraperResult;semistability::Bool=false,
        budget::HNBudget=HNBudget(),throw::Bool=false)
    issues=String[];checked=false
    try
        M=r.source;_input(M,budget);a=_vertex(M,r.anchor);n=length(M.dims);K=coeff_type(M.field)
        length(r.weights)==n && all(>=(0),r.weights) || error("invalid mass weights")
        length(r.slopes)==length(r.source_bases)==length(r.term_dimensions) || error("factor count mismatch")
        previous=zeros(K,M.dims[a],0);lastdims=zeros(Int,n)
        for i in eachindex(r.slopes)
            U=r.source_bases[i]
            size(U,1)==M.dims[a] && _rank(M.field,U)==size(U,2) || error("invalid anchor basis")
            size(U,2)>size(previous,2) && _contains(M.field,U,previous) || error("anchor flag is not strictly nested")
            dims=[leq(M.Q,a,v) ? _rank(M.field,map_leq(M,a,v)*U) : 0 for v in 1:n]
            dims==r.term_dimensions[i] || error("incorrect generated dimensions")
            slope=(size(U,2)-size(previous,2))/sum(r.weights.*(dims-lastdims))
            slope==r.slopes[i] || error("incorrect slope")
            i==1 || r.slopes[i-1]>slope || error("slopes must strictly decrease")
            if r.witnesses!==nothing
                j=r.witnesses.inclusions[i]
                j.cod===M && j.dom===r.witnesses.terms[i] && j.dom.dims==dims || error("filtration inclusion endpoints disagree")
                check_morphism(j;throw=true)
                for v in 1:n
                    image=leq(M.Q,a,v) ? map_leq(M,a,v)*U : zeros(K,M.dims[v],0)
                    _rank(M.field,j.comps[v])==dims[v] && _contains(M.field,j.comps[v],image) || error("incorrect generated inclusion")
                end
                factor=r.witnesses.factors[i];q=r.witnesses.factor_projections[i]
                factor.dims==dims-lastdims && q.dom===j.dom && q.cod===factor || error("factor projection endpoints disagree")
                check_morphism(q;throw=true)
                all(_rank(M.field,q.comps[v])==factor.dims[v] for v in 1:n) || error("factor projection is not epic")
                if i>1
                    prev=r.witnesses.inclusions[i-1]
                    for v in 1:n
                        coordinates=FL.solve_fullcolumn(M.field,j.comps[v],prev.comps[v])
                        iszero(q.comps[v]*coordinates) || error("factor projection does not annihilate the previous term")
                    end
                end
            end
            previous=U;lastdims=dims
        end
        size(previous,2)==M.dims[a] || error("filtration does not exhaust the anchor stalk")
        if semistability
            w=_Work(0,0);data,(ids,slopes),_=_anchor_query_data(M,a,r.weights,budget,w)
            slopes==r.slopes && data.dimensions[ids]==r.term_dimensions || error("filtration is not HN")
            for (U,i) in zip(r.source_bases,ids)
                _contains(M.field,U,data.spaces[i]) || error("incorrect HN anchor subspace")
            end
            checked=true
        end
    catch err
        err isa InterruptException && rethrow()
        push!(issues,sprint(showerror,err))
    end
    throw && !isempty(issues) && Base.throw(ArgumentError(join(issues,"; ")))
    HNValidationSummary(isempty(issues),issues,checked)
end
