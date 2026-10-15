# Certified bisection brackets for diagonal filtered landscapes. The query
# remains exact; only the reported landscape radius is enclosed to tolerance.
struct FilteredLandscapeResult{N,C}
    points::Vector{NTuple{N,QQ}}
    thresholds::Vector{QQ}
    layers::Vector{Int}
    lower::Array{QQ,3}
    upper::Array{QQ,3}
    context::C
end
"""Supply a bounded landscape array with explicit radius enclosures and
provenance. Axes are (point, threshold, layer). Supplied arrays are structurally
checked and labelled supplied; their mathematical source is not certified.
"""
function FilteredLandscapeResult(points,thresholds,layers,lower,upper;
        provenance=(;source=:supplied))
    ps=collect(points);isempty(ps) && throw(ArgumentError("supply at least one landscape point"))
    N=length(first(ps));N>0 || throw(ArgumentError("landscape points must have positive dimension"))
    all(p->length(p)==N,ps) || throw(DimensionMismatch("landscape point dimensions differ"))
    xs=NTuple{N,QQ}[Tuple(_rational.(p)) for p in ps]
    ts=_rational.(collect(thresholds));ks=collect(layers)
    !isempty(ts) && issorted(ts) && allunique(ts) || throw(ArgumentError("thresholds must be nonempty and strictly increasing"))
    !isempty(ks) && all(k->k isa Integer && !(k isa Bool) && 0<k<=typemax(Int),ks) && issorted(ks) && allunique(ks) ||
        throw(ArgumentError("layers must be nonempty, positive, strictly increasing integers"))
    shape=(length(xs),length(ts),length(ks))
    size(lower)==size(upper)==shape || throw(DimensionMismatch("landscape arrays need axes (point, threshold, layer)"))
    lo=_rational.(lower);hi=_rational.(upper)
    all(0 .<= lo) && all(lo .<= hi) || throw(ArgumentError("invalid nonnegative radius enclosures"))
    prefix=copy(hi)
    for i in axes(lo,1),j in axes(lo,2),k in axes(lo,3)
        j==1 || (prefix[i,j,k]=min(prefix[i,j,k],prefix[i,j-1,k]))
        k==1 || (prefix[i,j,k]=min(prefix[i,j,k],prefix[i,j,k-1]))
        lo[i,j,k]<=prefix[i,j,k] || throw(ArgumentError("threshold/layer enclosures contradict monotonicity"))
    end
    FilteredLandscapeResult(xs,ts,Int.(ks),lo,hi,(;scope=:supplied,provenance))
end

"""Enclose sup{r >= 0: s^threshold(point-r*1,point+r*1) >= layer}.
Bounds use exact skyscraper queries and monotonicity, with width at most atol.
The support box, volume, zero extension and binary-float interpretation are
those of SkyscraperGrid. Budget exhaustion throws instead of returning an
unqualified approximation. No statistical uncertainty is implied.
A prepared planar chamber result can replace the grid to reuse certified flags
throughout bisection, avoiding repeated matrix search and weight replication.
Its selected cells must cover every queried source point.
"""
filtered_landscape(g::Union{SkyscraperGrid,SkyscraperEncoding};kwargs...)=_filtered_landscape(g,nothing;kwargs...)
filtered_landscape(r::SkyscraperChambers;kwargs...)=_filtered_landscape(r.grid,r;kwargs...)
function _filtered_landscape(g::Union{SkyscraperGrid{N},SkyscraperEncoding{N}},prepared;points,thresholds=[0],layers=[1],
        atol=1//1024,max_steps=128,budget::HNBudget=HNBudget()) where N
    tolerance=_rational(atol);tolerance>0 || throw(ArgumentError("atol must be positive"))
    max_steps isa Integer && !(max_steps isa Bool) && 0<max_steps<=typemax(Int) || throw(ArgumentError("max_steps must be positive"))
    ps=collect(points);ts=collect(thresholds);ks=collect(layers)
    _bound(big(length(ps))*length(ts)*length(ks)*2,budget.max_entries,"landscape output")
    template=FilteredLandscapeResult(ps,ts,ks,zeros(Int,length(ps),length(ts),length(ks)),zeros(Int,length(ps),length(ts),length(ks)))
    length(first(template.points))==N || throw(DimensionMismatch("landscape point dimension differs from grid"))
    lower=copy(template.lower);upper=copy(template.upper);calls=0
    boxlo,boxhi=_support_box(g)
    for (i,p) in enumerate(template.points)
        label,_=_grid_label(g,p);label==0 && continue
        cap=minimum(min(p[j]-boxlo[j],boxhi[j]-p[j]) for j in 1:N)
        cache=Dict{QQ,Vector{Int}}()
        function ranks(radius)
            get!(cache,radius) do
                calls+=1;_bound(calls,budget.max_states,"landscape queries")
                source=ntuple(j->p[j]-radius,N);target=ntuple(j->p[j]+radius,N)
                a,_=_grid_label(g,source);b,_=_grid_label(g,target)
                (a==0 || b==0) && return zeros(Int,length(template.thresholds))
                r=prepared===nothing ? skyscraper(g;anchor=source,budget) :
                    skyscraper_chamber_at(prepared;anchor=source,budget).filtration
                [skyscraper_rank(r;target=b,threshold=t) for t in template.thresholds]
            end
        end
        for (j,t) in enumerate(template.thresholds),(k,layer) in enumerate(template.layers)
            ranks(QQ(0))[j]<layer && continue
            if ranks(cap)[j]>=layer
                lower[i,j,k]=upper[i,j,k]=cap;continue
            end
            lo=QQ(0);hi=cap;steps=0
            while hi-lo>tolerance
                steps+=1;steps<=max_steps || throw(ArgumentError("landscape bisection max_steps exceeded"))
                mid=(lo+hi)/2
                if ranks(mid)[j]>=layer;lo=mid;else;hi=mid;end
            end
            lower[i,j,k]=lo;upper[i,j,k]=hi
        end
    end
    FilteredLandscapeResult(template.points,template.thresholds,template.layers,lower,upper,
        (;scope=:computed,field=g.finite.field,box=(boxlo,boxhi),measure=_measure(g),density=copy(g.density),
          extension=:zero,definition=:symmetric_diagonal_radius,tolerance,queries=calls,
          query_route=prepared===nothing ? :selected_anchor : :prepared_chambers))
end
filtered_landscape_values(r::FilteredLandscapeResult)=(;lower=copy(r.lower),upper=copy(r.upper))
function filtered_landscape_difference(r::FilteredLandscapeResult;threshold,baseline)
    a=findfirst(==(_rational(threshold)),r.thresholds);b=findfirst(==(_rational(baseline)),r.thresholds)
    (a===nothing || b===nothing) && throw(ArgumentError("both thresholds must be retained"))
    (;lower=a==b ? zeros(QQ,length(r.points),length(r.layers)) : r.lower[:,a,:]-r.upper[:,b,:],
      upper=a==b ? zeros(QQ,length(r.points),length(r.layers)) : r.upper[:,a,:]-r.lower[:,b,:],
      definition=:threshold_minus_baseline,threshold=r.thresholds[a],baseline=r.thresholds[b])
end
describe(r::FilteredLandscapeResult)=(;kind=:filtered_landscape,points=length(r.points),
    thresholds=copy(r.thresholds),layers=copy(r.layers),context=r.context,
    maximum_width=maximum(r.upper-r.lower;init=zero(QQ)))
Base.show(io::IO,r::FilteredLandscapeResult)=print(io,"FilteredLandscapeResult(",length(r.points)," points, ",length(r.thresholds)," thresholds, ",r.context.scope,")")
