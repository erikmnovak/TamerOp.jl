# Bounded source preparation shared by static previews and exports.
const _SOURCE_BUDGET_KEYS = (:max_points,:max_edges,:max_labels)
const _SOURCE_GRAPH_KEYS = (:nn_backend,:max_distance_pairs,:max_graph_edges)
const _SourcePreviewInput = Union{DataTypes.PointCloud,DataTypes.GraphData,
    DataTypes.EmbeddedPlanarGraph2D,DataIngestion.PointCodensityResult}

"""
    visual_spec(source; kind=:auto, cache=:auto, max_points=5000,
                max_edges=10000, max_labels=50, kwargs...)

Prepare a bounded point-cloud or supplied-graph preview without rendering.
Point indices are spread deterministically through source order; edges retain
source order and require both displayed endpoints. Metadata retains original
IDs, full counts and omissions. Limits affect drawing, not ingestion or topology.
Three-dimensional previews support point/edge limits but no text labels.

For `kind=:knn_graph` or `:radius_graph`, `nn_backend=:auto` chooses an activated
exact NearestNeighbors backend, otherwise brute force. These graphs are metric
previews, not certificates of the graph used by an unrelated filtration. Use
retained `GraphData` or `FilteredComplex` for the supplied construction.
`max_distance_pairs=10_000_000` bounds brute-force distance work;
`max_graph_edges=1_000_000` bounds radius output and the kNN candidate upper bound.
Budgets are nonnegative Ints; an exceeded preparation budget raises an error.

Use `cache=sc::SessionCache` to reuse compatible sparse-ingestion graphs and
coordinate projections. `cache=:auto` and `cache=nothing` retain no preparation
between calls. Coordinate contents, graph settings and resolved backend identify
prepared graphs; in-place coordinate edits invalidate matching entries. Each
session retains at most four neighbor graphs and four projections. Metadata's
`graph_reused` / `geometry_reused` report actual reuse. Do not mutate source data
concurrently with a call; returned display arrays are independent of cache data.

```julia
pc = TamerOp.PointCloud([0.0 0.0; 1.0 2.0; 4.0 3.0])
sc = TamerOp.SessionCache()
spec = TamerOp.Advanced.visual_spec(pc; kind=:knn_graph, k=1, cache=sc)
TamerOp.Advanced.visual_metadata(spec).construction
```
"""
function visual_spec(obj::_SourcePreviewInput;kind::Symbol=:auto,cache=:auto,kwargs...)
    haskey(kwargs,:backend) && throw(ArgumentError("backend belongs to visualize/render, not visual_spec"))
    report=check_visual_request(obj;kind,kwargs...,throw=true)
    # One-shot views retain no hidden global geometry. Explicit sessions share
    # ingestion's bounded graph cache and this owner's projection preparation.
    sc=cache===:auto ? nothing : CoreModules._resolve_workflow_session_cache(cache)
    prepared=CoreModules._workflow_encoding_cache(sc)
    return _visual_spec(obj,report.requested_kind;_preview_cache=prepared,kwargs...)
end

function _source_budget_check(;max_points=5000,max_edges=10000,max_labels=50,
        max_distance_pairs=10_000_000,max_graph_edges=1_000_000)
    for (name,v) in pairs((;max_points,max_edges,max_labels,max_distance_pairs,max_graph_edges))
        v isa Int && v>=0 || throw(ArgumentError("$name must be a nonnegative Int."))
    end
    return nothing
end

function _source_ids(n,limit)
    m=min(n,limit)
    m==0 && return Int[]
    m==n && return collect(1:n)
    # Deterministic spread in original source order, including both ends.
    m==1 && return [1]
    return [1+div((i-1)*Int128(n-1),m-1) for i in 1:m] .|> Int
end

function _source_projection(A,target;dims=nothing,cache=nothing)
    all(x->isfinite(x) && isfinite(Float64(x)),A) ||
        throw(ArgumentError("Source coordinates must be finite and representable as Float64."))
    cache===nothing && return (_project_coords(A,target;dims)...,false)
    request=(target,dims===nothing ? nothing : Tuple(dims))
    lock(cache.lock)
    try
        bucket=get!(cache.geometry,(:source_preview_projections,)) do
            CoreModules.GeometryCachePayload(Pair{Any,Any}[])
        end
        for (old,value) in bucket.value
            isequal(old[2],request) && isequal(old[1],A) && return (value...,true)
        end
        value=_project_coords(A,target;dims)
        length(bucket.value)>=4 && popfirst!(bucket.value)
        push!(bucket.value,(copy(A),request)=>value)
        return (value...,false)
    finally
        unlock(cache.lock)
    end
end

function _source_edge_ids(edges,n,point_ids,max_edges;trusted=false)
    # Computed neighbor graphs already certify their endpoints. With every
    # source point displayed, their eligible count needs no full edge scan.
    if trusted && length(point_ids)==n
        return collect(1:min(length(edges),max_edges)),length(edges)
    end
    keep=falses(n);keep[point_ids].=true
    ids=Int[]
    sizehint!(ids,min(length(edges),max_edges))
    eligible=0
    for (i,(u,v)) in enumerate(edges)
        1<=u<=n && 1<=v<=n || throw(ArgumentError("Source edge $i has an endpoint outside 1:$n."))
        if keep[u] && keep[v]
            eligible+=1
            length(ids)<max_edges && push!(ids,i)
        end
    end
    return ids,eligible
end

function _source_labels!(layers,pts,labels,ids,max_labels)
    labels===nothing && return Int[]
    chosen=ids[1:min(length(ids),max_labels)]
    length(chosen)==0 && return chosen
    push!(layers,TextLayer([string(labels[i]) for i in chosen],
        [(pts[i,1]+0.02,pts[i,2]+0.02) for i in chosen],:black,9.0))
    return chosen
end

function _source_counts(n,m,ids,eids,lids;eligible=m)
    return (;point_ids=ids,edge_ids=eids,label_ids=lids,total_points=n,total_edges=m,
        displayed_points=length(ids),displayed_edges=length(eids),displayed_labels=length(lids),
        omitted_points=n-length(ids),omitted_edges=m-length(eids),eligible_edges=eligible,
        display_selection=:source_order_spread_points_then_source_order_edges,
        budget_semantics=:display_only)
end
_source_caption(c)="Showing $(c.displayed_points)/$(c.total_points) points and $(c.displayed_edges)/$(c.total_edges) edges; $(c.displayed_labels) labels"

function _source_points!(layers,pts,ids;colors=nothing,basecolor=:dodgerblue3,markersize=9.0,alpha=0.9)
    color=colors===nothing ? basecolor : Float64[colors[i] for i in ids]
    if size(pts,2)==3
        push!(layers,Point3Layer(NTuple{3,Float64}[(pts[i,1],pts[i,2],pts[i,3]) for i in ids],color,alpha,markersize,:viridis))
    else
        push!(layers,PointLayer(NTuple{2,Float64}[(pts[i,1],pts[i,2]) for i in ids],color,alpha,markersize,:viridis))
    end
end

# Integer/rational sources may carry more precision than a drawing coordinate.
_source_rounding(A,dims) = eltype(A)<:Union{Float16,Float32,Float64} ? false :
    any(A[i,d]!=Float64(A[i,d]) for i in axes(A,1),d in dims)
