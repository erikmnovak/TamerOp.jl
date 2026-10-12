# Shared preparation for ingestion and source previews. The cache owns at most
# four graphs, with structural coordinate snapshots rather than hash-only keys.
# Consumers treat the edge/distance arrays as read-only.
function _prepared_point_graph(points, kind::Symbol, parameter;
        backend::Symbol=:auto, approx_candidates::Int=0, cache=nothing,
        max_distance_pairs::Int=typemax(Int), max_graph_edges::Int=typemax(Int))
    n = points isa AbstractMatrix ? size(points,1) : length(points)
    d = points isa AbstractMatrix ? size(points,2) : isempty(points) ? 0 : length(first(points))
    resolved = _resolve_pointcloud_runtime_backend(backend,n,d,kind===:knn ? 1 : 2)
    kind in (:knn,:radius) || throw(ArgumentError("Unknown point graph kind."))
    request = (kind,parameter,resolved,approx_candidates)
    cache===nothing || Base.lock(cache.lock)
    try
        entries = cache===nothing ? nothing : get!(cache.geometry,(:prepared_point_graphs,)) do
            GeometryCachePayload(Pair{Any,Any}[])
        end
        if entries!==nothing
            for (old,graph) in entries.value
                if isequal(old[2],request) && isequal(old[1],points)
                    length(graph.edges)<=max_graph_edges || throw(ArgumentError("Prepared graph exceeds max_graph_edges."))
                    graph.backend===:none || _record_ingestion_backend(kind===:knn ? :knn_graph : :radius_graph,graph.backend)
                    return (;graph...,reused=true)
                end
            end
        end
        pairs = kind===:knn ? big(n)*max(n-1,0) : div(big(n)*max(n-1,0),2)
        resolved===:bruteforce && pairs>max_distance_pairs &&
            throw(ArgumentError("Preview needs up to $pairs distance pairs; increase max_distance_pairs, activate NearestNeighbors, or supply a retained graph."))
        upper = kind===:knn ? big(n)*min(Int(parameter),max(n-1,0)) : pairs
        kind===:knn && upper>max_graph_edges && throw(ArgumentError("Graph edge upper bound $upper exceeds max_graph_edges; raise the preparation budget explicitly or visualize retained GraphData/FilteredComplex."))
        trace=_IngestionExecutionTrace()
        edges,dists,kdist = with(_INGESTION_EXECUTION_TRACE=>trace) do
            if n<=1
                (NTuple{2,Int}[],Float64[],Float64[])
            elseif kind===:knn
                _point_cloud_knn_graph(points,Int(parameter);backend=resolved,approx_candidates)
            else
                es,ds=_point_cloud_radius_graph(points,Float64(parameter);backend=resolved,approx_candidates,max_edges=max_graph_edges)
                (es,ds,Float64[])
            end
        end
        effective=isempty(trace.backends) ? :none : last(only(trace.backends))
        effective===:none || _record_ingestion_backend(kind===:knn ? :knn_graph : :radius_graph,effective)
        graph=(;edges,dists,kdist,backend=effective)
        if entries!==nothing
            length(entries.value)>=4 && popfirst!(entries.value)
            snapshot=points isa AbstractMatrix ? copy(points) : deepcopy(points)
            push!(entries.value,(snapshot,request)=>graph)
        end
        return (;graph...,reused=false)
    finally
        cache===nothing || Base.unlock(cache.lock)
    end
end
