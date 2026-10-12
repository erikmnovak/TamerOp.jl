# Data-ingestion and graded-complex visualization builders.

@inline _image_array(img::DataTypes.ImageNd) = getfield(img, :data)

function _variance_order(A::AbstractMatrix{<:Real})
    d = size(A, 2)
    n = size(A, 1)
    n == 0 && return collect(1:d)
    means = Vector{Float64}(undef, d)
    vars = Vector{Float64}(undef, d)
    @inbounds for j in 1:d
        s = 0.0
        for i in 1:n
            s += float(A[i, j])
        end
        mu = s / n
        means[j] = mu
        ss = 0.0
        for i in 1:n
            delta = float(A[i, j]) - mu
            ss += delta * delta
        end
        vars[j] = ss / max(n - 1, 1)
    end
    return sortperm(vars; rev=true)
end

function _project_coords(A::AbstractMatrix{<:Real}, target_dim::Int; dims=nothing)
    d = size(A, 2)
    d >= target_dim || throw(ArgumentError("this view requires at least $target_dim coordinate axes."))
    chosen = if dims === nothing
        if d <= target_dim
            collect(1:d)
        else
            order = _variance_order(A)
            sort(order[1:min(target_dim, length(order))])
        end
    else
        collect(Int, dims)
    end
    length(chosen) == target_dim || throw(ArgumentError("dims must select exactly $target_dim coordinate axes."))
    length(unique(chosen)) == target_dim || throw(ArgumentError("dims must select distinct coordinate axes."))
    all(j -> 1 <= j <= d, chosen) || throw(ArgumentError("dims must lie between 1 and $d."))
    P = Matrix{Float64}(undef, size(A, 1), length(chosen))
    @inbounds for j in eachindex(chosen), i in 1:size(A, 1)
        P[i, j] = float(A[i, chosen[j]])
    end
    return P, chosen
end

function _bbox_from_matrix(A::AbstractMatrix{<:Real})
    size(A, 1) == 0 && return nothing
    mins = [minimum(Float64.(A[:, j])) for j in 1:size(A, 2)]
    maxs = [maximum(Float64.(A[:, j])) for j in 1:size(A, 2)]
    return mins, maxs
end

function _pad_limits_1d(lims::Tuple{<:Real,<:Real}; frac::Float64=0.06, minpad::Float64=0.05)
    lo = float(lims[1])
    hi = float(lims[2])
    pad = max(minpad, frac * max(abs(hi - lo), 1.0))
    return (lo - pad, hi + pad)
end

function _axes_from_matrix_2d(A::AbstractMatrix{<:Real}; labels=("x1", "x2"))
    bbox = _bbox_from_matrix(A)
    bbox === nothing && return _default_axes_2d(xlabel=labels[1], ylabel=labels[2])
    return _default_axes_2d(xlabel=labels[1], ylabel=labels[2],
                            xlimits=_pad_limits_1d((bbox[1][1], bbox[2][1])),
                            ylimits=_pad_limits_1d((bbox[1][2], bbox[2][2])),
                            aspect=:equal)
end

function _axes_from_matrix_3d(A::AbstractMatrix{<:Real}; labels=("x1", "x2", "x3"))
    bbox = _bbox_from_matrix(A)
    bbox === nothing && return _default_axes_3d(xlabel=labels[1], ylabel=labels[2], zlabel=labels[3])
    return _default_axes_3d(xlabel=labels[1], ylabel=labels[2], zlabel=labels[3],
                            xlimits=_pad_limits_1d((bbox[1][1], bbox[2][1])),
                            ylimits=_pad_limits_1d((bbox[1][2], bbox[2][2])),
                            zlimits=_pad_limits_1d((bbox[1][3], bbox[2][3])),
                            aspect=:data)
end

function _density_heatmap(points::AbstractMatrix{<:Real}; nbins::Int=24)
    size(points, 2) == 2 || throw(ArgumentError("density overlay requires 2-dimensional points."))
    size(points, 1) == 0 && return HeatmapLayer([0.0, 1.0], [0.0, 1.0], zeros(2, 2), :magma, 0.65, "point density")
    xs = Float64.(points[:, 1])
    ys = Float64.(points[:, 2])
    xlo, xhi = minimum(xs), maximum(xs)
    ylo, yhi = minimum(ys), maximum(ys)
    xlo == xhi && ((xlo, xhi) = _pad_limits_1d((xlo, xhi)))
    ylo == yhi && ((ylo, yhi) = _pad_limits_1d((ylo, yhi)))
    xspan = max(xhi - xlo, 1e-9)
    yspan = max(yhi - ylo, 1e-9)
    vals = zeros(Float64, nbins, nbins)
    @inbounds for i in eachindex(xs)
        bx = clamp(1 + floor(Int, ((xs[i] - xlo) / xspan) * (nbins - 1)), 1, nbins)
        by = clamp(1 + floor(Int, ((ys[i] - ylo) / yspan) * (nbins - 1)), 1, nbins)
        vals[by, bx] += 1.0
    end
    xgrid = collect(range(xlo, xhi; length=nbins))
    ygrid = collect(range(ylo, yhi; length=nbins))
    return HeatmapLayer(xgrid, ygrid, vals, :magma, 0.65, "point density")
end

@inline function _dist2(A::AbstractMatrix{<:Real}, i::Int, j::Int)
    s = 0.0
    @inbounds for k in 1:size(A, 2)
        delta = float(A[i, k]) - float(A[j, k])
        s += delta * delta
    end
    return s
end

function _segments_from_edges_2d(points::AbstractMatrix{<:Real}, edges)
    segs = NTuple{4,Float64}[]
    @inbounds for (u, v) in edges
        push!(segs, (float(points[u, 1]), float(points[u, 2]),
                     float(points[v, 1]), float(points[v, 2])))
    end
    return segs
end

function _segments_from_edges_3d(points::AbstractMatrix{<:Real}, edges)
    segs = NTuple{6,Float64}[]
    @inbounds for (u, v) in edges
        push!(segs, (float(points[u, 1]), float(points[u, 2]), float(points[u, 3]),
                     float(points[v, 1]), float(points[v, 2]), float(points[v, 3])))
    end
    return segs
end

function _graph_layout_2d(data::DataTypes.GraphData; dims=nothing)
    coords = DataTypes.coord_matrix(data)
    if coords === nothing
        n = DataTypes.nvertices(data)
        pts = Matrix{Float64}(undef, n, 2)
        @inbounds for i in 1:n
            theta = 2 * pi * (i - 1) / max(n, 1)
            pts[i, 1] = cos(theta)
            pts[i, 2] = sin(theta)
        end
        return pts, [1, 2], :circular
    end
    P, chosen = _project_coords(coords, 2; dims=dims)
    return P, chosen, :embedded
end

function _weighted_edge_layers_2d(points::AbstractMatrix{<:Real}, edges, weights::AbstractVector{<:Real}; weight_range=nothing)
    layers = AbstractVisualizationLayer[]
    legend_entries = NamedTuple[]
    isempty(edges) && return layers, legend_entries
    wmin,wmax = weight_range===nothing ? extrema(Float64.(weights)) : Float64.(weight_range)
    span = max(wmax - wmin, 1e-9)
    palette = (:steelblue3, :darkorange2, :crimson)
    bins = [NTuple{4,Float64}[] for _ in 1:3]
    @inbounds for (edge, w) in zip(edges, weights)
        q = clamp(1 + floor(Int, 3 * (float(w) - wmin) / span), 1, 3)
        push!(bins[q], (float(points[edge[1], 1]), float(points[edge[1], 2]),
                        float(points[edge[2], 1]), float(points[edge[2], 2])))
    end
    for i in 1:3
        isempty(bins[i]) && continue
        push!(layers, SegmentLayer(bins[i], palette[i], 0.7, 1.0 + 0.8 * (i - 1)))
        lo = wmin + (i - 1) * span / 3
        hi = wmin + i * span / 3
        push!(legend_entries, (; label="$(round(lo; digits=3)) - $(round(hi; digits=3))", color=palette[i], style=:line))
    end
    return layers, legend_entries
end

function _weighted_edge_layers_3d(points::AbstractMatrix{<:Real}, edges, weights::AbstractVector{<:Real}; weight_range=nothing)
    layers = AbstractVisualizationLayer[]
    legend_entries = NamedTuple[]
    isempty(edges) && return layers, legend_entries
    wmin,wmax = weight_range===nothing ? extrema(Float64.(weights)) : Float64.(weight_range)
    span = max(wmax - wmin, 1e-9)
    palette = (:steelblue3, :darkorange2, :crimson)
    bins = [NTuple{6,Float64}[] for _ in 1:3]
    @inbounds for (edge, w) in zip(edges, weights)
        q = clamp(1 + floor(Int, 3 * (float(w) - wmin) / span), 1, 3)
        push!(bins[q], (float(points[edge[1], 1]), float(points[edge[1], 2]), float(points[edge[1], 3]),
                        float(points[edge[2], 1]), float(points[edge[2], 2]), float(points[edge[2], 3])))
    end
    for i in 1:3
        isempty(bins[i]) && continue
        push!(layers, Segment3Layer(bins[i], palette[i], 0.7, 1.0 + 0.8 * (i - 1)))
        lo = wmin + (i - 1) * span / 3
        hi = wmin + i * span / 3
        push!(legend_entries, (; label="$(round(lo; digits=3)) - $(round(hi; digits=3))", color=palette[i], style=:line))
    end
    return layers, legend_entries
end

function _text_panel_spec(kind::Symbol, title::AbstractString, lines::AbstractVector{<:AbstractString}; metadata::NamedTuple=NamedTuple())
    n = max(length(lines), 1)
    positions = [(0.02, float(n - i + 1)) for i in 1:n]
    labels = n == 0 ? [""] : String[string(line) for line in lines]
    return VisualizationSpec(kind;
                             title=title,
                             layers=AbstractVisualizationLayer[
                                 TextLayer(labels, positions, :black, 12.0),
                             ],
                             axes=_default_axes_2d(xlabel="", ylabel="",
                                                   xlimits=(0.0, 1.0),
                                                   ylimits=(0.0, float(n + 1)),
                                                   aspect=:auto),
                             metadata=(; metadata..., panel_style=:text_only))
end

function _bar_spec(kind::Symbol, title::AbstractString, labels::AbstractVector{<:AbstractString},
                   values::AbstractVector{<:Real}; ylabel::AbstractString="value",
                   fill_color::Symbol=:dodgerblue, metadata::NamedTuple=NamedTuple())
    n = length(values)
    n == 0 && return _text_panel_spec(kind, title, ["No values are available for this diagnostic."];
                                      metadata=(; metadata..., empty=true))
    rects = NTuple{4,Float64}[]
    positions = NTuple{2,Float64}[]
    text_labels = String[]
    ymax = max(maximum(Float64.(values)), 1.0)
    @inbounds for i in 1:n
        xlo = float(i) - 0.4
        xhi = float(i) + 0.4
        yhi = max(0.0, float(values[i]))
        push!(rects, (xlo, 0.0, xhi, yhi))
        push!(positions, (float(i), yhi + 0.03 * ymax))
        push!(text_labels, string(round(float(values[i]); digits=3)))
    end
    return VisualizationSpec(kind;
                             title=title,
                             layers=AbstractVisualizationLayer[
                                 RectLayer(rects, fill_color, :black, 0.45, 1.0),
                                 TextLayer(text_labels, positions, :black, 10.0),
                             ],
                             axes=_default_axes_2d(xlabel="", ylabel=ylabel,
                                                   xlimits=(0.25, float(n) + 0.75),
                                                   ylimits=(0.0, 1.15 * ymax),
                                                   xticks=(Float64.(1:n), collect(labels))),
                             metadata=metadata)
end

function available_visuals(pc::DataTypes.PointCloud)
    d = DataTypes.ambient_dim(pc)
    d < 2 && return ()
    if d >= 3
        return (:points_3d, :points_2d, :point_density, :knn_graph, :radius_graph)
    end
    return (:points_2d, :point_density, :knn_graph, :radius_graph)
end

function available_visuals(res::DataIngestion.PointCodensityResult)
    base = collect(available_visuals(DataIngestion.source_data(res)))
    DataTypes.ambient_dim(DataIngestion.source_data(res)) == 2 &&
        push!(base, :codensity_radius_snapshots)
    return Tuple(base)
end

@inline function _point_layer_from_matrix(A::AbstractMatrix{<:Real}; color=:dodgerblue, alpha::Float64=1.0,
                                          markersize::Float64=8.0, colormap::Symbol=:viridis,
                                          markerspace::Symbol=:pixel)
    pts = NTuple{2,Float64}[(float(A[i, 1]), float(A[i, 2])) for i in 1:size(A, 1)]
    if color isa AbstractVector
        return PointLayer(pts, Float64[float(v) for v in color], alpha, markersize, colormap, markerspace)
    end
    return PointLayer(pts, color, alpha, markersize, colormap, markerspace)
end

@inline function _codensity_snapshot_cutoffs(vals::AbstractVector{<:Real}, levels)
    if levels isa Symbol
        levels === :quantiles ||
            throw(ArgumentError("codensity_radius_snapshots codensity_levels must be :quantiles or an explicit vector of cutoffs."))
        return Float64.(quantile(Float64.(vals), [0.38, 0.62, 0.84]))
    elseif levels isa AbstractVector
        isempty(levels) && throw(ArgumentError("codensity_radius_snapshots requires at least one codensity cutoff."))
        return Float64[float(v) for v in levels]
    end
    throw(ArgumentError("codensity_radius_snapshots codensity_levels must be :quantiles or an explicit vector of cutoffs."))
end

function _codensity_snapshot_panel_spec(coords::AbstractMatrix{<:Real},
                                        vals::AbstractVector{<:Real},
                                        cutoff::Real,
                                        radius::Real;max_points=5000)
    cutoff64 = float(cutoff)
    radius64 = float(radius)
    keep = Float64.(vals) .<= cutoff64
    ids=_source_ids(size(coords,1),max_points)
    selected=filter(i->keep[i],ids)
    retained = coords[selected, :]
    retained_vals = Float64.(vals[selected])
    layers = AbstractVisualizationLayer[
        _point_layer_from_matrix(coords[ids,:]; color=:gray88, alpha=0.95, markersize=5.0),
        _point_layer_from_matrix(retained; color=retained_vals, alpha=0.18,
                                 markersize=2.0 * radius64, colormap=:viridis, markerspace=:data),
        _point_layer_from_matrix(retained; color=retained_vals, alpha=0.95, markersize=8.0, colormap=:viridis),
    ]
    return VisualizationSpec(:codensity_radius_snapshot;
                             title="c <= $(round(cutoff64; digits=3)), r = $(round(radius64; digits=2))",
                             subtitle="$(length(selected))/$(count(identity, keep)) retained points displayed",
                             layers=layers,
                             axes=_axes_from_matrix_2d(coords),
                             metadata=(; cutoff=cutoff64, radius=radius64, nretained=count(identity, keep),
                                 point_ids=ids,retained_ids=selected,displayed_retained=length(selected),
                                 total_points=size(coords,1),displayed_points=length(ids)))
end

function _visual_spec(res::DataIngestion.PointCodensityResult, kind::Symbol;
                      radii::AbstractVector{<:Real}=[0.07, 0.16, 0.30],
                      codensity_levels=:quantiles,max_points=5000,_preview_cache=nothing,
                      kwargs...)
    vals = DataIngestion.codensity_values(res)
    data = DataIngestion.source_data(res)
    if kind === :codensity_radius_snapshots
        DataTypes.ambient_dim(data) == 2 ||
            throw(ArgumentError("codensity_radius_snapshots requires a 2-dimensional point cloud."))
        isempty(radii) &&
            throw(ArgumentError("codensity_radius_snapshots requires at least one radius."))
        all(float(r) > 0.0 for r in radii) ||
            throw(ArgumentError("codensity_radius_snapshots radii must be positive."))
        coords = Matrix{Float64}(DataTypes.point_matrix(data))
        cutoffs = _codensity_snapshot_cutoffs(vals, codensity_levels)
        panels = VisualizationSpec[]
        sizehint!(panels, length(cutoffs) * length(radii))
        for cutoff in reverse(cutoffs), radius in radii
            push!(panels, _codensity_snapshot_panel_spec(coords, vals, cutoff, radius;max_points))
        end
        return VisualizationSpec(:codensity_radius_snapshots;
                                 title="Codensity/radius snapshots",
                                 subtitle="rows filter by codensity cutoff; columns increase radius",
                                 panels=panels,
                                 metadata=(;
                                     object=:point_codensity_result,
                                     dtm_mass=DataIngestion.codensity_mass(res),
                                     neighbor_count=DataIngestion.neighbor_count(res),
                                     nlevels=length(cutoffs),
                                     total_points=size(coords,1),displayed_points=min(size(coords,1),max_points),
                                     point_ids=_source_ids(size(coords,1),max_points),
                                     construction=:codensity_ball_preview,
                                     nradii=length(radii),
                                     codensity_levels=Tuple(cutoffs),
                                     radii=Tuple(Float64.(radii)),
                                     panel_columns=max(length(radii), 1),
                                     figure_size=(980, 760),
                                 ))
    end
    base = _visual_spec(data, kind; color_values=Float64.(vals),max_points,_preview_cache,kwargs...)
    return VisualizationSpec(base.kind;
                             title=base.title,
                             subtitle=base.subtitle,
                             layers=base.layers,
                             panels=base.panels,
                             axes=base.axes,
                             legend=base.legend,
                             interaction=base.interaction,
                             metadata=(;
                                 base.metadata...,
                                 object=:point_codensity_result,
                                 dtm_mass=DataIngestion.codensity_mass(res),
                                 neighbor_count=DataIngestion.neighbor_count(res),
                                 value_range=isempty(vals) ? nothing : extrema(vals),
                             ))
end

function _visual_spec(pc::DataTypes.PointCloud, kind::Symbol; dims=nothing, labels=nothing,
        color_values=nothing,density::Bool=false,k::Int=6,radius=nothing,
        max_points=5000,max_edges=10000,max_labels=50,nn_backend=:auto,
        max_distance_pairs=10_000_000,max_graph_edges=1_000_000,_preview_cache=nothing)
    A=DataTypes.point_matrix(pc);n=size(A,1);target=kind===:points_3d ? 3 : 2
    P,chosen,geometry_reused=_source_projection(A,target;dims,cache=_preview_cache)
    ids=_source_ids(n,max_points);layers=AbstractVisualizationLayer[]
    rounded=_source_rounding(A,chosen)
    edges=NTuple{2,Int}[];eids=Int[];eligible=0;graph_reused=false;graph_backend=:none
    parameter=nothing
    graph_requested=kind in (:knn_graph,:radius_graph)
    if graph_requested
        parameter=kind===:knn_graph ? min(k,max(n-1,0)) : radius===nothing ?
            (n<=1 ? 0.0 : sqrt(maximum(_dist2(A,1,j) for j in 2:n))/2) : Float64(radius)
        # Automatic preview construction is exact in the supplied floating
        # metric. Ingestion's explicit approximate backend is never selected.
        backend=nn_backend===:auto ? (DataIngestion._have_pointcloud_nn_backend() ? :nearestneighbors : :bruteforce) : nn_backend
        graph=DataIngestion._prepared_point_graph(A,kind===:knn_graph ? :knn : :radius,parameter;
            backend,cache=_preview_cache,max_distance_pairs,max_graph_edges)
        edges=graph.edges;graph_reused=graph.reused;graph_backend=graph.backend
        eids,eligible=_source_edge_ids(edges,n,ids,max_edges;trusted=true)
        push!(layers,SegmentLayer(_segments_from_edges_2d(P,view(edges,eids)),:gray35,0.62,1.3))
    end
    show_density=kind!==:points_3d && (density || kind===:point_density)
    show_density && push!(layers,_density_heatmap(P))
    _source_points!(layers,P,ids;colors=color_values,markersize=target==3 ? 10.0 : 9.0,alpha=target==3 ? 0.85 : 0.9)
    lids=target==2 ? _source_labels!(layers,P,labels,ids,max_labels) : Int[]
    counts=_source_counts(n,length(edges),ids,eids,lids;eligible)
    provenance=graph_requested ? :reconstructed_metric_preview : :supplied_points
    axes=target==3 ? _axes_from_matrix_3d(P;labels=Tuple("x$d" for d in chosen)) :
        _axes_from_matrix_2d(P;labels=Tuple("x$d" for d in chosen))
    return VisualizationSpec(kind;title="Point cloud preview",
        subtitle=(graph_requested ? "Reconstructed $(kind===:knn_graph ? "kNN" : "radius") graph; " : "")*_source_caption(counts)*(rounded ? "; Float64 coordinates rounded" : ""),
        layers,axes,
        legend=_default_legend(visible=graph_requested,title="preview",
            entries=graph_requested ? [(;label="metric preview edges",color=:gray55,style=:line)] : NamedTuple[]),
        metadata=(;object=:point_cloud,projected_dims=Tuple(chosen),ambient_dim=size(A,2),npoints=n,
            overlay_density=show_density,density_population=show_density ? :all_source_points : :none,
            construction=provenance,graph_backend,graph_parameter=parameter,graph_reused,geometry_reused,
            drawing_rounding=rounded,source_edges=edges[eids],counts...,figure_size=(780,560)),
        interaction=_default_interaction(labels=!isempty(lids)))
end

function available_visuals(data::DataTypes.GraphData)
    d = DataTypes.ambient_dim(data)
    d !== nothing && d < 2 && return ()
    weighted = DataTypes.edge_weights(data) === nothing ? () : (:weighted_graph,)
    if d !== nothing && d >= 3
        return (:graph_3d, :graph, weighted...)
    end
    return (:graph, weighted...)
end

function _visual_spec(data::DataTypes.GraphData,kind::Symbol;dims=nothing,labels=nothing,
        max_points=5000,max_edges=10000,max_labels=50,_preview_cache=nothing)
    n=DataTypes.nvertices(data);target=kind===:graph_3d ? 3 : 2
    coords=DataTypes.coord_matrix(data)
    P,chosen,reused=coords===nothing ? (_graph_layout_2d(data)[1:2]...,false) :
        _source_projection(coords,target;dims,cache=_preview_cache)
    layout=coords===nothing ? :circular : :embedded
    rounded=coords===nothing ? false : _source_rounding(coords,chosen)
    ids=_source_ids(n,max_points);edges=DataTypes.edge_list(data)
    eids,eligible=_source_edge_ids(edges,n,ids,max_edges)
    drawn=edges[eids];w=DataTypes.edge_weights(data)
    w===nothing || all(isfinite,w) || throw(ArgumentError("Graph weights must be finite."))
    layers=AbstractVisualizationLayer[];entries=NamedTuple[]
    weighted=(kind===:weighted_graph || kind===:graph_3d) && w!==nothing
    if weighted
        ls,entries=target==3 ? _weighted_edge_layers_3d(P,drawn,w[eids];weight_range=isempty(w) ? nothing : extrema(w)) :
            _weighted_edge_layers_2d(P,drawn,w[eids];weight_range=isempty(w) ? nothing : extrema(w))
        append!(layers,ls)
    else
        push!(layers,target==3 ? Segment3Layer(_segments_from_edges_3d(P,drawn),:gray55,0.55,1.2) :
            SegmentLayer(_segments_from_edges_2d(P,drawn),:gray55,0.55,1.2))
        push!(entries,(;label="supplied edges",color=:gray55,style=:line))
    end
    _source_points!(layers,P,ids;basecolor=:black)
    target==2 && push!(entries,(;label="vertices",color=:black,style=:marker))
    lids=target==2 ? _source_labels!(layers,P,labels,ids,max_labels) : Int[]
    counts=_source_counts(n,length(edges),ids,eids,lids;eligible)
    axes=target==3 ? _axes_from_matrix_3d(P;labels=Tuple("x$d" for d in chosen)) :
        _axes_from_matrix_2d(P;labels=Tuple("x$d" for d in chosen))
    return VisualizationSpec(kind;title="Graph preview",subtitle="Supplied graph ($layout); "*_source_caption(counts)*(rounded ? "; Float64 coordinates rounded" : ""),
        layers,axes,legend=_default_legend(visible=true,title="edges",entries=entries),
        metadata=(;object=:graph_data,projected_dims=Tuple(chosen),layout,nvertices=n,nedges=length(edges),
            construction=:supplied_graph,drawing_rounding=rounded,geometry_reused=reused,source_edges=drawn,
            weight_range=w===nothing || isempty(w) ? nothing : extrema(w),counts...),
        interaction=_default_interaction(labels=!isempty(lids)))
end

available_visuals(::DataTypes.EmbeddedPlanarGraph2D) = (:embedded_planar_graph,)

function _append_visual_request_issues!(issues::Vector{String},data::DataTypes.EmbeddedPlanarGraph2D,kind::Symbol;kwargs...)
    p=(;kwargs...)
    _source_budget_check(; (k=>v for (k,v) in pairs(p) if k in _SOURCE_BUDGET_KEYS)...)
    limit=get(p,:max_path_points,100_000)
    limit isa Int && limit>=0 || push!(issues,"max_path_points must be a nonnegative Int.")
    labels=get(p,:labels,nothing)
    labels===nothing || (labels isa AbstractVector && length(labels)==DataTypes.nvertices(data)) || push!(issues,"labels must contain one value per source vertex.")
    return issues
end

function _visual_spec(data::DataTypes.EmbeddedPlanarGraph2D,kind::Symbol;labels=nothing,
        max_points=5000,max_edges=10000,max_labels=50,max_path_points=100_000,_preview_cache=nothing)
    positions=DataTypes.vertex_positions(data);n=length(positions)
    coords=[positions[i][j] for i in 1:n,j in 1:2]
    P,_,reused=_source_projection(coords,2;cache=_preview_cache)
    rounded=_source_rounding(coords,(1,2))
    ids=_source_ids(n,max_points);edges=DataTypes.edge_list(data)
    eids,eligible=_source_edge_ids(edges,n,ids,max_edges)
    poly=DataTypes.polylines(data);layers=AbstractVisualizationLayer[]
    if poly===nothing
        push!(layers,SegmentLayer(_segments_from_edges_2d(P,edges[eids]),:gray40,0.7,1.3))
    else
        length(poly)==length(edges) || throw(ArgumentError("Polylines must correspond to source edges in order."))
        sum(i->length(poly[i]),eids;init=0)<=max_path_points ||
            throw(ArgumentError("Selected polylines exceed max_path_points; increase it explicitly or display fewer edges."))
        paths=[[Tuple(Float64.(p)) for p in poly[i]] for i in eids]
        all(p->all(isfinite,p),Iterators.flatten(paths)) || throw(ArgumentError("Polyline coordinates must be finite."))
        push!(layers,PolylineLayer(paths,:gray40,0.8,1.6,false))
    end
    _source_points!(layers,P,ids;basecolor=:black,markersize=8.5)
    lids=_source_labels!(layers,P,labels,ids,max_labels)
    counts=_source_counts(n,length(edges),ids,eids,lids;eligible)
    bbox=DataTypes.bounding_box(data)
    axes=bbox===nothing ? _axes_from_matrix_2d(P;labels=("x","y")) :
        _default_axes_2d(xlabel="x",ylabel="y",xlimits=_pad_limits_1d((bbox[1],bbox[3])),
            ylimits=_pad_limits_1d((bbox[2],bbox[4])),aspect=:equal)
    return VisualizationSpec(kind;title="Embedded planar graph",subtitle="Supplied geometry; "*_source_caption(counts)*(rounded ? "; Float64 coordinates rounded" : ""),
        layers,axes,metadata=(;object=:embedded_planar_graph_2d,nvertices=n,nedges=length(edges),
            construction=:supplied_graph,drawing_rounding=rounded,geometry_reused=reused,source_edges=edges[eids],counts...),
        interaction=_default_interaction(labels=!isempty(lids)))
end

function available_visuals(img::DataTypes.ImageNd{T,N}) where {T,N}
    (N < 2 || isempty(_image_array(img))) && return ()
    if N == 2
        return (:image,)
    elseif N == 3
        if size(_image_array(img), 3) <= 4
            return (:image, :channels, :slice_viewer)
        end
        return (:image, :slice_viewer)
    end
    return (:image, :slice_viewer)
end

function _image_view_selection(img::DataTypes.ImageNd, kind::Symbol;
                               view_dims=nothing, slice_indices=nothing)
    A = _image_array(img)
    nd = ndims(A)
    nd >= 2 || throw(ArgumentError("image views require at least two array axes."))
    isempty(A) && throw(ArgumentError("image views require nonempty array axes."))
    kind in available_visuals(img) || throw(ArgumentError("kind=$kind is unsupported for this image shape."))
    # view_dims names (x, y); the default preserves Julia's row/column image layout.
    vd = if view_dims === nothing
        (2, 1)
    else
        length(view_dims) == 2 || throw(ArgumentError("view_dims must contain exactly two array axes (x, y)."))
        (Int(view_dims[1]), Int(view_dims[2]))
    end
    vd[1] != vd[2] || throw(ArgumentError("view_dims must select distinct array axes."))
    all(d -> 1 <= d <= nd, vd) || throw(ArgumentError("view_dims must lie between 1 and $nd."))
    if kind === :channels
        sort(collect(vd)) == [1, 2] || throw(ArgumentError("channels requires view_dims to select axes 1 and 2; axis 3 indexes panels."))
        slice_indices === nothing || throw(ArgumentError("channels displays every stored axis-3 slice; slice_indices is inapplicable."))
    end
    fixed = Dict{Int,Int}()
    for d in 1:nd
        d in vd && continue
        fixed[d] = nd == 3 && d == 3 && size(A, 3) <= 4 ? 1 : cld(size(A, d), 2)
    end
    if slice_indices !== nothing
        for (dim, pos) in pairs(slice_indices)
            d, p = Int(dim), Int(pos)
            1 <= d <= nd || throw(ArgumentError("slice_indices axis $d is outside 1:$nd."))
            d in vd && throw(ArgumentError("slice_indices cannot fix displayed axis $d."))
            1 <= p <= size(A, d) || throw(ArgumentError("slice index $p is outside axis $d."))
            fixed[d] = p
        end
    end
    return vd, fixed
end

# Shared with the live volume renderer: HeatmapLayer stores rows=y, columns=x.
function _image_slice_values(A::AbstractArray, view_dims::Tuple{Int,Int}, fixed::Dict{Int,Int})
    idx = ntuple(d -> d in view_dims ? Colon() : fixed[d], ndims(A))
    slice = Float64.(A[idx...])
    # Scalar indexing preserves the original order of the two remaining axes.
    return view_dims[2] < view_dims[1] ? slice : permutedims(slice)
end

function _image_colorrange(value)
    value === nothing && return nothing
    value isa Tuple && length(value) == 2 &&
        all(x -> x isa Real && !(x isa Bool) && isfinite(x), value) ||
        throw(ArgumentError("colorrange must be nothing or two finite real limits."))
    limits = Float64.(value)
    all(isfinite, limits) && limits[1] < limits[2] ||
        throw(ArgumentError("colorrange limits must remain finite and strictly increasing in display coordinates."))
    return limits
end

function _image_spec(img::DataTypes.ImageNd; kind::Symbol=:image, view_dims=nothing,
                     slice_indices=nothing, colormap::Symbol=:magma,
                     colorrange=nothing, colorbar_label::AbstractString="intensity",
                     title::AbstractString=(kind === :channels ? "Image channels" : "Image"), kwargs...)
    _ = kwargs
    A = _image_array(img)
    vd, fixed = _image_view_selection(img, kind; view_dims=view_dims, slice_indices=slice_indices)
    limits = _image_colorrange(colorrange)
    x = Float64.(1:size(A, vd[1]))
    y = Float64.(1:size(A, vd[2]))
    # Tiny arrays should show cell indices, not fractional positions between them.
    xticks = length(x) <= 12 ? (x, string.(1:length(x))) : nothing
    yticks = length(y) <= 12 ? (y, string.(1:length(y))) : nothing
    labels = ndims(A) == 2 ? ("Row index", "Column index") : nothing
    axes = _default_axes_2d(xlabel=labels === nothing ? "axis $(vd[1])" : labels[vd[1]],
        ylabel=labels === nothing ? "axis $(vd[2])" : labels[vd[2]],
        xlimits=(first(x)-0.5,last(x)+0.5), ylimits=(first(y)-0.5,last(y)+0.5),
        aspect=:equal, xticks=xticks, yticks=yticks)
    if kind === :channels
        panels = VisualizationSpec[]
        for c in 1:size(A, 3)
            vals = _image_slice_values(A, vd, Dict(3 => c))
            push!(panels, VisualizationSpec(:channels;
                                            title="channel $c",
                                            layers=AbstractVisualizationLayer[
                                                HeatmapLayer(x, y, vals, colormap, 1.0, colorbar_label),
                                            ],
                                            axes=axes,
                                            metadata=(; view_dims=vd, fixed_indices=Dict(3 => c), colorrange=limits,
                                                minimal_axes=true)))
        end
        return VisualizationSpec(:channels;
                                 title=title,
                                 subtitle="one panel per stored axis-3 slice",
                                 panels=panels,
                                 metadata=(; shape=size(A), nchannels=size(A, 3), view_dims=vd, colorrange=limits,
                                             coordinate_convention=:array_indices, channel_axis=3))
    end
    vals = _image_slice_values(A, vd, fixed)
    widgets = kind === :slice_viewer ? (:slice_index,) : ()
    subtitle = kind === :slice_viewer ? "slice controls require a live Julia session" : ""
    metadata = (; shape=size(A), view_dims=vd, fixed_indices=copy(fixed), colormap,
                 colorrange=limits, colorbar_label=String(colorbar_label),
                 coordinate_convention=:array_indices, minimal_axes=true, figure_size=(640,440))
    kind === :slice_viewer && (metadata = (; metadata..., volume=A, requires_live_julia=true))
    return VisualizationSpec(kind;
                             title=title,
                             subtitle=subtitle,
                             layers=AbstractVisualizationLayer[
                                 HeatmapLayer(x, y, vals, colormap, 1.0, colorbar_label),
                             ],
                             axes=axes,
                             metadata=metadata,
                             interaction=_default_interaction(widgets=widgets, notebook=kind === :slice_viewer ? :widget_viewer : :summary_card))
end

function _visual_spec(img::DataTypes.ImageNd, kind::Symbol; kwargs...)
    kind in (:image, :channels, :slice_viewer) || throw(ArgumentError("Unsupported ImageNd visualization kind $(kind)."))
    return _image_spec(img; kind=kind, kwargs...)
end

available_visuals(::DataIngestion.IngestionEstimate) = (:simplex_counts,)
available_visuals(::DataIngestion.IngestionPlan) = ()
available_visuals(::DataIngestion.GradedComplexBuildResult) = ()
available_visuals(::DataTypes.GradedComplex) = ()
available_visuals(::DataTypes.MultiCriticalGradedComplex) = ()
available_visuals(::DataTypes.SimplexTreeMulti) = (:simplex_counts,)

function _visual_spec(est::DataIngestion.IngestionEstimate, kind::Symbol; kwargs...)
    _ = kwargs
    if kind === :simplex_counts
        counts = DataIngestion.cell_counts_by_dim(est)
        return _bar_spec(:simplex_counts, "estimated cells by dimension",
                         ["dim $(d - 1)" for d in 1:length(counts)], counts;
                         ylabel="estimated cells",
                         fill_color=:steelblue3,
                         metadata=(; counts))
    end
    throw(ArgumentError("Unsupported IngestionEstimate visualization kind $(kind)."))
end

function _visual_spec(res::DataIngestion.GradedComplexBuildResult, kind::Symbol; kwargs...)
    _ = kwargs
    throw(ArgumentError("GradedComplexBuildResult has no visualization kinds. Use describe(buildres) or graded_complex_build_summary(buildres) for diagnostics."))
end

function _visual_spec(G::Union{DataTypes.GradedComplex,DataTypes.MultiCriticalGradedComplex}, kind::Symbol; kwargs...)
    _ = kwargs
    throw(ArgumentError("GradedComplex has no visualization kinds. Use describe(graded_complex) or data_summary(graded_complex) for diagnostics."))
end

function _visual_spec(ST::DataTypes.SimplexTreeMulti, kind::Symbol; kwargs...)
    _ = kwargs
    if kind === :simplex_counts
        counts = DataTypes.cell_counts(ST)
        return _bar_spec(:simplex_counts, "simplex counts by dimension",
                         ["dim $(d - 1)" for d in 1:length(counts)], counts;
                         ylabel="simplices",
                         fill_color=:teal,
                         metadata=(; counts, nsimplices=DataTypes.simplex_count(ST)))
    end
    throw(ArgumentError("Unsupported SimplexTreeMulti visualization kind $(kind)."))
end
