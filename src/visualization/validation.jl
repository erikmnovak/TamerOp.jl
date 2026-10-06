# Validation helpers for visualization requests and backend-agnostic specs.

@inline function _visual_issue_report(kind::Symbol, valid::Bool; kwargs...)
    return (; kind, valid, kwargs...)
end

function _throw_invalid_visual(kind::Symbol, issues::Vector{String})
    msg = isempty(issues) ? "invalid visualization request." : join(issues, " ")
    throw(ArgumentError(string(kind, ": ", msg)))
end

function check_visual_spec(spec::VisualizationSpec; throw::Bool=false)
    issues = String[]
    String(spec.kind) == "" && push!(issues, "kind must be a nonempty symbol")
    isempty(spec.layers) && isempty(spec.panels) &&
        push!(issues, "spec must contain at least one layer or one panel")
    !isempty(spec.layers) && !isempty(spec.panels) &&
        push!(issues, "spec cannot mix direct layers and panels")
    haskey(spec.axes, :xlabel) || push!(issues, "axes must define :xlabel")
    haskey(spec.axes, :ylabel) || push!(issues, "axes must define :ylabel")
    haskey(spec.axes, :xlimits) || push!(issues, "axes must define :xlimits")
    haskey(spec.axes, :ylimits) || push!(issues, "axes must define :ylimits")
    linked = spec.kind === :linked_inspector
    linked && _check_linked_inspection!(issues, spec)
    !linked && get(spec.interaction, :hover, false) && push!(issues, "Hover callbacks require a linked inspector.")
    !linked && get(spec.interaction, :clicks, false) && push!(issues, "Selection callbacks require a linked inspector.")
    widgets = get(spec.interaction, :widgets, ())
    if !linked && !isempty(widgets)
        widgets == (:slice_index,) && get(spec.interaction, :notebook, nothing) === :widget_viewer ||
            push!(issues, "The only implemented widget is :slice_index in a standalone slice viewer.")
        haskey(spec.metadata, :volume) && haskey(spec.metadata, :view_dims) && haskey(spec.metadata, :fixed_indices) ||
            push!(issues, "A slice widget requires volume, view_dims, and fixed_indices metadata.")
    end

    matrix_layers = count(layer -> layer isa MatrixLayer, spec.layers)
    matrix_layers > 0 && (matrix_layers != 1 || length(spec.layers) != 1) &&
        push!(issues, "A matrix panel must contain exactly one MatrixLayer and no other layers.")
    get(spec.metadata, :panel_style, nothing) === :matrix && matrix_layers != 1 &&
        push!(issues, "A matrix panel requires one MatrixLayer.")
    positions = get(spec.metadata, :panel_positions, nothing)
    valid_positions = false
    if haskey(spec.metadata, :panel_positions)
        if isempty(spec.panels)
            push!(issues, "panel_positions requires child panels.")
        elseif !(positions isa AbstractVector && length(positions) == length(spec.panels))
            push!(issues, "panel_positions must contain one (rowrange, columnrange) tuple per panel.")
        elseif !all(position -> position isa Tuple && length(position) == 2 &&
                    all(r -> r isa UnitRange{Int} && !isempty(r) && first(r) > 0, position), positions)
            push!(issues, "panel_positions must use nonempty, positive UnitRange{Int} row and column ranges.")
        else
            valid_positions = true
            # Compare interval endpoints without enumerating large spans.
            for j in eachindex(positions), i in firstindex(positions):(j - 1)
                a, b = positions[i], positions[j]
                overlap = all(k -> max(first(a[k]), first(b[k])) <= min(last(a[k]), last(b[k])), 1:2)
                overlap && push!(issues, "panel_positions for panels $i and $j overlap.")
            end
        end
    end
    if haskey(spec.metadata, :panel_row_weights)
        weights = spec.metadata.panel_row_weights
        haskey(spec.metadata, :panel_positions) ||
            push!(issues, "panel_row_weights requires panel_positions.")
        if !(weights isa AbstractVector && !isempty(weights) &&
             all(w -> w isa Real && !(w isa Bool) && isfinite(w) && w > 0 &&
                      isfinite(Float64(w)) && Float64(w) > 0, weights))
            push!(issues, "panel_row_weights must be a vector of positive finite real numbers.")
        elseif valid_positions && length(weights) != maximum(position -> last(position[1]), positions)
            push!(issues, "panel_row_weights must contain one weight per layout row.")
        end
    end
    for (idx, layer) in enumerate(spec.layers)
        if layer isa HeatmapLayer
            xok = size(layer.values, 2) == length(layer.x) || size(layer.values, 2) + 1 == length(layer.x)
            yok = size(layer.values, 1) == length(layer.y) || size(layer.values, 1) + 1 == length(layer.y)
            (xok && yok) ||
                push!(issues, "HeatmapLayer $idx values shape must match x/y lengths or cell-edge lengths.")
        elseif layer isa MatrixLayer
            size(layer.entries) == (length(layer.row_labels), length(layer.column_labels)) ||
                push!(issues, "MatrixLayer $idx entries must match target-row and source-column labels.")
            shape = get(spec.metadata, :matrix_size, size(layer.entries))
            if !(shape isa Tuple && length(shape) == 2 &&
                 all(n -> n isa Integer && !(n isa Bool) && n >= 0, shape))
                push!(issues, "matrix_size must be a pair of nonnegative integer dimensions.")
            elseif any(size(layer.entries, i) > shape[i] for i in 1:2)
                push!(issues, "Displayed MatrixLayer entries cannot exceed the full matrix_size.")
            end
            for (key, dims) in ((:row_colors, (size(layer.entries,1),)),
                                (:column_colors, (size(layer.entries,2),)),
                                (:cell_colors, size(layer.entries)))
                haskey(spec.metadata, key) || continue
                colors = spec.metadata[key]
                colors isa AbstractArray && size(colors) == dims ||
                    push!(issues, "$key must match the displayed matrix dimensions.")
            end
            for (key, dims) in ((:row_roles, (size(layer.entries,1),)),
                                (:column_roles, (size(layer.entries,2),)),
                                (:cell_roles, size(layer.entries)))
                haskey(spec.metadata, key) || continue
                roles = spec.metadata[key]
                if !(roles isa AbstractArray && size(roles) == dims)
                    push!(issues, "$key must match the displayed matrix dimensions.")
                elseif !all(role -> role isa Symbol && role in _VISUAL_COLOR_ROLES, roles)
                    push!(issues, "$key contains an unknown mathematical display role.")
                end
            end
            for key in (:matrix_corner, :matrix_row_heading, :empty_matrix_reason)
                get(spec.metadata, key, "") isa AbstractString ||
                    push!(issues, "$key must be a string.")
            end
        elseif layer isa RectLayer
            for rect in layer.rects
                rect[1] <= rect[3] || push!(issues, "RectLayer $idx has xlo > xhi.")
                rect[2] <= rect[4] || push!(issues, "RectLayer $idx has ylo > yhi.")
            end
        elseif layer isa PolygonLayer
            for polygon in layer.polygons
                length(polygon) >= 3 || push!(issues, "PolygonLayer $idx polygons need at least three vertices.")
                all(p -> all(isfinite, p), polygon) ||
                    push!(issues, "PolygonLayer $idx vertices must have finite display coordinates.")
            end
        elseif layer isa SegmentLayer
            layer.linestyle in (:solid, :dash, :dot) ||
                push!(issues, "SegmentLayer $idx linestyle must be :solid, :dash, or :dot.")
            all(seg -> all(isfinite, seg), layer.segments) ||
                push!(issues, "SegmentLayer $idx coordinates must be finite.")
        elseif layer isa TextLayer
            length(layer.labels) == length(layer.positions) ||
                push!(issues, "TextLayer $idx labels/positions length mismatch.")
        elseif layer isa BarcodeLayer
            length(layer.intervals) == length(layer.multiplicities) ||
                push!(issues, "BarcodeLayer $idx intervals/multiplicities length mismatch.")
        elseif layer isa PointLayer
            layer.color isa AbstractVector && length(layer.color) != length(layer.points) &&
                push!(issues, "PointLayer $idx color_values length must match the number of points.")
            layer.markerspace in (:pixel, :data) ||
                push!(issues, "PointLayer $idx markerspace must be :pixel or :data.")
        elseif layer isa Point3Layer
            isempty(layer.points) || all(length(p) == 3 for p in layer.points) ||
                push!(issues, "Point3Layer $idx points must be 3-vectors.")
            layer.color isa AbstractVector && length(layer.color) != length(layer.points) &&
                push!(issues, "Point3Layer $idx color_values length must match the number of points.")
        elseif layer isa Segment3Layer
            all(seg -> length(seg) == 6, layer.segments) ||
                push!(issues, "Segment3Layer $idx segments must store (x1,y1,z1,x2,y2,z2).")
        end
    end

    for (idx, panel) in enumerate(spec.panels)
        isempty(panel.panels) || push!(issues, "panel $idx cannot contain nested panels.")
        isempty(get(panel.interaction, :widgets, ())) ||
            push!(issues, "panel $idx contains a widget; render slice viewers as standalone figures.")
        panel_report = check_visual_spec(panel; throw=false)
        get(panel_report, :valid, false) || begin
            for issue in get(panel_report, :issues, String[])
                push!(issues, "panel $idx: " * issue)
            end
        end
    end

    valid = isempty(issues)
    throw && !valid && _throw_invalid_visual(:check_visual_spec, issues)
    return _visual_issue_report(:visual_spec, valid;
                                visual_kind=visual_kind(spec),
                                nlayers=length(spec.layers),
                                npanels=length(spec.panels),
                                issues=issues)
end

"""
    check_visual_request(obj; kind=:auto, backend=:auto, throw=false, kwargs...)

Check recipe availability, prerequisites, and effective recipe keywords without
constructing the picture. The report includes `supported_keywords`, qualitative
construction cost, and renderer availability. `backend=:auto` permits building
specifications without an activated renderer; a named backend must be activated.
Renderer controls (`figure`, `size`, `style`) belong to `render`/`visualize`/`save_visual`,
not to recipe construction. Unsupported keywords are errors, including keywords
that apply to a different kind on the same object.
"""
function check_visual_request(obj; kind::Symbol=:auto, backend::Symbol=:auto,
                              throw::Bool=false, kwargs...)
    supported = available_visuals(obj)
    issues = String[]
    isempty(supported) && push!(issues, "no visualization kinds are registered for $(nameof(typeof(obj))).")
    requested = kind === :auto ? (isempty(supported) ? :auto : supported[1]) : kind
    if kind !== :auto && !(kind in supported)
        push!(issues, "kind=$(kind) is unsupported for $(nameof(typeof(obj))); supported kinds are $(supported).")
    end
    keywords = _visual_request_keywords(obj, requested)
    for name in keys(kwargs)
        name in keywords || push!(issues, "keyword $(name) is unsupported for kind=$(requested); supported recipe keywords are $(keywords).")
    end
    backend === :auto || _visual_backend_available(backend) ||
        push!(issues, "visualization backend $(backend) is not activated. $(get(_VISUAL_BACKEND_HELP, backend, ""))")
    requested === :linked_inspector && !(backend in (:auto, :wglmakie)) &&
        push!(issues, "Live linked inspection requires WGLMakie; render inspection_snapshot(session) for a static view.")
    try
        _append_visual_request_issues!(issues, obj, requested; kwargs...)
    catch err
        err isa InterruptException && rethrow()
        push!(issues, sprint(showerror, err))
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_visual(:check_visual_request, issues)
    return _visual_issue_report(:visual_request, valid;
                                object_type=Symbol(nameof(typeof(obj))),
                                requested_kind=requested,
                                supported_kinds=supported,
                                supported_keywords=keywords,
                                construction_cost=_visual_request_cost(obj, requested),
                                rendering=_visual_render_capabilities(; kind=requested),
                                issues=issues)
end

# This is the validation contract for the existing recipes, not a second recipe
# registry. Dispatch follows their mathematical owners; kind-specific tuples
# deliberately exclude keywords that the selected recipe would ignore.
_visual_request_keywords(obj, kind::Symbol) = ()
_visual_request_keywords(obj::Union{AbstractPLikeEncodingMap,CompiledEncoding,EncodingResult}, kind::Symbol) =
    kind === :query_overlay ? (:box, :point, :points) : (:box,)
_visual_request_keywords(obj::Flange, kind::Symbol) =
    kind === :regions ? (:box, :alpha_up, :alpha_dn) : (:box,)
_visual_request_keywords(obj::CohomologyDimsResult, kind::Symbol) =
    kind in (:cohomology_support, :cohomology_support_plane) ? (:box,) : ()
const _INTERVAL_DISPLAY_KEYWORDS = (:window, :interval, :max_intervals)
_visual_request_keywords(obj::Union{AbstractDict{<:Tuple{<:Real,<:Real},<:Integer},
    AbstractVector{<:Tuple{<:Real,<:Real}},PackedBarcode}, kind::Symbol) =
    kind in (:barcode,:persistence_diagram) ? _INTERVAL_DISPLAY_KEYWORDS : ()
_visual_request_keywords(obj::InvariantResult, kind::Symbol) =
    obj.which in (:slice_barcode,:slice_barcodes) ? _visual_request_keywords(invariant_value(obj),kind) :
    kind === :rank_query_overlay ? (:box, :pair, :pairs) :
    kind === :hilbert_heatmap ? (:box,) : ()
_visual_request_keywords(obj::Union{SliceBarcodesResult,ProjectedBarcodesResult}, kind::Symbol) =
    kind in (:barcode,:persistence_diagram) ? (:index,_INTERVAL_DISPLAY_KEYWORDS...) : ()
_visual_request_keywords(obj::FiberedSliceResult, kind::Symbol) =
    kind in (:barcode,:persistence_diagram) ? _INTERVAL_DISPLAY_KEYWORDS :
    kind === :fibered_slice_overlay ? (:arrangement, :dir, :offset, :basepoint, :tie_break) : ()
function _visual_request_keywords(obj::Union{FiberedArrangement2D,FiberedBarcodeCache2D}, kind::Symbol)
    obj isa FiberedBarcodeCache2D && kind in (:barcode,:persistence_diagram) &&
        return (:dir,:offset,:basepoint,:tie_break,_INTERVAL_DISPLAY_KEYWORDS...)
    kind in (:fibered_query, :fibered_cell_highlight, :fibered_tie_break, :fibered_query_barcode) &&
        return (:dir, :offset, :basepoint, :tie_break)
    kind === :fibered_offset_intervals && return (:dir,)
    kind === :fibered_projected_comparison && return (:projected,)
    kind in (:fibered_family_contributions, :fibered_distance_diagnostic) && return (:caches,)
    return ()
end
_visual_request_keywords(obj::FiberedSliceFamily2D, kind::Symbol) =
    kind in (:fibered_family_contributions, :fibered_distance_diagnostic) ? (:caches,) : ()
_visual_request_keywords(obj::MPPLineSpec, kind::Symbol) = (:box,)
_visual_request_keywords(obj::MPPDecomposition, kind::Symbol) = (:layout,)
_visual_request_keywords(obj::MPLandscape, kind::Symbol) =
    kind === :landscape_slices ? (:idir, :ioff, :layer) : ()
_visual_request_keywords(obj::OrdinaryPersistence.PersistenceDiagram, kind::Symbol) = (:dim,_INTERVAL_DISPLAY_KEYWORDS...)
function _visual_request_keywords(obj::DataTypes.PointCloud, kind::Symbol)
    kind === :points_3d && return (:dims, :color_values)
    kind === :point_density && return (:dims, :labels, :color_values)
    common = (:dims, :labels, :color_values, :density)
    kind === :knn_graph && return (common..., :k)
    kind === :radius_graph && return (common..., :radius)
    return common
end
function _visual_request_keywords(obj::DataIngestion.PointCodensityResult, kind::Symbol)
    kind === :codensity_radius_snapshots && return (:radii, :codensity_levels)
    return Tuple(k for k in _visual_request_keywords(DataIngestion.source_data(obj), kind) if k !== :color_values)
end
_visual_request_keywords(obj::DataTypes.GraphData, kind::Symbol) =
    DataTypes.coord_matrix(obj) === nothing ? (:labels,) :
    kind === :graph_3d ? (:dims,) : (:dims, :labels)
_visual_request_keywords(obj::DataTypes.EmbeddedPlanarGraph2D, kind::Symbol) = (:labels,)
_visual_request_keywords(obj::DataTypes.ImageNd, kind::Symbol) =
    kind === :channels ? (:view_dims, :colormap, :colorrange, :colorbar_label, :title) :
        (:view_dims, :slice_indices, :colormap, :colorrange, :colorbar_label, :title)

function _visual_request_cost(obj, kind::Symbol)
    if kind === :rank_section
        return (; work=:one_rank_row_or_column_per_distinct_anchor_label,
            all_pairs_table=false, selected_matrices=:selected_pair_only,
            cache_reuse=:within_call_or_bounded_inspection_session,
            lazy_encoding=:rank_queries_may_materialize, timing=:not_measured)
    end
    if kind === :presentation_inspector
        return (; work=:selected_presentation_fibers, timing=:not_measured,
            cache_reuse=:none, default=:supports_and_coefficients_without_ranks,
            single_stalk=:active_block_and_rank, basis=:explicit_single_stalk_opt_in,
            map_queries=:defined_pair_computes_two_bases_and_induced_map,
            lazy_encoding=:no_module_materialization,
            geometry=:when_supported_by_encoding)
    end
    if kind in (:hasse, :module_inspector)
        return (; work=kind === :hasse ? :cover_graph_and_dimensions : :selected_stalk_or_map,
            timing=:not_measured, cache_reuse=:module_owner,
            default=:dimensions_without_structure_maps,
            map_queries=kind === :hasse ? :none : :selected_pair_only,
            lazy_encoding=kind === :hasse ? :dimensions_only : :defined_pair_may_materialize_cover_maps,
            geometry=kind === :module_inspector ? :when_supported_by_encoding : :schematic)
    end
    work = if kind in (:rank_heatmap, :rank_rectangles)
        :dense_pair_table
    elseif kind === :constant_subdivision
        :fiber_rank_grid
    elseif kind in (:knn_graph, :radius_graph)
        :all_pairs_distances
    elseif kind in (:fibered_family_contributions, :fibered_distance_diagnostic)
        :sampled_family_bottleneck_queries
    elseif kind === :fibered_query_barcode
        :slice_barcode_query
    elseif kind in (:regions, :region_labels, :query_overlay, :rank_query_overlay,
                     :hilbert_heatmap, :cohomology_support, :cohomology_support_plane)
        :geometry_materialization
    elseif kind === :density_image
        :dense_sample_grid
    else
        :recipe_materialization
    end
    return (; work, timing=:not_measured, cache_reuse=:recipe_dependent)
end

_append_visual_request_issues!(issues::Vector{String}, obj, kind::Symbol; kwargs...) = issues

function _interval_request_issues!(issues; window=nothing,interval=nothing,max_intervals=200,kwargs...)
    window === nothing || _interval_window((),window)
    interval === nothing || (interval isa Integer && !(interval isa Bool) && 0 < interval <= typemax(Int)) ||
        push!(issues,"interval must be a positive integer group ID or nothing.")
    max_intervals isa Integer && !(max_intervals isa Bool) && 0 < max_intervals <= typemax(Int) ||
        push!(issues,"max_intervals must be a positive integer fitting Int.")
    return issues
end

function _append_visual_request_issues!(issues::Vector{String},
        obj::Union{AbstractDict{<:Tuple{<:Real,<:Real},<:Integer},AbstractVector{<:Tuple{<:Real,<:Real}},PackedBarcode},
        kind::Symbol; kwargs...)
    kind in (:barcode,:persistence_diagram) || return issues
    _interval_request_issues!(issues;kwargs...)
    # Validate the supplied intervals without constructing layers or evaluating algebra.
    entries = obj isa AbstractVector ? ((iv,1) for iv in obj) : obj
    for (iv,mult) in entries
        length(iv) == 2 || throw(ArgumentError("Barcode entries require two endpoints."))
        _interval_record(iv[1],iv[2],mult)
    end
    return issues
end

function _append_visual_request_issues!(issues::Vector{String}, obj::Union{Flange,MPPLineSpec}, kind::Symbol; kwargs...)
    box = get(kwargs, :box, nothing)
    box === nothing || _visual_box_2d(box)
    for name in (:alpha_up, :alpha_dn)
        alpha = get(kwargs, name, nothing)
        alpha === nothing && continue
        alpha isa Real && isfinite(alpha) && 0 <= alpha <= 1 ||
            push!(issues, "$(name) must be a finite opacity between 0 and 1.")
    end
    return issues
end

function _require_one_of!(issues::Vector{String}, names::Tuple, kwargs::NamedTuple, context::AbstractString)
    any(name -> get(kwargs, name, nothing) !== nothing, names) ||
        push!(issues, context * " requires one of " * join(string.(names), ", ") * ".")
    return issues
end

function _require_all!(issues::Vector{String}, names::Tuple, kwargs::NamedTuple, context::AbstractString)
    for name in names
        get(kwargs, name, nothing) !== nothing || push!(issues, context * " requires keyword " * string(name) * ".")
    end
    return issues
end

function _append_visual_request_issues!(issues::Vector{String}, obj::Union{AbstractPLikeEncodingMap,CompiledEncoding,EncodingResult}, kind::Symbol; kwargs...)
    params = (; kwargs...)
    get(params, :box, nothing) === nothing || _visual_box_2d(params.box)
    if kind === :query_overlay
        _require_one_of!(issues, (:point, :points), params, "query_overlay")
        points = _collect_query_points(; point=get(params, :point, nothing), points=get(params, :points, nothing))
        isempty(points) && push!(issues, "query_overlay requires at least one query point.")
        foreach(_drawing_point, points)
    end
    return issues
end

function _append_visual_request_issues!(issues::Vector{String}, obj::DataTypes.PointCloud, kind::Symbol; kwargs...)
    params = (; kwargs...)
    target = kind === :points_3d ? 3 : 2
    dims = get(params, :dims, nothing)
    if dims !== nothing
        length(dims) == target && length(unique(dims)) == target &&
            all(d -> d isa Integer && !(d isa Bool) && 1 <= d <= DataTypes.ambient_dim(obj), dims) ||
            push!(issues, "dims must select $target distinct coordinate axes within the point cloud.")
    end
    colors = get(params, :color_values, nothing)
    colors === nothing || length(colors) == DataTypes.npoints(obj) ||
        push!(issues, "color_values must contain one value per point.")
    get(params, :labels, nothing) === nothing || _collect_point_labels(params.labels, DataTypes.npoints(obj))
    if kind === :knn_graph
        k = get(params, :k, nothing)
        k === nothing || Int(k) > 0 || push!(issues, "knn_graph k must be positive.")
    elseif kind === :radius_graph
        radius = get(params, :radius, nothing)
        radius === nothing || float(radius) > 0 || push!(issues, "radius_graph radius must be positive.")
    end
    return issues
end

function _append_visual_request_issues!(issues::Vector{String}, obj::DataTypes.ImageNd, kind::Symbol; kwargs...)
    params = (; kwargs...)
    _image_view_selection(obj, kind; view_dims=get(params, :view_dims, nothing),
                           slice_indices=get(params, :slice_indices, nothing))
    get(params, :colormap, :magma) isa Symbol || push!(issues, "colormap must be a Symbol naming a Makie colormap.")
    _image_colorrange(get(params, :colorrange, nothing))
    for key in (:title, :colorbar_label)
        get(params, key, "") isa AbstractString || push!(issues, "$key must be text.")
    end
    return issues
end

function _append_visual_request_issues!(issues::Vector{String}, obj::DataTypes.GraphData, kind::Symbol; kwargs...)
    params = (; kwargs...)
    dims = get(params, :dims, nothing)
    if dims !== nothing && DataTypes.coord_matrix(obj) !== nothing
        target = kind === :graph_3d ? 3 : 2
        length(dims) == target && length(unique(dims)) == target &&
            all(d -> d isa Integer && !(d isa Bool) && 1 <= d <= DataTypes.ambient_dim(obj), dims) ||
            push!(issues, "dims must select $target distinct axes within the graph coordinates.")
    end
    get(params, :labels, nothing) === nothing || _collect_point_labels(params.labels, DataTypes.nvertices(obj))
    return issues
end

function _append_visual_request_issues!(issues::Vector{String}, obj::CohomologyDimsResult, kind::Symbol; kwargs...)
    box = get(kwargs, :box, nothing)
    box === nothing || _visual_box_2d(box)
    return issues
end

function _append_visual_request_issues!(issues::Vector{String}, obj::DataIngestion.PointCodensityResult, kind::Symbol; kwargs...)
    params = (; kwargs...)
    if kind === :codensity_radius_snapshots
        DataTypes.ambient_dim(DataIngestion.source_data(obj)) == 2 ||
            push!(issues, "codensity_radius_snapshots requires a 2-dimensional point cloud.")
        radii = get(params, :radii, nothing)
        if radii !== nothing
            try
                isempty(radii) &&
                    push!(issues, "codensity_radius_snapshots requires at least one radius.")
                all(float(r) > 0.0 for r in radii) ||
                    push!(issues, "codensity_radius_snapshots radii must be positive.")
            catch err
                push!(issues, sprint(showerror, err))
            end
        end
        levels = get(params, :codensity_levels, nothing)
        if levels isa Symbol
            levels === :quantiles ||
                push!(issues, "codensity_radius_snapshots codensity_levels must be :quantiles or an explicit vector of cutoffs.")
        elseif levels !== nothing
            try
                isempty(levels) &&
                    push!(issues, "codensity_radius_snapshots requires at least one codensity cutoff.")
                all(isfinite(float(v)) for v in levels) ||
                    push!(issues, "codensity_radius_snapshots cutoffs must be finite real values.")
            catch err
                push!(issues, sprint(showerror, err))
            end
        end
        return issues
    end
    return _append_visual_request_issues!(issues, DataIngestion.source_data(obj), kind; kwargs...)
end

function _append_visual_request_issues!(issues::Vector{String}, obj::Union{FiberedArrangement2D,FiberedBarcodeCache2D}, kind::Symbol; kwargs...)
    params = (; kwargs...)
    if obj isa FiberedBarcodeCache2D && kind in (:barcode,:persistence_diagram)
        _interval_request_issues!(issues;kwargs...)
        _require_all!(issues,(:dir,),params,string(kind))
        _require_one_of!(issues,(:offset,:basepoint),params,string(kind))
        get(params,:tie_break,:up) in (:up,:down,:center) || push!(issues,"tie_break must be :up, :down, or :center.")
    elseif kind in (:fibered_query, :fibered_cell_highlight, :fibered_tie_break, :fibered_query_barcode)
        _require_all!(issues, (:dir,), params, string(kind))
        _require_one_of!(issues, (:offset, :basepoint), params, string(kind))
        if kind === :fibered_tie_break && isempty(issues)
            try
                arg = get(params, :offset, nothing)
                arg === nothing && (arg = get(params, :basepoint, nothing))
                report = fibered_query_summary(obj, params.dir, arg; tie_break=get(params, :tie_break, :up))
                report.valid || append!(issues, String.(report.issues))
                report.valid && !report.tie_break_relevant &&
                    push!(issues, "fibered_tie_break requires a boundary query where cell_up != cell_down.")
            catch err
                push!(issues, sprint(showerror, err))
            end
        end
    elseif kind === :fibered_offset_intervals
        _require_all!(issues, (:dir,), params, string(kind))
    elseif kind === :fibered_projected_comparison
        _require_all!(issues, (:projected,), params, string(kind))
        projected = get(params, :projected, nothing)
        projected isa ProjectedArrangement || push!(issues, "fibered_projected_comparison requires projected to be a ProjectedArrangement.")
        projected isa ProjectedArrangement && !(:projected_arrangement in available_visuals(projected)) &&
            push!(issues, "fibered_projected_comparison requires 2D projected directions.")
    elseif kind in (:fibered_family_contributions, :fibered_distance_diagnostic)
        _require_all!(issues, (:caches,), params, string(kind))
        if isempty(issues)
            try
                _resolve_fibered_caches(obj, params.caches)
            catch err
                push!(issues, sprint(showerror, err))
            end
        end
    end
    return issues
end

function _append_visual_request_issues!(issues::Vector{String}, obj::FiberedSliceFamily2D, kind::Symbol; kwargs...)
    params = (; kwargs...)
    if kind in (:fibered_family_contributions, :fibered_distance_diagnostic)
        _require_all!(issues, (:caches,), params, string(kind))
        if isempty(issues)
            try
                _resolve_fibered_caches(obj, params.caches)
            catch err
                push!(issues, sprint(showerror, err))
            end
        end
    end
    return issues
end

function _append_visual_request_issues!(issues::Vector{String}, obj::FiberedSliceResult, kind::Symbol; kwargs...)
    params = (; kwargs...)
    if kind in (:barcode,:persistence_diagram)
        _interval_request_issues!(issues;kwargs...)
    elseif kind === :fibered_slice_overlay
        _require_all!(issues, (:arrangement, :dir), params, string(kind))
        _require_one_of!(issues, (:offset, :basepoint), params, string(kind))
        arr = get(params, :arrangement, nothing)
        arr isa FiberedArrangement2D || push!(issues, "fibered_slice_overlay requires arrangement to be a FiberedArrangement2D.")
    end
    return issues
end

function _append_visual_request_issues!(issues::Vector{String}, obj::Union{SliceBarcodesResult,ProjectedBarcodesResult}, kind::Symbol; kwargs...)
    if kind in (:barcode,:persistence_diagram)
        _interval_request_issues!(issues;kwargs...)
        bars = obj isa SliceBarcodesResult ? slice_barcodes(obj) : projected_barcodes(obj)
        _interval_family_index(bars,get(kwargs,:index,nothing),string(nameof(typeof(obj))))
    end
    return issues
end

function _append_visual_request_issues!(issues::Vector{String}, obj::MPLandscape, kind::Symbol; kwargs...)
    params = (; kwargs...)
    if kind === :landscape_slices
        get(params, :idir, nothing) !== nothing || push!(issues, "landscape_slices requires keyword idir.")
        get(params, :ioff, nothing) !== nothing || push!(issues, "landscape_slices requires keyword ioff.")
        for (name, bound) in ((:idir, ndirections(obj)), (:ioff, noffsets(obj)), (:layer, landscape_layers(obj)))
            value = get(params, name, name === :layer ? 1 : nothing)
            value === nothing && continue
            value isa Integer && !(value isa Bool) && 1 <= value <= bound ||
                push!(issues, "$(name) must be an integer in 1:$(bound).")
        end
    end
    return issues
end

function _append_visual_request_issues!(issues::Vector{String}, obj::MPPDecomposition, kind::Symbol; kwargs...)
    params = (; kwargs...)
    if kind === :mpp_decomposition
        layout = get(params, :layout, :overlay)
        layout in (:overlay, :summands) ||
            push!(issues, "mpp_decomposition layout must be :overlay or :summands.")
        layout === :summands && nsummands(obj) == 0 &&
            push!(issues, "layout=:summands requires at least one sampled track; use layout=:overlay for an empty decomposition.")
    end
    return issues
end

function _append_visual_request_issues!(issues::Vector{String}, obj::InvariantResult, kind::Symbol; kwargs...)
    obj.which in (:slice_barcode,:slice_barcodes) &&
        return _append_visual_request_issues!(issues,invariant_value(obj),kind;kwargs...)
    params = (; kwargs...)
    get(params, :box, nothing) === nothing || _visual_box_2d(params.box)
    if kind === :rank_query_overlay
        _require_one_of!(issues, (:pair, :pairs), params, "rank_query_overlay")
        query_pairs = _collect_rank_query_pairs(; pair=get(params, :pair, nothing), pairs=get(params, :pairs, nothing))
        isempty(query_pairs) && push!(issues, "rank_query_overlay requires at least one query pair.")
        for (x, y) in query_pairs
            _drawing_point(x)
            _drawing_point(y)
            report = _visual_rank_query_points(encoding_map(obj), collect(x), collect(y); throw=false)
            report.valid || append!(issues, String.(report.issues))
        end
    end
    return issues
end

function _append_visual_request_issues!(issues::Vector{String}, obj::ModuleTranslationResult, kind::Symbol; kwargs...)
    if kind === :pushforward_overlay
        map = translation_map(obj)
        map isa EncodingMap || push!(issues, "pushforward_overlay currently requires translation_map(res) to be an EncodingMap.")
    end
    return issues
end
