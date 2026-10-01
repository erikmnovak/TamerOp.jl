# Selected finite fringe presentations: supports, active coefficients, and images.

available_visuals(::FiniteFringe.FringeModule) = (:presentation_inspector,)
_inspection_poset(H::FiniteFringe.FringeModule) = FiniteFringe.ambient_poset(H)
_inspection_field(H::FiniteFringe.FringeModule) = FiniteFringe.field(H)
_inspection_presentation(H::FiniteFringe.FringeModule) = H
function _inspection_presentation(enc::EncodingResult)
    H = Results.encoding_presentation(enc)
    H === nothing && throw(ArgumentError("This encoding has no retained finite fringe presentation on its current poset and field."))
    return H
end

function _presentation_request_keywords(obj)
    common = (:vertex, :pair, :matrix_limit, :basis, :upset, :downset)
    return obj isa EncodingResult && _inspection_has_geometry(obj) ?
        (common..., :point, :parameter_pair, :box) : common
end
_visual_request_keywords(H::FiniteFringe.FringeModule, ::Symbol) = _presentation_request_keywords(H)

function _check_presentation_selection!(issues, obj; kwargs...)
    _check_module_selection!(issues, obj, :presentation_inspector; kwargs...)
    H = _inspection_presentation(obj)
    for (key, supports) in ((:upset, FiniteFringe.birth_upsets(H)),
                            (:downset, FiniteFringe.death_downsets(H)))
        id = get(kwargs, key, nothing)
        id === nothing || (id isa Integer && !(id isa Bool) && 1 <= id <= length(supports)) ||
            push!(issues, "$key must be a support index in 1:$(length(supports)).")
    end
    if haskey(kwargs, :basis)
        kwargs[:basis] isa Bool || push!(issues, "basis must be true or false.")
        (get(kwargs, :vertex, nothing) !== nothing || get(kwargs, :point, nothing) !== nothing) ||
            push!(issues, "basis is only accepted for a selected stalk; a defined pair explicitly computes both bases.")
    end
    return issues
end
_append_visual_request_issues!(issues::Vector{String}, H::FiniteFringe.FringeModule, ::Symbol; kwargs...) =
    _check_presentation_selection!(issues, H; kwargs...)
_visual_spec(H::FiniteFringe.FringeModule, ::Symbol; kwargs...) = _presentation_visual_spec(H; kwargs...)

function _presentation_matrix_panel(A, title, row_labels, column_labels, limit;
                                    subtitle="", metadata=NamedTuple())
    rows, cols = 1:min(size(A, 1), limit[1]), 1:min(size(A, 2), limit[2])
    truncated = length(rows) < size(A, 1) || length(cols) < size(A, 2)
    subtitle *= (isempty(subtitle) ? "" : "\n") * "$(size(A,1)) x $(size(A,2))"
    truncated && (subtitle *= "; showing $(length(rows)) x $(length(cols)); full matrix in metadata")
    clipped = metadata
    for (key, inds) in ((:row_roles, (rows,)), (:column_roles, (cols,)), (:cell_roles, (rows, cols)))
        haskey(metadata, key) || continue
        # Cell styling is already limited to displayed entries, even for a very
        # large sparse coefficient matrix. Row/column metadata can be linear.
        value = key === :cell_roles ? copy(metadata[key]) : metadata[key][inds...]
        clipped = merge(clipped, NamedTuple{(key,)}((value,)))
    end
    return VisualizationSpec(:presentation_matrix; title, subtitle,
        layers=AbstractVisualizationLayer[MatrixLayer(
            [_matrix_coefficient_text(A[i,j]) for i in rows, j in cols],
            String.(row_labels[rows]), String.(column_labels[cols]))],
        metadata=merge((; panel_style=:matrix, matrix=copy(A), matrix_size=size(A),
            displayed_rows=rows, displayed_columns=cols, truncated), clipped))
end

function _presentation_support_layers(geometry, membership)
    colors = Dict(q => _VisualRole(membership[q] ? :support : :absent) for q in eachindex(membership))
    colors[0] = _VisualRole(:unrepresented)
    layers = _region_geometry_layers(geometry; colors)
    # Membership is a mathematical label, independent of the chosen palette.
    positions = NTuple{2,Float64}[_drawing_point((
        sum(p[1] for p in c.vertices) / length(c.vertices),
        sum(p[2] for p in c.vertices) / length(c.vertices))) for c in geometry.components]
    labels = [c.region_id == 0 ? "0: ?" :
        "$(c.region_id): $(membership[c.region_id] ? "+" : "-")" for c in geometry.components]
    push!(layers, TextLayer(labels, positions, _VisualRole(:foreground), 10.0))
    return layers
end

function _presentation_support_panel(H, family, id, region, limit)
    isup = family === :upset
    supports = isup ? FiniteFringe.birth_upsets(H) : FiniteFringe.death_downsets(H)
    prefix = isup ? "U" : "D"
    id = id === nothing ? (isempty(supports) ? nothing : 1) : Int(id)
    if id === nothing
        return _inspection_text_panel("No $(isup ? "upsets" : "downsets")", ["This presentation has no $family summands."])
    end
    support = supports[id]
    membership = [FiniteFringe.contains(support, q) for q in 1:nvertices(FiniteFringe.ambient_poset(H))]
    title = "$prefix$id: $(isup ? "upset column" : "downset row") support"
    meta = (; family, support_id=id, membership, support_coordinate=:finite_vertex)
    if region === nothing
        return _presentation_matrix_panel(reshape(Int.(membership), 1, :), title,
            ["member?"], ["q$q" for q in eachindex(membership)], limit;
            subtitle="1 = member; 0 = absent. Actual finite vertex IDs.",
            metadata=merge(meta, (; matrix_corner="support / vertex")))
    end
    layers = _presentation_support_layers(region.metadata.geometry, membership)
    queries = region.metadata.query_readout
    query_label_layer = nothing
    if !isempty(queries)
        points = NTuple{2,Float64}[q.display_point for q in queries]
        same_point = length(queries) == 2 && queries[1].point == queries[2].point
        marker_points = same_point ? points[1:1] : points
        for (i, point) in enumerate(marker_points)
            role = same_point ? :both : length(queries) == 1 ? :selected : i == 1 ? :source : :target
            push!(layers, PointLayer([point], _VisualRole(role), 0.95, 14.0))
        end
        labels = same_point ?
            ["source = target -> $(queries[1].region_id)$(queries[1].inside_viewport ? "" : " (outside view)")"] :
            ["$(length(queries) == 1 ? "q" : i == 1 ? "source" : "target") -> $(q.region_id)$(q.inside_viewport ? "" : " (outside view)")" for (i,q) in enumerate(queries)]
        push!(layers, TextLayer(labels, marker_points, _VisualRole(:foreground), 10.0))
        query_label_layer = length(layers)
    end
    subtitle = "Labels: + member; - absent.\nMembership on the actual encoding regions.\nFiber edges: solid included; dashed excluded;\ndotted at the viewing-window boundary."
    region.metadata.geometry.has_unrepresented_area &&
        (subtitle *= "\n?: unrepresented label 0;\nsupport membership is unknown.")
    region.metadata.geometry.geometry_kind === :nearest_lattice_tiles &&
        (subtitle *= "\nInteger fibers: nearest-lattice tiles;\nties round-to-even.")
    isempty(region.metadata.warnings) || (subtitle *= "\nDrawing precision warning: inspect\nexact geometry/query metadata.")
    return VisualizationSpec(:presentation_support; title, subtitle, layers, axes=region.axes,
        metadata=merge(region.metadata, meta, (; legend_position=:none, query_label_layer)))
end

function _presentation_active_panel(s, name, limit)
    IR = IndicatorResolutions
    d = IR.presentation_summary(s).dimension
    return _presentation_matrix_panel(IR.presentation_matrix(s), "$name: active block",
        ["D$i" for i in IR.active_rows(s)], ["U$j" for j in IR.active_columns(s)], limit;
        subtitle="Vertex $(IR.presentation_vertex(s)); image dimension $d. Active does not mean nonzero.",
        metadata=(; stalk=s, matrix_corner="downset / upset"))
end

function _presentation_basis_panel(s, name, limit)
    IR = IndicatorResolutions
    B = IR.image_basis(s)
    B === nothing && return _inspection_text_panel("$name: embedded image basis",
        ["Not computed. The default query computes rank only.",
         "Select this stalk with basis=true to inspect its image embedding."])
    return _presentation_matrix_panel(B, "$name: embedded image basis",
        ["D$i" for i in IR.active_rows(s)], ["b$j" for j in axes(B,2)], limit;
        subtitle="Columns span the active block's image in downset coordinates.",
        metadata=(; stalk=s, matrix_corner="downset / image basis",
            empty_matrix_reason="the image has no basis vectors",
            matrix_row_heading="Active downset coordinates"))
end

function _presentation_full_panel(H, stalks, limit)
    IR = IndicatorResolutions
    A = FiniteFringe.fringe_coefficients(H)
    row_labels = ["D$i" for i in axes(A,1)]
    column_labels = ["U$j" for j in axes(A,2)]
    displayed_rows = 1:min(size(A,1), limit[1])
    displayed_columns = 1:min(size(A,2), limit[2])
    cell_roles = fill(:inactive, length(displayed_rows), length(displayed_columns))
    row_roles, column_roles = fill(:inactive, size(A,1)), fill(:inactive, size(A,2))
    n = length(stalks)
    state_role(s, t) = s && t ? :both : s ? :source : t ? :target : :inactive
    for i in axes(A,1)
        active = [s !== nothing && i in IR.active_rows(s) for s in stalks]
        row_roles[i] = n == 1 ? (active[1] ? :selected : :inactive) : n == 2 ? state_role(active...) : :inactive
    end
    for j in axes(A,2)
        active = [s !== nothing && j in IR.active_columns(s) for s in stalks]
        column_roles[j] = n == 1 ? (active[1] ? :selected : :inactive) : n == 2 ? state_role(active...) : :inactive
    end
    for j in displayed_columns, i in displayed_rows
        active = [s !== nothing && i in IR.active_rows(s) && j in IR.active_columns(s) for s in stalks]
        cell_roles[i,j] = n == 1 ? (active[1] ? :selected : :inactive) : n == 2 ? state_role(active...) : :inactive
    end
    subtitle = n == 1 ? "Selected rows and columns define the active block, including its zeros." :
        n == 2 ? "Row and column labels identify source, target, both active blocks, or inactive coordinates." :
        "Rows index downsets; columns index upsets. Coefficients are field elements."
    return _presentation_matrix_panel(A, "Full coefficient matrix Phi", row_labels, column_labels, limit;
        subtitle, metadata=(; matrix_corner="downset / upset", row_roles, column_roles, cell_roles))
end

function _presentation_fibers(H, selection; basis=false)
    IR = IndicatorResolutions
    P = FiniteFringe.ambient_poset(H)
    stalks = Union{Nothing,IR.PresentationStalk}[]
    map = nothing
    relation = :not_selected
    if selection.vertex !== nothing
        q = selection.vertex
        push!(stalks, q == 0 ? nothing : IR.presentation_stalk(H; vertex=q, basis=basis === true))
        relation = q == 0 ? :outside : :stalk
    elseif selection.pair !== nothing
        u, v = selection.pair
        relation = u == 0 || v == 0 ? :outside :
            selection.parameter_relation in (:incomparable, :reverse_comparable) ? selection.parameter_relation :
            leq(P,u,v) ? (u == v ? :equal : :comparable) :
            leq(P,v,u) ? :reverse_comparable : :incomparable
        if relation in (:equal, :comparable)
            map = IR.presentation_map(H; source=u, target=v)
            append!(stalks, (IR.source_stalk(map), IR.target_stalk(map)))
        else
            append!(stalks, (u == 0 ? nothing : IR.presentation_stalk(H; vertex=u),
                            v == 0 ? nothing : IR.presentation_stalk(H; vertex=v)))
        end
    end
    return (; stalks, map, relation)
end

function _presentation_visual_spec(obj; vertex=nothing, pair=nothing, point=nothing,
                                   parameter_pair=nothing, box=nothing, basis=nothing,
                                   upset=nothing, downset=nothing, matrix_limit=(12,12),
                                   prepared=nothing, selection=nothing, presentation_data=nothing, graph=nothing)
    IR = IndicatorResolutions
    H = _inspection_presentation(obj)
    P, field = FiniteFringe.ambient_poset(H), FiniteFringe.field(H)
    selection = selection === nothing ? _inspection_selection(obj; vertex, pair, point, parameter_pair) : selection
    geometry = obj isa EncodingResult && _inspection_has_geometry(obj)
    region = geometry ? _inspection_region_panel(obj, selection; box,
        prepared=prepared === nothing ? nothing : prepared.geometry) : nothing
    panels = VisualizationSpec[
        _presentation_support_panel(H, :upset, upset, region, matrix_limit),
        _presentation_support_panel(H, :downset, downset, region, matrix_limit)]
    data = presentation_data === nothing ? _presentation_fibers(H, selection; basis=basis === true) : presentation_data
    stalks, map, relation = data.stalks, data.map, data.relation
    graph === nothing || pushfirst!(panels, graph)
    push!(panels, _presentation_full_panel(H, stalks, matrix_limit))
    if isempty(stalks)
        push!(panels, _inspection_text_panel("From a presentation to a stalk",
            ["Select vertex=q (or point=x for planar encodings).",
             "Membership selects rows and columns of Phi.",
             "The image of that active block is the stalk.",
             "Select pair=(u,v) to see an induced map.",
             "No selected presentation ranks or image bases have been computed."]))
    else
        for (i,s) in enumerate(stalks)
            name = length(stalks) == 1 ? "Selected stalk" : i == 1 ? "Source" : "Target"
            push!(panels, s === nothing ? _inspection_text_panel("$name: outside",
                ["Label 0 has no represented stalk.", "This is not a represented zero space."]) :
                _presentation_active_panel(s, name, matrix_limit))
        end
        if length(stalks) == 1
            s = only(stalks)
            s === nothing || push!(panels, _presentation_basis_panel(s, "Selected stalk", matrix_limit))
            push!(panels, _inspection_text_panel("Image, not the whole target",
                ["Phi_q maps active upset coordinates to active downset coordinates.",
                 "M_q is its image. Its dimension is rank(Phi_q).",
                 "An active coefficient may be zero, so support counts alone do not give dimension.",
                 "Image bases are embedded in downset coordinates."]))
        elseif map === nothing
            push!(panels, _inspection_text_panel("No forward map: $relation",
                ["The selected parameters are outside or not ordered in this direction.",
                 "No map exists here; this is different from a zero matrix.",
                 "The represented active blocks are shown without image bases."]))
        else
            s, t = IR.source_stalk(map), IR.target_stalk(map)
            C, R = IR.induced_map(map), IR.ambient_projection(map)
            push!(panels, _presentation_matrix_panel(C, "Induced map C",
                ["b$i @ target" for i in axes(C,1)], ["b$j @ source" for j in axes(C,2)], matrix_limit;
                subtitle="B_target * C = R * B_source. Image coordinates of this presentation."))
            push!(panels, _presentation_basis_panel(s, "Source", matrix_limit))
            push!(panels, _presentation_basis_panel(t, "Target", matrix_limit))
            push!(panels, _presentation_matrix_panel(R, "Ambient downset projection R",
                ["D$i" for i in IR.active_rows(t)], ["D$j" for j in IR.active_rows(s)], matrix_limit;
                subtitle="Keep surviving downset coordinates; discard those that have died.",
                metadata=(; matrix_corner="target downset / source downset")))
        end
    end
    return VisualizationSpec(:presentation_inspector; title="From indicator supports to spaces and maps",
        subtitle="Finite fringe image over $(_inspection_field_label(field)); embedded presentation coordinates",
        panels, metadata=(; object=obj isa EncodingResult ? :encoding_result : :fringe_module,
            selection, relation, stalks, presentation_map=map, field,
            category=:finite_poset_representations, basis_convention=:embedded_presentation_image,
            defined=map !== nothing, selected_fibers_only=true, materializes_module=false,
            panel_columns=isempty(stalks) ? 2 : 3,
            figure_size=(isempty(stalks) ? 1280 : 1560, map === nothing ? 1000 : 1420)),
        interaction=_default_interaction(labels=true))
end
