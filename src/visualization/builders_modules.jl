# Finite-poset modules: schematic covers, selected stalks/maps, and encoding panels.

available_visuals(::AbstractPoset) = (:hasse,)
available_visuals(::Modules.PModule) = (:hasse, :module_inspector, :rank_section)

_inspection_poset(P::AbstractPoset) = P
_inspection_poset(M::Modules.PModule) = M.Q
_inspection_poset(enc::EncodingResult) = Results.encoding_poset(enc)
_inspection_field(M::Modules.PModule) = M.field
_inspection_field(enc::EncodingResult) = Results.provenance(enc).field
_inspection_dimensions(M::Modules.PModule) = dimensions(M).stalks
_inspection_dimensions(enc::EncodingResult) = copy(dimensions(enc))

_inspection_classifier(pi) = pi
_inspection_classifier(pi::CompiledEncoding) = _inspection_classifier(encoding_map(pi))
function _inspection_has_geometry(enc::EncodingResult)
    pi = _inspection_classifier(encoding_map(enc))
    return pi isa _GeometricEncoding2D && :regions in available_visuals(pi)
end

function _inspection_field_label(field)
    field isa CoreModules.QQField && return "QQ"
    field isa CoreModules.PrimeField && return "F$(field.p)"
    field isa CoreModules.RealField && return string(CoreModules.coeff_type(field),
        " (numerical rank; atol=", field.atol, ", rtol=", field.rtol, ")")
    return string(field)
end

_visual_request_keywords(::Union{AbstractPoset,Modules.PModule}, kind::Symbol) =
    kind === :rank_section ? (:source, :target, :vertex, :matrix_limit) :
    kind === :module_inspector ? (:vertex, :pair, :matrix_limit) : (:vertex, :pair)
function _visual_request_keywords(enc::EncodingResult, kind::Symbol)
    kind === :rank_section && return (:source, :target, :vertex, :point, :box, :matrix_limit)
    kind === :presentation_inspector && return _presentation_request_keywords(enc)
    kind === :hasse && return (:vertex, :pair)
    if kind === :module_inspector
        return _inspection_has_geometry(enc) ?
            (:vertex, :pair, :point, :parameter_pair, :box, :matrix_limit) :
            (:vertex, :pair, :matrix_limit)
    end
    return _visual_request_keywords(encoding_map(enc), kind)
end

function _check_module_selection!(issues, obj, kind; kwargs...)
    n = nvertices(_inspection_poset(obj))
    options = (:vertex, :pair, :point, :parameter_pair)
    count(k -> get(kwargs, k, nothing) !== nothing, options) <= 1 ||
        push!(issues, "Select one of vertex, pair, point, or parameter_pair, not several.")
    valid_id(q) = q isa Integer && !(q isa Bool) && 1 <= q <= n
    vertex = get(kwargs, :vertex, nothing)
    vertex === nothing || valid_id(vertex) || push!(issues, "vertex must be a poset ID in 1:$n.")
    pair = get(kwargs, :pair, nothing)
    if pair !== nothing
        (pair isa Tuple || pair isa AbstractVector) && length(pair) == 2 && all(valid_id, pair) ||
            push!(issues, "pair must contain two poset IDs in source, target order, each in 1:$n.")
    end
    limit = get(kwargs, :matrix_limit, (12, 12))
    (limit isa Tuple || limit isa AbstractVector) && length(limit) == 2 &&
        all(x -> x isa Integer && !(x isa Bool) && x > 0, limit) ||
        push!(issues, "matrix_limit must contain positive row and column counts.")
    get(kwargs, :box, nothing) === nothing || _visual_box_2d(kwargs[:box])
    point = get(kwargs, :point, nothing)
    point === nothing || foreach(_drawing_point, _collect_query_points(; point))
    pp = get(kwargs, :parameter_pair, nothing)
    if pp !== nothing
        if (pp isa Tuple || pp isa AbstractVector) && length(pp) == 2
            foreach(p -> foreach(_drawing_point, _collect_query_points(; point=p)), pp)
        else
            push!(issues, "parameter_pair must contain two planar points in source, target order.")
        end
    end
    return issues
end

function _append_visual_request_issues!(issues::Vector{String}, obj::Union{AbstractPoset,Modules.PModule}, kind::Symbol; kwargs...)
    kind === :rank_section && return _check_rank_section!(issues, obj; kwargs...)
    return _check_module_selection!(issues, obj, kind; kwargs...)
end

function _append_visual_request_issues!(issues::Vector{String}, enc::EncodingResult, kind::Symbol; kwargs...)
    kind === :rank_section && return _check_rank_section!(issues, enc; kwargs...)
    kind === :presentation_inspector && return _check_presentation_selection!(issues, enc; kwargs...)
    if kind in (:hasse, :module_inspector)
        return _check_module_selection!(issues, enc, kind; kwargs...)
    end
    return _append_visual_request_issues!(issues, encoding_map(enc), kind; kwargs...)
end

function _visual_spec(P::AbstractPoset, kind::Symbol; vertex=nothing, pair=nothing)
    kind === :hasse || throw(ArgumentError("A finite poset supports kind=:hasse."))
    return _hasse_spec(P; vertex, pair=pair === nothing ? nothing : (Int(pair[1]), Int(pair[2])))
end

function _visual_spec(M::Modules.PModule, kind::Symbol; kwargs...)
    kind === :rank_section && return _rank_section_spec(M; kwargs...)
    return _module_visual_spec(M, kind; kwargs...)
end

# Ambient order must be tested before passing to finite labels: even equal
# labels can come from incomparable original parameters.
function _inspection_parameter_relation(pi, x, y)
    orientation = pi isa GridEncodingMap ? pi.orientation : (1, 1)
    xy = all(orientation[i] == 1 ? x[i] <= y[i] : x[i] >= y[i] for i in 1:2)
    yx = all(orientation[i] == 1 ? y[i] <= x[i] : y[i] >= x[i] for i in 1:2)
    return xy ? (yx ? :equal : :comparable) : (yx ? :reverse_comparable : :incomparable)
end

function _inspection_selection(obj; vertex=nothing, pair=nothing, point=nothing, parameter_pair=nothing)
    pi = obj isa EncodingResult ? _inspection_classifier(encoding_map(obj)) : nothing
    query_points = Tuple{Real,Real}[]
    parameter_relation = :not_applicable
    if point !== nothing
        query_points = _collect_query_points(; point)
        vertex = _visual_locate(pi, only(query_points))
    elseif parameter_pair !== nothing
        query_points = _collect_query_points(; points=parameter_pair)
        x, y = query_points
        pair = (_visual_locate(pi, x), _visual_locate(pi, y))
        parameter_relation = _inspection_parameter_relation(pi, x, y)
    end
    vertex = vertex === nothing ? nothing : Int(vertex)
    pair = pair === nothing ? nothing : (Int(pair[1]), Int(pair[2]))
    return (; vertex, pair, query_points, parameter_relation)
end

function _inspection_text_panel(title, lines; subtitle="", metadata=NamedTuple())
    return VisualizationSpec(:stalk_readout; title, subtitle,
        layers=AbstractVisualizationLayer[TextLayer(String.(lines),
            [(0.0, -Float64(i)) for i in eachindex(lines)], :black, 13.0)],
        metadata=merge((; panel_style=:text_only), metadata))
end

_matrix_coefficient_text(x) = string(x)
_matrix_coefficient_text(x::Rational) = denominator(x) == 1 ? string(numerator(x)) :
    string(numerator(x), "/", denominator(x))

function _inspection_readout(obj, dims, selection, label_relation, matrix_limit)
    field = _inspection_field(obj)
    field_label = _inspection_field_label(field)
    common = (; field, basis_convention=:module_coordinates,
                exact=!(field isa CoreModules.RealField))
    if selection.vertex !== nothing
        q = selection.vertex
        if q == 0
            info = merge(common, (; kind=:outside, vertex=0, dimension=nothing))
            return _inspection_text_panel("Outside the represented encoding",
                ["Classifier label 0 has no represented stalk.",
                 "This is different from a represented zero-dimensional space."]; subtitle=field_label), info
        end
        d = dims[q]
        info = merge(common, (; kind=:stalk, vertex=q, dimension=d))
        lines = ["Vertex $q: dimension $d", "Coordinates in the stored module's basis.",
                 d == 0 ? "The zero space has an empty basis." : "Coordinate basis: e1 through e$d.",
                 "These are not source cycles or presentation image embeddings."]
        return _inspection_text_panel("Space at vertex $q", lines; subtitle=field_label), info
    elseif selection.pair === nothing
        info = merge(common, (; kind=:overview, dimensions=dims))
        return _inspection_text_panel("Inspect a space or map",
            ["Select vertex=q to inspect one space.", "Select pair=(u,v) to inspect its structure map.",
             "Solid arrows in the poset are cover relations.", "No structure matrices were queried."];
            subtitle=field_label), info
    end
    u, v = selection.pair
    outside = u == 0 || v == 0
    ambient_ordered = selection.parameter_relation in (:not_applicable, :equal, :comparable)
    defined = !outside && ambient_ordered && label_relation in (:cover, :comparable, :equal)
    relation = outside ? :outside : !ambient_ordered ? selection.parameter_relation : label_relation
    info = merge(common, (; kind=:map, source=u, target=v, relation, label_relation,
        parameter_relation=selection.parameter_relation, defined,
        source_dimension=u == 0 ? nothing : dims[u], target_dimension=v == 0 ? nothing : dims[v]))
    if !defined
        reason = outside ? "A selected parameter has no represented label." :
                 relation === :reverse_comparable ? "Only the reverse direction is ordered." :
                 "The selected source and target are incomparable."
        lines = [reason, "There is no structure map in this direction; this is not a zero matrix."]
        selection.parameter_relation === :not_applicable ||
            push!(lines, "Finite-label relations do not establish order of original parameters.")
        info = merge(info, (; matrix=nothing, rank=nothing, kernel_dimension=nothing, image_dimension=nothing))
        return _inspection_text_panel("No map: $u -> $v", lines; subtitle=field_label), info
    end
    M = obj isa EncodingResult ? Results.encoding_module(obj) : obj
    # Map queries may return shared cached matrices. Own the inspection snapshot.
    A = copy(Modules.structure_map(M; source=u, target=v))
    r = isempty(A) ? 0 : FieldLinAlg.rank(field, A)
    rows = 1:min(size(A, 1), matrix_limit[1])
    cols = 1:min(size(A, 2), matrix_limit[2])
    entries = [_matrix_coefficient_text(A[i, j]) for i in rows, j in cols]
    truncated = length(rows) < size(A, 1) || length(cols) < size(A, 2)
    info = merge(info, (; matrix=A, matrix_size=size(A), rank=r, kernel_dimension=size(A, 2)-r,
        image_dimension=r, cokernel_dimension=size(A, 1)-r,
        displayed_rows=rows, displayed_columns=cols, truncated,
        atol=field isa CoreModules.RealField ? field.atol : nothing,
        rtol=field isa CoreModules.RealField ? field.rtol : nothing))
    subtitle = "$field_label\n$(size(A,1)) x $(size(A,2)); rank $r; kernel $(size(A,2)-r); image $r"
    truncated && (subtitle *= "\nShowing $(length(rows)) x $(length(cols)) entries; full matrix retained in metadata.")
    panel = VisualizationSpec(:structure_matrix; title="Map $u -> $v", subtitle,
        layers=AbstractVisualizationLayer[MatrixLayer(entries,
            ["e$i @ $v" for i in rows], ["e$j @ $u" for j in cols])],
        metadata=merge(info, (; panel_style=:matrix,
            row_roles=fill(:target, length(rows)), column_roles=fill(:source, length(cols)))))
    return panel, info
end

function _inspection_region_panel(enc, selection; box=nothing, prepared=nothing)
    pi = _inspection_classifier(encoding_map(enc))
    query_roles = length(selection.query_points) == 2 ? (:source, :target) : (:selected,)
    region = isempty(selection.query_points) ? _visual_spec(pi, :region_labels; box, prepared_geometry=prepared) :
        _visual_spec(pi, :query_overlay; points=selection.query_points, box, prepared_geometry=prepared, query_roles)
    subtitle = "Solid/dashed: included/excluded; dotted: window cut\nColors and IDs match the finite-poset panel"
    pi isa ZnEncodingMap && (subtitle *= "\nInteger fibers: nearest-lattice tiles; ties round-to-even")
    isempty(region.metadata.warnings) || (subtitle *= "\nDrawing precision warning: inspect exact query/geometry metadata.")
    return VisualizationSpec(region.kind; title="Parameter regions", subtitle,
        layers=region.layers, axes=region.axes, interaction=region.interaction,
        metadata=merge(region.metadata, (; legend_position=:none)))
end

function _module_visual_spec(obj, kind::Symbol; vertex=nothing, pair=nothing,
                             point=nothing, parameter_pair=nothing, box=nothing,
                             matrix_limit=(12, 12), prepared=nothing, selection=nothing,
                             inspection_data=nothing, graph=nothing, materialized_before=nothing)
    P = _inspection_poset(obj)
    selection = selection === nothing ? _inspection_selection(obj; vertex, pair, point, parameter_pair) : selection
    dims = prepared === nothing ? _inspection_dimensions(obj) : prepared.dims
    length(dims) == nvertices(P) || throw(ArgumentError("Module dimensions do not match its poset."))
    field = _inspection_field(obj)
    graph_pair = selection.pair === nothing || any(iszero, selection.pair) ? nothing : selection.pair
    graph_vertex = selection.vertex === 0 ? nothing : selection.vertex
    graph = graph === nothing ? _hasse_spec(P; dims, vertex=graph_vertex, pair=graph_pair,
        field_label=_inspection_field_label(field), prepared=prepared === nothing ? nothing : prepared.hasse) : graph
    kind === :hasse && return graph
    kind === :module_inspector || throw(ArgumentError("Unsupported module visualization kind=$kind."))
    was_materialized = materialized_before === nothing ?
        (obj isa EncodingResult ? Results.result_summary(obj).materialized : true) : materialized_before
    readout, inspection = inspection_data === nothing ?
        _inspection_readout(obj, dims, selection, graph.metadata.relation, matrix_limit) : inspection_data
    panels = VisualizationSpec[graph, readout]
    geometry = obj isa EncodingResult && _inspection_has_geometry(obj)
    geometry && pushfirst!(panels, _inspection_region_panel(obj, selection; box,
        prepared=prepared === nothing ? nothing : prepared.geometry))
    materialized = obj isa EncodingResult ? Results.result_summary(obj).materialized : true
    subtitle = "Finite-poset representation over $(_inspection_field_label(field)); arrows point from source to target"
    selection.parameter_relation in (:incomparable, :reverse_comparable) &&
        (subtitle *= "\nOriginal parameters are not ordered in the selected direction; no ambient map is shown.")
    !was_materialized && materialized &&
        (subtitle *= "\nThis map selection materialized the lazy encoded module.")
    return VisualizationSpec(:module_inspector; title="Spaces and structure maps", subtitle, panels,
        metadata=(; object=obj isa EncodingResult ? :encoding_result : :pmodule,
            selection, inspection, field, category=:finite_poset_representations,
            module_materialized_before=was_materialized, module_materialized_after=materialized,
            panel_columns=length(panels), figure_size=(geometry ? 1560 : 1100, 650)),
        interaction=_default_interaction(labels=true))
end
