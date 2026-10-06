# Anchored ordinary pair ranks. Geometry is clipped in the original parameter
# order; a finite classifier label alone never establishes ambient comparability.

function _rank_section_anchors(obj, source, target)
    (source === nothing) != (target === nothing) ||
        throw(ArgumentError("Fix exactly one source or target anchor."))
    from = source !== nothing
    supplied = from ? source : target
    planar(p) = (p isa Tuple || p isa AbstractVector) && length(p) == 2 &&
        all(x -> x isa Real && !(x isa Bool) && isfinite(x), p)
    finite = supplied isa Integer || obj isa Modules.PModule
    anchors = if supplied isa Integer
        [supplied]
    elseif finite
        (supplied isa Tuple || supplied isa AbstractVector) ||
            throw(ArgumentError("Finite anchors must be a vertex ID or a collection of IDs."))
        collect(supplied)
    else
        _inspection_has_geometry(obj) ||
            throw(ArgumentError("Parameter rank sections require a supported planar classifier; use an integer finite label instead."))
        planar(supplied) ? [supplied] :
            (supplied isa Tuple || supplied isa AbstractVector) ? collect(supplied) : []
    end
    isempty(anchors) && throw(ArgumentError("At least one anchor is required."))
    if finite
        n = nvertices(_inspection_poset(obj))
        all(a -> a isa Integer && !(a isa Bool) && 1 <= a <= n, anchors) ||
            throw(ArgumentError("Finite anchors must be vertex IDs in 1:$n."))
        return (; from, domain=:finite, anchors=Int.(anchors))
    end
    all(planar, anchors) || throw(ArgumentError("Parameter anchors must be a planar point or a collection of planar points."))
    foreach(_drawing_point, anchors)
    return (; from, domain=:parameter, anchors=_visual_point.(anchors))
end

function _check_rank_section!(issues, obj; source=nothing, target=nothing,
                              vertex=nothing, point=nothing, box=nothing, matrix_limit=(12,12))
    request = _rank_section_anchors(obj, source, target)
    _check_module_selection!(issues, obj, :rank_section; vertex, point, box, matrix_limit)
    if request.domain === :finite
        point === nothing || push!(issues, "Use vertex to select the other endpoint of a finite-label section.")
        box === nothing || push!(issues, "A finite-label section has schematic coordinates and does not accept box.")
    else
        vertex === nothing || push!(issues, "Use point to select the other endpoint of a parameter section.")
    end
    return issues
end

function _rank_section_row(obj, dims, anchor::Int, from::Bool)
    P = _inspection_poset(obj)
    ranks = Union{Nothing,Int}[nothing for _ in dims]
    queries = 0
    M = nothing
    if anchor != 0
        for q in eachindex(dims)
            u, v = from ? (anchor, q) : (q, anchor)
            leq(P, u, v) || continue
            ranks[q] = if u == v
                dims[u]
            elseif dims[u] == 0 || dims[v] == 0
                0
            else
                M === nothing && (M = obj isa EncodingResult ? Results.encoding_module(obj) : obj)
                queries += 1
                InvariantCore.rank_map(M, u, v)
            end
        end
    end
    return (; anchor_label=anchor, from, ranks, rank_queries=queries,
        retained_matrices=0, all_pairs_table=false)
end

struct _RankSectionClassifier{P,A}
    classifier::P
    anchor::A
    from::Bool
end
function _visual_locate(section::_RankSectionClassifier, point)
    x, y = section.from ? (section.anchor, point) : (point, section.anchor)
    _inspection_parameter_relation(section.classifier, x, y) in (:equal, :comparable) || return -1
    return _visual_locate(section.classifier, point)
end

function _rank_section_geometry(pi, geometry, anchor, from)
    # Live pointer selections retain their supplied floating coordinates. Use
    # their exact represented values for clipping, just as static requests do.
    anchor = _visual_point(anchor)
    section = _RankSectionClassifier(pi, anchor, from)
    orientation = pi isa GridEncodingMap ? pi.orientation : (1,1)
    # The comparable orthant is the intersection a_i*x <= b_i.
    normals = ntuple(i -> ntuple(j -> i == j ? (from ? -orientation[i] : orientation[i]) : 0, 2), 2)
    bounds = ntuple(i -> normals[i][i] * anchor[i], 2)
    components = NamedTuple[]
    for c in geometry.components
        vertices = c.vertices
        for i in 1:2
            vertices = _clip_visual_halfspace(vertices, normals[i], bounds[i])
        end
        isempty(vertices) && continue
        center = ntuple(i -> sum(p[i] for p in vertices) / length(vertices), 2)
        _visual_locate(section, center) == c.region_id || continue
        push!(components, _visual_component(section, c.region_id, vertices, geometry.box))
    end
    lo, hi = geometry.box
    rectangle = _VisualPoint[(lo[1],lo[2]), (hi[1],lo[2]), (hi[1],hi[2]), (lo[1],hi[2])]
    # Two disjoint strips fill the complement, including unrepresented holes.
    # Equality edges are excluded by the exact classifier above.
    for i in 1:2
        vertices = rectangle
        for j in 1:(i-1)
            vertices = _clip_visual_halfspace(vertices, normals[j], bounds[j])
        end
        vertices = _clip_visual_halfspace(vertices, .-normals[i], -bounds[i])
        isempty(vertices) && continue
        center = ntuple(j -> sum(p[j] for p in vertices) / length(vertices), 2)
        _visual_locate(section, center) == -1 || continue
        push!(components, _visual_component(section, -1, vertices, geometry.box))
    end
    return _geometry_result(components, geometry.box; geometry_kind=:anchored_rank_section)
end

# A common sequential scale across anchors; numeric labels disambiguate close
# shades and remain readable in grayscale. Missing/order masks are not ranks.
function _rank_section_color(r, maximum_rank)
    r === nothing && return _VisualRole(:unrepresented)
    r == 0 && return _VisualRole(:background)
    # Keep positive ranks darker than the no-map and unknown fills after the
    # shared renderer converts them to luminance for grayscale exports.
    palette = (:lightsteelblue3, :lightskyblue3, :skyblue3, :steelblue3)
    return palette[clamp(1 + round(Int, (length(palette)-1) * r / max(1, maximum_rank)), 1, length(palette))]
end

function _rank_section_legend(rows, maximum_rank, domain)
    values = sort!(unique(Int[r for row in rows for r in row.ranks if r !== nothing]))
    # Keep the scale key compact even for high-dimensional stalks. Region
    # labels and retained rows still give every exact integer rank.
    length(values) > 6 && (values = values[unique(round.(Int, range(1,length(values);length=6)))])
    entries = NamedTuple[(; label="rank $r", color=_rank_section_color(r, maximum_rank), style=:patch) for r in values]
    if domain === :parameter || any(row -> any(isnothing, row.ranks), rows)
        push!(entries, (; label="not ordered (no map)", color=_VisualRole(:absent), style=:patch))
    end
    domain === :parameter && push!(entries,
        (; label="unrepresented (unknown)", color=_VisualRole(:unrepresented), style=:patch))
    return _default_legend(; visible=true, entries)
end

function _rank_section_panel(obj, prepared, anchor, domain, row, selection, maximum_rank)
    fixed = row.from ? "source p" : "target q"
    varying = row.from ? "target q" : "source p"
    anchor_text = domain === :parameter ? "(" * join(_matrix_coefficient_text.(anchor), ", ") * ")" : string(anchor)
    title = "Fixed $fixed = $anchor_text"
    d = row.anchor_label == 0 ? "unknown" : string(prepared.dims[row.anchor_label])
    subtitle = "Vary $varying; anchor dimension $d"
    if domain === :finite
        graph = _hasse_spec(_inspection_poset(obj); dims=prepared.dims, prepared=prepared.hasse,
            vertex=row.anchor_label == 0 ? nothing : row.anchor_label,
            pair=selection.pair === nothing || any(iszero, selection.pair) ? nothing : selection.pair)
        layers = AbstractVisualizationLayer[]
        for layer in graph.layers
            if layer isa PointLayer && layer.color isa _VisualRole && layer.color.kind === :categorical
                r = row.ranks[layer.color.index]
                color = r === nothing ? _VisualRole(:absent) : _rank_section_color(r, maximum_rank)
                push!(layers, PointLayer(layer.points, color, layer.alpha, layer.markersize))
            elseif layer isa TextLayer && length(layer.labels) == length(row.ranks)
                labels = [layer.labels[i] * (row.ranks[i] === nothing ? "\nno map" : "\nrank = $(row.ranks[i])") for i in eachindex(row.ranks)]
                push!(layers, TextLayer(labels, layer.positions, layer.color, layer.textsize))
            else
                push!(layers, layer)
            end
        end
        return VisualizationSpec(:rank_section_hasse; title, subtitle=subtitle * "\nSchematic finite order; not parameter coordinates",
            layers, axes=graph.axes, metadata=merge(graph.metadata, (; row, anchor, domain, legend_position=:none)))
    end
    pi = _inspection_classifier(encoding_map(obj))
    geometry = _rank_section_geometry(pi, prepared.geometry, anchor, row.from)
    colors = Dict{Int,Union{Symbol,_VisualRole}}(-1 => _VisualRole(:absent), 0 => _VisualRole(:unrepresented))
    for i in eachindex(row.ranks)
        colors[i] = _rank_section_color(row.ranks[i], maximum_rank)
    end
    layers = _algebra_support_layers(geometry, colors)
    # Keep the numerical scale identical to its legend. Zero-valued faces
    # still need visible boundary ownership, including isolated points.
    for (i, layer) in pairs(layers)
        if layer isa PolygonLayer
            layers[i] = PolygonLayer(layer.polygons, layer.fill_color, layer.stroke_color, 1.0, layer.linewidth)
        elseif layer isa SegmentLayer && layer.color === _VisualRole(:background)
            layers[i] = SegmentLayer(layer.segments, _VisualRole(:border), layer.alpha, layer.linewidth, layer.linestyle)
        elseif layer isa PointLayer && layer.color === _VisualRole(:background) && layer.markersize != 3.0
            layers[i] = PointLayer(layer.points, _VisualRole(:border), layer.alpha, layer.markersize)
        end
    end
    labels, positions = String[], NTuple{2,Float64}[]
    labelled = Dict{Int,NamedTuple}()
    marks = _drawing_point.((anchor, selection.query_points...))
    spans = (geometry.axes.xlimits[2] - geometry.axes.xlimits[1],
             geometry.axes.ylimits[2] - geometry.axes.ylimits[1])
    clearance(p) = minimum(sum(((p[j] - mark[j]) / spans[j])^2 for j in 1:2) for mark in marks)
    area(c) = abs(sum(c.vertices[i][1] * c.vertices[mod1(i+1,length(c.vertices))][2] -
        c.vertices[mod1(i+1,length(c.vertices))][1] * c.vertices[i][2] for i in eachindex(c.vertices)))
    width(c) = maximum(first, c.vertices) - minimum(first, c.vertices)
    for c in geometry.components
        c.dimension == 2 && c.region_id > 0 || continue
        prepared.dims[c.region_id] > 0 || continue
        previous = get(labelled, c.region_id, nothing)
        if previous === nothing || (area(c), width(c)) > (area(previous), width(previous))
            labelled[c.region_id] = c
        end
    end
    # A fiber can have many geometric cells. One label in its largest cell
    # avoids repeating the same value across internal subdivision boundaries.
    # Zero stalks need no text: the shared key already identifies rank zero.
    for id in sort!(collect(keys(labelled)))
        c = labelled[id]
        r = row.ranks[c.region_id]
        p = _drawing_point(ntuple(i -> sum(v[i] for v in c.vertices) / length(c.vertices), 2))
        if clearance(p) < 0.07^2
            # Keep the label inside its convex cell and away from a selection
            # marker. These offsets affect drawing positions only.
            candidates = NTuple{2,Float64}[p]
            append!(candidates, (ntuple(j -> p[j]/2 + Float64(v[j])/2, 2) for v in c.vertices))
            p = argmax(clearance, candidates)
        end
        push!(labels, (r === nothing ? "unknown" : string(r)) * "\ndim $(prepared.dims[c.region_id])")
        push!(positions, p)
    end
    push!(layers, TextLayer(labels, positions, _VisualRole(:foreground), 12.0))
    anchor_role = row.from ? :source : :target
    other = length(selection.query_points) == 2 ? selection.query_points[row.from ? 2 : 1] : nothing
    push!(layers, PointLayer([_drawing_point(anchor)], _VisualRole(other == anchor ? :both : anchor_role), 1.0, 15.0))
    if other !== nothing && other != anchor
        push!(layers, PointLayer([_drawing_point(other)], _VisualRole(row.from ? :target : :source), 1.0, 15.0))
    end
    fixed_coordinates = row.from ? (:p1,:p2) : (:q1,:q2)
    varying_coordinates = row.from ? (:q1,:q2) : (:p1,:p2)
    orientation = pi isa GridEncodingMap ? pi.orientation : (1,1)
    axes = merge(geometry.axes, (; xlabel="$(varying_coordinates[1]) ($(orientation[1] == 1 ? "increasing" : "decreasing") order)",
        ylabel="$(varying_coordinates[2]) ($(orientation[2] == 1 ? "increasing" : "decreasing") order)"))
    subtitle *= "\n$(join(fixed_coordinates, ", ")) fixed; labels: rank / dim\nSolid/dashed = included/excluded; dotted = window cut"
    pi isa ZnEncodingMap && (subtitle *= "\nNearest-lattice tiles; ties round-to-even")
    warnings = String[]
    query_collisions = _display_collisions(other === nothing ? [anchor] : [anchor,other])
    if !isempty(geometry.coordinate_collisions) || !isempty(geometry.dimension_collapses)
        push!(warnings, "Distinct exact boundaries coincide at drawing precision; inspect geometry metadata.")
        subtitle *= "\nDrawing precision warning: exact boundaries retained in metadata."
    end
    if !isempty(query_collisions)
        push!(warnings, "Distinct selected parameters coincide at drawing precision.")
        subtitle *= "\nSelected parameters coincide on screen; the map uses their exact coordinates."
    end
    return VisualizationSpec(:rank_section_plane; title, subtitle, layers, axes,
        metadata=(; row, anchor, domain, geometry, fixed_coordinates, varying_coordinates,
            orientation, warnings, query_collisions, minimal_axes=true, legend_position=:none))
end

function _rank_section_snapshot(obj, prepared, anchors, domain, rows, selections, readouts)
    maximum_rank = maximum((r for row in rows for r in row.ranks if r !== nothing); init=0)
    panels = VisualizationSpec[]
    show_readouts = length(anchors) == 1 || any(s -> s.pair !== nothing, selections)
    for i in eachindex(anchors)
        push!(panels, _rank_section_panel(obj, prepared, anchors[i], domain, rows[i], selections[i], maximum_rank))
        if show_readouts
            readout, info = readouts[i]
            if info.kind === :stalk
                row = rows[i]
                fixed, varying = row.from ? ("source", "target") : ("target", "source")
                selector = domain === :parameter ? "point" : "vertex"
                readout = _inspection_text_panel("The $fixed stays fixed",
                    ["Finite label $(row.anchor_label); dimension $(prepared.dims[row.anchor_label]).",
                     "Each rank counts independent vectors reaching the target.",
                     "Choose $selector to inspect a map at a particular $varying."];
                    subtitle=_inspection_field_label(_inspection_field(obj)))
            elseif domain === :parameter && info.kind === :map
                p, q = selections[i].query_points
                text(point) = "(" * join(_matrix_coefficient_text.(point), ", ") * ")"
                readout = VisualizationSpec(readout.kind; title=readout.title,
                    subtitle=readout.subtitle * "\np = $(text(p)); q = $(text(q))",
                    layers=readout.layers, axes=readout.axes, legend=readout.legend,
                    interaction=readout.interaction, metadata=readout.metadata)
            end
            push!(panels, readout)
        end
    end
    from = first(rows).from
    columns = show_readouts ? 2 : min(3, length(anchors))
    nrows = cld(length(panels), columns)
    weights = fill(domain === :parameter ? 450 : 520, nrows)
    return VisualizationSpec(:rank_section; title=from ? "Rank surviving from a source" : "Rank reaching a target",
        subtitle="Ordinary pair rank over $(_inspection_field_label(_inspection_field(obj))); not generalized rank.\nRank zero, unordered pairs and unrepresented points have separate fills.",
        panels, legend=_rank_section_legend(rows, maximum_rank, domain),
        metadata=(; object=obj isa EncodingResult ? :encoding_result : :pmodule,
            category=:finite_poset_representations,
            field=_inspection_field(obj), domain, from, anchors, rank_sections=rows,
            inspections=last.(readouts), inspection=length(readouts) == 1 ? only(readouts)[2] : nothing,
            rank_range=(0,maximum_rank), all_pairs_table=false,
            retained_rank_count=sum(length(row.ranks) for row in rows),
            panel_columns=columns, panel_positions=[((1+div(i-1,columns)):(1+div(i-1,columns)),
                (1+mod(i-1,columns)):(1+mod(i-1,columns))) for i in eachindex(panels)],
            panel_row_weights=weights, figure_size=(columns == 3 ? 1560 : 1280, 160 + sum(weights)), legend_position=:bottom))
end

"""
    visual_spec(obj; kind=:rank_section, source=nothing, target=nothing,
                vertex=nothing, point=nothing, box=nothing, matrix_limit=(12,12))

Fix exactly one endpoint of the ordinary pair rank. On a `PModule`, `source=1`
shows `q -> rank M(1 <= q)` on its schematic finite poset; `target=3` is the
opposite section. A collection of finite IDs gives small multiples.

On a planar `EncodingResult`, `source=(x,y)` fixes both source coordinates
and varies both target coordinates. `target=(x,y)` fixes the target instead.
Use a collection of planar points for small multiples. An integer anchor on
an encoding explicitly selects its finite model; it is not a representative
parameter. For multiple finite anchors, use `encoding_module(enc)`.

Select the other endpoint with `vertex` (finite model) or `point` (parameter
section) to show its actual matrix, endpoint dimensions and rank. Incomparable
and reverse-ordered parameters are masked, even when their finite labels agree.
Label 0 means unrepresented, never a zero space. Grid axis orientations and
exact boundaries are retained; floating conversion is only for drawing.

Each distinct anchor label computes one rank row/column, reusing it across
small multiples with different exact anchors. No all-pairs table is made and
no row matrices are retained. Rank queries may materialize a lazy module;
only the selected map is copied into the readout. `matrix_limit` limits shown
entries, not matrix construction. Finite viewing `box` bounds the drawing,
not the mathematical rank. Numerical fields use their rank tolerances.

For interactive selections use `inspection_session(obj; view=:rank_from)`
or `view=:rank_to`, then `visualize(session)` with WGLMakie loaded.
"""
function _rank_section_spec(obj; source=nothing, target=nothing, vertex=nothing,
                            point=nothing, box=nothing, matrix_limit=(12,12))
    request = _rank_section_anchors(obj, source, target)
    prepared, _ = _inspection_prepare(obj, request.domain === :parameter ? box : nothing)
    before = obj isa EncodingResult ? Results.result_summary(obj).materialized : true
    pi = request.domain === :parameter ? _inspection_classifier(encoding_map(obj)) : nothing
    rows, selections, readouts = NamedTuple[], NamedTuple[], Tuple[]
    cache = Dict{Int,NamedTuple}()
    for anchor in request.anchors
        label = request.domain === :finite ? anchor : _visual_locate(pi, anchor)
        row = get!(cache, label) do
            _rank_section_row(obj, prepared.dims, label, request.from)
        end
        other = request.domain === :finite ? vertex : point
        selection = if other === nothing
            request.domain === :finite ? _inspection_selection(obj; vertex=anchor) : _inspection_selection(obj; point=anchor)
        else
            pair = request.from ? (anchor, other) : (other, anchor)
            request.domain === :finite ? _inspection_selection(obj; pair) : _inspection_selection(obj; parameter_pair=pair)
        end
        graph = _inspection_graph(obj, prepared, selection)
        push!(rows, row); push!(selections, selection)
        push!(readouts, _inspection_readout(obj, prepared.dims, selection, graph.metadata.relation, matrix_limit))
    end
    spec = _rank_section_snapshot(obj, prepared, request.anchors, request.domain, rows, selections, readouts)
    after = obj isa EncodingResult ? Results.result_summary(obj).materialized : true
    return VisualizationSpec(spec.kind; title=spec.title, subtitle=spec.subtitle, panels=spec.panels, legend=spec.legend,
        metadata=merge(spec.metadata, (; selections, rank_queries=sum(row.rank_queries for row in values(cache)),
            module_materialized_before=before, module_materialized_after=after)))
end
