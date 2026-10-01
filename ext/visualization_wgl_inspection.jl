# Live WGL/Bonito controls for backend-independent inspection sessions.

include("visualization_wgl_slices.jl")

const _INSPECTION_UI_HANDLES = WeakKeyDict{Any,Any}()
const _NEXT_INSPECTION_UI_ID = Ref(0)

_inspection_ui(app) = _INSPECTION_UI_HANDLES[app]
function _dispose_inspection_ui!(app; close_session::Bool=true)
    ui = get(_INSPECTION_UI_HANDLES, app, nothing)
    ui === nothing || ui.dispose(; close_session)
    return nothing
end

function _inspection_css_color(style, color)
    rgba = WGLMakie.Makie.to_color(VIZ._visual_color(style, color))
    r, g, b = rgba.r, rgba.g, rgba.b
    if style.palette === :grayscale && !(color isa VIZ._VisualRole)
        r = g = b = 0.2126 * r + 0.7152 * g + 0.0722 * b
    end
    return "rgba($(round(Int, 255 * r)),$(round(Int, 255 * g)),$(round(Int, 255 * b)),$(rgba.alpha))"
end

function _inspection_control_style(style)
    B = WGLMakie.Bonito
    foreground_css = _inspection_css_color(style, VIZ._VisualRole(:foreground))
    background_css = _inspection_css_color(style, VIZ._VisualRole(:background))
    surface_css = _inspection_css_color(style, VIZ._VisualRole(:surface))
    border_css = _inspection_css_color(style, VIZ._VisualRole(:border))
    return B.Styles(
        B.CSS("box-sizing" => "border-box", "font-family" => "'$(style.font)',sans-serif",
            "font-size" => "$(style.fontsize)px", "font-weight" => "400", "line-height" => "1.45",
            "color" => foreground_css, "background-color" => surface_css,
            "padding-left" => "$(style.gap/2)px", "padding-right" => "$(style.gap/2)px",
            "padding-top" => "$(style.gap/2)px", "padding-bottom" => "$(style.gap/2)px",
            "border-width" => "$(style.linewidth_scale)px", "border-style" => "solid",
            "border-color" => border_css, "border-radius" => "$(style.gap/3)px",
            "margin" => "0", "max-width" => "100%", "min-width" => "0", "box-shadow" => "none"),
        B.CSS(":hover", "background-color" => background_css, "box-shadow" => "none"),
        B.CSS(":focus", "outline" => "$(2 * style.linewidth_scale)px solid $foreground_css",
            "outline-offset" => "2px", "box-shadow" => "none"))
end

function _inspection_dom_readout(snapshot; style=VIZ.VisualStyle())
    B = WGLMakie.Bonito
    sections = Any[]
    border = _inspection_css_color(style, VIZ._VisualRole(:border))
    surface = _inspection_css_color(style, VIZ._VisualRole(:surface))
    heading_style = "font-family:'$(style.font)',sans-serif;font-size:$(1.125 * style.fontsize)px;margin:0 0 $(style.gap/2)px;overflow-wrap:anywhere"
    for panel in VIZ.visual_panels(snapshot)
        panel.kind in (:hasse, :regions, :region_labels, :query_overlay, :presentation_support,
            :slice_barcode, :slice_diagram, :barcode, :persistence_diagram) && continue
        content = Any[B.DOM.h3(panel.title; style=heading_style),
                      B.DOM.p(panel.subtitle; style="white-space:pre-line")]
        for layer in VIZ.visual_layers(panel)
            if layer isa VIZ.TextLayer
                append!(content, [B.DOM.p(line) for line in layer.labels])
            elseif layer isa VIZ.MatrixLayer
                nr, nc = size(layer.entries)
                if nr == 0 || nc == 0
                    shape = get(panel.metadata, :matrix_size, (nr, nc))
                    reason = get(panel.metadata, :empty_matrix_reason,
                        shape == (0, 0) ? "both spaces are zero-dimensional" :
                        shape[1] == 0 ? "the target is zero-dimensional" : "the source is zero-dimensional")
                    push!(content, B.DOM.p("Empty matrix ($(shape[1]) x $(shape[2])): $reason."))
                    for (heading, labels) in ((get(panel.metadata, :matrix_row_heading, "Target basis"), layer.row_labels),
                                               ("Source basis", layer.column_labels))
                        isempty(labels) || push!(content, B.DOM.p("$heading: " * join(labels, ", ")))
                    end
                else
                    row_roles = get(panel.metadata, :row_roles, fill(:foreground, nr))
                    column_roles = get(panel.metadata, :column_roles, fill(:foreground, nc))
                    cell_roles = get(panel.metadata, :cell_roles, fill(:foreground, nr, nc))
                    row_colors = haskey(panel.metadata, :row_roles) ? VIZ._VisualRole.(row_roles) :
                        get(panel.metadata, :row_colors, fill(VIZ._VisualRole(:foreground), nr))
                    column_colors = haskey(panel.metadata, :column_roles) ? VIZ._VisualRole.(column_roles) :
                        get(panel.metadata, :column_colors, fill(VIZ._VisualRole(:foreground), nc))
                    cell_colors = haskey(panel.metadata, :cell_roles) ? VIZ._VisualRole.(cell_roles) :
                        get(panel.metadata, :cell_colors, fill(VIZ._VisualRole(:foreground), nr, nc))
                    cell_padding = "padding:$(style.gap/2)px $(style.gap)px"
                    header = B.DOM.tr(B.DOM.th(get(panel.metadata, :matrix_corner, "target / source")),
                        [B.DOM.th(layer.column_labels[j] * VIZ._visual_role_label(column_roles[j]); scope="col",
                            style="$cell_padding;color:$(_inspection_css_color(style, column_colors[j]))") for j in 1:nc]...)
                    rows = Any[]
                    for i in 1:nr
                        entries = [B.DOM.td(layer.entries[i,j];
                            var"aria-label"=layer.entries[i,j] * VIZ._visual_role_label(cell_roles[i,j]),
                            title=layer.entries[i,j] * VIZ._visual_role_label(cell_roles[i,j]),
                            style="$cell_padding;text-align:right;white-space:nowrap;font-family:'$(style.mono_font)',monospace;color:$(_inspection_css_color(style, cell_colors[i,j]))") for j in 1:nc]
                        push!(rows, B.DOM.tr(B.DOM.th(layer.row_labels[i] * VIZ._visual_role_label(row_roles[i]); scope="row",
                            style="$cell_padding;color:$(_inspection_css_color(style, row_colors[i]))"), entries...))
                    end
                    push!(content, B.DOM.div(B.DOM.table(B.DOM.thead(header), B.DOM.tbody(rows...);
                        var"aria-label"=panel.title,
                        style="border-collapse:collapse;border:$(style.linewidth_scale)px solid $border;font:inherit");
                        style="max-width:100%;overflow:auto", tabindex="0", var"aria-label"="$(panel.title) matrix"))
                end
            end
        end
        push!(sections, B.DOM.section(content...;
            style="box-sizing:border-box;padding:$(style.padding)px;border:$(style.linewidth_scale)px solid $border;background:$surface;border-radius:$(style.gap/2)px;min-width:0;max-width:100%;flex:1 1 22rem"))
    end
    return B.DOM.div(B.DOM.p(snapshot.subtitle; style="white-space:pre-line"),
        B.DOM.div(sections...; style="display:flex;flex-wrap:wrap;gap:$(style.gap)px"))
end

function _inspection_markers!(ax; graph::Bool=false, style=VIZ.VisualStyle())
    M = WGLMakie.Makie
    stalk = M.Observable(M.Point2d[])
    source = M.Observable(M.Point2d[])
    target = M.Observable(M.Point2d[])
    both = M.Observable(M.Point2d[])
    for (points, kind, base_size) in ((stalk, :selected, 27), (source, :source, 31),
            (target, :target, 36), (both, :both, 36))
        role = VIZ._VisualRole(kind)
        color, marker = VIZ._visual_color(style, role), VIZ._visual_marker(role)
        markersize = base_size * style.markersize_scale
        if graph
            M.scatter!(ax, points; color=:transparent, strokecolor=color, marker,
                strokewidth=2.5 * style.linewidth_scale, markersize, depth_shift=-0.001)
        else
            M.scatter!(ax, points; color, marker, strokecolor=VIZ._visual_color(style, VIZ._VisualRole(:background)),
                strokewidth=style.linewidth_scale,
                markersize=markersize/2, depth_shift=-0.001)
        end
    end
    return (; stalk, source, target, both)
end

function _update_inspection_markers!(markers, selection, positions=nothing)
    M = WGLMakie.Makie
    for name in (:stalk, :source, :target, :both)
        getproperty(markers, name)[] = M.Point2d[]
    end
    if positions === nothing
        points = selection.query_points
        if length(points) == 1
            markers.stalk[] = [M.Point2d(Float64.(points[1]))]
        elseif length(points) == 2
            if points[1] == points[2]
                markers.both[] = [M.Point2d(Float64.(points[1]))]
            else
                markers.source[] = [M.Point2d(Float64.(points[1]))]
                markers.target[] = [M.Point2d(Float64.(points[2]))]
            end
        end
    elseif selection.vertex !== nothing && selection.vertex != 0
        markers.stalk[] = [M.Point2d(positions[selection.vertex])]
    elseif selection.pair !== nothing
        u, v = selection.pair
        if u == v && u != 0
            markers.both[] = [M.Point2d(positions[u])]
        else
            u == 0 || (markers.source[] = [M.Point2d(positions[u])])
            v == 0 || (markers.target[] = [M.Point2d(positions[v])])
        end
    end
    return nothing
end

function _inspection_nearest_vertex(ax, positions)
    isempty(positions) && return nothing
    M = WGLMakie.Makie
    point = M.mouseposition(ax.scene)
    limits = ax.finallimits[]
    spans = M.widths(limits)
    pixels = M.widths(ax.scene.viewport[])
    all(>(0), spans) && all(>(0), pixels) || return nothing
    best, distance = 0, 20.0^2
    for (q, pos) in enumerate(positions)
        d = ((pos[1]-point[1])*pixels[1]/spans[1])^2 +
            ((pos[2]-point[2])*pixels[2]/spans[2])^2
        if d <= distance
            best, distance = q, d
        end
    end
    return best == 0 ? nothing : best
end

function _render_linked_inspector(spec; display::Symbol=:inline, figure=nothing, size=nothing,
                                  style=VIZ.VisualStyle())
    VIZ._check_visual_render_options(; display, figure, size, style)
    session = spec.metadata.session
    summary = VIZ.inspection_summary(session)
    summary.closed && throw(ArgumentError("This inspection session is closed. Create a new inspection_session."))
    B, M = WGLMakie.Bonito, WGLMakie.Makie
    _NEXT_INSPECTION_UI_ID[] += 1
    dom_id = "tamerop-inspector-$(_NEXT_INSPECTION_UI_ID[])"
    prepared = VIZ._inspection_scene(session)
    positions = prepared.hasse.metadata.positions
    figure_size = figure === nothing ? (size === nothing ? (1150, 460) : size) :
        Tuple(Int.(M.widths(M.viewport(figure.scene)[])))
    background = VIZ._visual_color(style, VIZ._VisualRole(:background))
    fig = figure === nothing ? M.Figure(size=figure_size, figure_padding=style.padding,
        backgroundcolor=background) : figure
    existing_blocks = length(fig.content)
    region_axis = if prepared.region === nothing
        nothing
    else
        region_layout = fig[1, 1] = M.GridLayout()
        _HANDLERS.render_panel(fig, region_layout, prepared.region; style)
    end
    hasse_column = region_axis === nothing ? 1 : 2
    hasse_layout = fig[1, hasse_column] = M.GridLayout()
    hasse_axis = _HANDLERS.render_panel(fig, hasse_layout, prepared.hasse; style)
    _HANDLERS.style_figure(fig, style)
    navigation_blocks = copy(fig.content[existing_blocks+1:end])
    region_markers = region_axis === nothing ? nothing : _inspection_markers!(region_axis; style)
    hasse_markers = _inspection_markers!(hasse_axis; graph=true, style)

    disabled = M.Observable(false)
    basis_disabled = M.Observable(true)
    updating, closed = Ref(false), Ref(false)
    error_text = M.Observable("")
    status_text = M.Observable("Select a stalk or a source and target. Live Julia session required.")
    selection_text = M.Observable("")
    hover_text = M.Observable("Hover over a region or poset vertex to read its label and dimension.")
    readout = M.Observable{Any}(_inspection_dom_readout(VIZ.inspection_snapshot(session); style))
    navigation_content = M.Observable{Any}(fig)
    support_content = M.Observable{Any}(B.DOM.div())
    support_style = M.Observable("display:none")
    support_key = Ref{Any}(nothing)
    support_figure = Ref{Any}(nothing)
    support_markers = Any[]
    draft_source, draft_target = Ref{Any}(nothing), Ref{Any}(nothing)
    app_ref = Ref(WeakRef(nothing))
    observers = Any[]
    browser_client = Ref{Any}(nothing)
    browser_lock = ReentrantLock()
    session_token = Ref(0)
    slice_ui = Ref{Any}(nothing)

    foreground_css = _inspection_css_color(style, VIZ._VisualRole(:foreground))
    background_css = _inspection_css_color(style, VIZ._VisualRole(:background))
    surface_css = _inspection_css_color(style, VIZ._VisualRole(:surface))
    border_css = _inspection_css_color(style, VIZ._VisualRole(:border))
    error_css = _inspection_css_color(style, VIZ._VisualRole(:error))
    control_style = _inspection_control_style(style)
    attrs(name) = (; id="$dom_id-$name", disabled, style=control_style)
    endpoint = B.Dropdown(["Stalk", "Source", "Target"]; attrs("endpoint")...)
    views = summary.supported_views
    view = B.Dropdown(collect(views); option_to_string=x -> x === :module ? "Module coordinates" : "Presentation image coordinates",
        attrs("view")...)
    vertex = B.TextField("1"; attrs("vertex")...)
    select_vertex = B.Button("Inspect vertex"; attrs("select-vertex")...)
    previous_vertex = B.Button("Previous vertex"; attrs("previous-vertex")...)
    next_vertex = B.Button("Next vertex"; attrs("next-vertex")...)
    label_source = B.TextField("1"; attrs("label-source")...)
    label_target = B.TextField("1"; attrs("label-target")...)
    select_labels = B.Button("Inspect label pair"; attrs("select-labels")...)
    point_x, point_y = B.TextField("0"; attrs("point-x")...), B.TextField("0"; attrs("point-y")...)
    select_point = B.Button("Inspect exact point"; attrs("select-point")...)
    source_x, source_y = B.TextField("0"; attrs("source-x")...), B.TextField("0"; attrs("source-y")...)
    target_x, target_y = B.TextField("1"; attrs("target-x")...), B.TextField("1"; attrs("target-y")...)
    select_points = B.Button("Inspect exact parameter pair"; attrs("select-points")...)
    basis = B.Checkbox(false; id="$dom_id-basis", disabled=basis_disabled,
        style=B.Styles(control_style, B.Styles("width" => "1.1em", "height" => "1.1em",
            "accent-color" => foreground_css, "transform" => "none", "cursor" => "pointer")))
    ups = collect(1:summary.nupsets)
    downs = collect(1:summary.ndownsets)
    upset = isempty(ups) ? nothing : B.Dropdown(ups; option_to_string=i -> "U$i", attrs("upset")...)
    downset = isempty(downs) ? nothing : B.Dropdown(downs; option_to_string=i -> "D$i", attrs("downset")...)
    reset_button, close_button = B.Button("Reset selection"; attrs("reset")...), B.Button("Close inspector"; attrs("close")...)

    function commit(action)
        if closed[]
            error_text[] = "This inspector is closed. Create a new inspection session."
            return false
        end
        try
            action() === false && return false
            error_text[] = ""
            return true
        catch err
            error_text[] = sprint(showerror, err)
            return false
        end
    end

    function choose(domain, value; input=:provided)
        role = endpoint.value[]
        item = (; domain, value, input)
        if role == "Stalk"
            ok = commit() do
                domain === :vertex ? VIZ.select_inspection!(session; vertex=value, input) :
                    VIZ.select_inspection!(session; point=value, input)
            end
            ok && (draft_source[] = nothing; draft_target[] = nothing)
            return ok
        end
        s = role == "Source" ? item : draft_source[]
        t = role == "Target" ? item : draft_target[]
        if s !== nothing && t !== nothing && s.domain !== t.domain
            error_text[] = "Choose both endpoints as finite labels or both as original parameters. Reset selection to start a different pair."
            return false
        end
        ok = commit() do
            if s === nothing || t === nothing
                domain === :vertex ? VIZ.select_inspection!(session; vertex=value, input) :
                    VIZ.select_inspection!(session; point=value, input)
            else
                pair_input = s.input === :pointer || t.input === :pointer ? :pointer : :provided
                domain === :vertex ? VIZ.select_inspection!(session; pair=(s.value, t.value), input=pair_input) :
                    VIZ.select_inspection!(session; parameter_pair=(s.value, t.value), input=pair_input)
            end
        end
        if ok
            draft_source[], draft_target[] = s, t
            (s === nothing || t === nothing) && (status_text[] *= " Choose the other endpoint using Source / Target.")
        end
        return ok
    end

    function hover(; point=nothing, vertex=nothing)
        closed[] && return nothing
        result = VIZ._inspection_hover(session; point, vertex, input=:pointer)
        text = result.vertex == 0 ? "Unrepresented label 0; no stalk." :
            "Vertex $(result.vertex); dimension $(result.dimension)."
        result.point === nothing || (text *= " Pointer point $(result.point); coordinates are approximate.")
        hover_text[] = text
        return result
    end

    function dispose(; close_session::Bool=false)
        closed[] && return nothing
        closed[] = true
        disabled[] = true
        basis_disabled[] = true
        status_text[] = "Inspector closed. Its last static snapshot remains available from the inspection session."
        hover_text[] = ""
        session_token[] == 0 || VIZ._off_inspection!(session, session_token[])
        session_token[] = 0
        foreach(M.off, observers)
        empty!(observers)
        lock(browser_lock) do
            browser_client[] = nothing
        end
        # Replacing the DOM closes the figure subsessions before deleting plots.
        # This also avoids waiting for a never-connected frontend during cleanup.
        navigation_content[] = B.DOM.p("Inspector closed.")
        support_content[] = B.DOM.div()
        slice_ui[] === nothing || slice_ui[].dispose()
        foreach(delete!, navigation_blocks)
        empty!(navigation_blocks)
        figure === nothing && empty!(fig)
        support_figure[] === nothing || empty!(support_figure[])
        support_figure[] = nothing
        empty!(support_markers)
        live_app = app_ref[].value
        live_app === nothing || delete!(_INSPECTION_UI_HANDLES, live_app)
        close_session && VIZ.close_inspection!(session)
        return nothing
    end

    function refresh()
        state = VIZ.inspection_summary(session)
        state.closed && return dispose()
        selection = VIZ.inspection_selection(session)
        snapshot = VIZ.inspection_snapshot(session)
        if selection.pair !== nothing
            u, v = selection.pair
            if length(selection.query_points) == 2
                draft_source[] = (; domain=:point, value=selection.query_points[1], input=selection.input)
                draft_target[] = (; domain=:point, value=selection.query_points[2], input=selection.input)
            else
                draft_source[] = (; domain=:vertex, value=u, input=selection.input)
                draft_target[] = (; domain=:vertex, value=v, input=selection.input)
            end
        elseif selection.vertex !== nothing &&
               xor(draft_source[] === nothing, draft_target[] === nothing)
            # Display changes retain the same single query and its pending
            # endpoint. A replacement query, domain or input origin clears it.
            pending = something(draft_source[], draft_target[])
            domain = isempty(selection.query_points) ? :vertex : :point
            value = domain === :vertex ? selection.vertex : only(selection.query_points)
            if pending.domain !== domain || !isequal(pending.value, value) ||
               pending.input !== selection.input
                draft_source[] = nothing
                draft_target[] = nothing
            end
        else
            draft_source[] = nothing
            draft_target[] = nothing
        end
        updating[] = true
        try
            view.option_index[] = findfirst(==(selection.view), views)
            basis.value[] = selection.basis
            basis_disabled[] = !(selection.view === :presentation && selection.vertex !== nothing && selection.vertex != 0)
            upset === nothing || (upset.option_index[] = selection.upset)
            downset === nothing || (downset.option_index[] = selection.downset)
        finally
            updating[] = false
        end
        _update_inspection_markers!(hasse_markers, selection, positions)
        region_markers === nothing || _update_inspection_markers!(region_markers, selection)
        readout[] = _inspection_dom_readout(snapshot; style)
        relation = haskey(snapshot.metadata, :inspection) ? get(snapshot.metadata.inspection, :relation, :not_selected) :
            get(snapshot.metadata, :relation, :not_selected)
        label_text = selection.pair !== nothing ? "Finite labels: $(selection.pair[1]) -> $(selection.pair[2])" :
            selection.vertex !== nothing ? "Finite vertex: $(selection.vertex)" : "No selected vertex or pair"
        point_text = isempty(selection.query_points) ? "" :
            "\n$(selection.input === :pointer ? "Approximate pointer" : "Supplied") parameters: $(selection.query_points)"
        selection_text[] = "$label_text$point_text\nRelation: $relation; view: $(selection.view); revision: $(selection.revision)"
        status_text[] = isempty(selection.query_points) ?
            "Finite-label selections describe the finite model; they do not establish order of original parameters." :
            selection.input === :pointer ?
            "Pointer selection: coordinates are approximate. Use the exact fields to decide boundary membership." :
            "Supplied selection; exact coordinate text is classified before drawing conversion."
        support_style[] = selection.view === :presentation ? "display:block" : "display:none"
        if selection.view === :presentation
            key = (selection.upset, selection.downset)
            if support_key[] != key
                supports = [panel for panel in snapshot.panels if panel.kind === :presentation_support]
                empty!(support_markers)
                if !isempty(supports)
                    old_figure = support_figure[]
                    support_size = (figure_size[1], max(1, round(Int, 0.85 * figure_size[2])))
                    sf = M.Figure(size=support_size, figure_padding=style.padding,
                        backgroundcolor=background)
                    for (i, panel) in enumerate(supports)
                        # Support colors are fixed for this support choice;
                        # moving a selection changes only marker observables.
                        layers = VIZ._presentation_support_layers(panel.metadata.geometry,
                            panel.metadata.membership)
                        base = VIZ.VisualizationSpec(panel.kind; title=panel.title, subtitle=panel.subtitle,
                            layers, axes=panel.axes,
                            metadata=merge(panel.metadata, (; query_label_layer=nothing)))
                        support_layout = sf[1,i] = M.GridLayout()
                        ax = _HANDLERS.render_panel(sf, support_layout, base; style)
                        push!(support_markers, _inspection_markers!(ax; style))
                    end
                    M.colgap!(sf.layout, style.gap)
                    M.rowgap!(sf.layout, style.gap)
                    support_figure[] = sf
                    support_content[] = sf
                    old_figure === nothing || empty!(old_figure)
                else
                    support_content[] = B.DOM.div()
                end
                support_key[] = key
            end
            foreach(markers -> _update_inspection_markers!(markers, selection), support_markers)
        end
        slice_ui[] === nothing || slice_ui[].refresh()
        return nothing
    end

    function listen(action, observable)
        push!(observers, M.on(observable) do value
            updating[] || closed[] || action(value)
            return nothing
        end)
    end
    parse_vertex(text) = begin
        value = VIZ._parse_inspection_coordinate(text)
        denominator(value) == 1 || throw(ArgumentError("A finite vertex identifier must be an integer."))
        1 <= value <= summary.nvertices || throw(ArgumentError("A vertex identifier must be in 1:$(summary.nvertices)."))
        Int(value)
    end
    parse_point(x, y) = (VIZ._parse_inspection_coordinate(x), VIZ._parse_inspection_coordinate(y))
    listen(select_vertex.value) do _
        commit(() -> choose(:vertex, parse_vertex(vertex.value[])))
    end
    listen(select_point.value) do _
        commit(() -> choose(:point, parse_point(point_x.value[], point_y.value[])))
    end
    listen(select_labels.value) do _
        commit() do
            u, v = parse_vertex(label_source.value[]), parse_vertex(label_target.value[])
            VIZ.select_inspection!(session; pair=(u,v), input=:provided)
            draft_source[] = (; domain=:vertex, value=u, input=:provided)
            draft_target[] = (; domain=:vertex, value=v, input=:provided)
        end
    end
    listen(select_points.value) do _
        commit() do
            x, y = parse_point(source_x.value[], source_y.value[]), parse_point(target_x.value[], target_y.value[])
            VIZ.select_inspection!(session; parameter_pair=(x,y), input=:provided)
            draft_source[] = (; domain=:point, value=x, input=:provided)
            draft_target[] = (; domain=:point, value=y, input=:provided)
        end
    end
    function step_vertex(step)
        summary.nvertices == 0 && return nothing
        selection = VIZ.inspection_selection(session)
        current = selection.pair !== nothing ?
            selection.pair[endpoint.value[] == "Target" ? 2 : 1] :
            selection.vertex === nothing || selection.vertex == 0 ? 1 : selection.vertex
        next = clamp(current + step, 1, summary.nvertices)
        vertex.value[] = string(next)
        return choose(:vertex, next)
    end
    listen(_ -> step_vertex(-1), previous_vertex.value)
    listen(_ -> step_vertex(1), next_vertex.value)
    listen(view.value) do requested
        commit(() -> VIZ.select_inspection!(session; view=requested)) || refresh()
    end
    listen(basis.value) do requested
        commit(() -> VIZ.select_inspection!(session; basis=requested)) || refresh()
    end
    upset === nothing || listen(value -> commit(() -> VIZ.select_inspection!(session; upset=value)), upset.value)
    downset === nothing || listen(value -> commit(() -> VIZ.select_inspection!(session; downset=value)), downset.value)
    listen(reset_button.value) do _
        draft_source[] = nothing
        draft_target[] = nothing
        commit(() -> VIZ.reset_inspection!(session))
    end
    listen(_ -> VIZ.close_inspection!(session), close_button.value)

    function pointer_target()
        if region_axis !== nothing && M.is_mouseinside(region_axis.scene)
            p = M.mouseposition(region_axis.scene)
            return (; domain=:point, value=(Float64(p[1]), Float64(p[2])))
        elseif M.is_mouseinside(hasse_axis.scene)
            picked_vertex = _inspection_nearest_vertex(hasse_axis, positions)
            picked_vertex === nothing || return (; domain=:vertex, value=picked_vertex)
        end
        return nothing
    end
    push!(observers, M.on(M.events(fig.scene).mouseposition) do _
        closed[] && return nothing
        picked = pointer_target()
        if picked !== nothing
            picked.domain === :vertex ? hover(; vertex=picked.value) : hover(; point=picked.value)
        end
        return nothing
    end)
    push!(observers, M.on(M.events(fig.scene).mousebutton; priority=100) do event
        closed[] && return M.Consume(false)
        if event.button == M.Mouse.left && event.action == M.Mouse.press
            picked = pointer_target()
            if picked !== nothing
                choose(picked.domain, picked.value; input=:pointer)
                return M.Consume(true)
            end
        end
        return M.Consume(false)
    end)
    slice_ui[] = _inspection_slice_ui(session,region_axis,prepared;
        style,figure_size,disabled,closed,updating,commit,listen,control_style,dom_id)
    session_token[] = VIZ._on_inspection(session, _ -> refresh())
    refresh()

    label(text, control) = B.DOM.label(B.DOM.span(text; style="display:block;font-weight:600;margin-bottom:$(style.gap/3)px"), control;
        style="display:block;min-width:0;max-width:100%")
    row(children...) = B.DOM.div(children...;
        style="display:flex;align-items:end;flex-wrap:wrap;gap:$(style.gap)px;margin:$(style.gap)px 0;min-width:0")
    parameter_controls = region_axis === nothing ? B.DOM.div() : B.DOM.div(
        row(label("Exact point x", point_x), label("Exact point y", point_y), select_point),
        row(label("Exact source x", source_x), label("Exact source y", source_y),
            label("Exact target x", target_x), label("Exact target y", target_y), select_points),
        B.DOM.p("Exact fields accept integers, fractions such as 1/3, and decimal text. Point buttons honor Stalk / Source / Target; pair buttons set both endpoints."))
    support_controls = B.DOM.div(row(
        upset === nothing ? B.DOM.span("No upset supports") : label("Upset support", upset),
        downset === nothing ? B.DOM.span("No downset supports") : label("Downset support", downset));
        style=support_style)
    dom = B.DOM.div(
        B.DOM.h2("Linked spaces, maps, and presentations";
            style="font:inherit;font-size:$(1.5 * style.fontsize)px;font-weight:700;margin:0 0 $(style.gap)px"),
        B.DOM.p("Live Julia session required. Click a region or a finite-poset vertex; use exact fields for boundary questions. Hover reads only labels and dimensions."),
        row(label("Selection endpoint", endpoint), label("Coordinate view", view),
            label("Compute selected image basis", basis), reset_button, close_button),
        row(label("Finite vertex", vertex), select_vertex, previous_vertex, next_vertex),
        row(label("Finite source label", label_source), label("Finite target label", label_target), select_labels),
        parameter_controls, support_controls,
        B.DOM.p(error_text; id="$dom_id-error", role="alert", style="color:$error_css;white-space:pre-line"),
        B.DOM.p(status_text; id="$dom_id-status", role="status"),
        B.DOM.p(selection_text; id="$dom_id-selection", role="status", style="white-space:pre-line"),
        B.DOM.p(hover_text; id="$dom_id-hover", role="status"),
        B.DOM.p("Selection markers: triangle = stalk; circle = source; square = target; diamond = source and target at the same location."),
        B.DOM.div(navigation_content; style="max-width:100%;overflow:auto", tabindex="0",
            var"aria-label"="Parameter regions and finite poset"),
        B.DOM.div(B.DOM.div(support_content; style="max-width:100%;overflow:auto", tabindex="0",
            var"aria-label"="Presentation support figures"); style=support_style),
        slice_ui[].dom,
        B.DOM.div(readout; id="$dom_id-readout");
        id=dom_id,
        style="box-sizing:border-box;font-family:'$(style.font)',sans-serif;font-size:$(style.fontsize)px;line-height:1.45;color:$foreground_css;background:$background_css;padding:$(style.padding)px;width:100%;max-width:$(figure_size[1] + 2 * style.padding)px;min-width:0;overflow-wrap:anywhere")
    app = B.App(; title="TamerOp linked inspector") do browser_session
        return lock(browser_lock) do
            closed[] && return B.DOM.p("This inspector has been closed. Create a new inspection session.")
            if browser_client[] !== nothing && browser_client[] !== browser_session
                # Bonito assigns app.session before calling this handler. Keep
                # close(app) directed at the accepted client after a rejection.
                current_app = app_ref[].value
                current_app === nothing || (current_app.session[] = browser_client[])
                return B.DOM.p("This widget already has a browser client. Call visualize(session) again to open an independent linked view.")
            end
            if browser_client[] === nothing
                browser_client[] = browser_session
                push!(observers, M.on(browser_session.on_close) do isclosed
                    isclosed && dispose()
                    return nothing
                end)
            end
            return dom
        end
    end
    app_ref[] = WeakRef(app)
    controls = (; endpoint, view, vertex, select_vertex, previous_vertex, next_vertex,
        label_source, label_target, select_labels, point_x, point_y, select_point,
        source_x, source_y, target_x, target_y, select_points, basis, upset, downset,
        reset=reset_button, close=close_button,slice=slice_ui[].controls)
    callbacks = (; choose, hover, step_vertex, refresh)
    _INSPECTION_UI_HANDLES[app] = (; session, figure=fig, controls,
        axes=(; region=region_axis, hasse=hasse_axis), markers=(; region=region_markers, hasse=hasse_markers),
        callbacks, dispose, closed, disabled, basis_disabled, observers, session_token,
        error_text, status_text, selection_text, hover_text, readout, support_figure, support_markers,
        preparation_count=1, browser_client, navigation_content, dom_id, style, figure_size,
        slice=slice_ui[])
    return app
end
