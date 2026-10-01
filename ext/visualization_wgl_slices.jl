# Live slice controls share the inspection session's transactional selection.
# Draft changes never compute persistence: the Apply button commits one line.

function _inspection_slice_ui(session, region_axis, prepared;
                              style, figure_size, disabled, closed, updating,
                              commit, listen, control_style, dom_id)
    B, M = WGLMakie.Bonito, WGLMakie.Makie
    summary = VIZ.inspection_summary(session)
    available = summary.slice_available && region_axis !== nothing
    no_op() = nothing
    if !available
        reason = summary.slice_unavailable_reason
        return (; dom=B.DOM.p("Linked slice: $reason"), controls=nothing,
            refresh=no_op, dispose=no_op, callbacks=nothing)
    end

    attrs(name) = (; id="$dom_id-slice-$name", disabled, style=control_style)
    base_x = B.TextField("0"; attrs("base-x")...)
    base_y = B.TextField("0"; attrs("base-y")...)
    dir_x = B.TextField("1"; attrs("direction-x")...)
    dir_y = B.TextField("1"; attrs("direction-y")...)
    angle = B.Slider(collect(0:90); value=45, attrs("angle")...)
    offset = B.Slider(collect(-100:100); value=0, attrs("offset")...)
    scope = B.Dropdown([:window,:global]; option_to_string=x -> x === :window ?
        "Viewing-window restriction" : "Whole line (certified endpoints)", attrs("scope")...)
    apply_button = B.Button("Apply slice"; attrs("apply")...)
    disable_button = B.Button("Hide slice"; attrs("disable")...)
    interval_disabled = M.Observable(true)
    interval_labels = Ref(Dict(0 => "No selected interval"))
    interval = B.Dropdown([0]; option_to_string=i -> get(interval_labels[], i, string(i)),
        id="$dom_id-slice-interval", disabled=interval_disabled, style=control_style)
    draft_text = M.Observable("Enter a basepoint and a nonnegative, nonzero direction, then apply the slice.")
    status_text = M.Observable("No slice computed. Existing stalk and map selections are independent.")
    content = M.Observable{Any}(B.DOM.div())
    slice_figure = Ref{Any}(nothing)
    axes = Ref{Any}(nothing)
    chart_data = Ref{Any}(nothing)
    chart_observers = Any[]
    chart_key = Ref{Any}(nothing)
    last_line = Ref{Any}(nothing)
    rebuild_count = Ref(0)
    selected_bar = Ref{Any}(nothing)
    selected_bar_point = Ref{Any}(nothing)
    selected_point = Ref{Any}(nothing)
    line_points = M.Observable(M.Point2d[])
    selected_region = M.Observable(M.Point2d[])
    selected_region_point = M.Observable(M.Point2d[])
    line_color = VIZ._visual_color(style, VIZ._VisualRole(:foreground))
    selection_color = VIZ._visual_color(style, VIZ._VisualRole(:selected))
    M.lines!(region_axis, line_points; color=line_color,
        linewidth=2 * style.linewidth_scale, linestyle=:dash, depth_shift=-0.002)
    M.lines!(region_axis, selected_region; color=selection_color,
        linewidth=5 * style.linewidth_scale, depth_shift=-0.003)
    M.scatter!(region_axis,selected_region_point; color=selection_color, marker=:utriangle,
        markersize=16 * style.markersize_scale, depth_shift=-0.003)

    xlimits, ylimits = prepared.region.axes.xlimits, prepared.region.axes.ylimits
    center = (xlimits[1]/2+xlimits[2]/2, ylimits[1]/2+ylimits[2]/2)
    radius = hypot(xlimits[2]/2-xlimits[1]/2, ylimits[2]/2-ylimits[1]/2)
    text_fields = (base_x, base_y, dir_x, dir_y)
    function draft()
        theta = angle.value[]
        dx, dy = round(cosd(theta); digits=12), round(sind(theta); digits=12)
        distance = offset.value[] * radius / 100
        bx, by = center[1] - dy * distance, center[2] + dx * distance
        was_updating = updating[]
        updating[] = true
        try
            for (field, value) in zip(text_fields, (bx,by,dx,dy))
                field.value[] = string(round(value; digits=12))
            end
        finally
            updating[] = was_updating
        end
        draft_text[] = "Draft: angle $theta degrees; offset $(offset.value[])% of viewport radius. Apply to recompute. Decimal fields record the chosen approximation."
        return nothing
    end
    function apply_slice()
        ok = commit() do
            values = map(field -> VIZ._parse_inspection_coordinate(field.value[]), text_fields)
            VIZ.select_inspection!(session; slice=(basepoint=(values[1],values[2]),
                direction=(values[3],values[4])), slice_scope=scope.value[])
        end
        ok && (draft_text[] = "Fields show the committed line. Sliders draft a new line through the viewport center, shifted by the offset; press Apply slice to compute it.")
        return ok
    end
    select_interval(id) = commit(() -> VIZ.select_inspection!(session; interval=id))
    listen(_ -> draft(), angle.value)
    listen(_ -> draft(), offset.value)
    for field in (text_fields...,scope)
        listen(field.value) do _
            draft_text[] = "Exact fields have unapplied changes. The plots still show the committed slice; press Apply slice."
        end
    end
    listen(_ -> apply_slice(), apply_button.value)
    listen(_ -> commit(() -> VIZ.select_inspection!(session; slice=false)), disable_button.value)
    listen(id -> select_interval(id), interval.value)

    function clear_charts()
        foreach(M.off, chart_observers)
        empty!(chart_observers)
        content[] = B.DOM.div()
        slice_figure[] === nothing || empty!(slice_figure[])
        slice_figure[] = nothing
        axes[] = nothing
        chart_data[] = nothing
        selected_bar[] = nothing
        selected_bar_point[] = nothing
        selected_point[] = nothing
        chart_key[] = nothing
        return nothing
    end

    # Screen-distance hit testing consumes exactly the recipe's drawn positions.
    # When decorations coincide, repeated clicks cycle their distinct group IDs.
    function pick_interval(kind, point)
        data = chart_data[]
        data === nothing && return nothing
        ax = kind === :barcode ? axes[].barcode : axes[].diagram
        spans, pixels = M.widths(ax.finallimits[]), M.widths(ax.scene.viewport[])
        all(>(0), spans) && all(>(0), pixels) || return nothing
        scale = (pixels[1]/spans[1], pixels[2]/spans[2])
        distances = Float64[]
        if kind === :barcode
            for segment in data.bar_segments
                x = clamp(point[1], min(segment[1],segment[3]), max(segment[1],segment[3]))
                push!(distances, ((x-point[1])*scale[1])^2 + ((segment[2]-point[2])*scale[2])^2)
            end
        else
            for p in data.diagram_points
                push!(distances, ((p[1]-point[1])*scale[1])^2 + ((p[2]-point[2])*scale[2])^2)
            end
        end
        isempty(distances) && return nothing
        best = minimum(distances)
        best <= 20^2 || return nothing
        indices = findall(d -> abs(d-best) <= 1e-6, distances)
        ids = [data.records[i].id for i in indices]
        current = VIZ.inspection_selection(session).interval
        at = findfirst(==(current), ids)
        return at === nothing ? first(ids) : ids[mod1(at+1,length(ids))]
    end

    function rebuild(result, data)
        clear_charts()
        sf = M.Figure(size=(figure_size[1], max(420,round(Int,figure_size[2]))),
            figure_padding=style.padding,
            backgroundcolor=VIZ._visual_color(style,VIZ._VisualRole(:background)))
        panels = VIZ._inspection_slice_panels(result, VIZ.inspection_selection(session).interval; highlight=false)
        bar_layout = sf[1,1] = M.GridLayout()
        diagram_layout = sf[1,2] = M.GridLayout()
        bar_axis = _HANDLERS.render_panel(sf,bar_layout,panels[1]; style)
        diagram_axis = _HANDLERS.render_panel(sf,diagram_layout,panels[2]; style)
        _HANDLERS.style_figure(sf,style)
        bar_highlight, point_highlight = M.Observable(M.Point2d[]), M.Observable(M.Point2d[])
        bar_point = M.Observable(M.Point2d[])
        M.lines!(bar_axis,bar_highlight; color=selection_color,
            linewidth=6 * style.linewidth_scale, depth_shift=-0.002)
        M.scatter!(bar_axis,bar_point; color=selection_color,marker=:utriangle,
            markersize=12 * style.markersize_scale,depth_shift=-0.003)
        M.scatter!(diagram_axis,point_highlight; color=:transparent,
            strokecolor=selection_color, marker=:rect, strokewidth=2 * style.linewidth_scale,
            markersize=22 * style.markersize_scale, depth_shift=-0.002)
        slice_figure[] = sf
        axes[] = (; barcode=bar_axis, diagram=diagram_axis)
        chart_data[] = data
        selected_bar[], selected_point[] = bar_highlight, point_highlight
        selected_bar_point[] = bar_point
        push!(chart_observers,M.on(M.events(sf.scene).mousebutton; priority=100) do event
            closed[] && return M.Consume(false)
            if event.button == M.Mouse.left && event.action == M.Mouse.press
                kind = M.is_mouseinside(bar_axis.scene) ? :barcode :
                    M.is_mouseinside(diagram_axis.scene) ? :diagram : nothing
                if kind !== nothing
                    axis = kind === :barcode ? bar_axis : diagram_axis
                    id = pick_interval(kind,M.mouseposition(axis.scene))
                    if id !== nothing
                        select_interval(id)
                        return M.Consume(true)
                    end
                end
            end
            return M.Consume(false)
        end)
        content[] = sf
        chart_key[] = (result.line,result.window,result.endpoint_semantics,data.interval_ids)
        rebuild_count[] += 1
        return nothing
    end

    function refresh()
        selection = VIZ.inspection_selection(session)
        snapshot = VIZ.inspection_snapshot(session)
        result = get(snapshot.metadata,:slice_result,nothing)
        data = get(snapshot.metadata,:slice_view,nothing)
        was_updating = updating[]
        updating[] = true
        try
            if result === nothing
                line_points[] = M.Point2d[]
                selected_region[] = M.Point2d[]
                selected_region_point[] = M.Point2d[]
                interval_disabled[] = true
                interval.option_index[] = 1
                interval_labels[] = Dict(0 => "No selected interval")
                interval.options[] = Any[0]
                clear_charts()
                last_line[] = nothing
                status_text[] = "No slice computed. Existing stalk and map selections are independent."
                return nothing
            end
            line_points[] = [M.Point2d(p) for p in data.line_points]
            changed = last_line[] != (result.line,selection.slice_scope)
            if changed
                for (field,value) in zip(text_fields,(result.line.basepoint...,result.line.direction...))
                    field.value[] = string(value)
                end
                scope.option_index[] = selection.slice_scope === :window ? 1 : 2
                last_line[] = (result.line,selection.slice_scope)
                # Slider indices remain local drafts. Updating them here would
                # echo through Bonito's browser-side index -> value listener
                # after this synchronous refresh and overwrite exact fields.
                draft_text[] = "Fields show the committed line. Sliders draft a new line through the viewport center, shifted by the offset; press Apply slice to compute it."
            end
            key = (result.line,result.window,result.endpoint_semantics,data.interval_ids)
            chart_key[] == key || rebuild(result,data)
            labels = Dict(0 => "No selected interval")
            for record in result.intervals
                labels[record.id] = VIZ._inspection_slice_record_label(record)
            end
            options = Any[0; [record.id for record in result.intervals]]
            if interval.options[] != options || interval_labels[] != labels
                interval.option_index[] = 1
                interval_labels[] = labels
                interval.options[] = options
            end
            id = something(selection.interval,0)
            interval.option_index[] = something(findfirst(==(id),options),1)
            interval_disabled[] = isempty(result.intervals)
            selected_bar[][] = M.Point2d[]
            selected_bar_point[][] = M.Point2d[]
            selected_point[][] = M.Point2d[]
            selected_region[] = M.Point2d[]
            selected_region_point[] = M.Point2d[]
            record_index = findfirst(record -> record.id == id,data.records)
            if record_index !== nothing
                segment = data.bar_segments[record_index]
                selected_bar[][] = [M.Point2d(segment[1],segment[2]), M.Point2d(segment[3],segment[4])]
                selected_bar_point[][] = [M.Point2d(segment[1]/2+segment[3]/2,segment[2])]
                selected_point[][] = [M.Point2d(data.diagram_points[record_index])]
                record = data.records[record_index]
                region_segment = VIZ._inspection_slice_segment(result,record)
                if region_segment !== nothing
                    selected_region[] = [M.Point2d(p) for p in region_segment]
                    first(region_segment) == last(region_segment) && (selected_region_point[] = [first(selected_region[])])
                end
            end
            n = length(result.intervals)
            total = sum(record.multiplicity for record in result.intervals; init=0)
            chosen = findfirst(record -> record.id == id,result.intervals)
            selected_text = chosen === nothing ? "No selected interval." :
                VIZ._inspection_slice_record_label(result.intervals[chosen]) *
                (record_index === nothing ? " (outside display window)" : "")
            scope_text = selection.slice_scope === :global ?
                "Whole-line endpoints certified; infinity is shown on labelled lanes. Finite offscreen ends are separate." :
                "Viewing-window restriction; censored ends do not establish essential infinity."
            status_text[] = "Committed line x(t) = $(result.line.basepoint) + t * $(result.line.direction).\nViewing window: $(result.window); $(length(data.records)) displayed / $n groups, total multiplicity $total. $scope_text\n$selected_text"

        finally
            updating[] = was_updating
        end
        return nothing
    end

    function dispose()
        interval_disabled[] = true
        clear_charts()
        line_points[] = M.Point2d[]
        selected_region[] = M.Point2d[]
        selected_region_point[] = M.Point2d[]
        return nothing
    end
    label(text,control) = B.DOM.label(B.DOM.span(text; style="display:block;font-weight:600"),control;
        style="display:block;min-width:0;max-width:100%")
    row(children...) = B.DOM.div(children...;
        style="display:flex;align-items:end;flex-wrap:wrap;gap:$(style.gap)px;margin:$(style.gap)px 0;min-width:0")
    dom = B.DOM.section(
        B.DOM.h3("A movable slice and its persistence intervals";
            style="font:inherit;font-size:$(1.125*style.fontsize)px;font-weight:700"),
        B.DOM.p("Restrict the encoded module to x(t) = basepoint + t * direction. Directions must be nonnegative and nonzero. Choose a viewing-window restriction or certify the whole line. A global query rejects unrepresented parameters."),
        row(label("Slice scope",scope)),
        row(label("Draft angle (degrees)",angle),label("Draft offset (% of viewport radius)",offset)),
        row(label("Exact basepoint x",base_x),label("Exact basepoint y",base_y),
            label("Exact direction x",dir_x),label("Exact direction y",dir_y),apply_button,disable_button),
        B.DOM.p(draft_text; id="$dom_id-slice-draft",role="status"),
        row(label("Selected interval group",interval)),
        B.DOM.p("Click a bar or diagram point to select the same interval group in both charts. Repeated clicks cycle coincident diagram points; the selector provides every group. Selection IDs belong only to the current slice."),
        B.DOM.p(status_text; id="$dom_id-slice-status",role="status",style="white-space:pre-line"),
        B.DOM.div(content; style="max-width:100%;overflow:auto",tabindex="0",var"aria-label"="Linked slice barcode and persistence diagram");
        style="margin:$(style.gap)px 0;padding:$(style.padding)px;border:$(style.linewidth_scale)px solid $(_inspection_css_color(style,VIZ._VisualRole(:border)));min-width:0")
    controls = (; base_x,base_y,dir_x,dir_y,angle,offset,scope,apply=apply_button,disable=disable_button,interval)
    callbacks = (; draft,apply_slice,select_interval,pick_interval)
    return (; dom,controls,refresh,dispose,callbacks,figure=slice_figure,axes,content,
        chart_observers,line_points,selected_region,selected_region_point,selected_bar,selected_bar_point,selected_point,
        interval_disabled,draft_text,status_text,rebuild_count)
end
