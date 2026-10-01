# Linked interval-only inspection uses the same exact records as static export.
# Picking reads drawing positions; representative work requires an explicit opt-in.

function _render_interval_inspector(spec; display::Symbol=:inline, figure=nothing,
                                    size=nothing, style=VIZ.VisualStyle())
    VIZ._check_visual_render_options(; display, figure, size, style)
    session = spec.metadata.session
    summary = VIZ.inspection_summary(session)
    summary.closed && throw(ArgumentError("This inspection session is closed. Create a new inspection_session."))
    B, M = WGLMakie.Bonito, WGLMakie.Makie
    _NEXT_INSPECTION_UI_ID[] += 1
    dom_id = "tamerop-interval-inspector-$(_NEXT_INSPECTION_UI_ID[])"
    figure_size = figure === nothing ? (size === nothing ? (1150,520) : size) :
        Tuple(Int.(M.widths(M.viewport(figure.scene)[])))
    background = VIZ._visual_color(style,VIZ._VisualRole(:background))
    selection_color = VIZ._visual_color(style,VIZ._VisualRole(:selected))
    fig = figure === nothing ? M.Figure(size=figure_size,figure_padding=style.padding,
        backgroundcolor=background) : figure
    axes = Ref{Any}(nothing)
    chart_data = Ref{Any}(nothing)
    chart_key = Ref{Any}(nothing)
    navigation_blocks = Any[]
    rebuild_count = Ref(0)
    selected_bar, selected_bar_point, selected_point =
        M.Observable(M.Point2d[]), M.Observable(M.Point2d[]), M.Observable(M.Point2d[])
    navigation_content = M.Observable{Any}(B.DOM.div())
    snapshot = VIZ.inspection_snapshot(session)
    payload = snapshot.metadata.interval_payload
    readout = M.Observable{Any}(_inspection_dom_readout(snapshot;style))
    disabled, member_disabled, representative_disabled = M.Observable(false),M.Observable(true),M.Observable(true)
    closed, updating = Ref(false),Ref(false)
    error_text, status_text, hover_text = M.Observable(""),M.Observable(""),M.Observable("")
    observers = Any[]
    session_token = Ref(0)
    browser_client = Ref{Any}(nothing)
    browser_lock = ReentrantLock()
    app_ref = Ref(WeakRef(nothing))
    control_style = _inspection_control_style(style)
    foreground_css = _inspection_css_color(style,VIZ._VisualRole(:foreground))
    background_css = _inspection_css_color(style,VIZ._VisualRole(:background))
    error_css = _inspection_css_color(style,VIZ._VisualRole(:error))
    labels = Dict(0 => "No selected interval")
    for record in payload.records
        labels[record.id] = VIZ._interval_record_label(record)
    end
    options = Any[0; [r.id for r in payload.records]]
    interval = B.Dropdown(options;option_to_string=id -> labels[id],
        id="$dom_id-interval",disabled,style=control_style)
    member = B.TextField("";id="$dom_id-member",disabled=member_disabled,style=control_style)
    apply_member = B.Button("Select member";id="$dom_id-apply-member",disabled=member_disabled,style=control_style)
    representative = B.Checkbox(false;id="$dom_id-representative",disabled=representative_disabled,
        style=B.Styles(control_style,B.Styles("width"=>"1.1em","height"=>"1.1em",
            "accent-color"=>foreground_css,"transform"=>"none","cursor"=>"pointer")))
    reset_button = B.Button("Reset selection";id="$dom_id-reset",disabled,style=control_style)
    close_button = B.Button("Close inspector";id="$dom_id-close",disabled,style=control_style)

    function commit(action)
        if closed[]
            error_text[] = "This inspector is closed. Create a new inspection session."
            return false
        end
        try
            action()
            error_text[] = ""
            return true
        catch err
            err isa InterruptException && rethrow()
            error_text[] = sprint(showerror,err)
            return false
        end
    end
    function listen(action,observable)
        push!(observers,M.on(observable) do value
            updating[] || closed[] || action(value)
            return nothing
        end)
    end
    select_interval(id) = commit(() -> VIZ.select_inspection!(session;interval=id))
    function select_member()
        return commit() do
            value = tryparse(Int,strip(member.value[]))
            value !== nothing && value > 0 || throw(ArgumentError("Enter a positive original member number."))
            VIZ.select_inspection!(session;member=value)
        end
    end

    function nearby_intervals(kind,point)
        data = chart_data[]
        data === nothing && return Int[]
        kind in (:barcode,:diagram) || throw(ArgumentError("Pick a barcode or diagram panel."))
        ax = getproperty(axes[],kind)
        spans,pixels = M.widths(ax.finallimits[]),M.widths(ax.scene.viewport[])
        all(>(0),spans) && all(>(0),pixels) || return Int[]
        scale = (pixels[1]/spans[1],pixels[2]/spans[2])
        distances = Float64[]
        if kind === :barcode
            for segment in data.bar_segments
                x = clamp(point[1],min(segment[1],segment[3]),max(segment[1],segment[3]))
                push!(distances,((x-point[1])*scale[1])^2+((segment[2]-point[2])*scale[2])^2)
            end
        else
            for p in data.diagram_points
                push!(distances,((p[1]-point[1])*scale[1])^2+((p[2]-point[2])*scale[2])^2)
            end
        end
        isempty(distances) && return Int[]
        best = minimum(distances)
        best <= 20^2 || return Int[]
        return Int[data.interval_ids[i] for i in eachindex(distances) if abs(distances[i]-best) <= 1e-6]
    end
    function pick_interval(kind,point)
        ids = nearby_intervals(kind,point)
        isempty(ids) && return nothing
        current = VIZ.inspection_selection(session).interval
        at = findfirst(==(current),ids)
        return at === nothing ? first(ids) : ids[mod1(at+1,length(ids))]
    end
    function hover(kind,point)
        closed[] && return nothing
        ids = nearby_intervals(kind,point)
        hover_text[] = isempty(ids) ? "" : join((labels[id] for id in ids),"\n")
        return (;interval_ids=Tuple(ids))
    end

    function rebuild(data)
        # Remove only blocks owned by this viewer, preserving a supplied figure.
        axes[] = nothing
        foreach(delete!,navigation_blocks)
        empty!(navigation_blocks)
        before = length(fig.content)
        panels = VIZ._interval_panels(payload.records;window=summary.window,
            interval=VIZ.inspection_selection(session).interval,max_intervals=summary.max_intervals,
            order=payload.order,endpoint_semantics=payload.endpoint_semantics,
            essential_status=payload.essential_status,highlight=false)
        bar_layout = fig[1,1] = M.GridLayout()
        diagram_layout = fig[1,2] = M.GridLayout()
        bar_axis = _HANDLERS.render_panel(fig,bar_layout,panels[1];style)
        diagram_axis = _HANDLERS.render_panel(fig,diagram_layout,panels[2];style)
        _HANDLERS.style_figure(fig,style)
        append!(navigation_blocks,fig.content[before+1:end])
        M.lines!(bar_axis,selected_bar;color=selection_color,linewidth=6*style.linewidth_scale,depth_shift=-0.002)
        M.scatter!(bar_axis,selected_bar_point;color=selection_color,marker=:utriangle,
            markersize=12*style.markersize_scale,depth_shift=-0.003)
        M.scatter!(diagram_axis,selected_point;color=:transparent,strokecolor=selection_color,
            marker=:rect,strokewidth=2*style.linewidth_scale,markersize=22*style.markersize_scale,depth_shift=-0.002)
        axes[] = (;barcode=bar_axis,diagram=diagram_axis)
        chart_key[] = data.interval_ids
        rebuild_count[] += 1
        navigation_content[] = fig
        return nothing
    end

    function dispose(;close_session::Bool=false)
        closed[] && return nothing
        closed[] = true
        disabled[] = member_disabled[] = representative_disabled[] = true
        status_text[] = "Inspector closed. Its last static snapshot remains available."
        hover_text[] = ""
        session_token[] == 0 || VIZ._off_inspection!(session,session_token[])
        session_token[] = 0
        foreach(M.off,observers)
        empty!(observers)
        lock(browser_lock) do
            browser_client[] = nothing
        end
        navigation_content[] = B.DOM.p("Inspector closed.")
        foreach(delete!,navigation_blocks)
        empty!(navigation_blocks)
        figure === nothing && empty!(fig)
        axes[] = nothing
        chart_data[] = nothing
        selected_bar[] = M.Point2d[]
        selected_bar_point[] = M.Point2d[]
        selected_point[] = M.Point2d[]
        live_app = app_ref[].value
        live_app === nothing || delete!(_INSPECTION_UI_HANDLES,live_app)
        close_session && VIZ.close_inspection!(session)
        return nothing
    end
    function refresh()
        VIZ.inspection_summary(session).closed && return dispose()
        selection = VIZ.inspection_selection(session)
        current_snapshot = VIZ.inspection_snapshot(session)
        data = current_snapshot.metadata.interval_view
        updating[] = true
        try
            chart_data[] = data
            chart_key[] == data.interval_ids || rebuild(data)
            interval.option_index[] = something(findfirst(==(something(selection.interval,0)),options),1)
            member.value[] = selection.member === nothing ? "" : string(selection.member)
            representative.value[] = selection.representative
            chosen = data.selected_record
            members = chosen === nothing ? () : get(chosen,:members,())
            member_disabled[] = isempty(members)
            representative_disabled[] = chosen === nothing
            selected_bar[] = M.Point2d[]
            selected_bar_point[] = M.Point2d[]
            selected_point[] = M.Point2d[]
            index = findfirst(==(selection.interval),data.interval_ids)
            if index !== nothing
                segment = data.bar_segments[index]
                selected_bar[] = [M.Point2d(segment[1],segment[2]),M.Point2d(segment[3],segment[4])]
                selected_bar_point[] = [M.Point2d(segment[1]/2+segment[3]/2,segment[2])]
                selected_point[] = [M.Point2d(data.diagram_points[index])]
            end
            text = "$(data.displayed_groups) / $(data.total_groups) groups displayed; total multiplicity $(data.total_multiplicity). " *
                "$(data.offscreen_groups) outside the window; $(data.omitted_groups) omitted by the display budget."
            chosen === nothing || (text *= "\n" * labels[chosen.id] *
                (index === nothing ? " (outside the displayed window)" : "") *
                (isempty(members) ? "\nNo source-member correspondence is retained." :
                 "\n$(length(members)) original members; " * (selection.member === nothing ? "choose one before requesting a cycle." : "member $(selection.member) selected.")))
            status_text[] = text
            readout[] = _inspection_dom_readout(current_snapshot;style)
        finally
            updating[] = false
        end
        return nothing
    end

    listen(id -> select_interval(id),interval.value)
    listen(_ -> select_member(),apply_member.value)
    listen(representative.value) do wanted
        ok = commit(() -> VIZ.select_inspection!(session;representative=wanted))
        if !ok
            updating[] = true
            try
                representative.value[] = VIZ.inspection_selection(session).representative
            finally
                updating[] = false
            end
        end
    end
    listen(_ -> commit(() -> VIZ.reset_inspection!(session)),reset_button.value)
    listen(_ -> VIZ.close_inspection!(session),close_button.value)
    push!(observers,M.on(M.events(fig.scene).mousebutton;priority=100) do event
        closed[] && return M.Consume(false)
        if event.button == M.Mouse.left && event.action == M.Mouse.press && axes[] !== nothing
            for kind in (:barcode,:diagram)
                axis = getproperty(axes[],kind)
                M.is_mouseinside(axis.scene) || continue
                id = pick_interval(kind,M.mouseposition(axis.scene))
                if id !== nothing
                    select_interval(id)
                    return M.Consume(true)
                end
            end
        end
        return M.Consume(false)
    end)
    push!(observers,M.on(M.events(fig.scene).mouseposition) do _
        closed[] && return nothing
        if axes[] !== nothing
            for kind in (:barcode,:diagram)
                axis = getproperty(axes[],kind)
                M.is_mouseinside(axis.scene) || continue
                hover(kind,M.mouseposition(axis.scene))
                return nothing
            end
        end
        hover_text[] = ""
        return nothing
    end)
    session_token[] = VIZ._on_inspection(session,_ -> refresh())
    refresh()

    label(text,control) = B.DOM.label(B.DOM.span(text;style="display:block;font-weight:600"),control;
        style="display:block;min-width:0;max-width:100%")
    row(children...) = B.DOM.div(children...;style="display:flex;align-items:end;flex-wrap:wrap;gap:$(style.gap)px;margin:$(style.gap)px 0;min-width:0")
    dom = B.DOM.div(
        B.DOM.h2("Linked barcode and persistence diagram";
            style="font:inherit;font-size:$(1.5*style.fontsize)px;font-weight:700"),
        B.DOM.p("Live Julia session required. Click either chart to select an interval group. Repeated clicks cycle coincident points. The selector includes every group, including groups outside the window."),
        row(label("Selected interval group",interval),label("Original member number",member),apply_member),
        row(label("Show retained representative",representative),reset_button,close_button),
        B.DOM.p("Duplicate intervals describe several independent classes. Choose an original member before requesting its retained cycle. Cycles are reduction choices; no source geometry or correspondence between different lines is implied."),
        B.DOM.p(error_text;id="$dom_id-error",role="alert",style="color:$error_css;white-space:pre-line"),
        B.DOM.p(status_text;id="$dom_id-status",role="status",style="white-space:pre-line"),
        B.DOM.p(hover_text;id="$dom_id-hover",role="status",style="white-space:pre-line"),
        B.DOM.div(navigation_content;style="max-width:100%;overflow:auto",tabindex="0",
            var"aria-label"="Linked barcode and persistence diagram"),
        B.DOM.div(readout;id="$dom_id-readout");id=dom_id,
        style="box-sizing:border-box;font-family:'$(style.font)',sans-serif;font-size:$(style.fontsize)px;line-height:1.45;color:$foreground_css;background:$background_css;padding:$(style.padding)px;width:100%;max-width:$(figure_size[1]+2*style.padding)px;min-width:0;overflow-wrap:anywhere")
    app = B.App(;title="TamerOp interval inspector") do browser_session
        return lock(browser_lock) do
            closed[] && return B.DOM.p("This inspector has been closed. Create a new inspection session.")
            if browser_client[] !== nothing && browser_client[] !== browser_session
                current_app = app_ref[].value
                current_app === nothing || (current_app.session[] = browser_client[])
                return B.DOM.p("This widget already has a browser client. Call visualize(session) again to open an independent linked view.")
            end
            if browser_client[] === nothing
                browser_client[] = browser_session
                push!(observers,M.on(browser_session.on_close) do isclosed
                    isclosed && dispose()
                    return nothing
                end)
            end
            return dom
        end
    end
    app_ref[] = WeakRef(app)
    controls = (;interval,member,apply_member,representative,reset=reset_button,close=close_button)
    callbacks = (;select_interval,select_member,pick_interval,hover,refresh)
    _INSPECTION_UI_HANDLES[app] = (;session,figure=fig,controls,callbacks,axes,chart_data,
        selected_bar,selected_bar_point,selected_point,rebuild_count,dispose,closed,disabled,
        member_disabled,representative_disabled,observers,session_token,browser_client,
        navigation_content,readout,error_text,status_text,hover_text,dom_id,style,figure_size)
    return app
end
