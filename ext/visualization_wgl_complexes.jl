# Browser page navigation uses the same session and prepared specifications.
function _render_complex_inspector(spec;display=:inline,figure=nothing,size=nothing,style=VIZ.VisualStyle())
    VIZ._check_visual_render_options(;display,figure,size,style)
    figure === nothing || throw(ArgumentError("A live complex inspector owns its canvas; compose an inspection_snapshot for static figures."))
    session=spec.metadata.session
    VIZ.inspection_summary(session).closed && throw(ArgumentError("This complex session is closed."))
    B,M=WGLMakie.Bonito,WGLMakie.Makie
    _NEXT_INSPECTION_UI_ID[]+=1
    dom_id="tamerop-complex-inspector-$(_NEXT_INSPECTION_UI_ID[])"
    geometric=session.options.kind===:geometric_filtration
    figure_size=size === nothing ? (geometric ? (1200,1000) : (1050,700)) : size
    fig=Ref{Any}(nothing);axis=Ref{Any}(nothing);targets=Ref(NamedTuple[])
    disabled=M.Observable(false)
    error_text,status_text,hover_text=M.Observable(""),M.Observable(""),M.Observable("")
    content,readout=M.Observable{Any}(B.DOM.div()),M.Observable{Any}(B.DOM.div())
    control_style=_inspection_control_style(style)
    parameter=B.TextField("";id="$dom_id-parameter",disabled,style=control_style)
    apply=B.Button("Show parameter";id="$dom_id-apply",disabled,style=control_style)
    cell=B.TextField("";id="$dom_id-cell",disabled,style=control_style)
    select=B.Button("Inspect cell";id="$dom_id-select",disabled,style=control_style)
    dimensions=B.TextField("0,1,2";id="$dom_id-dimensions",disabled,style=control_style)
    apply_dims=B.Button("Show dimensions";id="$dom_id-apply-dimensions",disabled,style=control_style)
    inactive=B.Checkbox(false;id="$dom_id-inactive",disabled)
    boundary=B.Checkbox(true;id="$dom_id-boundary",disabled)
    clear=B.Button("Clear selection";id="$dom_id-clear",disabled,style=control_style)
    reset=B.Button("Reset";id="$dom_id-reset",disabled,style=control_style)
    close_button=B.Button("Close inspector";id="$dom_id-close",disabled,style=control_style)
    closed,updating=Ref(false),Ref(false)
    observers=Any[];pointer_observers=Any[]
    session_token,rebuild_count=Ref(0),Ref(0)
    browser_client=Ref{Any}(nothing);browser_lock=ReentrantLock();app_ref=Ref(WeakRef(nothing))
    control_owner=Ref(nothing)
    function control_status(stage,error)
        closed[] && return nothing
        error_text[]=stage===:error ? error : ""
        stage===:pending && (status_text[]="Query pending. The chart shows the last committed selection.")
        nothing
    end
    select_state(;kwargs...)=_inspection_control_selection!(session,control_owner,
        browser_client[],control_status;kwargs...)
    function commit(f)
        closed[] && return false
        disabled[]=true
        try;f();error_text[]="";return true
        catch err;err isa InterruptException && rethrow();error_text[]=sprint(showerror,err);return false
        finally;disabled[]=closed[];end
    end
    function listen(f,observable)
        push!(observers,M.on(observable) do value
            updating[] || closed[] || f(value)
            nothing
        end)
    end
    parameter_ui=_inspection_parameter_ui(session;style,disabled,closed,updating,listen,commit,dom_id)
    retained=Ref{Any}(nothing);update_count=Ref(0)
    function parse_integer(control,name)
        n=tryparse(Int,strip(control.value[]))
        n === nothing && throw(ArgumentError("Enter an integer $name."))
        return n
    end
    function nearby(point)
        axis[] === nothing && return nothing
        pixels=[M.project(axis[].scene,length(t.point)==2 ? M.Point2d(t.point) : M.Point3d(t.point)) + minimum(M.viewport(axis[].scene)[]) for t in targets[]]
        d=[sum((point[i]-p[i])^2 for i in 1:2) for p in pixels]
        isempty(d) && return nothing
        id=argmin(d)
        return d[id]<=20^2 ? targets[][id] : nothing
    end
    function hover(point)
        t=nearby(point)
        hover_text[]=t === nothing ? "" : "Cell $(t.index), dimension $(t.dimension): " * (t.active ? "active" : "inactive")
        return t
    end
    function pick(point)
        t=hover(point);t === nothing && return false
        return commit(()->select_state(;cell=t.index))
    end
    function dispose(;close_session=false)
        closed[] && return nothing
        closed[]=true;disabled[]=true
        VIZ._cancel_inspection_requests!(session,control_owner)
        parameter_ui.dispose()
        session_token[]==0 || VIZ._off_inspection!(session,session_token[])
        session_token[]=0
        foreach(M.off,observers);empty!(observers)
        foreach(M.off,pointer_observers);empty!(pointer_observers)
        fig[] === nothing || empty!(fig[])
        axis[]=nothing;empty!(targets[])
        content[]=B.DOM.p("Inspector closed. Its final static snapshot remains available.")
        lock(browser_lock) do;browser_client[]=nothing;end
        app=app_ref[].value;app === nothing || delete!(_INSPECTION_UI_HANDLES,app)
        close_session && VIZ.close_inspection!(session)
        return nothing
    end
    function refresh()
        (VIZ.inspection_summary(session).closed || _inspection_browser_ended(browser_client[])) && return dispose()
        snapshot=VIZ.inspection_snapshot(session)
        state=VIZ.inspection_selection(session)
        updating[]=true
        try
            panels=filter(p->get(p.metadata,:panel_style,nothing)!==:text_only,snapshot.panels)
            drawing=VIZ.VisualizationSpec(:complex_inspector;panels,metadata=(;panel_columns=2,figure_size))
            old=fig[]
            reused=retained[]!==nothing && _visual_update_retained!(M,VIZ,_HANDLERS,retained[],drawing;style,replace_plots=true)
            if reused
                current=fig[];update_count[]+=1
            else
                foreach(M.off,pointer_observers);empty!(pointer_observers)
                retained[]=_visual_retained_figure(M,VIZ,_HANDLERS,drawing;size=figure_size,style)
                current=retained[].figure;fig[]=current;rebuild_count[]+=1
                axis[]=retained[].panels[1].axis
                push!(pointer_observers,M.on(M.events(current.scene).mouseposition) do _
                    closed[] || (M.is_mouseinside(axis[].scene) ? hover(M.events(current.scene).mouseposition[]) : (hover_text[]=""))
                    nothing
                end)
                push!(pointer_observers,M.on(M.events(current.scene).mousebutton;priority=10) do event
                    if !closed[] && event.button==M.Mouse.left && event.action==M.Mouse.press && M.is_mouseinside(axis[].scene)
                        return M.Consume(pick(M.events(current.scene).mouseposition[]))
                    end
                    M.Consume(false)
                end)
            end
            targets[]=panels[1].metadata.pick_targets
            geometric && parameter_ui.bind_plane(current,retained[].panels[3].axis)
            parameter.value[]=join(string.(state.parameter),", ")
            cell.value[]=state.cell === nothing ? "" : string(state.cell)
            dimensions.value[]=join(state.dimensions,",")
            inactive.value[]=state.activity==:all;boundary.value[]=state.boundary
            status_text[]="Committed parameter $(state.parameter). Active counts: $(snapshot.metadata.active_counts) / $(snapshot.metadata.total_counts) total by dimension. Displayed: $(sum(snapshot.metadata.drawn_counts)); omitted by budget: $(snapshot.metadata.omitted_count)."
            hover_text[]=""
            if !reused
                content[]=current
                old === nothing || empty!(old)
            end
            parameter_ui.refresh()
            readout[]=_inspection_dom_readout(VIZ.VisualizationSpec(:complex_readout;
                panels=filter(p->get(p.metadata,:panel_style,nothing)===:text_only,snapshot.panels));style)
        finally;updating[]=false;end
        return nothing
    end
    listen(apply.value) do _
        commit() do
            p=Tuple(VIZ._parse_inspection_coordinate(strip(t)) for t in split(parameter.value[],','))
            select_state(;parameter=p)
        end
    end
    listen(apply_dims.value) do _
        commit() do
            ds=tryparse.(Int,strip.(split(dimensions.value[],',')))
            any(isnothing,ds) && throw(ArgumentError("Enter comma-separated dimensions from 0,1,2."))
            select_state(;dimensions=Int.(ds))
        end
    end
    listen(value->commit(()->select_state(;activity=value ? :all : :active)),inactive.value)
    listen(value->commit(()->select_state(;boundary=value)),boundary.value)
    listen(_->commit(()->select_state(;cell=parse_integer(cell,"cell index"))),select.value)
    listen(_->commit(()->select_state(;cell=nothing)),clear.value)
    listen(_->commit(()->select_state(;session.initial...)),reset.value)
    listen(_->commit(()->VIZ.close_inspection!(session)),close_button.value)
    session_token[]=VIZ._on_inspection(session,_ -> refresh())
    refresh()
    label(text,control)=B.DOM.label(B.DOM.span(text;style="display:block;font-weight:600"),control;style="display:block;min-width:0")
    row(children...)=B.DOM.div(children...;style="display:flex;flex-wrap:wrap;align-items:end;gap:$(style.gap)px;margin:$(style.gap)px 0")
    foreground=_inspection_css_color(style,VIZ._VisualRole(:foreground))
    background=_inspection_css_color(style,VIZ._VisualRole(:background))
    error_color=_inspection_css_color(style,VIZ._VisualRole(:error))
    dom=B.DOM.div(B.DOM.h2(geometric ? "Radius, depth and the geometric model" : "The filtered complex at a parameter"),
        B.DOM.p("Enter exact coordinates (integers, fractions or decimal text). Select a cell near its center or use its exact index. Hover reads retained cell data only."),
        parameter_ui.dom,
        row(label("Parameter coordinates",parameter),apply),
        row(label("Cell index",cell),select,clear),
        row(label("Cell dimensions",dimensions),apply_dims,label("Show inactive context",inactive),label("Highlight boundary",boundary)),
        row(reset,close_button),
        B.DOM.p(error_text;id="$dom_id-error",role="alert",style="color:$error_color;white-space:pre-line"),
        B.DOM.p(status_text;id="$dom_id-status",role="status"),B.DOM.p(hover_text;id="$dom_id-hover",role="status"),
        B.DOM.div(content;style="max-width:100%;overflow:auto",tabindex="0",var"aria-label"="Filtered complex chart"),
        B.DOM.div(readout;id="$dom_id-readout");id=dom_id,
        style="box-sizing:border-box;font-family:'$(style.font)',sans-serif;font-size:$(style.fontsize)px;line-height:1.45;color:$foreground;background:$background;padding:$(style.padding)px;max-width:100%;min-width:0;overflow-wrap:anywhere")
    app=B.App(;title="TamerOp complex inspector") do client
        lock(browser_lock) do
            closed[] && return B.DOM.p("This inspector has been closed.")
            if browser_client[] !== nothing && browser_client[] !== client
                existing=app_ref[].value;existing === nothing || (existing.session[]=browser_client[])
                return B.DOM.p("This widget already has a browser client. Call visualize(session) again to open an independent linked view.")
            end
            if browser_client[] === nothing
                browser_client[]=client
                _inspection_keyboard_navigation!(client, dom)
                push!(observers,M.on(client.on_close) do ended
                    ended && dispose();nothing
                end)
            end
            dom
        end
    end
    app_ref[]=WeakRef(app)
    _INSPECTION_UI_HANDLES[app]=(;session,figure=fig,axis,targets,controls=(;parameter,apply,cell,select,dimensions,apply_dims,inactive,boundary,clear,reset,close=close_button),
        callbacks=(;pick,hover,refresh),dispose,closed,disabled,observers,pointer_observers,session_token,browser_client,
        parameter_ui,retained,update_count,rebuild_count,error_text,status_text,hover_text,readout,dom_id,style,figure_size)
    return app
end
