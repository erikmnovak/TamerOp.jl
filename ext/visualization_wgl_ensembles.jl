# One shared query selection; callbacks consume the core's prepared snapshots.
function _render_ensemble_inspector(spec;display=:inline,figure=nothing,size=nothing,style=VIZ.VisualStyle())
    VIZ._check_visual_render_options(;display,figure,size,style)
    figure===nothing || throw(ArgumentError("An ensemble inspector owns its canvas."))
    session=spec.metadata.session
    VIZ.inspection_summary(session).closed && throw(ArgumentError("This ensemble session is closed."))
    B,M=WGLMakie.Bonito,WGLMakie.Makie
    _NEXT_INSPECTION_UI_ID[]+=1
    dom_id="tamerop-ensemble-inspector-$(_NEXT_INSPECTION_UI_ID[])"
    figure_size=size
    fig=Ref{Any}(nothing);axis=Ref{Any}(nothing);axes=Any[];targets=Ref(NamedTuple[])
    disabled=M.Observable(false)
    error_text,status_text,hover_text=M.Observable(""),M.Observable(""),M.Observable("")
    content,readout=M.Observable{Any}(B.DOM.div()),M.Observable{Any}(B.DOM.div())
    control_style=_inspection_control_style(style)
    sample=B.TextField("1";id="$dom_id-sample",disabled,style=control_style)
    probe=B.TextField("1";id="$dom_id-probe",disabled,style=control_style)
    apply=B.Button("Apply selection";id="$dom_id-apply",disabled,style=control_style)
    views=Dict(k=>B.Button(uppercasefirst(String(k));id="$dom_id-$k",disabled,style=control_style)
        for k in (:mean,:lower,:upper,:std,:frequency,:counts))
    reset=B.Button("Reset";id="$dom_id-reset",disabled,style=control_style)
    close_button=B.Button("Close inspector";id="$dom_id-close",disabled,style=control_style)
    closed,updating=Ref(false),Ref(false)
    observers=Any[];pointer_observers=Any[]
    session_token,rebuild_count=Ref(0),Ref(0)
    browser_client=Ref{Any}(nothing);browser_lock=ReentrantLock();app_ref=Ref(WeakRef(nothing))
    function commit(f)
        closed[] && return false
        disabled[]=true
        try;f();error_text[]="";true
        catch err;err isa InterruptException && rethrow();error_text[]=sprint(showerror,err);false
        finally;disabled[]=closed[];end
    end
    function listen(f,observable)
        push!(observers,M.on(observable) do value
            updating[] || closed[] || f(value)
            nothing
        end)
    end
    function integer(control)
        n=tryparse(Int,strip(control.value[]))
        n===nothing && throw(ArgumentError("Enter an integer replicate or probe index."))
        n
    end
    function nearby(point,ax=axis[])
        ax===nothing && return nothing
        candidates=filter(t->t.axis===ax,targets[])
        isempty(candidates) && return nothing
        # Measure hit tolerance in pixels, independently of embedding units.
        pixel=M.project(ax.scene,M.Point2d(point))
        distances=[sum(abs2,M.project(ax.scene,M.Point2d(t.point))-pixel) for t in candidates]
        k=argmin(distances)
        distances[k]<=22^2 ? candidates[k] : nothing
    end
    function hover(point,ax=axis[])
        t=nearby(point,ax)
        hover_text[]=t===nothing ? "" : t.label
        t
    end
    function pick(point,ax=axis[])
        t=hover(point,ax);t===nothing && return false
        commit(()->VIZ.select_inspection!(session;probe=t.probe))
    end
    function dispose(;close_session=false)
        closed[] && return nothing
        closed[]=true;disabled[]=true
        session_token[]==0 || VIZ._off_inspection!(session,session_token[])
        session_token[]=0
        foreach(M.off,observers);empty!(observers)
        foreach(M.off,pointer_observers);empty!(pointer_observers)
        fig[]===nothing || empty!(fig[])
        axis[]=nothing;empty!(targets[]);empty!(axes)
        content[]=B.DOM.p("Inspector closed. Its final static snapshot remains available.")
        lock(browser_lock) do;browser_client[]=nothing;end
        app=app_ref[].value;app===nothing || delete!(_INSPECTION_UI_HANDLES,app)
        close_session && VIZ.close_inspection!(session)
        nothing
    end
    function refresh()
        VIZ.inspection_summary(session).closed && return dispose()
        snapshot=VIZ.inspection_snapshot(session);state=VIZ.inspection_selection(session)
        updating[]=true
        try
            foreach(M.off,pointer_observers);empty!(pointer_observers)
            old=fig[]
            visible=[i for (i,p) in enumerate(snapshot.panels) if get(p.metadata,:panel_style,nothing)!==:text_only]
            visible_rows=sort!(unique([r for i in visible for r in snapshot.metadata.panel_positions[i][1]]))
            row_index=Dict(r=>i for (i,r) in enumerate(visible_rows))
            weights=snapshot.metadata.panel_row_weights[visible_rows]
            canvas_size=figure_size===nothing ? (snapshot.metadata.figure_size[1],ceil(Int,sum(weights)+2style.padding)) : figure_size
            current=M.Figure(size=canvas_size,figure_padding=style.padding,
                backgroundcolor=VIZ._visual_color(style,VIZ._VisualRole(:background)))
            empty!(axes);empty!(targets[]);axis[]=nothing

            for (i,panel) in enumerate(snapshot.panels)
                get(panel.metadata,:panel_style,nothing)===:text_only && continue
                original_rows,cols=snapshot.metadata.panel_positions[i]
                rows=row_index[first(original_rows)]:row_index[last(original_rows)]
                grid=current[rows,cols]=M.GridLayout()
                ax=_HANDLERS.render_panel(current,grid,panel;style)
                if panel.kind in (:ensemble_curve,:ensemble_map)
                    push!(axes,ax);axis[]===nothing && (axis[]=ax)
                    append!(targets[],[(;t...,axis=ax,kind=panel.kind) for t in panel.metadata.pick_targets])

                end
            end

            for (i,weight) in enumerate(weights)
                M.rowsize!(current.layout,i,M.Auto(false,weight))
            end
            _HANDLERS.style_figure(current,style)
            fig[]=current;rebuild_count[]+=1
            push!(pointer_observers,M.on(M.events(current.scene).mouseposition) do _
                closed[] && return nothing
                active=findfirst(ax->M.is_mouseinside(ax.scene),axes)
                active===nothing ? (hover_text[]="") : hover(M.mouseposition(axes[active].scene),axes[active])
                nothing
            end)
            push!(pointer_observers,M.on(M.events(current.scene).mousebutton;priority=10) do event
                if !closed[] && event.button==M.Mouse.left && event.action==M.Mouse.press
                    active=findfirst(ax->M.is_mouseinside(ax.scene),axes)
                    active===nothing || return M.Consume(pick(M.mouseposition(axes[active].scene),axes[active]))
                end
                M.Consume(false)
            end)
            sample.value[]=string(state.sample);probe.value[]=string(state.probe)
            selected=snapshot.metadata.selected
            status_text[]="$(state.view): replicate $(selected.sample), probe $(selected.probe) at $(selected.coordinates). Original value: $(selected.value); mean $(selected.mean); available $(selected.count)."
            hover_text[]="";content[]=current
            old===nothing || empty!(old)
            readout[]=_inspection_dom_readout(VIZ.VisualizationSpec(:comparison_readout;
                panels=filter(p->get(p.metadata,:panel_style,nothing)===:text_only,snapshot.panels));style)
        finally;updating[]=false;end
        nothing
    end
    listen(_->commit(()->VIZ.select_inspection!(session;sample=integer(sample),probe=integer(probe))),apply.value)
    for (key,button) in views
        listen(_->commit(()->VIZ.select_inspection!(session;view=key)),button.value)
    end
    listen(_->commit(()->VIZ.reset_inspection!(session)),reset.value)
    listen(_->commit(()->VIZ.close_inspection!(session)),close_button.value)
    session_token[]=VIZ._on_inspection(session,_ -> refresh())
    refresh()
    label(text,control)=B.DOM.label(B.DOM.span(text;style="display:block;font-weight:600"),control;style="display:block;min-width:0")
    row(children...)=B.DOM.div(children...;style="display:flex;flex-wrap:wrap;align-items:end;gap:$(style.gap)px;margin:$(style.gap)px 0")
    foreground=_inspection_css_color(style,VIZ._VisualRole(:foreground))
    background=_inspection_css_color(style,VIZ._VisualRole(:background))
    dom=B.DOM.div(B.DOM.h2("Explore an ensemble"),
        B.DOM.p("Select a probe in a curve/map or enter its index. Replicate selection links retained diagrams and source examples. Bands describe the declared probes only; empirical variation is distinct from confidence. Live Julia is required."),
        row((views[k] for k in (:mean,:lower,:upper,:std,:frequency,:counts))...),
        row(label("Replicate index",sample),label("Probe index",probe),apply,reset,close_button),
        B.DOM.p(error_text;id="$dom_id-error",role="alert"),
        B.DOM.p(status_text;id="$dom_id-status",role="status"),B.DOM.p(hover_text;id="$dom_id-hover",role="status",style="min-height:1.5em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis"),
        B.DOM.div(content;style="max-width:100%;overflow:auto",tabindex="0",var"aria-label"="Ensemble charts"),
        B.DOM.div(readout;id="$dom_id-readout");id=dom_id,
        style="box-sizing:border-box;font-family:'$(style.font)',sans-serif;font-size:$(style.fontsize)px;line-height:1.45;color:$foreground;background:$background;padding:$(style.padding)px;max-width:100%;min-width:0;overflow-wrap:anywhere")
    app=B.App(;title="TamerOp ensemble inspector") do client
        lock(browser_lock) do
            closed[] && return B.DOM.p("This inspector has been closed.")
            if browser_client[]!==nothing && browser_client[]!==client
                existing=app_ref[].value;existing===nothing || (existing.session[]=browser_client[])
                return B.DOM.p("This widget already has a browser client. Call visualize(session) again to open an independent linked view.")
            end
            if browser_client[]===nothing
                browser_client[]=client
                _inspection_keyboard_navigation!(client, dom)
                push!(observers,M.on(client.on_close) do ended;ended && dispose();nothing;end)
            end
            dom
        end
    end
    app_ref[]=WeakRef(app)
    _INSPECTION_UI_HANDLES[app]=(;session,figure=fig,axis,axes,targets,
        controls=(;sample,probe,apply,views,reset,close=close_button),
        callbacks=(;pick,hover,refresh),dispose,closed,disabled,observers,pointer_observers,session_token,browser_client,
        rebuild_count,error_text,status_text,hover_text,readout,dom_id,style,figure_size)
    app
end
