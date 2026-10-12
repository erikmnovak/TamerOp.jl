# One shared query selection; callbacks consume the core's prepared snapshots.
function _render_comparison_inspector(spec;display=:inline,figure=nothing,size=nothing,style=VIZ.VisualStyle())
    VIZ._check_visual_render_options(;display,figure,size,style)
    figure===nothing || throw(ArgumentError("A comparison inspector owns its canvas."))
    session=spec.metadata.session
    VIZ.inspection_summary(session).closed && throw(ArgumentError("This comparison session is closed."))
    B,M=WGLMakie.Bonito,WGLMakie.Makie
    _NEXT_INSPECTION_UI_ID[]+=1
    dom_id="tamerop-comparison-inspector-$(_NEXT_INSPECTION_UI_ID[])"
    figure_size=size===nothing ? (1250,300length(session.object.cases)) : size
    fig=Ref{Any}(nothing);axis=Ref{Any}(nothing);axes=Any[];targets=Ref(NamedTuple[])
    disabled=M.Observable(false)
    error_text,status_text,hover_text=M.Observable(""),M.Observable(""),M.Observable("")
    content,readout=M.Observable{Any}(B.DOM.div()),M.Observable{Any}(B.DOM.div())
    control_style=_inspection_control_style(style)
    source=B.TextField("1";id="$dom_id-source",disabled,style=control_style)
    probe=B.TextField("1";id="$dom_id-probe",disabled,style=control_style)
    baseline=B.TextField(string(session.state.baseline);id="$dom_id-baseline",disabled,style=control_style)
    apply=B.Button("Apply selection";id="$dom_id-apply",disabled,style=control_style)
    dimensions=B.Button("Dimensions";id="$dom_id-dimensions",disabled,style=control_style)
    ranks=B.Button("Anchored ranks";id="$dom_id-ranks",disabled,style=control_style)
    barcodes=session.object.domain===:ordinary ? B.Button("Barcodes";id="$dom_id-barcodes",disabled,style=control_style) : nothing
    matrices=session.object.domain!==:ordinary ? B.Button("Inspect selected matrices";id="$dom_id-matrices",disabled,style=control_style) : nothing
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
        n===nothing && throw(ArgumentError("Enter an integer probe index."))
        n
    end
    function nearby(point,ax=axis[])
        ax===nothing && return nothing
        # Strips are indexed by probe, not by invented ambient coordinates.
        0<=point[2]<=1 || return nothing
        i=round(Int,point[1])
        match=findfirst(t->t.axis===ax && t.probe==i,targets[])
        match===nothing ? nothing : targets[][match]
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
            current=M.Figure(size=figure_size,figure_padding=style.padding,
                backgroundcolor=VIZ._visual_color(style,VIZ._VisualRole(:background)))
            empty!(axes);empty!(targets[]);axis[]=nothing
            strips=Any[];bars=Any[]
            for (i,panel) in enumerate(snapshot.panels)
                get(panel.metadata,:panel_style,nothing)===:text_only && continue
                rows,cols=snapshot.metadata.panel_positions[i]
                grid=current[rows,cols]=M.GridLayout()
                ax=_HANDLERS.render_panel(current,grid,panel;style)
                if haskey(panel.metadata,:pick_targets)
                    push!(axes,ax);push!(strips,ax);axis[]===nothing && (axis[]=ax)
                    append!(targets[],[(;t...,axis=ax) for t in panel.metadata.pick_targets])
                elseif panel.kind===:barcode
                    push!(bars,ax)
                end
            end
            length(strips)>1 && M.linkxaxes!(strips...)
            length(bars)>1 && M.linkxaxes!(bars...)
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
            source.value[]=string(state.source);probe.value[]=string(state.probe);baseline.value[]=string(state.baseline)
            status_text[]="View: $(state.view); probe $(state.probe); source $(state.source); differences from $(state.baseline)."
            hover_text[]="";content[]=current
            old===nothing || empty!(old)
            readout[]=_inspection_dom_readout(VIZ.VisualizationSpec(:comparison_readout;
                panels=filter(p->get(p.metadata,:panel_style,nothing)===:text_only,snapshot.panels));style)
        finally;updating[]=false;end
        nothing
    end
    listen(_->commit(()->VIZ.select_inspection!(session;source=integer(source),probe=integer(probe),baseline=Symbol(strip(baseline.value[])))),apply.value)
    listen(_->commit(()->VIZ.select_inspection!(session;view=:hilbert)),dimensions.value)
    listen(_->commit(()->VIZ.select_inspection!(session;view=:rank)),ranks.value)
    barcodes===nothing || listen(_->commit(()->VIZ.select_inspection!(session;view=:barcode)),barcodes.value)
    matrices===nothing || listen(_->commit(()->VIZ.select_inspection!(session;view=:rank,matrix=true)),matrices.value)
    listen(_->commit(()->VIZ.reset_inspection!(session)),reset.value)
    listen(_->commit(()->VIZ.close_inspection!(session)),close_button.value)
    session_token[]=VIZ._on_inspection(session,_ -> refresh())
    refresh()
    label(text,control)=B.DOM.label(B.DOM.span(text;style="display:block;font-weight:600"),control;style="display:block;min-width:0")
    row(children...)=B.DOM.div(children...;style="display:flex;flex-wrap:wrap;align-items:end;gap:$(style.gap)px;margin:$(style.gap)px 0")
    foreground=_inspection_css_color(style,VIZ._VisualRole(:foreground))
    background=_inspection_css_color(style,VIZ._VisualRole(:background))
    dom=B.DOM.div(B.DOM.h2("Compare corresponding queries"),
        B.DOM.p("Select a probe in any value or difference strip. Colors share a scale; they do not identify bases or classes. Hover only reads retained values. Live Julia is required."),
        row(dimensions,ranks,barcodes===nothing ? B.DOM.span() : barcodes,matrices===nothing ? B.DOM.span() : matrices),
        row(label("Source probe (ranks)",source),label("Selected probe",probe),label("Baseline case",baseline),apply,reset,close_button),
        B.DOM.p(error_text;id="$dom_id-error",role="alert"),
        B.DOM.p(status_text;id="$dom_id-status",role="status"),B.DOM.p(hover_text;id="$dom_id-hover",role="status"),
        B.DOM.div(content;style="max-width:100%;overflow:auto",tabindex="0",var"aria-label"="Comparison charts"),
        B.DOM.div(readout;id="$dom_id-readout");id=dom_id,
        style="box-sizing:border-box;font-family:'$(style.font)',sans-serif;font-size:$(style.fontsize)px;line-height:1.45;color:$foreground;background:$background;padding:$(style.padding)px;max-width:100%;min-width:0;overflow-wrap:anywhere")
    app=B.App(;title="TamerOp comparison inspector") do client
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
        controls=(;source,probe,baseline,apply,dimensions,ranks,barcodes,matrices,reset,close=close_button),
        callbacks=(;pick,hover,refresh),dispose,closed,disabled,observers,pointer_observers,session_token,browser_client,
        rebuild_count,error_text,status_text,hover_text,readout,dom_id,style,figure_size)
    app
end
