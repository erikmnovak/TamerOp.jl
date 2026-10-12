# Browser page navigation uses the same session and prepared specifications.
function _render_spectral_inspector(spec;display=:inline,figure=nothing,size=nothing,style=VIZ.VisualStyle())
    VIZ._check_visual_render_options(;display,figure,size,style)
    figure === nothing || throw(ArgumentError("A live spectral inspector owns its canvas; compose an inspection_snapshot for static figures."))
    session=spec.metadata.session
    VIZ.inspection_summary(session).closed && throw(ArgumentError("This spectral session is closed."))
    B,M=WGLMakie.Bonito,WGLMakie.Makie
    _NEXT_INSPECTION_UI_ID[]+=1
    dom_id="tamerop-spectral-inspector-$(_NEXT_INSPECTION_UI_ID[])"
    figure_size=size === nothing ? (1200,900) : size
    fig=Ref{Any}(nothing);axis=Ref{Any}(nothing);targets=Ref(NamedTuple[])
    disabled=M.Observable(false)
    error_text,status_text,hover_text=M.Observable(""),M.Observable(""),M.Observable("")
    content,readout=M.Observable{Any}(B.DOM.div()),M.Observable{Any}(B.DOM.div())
    control_style=_inspection_control_style(style)
    page=B.TextField("2";id="$dom_id-page",disabled,style=control_style)
    apply_page=B.Button("Show page";id="$dom_id-apply-page",disabled,style=control_style)
    previous=B.Button("Previous page";id="$dom_id-previous",disabled,style=control_style)
    next=B.Button("Next page";id="$dom_id-next",disabled,style=control_style)
    limit=B.Button("Associated graded limit";id="$dom_id-limit",disabled,style=control_style)
    compute=B.Checkbox(true;id="$dom_id-compute",disabled)
    p=B.TextField("0";id="$dom_id-p",disabled,style=control_style)
    q=B.TextField("0";id="$dom_id-q",disabled,style=control_style)
    select=B.Button("Inspect outgoing differential";id="$dom_id-select",disabled,style=control_style)
    clear=B.Button("Clear selection";id="$dom_id-clear",disabled,style=control_style)
    reset=B.Button("Reset";id="$dom_id-reset",disabled,style=control_style)
    close_button=B.Button("Close inspector";id="$dom_id-close",disabled,style=control_style)
    closed,updating=Ref(false),Ref(false)
    observers=Any[];pointer_observers=Any[]
    session_token,rebuild_count=Ref(0),Ref(0)
    browser_client=Ref{Any}(nothing);browser_lock=ReentrantLock();app_ref=Ref(WeakRef(nothing))
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
    function parse_integer(control,name)
        n=tryparse(Int,strip(control.value[]))
        n === nothing && throw(ArgumentError("Enter an integer $name."))
        return n
    end
    function nearby(point)
        axis[] === nothing && return nothing
        spans,pixels=M.widths(axis[].finallimits[]),M.widths(axis[].scene.viewport[])
        all(>(0),spans) && all(>(0),pixels) || return nothing
        scale=pixels ./ spans
        d=[sum(((point[i]-t.point[i])*scale[i])^2 for i in 1:2) for t in targets[]]
        isempty(d) && return nothing
        id=argmin(d)
        return d[id]<=20^2 ? targets[][id] : nothing
    end
    function hover(point)
        t=nearby(point)
        hover_text[]=t === nothing ? "" : "$(t.bidegree): " * (t.dimension === nothing ? "not computed" : "dimension $(t.dimension)")
        return t
    end
    function pick(point)
        t=hover(point);t === nothing && return false
        return commit(()->VIZ.select_inspection!(session;source=t.bidegree))
    end
    function dispose(;close_session=false)
        closed[] && return nothing
        closed[]=true;disabled[]=true
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
        VIZ.inspection_summary(session).closed && return dispose()
        snapshot=VIZ.inspection_snapshot(session)
        state=VIZ.inspection_selection(session)
        updating[]=true
        try
            foreach(M.off,pointer_observers);empty!(pointer_observers)
            old=fig[]
            current=M.Figure(size=figure_size,figure_padding=style.padding,
                backgroundcolor=VIZ._visual_color(style,VIZ._VisualRole(:background)))
            panels=filter(p->get(p.metadata,:panel_style,nothing)!==:text_only,snapshot.panels)
            for (i,panel) in enumerate(panels)
                grid = if length(panels)==1
                    current[1,1]=M.GridLayout()
                elseif i==1
                    current[1:(length(panels)-1),1]=M.GridLayout()
                else
                    current[i-1,2]=M.GridLayout()
                end
                ax=_HANDLERS.render_panel(current,grid,panel;style)
                if i==1
                    axis[]=ax;targets[]=panel.metadata.pick_targets
                end
            end
            _HANDLERS.style_figure(current,style)
            fig[]=current;rebuild_count[]+=1
            push!(pointer_observers,M.on(M.events(current.scene).mouseposition) do _
                closed[] || (M.is_mouseinside(axis[].scene) ? hover(M.mouseposition(axis[].scene)) : (hover_text[]=""))
                nothing
            end)
            push!(pointer_observers,M.on(M.events(current.scene).mousebutton;priority=10) do event
                if !closed[] && event.button==M.Mouse.left && event.action==M.Mouse.press && M.is_mouseinside(axis[].scene)
                    return M.Consume(pick(M.mouseposition(axis[].scene)))
                end
                M.Consume(false)
            end)
            page.value[]=string(state.page);compute.value[]=state.compute
            if state.source !== nothing
                p.value[]=string(state.source[1]);q.value[]=string(state.source[2])
            end
            status_text[]="$(VIZ._spectral_label(state.page)): $(snapshot.metadata.page_status). " *
                (snapshot.metadata.limiting_dimensions ? "Limiting dimensions of the supplied complex." : "No limiting-page claim.")
            hover_text[]=""
            content[]=current
            old === nothing || empty!(old)
            readout[]=_inspection_dom_readout(VIZ.VisualizationSpec(:spectral_readout;
                panels=filter(p->get(p.metadata,:panel_style,nothing)===:text_only,snapshot.panels));style)
        finally;updating[]=false;end
        return nothing
    end
    listen(apply_page.value) do _
        commit() do
            r=strip(page.value[])=="inf" ? :inf : parse_integer(page,"page")
            VIZ.select_inspection!(session;page=r,source=nothing)
        end
    end
    for (button,delta) in ((previous,-1),(next,1))
        listen(button.value) do _
            commit() do
                r=VIZ.inspection_selection(session).page
                bound=VIZ.inspection_summary(session).convergence_bound
                r=r === :inf ? bound : clamp(r,1,bound)
                VIZ.select_inspection!(session;page=clamp(r+delta,1,bound),source=nothing)
            end
        end
    end
    listen(_->commit(()->VIZ.select_inspection!(session;page=:inf,source=nothing)),limit.value)
    listen(value->commit(()->VIZ.select_inspection!(session;compute=value)),compute.value)
    listen(_->commit(()->VIZ.select_inspection!(session;source=(parse_integer(p,"p"),parse_integer(q,"q")))),select.value)
    listen(_->commit(()->VIZ.select_inspection!(session;source=nothing)),clear.value)
    listen(_->commit(()->VIZ.reset_inspection!(session)),reset.value)
    listen(_->commit(()->VIZ.close_inspection!(session)),close_button.value)
    session_token[]=VIZ._on_inspection(session,_ -> refresh())
    refresh()
    label(text,control)=B.DOM.label(B.DOM.span(text;style="display:block;font-weight:600"),control;style="display:block;min-width:0")
    row(children...)=B.DOM.div(children...;style="display:flex;flex-wrap:wrap;align-items:end;gap:$(style.gap)px;margin:$(style.gap)px 0")
    foreground=_inspection_css_color(style,VIZ._VisualRole(:foreground))
    background=_inspection_css_color(style,VIZ._VisualRole(:background))
    error_color=_inspection_css_color(style,VIZ._VisualRole(:error))
    dom=B.DOM.div(B.DOM.h2("Spectral pages and differentials"),
        B.DOM.p("A dot is a page term; its label is the dimension. Select a source to inspect its actual outgoing differential. Hover reads retained dimensions only."),
        row(label("Page (or inf)",page),apply_page,previous,next,limit),
        row(label("Compute requested missing data",compute)),
        row(label("Source p",p),label("Source q",q),select,clear,reset,close_button),
        B.DOM.p(error_text;id="$dom_id-error",role="alert",style="color:$error_color;white-space:pre-line"),
        B.DOM.p(status_text;id="$dom_id-status",role="status"),B.DOM.p(hover_text;id="$dom_id-hover",role="status"),
        B.DOM.div(content;style="max-width:100%;overflow:auto",tabindex="0",var"aria-label"="Spectral page charts"),
        B.DOM.div(readout;id="$dom_id-readout");id=dom_id,
        style="box-sizing:border-box;font-family:'$(style.font)',sans-serif;font-size:$(style.fontsize)px;line-height:1.45;color:$foreground;background:$background;padding:$(style.padding)px;max-width:100%;min-width:0;overflow-wrap:anywhere")
    app=B.App(;title="TamerOp spectral inspector") do client
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
    _INSPECTION_UI_HANDLES[app]=(;session,figure=fig,axis,targets,controls=(;page,apply_page,previous,next,limit,compute,p,q,select,clear,reset,close=close_button),
        callbacks=(;pick,hover,refresh),dispose,closed,disabled,observers,pointer_observers,session_token,browser_client,
        rebuild_count,error_text,status_text,hover_text,readout,dom_id,style,figure_size)
    return app
end
