# Browser transport selection uses the same verified preparation and snapshots.
function _render_transport_inspector(spec;display=:inline,figure=nothing,size=nothing,style=VIZ.VisualStyle())
    VIZ._check_visual_render_options(;display,figure,size,style)
    figure === nothing || throw(ArgumentError("A live transport inspector owns its canvas; compose an inspection_snapshot for static figures."))
    session=spec.metadata.session
    VIZ.inspection_summary(session).closed && throw(ArgumentError("This transport session is closed."))
    B,M=WGLMakie.Bonito,WGLMakie.Makie
    _NEXT_INSPECTION_UI_ID[]+=1
    dom_id="tamerop-transport-inspector-$(_NEXT_INSPECTION_UI_ID[])"
    figure_size=size === nothing ? (1500,1150) : size
    fig=Ref{Any}(nothing);axis=Ref{Any}(nothing);targets=Ref(NamedTuple[]);axes=Any[]
    disabled=M.Observable(false)
    error_text,status_text,hover_text=M.Observable(""),M.Observable(""),M.Observable("")
    content,readout=M.Observable{Any}(B.DOM.div()),M.Observable{Any}(B.DOM.div())
    control_style=_inspection_control_style(style)
    side=B.TextField("base";id="$dom_id-side",disabled,style=control_style)
    vertex=B.TextField("1";id="$dom_id-vertex",disabled,style=control_style)
    select=B.Button("Select label";id="$dom_id-select",disabled,style=control_style)
    p=B.TextField("1";id="$dom_id-p",disabled,style=control_style)
    q=B.TextField("2";id="$dom_id-q",disabled,style=control_style)
    square=B.Button("Inspect square";id="$dom_id-square",disabled,style=control_style)
    overview=B.Button("Overview";id="$dom_id-overview",disabled,style=control_style)
    x=B.TextField("0";id="$dom_id-x",disabled,style=control_style)
    y=B.TextField("0";id="$dom_id-y",disabled,style=control_style)
    point=B.Button("Locate parameter";id="$dom_id-point",disabled,style=control_style)
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
    function nearby(point,ax=axis[])
        ax===nothing && return nothing
        spans,pixels=M.widths(ax.finallimits[]),M.widths(ax.scene.viewport[])
        all(>(0),spans) && all(>(0),pixels) || return nothing
        scale=pixels ./ spans
        ts=filter(t->t.axis===ax,targets[])
        isempty(ts) && return nothing
        d=[sum(((point[i]-t.point[i])*scale[i])^2 for i in 1:2) for t in ts]
        id=argmin(d)
        return d[id]<=20^2 ? ts[id] : nothing
    end
    function hover(point,ax=axis[])
        t=nearby(point,ax)
        hover_text[]=t===nothing ? "" : "$(t.side) label $(t.label); click to show all common preimages"
        return t
    end
    function pick(point,ax=axis[])
        t=hover(point,ax);t===nothing && return false
        return commit(()->VIZ.select_inspection!(session;side=t.side,label=t.label,view=:overview))
    end
    function dispose(;close_session=false)
        closed[] && return nothing
        closed[]=true;disabled[]=true
        session_token[]==0 || VIZ._off_inspection!(session,session_token[])
        session_token[]=0
        foreach(M.off,observers);empty!(observers)
        foreach(M.off,pointer_observers);empty!(pointer_observers)
        fig[] === nothing || empty!(fig[])
        axis[]=nothing;empty!(targets[]);empty!(axes)
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
            empty!(axes);empty!(targets[]);axis[]=nothing
            cols=snapshot.metadata.view===:overview ? 3 : 2
            for (i,panel) in enumerate(panels)
                grid=current[cld(i,cols),mod1(i,cols)]=M.GridLayout()
                ax=_HANDLERS.render_panel(current,grid,panel;style)
                if haskey(panel.metadata,:pick_targets)
                    push!(axes,ax);axis[]===nothing && (axis[]=ax)
                    append!(targets[],[(;t...,axis=ax) for t in panel.metadata.pick_targets])
                end
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
            if state.pair!==nothing
                p.value[]=string(state.pair[1]);q.value[]=string(state.pair[2])
            end
            status_text[]="$(snapshot.metadata.refinement); selection $(snapshot.metadata.selection.status). " * VIZ._TRANSPORT_CATEGORY
            hover_text[]=""
            content[]=current
            old === nothing || empty!(old)
            readout[]=_inspection_dom_readout(VIZ.VisualizationSpec(:transport_readout;
                panels=filter(p->get(p.metadata,:panel_style,nothing)===:text_only,snapshot.panels));style)
        finally;updating[]=false;end
        return nothing
    end
    listen(_->commit(()->VIZ.select_inspection!(session;side=Symbol(strip(side.value[])),label=parse_integer(vertex,"label"),view=:overview)),select.value)
    listen(_->commit(()->VIZ.select_inspection!(session;pair=(parse_integer(p,"source"),parse_integer(q,"target")),view=:square)),square.value)
    listen(_->commit(()->VIZ.select_inspection!(session;view=:overview)),overview.value)
    listen(_->commit(()->VIZ.select_inspection!(session;point=(VIZ._parse_inspection_coordinate(x.value[]),VIZ._parse_inspection_coordinate(y.value[])),view=:overview)),point.value)
    listen(_->commit(()->VIZ.select_inspection!(session;vertex=nothing,view=:overview)),clear.value)
    listen(_->commit(()->VIZ.reset_inspection!(session)),reset.value)
    listen(_->commit(()->VIZ.close_inspection!(session)),close_button.value)
    session_token[]=VIZ._on_inspection(session,_ -> refresh())
    refresh()
    label(text,control)=B.DOM.label(B.DOM.span(text;style="display:block;font-weight:600"),control;style="display:block;min-width:0")
    row(children...)=B.DOM.div(children...;style="display:flex;flex-wrap:wrap;align-items:end;gap:$(style.gap)px;margin:$(style.gap)px 0")
    foreground=_inspection_css_color(style,VIZ._VisualRole(:foreground))
    background=_inspection_css_color(style,VIZ._VisualRole(:background))
    error_color=_inspection_css_color(style,VIZ._VisualRole(:error))
    dom=B.DOM.div(B.DOM.h2("Refinement and change of poset"),
        B.DOM.p("Select labels in any finite-poset panel. A label with several common preimages highlights all of them. Inspect a directed pair for actual matrices and a supplied comparison square. Hover reads retained labels only."),
        row(label("Side: base / left / right",side),label("Label",vertex),select),
        row(label("Common source",p),label("Common target",q),square,overview),
        row(label("Parameter x (exact fractions accepted)",x),label("Parameter y",y),point),
        row(clear,reset,close_button),
        B.DOM.p(error_text;id="$dom_id-error",role="alert",style="color:$error_color;white-space:pre-line"),
        B.DOM.p(status_text;id="$dom_id-status",role="status"),B.DOM.p(hover_text;id="$dom_id-hover",role="status"),
        B.DOM.div(content;style="max-width:100%;overflow:auto",tabindex="0",var"aria-label"="Transport comparison charts"),
        B.DOM.div(readout;id="$dom_id-readout");id=dom_id,
        style="box-sizing:border-box;font-family:'$(style.font)',sans-serif;font-size:$(style.fontsize)px;line-height:1.45;color:$foreground;background:$background;padding:$(style.padding)px;max-width:100%;min-width:0;overflow-wrap:anywhere")
    app=B.App(;title="TamerOp transport inspector") do client
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
    _INSPECTION_UI_HANDLES[app]=(;session,figure=fig,axis,axes,targets,controls=(;side,vertex,p,q,select,square,overview,x,y,point,clear,reset,close=close_button),
        callbacks=(;pick,hover,refresh),dispose,closed,disabled,observers,pointer_observers,session_token,browser_client,
        rebuild_count,error_text,status_text,hover_text,readout,dom_id,style,figure_size)
    return app
end
