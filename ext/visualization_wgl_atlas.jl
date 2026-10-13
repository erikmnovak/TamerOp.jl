# A fixed contact sheet and bounded, replaceable transition detail.
function _render_atlas_inspector(spec;display=:inline,figure=nothing,size=nothing,style=VIZ.VisualStyle())
    VIZ._check_visual_render_options(;display,figure,size,style)
    figure===nothing || throw(ArgumentError("An atlas inspector owns its canvas; export its snapshot instead."))
    session=spec.metadata.session;B,M=WGLMakie.Bonito,WGLMakie.Makie
    VIZ.inspection_summary(session).closed && throw(ArgumentError("This atlas is closed."))
    _NEXT_INSPECTION_UI_ID[]+=1;dom_id="tamerop-atlas-inspector-$(_NEXT_INSPECTION_UI_ID[])"
    disabled=M.Observable(false);closed=Ref(false);updating=Ref(false)
    browser_client=Ref{Any}(nothing);app_ref=Ref(WeakRef(nothing));owner=Ref(nothing)
    error_text=M.Observable("");status_text=M.Observable("");hover_text=M.Observable("")
    readout=M.Observable{Any}(B.DOM.div());detail=M.Observable{Any}(B.DOM.div());detail_figure=Ref{Any}(nothing)
    cs=_inspection_control_style(style)
    source=B.TextField(string(session.state.source);id="$dom_id-source",disabled,style=cs)
    target=B.TextField(string(something(session.state.target,session.state.source));id="$dom_id-target",disabled,style=cs)
    witness=B.TextField("1";id="$dom_id-witness",disabled,style=cs)
    endpoint=B.Checkbox(false;id="$dom_id-endpoint",disabled)
    inspect=B.Button("Inspect map";id="$dom_id-inspect",disabled,style=cs)
    entries=B.Button("Cell entries";id="$dom_id-entries",disabled,style=cs)
    kernel=B.Button("Vanishing class";id="$dom_id-kernel",disabled,style=cs)
    cokernel=B.Button("New target class";id="$dom_id-cokernel",disabled,style=cs)
    reset=B.Button("Reset";id="$dom_id-reset",disabled,style=cs)
    close_button=B.Button("Close inspector";id="$dom_id-close",disabled,style=cs)
    observers=Any[];pointer_observers=Any[];token=Ref(0);rebuild_count=Ref(0)
    initial=VIZ.inspection_snapshot(session);n=initial.metadata.state_panel_count
    columns=initial.metadata.panel_columns
    contact=VIZ.VisualizationSpec(:atlas_contact;panels=[s.view for s in session.object.states],
        metadata=(;panel_columns=columns,figure_size=(max(900,columns*420),cld(n,columns)*360+70)))
    # Use parameter/dimension titles from the public specification.
    contact=VIZ._composition_copy(contact;panels=[VIZ._composition_copy(initial.panels[i];
        title="$(i). $(session.object.states[i].parameter)") for i in 1:n])
    fig=Base.invokelatest(_BASE_WGL_RENDER,contact;size,style)
    axes=filter(x->x isa Union{M.Axis,M.Axis3},fig.content)
    function status(stage,error)
        closed[] && return nothing
        error_text[]=stage===:error ? "Selection unchanged: $error" : ""
        stage===:pending && (status_text[]="Updating the selected transition\u2026")
        nothing
    end
    function request(;kwargs...)
        closed[] && return false
        try
            _inspection_control_selection!(session,owner,browser_client[],status;kwargs...)
            return true
        catch err
            err isa InterruptException && rethrow();error_text[]=sprint(showerror,err);false
        end
    end
    integer(c,name)=let value=tryparse(Int,strip(c.value[]))
        value===nothing && throw(ArgumentError("Enter an integer $name."));value
    end
    function controls_request(event)
        try
            request(;source=integer(source,"source state"),target=integer(target,"target state"),event,
                witness=event in (:kernel,:cokernel) ? integer(witness,"witness index") : 1)
        catch err
            err isa InterruptException && rethrow();error_text[]=sprint(showerror,err)
        end
    end
    function listen(f,observable)
        push!(observers,M.on(observable) do value
            updating[] || closed[] || f(value);nothing
        end)
    end
    function pick(index)
        request(;(endpoint.value[] ? (;target=index) : (;source=index))...)
    end
    function refresh()
        VIZ.inspection_summary(session).closed && return dispose()
        _inspection_browser_ended(browser_client[]) && return dispose()
        snapshot=VIZ.inspection_snapshot(session);state=VIZ.inspection_selection(session)
        updating[]=true
        try
            source.value[]=string(state.source);target.value[]=string(something(state.target,state.source));witness.value[]=string(state.witness)
            t=snapshot.metadata.transition
            status_text[]=t===nothing ? "Selected state $(state.source)." : t.defined ?
                "State $(state.source) \u2192 $(state.target): rank $(t.rank), kernel $(t.kernel_dimension), cokernel $(t.cokernel_dimension)." :
                "State $(state.source) \u2192 $(state.target): no map ($(t.relation))."
            readout[]=_inspection_dom_readout(VIZ.VisualizationSpec(:atlas_readout;
                panels=filter(p->get(p.metadata,:panel_style,nothing) in (:text_only,:matrix),snapshot.panels));style)
            panels=filter(p->!(get(p.metadata,:panel_style,nothing) in (:text_only,:matrix)),snapshot.panels[n+1:end])
            old=detail_figure[]
            if isempty(panels)
                detail_figure[]=nothing;detail[]=B.DOM.div()
            else
                drawing=VIZ.VisualizationSpec(:atlas_transition;panels,
                    metadata=(;panel_columns=min(3,length(panels)),figure_size=(max(720,420*length(panels)),460)))
                current=Base.invokelatest(_BASE_WGL_RENDER,drawing;style)
                detail_figure[]=current;detail[]=current
            end
            rebuild_count[]+=1;old===nothing || empty!(old)
        finally;updating[]=false;end
        nothing
    end
    function dispose(;close_session=false)
        closed[] && return nothing
        closed[]=true;disabled[]=true;VIZ._cancel_inspection_requests!(session,owner)
        token[]==0 || VIZ._off_inspection!(session,token[]);token[]=0
        foreach(M.off,observers);empty!(observers);foreach(M.off,pointer_observers);empty!(pointer_observers)
        empty!(fig);detail_figure[]===nothing || empty!(detail_figure[])
        detail_figure[]=nothing;browser_client[]=nothing
        detail[]=B.DOM.p("Inspector closed. The final static snapshot remains available.")
        app=app_ref[].value;app===nothing || delete!(_INSPECTION_UI_HANDLES,app)
        close_session && VIZ.close_inspection!(session)
        nothing
    end
    push!(pointer_observers,M.on(M.events(fig.scene).mouseposition) do _
        closed[] && return nothing
        i=findfirst(ax->M.is_mouseinside(ax.scene),axes)
        hover_text[]=i===nothing ? "" : "State $i: parameter $(initial.metadata.states[i].parameter), dimension $(initial.metadata.states[i].dimension)."
        nothing
    end)
    pressed=Ref{Any}(nothing)
    push!(pointer_observers,M.on(M.events(fig.scene).mousebutton;priority=10) do e
        if !closed[] && e.button==M.Mouse.left
            if e.action==M.Mouse.press
                i=findfirst(ax->M.is_mouseinside(ax.scene),axes)
                pressed[]=i===nothing ? nothing : (;index=i,point=M.events(fig.scene).mouseposition[])
            elseif e.action==M.Mouse.release && pressed[]!==nothing
                start=pressed[];pressed[]=nothing
                point=M.events(fig.scene).mouseposition[]
                # A drag remains a camera gesture; a short click selects a state.
                sum((point[k]-start.point[k])^2 for k in 1:2)<=25 && return M.Consume(pick(start.index))
            end
        end
        M.Consume(false)
    end)
    listen(_->controls_request(:summary),inspect.value)
    listen(_->controls_request(:entries),entries.value)
    listen(_->controls_request(:kernel),kernel.value)
    listen(_->controls_request(:cokernel),cokernel.value)
    listen(_->request(;session.initial...),reset.value)
    listen(_->dispose(;close_session=true),close_button.value)
    token[]=VIZ._on_inspection(session,_->refresh());refresh()
    label(text,control)=B.DOM.label(B.DOM.span(text;style="display:block;font-weight:600"),control;style="min-width:0")
    row(children...)=B.DOM.div(children...;style="display:flex;flex-wrap:wrap;align-items:end;gap:$(style.gap)px;margin:$(style.gap)px 0")
    state_buttons=map(1:n) do i
        button=B.Button("State $i";id="$dom_id-state-$i",disabled,style=cs)
        listen(_->pick(i),button.value);button
    end
    foreground=_inspection_css_color(style,VIZ._VisualRole(:foreground))
    background=_inspection_css_color(style,VIZ._VisualRole(:background))
    dom=B.DOM.div(B.DOM.h2("A parameter atlas"),
        B.DOM.p("Explore the declared states and their actual maps. The samples do not certify every intervening event. Click a panel or use its state button; hovering only reads retained information."),
        row(label("Source state",source),label("Target state",target),inspect,label("Select target with a panel",endpoint),reset,close_button),
        row(state_buttons...),
        B.DOM.p(error_text;id="$dom_id-error",role="alert",style="white-space:pre-line"),
        B.DOM.p(status_text;id="$dom_id-status",role="status"),B.DOM.p(hover_text;id="$dom_id-hover",role="status"),
        B.DOM.div(fig;style="max-width:100%;overflow:auto",tabindex="0",var"aria-label"="Aligned parameter states"),
        row(entries,label("Witness index",witness),kernel,cokernel),
        B.DOM.div(detail;id="$dom_id-detail",style="max-width:100%;overflow:auto",tabindex="0",var"aria-label"="Selected transition and witnesses"),
        B.DOM.div(readout;id="$dom_id-readout");id=dom_id,
        style="box-sizing:border-box;font-family:'$(style.font)',sans-serif;font-size:$(style.fontsize)px;line-height:1.45;color:$foreground;background:$background;padding:$(style.padding)px;max-width:100%;overflow-wrap:anywhere")
    app=B.App(;title="TamerOp parameter atlas") do client
        closed[] && return B.DOM.p("This inspector has been closed.")
        if browser_client[]!==nothing && browser_client[]!==client
            return B.DOM.p("This widget already has a browser client. Call visualize(session) again for a linked view.")
        end
        if browser_client[]===nothing
            browser_client[]=client;_inspection_keyboard_navigation!(client,dom)
            push!(observers,M.on(client.on_close) do ended
                ended && dispose();nothing
            end)
        end
        dom
    end
    app_ref[]=WeakRef(app)
    _INSPECTION_UI_HANDLES[app]=(;session,figure=fig,axes,detail_figure,
        controls=(;source,target,witness,endpoint,inspect,entries,kernel,cokernel,reset,close=close_button),
        callbacks=(;pick,refresh),dispose,closed,disabled,observers,pointer_observers,session_token=token,browser_client,
        rebuild_count,error_text,status_text,hover_text,readout,dom_id,style)
    app
end
