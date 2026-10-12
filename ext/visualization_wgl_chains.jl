# A89's browser consumes prepared chain/sequence specifications.
function _render_chain_inspector(spec;display=:inline,figure=nothing,size=nothing,style=VIZ.VisualStyle())
    VIZ._check_visual_render_options(;display,figure,size,style)
    figure===nothing || throw(ArgumentError("A live chain inspector owns its canvas; export inspection_snapshot instead."))
    session=spec.metadata.session
    VIZ.inspection_summary(session).closed && throw(ArgumentError("This chain session is closed."))
    B,M=WGLMakie.Bonito,WGLMakie.Makie
    _NEXT_INSPECTION_UI_ID[]+=1
    dom_id="tamerop-chain-inspector-$(_NEXT_INSPECTION_UI_ID[])"
    fig=Ref{Any}(nothing);axis=Ref{Any}(nothing);targets=Ref(NamedTuple[])
    disabled=M.Observable(false)
    error_text,status_text,hover_text=M.Observable(""),M.Observable(""),M.Observable("")
    content,readout=M.Observable{Any}(B.DOM.div()),M.Observable{Any}(B.DOM.div())
    control_style=_inspection_control_style(style)
    degree=B.TextField("0";id="$dom_id-degree",disabled,style=control_style)
    vertex=B.TextField("1";id="$dom_id-vertex",disabled,style=control_style)
    parameter=B.TextField("0";id="$dom_id-parameter",disabled,style=control_style)
    map_control=B.TextField("delta";id="$dom_id-map",disabled,style=control_style)
    vector=B.TextField("";id="$dom_id-vector",disabled,style=control_style)
    apply_location=B.Button("Show location";id="$dom_id-apply-location",disabled,style=control_style)
    select=B.Button("Inspect vector";id="$dom_id-select",disabled,style=control_style)
    compute=B.Checkbox(false;id="$dom_id-compute",disabled)
    class_coordinates=B.Checkbox(false;id="$dom_id-class-coordinates",disabled)
    clear=B.Button("Clear vector";id="$dom_id-clear",disabled,style=control_style)
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
        hover_text[]=t === nothing ? "" : "$(t.selection): dimension $(t.dimension)"
        return t
    end
    function pick(point)
        t=hover(point);t === nothing && return false
        return commit(()->VIZ.select_inspection!(session;t.selection...))
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
    connecting=VIZ.inspection_snapshot(session).kind==:connecting_map
    format_coefficient(x)=replace(string(hasproperty(x,:val) ? x.val : x),"//"=>"/")
    function refresh()
        VIZ.inspection_summary(session).closed && return dispose()
        snapshot=VIZ.inspection_snapshot(session);state=VIZ.inspection_selection(session)
        updating[]=true
        try
            foreach(M.off,pointer_observers);empty!(pointer_observers)
            old=fig[]
            panels=filter(p->get(p.metadata,:panel_style,nothing)!==:text_only,snapshot.panels)
            drawing=VIZ.VisualizationSpec(:chain_live;panels,metadata=VIZ._chain_layout(panels))
            current=Base.invokelatest(_BASE_WGL_RENDER,drawing;size,style)
            axis[]=first(filter(x->x isa M.Axis,current.content))
            targets[]=first(panels).metadata.pick_targets
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
            state.degree===nothing || (degree.value[]=string(state.degree))
            state.vertex===nothing || (vertex.value[]=string(state.vertex))
            get(state,:parameter,nothing)===nothing || (parameter.value[]=join(format_coefficient.(state.parameter),", "))
            connecting ? (map_control.value[]=string(state.map)) :
                (compute.value[]=state.compute;class_coordinates.value[]=state.coordinates==:class)
            vector.value[]=state.vector===nothing ? "" : isempty(state.vector) ? "[]" : join(format_coefficient.(state.vector),", ")
            status_text[]=connecting ? "Selected $(state.map). Coefficients are in the sequence's retained bases." :
                "Degree $(state.degree). "*(snapshot.metadata.quotient_computed ? "Quotient data computed for this selection." : "Quotient bases not requested.")
            hover_text[]="";content[]=current
            old===nothing || empty!(old)
            readout[]=_inspection_dom_readout(VIZ.VisualizationSpec(:chain_readout;
                panels=filter(p->get(p.metadata,:panel_style,nothing)===:text_only,snapshot.panels));style)
        finally;updating[]=false;end
        return nothing
    end
    function parsed_vector()
        text=strip(vector.value[])
        text=="[]" && return ()
        isempty(text) && throw(ArgumentError("Enter comma-separated coefficients, or [] for a zero-dimensional space."))
        f=VIZ.inspection_summary(session).field
        return Tuple(TO.CoreModules.coerce(f,VIZ._parse_inspection_coordinate(strip(s))) for s in split(text,','))
    end
    listen(apply_location.value) do _
        commit() do
            state=VIZ.inspection_selection(session)
            selections=Pair{Symbol,Any}[]
            state.degree===nothing || push!(selections,:degree=>parse_integer(degree,"degree"))
            state.vertex===nothing || push!(selections,:vertex=>parse_integer(vertex,"vertex"))
            get(state,:parameter,nothing)===nothing || push!(selections,:parameter=>Tuple(VIZ._parse_inspection_coordinate(strip(s)) for s in split(parameter.value[],',')))
            connecting && push!(selections,:map=>Symbol(strip(map_control.value[])))
            VIZ.select_inspection!(session;selections...)
        end
    end
    listen(_->commit(()->VIZ.select_inspection!(session;vector=parsed_vector())),select.value)
    if !connecting
        listen(value->commit(()->VIZ.select_inspection!(session;compute=value)),compute.value)
        listen(value->commit(()->VIZ.select_inspection!(session;coordinates=value ? :class : :ambient)),class_coordinates.value)
    end
    listen(_->commit(()->VIZ.select_inspection!(session;vector=nothing)),clear.value)
    listen(_->commit(()->VIZ.reset_inspection!(session)),reset.value)
    listen(_->commit(()->VIZ.close_inspection!(session)),close_button.value)
    session_token[]=VIZ._on_inspection(session,_ -> refresh())
    refresh()
    label(text,control)=B.DOM.label(B.DOM.span(text;style="display:block;font-weight:600"),control;style="display:block;min-width:0")
    row(children...)=B.DOM.div(children...;style="display:flex;flex-wrap:wrap;align-items:end;gap:$(style.gap)px;margin:$(style.gap)px 0")
    foreground=_inspection_css_color(style,VIZ._VisualRole(:foreground))
    background=_inspection_css_color(style,VIZ._VisualRole(:background))
    error_color=_inspection_css_color(style,VIZ._VisualRole(:error))
    initial=VIZ.inspection_selection(session)
    locations=Any[]
    initial.degree===nothing || push!(locations,label("Degree",degree))
    initial.vertex===nothing || push!(locations,label("Finite vertex",vertex))
    get(initial,:parameter,nothing)===nothing || push!(locations,label("Exact parameter",parameter))
    connecting && push!(locations,label("Sequence map",map_control))
    push!(locations,apply_location)
    choices=connecting ? B.DOM.p("Enter coordinates in the selected source space of the sequence.") :
        row(label("Compute cycle/boundary/quotient bases",compute),label("Use quotient coordinates",class_coordinates))
    dom=B.DOM.div(B.DOM.h2("Chains, cycles and connecting maps"),
        B.DOM.p("Select a degree or sequence arrow, then enter a vector. Hover reads dimensions only. Coefficients use the declared field; fractions are interpreted in that field."),
        row(locations...),choices,row(label("Vector coefficients",vector),select,clear,reset,close_button),
        B.DOM.p(error_text;id="$dom_id-error",role="alert",style="color:$error_color;white-space:pre-line"),
        B.DOM.p(status_text;id="$dom_id-status",role="status"),B.DOM.p(hover_text;id="$dom_id-hover",role="status"),
        B.DOM.div(content;style="max-width:100%;overflow:auto",tabindex="0",var"aria-label"="Chain and connecting-map charts"),
        B.DOM.div(readout;id="$dom_id-readout");id=dom_id,
        style="box-sizing:border-box;font-family:'$(style.font)',sans-serif;font-size:$(style.fontsize)px;line-height:1.45;color:$foreground;background:$background;padding:$(style.padding)px;max-width:100%;min-width:0;overflow-wrap:anywhere")
    app=B.App(;title="TamerOp chain inspector") do client
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
    _INSPECTION_UI_HANDLES[app]=(;session,figure=fig,axis,targets,
        controls=(;degree,vertex,parameter,map=map_control,vector,apply_location,select,compute,class_coordinates,clear,reset,close=close_button),
        callbacks=(;pick,hover,refresh),dispose,closed,disabled,observers,pointer_observers,session_token,browser_client,
        rebuild_count,error_text,status_text,hover_text,readout,dom_id,style)
    return app
end
