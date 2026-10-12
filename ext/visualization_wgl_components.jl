# Prepared source memberships and merge records drive all live interactions.
function _render_component_inspector(spec;display=:inline,figure=nothing,size=nothing,style=VIZ.VisualStyle())
    VIZ._check_visual_render_options(;display,figure,size,style)
    figure===nothing || throw(ArgumentError("A component inspector owns its canvas; export its static snapshot instead."))
    session=spec.metadata.session
    VIZ.inspection_summary(session).closed && throw(ArgumentError("This component session is closed."))
    B,M=WGLMakie.Bonito,WGLMakie.Makie
    _NEXT_INSPECTION_UI_ID[]+=1;dom_id="tamerop-components-$(_NEXT_INSPECTION_UI_ID[])"
    fig=Ref{Any}(nothing);axis=Ref{Any}(nothing);targets=Ref(NamedTuple[]);chart_axes=Ref(Any[])
    disabled=M.Observable(false);error_text=M.Observable("");status_text=M.Observable("");hover_text=M.Observable("")
    content=M.Observable{Any}(B.DOM.div());readout=M.Observable{Any}(B.DOM.div())
    control_style=_inspection_control_style(style)
    scale=B.TextField("0";id="$dom_id-scale",disabled,style=control_style)
    component=B.TextField("";id="$dom_id-component",disabled,style=control_style)
    event=B.TextField("";id="$dom_id-event",disabled,style=control_style)
    apply_scale=B.Button("Show scale";id="$dom_id-apply-scale",disabled,style=control_style)
    select=B.Button("Select component";id="$dom_id-select",disabled,style=control_style)
    select_event=B.Button("Inspect merge";id="$dom_id-select-event",disabled,style=control_style)
    previous=B.Button("Previous event";id="$dom_id-previous",disabled,style=control_style)
    next=B.Button("Next event";id="$dom_id-next",disabled,style=control_style)
    clear=B.Button("Clear selection";id="$dom_id-clear",disabled,style=control_style)
    reset=B.Button("Reset";id="$dom_id-reset",disabled,style=control_style)
    close_button=B.Button("Close inspector";id="$dom_id-close",disabled,style=control_style)
    closed,updating=Ref(false),Ref(false);observers=Any[];pointer_observers=Any[]
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
            updating[] || closed[] || f(value);nothing
        end)
    end
    function parsed_id(control,name)
        text=strip(control.value[]);isempty(text) && return nothing
        id=tryparse(Int,text);id===nothing && throw(ArgumentError("Enter an integer $name, or leave it blank."));id
    end
    function nearby(point,index=1)
        index<=length(chart_axes[]) || return nothing
        ax=chart_axes[][index];ax isa M.Axis || return nothing
        spans,pixels=M.widths(ax.finallimits[]),M.widths(ax.scene.viewport[])
        all(>(0),spans) && all(>(0),pixels) || return nothing
        candidates=filter(t->t.axis_index==index,targets[]);isempty(candidates) && return nothing
        scaling=pixels ./ spans;distances=[sum(((point[k]-t.point[k])*scaling[k])^2 for k in 1:2) for t in candidates]
        i=argmin(distances);distances[i]<=22^2 ? candidates[i] : nothing
    end
    function hover(point,index=1)
        t=nearby(point,index);hover_text[]=t===nothing ? "" : t.label;t
    end
    function pick(point,index=1)
        t=hover(point,index);t===nothing && return false
        commit(()->VIZ.select_inspection!(session;t.selection...))
    end
    function dispose(;close_session=false)
        closed[] && return nothing
        closed[]=true;disabled[]=true
        session_token[]==0 || VIZ._off_inspection!(session,session_token[]);session_token[]=0
        foreach(M.off,observers);empty!(observers);foreach(M.off,pointer_observers);empty!(pointer_observers)
        fig[]===nothing || empty!(fig[])
        axis[]=nothing;empty!(chart_axes[]);empty!(targets[])
        content[]=B.DOM.p("Inspector closed. Its final static snapshot remains available.")
        lock(browser_lock) do;browser_client[]=nothing;end
        app=app_ref[].value;app===nothing || delete!(_INSPECTION_UI_HANDLES,app)
        close_session && VIZ.close_inspection!(session);nothing
    end
    function refresh()
        VIZ.inspection_summary(session).closed && return dispose()
        snapshot=VIZ.inspection_snapshot(session);state=VIZ.inspection_selection(session)
        updating[]=true
        try
            foreach(M.off,pointer_observers);empty!(pointer_observers)
            old=fig[];panels=filter(p->get(p.metadata,:panel_style,nothing)!==:text_only,snapshot.panels)
            drawing=VIZ.VisualizationSpec(:component_live;panels,metadata=(;panel_columns=2,figure_size=(1400,950)))
            current=Base.invokelatest(_BASE_WGL_RENDER,drawing;size,style)
            chart_axes[]=filter(a->a isa Union{M.Axis,M.Axis3},current.content)
            axis[]=first(chart_axes[]);targets[]=NamedTuple[]
            for (i,p) in enumerate(panels),t in p.metadata.pick_targets
                push!(targets[],merge(t,(;axis_index=i)))
            end
            fig[]=current;rebuild_count[]+=1
            push!(pointer_observers,M.on(M.events(current.scene).mouseposition) do _
                closed[] && return nothing
                hovered=false
                for (i,ax) in enumerate(chart_axes[])
                    ax isa M.Axis && M.is_mouseinside(ax.scene) || continue
                    hover(M.mouseposition(ax.scene),i);hovered=true;break
                end
                hovered || (hover_text[]="");nothing
            end)
            push!(pointer_observers,M.on(M.events(current.scene).mousebutton;priority=10) do e
                if !closed[] && e.button==M.Mouse.left && e.action==M.Mouse.press
                    for (i,ax) in enumerate(chart_axes[])
                        ax isa M.Axis && M.is_mouseinside(ax.scene) || continue
                        return M.Consume(pick(M.mouseposition(ax.scene),i))
                    end
                end
                M.Consume(false)
            end)
            scale.value[]=replace(string(state.scale),"//"=>"/")
            component.value[]=state.component===nothing ? "" : string(state.component)
            event.value[]=state.event===nothing ? "" : string(state.event)
            status_text[]="$(length(snapshot.metadata.partition)) components at source parameter $(snapshot.metadata.parameter)."
            hover_text[]="";content[]=current;old===nothing || empty!(old)
            readout[]=_inspection_dom_readout(VIZ.VisualizationSpec(:component_readout;
                panels=filter(p->get(p.metadata,:panel_style,nothing)===:text_only,snapshot.panels));style)
        finally;updating[]=false;end
        nothing
    end
    listen(_->commit(()->VIZ.select_inspection!(session;scale=VIZ._parse_inspection_coordinate(strip(scale.value[])))),apply_scale.value)
    listen(_->commit(()->VIZ.select_inspection!(session;component=parsed_id(component,"birth-vertex index"),event=nothing)),select.value)
    listen(_->commit(()->VIZ.select_inspection!(session;event=parsed_id(event,"merge-node index"))),select_event.value)
    function step(forward)
        h=session.object;t=TO.OrdinaryPersistence._component_progress(h,VIZ.inspection_selection(session).scale)
        events=sort!(unique(vcat(h.births,h.edge_times)))
        candidates=filter(v->forward ? v>t : v<t,events)
        isempty(candidates) && throw(ArgumentError(forward ? "There is no later source event." : "There is no earlier source event."))
        next_t=forward ? first(candidates) : last(candidates)
        VIZ.select_inspection!(session;scale=TO.OrdinaryPersistence._component_external(h,next_t))
    end
    listen(_->commit(()->step(false)),previous.value);listen(_->commit(()->step(true)),next.value)
    listen(_->commit(()->VIZ.select_inspection!(session;component=nothing,event=nothing)),clear.value)
    listen(_->commit(()->VIZ.reset_inspection!(session)),reset.value)
    listen(_->commit(()->VIZ.close_inspection!(session)),close_button.value)
    session_token[]=VIZ._on_inspection(session,_ -> refresh());refresh()
    label(text,control)=B.DOM.label(B.DOM.span(text;style="display:block;font-weight:600"),control;style="display:block;min-width:0")
    row(children...)=B.DOM.div(children...;style="display:flex;flex-wrap:wrap;align-items:end;gap:$(style.gap)px;margin:$(style.gap)px 0")
    foreground=_inspection_css_color(style,VIZ._VisualRole(:foreground));background=_inspection_css_color(style,VIZ._VisualRole(:background))
    error_color=_inspection_css_color(style,VIZ._VisualRole(:error))
    dom=B.DOM.div(B.DOM.h2("Components and merge history"),
        B.DOM.p("Select a bar, a source vertex or a merge node. Exact scale fields include all simultaneous events. Hover reads prepared identities only."),
        row(label("Exact scale",scale),apply_scale,previous,next),row(label("Component birth vertex",component),select,label("Merge node",event),select_event),
        row(clear,reset,close_button),
        B.DOM.p(error_text;id="$dom_id-error",role="alert",style="color:$error_color;white-space:pre-line"),
        B.DOM.p(status_text;id="$dom_id-status",role="status"),B.DOM.p(hover_text;id="$dom_id-hover",role="status"),
        B.DOM.div(content;style="max-width:100%;overflow:auto",tabindex="0",var"aria-label"="Component and merge-history charts"),
        B.DOM.div(readout;id="$dom_id-readout");id=dom_id,
        style="box-sizing:border-box;font-family:'$(style.font)',sans-serif;font-size:$(style.fontsize)px;line-height:1.45;color:$foreground;background:$background;padding:$(style.padding)px;max-width:100%;min-width:0;overflow-wrap:anywhere")
    app=B.App(;title="TamerOp component inspector") do client
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
    _INSPECTION_UI_HANDLES[app]=(;session,figure=fig,axis,targets,chart_axes,
        controls=(;scale,component,event,apply_scale,select,select_event,previous,next,clear,reset,close=close_button),
        callbacks=(;pick,hover,refresh),dispose,closed,disabled,observers,pointer_observers,session_token,browser_client,
        rebuild_count,error_text,status_text,hover_text,readout,dom_id,style)
    app
end
