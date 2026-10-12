# Exact entry and pointer selection share the transactional region session.
function _render_generalized_rank_inspector(spec;display=:inline,figure=nothing,size=nothing,style=VIZ.VisualStyle())
    VIZ._check_visual_render_options(;display,figure,size,style)
    figure===nothing || throw(ArgumentError("A generalized-rank inspector owns its canvas."))
    session=spec.metadata.session
    VIZ.inspection_summary(session).closed && throw(ArgumentError("This region session is closed."))
    B,M=WGLMakie.Bonito,WGLMakie.Makie
    _NEXT_INSPECTION_UI_ID[]+=1;dom_id="tamerop-region-inspector-$(_NEXT_INSPECTION_UI_ID[])"
    figure_size=size;fig=Ref{Any}(nothing);axis=Ref{Any}(nothing);axes=Any[];targets=Ref(NamedTuple[])
    disabled=M.Observable(false);error_text=M.Observable("");status_text=M.Observable("");hover_text=M.Observable("")
    content=M.Observable{Any}(B.DOM.div());readout=M.Observable{Any}(B.DOM.div())
    control_style=_inspection_control_style(style)
    input(name,value)=B.TextField(value;id="$dom_id-$name",disabled,style=control_style)
    button(name,text)=B.Button(text;id="$dom_id-$name",disabled,style=control_style)
    vertices=input("vertices","");cx=input("center-x","");cy=input("center-y","")
    width=input("width","0");length_control=input("length","1");level=input("level","1")
    occurrence=input("occurrence","0");bar=input("bar","0")
    apply=button("apply","Apply query");select=button("select","Select occurrence / bar")
    boundary=button("boundary","Toggle boundary barcode");witnesses=button("witnesses","Toggle matrices")
    reset=button("reset","Reset");close_button=button("close","Close inspector")
    closed=Ref(false);updating=Ref(false);observers=Any[];pointer_observers=Any[]
    session_token=Ref(0);rebuild_count=Ref(0)
    browser_client=Ref{Any}(nothing);browser_lock=ReentrantLock();app_ref=Ref(WeakRef(nothing))
    finite=VIZ.inspection_selection(session).mode===:finite
    function commit(f)
        closed[] && return false
        disabled[]=true
        try;f();error_text[]="";true
        catch err;err isa InterruptException && rethrow();error_text[]=sprint(showerror,err);false
        finally;disabled[]=closed[];end
    end
    function listen(f,observable)
        push!(observers,M.on(observable) do value
            updating[] || closed[] || f(value);nothing
        end)
    end
    integer(control)=begin
        value=tryparse(Int,strip(control.value[]))
        value===nothing && throw(ArgumentError("Enter an integer selection."));value
    end
    coordinate(control)=VIZ._parse_inspection_coordinate(strip(control.value[]))
    function nearby(point,ax)
        ax===nothing && return nothing
        candidates=filter(t->t.axis===ax,targets[])
        isempty(candidates) && return nothing
        pixel=M.project(ax.scene,M.Point2d(point))
        distances=[sum(abs2,M.project(ax.scene,M.Point2d(t.point))-pixel) for t in candidates]
        k=argmin(distances);distances[k]<=24^2 ? candidates[k] : nothing
    end
    function hover(point,ax=axis[])
        t=nearby(point,ax)
        hover_text[]=t===nothing ? (!finite && ax===axis[] ? "Approximate pointer coordinates: $(Tuple(point)); click to move the worm center." : "") : t.label
        t
    end
    function pick(point,ax=axis[])
        t=hover(point,ax)
        if t===nothing
            return !finite && ax===axis[] ? commit(()->VIZ.select_inspection!(session;center=Tuple(point))) : false
        end
        commit() do
            if t.kind===:vertex
                selected=copy(VIZ.inspection_selection(session).vertices)
                t.vertex in selected ? filter!(!=(t.vertex),selected) : push!(selected,t.vertex)
                VIZ.select_inspection!(session;vertices=selected)
            elseif t.kind===:occurrence
                VIZ.select_inspection!(session;occurrence=t.occurrence)
            elseif t.kind===:bar
                VIZ.select_inspection!(session;bar=t.bar)
            elseif t.kind===:level
                VIZ.select_inspection!(session;level=t.level)
            end
        end
    end
    function dispose(;close_session=false)
        closed[] && return nothing
        closed[]=true;disabled[]=true
        session_token[]==0 || VIZ._off_inspection!(session,session_token[]);session_token[]=0
        foreach(M.off,observers);empty!(observers)
        foreach(M.off,pointer_observers);empty!(pointer_observers)
        fig[]===nothing || empty!(fig[]);axis[]=nothing;empty!(targets[]);empty!(axes)
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
            old=fig[];visible=findall(p->get(p.metadata,:panel_style,nothing)!==:text_only,snapshot.panels)
            weights=snapshot.metadata.panel_row_weights[1:end-1]
            canvas_size=figure_size===nothing ? (1250,sum(weights)+50) : figure_size
            current=M.Figure(size=canvas_size,figure_padding=style.padding,
                backgroundcolor=VIZ._visual_color(style,VIZ._VisualRole(:background)))
            empty!(axes);empty!(targets[]);axis[]=nothing
            for i in visible
                panel=snapshot.panels[i];rows,cols=snapshot.metadata.panel_positions[i]
                grid=current[rows,cols]=M.GridLayout();ax=_HANDLERS.render_panel(current,grid,panel;style)
                if panel.kind in (:generalized_rank_domain,:generalized_rank_worm,:boundary_path,:boundary_barcode,:gril_layers)
                    push!(axes,ax);axis[]===nothing && (axis[]=ax)
                    append!(targets[],[(;t...,axis=ax) for t in get(panel.metadata,:pick_targets,NamedTuple[])])
                end
            end
            for (i,weight) in enumerate(weights);M.rowsize!(current.layout,i,M.Auto(false,weight));end
            _HANDLERS.style_figure(current,style);fig[]=current;rebuild_count[]+=1
            push!(pointer_observers,M.on(M.events(current.scene).mouseposition) do _
                closed[] && return nothing
                active=findfirst(ax->M.is_mouseinside(ax.scene),axes)
                active===nothing ? (hover_text[]="") : hover(M.mouseposition(axes[active].scene),axes[active]);nothing
            end)
            push!(pointer_observers,M.on(M.events(current.scene).mousebutton;priority=10) do event
                if !closed[] && event.button==M.Mouse.left && event.action==M.Mouse.press
                    active=findfirst(ax->M.is_mouseinside(ax.scene),axes)
                    active===nothing || return M.Consume(pick(M.mouseposition(axes[active].scene),axes[active]))
                end
                M.Consume(false)
            end)
            vertices.value[]=join(state.vertices,",");occurrence.value[]=string(state.occurrence);bar.value[]=string(state.bar)
            if !finite
                cx.value[]=VIZ._signed_label(state.center[1]);cy.value[]=VIZ._signed_label(state.center[2])
                width.value[]=VIZ._signed_label(state.width);length_control.value[]=string(state.length);level.value[]=string(state.level)
            end
            status_text[]="Exact generalized rank: $(snapshot.metadata.rank). "*(finite ?
                "Finite labels; matrices $(state.witnesses ? "shown" : "off"), boundary $(state.boundary ? "shown" : "off")." :
                "Closed worm, length $(state.length), width $(VIZ._signed_label(state.width)); displayed level $(state.level).")
            hover_text[]="";content[]=current;old===nothing || empty!(old)
            readout[]=_inspection_dom_readout(VIZ.VisualizationSpec(:region_readout;
                panels=filter(p->get(p.metadata,:panel_style,nothing)===:text_only,snapshot.panels));style)
        finally;updating[]=false;end
        nothing
    end
    listen(_->commit() do
        if finite
            pieces=split(vertices.value[],',');selected=Int[]
            for p in pieces
                v=tryparse(Int,strip(p));v===nothing && throw(ArgumentError("Enter comma-separated finite labels."));push!(selected,v)
            end
            VIZ.select_inspection!(session;vertices=selected)
        else
            VIZ.select_inspection!(session;center=(coordinate(cx),coordinate(cy)),width=coordinate(width),
                length=integer(length_control),level=integer(level))
        end
    end,apply.value)
    listen(_->commit(()->VIZ.select_inspection!(session;occurrence=integer(occurrence),bar=integer(bar))),select.value)
    listen(_->commit(()->VIZ.select_inspection!(session;boundary=!VIZ.inspection_selection(session).boundary)),boundary.value)
    listen(_->commit(()->VIZ.select_inspection!(session;witnesses=!VIZ.inspection_selection(session).witnesses)),witnesses.value)
    listen(_->commit(()->VIZ.reset_inspection!(session)),reset.value)
    listen(_->commit(()->VIZ.close_inspection!(session)),close_button.value)
    session_token[]=VIZ._on_inspection(session,_ -> refresh());refresh()
    label(text,control)=B.DOM.label(B.DOM.span(text;style="display:block;font-weight:600"),control;style="display:block;min-width:0")
    row(children...)=B.DOM.div(children...;style="display:flex;flex-wrap:wrap;align-items:end;gap:$(style.gap)px;margin:$(style.gap)px 0")
    query_controls=finite ? row(label("Finite labels (comma separated)",vertices),apply,boundary,witnesses) :
        row(label("Center x (exact entry)",cx),label("Center y (exact entry)",cy),label("Width",width),
            label("Worm length",length_control),label("Rank level",level),apply)
    selection_controls=finite ? row(label("Occurrence (0 clears)",occurrence),label("Bar (0 clears)",bar),select) : B.DOM.div()
    foreground=_inspection_css_color(style,VIZ._VisualRole(:foreground));background=_inspection_css_color(style,VIZ._VisualRole(:background))
    dom=B.DOM.div(B.DOM.h2("Explore generalized rank"),
        B.DOM.p(finite ? "Select a connected convex set of finite labels. Click a node to toggle it; invalid selections keep the last valid region. Boundary bars use distinct path occurrences." :
            "Move a continuous closed worm and compare its rank with exact GRIL widths. Click the parameter plane for an approximate center, or enter exact coordinates. A layer width is a supremum and may not be attained."),
        B.DOM.p("Live queries require Julia. Hover and retained bar/layer selection do not recompute ranks."),
        query_controls,selection_controls,row(reset,close_button),
        B.DOM.p(error_text;id="$dom_id-error",role="alert"),B.DOM.p(status_text;id="$dom_id-status",role="status"),
        B.DOM.p(hover_text;id="$dom_id-hover",role="status",style="min-height:1.5em"),
        B.DOM.div(content;style="max-width:100%;overflow:auto",tabindex="0",var"aria-label"="Generalized rank charts"),
        B.DOM.div(readout;id="$dom_id-readout");id=dom_id,
        style="box-sizing:border-box;font-family:'$(style.font)',sans-serif;font-size:$(style.fontsize)px;line-height:1.45;color:$foreground;background:$background;padding:$(style.padding)px;max-width:100%;min-width:0;overflow-wrap:anywhere")
    app=B.App(;title="TamerOp generalized rank inspector") do client
        lock(browser_lock) do
            closed[] && return B.DOM.p("This inspector has been closed.")
            if browser_client[]!==nothing && browser_client[]!==client
                existing=app_ref[].value;existing===nothing || (existing.session[]=browser_client[])
                return B.DOM.p("This widget already has a browser client. Call visualize(session) for an independent linked view.")
            end
            if browser_client[]===nothing
                browser_client[]=client;_inspection_keyboard_navigation!(client,dom)
                push!(observers,M.on(client.on_close) do ended;ended && dispose();nothing;end)
            end
            dom
        end
    end
    app_ref[]=WeakRef(app)
    _INSPECTION_UI_HANDLES[app]=(;session,figure=fig,axis,axes,targets,
        controls=(;vertices,cx,cy,width,length=length_control,level,occurrence,bar,apply,select,boundary,witnesses,reset,close=close_button),
        callbacks=(;pick,hover,refresh),dispose,closed,disabled,observers,pointer_observers,session_token,browser_client,
        rebuild_count,error_text,status_text,hover_text,readout,dom_id,style,figure_size)
    app
end
