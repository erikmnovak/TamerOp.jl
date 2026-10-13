# WGL controls for retained, checked summands. Browser input never supplies a certificate.
function _render_summand_inspector(spec;display=:inline,figure=nothing,size=nothing,style=VIZ.VisualStyle())
    VIZ._check_visual_render_options(;display,figure,size,style)
    figure===nothing || throw(ArgumentError("A summand inspector owns its canvas."))
    session=spec.metadata.session
    VIZ.inspection_summary(session).closed && throw(ArgumentError("This summand session is closed."))
    B,M=WGLMakie.Bonito,WGLMakie.Makie
    _NEXT_INSPECTION_UI_ID[]+=1;dom_id="tamerop-summand-inspector-$(_NEXT_INSPECTION_UI_ID[])"
    fig=Ref{Any}(nothing);axis=Ref{Any}(nothing);axes=Any[];targets=Ref(NamedTuple[])
    disabled=M.Observable(false);error_text=M.Observable("");status_text=M.Observable("");hover_text=M.Observable("")
    content=M.Observable{Any}(B.DOM.div());readout=M.Observable{Any}(B.DOM.div())
    control_style=_inspection_control_style(style)
    input(name,value)=B.TextField(value;id="$dom_id-$name",disabled,style=control_style)
    button(name,text)=B.Button(text;id="$dom_id-$name",disabled,style=control_style)
    selected=input("selected","");focus=input("focus","1");vertex=input("vertex","1")
    layout=input("layout","facets");ranking=input("ranking","index");type_filter=input("type","all")
    minimum_=input("minimum","0");limit=input("limit","8");pair=input("pair","")
    hom_pair=input("hom-pair","");basis=input("basis","1")
    apply=button("apply","Apply selection");previous=button("previous","Previous block");next=button("next","Next block")
    hom=button("hom","Toggle Hom table");scale=button("scale","Toggle linear / log colors")
    shared=button("shared","Toggle shared / independent scales")
    reset=button("reset","Reset");close_button=button("close","Close inspector")
    closed=Ref(false);updating=Ref(false);observers=Any[];pointer_observers=Any[]
    session_token=Ref(0);rebuild_count=Ref(0)
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
            updating[] || closed[] || f(value);nothing
        end)
    end
    integer(control)=begin
        value=tryparse(Int,strip(control.value[]));value===nothing && throw(ArgumentError("Enter integer IDs and counts."));value
    end
    function integers(control)
        isempty(strip(control.value[])) && return Int[]
        [begin v=tryparse(Int,strip(p));v===nothing && throw(ArgumentError("Enter comma-separated integer IDs."));v end for p in split(control.value[],',')]
    end
    function nearby(point,ax)
        ax===nothing && return nothing
        candidates=filter(t->t.axis===ax,targets[]);isempty(candidates) && return nothing
        pixel=M.project(ax.scene,M.Point2d(point))
        distances=[sum(abs2,M.project(ax.scene,M.Point2d(t.point))-pixel) for t in candidates]
        k=argmin(distances);distances[k]<=26^2 ? candidates[k] : nothing
    end
    function hover(point,ax=axis[])
        t=nearby(point,ax);hover_text[]=t===nothing ? "" : t.label;t
    end
    function pick(point,ax=axis[])
        t=hover(point,ax);t===nothing && return false
        commit() do
            VIZ.select_inspection!(session;vertex=t.vertex,
                focus=get(t,:block,VIZ.inspection_selection(session).focus))
        end
    end
    function dispose(;close_session=false)
        closed[] && return nothing
        closed[]=true;disabled[]=true
        session_token[]==0 || VIZ._off_inspection!(session,session_token[]);session_token[]=0
        foreach(M.off,observers);empty!(observers);foreach(M.off,pointer_observers);empty!(pointer_observers)
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
            # The final readout has its own accessible DOM. Matrix/text-only
            # content remains in the static export as well.
            weights=snapshot.metadata.panel_row_weights[1:end-1]
            canvas_size=size===nothing ? (1450,sum(weights)+60) : size
            current=M.Figure(size=canvas_size,figure_padding=style.padding,backgroundcolor=VIZ._visual_color(style,VIZ._VisualRole(:background)))
            empty!(axes);empty!(targets[]);axis[]=nothing
            for i in visible
                panel=snapshot.panels[i];rows,cols=snapshot.metadata.panel_positions[i]
                grid=current[rows,cols]=M.GridLayout();ax=_HANDLERS.render_panel(current,grid,panel;style)
                if !isempty(get(panel.metadata,:pick_targets,()))
                    push!(axes,ax);axis[]===nothing && (axis[]=ax)
                    append!(targets[],[(;t...,axis=ax) for t in panel.metadata.pick_targets])
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
            selected.value[]=state.selected===nothing ? "all" : join(state.selected,",")
            focus.value[]=string(state.focus);vertex.value[]=string(state.vertex);layout.value[]=string(state.layout)
            ranking.value[]=string(state.sort_by);type_filter.value[]=string(state.type_filter)
            minimum_.value[]=string(state.min_value);limit.value[]=string(state.limit)
            pair.value[]=state.pair===nothing ? "" : join(state.pair,",")
            hom_pair.value[]=state.hom_pair===nothing ? "" : join(state.hom_pair,",");basis.value[]=string(state.basis_index)
            status_text[]="Displayed blocks: $(join(state.ids,", ")); focus $(state.focus), vertex $(state.vertex). Splitting checked; geometry $(snapshot.metadata.geometry_status). $(state.scale) colors; $(state.shared_scale ? "shared" : "independent") scales. Hom $(state.hom ? "on" : "off")."
            hover_text[]="";content[]=current;old===nothing || empty!(old)
            readout[]=_inspection_dom_readout(VIZ.VisualizationSpec(:summand_readout;panels=filter(p->get(p.metadata,:panel_style,nothing)===:text_only,snapshot.panels));style)
        finally;updating[]=false;end
        nothing
    end
    listen(_->commit() do
        sel=strip(selected.value[])=="all" ? nothing : integers(selected)
        pq=integers(pair);hp=integers(hom_pair)
        state=VIZ.inspection_selection(session)
        rank=Symbol(strip(ranking.value[]));kind=Symbol(strip(type_filter.value[]))
        lower=integer(minimum_);count_=integer(limit);chosen=integer(focus)
        filters_changed=sel!=state.selected || rank!=state.sort_by || kind!=state.type_filter ||
            lower!=state.min_value || count_!=state.limit
        chosen_pair=isempty(hp) ? nothing : Tuple(hp)
        # An unchanged detail field follows a newly filtered gallery. Explicit
        # edits still receive the ordinary strict validation and rollback.
        filters_changed && chosen_pair==state.hom_pair && (chosen_pair=nothing)
        VIZ.select_inspection!(session;selected=sel,
            focus=filters_changed && chosen==state.focus ? nothing : chosen,vertex=integer(vertex),
            layout=Symbol(strip(layout.value[])),sort_by=rank,type_filter=kind,
            min_value=lower,limit=count_,pair=isempty(pq) ? nothing : Tuple(pq),
            hom_pair=chosen_pair,basis_index=integer(basis))
    end,apply.value)
    function step(delta)
        state=VIZ.inspection_selection(session);isempty(state.ids) && return
        k=findfirst(==(state.focus),state.ids);k=k===nothing ? 1 : mod1(k+delta,length(state.ids))
        VIZ.select_inspection!(session;focus=state.ids[k])
    end
    listen(_->commit(()->step(-1)),previous.value);listen(_->commit(()->step(1)),next.value)
    listen(_->commit(()->VIZ.select_inspection!(session;hom=!VIZ.inspection_selection(session).hom)),hom.value)
    listen(_->commit(()->VIZ.select_inspection!(session;scale=VIZ.inspection_selection(session).scale===:linear ? :log : :linear)),scale.value)
    listen(_->commit(()->VIZ.select_inspection!(session;shared_scale=!VIZ.inspection_selection(session).shared_scale)),shared.value)
    listen(_->commit(()->VIZ.reset_inspection!(session)),reset.value)
    listen(_->commit(()->VIZ.close_inspection!(session)),close_button.value)
    session_token[]=VIZ._on_inspection(session,_ -> refresh());refresh()
    label(text,control)=B.DOM.label(B.DOM.span(text;style="display:block;font-weight:600"),control;style="display:block;min-width:0")
    row(children...)=B.DOM.div(children...;style="display:flex;flex-wrap:wrap;align-items:end;gap:$(style.gap)px;margin:$(style.gap)px 0")
    foreground=_inspection_css_color(style,VIZ._VisualRole(:foreground));background=_inspection_css_color(style,VIZ._VisualRole(:background))
    dom=B.DOM.div(B.DOM.h2("Explore module summands"),
        B.DOM.p("The full module is the selected blocks plus their complement. Blocks retain their local IDs while ranking and filtering change their display order. A block may contain several copies of an interval."),
        B.DOM.p("Live controls require Julia. Hover reads retained values; selecting a block does not recompute a decomposition."),
        row(label("Block IDs (all or comma separated; blank clears)",selected),label("Focus ID (0 clears)",focus),label("Finite vertex",vertex),previous,next),
        row(label("Layout: facets, overlay, outline, isolate",layout),label("Rank by: index, multiplicity, support_vertices, total_dimension, max_dimension, generators, relations",ranking),label("Type: all, interval, thin, thick",type_filter)),
        row(label("Minimum rank statistic",minimum_),label("Display limit",limit),label("Structure map source,target (blank clears)",pair)),
        row(label("Hom source,target (blank clears)",hom_pair),label("Hom basis index",basis),apply,hom),
        row(scale,shared,reset,close_button),
        B.DOM.p(error_text;id="$dom_id-error",role="alert"),B.DOM.p(status_text;id="$dom_id-status",role="status"),
        B.DOM.p(hover_text;id="$dom_id-hover",role="status",style="min-height:1.5em"),
        B.DOM.div(content;style="max-width:100%;overflow:auto",tabindex="0",var"aria-label"="Summand charts"),
        B.DOM.div(readout;id="$dom_id-readout");id=dom_id,
        style="box-sizing:border-box;font-family:'$(style.font)',sans-serif;font-size:$(style.fontsize)px;line-height:1.45;color:$foreground;background:$background;padding:$(style.padding)px;max-width:100%;min-width:0;overflow-wrap:anywhere")
    app=B.App(;title="TamerOp summand inspector") do client
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
        controls=(;selected,focus,vertex,layout,ranking,type_filter,minimum=minimum_,limit,pair,hom_pair,basis,apply,previous,next,hom,scale,shared,reset,close=close_button),
        callbacks=(;pick,hover,refresh),dispose,closed,disabled,observers,pointer_observers,session_token,browser_client,
        rebuild_count,error_text,status_text,hover_text,readout,dom_id,style,figure_size=size)
    app
end
