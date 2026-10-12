# One shared query selection; callbacks consume the core's prepared snapshots.
function _render_collection_inspector(spec;display=:inline,figure=nothing,size=nothing,style=VIZ.VisualStyle())
    VIZ._check_visual_render_options(;display,figure,size,style)
    figure===nothing || throw(ArgumentError("A collection inspector owns its canvas."))
    session=spec.metadata.session
    VIZ.inspection_summary(session).closed && throw(ArgumentError("This collection session is closed."))
    B,M=WGLMakie.Bonito,WGLMakie.Makie
    _NEXT_INSPECTION_UI_ID[]+=1
    dom_id="tamerop-collection-inspector-$(_NEXT_INSPECTION_UI_ID[])"
    figure_size=size
    fig=Ref{Any}(nothing);axis=Ref{Any}(nothing);axes=Any[];targets=Ref(NamedTuple[])
    disabled=M.Observable(false)
    error_text,status_text,hover_text=M.Observable(""),M.Observable(""),M.Observable("")
    content,readout=M.Observable{Any}(B.DOM.div()),M.Observable{Any}(B.DOM.div())
    control_style=_inspection_control_style(style)
    sample=B.TextField("1";id="$dom_id-sample",disabled,style=control_style)
    target=B.TextField("2";id="$dom_id-target",disabled,style=control_style)
    feature=B.TextField("1";id="$dom_id-feature",disabled,style=control_style)
    apply=B.Button("Apply selection";id="$dom_id-apply",disabled,style=control_style)
    views=Dict(k=>B.Button(uppercasefirst(String(k));id="$dom_id-$k",disabled,style=control_style)
        for k in VIZ.describe(session.object).tables)
    matching=B.Button("Open supplied matching";id="$dom_id-matching",disabled,style=control_style)
    pair_content=M.Observable{Any}(B.DOM.div());pair_view=Ref{Any}(nothing);pair_session=Ref{Any}(nothing)
    function close_pair()
        if pair_view[]!==nothing
            ui=get(_INSPECTION_UI_HANDLES,pair_view[],nothing)
            ui===nothing || ui.dispose()
            pair_view[]=nothing
        end
        pair_session[]===nothing || VIZ.close_inspection!(pair_session[])
        pair_session[]=nothing;pair_content[]=B.DOM.div()
    end
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
        n===nothing && throw(ArgumentError("Enter an integer sample or feature index."))
        n
    end
    function nearby(point,ax=axis[])
        ax===nothing && return nothing
        candidates=filter(t->t.axis===ax,targets[])
        isempty(candidates) && return nothing
        if first(candidates).kind===:collection_matrix
            i,j=round(Int,point[2]),round(Int,point[1])
            k=findfirst(t->t.point==(Float64(j),Float64(i)),candidates)
            return k===nothing ? nothing : candidates[k]
        end
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
        commit(()->VIZ.select_inspection!(session;sample=t.sample,target=t.target,feature=t.feature))
    end
    function dispose(;close_session=false)
        closed[] && return nothing
        closed[]=true;disabled[]=true
        close_pair()
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
        close_pair()
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
                if panel.kind in (:collection_matrix,:collection_embedding)
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
            sample.value[]=string(state.sample);target.value[]=string(state.target)
            feature.value[]=state.feature===nothing ? "" : string(state.feature)
            selected=snapshot.metadata.selected
            status_text[]="$(state.view): $(selected.sample) / $(state.view===:features ? selected.feature : selected.target). Original value: $(selected.value)."
            hover_text[]="";content[]=current
            old===nothing || empty!(old)
            readout[]=_inspection_dom_readout(VIZ.VisualizationSpec(:comparison_readout;
                panels=filter(p->get(p.metadata,:panel_style,nothing)===:text_only,snapshot.panels));style)
        finally;updating[]=false;end
        nothing
    end
    listen(_->commit(()->VIZ.select_inspection!(session;sample=integer(sample),target=integer(target),
        feature=isempty(strip(feature.value[])) ? nothing : integer(feature))),apply.value)
    for (key,button) in views
        listen(_->commit(()->VIZ.select_inspection!(session;view=key)),button.value)
    end
    listen(_->commit() do
        child=VIZ.collection_matching(session)
        close_pair();pair_session[]=child
        try
            viewer=VIZ.visualize(child;backend=:wglmakie,style)
            pair_view[]=viewer;pair_content[]=viewer
        catch
            close_pair();rethrow()
        end
    end,matching.value)
    listen(_->commit(()->VIZ.reset_inspection!(session)),reset.value)
    listen(_->commit(()->VIZ.close_inspection!(session)),close_button.value)
    session_token[]=VIZ._on_inspection(session,_ -> refresh())
    refresh()
    label(text,control)=B.DOM.label(B.DOM.span(text;style="display:block;font-weight:600"),control;style="display:block;min-width:0")
    row(children...)=B.DOM.div(children...;style="display:flex;flex-wrap:wrap;align-items:end;gap:$(style.gap)px;margin:$(style.gap)px 0")
    foreground=_inspection_css_color(style,VIZ._VisualRole(:foreground))
    background=_inspection_css_color(style,VIZ._VisualRole(:background))
    dom=B.DOM.div(B.DOM.h2("Explore a collection"),
        B.DOM.p("Select a table cell or an embedded sample. Hover reads retained values. Sample panels show supplied invariants and source data. An embedding is not evidence of preserved distances. Live Julia is required."),
        row((views[k] for k in VIZ.describe(session.object).tables)...),
        row(label("Sample index",sample),label("Target sample index",target),label("Feature index",feature),apply,reset,close_button),
        row(matching,B.DOM.span("Uses a supplied witness for the ordered pair; its distance semantics may differ from the table.")),
        B.DOM.p(error_text;id="$dom_id-error",role="alert"),
        B.DOM.p(status_text;id="$dom_id-status",role="status"),B.DOM.p(hover_text;id="$dom_id-hover",role="status",style="min-height:1.5em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis"),
        B.DOM.div(content;style="max-width:100%;overflow:auto",tabindex="0",var"aria-label"="Collection charts"),
        B.DOM.div(readout;id="$dom_id-readout"),B.DOM.div(pair_content;id="$dom_id-pair");id=dom_id,
        style="box-sizing:border-box;font-family:'$(style.font)',sans-serif;font-size:$(style.fontsize)px;line-height:1.45;color:$foreground;background:$background;padding:$(style.padding)px;max-width:100%;min-width:0;overflow-wrap:anywhere")
    app=B.App(;title="TamerOp collection inspector") do client
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
        controls=(;sample,target,feature,apply,views,matching,reset,close=close_button),
        callbacks=(;pick,hover,refresh),dispose,closed,disabled,observers,pointer_observers,session_token,browser_client,
        rebuild_count,error_text,status_text,hover_text,readout,dom_id,style,figure_size,pair_session)
    app
end
