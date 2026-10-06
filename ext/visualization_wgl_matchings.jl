# Bonito controls and picking for retained matching records. All mathematical
# changes go through the same session methods used by notebooks and scripts.
function _render_matching_inspector(spec;display=:inline,figure=nothing,size=nothing,style=VIZ.VisualStyle())
    VIZ._check_visual_render_options(;display,figure,size,style)
    session=spec.metadata.session
    VIZ.inspection_summary(session).closed && throw(ArgumentError("This matching session is closed."))
    B,M=WGLMakie.Bonito,WGLMakie.Makie
    _NEXT_INSPECTION_UI_ID[]+=1
    dom_id="tamerop-matching-inspector-$(_NEXT_INSPECTION_UI_ID[])"
    figure_size=size === nothing ? (1200,session.inputs === nothing ? 900 : 1400) : size
    figure === nothing || throw(ArgumentError("A live matching inspector owns its figure. Render inspection_snapshot(session) to compose a static view into a supplied figure."))
    fig=Ref{Any}(nothing)
    disabled=M.Observable(false)
    error_text,status_text,hover_text=M.Observable(""),M.Observable(""),M.Observable("")
    content,readout=M.Observable{Any}(B.DOM.div()),M.Observable{Any}(B.DOM.div())
    control_style=_inspection_control_style(style)
    pair=B.TextField("";id="$dom_id-pair",disabled,style=control_style)
    apply_pair=B.Button("Select pair";id="$dom_id-apply-pair",disabled,style=control_style)
    next=B.Button("Next pair";id="$dom_id-next-pair",disabled,style=control_style)
    previous=B.Button("Previous pair";id="$dom_id-previous-pair",disabled,style=control_style)
    sample=B.TextField("1";id="$dom_id-sample",disabled,style=control_style)
    apply_sample=B.Button("Select sample";id="$dom_id-apply-sample",disabled,style=control_style)
    best_sample=B.Button("Best sample";id="$dom_id-best-sample",disabled,style=control_style)
    optimum=B.Button("Compute exact window optimum";id="$dom_id-optimum",disabled,style=control_style)
    fields=ntuple(i->B.TextField(i <= 2 ? "0" : "1";id="$dom_id-line-$i",disabled,style=control_style),4)
    apply_slice=B.Button("Compare this slice";id="$dom_id-apply-slice",disabled,style=control_style)
    reset=B.Button("Reset selection";id="$dom_id-reset",disabled,style=control_style)
    close_button=B.Button("Close inspector";id="$dom_id-close",disabled,style=control_style)
    closed,updating=Ref(false),Ref(false)
    axes=Dict{Symbol,Any}();targets=Dict{Symbol,Any}()
    observers=Any[];pointer_observers=Any[]
    rebuild_count,session_token=Ref(0),Ref(0)
    browser_client=Ref{Any}(nothing);browser_lock=ReentrantLock();app_ref=Ref(WeakRef(nothing))
    function commit(f)
        closed[] && return false
        disabled[]=true
        try
            f();error_text[]="";return true
        catch err
            err isa InterruptException && rethrow()
            error_text[]=sprint(showerror,err);return false
        finally
            disabled[]=closed[]
        end
    end
    function listen(f,observable)
        push!(observers,M.on(observable) do value
            updating[] || closed[] || f(value)
            nothing
        end)
    end
    function parse_id(control,name)
        id=tryparse(Int,strip(control.value[]))
        id === nothing && throw(ArgumentError("Enter an integer $name."))
        return id
    end
    function nearby(kind,point)
        haskey(axes,kind) && haskey(targets,kind) || return Int[]
        ax=axes[kind]
        spans,pixels=M.widths(ax.finallimits[]),M.widths(ax.scene.viewport[])
        all(>(0),spans) && all(>(0),pixels) || return Int[]
        scale=pixels ./ spans
        distances=map(targets[kind]) do target
            p=if haskey(target,:segment)
                a=target.segment
                (clamp(point[1],min(a[1],a[3]),max(a[1],a[3])),a[2])
            else;target.point;end
            ((p[1]-point[1])*scale[1])^2+((p[2]-point[2])*scale[2])^2
        end
        isempty(distances) && return Int[]
        best=minimum(distances)
        best <= 20^2 || return Int[]
        return unique(Int[targets[kind][i].id for i in eachindex(distances) if abs(distances[i]-best)<=1e-6])
    end
    function hover(kind,point)
        ids=nearby(kind,point)
        records=VIZ.inspection_snapshot(session).metadata.matching_records
        hover_text[]=isempty(ids) ? "" : kind === :matching_sample_map ?
            join(("Sample $id; weighted cost $(session.initial.context.samples[id].weighted_distance)" for id in ids),"\n") :
            join((VIZ._matching_record_label(records[id]) for id in ids),"\n")
        return ids
    end
    function pick(kind,point)
        ids=nearby(kind,point);isempty(ids) && return false
        current=VIZ.inspection_selection(session)
        active=kind === :matching_sample_map ? current.sample : current.pair
        position=findfirst(==(active),ids)
        id=position === nothing ? first(ids) : ids[mod1(position+1,length(ids))]
        return commit(() -> kind === :matching_sample_map ?
            VIZ.select_inspection!(session;sample=id) : VIZ.select_inspection!(session;pair=id))
    end
    function dispose(;close_session=false)
        closed[] && return nothing
        closed[]=true;disabled[]=true
        status_text[]="Inspector closed. Its last static snapshot remains available."
        session_token[] == 0 || VIZ._off_inspection!(session,session_token[])
        session_token[]=0
        foreach(M.off,observers);empty!(observers)
        foreach(M.off,pointer_observers);empty!(pointer_observers)
        content[]=B.DOM.p("Inspector closed.")
        empty!(axes);empty!(targets)
        fig[] === nothing || empty!(fig[])
        lock(browser_lock) do;browser_client[]=nothing;end
        app=app_ref[].value;app === nothing || delete!(_INSPECTION_UI_HANDLES,app)
        close_session && VIZ.close_inspection!(session)
        return nothing
    end
    function refresh()
        VIZ.inspection_summary(session).closed && return dispose()
        snapshot=VIZ.inspection_snapshot(session)
        updating[]=true
        try
            foreach(M.off,pointer_observers);empty!(pointer_observers)
            empty!(axes);empty!(targets)
            # Build offscreen, then replace the DOM once. Mutating a displayed
            # axis inserts many WGL scenes synchronously during a browser event.
            # A fresh figure avoids those round trips and releases the old scene.
            old_figure=fig[]
            current=M.Figure(size=figure_size,figure_padding=style.padding,
                backgroundcolor=VIZ._visual_color(style,VIZ._VisualRole(:background)))
            panels=filter(p->get(p.metadata,:panel_style,nothing)!==:text_only,snapshot.panels)
            for (i,p) in enumerate(panels)
                grid=current[cld(i,2),mod1(i,2)]=M.GridLayout()
                ax=_HANDLERS.render_panel(current,grid,p;style)
                axes[p.kind]=ax
                targets[p.kind]=get(p.metadata,:pick_targets,NamedTuple[])
            end
            _HANDLERS.style_figure(current,style)
            fig[]=current
            attach_pointer(current)
            rebuild_count[]+=1
            pair.value[]=session.state.pair === nothing ? "" : string(session.state.pair)
            sample.value[]=session.state.sample === nothing ? "1" : string(session.state.sample)
            q=session.payload.context.query
            if q !== nothing
                for (f,x) in zip(fields,(q.basepoint...,q.direction...))
                    f.value[]=string(x)
                end
            end
            status_text[]="$(VIZ._matching_scope_label(session.payload.context.scope)): bottleneck cost $(VIZ._interval_value(session.payload.matching.distance)). " *
                "$(length(snapshot.metadata.displayed_pair_ids)) / $(length(snapshot.metadata.matching_records)) pairs displayed."
            content[]=current
            old_figure === nothing || empty!(old_figure)
            readout[]=_inspection_dom_readout(VIZ.VisualizationSpec(:matching_readout;
                panels=filter(p->get(p.metadata,:panel_style,nothing)===:text_only,snapshot.panels));style)
        finally;updating[]=false;end
        return nothing
    end
    listen(_->commit(()->VIZ.select_inspection!(session;pair=parse_id(pair,"pair ID"))),apply_pair.value)
    for (button,delta) in ((next,1),(previous,-1))
        listen(button.value) do _
            n=length(VIZ.inspection_snapshot(session).metadata.matching_records)
            n==0 || commit(()->VIZ.select_inspection!(session;pair=mod1(something(session.state.pair,delta>0 ? 0 : 1)+delta,n)))
        end
    end
    listen(_->commit(()->VIZ.select_inspection!(session;sample=parse_id(sample,"sample ID"))),apply_sample.value)
    listen(best_sample.value) do _
        commit() do
            isempty(session.samples) && throw(ArgumentError("This session has no sample family."))
            id=argmax([s.weighted_distance for s in session.initial.context.samples])
            VIZ.select_inspection!(session;sample=id)
        end
    end
    listen(_->commit(()->VIZ.select_inspection!(session;optimum=true)),optimum.value)
    listen(apply_slice.value) do _
        commit() do
            values=map(f->VIZ._parse_inspection_coordinate(f.value[]),fields)
            VIZ.select_inspection!(session;slice=(;basepoint=values[1:2],direction=values[3:4]))
        end
    end
    listen(_->commit(()->VIZ.reset_inspection!(session)),reset.value)
    listen(_->VIZ.close_inspection!(session),close_button.value)
    function attach_pointer(current)
        push!(pointer_observers,M.on(M.events(current.scene).mousebutton;priority=100) do event
            closed[] && return M.Consume(false)
            if event.button==M.Mouse.left && event.action==M.Mouse.press
                for (kind,axis) in axes
                    M.is_mouseinside(axis.scene) || continue
                    return M.Consume(pick(kind,M.mouseposition(axis.scene)))
                end
            end
            M.Consume(false)
        end)
        push!(pointer_observers,M.on(M.events(current.scene).mouseposition) do _
            closed[] && return nothing
            for (kind,axis) in axes
                M.is_mouseinside(axis.scene) || continue
                hover(kind,M.mouseposition(axis.scene));return nothing
            end
            hover_text[]="";nothing
        end)
        return nothing
    end
    session_token[]=VIZ._on_inspection(session,_ -> refresh())
    refresh()
    label(text,control)=B.DOM.label(B.DOM.span(text;style="display:block;font-weight:600"),control;style="display:block;min-width:0")
    row(children...)=B.DOM.div(children...;style="display:flex;flex-wrap:wrap;align-items:end;gap:$(style.gap)px;margin:$(style.gap)px 0")
    slice_controls=session.inputs === nothing ? B.DOM.div() : B.DOM.details(
        B.DOM.summary("Compare slices and search the window"),
        B.DOM.p("The map contains $(length(session.samples)) discrete samples. Exact optimization is explicit and budgeted. Both barcodes are clipped to the same window."),
        row(label("Sample ID",sample),apply_sample,best_sample),
        B.DOM.p("Use integers, fractions or decimals for a new slice."),
        row(label("Basepoint x",fields[1]),label("Basepoint y",fields[2]),label("Direction x",fields[3]),label("Direction y",fields[4]),apply_slice),
        row(optimum))
    foreground=_inspection_css_color(style,VIZ._VisualRole(:foreground))
    background=_inspection_css_color(style,VIZ._VisualRole(:background))
    error_color=_inspection_css_color(style,VIZ._VisualRole(:error))
    dom=B.DOM.div(B.DOM.h2("Distance and matching witness"),
        B.DOM.p("Select a pair in either chart, or enter its ID. Repeated clicks cycle coincident members. The assignment is one optimal choice; uniqueness is not asserted."),
        row(label("Pair ID (0 clears)",pair),apply_pair,previous,next,reset,close_button),slice_controls,
        B.DOM.p(error_text;id="$dom_id-error",role="alert",style="color:$error_color;white-space:pre-line"),
        B.DOM.p(status_text;id="$dom_id-status",role="status"),
        B.DOM.p(hover_text;id="$dom_id-hover",role="status",style="white-space:pre-line"),
        B.DOM.div(content;style="max-width:100%;overflow:auto",tabindex="0",var"aria-label"="Linked matching charts"),
        B.DOM.div(readout;id="$dom_id-readout");id=dom_id,
        style="box-sizing:border-box;font-family:'$(style.font)',sans-serif;font-size:$(style.fontsize)px;line-height:1.45;color:$foreground;background:$background;padding:$(style.padding)px;max-width:100%;min-width:0;overflow-wrap:anywhere")
    app=B.App(;title="TamerOp matching inspector") do client
        lock(browser_lock) do
            closed[] && return B.DOM.p("This inspector has been closed.")
            if browser_client[] !== nothing && browser_client[] !== client
                existing=app_ref[].value;existing === nothing || (existing.session[]=browser_client[])
                return B.DOM.p("This widget already has a browser client. Call visualize(session) again to open an independent linked view.")
            end
            if browser_client[] === nothing
                browser_client[]=client
                push!(observers,M.on(client.on_close) do ended
                    ended && dispose();nothing
                end)
            end
            dom
        end
    end
    app_ref[]=WeakRef(app)
    _INSPECTION_UI_HANDLES[app]=(;session,figure=fig,axes,targets,controls=(;pair,apply_pair,next,previous,sample,apply_sample,best_sample,optimum,fields,apply_slice,reset,close=close_button),
        callbacks=(;pick,hover,refresh),dispose,closed,disabled,observers,pointer_observers,session_token,browser_client,
        rebuild_count,error_text,status_text,hover_text,readout,dom_id,style,figure_size)
    return app
end
