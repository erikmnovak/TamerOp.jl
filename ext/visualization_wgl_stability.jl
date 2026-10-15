# Exact HN selections use bounded queries; hover reads retained targets only.
function _render_stability_inspector(spec;display=:inline,figure=nothing,size=nothing,style=VIZ.VisualStyle())
    VIZ._check_visual_render_options(;display,figure,size,style)
    figure===nothing || throw(ArgumentError("The stability inspector owns its canvas."))
    s=spec.metadata.session;VIZ._inspection_require_open(s)
    B,M=WGLMakie.Bonito,WGLMakie.Makie
    _NEXT_INSPECTION_UI_ID[]+=1;dom_id="tamerop-stability-$(_NEXT_INSPECTION_UI_ID[])"
    fig=Ref{Any}(nothing);axis=Ref{Any}(nothing);chart_axes=Ref(Any[]);targets=Ref(NamedTuple[])
    closed=Ref(false);updating=Ref(false);closing=Ref(false);close_task=Ref{Union{Nothing,Task}}(nothing);disabled=M.Observable(false)
    error_text=M.Observable("");status_text=M.Observable("");hover_text=M.Observable("")
    content=M.Observable{Any}(B.DOM.div());readout=M.Observable{Any}(B.DOM.div())
    observers=Any[];pointer_observers=Any[];session_token=Ref(0);rebuild_count=Ref(0)
    retained=Ref{Any}(nothing);update_count=Ref(0)
    browser_client=Ref{Any}(nothing);app_ref=Ref(WeakRef(nothing));browser_lock=ReentrantLock();owner=Ref(nothing)
    cs=_inspection_control_style(style)
    object=s.object;H=TO.Stability
    chambers=object isa H.SkyscraperChambers;landscape=object isa H.FilteredLandscapeResult
    views=VIZ._stability_views(object)
    view=B.Dropdown(collect(views);option_to_string=string,id="$dom_id-view",disabled,style=cs)
    point_x=B.TextField("0";id="$dom_id-x",disabled,style=cs)
    point_y=B.TextField("0";id="$dom_id-y",disabled,style=cs)
    threshold=B.TextField("0";id="$dom_id-threshold",disabled,style=cs)
    baseline=B.TextField("0";id="$dom_id-baseline",disabled,style=cs)
    layer=B.TextField("1";id="$dom_id-layer",disabled,style=cs)
    factor=B.TextField("0";id="$dom_id-factor",disabled,style=cs)
    vertex=B.TextField("1";id="$dom_id-vertex",disabled,style=cs)
    apply=B.Button("Inspect selection";id="$dom_id-apply",disabled,style=cs)
    reset=B.Button("Reset";id="$dom_id-reset",disabled,style=cs)
    close_button=B.Button("Close inspector";id="$dom_id-close",disabled,style=cs)
    function listen(f,o)
        push!(observers,M.on(o) do v
            closed[] || updating[] || f(v);nothing
        end)
    end
    function commit(f)
        closed[] && return false
        try;f();error_text[]="";true
        catch err;err isa InterruptException && rethrow();error_text[]=sprint(showerror,err);false;end
    end
    function status(stage,error)
        closed[] && return
        stage===:error && (error_text[]=string(error))
        status_text[]=stage===:pending ? "Preparing the requested selection; the previous result remains visible." :
            stage===:error ? "Selection unchanged." : "Stability selection committed."
    end
    request(;kwargs...)=commit(()->VIZ._request_inspection!(s,owner;status,kwargs...))
    function nearby(point,index)
        index<=length(chart_axes[]) || return nothing
        ax=chart_axes[][index];spans=M.widths(ax.finallimits[]);pixels=M.widths(ax.scene.viewport[])
        all(>(0),spans) && all(>(0),pixels) || return nothing
        choices=filter(t->t.axis_index==index,targets[]);isempty(choices) && return nothing
        scale=pixels./spans
        function distance(t)
            a=M.Point2d(t.segment[1:2]).*scale;b=M.Point2d(t.segment[3:4]).*scale
            p=M.Point2d(point).*scale;v=b-a;den=sum(abs2,v)
            u=den==0 ? 0. : clamp(sum((p-a).*v)/den,0.,1.)
            sum(abs2,p-(a+u*v))
        end
        ds=distance.(choices);i=argmin(ds);ds[i]<=12^2 ? choices[i] : nothing
    end
    function hover(point,index=1)
        t=nearby(point,index);hover_text[]=t===nothing ? "" : t.label;t
    end
    function pick(point,index=1)
        t=hover(point,index);t===nothing && return false
        isempty(t.selection) && return false
        request(;t.selection...)
    end
    function dispose(;close_session=false)
        closed[] && return
        closed[]=true;disabled[]=true;VIZ._cancel_inspection_requests!(s,owner)
        session_token[]==0 || VIZ._off_inspection!(s,session_token[]);session_token[]=0
        foreach(M.off,observers);empty!(observers);foreach(M.off,pointer_observers);empty!(pointer_observers)
        fig[]===nothing || empty!(fig[]);retained[]=nothing;axis[]=nothing;empty!(chart_axes[]);empty!(targets[])
        content[]=B.DOM.p("Inspector closed. The final snapshot and exact values remain available.")
        browser_client[]=nothing;app=app_ref[].value;app===nothing || delete!(_INSPECTION_UI_HANDLES,app)
        close_session && VIZ.close_inspection!(s)
        nothing
    end
    function request_close()
        (closed[] || closing[]) && return nothing
        closing[]=true;disabled[]=true;VIZ._cancel_inspection_requests!(s)
        close_task[]=@async begin
            while updating[] || VIZ.inspection_summary(s).updating;sleep(0.01);end
            VIZ.inspection_summary(s).closed || VIZ.close_inspection!(s)
            closed[] || dispose()
        end
        nothing
    end
    function refresh()
        (VIZ.inspection_summary(s).closed || _inspection_browser_ended(browser_client[])) && return dispose()
        snap=VIZ.inspection_snapshot(s);state=VIZ.inspection_selection(s);updating[]=true
        try
            hover_text[]=""
            old=fig[];panels=filter(p->!(get(p.metadata,:panel_style,nothing) in (:text_only,:matrix)),snap.panels)
            drawing=VIZ.VisualizationSpec(:stability_live;panels,metadata=(;panel_columns=2,figure_size=(1200,chambers ? 850 : 550)))
            reused=retained[]!==nothing && _visual_update_retained!(M,VIZ,_HANDLERS,retained[],drawing;style,replace_plots=true)
            if reused
                current=fig[];update_count[]+=1
            else
                foreach(M.off,pointer_observers);empty!(pointer_observers)
                retained[]=_visual_retained_figure(M,VIZ,_HANDLERS,drawing;size,style)
                current=retained[].figure;fig[]=current;rebuild_count[]+=1
                chart_axes[]=filter(a->a isa M.Axis,current.content);axis[]=first(chart_axes[])
                push!(pointer_observers,M.on(M.events(current.scene).mouseposition) do _
                    (closed[] || updating[]) && return nothing
                    for (i,ax) in enumerate(chart_axes[])
                        M.is_mouseinside(ax.scene) || continue
                        hover(M.mouseposition(ax.scene),i);return nothing
                    end
                    hover_text[]="";nothing
                end)
                push!(pointer_observers,M.on(M.events(current.scene).mousebutton;priority=10) do e
                    if !closed[] && !updating[] && e.button==M.Mouse.left && e.action==M.Mouse.press
                        for (i,ax) in enumerate(chart_axes[])
                            M.is_mouseinside(ax.scene) || continue
                            return M.Consume(pick(M.mouseposition(ax.scene),i))
                        end
                    end
                    M.Consume(false)
                end)
            end
            targets[]=NamedTuple[]
            for (i,p) in enumerate(panels),t in get(p.metadata,:pick_targets,());push!(targets[],merge(t,(;axis_index=i)));end
            view.option_index[]=findfirst(==(state.view),views)
            if chambers;point_x.value[]=string(state.anchor[1]);point_y.value[]=string(state.anchor[2]);end
            threshold.value[]=string(state.threshold);baseline.value[]=string(state.baseline)
            layer.value[]=string(state.layer);factor.value[]=string(state.factor);vertex.value[]=string(state.vertex)
            status_text[]="$(state.view); threshold $(state.threshold); selected vertex $(state.vertex)."
            readout[]=_inspection_dom_readout(VIZ.VisualizationSpec(:stability_readout;panels=filter(p->get(p.metadata,:panel_style,nothing) in (:text_only,:matrix),snap.panels));style)
            if !reused
                content[]=current;old===nothing || empty!(old)
            end
        finally;updating[]=false;end
        nothing
    end
    function parse_int(c)
        v=tryparse(Int,strip(c.value[]));v===nothing && throw(ArgumentError("Enter an integer."));v
    end
    function parse_exact(c)
        text=strip(c.value[]);parts=split(text,"//")
        if length(parts)==2
            a=tryparse(BigInt,strip(parts[1]));b=tryparse(BigInt,strip(parts[2]))
            (a===nothing || b===nothing || iszero(b)) && throw(ArgumentError("Enter a finite number or an exact fraction a//b."))
            return a//b
        end
        a=tryparse(BigInt,text);a===nothing || return a//big(1)
        x=tryparse(Float64,text)
        x===nothing || !isfinite(x) ? throw(ArgumentError("Enter a finite number or an exact fraction a//b.")) : TO.CoreModules.QQ(x)
    end
    listen(_->commit(()->request(view=view.value[],threshold=parse_exact(threshold),vertex=parse_int(vertex),
        anchor=chambers ? (parse_exact(point_x),parse_exact(point_y)) : nothing,
        factor=landscape ? 0 : parse_int(factor),layer=landscape ? parse_int(layer) : 1,
        baseline=landscape ? parse_exact(baseline) : 0)),apply.value)
    listen(v->request(view=v),view.value)
    listen(_->request(;s.initial...),reset.value)
    listen(_->commit(request_close),close_button.value)
    session_token[]=VIZ._on_inspection(s,_ -> refresh());refresh()
    label(text,c)=B.DOM.label(B.DOM.span(text;style="display:block;font-weight:600"),c;style="display:block;min-width:0")
    row(children...)=B.DOM.div(children...;style="display:flex;flex-wrap:wrap;gap:$(style.gap)px;align-items:end;margin:$(style.gap)px 0")
    fg=_inspection_css_color(style,VIZ._VisualRole(:foreground));bg=_inspection_css_color(style,VIZ._VisualRole(:background))
    selectors=Any[label("View",view),label("Threshold",threshold),label(landscape ? "Sample vertex" : "Finite vertex",vertex)]
    chambers && append!(selectors,[label("Exact anchor x",point_x),label("Exact anchor y",point_y)])
    landscape ? append!(selectors,[label("Baseline threshold",baseline),label("Landscape layer",layer)]) : push!(selectors,label("Factor (0 uses threshold)",factor))
    dom=B.DOM.div(B.DOM.h2("HN and skyscraper inspection"),B.DOM.p("Select exact values or a marked chart point. Thresholds retain slopes greater than or equal to the chosen value. Fractions may be entered as a//b."),
        row(selectors...,apply,reset,close_button),
        B.DOM.p(error_text;id="$dom_id-error",role="alert"),B.DOM.p(status_text;id="$dom_id-status",role="status"),
        B.DOM.p(hover_text;id="$dom_id-hover",role="status",style="min-height:1.5em"),
        B.DOM.div(content;style="max-width:100%;overflow:auto",tabindex="0",var"aria-label"="Stability charts"),
        B.DOM.div(readout;id="$dom_id-readout");id=dom_id,
        style="box-sizing:border-box;font-family:'$(style.font)',sans-serif;font-size:$(style.fontsize)px;color:$fg;background:$bg;padding:$(style.padding)px;max-width:100%;overflow-wrap:anywhere")
    app=B.App(;title="TamerOp stability") do client
        lock(browser_lock) do
            closed[] && return B.DOM.p("This inspector has been closed.")
            if browser_client[]!==nothing && browser_client[]!==client
                existing=app_ref[].value;existing===nothing || (existing.session[]=browser_client[])
                return B.DOM.p("This widget already has a browser client. Call visualize(session) again to open an independent linked view.")
            end
            if browser_client[]===nothing
                browser_client[]=client;_inspection_keyboard_navigation!(client,dom)
                push!(observers,M.on(client.on_close) do ended;ended && dispose();nothing;end)
            end
            dom
        end
    end
    app_ref[]=WeakRef(app)
    _INSPECTION_UI_HANDLES[app]=(;session=s,figure=fig,axis,chart_axes,targets,controls=(;view,x=point_x,y=point_y,threshold,baseline,layer,factor,vertex,apply,reset,close=close_button),
        callbacks=(;pick,hover,refresh),dispose,closed,closing,close_task,disabled,observers,pointer_observers,session_token,browser_client,
        retained,rebuild_count,update_count,error_text,status_text,hover_text,readout,dom_id,style)
    app
end
