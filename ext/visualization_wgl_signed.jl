# Signed-term and exact-query controls use the same transactional session API as
# static notebook selections. Browser disconnect releases its own listeners.
function _render_signed_inspector(spec;display=:inline,figure=nothing,size=nothing,style=VIZ.VisualStyle())
    VIZ._check_visual_render_options(;display,figure,size,style)
    s=spec.metadata.session
    VIZ.inspection_summary(s).closed && throw(ArgumentError("This signed inspector is closed."))
    figure===nothing || throw(ArgumentError("A live signed inspector owns its figure; compose its static snapshot instead."))
    B,M=WGLMakie.Bonito,WGLMakie.Makie
    _NEXT_INSPECTION_UI_ID[]+=1;dom_id="tamerop-signed-inspector-$(_NEXT_INSPECTION_UI_ID[])"
    figure_size=size===nothing ? (1200,1000) : size
    disabled=M.Observable(false);error_text=M.Observable("");status_text=M.Observable("");hover_text=M.Observable("")
    content=M.Observable{Any}(B.DOM.div());readout=M.Observable{Any}(B.DOM.div())
    cs=_inspection_control_style(style)
    term=B.TextField("0";id="$dom_id-term",disabled,style=cs)
    query=B.TextField("";id="$dom_id-query",disabled,style=cs)
    smoothing=B.TextField("0, 0";id="$dom_id-smoothing",disabled,style=cs)
    button(name,label)=B.Button(label;id="$dom_id-$name",disabled,style=cs)
    apply_term=button("apply-term","Select term");apply_query=button("apply-query","Evaluate query")
    apply_smoothing=button("apply-smoothing","Apply smoothing")
    next=button("next","Next term");previous=button("previous","Previous term")
    rectangles=button("rectangles","Rectangles");diagonals=button("diagonals","Diagonals")
    reset=button("reset","Reset selection");close_button=button("close","Close inspector")
    closed=Ref(false);updating=Ref(false);fig=Ref{Any}(nothing);session_token=Ref(0);rebuild_count=Ref(0)
    observers=Any[];pointer_observers=Any[];axes=Dict{Symbol,Any}();targets=Dict{Symbol,Any}()
    browser_client=Ref{Any}(nothing);app_ref=Ref(WeakRef(nothing));browser_lock=ReentrantLock()
    family=s.object isa VIZ.Invariants.IntervalRankSummary ? :intervals : s.snapshot.metadata.observable===:rank ? :rectangles : :points
    function commit(f)
        closed[] && return false
        disabled[]=true
        try;f();error_text[]="";return true
        catch err
            err isa InterruptException && rethrow()
            error_text[]=sprint(showerror,err);return false
        finally;disabled[]=closed[];end
    end
    function listen(f,observable)
        push!(observers,M.on(observable) do value
            closed[] || updating[] || f(value)
            nothing
        end)
    end
    function coordinates(control,n)
        parts=split(strip(control.value[]),',')
        length(parts)==n || throw(ArgumentError("Enter $n comma-separated coordinates (integers, fractions or decimals)."))
        Tuple(VIZ._parse_inspection_coordinate(strip(p)) for p in parts)
    end
    function nearby(kind,point)
        haskey(axes,kind) && haskey(targets,kind) || return Int[]
        ax=axes[kind];spans=M.widths(ax.finallimits[]);pixels=M.widths(ax.scene.viewport[])
        all(>(0),spans) && all(>(0),pixels) || return Int[]
        scale=pixels ./ spans
        ds=[sum(((t.point[k]-point[k])*scale[k])^2 for k in 1:2) for t in targets[kind]]
        isempty(ds) && return Int[]
        best=minimum(ds);best<=20^2 || return Int[]
        unique(Int[targets[kind][i].id for i in eachindex(ds) if abs(ds[i]-best)<=1e-6])
    end
    function hover(kind,point)
        ids=nearby(kind,point);rs=VIZ.inspection_snapshot(s).metadata.records
        hover_text[]=join(("Term #$id: weight $(rs[id].weight), contribution $(rs[id].contribution)" for id in ids),"\n")
        ids
    end
    function pick(kind,point)
        ids=nearby(kind,point);isempty(ids) && return false
        current=VIZ.inspection_selection(s).term;position=findfirst(==(current),ids)
        id=position===nothing ? first(ids) : ids[mod1(position+1,length(ids))]
        commit(()->VIZ.select_inspection!(s;term=id))
    end
    function attach_pointer(current)
        push!(pointer_observers,M.on(M.events(current.scene).mousebutton;priority=100) do event
            closed[] && return M.Consume(false)
            if event.button==M.Mouse.left && event.action==M.Mouse.press
                for (kind,ax) in axes
                    M.is_mouseinside(ax.scene) || continue
                    return M.Consume(pick(kind,M.mouseposition(ax.scene)))
                end
            end
            M.Consume(false)
        end)
        push!(pointer_observers,M.on(M.events(current.scene).mouseposition) do _
            closed[] && return nothing
            for (kind,ax) in axes
                M.is_mouseinside(ax.scene) || continue
                hover(kind,M.mouseposition(ax.scene));return nothing
            end
            hover_text[]="";nothing
        end)
    end
    function dispose()
        closed[] && return nothing
        closed[]=true;disabled[]=true
        session_token[]==0 || VIZ._off_inspection!(s,session_token[]);session_token[]=0
        foreach(M.off,observers);empty!(observers);foreach(M.off,pointer_observers);empty!(pointer_observers)
        empty!(axes);empty!(targets);fig[]===nothing || empty!(fig[])
        content[]=B.DOM.p("Inspector closed. Its last static snapshot remains available.")
        status_text[]="Inspector closed."
        lock(browser_lock) do;browser_client[]=nothing;end
        app=app_ref[].value;app===nothing || delete!(_INSPECTION_UI_HANDLES,app)
        nothing
    end
    function refresh()
        VIZ.inspection_summary(s).closed && return dispose()
        shot=VIZ.inspection_snapshot(s);updating[]=true
        try
            # A query or smoothing change invalidates the previous hover's
            # contribution even when the pointer has not moved.
            hover_text[]=""
            foreach(M.off,pointer_observers);empty!(pointer_observers);empty!(axes);empty!(targets)
            old=fig[];current=M.Figure(size=figure_size,figure_padding=style.padding,
                backgroundcolor=VIZ._visual_color(style,VIZ._VisualRole(:background)))
            panels=filter(p->get(p.metadata,:panel_style,nothing)!==:text_only,shot.panels)
            for (i,p) in enumerate(panels)
                grid=current[cld(i,2),mod1(i,2)]=M.GridLayout()
                ax=_HANDLERS.render_panel(current,grid,p;style)
                if ax isa M.Axis
                    axes[p.kind]=ax;targets[p.kind]=get(p.metadata,:pick_targets,NamedTuple[])
                end
            end
            _HANDLERS.style_figure(current,style);fig[]=current;attach_pointer(current)
            term.value[]=string(something(s.state.term,0))
            q=s.state.query
            query.value[]=q===nothing ? "" : join(family===:rectangles ? (q.p...,q.q...) : family===:points ? q : q.vertices,", ")
            smoothing.value[]=join(s.state.smoothing,", ")
            m=shot.metadata
            status_text[]="$(m.observable): $(m.reconstructed===nothing ? "choose a query" : "$(m.positive) + ($(m.negative)) = $(m.reconstructed)"). " *
                "Direct check: $(m.agreement). $(length(m.displayed_term_ids)) / $(length(m.records)) terms displayed. Selected: $(something(m.selected_term,0))."
            content[]=current;old===nothing || empty!(old)
            readout[]=_inspection_dom_readout(VIZ.VisualizationSpec(:signed_readout;
                panels=filter(p->get(p.metadata,:panel_style,nothing)===:text_only,shot.panels));style)
            rebuild_count[]+=1
        finally;updating[]=false;end
        nothing
    end
    listen(apply_term.value) do _
        commit() do
            id=tryparse(Int,strip(term.value[]));id===nothing && throw(ArgumentError("Enter an integer term ID."))
            VIZ.select_inspection!(s;term=id)
        end
    end
    for (b,delta) in ((next,1),(previous,-1))
        listen(b.value) do _
            n=length(VIZ.inspection_snapshot(s).metadata.records)
            n==0 || commit(()->VIZ.select_inspection!(s;term=mod1(something(s.state.term,delta>0 ? 0 : 1)+delta,n)))
        end
    end
    listen(apply_query.value) do _
        commit() do
            q=if family===:rectangles
                c=coordinates(query,4);(;p=c[1:2],q=c[3:4])
            elseif family===:points
                coordinates(query,2)
            else
                parts=split(strip(query.value[]),',');ids=tryparse.(Int,strip.(parts))
                any(isnothing,ids) && throw(ArgumentError("Enter comma-separated finite label IDs."))
                (;vertices=Int[ids...])
            end
            VIZ.select_inspection!(s;query=q)
        end
    end
    listen(_->commit(()->VIZ.select_inspection!(s;smoothing=coordinates(smoothing,2))),apply_smoothing.value)
    listen(_->commit(()->VIZ.select_inspection!(s;view=:signed_cancellation)),rectangles.value)
    listen(_->commit(()->VIZ.select_inspection!(s;view=:signed_diagonals)),diagonals.value)
    listen(_->commit(()->VIZ.reset_inspection!(s)),reset.value)
    listen(_->commit(()->VIZ.close_inspection!(s)),close_button.value)
    session_token[]=VIZ._on_inspection(s,_->refresh());refresh()
    label(t,c)=B.DOM.label(B.DOM.span(t;style="display:block;font-weight:600"),c)
    row(children...)=B.DOM.div(children...;style="display:flex;flex-wrap:wrap;align-items:end;gap:12px;margin:12px 0")
    query_label=family===:rectangles ? "Rank query: px, py, qx, qy" : family===:points ? "Grid query: x, y" : "Declared interval: finite label IDs"
    extra=family===:rectangles ? row(label("Smoothing: epsilon x, epsilon y",smoothing),apply_smoothing,rectangles,diagonals) : B.DOM.div()
    dom=B.DOM.div(B.DOM.h2("Signed contributions and cancellation"),
        B.DOM.p("Select a labelled term, prominence point or contribution. Repeated clicks cycle coincident terms. Query sums always include every term."),
        row(label("Term ID (0 clears)",term),apply_term,previous,next),row(label(query_label,query),apply_query),extra,row(reset,close_button),
        B.DOM.p(error_text;id="$dom_id-error",role="alert",style="color:$(_inspection_css_color(style,VIZ._VisualRole(:error)))"),
        B.DOM.p(status_text;id="$dom_id-status",role="status"),B.DOM.p(hover_text;id="$dom_id-hover",style="white-space:pre-line"),
        B.DOM.div(content;style="max-width:100%;overflow:auto",tabindex="0",var"aria-label"="Linked signed contribution charts"),
        B.DOM.div(readout;id="$dom_id-readout");id=dom_id,
        style="font-family:'$(style.font)',sans-serif;line-height:1.45;padding:$(style.padding)px;color:$(_inspection_css_color(style,VIZ._VisualRole(:foreground)));background:$(_inspection_css_color(style,VIZ._VisualRole(:background)))")
    app=B.App(;title="TamerOp signed inspector") do client
        lock(browser_lock) do
            closed[] && return B.DOM.p("This inspector has been closed.")
            if browser_client[]!==nothing && browser_client[]!==client
                existing=app_ref[].value;existing===nothing || (existing.session[]=browser_client[])
                return B.DOM.p("This widget already has a client. Call visualize(session) for another linked view.")
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
    _INSPECTION_UI_HANDLES[app]=(;session=s,figure=fig,axes,targets,controls=(;term,query,smoothing,apply_term,apply_query,apply_smoothing,next,previous,rectangles,diagonals,reset,close=close_button),
        callbacks=(;pick,hover,refresh),dispose,closed,disabled,observers,pointer_observers,session_token,browser_client,rebuild_count,error_text,status_text,hover_text,readout,dom_id,style,figure_size)
    app
end
