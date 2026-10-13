# Source gestures create validated pairs; hover never runs algebra.
function _render_relative_inspector(spec;display=:inline,figure=nothing,size=nothing,style=VIZ.VisualStyle())
    VIZ._check_visual_render_options(;display,figure,size,style)
    figure===nothing || throw(ArgumentError("A relative inspector owns its canvas."))
    session=spec.metadata.session;B,M=WGLMakie.Bonito,WGLMakie.Makie
    VIZ.inspection_summary(session).closed && throw(ArgumentError("This relative inspector is closed."))
    _NEXT_INSPECTION_UI_ID[]+=1;dom_id="tamerop-relative-inspector-$(_NEXT_INSPECTION_UI_ID[])"
    disabled=M.Observable(false);closed=Ref(false);updating=Ref(false)
    browser_client=Ref{Any}(nothing);app_ref=Ref(WeakRef(nothing));owner=Ref(nothing)
    error_text=M.Observable("");status_text=M.Observable("");hover_text=M.Observable("")
    content=M.Observable{Any}(B.DOM.div());readout=M.Observable{Any}(B.DOM.div());table=M.Observable{Any}(B.DOM.div())
    fig=Ref{Any}(nothing);axis=Ref{Any}(nothing);targets=Ref(NamedTuple[])
    observers=Any[];pointer_observers=Any[];token=Ref(0);rebuild_count=Ref(0)
    cs=_inspection_control_style(style)
    input(name,value)=B.TextField(value;id="$dom_id-$name",disabled,style=cs)
    button(name,text)=B.Button(text;id="$dom_id-$name",disabled,style=cs)
    cells=input("cells","");scale=input("scale","");degree=input("degree","");entry=input("entry","")
    lower=input("lower","");upper=input("upper","");witness=input("witness","1")
    apply=button("apply","Select cells");query=button("query","Query scale and degree")
    brush=button("brush","Apply closed brush");delay=button("delay","Set A entry")
    closure=button("closure","Use face closure");induced=button("induced","Use induced vertices");chain=button("chain","Use chain closure")
    clear=button("clear","Empty selection");full=button("full","Select all")
    maps=button("maps","Compute maps");representative=button("representative","Show quotient witness")
    reset=button("reset","Reset");close_button=button("close","Close inspector")
    function status(stage,error)
        closed[] && return nothing
        error_text[]=stage===:error ? "Selection unchanged: $error" : ""
        stage===:error && (status_text[]="Query failed; the previous committed answer remains visible.")
        stage===:pending && (status_text[]="Computing the selected pair; the chart shows the previous answer.")
        nothing
    end
    function request(;kwargs...)
        closed[] && return false
        try
            VIZ._inspection_require_open(session)
            _inspection_control_selection!(session,owner,browser_client[],status;kwargs...)
            true
        catch err
            err isa InterruptException && rethrow();error_text[]=sprint(showerror,err);false
        end
    end
    function commit(f)
        try;f()
        catch err;err isa InterruptException && rethrow();error_text[]=sprint(showerror,err);false;end
    end
    function listen(f,observable)
        push!(observers,M.on(observable) do value
            updating[] || closed[] || f(value);nothing
        end)
    end
    coordinate(control)=VIZ._parse_inspection_coordinate(strip(control.value[]))
    coordinates(control)=Tuple(VIZ._parse_inspection_coordinate(strip(v)) for v in split(control.value[],','))
    integer(control)=let v=tryparse(Int,strip(control.value[]));v===nothing && throw(ArgumentError("Enter an integer index or degree."));v;end
    function toggle(i)
        ids=copy(session.state.cells);i in ids ? filter!(!=(i),ids) : push!(ids,i)
        request(;cells=ids,details=:summary,witness=1)
    end
    function nearby(point)
        axis[]===nothing && return nothing
        distances=[let p=M.project(axis[].scene,length(t.point)==2 ? M.Point2d(t.point) : M.Point3d(t.point))+minimum(M.viewport(axis[].scene)[])
            sum((point[k]-p[k])^2 for k in 1:2)
        end for t in targets[]]
        isempty(distances) && return nothing
        i=argmin(distances);distances[i]<=20^2 ? targets[][i] : nothing
    end
    function hover(point)
        t=nearby(point)
        hover_text[]=t===nothing ? "" : "Cell $(t.index), dimension $(t.dimension), $(t.role)."
        t
    end
    function pick(point)
        t=hover(point);t===nothing ? false : toggle(t.index)
    end
    function dispose(;close_session=false)
        closed[] && return nothing
        closed[]=true;disabled[]=true;VIZ._cancel_inspection_requests!(session,owner)
        token[]==0 || VIZ._off_inspection!(session,token[]);token[]=0
        foreach(M.off,observers);empty!(observers);foreach(M.off,pointer_observers);empty!(pointer_observers)
        fig[]===nothing || empty!(fig[]);axis[]=nothing;empty!(targets[]);browser_client[]=nothing
        content[]=B.DOM.p("Inspector closed. The final static snapshot remains available.")
        table[]=B.DOM.div()
        app=app_ref[].value;app===nothing || delete!(_INSPECTION_UI_HANDLES,app)
        close_session && VIZ.close_inspection!(session)
        nothing
    end
    function refresh()
        (VIZ.inspection_summary(session).closed || _inspection_browser_ended(browser_client[])) && return dispose()
        snapshot=VIZ.inspection_snapshot(session);state=VIZ.inspection_selection(session)
        updating[]=true
        try
            hover_text[]=""
            cells.value[]=join(state.cells,", ");scale.value[]=string(state.scale);degree.value[]=string(state.degree);witness.value[]=string(state.witness);entry.value[]=""
            if state.brush!==nothing
                lower.value[]=join(state.brush.lower,", ");upper.value[]=join(state.brush.upper,", ")
            else
                lower.value[]="";upper.value[]=""
            end
            dims=snapshot.metadata.counts
            status_text[]="H$(state.degree) at $(state.scale): K=$(dims.ambient), A=$(dims.subcomplex), (K,A)=$(dims.relative). Rule: $(state.rule). $(length(snapshot.metadata.selected)) selected; $(length(snapshot.metadata.added)) added."
            readout[]=_inspection_dom_readout(VIZ.VisualizationSpec(:relative_readout;
                panels=filter(p->get(p.metadata,:panel_style,nothing) in (:text_only,:matrix),snapshot.panels));style)
            rows=map(Iterators.take(snapshot.metadata.table,50)) do r
                B.DOM.tr(B.DOM.td(string(r.index)),B.DOM.td(string(r.source)),B.DOM.td(string(r.dimension)),
                    B.DOM.td(string(r.role)),B.DOM.td(string(r.ambient_grade)),B.DOM.td(r.subcomplex_grade===nothing ? "-" : string(r.subcomplex_grade)))
            end
            table[]=B.DOM.div(B.DOM.p("Source table: first $(min(50,length(snapshot.metadata.table))) of $(length(snapshot.metadata.table)) cells. Select their exact global indices above; all entries are available through spatial_cells."),
                B.DOM.table(B.DOM.thead(B.DOM.tr((B.DOM.th(t) for t in ("Index","Source ID","Degree","Selection","K entry","A entry"))...)),B.DOM.tbody(rows...);
                    style="border-collapse:separate;border-spacing:16px 4px;white-space:nowrap"))
            panels=filter(p->!(get(p.metadata,:panel_style,nothing) in (:text_only,:matrix)),snapshot.panels)
            drawing=VIZ.VisualizationSpec(:relative_charts;panels,metadata=(;panel_columns=2,figure_size=(1100,380*cld(length(panels),2))))
            foreach(M.off,pointer_observers);empty!(pointer_observers)
            current=Base.invokelatest(_BASE_WGL_RENDER,drawing;size,style)
            old=fig[];fig[]=current;rebuild_count[]+=1
            geometry=session.object.source isa TO.DataIngestion.FilteredComplex
            axis[]=geometry ? first(filter(a->a isa Union{M.Axis,M.Axis3},current.content)) : nothing
            targets[]=geometry ? panels[1].metadata.pick_targets : NamedTuple[]
            if geometry
                pressed=Ref{Any}(nothing)
                push!(pointer_observers,M.on(M.events(current.scene).mouseposition) do _
                    if !closed[] && axis[]!==nothing
                        M.is_mouseinside(axis[].scene) ? hover(M.events(current.scene).mouseposition[]) : (hover_text[]="")
                    end
                    nothing
                end)
                push!(pointer_observers,M.on(M.events(current.scene).mousebutton;priority=20) do event
                    if !closed[] && event.button==M.Mouse.left
                        if event.action==M.Mouse.press && M.is_mouseinside(axis[].scene)
                            pressed[]=(;pixel=Tuple(M.events(current.scene).mouseposition[]),data=axis[] isa M.Axis ? Tuple(M.mouseposition(axis[])) : nothing)
                            return M.Consume(axis[] isa M.Axis)
                        elseif event.action==M.Mouse.release && pressed[]!==nothing
                            start=pressed[];pressed[]=nothing
                            pixel=M.events(current.scene).mouseposition[]
                            if sum((pixel[k]-start.pixel[k])^2 for k in 1:2)<=25
                                return M.Consume(pick(pixel))
                            elseif axis[] isa M.Axis && state.rule!==:chain_closure
                                stop=Tuple(M.mouseposition(axis[]));vd=session.options.view_dims
                                box=(;dims=vd,lower=Tuple(min.(start.data,stop)),upper=Tuple(max.(start.data,stop)))
                                return M.Consume(request(;brush=box,details=:summary,witness=1))
                            end
                        end
                    end
                    M.Consume(false)
                end)
            end
            content[]=current;old===nothing || empty!(old)
        finally;updating[]=false;end
        nothing
    end
    listen(_->commit(()->request(;cells=isempty(strip(cells.value[])) ? Int[] : [let n=tryparse(Int,strip(v));n===nothing && throw(ArgumentError("Enter comma-separated global cell indices."));n;end for v in split(cells.value[],',')],details=:summary,witness=1)),apply.value)
    listen(_->commit(()->request(;scale=coordinate(scale),degree=integer(degree),details=:summary,witness=1)),query.value)
    listen(_->commit(()->request(;brush=(;dims=session.options.view_dims,lower=coordinates(lower),upper=coordinates(upper)),details=:summary,witness=1)),brush.value)
    listen(_->commit(()->request(;entry=isempty(strip(entry.value[])) ? nothing : coordinate(entry),details=:summary,witness=1)),delay.value)
    for (control,rule) in ((closure,:face_closure),(induced,:induced),(chain,:chain_closure))
        # Changing the rule starts an empty selection, so edge seeds do not
        # silently acquire a different meaning in induced-vertex mode.
        listen(_->request(;cells=Int[],rule,details=:summary,witness=1),control.value)
    end
    listen(_->request(;cells=Int[],details=:summary,witness=1),clear.value)
    listen(full.value) do _
        G=session.object.pair.ambient
        ids=session.state.rule===:induced ? findall(==(0),G.cell_dims) : collect(eachindex(G.grades))
        request(;cells=ids,details=:summary,witness=1)
    end
    listen(_->request(;details=:maps),maps.value)
    listen(_->commit(()->request(;details=:representative,witness=integer(witness))),representative.value)
    listen(_->request(;session.initial...),reset.value)
    listen(_->dispose(;close_session=true),close_button.value)
    token[]=VIZ._on_inspection(session,_->refresh());refresh()
    label(text,control)=B.DOM.label(B.DOM.span(text;style="display:block;font-weight:600"),control;style="min-width:0")
    row(children...)=B.DOM.div(children...;style="display:flex;flex-wrap:wrap;align-items:end;gap:$(style.gap)px;margin:$(style.gap)px 0")
    foreground=_inspection_css_color(style,VIZ._VisualRole(:foreground));background=_inspection_css_color(style,VIZ._VisualRole(:background))
    dom=B.DOM.div(B.DOM.h2("Select a region and ask a relative question"),
        B.DOM.p("Choose a rule, then select exact cell indices or drag a rectangle in the ambient 2D panel. A click toggles one cell; hovering only reads its identity. The closed brush selects vertices in induced mode and cell vertex-centroids in face-closure mode. It does not cut cells. In 3D use the exact box fields."),
        row(closure,induced,chain),row(label("Global cell indices",cells),apply,clear,full),
        row(label("Brush lower coordinates",lower),label("Brush upper coordinates",upper),brush),
        row(label("Scale",scale),label("Homology degree",degree),query),
        row(label("Set common A entry (blank restores K)",entry),delay),
        row(maps,label("Quotient witness index",witness),representative,reset,close_button),
        B.DOM.p(error_text;id="$dom_id-error",role="alert",style="white-space:pre-line"),
        B.DOM.p(status_text;id="$dom_id-status",role="status"),B.DOM.p(hover_text;id="$dom_id-hover",role="status"),
        B.DOM.div(content;style="max-width:100%;overflow:auto",tabindex="0",var"aria-label"="Source geometry and persistence charts"),
        B.DOM.div(readout;id="$dom_id-readout"),
        B.DOM.div(table;id="$dom_id-table",style="max-width:100%;overflow:auto",tabindex="0",var"aria-label"="Exact source cell and entry grade table");id=dom_id,
        style="box-sizing:border-box;font-family:'$(style.font)',sans-serif;font-size:$(style.fontsize)px;line-height:1.45;color:$foreground;background:$background;padding:$(style.padding)px;max-width:100%;overflow-wrap:anywhere")
    app=B.App(;title="TamerOp relative persistence") do client
        closed[] && return B.DOM.p("This inspector has been closed.")
        if browser_client[]!==nothing && browser_client[]!==client
            return B.DOM.p("This widget already has a browser client. Call visualize(session) again for a linked view.")
        end
        if browser_client[]===nothing
            browser_client[]=client;_inspection_keyboard_navigation!(client,dom)
            push!(observers,M.on(client.on_close) do ended;ended && dispose();nothing;end)
        end
        dom
    end
    app_ref[]=WeakRef(app)
    _INSPECTION_UI_HANDLES[app]=(;session,figure=fig,axis,targets,
        controls=(;cells,scale,degree,entry,lower,upper,witness,apply,query,brush,delay,closure,induced,chain,clear,full,maps,representative,reset,close=close_button),
        callbacks=(;pick,hover,refresh,toggle),dispose,closed,disabled,observers,pointer_observers,session_token=token,browser_client,
        rebuild_count,error_text,status_text,hover_text,readout,dom_id,style)
    app
end
