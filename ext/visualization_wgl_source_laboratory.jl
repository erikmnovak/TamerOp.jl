# Draft gestures never run construction. Rebuilds use the common bounded,
# replaceable request worker, with stale results rejected before publication.
function _render_source_laboratory(spec;display=:inline,figure=nothing,size=nothing,style=VIZ.VisualStyle())
    VIZ._check_visual_render_options(;display,figure,size,style)
    figure===nothing || throw(ArgumentError("The source laboratory owns its figure."))
    s=spec.metadata.session;B,M=WGLMakie.Bonito,WGLMakie.Makie
    VIZ.inspection_summary(s).closed && throw(ArgumentError("This laboratory is closed."))
    _NEXT_INSPECTION_UI_ID[]+=1;dom_id="tamerop-source-laboratory-$(_NEXT_INSPECTION_UI_ID[])"
    disabled=M.Observable(false);closed=Ref(false);updating=Ref(false);closing=Ref(false)
    close_task=Ref{Union{Nothing,Task}}(nothing);owner=Ref(nothing);browser_client=Ref{Any}(nothing);app_ref=Ref(WeakRef(nothing))
    error_text=M.Observable("");status_text=M.Observable("");hover_text=M.Observable("")
    content=M.Observable{Any}(B.DOM.div());readout=M.Observable{Any}(B.DOM.div())
    fig=Ref{Any}(nothing);axes=Ref(Any[]);retained=Ref{Any}(nothing);targets=Ref(NamedTuple[])
    observers=Any[];pointer_observers=Any[];token=Ref(0);drag=Ref{Any}(nothing)
    preview=M.Observable(M.Point2d[])
    rebuild_count=Ref(0);update_count=Ref(0)
    cs=_inspection_control_style(style)
    input(name,value)=B.TextField(string(value);id="$dom_id-$name",disabled,style=cs)
    button(name,value)=B.Button(value;id="$dom_id-$name",disabled,style=cs)
    point=input("point",1);x=input("x",0);y=input("y",0)
    mass=input("mass",s.draft.mass);cap=input("cap",s.draft.cap)
    diameter=input("diameter",s.state.diameter);codensity=input("codensity",s.state.codensity)
    move=button("move","Move point in draft");add=button("add","Add point to draft");remove=button("remove","Remove point from draft")
    settings=button("settings","Apply draft settings");rebuild=button("rebuild","Rebuild source")
    cancel=button("cancel","Cancel pending rebuild");undo=button("undo","Undo draft edit");redo=button("redo","Redo draft edit")
    reset=button("reset","Restore original draft");select=button("select","Inspect parameters");close_button=button("close","Close laboratory")
    coordinate(c)=VIZ._parse_inspection_coordinate(c.value[])
    integer(c)=let n=tryparse(Int,strip(c.value[]));n===nothing && throw(ArgumentError("Use an integer point ID."));n;end
    function attempt(f)
        (closed[]||closing[]) && return false
        try;result=f();result===false && return false;error_text[]="";true
        catch err;err isa InterruptException && rethrow();error_text[]=sprint(showerror,err);false;end
    end
    function status(stage,error)
        closed[] && return nothing
        error_text[]=stage===:error ? "Committed result unchanged: $error" : ""
        stage===:pending && (status_text[]="Computing requested result. Showing the last committed mathematics.")
        nothing
    end
    function request(;kwargs...)
        attempt() do
            client=browser_client[]
            if client!==nothing && isready(client)
                VIZ._request_inspection!(s,owner;status,kwargs...)
            else
                VIZ._apply_inspection_request!(s;kwargs...)
            end
        end
    end
    function listen(f,observable;during_refresh=false)
        push!(observers,M.on(observable) do value
            (updating[] && !during_refresh)||closed[]||closing[]||f(value);nothing
        end)
    end
    function selected(id)
        d=VIZ.source_draft(s);i=findfirst(==(id),d.ids);i===nothing && return false
        point.value[]=string(id);x.value[]=string(d.points[i,1]);y.value[]=string(d.points[i,2]);true
    end
    function nearby(pixel)
        length(axes[])>=2 || return nothing
        ax=axes[][2];M.is_mouseinside(ax.scene) || return nothing
        best=nothing;distance=18.0^2
        for t in targets[]
            q=M.project(ax.scene,M.Point2d(t.point))+minimum(M.viewport(ax.scene)[])
            delta=sum((pixel[k]-q[k])^2 for k in 1:2)
            if delta<distance;best=t;distance=delta;end
        end
        best
    end
    function hover(pixel)
        t=nearby(pixel);hover_text[]=t===nothing ? "" : "Source point $(t.id) at $(t.point). Drag to edit the draft; rebuild explicitly."
        t
    end
    function dispose(;close_session=false)
        closed[] && return nothing
        closed[]=true;disabled[]=true;drag[]=nothing;VIZ._cancel_inspection_requests!(s,owner)
        token[]==0 || VIZ._off_inspection!(s,token[]);token[]=0
        foreach(M.off,observers);empty!(observers);foreach(M.off,pointer_observers);empty!(pointer_observers)
        fig[]===nothing || empty!(fig[]);retained[]=nothing;empty!(axes[]);empty!(targets[]);browser_client[]=nothing
        content[]=B.DOM.p("Laboratory closed. The final static comparison remains available.")
        app=app_ref[].value;app===nothing || delete!(_INSPECTION_UI_HANDLES,app)
        close_session && VIZ.close_inspection!(s);nothing
    end
    function request_close()
        (closed[]||closing[]) && return nothing
        closing[]=true;disabled[]=true;drag[]=nothing;VIZ._cancel_inspection_requests!(s)
        close_task[]=@async begin
            while updating[] || VIZ.inspection_summary(s).updating;sleep(0.01);end
            VIZ.inspection_summary(s).closed || VIZ.close_inspection!(s)
            closed[] || dispose()
        end
        nothing
    end
    function refresh()
        (VIZ.inspection_summary(s).closed || _inspection_browser_ended(browser_client[])) && return dispose()
        updating[]=true
        try
            preview[]=M.Point2d[];drag[]=nothing
            snapshot=VIZ.inspection_snapshot(s);d=VIZ.source_draft(s);state=VIZ.inspection_selection(s)
            mass.value[]=string(d.dtm_mass);cap.value[]=string(d.max_diameter)
            diameter.value[]=string(state.diameter);codensity.value[]=string(state.codensity)
            id=tryparse(Int,point.value[]);id in d.ids || (id=first(d.ids));selected(id)
            status_text[]=snapshot.metadata.dirty ? "Draft input $(d.input_version) awaits rebuild. Showing committed generation $(state.generation)." : "Committed generation $(state.generation). Source and barcode agree."
            hover_text[]=""
            readout[]=_inspection_dom_readout(VIZ.VisualizationSpec(:laboratory_readout;panels=[last(snapshot.panels)]);style)
            drawing=VIZ.VisualizationSpec(:laboratory_drawing;panels=snapshot.panels[1:4],metadata=(;panel_columns=2,figure_size=(1120,850)))
            old=fig[]
            if retained[]!==nothing && _visual_update_retained!(M,VIZ,_HANDLERS,retained[],drawing;style,replace_plots=true)
                update_count[]+=1
            else
                foreach(M.off,pointer_observers);empty!(pointer_observers)
                retained[]=_visual_retained_figure(M,VIZ,_HANDLERS,drawing;size,style);fig[]=retained[].figure;rebuild_count[]+=1
                axes[]=filter(a->a isa M.Axis,fig[].content)
                M.scatter!(axes[][2],preview;color=_inspection_css_color(style,VIZ._VisualRole(:selected)),markersize=16,strokewidth=2,strokecolor=:white)
                push!(pointer_observers,M.on(M.events(fig[].scene).mouseposition) do pixel
                    if !closed[] && !updating[]
                        if drag[]===nothing;hover(pixel)
                        else
                            p=Tuple(M.mouseposition(axes[][2]));preview[]=[M.Point2d(p)];x.value[]=string(p[1]);y.value[]=string(p[2])
                            hover_text[]="Draft move preview: $(p). Release to keep the edit; rebuild explicitly."
                        end
                    end
                    nothing
                end)
                push!(pointer_observers,M.on(M.events(fig[].scene).mousebutton;priority=30) do e
                    (closed[]||closing[]||updating[]) && return M.Consume(false)
                    if e.button==M.Mouse.left && e.action==M.Mouse.press
                        t=nearby(M.events(fig[].scene).mouseposition[])
                        if t!==nothing;selected(t.id);drag[]=(;id=t.id,start=t.point,pixel=Tuple(M.events(fig[].scene).mouseposition[]));return M.Consume(true);end
                    elseif e.button==M.Mouse.left && e.action==M.Mouse.release && drag[]!==nothing
                        origin=drag[];drag[]=nothing;preview[]=M.Point2d[];p=Tuple(M.mouseposition(axes[][2]))
                        pixel=M.events(fig[].scene).mouseposition[]
                        all(isfinite,p) && sum((pixel[i]-origin.pixel[i])^2 for i in 1:2)>16 &&
                            request(;operation=:edit,edit=(;move=(origin.id,p)))
                        return M.Consume(true)
                    end
                    M.Consume(false)
                end)
                content[]=fig[];old===nothing || empty!(old)
            end
            targets[]=collect(snapshot.panels[2].metadata.pick_targets)
        finally;updating[]=false;end
        nothing
    end
    listen(_->attempt(()->request(;operation=:edit,edit=(;move=(integer(point),(coordinate(x),coordinate(y)))))),move.value)
    listen(_->attempt(()->request(;operation=:edit,edit=(;add=(coordinate(x),coordinate(y))))),add.value)
    listen(_->attempt(()->request(;operation=:edit,edit=(;remove=integer(point)))),remove.value)
    listen(_->attempt(()->request(;operation=:edit,edit=(;dtm_mass=coordinate(mass),max_diameter=coordinate(cap)))),settings.value)
    listen(_->request(;rebuild=true),rebuild.value)
    listen(cancel.value;during_refresh=true) do _
        VIZ._cancel_inspection_requests!(s,owner);status_text[]="Pending work cancelled. The last committed result is retained."
    end
    listen(_->request(;operation=:undo),undo.value)
    listen(_->request(;operation=:redo),redo.value)
    listen(_->request(;operation=:reset),reset.value)
    listen(select.value) do _;attempt(()->request(;diameter=coordinate(diameter),codensity=coordinate(codensity)));end
    listen(_->request_close(),close_button.value;during_refresh=true)
    token[]=VIZ._on_inspection(s,_->refresh());refresh()
    label(text,control)=B.DOM.label(B.DOM.span(text;style="display:block;font-weight:600"),control;style="min-width:0;max-width:100%")
    row(children...)=B.DOM.div(children...;style="display:flex;flex-wrap:wrap;align-items:end;gap:$(style.gap)px;margin:$(style.gap)px 0")
    foreground=_inspection_css_color(style,VIZ._VisualRole(:foreground));background=_inspection_css_color(style,VIZ._VisualRole(:background))
    dom=B.DOM.div(B.DOM.h2("Source and construction laboratory"),
        B.DOM.p("Drag a point in the edited source, or use its ID and coordinates. Draft edits leave the committed barcode intact. Rebuild when ready to compare the changed mathematics."),
        row(label("Point ID",point),label("Point x",x),label("Point y",y),move,add,remove),
        row(label("DTM mass",mass),label("Maximum Rips diameter",cap),settings),
        row(rebuild,cancel,undo,redo,reset,close_button),
        B.DOM.details(B.DOM.summary("Inspect the constructed filtration"),
            B.DOM.p("These parameters select the retained construction. Changing codensity recomputes its diameter barcode; changing diameter only moves the view."),
            row(label("Displayed diameter",diameter),label("Codensity sublevel",codensity),select)),
        B.DOM.p(error_text;id="$dom_id-error",role="alert"),B.DOM.p(status_text;id="$dom_id-status",role="status"),B.DOM.p(hover_text;id="$dom_id-hover",role="status",style="height:3em;overflow:auto"),
        B.DOM.div(content;id="$dom_id-chart",style="max-width:100%;overflow:auto",tabindex="0",var"aria-label"="Original and edited source comparison"),
        B.DOM.div(readout;id="$dom_id-readout");id=dom_id,
        style="box-sizing:border-box;font-family:'$(style.font)',sans-serif;font-size:$(style.fontsize)px;line-height:1.45;color:$foreground;background:$background;padding:$(style.padding)px;max-width:100%;overflow-wrap:anywhere")
    app=B.App(;title="TamerOp source laboratory") do client
        closed[] && return B.DOM.p("This laboratory has been closed.")
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
    _INSPECTION_UI_HANDLES[app]=(;session=s,figure=fig,axes,targets,retained,
        controls=(;point,x,y,mass,cap,diameter,codensity,move,add,remove,settings,rebuild,cancel,undo,redo,reset,select,close=close_button),
        callbacks=(;refresh,selected,hover,request),dispose,closed,disabled,observers,pointer_observers,session_token=token,browser_client,
        close_task,drag,rebuild_count,update_count,error_text,status_text,hover_text,readout,dom_id,style)
    app
end
