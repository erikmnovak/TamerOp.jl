# Playback changes one retained selection at a time; no reduction runs here.
function _render_path_inspector(spec;display=:inline,figure=nothing,size=nothing,style=VIZ.VisualStyle())
    VIZ._check_visual_render_options(;display,figure,size,style)
    figure===nothing || throw(ArgumentError("A path inspector owns its canvas; export its snapshot instead."))
    session=spec.metadata.session;B,M=WGLMakie.Bonito,WGLMakie.Makie
    VIZ.inspection_summary(session).closed && throw(ArgumentError("This path inspector is closed."))
    _NEXT_INSPECTION_UI_ID[]+=1;dom_id="tamerop-path-inspector-$(_NEXT_INSPECTION_UI_ID[])"
    n=length(session.object.frames);event_frames=findall(f->!isempty(f.events),session.object.frames)
    disabled=M.Observable(false);closed=Ref(false);updating=Ref(false);closing=Ref(false)
    close_task=Ref{Union{Nothing,Task}}(nothing)
    browser_client=Ref{Any}(nothing);app_ref=Ref(WeakRef(nothing));owner=Ref(nothing)
    error_text=M.Observable("");status_text=M.Observable("");hover_text=M.Observable("")
    content=M.Observable{Any}(B.DOM.div());readout=M.Observable{Any}(B.DOM.div())
    fig=Ref{Any}(nothing);axes=Ref(Any[]);targets=Ref(Vector{NamedTuple}[])
    retained=Ref{Any}(nothing);rebuild_count=Ref(0);update_count=Ref(0)
    playing=M.Observable(false);direction=Ref(1);timer=Ref{Union{Nothing,Timer}}(nothing)
    observers=Any[];pointer_observers=Any[];token=Ref(0)
    cs=_inspection_control_style(style)
    input(name,value)=B.TextField(value;id="$dom_id-$name",disabled,style=cs)
    button(name,text)=B.Button(text;id="$dom_id-$name",disabled,style=cs)
    frame=input("frame","1");track=input("track","0");scale=input("scale",string(session.state.scale))
    slider=_inspection_range(1:n;value=session.state.frame,id="$dom_id-slider",disabled,style=cs)
    apply=button("apply","Inspect selection");previous=button("previous","Previous state");next=button("next","Next state")
    previous_event=button("previous-event","Previous crossing");next_event=button("next-event","Next crossing")
    play=button("play","Play forward");reverse=button("reverse","Play backward");pause=button("pause","Pause")
    representative=B.Checkbox(false;id="$dom_id-representative",disabled)
    reset=button("reset","Reset");close_button=button("close","Close inspector")
    function stop()
        timer[]===nothing || close(timer[]);timer[]=nothing
        wasplaying=playing[];playing[]=false
        wasplaying && VIZ._cancel_inspection_requests!(session,owner)
        nothing
    end
    function status(stage,error)
        closed[] && return nothing
        error_text[]=stage===:error ? "Selection unchanged: $error" : ""
        stage===:error && stop()
        stage===:pending && (status_text[]="Selecting a retained state; the chart shows the last committed answer.")
        nothing
    end
    function request(;kwargs...)
        (closed[] || closing[]) && return false
        try
            _inspection_control_selection!(session,owner,browser_client[],status;kwargs...);true
        catch err
            err isa InterruptException && rethrow();error_text[]=sprint(showerror,err);stop();false
        end
    end
    function listen(f,observable;during_refresh=false)
        push!(observers,M.on(observable) do value
            (updating[] && !during_refresh) || closed[] || closing[] || f(value);nothing
        end)
    end
    function step(delta;events=false)
        stop();i=VIZ.inspection_selection(session).frame
        if events
            candidates=filter(j->delta>0 ? j>i : j<i,event_frames)
            isempty(candidates) && return (status_text[]="No further crossing in that direction.")
            request(;frame=delta>0 ? first(candidates) : last(candidates))
        else
            j=clamp(i+delta,1,n);j==i || request(;frame=j)
        end
    end
    function start(delta)
        stop();direction[]=delta;playing[]=true
        timer[]=Timer(0.7;interval=0.7) do _
            if closed[] || _inspection_browser_ended(browser_client[]) || VIZ.inspection_summary(session).closed
                stop();return
            end
            summary=VIZ.inspection_summary(session)
            (summary.updating||summary.requests.pending||summary.requests.running) && return
            j=summary.selection.frame+direction[]
            1<=j<=n ? request(;frame=j) : stop()
        end
        nothing
    end
    function slide(i)
        stop()
        # Browser transports can deliver a different numeric storage type.
        # Keep the public frame contract strict and convert at this boundary.
        if i isa Real && !(i isa Bool) && isinteger(i) && 1<=i<=n
            request(;frame=Int(i))
        else
            status(:error,"The slider must select an integer frame in 1:$n.")
        end
    end
    function nearby(point)
        best=nothing;distance=20.0^2
        for (i,ts) in enumerate(targets[])
            i<=length(axes[]) || continue
            ax=axes[][i];M.is_mouseinside(ax.scene) || continue
            for t in ts
                q=M.project(ax.scene,M.Point2d(t.point))+minimum(M.viewport(ax.scene)[])
                d=sum((point[k]-q[k])^2 for k in 1:2)
                if d<distance;best=t;distance=d;end
            end
        end
        best
    end
    function hover(point)
        t=nearby(point)
        hover_text[]=t===nothing ? "" : "Track $(t.id), state $(t.frame). Hover reads retained values only."
        t
    end
    function pick(point)
        t=hover(point);t===nothing && return false
        stop();request(;frame=t.frame,track=t.id)
    end
    function dispose(;close_session=false)
        closed[] && return nothing
        stop();closed[]=true;disabled[]=true;VIZ._cancel_inspection_requests!(session,owner)
        token[]==0 || VIZ._off_inspection!(session,token[]);token[]=0
        foreach(M.off,observers);empty!(observers);foreach(M.off,pointer_observers);empty!(pointer_observers)
        fig[]===nothing || empty!(fig[]);retained[]=nothing;empty!(axes[]);empty!(targets[]);browser_client[]=nothing
        content[]=B.DOM.p("Inspector closed. The final static snapshot remains available.")
        app=app_ref[].value;app===nothing || delete!(_INSPECTION_UI_HANDLES,app)
        close_session && VIZ.close_inspection!(session)
        nothing
    end
    function request_close()
        (closed[] || closing[]) && return nothing
        stop();closing[]=true;disabled[]=true
        VIZ._cancel_inspection_requests!(session)
        # A browser close can arrive while a refresh yields to its transport.
        # Stop playback immediately, then release the figure after that refresh.
        close_task[]=@async begin
            while updating[] || VIZ.inspection_summary(session).updating
                sleep(0.01)
            end
            VIZ.inspection_summary(session).closed || VIZ.close_inspection!(session)
            closed[] || dispose()
        end
        nothing
    end
    function refresh()
        (VIZ.inspection_summary(session).closed || _inspection_browser_ended(browser_client[])) && return dispose()
        snapshot=VIZ.inspection_snapshot(session);state=VIZ.inspection_selection(session)
        updating[]=true
        try
            hover_text[]="";error_text[]=""
            frame.value[]=string(state.frame);slider.value[]=state.frame;track.value[]=string(something(state.track,0))
            scale.value[]=string(state.scale);representative.value[]=state.representative
            status_text[]="State $(state.frame)/$n, path parameter $(VIZ._interval_value(snapshot.metadata.parameter)); $(snapshot.metadata.mode) history."
            readout[]=_inspection_dom_readout(VIZ.VisualizationSpec(:path_readout;
                panels=filter(p->get(p.metadata,:panel_style,nothing)===:text_only,snapshot.panels));style)
            panels=filter(p->get(p.metadata,:panel_style,nothing)!==:text_only,snapshot.panels)
            drawing=VIZ.VisualizationSpec(:path_drawing;panels,
                metadata=(;panel_columns=2,figure_size=(1120,400*cld(length(panels),2))))
            old=fig[]
            reused=retained[]!==nothing && _visual_update_retained!(M,VIZ,_HANDLERS,retained[],drawing;style,replace_plots=true)
            if reused
                update_count[]+=1
            else
                foreach(M.off,pointer_observers);empty!(pointer_observers)
                retained[]=_visual_retained_figure(M,VIZ,_HANDLERS,drawing;size,style)
                fig[]=retained[].figure;rebuild_count[]+=1
                axes[]=filter(x->x isa M.Axis,fig[].content)
                push!(pointer_observers,M.on(M.events(fig[].scene).mouseposition) do _
                    closed[] || hover(M.events(fig[].scene).mouseposition[]);nothing
                end)
                pressed=Ref{Any}(nothing)
                push!(pointer_observers,M.on(M.events(fig[].scene).mousebutton;priority=10) do e
                    if !closed[] && e.button==M.Mouse.left
                        if e.action==M.Mouse.press
                            pressed[]=M.events(fig[].scene).mouseposition[]
                        elseif e.action==M.Mouse.release && pressed[]!==nothing
                            point=M.events(fig[].scene).mouseposition[];a=pressed[];pressed[]=nothing
                            sum((point[k]-a[k])^2 for k in 1:2)<=25 && return M.Consume(pick(point))
                        end
                    end
                    M.Consume(false)
                end)
                content[]=fig[];old===nothing || empty!(old)
            end
            targets[]=[collect(get(p.metadata,:pick_targets,NamedTuple[])) for p in panels[1:4]]
        finally;updating[]=false;end
        nothing
    end
    integer(c)=let v=tryparse(Int,strip(c.value[]));v===nothing && throw(ArgumentError("Enter integer frame and track IDs."));v;end
    listen(apply.value) do _
        stop()
        try;request(;frame=integer(frame),track=integer(track),representative=representative.value[],scale=VIZ._parse_inspection_coordinate(scale.value[]))
        catch err;err isa InterruptException && rethrow();error_text[]=sprint(showerror,err);end
    end
    listen(slide,slider.value)
    listen(_->step(-1),previous.value);listen(_->step(1),next.value)
    listen(_->step(-1;events=true),previous_event.value);listen(_->step(1;events=true),next_event.value)
    listen(_->start(1),play.value);listen(_->start(-1),reverse.value);listen(_->stop(),pause.value;during_refresh=true)
    listen(reset.value) do _;stop();request(;session.initial...);end
    listen(_->request_close(),close_button.value;during_refresh=true)
    token[]=VIZ._on_inspection(session,_->refresh());refresh()
    label(text,control)=B.DOM.label(B.DOM.span(text;style="display:block;font-weight:600"),control;style="min-width:0;max-width:100%")
    row(children...)=B.DOM.div(children...;style="display:flex;flex-wrap:wrap;align-items:end;gap:$(style.gap)px;margin:$(style.gap)px 0")
    foreground=_inspection_css_color(style,VIZ._VisualRole(:foreground));background=_inspection_css_color(style,VIZ._VisualRole(:background))
    dom=B.DOM.div(B.DOM.h2("Persistence along a path"),
        B.DOM.p("Move through the retained states. Click a trajectory or diagram point to follow its track; exact frame and track IDs provide the same selection. Playback follows frame order, not elapsed physical time."),
        row(previous,play,reverse,pause,next,previous_event,next_event,reset,close_button),
        label("Path state (including exact crossings)",slider.node),
        row(label("Frame",frame),label("Track (0 clears)",track),(session.object.geometry===nothing ? B.DOM.span() : label("Filtration scale",scale)),label("Show retained representative",representative),apply),
        B.DOM.p(M.lift(x->x ? "Playing" : "Paused",playing);id="$dom_id-playing",var"aria-label"="Playback status"),
        B.DOM.p(error_text;id="$dom_id-error",role="alert"),B.DOM.p(status_text;id="$dom_id-status",role="status"),
        B.DOM.p(hover_text;id="$dom_id-hover",role="status"),
        B.DOM.div(content;id="$dom_id-chart",style="max-width:100%;overflow:auto",tabindex="0",var"aria-label"="Path trajectories and selected state"),
        B.DOM.div(readout;id="$dom_id-readout");id=dom_id,
        style="box-sizing:border-box;font-family:'$(style.font)',sans-serif;font-size:$(style.fontsize)px;line-height:1.45;color:$foreground;background:$background;padding:$(style.padding)px;max-width:100%;overflow-wrap:anywhere")
    app=B.App(;title="TamerOp persistence path") do client
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
    _INSPECTION_UI_HANDLES[app]=(;session,figure=fig,axes,targets,retained,
        controls=(;frame,track,scale,slider,apply,previous,next,previous_event,next_event,play,reverse,pause,representative,reset,close=close_button),
        callbacks=(;pick,hover,refresh,step,start,stop,slide),dispose,closed,disabled,observers,pointer_observers,session_token=token,browser_client,
        playing,timer,close_task,rebuild_count,update_count,error_text,status_text,hover_text,readout,dom_id,style)
    app
end
