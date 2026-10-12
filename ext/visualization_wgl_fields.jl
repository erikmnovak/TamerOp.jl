# A bounded retained-grid explorer. Every input goes through the session contract.
function _render_field_inspector(spec;display=:inline,figure=nothing,size=nothing,style=VIZ.VisualStyle())
    VIZ._check_visual_render_options(;display,figure,size,style)
    figure===nothing||throw(ArgumentError("A field inspector owns its canvas."))
    session=spec.metadata.session
    VIZ.inspection_summary(session).closed&&throw(ArgumentError("This field session is closed."))
    B,M=WGLMakie.Bonito,WGLMakie.Makie
    _NEXT_INSPECTION_UI_ID[]+=1;dom_id="tamerop-field-inspector-$(_NEXT_INSPECTION_UI_ID[])"
    fig=Ref{Any}(nothing);retained=Ref{Any}(nothing);rebuild_count=Ref(0);update_count=Ref(0);axes=Any[];closed=Ref(false);updating=Ref(false)
    disabled=M.Observable(false);error_text=M.Observable("");status_text=M.Observable("");hover_text=M.Observable("")
    content=M.Observable{Any}(B.DOM.div());observers=Any[];pointer_observers=Any[];session_token=Ref(0)
    browser_client=Ref{Any}(nothing);browser_lock=ReentrantLock();app_ref=Ref(WeakRef(nothing))
    f=first(session.fields);coordinate_axes=deepcopy(f.axes);axis_labels=f.axis_labels
    state=VIZ.inspection_selection(session);control_style=_inspection_control_style(style)
    sliders=[B.Slider(1:length(a);value=state.indices[d],id="$dom_id-axis-$d",disabled,style=control_style) for (d,a) in enumerate(coordinate_axes)]
    input(name,value)=B.TextField(value;id="$dom_id-$name",disabled,style=control_style)
    planes=input("planes","");layer=input("layer",state.layer);baseline=input("baseline","");contours=input("contours","")
    button(name,text)=B.Button(text;id="$dom_id-$name",disabled,style=control_style)
    apply=button("apply","Apply view");reset=button("reset","Reset");close_button=button("close","Close inspector")
    function commit(f)
        closed[]&&return false
        disabled[]=true
        try;f();error_text[]="";true
        catch err;err isa InterruptException&&rethrow();error_text[]=sprint(showerror,err);false
        finally;disabled[]=closed[];end
    end
    function listen(f,observable)
        push!(observers,M.on(observable) do value
            updating[]||closed[]||f(value);nothing
        end)
    end
    function nearest(point,panel)
        plane=panel.metadata.plane
        (argmin(abs.(coordinate_axes[plane[1]].-point[1])),argmin(abs.(coordinate_axes[plane[2]].-point[2])))
    end
    function hover(point,i)
        panel=VIZ.inspection_snapshot(session).panels[i];a,b=nearest(point,panel)
        values=only(filter(l->l isa VIZ.HeatmapLayer,panel.layers)).values
        hover_text[]="$(panel.title): ($(coordinate_axes[panel.metadata.plane[1]][a]), $(coordinate_axes[panel.metadata.plane[2]][b])); value=$(values[b,a]). Nearest stored sample."
        (a,b)
    end
    function pick(point,i)
        a,b=hover(point,i);plane=VIZ.inspection_snapshot(session).panels[i].metadata.plane
        indices=collect(VIZ.inspection_selection(session).indices);indices[plane[1]]=a;indices[plane[2]]=b
        commit(()->VIZ.select_inspection!(session;indices))
    end
    function dispose(;close_session=false)
        closed[]&&return nothing
        closed[]=true;disabled[]=true
        session_token[]==0||VIZ._off_inspection!(session,session_token[]);session_token[]=0
        foreach(M.off,observers);empty!(observers);foreach(M.off,pointer_observers);empty!(pointer_observers)
        fig[]===nothing||empty!(fig[]);retained[]=nothing;empty!(axes)
        content[]=B.DOM.p("Inspector closed. Its final static snapshot remains available.")
        lock(browser_lock) do;browser_client[]=nothing;end
        app=app_ref[].value;app===nothing||delete!(_INSPECTION_UI_HANDLES,app)
        close_session&&VIZ.close_inspection!(session);nothing
    end
    function refresh()
        VIZ.inspection_summary(session).closed&&return dispose()
        snapshot=VIZ.inspection_snapshot(session);state=VIZ.inspection_selection(session)
        updating[]=true
        try
            reused=retained[]!==nothing && _visual_update_retained!(M,VIZ,_HANDLERS,retained[],snapshot;style)
            old=fig[]
            if reused
                current=fig[];update_count[]+=1
            else
                foreach(M.off,pointer_observers);empty!(pointer_observers)
                retained[]=_visual_retained_figure(M,VIZ,_HANDLERS,snapshot;size,style)
                current=retained[].figure;fig[]=current;rebuild_count[]+=1
                empty!(axes);append!(axes,filter(b->b isa M.Axis,current.content))
            end
            # The public setter updates the widget index and its derived value.
            # Writing only .value leaves the browser's range thumb unchanged.
            for (d,slider) in enumerate(sliders);slider[]=state.indices[d];end
            planes.value[]=join((join(p,",") for p in state.planes),"; ")
            layer.value[]=state.layer;baseline.value[]=something(state.baseline,"");contours.value[]=join(state.contours,",")
            point=join(("$(axis_labels[d])=$(coordinate_axes[d][state.indices[d]])" for d in eachindex(coordinate_axes)),", ")
            status_text[]="Selected $point; layer $(state.layer). Shared $(snapshot.metadata.scale.normalization) range $(snapshot.metadata.scale.limits). Blank samples are unavailable."
            error_text[]="";hover_text[]=""
            if !reused
                content[]=current;old===nothing||empty!(old)
                push!(pointer_observers,M.on(M.events(current.scene).mouseposition) do _
                    closed[]&&return nothing
                    i=findfirst(ax->M.is_mouseinside(ax.scene),axes)
                    i===nothing ? (hover_text[]="") : hover(M.mouseposition(axes[i].scene),i);nothing
                end)
                push!(pointer_observers,M.on(M.events(current.scene).mousebutton;priority=10) do event
                    if !closed[]&&event.button==M.Mouse.left&&event.action==M.Mouse.press
                        i=findfirst(ax->M.is_mouseinside(ax.scene),axes)
                        i===nothing||return M.Consume(pick(M.mouseposition(axes[i].scene),i))
                    end
                    M.Consume(false)
                end)
            end
        finally;updating[]=false;end
        nothing
    end
    for (d,slider) in enumerate(sliders)
        listen(slider.value) do value
            indices=collect(VIZ.inspection_selection(session).indices);indices[d]=Int(value)
            commit(()->VIZ.select_inspection!(session;indices))
        end
    end
    listen(apply.value) do _
        commit() do
            ps=[Tuple(parse.(Int,strip.(split(p,',')))) for p in split(planes.value[],';')]
            cs=isempty(strip(contours.value[])) ? Float64[] : parse.(Float64,strip.(split(contours.value[],',')))
            base=isempty(strip(baseline.value[])) ? nothing : strip(baseline.value[])
            VIZ.select_inspection!(session;planes=ps,layer=strip(layer.value[]),baseline=base,contours=cs)
        end
    end
    listen(_->commit(()->VIZ.reset_inspection!(session)),reset.value)
    listen(_->commit(()->VIZ.close_inspection!(session)),close_button.value)
    session_token[]=VIZ._on_inspection(session,_ -> refresh());refresh()
    label(text,control)=B.DOM.label(B.DOM.span(text;style="display:block;font-weight:600"),control;style="display:block;min-width:0")
    row(children...)=B.DOM.div(children...;style="display:flex;flex-wrap:wrap;align-items:end;gap:$(style.gap)px;margin:$(style.gap)px 0")
    foreground=_inspection_css_color(style,VIZ._VisualRole(:foreground));background=_inspection_css_color(style,VIZ._VisualRole(:background))
    dom=B.DOM.div(B.DOM.h2("Explore coordinate sections"),
        B.DOM.p("Each panel varies two coordinates and fixes the others at the selected point. Click a chart to select its nearest stored sample. Julia must remain running."),
        row([label("Selected index on $(axis_labels[d])",slider) for (d,slider) in enumerate(sliders)]...),
        row(label("Ordered planes (1,2; 1,3)",planes),label("Layer ID",layer),label("Baseline ID (blank for values)",baseline),label("Contour levels (comma separated)",contours),apply),
        row(reset,close_button),B.DOM.p(error_text;id="$dom_id-error",role="alert"),
        B.DOM.p(status_text;id="$dom_id-status",role="status"),B.DOM.p(hover_text;id="$dom_id-hover",role="status",style="min-height:1.5em"),
        B.DOM.div(content;style="max-width:100%;overflow:auto",tabindex="0",var"aria-label"="Coordinate section charts");
        id=dom_id,style="font-family:'$(style.font)',sans-serif;font-size:$(style.fontsize)px;color:$foreground;background:$background;line-height:1.45;max-width:100%;padding:$(style.padding)px;box-sizing:border-box")
    app=B.App(;title="TamerOp coordinate sections") do client
        lock(browser_lock) do
            closed[]&&return B.DOM.p("This field viewer has been closed.")
            if browser_client[]!==nothing&&browser_client[]!==client
                existing=app_ref[].value;existing===nothing||(existing.session[]=browser_client[])
                return B.DOM.p("This widget already has a browser client. Render the session again for a linked view.")
            end
            if browser_client[]===nothing
                browser_client[]=client;_inspection_keyboard_navigation!(client,dom)
                push!(observers,M.on(client.on_close) do ended;ended&&dispose();nothing;end)
            end
            dom
        end
    end
    app_ref[]=WeakRef(app)
    _INSPECTION_UI_HANDLES[app]=(;session,figure=fig,retained,rebuild_count,update_count,axes,controls=(;sliders,planes,layer,baseline,contours,apply,reset,close=close_button),
        closed,disabled,observers,pointer_observers,session_token,dispose,browser_client,dom_id,error_text,status_text,hover_text)
    app
end
