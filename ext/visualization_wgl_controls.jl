# Native scene gestures and keyboard sliders share the same mathematical
# request path. Pointer coordinates are explicit decimal approximations.
function _inspection_drag_handles!(axis, handles, move, closed;start=()->nothing)
    M=WGLMakie.Makie
    dragging=Ref(0)
    observers=Any[]
    push!(observers,M.on(M.events(axis.scene).mousebutton;priority=300) do event
        closed[] && return M.Consume(false)
        event.button==M.Mouse.left || return M.Consume(false)
        if event.action==M.Mouse.press && M.is_mouseinside(axis.scene)
            point=M.mouseposition(axis.scene)
            spans=M.widths(axis.finallimits[]);pixels=M.widths(axis.scene.viewport[])
            points=handles()
            distances=[sum(((p[d]-point[d])*pixels[d]/spans[d])^2 for d in 1:2) for p in points]
            if !isempty(distances) && minimum(distances)<=24^2
                dragging[]=argmin(distances)
                start()
                return M.Consume(true)
            end
        elseif event.action==M.Mouse.release && dragging[]!=0
            id=dragging[];dragging[]=0
            move(id,M.mouseposition(axis.scene),true)
            return M.Consume(true)
        end
        M.Consume(false)
    end)
    push!(observers,M.on(M.events(axis.scene).mouseposition;priority=300) do _
        if !closed[] && dragging[]!=0
            move(dragging[],M.mouseposition(axis.scene),false)
            return M.Consume(true)
        end
        M.Consume(false)
    end)
    (;observers,dragging)
end

_inspection_pointer_coordinate(value)=VIZ._parse_inspection_coordinate(string(round(Float64(value);digits=10)))

# WGL plot insertion can wait for a browser acknowledgement. A connected
# control must release Bonito's incoming-message task before that drawing work.
# Direct Julia calls and native, unconnected UI checks remain synchronous.
function _inspection_control_selection!(session,owner,client,status;kwargs...)
    if client!==nothing && isready(client)
        VIZ._request_inspection!(session,owner;status,kwargs...)
        return :queued
    end
    VIZ.select_inspection!(session;kwargs...)
    return :committed
end

function _inspection_browser_ended(client)
    client===nothing && return false
    B=WGLMakie.Bonito
    root=B.root_session(client)
    # Transport shutdown can precede the App's on_close notification. Avoid
    # inserting plots into a child screen that Bonito has already released.
    B.isclosed(client) || B.isclosed(root) || root.status===B.SOFT_CLOSED
end

# A server refresh updates the thumb without echoing a synthetic user request.
# Bonito's two-observable Slider also sends index->value updates from the
# browser, which can otherwise replace an exact event with a rounded tick.
function _inspection_range(values;value=first(values),style=nothing,kwargs...)
    B=WGLMakie.Bonito
    selected=WGLMakie.Makie.Observable(Int(value))
    width=B.Styles("width"=>"100%")
    range_style=style===nothing ? width : B.Styles(style,width)
    node=B.DOM.input(;type="range",min=first(values),max=last(values),step=1,value=selected,style=range_style,
        oninput=WGLMakie.Bonito.js"""event => $(selected).notify(event.target.valueAsNumber)""",kwargs...)
    (;node,value=selected,values)
end

function _inspection_parameter_ui(session;style,disabled,closed,updating,listen,commit,dom_id)
    B,M=WGLMakie.Bonito,WGLMakie.Makie
    controls=VIZ._filtration_controls(session.object,VIZ.inspection_selection(session).parameter)
    owner=Ref(nothing)
    message=M.Observable("Controls query the retained grades. Exact coordinates remain available below.")
    cs=_inspection_control_style(style)
    sliders=Any[];rows=Any[];observers=Any[]
    requested=Ref{Any}(VIZ.inspection_selection(session).parameter)
    plane=Ref{Any}(nothing);axis=Ref{Any}(nothing);cursor=M.Observable(M.Point2d[])
    geometric=session.options.kind===:geometric_filtration
    function status(stage,error)
        closed[] && return nothing
        stage===:error && refresh()
        message[]=stage===:pending ? "Requested $(requested[]). The chart shows the last committed parameter while this query is pending." :
            stage===:error ? "Parameter unchanged: $error" : "Committed parameter $(VIZ.inspection_selection(session).parameter)."
        nothing
    end
    function request(values)
        closed[] && return nothing
        try
            p=VIZ._filtration_control_parameter(controls,values)
            requested[]=p
            VIZ._request_inspection!(session,owner;status,parameter=p)
        catch err
            err isa InterruptException && rethrow()
            VIZ._cancel_inspection_requests!(session,owner)
            refresh()
            message[]="Parameter unchanged: $(sprint(showerror,err))"
        end
        nothing
    end
    for (d,c) in enumerate(controls)
        direction=c.orientation==1 ? "sublevel ≤" : "superlevel ≥"
        if !c.drawable
            push!(sliders,nothing)
            push!(rows,B.DOM.p("$(c.label): use exact entry; the range cannot be drawn faithfully."))
            continue
        end
        if c.lower==c.upper
            push!(sliders,nothing)
            push!(rows,B.DOM.p("$(c.label) = $(c.lower) (fixed construction window)."))
            continue
        end
        # The range input stores bounded integer ticks, not a serialized array
        # proportional to an arbitrarily large number of grade values.
        ticks=c.discrete ? (0:Int(c.upper-c.lower)) : (0:1000)
        slider=_inspection_range(ticks;value=first(ticks),id="$dom_id-grade-$d",disabled,style=cs)
        push!(sliders,slider)
        listen(slider.value) do tick
            value=c.discrete ? c.lower+tick : c.lower+(c.upper-c.lower)*(tick//1000)
            request(ntuple(i->i==d ? value : requested[][i],length(controls)))
        end
        previous=B.Button("Previous event";id="$dom_id-grade-$d-previous",disabled,style=cs)
        next=B.Button("Next event";id="$dom_id-grade-$d-next",disabled,style=cs)
        for (button,step) in ((previous,-1),(next,1))
            listen(button.value) do _
                p=VIZ._filtration_event_parameter(controls,VIZ.inspection_selection(session).parameter,d,step)
                p===nothing ? (message[]="No further stored event on $(c.label).") : request(p)
            end
        end
        label=B.DOM.label(B.DOM.span("$(c.label) ($direction)";style="display:block;font-weight:600"),slider.node;
            style="display:block;min-width:12rem;max-width:100%;flex:0 1 32rem")
        push!(rows,B.DOM.div(label,previous,next;style="display:flex;flex-wrap:wrap;gap:$(style.gap)px;align-items:end;margin:$(style.gap)px 0"))
    end
    function move(_,point,final)
        values=ntuple(length(controls)) do d
            c=controls[d]
            c.lower==c.upper ? c.lower : c.discrete ? round(Int,point[d]) : _inspection_pointer_coordinate(point[d])
        end
        request(values)
    end
    function bind_plane(f,ax)
        axis[]===ax && return nothing
        foreach(M.off,observers);empty!(observers)
        geometric && M.scatter!(ax,cursor;marker=:rect,markersize=18*style.markersize_scale,
            color=VIZ._visual_color(style,VIZ._VisualRole(:selected)),depth_shift=-0.004)
        drag=_inspection_drag_handles!(ax,()->cursor[],move,closed)
        append!(observers,drag.observers)
        plane[]=f;axis[]=ax
        nothing
    end
    if geometric
        push!(rows,B.DOM.p("The parameter plot below also has a draggable square. Pointer values are rounded to ten decimal places; depth snaps to integers."))
    elseif length(controls)==2 && all(c->c.drawable && c.lower<c.upper,controls)
        f=M.Figure(size=(560,320),figure_padding=style.padding)
        ax=M.Axis(f[1,1];xlabel=controls[1].label,ylabel=controls[2].label,
            xlabelsize=style.fontsize,ylabelsize=style.fontsize)
        lo=Float64.((controls[1].lower,controls[2].lower))
        hi=Float64.((controls[1].upper,controls[2].upper))
        padding=0.06 .* (hi.-lo)
        M.limits!(ax,lo[1]-padding[1],hi[1]+padding[1],lo[2]-padding[2],hi[2]+padding[2])
        M.lines!(ax,M.Point2d[(lo[1],lo[2]),(hi[1],lo[2]),(hi[1],hi[2]),(lo[1],hi[2]),(lo[1],lo[2])];
            color=VIZ._visual_color(style,VIZ._VisualRole(:border)),linestyle=:dash,linewidth=style.linewidth_scale)
        M.scatter!(ax,cursor;marker=:rect,markersize=18,color=VIZ._visual_color(style,VIZ._VisualRole(:selected)))
        _HANDLERS.style_figure(f,style)
        bind_plane(f,ax)
        push!(rows,B.DOM.p("Drag the square within the outlined control window, or use the labelled sliders and exact fields. Pointer positions are rounded to ten decimal places; discrete coordinates snap to integers."))
        push!(rows,B.DOM.div(f;style="max-width:100%;overflow:auto",tabindex="0",var"aria-label"="Filtration parameter plane"))
    end
    function refresh()
        p=VIZ.inspection_selection(session).parameter
        requested[]=p
        message[]="Committed parameter $p. Controls query the retained grades; exact coordinates remain available below."
        was=updating[];updating[]=true
        try
            for (d,(c,slider)) in enumerate(zip(controls,sliders))
                slider===nothing && continue
                tick=c.discrete ? round(Int,p[d]-c.lower) : round(Int,1000*Float64((p[d]-c.lower)/(c.upper-c.lower)))
                slider.value[]=clamp(tick,first(slider.values),last(slider.values))
            end
            length(controls)==2 && (cursor[]=[M.Point2d(Float64.(p))])
        finally;updating[]=was;end
        nothing
    end
    function dispose()
        VIZ._cancel_inspection_requests!(session,owner)
        foreach(M.off,observers);empty!(observers)
        geometric || plane[]===nothing || empty!(plane[])
        nothing
    end
    dom=B.DOM.section(rows...,B.DOM.p(message;id="$dom_id-continuous-status",role="status"))
    (;dom,controls,sliders,refresh,dispose,plane,axis,cursor,request,message,owner,bind_plane)
end
