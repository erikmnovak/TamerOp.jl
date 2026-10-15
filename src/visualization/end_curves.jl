# Exact lattice curves and Betti-linked inspection; algebra belongs to EndCurves.
mutable struct EndCurveInspectionSession <: _AbstractInspectionSession
    object::Union{Nothing,EndCurves.EndCurveResult}
    initial::NamedTuple
    state::NamedTuple
    snapshot::VisualizationSpec
    snapshot_builds::Int
    listeners::Dict{Int,Any}
    next_listener::Int
    listener_errors::Vector{String}
    closed::Bool
    busy::Bool
    lock::ReentrantLock
    requests::_InspectionRequests
end

function _end_curve_selection(r;kind=:birth,curve=0,point=r.origin,betti_degree=0,show_betti=true)
    kind in (:birth,:death) || throw(ArgumentError("kind must be :birth or :death"))
    cs=kind===:birth ? r.births : r.deaths
    curve isa Integer && !(curve isa Bool) && 0<=curve<=length(cs) || throw(ArgumentError("curve must be in 0:$(length(cs)); zero shows all curves"))
    betti_degree isa Integer && !(betti_degree isa Bool) && 0<=betti_degree<=2 || throw(ArgumentError("Betti degree must be 0, 1 or 2"))
    show_betti isa Bool || throw(ArgumentError("show_betti must be true or false"))
    q=EndCurves.end_curve_at(r;point)
    (;kind,curve=Int(curve),point=q.point,betti_degree=Int(betti_degree),show_betti)
end
function _end_curve_display_budget(r;max_curves=200,max_points=4000)
    for (v,name) in ((max_curves,"max_curves"),(max_points,"max_points"))
        v isa Integer && !(v isa Bool) && v>=0 || throw(ArgumentError("$name must be a nonnegative integer"))
    end
    length(r.births)+length(r.deaths)<=max_curves || throw(ArgumentError("end-curve display curve budget exceeded"))
    big(r.shape[1])+r.shape[2]+2<=max_points || throw(ArgumentError("end-curve display lattice-axis budget exceeded"))
    sum(c->length(c.vertices),r.births;init=0)+sum(c->length(c.vertices),r.deaths;init=0)<=max_points ||
        throw(ArgumentError("end-curve display point budget exceeded"))
    nothing
end
function _end_curve_snapshot(r,p)
    nx,ny=r.shape;ox,oy=r.origin
    xs=Float64.(ox:ox+nx);ys=Float64.(oy:oy+ny)
    all(diff(xs).>0) && all(diff(ys).>0) || throw(ArgumentError("lattice coordinates cannot be distinguished in Float64 drawing coordinates"))
    panels=VisualizationSpec[]
    for (kind,curves,color) in ((:birth,r.births,1),(:death,r.deaths,2))
        layers=AbstractVisualizationLayer[];targets=NamedTuple[]
        labels=Dict{Tuple{Float64,Float64},Vector{String}}()
        # The solid rectangle encloses the input lattice, the dashed lines its zero halo.
        outline=[(xs[1],ys[1],xs[end-1],ys[1]),(xs[end-1],ys[1],xs[end-1],ys[end-1]),
            (xs[end-1],ys[end-1],xs[1],ys[end-1]),(xs[1],ys[end-1],xs[1],ys[1])]
        push!(layers,SegmentLayer(outline,_VisualRole(:foreground),0.25,1.))
        push!(layers,SegmentLayer([(xs[end],ys[1],xs[end],ys[end]),(xs[1],ys[end],xs[end],ys[end])],_VisualRole(:foreground),0.3,1.,:dash))
        for (k,c) in enumerate(curves)
            ps=Tuple{Float64,Float64}[(Float64(x),Float64(y)) for (x,y) in c.vertices]
            selected=p.kind===kind && p.curve==k;role=selected ? _VisualRole(:selected) : _VisualRole(:categorical,color)
            alpha=p.curve==0 || p.kind!==kind || selected ? 1. : 0.25
            segments=[(ps[i]...,ps[i+1]...) for i in 1:length(ps)-1]
            push!(layers,SegmentLayer(segments,role,alpha,selected ? 5.0 : 2.0 + log2(c.multiplicity)))
            push!(layers,PointLayer(ps,role,alpha,selected ? 11. : 7.))
            center=ps[cld(length(ps),2)];prefix=kind===:birth ? "B" : "D"
            if length(curves)<=12 || selected
                push!(get!(Vector{String},labels,center),"$prefix$k"*(c.multiplicity==1 ? "" : " \u00d7 $(c.multiplicity)"))
            end
            for (i,v) in enumerate(c.vertices)
                push!(targets,(;point=ps[i],segment=(ps[i]...,ps[i]...),selection=(;kind,curve=k,point=v),label="$prefix$k at $v; multiplicity $(c.multiplicity)"))
            end
            for i in 1:length(ps)-1
                push!(targets,(;point=((ps[i][1]+ps[i+1][1])/2,(ps[i][2]+ps[i+1][2])/2),segment=segments[i],
                    selection=(;kind,curve=k,point=c.vertices[i]),label="$prefix$k: $(c.vertices[i]) to $(c.vertices[i+1]); multiplicity $(c.multiplicity)"))
            end
        end
        # Merge coincident labels without moving the mathematical curve coordinates.
        for (center,ids) in sort!(collect(labels);by=first)
            push!(layers,TextLayer([join(ids,", ")],[(center[1]+0.12,center[2]+0.17)],_VisualRole(:foreground),11.))
        end
        if p.show_betti && r.betti!==nothing
            for j in 1:ny+1,i in 1:nx+1
                v=r.betti[i,j,p.betti_degree+1];v==0 && continue
                x,y=xs[i],ys[j];role=_VisualRole(:foreground)
                push!(layers,TextLayer(["\u03b2$(p.betti_degree)=$v"],[(x-0.16,y-0.22)],role,10.))
                push!(targets,(;point=(x,y),segment=(x,y,x,y),selection=(;point=(ox+i-1,oy+j-1)),label="\u03b2$(p.betti_degree)($((ox+i-1,oy+j-1))) = $v"))
            end
        end
        x,y=Float64.(p.point)
        push!(layers,SegmentLayer([(x-0.1,y,x+0.1,y),(x,y-0.1,x,y+0.1)],_VisualRole(:selected),1.,2.))
        title=kind===:birth ? "Birth curves" : "Death curves"
        subtitle=kind===:birth ? "Cokernel of xy; multiplicities count copies" : "Kernel of xy in shifted grades; zero extension"
        push!(panels,VisualizationSpec(:end_curve_lattice;title,subtitle,layers,
            axes=(;xlabel="Lattice x (module order)",ylabel="Lattice y (module order)",aspect=:equal,
                xlimits=(xs[1]-0.6,xs[end]+0.6),ylimits=(ys[1]-0.6,ys[end]+0.6),
                xticks=(xs,string.(ox:ox+nx)),yticks=(ys,string.(oy:oy+ny))),
            metadata=(;kind,curves=[(;vertices=copy(c.vertices),multiplicity=c.multiplicity) for c in curves],
                pick_targets=targets,extension=:zero,origin=r.origin,shape=r.shape)))
    end
    at=EndCurves.end_curve_at(r;point=p.point);cs=p.kind===:birth ? r.births : r.deaths
    selected=p.curve==0 ? nothing : cs[p.curve]
    support = if selected===nothing
        ""
    elseif length(selected.vertices)<=8
        join(string.(selected.vertices), " \u2192 ")
    else
        "$(first(selected.vertices)) \u2192 \u2026 \u2192 $(last(selected.vertices)) ($(length(selected.vertices)) lattice points; full support in exported values)"
    end
    terms=r.count_terms
    lines=["Two-parameter count: $(EndCurves.two_parameter_count(r)). Both curve multisets have this many copies.",
        "Selected lattice point: $(p.point)$(at.zero_halo ? " (zero halo)" : ""). Birth curves: $(Tuple(at.births)); death curves: $(Tuple(at.deaths)).",
        selected===nothing ? "Select a curve to inspect its exact support and multiplicity." :
            "Selected $(p.kind) curve $(p.curve), multiplicity $(selected.multiplicity): $support.",
        at.betti===nothing ? "No Betti data were supplied." : "Source Betti numbers at this point: \u03b20=$(at.betti[1]), \u03b21=$(at.betti[2]), \u03b22=$(at.betti[3]).",
        "Betti labels refer to the zero-extended source module. Curve corners alone do not identify its presentation.",
        "Solid outline: input lattice. Dashed upper/right lines: added zero halo. All curve vertices are included lattice points.",
        "$(r.scope===:computed ? "Computed over $(_inspection_field_label(r.field))" : "Supplied curve data; source realization not certified"). Source axis orientation: $(r.orientation).",
        "Birth and death curves are separate multisets. Colors do not pair them into boundary components."]
    terms===nothing || pushfirst!(lines,"Count = $(terms.dimension) \u2212 $(terms.x_rank) \u2212 $(terms.y_rank) + $(terms.xy_rank), from dim M, rank x, rank y and rank xy.")
    push!(panels,_inspection_text_panel("Count, support and Betti data",lines))
    VisualizationSpec(:end_curves;title="Counts and end-curves",panels,
        metadata=(;selection=p,count=EndCurves.two_parameter_count(r),point=at,scope=r.scope,extension=:zero,
            betti=r.betti===nothing ? nothing : copy(r.betti),count_terms=terms,
            panel_positions=[(1:1,1:1),(1:1,2:2),(2:2,1:2)],panel_row_weights=[550.,320.],figure_size=(1250,870)))
end

available_visuals(::EndCurves.EndCurveResult)=(:end_curves,)
_visual_request_keywords(::EndCurves.EndCurveResult,::Symbol)=(:kind,:curve,:point,:betti_degree,:show_betti,:max_curves,:max_points)
function _visual_spec(r::EndCurves.EndCurveResult,::Symbol;max_curves=200,max_points=4000,kwargs...)
    _end_curve_display_budget(r;max_curves,max_points)
    _end_curve_snapshot(r,_end_curve_selection(r;kwargs...))
end
"""Explore exact birth/death supports, multiplicities and source Betti numbers.
Live pointer selection has exact curve/point inputs as a keyboard alternative.
Saved HTML retains explicitly prepared selections without requiring Julia.
"""
function end_curve_explorer(r::EndCurves.EndCurveResult;max_curves=200,max_points=4000,kwargs...)
    _end_curve_display_budget(r;max_curves,max_points)
    # The viewer owns only displayed mathematical data, not source caches or splittings.
    owned=EndCurves.EndCurveResult(EndCurves.birth_curves(r),EndCurves.death_curves(r),r.shape,r.origin,r.field,
        r.betti===nothing ? nothing : copy(r.betti),r.count_terms,nothing,r.scope,r.orientation)
    EndCurves.check_end_curves(owned;throw=true)
    p=_end_curve_selection(owned;kwargs...);state=merge(p,(;revision=0));snap=_end_curve_snapshot(owned,state)
    EndCurveInspectionSession(owned,p,state,snap,1,Dict{Int,Any}(),1,String[],false,false,ReentrantLock(),_InspectionRequests())
end

inspection_selection(s::EndCurveInspectionSession)=lock(s.lock) do;s.state;end
inspection_snapshot(s::EndCurveInspectionSession)=lock(s.lock) do;s.snapshot;end
function inspection_summary(s::EndCurveInspectionSession)
    lock(s.lock) do
        (;kind=:end_curve_explorer,selection=s.state,revision=s.state.revision,closed=s.closed,updating=s.busy,
            snapshot_builds=s.snapshot_builds,listener_count=length(s.listeners),listener_errors=Tuple(s.listener_errors),requests=_inspection_request_summary(s))
    end
end
describe(s::EndCurveInspectionSession)=inspection_summary(s)
Base.show(io::IO,s::EndCurveInspectionSession)=print(io,"EndCurveInspectionSession(curve=",s.state.curve,", closed=",s.closed,")")
function _end_curve_candidate(s;kwargs...)
    _inspection_require_open(s)
    all(k->k in keys(s.initial),keys(kwargs)) || throw(ArgumentError("Select kind, curve, point, betti_degree or show_betti."))
    p=merge(Base.structdiff(s.state,(;revision=s.state.revision)),(;kwargs...))
    merge(_end_curve_selection(s.object;p...),(;revision=s.state.revision+1))
end
function check_inspection_selection(s::EndCurveInspectionSession;throw=false,kwargs...)
    issues=String[]
    lock(s.lock) do
        try;_end_curve_candidate(s;kwargs...)
        catch err;err isa InterruptException && rethrow();push!(issues,sprint(showerror,err));end
    end
    valid=isempty(issues);throw && !valid && _throw_invalid_visual(:check_inspection_selection,issues)
    (;kind=:inspection_selection,valid,issues)
end
function select_inspection!(s::EndCurveInspectionSession;kwargs...)
    _cancel_inspection_requests!(s)
    lock(s.lock) do
        s.busy && throw(ArgumentError("An end-curve update is in progress."))
        p=_end_curve_candidate(s;kwargs...);snap=_end_curve_snapshot(s.object,p)
        s.state=p;s.snapshot=snap;s.snapshot_builds+=1;s.busy=true
    end
    try;_inspection_notify(s)
    finally;lock(s.lock) do;s.busy=false;end;end
    s
end
reset_inspection!(s::EndCurveInspectionSession)=select_inspection!(s;s.initial...)
function close_inspection!(s::EndCurveInspectionSession)
    _cancel_inspection_requests!(s)
    lock(s.lock) do
        s.closed && return s
        s.busy && throw(ArgumentError("Wait for the current end-curve update before closing."))
        s.closed=true;s.object=nothing
    end
    try;_inspection_notify(s)
    finally;lock(s.lock) do;empty!(s.listeners);end;end
    s
end
function check_inspection_session(s::EndCurveInspectionSession;throw=false)
    issues=String[]
    lock(s.lock) do
        s.snapshot.metadata.selection==s.state || push!(issues,"Snapshot selection mismatch.")
        s.closed && (s.object!==nothing || !isempty(s.listeners)) && push!(issues,"Closed explorer retains input or listeners.")
        append!(issues,check_visual_spec(s.snapshot).issues)
    end
    valid=isempty(issues);throw && !valid && _throw_invalid_visual(:check_inspection_session,issues)
    (;kind=:inspection_session,valid,issues)
end
function _visual_spec(s::EndCurveInspectionSession,::Symbol;kwargs...)
    isempty(kwargs) || throw(ArgumentError("Use select_inspection! to change the end-curve selection."))
    VisualizationSpec(:linked_inspector;title="Counts and end-curves",panels=s.snapshot.panels,
        interaction=(;hover=true,clicks=true,labels=true,widgets=(:end_curve_selection,),notebook=:linked_inspector,
            mode=:live_julia,requires_live_julia=true,offline_widgets=false),metadata=(;session=s,inspector_kind=:end_curves))
end
_visual_request_cost(::EndCurveInspectionSession,::Symbol)=(;work=:retained_end_curves,hover=:read_only,requires_live_julia=true,timing=:not_measured)
