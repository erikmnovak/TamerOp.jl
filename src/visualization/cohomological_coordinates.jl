# Linked source/coordinate selections reuse checked lifts and harmonic solutions.
mutable struct CoordinateInspectionSession <: _AbstractInspectionSession
    object::Union{Nothing,CohomologicalCoordinates.CohomologicalCoordinateResult}
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

function _coordinate_selection(r;coordinate=1,vertex=first(r.vertices),phase=0.,source_axes=(1,2))
    coordinate isa Integer && !(coordinate isa Bool) && 1<=coordinate<=size(r.values,2) || throw(ArgumentError("Select an available coordinate."))
    vertex isa Integer && !(vertex isa Bool) && vertex in r.vertices || throw(ArgumentError("Select an active source vertex."))
    phase isa Real && isfinite(phase) || throw(ArgumentError("Phase must be finite turns."))
    D=r.source.provenance.ambient_dim
    length(source_axes)==2 && allunique(source_axes) && all(i->i isa Integer && !(i isa Bool) && 1<=i<=D,source_axes) ||
        throw(ArgumentError("source_axes must select two distinct source coordinates."))
    (;coordinate=Int(coordinate),vertex=Int(vertex),phase=mod(Float64(phase),1.),source_axes=Tuple(Int.(source_axes)))
end

function _coordinate_snapshot(r,p)
    length(r.vertices)<=5000 && length(r.edges)<=20_000 || throw(ArgumentError("Coordinate display budget is 5,000 vertices and 20,000 edges."))
    C=CohomologicalCoordinates;j=p.coordinate;idx=findfirst(==(p.vertex),r.vertices)
    phase=mod.(r.values[:,j].+p.phase,1.)
    xy=Tuple{Float64,Float64}[(Float64(r.source.geometry[v][1][p.source_axes[1]]),Float64(r.source.geometry[v][1][p.source_axes[2]])) for v in r.vertices]
    all(q->all(isfinite,q),xy) || throw(ArgumentError("Source drawing coordinates overflow Float64."))
    circle=Tuple{Float64,Float64}[(cospi(2t),sinpi(2t)) for t in phase]
    panels=VisualizationSpec[]
    edges=NTuple{4,Float64}[]
    for vs in C._edge_vertices(r.delta0)
        length(vs)==2 || continue
        push!(edges,(xy[vs[1]]...,xy[vs[2]]...))
    end
    source_layers=AbstractVisualizationLayer[SegmentLayer(edges,_VisualRole(:foreground),0.18,1.),
        PointLayer(xy,phase,1.,12.,:hsv),PointLayer([xy[idx]],_VisualRole(:selected),1.,19.)]
    targets(points)=[(;point=points[i],segment=(points[i]...,points[i]...),selection=(;vertex=v),label="Vertex $v; coordinate $j = $(r.values[i,j]) turns") for (i,v) in enumerate(r.vertices)]
    shared=(;point_colorrange=(0.,1.),point_colorbar_label="Phase (turns)")
    push!(panels,VisualizationSpec(:coordinate_source;title="Source colored by coordinate $j",
        subtitle="Source axes $(p.source_axes); scale $(r.scale)",layers=source_layers,
        axes=(;xlabel="Source coordinate $(p.source_axes[1])",ylabel="Source coordinate $(p.source_axes[2])",aspect=:equal,
            xlimits=_pad_limits_1d(extrema(first.(xy))),ylimits=_pad_limits_1d(extrema(last.(xy)))),
        metadata=merge(shared,(;pick_targets=targets(xy),source_axes=p.source_axes,vertices=copy(r.vertices),values=copy(phase)))))
    rim=[(cospi(t),sinpi(t)) for t in range(0,2;length=129)]
    ring=[(rim[i]...,rim[i+1]...) for i in 1:length(rim)-1]
    push!(panels,VisualizationSpec(:coordinate_circle;title="Circle coordinate",subtitle="One turn identifies 0 and 1",
        layers=AbstractVisualizationLayer[SegmentLayer(ring,_VisualRole(:foreground),0.3,1.),
            PointLayer(circle,phase,1.,11.,:hsv),PointLayer([circle[idx]],_VisualRole(:selected),1.,19.),
            SegmentLayer([(0.,0.,circle[idx]...)],_VisualRole(:selected),1.,2.)],
        axes=(;xlabel="cos(2\u03c0\u03b8)",ylabel="sin(2\u03c0\u03b8)",aspect=:equal,xlimits=(-1.2,1.2),ylimits=(-1.2,1.2)),
        metadata=merge(shared,(;pick_targets=targets(circle),coordinate=j,values=copy(phase)))))
    if size(r.values,2)>1
        other=j==size(r.values,2) ? 1 : j+1
        square=Tuple{Float64,Float64}[(phase[i],r.values[i,other]) for i in eachindex(phase)]
        push!(panels,VisualizationSpec(:coordinate_torus;title="Two circle factors",subtitle="Opposite sides are identified; selected basis",
            layers=AbstractVisualizationLayer[PointLayer(square,phase,1.,11.,:hsv),PointLayer([square[idx]],_VisualRole(:selected),1.,19.)],
            axes=(;xlabel="Coordinate $j (turns)",ylabel="Coordinate $other (turns)",aspect=:equal,xlimits=(-0.04,1.04),ylimits=(-0.04,1.04)),
            metadata=merge(shared,(;pick_targets=targets(square),coordinates=(j,other)))))
    end
    incident=findall(!iszero,r.delta0[:,idx]);sel=r.classes[j]
    lines=["Selected source vertex $(p.vertex), component $(r.components[idx]); coordinate $j.",
        "Stored phase: $(r.values[idx,j]) turns. Display rotation: $(p.phase) turns.",
        "Class: $(sel.kind) $(sel.index); interval $(sel.interval). Field: $(_inspection_field_label(r.field)).",
        "Scale $(r.scale); $(r.provenance.source_order). Integer lift: $(r.provenance.lift).",
        "Weighted energy: $(C.coordinate_summary(r).energy[j]). Anchors: $(Tuple(r.anchors)).",
        "Components have independent phase origins. The selected circle factors need not embed the source."]
    for e in Iterators.take(incident,6)
        push!(lines,"Edge $(r.edges[e]): residue $(r.cocycles[e,j]), integer lift $(r.lifts[e,j]), harmonic value $(r.harmonic[e,j]).")
    end
    length(incident)>6 && push!(lines,"$(length(incident)-6) further incident edges retained in the result.")
    push!(panels,_inspection_text_panel("Class and coordinate",lines))
    nplots=length(panels)-1
    readout_height = 100.0 + 30 * length(lines)
    VisualizationSpec(:cohomological_coordinates;title="Cohomological coordinates",panels,
        metadata=(;selection=p,vertices=copy(r.vertices),values=copy(r.values),cocycles=copy(r.cocycles),
            lifts=copy(r.lifts),harmonic=copy(r.harmonic),weights=copy(r.weights),provenance=r.provenance,
            field=string(r.field),scale=r.scale,classes=copy(r.classes),anchors=copy(r.anchors),
            panel_positions=[[(1:1,i:i) for i in 1:nplots];[(2:2,1:nplots)]],
            panel_row_weights=[500.,readout_height],figure_size=(600*nplots,ceil(Int,550+readout_height))))
end

available_visuals(::CohomologicalCoordinates.CohomologicalCoordinateResult)=(:cohomological_coordinates,)
_visual_request_keywords(::CohomologicalCoordinates.CohomologicalCoordinateResult,::Symbol)=(:coordinate,:vertex,:phase,:source_axes)
function _visual_spec(r::CohomologicalCoordinates.CohomologicalCoordinateResult,::Symbol;kwargs...)
    CohomologicalCoordinates.check_coordinates(r;throw=true)
    _coordinate_snapshot(r,_coordinate_selection(r;kwargs...))
end
"""Link source vertices, selected circle factors and checked cocycle readouts."""
function coordinate_explorer(r::CohomologicalCoordinates.CohomologicalCoordinateResult;kwargs...)
    CohomologicalCoordinates.check_coordinates(r;throw=true)
    p=_coordinate_selection(r;kwargs...);state=merge(p,(;revision=0))
    CoordinateInspectionSession(r,p,state,_coordinate_snapshot(r,state),1,Dict{Int,Any}(),1,String[],false,false,ReentrantLock(),_InspectionRequests())
end

inspection_selection(s::CoordinateInspectionSession)=lock(s.lock) do;s.state;end
inspection_snapshot(s::CoordinateInspectionSession)=lock(s.lock) do;s.snapshot;end
function inspection_summary(s::CoordinateInspectionSession)
    lock(s.lock) do
        (;kind=:coordinate_explorer,selection=s.state,revision=s.state.revision,closed=s.closed,updating=s.busy,
            snapshot_builds=s.snapshot_builds,listener_count=length(s.listeners),listener_errors=Tuple(s.listener_errors),requests=_inspection_request_summary(s))
    end
end
describe(s::CoordinateInspectionSession)=inspection_summary(s)
Base.show(io::IO,s::CoordinateInspectionSession)=print(io,"CoordinateInspectionSession(vertex=",s.state.vertex,", closed=",s.closed,")")
function _coordinate_candidate(s;kwargs...)
    _inspection_require_open(s)
    all(k->k in keys(s.initial),keys(kwargs)) || throw(ArgumentError("Select coordinate, vertex, phase or source_axes."))
    p=merge(Base.structdiff(s.state,(;revision=s.state.revision)),(;kwargs...))
    merge(_coordinate_selection(s.object;p...),(;revision=s.state.revision+1))
end
function check_inspection_selection(s::CoordinateInspectionSession;throw=false,kwargs...)
    issues=String[]
    lock(s.lock) do
        try;_coordinate_candidate(s;kwargs...)
        catch err;err isa InterruptException && rethrow();push!(issues,sprint(showerror,err));end
    end
    valid=isempty(issues);throw && !valid && _throw_invalid_visual(:check_inspection_selection,issues)
    (;kind=:inspection_selection,valid,issues)
end
function select_inspection!(s::CoordinateInspectionSession;kwargs...)
    _cancel_inspection_requests!(s)
    lock(s.lock) do
        s.busy && throw(ArgumentError("A coordinate selection is in progress."))
        p=_coordinate_candidate(s;kwargs...);snap=_coordinate_snapshot(s.object,p)
        s.state=p;s.snapshot=snap;s.snapshot_builds+=1;s.busy=true
    end
    try;_inspection_notify(s)
    finally;lock(s.lock) do;s.busy=false;end;end
    s
end
reset_inspection!(s::CoordinateInspectionSession)=select_inspection!(s;s.initial...)
function close_inspection!(s::CoordinateInspectionSession)
    _cancel_inspection_requests!(s)
    lock(s.lock) do
        s.closed && return s
        s.busy && throw(ArgumentError("Wait for the current coordinate update before closing."))
        s.closed=true;s.object=nothing
    end
    try;_inspection_notify(s)
    finally;lock(s.lock) do;empty!(s.listeners);end;end
    s
end
function check_inspection_session(s::CoordinateInspectionSession;throw=false)
    issues=String[]
    lock(s.lock) do
        s.snapshot.metadata.selection==s.state || push!(issues,"Snapshot selection mismatch.")
        s.closed && (s.object!==nothing || !isempty(s.listeners)) && push!(issues,"Closed explorer retains input or listeners.")
        append!(issues,check_visual_spec(s.snapshot).issues)
    end
    valid=isempty(issues);throw && !valid && _throw_invalid_visual(:check_inspection_session,issues)
    (;kind=:inspection_session,valid,issues)
end
function _visual_spec(s::CoordinateInspectionSession,::Symbol;kwargs...)
    isempty(kwargs) || throw(ArgumentError("Use select_inspection! to change the coordinate selection."))
    VisualizationSpec(:linked_inspector;title="Cohomological coordinates",panels=s.snapshot.panels,
        interaction=(;hover=true,clicks=true,labels=true,widgets=(:coordinate_selection,),notebook=:linked_inspector,
            mode=:live_julia,requires_live_julia=true,offline_widgets=false),metadata=(;session=s,inspector_kind=:cohomological_coordinates))
end
_visual_request_cost(::CoordinateInspectionSession,::Symbol)=(;work=:retained_coordinates,hover=:read_only,requires_live_julia=true,timing=:not_measured)
