"""
    FieldInspectionSession

Linked coordinate sections with one shared selected grid point, layer and scale.
Construct with `field_session`; query using the common inspection accessors.
Closing releases sampled arrays and listeners, retaining the last static view.
"""
mutable struct FieldInspectionSession <: _AbstractInspectionSession
    fields::Any
    initial::NamedTuple
    state::NamedTuple
    scale::NamedTuple
    snapshot::VisualizationSpec
    listeners::Dict{Int,Any}
    next_listener::Int
    listener_errors::Vector{String}
    closed::Bool
    busy::Bool
    lock::ReentrantLock
end
"""
    field_session(field_or_fields; planes, indices, layer, normalization=:auto,
        colorrange=nothing, contours=(), baseline=nothing, max_cells=250_000)

Explore sampled Cartesian scalar fields in linked ordered coordinate planes.
`indices` selects one coordinate on every axis. Each plane varies its two axes
and holds the others at that shared point. The default shows the three planes
of the first three coordinates (one in dimension two). `layer` is a stored ID.

All fields use one fixed range over every supplied sample and layer. Signed
values use a zero-centered diverging range; `baseline="id"` displays differences
on the intersection of available samples. `contours` explicitly requests levels
of the piecewise-affine interpolant, never an exact support boundary. At most
12 panels and `max_cells` displayed samples times contour levels are allowed.
The session copies input fields; subsequent selections do not recompute an
invariant. Change options with `select_inspection!`, reset with
`reset_inspection!`, and use `inspection_snapshot` for static exports.
"""
function field_session(input::_FieldInput;kwargs...)
    fields=deepcopy(_field_list(input));o=_field_options(fields;kwargs...);scale=_field_scale(fields,o)
    FieldInspectionSession(fields,deepcopy(o),merge(o,(;revision=0)),scale,_field_snapshot(fields,o,scale),
        Dict{Int,Any}(),1,String[],false,false,ReentrantLock())
end
inspection_selection(s::FieldInspectionSession)=lock(s.lock) do;deepcopy(s.state);end
inspection_snapshot(s::FieldInspectionSession)=lock(s.lock) do;s.snapshot;end
inspection_summary(s::FieldInspectionSession)=lock(s.lock) do
    (;kind=:field_inspection_session,selection=deepcopy(s.state),revision=s.state.revision,
        scale=s.scale,selected_values=s.snapshot.metadata.selected_values,closed=s.closed,updating=s.busy,
        listener_count=length(s.listeners),listener_errors=Tuple(s.listener_errors))
end
describe(s::FieldInspectionSession)=inspection_summary(s)
Base.show(io::IO,s::FieldInspectionSession)=print(io,"FieldInspectionSession(layer=",s.state.layer,", closed=",s.closed,")")
function _field_candidate(s;kwargs...)
    _inspection_require_open(s)
    _field_options(s.fields;merge(Base.structdiff(s.state,(;revision=s.state.revision)),(;kwargs...))...)
end
function check_inspection_selection(s::FieldInspectionSession;throw=false,kwargs...)
    lock(s.lock) do
        issues=String[];candidate=nothing
        try;candidate=_field_candidate(s;kwargs...);_field_scale(s.fields,candidate)
        catch err;err isa InterruptException&&rethrow();push!(issues,sprint(showerror,err));end
        valid=isempty(issues)
        throw&&!valid&&_throw_invalid_visual(:check_inspection_selection,issues)
        (;kind=:inspection_selection,valid,issues,selection=candidate)
    end
end
function select_inspection!(s::FieldInspectionSession;reset=false,kwargs...)
    lock(s.lock)
    try
        _inspection_require_open(s);s.busy&&throw(ArgumentError("An inspection update is already in progress."))
        reset isa Bool || throw(ArgumentError("reset must be Bool."))
        reset&&!isempty(kwargs)&&throw(ArgumentError("Reset must be a separate action."))
        o=reset ? deepcopy(s.initial) : _field_candidate(s;kwargs...)
        s.busy=true
        try
            scale=all(k->getproperty(o,k)==getproperty(s.state,k),(:normalization,:colorrange,:baseline)) ? s.scale : _field_scale(s.fields,o)
            snapshot=_field_snapshot(s.fields,o,scale)
            s.state=merge(o,(;revision=s.state.revision+1));s.scale=scale;s.snapshot=snapshot
        catch;s.busy=false;rethrow();end
    finally;unlock(s.lock);end
    try;_inspection_notify(s)
    finally;lock(s.lock) do;s.busy=false;end;end
    s
end
reset_inspection!(s::FieldInspectionSession)=select_inspection!(s;reset=true)
function close_inspection!(s::FieldInspectionSession)
    lock(s.lock)
    try
        s.closed&&return s;s.busy&&throw(ArgumentError("Cannot close during an inspection update."))
        s.closed=true;s.busy=true
    finally;unlock(s.lock);end
    try;_inspection_notify(s)
    finally;lock(s.lock) do;empty!(s.listeners);s.fields=nothing;s.busy=false;end;end
    s
end
function check_inspection_session(s::FieldInspectionSession;throw=false)
    lock(s.lock) do
        issues=copy(check_visual_spec(s.snapshot).issues)
        s.snapshot.metadata.selection==Base.structdiff(s.state,(;revision=s.state.revision))||push!(issues,"Snapshot and selection disagree.")
        s.closed&&!s.busy&&!(s.fields===nothing&&isempty(s.listeners))&&push!(issues,"Closed session retains fields/listeners.")
        valid=isempty(issues);throw&&!valid&&_throw_invalid_visual(:check_inspection_session,issues)
        (;kind=:inspection_session,valid,issues,closed=s.closed,revision=s.state.revision)
    end
end
function _visual_spec(s::FieldInspectionSession,::Symbol;kwargs...)
    isempty(kwargs)||throw(ArgumentError("Change a field session with select_inspection!."))
    VisualizationSpec(:linked_inspector;title="Explore coordinate sections",panels=inspection_snapshot(s).panels,
        interaction=(;hover=true,clicks=true,labels=true,widgets=(:field_point,:field_planes,:field_layer),
            notebook=:linked_inspector,mode=:live_julia,requires_live_julia=true,offline_widgets=false),
        metadata=(;session=s,inspector_kind=:fields))
end
_visual_request_cost(::FieldInspectionSession,::Symbol)=(;work=:retained_grid_sections,
    selection=:no_invariant_queries,scale=:cached_over_all_layers,requires_live_julia=true,timing=:not_measured)
