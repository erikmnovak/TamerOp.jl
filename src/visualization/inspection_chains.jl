"""
    ChainInspectionSession

Linked chain/cochain or connecting-map inspection, created by
`inspection_session(object; ...)`. For a `FilteredComplex`, select
`kind=:chain_inspector`. Degree/vertex/parameter/map changes clear a vector
unless a replacement is supplied. Vector input requests its quotient analysis;
`compute=true` requests bases without a vector. Hover is read-only. Inputs must
remain read-only while viewers use them. Closing releases inputs and listeners,
retaining the final static snapshot.
"""
mutable struct ChainInspectionSession <: _AbstractInspectionSession
    object::Union{Nothing,_ChainInspectionInput}
    initial::NamedTuple
    options::NamedTuple
    state::NamedTuple
    snapshot::VisualizationSpec
    snapshot_builds::Int
    listeners::Dict{Int,Any}
    next_listener::Int
    listener_errors::Vector{String}
    closed::Bool
    busy::Bool
    lock::ReentrantLock
end
function _chain_inspection_session(x::_ChainInspectionInput;kwargs...)
    connecting=x isa _ConnectingInput
    o=connecting ? _connecting_options(x;kwargs...) : _chain_parameters(x;kwargs...)
    initial=connecting ? (;degree=o.degree,vertex=o.vertex,map=o.map,vector=o.vector) :
        (;degree=o.degree,vertex=o.vertex,parameter=o.parameter,vector=o.vector,coordinates=o.coordinates,compute=o.compute)
    options=Base.structdiff(o,initial)
    snapshot=visual_spec(x;kind=connecting ? :connecting_map : :chain_inspector,o...)
    ChainInspectionSession(x,initial,options,merge(initial,(;revision=0)),snapshot,1,Dict{Int,Any}(),1,String[],false,false,ReentrantLock())
end
"""
    inspection_session(complex_or_map; degree, vertex=nothing, vector=nothing,
                       coordinates=:ambient, compute=false, kwargs...)
    inspection_session(sequence; degree, vertex, map=:delta, vector=nothing)

Explore a supplied chain/cochain complex or an implemented connecting-map
result. A complex starts with its differentials; quotient bases are opt-in via
`compute=true` or a vector query. Module complexes select a finite `vertex`,
defaulting to the first vertex; vector-space complexes have no vertex. A cone sequence selects `degree` and
`:f`, `:i` or `:delta`; a snake result selects a finite vertex and one of
`:k1`, `:k2`, `:delta`, `:c1`, `:c2`. Sequence vectors use retained class bases.

Render with `visualize(session; backend=:wglmakie)`. Use `inspection_summary`
for cheap state, `select_inspection!` for explicit work, `inspection_snapshot`
for a static export, and `close_inspection!` to release inputs and callbacks.
Changing degree, vertex, parameter, sequence map or coordinate mode clears the
old vector unless a replacement is supplied. Failed updates preserve the prior
snapshot. Hover does not compute algebra. Inputs and returned arrays must remain
read-only while a session uses them.
"""
inspection_session(x::Union{ChainComplexes.CochainComplex,ChainComplexes.CochainMap,
    ModuleComplexes.ModuleCochainComplex,ModuleComplexes.ModuleCochainMap,
    DataIngestion.SourceChain,_ConnectingInput};kwargs...)=_chain_inspection_session(x;kwargs...)
function inspection_session(x::DataIngestion.SourceChain;kwargs...)
    o=_chain_parameters(x;kwargs...)
    o.vector===nothing && o.coordinates!=:ambient && throw(ArgumentError("A retained chain uses ambient coordinates; supply a vector to select class coordinates."))
    model=_chain_model(x,o)
    vector=o.vector===nothing ? x.coefficients[model.active_ids[o.degree+1]] : o.vector
    # Navigate the actual source complex, starting at the retained chain. This
    # lets clearing/changing a selection remove that vector without fabricating
    # a new SourceChain record at another parameter.
    _chain_inspection_session(x.source;merge(o,(;vector))...)
end
inspection_selection(s::ChainInspectionSession)=lock(s.lock) do;s.state;end
inspection_snapshot(s::ChainInspectionSession)=lock(s.lock) do;s.snapshot;end
function inspection_summary(s::ChainInspectionSession)
    lock(s.lock) do
        m=s.snapshot.metadata
        (;kind=:chain_inspection_session,selection=s.state,revision=s.state.revision,
            view=s.snapshot.kind,quotient_computed=m.quotient_computed,
            field=m.field,closed=s.closed,updating=s.busy,snapshot_builds=s.snapshot_builds,
            listener_count=length(s.listeners),listener_errors=Tuple(s.listener_errors))
    end
end
describe(s::ChainInspectionSession)=inspection_summary(s)
Base.show(io::IO,s::ChainInspectionSession)=print(io,"ChainInspectionSession(degree=",s.state.degree,", closed=",s.closed,")")
function _chain_candidate(s;kwargs...)
    _inspection_require_open(s)
    all(k->k in keys(s.initial),keys(kwargs)) || throw(ArgumentError("Only the live degree, vertex, parameter, vector, coordinate mode, compute or map selection can change."))
    state=Base.structdiff(s.state,(revision=s.state.revision,))
    changed=any(k->haskey(kwargs,k) && kwargs[k]!=get(state,k,nothing),(:degree,:vertex,:parameter,:map,:coordinates))
    changed && !haskey(kwargs,:vector) && (state=merge(state,(;vector=nothing)))
    candidate=merge(state,(;kwargs...))
    o=s.object isa _ConnectingInput ? _connecting_options(s.object;candidate...,s.options...) :
        _chain_parameters(s.object;candidate...,s.options...)
    return merge(NamedTuple{keys(s.initial)}(Tuple(o[k] for k in keys(s.initial))),(;revision=s.state.revision+1))
end
function check_inspection_selection(s::ChainInspectionSession;throw=false,kwargs...)
    lock(s.lock) do
        issues=String[];candidate=nothing
        try
            candidate=_chain_candidate(s;kwargs...)
            # Selection option validation precedes mutation; algebra is deferred to
            # the explicitly requested snapshot build in select_inspection!.
        catch err
            err isa InterruptException && rethrow();push!(issues,sprint(showerror,err))
        end
        valid=isempty(issues)
        throw && !valid && _throw_invalid_visual(:check_inspection_selection,issues)
        (;kind=:inspection_selection,valid,issues,selection=candidate)
    end
end
function select_inspection!(s::ChainInspectionSession;kwargs...)
    lock(s.lock)
    try
        _inspection_require_open(s)
        s.busy && throw(ArgumentError("An inspection update or callback is already in progress."))
        c=_chain_candidate(s;kwargs...);s.busy=true
        try
            state=Base.structdiff(c,(revision=c.revision,))
            snapshot=visual_spec(s.object;kind=s.snapshot.kind,state...,s.options...)
            s.snapshot=snapshot;s.state=c;s.snapshot_builds+=1
        catch;s.busy=false;rethrow();end
    finally;unlock(s.lock);end
    try;_inspection_notify(s)
    finally;lock(s.lock) do;s.busy=false;end;end
    return s
end
reset_inspection!(s::ChainInspectionSession)=select_inspection!(s;s.initial...)
function close_inspection!(s::ChainInspectionSession)
    lock(s.lock)
    try
        s.closed && return s
        s.busy && throw(ArgumentError("Cannot close during an inspection update or callback."))
        s.closed=true;s.busy=true
    finally;unlock(s.lock);end
    try;_inspection_notify(s)
    finally
        lock(s.lock) do
            empty!(s.listeners);s.object=nothing;s.options=NamedTuple();s.busy=false
        end
    end
    return s
end
function check_inspection_session(s::ChainInspectionSession;throw=false)
    lock(s.lock) do
        issues=String[]
        s.snapshot.metadata.degree==s.state.degree && s.snapshot.metadata.vertex==s.state.vertex ||
            push!(issues,"Snapshot and degree/vertex selection disagree.")
        if s.closed && !s.busy
            s.object===nothing && isempty(s.listeners) || push!(issues,"Closed inspector retains input or callbacks.")
        end
        append!(issues,check_visual_spec(s.snapshot).issues)
        valid=isempty(issues)
        throw && !valid && _throw_invalid_visual(:check_inspection_session,issues)
        (;kind=:inspection_session,valid,issues,closed=s.closed,revision=s.state.revision)
    end
end
function _visual_spec(s::ChainInspectionSession,::Symbol;kwargs...)
    VisualizationSpec(:linked_inspector;title="Explore chains and classes",panels=inspection_snapshot(s).panels,
        interaction=(;hover=true,clicks=true,labels=true,widgets=(:chain_degree,:chain_vector,:chain_map),
            notebook=:linked_inspector,mode=:live_julia,requires_live_julia=true,offline_widgets=false),
        metadata=(;session=s,inspector_kind=:chains))
end
_visual_request_cost(::ChainInspectionSession,::Symbol)=merge(_chain_cost(),(;requires_live_julia=true,hover=:retained_dimensions))
