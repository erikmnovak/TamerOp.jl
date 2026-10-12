# Page navigation and selected local homology share the static spectral builder.
"""
    SpectralInspectionSession

Created by `inspection_session(ss; page=2, compute=true, ...)`. Navigate with
`select_inspection!(session; page=r)`; inspect the outgoing differential with
`source=(p,q)`. `source=nothing` clears it. Moving to a new page clears selection
unless a source is supplied explicitly. `compute=false` shows missing pages as
unknown; `compute=true` permits requested page algebra. Hover never computes.
The final static snapshot survives closure. Treat the supplied sequence as
read-only outside this session while live viewers are using it.
"""
mutable struct SpectralInspectionSession <: _AbstractInspectionSession
    object::Union{Nothing,_SpectralVisualInput}
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

function inspection_session(ss::_SpectralVisualInput;page=2,compute=true,window=nothing,
        truncated=false,max_terms=400,matrix_limit=(12,12))
    options=(;window,truncated,max_terms,matrix_limit)
    snapshot=visual_spec(ss;page,compute,options...)
    initial=(;page,source=nothing,compute)
    state=merge(initial,(;revision=0))
    return SpectralInspectionSession(ss,initial,options,state,snapshot,1,Dict{Int,Any}(),1,String[],false,false,ReentrantLock())
end
inspection_selection(s::SpectralInspectionSession)=lock(s.lock) do;s.state;end
inspection_snapshot(s::SpectralInspectionSession)=lock(s.lock) do;s.snapshot;end
function inspection_summary(s::SpectralInspectionSession)
    return lock(s.lock) do
        m=s.snapshot.metadata
        (;kind=:spectral_inspection_session,selection=s.state,revision=s.state.revision,
            page_status=m.page_status,differential_status=m.differential_status,
            convergence_bound=m.convergence_bound,limiting_dimensions=m.limiting_dimensions,
            convention=m.convention,closed=s.closed,updating=s.busy,
            snapshot_builds=s.snapshot_builds,listener_count=length(s.listeners),listener_errors=Tuple(s.listener_errors))
    end
end
describe(s::SpectralInspectionSession)=inspection_summary(s)
Base.show(io::IO,s::SpectralInspectionSession)=print(io,"SpectralInspectionSession(page=",s.state.page,
    ", source=",s.state.source,", closed=",s.closed,")")

function _spectral_candidate(s;page=s.state.page,source=page==s.state.page ? s.state.source : nothing,compute=s.state.compute)
    _inspection_require_open(s)
    kind=source === nothing ? :spectral_page : :spectral_differential
    check_visual_request(s.object;kind,page,source,compute,s.options...,throw=true)
    return (;page,source,compute,revision=s.state.revision+1)
end
function check_inspection_selection(s::SpectralInspectionSession;throw=false,kwargs...)
    return lock(s.lock) do
        issues=String[];candidate=nothing
        try;candidate=_spectral_candidate(s;kwargs...)
        catch err;err isa InterruptException && rethrow();push!(issues,sprint(showerror,err));end
        valid=isempty(issues)
        throw && !valid && _throw_invalid_visual(:check_inspection_selection,issues)
        (;kind=:inspection_selection,valid,issues,selection=candidate)
    end
end
function select_inspection!(s::SpectralInspectionSession;kwargs...)
    lock(s.lock)
    try
        _inspection_require_open(s)
        s.busy && throw(ArgumentError("An inspection update or callback is already in progress."))
        c=_spectral_candidate(s;kwargs...)
        s.busy=true
        try
            snapshot=visual_spec(s.object;kind=c.source === nothing ? :spectral_page : :spectral_differential,
                page=c.page,source=c.source,compute=c.compute,s.options...)
            s.snapshot=snapshot;s.state=c;s.snapshot_builds+=1
        catch;s.busy=false;rethrow();end
    finally;unlock(s.lock);end
    try;_inspection_notify(s)
    finally;lock(s.lock) do;s.busy=false;end;end
    return s
end
reset_inspection!(s::SpectralInspectionSession)=select_inspection!(s;s.initial...)
function close_inspection!(s::SpectralInspectionSession)
    lock(s.lock)
    try
        s.closed && return s
        s.busy && throw(ArgumentError("Cannot close during an inspection update or callback."))
        s.closed=true;s.busy=true
    finally;unlock(s.lock);end
    try;_inspection_notify(s)
    finally
        lock(s.lock) do
            empty!(s.listeners);s.object=nothing;s.busy=false
        end
    end
    return s
end
function check_inspection_session(s::SpectralInspectionSession;throw=false)
    return lock(s.lock) do
        issues=String[]
        s.snapshot.metadata.page==s.state.page && s.snapshot.metadata.source==s.state.source ||
            push!(issues,"Snapshot and selected page/source disagree.")
        if s.closed && !s.busy
            isempty(s.listeners) && s.object === nothing || push!(issues,"Closed spectral session retains callbacks or input.")
        end
        append!(issues,check_visual_spec(s.snapshot).issues)
        valid=isempty(issues)
        throw && !valid && _throw_invalid_visual(:check_inspection_session,issues)
        (;kind=:inspection_session,valid,issues,closed=s.closed,revision=s.state.revision)
    end
end

function _visual_spec(s::SpectralInspectionSession,::Symbol;kwargs...)
    return VisualizationSpec(:linked_inspector;title="Explore spectral pages",panels=inspection_snapshot(s).panels,
        interaction=(;hover=true,clicks=true,labels=true,widgets=(:spectral_page,:spectral_differential),
            notebook=:linked_inspector,mode=:live_julia,requires_live_julia=true,offline_widgets=false),
        metadata=(;session=s,inspector_kind=:spectral))
end
_visual_request_cost(::SpectralInspectionSession,::Symbol) =
    (;work=:requested_spectral_page_and_selected_local_homology,hover=:retained_dimensions,
        requires_live_julia=true,timing=:not_measured)
