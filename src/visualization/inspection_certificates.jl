# One prepared, owned certificate; navigation never searches for witnesses.
"""Linked exact-map explorer for a validated interleaving or epsilon-pruning result."""
mutable struct CertificateInspectionSession <: _AbstractInspectionSession
    prepared::Any
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
end

function inspection_session(x::_CertificateInput;budget=Interleavings.InterleavingBudget(),kwargs...)
    c=_prepare_certificate(x;budget)
    initial=_certificate_options(c;kwargs...)
    snapshot=_certificate_spec(c;initial...)
    CertificateInspectionSession(c,initial,merge(initial,(;revision=0)),snapshot,1,
        Dict{Int,Any}(),1,String[],false,false,ReentrantLock())
end
inspection_selection(s::CertificateInspectionSession)=lock(s.lock) do;s.state;end
inspection_snapshot(s::CertificateInspectionSession)=lock(s.lock) do;s.snapshot;end
function inspection_summary(s::CertificateInspectionSession)
    lock(s.lock) do
        (;kind=:certificate_inspection_session,selection=s.state,revision=s.state.revision,
            verified=s.snapshot.metadata.verified,guarantee=s.snapshot.metadata.guarantee,
            locations=s.closed ? 0 : length(s.prepared.certificate.problem.points),
            edges=s.closed ? 0 : length(s.prepared.edges),pruning=s.closed ? false : s.prepared.pruning!==nothing,
            closed=s.closed,updating=s.busy,snapshot_builds=s.snapshot_builds,
            listener_count=length(s.listeners),listener_errors=Tuple(s.listener_errors))
    end
end
describe(s::CertificateInspectionSession)=inspection_summary(s)
Base.show(io::IO,s::CertificateInspectionSession)=print(io,"CertificateInspectionSession(location=",s.state.vertex,", view=",s.state.view,", closed=",s.closed,")")
function _certificate_candidate(s;kwargs...)
    _inspection_require_open(s)
    all(k->k in (:vertex,:point,:edge,:side,:view,:matrix_limit),keys(kwargs)) || throw(ArgumentError("Unknown certificate selection keyword."))
    prior=(;vertex=s.state.vertex,edge=s.state.edge,side=s.state.side,view=s.state.view,matrix_limit=s.state.matrix_limit)
    merge(_certificate_options(s.prepared;merge(prior,(;kwargs...))...),(;revision=s.state.revision+1))
end
function check_inspection_selection(s::CertificateInspectionSession;throw=false,kwargs...)
    lock(s.lock) do
        issues=String[];candidate=nothing
        try;candidate=_certificate_candidate(s;kwargs...)
        catch err;err isa InterruptException && rethrow();push!(issues,sprint(showerror,err));end
        throw && !isempty(issues) && _throw_invalid_visual(:check_inspection_selection,issues)
        (;kind=:inspection_selection,valid=isempty(issues),issues,selection=candidate)
    end
end
function select_inspection!(s::CertificateInspectionSession;kwargs...)
    lock(s.lock)
    try
        _inspection_require_open(s)
        s.busy && throw(ArgumentError("An inspection update or callback is already in progress."))
        c=_certificate_candidate(s;kwargs...)
        s.busy=true
        try
            snapshot=_certificate_spec(s.prepared;vertex=c.vertex,edge=c.edge,side=c.side,view=c.view,matrix_limit=c.matrix_limit)
            s.snapshot=snapshot;s.state=c;s.snapshot_builds+=1
        catch;s.busy=false;rethrow();end
    finally;unlock(s.lock);end
    try;_inspection_notify(s)
    finally;lock(s.lock) do;s.busy=false;end;end
    s
end
reset_inspection!(s::CertificateInspectionSession)=select_inspection!(s;s.initial...)
function close_inspection!(s::CertificateInspectionSession)
    lock(s.lock)
    try
        s.closed && return s
        s.busy && throw(ArgumentError("Cannot close during an update or callback."))
        s.closed=true;s.busy=true
    finally;unlock(s.lock);end
    try;_inspection_notify(s)
    finally
        lock(s.lock) do
            empty!(s.listeners);s.prepared=nothing;s.busy=false
        end
    end
    s
end
function check_inspection_session(s::CertificateInspectionSession;throw=false)
    lock(s.lock) do
        issues=copy(check_visual_spec(s.snapshot).issues)
        o=s.snapshot.metadata.selection
        all(k->getproperty(o,k)==getproperty(s.state,k),keys(o)) || push!(issues,"Snapshot differs from selection.")
        s.closed && !s.busy && !(isempty(s.listeners) && s.prepared===nothing) && push!(issues,"Closed session retains input or callbacks.")
        throw && !isempty(issues) && _throw_invalid_visual(:check_inspection_session,issues)
        (;kind=:inspection_session,valid=isempty(issues),issues,closed=s.closed,revision=s.state.revision)
    end
end
function _visual_spec(s::CertificateInspectionSession,::Symbol;kwargs...)
    isempty(kwargs) || throw(ArgumentError("Use select_inspection! to change the certificate selection."))
    VisualizationSpec(:linked_inspector;title="Explore a certificate",panels=inspection_snapshot(s).panels,
        interaction=(;hover=true,clicks=true,labels=true,widgets=(:certificate_location,:certificate_equations),
            notebook=:linked_inspector,mode=:live_julia,requires_live_julia=true,offline_widgets=false),
        metadata=(;session=s,inspector_kind=:certificate))
end
_visual_request_cost(::CertificateInspectionSession,::Symbol)=(;work=:retained_exact_matrices,
    search=:none,hover=:retained_data,requires_live_julia=true,timing=:not_measured)
