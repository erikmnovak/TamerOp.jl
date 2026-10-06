# Linked interval-group/member selection over retained barcode data. Selection
# never recomputes persistence or creates representatives absent from its input.

"""
    IntervalInspectionSession

A linked barcode/diagram inspection created with `inspection_session(barcode)`.
Group IDs identify equal decorated intervals; `member` chooses an original
interval within a multiplicity group. Retained ordinary persistence cycles are
read only after `representative=true`. They are reduction choices, not canonical
classes or assertions about source geometry. Use `inspection_snapshot` for an
exportable static view, and `close_inspection!` to release live callbacks.
"""
mutable struct IntervalInspectionSession{T,P} <: _AbstractInspectionSession
    object::Union{Nothing,T}
    payload::P
    state::NamedTuple
    snapshot::VisualizationSpec
    window::Union{Nothing,Tuple}
    max_intervals::Int
    selected_representative::Union{Nothing,NamedTuple}
    snapshot_builds::Int
    listeners::Dict{Int,Any}
    next_listener::Int
    listener_errors::Vector{String}
    closed::Bool
    busy::Bool
    lock::ReentrantLock
end

const _IntervalInspectionInput = Union{OrdinaryPersistence.PersistenceDiagram,
    _RawVisualBarcode,SliceBarcodesResult,FiberedSliceResult,ProjectedBarcodesResult,InvariantResult}

function _interval_inspection_readout(payload, state, representative)
    if state.interval === nothing
        return _inspection_text_panel("Inspect an interval", [
            "Select a group in the barcode, diagram, or interval selector.",
            "Equal decorated intervals share a multiplicity group.",
            "Choose an original member before requesting a retained cycle."])
    end
    record = payload.records[state.interval]
    lines = [_interval_record_label(record)]
    members = get(record,:members,())
    if isempty(members)
        push!(lines,"The supplied barcode has no original source-member correspondence.")
    elseif state.member === nothing
        push!(lines,"Choose one of $(length(members)) original members; the group alone does not identify a cycle.")
    else
        source = members[state.member]
        push!(lines,"Member $(state.member): $(source.kind) interval $(source.index), homological degree $(source.dim).")
    end
    if representative === nothing
        push!(lines,"Retained cycles are read only after an explicit representative request.")
    elseif !representative.available
        push!(lines,"Representative unavailable: $(representative.reason).")
        push!(lines,"For ordinary diagrams, compute persistence with representatives=true to retain reduction cycles.")
    else
        push!(lines,"F$(representative.field.p) reduction representative; this choice is not canonical and carries no geometric embedding.")
        for (name,chain) in (("Cycle",representative.cycle),("Bounding chain at death",representative.bounding_chain))
            chain === nothing && continue
            shown = min(length(chain.cell_ids),12)
            push!(lines,"$name: showing $shown of $(length(chain.cell_ids)) cells in dimension $(chain.dimension).")
            for i in 1:shown
                push!(lines,"  $(chain.coefficients[i]) * cell $(chain.cell_ids[i]) " *
                    "(index $(chain.cell_indices[i]), grade $(_interval_value(chain.cell_grades[i])))")
            end
            length(chain.cell_ids) <= 12 || push!(lines,"  Further cells remain in the exact snapshot metadata.")
        end
    end
    return _inspection_text_panel("Selected interval and source member", lines;
        metadata=(; selected_representative=representative))
end

function _interval_inspection_snapshot(payload, state, window, max_intervals, representative)
    view = _interval_view(payload.records;window,interval=state.interval,max_intervals,order=payload.order)
    barcode,diagram = _interval_panels(payload.records;window,interval=state.interval,max_intervals,
        order=payload.order,endpoint_semantics=payload.endpoint_semantics,
        essential_status=payload.essential_status,metadata=payload.metadata)
    readout = _interval_inspection_readout(payload,state,representative)
    return VisualizationSpec(:interval_inspector;title="Intervals and their source members",
        subtitle="$(payload.order) parameter order; selection shares exact interval-group IDs.",
        panels=[barcode,diagram,readout],
        metadata=(;interval_payload=payload,interval_view=view,inspection_selection=state,
            selected_representative=representative,snapshot_semantics=:static,panel_columns=2,
            figure_size=(1600,1100)))
end

"""
    inspection_session(barcode; dim=0, index=nothing, window=nothing, max_intervals=200)

Link a barcode and persistence diagram with shared exact interval selection.
Accepted inputs are an ordinary `PersistenceDiagram`, a raw interval dictionary,
vector or packed barcode, a `FiberedSliceResult`, a `SliceBarcodesResult`, or a
`ProjectedBarcodesResult` (also the corresponding slice invariant wrapper).
Use `dim` for ordinary homology degree and `index` for a barcode family member.

`window=(lo,hi)` clips the drawing, not the retained endpoints. `max_intervals`
bounds the displayed groups; all group IDs and counts remain in the snapshot.
`select_inspection!(session; interval=i)` selects a group, `member=j` chooses an
original member, and `representative=true` explicitly requests its retained
ordinary cycle. Multiplicity groups require a member choice; a raw barcode
does not imply source representatives. Window and display budget are fixed for
one session. The input and returned exact metadata must be treated as read-only.
"""
function inspection_session(obj::_IntervalInspectionInput;
                            dim=0,index=nothing,window=nothing,max_intervals=200)
    payload = _interval_payload(obj;dim,index)
    budget = _inspection_integer(max_intervals,:max_intervals;lower=1)
    _interval_window(payload.records,window)
    fixed_window = window === nothing ? nothing : Tuple(window)
    state = (;interval=nothing,member=nothing,representative=false,revision=0)
    snapshot = _interval_inspection_snapshot(payload,state,fixed_window,budget,nothing)
    return IntervalInspectionSession(obj,payload,state,snapshot,fixed_window,budget,nothing,1,
        Dict{Int,Any}(),1,String[],false,false,ReentrantLock())
end

inspection_selection(s::IntervalInspectionSession) = lock(s.lock) do
    s.state
end

inspection_snapshot(s::IntervalInspectionSession) = lock(s.lock) do
    s.snapshot
end

function inspection_summary(s::IntervalInspectionSession)
    return lock(s.lock) do
        view = s.snapshot.metadata.interval_view
        (;kind=:interval_inspection_session,closed=s.closed,selection=s.state,revision=s.state.revision,
            window=s.window,max_intervals=s.max_intervals,order=s.payload.order,
            total_groups=view.total_groups,displayed_groups=view.displayed_groups,
            total_multiplicity=view.total_multiplicity,
            representative_available=get(s.payload.metadata,:representative_available,false),
            representative_reason=get(s.payload.metadata,:representative_reason,:not_retained),
            snapshot_builds=s.snapshot_builds,listener_count=length(s.listeners),
            listener_errors=Tuple(s.listener_errors),updating=s.busy,
            cost=(;intervals=:retained,selection=:no_persistence_recomputation,
                representative=:explicit_retained_member_only,updates=:synchronous))
    end
end

describe(s::IntervalInspectionSession) = inspection_summary(s)

function Base.show(io::IO,s::IntervalInspectionSession)
    d = inspection_summary(s)
    print(io,"IntervalInspectionSession(groups=",d.total_groups,", selected=",d.selection.interval,
        ", revision=",d.revision,", closed=",d.closed,")")
end

function Base.show(io::IO,::MIME"text/plain",s::IntervalInspectionSession)
    show(io,s)
    d = inspection_summary(s)
    print(io,"\n  selection: ",d.selection,"\n  displayed groups: ",d.displayed_groups,
        "/",d.total_groups,"\n  retained representatives: ",d.representative_available)
end

function _interval_inspection_candidate(s;interval=nothing,member=nothing,representative=nothing)
    _inspection_require_open(s)
    group = interval === nothing ? s.state.interval : _inspection_integer(interval,:interval)
    group === 0 && (group=nothing)
    group === nothing || group <= length(s.payload.records) ||
        throw(ArgumentError("interval must be an existing group ID or 0 to clear."))
    changed_group = group != s.state.interval
    members = group === nothing ? () : get(s.payload.records[group],:members,())
    chosen_member = if member === nothing
        changed_group ? (length(members) == 1 ? 1 : nothing) : s.state.member
    else
        q = _inspection_integer(member,:member)
        iszero(q) ? nothing : q
    end
    chosen_member === nothing || (group !== nothing && chosen_member <= length(members)) ||
        throw(ArgumentError("member must identify a retained original interval within the selected group, or 0 to clear."))
    wants_representative = representative === nothing ?
        ((changed_group || chosen_member != s.state.member) ? false : s.state.representative) : representative
    wants_representative isa Bool || throw(ArgumentError("representative must be Bool."))
    wants_representative && group === nothing &&
        throw(ArgumentError("Select an interval before requesting a representative."))
    wants_representative && !isempty(members) && chosen_member === nothing &&
        throw(ArgumentError("Select an original member of this multiplicity group before requesting its representative."))
    return (;interval=group,member=chosen_member,representative=wants_representative,
        revision=s.state.revision+1)
end

function check_inspection_selection(s::IntervalInspectionSession;throw::Bool=false,kwargs...)
    return lock(s.lock) do
        issues = String[]
        candidate = nothing
        try
            candidate = _interval_inspection_candidate(s;kwargs...)
        catch err
            err isa InterruptException && rethrow()
            push!(issues,sprint(showerror,err))
        end
        valid = isempty(issues)
        throw && !valid && _throw_invalid_visual(:check_inspection_selection,issues)
        (;kind=:inspection_selection,valid,issues,selection=candidate)
    end
end

function _interval_inspection_representative(s,state)
    state.representative || return nothing
    record = s.payload.records[state.interval]
    members = get(record,:members,())
    if isempty(members) || !(s.object isa OrdinaryPersistence.PersistenceDiagram)
        return (;available=false,reason=:no_source_member_correspondence,cycle=nothing,bounding_chain=nothing)
    end
    source = members[state.member]
    return OrdinaryPersistence.persistence_representative(s.object;
        dim=source.dim,kind=source.kind,index=source.index)
end

function select_inspection!(s::IntervalInspectionSession;kwargs...)
    lock(s.lock)
    try
        _inspection_require_open(s)
        s.busy && throw(ArgumentError("An inspection update or callback is already in progress."))
        s.busy = true
        try
            state = _interval_inspection_candidate(s;kwargs...)
            same_member = (state.interval,state.member,state.representative) ==
                (s.state.interval,s.state.member,s.state.representative)
            representative = same_member ? s.selected_representative : _interval_inspection_representative(s,state)
            snapshot = _interval_inspection_snapshot(s.payload,state,s.window,s.max_intervals,representative)
            s.state,s.snapshot,s.selected_representative = state,snapshot,representative
            s.snapshot_builds += 1
        catch
            s.busy = false
            rethrow()
        end
    finally
        unlock(s.lock)
    end
    try
        _inspection_notify(s)
    finally
        lock(s.lock) do
            s.busy = false
        end
    end
    return s
end

reset_inspection!(s::IntervalInspectionSession) =
    select_inspection!(s;interval=0,member=0,representative=false)

function close_inspection!(s::IntervalInspectionSession)
    lock(s.lock)
    try
        s.closed && return s
        s.busy && throw(ArgumentError("Cannot close during an inspection update or callback."))
        s.busy = true
        s.closed = true
    finally
        unlock(s.lock)
    end
    try
        _inspection_notify(s)
    finally
        lock(s.lock) do
            empty!(s.listeners)
            s.object = nothing
            s.busy = false
        end
    end
    return s
end

function check_inspection_session(s::IntervalInspectionSession;throw::Bool=false)
    return lock(s.lock) do
        issues = String[]
        get(s.snapshot.metadata,:inspection_selection,nothing) == s.state ||
            push!(issues,"snapshot and current interval selection disagree")
        get(s.snapshot.metadata,:selected_representative,nothing) == s.selected_representative ||
            push!(issues,"snapshot and retained representative disagree")
        valid_group = s.state.interval === nothing || 1 <= s.state.interval <= length(s.payload.records)
        valid_group || push!(issues,"selected interval is outside the retained groups")
        members = !valid_group || s.state.interval === nothing ? () :
            get(s.payload.records[s.state.interval],:members,())
        s.state.member === nothing || 1 <= s.state.member <= length(members) ||
            push!(issues,"selected member is outside the retained original intervals")
        s.state.representative || s.selected_representative === nothing ||
            push!(issues,"a representative remains without its explicit selection")
        if s.closed && !s.busy
            isempty(s.listeners) && s.object === nothing || push!(issues,"closed session retains its live input or callbacks")
        elseif !s.closed
            s.object === nothing && push!(issues,"open interval session has no input")
        end
        append!(issues,check_visual_spec(s.snapshot).issues)
        valid = isempty(issues)
        throw && !valid && _throw_invalid_visual(:check_inspection_session,issues)
        (;kind=:inspection_session,valid,issues,closed=s.closed,revision=s.state.revision)
    end
end
