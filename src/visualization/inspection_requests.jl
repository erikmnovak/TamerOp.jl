# One worker and one replaceable pending request per mathematical session.
# Work runs on a private transaction; obsolete results never enter the visible
# state or its caches. Owners may retain benign caches in the read-only input.
mutable struct _InspectionRequests
    generation::Int
    pending::Any
    active::Any
    task::Union{Nothing,Task}
    submitted::Int
    committed::Int
    discarded::Int
end
_InspectionRequests() = _InspectionRequests(0,nothing,nothing,nothing,0,0,0)

function _inspection_request_summary(s)
    q=s.requests
    (;pending=q.pending!==nothing,running=q.active!==nothing,
      submitted=q.submitted,committed=q.committed,discarded=q.discarded)
end

function _cancel_inspection_requests!(s, owner=nothing)
    lock(s.lock) do
        q=s.requests
        if owner===nothing || (q.pending!==nothing && q.pending.owner===owner) ||
                (q.pending===nothing && q.active!==nothing && q.active.owner===owner)
            q.generation+=1
            if q.pending!==nothing && (owner===nothing || q.pending.owner===owner)
                q.pending=nothing
            end
        end
    end
    nothing
end

# Copy only transaction-owned containers. The finite object, prepared geometry
# and immutable specifications are shared; no source reconstruction is involved.
function _inspection_fork(s)
    values=map(fieldnames(typeof(s))) do name
        value=getfield(s,name)
        name===:requests && return _InspectionRequests()
        name===:lock && return ReentrantLock()
        name===:listeners && return Dict{Int,Any}()
        name===:listener_errors && return String[]
        name===:busy && return false
        name in (:cache,:cache_order,:slice_cache,:slice_cache_order) && return copy(value)
        return value
    end
    typeof(s)(values...)
end

function _inspection_adopt!(s, transaction)
    for name in (:state,:snapshot,:snapshot_builds,:cache,:cache_order,:cache_hits,
                 :cache_misses,:slice_result,:slice_cache,:slice_cache_order,
                 :slice_cache_hits,:slice_cache_misses)
        hasfield(typeof(s),name) && setfield!(s,name,getfield(transaction,name))
    end
end

function _request_inspection!(s, owner; status=(_,_)->nothing, kwargs...)
    lock(s.lock) do
        _inspection_require_open(s)
        q=s.requests
        q.generation+=1;q.submitted+=1
        q.pending!==nothing && (q.discarded+=1)
        q.pending=(;owner,status,kwargs=(;kwargs...),generation=q.generation)
        status(:pending,nothing)
        if q.task===nothing
            q.task=@async _run_inspection_requests!(s)
        end
    end
    nothing
end

function _run_inspection_requests!(s)
    q=s.requests
    try
        while true
            # Allow a browser burst to replace the pending coordinates before
            # starting another nonpreemptible mathematical query.
            sleep(0.025)
            work=lock(s.lock) do
                (s.closed || q.pending===nothing) && return nothing
                request=q.pending;q.pending=nothing;q.active=request
                (;request,transaction=_inspection_fork(s),revision=s.state.revision)
            end
            work===nothing && break
            result_error=nothing
            try
                select_inspection!(work.transaction;work.request.kwargs...)
            catch err
                err isa InterruptException && rethrow()
                result_error=sprint(showerror,err)
            end
            # A query may run without yielding. Drain ready transport tasks
            # before deciding whether its result is still wanted.
            yield()
            accepted=lock(s.lock) do
                q.active=nothing
                if s.closed || q.generation!=work.request.generation || s.state.revision!=work.revision
                    q.discarded+=1
                    return false
                end
                if result_error===nothing
                    _inspection_adopt!(s,work.transaction)
                    q.committed+=1;s.busy=true
                end
                return true
            end
            if accepted
                if result_error===nothing
                    try;_inspection_notify(s)
                    finally;lock(s.lock) do;s.busy=false;end;end
                end
                lock(s.lock) do
                    if !s.closed && q.generation==work.request.generation
                        work.request.status(result_error===nothing ? :committed : :error,result_error)
                    end
                end
            end
        end
    finally
        lock(s.lock) do
            q.active=nothing;q.task=nothing
            # No yielding between this check and releasing the session lock.
            if !s.closed && q.pending!==nothing
                q.task=@async _run_inspection_requests!(s)
            end
        end
    end
    nothing
end
