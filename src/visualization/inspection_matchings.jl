# Shared matching session: retained assignments for ordinary intervals, selected
# fibered queries, bounded sample maps and explicit exact optimization.

"""
    MatchingInspectionSession

Linked diagram, barcode and cost selection, made by `inspection_session(witness)`
(or `inspection_session(a,b)` for barcodes/ordinary diagrams). For two compatible
encodings or fibered caches it also compares a common slice and a bounded family
of samples. `select_inspection!(s; pair=i)`, `sample=i`, or
`slice=(basepoint=(x,y),direction=(dx,dy))` changes the selection. Exact finite-window
optimization is explicit: `select_inspection!(s; optimum=true)`. Hover and pair
selection never run an optimizer. Snapshots retain the last mathematical witness;
closing releases live inputs and callbacks. Treat retained inputs as read-only.
"""
mutable struct MatchingInspectionSession <: _AbstractInspectionSession
    inputs::Any
    initial::NamedTuple
    payload::NamedTuple
    samples::Vector{NamedTuple}
    optimum::Any
    state::NamedTuple
    snapshot::VisualizationSpec
    max_pairs::Int
    max_candidates::Int
    matching_queries::Int
    snapshot_builds::Int
    listeners::Dict{Int,Any}
    next_listener::Int
    listener_errors::Vector{String}
    closed::Bool
    busy::Bool
    lock::ReentrantLock
end

function _matching_session(payload;inputs=nothing,samples=NamedTuple[],max_pairs=100,max_candidates=200_000)
    budget=_inspection_integer(max_pairs,:max_pairs;lower=1)
    candidates=_inspection_integer(max_candidates,:max_candidates;lower=1)
    state=(;pair=nothing,sample=nothing,mode=payload.context.scope,revision=0)
    snapshot=_matching_spec(payload.matching,payload.context;max_pairs=budget)
    return MatchingInspectionSession(inputs,payload,payload,samples,nothing,state,snapshot,budget,candidates,
        0,1,Dict{Int,Any}(),1,String[],false,false,ReentrantLock())
end

inspection_session(w::_MatchingVisualInput;max_pairs=100) = _matching_session(
    (;matching=_matching_assignment(w),context=_matching_context(w));max_pairs)

function inspection_session(a::_RawVisualBarcode,b::_RawVisualBarcode;max_pairs=100,backend=:auto)
    return inspection_session(bottleneck_matching(a,b;backend);max_pairs)
end
function inspection_session(a::OrdinaryPersistence.PersistenceDiagram,b::OrdinaryPersistence.PersistenceDiagram;
        dim,essential=:keep,essential_cap=nothing,origin=0,scale=1,backend=:auto,max_pairs=100)
    w=OrdinaryPersistence.bottleneck_matching(a,b;dim,essential,essential_cap,origin,scale,backend)
    payload=(;matching=w,context=merge(_matching_context(w),(;ordinary=(;dim,essential,essential_cap,origin,scale,
        order=OrdinaryPersistence.filtration_order(a),field_a=describe(a).field,field_b=describe(b).field))))
    return _matching_session(payload;max_pairs)
end

function _matching_slice_payload(caches,line)
    line=_inspection_slice_config(line)
    all(>(0),line.direction) || throw(ArgumentError("Matching slices require strictly positive directions."))
    arr=Fibered2D.shared_arrangement(caches[1])
    normalization=arr.normalize_dirs
    normalization in (:L1,:Linf) || throw(ArgumentError("Matching inspection requires :L1 or :Linf directions."))
    # Match the owner's actual query coordinate type, not an intended decimal.
    T=eltype(arr.box[1])
    d=T.(collect(line.direction))
    d ./= normalization === :L1 ? sum(d) : maximum(d)
    x=T.(collect(line.basepoint))
    off=-d[2]*x[1]+d[1]*x[2]
    b=off .* [-d[2],d[1]] ./ sum(abs2,d)
    query=(;basepoint=Tuple(b),direction=Tuple(d),weight=minimum(d),parameter=:normalized_line)
    bars=map(c->Fibered2D.fibered_barcode(c,d,off),caches)
    w=bottleneck_matching(bars...)
    context=(;scope=:selected_slice,status=:selected,query,box=Tuple.(arr.input_box),
        normalization,weight_convention=normalization === :L1 ? :lesnick_l1 : :lesnick_linf,
        weighted_distance=query.weight*w.distance,samples=(),sample=nothing)
    return (;matching=w,context)
end

function _matching_default_samples(arr)
    # Sampling is deliberately visible and small. Offsets are relative to the
    # projected window range for each normalized direction, not an image grid.
    box=arr.input_box
    lines=NamedTuple[]
    for degrees in (15,30,45,60,75)
        d=[cosd(degrees),sind(degrees)]
        d ./= arr.normalize_dirs === :L1 ? sum(d) : maximum(d)
        corners=[-d[2]*x+d[1]*y for x in (box[1][1],box[2][1]),y in (box[1][2],box[2][2])]
        lo,hi=extrema(corners)
        for fraction in (0.25,0.5,0.75)
            off=lo+fraction*(hi-lo)
            b=off .* [-d[2],d[1]] ./ sum(abs2,d)
            push!(lines,(;basepoint=Tuple(b),direction=Tuple(d)))
        end
    end
    return lines
end

"""
    inspection_session(a, b; opts=InvariantOptions(), max_pairs=100,
                       samples=nothing, max_candidates=200_000)

Compare compatible `EncodingResult`s on a common planar classifier and poset.
The initial view is a diagonal slice. By default 15 explicit angle/offset samples
supply a cost map; `samples=[]` disables it or supply at most 1000 line queries.
The map is a finite sample maximum, with no approximation error certificate.
`max_pairs` bounds only the drawing; matching uses every interval. Exact
optimization is never automatic and uses the explicit `max_candidates` budget.
Also accepts two `FiberedBarcodeCache2D`s sharing an arrangement, poset and field.
The cache form takes its fixed mathematical window from that arrangement.
"""
function inspection_session(a::EncodingResult,b::EncodingResult;
        opts=InvariantCore.InvariantOptions(),max_pairs=100,samples=nothing,max_candidates=200_000)
    a.P === b.P && a.pi === b.pi || throw(ArgumentError("Matching encodings must share their poset and classifier; common-encode first."))
    arr=Fibered2D.fibered_arrangement_2d(a.pi,opts;normalize_dirs=:L1,precompute=:none,threads=false)
    ca=Fibered2D.fibered_barcode_cache_2d(Results.encoding_module(a),arr;precompute=:none)
    cb=Fibered2D.fibered_barcode_cache_2d(Results.encoding_module(b),arr;precompute=:none)
    return inspection_session(ca,cb;max_pairs,samples,max_candidates)
end
function inspection_session(a::FiberedBarcodeCache2D,b::FiberedBarcodeCache2D;
        max_pairs=100,samples=nothing,max_candidates=200_000)
    Fibered2D.check_fibered_cache_pair(a,b;throw=true)
    a.M.Q === b.M.Q || throw(ArgumentError("Matching caches must share the same finite poset."))
    arr=Fibered2D.shared_arrangement(a)
    arr.normalize_dirs in (:L1,:Linf) || throw(ArgumentError("Matching inspection requires :L1 or :Linf directions."))
    lines=samples === nothing ? _matching_default_samples(arr) : collect(samples)
    length(lines) <= 1000 || throw(ArgumentError("At most 1000 sample queries are allowed per session."))
    payloads=NamedTuple[_matching_slice_payload((a,b),line) for line in lines]
    metrics=map(payloads) do payload
        c=payload.context;q=c.query;d=q.direction;basepoint=q.basepoint
        (;angle=rad2deg(atan(Float64(d[2]),Float64(d[1]))),offset=Float64(-d[2]*basepoint[1]+d[1]*basepoint[2]),
            weighted_distance=Float64(c.weighted_distance),query=q)
    end
    box=arr.input_box
    center=ntuple(i->(box[1][i]+box[2][i])/2,2)
    initial=_matching_slice_payload((a,b),(;basepoint=center,direction=(1,1)))
    initial=(;matching=initial.matching,context=merge(initial.context,(;samples=metrics)))
    s=_matching_session(initial;inputs=(a,b),samples=payloads,max_pairs,max_candidates)
    s.matching_queries=length(payloads)+1
    return s
end

inspection_selection(s::MatchingInspectionSession)=lock(s.lock) do;s.state;end
inspection_snapshot(s::MatchingInspectionSession)=lock(s.lock) do;s.snapshot;end
function inspection_summary(s::MatchingInspectionSession)
    return lock(s.lock) do
        (;kind=:matching_inspection_session,closed=s.closed,selection=s.state,revision=s.state.revision,
            pairs=length(s.snapshot.metadata.matching_records),samples=length(s.samples),
            scope=s.payload.context.scope,distance=s.payload.matching.distance,
            weighted_distance=s.payload.context.weighted_distance,matching_queries=s.matching_queries,
            optimum_available=s.optimum !== nothing,can_query_slices=s.inputs !== nothing,
            snapshot_builds=s.snapshot_builds,listener_count=length(s.listeners),
            listener_errors=Tuple(s.listener_errors),updating=s.busy)
    end
end
describe(s::MatchingInspectionSession)=inspection_summary(s)
Base.show(io::IO,s::MatchingInspectionSession)=print(io,"MatchingInspectionSession(pairs=",inspection_summary(s).pairs,
    ", scope=",s.payload.context.scope,", revision=",s.state.revision,", closed=",s.closed,")")

function _matching_candidate(s;pair=nothing,sample=nothing,slice=nothing,optimum=false,reset=false)
    _inspection_require_open(s)
    optimum isa Bool && reset isa Bool || throw(ArgumentError("optimum and reset must be Bool."))
    count(identity,(sample !== nothing,slice !== nothing,optimum,reset)) <= 1 ||
        throw(ArgumentError("Choose one of sample, slice, optimum or reset."))
    (sample === nothing && slice === nothing && !optimum) || s.inputs !== nothing ||
        throw(ArgumentError("This matching session has no module slice inputs."))
    sample === nothing || (_inspection_integer(sample,:sample;lower=1) <= length(s.samples)) ||
        throw(ArgumentError("sample must identify a retained sampled query."))
    line=slice === nothing ? nothing : _inspection_slice_config(slice)
    line === nothing || all(>(0),line.direction) || throw(ArgumentError("Matching directions must be strictly positive."))
    pair === nothing || _inspection_integer(pair,:pair)
    if sample === nothing && slice === nothing && !optimum && !reset
        _matching_options(s.payload.matching;pair,max_pairs=s.max_pairs)
    end
    return (;pair,sample,slice=line,optimum,reset)
end
function check_inspection_selection(s::MatchingInspectionSession;throw=false,kwargs...)
    return lock(s.lock) do
        issues=String[];candidate=nothing
        try;candidate=_matching_candidate(s;kwargs...)
        catch err;err isa InterruptException && rethrow();push!(issues,sprint(showerror,err));end
        valid=isempty(issues)
        throw && !valid && _throw_invalid_visual(:check_inspection_selection,issues)
        (;kind=:inspection_selection,valid,issues,selection=candidate)
    end
end

function select_inspection!(s::MatchingInspectionSession;kwargs...)
    lock(s.lock)
    try
        _inspection_require_open(s)
        s.busy && throw(ArgumentError("An inspection update or callback is already in progress."))
        c=_matching_candidate(s;kwargs...)
        s.busy=true
        try
            payload=s.payload;opt=s.optimum;queries=0
            changed=c.reset || c.sample !== nothing || c.slice !== nothing || c.optimum
            if c.reset
                payload=s.initial
            elseif c.sample !== nothing
                p=s.samples[c.sample]
                payload=(;matching=p.matching,context=merge(p.context,(;scope=:sampled_slice,status=:sampled,
                    samples=s.initial.context.samples,sample=c.sample)))
            elseif c.slice !== nothing
                p=_matching_slice_payload(s.inputs,c.slice)
                payload=(;matching=p.matching,context=merge(p.context,(;samples=s.initial.context.samples)))
                queries=1
            elseif c.optimum
                if opt === nothing
                    opt=Fibered2D.matching_distance_exact_2d(s.inputs...;witness=true,
                        weight=s.initial.context.weight_convention,max_candidates=s.max_candidates,threads=false)
                    queries=1
                end
                payload=(;matching=_matching_assignment(opt),context=merge(_matching_context(opt),(;samples=s.initial.context.samples)))
            end
            pair=c.pair === nothing ? (changed ? nothing : s.state.pair) : c.pair
            snapshot=_matching_spec(payload.matching,payload.context;pair,max_pairs=s.max_pairs)
            state=(;pair=snapshot.metadata.selected_pair,sample=payload.context.sample,
                mode=payload.context.scope,revision=s.state.revision+1)
            s.payload,s.snapshot,s.state,s.optimum=payload,snapshot,state,opt
            s.matching_queries+=queries;s.snapshot_builds+=1
        catch;s.busy=false;rethrow();end
    finally;unlock(s.lock);end
    try;_inspection_notify(s)
    finally;lock(s.lock) do;s.busy=false;end;end
    return s
end
reset_inspection!(s::MatchingInspectionSession)=select_inspection!(s;reset=true)
function close_inspection!(s::MatchingInspectionSession)
    lock(s.lock)
    try
        s.closed && return s
        s.busy && throw(ArgumentError("Cannot close during an inspection update or callback."))
        s.closed=true;s.busy=true
    finally;unlock(s.lock);end
    try;_inspection_notify(s)
    finally
        lock(s.lock) do
            empty!(s.listeners);empty!(s.samples);s.inputs=nothing;s.optimum=nothing;s.busy=false
        end
    end
    return s
end
function check_inspection_session(s::MatchingInspectionSession;throw=false)
    return lock(s.lock) do
        issues=String[]
        s.snapshot.metadata.selected_pair == s.state.pair || push!(issues,"Snapshot and selected pair disagree.")
        if s.closed && !s.busy
            isempty(s.listeners) && s.inputs === nothing || push!(issues,"Closed matching session retains callbacks or inputs.")
        end
        append!(issues,check_visual_spec(s.snapshot).issues)
        valid=isempty(issues)
        throw && !valid && _throw_invalid_visual(:check_inspection_session,issues)
        (;kind=:inspection_session,valid,issues,closed=s.closed,revision=s.state.revision)
    end
end
