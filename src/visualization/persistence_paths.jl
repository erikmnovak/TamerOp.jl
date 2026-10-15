# Bounded histories of diagrams and certified, piecewise-linear grade paths.

"""An ordered persistence history. Construct with `persistence_path`, inspect with
`path_frames`, `path_tracks`, `path_transitions`, `persistence_path_summary` and `check_persistence_path`.
Certified paths retain the update engine's pairing choices, including zero pairs
and tied events. Those choices are not canonical identities of homology classes.
Treat its storage as read-only; accessors return independent records."""
struct PersistencePath
    frames::Vector{NamedTuple}
    transitions::Vector{NamedTuple}
    degree::Int
    field::CoreModules.PrimeField
    order::Symbol
    mode::Symbol
    geometry::Union{Nothing,DataIngestion.FilteredComplex}
    native::Union{Nothing,NamedTuple}
    limits::NamedTuple
end

struct PersistencePathValidationSummary
    valid::Bool
    issues::Vector{String}
end
Base.:(==)(a::PersistencePathValidationSummary,b::PersistencePathValidationSummary)=a.valid==b.valid && a.issues==b.issues
Base.isequal(a::PersistencePathValidationSummary,b::PersistencePathValidationSummary)=a==b
Base.hash(a::PersistencePathValidationSummary,h::UInt)=hash((a.valid,Tuple(a.issues)),h)
Base.show(io::IO,v::PersistencePathValidationSummary)=print(io,"PersistencePathValidationSummary(valid=",v.valid,")")
describe(v::PersistencePathValidationSummary)=(;v.valid,issues=copy(v.issues))
"""Copy ordered frames, retaining exact path parameters, interval members,
source views, optional representatives and recorded observations."""
path_frames(p::PersistencePath)=deepcopy(p.frames)
"""Copy consecutive correspondences, with their kind, event records, ambiguity,
unmatched member IDs and matching cost. Links relate path-local track IDs."""
path_transitions(p::PersistencePath)=deepcopy(p.transitions)
"""Collect the frame indices and exact endpoints along each path-local track.
Matched samples do not certify persistence between the recorded frames."""
function path_tracks(p::PersistencePath)
    ids=sort!(unique([r.id for f in p.frames for r in f.records]))
    [(;id,points=[(;frame=i,parameter=f.parameter,birth=r.birth,death=r.death,zero=r.zero,
        birth_cell=r.birth_cell,death_cell=r.death_cell) for (i,f) in enumerate(p.frames) for r in f.records if r.id==id]) for id in ids]
end
function persistence_path_summary(p::PersistencePath)
    (;kind=:persistence_path,frames=length(p.frames),tracks=length(unique(r.id for f in p.frames for r in f.records)),
        degree=p.degree,field=p.field,order=p.order,mode=p.mode,
        events=sum(length(t.events) for t in p.transitions;init=0),
        ambiguous_transitions=count(t->t.ambiguous,p.transitions),
        representatives=any(r->r.representative!==nothing,Iterators.flatten(f.records for f in p.frames)),
        source_geometry=p.geometry!==nothing,limits=p.limits)
end
describe(p::PersistencePath)=persistence_path_summary(p)
Base.show(io::IO,p::PersistencePath)=print(io,"PersistencePath(",length(p.frames)," frames, ",p.mode,", degree=",p.degree,")")

function _path_limits(;max_frames=200,max_records=20000,max_matching_entries=1000000)
    all(x->x isa Int && x>0,(max_frames,max_records,max_matching_entries)) ||
        throw(ArgumentError("Path budgets must be positive Ints."))
    (;max_frames,max_records,max_matching_entries)
end
function _path_parameters(parameters,n)
    length(parameters)==n && n>0 || throw(ArgumentError("Provide one path parameter per nonempty input sequence."))
    all(x->x isa Real && !(x isa Bool) && isfinite(x),parameters) || throw(ArgumentError("Path parameters must be finite real values."))
    all(i->parameters[i]<parameters[i+1],1:n-1) || throw(ArgumentError("Input path parameters must increase strictly."))
    collect(parameters)
end
function _path_degree(degree)
    degree isa Int && degree>=0 || throw(ArgumentError("degree must be a nonnegative Int."))
    degree
end
function _path_push!(frames,frame,limits)
    length(frames)<limits.max_frames || throw(ArgumentError("Path exceeds max_frames, including exact event states."))
    sum(length(f.records) for f in frames;init=0)+length(frame.records)<=limits.max_records ||
        throw(ArgumentError("Path exceeds max_records; no intervals are discarded."))
    push!(frames,frame)
end
_path_record(id,b,d;birth_cell=0,death_cell=0,representative=nothing)=
    (;id,birth=b,death=d,zero=b==d,birth_cell,death_cell,representative)
_path_duplicates(records)=!allunique((r.birth,r.death) for r in records)
_path_matching_point(r,order)=order===:sublevel ? (r.birth,r.death) :
    (-_visual_exact_coordinate(r.birth),isfinite(r.death) ? -_visual_exact_coordinate(r.death) : Inf)

"""
    persistence_path(diagrams; parameters=1:length(diagrams), degree=0,
                     matching=:none, sources=nothing, observations=nothing,
                     representatives=false, max_frames=200,
                     max_records=20000, max_matching_entries=1000000)

Prepare supplied ordinary diagrams in one field, degree and filtration order.
`matching=:bottleneck` computes consecutive matchings, preserving unmatched
births/deaths and multiplicity members. These are chosen matches, not certified
continuation; nonuniqueness is not excluded. `:none` asserts no correspondence.
Optional source `VisualizationSpec`s are supplied synchronized views, not a
verified reconstruction of the diagrams. All mathematics stays within budgets.
`observations` optionally supplies one named tuple per frame, with `settings`,
`objective`, `objective_label` and/or `invariants`. These are recorded quantities,
not recomputed claims. A finite scalar objective receives its own history plot.
`representatives=true` copies witnesses already retained by the input diagrams.
It does not compute missing representatives.
"""
function persistence_path(diagrams::AbstractVector{<:OrdinaryPersistence.PersistenceDiagram};
        parameters=1:length(diagrams),degree=0,matching=:none,sources=nothing,observations=nothing,representatives=false,
        max_frames=200,max_records=20000,max_matching_entries=1000000)
    limits=_path_limits(;max_frames,max_records,max_matching_entries);_path_degree(degree)
    representatives isa Bool || throw(ArgumentError("representatives must be Bool."))
    ts=_path_parameters(parameters,length(diagrams))
    length(ts)<=max_frames || throw(ArgumentError("Path exceeds max_frames."))
    matching in (:none,:bottleneck) || throw(ArgumentError("matching must be :none or :bottleneck; supplied diagrams cannot certify continuation."))
    sources===nothing || length(sources)==length(ts) || throw(ArgumentError("Provide one source view per diagram."))
    observations===nothing || length(observations)==length(ts) || throw(ArgumentError("Provide one observation record per diagram."))
    O=OrdinaryPersistence;field=O.field(first(diagrams));order=O.filtration_order(first(diagrams))
    frames=NamedTuple[];transitions=NamedTuple[];nextid=0
    for (i,d) in enumerate(diagrams)
        observation=observations===nothing ? (;) : deepcopy(observations[i])
        _path_observation(observation)
        O.check_persistence_diagram(d;throw=true)
        O.field(d)==field && O.filtration_order(d)==order || throw(ArgumentError("A path cannot change field or filtration orientation."))
        count=length(O.finite_intervals(d;dim=degree))+length(O.essential_births(d;dim=degree))
        sum(length(f.records) for f in frames;init=0)+count<=max_records || throw(ArgumentError("Path exceeds max_records."))
        records=NamedTuple[]
        for kind in (:finite,:essential)
            values=kind===:finite ? O.finite_intervals(d;dim=degree) : O.essential_births(d;dim=degree)
            for (j,v) in enumerate(values)
                b,e=kind===:finite ? v : (v,order===:sublevel ? Inf : -Inf)
                rep=representatives ? O.persistence_representative(d;dim=degree,kind,index=j) : nothing
                push!(records,_path_record(0,b,e;representative=rep!==nothing && rep.available ? rep : nothing))
            end
        end
        sum(length(f.records) for f in frames;init=0)+length(records)<=max_records || throw(ArgumentError("Path exceeds max_records."))
        witness=nothing;links=Tuple{Int,Int}[]
        if i>1 && matching===:bottleneck
            previous=frames[end].records
            (big(length(previous))+length(records))^2<=max_matching_entries || throw(ArgumentError("Matching exceeds max_matching_entries."))
            # Reflect superlevel coordinates into a sublevel barcode for the matcher.
            witness=bottleneck_matching([_path_matching_point(r,order) for r in previous],[_path_matching_point(r,order) for r in records])
            # Vector input order and distinct multiplicity members are retained.
            for (a,b) in enumerate(witness.a_to_b)
                b==0 && continue
                k=b;id=previous[a].id
                records[k]=merge(records[k],(;id));push!(links,(id,id))
            end
        end
        for k in eachindex(records)
            records[k].id==0 || continue
            nextid+=1;records[k]=merge(records[k],(;id=nextid))
        end
        source=sources===nothing ? nothing : deepcopy(sources[i])
        if source!==nothing
            source isa VisualizationSpec && !get(source.interaction,:requires_live_julia,false) ||
                throw(ArgumentError("Sources must be static VisualizationSpecs."))
            check_visual_spec(source;throw=true)
        end
        if i>1
            push!(transitions,(;kind=matching===:none ? :unrelated : :matched,links,
                events=NamedTuple[],ambiguous=matching!==:none,
                distance=witness===nothing ? nothing : witness.distance,
                unmatched_before=[r.id for r in frames[end].records if !(r.id in first.(links))],
                unmatched_after=[r.id for r in records if !(r.id in last.(links))]))
        end
        _path_push!(frames,(;parameter=ts[i],records,source,grades=nothing,keyframe=i,events=NamedTuple[],
            context=deepcopy(O.provenance(d)),observation),limits)
    end
    PersistencePath(frames,transitions,degree,field,order,matching===:none ? :sequence : :matched,nothing,nothing,limits)
end

function _path_observation(x)
    x isa NamedTuple && all(k->k in (:settings,:objective,:objective_label,:invariants),keys(x)) ||
        throw(ArgumentError("Observations are named tuples of settings, objective, objective_label and/or invariants."))
    value=get(x,:objective,nothing)
    value===nothing || (value isa Real && !(value isa Bool) && isfinite(value)) ||
        throw(ArgumentError("A recorded objective must be a finite real value or nothing."))
    label=get(x,:objective_label,"Objective")
    label isa AbstractString && !isempty(label) || throw(ArgumentError("objective_label must be nonempty text."))
    Serialization.visual_data_to_json(x)
    nothing
end

function _path_native_records(pairs,ids,grades,order;state=nothing,representatives=false)
    O=OrdinaryPersistence
    map(sort!(collect(values(pairs));by=p->ids[p.birth_cell])) do p
        b=grades[p.birth_cell];d=p.death_cell==0 ? (order===:sublevel ? Inf : -Inf) : grades[p.death_cell]
        rep=nothing
        if representatives && state!==nothing && b!=d
            cycle=O.persistence_column(state;cell=p.death_cell==0 ? p.birth_cell : p.death_cell,
                kind=p.death_cell==0 ? :basis : :reduced)
            filling=p.death_cell==0 ? nothing : O.persistence_column(state;cell=p.death_cell,kind=:basis)
            rep=(;available=true,field=O.field(state),cycle,bounding_chain=filling,choice=:retained_reduction,
                indexing=:stable_cells)
        end
        _path_record(ids[p.birth_cell],b,d;birth_cell=p.birth_cell,death_cell=p.death_cell,representative=rep)
    end
end

"""
    persistence_path(complex; grades, parameters=0:length(grades), degree=0,
                     field=F2(), order=:sublevel, representatives=false, ...)

Follow successive piecewise-linear grade changes of one based chain complex.
Each vector in `grades` contains the next grade of every cell, in global cell
order. The initial grading belongs to `complex`. Exact crossing states are
inserted automatically. Uses `update_persistence!` and its retained reduction and transpositions;
tie order is deterministic, not a unique homology-class identity. Prime fields
and one parameter are supported. A `FilteredComplex` also retains source geometry.
Optional representatives are available at the supplied keyframes only.
`max_frames` includes event states; `max_records` bounds all retained pairs.
`budget` bounds each reduction/update, including its event trace.
"""
function persistence_path(input::Union{DataTypes.GradedComplex,DataIngestion.FilteredComplex};grades,
        parameters=0:length(grades),degree=0,field=CoreModules.F2(),order=nothing,representatives=false,
        max_frames=200,max_records=20000,max_matching_entries=1000000,
        budget=OrdinaryPersistence.PersistenceUpdateBudget(max_cells=2000,max_swaps=10000,max_entries=1000000))
    limits=_path_limits(;max_frames,max_records,max_matching_entries);_path_degree(degree)
    representatives isa Bool || throw(ArgumentError("representatives must be Bool."))
    budget isa OrdinaryPersistence.PersistenceUpdateBudget || throw(ArgumentError("budget must be a PersistenceUpdateBudget."))
    original=input isa DataIngestion.FilteredComplex ? DataIngestion.graded_complex(input) : input
    length(original.grades)<=budget.max_cells || throw(ArgumentError("Path source exceeds max_cells."))
    length(grades)+1<=max_frames || throw(ArgumentError("Path exceeds max_frames before event insertion."))
    geometry=input isa DataIngestion.FilteredComplex ? deepcopy(input) : nothing
    G=geometry===nothing ? input : DataIngestion.graded_complex(geometry)
    DataTypes.parameter_dim(G)==1 || throw(ArgumentError("Certified grade paths require a one-parameter complex."))
    inferred=geometry===nothing || geometry.orientation[1]==1 ? :sublevel : :superlevel
    order=order===nothing ? inferred : order
    geometry===nothing || order===inferred || throw(ArgumentError("order disagrees with source geometry."))
    geometry===nothing || DataIngestion.check_filtered_complex(geometry;throw=true)
    ts=_path_parameters(parameters,length(grades)+1)
    length(ts)<=max_frames || throw(ArgumentError("Path exceeds max_frames before event insertion."))
    rational(x)=x isa Union{Integer,Rational,AbstractFloat} && !(x isa Bool) && isfinite(x) ?
        Rational{BigInt}(x) : throw(ArgumentError("Certified paths require finite integer, rational or floating coordinates."))
    ts=rational.(ts);start=rational.(first.(G.grades));n=length(start)
    targets=[rational.(collect(g)) for g in grades]
    all(g->length(g)==n,targets) || throw(ArgumentError("Each grade vector must cover every global cell."))
    G0=DataTypes.GradedComplex(deepcopy(G.cells_by_dim),deepcopy(G.boundaries),[(g,) for g in start])
    O=OrdinaryPersistence;s=O.persistence_state(G0;field,order,retain_reduction=true,budget)
    frames=NamedTuple[];transitions=NamedTuple[]
    try
        pairs=Dict(p.birth_cell=>p for p in O.persistence_pairing(s) if p.dimension==degree)
        ids=Dict(id=>i for (i,id) in enumerate(sort!(collect(keys(pairs)))))
        records=_path_native_records(pairs,ids,start,order;state=s,representatives)
        context=(;source=:fixed_complex,essential_interpretation=:constant_extension)
        _path_push!(frames,(;parameter=ts[1],records,source=nothing,grades=copy(start),keyframe=1,events=NamedTuple[],context,observation=(;)),limits)
        for (segment,target) in enumerate(targets)
            # Validate geometric faces too: coefficients may cancel in this field.
            if geometry!==nothing
                faces=DataIngestion.face_incidence(geometry)
                faces===nothing || all(i->all(j->order===:sublevel ? target[j]<=target[i] : target[j]>=target[i],faces[i]),1:n) ||
                    throw(ArgumentError("A target grading violates geometric face incidence."))
            end
            u=O.update_persistence!(s;grades=Dict(i=>target[i] for i in 1:n),continuation=true)
            O.check_persistence_update(u;throw=true)
            cursor=1
            while cursor<=length(u.events)
                t=u.events[cursor].time;events=NamedTuple[]
                while cursor<=length(u.events) && u.events[cursor].time==t
                    e=u.events[cursor];push!(events,O.describe(e))
                    old=copy(ids)
                    for p in e.before
                        p.dimension==degree || continue
                        get(pairs,p.birth_cell,nothing)==p || error("Update event does not continue the current pairing.")
                        delete!(pairs,p.birth_cell);delete!(ids,p.birth_cell)
                    end
                    for (a,b) in e.continuation
                        haskey(old,a) && (ids[b]=old[a])
                    end
                    for p in e.after;p.dimension==degree && (pairs[p.birth_cell]=p);end
                    cursor+=1
                end
                values=(1-t).*start.+t.*target
                records=_path_native_records(pairs,ids,values,order)
                links=[(r.id,r.id) for r in records]
                push!(transitions,(;kind=:certified,links,events,ambiguous=any(e->e.tie!==:isolated,events)||_path_duplicates(records),
                    distance=nothing,unmatched_before=Int[],unmatched_after=Int[]))
                _path_push!(frames,(;parameter=(1-t)*ts[segment]+t*ts[segment+1],records,source=nothing,
                    grades=values,keyframe=0,events,context,observation=(;)),limits)
            end
            actual=Dict(p.birth_cell=>p for p in O.persistence_pairing(s) if p.dimension==degree)
            actual==pairs || error("Event trace and final reduction disagree.")
            records=_path_native_records(pairs,ids,target,order;state=s,representatives)
            push!(transitions,(;kind=:certified,links=[(r.id,r.id) for r in records],events=NamedTuple[],
                ambiguous=_path_duplicates(records),distance=nothing,unmatched_before=Int[],unmatched_after=Int[]))
            _path_push!(frames,(;parameter=ts[segment+1],records,source=nothing,grades=copy(target),keyframe=segment+1,events=NamedTuple[],context,observation=(;)),limits)
            start=target
        end
    finally;O.close_persistence!(s);end
    native=(;complex=G0,grades=targets,parameters=ts,representatives,budget)
    PersistencePath(frames,transitions,degree,field,order,:certified,geometry,native,limits)
end

"""Validate stored frames and correspondences. For a certified path this also
replays its bounded native construction, independently of the stored frames.
No rendering or browser work is performed."""
function check_persistence_path(p::PersistencePath;throw=false)
    issues=String[]
    try
        _path_degree(p.degree);_path_limits(;p.limits...)
        p.mode in (:sequence,:matched,:certified) || error("Unknown path mode.")
        p.order in (:sublevel,:superlevel) || error("Invalid filtration orientation.")
        0<length(p.frames)<=p.limits.max_frames || error("Invalid frame count.")
        length(p.transitions)==length(p.frames)-1 || error("Transition count disagrees.")
        sum(length(f.records) for f in p.frames)<=p.limits.max_records || error("Record budget exceeded.")
        for (i,f) in enumerate(p.frames)
            f.context isa NamedTuple || error("Missing frame provenance.")
            _path_observation(f.observation)
            f.parameter isa Real && !(f.parameter isa Bool) && isfinite(f.parameter) || error("Invalid path parameter.")
            i==1 || p.frames[i-1].parameter<=f.parameter || error("Unordered path parameters.")
            allunique(r.id for r in f.records) || error("Duplicate track ID in a frame.")
            for r in f.records
                r.id isa Int && r.id>0 && r.birth isa Real && !(r.birth isa Bool) &&
                    r.death isa Real && !(r.death isa Bool) && isfinite(r.birth) && !isnan(r.death) || error("Invalid interval record.")
                (p.order===:sublevel ? r.birth<=r.death : r.death<=r.birth) && r.zero==(r.birth==r.death) || error("Invalid oriented endpoints.")
            end
            f.source===nothing || check_visual_spec(f.source;throw=true)
            i==1 && continue
            t=p.transitions[i-1];a=Set(r.id for r in p.frames[i-1].records);b=Set(r.id for r in f.records)
            allunique(first.(t.links)) && allunique(last.(t.links)) || error("Correspondence is not a partial bijection.")
            all(x->first(x) in a && last(x) in b && first(x)==last(x),t.links) || error("Invalid linked track IDs.")
            Set(first.(t.links))==intersect(a,b) || error("Shared track IDs lack their correspondence.")
            allunique(t.unmatched_before) && allunique(t.unmatched_after) || error("Repeated unmatched member.")
            Set(t.unmatched_before)==setdiff(a,Set(first.(t.links))) && Set(t.unmatched_after)==setdiff(b,Set(last.(t.links))) || error("Unmatched members disagree.")
            t.kind===(p.mode===:certified ? :certified : p.mode===:matched ? :matched : :unrelated) || error("Transition claim disagrees with path mode.")
            p.mode!==:sequence || isempty(t.links) || error("Unrelated frames claim tracks.")
            if p.mode===:matched
                # Audit the actual partial matching and objective, not merely
                # the fact that endpoint pictures share colors.
                left=Dict(r.id=>r for r in p.frames[i-1].records);right=Dict(r.id=>r for r in f.records)
                (big(length(left))+length(right))^2<=p.limits.max_matching_entries || error("Matching budget exceeded.")
                point(r)=_path_matching_point(r,p.order)
                cost=maximum(vcat([_matching_pair_cost(point(left[a]),point(right[b])) for (a,b) in t.links],
                    [_matching_diagonal_cost(point(left[a])) for a in t.unmatched_before],
                    [_matching_diagonal_cost(point(right[b])) for b in t.unmatched_after]);init=0)
                optimum=bottleneck_distance([point(r) for r in values(left)],[point(r) for r in values(right)])
                same(a,b)=a==b || isapprox(Float64(a),Float64(b);rtol=8eps(Float64),atol=0)
                same(cost,t.distance) && same(cost,optimum) || error("Stored correspondence is not a bottleneck optimum.")
                t.ambiguous || error("A matched path cannot certify unique identity.")
            end
        end
        if p.mode===:certified
            p.native===nothing && error("Missing native construction.")
            q=if haskey(p.native,:slice_scalarization)
                line=p.native.slice_scalarization
                slice_path(line.source;basepoints=line.basepoints,direction=line.direction,parameters=line.parameters,
                    degree=p.degree,field=p.field,representatives=p.native.representatives,budget=p.native.budget,
                    max_frames=p.limits.max_frames,max_records=p.limits.max_records)
            else
                persistence_path(p.geometry===nothing ? p.native.complex : p.geometry;
                    grades=p.native.grades,parameters=p.native.parameters,degree=p.degree,field=p.field,order=p.order,
                    representatives=p.native.representatives,budget=p.native.budget,p.limits...)
            end
            q.frames==p.frames && q.transitions==p.transitions || error("Stored path disagrees with native continuation.")
        else
            p.native===nothing || error("Supplied path claims a native certificate.")
        end
    catch err
        err isa InterruptException && rethrow();push!(issues,sprint(showerror,err))
    end
    result=PersistencePathValidationSummary(isempty(issues),issues)
    throw && !result.valid && Base.throw(ArgumentError(join(issues,"; ")))
    result
end
