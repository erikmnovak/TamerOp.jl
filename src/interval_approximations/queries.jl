# Exact support queries and finite materialization of the candidate, not a
# direct-sum certificate for the source module.
approximation_supports(r::IntervalApproximation)=deepcopy(r.rectangles)
approximation_slices(r::IntervalApproximation)=(;offsets=copy(r.offsets),barcodes=deepcopy(r.barcodes),matches=deepcopy(r.matches))
function approximation_at(r::IntervalApproximation,point)
    length(point)==2 || throw(ArgumentError("Supply two parameter coordinates."))
    p=Tuple(_number.(point))
    ids=[j for j in eachindex(r.rectangles) if any(b->_inside(p,b),r.rectangles[j])]
    (;point=p,within_window=_inside(p,r.window),dimension=length(ids),supports=ids)
end

function approximation_barcode(r::IntervalApproximation;offset)
    o=_number(offset);result=Tuple{QQ,QQ}[]
    for rectangles in r.rectangles
        intervals=sort!([_clip(b,o) for b in rectangles if _clip(b,o)[1]<_clip(b,o)[2]])
        isempty(intervals) && continue
        lo,hi=first(intervals)
        for (a,b) in intervals[2:end]
            if a<=hi;hi=max(hi,b)
            else;push!(result,(lo,hi));lo,hi=a,b;end
        end
        push!(result,(lo,hi))
    end
    sort!(result)
end

function approximation_error(r::IntervalApproximation,M::RectilinearModule;
        offsets=r.offsets,budget=IntervalApproximationBudget())
    _validate_source(M,budget)
    M.finite.field==r.metadata.field || throw(ArgumentError("Source and candidate fields differ."))
    os=_number.(collect(offsets));_limit(length(os),budget.max_lines,:max_lines)
    errors=QQ[]
    for o in os
        a=_slice(M,r.window,o,budget);b=approximation_barcode(r;offset=o)
        m=_match(a,b,budget)
        costs=QQ[j==0 ? (a[i][2]-a[i][1])/2 : max(abs(a[i][1]-b[j][1]),abs(a[i][2]-b[j][2])) for (i,j) in enumerate(m)]
        append!(costs,[(b[j][2]-b[j][1])/2 for j in eachindex(b) if !(j in m)])
        push!(errors,maximum(costs;init=QQ(0)))
    end
    (;offsets=os,distances=errors,sampled_max=isempty(errors) ? nothing : maximum(errors),
        global_bound=nothing,scope=:specified_diagonal_lines,window=r.window)
end

function approximation_module(r::IntervalApproximation;budget=IntervalApproximationBudget())
    boxes=reduce(vcat,r.rectangles;init=Tuple{NTuple{2,QQ},NTuple{2,QQ}}[])
    cuts=ntuple(i->sort!(unique!([r.window[1][i],r.window[2][i],
        [p[i] for b in boxes for p in b]...])),2)
    reps=map(c->[c[1]-1;reduce(vcat,([c[i],i==length(c) ? c[i]+1 : (c[i]+c[i+1])/2] for i in eachindex(c)))],cuts)
    _limit(prod(big.(length.(reps))),budget.max_vertices,:max_vertices)
    Q=GridPoset(map(a->collect(1:length(a)),reps))
    active=[approximation_at(r,p).supports for p in vec(collect(Iterators.product(reps...)))]
    _limit(sum(big(length(active[u]))*length(active[v]) for (u,v) in cover_edges(Q);init=big(0)),budget.max_matrix_entries,:max_matrix_entries)
    K=coeff_type(r.metadata.field)
    maps=Dict((u,v)=>K[i==j for i in active[v],j in active[u]] for (u,v) in cover_edges(Q))
    RectilinearModule(PModule{K}(Q,length.(active),maps;field=r.metadata.field),cuts;
        budget=InterleavingBudget(max_vertices=budget.max_vertices,max_matrix_entries=budget.max_matrix_entries),
        origin=(;kind=:candidate_interval_supports,source_splitting=false,window=r.window))
end

approximation_summary(r::IntervalApproximation)=(;kind=:interval_support_approximation,
    supports=length(r.rectangles),lines=length(r.offsets),bars=sum(length,r.barcodes;init=0),
    resolution=r.resolution,window=r.window,field=r.metadata.field,
    matching=r.metadata.matching,compatibility=r.metadata.compatibility,
    source_splitting=false,global_error_bound=nothing,boundaries=:lower_closed_upper_open,
    essential_bars=:clipped_at_window,geometry=:union_of_order_rectangles)
describe(r::IntervalApproximation)=approximation_summary(r)
provenance(r::IntervalApproximation)=merge(r.metadata,approximation_summary(r))
Base.show(io::IO,r::IntervalApproximation)=print(io,"IntervalApproximation(",length(r.rectangles)," candidate supports, ",length(r.offsets)," lines; no source splitting asserted)")

struct ApproximationValidationSummary
    valid::Bool
    issues::Vector{String}
end
Base.:(==)(a::ApproximationValidationSummary,b::ApproximationValidationSummary)=a.valid==b.valid && a.issues==b.issues
Base.isequal(a::ApproximationValidationSummary,b::ApproximationValidationSummary)=isequal((a.valid,a.issues),(b.valid,b.issues))
Base.hash(r::ApproximationValidationSummary,h::UInt)=hash((r.valid,r.issues),h)
describe(r::ApproximationValidationSummary)=(;valid=r.valid,issues=copy(r.issues))
Base.show(io::IO,r::ApproximationValidationSummary)=print(io,"ApproximationValidationSummary(valid=",r.valid,", issues=",r.issues,")")
function check_interval_approximation(r::IntervalApproximation;throw=false,budget=IntervalApproximationBudget())
    issues=String[]
    try
        w=_window(r.window)
        r.metadata.field isa Union{QQField,PrimeField} || error("Unsupported coefficient field.")
        r.metadata.compatibility===:unverified || error("Compatibility has not been certified.")
        r.offsets==_offsets(w,r.resolution,budget) || error("Incorrect diagonal grid.")
        fresh=_assemble(w,r.resolution,r.offsets,r.barcodes,r.matches,r.metadata,budget)
        fresh.rectangles==r.rectangles && fresh.tracks==r.tracks || error("Supports differ from the retained slice matching.")
    catch err
        err isa InterruptException && rethrow()
        push!(issues,sprint(showerror,err))
    end
    throw && !isempty(issues) && Base.throw(ArgumentError(join(issues,"; ")))
    ApproximationValidationSummary(isempty(issues),issues)
end
