# Owned exact inputs and event-indexed results. Times never identify vertices.
struct ZigzagBudget
    max_vertices::Int
    max_dimension::Int
    max_matrix_entries::Int
    max_representative_entries::Int
    max_cells::Int
    function ZigzagBudget(; max_vertices=10_000, max_dimension=2048,
                         max_matrix_entries=2_000_000,
                         max_representative_entries=1_000_000, max_cells=100_000)
        values = (max_vertices,max_dimension,max_matrix_entries,max_representative_entries,max_cells)
        all(x -> x isa Integer && !(x isa Bool) && 0 <= x <= typemax(Int), values) ||
            throw(ArgumentError("zigzag budgets must be nonnegative integers fitting Int"))
        new(Int.(values)...)
    end
end

function _bound(value, limit, name)
    value <= limit || throw(ArgumentError("zigzag $name budget exceeded ($value > $limit)"))
end

function _exact_field(F)
    F isa Union{QQField,PrimeField} ||
        throw(ArgumentError("zigzag persistence requires QQField() or a prime field"))
    F
end

function _dimension(x)
    x isa Integer && !(x isa Bool) && 0 <= x <= typemax(Int) ||
        throw(ArgumentError("dimensions must be nonnegative integers fitting Int"))
    Int(x)
end

function _matrix(F, A)
    A isa AbstractMatrix || throw(ArgumentError("zigzag maps must be matrices"))
    K = coeff_type(F)
    out = Matrix{K}(undef,size(A))
    for i in eachindex(A)
        x = A[i]
        if x isa FpElem
            typeof(x) === K || throw(ArgumentError("matrix coefficient belongs to a different field"))
            out[i] = x
        elseif x isa Union{Integer,Rational}
            out[i] = coerce(F,x)
        else
            throw(ArgumentError("zigzag coefficients must be exact integers, rationals or elements of the specified field"))
        end
    end
    out
end

function _directions(directions, n)
    length(directions) == n-1 || throw(ArgumentError("one direction is required per adjacent pair"))
    all(d -> d === :forward || d === :backward, directions) ||
        throw(ArgumentError("directions must be :forward or :backward"))
    Symbol[d for d in directions]
end

function _times(times, n)
    length(times) == n || throw(ArgumentError("one time label is required per vertex"))
    n > 0 || throw(ArgumentError("a zigzag needs at least one vertex"))
    all(t -> t isa Real && !(t isa Bool) && isfinite(t), times) ||
        throw(ArgumentError("time labels must be finite real values"))
    T = mapreduce(typeof,promote_type,times)
    isconcretetype(T) || throw(ArgumentError("time labels need a concrete common type"))
    ts = T[t for t in times]
    all(ts[i] == times[i] for i in eachindex(ts)) ||
        throw(ArgumentError("time conversion loses precision"))
    issorted(ts) || throw(ArgumentError("time labels must be nondecreasing in event order"))
    ts
end

struct ZigzagModule{K,F,T}
    dims::Vector{Int}
    maps::Vector{Matrix{K}}
    directions::Vector{Symbol}
    times::Vector{T}
    field::F
end

function ZigzagModule(dims, maps; directions, field=F2(), times=nothing,
                      budget::ZigzagBudget=ZigzagBudget())
    _exact_field(field)
    n=length(dims)
    _bound(n,budget.max_vertices,"vertices")
    ts=_times(times===nothing ? (1:n) : times,n); ds=_directions(directions,n)
    length(maps)==n-1 || throw(ArgumentError("one map is required per adjacent pair"))
    dd=Int[_dimension(d) for d in dims]
    _bound(maximum(dd),budget.max_dimension,"dimension")
    entries=big(0)
    for i in eachindex(maps)
        s,t = ds[i] === :forward ? (i,i+1) : (i+1,i)
        maps[i] isa AbstractMatrix && size(maps[i])==(dd[t],dd[s]) ||
            throw(ArgumentError("map $i has the wrong source/target shape for $(ds[i])"))
        entries += big(dd[s])*dd[t]
    end
    _bound(entries+6big(maximum(dd))^2,budget.max_matrix_entries,"dense algebra entries")
    K=coeff_type(field)
    ZigzagModule{K,typeof(field),eltype(ts)}(dd,Matrix{K}[_matrix(field,A) for A in maps],ds,ts,field)
end

dimensions(z::ZigzagModule)=copy(z.dims)
field(z::ZigzagModule)=z.field
zigzag_times(z::ZigzagModule)=copy(z.times)
zigzag_directions(z::ZigzagModule)=copy(z.directions)
function zigzag_map(z::ZigzagModule; edge::Integer)
    edge isa Bool && throw(ArgumentError("edge must be an integer index"))
    1<=edge<=length(z.maps) || throw(ArgumentError("edge index outside the zigzag"))
    copy(z.maps[edge])
end
describe(z::ZigzagModule)=(kind=:zigzag_module,vertices=length(z.dims),dimensions=Tuple(z.dims),
    directions=Tuple(z.directions),field=z.field,domain=:ordered_occurrences)
Base.show(io::IO,z::ZigzagModule)=print(io,"ZigzagModule(",length(z.dims)," vertices, ",z.field,")")

# Inclusive occurrence support is authoritative. Decorations use the source
# space on each open directed edge. Repeated time labels are never quotiented,
# and a terminal endpoint is censored, not an inferred essential death.
struct ZigzagInterval{T}
    first::Int
    last::Int
    left::T
    right::T
    left_closed::Bool
    right_closed::Bool
    left_kind::Symbol
    right_kind::Symbol
end
Base.:(==)(a::ZigzagInterval,b::ZigzagInterval)=all(getfield(a,k)==getfield(b,k) for k in fieldnames(typeof(a)))
Base.isequal(a::ZigzagInterval,b::ZigzagInterval)=all(isequal(getfield(a,k),getfield(b,k)) for k in fieldnames(typeof(a)))
Base.hash(b::ZigzagInterval,h::UInt)=hash(Tuple(getfield(b,k) for k in fieldnames(typeof(b))),h)
Base.show(io::IO,b::ZigzagInterval)=print(io,"ZigzagInterval(",b.first,":",b.last,", ",
    b.left_closed ? "[" : "(",b.left,", ",b.right,b.right_closed ? "]" : ")",")")
describe(b::ZigzagInterval)=(first=b.first,last=b.last,left=b.left,right=b.right,
    left_closed=b.left_closed,right_closed=b.right_closed,left_kind=b.left_kind,right_kind=b.right_kind)

function _interval(z,a,b)
    leftkind = a==1 ? :initial : z.directions[a-1]===:forward ? :forward_birth : :backward_birth
    rightkind = b==length(z.dims) ? :terminal : z.directions[b]===:forward ? :forward_death : :backward_death
    li = leftkind===:backward_birth ? a-1 : a
    ri = rightkind===:forward_death ? b+1 : b
    ZigzagInterval(a,b,z.times[li],z.times[ri],leftkind!==:backward_birth,
                   rightkind!==:forward_death,leftkind,rightkind)
end

struct ZigzagResult{K,Z,S}
    diagram::Z
    supports::Vector{Tuple{Int,Int}}
    representatives::Union{Nothing,Vector{Vector{Vector{K}}}}
    source::S
end
zigzag_intervals(r::ZigzagResult)=[_interval(r.diagram,a,b) for (a,b) in r.supports]
dimensions(r::ZigzagResult)=dimensions(r.diagram)
field(r::ZigzagResult)=field(r.diagram)
zigzag_times(r::ZigzagResult)=zigzag_times(r.diagram)
zigzag_directions(r::ZigzagResult)=zigzag_directions(r.diagram)
zigzag_map(r::ZigzagResult;edge::Integer)=zigzag_map(r.diagram;edge)
function zigzag_summary(r::ZigzagResult)
    (kind=:zigzag_persistence,vertices=length(r.diagram.dims),intervals=length(r.supports),
     field=field(r),representatives=r.representatives!==nothing,
     domain=:ordered_occurrences,terminal=:censored,source=r.source===nothing ? :linear_maps : :chain_maps)
end
describe(r::ZigzagResult)=zigzag_summary(r)
Base.show(io::IO,r::ZigzagResult)=print(io,"ZigzagResult(",length(r.supports)," intervals, ",field(r),")")
provenance(r::ZigzagResult)=(algorithm=:right_filtration,field=field(r),domain=:ordered_occurrences,
    repeated_times=:distinct_events,terminal=:censored,representatives=:noncanonical,
    endpoint_convention=:directed_edge_extension,
    degree=r.source===nothing ? nothing : r.source.degree)

function zigzag_representative(r::ZigzagResult; interval::Integer, vertex::Integer)
    (interval isa Bool || vertex isa Bool) && throw(ArgumentError("indices must not be Bool"))
    1<=interval<=length(r.supports) || throw(ArgumentError("interval index outside the barcode"))
    a,b=r.supports[interval]
    a<=vertex<=b || throw(ArgumentError("vertex is outside the selected interval"))
    r.representatives===nothing && throw(ArgumentError("request representatives=true when computing the zigzag"))
    copy(r.representatives[interval][vertex-a+1])
end

struct ZigzagValidationSummary{R}
    report::R
end
describe(s::ZigzagValidationSummary)=s.report
Base.getproperty(s::ZigzagValidationSummary,k::Symbol)=k===:report ? getfield(s,:report) : getproperty(getfield(s,:report),k)
Base.propertynames(s::ZigzagValidationSummary,private::Bool=false)=private ? (:report,keys(s.report)...) : keys(s.report)
Base.:(==)(a::ZigzagValidationSummary,b::ZigzagValidationSummary)=a.report==b.report
Base.isequal(a::ZigzagValidationSummary,b::ZigzagValidationSummary)=isequal(a.report,b.report)
Base.hash(s::ZigzagValidationSummary,h::UInt)=hash(s.report,h)
Base.show(io::IO,s::ZigzagValidationSummary)=print(io,"ZigzagValidationSummary(valid=",s.valid,", issues=",s.issues,")")

function _validation(f,do_throw)
    issues=String[]
    try
        f()
    catch e
        e isa Union{ArgumentError,DimensionMismatch,BoundsError} || rethrow()
        push!(issues,sprint(showerror,e))
    end
    do_throw && !isempty(issues) && throw(ArgumentError(join(issues,"; ")))
    ZigzagValidationSummary((valid=isempty(issues),issues=Tuple(issues)))
end

function check_zigzag(z::ZigzagModule; throw::Bool=false,budget::ZigzagBudget=ZigzagBudget())
    _validation(throw) do
        ZigzagModule(z.dims,z.maps;directions=z.directions,field=z.field,times=z.times,budget)
    end
end

function check_zigzag(r::ZigzagResult; throw::Bool=false,budget::ZigzagBudget=ZigzagBudget())
    _validation(throw) do
        z=r.diagram
        check_zigzag(z;throw=true,budget)
        n=length(z.dims)
        all(ab->1<=ab[1]<=ab[2]<=n,r.supports) || Base.throw(ArgumentError("invalid interval support"))
        [count(ab->ab[1]<=i<=ab[2],r.supports) for i in 1:n]==z.dims ||
            Base.throw(ArgumentError("intervals do not reconstruct dimensions"))
        if r.representatives===nothing
            expected=zigzag_persistence(z;budget)
            r.supports==expected.supports || Base.throw(ArgumentError("intervals disagree with directed input maps"))
        else
            _check_representatives(r,budget)
        end
    end
end
