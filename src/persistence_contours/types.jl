# Exact ambient coordinates, explicit directions, and bounded work contracts.
struct ContourBudget
    max_generators::Int
    max_vertices::Int
    max_events::Int
    max_work::Int
    function ContourBudget(;max_generators=100_000,max_vertices=100_000,
            max_events=100_000,max_work=10_000_000)
        xs=(max_generators,max_vertices,max_events,max_work)
        all(x->x isa Integer && !(x isa Bool) && 0<=x<=typemax(Int),xs) ||
            throw(ArgumentError("contour budgets must be nonnegative integers fitting Int"))
        new(Int.(xs)...)
    end
end
_limit(n,b,label)=n<=b || throw(ArgumentError("contour $label budget exceeded ($n > $b)"))
function _coordinate(x)
    x isa Union{Integer,Rational,AbstractFloat} && !(x isa Bool) && isfinite(x) ||
        throw(ArgumentError("contour coordinates must be finite real numbers"))
    QQ(x)
end
function _point(p)
    p isa Union{Tuple,AbstractVector} && length(p)==2 || throw(ArgumentError("supply two ambient coordinates"))
    Tuple(_coordinate.(p))
end
_precedes(p,q)=p[1]<=q[1] && p[2]<=q[2]
"""The exact contour C(x,t)=x+t*direction on R^2, for t>=0.
Direction coordinates must be nonnegative. Zero directions and axis-aligned
directions are permitted. Coordinate units are those of the supplied module.
"""
struct LinearContour
    direction::NTuple{2,QQ}
    function LinearContour(direction)
        v=_point(direction);all(>=(0),v) || throw(ArgumentError("a linear contour direction must be nonnegative"))
        new(v)
    end
end
function contour_point(c::LinearContour;point,scale)
    p=_point(point);t=_coordinate(scale);t>=0 || throw(ArgumentError("contour scale must be nonnegative"))
    p.+t.*c.direction
end
describe(c::LinearContour)=(;kind=:linear_contour,direction=c.direction,domain=:real_bivariate,
    scale_domain=:nonnegative,identity=true,order_preserving=true,semigroup_law=:equality)
Base.show(io::IO,c::LinearContour)=print(io,"LinearContour(",c.direction,")")

"""A finitely presented interval U\\V on R^2, where U and V are finitely
generated upsets. Birth and killing boundaries are included in their upsets;
thus a killed boundary point has zero stalk. Construction verifies connected
order-convex support. Empty support is permitted. The coefficient field does
not affect this interval's shift dimension.
"""
struct BivariateInterval{F}
    generators::Vector{NTuple{2,QQ}}
    relations::Vector{NTuple{2,QQ}}
    field::F
    function BivariateInterval(generators,relations=();field=QQField(),budget::ContourBudget=ContourBudget())
        field isa Union{QQField,PrimeField} || throw(ArgumentError("use an exact rational or prime coefficient field"))
        _limit(big(length(generators))+length(relations),budget.max_generators,"presentation corners")
        g=_minimal(_point.(collect(generators)));d=_minimal(_point.(collect(relations)))
        all(p->_in_upset(g,p),d) || throw(ArgumentError("every relation must lie in the generated upset"))
        filter!(p->!_in_upset(d,p),g)
        d=_intersect_upsets(g,d)
        all(i->!_in_upset(d,(g[i+1][1],g[i][2])),1:length(g)-1) ||
            throw(ArgumentError("the declared support is disconnected; stable rank is not computed by summing its pieces"))
        new{typeof(field)}(g,d,field)
    end
end
field(I::BivariateInterval)=I.field
describe(I::BivariateInterval)=(;kind=:bivariate_interval,generators=copy(I.generators),
    relations=copy(I.relations),field=I.field,ambient=:real_bivariate,
    support=:generated_upset_minus_killed_upset,boundaries=:birth_included_death_excluded)
Base.show(io::IO,I::BivariateInterval)=print(io,"BivariateInterval(",length(I.generators)," generators, ",length(I.relations)," killing corners)")

struct ShiftDimensionResult{I,W}
    interval::I
    shift::NTuple{2,QQ}
    value::Int
    witnesses::W
    statistics::NamedTuple
end
shift_dimension_value(r::ShiftDimensionResult)=r.value
function shift_generators(r::ShiftDimensionResult)
    r.witnesses===nothing && throw(ArgumentError("request witnesses=true to retain a minimal shifted generating family"))
    copy(r.witnesses.generators)
end
describe(r::ShiftDimensionResult)=(;kind=:shift_dimension,value=r.value,shift=r.shift,
    status=:exact,field=r.interval.field,route=:bivariate_interval_greedy,
    witnesses=r.witnesses!==nothing,statistics=r.statistics)
provenance(r::ShiftDimensionResult)=(;definition=:minimum_homogeneous_generators_containing_shifted_image,
    source=describe(r.interval),shift=r.shift,status=:exact)
Base.show(io::IO,r::ShiftDimensionResult)=print(io,"ShiftDimensionResult(",r.value,", shift=",r.shift,", exact)")

struct StableRankCurve{I,W}
    interval::I
    contour::LinearContour
    scales::Vector{QQ}
    values::Vector{Int}
    witnesses::W
    scope::Symbol
    statistics::NamedTuple
end
stable_rank_values(r::StableRankCurve)=(;scales=copy(r.scales),values=copy(r.values),scope=r.scope,status=:exact)
function stable_rank_at(r::StableRankCurve;scale)
    t=_coordinate(scale);t>=0 || throw(ArgumentError("scale must be nonnegative"))
    i=r.scope===:all_scales ? searchsortedlast(r.scales,t) : something(findfirst(==(t),r.scales),0)
    i>0 || throw(ArgumentError("this curve retains exact samples only; request one of its supplied scales"))
    r.values[i]
end
describe(r::StableRankCurve)=(;kind=:stable_rank,contour=describe(r.contour),scope=r.scope,
    scales=copy(r.scales),values=copy(r.values),status=:exact,continuity=:right,
    witnesses=r.witnesses!==nothing,statistics=r.statistics)
provenance(r::StableRankCurve)=(;definition=:linear_contour_stabilization_of_generator_number,
    source=describe(r.interval),contour=describe(r.contour),scope=r.scope,status=:exact)
Base.show(io::IO,r::StableRankCurve)=print(io,"StableRankCurve(",length(r.scales)," values, ",r.scope,")")

struct ContourValidationSummary
    valid::Bool
    issues::Vector{String}
    optimality_checked::Bool
end
Base.:(==)(a::ContourValidationSummary,b::ContourValidationSummary)=
    (a.valid,a.issues,a.optimality_checked)==(b.valid,b.issues,b.optimality_checked)
Base.isequal(a::ContourValidationSummary,b::ContourValidationSummary)=
    isequal((a.valid,a.issues,a.optimality_checked),(b.valid,b.issues,b.optimality_checked))
Base.hash(a::ContourValidationSummary,h::UInt)=hash((a.valid,Tuple(a.issues),a.optimality_checked),h)
describe(r::ContourValidationSummary)=(;valid=r.valid,issues=copy(r.issues),optimality_checked=r.optimality_checked)
Base.show(io::IO,r::ContourValidationSummary)=print(io,"ContourValidationSummary(valid=",r.valid,", optimality_checked=",r.optimality_checked,")")
function _validation(f;throw=false)
    issues=String[];checked=false
    try;checked=f()
    catch err;err isa InterruptException && rethrow();push!(issues,sprint(showerror,err));end
    throw && !isempty(issues) && Base.throw(ArgumentError(join(issues,"; ")))
    ContourValidationSummary(isempty(issues),issues,checked)
end
check_contour(c::LinearContour;throw=false)=_validation(;throw) do
    LinearContour(c.direction);false
end
