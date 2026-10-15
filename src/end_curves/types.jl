# Exact curve supports, bounded computation and inspectable result contracts.
struct EndCurveBudget
    decomposition::IntervalDecompositions.IntervalDecompositionBudget
end
"""Bound finite-grid and exact decomposition work; exhausted searches throw.
Keywords are those of `IntervalDecompositionBudget`. The vertex bound includes
the one-cell upper/right zero halo needed for death curves and Betti data.
"""
EndCurveBudget(;kwargs...)=EndCurveBudget(IntervalDecompositions.IntervalDecompositionBudget(;kwargs...))
_bound(n,b,label)=n<=b || throw(ArgumentError("end-curve $label budget exceeded ($n > $b)"))
function _point(p)
    p isa Union{Tuple,AbstractVector} && length(p)==2 &&
        all(x->x isa Integer && !(x isa Bool) && typemin(Int)<=x<=typemax(Int),p) ||
        throw(ArgumentError("lattice points must contain two machine integers"))
    (Int(p[1]),Int(p[2]))
end
function _shape_origin(shape,origin)
    s=_point(shape);o=_point(origin)
    all(x->0<x<typemax(Int),s) || throw(ArgumentError("grid sizes must be positive and leave room for a zero halo"))
    all(big(o[k])+s[k]<=typemax(Int) for k in 1:2) || throw(ArgumentError("zero halo overflows lattice coordinates"))
    s,o
end

"""A connected thin lattice support with a positive multiplicity and kind
`:birth` or `:death`. Points are ordered from upper left to lower right; each
step is one unit right or down. Coincident copies use `multiplicity`.
"""
struct EndCurve
    kind::Symbol
    vertices::Vector{Tuple{Int,Int}}
    multiplicity::Int
    function EndCurve(kind::Symbol,vertices; multiplicity=1)
        kind in (:birth,:death) || throw(ArgumentError("curve kind must be :birth or :death"))
        multiplicity isa Integer && !(multiplicity isa Bool) && 0<multiplicity<=typemax(Int) ||
            throw(ArgumentError("curve multiplicity must be a positive machine integer"))
        ps=_point.(collect(vertices));isempty(ps) && throw(ArgumentError("a curve must be nonempty"))
        sort!(ps;by=p->(p[1],-big(p[2])))
        for k in 2:length(ps)
            a,b=ps[k-1],ps[k]
            (big(b[1])-a[1],big(b[2])-a[2]) in ((1,0),(0,-1)) ||
                throw(ArgumentError("curve points must form a unit-step right/down lattice path without duplicates"))
        end
        new(kind,ps,Int(multiplicity))
    end
end
curve_vertices(c::EndCurve)=copy(c.vertices)
curve_multiplicity(c::EndCurve)=c.multiplicity
Base.copy(c::EndCurve)=EndCurve(c.kind,c.vertices;multiplicity=c.multiplicity)
Base.:(==)(a::EndCurve,b::EndCurve)=a.kind==b.kind && a.vertices==b.vertices && a.multiplicity==b.multiplicity
Base.isequal(a::EndCurve,b::EndCurve)=isequal((a.kind,a.vertices,a.multiplicity),(b.kind,b.vertices,b.multiplicity))
Base.hash(c::EndCurve,h::UInt)=hash((c.kind,Tuple(c.vertices),c.multiplicity),h)
end_curve_summary(c::EndCurve)=(;kind=c.kind,vertices=length(c.vertices),multiplicity=c.multiplicity)
describe(c::EndCurve)=end_curve_summary(c)
Base.show(io::IO,c::EndCurve)=print(io,"EndCurve(",c.kind,", ",length(c.vertices)," lattice points, multiplicity=",c.multiplicity,")")

"""Exact birth/death multisets on a finite lattice grid with declared zero
extension. Construct computed results with `end_curves(M)`. The constructor
`EndCurveResult(births, deaths; shape, origin=(0,0), field, betti=nothing)` accepts
supplied curves and labels them as supplied, without certifying a source module.
"""
struct EndCurveResult{F,W}
    births::Vector{EndCurve}
    deaths::Vector{EndCurve}
    shape::Tuple{Int,Int}
    origin::Tuple{Int,Int}
    field::F
    betti::Union{Nothing,Array{Int,3}}
    count_terms::Union{Nothing,NamedTuple{(:dimension,:x_rank,:y_rank,:xy_rank),NTuple{4,Int}}}
    witnesses::W
    scope::Symbol
    orientation::Tuple{Int,Int}
end
field(r::EndCurveResult)=r.field
birth_curves(r::EndCurveResult)=copy.(r.births)
death_curves(r::EndCurveResult)=copy.(r.deaths)
two_parameter_count(r::EndCurveResult)=foldl((n,c)->Base.checked_add(n,c.multiplicity),r.births;init=0)
function end_curve_betti(r::EndCurveResult)
    r.betti===nothing && throw(ArgumentError("no Betti data were supplied"))
    copy(r.betti)
end
function end_curve_witnesses(r::EndCurveResult)
    r.witnesses===nothing && throw(ArgumentError("use end_curves(M; witnesses=true) to retain algebraic witnesses"))
    r.witnesses
end
end_curve_summary(r::EndCurveResult)=(;kind=:end_curves,field=r.field,shape=r.shape,origin=r.origin,
    count=two_parameter_count(r),birth_types=length(r.births),death_types=length(r.deaths),
    extension=:zero,scope=r.scope,witnesses=r.witnesses!==nothing,betti=r.betti!==nothing,
    source_orientation=r.orientation,boundary_components=:not_computed)
describe(r::EndCurveResult)=end_curve_summary(r)
Base.show(io::IO,r::EndCurveResult)=print(io,"EndCurveResult(count=",two_parameter_count(r),", grid=",r.shape,", ",r.scope,", zero extension)")

function _coalesce(curves,kind)
    d=Dict{Tuple,Int}()
    for c in curves
        c isa EndCurve && c.kind===kind || throw(ArgumentError("expected $kind EndCurve values"))
        checked=copy(c);key=Tuple(checked.vertices)
        d[key]=Base.checked_add(get(d,key,0),checked.multiplicity)
    end
    [EndCurve(kind,collect(k);multiplicity=d[k]) for k in sort!(collect(keys(d)))]
end
function EndCurveResult(births,deaths;shape,origin=(0,0),field,betti=nothing)
    field isa Union{QQField,PrimeField} || throw(ArgumentError("end-curves require QQ or a prime field"))
    s,o=_shape_origin(shape,origin)
    if betti!==nothing
        betti isa AbstractArray && ndims(betti)==3 && size(betti)==(s[1]+1,s[2]+1,3) &&
            all(x->x isa Integer && !(x isa Bool) && 0<=x<=typemax(Int),betti) ||
            throw(ArgumentError("Betti data must be nonnegative integers of size (nx+1,ny+1,3)"))
    end
    r=EndCurveResult(_coalesce(births,:birth),_coalesce(deaths,:death),s,o,field,
        betti===nothing ? nothing : Int.(betti),nothing,nothing,:supplied,(1,1))
    check_end_curves(r;throw=true);r
end

"""Curve memberships and ordinary Betti numbers at an exact lattice point,
including the upper/right zero halo. Betti degrees occupy the last array axis.
"""
function end_curve_at(r::EndCurveResult;point)
    p=_point(point);i,j=big(p[1])-r.origin[1]+1,big(p[2])-r.origin[2]+1
    1<=i<=r.shape[1]+1 && 1<=j<=r.shape[2]+1 || throw(ArgumentError("point lies outside the grid and its one-cell zero halo"))
    i,j=Int(i),Int(j)
    (;point=p,births=findall(c->p in c.vertices,r.births),deaths=findall(c->p in c.vertices,r.deaths),
        betti=r.betti===nothing ? nothing : Tuple(r.betti[i,j,:]),
        zero_halo=i>r.shape[1] || j>r.shape[2])
end

struct EndCurveValidationSummary
    valid::Bool
    issues::Vector{String}
    witnesses_checked::Bool
end
describe(v::EndCurveValidationSummary)=(;v.valid,issues=copy(v.issues),v.witnesses_checked)
Base.show(io::IO,v::EndCurveValidationSummary)=print(io,"EndCurveValidationSummary(valid=",v.valid,", witnesses_checked=",v.witnesses_checked,")")
Base.:(==)(a::EndCurveValidationSummary,b::EndCurveValidationSummary)=describe(a)==describe(b)
Base.isequal(a::EndCurveValidationSummary,b::EndCurveValidationSummary)=isequal(describe(a),describe(b))
Base.hash(v::EndCurveValidationSummary,h::UInt)=hash((v.valid,Tuple(v.issues),v.witnesses_checked),h)
