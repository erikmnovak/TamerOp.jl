# Exact stability conditions, resource bounds and finite-category results.
"""Resource limits for exact HN, skyscraper and planar chamber computations.
`max_trials` bounds random matrix proposals across a query; `max_blowup` bounds
their total tensor multiplicity, including the extension degree.
`max_extension_degree` bounds internal finite-field extensions. Proposals use a local reproducible random stream and
are accepted only with an exact optimality certificate. Exhaustion throws.
`max_entries` checks coefficient workspaces before allocation; large cleared
integer weights can exhaust it even when the input module is small.
"""
struct HNBudget
    max_vertices::Int
    max_subspaces::Int
    max_states::Int
    max_entries::Int
    max_chambers::Int
    max_trials::Int
    max_blowup::Int
    max_extension_degree::Int
    function HNBudget(;max_vertices=4096,max_subspaces=100_000,max_states=1_000_000,
            max_entries=8_000_000,max_chambers=4096,max_trials=256,max_blowup=32,max_extension_degree=32)
        xs=(max_vertices,max_subspaces,max_states,max_entries,max_chambers,max_trials,max_blowup,max_extension_degree)
        all(x->x isa Integer && !(x isa Bool) && 0<=x<=typemax(Int),xs) ||
            throw(ArgumentError("HN budgets must be nonnegative integers fitting Int"))
        new(Int.(xs)...)
    end
end
_bound(n,b,label)=n<=b || throw(ArgumentError("HN $label budget exceeded ($n > $b)"))
function _rational(x)
    x isa Union{Integer,Rational,AbstractFloat} && !(x isa Bool) && isfinite(x) ||
        throw(ArgumentError("weights, thresholds and coordinates must be finite real numbers"))
    QQ(x)
end

"""A finite-poset central charge Z(d)=sum(mass .* d)+i*sum(degree .* d).
Mass weights must be strictly positive; supplied binary floats are interpreted
as exact rationals. Geometric weights require a separately declared measure.
"""
struct StabilityWeights
    mass::Vector{QQ}
    degree::Vector{QQ}
    function StabilityWeights(mass,degree)
        a=_rational.(collect(mass));b=_rational.(collect(degree))
        length(a)==length(b) || throw(DimensionMismatch("mass and degree weights must have equal lengths"))
        all(>(0),a) || throw(ArgumentError("every mass weight must be positive"))
        new(a,b)
    end
end
Base.copy(z::StabilityWeights)=StabilityWeights(z.mass,z.degree)
describe(z::StabilityWeights)=(;kind=:stability_weights,mass=copy(z.mass),degree=copy(z.degree),category=:finite_poset)
Base.show(io::IO,z::StabilityWeights)=print(io,"StabilityWeights(",length(z.mass)," vertices)")

"""HN slopes and successive dimension vectors, with optional actual inclusions.
Retained modules/maps are read-only; use the accessors to inspect the filtration.
"""
struct HNFiltration{M,W}
    source::M
    stability::StabilityWeights
    slopes::Vector{QQ}
    term_dimensions::Vector{Vector{Int}}
    witnesses::W
    route::Symbol
    statistics::NamedTuple
end
hn_slopes(r::HNFiltration)=copy(r.slopes)
hn_dimensions(r::HNFiltration)=copy.(r.term_dimensions)
function hn_factors(r::HNFiltration)
    previous=zeros(Int,length(r.source.dims))
    [begin d=t-previous;previous=t;d end for t in r.term_dimensions]
end
function hn_maps(r::HNFiltration)
    r.witnesses===nothing && throw(ArgumentError("request witnesses=true to retain filtration terms and maps"))
    r.witnesses
end
function hn_summary(r::HNFiltration)
    (;kind=:hn_filtration,field=r.source.field,category=:finite_poset,
      slopes=hn_slopes(r),term_dimensions=hn_dimensions(r),factor_dimensions=hn_factors(r),
      stability=describe(r.stability),route=r.route,exact=true,
      witnesses=r.witnesses!==nothing,statistics=r.statistics)
end
describe(r::HNFiltration)=hn_summary(r)
field(r::HNFiltration)=r.source.field
dimensions(r::HNFiltration)=hn_dimensions(r)
provenance(r::HNFiltration)=(;category=:finite_poset,field=r.source.field,
    stability=describe(r.stability),route=r.route,exact=true,ambient_measure=:not_inferred)
Base.show(io::IO,r::HNFiltration)=print(io,"HNFiltration(",length(r.slopes)," factors, ",r.route,")")

struct HNValidationSummary
    valid::Bool
    issues::Vector{String}
    semistability_checked::Bool
end
Base.:(==)(a::HNValidationSummary,b::HNValidationSummary)=
    (a.valid,a.issues,a.semistability_checked)==(b.valid,b.issues,b.semistability_checked)
Base.isequal(a::HNValidationSummary,b::HNValidationSummary)=
    isequal((a.valid,a.issues,a.semistability_checked),(b.valid,b.issues,b.semistability_checked))
Base.hash(a::HNValidationSummary,h::UInt)=hash((a.valid,Tuple(a.issues),a.semistability_checked),h)
describe(a::HNValidationSummary)=(;valid=a.valid,issues=copy(a.issues),semistability_checked=a.semistability_checked)
Base.show(io::IO,a::HNValidationSummary)=print(io,"HNValidationSummary(valid=",a.valid,", semistability_checked=",a.semistability_checked,")")

mutable struct _Work
    states::Int
    subspaces::Int
    matrix_trials::Int
    largest_blowup::Int
    largest_matrix::Int
    extension_trials::Int
    largest_extension::Int
    rng::MersenneTwister
end
_Work(states,subspaces)=_Work(states,subspaces,0,0,0,0,1,MersenneTwister(0x51))
_statistics(w)=(;states=w.states,subspaces=w.subspaces,matrix_trials=w.matrix_trials,
    largest_blowup=w.largest_blowup,largest_matrix=w.largest_matrix,
    extension_trials=w.extension_trials,largest_extension=w.largest_extension)
function _tick!(w,b)
    _bound(w.states+1,b.max_states,"search states")
    w.states+=1
end
function _input(M,b)
    M.field isa Union{QQField,PrimeField} || throw(ArgumentError("HN computation requires exact QQ or prime-field coefficients"))
    _bound(nvertices(M.Q),b.max_vertices,"vertices")
    _bound(sum(big(d)^2 for d in M.dims;init=big(0))+
        sum(big(length(A)) for (_,A) in M.edge_maps;init=big(0)),b.max_entries,"input/workspace entries")
    check_module_functoriality(M;throw=true)
end
function _snapshot(M::PModule{K}) where K
    PModule{K}(M.Q,copy(M.dims),Dict(e=>Matrix{K}(A) for (e,A) in M.edge_maps);field=M.field)
end
_mass(z,d)=sum((a*x for (a,x) in zip(z.mass,d));init=zero(QQ))
_degree(z,d)=sum((a*x for (a,x) in zip(z.degree,d));init=zero(QQ))
_slope(z,d)=_degree(z,d)/_mass(z,d)

# Count retained coefficient entries before constructing a filtration prefix
# and its successive quotient. Shared source/poset storage is counted at input.
function _filtration_entries(M,d,previous;witnesses=false)
    count=big(length(d))+1
    if witnesses
        factor=d-previous
        count+=sum(big(M.dims[v])*d[v]+big(d[v])*factor[v] for v in eachindex(d);init=big(0))
        count+=sum(big(d[u])*d[v]+big(factor[u])*factor[v] for (u,v) in cover_edges(M.Q);init=big(0))
    end
    count
end

function _density(value,n)
    rho=value isa Number ? fill(_rational(value),n) : _rational.(vec(collect(value)))
    length(rho)==n || throw(DimensionMismatch("supply one density per geometric cell or region"))
    all(>(0),rho) || throw(ArgumentError("geometric densities must be strictly positive"))
    rho
end
_measure(g)=all(isone,g.density) ? :lebesgue : :piecewise_constant_density
