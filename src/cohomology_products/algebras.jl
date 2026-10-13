# Exact finite graded rings, validation, and explicit work bounds.

"""Bounds for exact cup computations. `max_entries` limits explicit matrix/workspace
size accounting per operation, not process memory or internal solver allocations.
`max_products` bounds basis-product evaluations (including algebra validation).
Exceeding a bound throws, never approximates."""
struct CupProductBudget
    max_cells::Int
    max_stages::Int
    max_entries::Int
    max_products::Int
    function CupProductBudget(;max_cells=10_000,max_stages=1000,
                              max_entries=20_000_000,max_products=2_000_000)
        xs=(max_cells,max_stages,max_entries,max_products)
        all(x->x isa Integer && !(x isa Bool) && 0<=x<=typemax(Int),xs) ||
            throw(ArgumentError("cup budgets must be nonnegative integers fitting Int"))
        new(Int.(xs)...)
    end
end
mutable struct _CupWork
    budget::CupProductBudget
    entries::BigInt
    products::BigInt
end
_CupWork(b::CupProductBudget)=_CupWork(b,big(0),big(0))
function _spend!(w::_CupWork;entries=0,products=0)
    w.entries+=entries;w.products+=products
    w.entries<=w.budget.max_entries || throw(ArgumentError("cup matrix-entry budget exceeded"))
    w.products<=w.budget.max_products || throw(ArgumentError("cup product-work budget exceeded"))
end
function _cup_int(x;positive=false)
    x isa Integer && !(x isa Bool) && (positive ? 1 : 0)<=x<=typemax(Int) ||
        throw(ArgumentError("expected a $(positive ? "positive" : "nonnegative") integer fitting Int"))
    Int(x)
end
function _cup_field(F)
    F isa Union{QQField,PrimeField} || throw(ArgumentError("cup products require QQField() or a prime field"))
    F
end
function _cup_coefficients(F,A)
    K=coeff_type(F)
    map(A) do x
        if x isa FpElem
            typeof(x)===K || throw(ArgumentError("coefficient belongs to a different field"))
            x
        elseif x isa Union{Integer,Rational}
            coerce(F,x)
        else
            throw(ArgumentError("cup coefficients must be exact integers, rationals, or elements of the specified field"))
        end
    end
end

struct _SimplicialCochains{K}
    cells::Vector{Vector{Tuple{Vararg{Int}}}}
    deltas::Vector{Matrix{K}}
    boundaries::Vector{Matrix{K}}
    representatives::Vector{Matrix{K}}
    quotient_bases::Vector{Matrix{K}}
end

"""A finite nonnegatively graded unital algebra over an exact field. Construct
with `cohomology_algebra`; use `dimensions`, `cup_product`, `cup_unit`, and
`cup_product_matrix`. Missing degrees above the stored range are zero, not
unknown. A supplied algebra is not asserted to be realizable by a space."""
struct CohomologyAlgebra{K,F}
    dims::Vector{Int}
    products::Dict{Tuple{Int,Int},Matrix{K}}
    unit::Vector{K}
    field::F
    cochains::Union{Nothing,_SimplicialCochains{K}}
end

field(A::CohomologyAlgebra)=A.field
dimensions(A::CohomologyAlgebra)=copy(A.dims)
_hdim(A,d)=0<=d<length(A.dims) ? A.dims[d+1] : 0
"""Coordinates of the multiplicative identity in degree zero."""
cup_unit(A::CohomologyAlgebra)=copy(A.unit)
"""The degree `(p,q)` product matrix. Column `i + (j-1)*dim(H^p)` is
`basis_p[i] cup basis_q[j]`; rows are coordinates in H^(p+q)."""
function cup_product_matrix(A::CohomologyAlgebra{K};left_degree,right_degree) where {K}
    p=_cup_int(left_degree);q=_cup_int(right_degree)
    p<=typemax(Int)-q || throw(ArgumentError("degree sum overflow"))
    copy(get(()->zeros(K,_hdim(A,p+q),_hdim(A,p)*_hdim(A,q)),A.products,(p,q)))
end
function _multiply(A::CohomologyAlgebra{K},x,y,p,q) where {K}
    out=zeros(K,_hdim(A,p+q));M=get(A.products,(p,q),nothing)
    M===nothing && return out
    for j in eachindex(y),i in eachindex(x)
        c=x[i]*y[j];iszero(c) && continue
        for r in eachindex(out);out[r]+=c*M[r,i+(j-1)*length(x)];end
    end
    out
end
"""Multiply cohomology coordinate vectors, with explicit nonnegative degrees.
This is multiplication in cohomology, not the generally noncommutative
cochain-level product. Inputs must belong to the recorded coefficient field."""
function cup_product(A::CohomologyAlgebra,x::AbstractVector,y::AbstractVector;
                     left_degree,right_degree)
    p=_cup_int(left_degree);q=_cup_int(right_degree)
    p<=typemax(Int)-q || throw(ArgumentError("degree sum overflow"))
    length(x)==_hdim(A,p) && length(y)==_hdim(A,q) || throw(ArgumentError("product coordinate dimensions disagree"))
    _multiply(A,_cup_coefficients(field(A),x),_cup_coefficients(field(A),y),p,q)
end

function _algebra_issues(A::CohomologyAlgebra{K},w) where {K}
    ds=A.dims;D=length(ds)-1;issues=String[]
    isempty(ds) && return ["dimensions must include degree zero"]
    all(>=(0),ds) || return ["negative graded dimension"]
    length(A.unit)==ds[1] || return ["unit has the wrong degree-zero dimension"]
    for ((p,q),M) in A.products
        0<=p<=D && 0<=q<=D && p+q<=D || return ["product has a degree outside the algebra"]
        size(M)==(ds[p+q+1],ds[p+1]*ds[q+1]) || return ["product matrix has the wrong shape"]
    end
    field(A) isa Union{QQField,PrimeField} || return ["unsupported coefficient field"]
    coeff_type(field(A))===K || return ["coefficient storage disagrees with field"]
    _spend!(w;entries=sum(big(d)^2 for d in ds))
    es=[eye(field(A),d) for d in ds]
    degrees=[d for d in 0:D if ds[d+1]>0]
    for p in 0:D,i in 1:ds[p+1]
        x=es[p+1][:,i];_spend!(w;products=2)
        _multiply(A,A.unit,x,0,p)==x && _multiply(A,x,A.unit,p,0)==x ||
            return ["multiplicative identity law fails"]
    end
    for p in degrees,q in degrees
        p+q<=D || continue
        for i in 1:ds[p+1],j in 1:ds[q+1]
            _spend!(w;products=2)
            x=es[p+1][:,i];y=es[q+1][:,j]
            _multiply(A,x,y,p,q)==(isodd(p*q) ? -one(K) : one(K))*_multiply(A,y,x,q,p) ||
                return ["graded commutativity fails in degrees ($p,$q)"]
        end
    end
    for p in degrees,q in degrees,r in degrees
        p+q+r<=D || continue
        for i in 1:ds[p+1],j in 1:ds[q+1],k in 1:ds[r+1]
            _spend!(w;products=4)
            x=es[p+1][:,i];y=es[q+1][:,j];z=es[r+1][:,k]
            _multiply(A,_multiply(A,x,y,p,q),z,p+q,r)==
                _multiply(A,x,_multiply(A,y,z,q,r),p,q+r) ||
                return ["associativity fails in degrees ($p,$q,$r)"]
        end
    end
    issues
end

"""Typed result of exact algebra, ring-diagram, or product-module validation.
Use `describe(report)` to inspect issues and `.valid` for a Boolean."""
struct CupProductValidationSummary
    valid::Bool
    issues::Vector{String}
end
describe(r::CupProductValidationSummary)=(;valid=r.valid,issues=copy(r.issues))
Base.show(io::IO,r::CupProductValidationSummary)=print(io,"CupProductValidationSummary(valid=",r.valid,", issues=",r.issues,")")
Base.:(==)(a::CupProductValidationSummary,b::CupProductValidationSummary)=a.valid==b.valid && a.issues==b.issues
Base.isequal(a::CupProductValidationSummary,b::CupProductValidationSummary)=a==b
Base.hash(a::CupProductValidationSummary,h::UInt)=hash((a.valid,Tuple(a.issues)),h)
function _cup_report(issues,throw_errors)
    throw_errors && !isempty(issues) && throw(ArgumentError(join(issues,"; ")))
    CupProductValidationSummary(isempty(issues),issues)
end
"""Check shapes, identity, graded commutativity, and associativity exactly.
Resource exhaustion is an error, not a successful or negative mathematical verdict."""
function check_cohomology_algebra(A::CohomologyAlgebra;throw=false,budget=CupProductBudget())
    _cup_report(_algebra_issues(A,_CupWork(budget)),throw)
end

"""
    cohomology_algebra(dimensions, products; unit, field=F2(), budget=CupProductBudget())

Build a checked finite graded algebra. `dimensions[d+1]` is dim H^d, and
`products[(p,q)]` has shape (dim H^(p+q), dim H^p * dim H^q), with the first
factor varying fastest. Omitted matrices are zero. Degrees above the supplied
range are zero. All ordered products, including degree zero, are checked.
The zero algebra has an empty (zero) unit. Coefficients are exact; no float
conversion or change of prime field is implicit.
"""
function cohomology_algebra(ds::AbstractVector,products::AbstractDict;unit,
                           field=F2(),budget=CupProductBudget())
    F=_cup_field(field);K=coeff_type(F);dims=Int[_cup_int(d) for d in ds]
    isempty(dims) && throw(ArgumentError("dimensions must include degree zero"))
    w=_CupWork(budget);_spend!(w;entries=sum(big(d)^2 for d in dims))
    tables=Dict{Tuple{Int,Int},Matrix{K}}();D=length(dims)-1
    for (key,M) in products
        key isa Tuple && length(key)==2 || throw(ArgumentError("product keys must be degree pairs"))
        p,q=_cup_int.(key)
        p<=D && q<=D-p || throw(ArgumentError("product degree lies outside the algebra"))
        M isa AbstractMatrix && size(M)==(dims[p+q+1],big(dims[p+1])*dims[q+1]) ||
            throw(ArgumentError("product matrix has the wrong shape"))
        _spend!(w;entries=length(M));tables[(p,q)]=Matrix{K}(_cup_coefficients(F,M))
    end
    unit isa AbstractVector && length(unit)==dims[1] || throw(ArgumentError("unit must be a degree-zero vector"))
    A=CohomologyAlgebra(dims,tables,Vector{K}(_cup_coefficients(F,unit)),F,nothing)
    _cup_report(_algebra_issues(A,w),true);A
end

describe(A::CohomologyAlgebra)=(;kind=:cohomology_algebra,field=field(A),dimensions=Tuple(A.dims),
    graded=true,unital=true,source=A.cochains===nothing ? :supplied_algebra : :ordered_simplicial_cochains)
Base.show(io::IO,A::CohomologyAlgebra)=print(io,"CohomologyAlgebra(field=",field(A),", dimensions=",Tuple(A.dims),")")
