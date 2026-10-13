struct InterleavingSearchResult{P,C}
    problem::P
    status::Symbol
    certificate::C
    reason::Symbol
    candidates::Int
    hom_dimensions::Tuple{Union{Nothing,Int},Union{Nothing,Int}}
end

interleaving_witness(r::InterleavingSearchResult) = r.certificate
interleaving_summary(r::InterleavingSearchResult) = (;kind=:interleaving_search,
    status=r.status,reason=r.reason,candidates=r.candidates,hom_dimensions=r.hom_dimensions,
    certified=r.certificate !== nothing,provenance=provenance(r.problem))
describe(r::InterleavingSearchResult) = interleaving_summary(r)
Base.show(io::IO,r::InterleavingSearchResult) = print(io,"InterleavingSearchResult(",r.status,", ",r.reason,", ",r.candidates," candidates)")

# Linear equations after fixing one morphism; free coordinates are set to zero.
# rref is the shared exact field engine, including its tiny/prime/QQ backends.
function _affine_solution(field,A::AbstractMatrix{K},b::AbstractVector{K}) where K
    n=size(A,2)
    R,pivots=FieldLinAlg.rref(field,hcat(A,b))
    (n+1) in pivots && return nothing
    x=zeros(K,n)
    for (i,j) in enumerate(pivots)
        x[j]=R[i,n+1]
    end
    return x
end

function _hom_basis(M,N,budget)
    vars=sum(big(a)*b for (a,b) in zip(M.dims,N.dims);init=big(0))
    _limit(vars,budget.max_hom_variables,"Hom variables")
    _limit(vars^2,budget.max_matrix_entries,"Hom workspace")
    return basis(Hom(M,N))
end

function _combination(M::PModule{K},N,B,coeffs) where K
    maps=Matrix{K}[zeros(K,N.dims[i],M.dims[i]) for i in eachindex(M.dims)]
    for j in eachindex(coeffs)
        iszero(coeffs[j]) && continue
        for i in eachindex(maps)
            maps[i] .+= coeffs[j].*B[j].comps[i]
        end
    end
    return PMorphism(M,N,maps)
end

function _composition_vector(p,f,g)
    K=coeff_type(p.sources[1].field)
    out=K[]
    for side in 1:2, i in eachindex(p.at)
        a,b=p.at[i],p.shifted_at[i]
        append!(out,vec(side==1 ? g.comps[b]*f.comps[a] : f.comps[b]*g.comps[a]))
    end
    return out
end

function _target_vector(p)
    K=coeff_type(p.sources[1].field)
    out=K[]
    for side in 1:2, A in p.canonical[side]
        append!(out,vec(A))
    end
    return out
end

function _try_fixed(p,f,free_basis,b,fixed_side,budget)
    K=coeff_type(p.sources[1].field)
    _limit(big(length(b))*(length(free_basis)+1),budget.max_matrix_entries,"witness solve")
    A=zeros(K,length(b),length(free_basis))
    for j in eachindex(free_basis)
        A[:,j]=fixed_side==1 ? _composition_vector(p,f,free_basis[j]) : _composition_vector(p,free_basis[j],f)
    end
    y=_affine_solution(p.sources[1].field,A,b)
    y===nothing && return nothing
    side=3-fixed_side
    g=_combination(p.sources[side],p.targets[side],free_basis,y)
    return fixed_side==1 ? interleaving_certificate(p,f,g;budget) : interleaving_certificate(p,g,f;budget)
end

# Enumerate projective coefficient vectors: simultaneous rescaling of f and
# inverse rescaling of g preserves both equations. This removes redundant
# scalar copies without restricting the Hom spaces. Return false on budget.
function _finite_candidates(visit,h,field)
    K=coeff_type(field); q=big(field.p)
    for pivot in h:-1:1
        total=q^(h-pivot)
        code=big(0)
        while code<total
            x=zeros(K,h); x[pivot]=one(K); value=code
            for j in pivot+1:h
                value,digit=divrem(value,q); x[j]=coerce(field,digit)
            end
            visit(x) || return false
            code+=1
        end
    end
    return true
end

function _rational_candidates(visit,h,maxheight,budget)
    # Every projective rational vector occurs at some finite height. Failure
    # at the configured height is not a rational nonexistence certificate.
    alphabet=QQ[0]; known=Set(alphabet)
    for height in 1:maxheight
        previous=length(alphabet)
        _limit(big(2height+1)*height,budget.max_matrix_entries,"rational search alphabet")
        for denominator in 1:height, numerator in -height:height
            max(abs(numerator),denominator)==height || continue
            q=big(numerator)//big(denominator)
            if !(q in known)
                push!(alphabet,q);push!(known,q)
            end
        end
        radix=big(length(alphabet))
        for pivot in h:-1:1
            code=big(0); total=radix^(h-pivot)
            while code<total
                x=zeros(QQ,h);x[pivot]=1;value=code;new=height==1
                for j in pivot+1:h
                    value,digit=divrem(value,radix)
                    index=Int(digit)+1; x[j]=alphabet[index]
                    new |= index>previous
                end
                if new
                    visit(x) || return false
                end
                code+=1
            end
        end
    end
    return true
end

"""
    find_interleaving(problem; budget=InterleavingBudget(), forward=nothing, reverse=nothing)
    find_interleaving(M, N, delta; budget=InterleavingBudget())

Search the full Hom spaces, with no interval-decomposition assumption. Each
trial fixes one morphism and solves the remaining equations linearly. Finite
prime-field exhaustion proves nonexistence. Rational projective coefficients
are searched up to rational_height and max_candidates; unsuccessful bounded
search returns :unknown. An inconsistent linear relaxation proves nonexistence
over either field. A supplied forward or reverse morphism restricts the search
to extensions of that map, and an unsuccessful extension returns :unknown.

The returned status is :interleaved, :not_interleaved, or :unknown. Successful
results contain both exact natural transformations, independently validated.
General interleaving decision is computationally hard; see
https://arxiv.org/abs/1811.09165. Budgets bound this exponential search.
"""
function find_interleaving(p::InterleavingProblem; budget=InterleavingBudget(),
                           forward=nothing,reverse=nothing)
    forward===nothing || reverse===nothing || throw(ArgumentError("Supply at most one fixed map; use interleaving_certificate to check two maps."))
    check_interleaving_problem(p;throw=true,budget)
    for M in (p.sources...,p.targets...); _module_check(M,budget);end
    K=coeff_type(p.sources[1].field); b=_target_vector(p)
    result(status,certificate,reason,trials,dims)=InterleavingSearchResult(p,status,certificate,reason,trials,dims)
    if forward!==nothing || reverse!==nothing
        side=forward!==nothing ? 1 : 2
        f=_components(p,side,side==1 ? forward : reverse)
        check_morphism(f;throw=true)
        B=_hom_basis(p.sources[3-side],p.targets[3-side],budget)
        c=_try_fixed(p,f,B,b,side,budget)
        dims=side==1 ? (nothing,length(B)) : (length(B),nothing)
        return result(c===nothing ? :unknown : :interleaved,c,
                      c===nothing ? :fixed_map_obstruction : :witness,1,dims)
    end
    if all(iszero,b)
        f=_combination(p.sources[1],p.targets[1],(),K[])
        g=_combination(p.sources[2],p.targets[2],(),K[])
        return result(:interleaved,interleaving_certificate(p,f,g;budget),:zero_witness,0,(nothing,nothing))
    end
    F=_hom_basis(p.sources[1],p.targets[1],budget)
    G=_hom_basis(p.sources[2],p.targets[2],budget)
    dims=(length(F),length(G))
    if isempty(F) || isempty(G)
        return result(:not_interleaved,nothing,:zero_hom_obstruction,0,dims)
    end
    _limit(big(length(b))*(big(length(F))*length(G)+1),budget.max_matrix_entries,"bilinear equations")
    tensor=zeros(K,length(b),length(G),length(F))
    for i in eachindex(F),j in eachindex(G)
        tensor[:,j,i]=_composition_vector(p,F[i],G[j])
    end
    if _affine_solution(p.sources[1].field,reshape(tensor,length(b),:),b)===nothing
        return result(:not_interleaved,nothing,:linear_obstruction,0,dims)
    end
    # Enumerate the smaller Hom, solving for the larger one at each trial.
    side=length(F)<=length(G) ? 1 : 2
    fixed,free=side==1 ? (F,G) : (G,F)
    trials=0; found=nothing
    function visit(x)
        trials>=budget.max_candidates && return false
        trials+=1
        A=zeros(K,length(b),length(free))
        for i in eachindex(x)
            iszero(x[i]) && continue
            if side==1
                A .+= x[i].*view(tensor,:,:,i)
            else
                A .+= x[i].*view(tensor,:,i,:)
            end
        end
        y=_affine_solution(p.sources[1].field,A,b)
        y===nothing && return true
        f=_combination(p.sources[side],p.targets[side],fixed,x)
        g=_combination(p.sources[3-side],p.targets[3-side],free,y)
        found=side==1 ? interleaving_certificate(p,f,g;budget) : interleaving_certificate(p,g,f;budget)
        return false
    end
    field=p.sources[1].field
    exhausted=field isa PrimeField ? _finite_candidates(visit,length(fixed),field) :
        _rational_candidates(visit,length(fixed),budget.rational_height,budget)
    found!==nothing && return result(:interleaved,found,:witness,trials,dims)
    if field isa PrimeField && exhausted
        return result(:not_interleaved,nothing,:exhaustive_finite_field,trials,dims)
    end
    return result(:unknown,nothing,exhausted ? :rational_height : :candidate_budget,trials,dims)
end

function find_interleaving(M,N,delta; budget=InterleavingBudget(),kwargs...)
    return find_interleaving(interleaving_problem(M,N,delta;budget);budget,kwargs...)
end

function find_interleaving(M::EncodingResult,N::EncodingResult,delta;
                           extension=:none,budget=InterleavingBudget(),kwargs...)
    return find_interleaving(interleaving_problem(M,N,delta;extension,budget);budget,kwargs...)
end

struct InterleavingDistanceBounds{C,R}
    lower::QQ
    upper::Union{Nothing,QQ}
    certificate::C
    reason::Symbol
    decisions::Vector{R}
end

function interleaving_distance_bounds(M::EncodingResult,N::EncodingResult;
                                      extension=:none,budget=InterleavingBudget(),kwargs...)
    return interleaving_distance_bounds(ambient_module(M;extension,budget),
        ambient_module(N;extension,budget);budget,kwargs...)
end
interleaving_witness(r::InterleavingDistanceBounds)=r.certificate
interleaving_summary(r::InterleavingDistanceBounds)=(;kind=:interleaving_distance_bounds,
    lower=r.lower,upper=r.upper,reason=r.reason,decisions=length(r.decisions),
    certified_upper=r.certificate!==nothing)
describe(r::InterleavingDistanceBounds)=interleaving_summary(r)
Base.show(io::IO,r::InterleavingDistanceBounds)=print(io,"InterleavingDistanceBounds(",r.lower,", ",r.upper,"; ",r.reason,")")

"""
    interleaving_distance_bounds(M, N; upper, tolerance=1//100, max_steps=32, budget)

Certify bounds on diagonal ambient interleaving distance by exact decisions
and bisection. `upper` is a trial, not an assumed bound: the result reports
upper=nothing unless a witness is found. Unknown decisions stop refinement;
only proved nonexistence raises the lower bound. Integer domains use integral
translations and stop at adjacent integer bounds. No sampled slice distance
is substituted for module interleaving distance.
"""
function interleaving_distance_bounds(M::RectilinearModule,N::RectilinearModule;
        upper,tolerance=1//100,max_steps::Integer=32,budget=InterleavingBudget())
    hi=_coordinate(upper);tol=_coordinate(tolerance)
    hi>=0 && tol>0 && max_steps>0 || throw(ArgumentError("Require upper >= 0, tolerance > 0 and max_steps > 0."))
    M.domain===:integer && !isinteger(hi) && throw(ArgumentError("Integer-domain upper bound must be integral."))
    records=NamedTuple[]
    lo=QQ(0); witness=nothing; certified_hi=nothing; reason=:step_budget
    function decide(e)
        r=find_interleaving(M,N,e;budget)
        push!(records,(;epsilon=_coordinate(e),status=r.status,reason=r.reason,candidates=r.candidates))
        r
    end
    zero_result=decide(0)
    if zero_result.status===:interleaved
        return InterleavingDistanceBounds(QQ(0),QQ(0),zero_result.certificate,:exact_zero,records)
    end
    r=hi==0 ? zero_result : decide(hi)
    if r.status!==:interleaved
        r.status===:not_interleaved && (lo=hi)
        return InterleavingDistanceBounds(lo,nothing,nothing,
            r.status===:unknown ? :undecided_upper : :upper_not_feasible,records)
    end
    witness=r.certificate;certified_hi=hi
    for _ in 1:max_steps
        if hi-lo<=tol || (M.domain===:integer && hi-lo<=1)
            reason=:tolerance;break
        end
        mid=(lo+hi)/2
        M.domain===:integer && (mid=QQ(floor(BigInt,mid)))
        r=decide(mid)
        if r.status===:interleaved
            hi=mid;certified_hi=hi;witness=r.certificate
        elseif r.status===:not_interleaved
            lo=mid
        else
            reason=:undecided;break
        end
    end
    return InterleavingDistanceBounds(lo,certified_hi,witness,reason,records)
end
