# Exhaustive finite-field subobjects provide a bounded exact reference engine.
# Maximal dimension resolves highest-slope ties to the unique destabilizer.
# See Fersztand--Jendrysiak, arXiv:2603.23560, Sections 2.4 and 5.
function _subspaces(F,d,b,w)
    K=coeff_type(F)
    F isa QQField && d>1 && throw(ArgumentError("exhaustive rational subspace enumeration is not finite; use certified matrix-space optimization"))
    spaces=Matrix{K}[]
    pivots=Int[]
    function emit()
        k=length(pivots)
        free=Tuple{Int,Int}[(r,c) for c in 1:d if !(c in pivots) for r in 1:k if pivots[r]<c]
        count=F isa PrimeField ? big(F.p)^length(free) : big(1)
        _bound(big(w.subspaces)+count,b.max_subspaces,"enumerated subspaces")
        _bound((big(length(spaces))+count)*d*max(d,1),b.max_entries,"subspace matrices")
        for code in 0:Int(count)-1
            A=zeros(K,d,k)
            for (r,c) in enumerate(pivots);A[c,r]=one(K);end
            remaining=code
            for (r,c) in free
                A[c,r]=K(rem(remaining,F.p));remaining=div(remaining,F.p)
            end
            push!(spaces,A)
        end
        w.subspaces+=Int(count)
    end
    function choose(start,k)
        k==0 && return emit()
        for p in start:d-k+1
            push!(pivots,p);choose(p+1,k-1);pop!(pivots)
        end
    end
    for k in 0:d;choose(1,k);end
    spaces
end
_rank(F,A)=FL.rank_dim(F,A)
_basis(F,A)=Matrix(FL.colspace(F,A))
_contains(F,A,B)=size(B,2)==0 || _rank(F,hcat(A,B))==size(A,2)
function _subobject(M::PModule{K},bases) where K
    F=M.field
    maps=Dict{Tuple{Int,Int},Matrix{K}}()
    for (u,v) in cover_edges(M.Q)
        maps[u,v]=Matrix{K}(FL.solve_fullcolumn(F,bases[v],M.edge_maps[u,v]*bases[u]))
    end
    N=PModule{K}(M.Q,[size(A,2) for A in bases],maps;field=F)
    N,PMorphism(N,M,copy.(bases))
end

function _highest(M,z,b,w)
    n=length(M.dims);F=M.field;K=coeff_type(F)
    ratios=unique([z.degree[v]/z.mass[v] for v in 1:n if M.dims[v]>0])
    length(ratios)==1 && return [eye(F,d) for d in M.dims],only(ratios)
    # Antichains have no coupled subspace choices, over any exact field.
    if isempty(cover_edges(M.Q))
        slope=maximum((z.degree[v]/z.mass[v] for v in 1:n if M.dims[v]>0))
        return [z.degree[v]/z.mass[v]==slope ? eye(F,M.dims[v]) : zeros(K,M.dims[v],0) for v in 1:n],slope
    end
    _highest_general(M,z,b,w)
end

"""Compute the unique HN filtration for explicit finite-poset stability weights.
All finite stalk dimensions over QQ and prime fields are supported. The general
route uses exact, certified matrix-space optimization. Random matrix proposals
never substitute for an optimality certificate; budget exhaustion throws.
Cleared integer weights affect workspace size. Optional witnesses retain actual
terms, inclusions and quotient maps.
"""
function hn_filtration(M::PModule,z::StabilityWeights;
        witnesses::Bool=false,budget::HNBudget=HNBudget())
    _input(M,budget);z=copy(z)
    length(z.mass)==length(M.dims) || throw(DimensionMismatch("one stability weight is required per vertex"))
    M=_snapshot(M);w=_Work(0,0)
    slopes,dims,maps=_hn_data(M,z,budget,w;witnesses)
    route=w.matrix_trials>0 ? :certified_matrix_space : :exact_finite_poset
    HNFiltration(M,z,slopes,dims,maps,route,_statistics(w))
end
function _hn_data(M::PModule{K},z,budget,w;witnesses=false) where K
    C=M;F=M.field;n=length(M.dims)
    projections=witnesses ? [eye(F,d) for d in M.dims] : Matrix{K}[]
    terms=PModule{K}[];inclusions=PMorphism{K}[];slopes=QQ[];dims=Vector{Int}[]
    retained=big(0);previous=zeros(Int,n)
    while any(>(0),C.dims)
        B,slope=_highest(C,z,budget,w)
        N,i=_subobject(C,B)
        D,q=AC.quotient_with_projection(i)
        current=M.dims-C.dims+N.dims
        retained+=_filtration_entries(M,current,previous;witnesses)
        _bound(retained,budget.max_entries,"retained filtration entries")
        push!(slopes,slope);push!(dims,current);previous=current
        if witnesses
            prefix=[Matrix{K}(FL.nullspace(F,transpose(FL.nullspace(F,transpose(B[v])))*projections[v])) for v in 1:n]
            T,j=_subobject(M,prefix)
            push!(terms,T);push!(inclusions,j)
            projections=[q.comps[v]*projections[v] for v in 1:n]
        end
        C=D
    end
    maps=if witnesses
        factors,factor_projections=_factor_maps(M,terms,inclusions)
        (;terms,inclusions,factors,factor_projections)
    else
        nothing
    end
    slopes,dims,maps
end

function check_hn_filtration(r::HNFiltration;semistability::Bool=false,
        budget::HNBudget=HNBudget(),throw::Bool=false)
    issues=String[];checked=false
    try
        _input(r.source,budget);z=copy(r.stability);n=length(r.source.dims)
        length(z.mass)==n || error("stability length mismatch")
        length(r.slopes)==length(r.term_dimensions) || error("factor count mismatch")
        previous=zeros(Int,n)
        for (i,d) in enumerate(r.term_dimensions)
            length(d)==n && all(previous .<= d) && all(d .<= r.source.dims) && d!=previous || error("invalid filtration dimensions")
            r.slopes[i]==_slope(z,d-previous) || error("incorrect factor slope")
            i==1 || r.slopes[i-1]>r.slopes[i] || error("slopes must strictly decrease")
            previous=d
        end
        previous==r.source.dims || error("filtration does not exhaust the source")
        if r.witnesses!==nothing
            ms=r.witnesses
            length(ms.terms)==length(ms.inclusions)==length(ms.factors)==length(ms.factor_projections)==length(r.slopes) || error("witness count mismatch")
            last=[zeros(coeff_type(r.source.field),d,0) for d in r.source.dims]
            for i in eachindex(ms.terms)
                j=ms.inclusions[i]
                j.dom===ms.terms[i] && j.cod===r.source && j.dom.dims==r.term_dimensions[i] || error("inclusion endpoints disagree")
                check_morphism(j;throw=true)
                q=ms.factor_projections[i];factor=ms.factors[i]
                q.dom===j.dom && q.cod===factor || error("factor projection endpoints disagree")
                check_morphism(q;throw=true)
                for v in 1:n
                    _rank(r.source.field,j.comps[v])==r.term_dimensions[i][v] || error("inclusion is not monic")
                    _contains(r.source.field,j.comps[v],last[v]) || error("terms are not nested")
                    factor.dims[v]==j.dom.dims[v]-size(last[v],2) || error("incorrect factor dimension")
                    _rank(r.source.field,q.comps[v])==factor.dims[v] || error("factor projection is not epic")
                    coordinates=FL.solve_fullcolumn(r.source.field,j.comps[v],last[v])
                    iszero(q.comps[v]*coordinates) || error("factor projection does not annihilate the previous term")
                end
                last=j.comps
            end
        end
        if semistability
            expected=hn_filtration(r.source,z;witnesses=true,budget)
            expected.slopes==r.slopes && expected.term_dimensions==r.term_dimensions || error("filtration is not HN")
            if r.witnesses!==nothing
                for (a,b) in zip(r.witnesses.inclusions,expected.witnesses.inclusions),v in 1:n
                    _contains(r.source.field,a.comps[v],b.comps[v]) || error("incorrect HN subspace")
                end
            end
            checked=true
        end
    catch err
        err isa InterruptException && rethrow()
        push!(issues,sprint(showerror,err))
    end
    throw && !isempty(issues) && Base.throw(ArgumentError(join(issues,"; ")))
    HNValidationSummary(isempty(issues),issues,checked)
end
