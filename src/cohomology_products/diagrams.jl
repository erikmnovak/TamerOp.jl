# Contravariant graded algebra diagrams on arbitrary finite posets.

"""A checked diagram of graded algebras on a finite poset. For `u <= v`,
cohomology restricts from v to u. Construct with `persistent_cup_products`;
`cup_poset` returns the original indexing order. Chain inputs also retain their
scalar event parameters. General finite diagrams assert no extension outside
their indexing poset."""
struct PersistentCupProducts{K,F,T}
    poset::FinitePoset
    algebras::Vector{CohomologyAlgebra{K,F}}
    restrictions::Dict{Tuple{Int,Int},Vector{Matrix{K}}}
    parameters::T
    order::Symbol
    field::F
    origin::Symbol
end
field(R::PersistentCupProducts)=R.field
"""An owned copy of the indexing poset, before reversing cohomology arrows."""
cup_poset(R::PersistentCupProducts)=deepcopy(R.poset)
"""Recorded scalar or grid parameters; `nothing` for an unparameterized poset."""
cup_parameters(R::PersistentCupProducts)=deepcopy(R.parameters)
_parameter_chain(R)=R.parameters isa AbstractVector{<:Real}
function _stage(R,i)
    i=_cup_int(i;positive=true)
    i<=length(R.algebras) || throw(ArgumentError("vertex outside the represented diagram"))
    i
end
"""A copy of the algebra at a one-based vertex label."""
cohomology_algebra(R::PersistentCupProducts;stage)=deepcopy(R.algebras[_stage(R,stage)])
_ring_degree(R)=maximum(length(A.dims) for A in R.algebras)-1

function _cup_poset(P,budget)
    n=nvertices(P)
    0<n<=budget.max_stages || throw(ArgumentError("cup poset must have between 1 and max_stages vertices"))
    big(n)^2<=budget.max_entries || throw(ArgumentError("cup poset exceeds matrix-entry budget"))
    FinitePoset([leq(P,i,j) for i in 1:n,j in 1:n])
end
function _diagram_module(P,F,dims,maps)
    n=nvertices(P)
    opposite=FinitePoset([leq(P,j,i) for i in 1:n,j in 1:n])
    PModule{coeff_type(F)}(opposite,copy(dims),maps;field=F)
end

function _restriction_issues(P,As,maps,w)
    K=coeff_type(field(first(As)));D=maximum(length(A.dims) for A in As)-1
    Set(keys(maps))==Set((v,u) for (u,v) in cover_edges(P)) ||
        return ["one restriction (source,target) is required per reversed poset cover"]
    for ((s,t),fs) in maps
        A=As[s];B=As[t]
        length(fs)==D+1 || return ["restriction does not cover all degrees"]
        all(d->size(fs[d+1])==(_hdim(B,d),_hdim(A,d)),0:D) || return ["restriction matrix shape disagrees"]
        fs[1]*A.unit==B.unit || return ["restriction does not preserve the unit"]
        for p in 0:D,q in 0:D-p
            np=_hdim(A,p);nq=_hdim(A,q)
            _spend!(w;products=big(np)*nq)
            table=get(A.products,(p,q),nothing)
            for j in 1:nq,k in 1:np
                lhs=table===nothing ? zeros(K,_hdim(B,p+q)) : fs[p+q+1]*view(table,:,k+(j-1)*np)
                lhs==_multiply(B,view(fs[p+1],:,k),view(fs[q+1],:,j),p,q) ||
                    return ["restriction ($s,$t) does not preserve products in degrees ($p,$q)"]
            end
        end
    end
    # Check every cover-path relation, including unequal-length merging paths.
    for d in 0:D
        dims=[_hdim(A,d) for A in As]
        _spend!(w;entries=big(nvertices(P))^2+sum(big(dims[s])*dims[t] for (s,t) in keys(maps);init=big(0)))
        M=_diagram_module(P,field(first(As)),dims,Dict(e=>fs[d+1] for (e,fs) in maps))
        report=check_module_functoriality(M)
        report.valid || return ["degree $d restrictions are not path independent: "*join(report.issues,"; ")]
    end
    String[]
end

"""
    persistent_cup_products(P, algebras; restrictions, budget=CupProductBudget())

Build a contravariant graded algebra diagram on a finite poset P. Supply one
algebra per vertex and `restrictions[(v,u)][d]` for every cover u < v of P:
this is the degree-d matrix from the algebra at v to the algebra at u.
Missing degree matrices are zero. Maps must preserve units and products, and
all cover-path composites must agree. Inputs are copied. A supplied algebra
diagram need not have a geometric realization.
"""
function persistent_cup_products(P::AbstractPoset,algebras::AbstractVector{<:CohomologyAlgebra};
        restrictions,budget=CupProductBudget())
    Q=_cup_poset(P,budget);n=nvertices(Q)
    length(algebras)==n || throw(ArgumentError("one algebra per poset vertex is required"))
    restrictions isa AbstractDict || throw(ArgumentError("poset restrictions must be keyed by (source,target)"))
    expected=Set((v,u) for (u,v) in cover_edges(Q))
    all(e->e isa Tuple && length(e)==2 && all(x->x isa Integer && !(x isa Bool),e),keys(restrictions)) &&
        Set(keys(restrictions))==expected || throw(ArgumentError("supply exactly the reversed cover restrictions"))
    F=field(first(algebras));K=coeff_type(F);_cup_field(F);w=_CupWork(budget)
    all(A->field(A)==F,algebras) || throw(ArgumentError("all algebras must use the same field"))
    for A in algebras
        _spend!(w;entries=sum(big(length(M)) for M in values(A.products);init=big(0)))
        _cup_report(_algebra_issues(A,w),true)
    end
    As=CohomologyAlgebra{K,typeof(F)}[deepcopy(A) for A in algebras]
    D=maximum(length(A.dims) for A in As)-1;maps=Dict{Tuple{Int,Int},Vector{Matrix{K}}}()
    for (s,t) in expected
        input=restrictions[s,t]
        input isa AbstractDict || throw(ArgumentError("each restriction must be keyed by degree"))
        all(d->d isa Integer && !(d isa Bool) && 0<=d<=D,keys(input)) || throw(ArgumentError("invalid restriction degree"))
        ms=Matrix{K}[]
        for d in 0:D
            shape=(_hdim(As[t],d),_hdim(As[s],d))
            _spend!(w;entries=big(shape[1])*shape[2]);M=get(input,d,nothing)
            M===nothing || (M isa AbstractMatrix && size(M)==shape) || throw(ArgumentError("restriction has wrong shape"))
            push!(ms,M===nothing ? zeros(K,shape) : Matrix{K}(_cup_coefficients(F,M)))
        end
        maps[s,t]=ms
    end
    _cup_report(_restriction_issues(Q,As,maps,w),true)
    PersistentCupProducts(Q,As,maps,nothing,:poset,F,:supplied_algebra_diagram)
end

"""
    persistent_cup_products(algebras; parameters, restrictions, order=:sublevel,
                            budget=CupProductBudget())

Build a scalar-parameter chain. `restrictions[i][d]` maps H^d at stage i+1 to
stage i. Parameters increase for sublevels and decrease for superlevels. The
diagram is constant between events, zero before the first, and constant after
the last. For a general finite diagram use the poset overload.
"""
function persistent_cup_products(algebras::AbstractVector{<:CohomologyAlgebra};
        parameters,restrictions,order=:sublevel,budget=CupProductBudget())
    ts=_cup_parameters(parameters,order);n=length(ts)
    length(algebras)==n || throw(ArgumentError("one parameter per algebra is required"))
    length(restrictions)==n-1 || throw(ArgumentError("one restriction per adjacent stage is required"))
    n<=budget.max_stages && big(n)^2<=budget.max_entries || throw(ArgumentError("cup chain exceeds budget"))
    P=FinitePoset([i<=j for i in 1:n,j in 1:n])
    R=persistent_cup_products(P,algebras;restrictions=Dict((i+1,i)=>r for (i,r) in enumerate(restrictions)),budget)
    PersistentCupProducts(R.poset,R.algebras,R.restrictions,ts,order,R.field,R.origin)
end

"""Compose the cohomology restriction from `source` to `target`, requiring
target <= source in the indexing poset. Arguments are vertex labels."""
function cohomology_restriction(R::PersistentCupProducts{K};source,target,degree) where {K}
    s=_stage(R,source);t=_stage(R,target);d=_cup_int(degree)
    leq(R.poset,t,s) || throw(ArgumentError("cohomology restrictions require target <= source"))
    d<=_ring_degree(R) || throw(ArgumentError("degree outside the stored range"))
    M=eye(field(R),_hdim(R.algebras[s],d));u=s;edges=cover_edges(R.poset)
    while u!=t
        v=first(a for (a,b) in edges if b==u && leq(R.poset,t,a))
        M=R.restrictions[u,v][d+1]*M;u=v
    end
    M
end
"""The degree-d cohomology PModule on the opposite indexing poset."""
function cohomology_module(R::PersistentCupProducts;degree)
    d=_cup_int(degree);d<=_ring_degree(R) || throw(ArgumentError("degree outside stored range"))
    _diagram_module(R.poset,field(R),[_hdim(A,d) for A in R.algebras],
        Dict(e=>copy(ms[d+1]) for (e,ms) in R.restrictions))
end

"""Check each algebra, product-preserving map, and every cover-path relation."""
function check_persistent_cup_products(R::PersistentCupProducts;throw=false,budget=CupProductBudget())
    issues=String[];w=_CupWork(budget)
    try
        Q=_cup_poset(R.poset,budget)
        length(R.algebras)==nvertices(Q) || Base.throw(ArgumentError("algebras and poset disagree"))
        if _parameter_chain(R)
            _cup_parameters(R.parameters,R.order)
            all(leq(Q,i,j)==(i<=j) for i in 1:Q.n,j in 1:Q.n) || Base.throw(ArgumentError("scalar parameters require the ordered stage chain"))
        else
            R.order in (:poset,:sublevel,:superlevel) || Base.throw(ArgumentError("invalid diagram order"))
            if R.parameters!==nothing
                _cup_order(R.order)
                ps=R.parameters
                isempty(ps) && Base.throw(ArgumentError("parameter list must be nonempty"))
                first(ps) isa Tuple || Base.throw(ArgumentError("grid parameters must be tuples"))
                N=length(first(ps))
                N>0 && all(p->p isa Tuple && length(p)==N && all(x->x isa Real && !(x isa Bool) && isfinite(x),p),ps) ||
                    Base.throw(ArgumentError("invalid grid parameters"))
                length(ps)==Q.n || Base.throw(ArgumentError("parameters and poset disagree"))
                all(leq(Q,i,j)==all(k->R.order===:sublevel ? ps[i][k]<=ps[j][k] : ps[i][k]>=ps[j][k],1:N)
                    for i in 1:Q.n,j in 1:Q.n) || Base.throw(ArgumentError("grid parameters disagree with indexing order"))
            end
        end
        R.parameters===nothing || length(R.parameters)==Q.n || Base.throw(ArgumentError("parameters and poset disagree"))
        for A in R.algebras
            field(A)==field(R) || push!(issues,"coefficient fields disagree")
            append!(issues,_algebra_issues(A,w))
        end
        isempty(issues) && append!(issues,_restriction_issues(Q,R.algebras,R.restrictions,w))
    catch err
        err isa InterruptException && rethrow()
        push!(issues,sprint(showerror,err))
    end
    _cup_report(issues,throw)
end
