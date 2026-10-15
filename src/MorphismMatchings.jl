"""Additive interval matchings of exact persistence morphisms on finite chains."""
module MorphismMatchings

using LinearAlgebra: I
using SparseArrays
using ..CoreModules: QQField, PrimeField
using ..FiniteFringe: nvertices, cover_edges, leq, FinitePoset, ProductOfChainsPoset
using ..Modules: PModule, PMorphism, map_leq, check_morphism, check_module_functoriality
import ..FieldLinAlg
import ..ZigzagPersistence
import ..ChainComplexes: describe, dimensions
import ..FiniteFringe: field
import ..Results: provenance

struct AdditiveMatchingBudget
    max_vertices::Int
    max_dimension::Int
    max_entries::Int
    max_intervals::Int
    function AdditiveMatchingBudget(;max_vertices=10_000,max_dimension=2048,
                                    max_entries=4_000_000,max_intervals=10_000)
        xs=(max_vertices,max_dimension,max_entries,max_intervals)
        all(x->x isa Integer && !(x isa Bool) && 0<=x<=typemax(Int),xs) ||
            throw(ArgumentError("matching budgets must be nonnegative integers fitting Int"))
        new(Int.(xs)...)
    end
end

_bound(x,b,label)=x<=b || throw(ArgumentError("additive matching $label budget exceeded ($x > $b)"))

struct _AdditiveMatchingWitness{K}
    source_supports::Vector{Tuple{Int,Int}}
    target_supports::Vector{Tuple{Int,Int}}
    source_bases::Vector{Matrix{K}}
    target_bases::Vector{Matrix{K}}
    coefficients::Matrix{K}
    reduced::Matrix{K}
    pivots::Vector{Int}
end

struct AdditiveMatchingResult{M,T,K}
    morphism::M
    chain::Vector{Int}
    parameters::Vector{T}
    ambient_vertices::Int
    scope::Symbol
    source_supports::Vector{Tuple{Int,Int}}
    target_supports::Vector{Tuple{Int,Int}}
    source_counts::Vector{Int}
    target_counts::Vector{Int}
    counts::SparseMatrixCSC{Int,Int}
    witness::Union{Nothing,_AdditiveMatchingWitness{K}}
end

function _chain(Q,chain,budget)
    n=nvertices(Q)
    if chain===nothing
        _bound(n,budget.max_vertices,"vertices")
        n==0 && return Int[]
        next=zeros(Int,n); indegree=zeros(Int,n)
        for (u,v) in cover_edges(Q)
            (next[u]==0 && indegree[v]==0) ||
                throw(ArgumentError("the input poset is not a chain; supply an explicit chain"))
            next[u]=v; indegree[v]=1
        end
        heads=findall(iszero,indegree)
        length(heads)==1 || throw(ArgumentError("the input poset is not a chain; supply an explicit chain"))
        ids=Int[];q=only(heads)
        while q!=0 && length(ids)<n
            push!(ids,q);q=next[q]
        end
        length(ids)==n && q==0 || throw(ArgumentError("invalid finite chain"))
        return ids
    end
    chain isa AbstractVector || throw(ArgumentError("chain must be a vector of vertex indices"))
    _bound(length(chain),budget.max_vertices,"vertices")
    all(q->q isa Integer && !(q isa Bool) && 1<=q<=n,chain) ||
        throw(ArgumentError("chain contains an invalid vertex"))
    ids=Int.(chain)
    length(unique(ids))==length(ids) || throw(ArgumentError("chain vertices must be distinct"))
    all(leq(Q,ids[i],ids[i+1]) for i in 1:length(ids)-1) ||
        throw(ArgumentError("chain must follow the poset order"))
    ids
end

function _parameters(values,n)
    values===nothing && return collect(1:n)
    values isa AbstractVector || throw(ArgumentError("parameters must be a vector"))
    length(values)==n || throw(ArgumentError("one parameter per selected vertex is required"))
    isempty(values) && return Int[]
    all(x->x isa Real && !(x isa Bool) && isfinite(x),values) ||
        throw(ArgumentError("parameters must be finite real values"))
    T=mapreduce(typeof,promote_type,values);isconcretetype(T) ||
        throw(ArgumentError("parameters require a concrete common type"))
    ts=T[x for x in values]
    all(ts[i]==values[i] for i in eachindex(ts)) || throw(ArgumentError("parameter conversion loses precision"))
    all(ts[i]<ts[i+1] for i in 1:n-1) || throw(ArgumentError("parameters must strictly increase"))
    ts
end

function _restricted_map(f::PMorphism{K},ids,budget,check) where {K}
    F=f.dom.field
    F isa Union{QQField,PrimeField} || throw(ArgumentError("additive matching requires QQ or a prime field"))
    f.dom.Q===f.cod.Q && f.dom.field==f.cod.field || throw(ArgumentError("morphism domains disagree"))
    if check
        check_module_functoriality(f.dom;throw=true)
        check_module_functoriality(f.cod;throw=true)
        check_morphism(f;throw=true)
    end
    ds=f.dom.dims[ids];dt=f.cod.dims[ids];n=length(ids)
    _bound(maximum(vcat(ds,dt);init=0),budget.max_dimension,"stalk dimension")
    entries=sum(big(d)^2 for d in vcat(ds,dt);init=big(0))
    entries+=sum(big(ds[i])*dt[i] for i in 1:n;init=big(0))
    entries+=sum(big(ds[i])*ds[i+1]+big(dt[i])*dt[i+1] for i in 1:n-1;init=big(0))
    _bound(entries,budget.max_entries,"input and persistence bases")
    Q=n==0 ? FinitePoset(falses(0,0)) : ProductOfChainsPoset((n,))
    A=PModule{K}(Q,ds,Dict{Tuple{Int,Int},Matrix{K}}((i,i+1)=>Matrix(map_leq(f.dom,ids[i],ids[i+1])) for i in 1:n-1);field=F)
    B=PModule{K}(Q,dt,Dict{Tuple{Int,Int},Matrix{K}}((i,i+1)=>Matrix(map_leq(f.cod,ids[i],ids[i+1])) for i in 1:n-1);field=F)
    PMorphism(A,B,Matrix{K}[Matrix(f.comps[q]) for q in ids])
end

function _persistence_basis(M::PModule{K},budget;target=false) where {K}
    n=length(M.dims)
    n==0 && return Tuple{Int,Int}[],Matrix{K}[]
    Z=ZigzagPersistence
    b=Z.ZigzagBudget(max_vertices=budget.max_vertices,max_dimension=budget.max_dimension,
        max_matrix_entries=budget.max_entries,max_representative_entries=budget.max_entries)
    z=Z.ZigzagModule(M.dims,[Matrix(map_leq(M,i,i+1)) for i in 1:n-1];
        directions=fill(:forward,n-1),field=M.field,budget=b)
    r=Z.zigzag_persistence(z;representatives=true,budget=b)
    intervals=Z.zigzag_intervals(r)
    _bound(length(intervals),budget.max_intervals,"intervals")
    perm=sortperm(intervals;by=x->target ? (x.last,x.first) : (x.first,x.last))
    supports=[(intervals[j].first,intervals[j].last) for j in perm]
    bases=Matrix{K}[]
    for t in 1:n
        B=zeros(K,M.dims[t],M.dims[t]);c=0
        for (j,(a,b)) in enumerate(supports)
            a<=t<=b || continue
            c+=1;B[:,c]=Z.zigzag_representative(r;interval=perm[j],vertex=t)
        end
        push!(bases,B)
    end
    supports,bases
end

# Gonzalez-Diaz--Soriano-Trigueros--Torras-Casas, Algorithm 1:
# https://arxiv.org/html/2006.11100v5#S3 . Source order is (birth,death),
# target order is (death,birth). Elimination is clipped to overlapping support.
function _reduce_matching(F,source,target)
    R=copy(F);owners=zeros(Int,length(target));pivots=zeros(Int,length(source))
    for j in eachindex(source)
        while true
            p=findlast(!iszero,view(R,:,j));p===nothing && break
            k=owners[p]
            if k==0
                owners[p]=j;pivots[j]=p;break
            end
            scale=R[p,j]/R[p,k]
            for i in 1:p
                overlap=max(source[j][1],target[i][1])<=min(source[j][2],target[i][2])
                R[i,j]=overlap ? R[i,j]-scale*R[i,k] : zero(eltype(R))
            end
        end
    end
    R,pivots
end

function _multiplicities(supports)
    distinct=sort!(unique(supports));index=Dict(s=>i for (i,s) in enumerate(distinct))
    counts=zeros(Int,length(distinct))
    for s in supports;counts[index[s]]+=1;end
    distinct,index,counts
end

"""Compute the additive matching of an exact module morphism, optionally on a
specified common chain. Parameters label selected vertices; the last endpoint
is censored. `witness=true` retains noncanonical persistence bases and reduction
matrices. The result owns its restricted input and is not a composition functor."""
function additive_matching(f::PMorphism{K};chain=nothing,values=nothing,
        witness::Bool=false,check::Bool=true,budget::AdditiveMatchingBudget=AdditiveMatchingBudget()) where {K}
    ids=_chain(f.dom.Q,chain,budget);ts=_parameters(values,length(ids))
    g=_restricted_map(f,ids,budget,check)
    source,A=_persistence_basis(g.dom,budget)
    target,B=_persistence_basis(g.cod,budget;target=true)
    _bound(2big(length(source))*length(target),budget.max_entries,"coefficient matrices")
    F=zeros(K,length(target),length(source))
    for t in eachindex(ids)
        born=findall(s->s[1]==t,source);isempty(born) && continue
        active_source=findall(s->s[1]<=t<=s[2],source)
        active_target=findall(s->s[1]<=t<=s[2],target)
        cols=Int[searchsortedfirst(active_source,j) for j in born]
        coeff=FieldLinAlg.solve_fullcolumn(g.dom.field,B[t],g.comps[t]*A[t][:,cols])
        F[active_target,born]=coeff
    end
    R,pivots=_reduce_matching(F,source,target)
    ss,si,sc=_multiplicities(source);tt,ti,tc=_multiplicities(target)
    rows=Int[];cols=Int[]
    for (j,i) in enumerate(pivots)
        i==0 && continue
        a,b=source[j];c,d=target[i]
        c<=a<=d<=b || error("invalid interval pair in additive reduction")
        push!(rows,si[(a,b)]);push!(cols,ti[(c,d)])
    end
    counts=sparse(rows,cols,ones(Int,length(rows)),length(ss),length(tt))
    w=witness ? _AdditiveMatchingWitness(source,target,A,B,F,R,pivots) : nothing
    AdditiveMatchingResult{typeof(g),eltype(ts),K}(g,ids,ts,nvertices(f.dom.Q),
        chain===nothing ? :finite_chain : :chain_restriction,ss,tt,sc,tc,counts,w)
end

function _interval(r,s)
    a,b=s
    (first=a,last=b,birth=r.parameters[a],death=b==length(r.chain) ? nothing : r.parameters[b+1],
     left_closed=true,right_closed=false,terminal=b==length(r.chain) ? :censored : :observed)
end
function matched_intervals(r::AdditiveMatchingResult)
    rows,cols,ns=findnz(r.counts)
    [(source=_interval(r,r.source_supports[i]),target=_interval(r,r.target_supports[j]),multiplicity=k)
        for (i,j,k) in zip(rows,cols,ns)]
end
function unmatched_intervals(r::AdditiveMatchingResult;side::Symbol=:source)
    side in (:source,:target) || throw(ArgumentError("side must be :source or :target"))
    supports,counts,used=side===:source ? (r.source_supports,r.source_counts,vec(sum(r.counts;dims=2))) :
        (r.target_supports,r.target_counts,vec(sum(r.counts;dims=1)))
    [(interval=_interval(r,supports[i]),multiplicity=counts[i]-used[i])
        for i in eachindex(counts) if counts[i]>used[i]]
end
function matching_intervals(r::AdditiveMatchingResult;side::Symbol=:image)
    side in (:source,:target,:image) || throw(ArgumentError("side must be :source, :target or :image"))
    if side!==:image
        supports,counts=side===:source ? (r.source_supports,r.source_counts) : (r.target_supports,r.target_counts)
        return [(interval=_interval(r,s),multiplicity=k) for (s,k) in zip(supports,counts)]
    end
    counts=Dict{Tuple{Int,Int},Int}()
    rows,cols,ns=findnz(r.counts)
    for (i,j,k) in zip(rows,cols,ns)
        s=(max(r.source_supports[i][1],r.target_supports[j][1]),min(r.source_supports[i][2],r.target_supports[j][2]))
        counts[s]=get(counts,s,0)+k
    end
    [(interval=_interval(r,s),multiplicity=counts[s]) for s in sort!(collect(keys(counts)))]
end
matching_chain(r::AdditiveMatchingResult)=copy(r.chain)
matching_parameters(r::AdditiveMatchingResult)=copy(r.parameters)
matching_morphism(r::AdditiveMatchingResult)=deepcopy(r.morphism)
function additive_matching_witness(r::AdditiveMatchingResult)
    r.witness===nothing && throw(ArgumentError("request witness=true when computing the matching"))
    w=r.witness
    NamedTuple{fieldnames(typeof(w))}(map(k->deepcopy(getfield(w,k)),fieldnames(typeof(w))))
end
field(r::AdditiveMatchingResult)=r.morphism.dom.field
dimensions(r::AdditiveMatchingResult)=(source=copy(r.morphism.dom.dims),target=copy(r.morphism.cod.dims))
provenance(r::AdditiveMatchingResult)=(construction=:gonzalez_diaz_soriano_trigueros_torras_casas,
    field=field(r),scope=r.scope,chain=Tuple(r.chain),ambient_vertices=r.ambient_vertices,
    endpoints=:half_open_with_censored_terminal,composition=:not_functorial,retained_map=:owned_chain_restriction)
additive_matching_summary(r::AdditiveMatchingResult)=(kind=:additive_matching,field=field(r),
    vertices=length(r.chain),source_intervals=sum(r.source_counts),target_intervals=sum(r.target_counts),
    matched=sum(r.counts),unmatched_source=sum(r.source_counts)-sum(r.counts),
    unmatched_target=sum(r.target_counts)-sum(r.counts),witness=r.witness!==nothing,scope=r.scope)
describe(r::AdditiveMatchingResult)=additive_matching_summary(r)
Base.show(io::IO,r::AdditiveMatchingResult)=print(io,"AdditiveMatchingResult(",sum(r.counts)," matched, ",field(r),")")
describe(w::_AdditiveMatchingWitness)=(kind=:additive_matching_witness,source_intervals=length(w.source_supports),
    target_intervals=length(w.target_supports),matched=count(!iszero,w.pivots),bases=:noncanonical)
Base.show(io::IO,w::_AdditiveMatchingWitness)=print(io,"_AdditiveMatchingWitness(",size(w.coefficients),")")

struct AdditiveMatchingValidationSummary{R}
    report::R
end
describe(s::AdditiveMatchingValidationSummary)=s.report
Base.getproperty(s::AdditiveMatchingValidationSummary,k::Symbol)=k===:report ? getfield(s,k) : getproperty(getfield(s,:report),k)
Base.propertynames(s::AdditiveMatchingValidationSummary,private::Bool=false)=private ? (:report,keys(s.report)...) : keys(s.report)
Base.:(==)(a::AdditiveMatchingValidationSummary,b::AdditiveMatchingValidationSummary)=a.report==b.report
Base.isequal(a::AdditiveMatchingValidationSummary,b::AdditiveMatchingValidationSummary)=isequal(a.report,b.report)
Base.hash(s::AdditiveMatchingValidationSummary,h::UInt)=hash(s.report,h)
Base.show(io::IO,s::AdditiveMatchingValidationSummary)=print(io,"AdditiveMatchingValidationSummary(valid=",s.valid,")")

function check_additive_matching(r::AdditiveMatchingResult;throw::Bool=false,
        budget::AdditiveMatchingBudget=AdditiveMatchingBudget())
    issues=String[]
    try
        length(r.chain)==nvertices(r.morphism.dom.Q) || Base.throw(ArgumentError("chain size disagrees"))
        r.scope in (:finite_chain,:chain_restriction) || Base.throw(ArgumentError("invalid matching scope"))
        length(unique(r.chain))==length(r.chain) && all(q->1<=q<=r.ambient_vertices,r.chain) ||
            Base.throw(ArgumentError("invalid ambient vertex provenance"))
        fresh=additive_matching(r.morphism;values=r.parameters,witness=r.witness!==nothing,budget)
        (r.source_supports==fresh.source_supports && r.target_supports==fresh.target_supports &&
         r.source_counts==fresh.source_counts && r.target_counts==fresh.target_counts && r.counts==fresh.counts) ||
            Base.throw(ArgumentError("matching disagrees with the retained morphism"))
        if r.witness!==nothing
            w=r.witness;v=fresh.witness
            all(getfield(w,k)==getfield(v,k) for k in fieldnames(typeof(w))) ||
                Base.throw(ArgumentError("matching witness disagrees with the retained morphism"))
        end
    catch e
        e isa Union{ArgumentError,DimensionMismatch,BoundsError,ErrorException} || rethrow()
        push!(issues,sprint(showerror,e))
    end
    throw && !isempty(issues) && Base.throw(ArgumentError(join(issues,"; ")))
    AdditiveMatchingValidationSummary((valid=isempty(issues),issues=Tuple(issues)))
end

end
