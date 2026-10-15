"""Rank-exact Betti data and minimal lower-hook resolutions on finite posets."""
module RankExact

using LinearAlgebra: I
using SparseArrays
using ..CoreModules: QQField, PrimeField, coeff_type
using ..FiniteFringe: AbstractPoset, FinitePoset, nvertices, leq, cover_edges
using ..Modules: PModule, PMorphism, map_leq, check_morphism, check_module_functoriality
import ..FieldLinAlg
import ..IndicatorResolutions
import ..AbelianCategories
import ..DerivedFunctors
import ..ChainComplexes: describe, dimensions
import ..FiniteFringe: field
import ..Results: provenance

struct RankExactBudget
    max_vertices::Int
    max_intervals::Int
    max_entries::Int
    max_generators::Int
    max_degrees::Int
    function RankExactBudget(;max_vertices=128,max_intervals=4096,max_entries=20_000_000,
                             max_generators=2048,max_degrees=256)
        xs=(max_vertices,max_intervals,max_entries,max_generators,max_degrees)
        all(x->x isa Integer && !(x isa Bool) && 0<=x<=typemax(Int),xs) ||
            throw(ArgumentError("rank-exact budgets must be nonnegative integers fitting Int"))
        new(Int.(xs)...)
    end
end
_bound(x,b,label)=x<=b || throw(ArgumentError("rank-exact $label budget exceeded ($x > $b)"))
const _Hook = Tuple{Int,Union{Int,Nothing}}

struct RankExactBetti{M}
    module_object::M
    hooks::Vector{_Hook}
    table::Matrix{Int}
    complete::Bool
    computed_through::Int
end
struct HookResolution{K,M}
    betti::RankExactBetti{M}
    terms::Vector{PModule{K}}
    summands::Vector{Vector{Int}}
    differentials::Vector{PMorphism{K}}
    coefficients::Vector{SparseMatrixCSC{K,Int}}
    augmentation::PMorphism{K}
end

_extended_leq(Q,n,a,b)=b==n+1 || (a<=n && b<=n && leq(Q,a,b))
_active(Q,h,v)=leq(Q,h[1],v) && (h[2]===nothing || !leq(Q,h[2],v))
_hook_hom(Q,source,target)=_active(Q,target,source[1]) &&
    (source[2]===nothing || (target[2]!==nothing && leq(Q,target[2],source[2])))
function _family(Q)
    n=nvertices(Q)
    _Hook[(a,b==n+1 ? nothing : b) for a in 1:n for b in 1:n+1
        if a!=b && _extended_leq(Q,n,a,b)]
end

function _check_input(M,budget)
    M.field isa Union{QQField,PrimeField} || throw(ArgumentError("rank-exact algebra requires QQ or a prime field"))
    _bound(nvertices(M.Q),budget.max_vertices,"vertices")
    entries=sum(big(d)^2 for d in M.dims;init=big(0))
    entries+=sum(big(M.dims[u])*M.dims[v] for (u,v) in cover_edges(M.Q);init=big(0))
    _bound(entries,budget.max_entries,"input matrices")
    check_module_functoriality(M;throw=true)
end

# Covariant product order on pairs (a,b), a<=b, including ALL diagonals and
# one adjoined top infinity with M(infinity)=0. Deleting diagonal vertices
# before taking ordinary Betti data gives the wrong higher degrees.
# Bauer--Brustle--Scoccola, Appendix A.1, Theorem A.2:
# https://arxiv.org/html/2604.07022v1#A1.SS1
function _kernel_diagram(M::PModule{K},budget) where {K}
    Q=M.Q;n=nvertices(Q);F=M.field;pairs=Tuple{Int,Int}[]
    for a in 1:n+1,b in 1:n+1
        _extended_leq(Q,n,a,b) || continue
        _bound(length(pairs)+1,budget.max_intervals,"interval vertices")
        push!(pairs,(a,b))
    end
    m=length(pairs);_bound(big(m)^2,budget.max_entries,"interval order matrix")
    L=BitMatrix([_extended_leq(Q,n,a,c) && _extended_leq(Q,n,b,d)
        for (a,b) in pairs,(c,d) in pairs])
    IP=FinitePoset(L);bases=Matrix{K}[];entries=big(m)^2
    for (a,b) in pairs
        da=a==n+1 ? 0 : M.dims[a]
        entries+=big(da)^2;_bound(entries,budget.max_entries,"kernel bases")
        B=a==b ? zeros(K,da,0) : b==n+1 ? Matrix{K}(I,da,da) :
            Matrix(FieldLinAlg.nullspace(F,map_leq(M,a,b)))
        push!(bases,B)
    end
    dims=[size(B,2) for B in bases];maps=Dict{Tuple{Int,Int},Matrix{K}}()
    edges=cover_edges(IP)
    entries+=sum(big(dims[u])*dims[v] for (u,v) in edges;init=big(0))
    _bound(entries,budget.max_entries,"kernel diagram")
    for (u,v) in edges
        a,b=pairs[u];c,d=pairs[v]
        maps[(u,v)]=if dims[u]==0 || dims[v]==0
            zeros(K,dims[v],dims[u])
        else
            FieldLinAlg.solve_fullcolumn(F,bases[v],map_leq(M,a,c)*bases[u])
        end
    end
    PModule{K}(IP,dims,maps;field=F),pairs,bases
end

function _hook_sum(Q,hooks,ids,F,budget)
    K=coeff_type(F);n=nvertices(Q)
    _bound(length(ids),budget.max_generators,"hook summands")
    active=[findall(j->_active(Q,hooks[ids[j]],v),eachindex(ids)) for v in 1:n]
    dims=length.(active);edges=cover_edges(Q)
    entries=sum(big(d)^2 for d in dims;init=big(0))+
        sum(big(dims[u])*dims[v] for (u,v) in edges;init=big(0))
    _bound(entries,budget.max_entries,"hook term")
    maps=Dict{Tuple{Int,Int},SparseMatrixCSC{K,Int}}()
    for (u,v) in edges
        rows=Int[];cols=Int[];indices=Dict(j=>i for (i,j) in enumerate(active[v]))
        for (i,j) in enumerate(active[u])
            k=get(indices,j,0);k==0 || (push!(rows,k);push!(cols,i))
        end
        maps[(u,v)]=sparse(rows,cols,ones(K,length(rows)),dims[v],dims[u])
    end
    PModule{K}(Q,dims,maps;field=F),active
end

"""The lower hook with support {v : birth <= v and death <= v is false}.
`death=nothing` means the principal upset on the supplied finite category."""
function hook_module(Q::AbstractPoset;birth,death=nothing,field=QQField(),budget=RankExactBudget())
    n=nvertices(Q);_bound(n,budget.max_vertices,"vertices")
    field isa Union{QQField,PrimeField} || throw(ArgumentError("hook modules require QQ or a prime field"))
    birth isa Integer && !(birth isa Bool) && 1<=birth<=n || throw(ArgumentError("invalid hook birth"))
    (death===nothing || (death isa Integer && !(death isa Bool) && 1<=death<=n &&
        birth!=death && leq(Q,birth,death))) || throw(ArgumentError("hook death must be strictly above its birth, or nothing"))
    first(_hook_sum(Q,_Hook[(Int(birth),death===nothing ? nothing : Int(death))],[1],field,budget))
end

# Each cover and kernel uses the existing ordinary resolution engines. The
# loop is kept here to check workspace bounds BEFORE each cover, stop without
# padded zero terms, and discard ordinary terms as soon as they are consumed.
function _cover_preflight(X,budget)
    s=sum(big(d) for d in X.dims;init=big(0));n=nvertices(X.Q)
    _bound(s,budget.max_generators,"upper bound on cover generators")
    _bound((big(n)+length(cover_edges(X.Q))+2)*s^2,budget.max_entries,"upper bound on cover workspace")
end
function _compute(M::PModule{K};maps::Bool,maxdegree,budget,threads) where {K}
    _check_input(M,budget)
    maxdegree===nothing || (maxdegree isa Integer && !(maxdegree isa Bool) && 0<=maxdegree<=budget.max_degrees) ||
        throw(ArgumentError("maxdegree must be nothing or a nonnegative integer within the degree budget"))
    owned=deepcopy(M);X,pairs,bases=_kernel_diagram(owned,budget)
    hooks=_family(owned.Q);hi=Dict(h=>i for (i,h) in enumerate(hooks));n=nvertices(owned.Q)
    off=[a!=b for (a,b) in pairs]
    tohook=[off[j] ? hi[(a,b==n+1 ? nothing : b)] : 0 for (j,(a,b)) in enumerate(pairs)]
    limit=maxdegree===nothing ? max(0,length(pairs)-1) : Int(maxdegree)
    rows=Vector{Int}[];terms=PModule{K}[];summands=Vector{Int}[]
    diffs=PMorphism{K}[];coeffs=SparseMatrixCSC{K,Int}[]
    aug=nothing;prev_inclusion=nothing;prev_gens=Int[];prev_kept=Int[]
    prev_active=Vector{Int}[];complete=false;used=big(0);computed=0
    relative_dims=copy(owned.dims)
    for degree in 0:limit
        _bound(degree,budget.max_degrees,"homological degree")
        _cover_preflight(X,budget)
        P,pi,gens=IndicatorResolutions.projective_cover(X;threads)
        labels=Int[p for block in gens for (p,_) in block]
        kept=findall(j->off[labels[j]],eachindex(labels))
        ids=[tohook[labels[j]] for j in kept];row=zeros(Int,length(hooks))
        for j in ids;row[j]+=1;end
        push!(rows,row);computed=degree
        # Exactness after killing diagonals determines the next relative
        # syzygy dimensions. This also certifies completion before spending
        # time resolving ordinary diagonal summands that will disappear.
        for v in 1:n
            relative_dims[v]=sum(row[j] for j in eachindex(hooks) if _active(owned.Q,hooks[j],v);init=0)-relative_dims[v]
        end
        all(>=(0),relative_dims) || error("negative relative syzygy dimension")
        if maps
            H,active=_hook_sum(owned.Q,hooks,ids,owned.field,budget)
            used+=sum(big(d)^2 for d in H.dims;init=big(0))
            _bound(used,budget.max_entries,"retained hook terms")
            push!(terms,H);push!(summands,ids)
            if degree==0
                vectors=Vector{K}[]
                for j in kept
                    p=labels[j];a,b=pairs[p]
                    localids=findall(q->leq(X.Q,labels[q],p),eachindex(labels))
                    col=searchsortedfirst(localids,j)
                    push!(vectors,bases[p]*pi.comps[p][:,col])
                end
                comps=Matrix{K}[]
                used+=sum(big(owned.dims[v])*H.dims[v] for v in 1:n;init=big(0))
                _bound(used,budget.max_entries,"retained augmentation")
                for v in 1:n
                    C=zeros(K,owned.dims[v],H.dims[v])
                    for (col,j) in enumerate(active[v]);C[:,col]=map_leq(owned,hooks[ids[j]][1],v)*vectors[j];end
                    push!(comps,C)
                end
                aug=PMorphism(H,owned,comps)
            else
                d=DerivedFunctors.Utils.compose(prev_inclusion,pi)
                used+=big(length(prev_kept))*length(kept)+sum(big(terms[end-1].dims[v])*H.dims[v] for v in 1:n;init=big(0))
                _bound(used,budget.max_entries,"retained resolution maps")
                C=zeros(K,length(prev_kept),length(kept))
                for (col,j) in enumerate(kept)
                    p=labels[j]
                    src=findall(q->leq(X.Q,labels[q],p),eachindex(labels))
                    dst=findall(q->leq(X.Q,prev_gens[q],p),eachindex(prev_gens))
                    sc=searchsortedfirst(src,j)
                    for (row,k) in enumerate(prev_kept)
                        pos=searchsortedfirst(dst,k)
                        pos<=length(dst) && dst[pos]==k || continue
                        # Maps through a killed diagonal vertex vanish in the
                        # quotient algebra. The target hook must contain a.
                        _hook_hom(owned.Q,hooks[ids[col]],hooks[summands[end-1][row]]) || continue
                        C[row,col]=d.comps[p][pos,sc]
                    end
                end
                comps=[C[prev_active[v],active[v]] for v in 1:n]
                push!(diffs,PMorphism(H,terms[end-1],comps));push!(coeffs,sparse(C))
            end
            prev_active=active
        end
        if all(iszero,relative_dims);complete=true;break;end
        degree==limit && break
        X,prev_inclusion=AbelianCategories.kernel_with_inclusion(pi)
        prev_gens=labels;prev_kept=kept
        all(iszero,X.dims) && error("ordinary syzygy vanished before the relative syzygy")
    end
    maxdegree===nothing && !complete && error("ordinary interval-poset resolution exceeded its finite-poset bound")
    lastrow=findlast(r->any(!iszero,r),rows);length_out=lastrow===nothing ? 1 : lastrow
    B=reduce(vcat,[reshape(r,1,:) for r in rows[1:length_out]])
    beta=RankExactBetti(owned,hooks,B,complete,computed)
    maps || return beta
    resize!(terms,length_out);resize!(summands,length_out)
    resize!(diffs,length_out-1);resize!(coeffs,length_out-1)
    HookResolution{K,typeof(owned)}(beta,terms,summands,diffs,coeffs,aug)
end

"""Minimal rank-exact hook multiplicities, with degrees as rows. Full finite
category by default; `maxdegree` requests a certified prefix. No relative
terms or maps are retained. Fields are exact and resource bounds explicit."""
rank_exact_betti(M::PModule;maxdegree=nothing,budget=RankExactBudget(),threads::Bool=Threads.nthreads()>1)=
    _compute(M;maps=false,maxdegree,budget,threads)
"""Construct actual minimal lower-hook terms, differentials and augmentation
for the rank-exact structure on the supplied finite poset."""
rank_exact_resolution(M::PModule;maxdegree=nothing,budget=RankExactBudget(),threads::Bool=Threads.nthreads()>1)=
    _compute(M;maps=true,maxdegree,budget,threads)
rank_exact_betti(r::HookResolution)=deepcopy(r.betti)
_data(r::RankExactBetti)=r
_data(r::HookResolution)=r.betti
rank_exact_table(r::Union{RankExactBetti,HookResolution})=copy(_data(r).table)
hook_labels(r::Union{RankExactBetti,HookResolution})=[(birth=a,death=b) for (a,b) in _data(r).hooks]
rank_exact_module(r::Union{RankExactBetti,HookResolution})=deepcopy(_data(r).module_object)
function hook_summands(r::HookResolution;degree::Integer=0)
    !(degree isa Bool) && 0<=degree<length(r.summands) || throw(ArgumentError("degree outside retained resolution"))
    [(birth=r.betti.hooks[j][1],death=r.betti.hooks[j][2]) for j in r.summands[degree+1]]
end
"""Return one owned copy of the terms, maps and coefficient model, preserving
shared domain/codomain identities within that copy."""
hook_complex(r::HookResolution)=deepcopy((terms=r.terms,differentials=r.differentials,
    augmentation=r.augmentation,coefficients=r.coefficients,summands=r.summands))
field(r::Union{RankExactBetti,HookResolution})=_data(r).module_object.field
dimensions(r::Union{RankExactBetti,HookResolution})=copy(_data(r).module_object.dims)
function rank_exact_summary(r::Union{RankExactBetti,HookResolution})
    b=_data(r)
    (kind=r isa HookResolution ? :hook_resolution : :rank_exact_betti,field=field(r),
     vertices=nvertices(b.module_object.Q),family=:all_lower_hooks,exact_structure=:rank_exact,
     betti_totals=Tuple(vec(sum(b.table;dims=2))),complete=b.complete,computed_through=b.computed_through,
     maps=r isa HookResolution,minimal=true,scope=:finite_poset)
end
describe(r::Union{RankExactBetti,HookResolution})=rank_exact_summary(r)
provenance(r::Union{RankExactBetti,HookResolution})=(scope=:finite_poset,field=field(r),
    family=:all_lower_hooks,exact_structure=:rank_exact,construction=:interval_poset_kernel,
    interval_order=:componentwise_covariant,diagonal=:retained_then_discarded,infinity=:adjoined_zero_stalk,
    degree_shift=0,complete=_data(r).complete)
Base.show(io::IO,r::RankExactBetti)=print(io,"RankExactBetti(",size(r.table),", complete=",r.complete,")")
Base.show(io::IO,r::HookResolution)=print(io,"HookResolution(",length(r.terms)," terms, complete=",r.betti.complete,")")

function signed_hook_decomposition(r::Union{RankExactBetti,HookResolution})
    b=_data(r);b.complete || throw(ArgumentError("a truncated Betti table does not determine the signed rank decomposition"))
    pos=vec(sum(b.table[1:2:end,:];dims=1));neg=vec(sum(b.table[2:2:end,:];dims=1))
    [(birth=a,death=d,positive=pos[j],negative=neg[j],coefficient=pos[j]-neg[j])
        for (j,(a,d)) in enumerate(b.hooks) if pos[j]!=0 || neg[j]!=0]
end
function reconstructed_rank(r::Union{RankExactBetti,HookResolution};source::Integer,target::Integer)
    b=_data(r);Q=b.module_object.Q;n=nvertices(Q)
    !(source isa Bool) && !(target isa Bool) && 1<=source<=n && 1<=target<=n && leq(Q,source,target) ||
        throw(ArgumentError("rank requires a comparable source and target in the finite category"))
    sum(x.coefficient for x in signed_hook_decomposition(r)
        if _active(Q,(x.birth,x.death),source) && _active(Q,(x.birth,x.death),target);init=0)
end

struct RankExactValidationSummary{R}
    report::R
end
describe(s::RankExactValidationSummary)=s.report
Base.getproperty(s::RankExactValidationSummary,k::Symbol)=k===:report ? getfield(s,k) : getproperty(getfield(s,:report),k)
Base.propertynames(s::RankExactValidationSummary,private::Bool=false)=private ? (:report,keys(s.report)...) : keys(s.report)
Base.:(==)(a::RankExactValidationSummary,b::RankExactValidationSummary)=a.report==b.report
Base.isequal(a::RankExactValidationSummary,b::RankExactValidationSummary)=isequal(a.report,b.report)
Base.hash(s::RankExactValidationSummary,h::UInt)=hash(s.report,h)
Base.show(io::IO,s::RankExactValidationSummary)=print(io,"RankExactValidationSummary(valid=",s.valid,")")

_require(ok,msg)=ok || throw(ArgumentError(msg))
function _validate_maps(r::HookResolution,budget)
    b=r.betti;M=b.module_object;Q=M.Q;n=nvertices(Q);F=M.field
    L=length(r.terms)
    _require(L==size(b.table,1)==length(r.summands) && length(r.differentials)==length(r.coefficients)==L-1,"inconsistent resolution lengths")
    _require(r.augmentation.dom===r.terms[1] && r.augmentation.cod===M,"augmentation objects disagree")
    for degree in 0:L-1
        P=r.terms[degree+1];ids=r.summands[degree+1]
        _require(all(j->1<=j<=length(b.hooks),ids),"invalid hook label")
        expected,active=_hook_sum(Q,b.hooks,ids,F,budget)
        _require(P.Q===Q && P.field==F && P.dims==expected.dims,"invalid hook term")
        _require(all(P.edge_maps[u,v]==expected.edge_maps[u,v] for (u,v) in cover_edges(Q)),"hook term maps disagree with its summands")
        _require([count(==(j),ids) for j in eachindex(b.hooks)]==b.table[degree+1,:],"hook counts disagree with Betti data")
        d=degree==0 ? r.augmentation : r.differentials[degree]
        _require(d.dom===P && (degree==0 || d.cod===r.terms[degree]),"differential objects disagree")
        check_morphism(d;throw=true)
        if degree==0
            _require(all(FieldLinAlg.rank(F,d.comps[v])==M.dims[v] for v in 1:n),"augmentation is not surjective")
        else
            before=degree==1 ? r.augmentation : r.differentials[degree-1]
            _require(all(iszero(before.comps[v]*d.comps[v]) for v in 1:n),"consecutive maps do not compose to zero")
            _require(all(FieldLinAlg.rank(F,before.comps[v])+FieldLinAlg.rank(F,d.comps[v])==d.cod.dims[v] for v in 1:n),"resolution is not exact")
            C=r.coefficients[degree];prev=r.summands[degree]
            _require(size(C)==(length(prev),length(ids)),"wrong differential coefficient shape")
            _require(all(iszero(C[i,j]) || _hook_hom(Q,b.hooks[ids[j]],b.hooks[prev[i]])
                for i in eachindex(prev),j in eachindex(ids)),"coefficient has no corresponding hook morphism")
            _require(all(iszero(C[i,j]) for i in eachindex(prev),j in eachindex(ids) if prev[i]==ids[j]),"differential is not minimal")
            for v in 1:n
                target=findall(i->_active(Q,b.hooks[prev[i]],v),eachindex(prev))
                _require(Matrix(C[target,active[v]])==d.comps[v],"coefficient witness disagrees with differential")
            end
        end
        kernels=[FieldLinAlg.nullspace(F,A) for A in d.comps]
        for u in 1:n,v in 1:n
            leq(Q,u,v) || continue
            T=map_leq(P,u,v)
            _require(FieldLinAlg.rank(F,T)==FieldLinAlg.rank(F,T*kernels[u])+FieldLinAlg.rank(F,d.comps[v]*T),
                "short sequence at degree $degree is not rank-exact")
        end
        if degree==L-1 && b.complete
            _require(all(iszero(size(A,2)) for A in kernels),"complete resolution has a nonzero final kernel")
        end
    end
end
function check_rank_exact(r::Union{RankExactBetti,HookResolution};throw::Bool=false,budget=RankExactBudget())
    issues=String[]
    try
        b=_data(r);_check_input(b.module_object,budget)
        _require(b.hooks==_family(b.module_object.Q),"hook family disagrees with the finite category")
        _require(b.computed_through>=0 && size(b.table,1)>=1 && size(b.table,1)<=b.computed_through+1 &&
            size(b.table,2)==length(b.hooks) && all(>=(0),b.table),"invalid Betti table shape or multiplicities")
        fresh=rank_exact_betti(b.module_object;maxdegree=b.complete ? nothing : b.computed_through,budget,threads=false)
        _require(b.table==fresh.table && (!b.complete || fresh.complete),"Betti data disagree with the kernel-module computation")
        r isa HookResolution && _validate_maps(r,budget)
        if b.complete
            Q=b.module_object.Q
            for u in 1:nvertices(Q),v in 1:nvertices(Q)
                leq(Q,u,v) || continue
                _require(reconstructed_rank(r;source=u,target=v)==FieldLinAlg.rank(field(r),map_leq(b.module_object,u,v)),
                    "signed hook data do not reconstruct ranks")
            end
        end
    catch e
        e isa Union{ArgumentError,DimensionMismatch,BoundsError,ErrorException} || rethrow()
        push!(issues,sprint(showerror,e))
    end
    throw && !isempty(issues) && Base.throw(ArgumentError(join(issues,"; ")))
    RankExactValidationSummary((valid=isempty(issues),issues=Tuple(issues)))
end

end
