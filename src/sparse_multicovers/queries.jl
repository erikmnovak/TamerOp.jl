# Mathematical set inclusions, retained cell grades, and owned validation.
function sparse_covering(r::SparseMulticover;radius)
    t=_radius(radius)
    mapping=_cover(r.covering_sequences,r.deaths,t)
    weights=zeros(Int,size(r.points,1))
    for j in mapping;weights[j]+=1;end
    (;radius=t,mapping,weights,active=findall(>(0),weights))
end
function sparse_ball_radii(r::SparseMulticover;radius)
    t=_radius(radius);e=r.internal_epsilon;kappa=1/(3(1+e)^2)
    Union{Nothing,AlgebraicReal}[
        r.deaths[i]!==nothing && t>r.deaths[i] ? nothing :
        r.slowing_squared[i]===nothing ? t :
        sqrt(min(t^2,kappa*t^2+(1-kappa)*r.slowing_squared[i])) for i in axes(r.points,1)]
end
function _point(r,point)
    length(point)==size(r.points,2) || throw(ArgumentError("Point has the wrong ambient dimension."))
    AlgebraicReal[x isa AlgebraicReal ? x : _rational(x) for x in point]
end
function sparse_multicover_at(r::SparseMulticover;point,radius)
    p=_point(r,point);t=_radius(radius)
    radii=sparse_ball_radii(r;radius=t);cover=sparse_covering(r;radius=t)
    distances=[sum((p[a]-r.points[i,a])^2 for a in eachindex(p)) for i in axes(r.points,1)]
    sites=[i for i in eachindex(radii) if radii[i]!==nothing && distances[i]<=radii[i]^2]
    original=findall(<=(t^2),distances)
    (;point=p,radius=t,sparse_depth=sum(cover.weights[sites]),exact_depth=length(original),
        sparse_sites=sites,exact_sites=original,weights=cover.weights)
end

"""
    multicover_witness(result; point, radius, depth, direction=:to_sparse)

Check the identity-on-space inclusion at a specified point. `:to_sparse` maps
Cov(radius,depth) into SCov((1+epsilon)radius,depth). `:to_exact` maps SCov into
Cov(((1+2e)/(1+e))radius,depth), where e=epsilon/3. Returned site lists certify
membership through the covering map. The maps of spaces are identities, so
their compositions and naturality commute; these are not supplied chain-basis
matrices. A point outside the source set is rejected.
"""
function multicover_witness(r::SparseMulticover;point,radius,depth,direction=:to_sparse)
    depth isa Integer && !(depth isa Bool) && 1<=depth<=size(r.points,1) || throw(ArgumentError("depth must be in 1:npoints."))
    direction in (:to_sparse,:to_exact) || throw(ArgumentError("direction must be :to_sparse or :to_exact."))
    p=_point(r,point);t=_radius(radius);e=r.internal_epsilon
    source=sparse_multicover_at(r;point=p,radius=t)
    factor=direction===:to_sparse ? 1+3e : (1+2e)/(1+e)
    target=sparse_multicover_at(r;point=p,radius=factor*t)
    if direction===:to_sparse
        source.exact_depth>=depth || throw(ArgumentError("Point is outside the source exact multicover."))
        c=sparse_covering(r;radius=factor*t)
        original=source.exact_sites;sites=sort!(unique(c.mapping[original]))
        valid=issubset(sites,target.sparse_sites) && sum(c.weights[sites])>=depth
    else
        source.sparse_depth>=depth || throw(ArgumentError("Point is outside the source sparse multicover."))
        c=sparse_covering(r;radius=t);sites=source.sparse_sites
        original=findall(i->c.mapping[i] in sites,eachindex(c.mapping))
        valid=issubset(original,target.exact_sites) && length(original)>=depth
    end
    valid || error("Sparse multicover inclusion certificate failed.")
    (;valid,point=p,depth=Int(depth),direction,source_radius=t,target_radius=factor*t,
        factor,original_sites=original,sparse_sites=sites,map=:identity_on_ambient_space)
end

function sparse_multicover_cells(r::SparseMulticover;radius=nothing,depth=1)
    depth isa Integer && !(depth isa Bool) && 1<=depth<=size(r.points,1) || throw(ArgumentError("depth must be in 1:npoints."))
    t=radius===nothing ? nothing : _radius(radius)
    output=NamedTuple[];id=0
    for (d,group) in enumerate(r.simplices),(j,simplex) in enumerate(group)
        id+=1;gs=r.simplex_grades[id]
        t!==nothing && !any(g->_reached(g,t,depth),gs) && continue
        push!(output,(;dimension=d-1,index=j,subsets=deepcopy(r.subsets[simplex]),
            grades=[(;radius=g.radius,lower_closed=!g.open,depth=g.depth) for g in gs]))
    end
    output
end

function sparse_multicover_summary(r::SparseMulticover)
    e=r.internal_epsilon
    (;kind=:sparse_multicover,points=size(r.points,1),ambient_dimension=size(r.points,2),
        metric=:euclidean,epsilon=r.epsilon,paper_epsilon=e,
        exact_to_sparse_factor=1+3e,sparse_to_exact_factor=(1+2e)/(1+e),
        interleaving_factor=1+r.epsilon,subsets=length(r.subsets),
        cells=length.(r.simplices),statistics=r.statistics,
        radius_boundaries=:closed_intersections_open_weight_increases,
        geometry=:exact_algebraic,construction=:sparse_subdivision_cech)
end
describe(r::SparseMulticover)=sparse_multicover_summary(r)
provenance(r::SparseMulticover)=merge(sparse_multicover_summary(r),
    (;reference="https://arxiv.org/abs/2411.06986v2",hypotheses=:finite_distinct_euclidean_sites,
        approximation=:multiplicative_multicover_interleaving,
        size_guarantee=:fixed_dimension_and_epsilon,net_algorithm=:quadratic_greedy,
        intersection_algorithm=:exact_active_bases,chain_coordinates=:subsets_and_flags))
Base.show(io::IO,r::SparseMulticover)=print(io,"SparseMulticover(",size(r.points,1)," sites, ",length(r.subsets)," subsets, ",r.statistics.simplices," cells; epsilon=",r.epsilon,")")

struct SparseMulticoverValidationSummary
    valid::Bool
    issues::Vector{String}
    geometry_checked::Bool
end
Base.:(==)(a::SparseMulticoverValidationSummary,b::SparseMulticoverValidationSummary)=(a.valid,a.issues,a.geometry_checked)==(b.valid,b.issues,b.geometry_checked)
Base.isequal(a::SparseMulticoverValidationSummary,b::SparseMulticoverValidationSummary)=isequal((a.valid,a.issues,a.geometry_checked),(b.valid,b.issues,b.geometry_checked))
Base.hash(r::SparseMulticoverValidationSummary,h::UInt)=hash((r.valid,r.issues,r.geometry_checked),h)
describe(r::SparseMulticoverValidationSummary)=(;valid=r.valid,issues=copy(r.issues),geometry_checked=r.geometry_checked)
Base.show(io::IO,r::SparseMulticoverValidationSummary)=print(io,"SparseMulticoverValidationSummary(valid=",r.valid,", geometry_checked=",r.geometry_checked,")")
function check_sparse_multicover(r::SparseMulticover;throw=false,recompute=false,budget=SparseMulticoverBudget())
    issues=String[]
    try
        n,d=size(r.points)
        0<r.epsilon<=1 && r.internal_epsilon==r.epsilon/3 || error("Invalid approximation parameter.")
        length(r.permutation)==length(r.insertion_squared)==length(r.slowing_squared)==length(r.deaths)==length(r.covering_sequences)==n || error("Net storage lengths disagree.")
        perm,ins=_net(r.points)
        perm==r.permutation && ins==r.insertion_squared || error("Incorrect persistent net.")
        e=r.internal_epsilon;kappa=1/(3(1+e)^2)
        expected=Union{Nothing,QQ}[v===nothing ? nothing : ((1+e)/e)^2*v for v in ins]
        expected==r.slowing_squared || error("Incorrect slowing times.")
        expected_deaths=Union{Nothing,AlgebraicReal}[v===nothing ? nothing : (1+3e)*sqrt(AlgebraicReal(v)) for v in expected]
        expected_deaths==r.deaths || error("Incorrect disappearance times.")
        _cover_sequences(r.points,expected,e)==r.covering_sequences || error("Incorrect covering sequences.")
        length(r.subsets)==length(r.intersection_radii)==length(r.intersection_centers)==length(r.grades) || error("Subset storage lengths disagree.")
        !isempty(r.subsets) && allunique(r.subsets) || error("Missing or repeated subset vertices.")
        r.statistics.max_dim==length(r.simplices)-1 || error("Skeleton dimension disagrees with storage.")
        r.statistics.complete_skeleton==(length(r.simplices)==maximum(length,r.subsets)) || error("Incorrect full-skeleton claim.")
        r.statistics.simplices==sum(length,r.simplices) || error("Cell count disagrees with storage.")
        r.statistics.grades==sum(length,r.simplex_grades) || error("Grade count disagrees with storage.")
        events=sort!(unique(AlgebraicReal[x for x in r.deaths if x!==nothing]))
        event_weights=[_weights(r.covering_sequences,r.deaths,event;after=true) for event in events]
        for (sites,t,z,gs) in zip(r.subsets,r.intersection_radii,r.intersection_centers,r.grades)
            !isempty(sites) && issorted(sites) && allunique(sites) && all(i->1<=i<=n,sites) || error("Invalid site subset.")
            length(z)==d && t>=0 || error("Invalid intersection witness.")
            rs=sparse_ball_radii(r;radius=t)
            all(i->rs[i]!==nothing && sum((z[a]-r.points[i,a])^2 for a in 1:d)<=rs[i]^2,sites) || error("Invalid sparse-ball intersection.")
            gs==_antichain(copy(gs)) && all(g->g.radius>=t && 1<=g.depth<=n,gs) || error("Invalid subset grade antichain.")
            weights=_weights(r.covering_sequences,r.deaths,t)
            expected_grades=SparseMulticoverGrade[SparseMulticoverGrade(t,false,sum(weights[sites]))]
            death=minimum((r.deaths[i] for i in sites if r.deaths[i]!==nothing);init=Inf)
            for (event,weights) in zip(events,event_weights)
                t<=event<death || continue
                push!(expected_grades,SparseMulticoverGrade(event,true,sum(weights[sites])))
            end
            gs==_antichain(expected_grades) || error("Subset weights or open thresholds disagree.")
        end
        # Check the declared order-complex skeleton, not just its boundary
        # equation: an unrelated simplicial complex can also satisfy d^2=0.
        successors=[Int[] for _ in r.subsets];nedges=0
        if length(r.simplices)>1
            for u in eachindex(r.subsets),v in eachindex(r.subsets)
                length(r.subsets[u])<length(r.subsets[v]) && issubset(r.subsets[u],r.subsets[v]) || continue
                nedges+=1;_limit(nedges,budget.max_simplices,:max_simplices)
                push!(successors[u],v)
            end
        end
        previous=Set{Tuple}([()])
        for group in r.simplices
            actual=Set(Tuple(flag) for flag in group)
            length(actual)==length(group) || error("Repeated subdivision flags.")
            expected_flags=Set{Tuple}()
            for flag in previous
                for v in (isempty(flag) ? eachindex(r.subsets) : successors[last(flag)])
                    push!(expected_flags,(flag...,v))
                    _limit(length(expected_flags),budget.max_simplices,:max_simplices)
                end
            end
            actual==expected_flags || error("Stored flags are not the complete declared order-complex skeleton.")
            previous=actual
        end
        cells,boundaries=_boundaries(r)
        length(r.simplex_grades)==sum(length,cells) || error("Flag grade counts disagree.")
        id=0
        for group in r.simplices,flag in group
            id+=1;gs=r.grades[first(flag)]
            for v in flag[2:end];gs=_antichain([_join(a,b) for a in gs for b in r.grades[v]]);end
            gs==r.simplex_grades[id] || error("Flag grades disagree with the intersection of its vertex grades.")
        end
        for i in 1:length(boundaries)-1
            iszero(boundaries[i]*boundaries[i+1]) || error("Subdivision boundary does not square to zero.")
        end
        if recompute
            fresh=sparse_multicover(r.points;epsilon=r.epsilon,max_dim=r.statistics.complete_skeleton ? nothing : r.statistics.max_dim,budget)
            fresh.subsets==r.subsets && fresh.intersection_radii==r.intersection_radii &&
                fresh.grades==r.grades && fresh.simplices==r.simplices && fresh.simplex_grades==r.simplex_grades || error("Construction differs from recomputed geometry and grades.")
        end
    catch err
        err isa InterruptException && rethrow()
        push!(issues,sprint(showerror,err))
    end
    throw && !isempty(issues) && Base.throw(ArgumentError(join(issues,"; ")))
    SparseMulticoverValidationSummary(isempty(issues),issues,recompute && isempty(issues))
end
