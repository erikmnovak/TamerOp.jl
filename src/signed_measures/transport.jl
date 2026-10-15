# Balanced finite signed transport, exact residual flows, and projected queries.
const _TransportQ = Rational{BigInt}

"""Work limits for finite signed transport, projections and convolution."""
struct SignedMeasureBudget
    max_atoms::Int
    max_pairs::Int
    max_augmentations::Int
    max_queries::Int
    max_evaluations::Int
    function SignedMeasureBudget(; max_atoms=4096, max_pairs=262144,
            max_augmentations=100000, max_queries=1_000_000, max_evaluations=10_000_000)
        xs=(max_atoms,max_pairs,max_augmentations,max_queries,max_evaluations)
        all(x->x isa Integer && !(x isa Bool) && 0<=x<=typemax(Int),xs) ||
            throw(ArgumentError("signed-measure budgets must be nonnegative integers fitting Int"))
        new(Int.(xs)...)
    end
end
_transport_bound(n,b,label) = n<=b || throw(ArgumentError("signed-measure $label budget exceeded ($n > $b)"))
function _transport_rational(x)
    (x isa Union{Integer,Rational,AbstractFloat} && isfinite(x)) ||
        throw(ArgumentError("signed measures require finite integer, rational or floating coordinates and weights"))
    return _TransportQ(x)
end
function _transport_scale(scale,N)
    s=scale===nothing ? ones(_TransportQ,N) : _transport_rational.(collect(scale))
    length(s)==N && all(>(0),s) || throw(ArgumentError("scale must contain one positive value per coordinate"))
    return s
end
function _transport_atoms(pm::PointSignedMeasure{N}, budget) where N
    N>0 || throw(ArgumentError("signed measures require positive ambient dimension"))
    _transport_bound(length(pm),budget.max_atoms,"input atoms")
    check_point_signed_measure(pm;throw=true)
    atoms=Dict{NTuple{N,_TransportQ},_TransportQ}()
    for (ind,w0) in zip(pm.inds,pm.wts)
        w=_transport_rational(w0)
        x=ntuple(k->_transport_rational(pm.axes[k][ind[k]]),N)
        atoms[x]=get(atoms,x,zero(_TransportQ))+w
    end
    filter!(p->!iszero(last(p)),atoms)
    return atoms
end
function _transport_difference(mu::PointSignedMeasure{N},nu::PointSignedMeasure{M},scale,budget) where {N,M}
    N==M || throw(ArgumentError("signed measures have different ambient dimensions"))
    a=_transport_atoms(mu,budget);b=_transport_atoms(nu,budget)
    sum(values(a);init=zero(_TransportQ))==sum(values(b);init=zero(_TransportQ)) ||
        throw(ArgumentError("balanced signed transport requires equal total signed mass; no mass normalization or boundary reservoir is implicit"))
    for (x,w) in b
        a[x]=get(a,x,zero(_TransportQ))-w
    end
    filter!(p->!iszero(last(p)),a)
    _transport_bound(length(a),budget.max_atoms,"difference atoms")
    s=_transport_scale(scale,N)
    points=sort!(collect(keys(a)))
    return [ntuple(k->x[k]*s[k],N) for x in points], [a[x] for x in points], s
end

"""Retained transport marginals, flows, costs and dual potentials after cancellation."""
struct SignedTransportPlan{N}
    positive::Vector{NTuple{N,_TransportQ}}
    negative::Vector{NTuple{N,_TransportQ}}
    supply::Vector{_TransportQ}
    demand::Vector{_TransportQ}
    flow::Matrix{_TransportQ}
    costs::Matrix{_TransportQ}
    cost_errors::Matrix{_TransportQ}
    source_potential::Vector{_TransportQ}
    target_potential::Vector{_TransportQ}
end
"""Balanced signed transport cost, certified bounds and optional optimal flow."""
struct SignedTransportResult{N}
    cost::_TransportQ
    lower_bound::_TransportQ
    upper_bound::_TransportQ
    mass::_TransportQ
    metric::Symbol
    scale::Vector{_TransportQ}
    positive_count::Int
    negative_count::Int
    augmentations::Int
    plan::Union{Nothing,SignedTransportPlan{N}}
end
"""A weighted finite set of projected signed transport queries, with no sampling-error guarantee."""
struct SlicedSignedTransportResult
    cost::_TransportQ
    costs::Vector{_TransportQ}
    directions::Matrix{Float64}
    weights::Vector{_TransportQ}
    scale::Vector{_TransportQ}
end
"""Validation of signed transport, projected queries or sampled convolution."""
struct SignedTransportValidationSummary
    valid::Bool
    errors::Vector{String}
end
Base.:(==)(a::SignedTransportValidationSummary,b::SignedTransportValidationSummary)=a.valid==b.valid && a.errors==b.errors
Base.isequal(a::SignedTransportValidationSummary,b::SignedTransportValidationSummary)=isequal(a.valid,b.valid)&&isequal(a.errors,b.errors)
Base.hash(x::SignedTransportValidationSummary,h::UInt)=hash((x.valid,Tuple(x.errors)),h)
describe(x::SignedTransportValidationSummary)=(valid=x.valid,errors=copy(x.errors))
Base.show(io::IO,x::SignedTransportValidationSummary)=print(io,"SignedTransportValidationSummary(valid=",x.valid,", errors=",length(x.errors),")")
Base.show(io::IO,b::SignedMeasureBudget)=print(io,"SignedMeasureBudget(max_atoms=",b.max_atoms,", max_pairs=",b.max_pairs,")")
describe(b::SignedMeasureBudget)=(max_atoms=b.max_atoms,max_pairs=b.max_pairs,max_augmentations=b.max_augmentations,max_queries=b.max_queries,max_evaluations=b.max_evaluations)
"""Return the optimal cost for the stored cost matrix, or the weighted projected cost."""
transport_cost(r::Union{SignedTransportResult,SlicedSignedTransportResult})=r.cost
ambient_dimension(::SignedTransportResult{N}) where N = N
"""Return an owned copy of the retained transport certificate; request `witness=true` to retain it."""
function transport_plan(r::SignedTransportResult)
    r.plan===nothing && throw(ArgumentError("transport plan was not retained; request witness=true"))
    return deepcopy(r.plan)
end
"""Describe the transport contract, arithmetic, support and certified bounds."""
transport_summary(r::Union{SignedTransportResult,SlicedSignedTransportResult})=describe(r)
function describe(r::SignedTransportResult)
    return (kind=:balanced_signed_transport,cost=r.cost,bounds=(r.lower_bound,r.upper_bound),
        status=r.lower_bound==r.upper_bound ? :exact : :bounded_euclidean,
        arithmetic=:rational,metric=r.metric,scale=copy(r.scale),mass=r.mass,
        support=(positive=r.positive_count,negative=r.negative_count),
        window=:full_finite_support,boundary=:none,cancellation=:coincident,
        augmentations=r.augmentations,witness=r.plan!==nothing)
end
function describe(r::SlicedSignedTransportResult)
    return (kind=:sliced_signed_transport,status=:sampled_projections,cost=r.cost,
        projections=length(r.costs),directions=copy(r.directions),weights=copy(r.weights),
        scale=copy(r.scale),sampling_error=nothing,window=:full_finite_support,boundary=:none)
end
Base.show(io::IO,r::SignedTransportResult)=print(io,"SignedTransportResult(cost=",r.cost,", status=",describe(r).status,")")
Base.show(io::IO,r::SlicedSignedTransportResult)=print(io,"SlicedSignedTransportResult(cost=",r.cost,", projections=",length(r.costs),")")
Base.show(io::IO,p::SignedTransportPlan)=print(io,"SignedTransportPlan(",length(p.supply)," x ",length(p.demand),")")
describe(p::SignedTransportPlan)=(positive_atoms=length(p.supply),negative_atoms=length(p.demand),mass=sum(p.supply;init=zero(_TransportQ)),nonzero_flows=count(!iszero,p.flow))

function _transport_ground(x,y,metric)
    d=map((a,b)->abs(a-b),x,y)
    metric===:l1 && return (sum(d),zero(_TransportQ))
    metric===:linf && return (maximum(d),zero(_TransportQ))
    squared=sum(abs2,d)
    a=isqrt(numerator(squared));b=isqrt(denominator(squared))
    a*a==numerator(squared) && b*b==denominator(squared) && return (a//b,zero(_TransportQ))
    c=Float64(sqrt(BigFloat(squared)))
    isfinite(c) && c>0 || throw(ArgumentError("Euclidean ground cost is outside the supported floating range; rescale coordinates"))
    q=_TransportQ(c)
    # |q-sqrt(s)| = |q^2-s|/(q+sqrt(s)) <= |q^2-s|/q.
    return q,abs(q*q-squared)/q
end
mutable struct _TransportEdge
    target::Int
    reverse::Int
    capacity::_TransportQ
    cost::_TransportQ
end
function _transport_edge!(graph,u,v,capacity,cost)
    push!(graph[u],_TransportEdge(v,length(graph[v])+1,capacity,cost))
    push!(graph[v],_TransportEdge(u,length(graph[u]),zero(_TransportQ),-cost))
end
function _transport_flow(a,b,c,budget)
    m,n=size(c);nv=m+n+2;source=nv-1;sink=nv
    graph=[_TransportEdge[] for _ in 1:nv]
    total=sum(a;init=zero(_TransportQ))
    for i in 1:m;_transport_edge!(graph,source,i,a[i],zero(_TransportQ));end
    for i in 1:m,j in 1:n;_transport_edge!(graph,i,m+j,total+1,c[i,j]);end
    for j in 1:n;_transport_edge!(graph,m+j,sink,b[j],zero(_TransportQ));end
    potential=zeros(_TransportQ,nv);sent=zero(_TransportQ);iterations=0
    while sent<total
        iterations+=1;_transport_bound(iterations,budget.max_augmentations,"augmentations")
        dist=Vector{Union{Nothing,_TransportQ}}(nothing,nv);dist[source]=zero(_TransportQ)
        prev=zeros(Int,nv);edge=zeros(Int,nv);seen=falses(nv)
        for _ in 1:nv
            u=0
            for v in 1:nv
                !seen[v] && dist[v]!==nothing && (u==0 || dist[v]<dist[u]) && (u=v)
            end
            u==0 && break
            seen[u]=true
            for (k,e) in enumerate(graph[u])
                e.capacity>0 && !seen[e.target] || continue
                reduced=e.cost+potential[u]-potential[e.target]
                reduced>=0 || error("negative reduced transport cost")
                nd=dist[u]+reduced
                if dist[e.target]===nothing || nd<dist[e.target]
                    dist[e.target]=nd;prev[e.target]=u;edge[e.target]=k
                end
            end
        end
        dist[sink]===nothing && error("balanced transport has no augmenting path")
        for v in 1:nv;dist[v]===nothing || (potential[v]+=dist[v]);end
        amount=total-sent;v=sink
        while v!=source
            u=prev[v];amount=min(amount,graph[u][edge[v]].capacity);v=u
        end
        v=sink
        while v!=source
            u=prev[v];e=graph[u][edge[v]];e.capacity-=amount
            graph[v][e.reverse].capacity+=amount;v=u
        end
        sent+=amount
    end
    flow=zeros(_TransportQ,m,n)
    for i in 1:m,j in 1:n
        # The source reverse edge precedes all demand edges.
        e=graph[i][j+1];flow[i,j]=graph[e.target][e.reverse].capacity
    end
    return flow,-potential[1:m],potential[m+1:m+n],iterations
end

"""
    signed_transport(mu, nu; metric=:l1, scale=nothing, witness=false, budget=SignedMeasureBudget())

Compare finite point measures of equal signed mass by transporting the positive
part of `mu - nu` to its negative part. Coincident atoms cancel exactly and
weights are never expanded into repeated points. `scale` multiplies coordinates
before measuring distance. All supplied support is used, with no boundary reservoir.

Integer, rational and finite floating input is interpreted exactly as stored.
The `:l1` and `:linf` costs are rational and exact. For `:euclidean`, irrational
costs use rational approximations with certified lower/upper bounds on the true
optimum. `transport_cost` returns the optimum for the stored cost matrix;
`transport_summary` reports these bounds. Retaining a witness makes the marginals,
flow and dual optimality certificate available through `transport_plan`.
"""
function signed_transport(mu::PointSignedMeasure{N},nu::PointSignedMeasure;
        metric::Symbol=:l1,scale=nothing,witness::Bool=false,budget::SignedMeasureBudget=SignedMeasureBudget()) where N
    metric in (:l1,:linf,:euclidean) || throw(ArgumentError("metric must be :l1, :linf or :euclidean"))
    points,w,s=_transport_difference(mu,nu,scale,budget)
    pos=findall(>(0),w);neg=findall(<(0),w)
    x=points[pos];y=points[neg];a=w[pos];b=-w[neg];m=length(a);n=length(b)
    _transport_bound(big(m)*n,budget.max_pairs,"transport pairs")
    costs=zeros(_TransportQ,m,n);errors=similar(costs)
    for i in 1:m,j in 1:n;costs[i,j],errors[i,j]=_transport_ground(x[i],y[j],metric);end
    flow,u,v,iterations=_transport_flow(a,b,costs,budget)
    cost=sum(flow.*costs;init=zero(_TransportQ));mass=sum(a;init=zero(_TransportQ))
    radius=mass*maximum(errors;init=zero(_TransportQ))
    lo=max(zero(_TransportQ),cost-radius);hi=cost+sum(flow.*errors;init=zero(_TransportQ))
    plan=witness ? SignedTransportPlan(x,y,a,b,flow,costs,errors,u,v) : nothing
    return SignedTransportResult{N}(cost,lo,hi,mass,metric,s,m,n,iterations,plan)
end

function _projected_transport(points,w,direction)
    isempty(w) && return zero(_TransportQ)
    projected=[sum(x[k]*direction[k] for k in eachindex(direction)) for x in points]
    order=sortperm(projected);mass=zero(_TransportQ);cost=zero(_TransportQ)
    last=projected[first(order)]
    for i in order
        cost+=abs(mass)*(projected[i]-last);mass+=w[i];last=projected[i]
    end
    iszero(mass) || error("unbalanced projected transport")
    return cost
end
"""
    sliced_signed_transport(mu, nu; directions, weights=nothing, scale=nothing, budget=SignedMeasureBudget())

Compute a weighted average of one-dimensional signed W1 costs along supplied
rows of `directions`, normalized to Euclidean unit length. Positive weights
are normalized to sum to one. Each projection uses sorted cumulative masses.
The result describes this finite quadrature, with no certified error relative
to a sphere integral. Finite directions can fail to distinguish different measures.
"""
function sliced_signed_transport(mu::PointSignedMeasure{N},nu::PointSignedMeasure;
        directions::AbstractMatrix,weights=nothing,scale=nothing,budget::SignedMeasureBudget=SignedMeasureBudget()) where N
    size(directions,2)==N && size(directions,1)>0 || throw(ArgumentError("directions must have nonempty rows of ambient dimension"))
    k=size(directions,1);_transport_bound(k,budget.max_queries,"projections")
    points,w,s=_transport_difference(mu,nu,scale,budget)
    _transport_bound(big(k)*length(points)*N,budget.max_evaluations,"projection evaluations")
    dirs=Matrix{Float64}(directions)
    all(isfinite,dirs) || throw(ArgumentError("directions must be finite"))
    for i in 1:k
        d=norm(view(dirs,i,:));isfinite(d) && d>0 || throw(ArgumentError("directions must have positive finite norm"))
        dirs[i,:]./=d
    end
    weights0=weights===nothing ? ones(_TransportQ,k) : _transport_rational.(collect(weights))
    length(weights0)==k && all(>(0),weights0) || throw(ArgumentError("projection weights must be positive and match directions"))
    weights0./=sum(weights0)
    costs=[_projected_transport(points,w,_TransportQ.(dirs[i,:])) for i in 1:k]
    return SlicedSignedTransportResult(sum(weights0.*costs),costs,dirs,weights0,s)
end

function _transport_plan_issues(p::SignedTransportPlan)
    errors=String[];m=length(p.supply);n=length(p.demand)
    length(p.positive)==m && length(p.negative)==n && size(p.flow)==size(p.costs)==size(p.cost_errors)==(m,n) &&
        length(p.source_potential)==m && length(p.target_potential)==n || return ["transport certificate dimensions disagree"]
    all(isfinite,p.supply) && all(isfinite,p.demand) && all(isfinite,p.flow) && all(isfinite,p.costs) && all(isfinite,p.cost_errors) &&
        all(isfinite,p.source_potential) && all(isfinite,p.target_potential) || return ["nonfinite transport certificate"]
    all(>(0),p.supply) && all(>(0),p.demand) || push!(errors,"transport marginals must be positive")
    all(>=(0),p.flow) && all(>=(0),p.costs) && all(>=(0),p.cost_errors) || push!(errors,"negative flow, cost or error")
    vec(sum(p.flow;dims=2))==p.supply && vec(sum(p.flow;dims=1))==p.demand || push!(errors,"flow marginals disagree")
    for i in 1:m,j in 1:n
        dual=p.source_potential[i]+p.target_potential[j]
        dual<=p.costs[i,j] || push!(errors,"infeasible dual potential at ($i,$j)")
        iszero(p.flow[i,j]) || dual==p.costs[i,j] || push!(errors,"positive flow violates complementary slackness")
    end
    return errors
end
"""Check dimensions, marginals and primal/dual optimality of a retained transport result."""
function check_signed_transport(r::Union{SignedTransportResult,SignedTransportPlan,SlicedSignedTransportResult};throw::Bool=false)
    errors=String[]
    if r isa SignedTransportPlan
        append!(errors,_transport_plan_issues(r))
    elseif r isa SignedTransportResult
        length(r.scale)==ambient_dimension(r) || push!(errors,"invalid dimension")
        r.metric in (:l1,:linf,:euclidean) || push!(errors,"invalid ground metric")
        r.positive_count>=0 && r.negative_count>=0 && r.augmentations>=0 || push!(errors,"negative work or support count")
        all(isfinite,(r.lower_bound,r.cost,r.upper_bound,r.mass)) || push!(errors,"nonfinite cost or mass")
        0<=r.lower_bound<=r.cost<=r.upper_bound || push!(errors,"invalid transport bounds")
        r.mass>=0 && length(r.scale)>0 && all(x->isfinite(x)&&x>0,r.scale) || push!(errors,"invalid mass or coordinate scale")
        if r.plan!==nothing
            p=r.plan;append!(errors,_transport_plan_issues(p))
            if isempty(errors)
                sum(p.supply;init=zero(_TransportQ))==r.mass || push!(errors,"mass disagrees with witness")
                (length(p.supply),length(p.demand))==(r.positive_count,r.negative_count) || push!(errors,"support counts disagree")
                sum(p.flow.*p.costs;init=zero(_TransportQ))==r.cost || push!(errors,"cost disagrees with witness")
                for i in eachindex(p.supply),j in eachindex(p.demand)
                    c,e=_transport_ground(p.positive[i],p.negative[j],r.metric)
                    (c,e)==(p.costs[i,j],p.cost_errors[i,j]) || push!(errors,"ground cost disagrees with witness")
                end
                radius=r.mass*maximum(p.cost_errors;init=zero(_TransportQ))
                (max(zero(_TransportQ),r.cost-radius),r.cost+sum(p.flow.*p.cost_errors;init=zero(_TransportQ)))==
                    (r.lower_bound,r.upper_bound) || push!(errors,"bounds disagree with witness")
            end
        end
    else
        k=length(r.costs)
        if k==0 || length(r.weights)!=k || size(r.directions)!=(k,length(r.scale))
            push!(errors,"projection dimensions disagree")
        else
            all(isfinite,r.costs) && all(isfinite,r.weights) && isfinite(r.cost) || push!(errors,"nonfinite projection data")
            all(x->isfinite(x)&&x>0,r.scale) || push!(errors,"invalid projection scale")
            all(>=(0),r.costs) && all(>(0),r.weights) && sum(r.weights)==1 || push!(errors,"invalid projection costs or weights")
            all(isfinite,r.directions) && all(i->isapprox(norm(r.directions[i,:]),1;atol=1e-14),1:k) || push!(errors,"directions must be unit vectors")
            sum(r.weights.*r.costs)==r.cost || push!(errors,"weighted cost disagrees")
        end
    end
    throw && !isempty(errors) && Base.throw(ArgumentError(join(errors,"; ")))
    return SignedTransportValidationSummary(isempty(errors),errors)
end
