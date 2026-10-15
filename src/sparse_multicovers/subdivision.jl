# The persistent union of weighted sparse Cech posets and its order complex.
struct SparseMulticoverGrade
    radius::AlgebraicReal
    open::Bool
    depth::Int
end
Base.:(==)(a::SparseMulticoverGrade,b::SparseMulticoverGrade)=(a.radius,a.open,a.depth)==(b.radius,b.open,b.depth)
Base.isless(a::SparseMulticoverGrade,b::SparseMulticoverGrade)=isless((a.radius,a.open,-a.depth),(b.radius,b.open,-b.depth))
_reached(g,r,k)=(g.radius<r || g.radius==r && !g.open) && k<=g.depth

struct SparseMulticover
    points::Matrix{QQ}
    epsilon::QQ
    internal_epsilon::QQ
    permutation::Vector{Int}
    insertion_squared::Vector{Union{Nothing,QQ}}
    slowing_squared::Vector{Union{Nothing,QQ}}
    deaths::Vector{Union{Nothing,AlgebraicReal}}
    covering_sequences::Vector{Vector{Int}}
    subsets::Vector{Vector{Int}}
    intersection_radii::Vector{AlgebraicReal}
    intersection_centers::Vector{Vector{AlgebraicReal}}
    grades::Vector{Vector{SparseMulticoverGrade}}
    simplices::Vector{Vector{Vector{Int}}}
    simplex_grades::Vector{Vector{SparseMulticoverGrade}}
    statistics::NamedTuple
end

function _antichain(gs)
    sort!(gs);result=SparseMulticoverGrade[];depth=0
    for g in gs
        g.depth>depth || continue
        push!(result,g);depth=g.depth
    end
    result
end
function _join(a,b)
    radius=max(a.radius,b.radius)
    open=(a.radius==radius && a.open) || (b.radius==radius && b.open)
    SparseMulticoverGrade(radius,open,min(a.depth,b.depth))
end

# Each antichain describes an increasing depth staircase. Intersect their
# supports by merging the radius events and taking the smaller current depth.
# Closed and open events at the same radius remain distinct.
function _intersect_grades(a,b)
    result=SparseMulticoverGrade[]
    (isempty(a) || isempty(b)) && return result
    i=1;j=1;adepth=0;bdepth=0;depth=0
    while i<=length(a) || j<=length(b)
        if j>length(b) || (i<=length(a) && isless((a[i].radius,a[i].open),(b[j].radius,b[j].open)))
            g=a[i];adepth=g.depth;i+=1
        else
            g=b[j];bdepth=g.depth;j+=1
            if i<=length(a) && (a[i].radius,a[i].open)==(g.radius,g.open)
                adepth=a[i].depth;i+=1
            end
        end
        nextdepth=min(adepth,bdepth)
        if nextdepth>depth
            push!(result,SparseMulticoverGrade(g.radius,g.open,nextdepth));depth=nextdepth
        end
    end
    result
end

"""
    sparse_multicover(points; epsilon, max_dim=2, budget=SparseMulticoverBudget())

Construct Alonso's sparse Euclidean multicover subdivision for distinct sites.
Input matrices have one point per row; `PointCloud` is also accepted. The
requested 0 < epsilon <= 1 gives a (1+epsilon) multiplicative interleaving
of the underlying multicover spaces. Internally the paper's parameter is
epsilon/3. Geometry, weights and open/closed event thresholds are exact.

The output retains the `max_dim` skeleton; homology is complete through degree
max_dim-1. `max_dim=nothing` requests all flags, subject to explicit budgets.
The packing theorem bounds size for fixed dimension and epsilon; this bounded
reference implementation uses quadratic greedy sampling and active-basis search.
"""
function sparse_multicover(points;epsilon,max_dim=2,budget=SparseMulticoverBudget())
    input=points isa PointCloud ? point_matrix(points) : points
    input isa AbstractMatrix || throw(ArgumentError("Supply a point matrix or PointCloud."))
    n,d=size(input)
    n>0 && d>0 || throw(ArgumentError("A nonempty point cloud in positive ambient dimension is required."))
    _limit(n,budget.max_points,:max_points)
    max_dim===nothing || (max_dim isa Integer && !(max_dim isa Bool) && 0<=max_dim<=typemax(Int)) || throw(ArgumentError("max_dim must be nonnegative or nothing."))
    X=map(_rational,input);eps=_rational(epsilon)
    0<eps<=1 || throw(ArgumentError("epsilon must lie in (0,1]."))
    e=eps/3;perm,ins=_net(X)
    slow2=Union{Nothing,QQ}[v===nothing ? nothing : ((1+e)/e)^2*v for v in ins]
    deaths=Union{Nothing,AlgebraicReal}[v===nothing ? nothing : (1+3e)*sqrt(AlgebraicReal(v)) for v in slow2]
    sequences=_cover_sequences(X,slow2,e)
    intersections=_IntersectionWorkspace(X,slow2,e)
    subsets=Vector{Int}[];births=AlgebraicReal[];centers=Vector{AlgebraicReal}[]
    grades=Vector{SparseMulticoverGrade}[];work=Ref(0);candidates=0
    events=sort!(unique(AlgebraicReal[x for x in deaths if x!==nothing]))
    event_weights=[_weights(sequences,deaths,event;after=true) for event in events]
    for (position,x) in enumerate(perm)
        friends=[y for y in perm[1:position-1] if _distance2(X,x,y)<=4(1+3e)^2*slow2[x]]
        nc=big(2)^length(friends)
        _limit(big(candidates)+nc,budget.max_subsets,:max_subsets)
        for mask in big(0):nc-1
            sites=sort!([x;[friends[j] for j in eachindex(friends) if !iszero(mask & (big(1)<<(j-1)))]] )
            candidates+=1
            birth,center=_intersection!(intersections,sites,budget,work)
            death=deaths[x]
            death!==nothing && birth>death && continue
            weights=_weights(sequences,deaths,birth)
            gs=SparseMulticoverGrade[SparseMulticoverGrade(birth,false,sum(weights[sites]))]
            for (event,after) in zip(events,event_weights)
                birth<=event && (death===nothing || event<death) || continue
                push!(gs,SparseMulticoverGrade(event,true,sum(after[sites])))
            end
            push!(subsets,sites);push!(births,birth);push!(centers,center);push!(grades,_antichain(gs))
        end
    end
    order=sortperm(subsets;by=s->(length(s),Tuple(s)))
    subsets,births,centers,grades=subsets[order],births[order],centers[order],grades[order]
    m=length(subsets);limit=max_dim===nothing ? maximum(length,subsets)-1 : Int(min(max_dim,maximum(length,subsets)-1))
    successors=[Int[] for _ in 1:m];edges=0
    if limit>0
        for i in 1:m,j in i+1:m
            length(subsets[i])<length(subsets[j]) && issubset(subsets[i],subsets[j]) || continue
            edges+=1;_limit(big(m)+edges,budget.max_simplices,:max_simplices)
            push!(successors[i],j)
        end
    end
    simplices=[Vector{Int}[] for _ in 0:limit]
    grade_groups=[Vector{SparseMulticoverGrade}[] for _ in 0:limit]
    nsimp=0;ngrade=0
    chain=Int[]
    function extend(v,previous)
        gs=isempty(chain) ? grades[v] : _intersect_grades(previous,grades[v])
        push!(chain,v);nsimp+=1;ngrade+=length(gs)
        _limit(nsimp,budget.max_simplices,:max_simplices);_limit(ngrade,budget.max_grades,:max_grades)
        push!(simplices[length(chain)],copy(chain));push!(grade_groups[length(chain)],gs)
        if length(chain)<=limit
            for w in successors[v];extend(w,gs);end
        end
        pop!(chain)
    end
    for v in 1:m;extend(v,SparseMulticoverGrade[]);end
    SparseMulticover(X,eps,e,perm,ins,slow2,deaths,sequences,subsets,births,centers,grades,
        simplices,reduce(vcat,grade_groups),
        (;candidate_subsets=candidates,active_bases=work[],
            distinct_bases=length(intersections.candidates),distinct_site_bases=length(intersections.geometry),
            simplices=nsimp,grades=ngrade,
            max_dim=limit,complete_skeleton=max_dim===nothing || max_dim>=maximum(length,subsets)-1))
end

function _boundaries(r)
    cells=[collect(1:length(s)) for s in r.simplices]
    boundaries=SparseMatrixCSC{Int,Int}[]
    for d in 2:length(cells)
        faces=Dict(Tuple(s)=>i for (i,s) in enumerate(r.simplices[d-1]))
        rows=Int[];cols=Int[];values=Int[]
        for (j,s) in enumerate(r.simplices[d]),i in eachindex(s)
            push!(rows,faces[Tuple(s[k] for k in eachindex(s) if k!=i)])
            push!(cols,j);push!(values,isodd(i) ? 1 : -1)
        end
        push!(boundaries,sparse(rows,cols,values,length(cells[d-1]),length(cells[d])))
    end
    cells,boundaries
end

function _stratified_complex(r)
    cuts=sort!(unique([g.radius for gs in r.simplex_grades for g in gs]))
    cells,boundaries=_boundaries(r)
    gs=[[(2searchsortedfirst(cuts,g.radius)+(g.open ? 1 : 0),-g.depth) for g in grades] for grades in r.simplex_grades]
    MultiCriticalGradedComplex(cells,boundaries,gs),cuts
end

struct SparseMulticoverEncodingMap{I,P} <: EncodingCore.AbstractPLikeEncodingMap
    inner::I
    P::P
    cuts::Vector{AlgebraicReal}
    npoints::Int
end
EncodingCore.dimension(::SparseMulticoverEncodingMap)=2
EncodingCore.encoding_poset(pi::SparseMulticoverEncodingMap)=pi.P
EncodingCore.axes_from_encoding(pi::SparseMulticoverEncodingMap)=(copy(pi.cuts),collect(-pi.npoints:-1))
function EncodingCore.locate(pi::SparseMulticoverEncodingMap,x::AbstractVector{<:Real})
    length(x)==2 || throw(ArgumentError("Use (radius, -depth) coordinates."))
    r=_radius(x[1]);k=-x[2]
    k isa Real && isfinite(k) && isinteger(k) && k>=1 || throw(ArgumentError("The second coordinate must be a negative integral depth."))
    k>pi.npoints && return 0
    j=searchsortedlast(pi.cuts,r)
    label=2j+1-(j>0 && pi.cuts[j]==r)
    EncodingCore.locate(pi.inner,[label,-Int(k)])
end
EncodingCore.locate(pi::SparseMulticoverEncodingMap,x::Tuple)=EncodingCore.locate(pi,collect(promote(x...)))
EncodingCore.locate(pi::SparseMulticoverEncodingMap,x::NTuple{N,T}) where {N,T<:Real}=
    EncodingCore.locate(pi,collect(x))

function encode(r::SparseMulticover;degree=0,field=QQField(),cache=:auto)
    check_sparse_multicover(r;throw=true)
    degree isa Integer && !(degree isa Bool) && degree>=0 || throw(ArgumentError("degree must be nonnegative."))
    (r.statistics.complete_skeleton || degree<r.statistics.max_dim) || throw(ArgumentError("Construct the subdivision through dimension degree+1."))
    G,cuts=_stratified_complex(r)
    enc=encode(G,DataIngestion.GradedFiltration();degree=Int(degree),field,cache)
    pi=SparseMulticoverEncodingMap(enc.pi,enc.P,cuts,size(r.points,1))
    source_provenance=merge(provenance(enc),(;
        construction=(requested=:sparse_multicover,effective=:sparse_subdivision_cech,substitution=:none),
        discretization=(kind=:exact_radius_stratification,boundary=:singleton_and_open_intervals),
        approximation=provenance(r),orientation=(1,1),window=:nonnegative_radius_positive_integer_depth,
        parameter_coordinates=(:radius,:negative_depth)))
    meta=merge(enc.meta,(;sparse_multicover=provenance(r),provenance=source_provenance,
        parameter_coordinates=(:radius,:negative_depth)))
    EncodingResult(enc.P,enc.M,pi;opts=enc.opts,backend=enc.backend,
        presentation=(;sparse_multicover=r),meta)
end
