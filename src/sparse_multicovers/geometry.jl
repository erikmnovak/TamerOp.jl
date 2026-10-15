# Greedy persistent nets and certified weighted minimum enclosing balls.
struct SparseMulticoverBudget
    max_points::Int
    max_subsets::Int
    max_bases::Int
    max_simplices::Int
    max_grades::Int
    function SparseMulticoverBudget(;max_points=200,max_subsets=20_000,
            max_bases=500_000,max_simplices=100_000,max_grades=500_000)
        v=(max_points,max_subsets,max_bases,max_simplices,max_grades)
        all(x->x isa Integer && !(x isa Bool) && 0<x<=typemax(Int),v) ||
            throw(ArgumentError("Sparse multicover budgets must be positive representable integers."))
        new(Int.(v)...)
    end
end
function _limit(n,cap,name)
    n<=cap || throw(ArgumentError("Sparse multicover exceeds $name=$cap."))
end
function _rational(x)
    x isa Union{Integer,Rational,AbstractFloat} && !(x isa Bool) && isfinite(x) ||
        throw(ArgumentError("Input coordinates and epsilon must be finite integers, rationals or represented floats."))
    QQ(x)
end
function _radius(r)
    r isa Real && !(r isa Bool) && isfinite(r) && r>=0 || throw(ArgumentError("radius must be finite and nonnegative."))
    AlgebraicReal(r)
end
_distance2(X,i,j)=sum((X[i,a]-X[j,a])^2 for a in axes(X,2);init=QQ(0))

function _net(X)
    n=size(X,1);perm=Int[1];ins=Union{Nothing,QQ}[nothing for _ in 1:n]
    chosen=falses(n);chosen[1]=true
    distances=[_distance2(X,i,1) for i in 1:n]
    for _ in 2:n
        j=0
        for i in 1:n
            !chosen[i] && (j==0 || distances[i]>distances[j]) && (j=i)
        end
        iszero(distances[j]) && throw(ArgumentError("Sparse multicover requires distinct sites; repeated coordinates are not perturbed or silently merged."))
        push!(perm,j);chosen[j]=true;ins[j]=distances[j]
        for i in 1:n;distances[i]=min(distances[i],_distance2(X,i,j));end
    end
    perm,ins
end

# These caches belong to one construction, and are discarded with its workspace.
# A point set determines its rational affine solve independently of the choice
# of ordinary/slowed constraints. Each resulting candidate also has a fixed
# containment answer for each input site, shared by all queried subsets.
struct _ActiveBasisGeometry
    projector::Matrix{QQ}
    squared_distances::Vector{QQ}
end
mutable struct _IntersectionCandidate
    squared_radius::AlgebraicReal
    offset::Vector{QQ}
    slope::Vector{QQ}
    distance_constant::QQ
    distance_linear::QQ
    radius::Union{Nothing,AlgebraicReal}
    center::Union{Nothing,Vector{AlgebraicReal}}
    containment::Vector{Int8} # 0 untested, 1 inside, -1 outside
end
struct _IntersectionWorkspace
    points::Matrix{QQ}
    slopes::Vector{QQ}
    offsets::Vector{QQ}
    slowing_squared::Vector{Union{Nothing,QQ}}
    geometry::Dict{Vector{Int},Union{Nothing,_ActiveBasisGeometry}}
    candidates::Dict{Vector{Int},Vector{_IntersectionCandidate}}
end
function _IntersectionWorkspace(X,slow2,e)
    kappa=1/(3(1+e)^2);n=size(X,1)
    slopes=repeat(QQ[1,kappa],n);offsets=zeros(QQ,2n)
    for i in 1:n
        slow2[i]!==nothing && (offsets[2i]=(1-kappa)*slow2[i])
    end
    _IntersectionWorkspace(X,slopes,offsets,slow2,
        Dict{Vector{Int},Union{Nothing,_ActiveBasisGeometry}}(),
        Dict{Vector{Int},Vector{_IntersectionCandidate}}())
end
function _nonpositive_at(constant::QQ,linear::QQ,s::AlgebraicReal)
    iszero(linear) && return constant<=0
    bound=-constant/linear
    linear>0 ? s<=bound : s>=bound
end
function _candidate_contains!(W::_IntersectionWorkspace,c::_IntersectionCandidate,i)
    known=c.containment[i]
    known!=0 && return known==1
    # z = offset + slope*s. Reduce ||z-p||^2 using the quadratic
    # for s, avoiding resultants between algebraic coordinate squares.
    constant=c.distance_constant;linear=c.distance_linear
    for a in axes(W.points,2)
        delta=c.offset[a]-W.points[i,a]
        constant+=delta^2;linear+=2delta*c.slope[a]
    end
    inside=_nonpositive_at(constant,linear-1,c.squared_radius) && (W.slowing_squared[i]===nothing ||
        _nonpositive_at(constant-W.offsets[2i],linear-W.slopes[2i],c.squared_radius))
    c.containment[i]=inside ? 1 : -1
    inside
end
function _active_basis_geometry(X,sites)
    n=length(sites)-1;p0=first(sites)
    V=QQ[X[sites[j+1],a]-X[p0,a] for j in 1:n,a in axes(X,2)]
    G=V*transpose(V)
    iszero(det(G)) && return nothing
    # G is symmetric, so transpose(G\V) = transpose(V)*inv(G).
    _ActiveBasisGeometry(Matrix(transpose(G\V)),[G[j,j] for j in 1:n])
end
function _intersection_candidates(W::_IntersectionWorkspace,constraints)
    sites=[(i+1)>>1 for i in constraints];p0=first(sites)
    n=length(sites)-1;d=size(W.points,2)
    if n==0
        containment=zeros(Int8,size(W.points,1));containment[p0]=1
        return [_IntersectionCandidate(AlgebraicReal(0),W.points[p0,:],zeros(QQ,d),
            QQ(0),QQ(0),nothing,nothing,containment)]
    end
    geometry=get!(W.geometry,sites) do
        _active_basis_geometry(W.points,sites)
    end
    geometry===nothing && return _IntersectionCandidate[]
    alpha0=W.slopes[first(constraints)];beta0=W.offsets[first(constraints)]
    c0=QQ[(geometry.squared_distances[j]+beta0-W.offsets[constraints[j+1]])/2 for j in 1:n]
    c1=QQ[(alpha0-W.slopes[constraints[j+1]])/2 for j in 1:n]
    u=geometry.projector*c0;v=geometry.projector*c1
    aa=sum(v.^2);bb=2sum(u.*v)-alpha0;cc=sum(u.^2)-beta0
    roots=AlgebraicReal[]
    if iszero(aa)
        iszero(bb) || push!(roots,AlgebraicReal(-cc/bb))
    else
        disc=bb^2-4aa*cc
        if disc>=0
            q=sqrt(AlgebraicReal(disc))
            push!(roots,(-bb-q)/(2aa),(-bb+q)/(2aa))
        end
    end
    result=_IntersectionCandidate[]
    offset=W.points[p0,:]+u
    # aa = ||v||^2. When it is nonzero, aa*s^2 = -bb*s-cc;
    # otherwise v vanishes and every squared distance is rational.
    constant,linear=iszero(aa) ? (QQ(0),QQ(0)) : (-cc,-bb)
    for s in roots
        signbit(s) && continue
        # Each selected constraint is an equality by construction. Its other
        # (ordinary/slowed) bound holds exactly when the radius lies on the
        # correct side of that site's slowing threshold.
        valid=true
        for (constraint,i) in zip(constraints,sites)
            threshold=W.slowing_squared[i]
            if threshold!==nothing && (isodd(constraint) ? s>threshold : s<threshold)
                valid=false;break
            end
        end
        valid || continue
        containment=zeros(Int8,size(W.points,1));containment[sites].=1
        push!(result,_IntersectionCandidate(s,offset,v,constant,linear,nothing,nothing,containment))
    end
    result
end

# Caratheodory bounds an optimum's active basis by d+1 distinct centers.
# Subtracting their squared constraints gives a center affine in s=r^2 and
# hence a quadratic equation. Reuse its candidates, but still require every
# constraint of the requested subset to hold before accepting a candidate.
function _intersection!(W::_IntersectionWorkspace,sites,budget,work)
    constraints=Int[]
    for i in sites
        push!(constraints,2i-1)
        W.slowing_squared[i]!==nothing && push!(constraints,2i)
    end
    best=Ref{Union{Nothing,_IntersectionCandidate}}(nothing)
    function consider(selected)
        # Cache hits consume the same search budget as fresh evaluations.
        work[]+=1;_limit(work[],budget.max_bases,:max_bases)
        for j in 2:length(selected)
            ((selected[j]+1)>>1)==((selected[j-1]+1)>>1) && return
        end
        candidates=get(W.candidates,selected,nothing)
        if candidates===nothing
            candidates=_intersection_candidates(W,selected)
            W.candidates[copy(selected)]=candidates
        end
        for candidate in candidates
            previous=best[]
            previous!==nothing && candidate.squared_radius>=previous.squared_radius && continue
            all(i->_candidate_contains!(W,candidate,i),sites) || continue
            best[]=candidate
        end
    end
    selected=Int[]
    function visit(start,remaining)
        if remaining==0;consider(selected);return;end
        for i in start:length(constraints)-remaining+1
            push!(selected,constraints[i]);visit(i+1,remaining-1);pop!(selected)
        end
    end
    for n in 1:min(length(sites),size(W.points,2)+1);visit(1,n);end
    chosen=best[]
    chosen===nothing && error("Weighted-ball intersection has no active-basis certificate.")
    chosen.radius===nothing && (chosen.radius=sqrt(chosen.squared_radius))
    if chosen.center===nothing
        chosen.center=AlgebraicReal[iszero(v) ? AlgebraicReal(u) : u+v*chosen.squared_radius
            for (u,v) in zip(chosen.offset,chosen.slope)]
    end
    chosen.radius,copy(chosen.center)
end

function _cover_sequences(X,slow2,e)
    result=Vector{Int}[]
    for x in axes(X,1)
        seq=Int[x]
        while slow2[last(seq)]!==nothing
            threshold=(1+3e)^2*slow2[last(seq)]
            eligible=[i for i in axes(X,1) if slow2[i]===nothing || slow2[i]>=threshold]
            y=first(eligible);best=_distance2(X,x,y)
            for i in eligible
                distance=_distance2(X,x,i)
                if distance<best;best=distance;y=i;end
            end
            push!(seq,y)
        end
        push!(result,seq)
    end
    result
end

function _cover(sequences,deaths,r;after=false)
    [first(y for y in seq if deaths[y]===nothing || (after ? r<deaths[y] : r<=deaths[y])) for seq in sequences]
end
function _weights(sequences,deaths,r;after=false)
    result=zeros(Int,length(sequences))
    for y in _cover(sequences,deaths,r;after);result[y]+=1;end
    result
end
