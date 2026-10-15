# MMA endpoint-corner construction (Loiseaux, Carriere, Blumberg,
# arXiv:2206.02026v4, Remark 8). Matching compatibility is not presumed.
struct IntervalApproximationBudget
    max_lines::Int
    max_bars::Int
    max_match_entries::Int
    max_rectangles::Int
    max_vertices::Int
    max_matrix_entries::Int
    function IntervalApproximationBudget(;max_lines=256,max_bars=4096,
            max_match_entries=1_000_000,max_rectangles=100_000,
            max_vertices=10_000,max_matrix_entries=2_000_000)
        v=(max_lines,max_bars,max_match_entries,max_rectangles,max_vertices,max_matrix_entries)
        all(x->x isa Integer && !(x isa Bool) && 0<x<=typemax(Int),v) ||
            throw(ArgumentError("Approximation budgets must be positive representable integers."))
        new(Int.(v)...)
    end
end

struct IntervalApproximation
    window::Tuple{NTuple{2,QQ},NTuple{2,QQ}}
    resolution::QQ
    offsets::Vector{QQ}
    barcodes::Vector{Vector{Tuple{QQ,QQ}}}
    matches::Vector{Vector{Int}}
    rectangles::Vector{Vector{Tuple{NTuple{2,QQ},NTuple{2,QQ}}}}
    tracks::Vector{Vector{Tuple{Int,Int}}}
    metadata::NamedTuple
end

function _number(x)
    x isa Union{Integer,Rational,AbstractFloat} && !(x isa Bool) && isfinite(x) ||
        throw(ArgumentError("Parameters must be finite real coordinates."))
    QQ(x)
end
function _window(window)
    length(window)==2 && all(x->length(x)==2,window) || throw(ArgumentError("window must be ((xmin,ymin),(xmax,ymax))."))
    a,b=map(x->Tuple(_number.(x)),window)
    all(a .< b) || throw(ArgumentError("Window bounds must be strictly increasing."))
    (a,b)
end
function _limit(n,cap,name)
    n<=cap || throw(ArgumentError("Interval approximation exceeds $name=$cap."))
end
function _offsets(window,delta,budget)
    delta>0 || throw(ArgumentError("resolution must be positive."))
    a,b=window
    lo=floor(BigInt,(a[2]-b[1])/delta)+1
    hi=ceil(BigInt,(b[2]-a[1])/delta)-1
    _limit(max(big(0),hi-lo+1),budget.max_lines,:max_lines)
    [QQ(i)*delta for i in lo:hi]
end
_clip(window,o)=(max(window[1][1],window[1][2]-o),min(window[2][1],window[2][2]-o))
_inside(p,box)=all(box[1] .<= p) && all(p .< box[2])

function _components(boxes)
    # These boxes are the complete valid birth/death corner product. Any two
    # comparable boxes have a valid cross-corner box. Thus components equal
    # those of the corner bipartite graph; no quadratic rectangle-pair scan.
    parents=collect(eachindex(boxes))
    function root(i)
        while parents[i]!=i
            parents[i]=parents[parents[i]];i=parents[i]
        end
        i
    end
    births=Dict{NTuple{2,QQ},Int}();deaths=Dict{NTuple{2,QQ},Int}()
    for (i,(a,b)) in enumerate(boxes)
        for (corner,owners) in ((a,births),(b,deaths))
            j=get(owners,corner,0)
            if j==0;owners[corner]=i
            else;parents[root(i)]=root(j);end
        end
    end
    groups=Vector{Int}[];indices=Dict{Int,Int}()
    for i in eachindex(boxes)
        key=root(i)
        if !haskey(indices,key);push!(groups,Int[]);indices[key]=length(groups);end
        push!(groups[indices[key]],i)
    end
    groups
end

function _validate_source(M,budget)
    M isa RectilinearModule{2} && M.domain===:real ||
        throw(ArgumentError("Supply a two-parameter real RectilinearModule or supported encoding."))
    nvertices(M.finite.Q)<=budget.max_vertices || throw(ArgumentError("Source exceeds max_vertices."))
    # Endpoint-corner supports use [birth,death). Ephemeral/open-birth inputs
    # cannot be silently rounded into this convention.
    sx,sy=M.finite.Q.sizes
    for y in 1:sy,x in 1:sx
        u=x+(y-1)*sx
        for v in (iseven(x) ? u+1 : 0,iseven(y) ? u+sx : 0)
            v==0 && continue
            M.finite.dims[u]==M.finite.dims[v] &&
                FieldLinAlg.rank_dim(M.finite.field,map_leq(M.finite,u,v))==M.finite.dims[u] ||
                throw(ArgumentError("Endpoint-corner approximation requires right-continuous axes; a singleton stratum differs from its right neighbor."))
        end
    end
end

function _slice(M,window,o,budget)
    lo,hi=_clip(window,o)
    lo<hi || return Tuple{QQ,QQ}[]
    times=sort!(unique!([lo;[x for x in M.cuts[1] if lo<x<hi];
        [y-o for y in M.cuts[2] if lo<y-o<hi];hi]))
    _limit(big(length(times))^2,budget.max_matrix_entries,:max_matrix_entries)
    labels=[locate(M,(t,t+o)) for t in times[1:end-1]]
    bars=SliceInvariants.slice_barcode(M.finite,labels;values=times)
    _limit(sum(values(bars);init=0),budget.max_bars,:max_bars)
    sort!([(QQ(b),QQ(d)) for ((b,d),m) in bars for _ in 1:m if b<d])
end

function _match(A,B,budget)
    _limit(big(length(A)+length(B))^2,budget.max_match_entries,:max_match_entries)
    SliceInvariants.bottleneck_matching(A,B).a_to_b
end

function _assemble(window,delta,offsets,bars,matches,metadata,budget)
    length(bars)==length(offsets) && length(matches)==max(0,length(offsets)-1) ||
        throw(ArgumentError("Supply one barcode per line and one matching per adjacent line pair."))
    _limit(sum(length,bars;init=0),budget.max_bars,:max_bars)
    tracks=Vector{Tuple{Int,Int}}[]
    previous=Int[]
    for k in eachindex(bars)
        lo,hi=_clip(window,offsets[k])
        all(b->lo<=b[1]<b[2]<=hi,bars[k]) || throw(ArgumentError("Slice bars must be positive-length and clipped to the half-open window."))
        current=zeros(Int,length(bars[k]))
        if k>1
            m=matches[k-1]
            length(m)==length(previous) && all(j->0<=j<=length(current),m) && allunique(filter(!iszero,m)) ||
                throw(ArgumentError("Matchings must be partial injections; 0 denotes a diagonal match."))
            for i in eachindex(m)
                j=m[i]; j==0 && continue
                current[j]=previous[i]
                push!(tracks[current[j]],(k,j))
            end
        end
        for j in eachindex(current)
            current[j]!=0 && continue
            push!(tracks,[(k,j)]);current[j]=length(tracks)
        end
        previous=current
    end
    boxes=Vector{Tuple{NTuple{2,QQ},NTuple{2,QQ}}}[]
    connected_tracks=Vector{Tuple{Int,Int}}[]
    count=big(0)
    for track in tracks
        births=[(bars[k][j][1],bars[k][j][1]+offsets[k]) for (k,j) in track]
        deaths=[(bars[k][j][2],bars[k][j][2]+offsets[k]) for (k,j) in track]
        # Minimal birth corners and maximal death corners describe the same
        # upset/downset intersection without redundant rectangle products.
        births=unique(births);deaths=unique(deaths)
        births=[a for a in births if !any(b->b!=a && all(b .<= a),births)]
        deaths=[a for a in deaths if !any(b->b!=a && all(a .<= b),deaths)]
        count+=big(length(births))*length(deaths)
        _limit(count,budget.max_rectangles,:max_rectangles)
        rectangles=[(a,b) for a in births for b in deaths if all(a .< b)]
        # Arbitrary supplied matchings can join incomparable regions. Their
        # order-connected components are separate interval summands.
        for ids in _components(rectangles)
            component=rectangles[ids];push!(boxes,component)
            push!(connected_tracks,[(k,j) for (k,j) in track if any(b->begin
                lo,hi=_clip(b,offsets[k]);max(lo,bars[k][j][1])<min(hi,bars[k][j][2])
            end,component)])
        end
    end
    IntervalApproximation(window,delta,copy(offsets),deepcopy(bars),deepcopy(matches),boxes,connected_tracks,metadata)
end

"""
    interval_approximation(M; window, resolution, matching=:bottleneck, budget=...)

Construct planar candidate supports from exact diagonal slice barcodes, using
the endpoint-corner variant of MMA. The window is lower closed and upper open;
bars crossing its boundary are clipped, including essential bars. Coordinates
retain their exact represented rational values. Input must be right continuous.
`matching` may instead contain one partial injection per adjacent barcode pair.
No compatible-matching, source splitting, or global error guarantee is inferred.
Use `approximation_error` for explicitly sampled comparison lines.
"""
function interval_approximation(M::RectilinearModule;window,resolution,matching=:bottleneck,
        budget=IntervalApproximationBudget())
    _validate_source(M,budget)
    w=_window(window);d=_number(resolution);offsets=_offsets(w,d,budget)
    bars=Vector{Tuple{QQ,QQ}}[];total=0
    for o in offsets
        bar=_slice(M,w,o,budget);total+=length(bar)
        _limit(total,budget.max_bars,:max_bars);push!(bars,bar)
    end
    matches=if matching===:bottleneck
        [_match(bars[k],bars[k+1],budget) for k in 1:length(bars)-1]
    elseif matching isa AbstractVector
        all(m->all(x->x isa Integer && !(x isa Bool) && 0<=x<=typemax(Int),m),matching) ||
            throw(ArgumentError("Matching indices must be nonnegative integers."))
        [Int.(m) for m in matching]
    else
        throw(ArgumentError("matching must be :bottleneck or supplied partial injections."))
    end
    _assemble(w,d,offsets,bars,matches,(;field=M.finite.field,
        matching=matching===:bottleneck ? :bottleneck : :supplied,
        compatibility=:unverified,source=provenance(M)),budget)
end

function interval_approximation(enc::EncodingResult;extension=:none,budget=IntervalApproximationBudget(),kwargs...)
    M=ambient_module(enc;extension,budget=InterleavingBudget(max_vertices=budget.max_vertices,
        max_matrix_entries=budget.max_matrix_entries))
    interval_approximation(M;budget,kwargs...)
end
function interval_approximation(M::PModule,pi;extension=:none,budget=IntervalApproximationBudget(),kwargs...)
    R=ambient_module(M,pi;extension,budget=InterleavingBudget(max_vertices=budget.max_vertices,
        max_matrix_entries=budget.max_matrix_entries))
    interval_approximation(R;budget,kwargs...)
end
