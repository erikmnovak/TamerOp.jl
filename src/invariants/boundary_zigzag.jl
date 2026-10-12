# The Dey--Kim--Memoli boundary cap, with distinct path occurrences.
"""
    BoundaryZigzagResult

Exact barcode of the boundary cap of a connected convex finite grid interval.
`boundary_path` retains repeated labels as distinct occurrences; `boundary_bars`
has inclusive occurrence endpoints and positive multiplicities. Full bars count
the generalized rank of the selected finite interval, not of a continuous worm.
Construct with `boundary_zigzag`; no ambient interval decomposition is implied.
"""
struct BoundaryZigzagResult{P,F}
    Q::P
    field::F
    vertices::Vector{Int}
    labels::Vector{Int}
    indices::Vector{NTuple{2,Int}}
    arrows::Vector{Int}
    dims::Vector{Int}
    bars::Vector{NTuple{3,Int}}
    rank::Int
    queries::Int
end
source_poset(r::BoundaryZigzagResult) = r.Q
selected_vertices(r::BoundaryZigzagResult) = copy(r.vertices)
"""Return distinct path occurrences, their finite labels, lattice indices and arrow signs (+1 forward)."""
boundary_path(r::BoundaryZigzagResult) =
    (;vertices=copy(r.labels), indices=copy(r.indices), arrows=copy(r.arrows), dimensions=copy(r.dims))
"""Return inclusive `(first, last, multiplicity)` bars in boundary-occurrence coordinates."""
boundary_bars(r::BoundaryZigzagResult) = copy(r.bars)
generalized_rank(r::BoundaryZigzagResult) = r.rank
generalized_rank_summary(r::BoundaryZigzagResult) =
    (;kind=:boundary_zigzag, rank=r.rank, occurrences=length(r.labels),
      distinct_labels=length(unique(r.labels)), intervals=length(r.bars), queries=r.queries,
      interpretation=:finite_grid_interval, field=r.field)
describe(r::BoundaryZigzagResult) = generalized_rank_summary(r)
Base.show(io::IO,r::BoundaryZigzagResult) =
    print(io,"BoundaryZigzagResult(rank=",r.rank,", occurrences=",length(r.labels),")")

"""
    check_boundary_zigzag(result; throw=false)

Check retained occurrence/barcode shapes, positive multiplicities, stalk-dimension
reconstruction, and the full-bar count. This does not recompute input maps.
"""
function check_boundary_zigzag(r::BoundaryZigzagResult; throw::Bool=false)
    issues=String[]; n=length(r.labels)
    n>=2 && length(r.indices)==length(r.dims)==n && length(r.arrows)==n-1 ||
        push!(issues,"boundary occurrence shapes disagree")
    all(a->a in (-1,1),r.arrows) || push!(issues,"arrows must be +1 or -1")
    all(v->v in r.vertices,r.labels) || push!(issues,"path leaves the selected interval")
    all(b->1<=b[1]<=b[2]<=n && b[3]>0,r.bars) || push!(issues,"invalid boundary bars")
    if isempty(issues)
        reconstructed=[sum((big(m) for (a,b,m) in r.bars if a<=i<=b);init=big(0)) for i in 1:n]
        reconstructed==r.dims || push!(issues,"bars do not reconstruct occurrence dimensions")
        sum((big(m) for (a,b,m) in r.bars if a==1 && b==n);init=big(0))==r.rank ||
            push!(issues,"full bars disagree with the region rank")
    end
    report=GeneralizedRankValidationSummary((;valid=isempty(issues),issues,interpretation=:finite_grid_interval))
    throw && !report.valid && Base.throw(ArgumentError(join(issues,"; ")))
    report
end

"""
    boundary_zigzag(M, grid; vertices, budget=GeneralizedRankBudget())

Compute the boundary-cap barcode of a connected convex interval in the positive
2D grid's **index lattice**. Reverse the lower fence, join it to the upper fence,
and retain every occurrence, including repeated points and identity arrows.
The full-bar multiplicity is checked against the canonical region rank.

Uses exact ranks of all contiguous path subintervals and finite differences to
recover the zigzag barcode with quadratically many queries in the path length.
This is a bounded reference algorithm, not a specialized fast zigzag reduction.
`max_queries` includes the region check; algebra budgets count repeated stalks.
Nonuniform axis spacing is permitted: the domain is finite labels, not all
continuous points between those coordinates. QQ and prime fields are supported.
"""
function boundary_zigzag(M::PModule,pi;vertices,budget::GeneralizedRankBudget=GeneralizedRankBudget())
    ax=_gr_grid(M,pi,budget)
    labels=check_generalized_rank_region(M;vertices,budget,throw=true).vertices
    nx=length(ax[1]); xy(v)=((v-1)%nx+1,div(v-1,nx)+1)
    minima=sort!([v for v in labels if !any(w->w!=v && leq(M.Q,w,v),labels)];by=v->xy(v)[1])
    maxima=sort!([v for v in labels if !any(w->w!=v && leq(M.Q,v,w),labels)];by=v->xy(v)[1])
    label(p)=p[1]+(p[2]-1)*nx
    lower=Int[first(minima)];upper=Int[first(maxima)]
    for i in 2:length(minima)
        push!(lower,label(max.(xy(minima[i-1]),xy(minima[i]))),minima[i])
    end
    for i in 2:length(maxima)
        push!(upper,label(min.(xy(maxima[i-1]),xy(maxima[i]))),maxima[i])
    end
    path=[reverse(lower);upper]; n=length(path)
    all(in(labels),path) || error("Boundary cap left a validated grid interval")
    _gr_bound(n,budget.max_vertices,"boundary occurrences")
    queries=div(big(n)*(n+1),2)+1
    _gr_bound(queries,budget.max_queries,"boundary rank queries")
    _gr_bound(big(n+2)^2,budget.max_matrix_entries,"boundary rank table")
    arrows=[leq(M.Q,path[i],path[i+1]) ? 1 : -1 for i in 1:n-1]
    ranks=zeros(Int,n+2,n+2)
    for a in 1:n,b in a:n
        # Only adjacent arrows, never the induced grid order on path labels.
        edges=[arrows[i]==1 ? (i-a+1,i-a+2) : (i-a+2,i-a+1) for i in a:b-1]
        ranks[a+1,b+1]=_gr_diagram(M,path[a:b],edges,budget)
    end
    bars=NTuple{3,Int}[]
    for a in 1:n,b in a:n
        multiplicity=ranks[a+1,b+1]-ranks[a,b+1]-ranks[a+1,b+2]+ranks[a,b+2]
        multiplicity>=0 || error("Boundary rank inversion gave a negative zigzag multiplicity")
        multiplicity>0 && push!(bars,(a,b,multiplicity))
    end
    rank=generalized_rank(M;vertices=labels,budget)
    rank==ranks[2,n+1] || error("Boundary full-bar rank disagrees with the selected finite interval")
    result=BoundaryZigzagResult(M.Q,M.field,labels,path,xy.(path),arrows,M.dims[path],bars,rank,Int(queries))
    check_boundary_zigzag(result;throw=true)
    result
end
