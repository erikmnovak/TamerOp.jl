# Retain construction-owned identities. Drawing never rediscovers cells.
function _rhomboid_retained_geometry(groups,radii2,X,G)
    n,D=size(X)
    geometry=Vector{Vector{Tuple}}();sources=NamedTuple[]
    for group in groups, cell in group
        sliced=cell isa _RhomboidDepthCell
        carrier=sliced ? cell.carrier : cell
        inside=Tuple(_rhomboid_indices(carrier.inside,n))
        on=Tuple(_rhomboid_indices(carrier.on,n));q=length(on)
        vertices=Tuple[]
        # Dimension two needs a cyclic boundary. All other dimensions retain
        # the complete corner set; only their 2-skeleton is drawn.
        bits=q==2 ? (0,1,3,2) : (0:(2^q-1))
        for b in bits
            sites=vcat(collect(inside),[on[j] for j in 1:q if (b >> (j-1)) & 1 == 1])
            depth=length(sites)
            if sliced
                (cell.slab ? cell.level<=depth<=cell.level+1 : depth==cell.level) || continue
            end
            push!(vertices,Tuple(vcat([sum((X[i,a] for i in sites);init=zero(QQ)) for a in 1:D],QQ[depth])))
        end
        push!(geometry,vertices)
        push!(sources,(;inside,on,radius_squared=radii2[carrier],
            section=sliced ? (cell.slab ? :closed_slab : :horizontal) : :unsliced,
            level=sliced ? cell.level : length(inside)))
    end
    # A sliced 2-cell can be a hypersimplex. Walk its actual edge incidence to
    # order its vertices, without a floating convex hull or proximity matching.
    counts=DataTypes.cell_counts(G)
    if length(counts)>=3
        offset=counts[1]+counts[2];B=DataTypes.boundary_map(G;dim=2)
        for col in 1:counts[3]
            adjacent=Dict{Tuple,Vector{Tuple}}()
            for row in rowvals(B)[nzrange(B,col)]
                ends=geometry[counts[1]+row]
                length(ends)==2 || error("Rhomboid edge must have two endpoints.")
                a,b=ends;push!(get!(adjacent,a,Tuple[]),b);push!(get!(adjacent,b,Tuple[]),a)
            end
            isempty(adjacent) && continue
            all(v->length(v)==2,values(adjacent)) || error("Rhomboid polygon has no boundary cycle.")
            start=minimum(keys(adjacent));cycle=Tuple[start];previous=nothing;current=start
            while true
                next=first(filter(!=(previous),adjacent[current]))
                next==start && break
                next in cycle && error("Rhomboid boundary has multiple cycles.")
                push!(cycle,next);previous,current=current,next
            end
            length(cycle)==length(adjacent) || error("Incomplete rhomboid boundary cycle.")
            geometry[offset+col]=cycle
        end
    end
    return (;geometry,sources)
end

function _filtered_rhomboid(data,spec,provenance;cache)
    G,axes,orientation,retained=_graded_complex_from_point_cloud_rhomboid(data,spec;retain_geometry=true,cache)
    X=_rhomboid_points(data;allow_coincident=true)
    geometric=(;sites=X,family=:multicover,representation=retained.representation,
        backend=retained.backend,axes,radius_cutoff=get(spec.params,:radius,nothing),
        depth_window=something(get(spec.params,:depth_range,nothing),(0,size(X,1))),
        grade_precision=:exact_algebraic,max_dim=get(spec.params,:max_dim,nothing))
    return filtered_complex(G;geometry=retained.geometry,source_ids=retained.sources,orientation,
        construction="rhomboid",provenance=merge(provenance,(;geometric,ambient_dim=size(X,2)+1,
            source_mapping=retained.representation===:schematic_flags ? :labelled_subset_flags : :inside_on_partitions)))
end

"""
    geometric_data(x::FilteredComplex)

Return the retained indexed sites, construction family, geometric representation,
backend, radius/depth window, axes and grade precision for geometric inspection.
Available for rhomboid/subdivision-Cech and unreduced core constructions created
by `filtered_complex`. Coordinates are exact rational representations of input
values. Treat the returned arrays as read-only. Core grades remain numerical.
"""
function geometric_data(x::FilteredComplex)
    haskey(x.provenance,:geometric) || throw(ArgumentError("This complex has no retained geometric construction data."))
    return x.provenance.geometric
end

function _geometric_parameter(x,parameter)
    m=geometric_data(x)
    parameter isa Union{Tuple,AbstractVector} && length(parameter)==2 ||
        throw(ArgumentError("parameter must be (physical_radius, integer_depth)."))
    r,k=parameter
    r isa Real && !(r isa Bool) && isfinite(r) && r>=0 || throw(ArgumentError("The physical radius must be finite and nonnegative."))
    k isa Real && !(k isa Bool) && isfinite(k) && isinteger(k) && m.depth_window[1]<=k<=m.depth_window[2] ||
        throw(ArgumentError("Depth must be an integer in the closed construction window $(m.depth_window)."))
    m.radius_cutoff===nothing || r<=m.radius_cutoff || throw(ArgumentError("The radius is outside the constructed window."))
    return (r,Int(k))
end

"""
    coverage_at(x::FilteredComplex, point; parameter)

Inspect source geometry at `(physical_radius, integer_depth)`. Return indexed
closed-ball membership and its count, eligible source sites and membership of
the represented region. Multicover counts labels, including repeated sites.
Core-Cech uses eligible balls; core-Delaunay additionally restricts each ball to
its full-cloud Voronoi cell (including ties). Eligibility uses retained vertex
grades. Exact distance comparisons rationalize finite inputs before squaring;
this does not make numerical core construction grades exact geometry.
"""
function coverage_at(x::FilteredComplex,point;parameter)
    r,k=_geometric_parameter(x,parameter);m=geometric_data(x);X=m.sites
    point isa Union{Tuple,AbstractVector} && length(point)==size(X,2) &&
        all(v->v isa Real && !(v isa Bool) && isfinite(v),point) ||
        throw(ArgumentError("point must contain $(size(X,2)) finite coordinates."))
    p=QQ.(point);r2=r isa AlgebraicReal ? r^2 : QQ(r)^2
    d2=[sum((X[i,a]-p[a])^2 for a in axes(X,2)) for i in axes(X,1)]
    covered=findall(<=(r2),d2)
    eligible=if m.family===:multicover
        collect(axes(X,1))
    else
        G=graded_complex(x)
        [only(cell_source(x,i)) for i in 1:DataTypes.cell_counts(G)[1]
         if any(g->g[1]<=r && g[2]>=k,DataTypes.cell_grade_set(G,i))]
    end
    contributing=intersect(covered,eligible)
    m.backend===:core_delaunay && filter!(i->d2[i]==minimum(d2),contributing)
    member=m.family===:multicover ? length(covered)>=k : !isempty(contributing)
    return (;count=length(covered),covered_sites=covered,eligible_sites=eligible,contributing_sites=contributing,member,
        parameter=(r,k),radius_squared=r2,family=m.family)
end

"""
    radius_slice(x::FilteredComplex; depth)

Restrict the actual retained geometric complex to a fixed depth and return its
one-parameter graded complex in physical radius. A multicritical cell receives
its earliest radius among occurrences allowed at this depth. Original cell IDs
and signed boundaries are retained; no re-triangulation, encoding grid snapping,
or homology computation occurs. The represented radius window and any skeleton
truncation are inherited. Use `persistence_diagram` for the resulting barcode.
"""
function radius_slice(x::FilteredComplex;depth)
    _geometric_parameter(x,(0,depth));G=graded_complex(x);counts=DataTypes.cell_counts(G)
    offsets=cumsum(vcat(0,counts));ids=[Int[] for _ in counts]
    T=_geometric_grade_type(G);grades=NTuple{1,T}[]
    # G.grades may be empty after a cutoff. Its element type still owns the scale.
    for d in eachindex(counts),j in 1:counts[d]
        candidates=[g[1] for g in DataTypes.cell_grade_set(G,offsets[d]+j) if g[2]>=depth]
        isempty(candidates) && continue
        push!(ids[d],j);push!(grades,(minimum(candidates),))
    end
    boundaries=[DataTypes.boundary_map(G;dim=d)[ids[d],ids[d+1]] for d in 1:length(counts)-1]
    return GradedComplex([DataTypes.cells(G;dim=d-1)[ids[d]] for d in eachindex(counts)],boundaries,grades)
end

_geometric_grade_type(::Union{GradedComplex{N,T},MultiCriticalGradedComplex{N,T}}) where {N,T}=T
