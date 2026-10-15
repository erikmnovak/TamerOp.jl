# Native degree-Rips component cuts, with a reproducible observed-lifetime rule.
struct DensityClusteringAnalysis
    source::DataIngestion.FilteredComplex
    histories::Vector{ComponentHistory}
    diameters::Vector{Float64}
    counts::Matrix{Int}
    cap::Float64
    npoints::Int
end
struct DensityClusteringResult
    labels::Tuple
    clusters::Tuple
    noise::Tuple
    parameters::NamedTuple
    provenance::NamedTuple
end

function _degree_slice(source,k)
    G=source.complex;counts=DataTypes.cell_counts(G);offsets=cumsum(vcat(0,counts))
    keep=Vector{Int}[];grades=Tuple{Float64}[];global_ids=Int[]
    for d in eachindex(counts)
        ids=Int[]
        for i in 1:counts[d]
            id=offsets[d]+i
            candidates=[g[1] for g in DataTypes.cell_grade_set(G,id) if g[2]>=k]
            isempty(candidates) && continue
            push!(ids,i);push!(global_ids,id);push!(grades,(minimum(candidates),))
        end
        push!(keep,ids)
    end
    cells=[G.cells_by_dim[d][keep[d]] for d in eachindex(counts)]
    boundaries=[G.boundaries[d][keep[d],keep[d+1]] for d in eachindex(G.boundaries)]
    scalar=DataTypes.GradedComplex(cells,boundaries,grades)
    DataIngestion.filtered_complex(scalar;geometry=source.geometry[global_ids],source_ids=source.sources[global_ids],
        orientation=(1,),construction="Degree\u2013Rips at fixed minimum neighbor count",
        provenance=(;source.provenance.ambient_dim,degree_threshold=k,grade_scale=:diameter))
end

"""`clustering_analysis(cloud; max_diameter, max_points=128, max_cells=20000, max_grid_entries=30000)`
retains degree\u2013Rips H0 histories for every integer neighbor threshold, plus exact
event-grid component counts. Geometry uses Float64 and includes distance ties.
The grid is exhaustive within the diameter cap, not a sampled approximation.
Treat the retained analysis as read-only; query results own their labels.
"""
function clustering_analysis(cloud::DataTypes.PointCloud;max_diameter,max_points=128,max_cells=20000,max_grid_entries=30000)
    all(v->v isa Int && v>0,(max_points,max_cells,max_grid_entries)) || throw(ArgumentError("Budgets must be positive Ints."))
    n=size(cloud.coords,1)
    1<=n<=max_points || throw(ArgumentError("Point count exceeds the clustering budget, or the cloud is empty."))
    cap=max_diameter
    cap isa Real && !(cap isa Bool) && isfinite(cap) && cap>=0 && isfinite(Float64(cap)) || throw(ArgumentError("max_diameter must be finite and nonnegative."))
    construction=Options.ConstructionOptions(budget=Options.ConstructionBudget(max_simplices=max_cells))
    x=DataIngestion.filtered_complex(DataTypes.PointCloud(copy(cloud.coords)),
        DataIngestion.RadiusDegreeRipsFiltration(;max_dim=1,radius=cap,construction))
    events=sort!(unique(vcat([0.,Float64(cap)],[g[1] for i in eachindex(x.geometry) for g in DataTypes.cell_grade_set(x.complex,i)])))
    big(length(events))*n<=max_grid_entries || throw(ArgumentError("Degree\u2013Rips event grid exceeds max_grid_entries."))
    histories=ComponentHistory[component_history(_degree_slice(x,k)) for k in 0:n-1]
    counts=[component_count(h;scale=r) for r in events,h in histories]
    DensityClusteringAnalysis(x,histories,events,counts,Float64(cap),n)
end

function _clustering_parameters(a;diameter,neighbors,min_size=1,min_persistence=0)
    diameter isa Real && !(diameter isa Bool) && isfinite(diameter) && 0<=diameter<=a.cap || throw(ArgumentError("diameter must lie in [0,$(a.cap)]."))
    neighbors isa Int && 0<=neighbors<a.npoints || throw(ArgumentError("neighbors must be an integer in 0:$(a.npoints-1)."))
    min_size isa Int && min_size>0 || throw(ArgumentError("min_size must be a positive Int."))
    min_persistence isa Real && !(min_persistence isa Bool) && isfinite(min_persistence) && min_persistence>=0 || throw(ArgumentError("min_persistence must be finite and nonnegative."))
    (;diameter,neighbors,min_size,min_persistence)
end

"""`density_clustering(analysis; diameter, neighbors, min_size=1, min_persistence=0)`
labels current degree\u2013Rips connected components passing both size and observed
H0 elder-bar lifetime thresholds. The lifetime is min(death, construction cap)
minus birth; cap survivors are censored. Vertices outside accepted components
have label 0 (noise); accepted components are ordered by minimum original point
index. No border-point reassignment, automatic optimal cut or statistical
recovery guarantee is asserted. Distance ties are inclusive; elder ties use
source order. `density_clustering(cloud; max_diameter, ...)` also prepares the analysis.
"""
function density_clustering(a::DensityClusteringAnalysis;diameter,neighbors,min_size=1,min_persistence=0)
    p=_clustering_parameters(a;diameter,neighbors,min_size,min_persistence)
    h=a.histories[neighbors+1];parts=components_at(h;scale=diameter)
    records=NamedTuple[]
    for c in parts
        ids=sort!([only(s) for s in c.sources]);i=c.component
        lifetime=(h.deaths[i]===nothing ? QQ(a.cap) : h.deaths[i])-h.births[i]
        length(ids)>=min_size && lifetime>=min_persistence || continue
        push!(records,(;source_ids=Tuple(ids),component=i,birth=h.births[i],death=h.deaths[i],
            observed_lifetime=lifetime,censored=h.deaths[i]===nothing))
    end
    sort!(records;by=c->first(c.source_ids));labels=zeros(Int,a.npoints)
    for (label,c) in enumerate(records);labels[collect(c.source_ids)].=label;end
    DensityClusteringResult(Tuple(labels),Tuple(merge(c,(;label=i)) for (i,c) in enumerate(records)),
        Tuple(findall(iszero,labels)),p,(;construction=:radius_degree_rips,diameter_cap=a.cap,
            neighbor_count=:other_vertices_in_full_radius_graph,orientation=(1,-1),rule=:persistent_component_cut,
            ties=:source_order,noise_label=0,arithmetic=:float64,guarantee=:declared_rule_only))
end
function density_clustering(cloud::DataTypes.PointCloud;max_diameter,diameter,neighbors,
        min_size=1,min_persistence=0,max_points=128,max_cells=20000,max_grid_entries=30000)
    a=clustering_analysis(cloud;max_diameter,max_points,max_cells,max_grid_entries)
    density_clustering(a;diameter,neighbors,min_size,min_persistence)
end
"""Return a copied vector of labels in original point order; zero denotes noise."""
cluster_labels(r::DensityClusteringResult)=collect(r.labels)
"""Accepted cluster memberships and observed elder-bar data in original point IDs."""
cluster_memberships(r::DensityClusteringResult)=r.clusters
"""Cheap retained counts, parameters and construction provenance for a clustering result or analysis."""
clustering_summary(r::DensityClusteringResult)=(;kind=:density_clustering,clusters=length(r.clusters),noise=length(r.noise),r.parameters,r.provenance)
clustering_summary(a::DensityClusteringAnalysis)=(;kind=:clustering_analysis,points=a.npoints,diameter_cap=a.cap,event_count=length(a.diameters),degree_count=length(a.histories))
describe(x::Union{DensityClusteringAnalysis,DensityClusteringResult})=clustering_summary(x)
Base.show(io::IO,x::DensityClusteringAnalysis)=print(io,"DensityClusteringAnalysis(",x.npoints," points, ",length(x.diameters)," events)")
Base.show(io::IO,x::DensityClusteringResult)=print(io,"DensityClusteringResult(",length(x.clusters)," clusters, ",length(x.noise)," noise points)")
"""Observed positive lifetime gaps at a fixed degree. Equal lifetimes stay tied;
cap survivors carry lower bounds. A suggested midpoint is a choice, not a certificate.
"""
function clustering_gaps(a::DensityClusteringAnalysis;neighbors)
    _clustering_parameters(a;diameter=0,neighbors)
    h=a.histories[neighbors+1]
    values=sort!(unique([(h.deaths[i]===nothing ? QQ(a.cap) : h.deaths[i])-h.births[i] for i in eachindex(h.births)]))
    [(;lower=values[i],upper=values[i+1],width=values[i+1]-values[i],threshold=(values[i]+values[i+1])/2,
        includes_censored=any(isnothing,h.deaths)) for i in 1:length(values)-1]
end
