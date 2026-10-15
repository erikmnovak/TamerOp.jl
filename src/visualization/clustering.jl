# Linked native density clustering, sharing the verified source and H0 owners.
mutable struct ClusteringInspectionSession <: _AbstractInspectionSession
    object::Union{Nothing,OrdinaryPersistence.DensityClusteringAnalysis}
    initial::NamedTuple
    state::NamedTuple
    snapshot::VisualizationSpec
    snapshot_builds::Int
    listeners::Dict{Int,Any}
    next_listener::Int
    listener_errors::Vector{String}
    closed::Bool
    busy::Bool
    lock::ReentrantLock
    requests::_InspectionRequests
end

function _clustering_snapshot(a,p)
    result=OrdinaryPersistence.density_clustering(a;p.diameter,p.neighbors,p.min_size,p.min_persistence)
    points=[Tuple(Float64.(first(a.source.geometry[i]))) for i in 1:a.npoints]
    all(q->length(q)==2,points) || throw(ArgumentError("The clustering explorer displays planar clouds."))
    layers=AbstractVisualizationLayer[];targets=NamedTuple[]
    n=a.npoints
    for i in n+1:length(a.source.geometry)
        DataIngestion._source_active(a.source,i,(p.diameter,p.neighbors)) || continue
        u,v=a.source.sources[i]
        push!(layers,SegmentLayer([(points[u]...,points[v]...)],_VisualRole(:muted),0.35,1.))
    end
    for label in unique(result.labels)
        ids=findall(==(label),result.labels)
        role=label==0 ? _VisualRole(:muted) : _VisualRole(:categorical,label)
        push!(layers,PointLayer(points[ids],role,label==0 ? 0.4 : 1.,10.))
    end
    positions=unique(points);point_labels=String[]
    for point in positions
        ids=findall(==(point),points);label=result.labels[first(ids)]
        push!(point_labels,join(ids,", "))
        push!(targets,(;point,selection=(;),label="Point IDs $(last(point_labels)): "*(label==0 ? "noise" : "cluster $label")))
    end
    n<=50 && push!(layers,TextLayer(point_labels,positions,_VisualRole(:foreground),11.))
    x=first.(points);y=last.(points);pad=max(maximum(x)-minimum(x),maximum(y)-minimum(y),1.)*0.1
    source=VisualizationSpec(:clustering_source;title="$(length(result.clusters)) clusters; $(length(result.noise)) noise points",
        subtitle="Diameter $(p.diameter), minimum neighbors $(p.neighbors). Faint points have label 0.",layers,
        axes=(;xlabel="x",ylabel="y",aspect=:data,xlimits=(minimum(x)-pad,maximum(x)+pad),ylimits=(minimum(y)-pad,maximum(y)+pad)),
        metadata=(;pick_targets=targets,labels=result.labels,points))
    grid_targets=NamedTuple[]
    for j in axes(a.counts,2),i in axes(a.counts,1)
        r=a.diameters[i];k=j-1;count=a.counts[i,j]
        push!(grid_targets,(;point=(r,Float64(k)),selection=(;diameter=r,neighbors=k),label="$count components at diameter $r, minimum neighbors $k"))
    end
    locations=[t.point for t in grid_targets]
    values=Float64[a.counts[searchsortedfirst(a.diameters,t.point[1]),Int(t.point[2])+1] for t in grid_targets]
    grid_layers=AbstractVisualizationLayer[]
    if length(a.diameters)>1
        # Counts are constant on [event_i,event_{i+1}); degree rows are discrete.
        push!(grid_layers,HeatmapLayer(copy(a.diameters),collect(-0.5:1:n-0.5),
            Float64.(permutedims(a.counts[1:end-1,:])),:viridis,0.35,"",false))
    end
    push!(grid_layers,PointLayer(locations,values,0.8,9.))
    length(values)<=100 && push!(grid_layers,TextLayer(string.(Int.(values)),locations,_VisualRole(:foreground),10.))
    push!(grid_layers,PointLayer([(Float64(p.diameter),Float64(p.neighbors))],_VisualRole(:selected),1.,15.))
    grid=VisualizationSpec(:clustering_grid;title="Component-count event grid",
        subtitle="Click an event dot. Bands show constant counts up to the next diameter; rows are integer k. Counts precede filtering.",layers=grid_layers,
        axes=(;xlabel="Rips diameter",ylabel="Minimum neighbors k",xlimits=(-0.04max(a.cap,1.),1.04max(a.cap,0.1)),ylimits=(-0.5,n-0.5),aspect=:auto),
        metadata=(;pick_targets=grid_targets,diameters=copy(a.diameters),counts=copy(a.counts),
            colorrange=(0.,Float64(n)),point_colorrange=(0.,Float64(n)),point_colorbar_label="Components",
            sampling=:all_events,interpolation=:none,diameter_cells=:left_closed_right_open,degree_rows=:integer))
    h=a.histories[p.neighbors+1];payload=_interval_payload(OrdinaryPersistence.persistence_diagram(h);dim=0)
    records=Tuple(isfinite(r.death) ? r : merge(r,(;death=a.cap,right_closed=true,right_status=:censored,right_clipped=true,singleton=r.birth==a.cap)) for r in payload.records)
    bar,_=_interval_panels(records;window=(0.,max(a.cap,0.1)),max_intervals=a.npoints,barcode_title="H0 at k = $(p.neighbors)")
    bar_layers=copy(bar.layers);lo,hi=bar.axes.ylimits
    push!(bar_layers,SegmentLayer([(Float64(p.diameter),lo,Float64(p.diameter),hi)],_VisualRole(:selected),1.,1.,:dash))
    bar=_composition_copy(bar;layers=bar_layers,subtitle="Colors identify interval groups; ? marks survival at the construction cap.",metadata=merge(bar.metadata,(;pick_targets=NamedTuple[])))
    gaps=OrdinaryPersistence.clustering_gaps(a;neighbors=p.neighbors)
    approximate(x)=string(round(Float64(x);sigdigits=5))
    lines=["Rule: persistent component cut. Minimum size $(p.min_size); minimum observed lifetime $(p.min_persistence).",
        "Degree counts other vertices in the full radius graph. Distance ties are included; elder ties follow source order.",
        "Cap survivors provide lifetime lower bounds. Labels have no identity across different selections. No automatic optimal clustering is asserted.",
        "Labels in original point order: $(result.labels)",
        "Positive lifetime gaps (rounded here; full values are retained): "*
            (isempty(gaps) ? "none" : join(["$i: $(approximate(g.lower)) to $(approximate(g.upper)), width $(approximate(g.width))" for (i,g) in enumerate(gaps)],"; "))]
    VisualizationSpec(:density_clustering;title="Density clustering",panels=[source,grid,bar,_inspection_text_panel("Clustering decision",lines)],
        metadata=(;selection=p,result,labels=result.labels,clusters=result.clusters,noise=result.noise,gaps=Tuple(gaps),
            panel_columns=2,figure_size=(1150,1000),rule=:persistent_component_cut))
end

"""`clustering_explorer(cloud; max_diameter, diameter=0, neighbors=0, min_size=1, min_persistence=0, ...)`
prepares the native degree\u2013Rips event grid and links source memberships,
component counts and fixed-degree barcodes. Select a cut explicitly; results
are reproduced with `density_clustering` and retrieved by `clustering_result`.
Live exploration is planar and budgeted; offline exports contain saved selections.
"""
function clustering_explorer(cloud::DataTypes.PointCloud;max_diameter,diameter=0,neighbors=0,min_size=1,min_persistence=0,
        max_points=128,max_cells=20000,max_grid_entries=30000)
    size(cloud.coords,2)==2 || throw(ArgumentError("The clustering explorer requires planar points."))
    a=OrdinaryPersistence.clustering_analysis(cloud;max_diameter,max_points,max_cells,max_grid_entries)
    p=OrdinaryPersistence._clustering_parameters(a;diameter,neighbors,min_size,min_persistence)
    initial=p;state=merge(initial,(;revision=0))
    snapshot=_clustering_snapshot(a,state)
    ClusteringInspectionSession(a,initial,state,snapshot,1,Dict{Int,Any}(),1,String[],false,false,ReentrantLock(),_InspectionRequests())
end
inspection_selection(s::ClusteringInspectionSession)=lock(s.lock) do;s.state;end
inspection_snapshot(s::ClusteringInspectionSession)=lock(s.lock) do;s.snapshot;end
"""Return the committed immutable clustering result, also available after closure."""
clustering_result(s::ClusteringInspectionSession)=lock(s.lock) do;s.snapshot.metadata.result;end
function inspection_summary(s::ClusteringInspectionSession)
    lock(s.lock) do
        (;kind=:clustering_explorer,selection=s.state,revision=s.state.revision,closed=s.closed,updating=s.busy,
            snapshot_builds=s.snapshot_builds,clusters=length(s.snapshot.metadata.clusters),noise=length(s.snapshot.metadata.noise),
            listener_count=length(s.listeners),listener_errors=Tuple(s.listener_errors),requests=_inspection_request_summary(s))
    end
end
describe(s::ClusteringInspectionSession)=inspection_summary(s)
Base.show(io::IO,s::ClusteringInspectionSession)=print(io,"ClusteringInspectionSession(",length(s.snapshot.metadata.clusters)," clusters, closed=",s.closed,")")
function _clustering_candidate(s;kwargs...)
    _inspection_require_open(s)
    all(k->k in keys(s.initial),keys(kwargs)) || throw(ArgumentError("Select diameter, neighbors, min_size or min_persistence."))
    p=merge(Base.structdiff(s.state,(;revision=s.state.revision)),(;kwargs...))
    OrdinaryPersistence._clustering_parameters(s.object;p...)
    merge(p,(;revision=s.state.revision+1))
end
function check_inspection_selection(s::ClusteringInspectionSession;throw=false,kwargs...)
    issues=String[]
    lock(s.lock) do
        try;_clustering_candidate(s;kwargs...)
        catch err;err isa InterruptException && rethrow();push!(issues,sprint(showerror,err));end
    end
    valid=isempty(issues);throw && !valid && _throw_invalid_visual(:check_inspection_selection,issues)
    (;kind=:inspection_selection,valid,issues)
end
function select_inspection!(s::ClusteringInspectionSession;kwargs...)
    _cancel_inspection_requests!(s)
    lock(s.lock) do
        s.busy && throw(ArgumentError("A clustering update is in progress."))
        p=_clustering_candidate(s;kwargs...);snapshot=_clustering_snapshot(s.object,p)
        s.state=p;s.snapshot=snapshot;s.snapshot_builds+=1;s.busy=true
    end
    try;_inspection_notify(s)
    finally;lock(s.lock) do;s.busy=false;end;end
    s
end
reset_inspection!(s::ClusteringInspectionSession)=select_inspection!(s;s.initial...)
function close_inspection!(s::ClusteringInspectionSession)
    _cancel_inspection_requests!(s)
    lock(s.lock) do
        s.closed && return s
        s.busy && throw(ArgumentError("Wait for the current clustering update before closing."))
        s.closed=true;s.object=nothing
    end
    try;_inspection_notify(s)
    finally;lock(s.lock) do;empty!(s.listeners);end;end
    s
end
function check_inspection_session(s::ClusteringInspectionSession;throw=false)
    issues=String[]
    lock(s.lock) do
        s.snapshot.metadata.selection==s.state || push!(issues,"Snapshot selection mismatch.")
        s.closed && (s.object!==nothing || !isempty(s.listeners)) && push!(issues,"Closed explorer retains input or listeners.")
        append!(issues,check_visual_spec(s.snapshot).issues)
    end
    valid=isempty(issues);throw && !valid && _throw_invalid_visual(:check_inspection_session,issues)
    (;kind=:inspection_session,valid,issues)
end
function _visual_spec(s::ClusteringInspectionSession,::Symbol;kwargs...)
    isempty(kwargs) || throw(ArgumentError("Use select_inspection! to change the clustering cut."))
    VisualizationSpec(:linked_inspector;title="Density clustering",panels=s.snapshot.panels,
        interaction=(;hover=true,clicks=true,labels=true,widgets=(:clustering_cut,:clustering_rule),notebook=:linked_inspector,
            mode=:live_julia,requires_live_julia=true,offline_widgets=false),metadata=(;session=s,inspector_kind=:clustering))
end
_visual_request_cost(::ClusteringInspectionSession,::Symbol)=(;work=:retained_degree_histories,hover=:read_only,requires_live_julia=true,timing=:not_measured)
