# A temporal selection reuses the finite module, barcode rows and rank landscape.
mutable struct TemporalInspectionSession <: _AbstractInspectionSession
    object::Union{Nothing,TemporalPersistence.TemporalPersistenceResult}
    prepared::Union{Nothing,NamedTuple}
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
function _temporal_selection(r,l;occurrence=1,scale_index=1,layer=first(l.layers),interval=0)
    index(x,n,name)=_inspection_integer(x,name;lower=1)<=n ? Int(x) : throw(ArgumentError("$name must be in 1:$n."))
    i=index(occurrence,length(r.times),:occurrence);j=index(scale_index,length(r.scales),:scale_index)
    layer isa Integer && !(layer isa Bool) && layer in l.layers || throw(ArgumentError("Choose a retained landscape layer $(l.layers)."))
    interval isa Integer && !(interval isa Bool) && 0<=interval<=length(r.barcodes[j].supports) || throw(ArgumentError("Interval must be zero or an interval ID at this scale."))
    (;occurrence=i,scale_index=j,layer=Int(layer),interval=Int(interval))
end
function _temporal_time_panel(r,p)
    n=length(r.times);layers=AbstractVisualizationLayer[];targets=NamedTuple[]
    kinds=get(r.provenance,:occurrence_kinds,fill(:source,n))
    ticks=unique([1;collect(1:max(1,cld(n,8)):n);n])
    union_sequence=get(r.provenance,:construction,nothing)===:rips_union_zigzag
    for i in 1:n
        point=(Float64(i),0.);role=i==p.occurrence ? :selected : kinds[i]===:union ? :target : :source
        push!(layers,PointLayer([point],_VisualRole(role),1.,i==p.occurrence ? 15. : clamp(500/n,5.,15.)))
        if n<=12 || i==p.occurrence
            push!(layers,TextLayer([kinds[i]===:union ? "union" : "source"],[(point[1],0.3)],_VisualRole(:foreground),11.))
        end
        push!(targets,(;point,segment=(point...,point...),selection=(;occurrence=i),label="Occurrence $i; time label $(r.times[i]); $(kinds[i])"))
        if i<n
            push!(layers,SegmentLayer([(i+0.13,0.,i+0.87,0.)],_VisualRole(:foreground),0.8,1.5))
            push!(layers,TextLayer([r.directions[i]===:forward ? "\u2192" : "\u2190"],[(i+0.5,0.12)],_VisualRole(:foreground),clamp(450/n,8.,22.)))
        end
    end
    VisualizationSpec(:temporal_arrows;title="Source states and actual map directions",subtitle=union_sequence ? "Odd occurrences: sources. Even occurrences: union bridges.\nRepeated time labels remain distinct." : "Positions are occurrence indices; repeated time labels remain distinct.",layers,
        axes=(;aspect=:auto,xlabel="Occurrence",ylabel="",xlimits=(0.5,n+0.5),ylimits=(-0.4,0.6),xticks=(Float64.(ticks),string.(ticks)),yticks=(Float64[],String[])),
        metadata=(;directions=copy(r.directions),times=copy(r.times),occurrence_kinds=kinds,pick_targets=targets))
end
function _temporal_barcode_panel(r,p,max_intervals)
    bars=r.barcodes[p.scale_index];records=ZigzagPersistence.zigzag_intervals(bars);n=length(r.times)
    layers=AbstractVisualizationLayer[];targets=NamedTuple[]
    ids=collect(1:min(length(records),max_intervals))
    p.interval>0 && !(p.interval in ids) && (ids[end]=p.interval;sort!(ids))
    for (row,id) in enumerate(ids)
        b=records[id]
        a,z=b.first,b.last
        left=b.left_kind===:backward_birth ? a-1 : a
        right=b.right_kind===:forward_death ? z+1 : z
        color=_VisualRole(id==p.interval ? :selected : :source);y=Float64(row)
        push!(layers,SegmentLayer([(Float64(left),y,Float64(right),y)],color,0.9,id==p.interval ? 4. : 2.))
        for (x,closed,censored) in ((left,b.left_closed,a==1),(right,b.right_closed,z==n))
            push!(layers,TextLayer([censored ? "|" : closed ? "\u25cf" : "\u25cb"],[(Float64(x),y)],color,16.))
        end
        push!(targets,(;point=((left+right)/2,y),segment=(Float64(left),y,Float64(right),y),
            selection=(;interval=id,occurrence=clamp(p.occurrence,a,z)),label="Interval $id: occurrences $a:$z; $(b.left_kind) / $(b.right_kind)"))
    end
    isempty(records) && push!(layers,TextLayer(["No classes at this scale"],[(max(1,n/2),1.)],_VisualRole(:foreground),13.))
    push!(layers,SegmentLayer([(Float64(p.occurrence),0.5,Float64(p.occurrence),max(1.,length(ids))+0.5)],_VisualRole(:selected),0.7,1.))
    VisualizationSpec(:temporal_barcode;title="H$(r.degree) zigzag at scale $(r.scales[p.scale_index])",
        subtitle="$(length(ids)) of $(length(records)) intervals. Filled/open endpoints follow the arrows; end marks are observed boundaries.",layers,
        axes=(;aspect=:auto,xlabel="Occurrence (arrow-dependent endpoints)",ylabel="Interval",xlimits=(0.5,n+0.5),ylimits=(0.5,max(1,length(ids))+0.5)),
        metadata=(;intervals=[describe(b) for b in records],supports=copy(bars.supports),displayed_intervals=ids,omitted_intervals=length(records)-length(ids),pick_targets=targets))
end
function _temporal_landscape_panel(r,l,p)
    n,m=length(r.times),length(r.scales);k=findfirst(==(p.layer),l.layers)
    values=Matrix(transpose(l.values[:,:,k]));flags=l.censored[:,:,k]
    color_max=Float64(max(1,maximum(l.values)))
    layers=AbstractVisualizationLayer[HeatmapLayer(collect(0.5:1:n+0.5),collect(0.5:1:m+0.5),Float64.(values),:viridis,1.,"Integer radius")]
    # These annotation colors contrast with the fixed viridis data colors,
    # independently of the surrounding page's light or dark theme.
    threshold=0.4color_max
    contrast(i,j)=l.values[i,j,k]<=threshold ? :white : :black
    for color in (:white,:black)
        points=[(Float64(i),Float64(j)) for i in 1:n,j in 1:m if flags[i,j] && contrast(i,j)===color]
        isempty(points) || push!(layers,TextLayer(fill("+",length(points)),points,color,12.))
    end
    push!(layers,RectLayer([(p.occurrence-0.48,p.scale_index-0.48,p.occurrence+0.48,p.scale_index+0.48)],_VisualRole(:selected),contrast(p.occurrence,p.scale_index),0.16,2.))
    targets=[(;point=(Float64(i),Float64(j)),segment=(Float64(i),Float64(j),Float64(i),Float64(j)),
        selection=(;occurrence=i,scale_index=j),label="Layer $(p.layer), occurrence $i, scale index $j: $(flags[i,j] ? "at least " : "")$(l.values[i,j,k])") for j in 1:m for i in 1:n]
    VisualizationSpec(:spatiotemporal_landscape;title="Spatiotemporal landscape, layer $(p.layer)",
        subtitle=l.boundary===:censored ? "+ marks a lower bound at the observed boundary." : "Exact for the declared zero extension outside this finite grid.",layers,
        axes=(;aspect=:auto,xlabel="Occurrence index",ylabel="Scale index",xlimits=(0.5,n+0.5),ylimits=(0.5,m+0.5)),
        metadata=(;values=copy(l.values[:,:,k]),censored=copy(flags),boundary=l.boundary,metric=:max_norm_on_integer_indices,
            colorrange=(0.,color_max),pick_targets=targets))
end
function _temporal_section_panel(r,l,p)
    k=findfirst(==(p.layer),l.layers);n=length(r.times)
    points=[(Float64(i),Float64(l.values[i,p.scale_index,k])) for i in 1:n]
    layers=AbstractVisualizationLayer[PointLayer(points,_VisualRole(:source),1.,8.),
        PointLayer([points[p.occurrence]],_VisualRole(:selected),1.,12.)]
    n>1 && pushfirst!(layers,PolylineLayer([points],_VisualRole(:source),1.,2.,false))
    lower_bounds=[(point[1],point[2]+0.08) for (i,point) in enumerate(points) if l.censored[i,p.scale_index,k]]
    isempty(lower_bounds) || push!(layers,TextLayer(fill("+",length(lower_bounds)),lower_bounds,_VisualRole(:foreground),12.))
    targets=[(;point,segment=(point...,point...),selection=(;occurrence=i),label="Occurrence $i; time label $(r.times[i]); radius $(point[2])") for (i,point) in enumerate(points)]
    VisualizationSpec(:temporal_landscape_section;title="At scale index $(p.scale_index)",subtitle=l.boundary===:censored ? "Discrete samples; + marks a lower bound at the observed boundary." : "Discrete samples with the declared zero extension; the metric uses index steps.",layers,
        axes=(;aspect=:auto,xlabel="Occurrence index",ylabel="Landscape radius",xlimits=(0.5,n+0.5),ylimits=(-0.25,max(1.,maximum(last,points))+0.25)),metadata=(;pick_targets=targets,values=copy(l.values[:,p.scale_index,k])))
end
Base.@constprop :none Base.@noinline Base.@nospecializeinfer function _temporal_snapshot(Base.@nospecialize(r),Base.@nospecialize(prepared),Base.@nospecialize(p))
    l=prepared.landscape;x=r.sources[p.occurrence];scale=r.scales[p.scale_index]
    source=first(_visual_spec(x,:filtered_complex;parameter=(scale,),view_dims=prepared.view_dims,max_cells=prepared.max_cells).panels)
    cycle=nothing
    alive=p.interval>0 && r.barcodes[p.scale_index].supports[p.interval][1]<=p.occurrence<=r.barcodes[p.scale_index].supports[p.interval][2]
    if alive && r.homology!==nothing
        cycle=TemporalPersistence.temporal_cycle(r;scale_index=p.scale_index,interval=p.interval,occurrence=p.occurrence)
        overlay=first(_visual_spec(cycle,:source_chain;view_dims=prepared.view_dims).panels)
        source=VisualizationSpec(source.kind;title="Source and selected cycle",subtitle=source.subtitle,
            layers=vcat(source.layers,overlay.layers),axes=source.axes,legend=source.legend,
            metadata=merge(source.metadata,(;selected_chain=DataIngestion.source_chain_data(cycle))))
    end
    source=VisualizationSpec(source.kind;title=source.title,subtitle=source.subtitle,layers=source.layers,axes=source.axes,
        legend=source.legend,metadata=merge(source.metadata,(;pick_targets=NamedTuple[])))
    k=findfirst(==(p.layer),l.layers);value=l.values[p.occurrence,p.scale_index,k];censored=l.censored[p.occurrence,p.scale_index,k]
    lines=["Occurrence $(p.occurrence) of $(length(r.times)); time label $(r.times[p.occurrence]).",
        "Spatial scale $(scale), index $(p.scale_index); H$(r.degree) over $(_inspection_field_label(r.field)).",
        "Landscape layer $(p.layer): $(censored ? "at least " : "")$value index steps.",
        "Each square uses the canonical limit-to-colimit rank of the entire time/scale diagram.",
        "Boundary convention: $(l.boundary). The scale grid is sampled; no unobserved scales are inferred.",
        p.interval==0 ? "Choose a barcode interval to inspect its retained source representative." : !alive ?
            "Interval $(p.interval) is absent at this occurrence." : r.homology===nothing ?
            "Interval $(p.interval) is present. Construct with representatives=true for source cycles." :
            "Interval $(p.interval) is present; its displayed source chain is a checked cycle in a chosen compatible interval basis."]
    if p.interval>0
        b=ZigzagPersistence.zigzag_intervals(r.barcodes[p.scale_index])[p.interval]
        push!(lines,"Exact endpoint labels: $(b.left_closed ? "[" : "(")$(b.left), $(b.right)$(b.right_closed ? "]" : ")"); $(b.left_kind), $(b.right_kind).")
    end
    cycle===nothing || push!(lines,"Source coefficients (first $(min(12,length(cycle.coefficients))) of $(length(cycle.coefficients))): [$(join(cycle.coefficients[1:min(12,end)],", "))].")
    panels=[source,_temporal_time_panel(r,p),_temporal_barcode_panel(r,p,prepared.max_intervals),_temporal_landscape_panel(r,l,p),_temporal_section_panel(r,l,p),_inspection_text_panel("Temporal selection",lines)]
    VisualizationSpec(:temporal_persistence;title="Follow topology through time and scale",panels,
        metadata=(;selection=p,field=_inspection_field_label(r.field),degree=r.degree,times=copy(r.times),scales=copy(r.scales),directions=copy(r.directions),
            landscape=(;values=copy(l.values),censored=copy(l.censored),layers=copy(l.layers),boundary=l.boundary,queries=l.queries,metric=:max_norm_on_integer_indices),
            active=alive,source_cycle=cycle===nothing ? nothing : DataIngestion.source_chain_data(cycle),provenance=r.provenance,
            panel_positions=[(1:1,1:1),(1:1,2:2),(2:2,1:2),(3:3,1:1),(3:3,2:2),(4:4,1:2)],panel_row_weights=[400.,280.,400.,300.],figure_size=(1400,1380)))
end

function _temporal_prepare(r;layers=1:3,boundary=:censored,view_dims=nothing,max_cells=1000,max_intervals=200,rank_budget=Invariants.GeneralizedRankBudget())
    TemporalPersistence.check_temporal(r;throw=true)
    max_cells=_inspection_integer(max_cells,:max_cells;lower=1)
    max_intervals=_inspection_integer(max_intervals,:max_intervals;lower=1)
    l=TemporalPersistence.spatiotemporal_landscape(r;layers,boundary,budget=rank_budget)
    (;landscape=l,view_dims,max_cells,max_intervals)
end
available_visuals(::TemporalPersistence.TemporalPersistenceResult)=(:temporal_persistence,)
_visual_request_keywords(::TemporalPersistence.TemporalPersistenceResult,::Symbol)=(:occurrence,:scale_index,:layer,:interval,:layers,:boundary,:view_dims,:max_cells,:max_intervals,:rank_budget)
function _visual_spec(r::TemporalPersistence.TemporalPersistenceResult,::Symbol;layers=1:3,boundary=:censored,view_dims=nothing,max_cells=1000,max_intervals=200,rank_budget=Invariants.GeneralizedRankBudget(),kwargs...)
    prepared=_temporal_prepare(r;layers,boundary,view_dims,max_cells,max_intervals,rank_budget)
    _temporal_snapshot(r,prepared,_temporal_selection(r,prepared.landscape;kwargs...))
end
"""
    temporal_explorer(result; layers=1:3, boundary=:censored, ...)

Link actual source states, directed time maps, the selected zigzag barcode,
a spatiotemporal landscape and its temporal section. Prepare the bounded rank
queries once. Time, scale, layer and interval selection reuse the result; they
do not rerun ingestion or landscape ranks. Retain source cycles with
`temporal_persistence(...; representatives=true)`. Higher-dimensional geometry
uses the source viewer's explicit `view_dims` contract. Saved sessions contain
selected snapshots; an arbitrary new temporal query requires Julia.
"""
function temporal_explorer(r::TemporalPersistence.TemporalPersistenceResult;layers=1:3,boundary=:censored,view_dims=nothing,max_cells=1000,max_intervals=200,rank_budget=Invariants.GeneralizedRankBudget(),kwargs...)
    prepared=_temporal_prepare(r;layers,boundary,view_dims,max_cells,max_intervals,rank_budget)
    p=_temporal_selection(r,prepared.landscape;kwargs...);state=merge(p,(;revision=0))
    TemporalInspectionSession(r,prepared,p,state,_temporal_snapshot(r,prepared,state),1,Dict{Int,Any}(),1,String[],false,false,ReentrantLock(),_InspectionRequests())
end
inspection_selection(s::TemporalInspectionSession)=lock(s.lock) do;s.state;end
inspection_snapshot(s::TemporalInspectionSession)=lock(s.lock) do;s.snapshot;end
function inspection_summary(s::TemporalInspectionSession)
    lock(s.lock) do
        (;kind=:temporal_explorer,selection=s.state,revision=s.state.revision,closed=s.closed,updating=s.busy,
          snapshot_builds=s.snapshot_builds,landscape_preparations=1,landscape_queries=s.snapshot.metadata.landscape.queries,
          listener_count=length(s.listeners),listener_errors=Tuple(s.listener_errors),requests=_inspection_request_summary(s))
    end
end
describe(s::TemporalInspectionSession)=inspection_summary(s)
Base.show(io::IO,s::TemporalInspectionSession)=print(io,"TemporalInspectionSession(occurrence=",s.state.occurrence,", closed=",s.closed,")")
function _temporal_candidate(s;kwargs...)
    _inspection_require_open(s)
    all(k->k in keys(s.initial),keys(kwargs)) || throw(ArgumentError("Select occurrence, scale_index, layer or interval."))
    p=merge(Base.structdiff(s.state,(;revision=s.state.revision)),(;kwargs...))
    haskey(kwargs,:scale_index) && p.scale_index!=s.state.scale_index && !haskey(kwargs,:interval) && (p=merge(p,(;interval=0)))
    merge(_temporal_selection(s.object,s.prepared.landscape;p...),(;revision=s.state.revision+1))
end
function check_inspection_selection(s::TemporalInspectionSession;throw=false,kwargs...)
    issues=String[]
    lock(s.lock) do
        try;_temporal_candidate(s;kwargs...)
        catch err;err isa InterruptException && rethrow();push!(issues,sprint(showerror,err));end
    end
    valid=isempty(issues);throw && !valid && _throw_invalid_visual(:check_inspection_selection,issues)
    (;kind=:inspection_selection,valid,issues)
end
function select_inspection!(s::TemporalInspectionSession;kwargs...)
    _cancel_inspection_requests!(s)
    lock(s.lock) do
        s.busy && throw(ArgumentError("A temporal selection update is in progress."))
        p=_temporal_candidate(s;kwargs...);snap=_temporal_snapshot(s.object,s.prepared,p)
        s.state=p;s.snapshot=snap;s.snapshot_builds+=1;s.busy=true
    end
    try;_inspection_notify(s)
    finally;lock(s.lock) do;s.busy=false;end;end
    s
end
reset_inspection!(s::TemporalInspectionSession)=select_inspection!(s;s.initial...)
function close_inspection!(s::TemporalInspectionSession)
    _cancel_inspection_requests!(s)
    lock(s.lock) do
        s.closed && return s
        s.busy && throw(ArgumentError("Wait for the current temporal selection before closing."))
        s.closed=true;s.object=nothing;s.prepared=nothing
    end
    try;_inspection_notify(s)
    finally;lock(s.lock) do;empty!(s.listeners);end;end
    s
end
function check_inspection_session(s::TemporalInspectionSession;throw=false)
    issues=String[]
    lock(s.lock) do
        s.snapshot.metadata.selection==s.state || push!(issues,"Snapshot selection mismatch.")
        s.closed && (s.object!==nothing || s.prepared!==nothing || !isempty(s.listeners)) && push!(issues,"Closed temporal explorer retains input or listeners.")
        append!(issues,check_visual_spec(s.snapshot).issues)
    end
    valid=isempty(issues);throw && !valid && _throw_invalid_visual(:check_inspection_session,issues)
    (;kind=:inspection_session,valid,issues)
end
function _visual_spec(s::TemporalInspectionSession,::Symbol;kwargs...)
    isempty(kwargs) || throw(ArgumentError("Use select_inspection! to change a temporal selection."))
    VisualizationSpec(:linked_inspector;title="Follow topology through time and scale",panels=s.snapshot.panels,
        interaction=(;hover=true,clicks=true,labels=true,widgets=(:temporal_selection,),notebook=:linked_inspector,
            mode=:live_julia,requires_live_julia=true,offline_widgets=false),metadata=(;session=s,inspector_kind=:temporal))
end
_visual_request_cost(::TemporalInspectionSession,::Symbol)=(;work=:retained_temporal_selection,hover=:read_only,requires_live_julia=true,timing=:not_measured)
