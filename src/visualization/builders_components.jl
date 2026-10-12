available_visuals(::OrdinaryPersistence.ComponentHistory)=(:components,)
_visual_request_keywords(::OrdinaryPersistence.ComponentHistory,::Symbol)=
    (:scale,:component,:event,:view_dims,:max_cells,:max_branches,:max_labels,:show_ephemeral)
_visual_request_cost(::OrdinaryPersistence.ComponentHistory,::Symbol)=
    (;work=:retained_merge_forest_cut,geometry=:actual_source_cells,memberships=:requested_partition,
        budget=:display_only,homology=:not_reduced_again,timing=:not_measured)

function _component_view_options(h;scale=nothing,component=nothing,event=nothing,view_dims=nothing,
        max_cells=10000,max_branches=100,max_labels=30,show_ephemeral=false)
    n=length(h.births)
    event===nothing || (event isa Int && n<event<=length(h.times)) || throw(ArgumentError("event must be a merge-node index, or nothing."))
    component===nothing || (component isa Int && 1<=component<=n) || throw(ArgumentError("component must be a birth-vertex index, or nothing."))
    t=scale===nothing ? (isempty(h.births) ? 0 : OrdinaryPersistence._component_external(h,minimum(h.births))) : scale
    OrdinaryPersistence.component_parameter(h,t)
    if event!==nothing
        t=OrdinaryPersistence._component_external(h,h.times[event])
        component===nothing && (component=h.elders[last(h.children[event])])
    end
    max_cells isa Int && max_cells>0 && max_branches isa Int && max_branches>0 &&
        max_labels isa Int && max_labels>=0 || throw(ArgumentError("Display budgets must be positive Ints; max_labels may be zero."))
    show_ephemeral isa Bool || throw(ArgumentError("show_ephemeral must be Bool."))
    D=h.source.provenance.ambient_dim
    vd=view_dims===nothing ? (D<=3 ? Tuple(1:D) : throw(ArgumentError("Select view_dims for source geometry above dimension three."))) : view_dims
    vd isa Tuple && length(vd) in (2,3) && all(d->d isa Int && 1<=d<=D,vd) && allunique(vd) ||
        throw(ArgumentError("view_dims must select two or three distinct source coordinates."))
    return (;scale=t,component,event,view_dims=vd,max_cells,max_branches,max_labels,show_ephemeral)
end
function _append_visual_request_issues!(issues::Vector{String},h::OrdinaryPersistence.ComponentHistory,::Symbol;kwargs...)
    try;_component_view_options(h;kwargs...)
    catch err;err isa InterruptException && rethrow();push!(issues,sprint(showerror,err));end
    issues
end

function _component_bars(h,o)
    rows=OrdinaryPersistence.component_intervals(h;include_ephemeral=o.show_ephemeral)
    o.component===nothing || sort!(rows;by=r->(r.component!=o.component,r.component))
    total=length(rows);resize!(rows,min(total,o.max_branches));sort!(rows;by=r->r.component)
    progress=Float64.(h.times);all(isfinite,progress) || throw(ArgumentError("Component event coordinates cannot be drawn as finite values."))
    lo,hi=isempty(progress) ? (0.0,1.0) : extrema(progress)
    t=Float64(OrdinaryPersistence._component_progress(h,o.scale))
    isfinite(t) || throw(ArgumentError("The selected scale cannot be drawn as a finite value."))
    span=hi>lo ? hi-lo : 1.0;upper=max(hi,t)+0.16span;lower=min(lo,t)-0.08span
    exact_scale=OrdinaryPersistence._component_progress(h,o.scale)
    collision=length(unique(progress))<length(unique(h.times)) ||
        any(i->progress[i]==t && h.times[i]!=exact_scale,eachindex(progress))
    layers=AbstractVisualizationLayer[];targets=NamedTuple[]
    for (j,r) in enumerate(rows)
        b=Float64(h.births[r.component]);d=r.death===nothing ? upper : Float64(h.deaths[r.component]);y=Float64(j)
        role=_VisualRole(:categorical,r.component)
        push!(layers,SegmentLayer([(b,y,d,y)],role,1.0,r.component==o.component ? 5.0 : 2.5))
        push!(layers,TextLayer(["[",r.death===nothing ? ">" : ")"],[(b,y),(d,y)],role,13.0))
        r.ephemeral && push!(layers,TextLayer(["empty"],[(b,y+0.22)],_VisualRole(:muted),10.0))
        push!(targets,(;point=((b+d)/2,y),selection=(;component=r.component),label="Component $(r.component); birth $(r.birth), death $(r.death)"))
    end
    push!(layers,SegmentLayer([(t,0.4,t,max(1.6,length(rows)+0.6))],_VisualRole(:foreground),0.65,1.0,:dash))
    ylabel="birth vertex";xlabel=h.slice.kind==:original && h.slice.direction[1]<0 ? "oriented progress = -parameter" : h.slice.kind==:affine ? "slice parameter t" : "parameter"
    chart=VisualizationSpec(:component_bars;title="H0 intervals",subtitle="[ includes birth; ) excludes death; > continues. $(total-length(rows)) bars omitted.",layers,
        axes=(;xlabel,ylabel,xlimits=(min(lower,t-0.03span),max(upper+0.05span,t+0.03span)),ylimits=(0.4,max(1.6,length(rows)+0.6)),
            yticks=(Float64.(1:length(rows)),[string(r.component) for r in rows]),aspect=:auto),
        metadata=(;pick_targets=targets,records=rows,omitted=total-length(rows),display_coordinates=:oriented_progress,
            exact_records=true,essential_continuation=true,rounded_event_collision=collision))
    chart,(lower,upper,span)
end

function _component_tree(h,o,bounds)
    n=length(h.births);order=Int[];stack=reverse(copy(h.roots))
    while !isempty(stack)
        i=pop!(stack)
        if i<=n;push!(order,i)
        else;push!(stack,last(h.children[i]));push!(stack,first(h.children[i]));end
    end
    x=zeros(Float64,length(h.times))
    for (j,i) in enumerate(order);x[i]=j;end
    for i in n+1:length(x);a,b=h.children[i];x[i]=(x[a]+x[b])/2;end
    # Keep at most max_branches displayed nodes. Selected records have priority;
    # then nearby ancestors preserve local context without unbounded
    # ancestor expansion on a comb-shaped merge forest.
    priority=Int[]
    o.event===nothing || push!(priority,o.event)
    o.component===nothing || push!(priority,o.component)
    queue=unique(vcat(priority,order));kept=Set{Int}()
    for node in priority
        length(kept)>=o.max_branches && break
        push!(kept,node)
    end
    cursor=1
    while cursor<=length(queue) && length(kept)<o.max_branches
        node=queue[cursor];cursor+=1
        push!(kept,node)
        parent=h.parent[node]
        parent==0 || parent in kept || insert!(queue,cursor,parent)
    end
    leaves=[i for i in kept if i<=n]
    layers=AbstractVisualizationLayer[];targets=NamedTuple[];missing=0
    for i in sort!(collect(kept))
        h.parent[i]!=0 && !(h.parent[i] in kept) && (missing+=1)
        y=Float64(h.times[i]);role=_VisualRole(:categorical,h.elders[i])
        if i>n
            for child in h.children[i]
                if child in kept
                    push!(layers,PolylineLayer([[(x[child],Float64(h.times[child])),(x[child],y),(x[i],y)]],role,0.85,1.7,false))
                else;missing+=1;end
            end
        elseif length(leaves)<=o.max_labels
            push!(layers,TextLayer([string(i)],[(x[i],y-0.055bounds[3])],_VisualRole(:foreground),11.0))
        end
        selected=i==o.event || i==o.component
        point=(x[i],y)
        push!(layers,PointLayer([point],selected ? _VisualRole(:selected) : role,1.0,selected ? 14.0 : 7.0))
        selection=i<=n ? (;component=i) : (;event=i,component=h.elders[last(h.children[i])])
        push!(targets,(;point,selection,label=i<=n ? "Birth vertex $i" : "Merge at $(OrdinaryPersistence._component_external(h,h.times[i])); edge $(h.merge_edges[i])"))
    end
    for i in h.roots
        i in kept || continue
        push!(layers,SegmentLayer([(x[i],Float64(h.times[i]),x[i],bounds[2])],_VisualRole(:categorical,h.elders[i]),0.8,1.7,:dash))
    end
    t=Float64(OrdinaryPersistence._component_progress(h,o.scale))
    push!(layers,SegmentLayer([(0.3,t,max(1.7,n+0.7),t)],_VisualRole(:foreground),0.65,1.0,:dash))
    VisualizationSpec(:component_tree;title="Merge forest",subtitle="Equal heights are simultaneous; tie order is a display choice. $(length(h.times)-length(kept)) nodes and $missing incident branches omitted.",layers,
        axes=(;xlabel="source-vertex leaf order",ylabel="oriented filtration progress",xlimits=(0.3,max(1.7,n+0.7)),
            ylimits=(bounds[1]-0.05bounds[3],bounds[2]),aspect=:auto),
        metadata=(;pick_targets=targets,displayed_nodes=sort!(collect(kept)),omitted_nodes=length(h.times)-length(kept),omitted_branches=missing))
end

function _component_source(h,o,partition,selected)
    p=OrdinaryPersistence.component_parameter(h,o.scale);n=length(h.births)
    if isempty(h.source.geometry)
        return VisualizationSpec(:component_geometry;title="Source components",layers=AbstractVisualizationLayer[PointLayer(NTuple{2,Float64}[],_VisualRole(:foreground),1.0,8.0)],
            axes=(;xlabel="source x",ylabel="source y",xlimits=(0.0,1.0),ylimits=(0.0,1.0),aspect=:auto),
            metadata=(;pick_targets=NamedTuple[],displayed_indices=Int[],omitted=0,minimal_axes=true))
    end
    context=_visual_spec(h.source,:filtered_complex;parameter=p,dimensions=(0,1),view_dims=o.view_dims,max_cells=o.max_cells,labels=false)
    chart=first(context.panels);layers=AbstractVisualizationLayer[];targets=NamedTuple[]
    membership=zeros(Int,n)
    for c in partition,i in c.vertices;membership[i]=c.component;end
    active=copy(context.metadata.active);eligible=[i for i in 1:n+length(h.edge_times) if active[i]]
    priority=selected===nothing ? Int[] : selected.vertices
    o.event===nothing || (priority=vcat([n+h.merge_edges[o.event]],priority))
    ids=unique(vcat(priority,eligible));resize!(ids,min(length(ids),o.max_cells))
    for i in ids
        d=i<=n ? 0 : 1
        root=i<=n ? membership[i] : h.endpoints[i-n][1]==0 ? 0 : membership[first(h.endpoints[i-n])]
        points=[Tuple(Float64(p[k]) for k in o.view_dims) for p in DataIngestion.cell_geometry(h.source,i)]
        color=root==0 ? _VisualRole(:muted) : _VisualRole(:categorical,root)
        chosen=selected!==nothing && root==selected.descendant
        event_edge=o.event!==nothing && i==n+h.merge_edges[o.event]
        if d==0
            chosen && push!(layers,length(o.view_dims)==2 ? PointLayer(points,_VisualRole(:selected),1.0,17.0) : Point3Layer(points,_VisualRole(:selected),1.0,17.0))
            push!(layers,length(o.view_dims)==2 ? PointLayer(points,color,1.0,10.0) : Point3Layer(points,color,1.0,10.0))
            length(o.view_dims)==2 && push!(targets,(;point=only(points),selection=(;component=root),label="Source vertex $i; component $root"))
        else
            width=event_edge ? 6.0 : chosen ? 3.5 : 1.7
            push!(layers,length(o.view_dims)==2 ? PolylineLayer([points],event_edge ? _VisualRole(:selected) : color,0.8,width,false) :
                Segment3Layer([Tuple((points[j]...,points[j+1]...)) for j in 1:length(points)-1],event_edge ? _VisualRole(:selected) : color,0.8,width))
        end
    end
    labels=[i for i in ids if i<=n][1:min(count(<=(n),ids),o.max_labels)]
    if length(o.view_dims)==2 && !isempty(labels)
        push!(layers,TextLayer(string.(labels),[Tuple(Float64(only(DataIngestion.cell_geometry(h.source,i))[k]) for k in o.view_dims) for i in labels],_VisualRole(:foreground),11.0))
    end
    isempty(layers) && push!(layers,length(o.view_dims)==2 ? PointLayer(NTuple{2,Float64}[],_VisualRole(:foreground),1.0,8.0) : Point3Layer(NTuple{3,Float64}[],_VisualRole(:foreground),1.0,8.0))
    VisualizationSpec(:component_geometry;title="Actual source components",subtitle="Parameter $p; $(length(partition)) components; $(length(eligible)-length(ids)) active cells omitted.",layers,axes=chart.axes,
        metadata=(;pick_targets=targets,displayed_indices=ids,omitted=length(eligible)-length(ids),minimal_axes=true,
            projection=chart.metadata.projection,source_view_dimension=length(o.view_dims),warnings=chart.metadata.warnings))
end

function _visual_spec(h::OrdinaryPersistence.ComponentHistory,::Symbol;kwargs...)
    o=_component_view_options(h;kwargs...)
    partition=OrdinaryPersistence.components_at(h;scale=o.scale)
    selected=o.component===nothing ? nothing : OrdinaryPersistence.component_members(h;scale=o.scale,component=o.component)
    bars,bounds=_component_bars(h,o);tree=_component_tree(h,o,bounds)
    source=_component_source(h,o,partition,selected)
    lines=["$(length(partition)) actual connected components at parameter $(OrdinaryPersistence.component_parameter(h,o.scale)).",
        "Membership uses oriented source edge boundaries; geometry and color do not define connectivity.",
        "Birth labels are source-vertex indices. Older births survive; equal births use the smaller index.",
        "All tied events are included at their exact scale. Zero-length bars have no nonempty lifetime."]
    bars.metadata.rounded_event_collision && push!(lines,"Distinct exact events or query scales share drawing coordinates. Use the exact readout to distinguish them.")
    if selected!==nothing
        push!(lines,"Selected birth vertex $(selected.component): $(selected.status).")
        selected.status==:merged && push!(lines,"Its bar has died. Highlighting follows its current descendant component $(selected.descendant).")
        push!(lines,"$(length(selected.vertices)) source vertices: "*join(selected.vertices[1:min(end,o.max_labels)],", ")*
            (length(selected.vertices)>o.max_labels ? " ... (complete IDs remain in metadata)" : ""))
    end
    if o.event!==nothing
        e=only(filter(e->e.node==o.event,OrdinaryPersistence.component_merges(h)))
        push!(lines,"Merge at $(e.scale): component $(e.absorbed) joins $(e.survivor) across source edge $(e.source_edge).")
    end
    h.slice.kind==:affine && push!(lines,"This forest belongs to p(t) = $(h.slice.base) + t * $(h.slice.direction), not to every slice of the multiparameter object.")
    any(h.source.provenance.periodic) && push!(lines,"Periodic source faces are identified. Wrap cells use retained unwrapped geometry.")
    h.provenance.diagram_correspondence==:endpoint_multisets && push!(lines,"Original/computed H0 interval multisets agree. Equal bars do not identify arbitrary reduction-basis representatives.")
    append!(lines,get(source.metadata,:warnings,String[]))
    readout=_inspection_text_panel("Components, events and identities",lines)
    VisualizationSpec(:components;title="Components and merge history",panels=[bars,source,tree,readout],
        metadata=(;scale=o.scale,component=o.component,event=o.event,partition,selection=selected,
            parameter=OrdinaryPersistence.component_parameter(h,o.scale),summary=OrdinaryPersistence.component_history_summary(h),
            panel_columns=2,figure_size=(1400,1000)))
end
