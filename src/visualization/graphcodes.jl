# Barcode-row geometry and linked coefficient selection; algebra belongs to Graphcodes.
mutable struct GraphcodeInspectionSession <: _AbstractInspectionSession
    object::Union{Nothing,Graphcodes.Graphcode}
    compressed::Union{Nothing,Graphcodes.CompressedGraphcode}
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

_graphcode_active(row,t)=findall(s->s[1]<=t<=s[2],row)
_graphcode_visual_index(i,n,label)=i isa Integer && !(i isa Bool) && 1<=i<=n ? Int(i) : throw(ArgumentError("$label must be an integer in 1:$n"))

function _graphcode_selection(g;row=1,source_interval=0,target_interval=0,column=1,component=0,compressed=false)
    r=_graphcode_visual_index(row,length(g.y),"row");c=_graphcode_visual_index(column,length(g.x),"column")
    for (k,v,n) in (("source interval",source_interval,length(g.rows[r])),
            ("target interval",target_interval,r<length(g.y) ? length(g.rows[r+1]) : 0),
            ("component",component,length(Graphcodes.graphcode_components(g))))
        v isa Integer && !(v isa Bool) && 0<=v<=n || throw(ArgumentError("$k must be an integer in 0:$n; zero clears selection"))
    end
    compressed isa Bool || throw(ArgumentError("compressed must be true or false"))
    (;row=r,source_interval=Int(source_interval),target_interval=Int(target_interval),column=c,component=Int(component),compressed)
end
function _graphcode_snapshot(g,c,p)
    m,n=length(g.x),length(g.y);xx=Float64.(g.x)
    F=Graphcodes.field(g);field_label=_inspection_field_label(F)
    all(isfinite,xx) && (m==1 || all(diff(xx).>0) || all(diff(xx).<0)) ||
        throw(ArgumentError("graphcode coordinates cannot be distinguished in Float64 drawing coordinates"))
    parts=Graphcodes.graphcode_components(g);membership=Dict(v=>k for (k,vs) in enumerate(parts) for v in vs)
    offsets=cumsum(vcat(0,[max(length(row),1)+1 for row in g.rows]))
    location(j,i)=Float64(offsets[j]+i)
    row_centers=[offsets[j]+(max(length(g.rows[j]),1)+1)/2 for j in 1:n]
    chart_height=max(600,24last(offsets))
    nodes=p.compressed ? [(first(v)...,) for v in c.members] : [(j,i) for j in 1:n for i in eachindex(g.rows[j])]
    spans=p.compressed ? c.members : [[v] for v in nodes]
    edgelist=p.compressed ? [(;row=last(c.members[u])[1],source=last(c.members[u])[2],target=first(c.members[v])[2],
        from=first(c.members[u]),to=first(c.members[v]),coefficient=c.coefficients[k]) for (k,(u,v)) in enumerate(c.edges)] :
        [merge(e,(;from=(e.row,e.source),to=(e.row+1,e.target))) for e in Graphcodes.graphcode_edges(g)]
    pos=Dict{Tuple{Int,Int},Tuple{Float64,Float64}}();layers=AbstractVisualizationLayer[];targets=NamedTuple[]
    for (j,i) in nodes
        a,b=g.rows[j][i];right=xx[min(b+1,m)];pos[(j,i)]=((xx[a]+right)/2,location(j,i))
    end
    for e in edgelist
        a,b=pos[e.from],pos[e.to];selected=p.row==e.row && p.source_interval==e.source && p.target_interval==e.target
        visible=p.component==0 || membership[e.from]==p.component
        role=selected ? _VisualRole(:selected) : _VisualRole(:categorical,membership[e.from])
        push!(layers,SegmentLayer([(a...,b...)],role,visible ? 0.65 : 0.12,selected ? 3. : 1.))
        # Arrowheads mark incidence even when an edge crosses another barcode.
        scale=(max(maximum(xx)-minimum(xx),1.)/1000,last(offsets)/Float64(chart_height))
        dx,dy=(b[1]-a[1])/scale[1],(b[2]-a[2])/scale[2];d=hypot(dx,dy)
        if d>0
            ux,uy=dx/d,dy/d
            heads=[(b[1],b[2],b[1]+(-8ux+side*4uy)*scale[1],b[2]+(-8uy-side*4ux)*scale[2]) for side in (-1,1)]
            push!(layers,SegmentLayer(heads,role,visible ? 0.85 : 0.12,selected ? 2. : 1.))
        end
        midpoint=((a[1]+b[1])/2,(a[2]+b[2])/2)
        push!(targets,(;point=midpoint,segment=(a...,b...),selection=(;row=e.row,source_interval=e.source,target_interval=e.target),
            label="Map $(e.row) -> $(e.row+1): interval $(e.source) -> $(e.target), coefficient $(_matrix_coefficient_text(e.coefficient)) in $field_label"))
    end
    for (k,(j,i)) in enumerate(nodes)
        a,b=g.rows[j][i];yl=location(j,i);right=xx[min(b+1,m)];part=membership[(j,i)]
        selected=(p.row,p.source_interval) in spans[k] || (p.row+1,p.target_interval) in spans[k]
        alpha=p.component==0 || p.component==part ? 1. : 0.15
        role=selected ? _VisualRole(:selected) : _VisualRole(:categorical,part)
        push!(layers,SegmentLayer([(xx[a],yl,right,yl)],role,alpha,selected ? 5. : 3.))
        push!(layers,PointLayer([(xx[a],yl)],role,alpha,8.))
        # Censored boundary is '?' rather than an inferred finite death.
        push!(layers,TextLayer([b==m ? "?" : "o"],[(right,yl)],role,10.))
        label="$(j).$(i)"*(length(spans[k])>1 ? " [rows $j:$(last(spans[k])[1])]" : "")
        push!(layers,TextLayer([label],[(pos[(j,i)][1],yl+0.25)],_VisualRole(:foreground),10.))
        push!(targets,(;point=pos[(j,i)],segment=(xx[a],yl,right,yl),selection=(;row=j,source_interval=i,target_interval=0),
            label="Row $j, interval $i: indices $a:$b, component $part; y= $(_matrix_coefficient_text(g.y[j]))"))
    end
    sort!(targets;by=t->t.selection.target_interval==0 ? 0 : 1)
    push!(layers,SegmentLayer([(xx[p.column],0.,xx[p.column],Float64(last(offsets)))],_VisualRole(:selected),0.5,1.,:dash))
    span=max(maximum(xx)-minimum(xx),1.);rowlabels=["$j:  $(_matrix_coefficient_text(g.y[j]))" for j in 1:n]
    chart=VisualizationSpec(:graphcode_rows;title=p.compressed ? "Compressed graphcode" : "Barcode rows and connecting maps",
        subtitle="$field_label; maps run toward increasing row numbers. Colors mark graph components, not necessarily indecomposables.",layers,
        axes=(;xlabel="First parameter",ylabel="Row: second parameter (schematic spacing)",aspect=:auto,
            xlimits=(minimum(xx)-0.08span,maximum(xx)+0.1span),ylimits=(0.,Float64(last(offsets))),yticks=(Float64.(row_centers),rowlabels)),
        metadata=(;pick_targets=targets,rows=Graphcodes.graphcode_rows(g),edges=Graphcodes.graphcode_edges(g),
            original_axes=Graphcodes.graphcode_axes(g),compression=p.compressed ? :superfluous_vertices : :none,
            compressed_members=deepcopy(c.members),compressed_edges=Graphcodes.graphcode_edges(c),
            continuations=Graphcodes.graphcode_continuations(c),field=field_label,basis=:noncanonical))
    r=p.row;K=CoreModules.coeff_type(F)
    C=r<n ? g.maps[r] : zeros(K,0,length(g.rows[r]))
    matrix=_presentation_matrix_panel(C,"Connecting coefficients: row $r to $(r+1)",
        ["$(r+1).$i" for i in axes(C,1)],["$r.$i" for i in axes(C,2)],(12,12);
        subtitle=r<n ? "Rows are target intervals; columns are source intervals. Entries are in $field_label." : "The final row has no outgoing map.",
        metadata=(;cell_roles=[i==p.target_interval && j==p.source_interval ? :selected : :foreground for i in 1:min(12,size(C,1)),j in 1:min(12,size(C,2))]))
    source=_graphcode_active(g.rows[r],p.column);target=r<n ? _graphcode_active(g.rows[r+1],p.column) : Int[]
    A=C[target,source];value=p.source_interval>0 && p.target_interval>0 ? C[p.target_interval,p.source_interval] : nothing
    relation=p.source_interval>0 && p.target_interval>0 ? (g.rows[r+1][p.target_interval][1]<=g.rows[r][p.source_interval][1]<=g.rows[r+1][p.target_interval][2]<=g.rows[r][p.source_interval][2]) : nothing
    lines=["Selection: row $r (second parameter $(_matrix_coefficient_text(g.y[r]))); column $(p.column) (first parameter $(_matrix_coefficient_text(g.x[p.column]))).",
        "Source interval $(p.source_interval); target interval $(p.target_interval). Zero clears an interval selection.",
        "Selected coefficient: $(value===nothing ? "none" : _matrix_coefficient_text(value)); interval Hom condition: $(relation===nothing ? "select both intervals" : relation).",
        "Active source intervals: $(Tuple(source)). Active target intervals: $(Tuple(target)).",
        "$(length(parts)) graph components in these bases. A connected component may still split after a change of basis.",
        "Compose coefficients in $field_label: paths can cancel, so a drawn path alone does not certify a nonzero composite.",
        "$(sum(length,g.rows)) row intervals; $(length(c.members)) compressed vertices. Compression retains every intermediate scalar map and all original row/member IDs.",
        "The dashed line selects a finite-grid column. Endpoint o is open; ? means censored at the final grid coordinate.",
        "Scope: $(g.scope===:supplied_rows ? "supplied barcode rows" : g.scope===:finite_grid ? "complete finite grid" : "declared grid restriction"). Row spacing is schematic; exact parameter labels are retained. No extension beyond this grid is asserted."]
    localmap=_presentation_matrix_panel(A,"Actual stalk map at the selected column",string.(target),string.(source),(12,12);
        subtitle="Restriction of the row coefficient matrix to intervals active at this column.")
    VisualizationSpec(:graphcode;title="Graphcode",panels=[chart,matrix,localmap,_inspection_text_panel("Selection and basis",lines)],
        metadata=(;selection=p,field=field_label,coefficients=copy(C),stalk_map=copy(A),coefficient=value,hom_admissible=relation,
            active_source=Tuple(source),active_target=Tuple(target),components=Tuple(Tuple(v) for v in parts),
            panel_positions=[(1:1,1:2),(2:2,1:1),(2:2,2:2),(3:3,1:2)],
            panel_row_weights=[Float64(chart_height),220.,300.],figure_size=(1200,chart_height+520),chart_height,basis=:noncanonical))
end

function _graphcode_display_budget(g;max_intervals=200,max_edges=1000,max_rows=64)
    for (v,label) in ((max_intervals,"max_intervals"),(max_edges,"max_edges"),(max_rows,"max_rows"))
        v isa Integer && !(v isa Bool) && v>=0 || throw(ArgumentError("$label must be a nonnegative integer"))
    end
    sum(length,g.rows)<=max_intervals || throw(ArgumentError("graphcode display interval budget exceeded"))
    sum(A->count(!iszero,A),g.maps;init=0)<=max_edges || throw(ArgumentError("graphcode display edge budget exceeded"))
    length(g.y)<=max_rows || throw(ArgumentError("graphcode display row budget exceeded"))
    nothing
end

available_visuals(::Union{Graphcodes.Graphcode,Graphcodes.CompressedGraphcode})=(:graphcode,)
_visual_request_keywords(::Union{Graphcodes.Graphcode,Graphcodes.CompressedGraphcode},::Symbol)=(:row,:source_interval,:target_interval,:column,:component,:compressed,:max_intervals,:max_edges,:max_rows)
function _visual_spec(g::Graphcodes.Graphcode,::Symbol;max_intervals=200,max_edges=1000,max_rows=64,kwargs...)
    _graphcode_display_budget(g;max_intervals,max_edges,max_rows)
    p=_graphcode_selection(g;kwargs...);_graphcode_snapshot(g,Graphcodes.compress_graphcode(g),p)
end
function _visual_spec(c::Graphcodes.CompressedGraphcode,::Symbol;compressed=true,max_intervals=200,max_edges=1000,max_rows=64,kwargs...)
    g=Graphcodes.expand_graphcode(c);_graphcode_display_budget(g;max_intervals,max_edges,max_rows)
    _graphcode_snapshot(g,c,_graphcode_selection(g;compressed,kwargs...))
end
"""Link exact barcode-row coefficients, graph components and selected stalk maps.
Pointer selection has exact row/interval/column controls as a keyboard alternative.
The live explorer is bounded; saved HTML retains explicitly prepared selections.
"""
function graphcode_explorer(g::Graphcodes.Graphcode;max_intervals=200,max_edges=1000,max_rows=64,kwargs...)
    _graphcode_display_budget(g;max_intervals,max_edges,max_rows)
    Graphcodes.check_graphcode(g;throw=true)
    # The session owns its graph data; external mutations cannot invalidate a queued view.
    owned=copy(g);c=Graphcodes.compress_graphcode(owned);p=_graphcode_selection(owned;kwargs...)
    state=merge(p,(;revision=0));snap=_graphcode_snapshot(owned,c,state)
    GraphcodeInspectionSession(owned,c,p,state,snap,1,Dict{Int,Any}(),1,String[],false,false,ReentrantLock(),_InspectionRequests())
end
inspection_selection(s::GraphcodeInspectionSession)=lock(s.lock) do;s.state;end
inspection_snapshot(s::GraphcodeInspectionSession)=lock(s.lock) do;s.snapshot;end
function inspection_summary(s::GraphcodeInspectionSession)
    lock(s.lock) do
        (;kind=:graphcode_explorer,selection=s.state,revision=s.state.revision,closed=s.closed,updating=s.busy,
            snapshot_builds=s.snapshot_builds,listener_count=length(s.listeners),listener_errors=Tuple(s.listener_errors),requests=_inspection_request_summary(s))
    end
end
describe(s::GraphcodeInspectionSession)=inspection_summary(s)
Base.show(io::IO,s::GraphcodeInspectionSession)=print(io,"GraphcodeInspectionSession(row=",s.state.row,", closed=",s.closed,")")
function _graphcode_candidate(s;kwargs...)
    _inspection_require_open(s)
    all(k->k in keys(s.initial),keys(kwargs)) || throw(ArgumentError("Select row, source_interval, target_interval, column, component or compressed."))
    p=merge(Base.structdiff(s.state,(;revision=s.state.revision)),(;kwargs...))
    merge(_graphcode_selection(s.object;p...),(;revision=s.state.revision+1))
end
function check_inspection_selection(s::GraphcodeInspectionSession;throw=false,kwargs...)
    issues=String[]
    lock(s.lock) do
        try;_graphcode_candidate(s;kwargs...)
        catch err;err isa InterruptException && rethrow();push!(issues,sprint(showerror,err));end
    end
    valid=isempty(issues);throw && !valid && _throw_invalid_visual(:check_inspection_selection,issues)
    (;kind=:inspection_selection,valid,issues)
end
function select_inspection!(s::GraphcodeInspectionSession;kwargs...)
    _cancel_inspection_requests!(s)
    lock(s.lock) do
        s.busy && throw(ArgumentError("A graphcode update is in progress."))
        p=_graphcode_candidate(s;kwargs...);snap=_graphcode_snapshot(s.object,s.compressed,p)
        s.state=p;s.snapshot=snap;s.snapshot_builds+=1;s.busy=true
    end
    try;_inspection_notify(s)
    finally;lock(s.lock) do;s.busy=false;end;end
    s
end
reset_inspection!(s::GraphcodeInspectionSession)=select_inspection!(s;s.initial...)
function close_inspection!(s::GraphcodeInspectionSession)
    _cancel_inspection_requests!(s)
    lock(s.lock) do
        s.closed && return s
        s.busy && throw(ArgumentError("Wait for the current graphcode update before closing."))
        s.closed=true;s.object=nothing;s.compressed=nothing
    end
    try;_inspection_notify(s)
    finally;lock(s.lock) do;empty!(s.listeners);end;end
    s
end
function check_inspection_session(s::GraphcodeInspectionSession;throw=false)
    issues=String[]
    lock(s.lock) do
        s.snapshot.metadata.selection==s.state || push!(issues,"Snapshot selection mismatch.")
        s.closed && (s.object!==nothing || s.compressed!==nothing || !isempty(s.listeners)) && push!(issues,"Closed explorer retains input or listeners.")
        append!(issues,check_visual_spec(s.snapshot).issues)
    end
    valid=isempty(issues);throw && !valid && _throw_invalid_visual(:check_inspection_session,issues)
    (;kind=:inspection_session,valid,issues)
end
function _visual_spec(s::GraphcodeInspectionSession,::Symbol;kwargs...)
    isempty(kwargs) || throw(ArgumentError("Use select_inspection! to change the graphcode selection."))
    VisualizationSpec(:linked_inspector;title="Graphcode",panels=s.snapshot.panels,
        interaction=(;hover=true,clicks=true,labels=true,widgets=(:graphcode_selection,),notebook=:linked_inspector,
            mode=:live_julia,requires_live_julia=true,offline_widgets=false),metadata=(;session=s,inspector_kind=:graphcode))
end
_visual_request_cost(::GraphcodeInspectionSession,::Symbol)=(;work=:retained_barcode_maps,hover=:read_only,requires_live_julia=true,timing=:not_measured)
