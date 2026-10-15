# HN factors, exact chamber supports and bounded filtered-landscape values.
const _StabilityObject=Union{Stability.HNFiltration,Stability.SkyscraperResult,Stability.SkyscraperChambers,Stability.FilteredLandscapeResult}
mutable struct StabilityInspectionSession <: _AbstractInspectionSession
    object::Union{Nothing,_StabilityObject}
    budget::Stability.HNBudget
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
_stability_views(r)=r isa Stability.SkyscraperChambers ? (:chambers,:filtration) :
    r isa Stability.FilteredLandscapeResult ? (:landscape,:difference) : (:filtration,)
function _stability_rational(x)
    x isa Union{Integer,Rational,AbstractFloat} && !(x isa Bool) && isfinite(x) ||
        throw(ArgumentError("selection coordinates and thresholds must be finite real numbers"))
    CoreModules.QQ(x)
end
function _stability_initial_anchor(r)
    r isa Stability.SkyscraperChambers || return nothing
    isempty(r.cells) && throw(ArgumentError("an explorer requires at least one prepared source cell"))
    c=CartesianIndices(r.grid.shape)[first(r.cells)]
    Tuple((r.grid.axes[j][c[j]]+r.grid.axes[j][c[j]+1])/2 for j in 1:2)
end
function _stability_selection(r;view=first(_stability_views(r)),anchor=_stability_initial_anchor(r),
        threshold=r isa Stability.FilteredLandscapeResult ? first(r.thresholds) : 0,
        baseline=r isa Stability.FilteredLandscapeResult ? first(r.thresholds) : 0,
        layer=r isa Stability.FilteredLandscapeResult ? first(r.layers) : 1,factor=0,vertex=1)
    view in _stability_views(r) || throw(ArgumentError("choose a supported view $(_stability_views(r))"))
    threshold=_stability_rational(threshold);baseline=_stability_rational(baseline)
    factor isa Integer && !(factor isa Bool) && 0<=factor<=typemax(Int) || throw(ArgumentError("factor must be nonnegative and fit Int; zero uses the threshold"))
    vertex isa Integer && !(vertex isa Bool) && 0<vertex<=typemax(Int) || throw(ArgumentError("vertex must be positive and fit Int"))
    layer isa Integer && !(layer isa Bool) && 0<layer<=typemax(Int) || throw(ArgumentError("layer must be positive and fit Int"))
    if r isa Stability.FilteredLandscapeResult
        threshold in r.thresholds && baseline in r.thresholds && layer in r.layers || throw(ArgumentError("choose retained thresholds and a retained landscape layer"))
        factor==0 && anchor===nothing && vertex<=length(r.points) || throw(ArgumentError("landscape selection uses a sample vertex, threshold and layer"))
    else
        layer==1 && baseline==0 || throw(ArgumentError("layer and baseline apply only to filtered landscapes"))
        if r isa Stability.SkyscraperChambers
            anchor isa Union{Tuple,AbstractVector} && length(anchor)==2 || throw(ArgumentError("supply a planar anchor"))
            p=Tuple(_stability_rational.(anchor));xs=r.grid.axes
            all(first(xs[j])<=p[j]<last(xs[j]) for j in 1:2) || throw(ArgumentError("anchor is outside the declared support box"))
            cell=LinearIndices(r.grid.shape)[searchsortedlast(xs[1],p[1]),searchsortedlast(xs[2],p[2])]
            cell in r.cells || throw(ArgumentError("anchor must belong to a prepared source cell"))
            anchor=p;vertex<=length(r.grid.finite.dims) || throw(ArgumentError("vertex is outside the finite grid"))
        else
            anchor===nothing || throw(ArgumentError("this result has a fixed anchor/stability condition"))
            factor<=length(r.slopes) || throw(ArgumentError("factor is outside this filtration"))
            vertex<=length(r.source.dims) || throw(ArgumentError("vertex is outside the source module"))
        end
    end
    (;view,anchor,threshold,baseline,layer=Int(layer),factor=Int(factor),vertex=Int(vertex))
end
function _stability_generated_panel(c,p,r,k)
    dims=k==0 ? zeros(Int,length(r.source.dims)) : r.term_dimensions[k]
    layers=AbstractVisualizationLayer[];cells=NamedTuple[];xs,ys=c.grid.axes
    for (v,I) in enumerate(CartesianIndices(c.grid.shape))
        dims[v]>0 || continue
        x0=max(xs[I[1]],p.anchor[1]);x1=xs[I[1]+1]
        y0=max(ys[I[2]],p.anchor[2]);y1=ys[I[2]+1]
        x0<x1 && y0<y1 || continue
        poly=[(x0,y0),(x1,y0),(x1,y1),(x0,y1)]
        push!(layers,PolygonLayer([NTuple{2,Float64}[Tuple(_stability_float.(q)) for q in poly]],
            _VisualRole(:categorical,dims[v]),_VisualRole(:foreground),0.45,1.))
        push!(layers,TextLayer([string(dims[v])],[Tuple(_stability_float.(((x0+x1)/2,(y0+y1)/2)))],_VisualRole(:foreground),13.))
        push!(cells,(;vertex=v,polygon=poly,dimension=dims[v]))
    end
    push!(layers,PointLayer([Tuple(_stability_float.(p.anchor))],_VisualRole(:selected),1.,11.))
    VisualizationSpec(:skyscraper_generated;title="Retained generated submodule",subtitle="Term $k; numbers are stalk dimensions; cells begin at the exact anchor",
        layers,axes=(;xlabel="Parameter x",ylabel="Parameter y",aspect=:equal,
            xlimits=_pad_limits(_stability_float.((first(xs),last(xs)))),ylimits=_pad_limits(_stability_float.((first(ys),last(ys))))),
        metadata=(;cells,generated_dimensions=copy(dims),anchor=p.anchor,term=k,measure=:lebesgue))
end
function _stability_float(x)
    y=Float64(x);isfinite(y) || throw(ArgumentError("exact value is outside finite drawing coordinates"));y
end
function _stability_factors(r,p)
    ds=r.term_dimensions;previous=zeros(Int,length(r.source.dims));rows=Vector{Int}[]
    for d in ds;push!(rows,d-previous);previous=d;end
    k=p.factor==0 ? something(findlast(>=(p.threshold),r.slopes),0) : p.factor
    k<=length(r.slopes) || throw(ArgumentError("factor is outside the selected anchor filtration"))
    layers=AbstractVisualizationLayer[];targets=NamedTuple[]
    for (i,s) in enumerate(r.slopes)
        x=Float64(i);y=_stability_float(s);role=i==k ? _VisualRole(:selected) : _VisualRole(:categorical,i)
        push!(layers,SegmentLayer([(x-0.35,y,x+0.35,y)],role,1.,3.))
        push!(layers,PointLayer([(x,y)],role,1.,10.))
        push!(targets,(;point=(x,y),segment=(x,y,x,y),selection=(;factor=i),label="Factor $i: slope $s; dimensions $(rows[i])"))
    end
    push!(layers,SegmentLayer([(0.5,_stability_float(p.threshold),max(1.5,length(ds)+0.5),_stability_float(p.threshold))],_VisualRole(:foreground),0.5,1.,:dash))
    panel=VisualizationSpec(:hn_slopes;title="HN factor slopes",subtitle="Threshold retains slope >= $(p.threshold)",layers,
        axes=(;xlabel="Factor in filtration order",ylabel="Degree / mass",aspect=:auto,xticks=(Float64.(1:length(ds)),string.(1:length(ds))),
            xlimits=(0.5,max(1.5,length(ds)+0.5)),ylimits=_pad_limits(extrema(_stability_float.([r.slopes;p.threshold;0])))),
        metadata=(;slopes=copy(r.slopes),factor_dimensions=rows,selected_factor=k,pick_targets=targets))
    panels=VisualizationSpec[panel]
    !isempty(rows) && push!(panels,_presentation_matrix_panel(permutedims(hcat(rows...)),"Factor dimensions",
        ["factor $i" for i in eachindex(rows)],string.(eachindex(r.source.dims)),(12,16);
        metadata=(;dimensions=copy.(rows),selected_vertex=p.vertex,matrix_corner="factor / vertex")))
    if k>0 && r.witnesses!==nothing
        j=r.witnesses.inclusions[k]
        push!(panels,_algebra_matrix_panel(j.comps[p.vertex],"Term $k inclusion at vertex $(p.vertex)",r.source.field,(12,12)))
        qs=r.witnesses.factor_projections
        push!(panels,_algebra_matrix_panel(qs[k].comps[p.vertex],"Projection to factor $k",r.source.field,(12,12)))
    end
    panels,k
end
function _stability_chamber_panel(c,p,at)
    layers=AbstractVisualizationLayer[];targets=NamedTuple[]
    for (i,ch) in enumerate(c.chambers)
        ps=NTuple{2,Float64}[Tuple(_stability_float.(x)) for x in ch.polygon]
        length(unique(ps))==length(ps) || throw(ArgumentError("chamber vertices collide in drawing coordinates"))
        role=i in at.chambers ? _VisualRole(:selected) : _VisualRole(:categorical,i)
        push!(layers,PolygonLayer([ps],role,_VisualRole(:foreground),0.35,1.))
        center=Tuple(sum(x[j] for x in ch.polygon)/length(ch.polygon) for j in 1:2)
        xy=Tuple(_stability_float.(center))
        push!(layers,TextLayer([string(i)],[xy],_VisualRole(:foreground),12.))
        push!(targets,(;point=xy,segment=(xy...,xy...),selection=(;anchor=center,factor=0),label="Chamber $i; exact anchor $center"))
    end
    xs,ys=c.grid.axes
    lines=NTuple{4,Float64}[]
    for x in xs;push!(lines,(_stability_float(x),_stability_float(first(ys)),_stability_float(x),_stability_float(last(ys))));end
    for y in ys;push!(lines,(_stability_float(first(xs)),_stability_float(y),_stability_float(last(xs)),_stability_float(y)));end
    push!(layers,SegmentLayer(lines,_VisualRole(:foreground),0.35,1.,:dash))
    point=Tuple(_stability_float.(p.anchor));push!(layers,PointLayer([point],_VisualRole(:selected),1.,13.))
    VisualizationSpec(:hn_chambers;title="Exact HN chamber supports",subtitle="$(length(c.chambers)) regions in $(length(c.cells)) prepared cells; dashed lines are encoding cuts",
        layers,axes=(;xlabel="Parameter x",ylabel="Parameter y",aspect=:equal,
            xlimits=_pad_limits(_stability_float.((first(xs),last(xs)))),ylimits=_pad_limits(_stability_float.((first(ys),last(ys))))),
        metadata=(;polygons=Stability.skyscraper_chamber_polygons(c),anchor=p.anchor,
            selected_chambers=at.chambers,boundary=at.boundary,pick_targets=targets,
            encoding_axes=map(copy,c.grid.axes),prepared_cells=copy(c.cells)))
end
function _stability_landscape_panel(r,p,budget)
    length(first(r.points))==2 || throw(ArgumentError("landscape pictures require two-dimensional sample points"))
    j=findfirst(==(p.threshold),r.thresholds);k=findfirst(==(p.layer),r.layers)
    lower=copy(r.lower[:,j,k]);upper=copy(r.upper[:,j,k])
    if p.view===:difference
        difference=Stability.filtered_landscape_difference(r;threshold=p.threshold,baseline=p.baseline)
        lower=difference.lower[:,k];upper=difference.upper[:,k]
    end
    xs=sort!(unique(first.(r.points)));ys=sort!(unique(last.(r.points)))
    big(length(xs))*length(ys)<=budget.max_entries || throw(ArgumentError("landscape display grid exceeds the entry budget"))
    values=fill(NaN,length(ys),length(xs));targets=NamedTuple[]
    length(unique(r.points))==length(r.points) || throw(ArgumentError("landscape display points must be unique"))
    quantity=p.view===:difference ? "radius difference" : "radius"
    for (i,q) in enumerate(r.points)
        values[searchsortedfirst(ys,q[2]),searchsortedfirst(xs,q[1])]=_stability_float((lower[i]+upper[i])/2)
        xy=Tuple(_stability_float.(q))
        push!(targets,(;point=xy,segment=(xy...,xy...),selection=(;vertex=i),label="Sample $i at $q: $quantity in [$(lower[i]), $(upper[i])]"))
    end
    xf=_stability_float.(xs);yf=_stability_float.(ys)
    all(diff(xf).>0) && all(diff(yf).>0) || throw(ArgumentError("sample coordinates collide in Float64"))
    title=p.view===:difference ? "Landscape difference: $(p.threshold) minus $(p.baseline)" : "Filtered landscape at threshold $(p.threshold)"
    scale=_stability_float(maximum(r.upper));iszero(scale) && (scale=1.)
    colorrange=p.view===:difference ? (-scale,scale) : (0.,scale)
    layers=AbstractVisualizationLayer[HeatmapLayer(xf,yf,values,p.view===:difference ? :balance : :viridis,1.,"$(uppercasefirst(quantity)) enclosure midpoint")]
    push!(layers,PointLayer([Tuple(_stability_float.(r.points[p.vertex]))],_VisualRole(:selected),1.,9.))
    VisualizationSpec(:hn_landscape;title,subtitle="Layer $(p.layer); enclosure width <= $(maximum(upper-lower;init=zero(CoreModules.QQ)))",
        layers,axes=(;xlabel="Parameter x",ylabel="Parameter y",aspect=:equal,
            xlimits=_pad_limits((first(xf),last(xf))),ylimits=_pad_limits((first(yf),last(yf)))),
        metadata=(;points=copy(r.points),lower,upper,threshold=p.threshold,baseline=p.baseline,layer=p.layer,
            view=p.view,context=r.context,pick_targets=targets,colorrange)),lower,upper
end
function _stability_snapshot(object,p,budget)
    panels=VisualizationSpec[];lines=String[];result=nothing;boundary=false
    if object isa Stability.FilteredLandscapeResult
        panel,lower,upper=_stability_landscape_panel(object,p,budget);push!(panels,panel)
        quantity=p.view===:difference ? "radius difference" : "radius"
        push!(lines,"Sample $(p.vertex), point $(object.points[p.vertex]): $quantity in [$(lower[p.vertex]), $(upper[p.vertex])].",
            "Displayed colors use enclosure midpoints; exact bounds are retained in exported values.",
            "Source scope: $(object.context.scope). Radius error is numerical enclosure width, not statistical uncertainty.")
    else
        if object isa Stability.SkyscraperChambers
            at=Stability.skyscraper_chamber_at(object;anchor=p.anchor,witnesses=true,budget)
            result=at.filtration;boundary=at.boundary
            p.view===:chambers && push!(panels,_stability_chamber_panel(object,p,at))
            push!(lines,"Exact anchor $(p.anchor); chamber closures $(at.chambers); tie boundary: $(at.boundary).",
                "Mass integrates the declared cell density; subdivision preserves this measure.")
        else
            result=object
            push!(lines,object isa Stability.HNFiltration ? "Central charge: $(describe(object.stability))." : "Mass weights: $(object.weights).")
        end
        fs,k=_stability_factors(result,p);append!(panels,fs)
        object isa Stability.SkyscraperChambers && insert!(panels,2,_stability_generated_panel(object,p,result,k))
        push!(lines,"Selected factor: $k; selected finite vertex: $(p.vertex).",
            "Exact factor slopes: $(join(string.(result.slopes), ", ")).",
            "Slopes decrease; equal slopes form one semistable factor, not necessarily an interval.",
            result isa Stability.HNFiltration ? "Higher thresholds retain factors of higher slope." :
                "Higher thresholds retain smaller average support mass.",
            result.witnesses===nothing ? "Algebraic witnesses were not retained." : "Matrices use stored bases; inclusions need not split.")
    end
    push!(panels,_inspection_text_panel("Stability and exact selection",lines))
    VisualizationSpec(:stability;title="HN and skyscraper inspection",panels,
        metadata=(;selection=p,boundary,slopes=result===nothing ? CoreModules.QQ[] : copy(result.slopes),
            term_dimensions=result===nothing ? Vector{Int}[] : copy.(result.term_dimensions),
            panel_columns=2,figure_size=(1250,1100)))
end
available_visuals(::_StabilityObject)=(:stability,)
_visual_request_keywords(::_StabilityObject,::Symbol)=(:view,:anchor,:threshold,:baseline,:layer,:factor,:vertex,:budget)
function _visual_spec(r::_StabilityObject,::Symbol;budget=Stability.HNBudget(),kwargs...)
    _stability_snapshot(r,_stability_selection(r;kwargs...),budget)
end
function stability_explorer(r::_StabilityObject;budget=Stability.HNBudget(),kwargs...)
    r=deepcopy(r)
    p=_stability_selection(r;kwargs...);state=merge(p,(;revision=0))
    snap=_stability_snapshot(r,state,budget)
    StabilityInspectionSession(r,budget,p,state,snap,1,Dict{Int,Any}(),1,String[],false,false,ReentrantLock(),_InspectionRequests())
end
inspection_selection(s::StabilityInspectionSession)=lock(s.lock) do;s.state;end
inspection_snapshot(s::StabilityInspectionSession)=lock(s.lock) do;s.snapshot;end
inspection_summary(s::StabilityInspectionSession)=lock(s.lock) do
    (;kind=:stability_explorer,selection=s.state,revision=s.state.revision,closed=s.closed,updating=s.busy,
      snapshot_builds=s.snapshot_builds,listener_count=length(s.listeners),listener_errors=Tuple(s.listener_errors),requests=_inspection_request_summary(s))
end
describe(s::StabilityInspectionSession)=inspection_summary(s)
Base.show(io::IO,s::StabilityInspectionSession)=print(io,"StabilityInspectionSession(view=",s.state.view,", closed=",s.closed,")")
function _stability_candidate(s;kwargs...)
    _inspection_require_open(s)
    all(k->k in keys(s.initial),keys(kwargs)) || throw(ArgumentError("unsupported stability selection"))
    p=merge(Base.structdiff(s.state,(;revision=s.state.revision)),(;kwargs...))
    merge(_stability_selection(s.object;p...),(;revision=s.state.revision+1))
end
function check_inspection_selection(s::StabilityInspectionSession;throw=false,kwargs...)
    issues=String[]
    lock(s.lock) do
        try
            p=_stability_candidate(s;kwargs...)
            _stability_snapshot(s.object,p,s.budget)
        catch err;err isa InterruptException && rethrow();push!(issues,sprint(showerror,err));end
    end
    valid=isempty(issues);throw && !valid && _throw_invalid_visual(:check_inspection_selection,issues)
    (;kind=:inspection_selection,valid,issues)
end
function select_inspection!(s::StabilityInspectionSession;kwargs...)
    _cancel_inspection_requests!(s)
    lock(s.lock) do
        s.busy && throw(ArgumentError("a stability selection is in progress"))
        p=_stability_candidate(s;kwargs...);snap=_stability_snapshot(s.object,p,s.budget)
        s.state=p;s.snapshot=snap;s.snapshot_builds+=1;s.busy=true
    end
    try;_inspection_notify(s)
    finally;lock(s.lock) do;s.busy=false;end;end
    s
end
reset_inspection!(s::StabilityInspectionSession)=select_inspection!(s;s.initial...)
function close_inspection!(s::StabilityInspectionSession)
    _cancel_inspection_requests!(s)
    lock(s.lock) do
        s.closed && return s
        s.busy && throw(ArgumentError("wait for the current update before closing"))
        s.closed=true;s.object=nothing
    end
    try;_inspection_notify(s)
    finally;lock(s.lock) do;empty!(s.listeners);end;end
    s
end
function check_inspection_session(s::StabilityInspectionSession;throw=false)
    issues=String[]
    lock(s.lock) do
        s.snapshot.metadata.selection==s.state || push!(issues,"snapshot selection mismatch")
        s.closed && (s.object!==nothing || !isempty(s.listeners)) && push!(issues,"closed explorer retains input or listeners")
        append!(issues,check_visual_spec(s.snapshot).issues)
    end
    valid=isempty(issues);throw && !valid && _throw_invalid_visual(:check_inspection_session,issues)
    (;kind=:inspection_session,valid,issues)
end
function _visual_spec(s::StabilityInspectionSession,::Symbol;kwargs...)
    isempty(kwargs) || throw(ArgumentError("change the selection with select_inspection!"))
    VisualizationSpec(:linked_inspector;title="HN and skyscraper inspection",panels=s.snapshot.panels,
        interaction=(;hover=true,clicks=true,labels=true,widgets=(:stability_selection,),notebook=:linked_inspector,
            mode=:live_julia,requires_live_julia=true,offline_widgets=false),metadata=(;session=s,inspector_kind=:stability))
end
_visual_request_cost(::StabilityInspectionSession,::Symbol)=(;work=:bounded_exact_anchor_query,hover=:read_only,requires_live_julia=true,timing=:not_measured)
