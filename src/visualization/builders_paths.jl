# Shared coordinates for path histories, selected diagrams and source states.
available_visuals(::PersistencePath)=(:persistence_path,)
const _PATH_KEYS=(:frame,:track,:representative,:scale,:view_dims,:max_tracks,:max_cells)
_visual_request_keywords(::PersistencePath,::Symbol)=_PATH_KEYS
_visual_request_cost(::PersistencePath,::Symbol)=(;work=:retained_history,homology=:already_computed,
    representatives=:retained_keyframes_only,timing=:not_measured)
function _path_selection(p;frame=1,track=nothing,representative=false,scale=nothing,view_dims=nothing,max_tracks=100,max_cells=5000)
    frame isa Int && 1<=frame<=length(p.frames) || throw(ArgumentError("frame must select an existing path state."))
    track===0 && (track=nothing)
    track===nothing || (track isa Int && any(r->r.id==track,Iterators.flatten(f.records for f in p.frames))) ||
        throw(ArgumentError("track must select an existing trajectory, or 0 to clear."))
    representative isa Bool || throw(ArgumentError("representative must be Bool."))
    representative && track===nothing && throw(ArgumentError("Select a track before requesting its representative."))
    all(x->x isa Int && x>0,(max_tracks,max_cells)) || throw(ArgumentError("Display budgets must be positive Ints."))
    if scale===nothing
        values=[r.birth for f in p.frames for r in f.records]
        scale=isempty(values) ? 0 : p.order===:sublevel ? minimum(values) : maximum(values)
    end
    scale isa Real && !(scale isa Bool) && isfinite(scale) || throw(ArgumentError("scale must be a finite filtration value."))
    vd=p.geometry===nothing ? nothing : _complex_options(p.geometry;parameter=scale,view_dims,max_cells).view_dims
    view_dims===nothing || vd!==nothing || throw(ArgumentError("Supplied source views keep their own coordinates."))
    (;frame,track,representative,scale,view_dims=vd,max_tracks,max_cells)
end
function _append_visual_request_issues!(issues::Vector{String},p::PersistencePath,::Symbol;kwargs...)
    try;_path_selection(p;kwargs...)
    catch err;err isa InterruptException && rethrow();push!(issues,sprint(showerror,err));end
    issues
end

function _path_coordinates(p)
    values=sort!(unique(Real[x for f in p.frames for r in f.records for x in (r.birth,r.death) if isfinite(x)]))
    isempty(values) && (values=Real[0,1])
    parameters=sort!(unique(Real[f.parameter for f in p.frames]))
    for (name,x) in (("endpoints",values),("path parameters",parameters))
        y=Float64.(x)
        all(isfinite,y) && all(>(0),diff(y)) || throw(ArgumentError("Distinct $name collapse or exceed Float64 drawing coordinates; rescale the path."))
    end
    lo,hi=Float64.(extrema(values));span=max(hi-lo,max(abs(lo),abs(hi),1)*0.05)
    lane=p.order===:sublevel ? hi+0.2span : lo-0.2span
    essential=any(r->!isfinite(r.death),Iterators.flatten(f.records for f in p.frames))
    limits=(min(lo,essential ? lane : lo)-0.08span,max(hi,essential ? lane : hi)+0.08span)
    a,b=Float64.(extrema(parameters));margin=max(b-a,max(abs(a),abs(b),1)*0.05)*0.05
    all(isfinite,(limits...,lane,a-margin,b+margin)) || throw(ArgumentError("Path leaves no finite plotting margin."))
    ticks=unique([lo,hi]);labels=_interval_value.(ticks)
    if essential;push!(ticks,lane);push!(labels,p.order===:sublevel ? "+Inf" : "-Inf");end
    (;window=extrema(values),limits,lane,ticks=(ticks,labels),path_limits=(a-margin,b+margin),
        rounded=count(x->Float64(x)!=x,values)+count(x->Float64(x)!=x,parameters))
end

function _path_source(p,o,record)
    f=p.frames[o.frame]
    p.geometry===nothing && return f.source===nothing ? nothing :
        _composition_copy(f.source;title="Source state $(o.frame)",subtitle="Supplied view at path parameter $(_interval_value(f.parameter)).\n"*f.source.subtitle)
    x=p.geometry;G=x.complex
    graded=DataTypes.GradedComplex(G.cells_by_dim,G.boundaries,[(g,) for g in f.grades])
    source=DataIngestion.FilteredComplex(graded,x.geometry,x.sources,x.orientation,x.provenance)
    spec=visual_spec(source;parameter=o.scale,activity=:all,view_dims=o.view_dims,max_cells=o.max_cells)
    # Filtered-complex specs wrap the geometric panel and a readout panel.
    base=isempty(spec.panels) ? spec : first(spec.panels)
    layers=copy(base.layers);witness=record===nothing ? nothing : record.representative
    highlighted=Int[]
    in_interval=record!==nothing && (p.order===:sublevel ? record.birth<=o.scale<record.death : record.death<o.scale<=record.birth)
    if o.representative && witness!==nothing && in_interval
        geometry=Vector{NTuple{length(o.view_dims),Float64}}[]
        for (id,coefficient) in witness.cycle
            id in base.metadata.displayed_indices || continue
            v=[Tuple(Float64(q[k]) for k in o.view_dims) for q in x.geometry[id]]
            push!(geometry,v)
            push!(highlighted,id)
        end
        color=_VisualRole(:categorical,record.id)
        if !isempty(geometry)
            if p.degree==0
                push!(layers,length(o.view_dims)==2 ? PointLayer(only.(geometry),color,1.,18.) : Point3Layer(only.(geometry),color,1.,18.))
            elseif p.degree==1
                push!(layers,length(o.view_dims)==2 ? PolylineLayer(geometry,color,1.,5.,false) :
                    Segment3Layer([Tuple((v[i]...,v[i+1]...)) for v in geometry for i in 1:length(v)-1],color,1.,5.))
            elseif p.degree==2
                push!(layers,length(o.view_dims)==2 ? PolygonLayer(geometry,color,color,0.65,2.) : Polygon3Layer(geometry,color,0.65))
            end
        end
    end
    detail=o.representative && witness!==nothing && in_interval ?
        "\nBold cells in the track color: $(length(highlighted)) of $(length(witness.cycle)) representative cells drawn." : ""
    _composition_copy(base;title="Source at path parameter $(_interval_value(f.parameter))",
        subtitle="Filtration scale $(_interval_value(o.scale)); faint cells are inactive context.\n"*base.subtitle*detail,layers,
        metadata=merge(base.metadata,(;path_frame=o.frame,path_parameter=f.parameter,view_dims=o.view_dims,
            representative_visible=!isempty(highlighted),representative_cells=highlighted)))
end

function _visual_spec(p::PersistencePath,::Symbol;kwargs...)
    o=_path_selection(p;kwargs...);coords=_path_coordinates(p);f=p.frames[o.frame]
    ids=sort!(unique([r.id for frame in p.frames for r in frame.records]));shown=ids[1:min(o.max_tracks,length(ids))]
    o.track===nothing || o.track in shown || (shown[end]=o.track;sort!(shown))
    records=[merge((;id=r.id),p.order===:sublevel ? _interval_record(r.birth,r.death,1) :
        _interval_record(r.death,r.birth,1;left_closed=false,right_closed=true)) for r in f.records if !r.zero && r.id in shown]
    selected=findfirst(r->r.id==o.track,f.records);record=selected===nothing ? nothing : f.records[selected]
    bar,diagram=_interval_panels(records;window=coords.window,order=p.order,max_intervals=o.max_tracks,
        interval=any(r->r.id==o.track,records) ? o.track : nothing,
        barcode_title="Barcode at $(_interval_value(f.parameter))",diagram_title="Diagram at $(_interval_value(f.parameter))")
    bar=_composition_copy(bar;axes=merge(bar.axes,(;xlabel="Filtration parameter",xlimits=coords.limits,xticks=coords.ticks)),
        metadata=merge(bar.metadata,(;path_panel=:barcode,pick_targets=[(;id=r.id,frame=o.frame,point=((s[1]+s[3])/2,s[2])) for (r,s) in zip(bar.metadata.displayed_records,bar.metadata.bar_segments)])))
    diagram=_composition_copy(diagram;axes=merge(diagram.axes,(;xlimits=coords.limits,ylimits=coords.limits,xticks=coords.ticks,yticks=coords.ticks)),
        metadata=merge(diagram.metadata,(;path_panel=:diagram,pick_targets=[(;id=r.id,frame=o.frame,point=q) for (r,q) in zip(diagram.metadata.displayed_records,diagram.metadata.diagram_points)])))
    panels=VisualizationSpec[]
    for endpoint in (:birth,:death)
        layers=AbstractVisualizationLayer[];targets=NamedTuple[]
        for id in shown
            color=_VisualRole(:categorical,id)
            for (i,frame) in enumerate(p.frames)
                k=findfirst(r->r.id==id,frame.records);k===nothing && continue
                r=frame.records[k];value=r[endpoint];y=isfinite(value) ? Float64(value) : coords.lane
                point=(Float64(frame.parameter),y)
                push!(layers,PointLayer([point],color,r.zero ? 0.35 : 0.9,o.track==id ? 10.0 : 6.0))
                push!(targets,(;id,frame=i,point))
                if i>1 && (id,id) in p.transitions[i-1].links
                    previous=only(filter(r->r.id==id,p.frames[i-1].records));v=previous[endpoint]
                    old=(Float64(p.frames[i-1].parameter),isfinite(v) ? Float64(v) : coords.lane)
                    push!(layers,SegmentLayer([(old...,point...)],color,o.track===nothing||o.track==id ? 0.95 : 0.25,
                        o.track==id ? 3.5 : 1.5,p.mode===:certified ? :solid : :dash))
                end
            end
        end
        marker=Float64(f.parameter)
        push!(layers,SegmentLayer([(marker,coords.limits[1],marker,coords.limits[2])],_VisualRole(:foreground),0.7,1.,:dash))
        caption=p.mode===:certified ? "Exact piecewise-linear pairing continuation; tied choices need not be unique." :
            p.mode===:matched ? "Dashed links are chosen bottleneck matches. Between samples is not computed." : "Independent states; no correspondence is asserted."
        push!(panels,VisualizationSpec(:path_endpoint;title=endpoint===:birth ? "Birth along the path" : "Death along the path",
            subtitle=caption,layers,axes=_default_axes_2d(xlabel="Path parameter",ylabel=endpoint===:birth ? "Birth" : "Death",
                xlimits=coords.path_limits,ylimits=coords.limits,yticks=coords.ticks,aspect=:auto),
            metadata=(;path_panel=endpoint,pick_targets=targets,minimal_axes=true,exact_parameters=[q.parameter for q in p.frames])))
    end
    append!(panels,[diagram,bar])
    observations=[get(frame.observation,:objective,nothing) for frame in p.frames]
    if any(!isnothing,observations)
        exact=[(;parameter=frame.parameter,value) for (frame,value) in zip(p.frames,observations) if value!==nothing]
        ys=Float64[q.value for q in exact]
        all(isfinite,ys) || throw(ArgumentError("Recorded objective exceeds drawing coordinates."))
        ordered=sort!(unique([q.value for q in exact]))
        all(>(0),diff(Float64.(ordered))) || throw(ArgumentError("Distinct objective values collapse at drawing precision."))
        lo,hi=extrema(ys);margin=max(hi-lo,max(abs(lo),abs(hi),1)*0.05)*0.1
        all(isfinite,(lo-margin,hi+margin)) || throw(ArgumentError("Objective leaves no finite plotting margin."))
        layers=AbstractVisualizationLayer[PointLayer([(Float64(q.parameter),Float64(q.value)) for q in exact],_VisualRole(:source),1.,9.)]
        for i in 1:length(observations)-1
            a,b=observations[i],observations[i+1]
            (a===nothing || b===nothing) && continue
            push!(layers,SegmentLayer([(Float64(p.frames[i].parameter),Float64(a),Float64(p.frames[i+1].parameter),Float64(b))],_VisualRole(:source),0.8,1.5,:dash))
        end
        value=observations[o.frame]
        value===nothing || push!(layers,PointLayer([(Float64(f.parameter),Float64(value))],_VisualRole(:selected),1.,13.))
        label=get(f.observation,:objective_label,"Objective")
        push!(panels,VisualizationSpec(:path_objective;title="Recorded objective",subtitle="Supplied samples; dashed lines only connect consecutive recorded values.",
            layers,axes=_default_axes_2d(xlabel="Path parameter / iteration",ylabel=label,xlimits=coords.path_limits,ylimits=(lo-margin,hi+margin),aspect=:auto),
            metadata=(;exact_values=exact,minimal_axes=true)))
    end
    source=_path_source(p,o,record)
    if source!==nothing
        if isempty(source.panels)
            push!(panels,source)
        else
            # A supplied source can already contain geometry and its readout.
            # Keep the shared renderer's flat panel contract.
            append!(panels,[_composition_copy(child;subtitle=source.subtitle*"\n"*child.subtitle) for child in source.panels])
        end
    end
    transition=o.frame==1 ? nothing : p.transitions[o.frame-1]
    lines=["State $(o.frame) of $(length(p.frames)); path parameter $(_interval_value(f.parameter)). Degree $(p.degree), $(_inspection_field_label(p.field)), $(p.order).",
        p.mode===:certified ? "Certified algorithmic pairing continuation on a fixed chain complex. Track IDs are not canonical homology-class identities." :
        p.mode===:matched ? "Consecutive bottleneck matching. Nonunique choices and discontinuities may occur; no vineyard continuation is certified." :
        "Only the supplied states are known. Colors and IDs are local to each state.",
        "Showing $(length(shown)) of $(length(ids)) tracks. Exact records retain all intervals and multiplicities."]
    get(f.context,:essential_interpretation,nothing) in (:survives_radius_cutoff,:survives_threshold_cutoff) &&
        push!(lines,"Essential endpoints refer to the computed truncated filtration ($(f.context.essential_interpretation)); persistence beyond its construction cutoff is not established.")
    f.source===nothing || push!(lines,"The source panel is a supplied synchronized view; its relation to the diagram is not independently certified.")
    for key in (:settings,:objective,:invariants)
        haskey(f.observation,key) && push!(lines,"Recorded $key: $(f.observation[key]).")
    end
    zeros=count(r->r.zero,f.records)
    zeros>0 && push!(lines,"$zeros zero-length reduction pairs lie on the diagonal; they contribute no barcode interval. Their traces are retained faintly.")
    transition===nothing || push!(lines,"Incoming transition: $(length(transition.links)) linked pairs; $(length(transition.unmatched_before)) end, $(length(transition.unmatched_after)) start. "*(transition.ambiguous ? "Identity is ambiguous or not certified unique." : ""))
    transition!==nothing && transition.kind===:matched && isinf(transition.distance) && push!(lines,"The matching has infinite cost; these extended-cost links do not certify a finite perturbation.")
    isempty(f.events) || push!(lines,"Crossing cells: "*join(["$(e.cells) ($(e.tie))" for e in f.events],", ")*". Same-time events use deterministic engine order.")
    coords.rounded>0 && push!(lines,"$(coords.rounded) exact coordinates are rounded for drawing; exact values remain in the readout.")
    rep=nothing
    if o.track!==nothing
        push!(lines,record===nothing ? "Track $(o.track) is absent at this state; selection is retained along the history." :
            "Track $(o.track): birth $(_interval_value(record.birth)), death $(_interval_value(record.death)). Birth cell $(record.birth_cell), death cell $(record.death_cell) (0 means unavailable or essential).")
        if o.representative
            rep=record===nothing ? nothing : record.representative
            if rep===nothing
                push!(lines,f.keyframe==0 ? "No representative was retained at this inserted crossing state. Select a supplied keyframe." :
                    "No representative is retained for this selection. Compute the input with representatives=true to retain keyframe witnesses.")
            else
                push!(lines,"A chosen reduction cycle, not a canonical representative: $(rep.cycle).")
                rep.bounding_chain===nothing || push!(lines,"Bounding chain at death: $(rep.bounding_chain).")
                p.geometry===nothing || (p.order===:sublevel ? record.birth<=o.scale<record.death : record.death<o.scale<=record.birth) ||
                    push!(lines,"The chosen filtration scale is outside this interval; its cycle is not highlighted on the active source.")
            end
        end
    end
    push!(panels,_inspection_text_panel("The selected path state",lines))
    VisualizationSpec(:persistence_path;title="Persistence along a path",panels,
        metadata=(;selection=o,mode=p.mode,parameter=f.parameter,records=f.records,context=f.context,observation=f.observation,
            transition=transition===nothing ? nothing : merge(transition,(;events=[(;cells=e.cells,time=e.time,tie=e.tie,continuation=e.continuation) for e in transition.events])),
            events=[(;cells=e.cells,time=e.time,tie=e.tie,continuation=e.continuation) for e in f.events],
            selected_record=record,representative=rep,shown_tracks=shown,total_tracks=length(ids),
            exact_frames=[(;parameter=q.parameter,context=q.context,observation=q.observation,records=[Base.structdiff(r,(;representative=r.representative)) for r in q.records]) for q in p.frames],
            panel_columns=2,figure_size=(1120,400*cld(length(panels),2))))
end
