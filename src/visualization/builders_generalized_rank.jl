# Selected finite regions, boundary occurrences and exact continuous worms.
const _RegionRankInput = Union{Modules.PModule,EncodingResult}
const _REGION_RANK_KEYWORDS = (:vertices,:center,:width,:length,:levels,:lengths,:level,
    :witnesses,:boundary,:occurrence,:bar,:budget,:max_vertices,:matrix_limit)
_region_module(M::Modules.PModule)=M
_region_module(enc::EncodingResult)=Results.encoding_module(enc)
_region_grid(M::Modules.PModule)=nothing
_region_grid(enc::EncodingResult)=_inspection_classifier(encoding_map(enc))

function _region_options(obj;vertices=nothing,center=nothing,width=0,length=1,
        levels=(1,2,3),lengths=(1,2,3),level=1,witnesses=false,boundary=false,
        occurrence=0,bar=0,budget=Invariants.GeneralizedRankBudget(),max_vertices=120,matrix_limit=(8,8))
    M=_region_module(obj);grid=_region_grid(obj)
    (vertices===nothing)!=(center===nothing) || throw(ArgumentError("Supply vertices for a finite region or center for a continuous worm."))
    budget isa Invariants.GeneralizedRankBudget || throw(ArgumentError("budget must be a GeneralizedRankBudget."))
    limit=_signed_integer(max_vertices,:max_vertices;lower=1)
    nvertices(M.Q)<=limit || throw(ArgumentError("The finite drawing exceeds max_vertices."))
    M.field isa Union{CoreModules.QQField,CoreModules.PrimeField} || throw(ArgumentError("Region ranks require an exact coefficient field."))
    witnesses isa Bool && boundary isa Bool || throw(ArgumentError("witnesses and boundary must be Bool."))
    oc=_signed_integer(occurrence,:occurrence);ba=_signed_integer(bar,:bar)
    matrix_limit isa Tuple && Base.length(matrix_limit)==2 || throw(ArgumentError("matrix_limit must have two positive counts."))
    ml=Tuple(_signed_integer(x,:matrix_limit;lower=1) for x in matrix_limit)
    if vertices!==nothing
        labels=Invariants.check_generalized_rank_region(M;vertices,budget,throw=true).vertices
        if boundary
            grid===nothing && throw(ArgumentError("A boundary witness requires a positive 2D grid encoding."))
            Invariants.check_gril_query(M,grid;centers=[],budget,throw=true)
        end
        !boundary && (oc!=0 || ba!=0) && throw(ArgumentError("Occurrence and bar selection require boundary=true."))
        return (;mode=:finite,vertices=labels,center=nothing,width=0,length=1,levels=(),lengths=(),level=1,
            witnesses,boundary,occurrence=oc,bar=ba,budget,max_vertices=limit,matrix_limit=ml)
    end
    grid===nothing && throw(ArgumentError("A continuous worm requires a positive 2D grid encoding."))
    witnesses && throw(ArgumentError("Retained limit/colimit witnesses are available for finite-label regions."))
    boundary && throw(ArgumentError("The lattice boundary theorem does not apply to the continuous-worm restriction."))
    (oc==0 && ba==0) || throw(ArgumentError("Worm queries have no boundary occurrence or bar selection."))
    p=_signed_pair(center,"center")
    all(x->x isa Rational,p) || throw(ArgumentError("Worm centers require rational, integer or finite floating coordinates."))
    width isa Union{Integer,Rational,AbstractFloat} && !(width isa Bool) && isfinite(width) ||
        throw(ArgumentError("width must be a finite rational, integer or floating coordinate."))
    d=_signed_exact(width)
    d isa Rational && d>=0 || throw(ArgumentError("width must be finite, nonnegative and rational, integer or floating."))
    ell=_signed_integer(length,:length;lower=1);k=_signed_integer(level,:level;lower=1)
    Invariants.check_gril_query(M,grid;centers=[p],levels,lengths,budget,throw=true)
    ell in lengths && k in levels || throw(ArgumentError("Select a length and level from the declared probes."))
    _drawing_point(p)
    (;mode=:worm,vertices=Int[],center=p,width=d,length=ell,levels=Tuple(Int.(levels)),lengths=Tuple(Int.(lengths)),level=k,
        witnesses,boundary,occurrence=0,bar=0,budget,max_vertices=limit,matrix_limit=ml)
end

function _region_compute(obj,o;previous=nothing,old=nothing)
    M=_region_module(obj);grid=_region_grid(obj)
    if o.mode===:finite
        same=old!==nothing && old.vertices==o.vertices
        rankdata=same && (!o.witnesses || previous.witness!==nothing) ? previous : begin
            r=Invariants.generalized_rank(M;vertices=o.vertices,witnesses=o.witnesses,budget=o.budget)
            (;rank=o.witnesses ? Invariants.generalized_rank(r) : r,witness=o.witnesses ? r : nothing)
        end
        b=o.boundary ? (same && previous.boundary!==nothing ? previous.boundary :
            Invariants.boundary_zigzag(M,grid;vertices=o.vertices,budget=o.budget)) : nothing
        if b!==nothing
            o.occurrence<=Base.length(b.labels) || throw(ArgumentError("occurrence exceeds the boundary path."))
            o.bar<=Base.length(b.bars) || throw(ArgumentError("bar exceeds the boundary barcode."))
        end
        return (;rank=rankdata.rank,witness=rankdata.witness,boundary=b,landscape=nothing,
            rank_computed=!same || (o.witnesses && previous.witness===nothing),
            landscape_computed=false,boundary_computed=o.boundary && !(same && previous.boundary!==nothing))
    end
    same=old!==nothing && old.center==o.center
    landscape=same ? previous.landscape : Invariants.gril(M,grid;centers=[o.center],levels=o.levels,lengths=o.lengths,budget=o.budget)
    reuse=same && old.width==o.width && old.length==o.length
    rank=reuse ? previous.rank : Invariants.worm_rank(M,grid;center=o.center,width=o.width,length=o.length,budget=o.budget)
    (;rank,witness=nothing,boundary=nothing,landscape,rank_computed=!reuse,landscape_computed=!same,boundary_computed=false)
end

function _region_finite_panel(M,o,data)
    graph=_hasse_spec(M.Q;dims=M.dims)
    positions=graph.metadata.positions
    layers=AbstractVisualizationLayer[]
    for layer in graph.layers
        if layer isa PointLayer && layer.color isa _VisualRole && layer.color.kind===:categorical
            selected=layer.color.index in o.vertices
            push!(layers,PointLayer(layer.points,_VisualRole(selected ? :selected : :muted),selected ? 1.0 : 0.35,selected ? 22.0 : 13.0))
        else
            push!(layers,layer)
        end
    end
    targets=[(;kind=:vertex,vertex=i,point=p,label="Finite label $i; dimension $(M.dims[i]); $(i in o.vertices ? "selected" : "unselected")") for (i,p) in enumerate(positions)]
    VisualizationSpec(:generalized_rank_domain;title="Selected finite region: rank $(data.rank)",
        subtitle="Connected and order-convex; highlighted labels\nSchematic order, not ambient parameter coordinates",
        layers,axes=graph.axes,metadata=(;pick_targets=targets,exact_vertices=copy(o.vertices)))
end

function _region_worm_panel(obj,o,data)
    M=_region_module(obj);grid=_region_grid(obj);ax=grid.coords;p=o.center;d=o.width;e=o.length
    # Exact Minkowski sum of a square and an antidiagonal segment.
    x,y=p
    polygon=unique([(x-e*d,y+(e-2)*d),(x+(e-2)*d,y-e*d),(x+e*d,y-e*d),
        (x+e*d,y-(e-2)*d),(x-(e-2)*d,y+e*d),(x-e*d,y+e*d)])
    drawn=_drawing_point.(polygon)
    exact_points=[(a,b) for b in ax[2] for a in ax[1]]
    points=_drawing_point.(exact_points)
    collisions=_display_collisions([exact_points;polygon])
    xmin,xmax=extrema(first.([points;drawn]));ymin,ymax=extrema(last.([points;drawn]))
    padx=max((xmax-xmin)*0.12,0.5);pady=max((ymax-ymin)*0.12,0.5)
    xlim=(min(xmin-padx,prevfloat(xmin)),max(xmax+padx,nextfloat(xmax)))
    ylim=(min(ymin-pady,prevfloat(ymin)),max(ymax+pady,nextfloat(ymax)))
    layers=AbstractVisualizationLayer[]
    # True half-open grid boundaries; the lower-left sample is a cell label.
    lines=[[(Float64(a),ylim[1],Float64(a),ylim[2]) for a in ax[1]];
           [(xlim[1],Float64(b),xlim[2],Float64(b)) for b in ax[2]]]
    push!(layers,SegmentLayer(lines,_VisualRole(:muted),0.35,1.0,:dash))
    length(drawn)>2 && push!(layers,PolygonLayer([drawn],_VisualRole(:selected),_VisualRole(:selected),0.22,2.0))
    push!(layers,PointLayer(points,Float64.(M.dims),1.0,11.0))
    push!(layers,TextLayer(["$i: $(M.dims[i])" for i in eachindex(points)],
        [(a+0.04padx,b+0.22pady) for (a,b) in points],_VisualRole(:foreground),10.0))
    push!(layers,PointLayer([_drawing_point(p)],_VisualRole(:selected),1.0,19.0))
    VisualizationSpec(:generalized_rank_worm;title="Continuous worm: rank $(data.rank)",
        subtitle="Closed worm; length $e, width $(_signed_label(d))\nPositive axes; grid labels show label: dimension" *
            (isempty(collisions) ? "" : "\nDrawing precision: distinct exact coordinates overlap"),
        layers,axes=_default_axes_2d(;xlabel="Parameter 1",ylabel="Parameter 2",xlimits=xlim,ylimits=ylim),
        metadata=(;exact_polygon=polygon,center=p,width=d,length=e,grid_axes=deepcopy(ax),display_collisions=collisions,
            extension=:right_continuous_grid_zero_below_constant_above,pick_targets=NamedTuple[]))
end

function _boundary_panels(r;occurrence=0,bar=0)
    path=Invariants.boundary_path(r);bars=Invariants.boundary_bars(r);n=length(path.vertices)
    occurrence<=n && bar<=length(bars) || throw(ArgumentError("Boundary selection is outside the retained result."))
    layers=AbstractVisualizationLayer[];segments=NTuple{4,Float64}[];heads=Vector{NTuple{2,Float64}}[]
    for i in 1:n-1
        a,b=path.arrows[i]==1 ? (i,i+1) : (i+1,i)
        _hasse_arrow!(segments,heads,(Float64(a),0.0),(Float64(b),0.0))
    end
    push!(layers,SegmentLayer(segments,_VisualRole(:edge),1.0,1.6))
    push!(layers,PolygonLayer(heads,_VisualRole(:edge),_VisualRole(:edge),1.0,0.0))
    selected=bar==0 ? nothing : bars[bar]
    targets=NamedTuple[]
    for i in 1:n
        active=i==occurrence || (selected!==nothing && selected[1]<=i<=selected[2])
        push!(layers,PointLayer([(Float64(i),0.)],_VisualRole(active ? :selected : :foreground),1.,active ? 19. : 12.))
        push!(targets,(;kind=:occurrence,occurrence=i,point=(Float64(i),0.),label="Occurrence $i: label $(path.vertices[i]), lattice $(path.indices[i]), dimension $(path.dimensions[i])"))
    end
    push!(layers,TextLayer(["$i\nq=$(path.vertices[i])" for i in 1:n],[(Float64(i),0.35) for i in 1:n],_VisualRole(:foreground),11.))
    pathpanel=VisualizationSpec(:boundary_path;title="Boundary zigzag: distinct occurrences",
        subtitle="Adjacent arrows only; repeated finite labels stay separate",
        layers,axes=_default_axes_2d(;xlabel="Path occurrence",ylabel="",xlimits=(0.5,n+0.5),ylimits=(-0.6,1.0),aspect=:auto,
            yticks=(Float64[],String[])),metadata=(;pick_targets=targets,path))
    barlayers=AbstractVisualizationLayer[];targets=NamedTuple[]
    for (i,(a,b,m)) in enumerate(bars)
        full=a==1 && b==n;role=_VisualRole(i==bar ? :selected : full ? :positive : :muted)
        push!(barlayers,SegmentLayer([(Float64(a),Float64(i),Float64(b),Float64(i))],role,1.0,full ? 5.0 : 3.0))
        push!(barlayers,PointLayer([(Float64(a),Float64(i)),(Float64(b),Float64(i))],role,1.0,8.0))
        push!(barlayers,TextLayer(["x$m"],[(b+0.16,Float64(i))],_VisualRole(:foreground),11.))
        push!(targets,(;kind=:bar,bar=i,point=((a+b)/2,Float64(i)),label="Bar $i: occurrences [$a, $b], multiplicity $m$(full ? "; full bar" : "")"))
    end
    isempty(bars) && push!(barlayers,TextLayer(["No nonzero bars"],[(1.,1.)],_VisualRole(:muted),12.))
    barpanel=VisualizationSpec(:boundary_barcode;title="Full-bar multiplicity = $(r.rank)",
        subtitle="Inclusive occurrence endpoints; multiplicities are retained",
        layers=barlayers,axes=_default_axes_2d(;xlabel="Path occurrence",ylabel="Barcode entry",xlimits=(0.5,n+0.8),ylimits=(0.4,max(1,length(bars))+0.7),aspect=:auto),
        metadata=(;pick_targets=targets,bars,full_rank=r.rank))
    [pathpanel,barpanel]
end

function _gril_layers_panel(r;center=1,level=first(r.levels),length=first(r.lengths),width=nothing)
    center=_signed_integer(center,:center;lower=1)
    level=_signed_integer(level,:level;lower=1);length=_signed_integer(length,:length;lower=1)
    1<=center<=Base.length(r.centers) || throw(ArgumentError("center must index a retained GRIL probe."))
    level in r.levels && length in r.lengths || throw(ArgumentError("Choose a retained level and length."))
    layers=AbstractVisualizationLayer[];targets=NamedTuple[];largest=0.0
    for (row,k) in enumerate(r.levels)
        v=Invariants.value_at(r;center,level=k,length);x=Float64(v);isfinite(x) || throw(ArgumentError("GRIL width cannot be drawn as Float64."))
        largest=max(largest,x);role=_VisualRole(k==level ? :selected : :positive)
        push!(layers,SegmentLayer([(0.,Float64(row),x,Float64(row))],role,1.,k==level ? 5. : 3.))
        push!(layers,PointLayer([(x,Float64(row))],role,1.,10.))
        push!(layers,TextLayer([_signed_label(v)],[(x,Float64(row)+0.2)],_VisualRole(:foreground),11.))
        push!(targets,(;kind=:level,level=k,point=(x,Float64(row)),label="Rank level $k: exact supremum $(_signed_label(v)), length $length"))
    end
    if width!==nothing
        x=Float64(width);isfinite(x) || throw(ArgumentError("width cannot be drawn as Float64."));largest=max(largest,x)
        push!(layers,SegmentLayer([(x,0.5,x,Base.length(r.levels)+0.5)],_VisualRole(:selected),0.8,1.5,:dash))
    end
    VisualizationSpec(:gril_layers;title="Exact GRIL widths; length $length",
        subtitle="A supremum need not be attained; labels retain exact values",
        layers,axes=_default_axes_2d(;xlabel="Worm width",ylabel="Rank level",xlimits=(-0.05max(1,largest),1.3max(1,largest)),
            ylimits=(0.5,Base.length(r.levels)+0.6),yticks=(Float64.(1:Base.length(r.levels)),string.(r.levels)),aspect=:auto),
        metadata=(;pick_targets=targets,values=Invariants.gril_values(r),center,level,length))
end

function _region_snapshot(obj,o,data)
    M=_region_module(obj)
    panels=VisualizationSpec[o.mode===:finite ? _region_finite_panel(M,o,data) : _region_worm_panel(obj,o,data)]
    if o.mode===:worm
        push!(panels,_gril_layers_panel(data.landscape;level=o.level,length=o.length,width=o.width))
    elseif data.boundary!==nothing
        append!(panels,_boundary_panels(data.boundary;occurrence=o.occurrence,bar=o.bar))
    end
    if o.witnesses
        r=data.witness
        coordinates=["q$v[$j]" for v in o.vertices for j in 1:M.dims[v]]
        limit=Invariants.limit_basis(r);projection=Invariants.colimit_projection(r)
        comparison=Invariants.comparison_map(r)
        sections=["section $j" for j in axes(limit,2)]
        classes=["class $i" for i in axes(projection,1)]
        for (A,title,rows,cols) in ((limit,"Compatible sections",coordinates,sections),
                (projection,"Colimit projection",classes,coordinates),
                (comparison,"Limit-to-colimit comparison",classes,sections))
            push!(panels,_presentation_matrix_panel(A,title,rows,cols,o.matrix_limit;
                subtitle="Exact entries over $(_inspection_field_label(M.field))",metadata=(;field=M.field)))
        end
    end
    lines=["Exact rank $(data.rank) over $(_inspection_field_label(M.field))."]
    if o.mode===:finite
        push!(lines,"Finite labels: $(join(o.vertices, ", ")). Connected and order-convex in the finite base.")
        push!(lines,"This query uses the retained spaces and maps, including their branching compatibility.")
        o.boundary && push!(lines,"The boundary barcode is on path occurrences. Full bars recover this region rank, not an ambient decomposition.")
    else
        push!(lines,"Center $(_signed_pair_label(o.center)); width $(_signed_label(o.width)); length $(o.length).")
        push!(lines,"Closed continuous worm; no radius mesh. Half-open grid cells; zero below either first axis, constant above the last.")
        push!(lines,"The dashed width is a rank query. GRIL labels are suprema; test endpoint attainment with the displayed exact rank.")
    end
    push!(lines,"Drawing uses floating coordinates; query coordinates and mathematical answers remain exact.")
    push!(panels,_inspection_text_panel("Reading this query",lines))
    positions=Tuple{UnitRange{Int},UnitRange{Int}}[]
    if data.boundary!==nothing
        append!(positions,[(1:2,1:3),(1:1,4:6),(2:2,4:6)])
        weights=[260,260];nextrow=3
    elseif o.mode===:worm
        append!(positions,[(1:1,1:3),(1:1,4:6)])
        weights=[410];nextrow=2
    else
        push!(positions,(1:1,1:6));weights=[410];nextrow=2
    end
    if o.witnesses
        append!(positions,[(nextrow:nextrow,1:2),(nextrow:nextrow,3:4),(nextrow:nextrow,5:6)])
        push!(weights,max(250,160+24min(size(Invariants.limit_basis(data.witness),1),o.matrix_limit[1])));nextrow+=1
    end
    push!(positions,(nextrow:nextrow,1:6));push!(weights,80+26Base.length(lines))
    VisualizationSpec(:generalized_rank;title="What persists across this region?",panels,
        metadata=(;selection=o,rank=data.rank,field=M.field,witness=data.witness,boundary=data.boundary,
            landscape=data.landscape,panel_columns=6,panel_positions=positions,panel_row_weights=weights,
            figure_size=(1250,sum(weights)+80)))
end
function _generalized_rank_spec(obj;kwargs...)
    o=_region_options(obj;kwargs...)
    _region_snapshot(obj,o,_region_compute(obj,o))
end
function _check_region_request!(issues,obj;kwargs...)
    _region_options(obj;kwargs...);issues
end

available_visuals(::Invariants.BoundaryZigzagResult)=(:boundary_zigzag,)
_visual_request_keywords(::Invariants.BoundaryZigzagResult,::Symbol)=(:occurrence,:bar)
function _visual_spec(r::Invariants.BoundaryZigzagResult,::Symbol;occurrence=0,bar=0)
    Invariants.check_boundary_zigzag(r;throw=true)
    _signed_integer(occurrence,:occurrence);_signed_integer(bar,:bar)
    VisualizationSpec(:boundary_zigzag;title="Boundary witness",panels=_boundary_panels(r;occurrence,bar),
        metadata=(;panel_columns=1,figure_size=(1050,650),boundary=r))
end
available_visuals(::Invariants.GRILResult)=(:gril,)
_visual_request_keywords(::Invariants.GRILResult,::Symbol)=(:center,:level,:length)
function _visual_spec(r::Invariants.GRILResult,::Symbol;center=1,level=first(r.levels),length=first(r.lengths))
    isempty(r.centers) && throw(ArgumentError("An empty GRIL result has no probe to display."))
    layerspanel=_gril_layers_panel(r;center,level,length)
    points=_drawing_point.(r.centers);collisions=_display_collisions(r.centers)
    exactvals=[Invariants.value_at(r;center=i,level,length) for i in eachindex(points)]
    vals=Float64.(exactvals)
    xmin,xmax=extrema(first.(points));ymin,ymax=extrema(last.(points))
    padx=max((xmax-xmin)*0.25,0.5);pady=max((ymax-ymin)*0.25,0.5)
    xlim=(min(xmin-padx,prevfloat(xmin)),max(xmax+padx,nextfloat(xmax)))
    ylim=(min(ymin-pady,prevfloat(ymin)),max(ymax+pady,nextfloat(ymax)))
    all(isfinite,vals) || throw(ArgumentError("GRIL widths cannot be drawn as Float64."))
    panel=VisualizationSpec(:gril_centers;title="Fixed probes: level $level, length $length",
        subtitle="Labels give probe: exact width; no interpolation between probes" *
            (isempty(collisions) ? "" : "\nDrawing precision: distinct exact centers overlap"),
        layers=AbstractVisualizationLayer[PointLayer(points,vals,1.,16.),
            PointLayer([points[center]],_VisualRole(:selected),0.7,25.),
            TextLayer(["$i: $(_signed_label(exactvals[i]))" for i in eachindex(points)],
                [(x,y+0.3pady) for (x,y) in points],_VisualRole(:foreground),12.)],
        axes=_default_axes_2d(;xlabel="Parameter 1",ylabel="Parameter 2",xlimits=xlim,ylimits=ylim),
        metadata=(;centers=Invariants.gril_centers(r),values=Invariants.gril_values(r),display_collisions=collisions))
    VisualizationSpec(:gril;title="Generalized-rank landscape probes",panels=[panel,layerspanel],
        metadata=(;panel_columns=2,figure_size=(1200,460),center,level,length))
end
