# A128: exact source selection, then absolute/subcomplex/relative calculations.
available_visuals(::OrdinaryPersistence.SpatialPair)=(:relative_explorer,)
const _RELATIVE_KEYS=(:scale,:degree,:details,:witness,:view_dims,:max_cells,:max_algebra_cells,:max_matrix_entries)
_visual_request_keywords(::OrdinaryPersistence.SpatialPair,::Symbol)=_RELATIVE_KEYS
_visual_request_cost(::OrdinaryPersistence.SpatialPair,::Symbol)=(;work=:three_persistence_reductions,
    basis=:explicit_opt_in,geometry=:retained_source,hover=:read_only,timing=:not_measured)
function _relative_options(s;scale=nothing,degree=0,details=:summary,witness=1,view_dims=nothing,
        max_cells=10000,max_algebra_cells=2000,max_matrix_entries=1000000)
    G=s.pair.ambient
    degree isa Int && 0<=degree<length(G.cells_by_dim) || throw(ArgumentError("degree is outside the ambient chain complex."))
    details in (:summary,:maps,:representative) || throw(ArgumentError("details must be :summary, :maps or :representative."))
    witness isa Int && witness>=1 || throw(ArgumentError("witness must be a positive integer."))
    all(v->v isa Int && v>0,(max_cells,max_algebra_cells,max_matrix_entries)) || throw(ArgumentError("Budgets must be positive integers."))
    n=length(G.grades);n<=max_algebra_cells || throw(ArgumentError("Pair exceeds max_algebra_cells; no truncated pair is computed."))
    details===:summary || sum(big(c)^2 for c in DataTypes.cell_counts(G))<=max_matrix_entries ||
        throw(ArgumentError("Explicit homology exceeds max_matrix_entries."))
    scale=scale===nothing ? (isempty(G.grades) ? 0 : s.pair.order===:sublevel ? maximum(first,G.grades) : minimum(first,G.grades)) : scale
    scale isa Real && !(scale isa Bool) && isfinite(scale) || throw(ArgumentError("scale must be a finite real coordinate."))
    vd=s.source isa DataIngestion.FilteredComplex ? _complex_options(s.source;parameter=scale,view_dims,max_cells).view_dims : nothing
    view_dims===nothing || vd!==nothing || throw(ArgumentError("Matrix-only input has no spatial projection."))
    (;scale,degree,details,witness,view_dims=vd,max_cells,max_algebra_cells,max_matrix_entries)
end
function _append_visual_request_issues!(issues::Vector{String},s::OrdinaryPersistence.SpatialPair,::Symbol;kwargs...)
    try;_relative_options(s;kwargs...)
    catch err;err isa InterruptException && rethrow();push!(issues,sprint(showerror,err));end
    issues
end
function _relative_diagrams(s)
    NamedTuple{(:ambient,:subcomplex,:relative)}(Tuple(OrdinaryPersistence.persistence_diagram(s;part) for part in (:ambient,:subcomplex,:relative)))
end
function _relative_count(diagram,degree,scale)
    order=diagram.order
    count(v->order===:sublevel ? v[1]<=scale<v[2] : v[2]<scale<=v[1],OrdinaryPersistence.finite_intervals(diagram;dim=degree))+
    count(v->order===:sublevel ? v<=scale : v>=scale,OrdinaryPersistence.essential_births(diagram;dim=degree))
end
function _relative_geometry(s,o,table;part=:ambient,coefficients=nothing)
    x=s.source
    base=first(visual_spec(x;parameter=o.scale,activity=:all,view_dims=o.view_dims,max_cells=max(o.max_cells,length(table))).panels)
    layers=AbstractVisualizationLayer[];targets=NamedTuple[]
    active(g)=s.pair.order===:sublevel ? g<=o.scale : g>=o.scale
    eligible=base.metadata.displayed_indices
    ids=unique(vcat(filter(in(eligible),s.seeds),filter(in(eligible),s.added),eligible))
    resize!(ids,min(length(ids),o.max_cells))
    for i in sort(ids;by=i->(-x.complex.cell_dims[i],i))
        r=table[i];v=[Tuple(Float64(p[k]) for k in o.view_dims) for p in x.geometry[i]]
        inK=active(r.ambient_grade);inA=r.selected && active(r.subcomplex_grade)
        role=coefficients!==nothing && haskey(coefficients,i) ? :selected : r.role===:seed ? :selected : r.role===:added ? :target : :muted
        visible=part===:subcomplex ? inA : part===:relative ? inK && !inA : inK
        opacity=!visible ? 0.08 : r.dimension==2 ? 0.4 : 1.0
        _complex_layers!(layers,v,r.dimension,role,opacity)
        push!(targets,(;index=i,dimension=r.dimension,point=Tuple(sum(p[k] for p in v)/length(v) for k in eachindex(o.view_dims)),role=r.role))
    end
    if s.brush!==nothing && length(o.view_dims)==2 && Set(s.brush.dims)==Set(o.view_dims)
        indices=[findfirst(==(d),s.brush.dims) for d in o.view_dims]
        lo=Float64.([s.brush.lower[i] for i in indices]);hi=Float64.([s.brush.upper[i] for i in indices])
        push!(layers,PolylineLayer([[(lo[1],lo[2]),(hi[1],lo[2]),(hi[1],hi[2]),(lo[1],hi[2])]],_VisualRole(:foreground),0.8,1.5,true))
    end
    title=coefficients!==nothing ? "Chosen quotient cycle, lifted to K" : part===:ambient ? "Ambient K: selected cells and added faces" : "Selected subcomplex A"
    subtitle=coefficients===nothing ? "Scale $(o.scale). Seed cells and closure/induced additions have separate legend entries." :
        "Scale $(o.scale). The computed quotient cycle is lifted to ambient source cells."
    entries=coefficients===nothing ? [(;label="seed",color=_VisualRole(:selected),style=:patch),
        (;label="added",color=_VisualRole(:target),style=:patch)] :
        [(;label="witness support",color=_VisualRole(:selected),style=:patch)]
    _composition_copy(base;title,subtitle=subtitle*"\nFaint geometry is context. $(length(table)-length(ids)) cells outside the displayed skeleton/budget.",layers,
        legend=_default_legend(;visible=true,entries),
        metadata=merge(base.metadata,(;pick_targets=targets,displayed_indices=ids,part,coefficients)))
end
function _relative_matrix(A,title)
    nr,nc=min(size(A,1),12),min(size(A,2),12)
    VisualizationSpec(:relative_matrix;title,subtitle="$(size(A,1)) x $(size(A,2)); target rows, source columns. Chosen homology bases.",
        layers=AbstractVisualizationLayer[MatrixLayer([_matrix_coefficient_text(A[i,j]) for i in 1:nr,j in 1:nc],
            ["e$i" for i in 1:nr],["e$j" for j in 1:nc])],metadata=(;panel_style=:matrix,matrix=A,matrix_size=size(A),truncated=nr<size(A,1)||nc<size(A,2)))
end
function _relative_view(s;_diagrams=nothing,kwargs...)
    o=_relative_options(s;kwargs...);O=OrdinaryPersistence
    O.check_spatial_pair(s;throw=true)
    diagrams=_diagrams===nothing ? _relative_diagrams(s) : _diagrams
    table=O.spatial_cells(s);counts=map(d->_relative_count(d,o.degree,o.scale),diagrams)
    panels=VisualizationSpec[]
    if s.source isa DataIngestion.FilteredComplex
        push!(panels,_relative_geometry(s,o,table))
        push!(panels,_relative_geometry(s,o,table;part=:subcomplex))
    end
    endpoints=vcat([g[1] for g in s.pair.ambient.grades],collect(Iterators.flatten(s.pair.grades)),[o.scale])
    window=extrema(endpoints)
    barcode_start=length(panels)+1
    for (part,title) in ((:ambient,"Absolute H$(o.degree)(K)"),(:subcomplex,"Selected H$(o.degree)(A)"),(:relative,"Relative H$(o.degree)(K,A)"))
        panel=visual_spec(diagrams[part];kind=:barcode,dim=o.degree,window)
        push!(panels,_composition_copy(panel;title,subtitle="Dimension at $(o.scale): $(counts[part]).\n"*panel.subtitle))
    end
    limits=(minimum(p.axes.xlimits[1] for p in panels[barcode_start:end]),maximum(p.axes.xlimits[2] for p in panels[barcode_start:end]))
    for i in barcode_start:length(panels)
        panels[i]=_composition_copy(panels[i];axes=merge(panels[i].axes,(;xlimits=limits)))
    end
    lines=["Rule: $(s.rule). $(length(s.seeds)) seeds; $(length(s.added)) added cells; $(length(s.selected)) cells in A across all scales.",
        s.rule===:chain_closure ? "Coordinate chain pair over $(s.pair.field); geometric closure is not asserted." : "Geometric closure uses retained face incidence, independently of field cancellation.",
        "Both K-entry and A-entry grades are retained. Selection constructs a filtered pair; it does not clip cells.",
        "This is one-parameter relative persistence. Quotient bases at a scale are not interval correspondences."]
    if s.source isa DataIngestion.FilteredComplex
        any(s.source.provenance.periodic) && push!(lines,"Periodic fundamental-domain drawing: opposite faces are identified; source cells are not clipped or wrapped by the brush.")
        length(o.view_dims)<s.source.provenance.ambient_dim && push!(lines,"Source coordinate projection $(o.view_dims); overlapping drawings do not identify source cells.")
        append!(lines,unique(vcat(panels[1].metadata.warnings,panels[2].metadata.warnings)))
    end
    s.brush===nothing || push!(lines,"Closed brush $(s.brush). "*(s.rule===:induced ? "Selects vertices." : "Selects vertex-centroids.")*" Omitted axes are extruded; periodic coordinates use the unwrapped drawing.")
    maps=nothing;representative=nothing
    if o.details!==:summary
        h=O.relative_homology(s;scale=o.scale);maps=O.relative_maps(h;dim=o.degree)
        for (name,title) in ((:inclusion,"Inclusion H(A) -> H(K)"),(:projection,"Projection H(K) -> H(K,A)"),(:connecting,"Connecting H$(o.degree)(K,A) -> H$(o.degree-1)(A)"))
            push!(panels,_relative_matrix(maps[name],title))
        end
        if o.details===:representative
            Z=O.basis(h;dim=o.degree)
            o.witness<=size(Z,2) || throw(ArgumentError("There are $(size(Z,2)) quotient witnesses at this scale."))
            c=O.relative_complex(h);indices=c.indices.relative[o.degree+1];offsets=s.pair.ambient.dim_offsets
            vector=Z[:,o.witness]
            coefficients=Dict(offsets[o.degree+1]+i-1=>vector[j] for (j,i) in enumerate(indices) if !iszero(vector[j]))
            lift=zeros(CoreModules.coeff_type(s.pair.field),length(c.indices.ambient[o.degree+1]))
            for (j,i) in enumerate(c.indices.ambient[o.degree+1]);lift[j]=get(coefficients,offsets[o.degree+1]+i-1,zero(eltype(lift)));end
            boundary=O.relative_boundary(c;dim=o.degree,part=:ambient)*lift
            representative=(;dimension=o.degree,index=o.witness,coefficients,
                boundary,boundary_cell_indices=o.degree==0 ? Int[] : [offsets[o.degree]+i-1 for i in c.indices.ambient[o.degree]],
                connecting_coordinates=maps.connecting[:,o.witness],scale=o.scale)
            s.source isa DataIngestion.FilteredComplex && push!(panels,_relative_geometry(s,o,table;part=:relative,coefficients))
            terms=join(["cell $i: $(_matrix_coefficient_text(c))" for (i,c) in sort!(collect(coefficients);by=first)],", ")
            push!(lines,"Quotient witness $(o.witness), coefficients by source cell: $terms. Its ambient boundary lies in A; the displayed connecting coordinates are computed from that boundary.")
        end
    end
    push!(panels,_inspection_text_panel("The selected pair",lines))
    VisualizationSpec(:relative_explorer;title="Select a region and ask a relative question",panels,
        metadata=(;options=o,counts,table,seeds=copy(s.seeds),selected=copy(s.selected),added=copy(s.added),rule=s.rule,
            brush=s.brush,maps,representative,panel_columns=2,figure_size=(1100,420*cld(length(panels),2))))
end
_visual_spec(s::OrdinaryPersistence.SpatialPair,::Symbol;kwargs...)=_relative_view(s;kwargs...)
