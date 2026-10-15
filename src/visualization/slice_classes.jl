# Exact interval coordinates and ambient maps between selected slice stalks.
"""
    slice_explorer(source; degree=0, field=F2(), slice=(basepoint=(0,0),direction=(1,1)), ...)

Build a finite encoding with retained homology bases from a planar source
complex, then open its linked slice inspector. The finite grid contains every
critical coordinate, including multicritical appearances. Its maps are induced
by inclusions of actual chain groups. Source cycles therefore need no supplied
basis identification on this route. For an existing encoding use
`inspection_session(enc; slice_classes=true, source_basis=...)` instead.

This explicit exploratory construction retains dense homology data. `class_budget`
bounds its grid, chain algebra and representatives before computation. Counts,
geometry and derived bases are fixed for the lifetime of the session.
"""
function slice_explorer(source::DataIngestion.FilteredComplex;degree=0,field=CoreModules.F2(),
        slice=(basepoint=(0,0),direction=(1,1)),box=nothing,
        class_budget=ZigzagPersistence.ZigzagBudget(max_vertices=256),kwargs...)
    DataIngestion.check_filtered_complex(source;throw=true)
    G=source.complex;counts=DataTypes.cell_counts(G)
    DataTypes.parameter_dim(G)==2 && source.orientation==(1,1) ||
        throw(ArgumentError("A source slice explorer requires two positively oriented parameters."))
    degree isa Integer && !(degree isa Bool) && 0<=degree<length(counts) ||
        throw(ArgumentError("degree must index a retained source chain group."))
    field isa Union{CoreModules.QQField,CoreModules.PrimeField} || throw(ArgumentError("Source slices require an exact field."))
    n=sum(counts);n<=class_budget.max_cells || throw(ArgumentError("Source exceeds max_cells."))
    grades=[g for i in 1:n for g in DataTypes.cell_grade_set(G,i)]
    axes=ntuple(2) do k
        values=sort!(unique([_visual_exact_coordinate(g[k]) for g in grades]))
        isempty(values) ? [zero(_VisualRational),one(_VisualRational)] : [first(values)-1;values;last(values)+1]
    end
    size_=length.(axes);ng=prod(big.(size_))
    ng<=class_budget.max_vertices || throw(ArgumentError("Source grid exceeds max_vertices."))
    ng*sum(big(c)^2 for c in counts)*6<=class_budget.max_matrix_entries ||
        throw(ArgumentError("Source grid homology exceeds max_matrix_entries."))
    P=FiniteFringe.ProductOfChainsPoset(size_)
    points=[(axes[1][i],axes[2][j]) for j in eachindex(axes[2]) for i in eachindex(axes[1])]
    models=[_atlas_source_model(source,p,Int(degree),field) for p in points]
    K=CoreModules.coeff_type(field)
    maps=Dict((a,b)=>ChainComplexes.induced_map_on_homology(models[a].H,models[b].H,
        _atlas_inclusion(models[a],models[b],K)) for (a,b) in FiniteFringe.cover_edges(P))
    M=Modules.PModule{K}(P,[m.H.dimH for m in models],maps;field)
    pi=GridEncodingMap(P,axes)
    enc=Results.EncodingResult(P,M,pi)
    bases=map(models) do model
        Z=zeros(K,counts[degree+1],model.H.dimH)
        Z[model.ids,:]=ChainComplexes.basis(model.H)
        Z
    end
    link=DataIngestion.source_homology_basis(source;degree,basis=(label,point)->bases[label])
    box===nothing && (box=(Tuple(first.(axes)),Tuple(last.(axes))))
    inspection_session(enc;box,slice,slice_classes=true,source_basis=link,class_budget,kwargs...)
end

Base.@constprop :none Base.@noinline Base.@nospecializeinfer function _inspection_slice_classes(Base.@nospecialize(obj),Base.@nospecialize(result),Base.@nospecialize(prepared))
    get(prepared,:slice_classes,false) || return result
    isempty(result.chain) && return merge(result,(;classes=nothing))
    Z=ZigzagPersistence;M=Results.encoding_module(obj);F=M.field
    F isa Union{CoreModules.QQField,CoreModules.PrimeField} ||
        throw(ArgumentError("Slice classes require QQ or a prime field."))
    dims=[M.dims[q] for q in result.chain]
    budget=prepared.class_budget
    sum(big(d)^2 for d in dims)<=budget.max_representative_entries ||
        throw(ArgumentError("Slice class representatives exceed max_representative_entries."))
    sum((big(dims[i])*dims[i+1] for i in 1:length(dims)-1);init=big(0))+6big(maximum(dims))^2<=budget.max_matrix_entries ||
        throw(ArgumentError("Slice class maps exceed max_matrix_entries."))
    if prepared.source_basis!==nothing
        counts=DataTypes.cell_counts(prepared.source_basis.source.complex)
        big(length(dims))*sum(big(c)^2 for c in counts)<=budget.max_matrix_entries ||
            throw(ArgumentError("Retained source bases exceed max_matrix_entries."))
    end
    maps=[Matrix(Modules.structure_map(M;source=result.chain[i],target=result.chain[i+1]))
        for i in 1:length(dims)-1]
    z=Z.ZigzagModule(dims,maps;directions=fill(:forward,length(maps)),field=F,budget)
    r=Z.zigzag_persistence(z;representatives=true,budget)
    groups=[findall(==(record.strata),r.supports) for record in result.intervals]
    all(length(groups[i])==result.intervals[i].multiplicity for i in eachindex(groups)) ||
        error("Slice rank barcode and interval basis decomposition disagree.")
    source=prepared.source_basis
    bases=source===nothing ? nothing : [DataIngestion._slice_source_basis(source,result.chain[i],
        ntuple(k->result.line.basepoint[k]+result.sample_parameters[i]*result.line.direction[k],2),
        dims[i],F,budget.max_matrix_entries) for i in eachindex(dims)]
    if bases!==nothing
        for i in eachindex(maps);DataIngestion._slice_source_map(F,bases[i],bases[i+1],maps[i]);end
    end
    merge(result,(;classes=(;supports=r.supports,vectors=r.representatives,groups,bases)))
end

function _slice_class_vertex(result,t)
    t isa Real && !(t isa Bool) && isfinite(t) || throw(ArgumentError("Class parameter must be finite and real."))
    if result.scope===:window
        result.window!==nothing && result.window[1]<=t<=result.window[2] ||
            throw(ArgumentError("Class parameter lies outside the computed slice window."))
    end
    k=searchsortedlast(collect(result.events),t)
    event=k>0 && result.events[k]==t
    result.scope===:global ? (event ? 2k : 2k+1) : (event ? 2k-1 : 2k)
end

function _slice_class_vector(result,group,member,t)
    classes=get(result,:classes,nothing)
    classes===nothing && throw(ArgumentError("Open the session with slice_classes=true to retain interval coordinates."))
    group isa Integer && !(group isa Bool) && 1<=group<=length(result.intervals) ||
        throw(ArgumentError("Choose an interval group before inspecting a class."))
    members=classes.groups[group]
    member isa Integer && !(member isa Bool) && 1<=member<=length(members) ||
        throw(ArgumentError("Member must be in 1:$(length(members))."))
    id=members[member];a,b=classes.supports[id]
    t===nothing && (t=result.sample_parameters[a])
    v=_slice_class_vertex(result,t)
    a<=v<=b || throw(ArgumentError("The selected interval is not alive at parameter $t."))
    (;id,vertex=v,parameter=t,label=result.chain[v],
      point=ntuple(k->result.line.basepoint[k]+t*result.line.direction[k],2),
      coordinates=copy(classes.vectors[id][v-a+1]))
end

function _slice_basis_at(obj,prepared,result,point,label)
    b=prepared.source_basis;b===nothing && return nothing
    M=Results.encoding_module(obj)
    if result!==nothing && get(result,:classes,nothing)!==nothing && result.classes.bases!==nothing
        match=findfirst(x->x.label==label && x.point==point,result.classes.bases)
        match===nothing || return result.classes.bases[match]
    end
    DataIngestion._slice_source_basis(b,label,point,M.dims[label],M.field,prepared.class_budget.max_matrix_entries)
end

Base.@constprop :none Base.@noinline Base.@nospecializeinfer function _slice_selected_class(Base.@nospecialize(obj),Base.@nospecialize(prepared),Base.@nospecialize(result),Base.@nospecialize(state))
    (result===nothing || state.interval===nothing || !prepared.slice_classes) && return nothing
    c=_slice_class_vector(result,state.interval,state.member,state.class_parameter)
    source=_slice_basis_at(obj,prepared,result,c.point,c.label)
    F=_inspection_field(obj);cycle=nothing;filling=nothing
    if source!==nothing
        canonical=result.classes.bases[c.vertex]
        # The query may lie elsewhere in the same constant stratum. Verify
        # its identification too, using the identity of this finite stalk.
        I=CoreModules.eye(F,length(c.coordinates))
        if all(canonical.point[k]<=c.point[k] for k in 1:2)
            DataIngestion._slice_source_map(F,canonical,source,I)
        else
            DataIngestion._slice_source_map(F,source,canonical,I)
        end
        cycle=DataIngestion._slice_source_cycle(prepared.source_basis,source,c.coordinates,F)
        _,last=result.classes.supports[c.id]
        if last<length(result.chain)
            target=result.classes.bases[last+1]
            # A finite window cut is not a death. Only an actual next stratum
            # whose map kills this interval justifies a filling.
            filling=DataIngestion._slice_source_filling(prepared.source_basis,target,cycle.coefficients,F)
        end
    end
    merge(c,(;source_cycle=cycle,source_filling=filling,
        source_cycle_data=cycle===nothing ? nothing : DataIngestion.source_chain_data(cycle),
        source_filling_data=filling===nothing ? nothing : DataIngestion.source_chain_data(filling),basis_choice=:noncanonical_interval_decomposition))
end

Base.@constprop :none Base.@noinline Base.@nospecializeinfer function _slice_class_correspondence(Base.@nospecialize(obj),Base.@nospecialize(prepared),Base.@nospecialize(result),Base.@nospecialize(anchor),Base.@nospecialize(t))
    (anchor===nothing || result===nothing) && return nothing
    F=_inspection_field(obj);K=CoreModules.coeff_type(F);M=Results.encoding_module(obj)
    t=t===nothing ? anchor.parameter : t
    v=_slice_class_vertex(result,t);q=result.chain[v]
    point=ntuple(k->result.line.basepoint[k]+t*result.line.direction[k],2)
    joinpoint=ntuple(k->max(anchor.point[k],point[k]),2)
    pi=_inspection_classifier(encoding_map(obj));j=_inspection_slice_locate(pi,joinpoint)
    j>0 || throw(ArgumentError("The common upper parameter is not represented by this encoding."))
    budget=prepared.class_budget.max_matrix_entries
    big(M.dims[j])*(M.dims[anchor.label]+M.dims[q])+big(M.dims[q])^2<=budget ||
        throw(ArgumentError("Slice correspondence exceeds the matrix-entry budget."))
    A=Matrix(Modules.structure_map(M;source=anchor.label,target=j))
    B=Matrix(Modules.structure_map(M;source=q,target=j))
    ids=findall(ab->ab[1]<=v<=ab[2],result.classes.supports)
    basis=zeros(K,M.dims[q],length(ids))
    for (column,id) in enumerate(ids)
        basis[:,column]=result.classes.vectors[id][v-result.classes.supports[id][1]+1]
    end
    y=A*anchor.coordinates;Y=B*basis
    kernel=FieldLinAlg.nullspace(F,Y)
    _,pivots=FieldLinAlg.rref(F,Y);columns=Int[p for p in pivots]
    has_preimage=FieldLinAlg.rank(F,hcat(Y,y))==length(columns)
    coefficients=nothing
    if has_preimage
        coefficients=zeros(K,length(ids))
        coefficients[columns]=vec(FieldLinAlg.solve_fullcolumn(F,Y[:,columns],reshape(y,:,1)))
        Y*coefficients==y || error("Slice correspondence failed its map equation.")
    end
    status=all(iszero,y) ? :zero_at_join : !has_preimage ? :no_corresponding_class :
        size(kernel,2)>0 ? :ambiguous_preimages : :unique_preimage
    member_ids=[(;interval=g,member=m,id) for (g,group) in enumerate(result.classes.groups)
        for (m,id) in enumerate(group) if id in ids]
    sort!(member_ids;by=x->findfirst(==(x.id),ids))
    source_cycle=nothing;target_cycle=nothing;filling=nothing
    if prepared.source_basis!==nothing
        old=_slice_basis_at(obj,prepared,result,anchor.point,anchor.label)
        new=_slice_basis_at(obj,prepared,result,point,q)
        upper=_slice_basis_at(obj,prepared,result,joinpoint,j)
        DataIngestion._slice_source_map(F,old,upper,A)
        DataIngestion._slice_source_map(F,new,upper,B)
        source_cycle=DataIngestion._slice_source_cycle(prepared.source_basis,old,anchor.coordinates,F)
        if coefficients!==nothing
            target_cycle=DataIngestion._slice_source_cycle(prepared.source_basis,new,basis*coefficients,F)
            filling=DataIngestion._slice_source_filling(prepared.source_basis,upper,
                source_cycle.coefficients-target_cycle.coefficients,F)
        end
    end
    (;status,parameter=t,source_point=anchor.point,target_point=point,join_point=joinpoint,
      source_label=anchor.label,target_label=q,join_label=j,source_map=A,target_map=B,
      image=y,target_images=Y,coefficients,kernel,members=member_ids,source_cycle,target_cycle,filling,
      semantics=:equal_images_at_common_upper_parameter,canonical=false)
end

Base.@constprop :none Base.@noinline Base.@nospecializeinfer function _inspection_class_snapshot(Base.@nospecialize(spec),Base.@nospecialize(obj),Base.@nospecialize(prepared),Base.@nospecialize(result),Base.@nospecialize(state))
    prepared.slice_classes || return spec
    selected=_slice_selected_class(obj,prepared,result,state)
    correspondence=_slice_class_correspondence(obj,prepared,result,state.class_anchor,state.class_parameter)
    lines=["Interval members use a chosen exact basis, not canonical class names."]
    if selected!==nothing
        push!(lines,"Member $(state.member), t=$(selected.parameter), point $(selected.point), finite label $(selected.label).")
        coordinates=join(string.(selected.coordinates[1:min(12,end)]),", ")
        push!(lines,"Stalk coordinates (first $(min(12,length(selected.coordinates))) of $(length(selected.coordinates))): [$coordinates].")
        push!(lines,prepared.source_basis===nothing ? "Source cycles were not retained; these are module coordinates." :
            "Source basis and requested structure maps are checked modulo boundaries.")
    else
        push!(lines,"Select an interval member and a parameter where it is alive.")
    end
    if correspondence!==nothing
        c=correspondence
        push!(lines,"Pinned point $(c.source_point); current point $(c.target_point); common upper point $(c.join_point).")
        push!(lines,"Correspondence: $(replace(string(c.status),'_'=>' ')).")
        if c.coefficients!==nothing
            nonzero=findall(!iszero,c.coefficients)
            terms=["$(c.coefficients[i]) * group $(c.members[i].interval), member $(c.members[i].member)" for i in nonzero[1:min(12,end)]]
            push!(lines,"One solution: "*(isempty(terms) ? "0" : join(terms,"; "))*". Kernel dimension $(size(c.kernel,2)).")
            length(nonzero)>12 && push!(lines,"Further nonzero coordinates remain in the exact result.")
        end
        push!(lines,"Equal images at the common upper point do not identify entire barcode intervals.")
    end
    panels=copy(spec.panels)
    source_panels=Tuple{String,Any}[
        ("Selected source cycle",selected===nothing ? nothing : selected.source_cycle),
        ("Filling at the interval's death",selected===nothing ? nothing : selected.source_filling)]
    if correspondence!==nothing
        c=correspondence
        # A pinned source remains visible after changing the line, even though
        # the local interval selection is cleared to avoid false identities.
        if selected===nothing || selected.point!=c.source_point || selected.coordinates!=state.class_anchor.coordinates
            push!(source_panels,("Pinned source cycle",c.source_cycle))
        end
        push!(source_panels,("One preimage on the current slice",c.target_cycle))
        c.filling===nothing || all(iszero,c.filling.coefficients) ||
            push!(source_panels,("Boundary relating the cycles at the common upper parameter",c.filling))
    end
    for (title,c) in source_panels
        (c===nothing || c.dimension>2) && continue
        panel=first(_visual_spec(c,:source_chain).panels)
        push!(panels,VisualizationSpec(panel.kind;title,subtitle=panel.subtitle,layers=panel.layers,
            axes=panel.axes,legend=panel.legend,metadata=panel.metadata))
    end
    push!(panels,_inspection_text_panel("Slice classes and source correspondence",lines))
    VisualizationSpec(spec.kind;title=spec.title,subtitle=spec.subtitle,panels,
        axes=spec.axes,legend=spec.legend,interaction=spec.interaction,
        metadata=merge(spec.metadata,(;selected_slice_class=selected,slice_correspondence=correspondence,
            panel_columns=2,panel_positions=[((1+div(i-1,2)):(1+div(i-1,2)),
                (1+mod(i-1,2)):(1+mod(i-1,2))) for i in eachindex(panels)],
            panel_row_weights=fill(450,cld(length(panels),2)),figure_size=(1800,450*cld(length(panels),2)))))
end

"""Return the selected exact interval coordinates, source cycle and available
death filling. Requires `slice_classes=true` when opening the session."""
slice_class(s::InspectionSession)=get(inspection_snapshot(s).metadata,:selected_slice_class,nothing)
"""Return the equality-of-images relation from the pinned class to the current
slice stalk. A kernel records nonunique preimages; interval IDs are local choices."""
slice_correspondence(s::InspectionSession)=get(inspection_snapshot(s).metadata,:slice_correspondence,nothing)

"""
    slice_path(source; basepoints, direction=(1,1), parameters=1:length(basepoints),
               degree=0, field=F2(), representatives=false, ...)

Follow a piecewise linear translation of a strictly positive slice through a
retained planar source complex. Each cell enters at the minimum, over its
critical grades, of `maximum((grade-basepoint)./direction)`. Insert every
change of this affine envelope before asking the vineyard engine to continue
the reduction. This also supports multicritical cells.

Native continuation uses the prime-field engine. The class inspector also
supports QQ. The fixed direction is not normalized. Direction changes instead
use the inspector's explicit common-upper-parameter relation. The returned
`PersistencePath` records chosen vineyard pairings, including ties and zero
pairs; these are not canonical homology-class identities. All paths and
reduction events are bounded. Source geometry is retained and optional
representatives are available at the inserted scalarization keyframes.
"""
function slice_path(source::DataIngestion.FilteredComplex; basepoints,direction=(1,1),
        parameters=1:length(basepoints),degree=0,field=CoreModules.F2(),representatives=false,
        max_frames=200,max_records=20000,
        budget=OrdinaryPersistence.PersistenceUpdateBudget(max_cells=2000,max_swaps=10000,max_entries=1000000))
    DataIngestion.check_filtered_complex(source;throw=true)
    G=source.complex
    DataTypes.parameter_dim(G)==2 && source.orientation==(1,1) ||
        throw(ArgumentError("Slice continuation requires a positively oriented planar source filtration."))
    length(basepoints)>0 && length(basepoints)<=max_frames || throw(ArgumentError("Provide a bounded nonempty sequence of basepoints."))
    ts=_path_parameters(parameters,length(basepoints))
    rational(x)=x isa Union{Integer,Rational,AbstractFloat} && !(x isa Bool) && isfinite(x) ?
        Rational{BigInt}(x) : throw(ArgumentError("Slice continuation requires finite rational or floating coordinates."))
    length(direction)==2 && all(>(0),direction) || throw(ArgumentError("Slice continuation needs two strictly positive direction coordinates."))
    d=Tuple(rational.(direction));ts=rational.(ts)
    bases=[Tuple(rational.(b)) for b in basepoints]
    all(b->length(b)==2,bases) || throw(ArgumentError("Each slice basepoint needs two coordinates."))
    n=sum(DataTypes.cell_counts(G));n<=budget.max_cells || throw(ArgumentError("Slice path exceeds max_cells."))
    critical_counts=[length(DataTypes.cell_grade_set(G,i)) for i in 1:n]
    sum(c->4big(c)^2,critical_counts)*max(1,length(bases)-1)<=budget.max_entries ||
        throw(ArgumentError("Slice affine-envelope comparisons exceed max_entries."))
    grades=[[Tuple(rational.(g)) for g in DataTypes.cell_grade_set(G,i)] for i in 1:n]
    all(!isempty,grades) || throw(ArgumentError("Slice continuation requires each cell to have a nonempty critical-grade set."))
    scalar(b)=[minimum(maximum((g[k]-b[k])/d[k] for k in 1:2) for g in cell) for cell in grades]
    frames=[scalar(first(bases))];times=[first(ts)]
    for segment in 1:length(bases)-1
        a,b=bases[segment],bases[segment+1]
        knots=Set([zero(Rational{BigInt}),one(Rational{BigInt})])
        for cell in grades
            # All candidate affine pieces are sufficient for both max and min
            # envelopes. Extraneous intersections preserve the exact path.
            pieces=[((g[k]-a[k])/d[k],(a[k]-b[k])/d[k]) for g in cell for k in 1:2]
            big(length(pieces))^2<=budget.max_entries || throw(ArgumentError("Slice envelope exceeds the algebra budget."))
            for i in eachindex(pieces),j in 1:i-1
                u,v=pieces[i],pieces[j];u[2]==v[2] && continue
                t=(v[1]-u[1])/(u[2]-v[2])
                0<t<1 && push!(knots,t)
            end
            length(knots)+length(frames)-1<=max_frames || throw(ArgumentError("Slice envelope exceeds max_frames."))
        end
        for t in sort!(collect(knots))[2:end]
            location=ntuple(k->a[k]+t*(b[k]-a[k]),2)
            push!(frames,scalar(location))
            push!(times,ts[segment]+t*(ts[segment+1]-ts[segment]))
        end
    end
    oneparameter=DataTypes.GradedComplex([collect(c) for c in G.cells_by_dim],copy(G.boundaries),[(v,) for v in first(frames)])
    geometry=DataIngestion.filtered_complex(oneparameter;geometry=source.geometry,
        source_ids=source.sources,faces=DataIngestion.face_incidence(source),construction="translated planar slice")
    path=persistence_path(geometry;grades=frames[2:end],parameters=times,degree,field,representatives,max_frames,max_records,budget)
    annotated=[merge(f,(;context=merge(f.context,(;scalarization=:exact_multicritical_line_entry,
        direction=d,basepoint=begin
            k=min(searchsortedlast(ts,f.parameter),length(ts)-1)
            length(ts)==1 ? first(bases) : ntuple(j->bases[k][j]+(f.parameter-ts[k])/(ts[k+1]-ts[k])*(bases[k+1][j]-bases[k][j]),2)
        end)))) for f in path.frames]
    native=merge(path.native,(;slice_scalarization=(;source=deepcopy(source),basepoints=copy(bases),direction=d,parameters=copy(ts))))
    PersistencePath(annotated,path.transitions,path.degree,path.field,path.order,path.mode,path.geometry,native,path.limits)
end
