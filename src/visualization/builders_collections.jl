# Bounded supplied tables, stable sample identities, and loss-aware drawing data.

"""
    VisualCollection

Prepared, labelled tables made by `visual_collection`. Rows always follow
`sample_ids`; feature columns follow `feature_names`. No normalization, metric,
kernel, embedding, or all-pairs computation is performed. Use `describe` for
semantics, `collection_sample` for retained sample material, and `visual_spec`
for exact selected values alongside floating drawing data. Inputs, including
retained sample objects and views, must be treated as read-only.
"""
struct VisualCollection{T,E,S,W,C}
    tables::T
    ids::Vector{String}
    names::Vector{String}
    definitions::Vector{String}
    embedding::E
    samples::S
    witnesses::W
    context::C
end

_collection_description(x,name)=x isa AbstractString && !isempty(strip(x)) ? String(x) :
    throw(ArgumentError("$name must be a nonempty description."))
function _collection_labels(xs,n,name)
    xs isa Union{AbstractVector,Tuple} && length(xs)==n || throw(ArgumentError("$name must contain $n labels."))
    labels=[_collection_description(x,name) for x in xs]
    length(unique(labels))==n || throw(ArgumentError("$name must be unique."))
    labels
end
function _collection_table(A,n,kind,budget)
    A===nothing && return nothing
    A isa AbstractMatrix && size(A,1)==n && size(A,2)>0 || throw(ArgumentError("$kind must be a nonempty matrix with one row per sample."))
    length(A)<=budget || throw(ArgumentError("$kind exceeds max_cells; subset the collection explicitly."))
    kind===:features || size(A,2)==n || throw(ArgumentError("$kind must be square in sample order."))
    for x in A
        ismissing(x) && continue
        x isa Real && !(x isa Bool) && !isnan(x) || throw(ArgumentError("$kind accepts real values or missing, not NaN/Bool."))
        (isfinite(x) || kind===:distances && x==Inf) || throw(ArgumentError("Only distances may contain positive infinity."))
        kind===:distances && x<0 && throw(ArgumentError("Distances must be nonnegative."))
        isfinite(x) && !isfinite(Float64(x)) && throw(ArgumentError("A finite value exceeds the floating display range; rescale explicitly."))
    end
    if kind!==:features
        all(isequal(A[i,j],A[j,i]) for i in 1:n for j in 1:i) || throw(ArgumentError("$kind must be exactly symmetric, including missing entries; no silent symmetrization is performed."))
        kind===:distances && !all(i->!ismissing(A[i,i]) && iszero(A[i,i]),1:n) && throw(ArgumentError("Distance diagonals must be zero."))
    end
    copy(A)
end

"""
    visual_collection(; sample_ids, features=nothing, feature_names=nothing,
        definitions=nothing, distances=nothing, kernel=nothing,
        distance_semantics=nothing, kernel_semantics=nothing,
        normalization="not declared; viewer applies none", invariant_settings=(;), degree=nothing,
        field="not declared", embedding=nothing, samples=nothing,
        witnesses=Dict(), max_samples=200, max_cells=40_000)
    visual_collection(features::Featurizers.FeatureSet; definitions, kwargs...)

Explore supplied features (sample-by-feature), distances, and/or a kernel
(sample-by-sample). IDs and feature names are unique strings. Feature definitions
are one nonempty mathematical description per column. The FeatureSet overload
retains its IDs, names and mathematical interchange metadata (without caches). `distance_semantics` and `kernel_semantics`
are required descriptions when those tables are supplied. Symmetric nonnegative
zero-diagonal distances may contain `Inf`; symmetry is checked exactly, but
triangle inequalities and kernel positive semidefiniteness are NOT certified.
`missing` is explicitly masked, never zero-filled. Finite values must fit the
floating display range; exact supplied values remain in selection metadata.

`normalization`, `invariant_settings`, `degree`, and `field` document the inputs;
they do not transform or certify them. No clustering or reordering is implicit.
All tables must use the same sample order. Budgets reject, never truncate.

Optional `embedding=(coordinates=Y, method="...", settings=(;...), seed=...)`
supplies an n-by-2 matrix in that order. Record `seed=nothing` for deterministic
methods. Coordinates are a display, not distance-preservation evidence.
No dimensionality-reduction dependency is loaded or algorithm run.

`samples` is a vector of NamedTuples in sample order. Each may retain `object`
(for explicit pair matching), static `invariant` and `source` VisualizationSpecs,
and a `metadata` NamedTuple (for sample-specific degree/field/settings).
These views are displayed on sample selection without recomputation. Their
relationship to the tables is the caller's declaration, not a certificate.
`witnesses` maps ordered pairs of sample-ID strings to existing A85 matching
witnesses; both directions are distinct. See `collection_matching` to open one
or explicitly compute just the selected pair. No callback runs on hover.
"""
function visual_collection(;sample_ids,features=nothing,feature_names=nothing,definitions=nothing,
        distances=nothing,kernel=nothing,distance_semantics=nothing,kernel_semantics=nothing,
        normalization="not declared; viewer applies none",invariant_settings=NamedTuple(),degree=nothing,field="not declared",
        embedding=nothing,samples=nothing,witnesses=Dict(),max_samples=200,max_cells=40_000)
    max_samples isa Int && max_samples>0 && max_cells isa Int && max_cells>0 || throw(ArgumentError("Budgets must be positive Ints."))
    sample_ids isa Union{Tuple,AbstractVector} || throw(ArgumentError("sample_ids must be a finite vector or tuple."))
    n=length(sample_ids);0<n<=max_samples || throw(ArgumentError("Collection size must be in 1:max_samples."))
    ids=_collection_labels(sample_ids,n,:sample_ids)
    # Budget before allocation, including the total across supplied tables.
    tables=(;features,distances,kernel)
    all(A->A===nothing || A isa AbstractMatrix, tables) || throw(ArgumentError("Supply matrices, not computed callbacks."))
    sum(A===nothing ? 0 : length(A) for A in tables)<=max_cells || throw(ArgumentError("Total table cells exceed max_cells."))
    any(!isnothing,tables) || throw(ArgumentError("Supply at least one feature, distance or kernel table."))
    tables=map((A,k)->_collection_table(A,n,k,max_cells),tables,(;features=:features,distances=:distances,kernel=:kernel))
    p=features===nothing ? 0 : size(features,2)
    names=p==0 ? String[] : _collection_labels(feature_names,p,:feature_names)
    defs=if p==0
        feature_names===nothing && definitions===nothing || throw(ArgumentError("Feature names/definitions require features."))
        String[]
    else
        definitions isa Union{AbstractVector,Tuple} && length(definitions)==p || throw(ArgumentError("Supply one mathematical definition per feature."))
        [_collection_description(x,:definitions) for x in definitions]
    end
    ds=distances===nothing ? nothing : _collection_description(distance_semantics,:distance_semantics)
    ks=kernel===nothing ? nothing : _collection_description(kernel_semantics,:kernel_semantics)
    invariant_settings isa NamedTuple || throw(ArgumentError("invariant_settings must be a NamedTuple."))
    degree===nothing || degree isa Int && degree>=0 || throw(ArgumentError("degree must be nonnegative or nothing."))
    context=(;normalization=_collection_description(normalization,:normalization),invariant_settings,
        degree,field=_collection_description(field,:field),distance_semantics=ds,kernel_semantics=ks)
    emb=if embedding===nothing;nothing
    else
        embedding isa NamedTuple && Set(keys(embedding))==Set((:coordinates,:method,:settings,:seed)) ||
            throw(ArgumentError("Embedding requires coordinates, method, settings, seed."))
        Y=embedding.coordinates
        Y isa AbstractMatrix && size(Y)==(n,2) && all(x->x isa Real && !(x isa Bool) && isfinite(x) && isfinite(Float64(x)),Y) ||
            throw(ArgumentError("Embedding coordinates must be finite n-by-2 real values."))
        embedding.settings isa NamedTuple || throw(ArgumentError("Embedding settings must be a NamedTuple."))
        embedding.seed===nothing || embedding.seed isa Integer && !(embedding.seed isa Bool) || throw(ArgumentError("Embedding seed must be an integer or nothing."))
        (;coordinates=copy(Y),method=_collection_description(embedding.method,:method),settings=embedding.settings,seed=embedding.seed)
    end
    items=if samples===nothing
        fill(NamedTuple(),n)
    else
        samples isa Union{AbstractVector,Tuple} && length(samples)==n || throw(ArgumentError("samples must follow sample_ids."))
        for item in samples
            item isa NamedTuple && all(k->k in (:object,:invariant,:source,:metadata),keys(item)) || throw(ArgumentError("Unknown sample fields."))
            get(item,:metadata,NamedTuple()) isa NamedTuple || throw(ArgumentError("Sample metadata must be a NamedTuple."))
            for key in (:invariant,:source)
                panel=get(item,key,nothing);panel===nothing && continue
                panel isa VisualizationSpec && panel.kind!==:linked_inspector || throw(ArgumentError("Sample views must be static VisualizationSpecs."))
                check_visual_spec(panel;throw=true)
            end
        end
        collect(samples)
    end
    witnesses isa AbstractDict || throw(ArgumentError("witnesses must be a dictionary keyed by ordered ID pairs."))
    length(witnesses)<=max_cells || throw(ArgumentError("Too many supplied witnesses."))
    for (key,w) in witnesses
        key isa Tuple && length(key)==2 && all(id->id in ids,key) && key[1]!=key[2] || throw(ArgumentError("Witness keys must be distinct ordered sample-ID pairs."))
        w isa _MatchingVisualInput || throw(ArgumentError("Supply an existing A85 matching witness."))
        check_visual_spec(visual_spec(w);throw=true)
    end
    VisualCollection(tables,ids,names,defs,emb,items,copy(witnesses),context)
end
function visual_collection(fs::Featurizers.FeatureSet;normalization=nothing,invariant_settings=NamedTuple(),kwargs...)
    Featurizers.check_feature_set(fs;throw=true)
    invariant_settings isa NamedTuple || throw(ArgumentError("invariant_settings must be a NamedTuple."))
    haskey(invariant_settings,:feature_set) && throw(ArgumentError("feature_set is reserved for recorded FeatureSet metadata."))
    meta=Featurizers.metadata(fs)
    recorded=meta isa NamedTuple ? get(meta,:normalization,nothing) :
        meta isa AbstractDict ? get(meta,:normalization,get(meta,"normalization",nothing)) : nothing
    normalization===nothing && (normalization=recorded===nothing ? "not declared; viewer applies none" : string(recorded))
    # The owner's interchange metadata keeps mathematical settings and reference
    # identities without retaining its session cache or reference-bank objects.
    settings=merge((;feature_set=Featurizers.feature_metadata(fs;git_commit="not recorded")),invariant_settings)
    visual_collection(;features=Featurizers.feature_matrix(fs),sample_ids=Featurizers.sample_ids(fs),
        feature_names=string.(Featurizers.feature_names(fs)),invariant_settings=settings,normalization,kwargs...)
end
function describe(c::VisualCollection)
    (;kind=:visual_collection,sample_ids=Tuple(c.ids),feature_names=Tuple(c.names),definitions=Tuple(c.definitions),
        tables=Tuple(k for k in keys(c.tables) if c.tables[k]!==nothing),context=c.context,
        embedding=c.embedding===nothing ? nothing : Base.structdiff(c.embedding,(;coordinates=c.embedding.coordinates)),
        metric_certified=false,kernel_psd_certified=false,embedding_isometric=false,all_pairs_computed=false)
end
Base.show(io::IO,c::VisualCollection)=print(io,"VisualCollection(",length(c.ids)," samples, ",length(c.names)," features)")
function _collection_index(ids,x,name)
    i=x isa Int ? x : x isa AbstractString ? findfirst(==(x),ids) : nothing
    i!==nothing && 1<=i<=length(ids) || throw(ArgumentError("$name must be an existing label or an index in 1:$(length(ids))."))
    i
end
"""
    collection_sample(collection; sample)
    collection_sample(session)

Return `(id, material)`: the stable ID and supplied object, views and metadata
for one sample (the selected sample for a live collection session). No query
or representative computation runs. A closed session has released its inputs.
"""
function collection_sample(c::VisualCollection;sample)
    i=_collection_index(c.ids,sample,:sample)
    (;id=c.ids[i],material=c.samples[i])
end
available_visuals(::VisualCollection)=(:collection,)
_visual_request_keywords(::VisualCollection,::Symbol)=(:view,:sample,:target,:feature)
_visual_request_cost(::VisualCollection,::Symbol)=(;work=:supplied_tables,all_pairs=false,embedding=:supplied,timing=:not_measured)
function _collection_options(c;view=first(k for k in keys(c.tables) if c.tables[k]!==nothing),sample=1,target=min(2,length(c.ids)),feature=isempty(c.names) ? nothing : 1)
    view in keys(c.tables) && c.tables[view]!==nothing || throw(ArgumentError("view must name a supplied table."))
    i=_collection_index(c.ids,sample,:sample);j=_collection_index(c.ids,target,:target)
    f=isempty(c.names) ? (feature===nothing ? nothing : throw(ArgumentError("No features supplied."))) : _collection_index(c.names,feature,:feature)
    (;view,sample=i,target=j,feature=f)
end
function _append_visual_request_issues!(issues::Vector{String},c::VisualCollection,::Symbol;kwargs...)
    _collection_options(c;kwargs...)
    issues
end
_collection_value(x)=ismissing(x) ? "missing" : isinf(x) ? "+Inf" : string(x)
function _visual_spec(c::VisualCollection,::Symbol;kwargs...)
    _collection_spec(c,_collection_options(c;kwargs...))
end
function _collection_spec(c,o)
    A=c.tables[o.view];n,p=size(A);column=o.view===:features ? o.feature : o.target
    z=Float64[ismissing(x) || !isfinite(x) ? NaN : Float64(x) for x in A]
    finite=filter(isfinite,vec(z));lo,hi=isempty(finite) ? (0.,1.) : extrema(finite)
    if lo==hi
        pad=max(abs(lo)*0.05,0.5)
        lo-=pad;hi+=pad
    end
    isfinite(hi-lo) || throw(ArgumentError("The displayed value range overflows; rescale the supplied table explicitly."))
    symmetric=o.view===:kernel || lo<0
    range=symmetric ? (-max(abs(lo),abs(hi)),max(abs(lo),abs(hi))) : (lo,hi)
    columns=o.view===:features ? c.names : c.ids
    axes=merge(_default_axes_2d(xlabel=o.view===:features ? "Feature" : "Target sample",ylabel="Sample",
        xlimits=(0.5,p+0.5),ylimits=(0.5,n+0.5),aspect=:auto),
        (;xticks=(Float64.(1:max(1,cld(p,16)):p),columns[1:max(1,cld(p,16)):p]),
          yticks=(Float64.(1:max(1,cld(n,20)):n),c.ids[1:max(1,cld(n,20)):n])))
    layers=AbstractVisualizationLayer[HeatmapLayer(collect(0.5:p+0.5),collect(0.5:n+0.5),z,symmetric ? :RdBu : :viridis,1.,String(o.view))]
    labels=String[];positions=NTuple{2,Float64}[]
    for i in 1:n,j in 1:p
        if length(A)<=64
            push!(labels,_collection_value(A[i,j]));push!(positions,(Float64(j),Float64(i)))
        end
    end
    isempty(labels) || push!(layers,TextLayer(labels,positions,_VisualRole(:foreground),14.))
    rect=[(column-0.48,o.sample-0.48,column+0.48,o.sample+0.48)]
    push!(layers,RectLayer(rect,_VisualRole(:background),_VisualRole(:background),0.,6.))
    push!(layers,RectLayer(rect,_VisualRole(:background),_VisualRole(:selected),0.,3.))
    targets=[(;point=(Float64(j),Float64(i)),sample=i,target=o.view===:features ? o.target : j,
        feature=o.view===:features ? j : o.feature,label="$(c.ids[i]) / $(columns[j]): $(_collection_value(A[i,j]))") for i in 1:n,j in 1:p]
    semantics=o.view===:features ? "Supplied values; no column scaling" : c.context[Symbol(o.view===:distances ? :distance_semantics : :kernel_semantics)]
    matrix=VisualizationSpec(:collection_matrix;title=uppercasefirst(String(o.view)),subtitle=semantics,layers,axes,
        metadata=(;colorrange=range,values=A,pick_targets=vec(targets),minimal_axes=true,text_halo=true))
    panels=VisualizationSpec[matrix];layout=[(1:1,1:1)]
    if c.embedding!==nothing
        Y=c.embedding.coordinates;pts=[(Float64(Y[i,1]),Float64(Y[i,2])) for i in 1:n]
        margin(v)=begin a,b=extrema(v);pad=max((b-a)*0.08,0.1);(a-pad,b+pad);end
        ls=AbstractVisualizationLayer[PointLayer(pts,_VisualRole(:foreground),0.65,12.),
            PointLayer([pts[o.target]],_VisualRole(:target),1.,18.),PointLayer([pts[o.sample]],_VisualRole(:selected),1.,20.)]
        if n<=30
            unique_points=unique(pts)
            labels=[join(c.ids[findall(==(point),pts)]," / ") for point in unique_points]
            offset=max((maximum(last,pts)-minimum(last,pts))*0.025,0.04)
            push!(ls,TextLayer(labels,[(x,y+offset) for (x,y) in unique_points],_VisualRole(:foreground),14.))
        end
        push!(panels,VisualizationSpec(:collection_embedding;title=c.embedding.method,
            subtitle="Supplied embedding; proximity does not certify input distance",
            legend=_default_legend(visible=true,entries=(;
                selected=(;label="Selected: $(c.ids[o.sample])",color=_VisualRole(:selected),style=:marker),
                target=(;label="Target: $(c.ids[o.target])",color=_VisualRole(:target),style=:marker))),
            layers=ls,axes=_default_axes_2d(xlabel="Coordinate 1",ylabel="Coordinate 2",xlimits=margin(first.(pts)),ylimits=margin(last.(pts))),
            metadata=(;coordinates=Y,embedding=c.embedding,pick_targets=[(;point=pts[i],sample=i,target=o.target,feature=o.feature,label=c.ids[i]) for i in 1:n])))
        push!(layout,(1:1,2:2))
    end
    width=c.embedding===nothing ? 1 : 2
    selected=(;sample=c.ids[o.sample],target=c.ids[o.target],feature=o.feature===nothing ? nothing : c.names[o.feature],
        value=A[o.sample,column],definition=o.feature===nothing ? nothing : c.definitions[o.feature],
        sample_metadata=get(c.samples[o.sample],:metadata,NamedTuple()),
        target_metadata=get(c.samples[o.target],:metadata,NamedTuple()))
    lines=["Selected $(selected.sample) / $(o.view===:features ? selected.feature : selected.target): $(_collection_value(selected.value))",
        "Normalization (declared): $(c.context.normalization)","Degree: $(c.context.degree); field: $(c.context.field)",
        ]
    settings=c.context.invariant_settings
    if haskey(settings,:feature_set) && settings.feature_set isa AbstractDict
        recorded=settings.feature_set
        for key in ("spec","opts","feature_axes")
            value=get(recorded,key,nothing)
            value===nothing || push!(lines,"Recorded $key: $(sprint(show,value;context=:limit=>true))")
        end
        settings=Base.structdiff(settings,(;feature_set=recorded))
    end
    isempty(settings) || push!(lines,"Invariant settings: $(sprint(show,settings;context=:limit=>true))")
    selected.definition===nothing || push!(lines,"Feature definition: $(selected.definition)")
    for (id,meta) in ((selected.sample,selected.sample_metadata),(selected.target,selected.target_metadata))
        isempty(meta) || push!(lines,"$id: $(sprint(show,meta;context=:limit=>true))")
    end
    push!(lines,o.view===:features ? "Colors use supplied units; feature columns are not standardized. Missing values remain unavailable." :
        o.view===:distances ? "Symmetry is checked; triangle inequalities are not certified. Missing and infinite distances remain distinct." :
        "Symmetry is checked; positive semidefiniteness is not certified. Similarity is not converted to distance.")
    c.embedding===nothing || push!(lines,"Embedding settings: $(c.embedding.settings); seed: $(c.embedding.seed). The display need not preserve distances.")
    push!(panels,_inspection_text_panel("Read the selected value",lines));push!(layout,(2:2,1:width))
    row=3
    # Flatten prepared compositions for the same static/live panel contract.
    function append_sample(panel)
        if !isempty(panel.panels)
            foreach(append_sample,panel.panels)
        else
            push!(panels,_composition_copy(panel;title="$(selected.sample): $(panel.title)"))
            push!(layout,(row:row,1:width));row+=1
        end
    end
    for key in (:invariant,:source)
        panel=get(c.samples[o.sample],key,nothing)
        panel===nothing || append_sample(panel)
    end
    text_rows=sum(max(1,cld(length(line),width==1 ? 95 : 135)) for line in lines)
    weights=vcat([430.,max(220.,26text_rows+60.)],fill(350.,row-3))
    VisualizationSpec(:collection;title="Explore a collection",panels,
        metadata=(;selection=o,selected,context=c.context,sample_ids=Tuple(c.ids),feature_names=Tuple(c.names),
            embedding=c.embedding,matching_available=haskey(c.witnesses,(selected.sample,selected.target)),
            panel_positions=layout,panel_columns=width,panel_row_weights=weights,figure_size=(width==1 ? 950 : 1350,ceil(Int,sum(weights)+100))))
end

"""
    collection_matching(collection; sample, target, compute=false, kwargs...)
    collection_matching(session; compute=false, kwargs...)

Open an independent A85 matching session for the ordered selected pair. By
default use only its supplied witness. With `compute=true`, explicitly run A85
on the two retained sample `object`s, forwarding mathematical options (`dim`
for ordinary diagrams, `opts` for encoded pairs). Only this pair is computed;
no result is installed into the collection's supplied distance/kernel table.
The matching can have different semantics from that table. Inspect its context
before comparing numbers. Close the returned session independently.
"""
function collection_matching(c::VisualCollection;sample,target,compute=false,kwargs...)
    i=_collection_index(c.ids,sample,:sample);j=_collection_index(c.ids,target,:target)
    i!=j || throw(ArgumentError("Select two distinct samples."))
    compute isa Bool || throw(ArgumentError("compute must be Bool."))
    if !compute
        key=(c.ids[i],c.ids[j]);haskey(c.witnesses,key) || throw(ArgumentError("No supplied witness for this ordered pair. Set compute=true with explicit A85 options to compute it."))
        return inspection_session(c.witnesses[key];kwargs...)
    end
    a=get(c.samples[i],:object,nothing);b=get(c.samples[j],:object,nothing)
    a===nothing || b===nothing ? throw(ArgumentError("Both samples need retained matching-compatible objects.")) : inspection_session(a,b;kwargs...)
end
