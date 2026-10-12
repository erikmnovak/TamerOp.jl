# Compare observable values on declared corresponding probes. Equal labels,
# dimensions, or colors never manufacture maps between different inputs.

const _ComparisonObject = Union{Modules.PModule,EncodingResult,OrdinaryPersistence.PersistenceDiagram}

"""
    ComparisonCase(object; degree=nothing, filtration="not declared",
                   construction="not declared", approximation="not declared",
                   units="parameter")

One finite module, encoded module, or ordinary persistence diagram in a visual
comparison. The coefficient field and recorded provenance come from the object.
`degree` selects an ordinary diagram's homology degree (default zero); for a
module it documents the degree, if any, and must agree with recorded provenance.
The other strings are the author's modeling context, retained separately from
recorded provenance: they neither certify an approximation nor override it.
Supply informative descriptions when comparing constructions or discretizations.
Use `compare_visuals` to declare corresponding query locations.
"""
struct ComparisonCase{T}
    object::T
    degree::Union{Nothing,Int}
    context::NamedTuple
end
function ComparisonCase(object::_ComparisonObject;degree=nothing,filtration="not declared",
        construction="not declared",approximation="not declared",units="parameter")
    degree===nothing || (degree isa Int && degree>=0) || throw(ArgumentError("degree must be a nonnegative Int or nothing."))
    for (key,value) in pairs((;filtration,construction,approximation,units))
        value isa AbstractString && !isempty(strip(value)) || throw(ArgumentError("$key must be a nonempty description."))
    end
    native=object isa EncodingResult ? Results.provenance(object) :
        object isa OrdinaryPersistence.PersistenceDiagram ? OrdinaryPersistence.provenance(object) : NamedTuple()
    # Ordinary diagrams contain several degrees; their provenance describes the
    # convention rather than fixing the degree selected by this case.
    known=object isa OrdinaryPersistence.PersistenceDiagram ? nothing : get(native,:degree,nothing)
    known===nothing || degree===nothing || known==degree || throw(ArgumentError("degree disagrees with recorded provenance."))
    d=object isa OrdinaryPersistence.PersistenceDiagram ? something(degree,0) : (degree===nothing ? known : degree)
    ComparisonCase(object,d,(;filtration=String(filtration),construction=String(construction),
        approximation=String(approximation),units=String(units)))
end
_comparison_field(c::ComparisonCase)=c.object isa OrdinaryPersistence.PersistenceDiagram ?
    c.object.field : _inspection_field(c.object)
function describe(c::ComparisonCase)
    obj=c.object
    native=obj isa EncodingResult ? Results.provenance(obj) :
        obj isa OrdinaryPersistence.PersistenceDiagram ? OrdinaryPersistence.provenance(obj) : NamedTuple()
    orientation=if obj isa EncodingResult && _inspection_has_geometry(obj)
        pi=_inspection_classifier(encoding_map(obj))
        _inspection_parameter_orientation(pi)
    elseif obj isa OrdinaryPersistence.PersistenceDiagram
        obj.order===:sublevel ? (1,) : (-1,)
    else;nothing;end
    (;kind=:comparison_case,degree=c.degree,field=_comparison_field(c),declared=c.context,
        recorded=native,query_orientation=orientation,
        category=obj isa OrdinaryPersistence.PersistenceDiagram ? :ordinary_persistence : :finite_poset_representations)
end
Base.show(io::IO,c::ComparisonCase)=print(io,"ComparisonCase(",_inspection_field_label(_comparison_field(c)),", degree=",c.degree,")")

"""
    VisualComparison

Named results and explicit corresponding probes, constructed by `compare_visuals`.
`visualize(c)` starts with dimensions; `view=:rank` requests one anchored rank
row, and `view=:barcode` shows ordinary interval records with a common window.
`inspection_session(c)` links probe, anchor, baseline, and view selections.
Use `describe(c)` for the contract and `visual_spec(c).metadata` for the plotted
integer values, masks, exact probe locations, and recorded/declared provenance.
Inputs and prepared data must remain read-only; rebuild after mutation.
"""
struct VisualComparison{C,P,D}
    cases::C
    probes::P
    prepared::D
    domain::Symbol
    correspondence::String
    window::Any
    max_intervals::Int
end

"""
    compare_visuals(cases; vertices=nothing, points=nothing, parameters=nothing,
                   correspondence, window=nothing, max_queries=4096,
                   max_intervals=200)

Prepare two to eight named `ComparisonCase`s (bare supported objects use the
case defaults). Exactly one probe declaration is required:

* `vertices=(left=[...],right=[...])`: explicitly aligned finite-poset labels.
  Equal integers in different cases do not imply a correspondence.
* `points=[(x,y),...]`: shared original coordinates in supported planar encodings;
  a NamedTuple supplies case-specific corresponding coordinates, for example
  an explicitly declared discretization map.
* `parameters=[t,...]`: shared original-unit parameters for ordinary diagrams.

The nonempty `correspondence` explains what aligns these probes. It is an
authored declaration, not an isomorphism or approximation certificate.
Case names in a NamedTuple must match exactly. Probes need not be uniformly
spaced or ordered; charts use labelled probe indices, never an interpolated
ambient field. Units must agree for shared coordinates and barcodes.

Dimensions are prepared without structure maps. Rank views request only the
selected source-to-probe row; unrepresented and unordered queries are masked,
not set to zero. For a lazy encoding, the first nontrivial map materializes its
finite module through `encoding_module`; no all-pairs rank table is built.
Differences subtract the selected baseline's integer
observable, never coefficient matrices or matched bars. Barcode colors and
member IDs are local to each case; no class matching is inferred.

`window` is a common finite ordinary-barcode display interval in original
units. It does not change query coverage. Restricted diagrams return unknown
outside their recorded cutoff. Budgets reject excessive probe work before
queries; the interval budget only limits drawing, not dimension/rank sums.
"""
function compare_visuals(cases::NamedTuple;vertices=nothing,points=nothing,parameters=nothing,
        correspondence,window=nothing,max_queries=4096,max_intervals=200)
    2<=length(cases)<=8 || throw(ArgumentError("Compare two to eight named cases."))
    correspondence isa AbstractString && !isempty(strip(correspondence)) ||
        throw(ArgumentError("Explain the declared correspondence in a nonempty string."))
    max_queries isa Int && max_queries>0 && max_intervals isa Int && max_intervals>0 ||
        throw(ArgumentError("Query and interval budgets must be positive Ints."))
    count(!isnothing,(vertices,points,parameters))==1 || throw(ArgumentError("Declare exactly one of vertices, points or parameters."))
    cs=map(x->x isa ComparisonCase ? x : ComparisonCase(x),cases)
    domain=vertices!==nothing ? :finite : points!==nothing ? :parameter : :ordinary
    supplied=domain===:finite ? vertices : domain===:parameter ? points : parameters
    domain===:finite && !(supplied isa NamedTuple) &&
        throw(ArgumentError("Finite labels require a NamedTuple explicitly aligning each case."))
    supplied isa NamedTuple && keys(supplied)!=keys(cs) &&
        throw(ArgumentError("Probe case names and order must match the comparison cases."))
    domain===:ordinary && supplied isa NamedTuple &&
        throw(ArgumentError("Ordinary barcodes require shared original-unit parameters."))
    groups=supplied isa NamedTuple ? values(supplied) : (supplied,)
    all(p->p isa Union{AbstractVector,Tuple},groups) ||
        throw(ArgumentError("Supply finite vectors or tuples of probes."))
    n=length(first(groups))
    n>0 && all(p->length(p)==n,groups) || throw(ArgumentError("Each case needs the same nonzero number of probes."))
    n<=div(max_queries,length(cs)) || throw(ArgumentError("The case-by-probe query budget is exceeded."))
    probes=supplied isa NamedTuple ? map(collect,supplied) : map(_->collect(supplied),cs)
    shared=!(supplied isa NamedTuple)
    shared && !all(c->c.context.units==first(cs).context.units,cs) &&
        throw(ArgumentError("Shared coordinates require the same declared units. Convert inputs explicitly first."))
    if window!==nothing
        domain===:ordinary || throw(ArgumentError("window is only a barcode display window; module queries use explicit probes."))
        (window isa Tuple || window isa AbstractVector) && length(window)==2 &&
            all(x->x isa Real && !(x isa Bool) && isfinite(x),window) && window[1]<window[2] ||
            throw(ArgumentError("window must be a finite increasing pair."))
        window=Tuple(window)
    end
    prepared=map(cs,probes) do c,qs
        obj=c.object
        if domain===:ordinary
            obj isa OrdinaryPersistence.PersistenceDiagram || throw(ArgumentError("parameters requires ordinary diagrams in every case."))
            OrdinaryPersistence.check_persistence_diagram(obj;throw=true)
            all(_comparison_scalar,qs) || throw(ArgumentError("Parameters must be finite real values."))
            payload=_interval_payload(obj;dim=c.degree)
            dims=Union{Nothing,Int}[_comparison_covered(payload,q) ? _comparison_bar_rank(payload,q,q) : nothing for q in qs]
            return (;dims,labels=nothing,payload)
        end
        obj isa Union{Modules.PModule,EncodingResult} || throw(ArgumentError("Module probes require modules or encodings in every case."))
        ds=_inspection_dimensions(obj)
        labels=if domain===:finite
            all(q->q isa Int && 1<=q<=length(ds),qs) || throw(ArgumentError("Finite probes must be valid Int vertex labels."))
            copy(qs)
        else
            obj isa EncodingResult && _inspection_has_geometry(obj) || throw(ArgumentError("Point probes require a supported planar encoding."))
            all(q->(q isa Tuple || q isa AbstractVector) && length(q)==2 && all(_comparison_scalar,q),qs) ||
                throw(ArgumentError("Point probes must have two finite real coordinates."))
            pi=_inspection_classifier(encoding_map(obj))
            Int[_visual_locate(pi,_visual_point(q)) for q in qs]
        end
        dims=Union{Nothing,Int}[q==0 ? nothing : ds[q] for q in labels]
        (;dims,labels,payload=nothing)
    end
    if domain===:ordinary && window===nothing
        vals=vcat(first(probes),[x for p in prepared for r in p.payload.records for x in (r.birth,r.death) if isfinite(x)])
        lo,hi=extrema(vals);lo==hi && ((lo,hi)=(lo-1,hi+1));window=(lo,hi)
    end
    c=VisualComparison(cs,probes,prepared,domain,String(correspondence),window,max_intervals)
    # Check drawing ranges without constructing any matrices or ranks.
    domain===:ordinary && foreach(p->_interval_view(p.payload.records;window,max_intervals,order=p.payload.order),prepared)
    c
end
_comparison_scalar(x)=x isa Real && !(x isa Bool) && isfinite(x)
function _comparison_coverage(payload)
    window=get(payload.metadata,:restriction_window,nothing)
    window===nothing && return (-Inf,Inf)
    if window isa Tuple && length(window)==2
        all(x->x isa Real && !isnan(x),window) && first(window)<=last(window) || return nothing
        return window
    end
    cutoff=window isa NamedTuple ? get(window,:threshold,get(window,:radius,nothing)) : nothing
    return cutoff isa Real && isfinite(cutoff) ? (-Inf,cutoff) : nothing
end
function _comparison_covered(payload,t)
    bounds=_comparison_coverage(payload)
    bounds===nothing && return false # Do not invent coverage for an unrecognized restriction.
    return first(bounds)<=t<=last(bounds)
end
function _comparison_display_records(payload)
    get(payload.metadata,:restriction_window,nothing)===nothing && return payload.records
    bounds=_comparison_coverage(payload)
    bounds===nothing && return ()
    records=NamedTuple[]
    for r in payload.records
        b,d=max(r.birth,first(bounds)),min(r.death,last(bounds))
        b<=d || continue
        left,right=b>r.birth,d<r.death
        lc,rc=left || r.left_closed,right || r.right_closed
        b==d && !(lc && rc) && continue
        clipped=_interval_record(b,d,r.multiplicity;left_closed=lc,right_closed=rc,
            left_status=left ? :censored : r.left_status,
            right_status=right ? :censored : r.right_status,members=r.members)
        push!(records,merge((;id=r.id),clipped))
    end
    return Tuple(records)
end
function _comparison_bar_rank(payload,p,q)
    inside(r,t)=(r.birth<t || (r.birth==t && r.left_closed)) &&
        (t<r.death || (t==r.death && r.right_closed))
    sum((r.multiplicity for r in payload.records if inside(r,p) && inside(r,q));init=0)
end
function describe(c::VisualComparison)
    (;kind=:visual_comparison,cases=keys(c.cases),domain=c.domain,probes=length(first(c.probes)),
        correspondence=c.correspondence,correspondence_status=:declared_not_certified,
        context=map(describe,c.cases),window=c.window,max_intervals=c.max_intervals,
        difference=:integer_observable_minus_baseline,bar_identity=:local_to_case)
end
Base.show(io::IO,c::VisualComparison)=print(io,"VisualComparison(",join(keys(c.cases),", "),"; ",length(first(c.probes))," probes)")
available_visuals(::VisualComparison)=(:comparison,)
_visual_request_keywords(::VisualComparison,::Symbol)=(:view,:source,:probe,:baseline,:matrix,:matrix_limit)
_visual_request_cost(::VisualComparison,::Symbol)=(;default=:prepared_dimensions,rank=:one_anchor_row_per_case,
    all_pairs=false,lazy_module_materialization=:on_first_nontrivial_map,
    matrices=:selected_pair_opt_in,timing=:not_measured)
function _comparison_options(c;view=:hilbert,source=1,probe=1,baseline=first(keys(c.cases)),matrix=false,matrix_limit=(8,8))
    view in (:hilbert,:rank,:barcode) || throw(ArgumentError("view must be :hilbert, :rank or :barcode."))
    view===:barcode && c.domain!==:ordinary && throw(ArgumentError("Use ordinary diagrams for barcode comparison; restrict encoded modules through the slice API first."))
    n=length(first(c.probes))
    all(i->i isa Int && 1<=i<=n,(source,probe)) || throw(ArgumentError("source and probe are aligned query indices in 1:$n."))
    baseline in keys(c.cases) || throw(ArgumentError("baseline must name a case."))
    matrix isa Bool || throw(ArgumentError("matrix must be Bool."))
    matrix && (view!==:rank || c.domain===:ordinary) && throw(ArgumentError("Matrix inspection needs a module rank comparison."))
    matrix_limit isa Tuple && length(matrix_limit)==2 && all(i->i isa Int && i>0,matrix_limit) ||
        throw(ArgumentError("matrix_limit must contain positive row/column budgets."))
    (;view,source,probe,baseline,matrix,matrix_limit)
end
function _append_visual_request_issues!(issues::Vector{String},c::VisualComparison,::Symbol;kwargs...)
    try;_comparison_options(c;kwargs...)
    catch err;err isa InterruptException && rethrow();push!(issues,sprint(showerror,err));end
    issues
end

function _comparison_rank(c::ComparisonCase,prepared,probes,domain,i,j;retain=false)
    p,q=probes[i],probes[j]
    missing=(;value=nothing,status=:unrepresented,matrix=nothing,sensitivity=nothing)
    if domain===:ordinary
        data=prepared.payload
        _comparison_covered(data,p) && _comparison_covered(data,q) || return missing
        ordered=data.order===:sublevel ? p<=q : q<=p
        return ordered ? (;value=_comparison_bar_rank(data,p,q),status=:defined,matrix=nothing,sensitivity=nothing) :
            merge(missing,(;status=:unordered))
    end
    a,b=prepared.labels[i],prepared.labels[j]
    (a==0 || b==0) && return missing
    obj=c.object;P=_inspection_poset(obj)
    ordered=leq(P,a,b)
    if domain===:parameter
        pi=_inspection_classifier(encoding_map(obj))
        ordered &= _inspection_parameter_relation(pi,_visual_point(p),_visual_point(q)) in (:equal,:comparable)
    end
    ordered || return merge(missing,(;status=:unordered))
    field=_comparison_field(c)
    # Identity/zero dimensions answer the cheap paths without matrix requests.
    if !retain && (a==b || prepared.dims[i]==0 || prepared.dims[j]==0)
        return (;value=a==b ? prepared.dims[i] : 0,status=:defined,matrix=nothing,sensitivity=nothing)
    end
    M=obj isa EncodingResult ? Results.encoding_module(obj) : obj
    A=Modules.structure_map(M;source=a,target=b)
    # Retaining the identity's matrix must not change the structural rank
    # already reported by the dimension-only path.
    value=a==b ? prepared.dims[i] : isempty(A) ? 0 : FieldLinAlg.rank(field,A)
    sensitivity=if field isa CoreModules.RealField && a!=b && !isempty(A)
        T=CoreModules.coeff_type(field)
        ranks=Tuple(FieldLinAlg.rank(CoreModules.RealField(T;atol=T(factor)*field.atol,rtol=T(factor)*field.rtol),A) for factor in (0.5,2))
        (;atol=field.atol,rtol=field.rtol,factors=(0.5,2.0),ranks,
            sensitive=any(!=(value),ranks),meaning=:tolerance_sensitivity_not_error_bound)
    else;nothing;end
    (;value,status=:defined,matrix=retain ? copy(A) : nothing,sensitivity)
end

function _comparison_values(c,o)
    map(c.cases,c.prepared,c.probes) do case,p,qs
        if o.view!==:rank
            [(;value=v,status=v===nothing ? :unrepresented : :defined,matrix=nothing,sensitivity=nothing) for v in p.dims]
        else
            [_comparison_rank(case,p,qs,c.domain,o.source,j) for j in eachindex(qs)]
        end
    end
end

function _comparison_strip(name,values,range,o;difference=false,baseline="",context="")
    n=length(values);z=reshape(Float64[v.value===nothing ? NaN : v.value for v in values],1,n)
    quantity=o.view===:rank ? "rank" : "dimension"
    label=difference ? "difference" : quantity
    layers=AbstractVisualizationLayer[HeatmapLayer(collect(0.5:1:n+0.5),[0.,1.],z,difference ? :RdBu : :viridis,1.0,label)]
    # Text preserves exact small integers, signed meaning and masks in grayscale.
    n<=40 && push!(layers,TextLayer([v.value===nothing ? (v.status===:unordered ? "no map" : "?") :
        (difference && v.value>0 ? "+" : "")*string(v.value) for v in values],
        [(Float64(i),0.5) for i in 1:n],_VisualRole(:foreground),16.0))
    selection=[(o.probe-0.47,0.05,o.probe+0.47,0.95)]
    push!(layers,RectLayer(selection,_VisualRole(:background),_VisualRole(:background),0.0,6.0))
    push!(layers,RectLayer(selection,_VisualRole(:background),_VisualRole(:selected),0.0,3.0))
    title=difference ? "$name - $baseline" : String(name)
    subtitle=difference ? "$quantity minus $baseline; ? means unavailable" : context
    ticks=n<=16 ? (Float64.(1:n),string.(1:n)) : nothing
    axes=merge(_default_axes_2d(xlabel="Corresponding probe",ylabel="",xlimits=(0.5,n+0.5),ylimits=(0.,1.),aspect=:auto),
        (;yticks=(Float64[],String[])))
    ticks===nothing || (axes=merge(axes,(;xticks=ticks)))
    targets=[(;point=(Float64(i),0.5),probe=i,
        label="$title, probe $i: "*(values[i].value===nothing ?
            (values[i].status===:unordered ? "unordered; no forward map" : "unrepresented; value unknown") :
            string(values[i].value))) for i in 1:n]
    VisualizationSpec(:comparison_values;title,subtitle,layers,axes,
        metadata=(;colorrange=range,values,pick_targets=targets,
            minimal_axes=true,text_halo=true))
end
function _comparison_case_caption(case)
    d=case.degree===nothing ? "ungraded" : "degree $(case.degree)"
    "$(_inspection_field_label(_comparison_field(case))); $d\n$(case.context.filtration) | $(case.context.construction)"
end
function _visual_spec(c::VisualComparison,::Symbol;kwargs...)
    o=_comparison_options(c;kwargs...)
    _comparison_spec(c,o,_comparison_values(c,o))
end
function _comparison_spec(c,o,data)
    base=data[o.baseline]
    differences=map(data) do row
        [merge(v,(;value=v.value===nothing || b.value===nothing ? nothing : v.value-b.value,
            status=v.value===nothing ? v.status : b.value===nothing ? b.status : :defined)) for (v,b) in zip(row,base)]
    end
    vals=[v.value for row in data for v in row if v.value!==nothing]
    deltas=[abs(v.value) for row in differences for v in row if v.value!==nothing]
    common=(0.,Float64(max(1,isempty(vals) ? 0 : maximum(vals))))
    bound=Float64(max(1,isempty(deltas) ? 0 : maximum(deltas)))
    panels=VisualizationSpec[];positions=Tuple{UnitRange{Int},UnitRange{Int}}[]
    bars=if o.view===:barcode
        map(c.prepared) do prepared
            payload=prepared.payload
            first(_interval_panels(_comparison_display_records(payload);window=c.window,max_intervals=c.max_intervals,
                order=payload.order,endpoint_semantics=payload.endpoint_semantics,essential_status=payload.essential_status,
                metadata=merge(payload.metadata,(;stored_records=payload.records))))
        end
    else;nothing;end
    # Infinity lanes occupy real drawing space even when only one case has an
    # essential interval. A shared window alone would leave unequal axis scales.
    bar_limits=bars===nothing ? nothing :
        (minimum(p->p.axes.xlimits[1],bars),maximum(p->p.axes.xlimits[2],bars))
    for (row,name) in enumerate(keys(c.cases))
        case=c.cases[name]
        if o.view===:barcode
            bar=bars[name]
            # One neutral color prevents equal local IDs implying shared classes.
            ls=map(bar.layers) do l
                l isa SegmentLayer ? SegmentLayer(l.segments,_VisualRole(:foreground),l.alpha,l.linewidth,l.linestyle) :
                l isa PointLayer && l.color!==_VisualRole(:background) ? PointLayer(l.points,_VisualRole(:foreground),l.alpha,l.markersize) :
                l isa TextLayer ? TextLayer(l.labels,l.positions,_VisualRole(:foreground),l.textsize) : l
            end
            x=Float64(c.probes[name][o.probe])
            first(c.window)<=x<=last(c.window) && push!(ls,SegmentLayer([(x,bar.axes.ylimits[1],x,bar.axes.ylimits[2])],_VisualRole(:selected),0.8,2.,:dash))
            panel=_composition_copy(bar;title=String(name),subtitle=_comparison_case_caption(case)*"\n"*bar.subtitle,
                axes=merge(bar.axes,(;xlimits=bar_limits)),layers=AbstractVisualizationLayer[ls...])
            push!(panels,panel)
        else
            push!(panels,_comparison_strip(name,data[name],common,o;context=_comparison_case_caption(case)))
        end
        push!(positions,(row:row,1:1))
        push!(panels,_comparison_strip(name,differences[name],(-bound,bound),o;difference=true,baseline=String(o.baseline)))
        push!(positions,(row:row,2:2))
    end
    lines=String["Correspondence (declared): "*c.correspondence,
        "Probe indices align the supplied queries; they do not identify bases or classes.",
        "Selected probe $(o.probe)"*(o.view===:rank ? "; anchored source $(o.source)" : "")]
    selected=map(c.cases,c.prepared,c.probes,data) do case,p,qs,values
        o.view!==:rank ? nothing : o.matrix ? _comparison_rank(case,p,qs,c.domain,o.source,o.probe;retain=true) : values[o.probe]
    end
    matrix_count=0
    for name in keys(c.cases)
        case=c.cases[name];r=data[name][o.probe];s=selected[name]
        push!(lines,"$name: $(c.probes[name][o.probe]); $(r.status), value $(r.value), difference $(differences[name][o.probe].value)")
        push!(lines,"  approximation: $(case.context.approximation); units: $(case.context.units)")
        context=describe(case);native=context.recorded
        context.query_orientation===nothing || push!(lines,"  parameter orientation: $(context.query_orientation)")
        # Print the recorded modeling choices, not internal poset/storage data.
        for key in (:construction,:orientation,:window,:discretization,:approximation)
            value=get(native,key,nothing)
            value in (:not_recorded,:not_applicable) && continue
            key===:construction && value isa NamedTuple && all(==(:not_recorded),value) && continue
            value===nothing || push!(lines,"  recorded $key: $(sprint(show,value;context=:limit=>true))")
        end
        if c.domain===:ordinary
            payload=c.prepared[name].payload
            push!(lines,"  order: $(payload.order); display window: $(c.window)")
            cutoff=get(payload.metadata,:restriction_window,nothing)
            cutoff===nothing || push!(lines,"  query coverage: $cutoff; terminal bars survive only to the recorded cutoff")
        end
        numerical=_comparison_field(case) isa CoreModules.RealField
        numerical && push!(lines,"  numerical dimensions/ranks depend on tolerances; no exact-rank certificate.")
        r.sensitivity===nothing || push!(lines,"  ranks at half/double tolerance: $(r.sensitivity.ranks); sensitive=$(r.sensitivity.sensitive)")
        if o.matrix && s.matrix!==nothing
            A=s.matrix;rs=1:min(size(A,1),o.matrix_limit[1]);cols=1:min(size(A,2),o.matrix_limit[2])
            panel=VisualizationSpec(:comparison_matrix;title="$name: selected structure map",
                subtitle="Stored basis; $(_inspection_field_label(_comparison_field(case))); rank $(s.value). Matrices are not subtracted.",
                layers=AbstractVisualizationLayer[MatrixLayer([_matrix_coefficient_text(A[i,j]) for i in rs,j in cols],string.(rs),string.(cols))],
                metadata=(;panel_style=:matrix,matrix_size=size(A),matrix=A,truncated=size(A)!=(length(rs),length(cols))))
            matrix_count+=1
            row=length(c.cases)+cld(matrix_count,2);column=mod1(matrix_count,2)
            push!(panels,panel);push!(positions,(row:row,column:column))
        end
    end
    push!(panels,_inspection_text_panel("Interpret the comparison",lines))
    row=maximum(last(p[1]) for p in positions)+1;push!(positions,(row:row,1:2))
    # Reserve actual readout space; text-only panels do not constrain Makie's
    # neighboring axes. Matrix panels share a row instead of wasting a column.
    weights=vcat(fill(210.0,length(c.cases)),fill(230.0,cld(matrix_count,2)),[max(200.0,26length(lines)+70.0)])
    meaning=o.view===:rank ? "Rank from one selected source" : o.view===:barcode ? "Barcodes and sampled dimension differences" : "Dimensions at corresponding probes"
    VisualizationSpec(:comparison;title=meaning,
        subtitle="Shared scales; differences relative to $(o.baseline). ? is unknown; no map is unordered. Bar identities remain local.",
        panels,metadata=(;selection=o,values=data,differences,selected,context=map(describe,c.cases),
            probes=c.probes,correspondence=c.correspondence,correspondence_status=:declared_not_certified,
            window=c.window,domain=c.domain,colorrange=common,difference_range=(-bound,bound),panel_positions=positions,
            panel_columns=2,panel_row_weights=weights,figure_size=(1250,ceil(Int,sum(weights)+150))))
end
