# Semantic composition of prepared views. Rendering never infers a mathematical
# correspondence from matching colors, coordinates, or panel positions.

"""
    VisualContext(; domain, units, orientation=(1,1), field=nothing, degree=nothing,
                    provenance)

Declare the coordinate domain and provenance of a planar view. `domain` and
`provenance` are nonempty identifiers meaningful within your analysis; `units`
names both displayed units, and orientation entries are +1 or -1. Optional
`field` and `degree` name the represented coefficient field and homology degree.
These declarations are compared, not inferred from similar-looking pictures.
Attach with `visual_panel`. A declaration does not certify an encoding theorem.
"""
struct VisualContext
    domain::String
    units::NTuple{2,String}
    orientation::NTuple{2,Int}
    field::Union{Nothing,String}
    degree::Union{Nothing,Int}
    provenance::String
    function VisualContext(;domain,units,orientation=(1,1),field=nothing,degree=nothing,provenance)
        all(x->x isa AbstractString && !isempty(strip(x)),(domain,provenance)) ||
            throw(ArgumentError("domain and provenance need nonempty string identifiers."))
        length(units)==2 && all(x->x isa AbstractString && !isempty(strip(x)),units) ||
            throw(ArgumentError("units must name the two displayed coordinate units."))
        length(orientation)==2 && all(x->x isa Integer && !(x isa Bool) && x in (-1,1),orientation) ||
            throw(ArgumentError("orientation must contain two signs, +1 or -1."))
        degree===nothing || (degree isa Integer && !(degree isa Bool) && degree>=0) ||
            throw(ArgumentError("degree must be a nonnegative integer or nothing."))
        field===nothing || ((field isa AbstractString || field isa CoreModules.AbstractCoeffField) && !isempty(strip(string(field)))) ||
            throw(ArgumentError("field must be a coefficient field, nonempty field name, or nothing."))
        new(String(domain),Tuple(String.(units)),Tuple(Int.(orientation)),
            field===nothing ? nothing : string(field),degree===nothing ? nothing : Int(degree),String(provenance))
    end
end
describe(c::VisualContext)=(;domain=c.domain,units=c.units,orientation=c.orientation,field=c.field,degree=c.degree,provenance=c.provenance)
Base.show(io::IO,c::VisualContext)=print(io,"VisualContext(",repr(c.domain),", ",c.units,")")
Base.:(==)(a::VisualContext,b::VisualContext)=describe(a)==describe(b)
Base.isequal(a::VisualContext,b::VisualContext)=isequal(describe(a),describe(b))
Base.hash(c::VisualContext,h::UInt)=hash(describe(c),h)

"""
    VisualAnchor(panel; point=nothing, space=:data, target=nothing)

Locate an annotation endpoint by a stable panel name. `point=(x,y)` is in data
coordinates, or in normalized `[0,1]^2` panel/figure coordinates when `space`
is `:panel`/`:figure`. Figure anchors use panel `:figure`. Coordinates are not
changed when another panel is inserted or the figure is resized.

Instead of a point, `target` identifies retained drawing data:
`(:layer, layer_index, element_index)`, `(:overlay, name, layer, element)`,
`(:matrix, row, column)`,
`(:source_cell, dimension_local_index)`, or `(:interval, degree, kind, index)`.
Interval `kind` is `:finite` or `:essential`; grouped bars retain member IDs.
An omitted/clipped-out target is rejected, never redirected to a nearby object.
Layer targets use a point or the center of a segment/rectangle/polygon; matrix
targets follow the actual coefficient label's layout box. Only planar data
anchors are supported; project a 3D source explicitly before annotating it.
"""
struct VisualAnchor
    panel::Symbol
    space::Symbol
    point::Union{Nothing,NTuple{2,Float64}}
    target::Union{Nothing,Tuple}
    function VisualAnchor(panel::Symbol;point=nothing,space=:data,target=nothing)
        space in (:data,:panel,:figure) || throw(ArgumentError("space must be :data, :panel or :figure."))
        (point===nothing)!=(target===nothing) || throw(ArgumentError("Supply exactly one of point or target."))
        (space===:figure)==(panel===:figure) || throw(ArgumentError("Figure anchors use panel=:figure and space=:figure."))
        if target!==nothing
            space===:data || throw(ArgumentError("Object targets use space=:data."))
            target isa Tuple && !isempty(target) || throw(ArgumentError("target must be a semantic selector tuple."))
        else
            length(point)==2 && all(x->x isa Real && !(x isa Bool) && isfinite(x) && isfinite(Float64(x)),point) ||
                throw(ArgumentError("point must contain two finite displayable real coordinates."))
            space===:data || all(x->0<=x<=1,point) || throw(ArgumentError("Panel/figure coordinates must lie in [0,1]."))
        end
        new(panel,space,point===nothing ? nothing : Tuple(Float64.(point)),target)
    end
end
describe(a::VisualAnchor)=(;panel=a.panel,space=a.space,point=a.point,target=a.target)
Base.show(io::IO,a::VisualAnchor)=print(io,"VisualAnchor(",a.panel,", ",a.target===nothing ? a.point : a.target,")")

"""
    VisualAnnotation(from, to; kind=:arrow, label, curve=0.0)

An authored explanation between two `VisualAnchor`s. `kind` is `:arrow`,
`:callout`, `:bracket`, or `:highlight`. `curve` bends an arrow by a fraction
of the endpoint separation. Brackets/highlights require one panel/space.
Every annotation has a nonempty interpretation label; authored explanations
use dashed strokes and an explicit key. They assert no computed relation.
Use `correspondence_annotation` for a checked retained cycle-to-interval link.
"""
struct VisualAnnotation{E}
    from::VisualAnchor
    to::VisualAnchor
    kind::Symbol
    label::String
    curve::Float64
    evidence::E
end
function VisualAnnotation(from::VisualAnchor,to::VisualAnchor;kind=:arrow,label,curve=0.0)
    a=VisualAnnotation(from,to,kind,String(label),Float64(curve),nothing)
    _check_annotation(a)
    return a
end
function _check_annotation(a::VisualAnnotation)
    a.kind in (:arrow,:callout,:bracket,:highlight) || throw(ArgumentError("Unknown annotation kind."))
    isempty(strip(a.label)) && throw(ArgumentError("Annotations need a nonempty interpretation label."))
    isfinite(a.curve) && abs(a.curve)<=1 || throw(ArgumentError("curve must be finite and between -1 and 1."))
    a.kind in (:bracket,:highlight) && (a.from.panel!=a.to.panel || a.from.space!=a.to.space) &&
        throw(ArgumentError("Brackets and highlights must stay within one panel and coordinate space."))
    a.kind in (:bracket,:highlight) && !iszero(a.curve) && throw(ArgumentError("Only arrows/callouts may be curved."))
    return a
end
describe(a::VisualAnnotation)=(;kind=a.kind,label=a.label,from=describe(a.from),to=describe(a.to),
    meaning=a.evidence===nothing ? :authored_explanation :
        a.evidence isa NamedTuple && get(a.evidence,:mode,nothing)===:frozen_correspondence ? :captured_computed_correspondence : :computed_correspondence)
Base.show(io::IO,a::VisualAnnotation)=print(io,"VisualAnnotation(",a.kind,", ",repr(a.label),", ",describe(a).meaning,")")

function _composition_copy(s;kind=s.kind,title=s.title,subtitle=s.subtitle,layers=s.layers,
        panels=s.panels,axes=s.axes,legend=s.legend,interaction=s.interaction,metadata=s.metadata)
    VisualizationSpec(kind;title,subtitle,layers,panels,axes,legend,interaction,metadata)
end

"""
    visual_panel(object; panel=nothing, context=nothing, title=nothing, limits=nothing, kwargs...)
    visual_panel(values::AbstractMatrix; x, y, quantity, context, colormap=:Greys)

Prepare one leaf view for composition through the normal `visual_spec` recipe.
`panel` selects an existing child by its unique kind or integer position. With
no selector, a one-panel view is used directly, and a multi-panel recipe uses
its first plot. `context=VisualContext(...)` declares compatibility for overlays
and shared axes. Recipe keywords (such as `kind` or `dim`) remain explicit.
This retains exact metadata and the originating object for checked connectors;
it does not run representative selection or alter mathematical data.
`title` changes the panel heading; `limits=(x=(lo,hi),y=(lo,hi))` selects an
explicit display window without altering retained values. The matrix method
draws supplied scalar samples: rows follow `y`, columns follow `x`, which may
be centers or cell edges. `quantity` names the colorbar; `NaN` means missing.
`limits` changes only the prepared view's camera; recipe windows and budgets
still control which objects were prepared. Request a larger recipe window to
recover previously clipped glyphs or omitted objects.
"""
function visual_panel(obj;panel=nothing,context=nothing,title=nothing,limits=nothing,kwargs...)
    context===nothing || context isa VisualContext || throw(ArgumentError("context must be a VisualContext."))
    s=visual_spec(obj;kwargs...)
    s.kind===:linked_inspector && throw(ArgumentError("Compose a static inspection_snapshot, not a live widget."))
    leaf=if isempty(s.panels)
        panel===nothing || throw(ArgumentError("This view has no child panels."));s
    elseif panel===nothing
        first(s.panels)
    elseif panel isa Integer && !(panel isa Bool) && 1<=panel<=length(s.panels)
        s.panels[panel]
    elseif panel isa Symbol
        found=filter(p->p.kind===panel,s.panels)
        length(found)==1 || throw(ArgumentError("panel must identify exactly one child kind."));only(found)
    else
        throw(ArgumentError("panel must be an existing child index or unique kind."))
    end
    isempty(leaf.panels) || throw(ArgumentError("Choose a leaf panel."))
    isempty(get(leaf.interaction,:widgets,())) || throw(ArgumentError("Compose fixed views, not widgets."))
    ctx=context===nothing ? get(leaf.metadata,:composition_context,nothing) : context
    retained=get(leaf.metadata,:composition_observed,NamedTuple())
    known_field=hasproperty(obj,:field) ? string(getproperty(obj,:field)) :
        get(s.metadata,:field,get(leaf.metadata,:field,get(retained,:field,nothing)))
    known_degree=obj isa DataIngestion.SourceChain ? obj.dimension :
        get(s.metadata,:homological_dimension,get(s.metadata,:degree,
        get(leaf.metadata,:homological_dimension,get(leaf.metadata,:degree,get(retained,:degree,nothing)))))
    observed=(;field=known_field,degree=known_degree)
    if ctx!==nothing
        known_field===nothing || ctx.field===nothing || string(known_field)==ctx.field ||
            throw(ArgumentError("Declared field disagrees with the originating object."))
        known_degree===nothing || ctx.degree===nothing || known_degree==ctx.degree ||
            throw(ArgumentError("Declared degree disagrees with the originating object."))
        for meta in (s.metadata,leaf.metadata)
            known_field=get(meta,:field,nothing)
            known_field===nothing || ctx.field===nothing || string(known_field)==ctx.field ||
                throw(ArgumentError("Declared field disagrees with the prepared view."))
            known_degree=get(meta,:homological_dimension,get(meta,:degree,nothing))
            known_degree===nothing || ctx.degree===nothing || known_degree==ctx.degree ||
                throw(ArgumentError("Declared degree disagrees with the prepared view."))
        end
    end
    axes=ctx===nothing ? leaf.axes : merge(leaf.axes,(;xlabel=ctx.units[1],ylabel=ctx.units[2]))
    if limits!==nothing
        limits isa NamedTuple && keys(limits)==(:x,:y) || throw(ArgumentError("limits must be (x=(lo,hi),y=(lo,hi))."))
        all(r->length(r)==2 && all(v->v isa Real && !(v isa Bool) && isfinite(v),r) && r[1]<r[2],values(limits)) ||
            throw(ArgumentError("Display limits must be finite increasing pairs."))
        axes=merge(axes,(;xlimits=Tuple(limits.x),ylimits=Tuple(limits.y)))
    end
    return _composition_copy(leaf;axes,title=title===nothing ? leaf.title : String(title),metadata=merge(leaf.metadata,(;composition_context=ctx,composition_observed=observed,
        composition_object=obj isa VisualizationSpec ? get(obj.metadata,:composition_object,nothing) : obj)))
end

function visual_panel(values::AbstractMatrix;x,y,quantity,context::VisualContext,colormap::Symbol=:Greys,title="")
    quantity isa AbstractString && !isempty(strip(quantity)) || throw(ArgumentError("quantity must name the scalar being displayed."))
    for (axis,n) in ((x,size(values,2)),(y,size(values,1)))
        axis isa AbstractVector && length(axis) in (n,n+1) && length(axis)>=2 &&
            all(v->v isa Real && !(v isa Bool) && isfinite(v) && isfinite(Float64(v)),axis) && all(>(0),diff(axis)) ||
            throw(ArgumentError("Each scalar axis must be strictly increasing finite centers or edges, with at least two coordinates."))
        all(>(0),diff(Float64.(axis))) || throw(ArgumentError("Distinct scalar-axis coordinates coincide at drawing precision."))
    end
    all(v->v isa Real && (isfinite(v) || isnan(v)),values) || throw(ArgumentError("Scalar samples must be real, finite or NaN."))
    z=Float64.(values)
    all(i->!isfinite(values[i]) || isfinite(z[i]),eachindex(z)) || throw(ArgumentError("Scalar samples overflow display precision."))
    s=VisualizationSpec(:scalar_samples;title,subtitle="Supplied samples: $(context.provenance)",
        layers=AbstractVisualizationLayer[HeatmapLayer(Float64.(x),Float64.(y),z,colormap,1.0,String(quantity))],
        axes=_default_axes_2d(;xlabel=context.units[1],ylabel=context.units[2],xlimits=(first(x),last(x)),ylimits=(first(y),last(y))),
        metadata=(;composition_context=context,sample_axes=(copy(x),copy(y)),sample_values=copy(values),quantity=String(quantity)))
    check_visual_spec(s;throw=true)
    return s
end

function _composition_named(parts)
    parts isa NamedTuple && !isempty(parts) || throw(ArgumentError("Use a nonempty NamedTuple of stable panel/layer names."))
    :figure in keys(parts) && throw(ArgumentError(":figure is reserved for figure-space anchors."))
    for p in values(parts)
        p isa VisualizationSpec || throw(ArgumentError("Prepare each view with visual_panel or visual_spec."))
        check_visual_spec(p;throw=true)
        isempty(p.panels) || throw(ArgumentError("Select leaf views with visual_panel before composing."))
        isempty(get(p.interaction,:widgets,())) || throw(ArgumentError("Composition accepts fixed views, not widgets."))
    end
    return parts
end
function _composition_compatibility(parts;windows=true)
    differences=String[]
    firstid=first(keys(parts));base=first(values(parts));bc=get(base.metadata,:composition_context,nothing)
    bc===nothing && push!(differences,"$firstid lacks a declared VisualContext")
    for (id,p) in pairs(parts)
        c=get(p.metadata,:composition_context,nothing)
        if c===nothing
            id==firstid || push!(differences,"$id lacks a declared VisualContext")
        elseif bc!==nothing
            for k in fieldnames(VisualContext)
                isequal(getfield(c,k),getfield(bc,k)) || push!(differences,"$firstid / $id differ in $k")
            end
        end
        for key in (:field,:degree)
            expected=bc===nothing ? nothing : getfield(bc,key)
            actual=c===nothing ? nothing : getfield(c,key)
            known=get(get(base.metadata,:composition_observed,NamedTuple()),key,nothing)
            expected=known===nothing ? expected : known
            known=get(get(p.metadata,:composition_observed,NamedTuple()),key,nothing)
            actual=known===nothing ? actual : known
            isequal(expected,actual) || push!(differences,"$firstid / $id differ in observed $key")
        end
        if windows
            for k in (:xlimits,:ylimits)
                get(p.axes,k,nothing)===nothing && push!(differences,"$id needs explicit $k")
                isequal(get(p.axes,k,nothing),get(base.axes,k,nothing)) || push!(differences,"$firstid / $id differ in $k")
            end
        end
    end
    return unique(differences)
end
function _composition_comparison(differences,comparison)
    comparison===nothing || (comparison isa AbstractString && !isempty(strip(comparison))) ||
        throw(ArgumentError("comparison must be a nonempty explanation or nothing."))
    isempty(differences) || comparison!==nothing || throw(ArgumentError(
        "Incompatible views: "*join(differences,"; ")*". Supply comparison=\"reason\" for an explicitly labelled comparison."))
    return comparison===nothing ? "" : "Intentional comparison: "*String(comparison)*
        (isempty(differences) ? "" : " ("*join(differences,"; ")*")")
end
function _composition_legends(parts)
    entries=NamedTuple[];seen=Dict{String,Any}()
    for p in values(parts)
        get(p.legend,:visible,false) || continue
        raw=get(p.legend,:entries,())
        iterable=raw isa NamedTuple ? [v isa NamedTuple ? merge((;label=String(k)),v) : (;label=String(k),color=v) for (k,v) in pairs(raw)] : raw
        for e in iterable
            label=String(e.label)
            haskey(seen,label) && !isequal(seen[label],e) && throw(ArgumentError("Legend label '$label' has conflicting encodings."))
            haskey(seen,label) || (push!(entries,e);seen[label]=e)
        end
    end
    return entries
end

"""
    overlay_visuals(layers::NamedTuple; title="", comparison=nothing)

Overlay named, compatible planar leaf views in the supplied back-to-front
order. Preserve each layer's identity, exact metadata, legend and color scale.
Context, field, degree, units, orientation, provenance and explicit axis windows
must agree. `comparison="reason"` permits a visibly labelled intentional
comparison; it performs no coordinate conversion or mathematical identification.
Matrices, 3D scenes and text-only panels belong beside, not on, scalar plots.
"""
function overlay_visuals(parts::NamedTuple;title="",comparison=nothing)
    _composition_named(parts)
    for p in values(parts)
        any(l->l isa Union{MatrixLayer,Point3Layer,Segment3Layer,Polygon3Layer},p.layers) &&
            throw(ArgumentError("Overlays require planar graphical layers."))
        get(p.metadata,:panel_style,nothing)===:text_only && throw(ArgumentError("Text readout panels cannot be overlaid."))
    end
    differences=_composition_compatibility(parts)
    caption=_composition_comparison(differences,comparison)
    layers=AbstractVisualizationLayer[];ranges=UnitRange{Int}[]
    for p in values(parts)
        start=length(layers)+1;append!(layers,p.layers);push!(ranges,start:length(layers))
    end
    base=first(values(parts));entries=_composition_legends(parts)
    captions=unique(filter(!isempty,[p.subtitle for p in values(parts)]))
    isempty(caption) || push!(captions,caption)
    result=VisualizationSpec(:overlay;title,subtitle=join(captions,"\n"),layers,axes=base.axes,
        legend=_default_legend(;visible=!isempty(entries),entries),
        metadata=(;composition_context=get(base.metadata,:composition_context,nothing),
            composition_observed=get(base.metadata,:composition_observed,NamedTuple()),
            overlay_names=keys(parts),overlay_specs=values(parts),overlay_ranges=ranges,
            differences,comparison,figure_size=get(base.metadata,:figure_size,(880,640))))
    check_visual_spec(result;throw=true)
    return result
end

function _composition_layer_point(layer,i)
    i isa Integer && !(i isa Bool) && i>0 || throw(ArgumentError("Target element indices must be positive integers."))
    data=if layer isa PointLayer
        layer.points
    elseif layer isa TextLayer
        layer.positions
    elseif layer isa SegmentLayer
        i<=length(layer.segments) || throw(ArgumentError("Target element is not displayed."))
        s=layer.segments[i];return ((s[1]+s[3])/2,(s[2]+s[4])/2)
    elseif layer isa RectLayer
        i<=length(layer.rects) || throw(ArgumentError("Target element is not displayed."))
        return _midpoint(layer.rects[i])
    elseif layer isa Union{PolygonLayer,PolylineLayer}
        polys=layer isa PolygonLayer ? layer.polygons : layer.paths
        i<=length(polys) || throw(ArgumentError("Target element is not displayed."))
        poly=polys[i];isempty(poly) && throw(ArgumentError("An empty path has no anchor."))
        return Tuple(sum(p[k] for p in poly)/length(poly) for k in 1:2)
    elseif layer isa HeatmapLayer
        throw(ArgumentError("Use a data point for a heatmap anchor; flattened pixels are not stable object IDs."))
    else
        throw(ArgumentError("This layer has no planar object anchors."))
    end
    i<=length(data) || throw(ArgumentError("Target element is not displayed."))
    return data[i]
end
function _composition_anchor(panel,a::VisualAnchor)
    if a.target===nothing
        a.space===:data && (any(l->l isa MatrixLayer,panel.layers) || get(panel.metadata,:panel_style,nothing)===:text_only) &&
            throw(ArgumentError("Data-point anchors need a graphical axis. Use a matrix target or panel-space point."))
        return (;kind=a.space,point=a.point)
    end
    t=a.target
    if length(t)==3 && t[1]===:matrix
        length(panel.layers)==1 && only(panel.layers) isa MatrixLayer || throw(ArgumentError("Matrix anchor requires a matrix panel."))
        i,j=t[2:3]
        all(x->x isa Integer && !(x isa Bool) && x>0,(i,j)) && i<=size(only(panel.layers).entries,1) && j<=size(only(panel.layers).entries,2) ||
            throw(ArgumentError("Matrix coefficient is not displayed."))
        return (;kind=:matrix,entry=(Int(i),Int(j)))
    elseif length(t)==3 && t[1]===:layer
        i=t[2]
        i isa Integer && !(i isa Bool) && 1<=i<=length(panel.layers) || throw(ArgumentError("Layer index is not displayed."))
        return (;kind=:data,point=_composition_layer_point(panel.layers[i],t[3]))
    elseif length(t)==4 && t[1]===:overlay
        names=get(panel.metadata,:overlay_names,())
        t[2] in names || throw(ArgumentError("Unknown named overlay."))
        child=panel.metadata.overlay_specs[findfirst(==(t[2]),names)]
        return _composition_anchor(child,VisualAnchor(a.panel;target=(:layer,t[3],t[4])))
    elseif length(t)==2 && t[1]===:source_cell
        c=get(panel.metadata,:composition_object,nothing)
        c isa DataIngestion.SourceChain || throw(ArgumentError("Source-cell anchor requires a SourceChain panel."))
        i=t[2];i isa Integer && !(i isa Bool) && 1<=i<=length(c.coefficients) || throw(ArgumentError("Invalid source-cell index."))
        !iszero(c.coefficients[i]) || throw(ArgumentError("Source cell is outside the selected chain support."))
        global_i=sum(DataTypes.cell_counts(c.source.complex)[1:c.dimension])+i
        global_i in get(panel.metadata,:displayed_indices,()) || throw(ArgumentError("Selected source cell was omitted by the display budget."))
        targets=get(panel.metadata,:pick_targets,())
        hit=findfirst(r->r.index==global_i,targets)
        hit===nothing && throw(ArgumentError("No displayed geometry for source cell."))
        point=targets[hit].point
        length(point)==2 || throw(ArgumentError("Source annotations need an explicit planar projection."))
        return (;kind=:data,point)
    elseif length(t)==4 && t[1]===:interval
        t[2] isa Integer && !(t[2] isa Bool) && t[2]>=0 &&
            t[3] in (:finite,:essential) && t[4] isa Integer && !(t[4] isa Bool) && t[4]>0 ||
            throw(ArgumentError("Interval targets require a nonnegative degree, :finite/:essential kind and positive member index."))
        records=get(panel.metadata,:displayed_records,get(get(panel.metadata,:interval_view,NamedTuple()),:displayed_records,()))
        idx=findfirst(r->any(m->m.dim==t[2] && m.kind==t[3] && m.index==t[4],r.members),records)
        idx===nothing && throw(ArgumentError("Interval member is not displayed in this panel."))
        points=get(panel.metadata,:interval_points,nothing)
        if points!==nothing
            return (;kind=:data,point=points[idx])
        end
        segs=get(panel.metadata,:interval_segments,nothing)
        segs===nothing && throw(ArgumentError("Interval anchor requires a barcode or diagram panel."))
        s=segs[idx];return (;kind=:data,point=((s[1]+s[3])/2,(s[2]+s[4])/2))
    end
    throw(ArgumentError("Unsupported anchor target $t."))
end

"""
    correspondence_annotation(diagram, cycle; from=:source, to=:barcode,
                              source_cell=nothing, label="Selected cycle", curve=0.0)

Link an ordinary retained source cycle to its actual interval member. Requery
the representative and check source identity, field, degree, parameter and
coefficients. `cycle` must come from `persistence_representative(diagram)`;
use source/interval panels prepared from these same objects. Solid strokes
identify this checked relationship. The representative is a chosen basis
element, not a canonical hole or a multiparameter summand decomposition.
"""
function correspondence_annotation(diag::OrdinaryPersistence.PersistenceDiagram,c::DataIngestion.SourceChain;
        from=:source,to=:barcode,source_cell=nothing,label="Selected cycle",curve=0.0)
    c.kind===:cycle || throw(ArgumentError("This correspondence requires a retained cycle."))
    all(k->haskey(c.metadata,k),(:member,:interval_kind,:interval)) || throw(ArgumentError("Cycle lacks an ordinary interval identity."))
    DataIngestion.check_source_chain(c;throw=true)
    dim=c.dimension;kind=c.metadata.interval_kind;index=c.metadata.member
    actual=OrdinaryPersistence.persistence_representative(diag;dim,kind,index,scale=only(c.parameter)).source_cycle
    actual!==nothing && actual.source===c.source && actual.field==c.field && actual.coefficients==c.coefficients && actual.metadata.interval==c.metadata.interval ||
        throw(ArgumentError("Cycle does not match this diagram's retained representative."))
    i=source_cell===nothing ? findfirst(!iszero,c.coefficients) : source_cell
    i===nothing && throw(ArgumentError("A zero cycle supplies no geometric anchor."))
    a=VisualAnnotation(VisualAnchor(from;target=(:source_cell,i)),VisualAnchor(to;target=(:interval,dim,kind,index)),
        :arrow,String(label),Float64(curve),(;diagram=diag,cycle=c,dim,kind,index))
    return _check_annotation(a)
end

"""
    compose_visuals(panels::NamedTuple; annotations=(), columns=2, title="",
                   subtitle="", link_axes=(), link_colors=(), comparison=nothing)

Compose named leaf views into a reusable figure. `link_axes` and `link_colors`
are tuples of groups of panel names. Linked axes require matching declared
contexts/windows; color linking also requires a common scalar label and map.
Unknown provenance is not silently certified; a labelled `comparison` permits
an intentional comparison. Shared scales and stable legend encodings are used
without changing any value. Annotation endpoints follow data or layout boxes.

PNG/SVG/PDF retain the selected figure. HTML export of compositions uses Cairo
to embed a self-contained responsive SVG with an accessible interpretation
list: it is a fixed figure, not a saved live Julia inspector. WGL rendering
supports live pan/zoom with attached annotations while Julia is connected.
"""
function compose_visuals(parts::NamedTuple;annotations=(),columns=2,title="",subtitle="",
        link_axes=(),link_colors=(),comparison=nothing)
    _composition_named(parts)
    columns isa Integer && !(columns isa Bool) && columns>0 || throw(ArgumentError("columns must be positive."))
    panels=collect(values(parts));names=keys(parts);differences=String[]
    for groups in (link_axes,link_colors)
        groups isa Tuple || throw(ArgumentError("Linked groups must be supplied as a tuple of tuples."))
        all(g->g isa Tuple,groups) || throw(ArgumentError("Each linked group must be a tuple of panel names."))
        ids=[id for group in groups for id in group]
        allunique(ids) || throw(ArgumentError("Linked groups must be disjoint; combine overlapping groups into one group."))
    end
    for groups in (link_axes,link_colors),group in groups
        group isa Tuple && length(group)>=2 && allunique(group) && all(in(names),group) || throw(ArgumentError("Linked groups need at least two distinct existing panel names."))
        subset=NamedTuple{group}(Tuple(parts[k] for k in group))
        append!(differences,_composition_compatibility(subset))
        for p in values(subset)
            any(l->l isa Union{MatrixLayer,Point3Layer,Segment3Layer,Polygon3Layer},p.layers) &&
                throw(ArgumentError("Linked scales require planar graphical panels."))
            get(p.metadata,:panel_style,nothing)===:text_only && throw(ArgumentError("Readout panels have no shared axes."))
        end
    end
    caption=_composition_comparison(unique(differences),comparison)
    for group in link_colors
        ids=[findfirst(==(k),names) for k in group]
        hms=[filter(l->l isa HeatmapLayer,panels[i].layers) for i in ids]
        any(i->panels[i].kind===:overlay,ids) && throw(ArgumentError("Link color scales before overlaying multiple scalar fields; use plain scalar panels for linked color comparisons."))
        all(h->length(h)==1,hms) || throw(ArgumentError("Color linking requires one scalar heatmap in each selected panel."))
        hs=only.(hms)
        all(h->h.colorbar_label==hs[1].colorbar_label && h.colormap==hs[1].colormap,hs) ||
            throw(ArgumentError("Linked color scales must name the same scalar quantity and colormap."))
        vals=Float64[v for h in hs for v in h.values if isfinite(v)]
        isempty(vals) && throw(ArgumentError("Linked color scales need finite values."))
        lo,hi=extrema(vals);range=lo==hi ? (lo-0.5,hi+0.5) : (lo,hi)
        for i in ids
            p=panels[i];panels[i]=_composition_copy(p;metadata=merge(p.metadata,(;colorrange=range)))
        end
    end
    entries=_composition_legends(parts)
    all(a->a isa VisualAnnotation,annotations) || throw(ArgumentError("annotations must contain VisualAnnotation objects."))
    annotations=Tuple(annotations)
    any(a->a.evidence===nothing,annotations) && push!(entries,(;label="Authored explanation",color=_VisualRole(:muted),style=:line,linestyle=:dash))
    any(a->a.evidence!==nothing,annotations) && push!(entries,(;label="Checked representative correspondence",color=_VisualRole(:selected),style=:line))
    panels=[_composition_copy(p;legend=_default_legend()) for p in panels]
    result=VisualizationSpec(:composition;title,subtitle=join(filter(!isempty,[String(subtitle),caption]),"\n"),panels,
        legend=_default_legend(;visible=!isempty(entries),entries),
        metadata=(;panel_names=names,annotations,linked_axes=Tuple(link_axes),linked_colors=Tuple(link_colors),
            comparison,differences=unique(differences),panel_columns=min(Int(columns),length(panels)),
            figure_size=(min(Int(columns),length(panels))*580,cld(length(panels),Int(columns))*470+170),
            html_mode=:responsive_fixed_svg))
    check_visual_spec(result;throw=true)
    return result
end

function _check_composition!(issues,s)
    s.kind===:composition || return issues
    try
        names=s.metadata.panel_names
        length(names)==length(s.panels) && allunique(names) || throw(ArgumentError("Panel identities disagree with prepared panels."))
        for a in s.metadata.annotations
            _check_annotation(a)
            for endpoint in (a.from,a.to)
                endpoint.space===:figure && continue
                i=findfirst(==(endpoint.panel),names);i===nothing && throw(ArgumentError("Unknown panel $(endpoint.panel)."))
                p=s.panels[i]
                any(l->l isa Union{Point3Layer,Segment3Layer,Polygon3Layer},p.layers) && endpoint.space===:data &&
                    throw(ArgumentError("Data anchors require a planar view; use an explicit source projection."))
                _composition_anchor(p,endpoint)
            end
            if a.evidence!==nothing
                e=a.evidence
                if e isa NamedTuple && get(e,:mode,nothing)===:frozen_correspondence
                    _check_frozen_correspondence(s,a)
                    continue
                end
                e isa NamedTuple && keys(e)==(:diagram,:cycle,:dim,:kind,:index) || throw(ArgumentError("Unrecognized correspondence evidence."))
                checked=correspondence_annotation(e.diagram,e.cycle;from=a.from.panel,to=a.to.panel,
                    source_cell=a.from.target[2],label=a.label,curve=a.curve)
                isequal(describe(checked.from),describe(a.from)) && isequal(describe(checked.to),describe(a.to)) || throw(ArgumentError("Computed anchors differ from their evidence."))
                src=s.panels[findfirst(==(a.from.panel),names)];dst=s.panels[findfirst(==(a.to.panel),names)]
                get(src.metadata,:composition_object,nothing)===e.cycle && get(dst.metadata,:composition_object,nothing)===e.diagram ||
                    throw(ArgumentError("Computed connector panels must retain the checked cycle and diagram."))
            end
        end
    catch err
        err isa InterruptException && rethrow();push!(issues,sprint(showerror,err))
    end
    return issues
end
