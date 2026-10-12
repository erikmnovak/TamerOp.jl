"""
    ScalarField(axes, values; id="field", axis_labels, units="value",
        layers=nothing, provenance=(;))
    ScalarField(ambient_grid; kwargs...)

A sampled real field on declared Cartesian coordinates. Array dimensions follow
`axes` in order; with explicit `layers`, the final array dimension indexes their
stable string identities. Without `layers`, there is one scalar layer. `NaN`
means unavailable; infinite values are rejected. Inputs are copied. This is a
display adapter, not a construction of a persistence module or an interpolation
between samples. Use `visualize(field)` or `field_session(field)` to explore it.

Comparisons accept a vector of fields with distinct IDs and identical coordinates,
axis labels, units and ordered layer IDs. Supplying that vector declares the
coordinate correspondence. Unrelated finite-poset labels require `compare_visuals`.
"""
struct ScalarField{N}
    axes::NTuple{N,Vector{Float64}}
    values::Array{Float64}
    id::String
    axis_labels::NTuple{N,String}
    units::String
    layers::Vector{String}
    provenance::NamedTuple
    function ScalarField(axes,values;id="field",axis_labels=nothing,units="value",layers=nothing,provenance=(;))
        ax=Tuple(Float64.(collect(a)) for a in axes);n=length(ax)
        n>=2 && all(a->!isempty(a)&&all(isfinite,a)&&all(>(0),diff(a)),ax) ||
            throw(ArgumentError("Supply at least two finite, strictly increasing, nonempty coordinate axes."))
        labels=axis_labels===nothing ? ntuple(i->"x$i",n) : Tuple(string.(axis_labels))
        length(labels)==n && all(!isempty,labels) && length(unique(labels))==n ||
            throw(ArgumentError("axis_labels must be distinct nonempty labels, one per axis."))
        names=layers===nothing ? ["scalar"] : string.(collect(layers))
        !isempty(names)&&all(!isempty,names)&&length(unique(names))==length(names) ||
            throw(ArgumentError("layers must contain distinct nonempty identities."))
        expected=layers===nothing ? Tuple(length.(ax)) : (length.(ax)...,length(names))
        size(values)==expected || throw(ArgumentError("values shape must be $expected in coordinate-axis order."))
        data=reshape(Float64.(values),(length.(ax)...,length(names)))
        all(x->isfinite(x)||isnan(x),data) || throw(ArgumentError("Values must be finite or NaN (unavailable)."))
        !isempty(string(id))&&!isempty(string(units)) || throw(ArgumentError("id and units must be nonempty."))
        provenance isa NamedTuple || throw(ArgumentError("provenance must be a NamedTuple."))
        new{n}(ax,data,string(id),labels,string(units),names,deepcopy(provenance))
    end
end
ScalarField(g::AmbientLandscapeGrid;kwargs...) = ScalarField(landscape_grid(g),landscape_values(g);
    layers=string.(1:landscape_layers(g)),units="ambient landscape height",provenance=describe(g),kwargs...)
describe(f::ScalarField)=(;kind=:sampled_scalar_field,id=f.id,axes=deepcopy(f.axes),axis_labels=f.axis_labels,
    shape=size(f.values),units=f.units,layers=copy(f.layers),available=count(isfinite,f.values),provenance=deepcopy(f.provenance))
Base.show(io::IO,f::ScalarField)=print(io,"ScalarField(\"",f.id,"\", ",length(f.axes)," coordinates, ",length(f.layers)," layers)")
const _FieldInput=Union{ScalarField,AbstractVector{<:ScalarField}}
available_visuals(::_FieldInput)=(:coordinate_sections,)
_visual_request_keywords(::_FieldInput,::Symbol)=(:planes,:indices,:layer,:normalization,:colorrange,:contours,:baseline,:max_cells)
_field_list(f::ScalarField)=[f]
_field_list(f::AbstractVector{<:ScalarField})=collect(f)
function _field_compatible(fields)
    !isempty(fields) || throw(ArgumentError("Supply at least one field."))
    length(unique(f.id for f in fields))==length(fields) || throw(ArgumentError("Field IDs must be distinct."))
    a=first(fields)
    all(f->f.axes==a.axes&&f.axis_labels==a.axis_labels&&f.units==a.units&&f.layers==a.layers,fields) ||
        throw(ArgumentError("Compared fields require identical coordinates, axis labels, units and ordered layer IDs."))
    nothing
end
function _field_options(fields;planes=nothing,indices=nothing,layer=nothing,
        normalization=:auto,colorrange=nothing,contours=(),baseline=nothing,max_cells=250_000)
    _field_compatible(fields);f=first(fields);n=length(f.axes)
    planes===nothing && (planes=n==2 ? [(1,2)] : [(1,2),(1,3),(2,3)])
    all(p->length(p)==2&&all(i->i isa Integer&&!(i isa Bool)&&1<=i<=n,p)&&p[1]!=p[2],planes) ||
        throw(ArgumentError("Each ordered plane must select two distinct coordinate axes."))
    ps=Tuple(Tuple(Int.(p)) for p in planes)
    !isempty(ps)&&length(unique(ps))==length(ps)&&length(ps)*length(fields)<=12 || throw(ArgumentError("Select distinct planes with at most 12 total panels."))
    idx=indices===nothing ? ntuple(i->cld(length(f.axes[i]),2),n) : Tuple(indices)
    length(idx)==n&&all(i->idx[i] isa Integer&&!(idx[i] isa Bool)&&1<=idx[i]<=length(f.axes[i]),1:n) ||
        throw(ArgumentError("indices must select one stored coordinate on every axis."))
    lid=layer===nothing ? first(f.layers) : string(layer)
    lid in f.layers || throw(ArgumentError("Unknown layer identity $lid."))
    normalization in (:auto,:sequential,:symmetric) || throw(ArgumentError("normalization must be :auto, :sequential or :symmetric."))
    baseline===nothing || string(baseline) in [f.id for f in fields] || throw(ArgumentError("baseline must be a supplied field ID."))
    cs=Float64.(collect(contours));all(isfinite,cs)&&issorted(cs)&&length(unique(cs))==length(cs)&&length(cs)<=12 ||
        throw(ArgumentError("contours must be at most 12 increasing finite levels."))
    max_cells isa Integer&&!(max_cells isa Bool)&&max_cells>0 || throw(ArgumentError("max_cells must be a positive integer."))
    cells=sum(BigInt(length(f.axes[p[1]]))*length(f.axes[p[2]]) for p in ps)*length(fields)
    cells*max(1,length(cs))<=max_cells || throw(ArgumentError("Selected planes exceed max_cells (including contour levels)."))
    (;planes=ps,indices=Tuple(Int.(idx)),layer=lid,normalization,colorrange=_image_colorrange(colorrange),
        contours=Tuple(cs),baseline=baseline===nothing ? nothing : string(baseline),max_cells=Int(max_cells))
end
function _field_scale(fields,o)
    baseline=o.baseline===nothing ? nothing : only(filter(f->f.id==o.baseline,fields))
    lo=Inf;hi=-Inf
    for f in fields,I in eachindex(f.values)
        v=baseline===nothing ? f.values[I] : f.values[I]-baseline.values[I]
        isinf(v) && throw(ArgumentError("Differences overflowed display precision."))
        isfinite(v) || continue
        lo=min(lo,v);hi=max(hi,v)
    end
    norm=o.normalization===:auto ? (lo<0||baseline!==nothing ? :symmetric : :sequential) : o.normalization
    norm===:sequential&&lo<0 && throw(ArgumentError("Signed values require symmetric normalization."))
    limits=if o.colorrange!==nothing
        o.colorrange
    elseif norm===:symmetric
        r=isfinite(lo) ? max(abs(lo),abs(hi)) : 1.0;r==0&&(r=1.0);(-r,r)
    elseif !isfinite(lo)
        (0.,1.)
    elseif lo==hi
        pad=max(1.,abs(lo)*0.05);(lo-pad,hi+pad)
    else
        (lo,hi)
    end
    norm===:symmetric&&limits[1]!=-limits[2] && throw(ArgumentError("Signed color limits must be symmetric about zero."))
    all(isfinite,limits)&&limits[1]<limits[2] || throw(ArgumentError("Color limits overflow display precision."))
    (;limits,normalization=norm,colormap=norm===:symmetric ? :balance : :viridis)
end
# Contours of a stated piecewise-affine interpolant, split along the SW-NE
# diagonal. NaN triangles are omitted; equality uses one half-open edge rule.
function _field_contours(x,y,z,level)
    segments=NTuple{4,Float64}[]
    for j in 1:length(y)-1,i in 1:length(x)-1
        points=((x[i],y[j]),(x[i+1],y[j]),(x[i+1],y[j+1]),(x[i],y[j+1]))
        heights=(z[j,i],z[j,i+1],z[j+1,i+1],z[j+1,i])
        for tri in ((1,2,3),(1,3,4))
            all(q->isfinite(heights[q]),tri)||continue
            hits=NTuple{2,Float64}[]
            for (a,b) in ((tri[1],tri[2]),(tri[2],tri[3]),(tri[3],tri[1]))
                (heights[a]<=level<heights[b]||heights[b]<=level<heights[a])||continue
                w=(level-heights[a])/(heights[b]-heights[a])
                push!(hits,((1-w)*points[a][1]+w*points[b][1],(1-w)*points[a][2]+w*points[b][2]))
            end
            length(hits)==2&&hits[1]!=hits[2]&&push!(segments,(hits[1]...,hits[2]...))
        end
    end
    segments
end
_field_contour_color(level) = (:black,:darkorange,:dodgerblue,:purple,:forestgreen,:brown)[
    mod1(Int(first(sha1(string(iszero(level) ? 0.0 : level)))),6)]
_field_contour_style(level) = (:solid,:dash,:dot)[
    mod1(Int(sha1(string(iszero(level) ? 0.0 : level))[2]),3)]
function _field_snapshot(fields,o,scale)
    f=first(fields);n=length(f.axes);k=findfirst(==(o.layer),f.layers)
    base=o.baseline===nothing ? nothing : only(filter(f->f.id==o.baseline,fields))
    panels=VisualizationSpec[];rows=NamedTuple[]
    for field in fields,plane in o.planes
        x=field.axes[plane[1]];y=field.axes[plane[2]]
        index(i,j)=ntuple(d->d==plane[1] ? i : d==plane[2] ? j : o.indices[d],n)
        vals=[field.values[index(i,j)...,k] for j in eachindex(y),i in eachindex(x)]
        base===nothing || (vals .-= [base.values[index(i,j)...,k] for j in eachindex(y),i in eachindex(x)])
        all(v->isfinite(v)||isnan(v),vals) || throw(ArgumentError("Differences overflowed display precision."))
        fixed=Tuple((;axis=d,label=f.axis_labels[d],index=o.indices[d],value=f.axes[d][o.indices[d]]) for d in 1:n if !(d in plane))
        layers=AbstractVisualizationLayer[HeatmapLayer(x,y,vals,scale.colormap,1.,f.units)]
        for (q,level) in enumerate(o.contours)
            push!(layers,SegmentLayer(_field_contours(x,y,vals,level),_field_contour_color(level),1.,1.6,_field_contour_style(level)))
        end
        point=(x[o.indices[plane[1]]],y[o.indices[plane[2]]])
        push!(layers,PointLayer([point],:white,1.,13.));push!(layers,PointLayer([point],:black,1.,6.))
        value=vals[o.indices[plane[2]],o.indices[plane[1]]]
        title=base===nothing ? field.id : "$(field.id) - $(base.id)"
        fixed_text=isempty(fixed) ? "no fixed coordinates" : join(("$(a.label)=$(a.value)" for a in fixed),", ")
        subtitle="$(f.axis_labels[plane[1]]) x $(f.axis_labels[plane[2]]); $fixed_text\nlayer $(o.layer) ($(k)/$(length(f.layers))); selected=$(isnan(value) ? "unavailable" : string(value))"
        !isempty(o.contours)&&(subtitle*="; interpolated contours")
        if get(field.provenance,:kind,nothing)===:ambient_landscape_grid
            windows=unique(field.provenance.source_windows)
            window_label=length(windows)==1 ? string(only(windows)) : "vary by line"
            subtitle*="\n$(replace(string(field.provenance.interpretation),'_'=>' ')); source t-window $window_label"
        end
        limits(a)=length(a)>1 ? (first(a),last(a)) : (prevfloat(a[1]-0.5),nextfloat(a[1]+0.5))
        push!(panels,VisualizationSpec(:coordinate_section;title,subtitle,layers,
            axes=_default_axes_2d(xlabel=f.axis_labels[plane[1]],ylabel=f.axis_labels[plane[2]],xlimits=limits(x),ylimits=limits(y),aspect=:auto),
            legend=_default_legend(visible=!isempty(o.contours),title="Interpolated levels",
                entries=[(;label=string(level),color=_field_contour_color(level),style=:line,linestyle=_field_contour_style(level)) for level in o.contours]),
            metadata=(;field_id=field.id,plane,fixed,layer=o.layer,layer_index=k,total_layers=length(f.layers),
                selected_point=point,selected_value=value,colorrange=scale.limits,normalization=scale.normalization,
                coordinate_convention=:declared_cartesian_samples,contour_interpolation=:affine_SW_NE_triangles,
                contour_levels=o.contours,available=count(isfinite,vals),total=length(vals),provenance=field.provenance)))
        push!(rows,(;id=field.id,plane,value))
    end
    columns=length(o.planes)
    VisualizationSpec(:coordinate_sections;title="Coordinate sections",subtitle="Shared $(scale.normalization) scale over all supplied samples and layers; blank cells are unavailable",
        panels,metadata=(;selection=o,scale,selected_values=rows,field_ids=[f.id for f in fields],
            panel_columns=columns,figure_size=(480columns,430length(fields)+70),
            fixed_coordinate_contract=:shared_selected_grid_point,correspondence=:identical_declared_axes_and_layers))
end
function _visual_spec(input::_FieldInput,kind::Symbol;kwargs...)
    fields=_field_list(input);o=_field_options(fields;kwargs...)
    _field_snapshot(fields,o,_field_scale(fields,o))
end
