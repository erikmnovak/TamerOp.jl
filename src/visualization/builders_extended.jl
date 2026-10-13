# Extended interval types stay separate from ordinary/essential bar displays.
available_visuals(::OrdinaryPersistence.ExtendedPersistenceDiagram)=(:extended_persistence,:extended_barcode)
_visual_request_keywords(::OrdinaryPersistence.ExtendedPersistenceDiagram,::Symbol)=(:dim,:max_intervals)

function _append_visual_request_issues!(issues::Vector{String},d::OrdinaryPersistence.ExtendedPersistenceDiagram,::Symbol;
        dim=0,max_intervals=200,kwargs...)
    append!(issues,OrdinaryPersistence.check_extended_persistence(d).issues)
    dim isa Integer && !(dim isa Bool) && 0<=dim<typemax(Int) || push!(issues,"dim must be a nonnegative integer")
    max_intervals isa Integer && !(max_intervals isa Bool) && 0<max_intervals<=typemax(Int) || push!(issues,"max_intervals must be positive")
    return issues
end

function _visual_spec(d::OrdinaryPersistence.ExtendedPersistenceDiagram,kind::Symbol;dim::Integer=0,max_intervals::Integer=200)
    rows=OrdinaryPersistence.extended_intervals(d;dim)
    length(rows)<=max_intervals || throw(ArgumentError("$(length(rows)) extended members exceed max_intervals=$max_intervals; request a larger explicit budget"))
    original=sort!(unique!([r.birth for r in rows]));append!(original,[r.death for r in rows]);sort!(unique!(original))
    coords=Float64.(original)
    all(isfinite,coords) && all(diff(coords).>0) || throw(ArgumentError("extended endpoints exceed or collide in Float64 display coordinates"))
    lo,hi=isempty(coords) ? (0.0,1.0) : extrema(coords)
    margin=max((hi-lo)/10,abs(lo)/100,1e-3)
    limits=(lo-margin,hi+margin)
    all(isfinite,limits) && limits[1]<limits[2] || throw(ArgumentError("extended endpoints leave no finite display margin"))
    meta=(;source=:extended_persistence,dimension=Int(dim),intervals=OrdinaryPersistence.describe.(rows),
        provenance=OrdinaryPersistence.provenance(d),rounding=:float64_display_only,
        rounded_endpoint_count=count(i->coords[i]!=original[i],eachindex(coords)),
        endpoint_convention=:birth_included_death_excluded_on_tagged_copies)
    if kind===:extended_barcode
        n=length(d.filtration.values)
        layers=AbstractVisualizationLayer[]
        for (i,r) in enumerate(rows)
            color=_VisualRole(r.kind===:ordinary ? :source : r.kind===:relative ? :target : :selected)
            push!(layers,SegmentLayer([(Float64(r.birth_step),Float64(i),Float64(r.death_step),Float64(i))],color,0.85,2.0))
            push!(layers,PointLayer([(Float64(r.birth_step),Float64(i))],color,1.0,7.0))
            # A hollow endpoint is a separate white overpaint, shared by both renderers.
            push!(layers,PointLayer([(Float64(r.death_step),Float64(i))],color,1.0,7.0))
            push!(layers,PointLayer([(Float64(r.death_step),Float64(i))],_VisualRole(:background),1.0,4.0))
        end
        push!(layers,SegmentLayer([(n+0.5,0.0,n+0.5,Float64(length(rows)+1))],_VisualRole(:muted),0.7,1.0))
        steps=OrdinaryPersistence.extended_steps(d)
        # Bound tick density while preserving the complete exact event table.
        selected=length(steps)<=12 ? eachindex(steps) : unique(round.(Int,range(1,length(steps);length=12)))
        ticks=(Float64[steps[i].step for i in selected],
            ["$(steps[i].phase===:ordinary ? "O" : "R"):$(steps[i].value)" for i in selected])
        return VisualizationSpec(:extended_barcode;title="Extended barcode (degree $dim)",
            subtitle="Ordinary then relative; event spacing is ordinal, not scalar distance. Filled birth, open death.",layers,
            axes=_default_axes_2d(;xlabel="tagged parameter event",ylabel="interval member",xticks=ticks,
                xlimits=(0.5,max(1.5,2n+0.5)),ylimits=(0.0,Float64(length(rows)+1)),aspect=:auto),
            metadata=merge(meta,(;coordinate_space=:ordinal_two_copy_order,steps)))
    end
    panels=VisualizationSpec[]
    for (family,title) in ((:ordinary,"Ordinary"),(:relative,"Relative"),(:extended,"Extended"))
        members=OrdinaryPersistence.extended_intervals(d;dim,kind=family)
        layers=AbstractVisualizationLayer[SegmentLayer([(lo,lo,hi,hi)],_VisualRole(:muted),0.5,1.0)]
        entries=NamedTuple[]
        for (k,role) in ((:ordinary,:source),(:relative,:target),(:extended_positive,:source),(:extended_negative,:target),(:extended_zero,:selected))
            pts=[(Float64(r.birth),Float64(r.death)) for r in members if r.kind===k]
            isempty(pts) || push!(layers,PointLayer(pts,_VisualRole(role),0.85,10.0))
            if !isempty(pts) && family===:extended
                label=k===:extended_positive ? "Positive" : k===:extended_negative ? "Negative" : "Equal scalar endpoints"
                push!(entries,(;label,color=_VisualRole(role),style=:marker))
            end
            counts=Dict(p=>count(==(p),pts) for p in unique(pts))
            repeated=sort!([p for (p,n) in counts if n>1])
            isempty(repeated) || push!(layers,TextLayer(["x$(counts[p])" for p in repeated],
                [(p[1]+margin/4,p[2]+margin/4) for p in repeated],_VisualRole(:foreground),12.0))
        end
        push!(panels,VisualizationSpec(:extended_component;title="$title ($(length(members)))",
            subtitle=family===:ordinary ? "Both endpoints on the ordinary copy" : family===:relative ?
                "Both endpoints on the relative copy" : "Birth ordinary; death relative; equality retained",
            layers,axes=_default_axes_2d(;xlabel="birth value",ylabel="death value",xlimits=limits,ylimits=limits,aspect=:data),
            legend=_default_legend(;visible=!isempty(entries),entries),
            metadata=merge(meta,(;intervals=OrdinaryPersistence.describe.(members),interval_family=family,coordinate_space=:original_scalar_values))))
    end
    return VisualizationSpec(:extended_persistence;title="Extended persistence (degree $dim)",
        subtitle="$(OrdinaryPersistence.filtration_order(d)); $(_inspection_field_label(OrdinaryPersistence.field(d))). Separate parameter copies, original scalar values.",
        panels,metadata=merge(meta,(;panel_columns=3,figure_size=(1500,560))))
end
