# Candidate support geometry retains exact corners; no source splitting is implied.
available_visuals(::IntervalApproximations.IntervalApproximation)=(:interval_approximation,)
_visual_request_keywords(::IntervalApproximations.IntervalApproximation,::Symbol)=(:selected,:slices)
function _approximation_selection(r;selected=nothing,slices=false)
    slices isa Bool || throw(ArgumentError("slices must be Bool."))
    ids=selected===nothing ? collect(eachindex(r.rectangles)) : collect(selected)
    all(i->i isa Integer && !(i isa Bool) && 1<=i<=length(r.rectangles),ids) && allunique(ids) ||
        throw(ArgumentError("selected must contain distinct support indices."))
    (;selected=Int.(ids),slices)
end
function _append_visual_request_issues!(issues::Vector{String},r::IntervalApproximations.IntervalApproximation,::Symbol;kwargs...)
    try;_approximation_selection(r;kwargs...)
    catch err;err isa InterruptException && rethrow();push!(issues,sprint(showerror,err));end
    issues
end
function _visual_spec(r::IntervalApproximations.IntervalApproximation,::Symbol;kwargs...)
    IntervalApproximations.check_interval_approximation(r;throw=true)
    o=_approximation_selection(r;kwargs...)
    layers=AbstractVisualizationLayer[]
    # A disjoint Cartesian subdivision avoids alpha-darkening where boxes in
    # the same support overlap. Only true outer edges receive boundary marks.
    for j in o.selected
        boxes=r.rectangles[j];isempty(boxes) && continue
        xs=sort!(unique([p[1] for b in boxes for p in b]));ys=sort!(unique([p[2] for b in boxes for p in b]))
        big(length(xs))*length(ys)<=100_000 || throw(ArgumentError("Support drawing exceeds 100000 planar cells."))
        mask=[any(b->b[1][1]<=x<b[2][1] && b[1][2]<=y<b[2][2],boxes) for x in xs[1:end-1],y in ys[1:end-1]]
        rects=NTuple{4,Float64}[];closed=NTuple{4,Float64}[];opened=NTuple{4,Float64}[]
        for x in axes(mask,1),y in axes(mask,2)
            mask[x,y] || continue
            a,b,c,d=Float64.((xs[x],ys[y],xs[x+1],ys[y+1]))
            a<c && b<d || throw(ArgumentError("Exact support boundaries collide at Float64 display precision; inspect exact support queries."))
            push!(rects,(a,b,c,d))
            (x==1 || !mask[x-1,y]) && push!(closed,(a,b,a,d))
            (y==1 || !mask[x,y-1]) && push!(closed,(a,b,c,b))
            (x==size(mask,1) || !mask[x+1,y]) && push!(opened,(c,b,c,d))
            (y==size(mask,2) || !mask[x,y+1]) && push!(opened,(a,d,c,d))
        end
        role=_VisualRole(:categorical,j)
        push!(layers,RectLayer(rects,role,role,0.35,0.))
        push!(layers,SegmentLayer(closed,role,1.,2.,:solid))
        push!(layers,SegmentLayer(opened,role,1.,2.,:dash))
    end
    if o.slices
        for (k,bar) in enumerate(r.barcodes)
            segments=NTuple{4,Float64}[Float64.((a,a+r.offsets[k],b,b+r.offsets[k])) for (a,b) in bar]
            push!(layers,SegmentLayer(segments,_VisualRole(:muted),0.6,1.))
        end
    end
    isempty(layers) && push!(layers,TextLayer(["No candidate supports selected"],
        [Tuple(Float64.((r.window[1].+r.window[2])./2))],_VisualRole(:muted),14.))
    VisualizationSpec(:interval_approximation;title="Candidate interval supports",
        subtitle="$(length(o.selected)) of $(length(r.rectangles)) supports; $(length(r.offsets)) diagonal slices.\nSolid lower edges are included; dashed upper edges are excluded. No source splitting or global error bound is asserted.",
        layers,axes=_geometry_axes_2d(r.window),metadata=(;selection=o,exact_supports=IntervalApproximations.approximation_supports(r),
            provenance=IntervalApproximations.provenance(r),figure_size=(1000,700)))
end
