# Actual bottleneck witnesses: stable expanded member IDs, never screen-space
# matching. Exact metadata survives finite display coordinates and budgets.

const _BottleneckWitness = NamedTuple{(:distance,:a_to_b,:b_to_a,:points_a,:points_b)}
const _MatchingVisualInput = Union{_BottleneckWitness,Fibered2D.MatchingDistanceResult2D}

available_visuals(::_MatchingVisualInput) = (:matching,)
_visual_request_keywords(::_MatchingVisualInput,::Symbol) = (:pair,:max_pairs)
_visual_request_cost(::_MatchingVisualInput,::Symbol) =
    (;work=:retained_matching,timing=:not_measured,optimization=:none)

_matching_endpoint_cost(a,b) = a == b ? 0 : (!isfinite(a) || !isfinite(b)) ? Inf :
    abs(_visual_exact_coordinate(a)-_visual_exact_coordinate(b))
_matching_diagonal_cost(p) = all(isfinite,p) ?
    (_visual_exact_coordinate(p[2])-_visual_exact_coordinate(p[1]))/2 : Inf
_matching_pair_cost(p,q) = max(_matching_endpoint_cost(p[1],q[1]),_matching_endpoint_cost(p[2],q[2]))

function _matching_records(w::_BottleneckWitness)
    A,B = w.points_a,w.points_b
    length(A) == length(w.a_to_b) && length(B) == length(w.b_to_a) ||
        throw(ArgumentError("Witness assignments must have one entry per expanded interval."))
    for p in Iterators.flatten((A,B))
        length(p) == 2 && all(x->x isa Real && !isnan(x),p) && p[1] <= p[2] &&
            !(p[1] == p[2] && !isfinite(p[1])) || throw(ArgumentError("Invalid witness interval."))
    end
    for (indices,other,n) in ((w.a_to_b,w.b_to_a,length(B)),(w.b_to_a,w.a_to_b,length(A)))
        for (i,j) in enumerate(indices)
            j isa Integer && !(j isa Bool) && 0 <= j <= n || throw(ArgumentError("Invalid matching member index."))
            j == 0 || other[j] == i || throw(ArgumentError("Witness assignments are not inverse partial matchings."))
        end
    end
    countsA,countsB = Dict{Any,Int}(),Dict{Any,Int}()
    for (points,counts) in ((A,countsA),(B,countsB)), p in points
        counts[p] = get(counts,p,0)+1
    end
    records = NamedTuple[]
    function retain(i,j)
        a,b = i == 0 ? nothing : A[i],j == 0 ? nothing : B[j]
        cost = a === nothing ? _matching_diagonal_cost(b) : b === nothing ?
            _matching_diagonal_cost(a) : _matching_pair_cost(a,b)
        push!(records,(;id=length(records)+1,a=i,b=j,interval_a=a,interval_b=b,cost,
            diagonal=i == 0 || j == 0,essential=(a !== nothing && !all(isfinite,a)) ||
                (b !== nothing && !all(isfinite,b)),
            multiplicity_a=i == 0 ? 0 : countsA[a],multiplicity_b=j == 0 ? 0 : countsB[b]))
    end
    for (i,j) in enumerate(w.a_to_b)
        retain(i,j)
    end
    for (j,i) in enumerate(w.b_to_a)
        i == 0 && retain(0,j)
    end
    maxcost = maximum((r.cost for r in records);init=0)
    w.distance isa Real && !isnan(w.distance) && w.distance >= 0 || throw(ArgumentError("Invalid witness distance."))
    (maxcost == w.distance || isapprox(Float64(maxcost),Float64(w.distance);rtol=8eps(Float64),atol=0)) ||
        throw(ArgumentError("The largest witness cost disagrees with its distance."))
    return records
end

function _matching_context(r::Fibered2D.MatchingDistanceResult2D)
    return (; scope=:certified_finite_window,status=r.status,query=Fibered2D.slice_query(r),
        box=Fibered2D.working_box(r),weight_convention=r.weight_convention,
        normalization=r.normalization,weighted_distance=r.exact_distance,samples=(),sample=nothing)
end
_matching_context(::_BottleneckWitness) = (;scope=:bottleneck,status=:provided_witness,
    query=nothing,box=nothing,weight_convention=:none,normalization=:none,
    weighted_distance=nothing,samples=(),sample=nothing)
_matching_assignment(w::_BottleneckWitness) = w
function _matching_assignment(r::Fibered2D.MatchingDistanceResult2D)
    Fibered2D.check_matching_result_2d(r;throw=true)
    w = Fibered2D.matching_witness(r)
    return w === nothing ? bottleneck_matching(Tuple{Int,Int}[],Tuple{Int,Int}[]) : w
end

function _matching_options(w;pair=nothing,max_pairs=100)
    records = _matching_records(w)
    budget = _inspection_integer(max_pairs,:max_pairs;lower=1)
    selected = pair === nothing ? nothing : _inspection_integer(pair,:pair)
    selected == 0 && (selected=nothing)
    selected === nothing || selected <= length(records) || throw(ArgumentError("pair must identify a matching record, or 0 to clear."))
    return records,selected,budget
end
function _append_visual_request_issues!(issues::Vector{String},obj::_MatchingVisualInput,::Symbol;kwargs...)
    _matching_options(_matching_assignment(obj);kwargs...)
    return issues
end
function _visual_spec(obj::_MatchingVisualInput,kind::Symbol;pair=nothing,max_pairs=100)
    return _matching_spec(_matching_assignment(obj),_matching_context(obj);pair,max_pairs)
end

_matching_interval_label(p) = p === nothing ? "diagonal" : "(" * join(_interval_value.(p),", ") * ")"
_matching_scope_label(scope) = scope === :certified_finite_window ? "Exact finite-window optimum" :
    scope === :sampled_slice ? "Sampled slice" : scope === :selected_slice ? "Selected slice" : "Bottleneck matching"
function _matching_record_label(r)
    return "Pair $(r.id): A$(r.a == 0 ? "-" : r.a) $(_matching_interval_label(r.interval_a)) -> " *
        "B$(r.b == 0 ? "-" : r.b) $(_matching_interval_label(r.interval_b)); cost $(_interval_value(r.cost))"
end

function _matching_spec(w,context;pair=nothing,max_pairs=100)
    records,selected,budget = _matching_options(w;pair,max_pairs)
    ids = collect(1:min(length(records),budget))
    selected === nothing || selected in ids || (isempty(ids) ? push!(ids,selected) : (ids[end]=selected))
    finite = Float64[Float64(x) for p in Iterators.flatten((w.points_a,w.points_b)) for x in p if isfinite(x)]
    all(isfinite,finite) || throw(ArgumentError("Matching endpoints exceed finite display coordinates; rescale the input."))
    lo,hi = isempty(finite) ? (0.,1.) : extrema(finite)
    span = max(hi-lo,1.)
    isfinite(span) || throw(ArgumentError("Matching endpoint range exceeds display coordinates."))
    low,high = lo-0.12span,hi+0.18span
    # Infinity lives on labelled rails; it is never capped in the mathematics.
    coord(x) = isfinite(x) ? Float64(x) : x < 0 ? low : high
    point(p) = (coord(p[1]),coord(p[2]))
    diagram = AbstractVisualizationLayer[SegmentLayer([(low,low,high,high)],_VisualRole(:muted),0.7,1.,:dash)]
    bars,costs = AbstractVisualizationLayer[],AbstractVisualizationLayer[]
    pick_diagram,pick_bars = NamedTuple[],NamedTuple[]
    costmax = maximum((Float64(r.cost) for r in records if isfinite(r.cost));init=0.)
    isfinite(costmax) || throw(ArgumentError("Matching costs exceed display coordinates."))
    costtop = max(costmax*1.2,1.)
    largest_cost = maximum((r.cost for r in records);init=0)
    bottleneck_ids = Int[r.id for r in records if r.cost == largest_cost]
    for (row,id) in enumerate(ids)
        r = records[id]
        pa,pb = r.interval_a,r.interval_b
        for (name,p,role,shift) in ((:a,pa,:source,0.17),(:b,pb,:target,-0.17))
            p === nothing && continue
            pos = point(p)
            push!(diagram,PointLayer([pos],_VisualRole(role),0.95,11.))
            push!(pick_diagram,(;id,point=pos))
            segment=(coord(p[1]),Float64(row)+shift,coord(p[2]),Float64(row)+shift)
            push!(bars,SegmentLayer([segment],_VisualRole(role),0.95,selected == id ? 5. : 3.,name === :a ? :solid : :dash))
            push!(pick_bars,(;id,segment))
            for (endpoint,x) in ((p[1],low),(p[2],high))
                isfinite(endpoint) || push!(bars,TextLayer([endpoint < 0 ? "-inf" : "+inf"],
                    [(x,Float64(row)+shift)],_VisualRole(role),11.))
            end
            selected == id && push!(diagram,PointLayer([pos],_VisualRole(:selected),0.45,20.))
        end
        if !r.essential || !r.diagonal
            start = pa === nothing ? ((pb[1]+pb[2])/2,(pb[1]+pb[2])/2) : pa
            stop = pb === nothing ? ((pa[1]+pa[2])/2,(pa[1]+pa[2])/2) : pb
            p,q = point(start),point(stop)
            push!(diagram,SegmentLayer([(p...,q...)],_VisualRole(selected == id ? :selected : :edge),0.8,
                selected == id ? 3.5 : 1.3,r.diagonal ? :dash : :solid))
        end
        c = isfinite(r.cost) ? Float64(r.cost) : costtop
        push!(costs,SegmentLayer([(0.,Float64(row),c,Float64(row))],_VisualRole(selected == id ? :selected : :edge),1.,selected == id ? 4. : 2.))
        push!(costs,PointLayer([(c,Float64(row))],_VisualRole(:finite),1.,8.))
        label=_interval_value(r.cost)
        ncodeunits(label)>14 && (label="~"*string(round(Float64(r.cost);sigdigits=5)))
        push!(costs,TextLayer([label],[(c,Float64(row)+0.22)],_VisualRole(:foreground),11.))
    end
    multiplicity_labels=Dict{Any,Vector{String}}()
    for (points,role,name) in ((w.points_a,:source,"A"),(w.points_b,:target,"B"))
        counts=Dict{Any,Int}()
        for p in points;counts[p]=get(counts,p,0)+1;end
        for p in unique(points)
            counts[p]>1 || continue
            any(id->(role===:source ? records[id].interval_a : records[id].interval_b)==p,ids) || continue
            push!(get!(multiplicity_labels,p,String[]),"$name x$(counts[p])")
        end
    end
    for (p,labels) in multiplicity_labels
        xy=point(p)
        push!(diagram,TextLayer([join(labels," / ")],[(xy[1]+0.02span,xy[2]+0.04span)],_VisualRole(:muted),11.))
    end
    exact_coordinates=unique([x for p in Iterators.flatten((w.points_a,w.points_b)) for x in p if isfinite(x)])
    display_collisions=length(exact_coordinates)-length(unique(Float64.(exact_coordinates)))
    isempty(bars) && push!(bars,TextLayer(["Both barcodes are empty"],[(lo,0.)],_VisualRole(:muted),16.))
    isempty(costs) && push!(costs,TextLayer(["Distance = 0"],[(0.,0.)],_VisualRole(:muted),16.))
    finite_ticks=lo==hi ? [lo] : [lo,lo/2+hi/2,hi]
    finite_labels=_interval_value.(finite_ticks)
    all_points=Iterators.flatten((w.points_a,w.points_b))
    birth_ticks=any(p->!isfinite(p[1]),all_points) ?
        ([low;finite_ticks],["-inf";finite_labels]) : nothing
    death_ticks=any(p->!isfinite(p[2]),all_points) ?
        ([finite_ticks;high],[finite_labels;"+inf"]) : nothing
    finite_cost_ticks=unique([0.,costmax/2,costmax])
    cost_ticks=any(r->!isfinite(r.cost),records) ?
        ([finite_cost_ticks;costtop],[_interval_value.(finite_cost_ticks);"+inf"]) : nothing
    legend = _default_legend(visible=true,entries=(;a=(;label="A",color=_VisualRole(:source),style=:marker),
        b=(;label="B",color=_VisualRole(:target),style=:marker)))
    ticks=(Float64.(eachindex(ids)),string.(ids))
    panels = [VisualizationSpec(:matching_diagram;title="One optimal assignment",subtitle="Dashed links end on the diagonal; A/B shapes identify the inputs.",
        layers=diagram,axes=_default_axes_2d(xlabel="birth",ylabel="death",xlimits=(low-0.04span,high+0.04span),ylimits=(low-0.04span,high+0.08span),xticks=birth_ticks,yticks=death_ticks),legend,
        metadata=(;pick_targets=pick_diagram)),
        VisualizationSpec(:matching_barcodes;title="Intervals, aligned by pair",subtitle="A: solid; B: dashed. Rows retain expanded member IDs.",layers=bars,
            axes=_default_axes_2d(xlabel="line parameter",ylabel="pair",xlimits=(low-0.04span,high+0.12span),ylimits=(0.,max(length(ids)+1.,1.)),yticks=ticks,aspect=:auto),legend,
            metadata=(;pick_targets=pick_bars)),
        VisualizationSpec(:matching_costs;title="What sets the distance?",subtitle="Per-pair costs; infinity is labelled, never truncated numerically.",layers=costs,
            axes=_default_axes_2d(xlabel="bottleneck pair cost",ylabel="pair",xlimits=(0.,costtop*1.25),ylimits=(0.,max(length(ids)+1.,1.)),xticks=cost_ticks,yticks=ticks,aspect=:auto))]
    lines = ["Bottleneck distance: $(_interval_value(w.distance)); largest pair cost: $(_interval_value(largest_cost)).",
        "$(length(ids)) / $(length(records)) pairs displayed; $(length(w.points_a)) A members, $(length(w.points_b)) B members.",
        "$(count(r->r.diagonal,records)) diagonal assignments; $(count(r->r.essential,records)) pairs with infinite endpoints.",
        "$(length(bottleneck_ids)) pairs attain this witness's largest cost.",
        "Deterministic optimal assignment; uniqueness is not asserted. Coincident members stay distinct."]
    display_collisions == 0 || push!(lines,"Distinct exact endpoints coincide after drawing conversion; inspect the exact selected-pair readout.")
    if haskey(context,:ordinary)
        o=context.ordinary
        append!(lines,["Homological degree $(o.dim); $(o.order) inputs; increasing analysis coordinates.",
            "Essential policy $(o.essential), cap $(o.essential_cap); origin $(o.origin), scale $(o.scale).",
            "Fields: A=$(o.field_a), B=$(o.field_b). Matching compares intervals, not source chains."])
    end
    if selected !== nothing
        r=records[selected]
        append!(lines,[_matching_record_label(r),"Equal interval multiplicities: A=$(r.multiplicity_a), B=$(r.multiplicity_b)."])
    end
    if context.query !== nothing
        q=context.query
        append!(lines,["$(_matching_scope_label(context.scope)); status: $(context.status).",
            "$(context.normalization) direction $(_matching_interval_label(q.direction)), basepoint $(_matching_interval_label(q.basepoint)).",
            "Weight $(_interval_value(q.weight)) ($(context.weight_convention)); weighted distance $(_interval_value(context.weighted_distance)).",
            "Window clipping is mathematical truncation; this is not a computed interleaving distance.",
            "Member IDs belong to this query, not to tracked classes across different slices."])
        box=context.box
        win=_inspection_slice_window(q,box)
        layers=AbstractVisualizationLayer[RectLayer([(Float64(box[1][1]),Float64(box[1][2]),Float64(box[2][1]),Float64(box[2][2]))],_VisualRole(:surface),_VisualRole(:edge),0.4,1.)]
        if win !== nothing
            ends=[ntuple(i->Float64(q.basepoint[i]+t*q.direction[i]),2) for t in win]
            push!(layers,SegmentLayer([(ends[1]...,ends[2]...)],_VisualRole(:selected),1.,3.))
        end
        push!(panels,VisualizationSpec(:matching_slice;title="The same slice through both modules",subtitle="Line and finite comparison window",layers,
            axes=_default_axes_2d(xlabel="parameter 1",ylabel="parameter 2"),metadata=(;query=q,box)))
    elseif context.scope === :certified_finite_window
        push!(lines,"Degenerate window: value zero; no maximizing slice is asserted.")
    end
    if !isempty(context.samples)
        samples=context.samples
        push!(lines,"Sampled maximum: $(maximum(s.weighted_distance for s in samples)); $(length(samples)) explicit queries, no certified error bound.")
        xy=[(s.angle,s.offset) for s in samples]
        ls=AbstractVisualizationLayer[PointLayer(xy,_VisualRole(:foreground),0.8,13.),
            PointLayer(xy,Float64[s.weighted_distance for s in samples],1.,11.,:viridis)]
        context.sample === nothing || push!(ls,PointLayer([xy[context.sample]],_VisualRole(:selected),0.6,22.))
        push!(panels,VisualizationSpec(:matching_sample_map;title="Sampled angle / offset costs",subtitle="Discrete samples only; no certified optimum or interpolation.",layers=ls,
            axes=_default_axes_2d(xlabel="angle (degrees)",ylabel="normal offset",aspect=:auto),
            metadata=(;samples,point_colorbar_label="weighted bottleneck cost",pick_targets=[(;id=i,point=p) for (i,p) in enumerate(xy)])))
    end
    push!(panels,_inspection_text_panel("Read the witness",lines))
    return VisualizationSpec(:matching;title="Distance and matching witness",subtitle="Compare matched intervals, diagonal assignments and their costs.",panels,
        metadata=(;matching=w,matching_records=records,selected_pair=selected,displayed_pair_ids=ids,
            bottleneck_pair_ids=bottleneck_ids,display_collisions,context,panel_columns=2,figure_size=(1450,isempty(context.samples) && context.query === nothing ? 1000 : 1450),
            uniqueness=:not_asserted,inspection_selection=(;pair=selected)))
end
