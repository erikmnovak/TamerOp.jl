# Shared exact interval records and bounded barcode/diagram display geometry.
# Mathematical endpoint data is never replaced by drawing coordinates.

_interval_value(x::Rational) = denominator(x) == 1 ? string(numerator(x)) :
    string(numerator(x), "/", denominator(x))
_interval_value(x::AbstractFloat) = replace(string(x), r"\.0$" => "")
_interval_value(x) = string(x)

function _interval_record(b, d, mult; left_closed=true, right_closed=false,
                          left_status=isfinite(b) ? :finite : :essential,
                          right_status=isfinite(d) ? :finite : :essential,
                          members=())
    b isa Real && d isa Real && !(b isa Bool) && !(d isa Bool) ||
        throw(ArgumentError("Interval endpoints must be real numbers, not Bool."))
    !isnan(b) && !isnan(d) && b <= d && b != Inf && d != -Inf ||
        throw(ArgumentError("Intervals require ordered endpoints birth <= death, without NaN."))
    mult isa Integer && !(mult isa Bool) && 0 < mult <= typemax(Int) ||
        throw(ArgumentError("Interval multiplicities must be positive integers fitting Int."))
    left_status in (:finite, :essential, :censored) && right_status in (:finite, :essential, :censored) ||
        throw(ArgumentError("Endpoint status must be :finite, :essential, or :censored."))
    (isfinite(b) ? left_status !== :essential : left_status === :essential) &&
    (isfinite(d) ? right_status !== :essential : right_status === :essential) ||
        throw(ArgumentError("Only certified infinite endpoints have status :essential."))
    return (; birth=b, death=d, left_closed=isfinite(b) && left_closed,
        right_closed=isfinite(d) && right_closed, multiplicity=Int(mult),
        left_clipped=left_status === :censored, right_clipped=right_status === :censored,
        singleton=b == d, left_status, right_status, members=Tuple(members))
end

function _group_interval_records(records)
    # Group only identical decorated intervals with the same endpoint evidence.
    grouped = Dict{Tuple,NamedTuple}()
    for r in records
        key = (iszero(r.birth) ? zero(r.birth) : r.birth, iszero(r.death) ? zero(r.death) : r.death, r.left_closed, r.right_closed, r.left_status, r.right_status)
        if haskey(grouped, key)
            prev = grouped[key]
            grouped[key] = merge(prev, (; multiplicity=Base.checked_add(prev.multiplicity, r.multiplicity),
                members=(prev.members..., r.members...)))
        else
            grouped[key] = r
        end
    end
    ordered = sort!(collect(values(grouped)); by=r ->
        (r.birth, r.death, !r.left_closed, !r.right_closed, string(r.left_status), string(r.right_status)))
    return Tuple(merge((; id=i), r) for (i,r) in enumerate(ordered))
end

function _barcode_records(bar; boundary=nothing)
    entries = bar isa AbstractVector ? ((iv, 1) for iv in bar) : bar
    records = NamedTuple[]
    for (iv,mult) in entries
        length(iv) == 2 || throw(ArgumentError("Barcode entries require pairs of endpoints."))
        b, d = iv
        left_status = !isfinite(b) ? :essential : boundary !== nothing && b == first(boundary) ? :censored : :finite
        right_status = !isfinite(d) ? :essential : boundary !== nothing && d == last(boundary) ? :censored : :finite
        # Empty half-open intervals contribute no module and no diagram point.
        record = _interval_record(b, d, mult; left_status, right_status)
        b == d || push!(records, record)
    end
    return _group_interval_records(records)
end

_interval_status(r, side::Symbol) = side === :left ?
    get(r, :left_status, !isfinite(r.birth) ? :essential : r.left_clipped ? :censored : :finite) :
    get(r, :right_status, !isfinite(r.death) ? :essential : r.right_clipped ? :censored : :finite)

function _interval_record_label(r)
    suffix = String[]
    _interval_status(r, :left) === :censored && push!(suffix, "left endpoint censored")
    _interval_status(r, :right) === :censored && push!(suffix, "right endpoint censored")
    label = "#$(r.id) " * (r.left_closed ? "[" : "(") * _interval_value(r.birth) * ", " *
        _interval_value(r.death) * (r.right_closed ? "]" : ")") * " x$(r.multiplicity)"
    return isempty(suffix) ? label : label * " (" * join(suffix, "; ") * ")"
end

function _interval_window(records, window)
    if window !== nothing
        (window isa Tuple || window isa AbstractVector) && length(window) == 2 ||
            throw(ArgumentError("window must be a finite ordered pair (lo, hi)."))
        lo, hi = window
        all(x -> x isa Real && !(x isa Bool) && isfinite(x), (lo,hi)) && lo <= hi ||
            throw(ArgumentError("window must be a finite ordered pair (lo, hi)."))
        return (lo,hi)
    end
    finite = Real[x for r in records for x in (r.birth,r.death) if isfinite(x)]
    return isempty(finite) ? (0,1) : extrema(finite)
end

_interval_multiplicity(records) = foldl((n,r) -> Base.checked_add(n,r.multiplicity),records;init=0)

function _interval_view(records; window=nothing, interval=nothing, max_intervals=200, order=:sublevel)
    order in (:sublevel,:superlevel) || throw(ArgumentError("order must be :sublevel or :superlevel."))
    max_intervals isa Integer && !(max_intervals isa Bool) && 1 <= max_intervals <= typemax(Int) ||
        throw(ArgumentError("max_intervals must be a positive integer fitting Int."))
    ids = Tuple(r.id for r in records)
    length(unique(ids)) == length(ids) || throw(ArgumentError("Interval group IDs must be distinct."))
    interval === nothing || (interval isa Integer && !(interval isa Bool) && interval in ids) ||
        throw(ArgumentError("interval must be an existing interval group ID or nothing."))
    exact_window = _interval_window(records, window)
    lo, hi = Float64.(exact_window)
    all(isfinite, (lo,hi)) || throw(ArgumentError("Interval window exceeds the Float64 drawing range; supply a smaller window."))
    span = hi - lo
    isfinite(span) || throw(ArgumentError("Interval window leaves no finite drawing scale; use a smaller window."))
    span = max(span, max(abs(lo),abs(hi),1.0)*0.05)
    lanes = (negative=lo-0.20span, positive=hi+0.20span)
    essential_left = any(r -> _interval_status(r,:left) === :essential, records)
    essential_right = any(r -> _interval_status(r,:right) === :essential, records)
    limits = ((essential_left ? lanes.negative : lo)-0.08span,
              (essential_right ? lanes.positive : hi)+0.08span)
    all(isfinite, (limits...,lanes...)) && limits[1] < limits[2] ||
        throw(ArgumentError("Interval window leaves no finite plotting margin; rescale it."))
    visible = [r for r in records if r.death >= first(exact_window) && r.birth <= last(exact_window) &&
        !(r.death == first(exact_window) && !r.right_closed) &&
        !(r.birth == last(exact_window) && !r.left_closed)]
    displayed = visible[1:min(length(visible),Int(max_intervals))]
    if interval !== nothing && !(interval in (r.id for r in displayed))
        selected = findfirst(r -> r.id == interval, visible)
        if selected !== nothing
            length(displayed) == max_intervals && pop!(displayed)
            push!(displayed, visible[selected])
            sort!(displayed; by=r -> findfirst(==(r.id),ids))
        end
    end
    segments = NTuple{4,Float64}[]
    points = NTuple{2,Float64}[]
    statuses = Tuple{Symbol,Symbol}[]
    collisions = Int[]
    for (row,r) in enumerate(displayed)
        ls, rs = _interval_status(r,:left), _interval_status(r,:right)
        ls === :finite && r.birth < first(exact_window) && (ls=:offscreen)
        rs === :finite && r.death > last(exact_window) && (rs=:offscreen)
        # Censored restriction boundaries remain censored even after extra display clipping.
        b = ls === :essential ? lanes.negative : Float64(clamp(r.birth,exact_window...))
        d = rs === :essential ? lanes.positive : Float64(clamp(r.death,exact_window...))
        push!(segments,(b,Float64(row),d,Float64(row)))
        push!(points,order === :sublevel ? (b,d) : (d,b))
        push!(statuses,(ls,rs))
        r.birth != r.death && isfinite(r.birth) && isfinite(r.death) &&
            first(exact_window) <= r.birth <= r.death <= last(exact_window) && b == d && push!(collisions,r.id)
    end
    selected_record = interval === nothing ? nothing : records[findfirst(r -> r.id == interval, records)]
    return (; records=Tuple(records), displayed_records=Tuple(displayed),
        interval_ids=Tuple(r.id for r in displayed), all_interval_ids=ids,
        bar_segments=Tuple(segments), diagram_points=Tuple(points), endpoint_display=Tuple(statuses),
        window=exact_window, limits, infinity_lanes=lanes, selected_interval=interval, selected_record,
        coordinate_collisions=Tuple(collisions), total_groups=length(records), displayed_groups=length(displayed),
        offscreen_groups=length(records)-length(visible), omitted_groups=length(visible)-length(displayed),
        total_multiplicity=_interval_multiplicity(records),
        displayed_multiplicity=_interval_multiplicity(displayed), order)
end

function _interval_caption(view; diagram::Bool=false)
    grouped = view.total_groups != view.total_multiplicity
    count = "$(view.total_multiplicity) " * (view.total_multiplicity == 1 ? "interval" : "intervals")
    grouped && (count *= " in $(view.total_groups) groups")
    view.displayed_groups < view.total_groups && (count *= "; showing $(view.displayed_groups) of $(view.total_groups) groups")
    notes = String[count]
    view.offscreen_groups > 0 && push!(notes, "$(view.offscreen_groups) groups outside window.")
    view.omitted_groups > 0 && push!(notes, "$(view.omitted_groups) groups hidden by max_intervals.")
    view.selected_interval !== nothing && !(view.selected_interval in view.interval_ids) &&
        push!(notes, "Selected #$(view.selected_interval) is outside window.")
    statuses = Set(s for pair in view.endpoint_display for s in pair)
    if !isempty(statuses)
        if diagram
            push!(notes, "Brackets include [ ] or exclude ( ) endpoints.")
        elseif :finite in statuses
            closed = Set(included for (record,pair) in zip(view.displayed_records,view.endpoint_display)
                for (status,included) in zip(pair,(record.left_closed,record.right_closed)) if status === :finite)
            push!(notes, length(closed) == 2 ? "Filled/open circles include/exclude endpoints." :
                true in closed ? "Filled circles include endpoints." : "Open circles exclude endpoints.")
        end
        :essential in statuses && push!(notes, "Inf: continues indefinitely.")
        :censored in statuses && push!(notes, "?: censored endpoint; continuation is unknown.")
        :offscreen in statuses && push!(notes, diagram ? "outside: finite endpoint beyond the view." :
            "< or >: finite endpoint beyond the view.")
    end
    isempty(view.coordinate_collisions) ||
        push!(notes, "Distinct exact endpoints coincide in the drawing; inspect exact records.")
    return join(notes, "\n")
end

function _interval_panels(records; window=nothing, interval=nothing, max_intervals=200, order=:sublevel,
                          barcode_kind=:barcode, diagram_kind=:persistence_diagram,
                          barcode_title="Barcode", diagram_title="Persistence diagram",
                          endpoint_semantics=:decorated, essential_status=:certified_if_supplied,
                          metadata=NamedTuple(), highlight=true)
    view = _interval_view(records; window,interval,max_intervals,order)
    lo, hi = Float64.(view.window)
    bars = AbstractVisualizationLayer[]
    endpoint_layers = (left=Int[], right=Int[])
    diagram = AbstractVisualizationLayer[SegmentLayer([(lo,lo,hi,hi)], _VisualRole(:muted),0.6,1.0,:dash)]
    labels = Dict{NTuple{2,Float64},Vector{String}}()
    for (r,seg,point,statuses) in zip(view.displayed_records,view.bar_segments,view.diagram_points,view.endpoint_display)
        color = _VisualRole(:categorical,r.id)
        push!(bars,SegmentLayer([seg],color,1.0,highlight && r.id == interval ? 5.0 : 2.5))
        for (x,closed,status,arrow,side) in ((seg[1],r.left_closed,statuses[1],"<",:left),(seg[3],r.right_closed,statuses[2],">",:right))
            if status === :finite
                push!(bars,PointLayer([(x,seg[2])],color,1.0,9.0))
                closed || push!(bars,PointLayer([(x,seg[2])],_VisualRole(:background),1.0,5.0))
            else
                glyph = status === :essential ? (arrow == "<" ? "< -Inf" : "> +Inf") : status === :censored ? "?" : arrow
                push!(bars,TextLayer([glyph],[(x,seg[2])],color,13.0))
                push!(getproperty(endpoint_layers,side),length(bars))
            end
        end
        push!(diagram,PointLayer([point],color,1.0,highlight && r.id == interval ? 16.0 : 10.0))
        evidence = join((s === :censored ? "?" : s === :offscreen ? "outside" : s === :essential ? "Inf" : "" for s in statuses)," ")
        label = "#$(r.id) " * (r.left_closed ? "[" : "(") * (r.right_closed ? "]" : ")")
        r.multiplicity == 1 || (label *= " x$(r.multiplicity)")
        isempty(strip(evidence)) || (label *= " " * strip(evidence))
        push!(get!(Vector{String},labels,point),label)
        if highlight && r.id == interval
            push!(bars,PointLayer([(seg[1]/2+seg[3]/2,seg[2])],_VisualRole(:selected),1.0,12.0))
            push!(diagram,PointLayer([point],_VisualRole(:selected),1.0,12.0))
        end
    end
    # Coincident points share one multiline annotation. The renderer lays all
    # diagram annotations out together in pixel space, retaining these anchors.
    annotation_layers = Int[]
    for point in unique(view.diagram_points)
        push!(diagram,TextLayer([join(labels[point],"\n")],[point],_VisualRole(:foreground),12.0))
        push!(annotation_layers,length(diagram))
    end
    if isempty(view.displayed_records)
        message = isempty(records) ? "No nonzero intervals" : "No intervals intersect the displayed window"
        midpoint = lo/2+hi/2
        push!(bars,TextLayer([message],[(midpoint,1.0)],_VisualRole(:muted),13.0))
        push!(diagram,TextLayer([message],[(midpoint,midpoint)],_VisualRole(:muted),13.0))
    end
    negative = any(r -> _interval_status(r,:left) === :essential,view.displayed_records)
    positive = any(r -> _interval_status(r,:right) === :essential,view.displayed_records)
    tick_positions = unique([lo,hi])
    # Small examples should let a reader read the actual births and deaths.
    # Keep dense/near-coincident diagrams on the bounded window ticks instead.
    finite_ticks = sort!(unique(vcat(tick_positions,
        Float64[x for r in view.displayed_records for x in (r.birth,r.death)
            if isfinite(x) && lo <= x <= hi])))
    if length(finite_ticks) <= 8 && all(gap -> gap >= 0.1*(hi-lo), diff(finite_ticks))
        tick_positions = finite_ticks
    end
    tick_labels = _interval_value.(tick_positions)
    negative && (pushfirst!(tick_positions,view.infinity_lanes.negative); pushfirst!(tick_labels,"-Inf"))
    positive && (push!(tick_positions,view.infinity_lanes.positive); push!(tick_labels,"+Inf"))
    ticks = (tick_positions,tick_labels)
    common = merge(metadata,view,(; endpoint_semantics,essential_status,display_coordinates=:float64,
        barcode_count=length(records),minimal_axes=true,legend_position=:none))
    barcode = VisualizationSpec(barcode_kind;title=barcode_title,subtitle=_interval_caption(view),
        layers=bars,axes=_default_axes_2d(xlabel="Parameter",
            ylabel=view.total_groups == view.total_multiplicity ? "Interval" : "Interval group",xlimits=view.limits,
            ylimits=(0.3,max(1,view.displayed_groups)+0.7),aspect=:auto,xticks=ticks,
            yticks=(Float64.(1:view.displayed_groups),["#$(r.id)" * (r.multiplicity == 1 ? "" : " x$(r.multiplicity)") for r in view.displayed_records])),
        metadata=merge(common,(;interval_segments=view.bar_segments,
            figure_size=(760,clamp(300 + 24*view.displayed_groups,360,780)),
            barcode_endpoint_layers=(left=Tuple(endpoint_layers.left),right=Tuple(endpoint_layers.right)))),
        interaction=_default_interaction(labels=true))
    diagram_spec = VisualizationSpec(diagram_kind;title=diagram_title,subtitle=_interval_caption(view;diagram=true),
        layers=diagram,axes=_default_axes_2d(xlabel="Birth",ylabel="Death",xlimits=view.limits,ylimits=view.limits,
            aspect=:equal,xticks=ticks,yticks=ticks),metadata=merge(common,(;interval_points=view.diagram_points,
                figure_size=(640,560),
                interval_annotation_layers=Tuple(annotation_layers))),
        interaction=_default_interaction(labels=true))
    return (barcode,diagram_spec)
end

const _RawVisualBarcode = Union{AbstractDict{<:Tuple{<:Real,<:Real},<:Integer},
    AbstractVector{<:Tuple{<:Real,<:Real}},PackedBarcode}

function _interval_payload(bar::_RawVisualBarcode; dim=0,index=nothing)
    dim isa Integer && !(dim isa Bool) && dim == 0 || throw(ArgumentError("A raw barcode has no homological dimension selector."))
    index === nothing || throw(ArgumentError("A raw barcode is already one barcode; index is not applicable."))
    return (; records=_barcode_records(bar),order=:sublevel,
        endpoint_semantics=:supplied_half_open,essential_status=:explicit_infinity_only,
        metadata=(;source=:barcode,ambient_coverage=:unspecified,
            representative_available=false,representative_reason=:not_retained))
end

function _interval_payload(result::SliceBarcodesResult; dim=0,index=nothing)
    dim isa Integer && !(dim isa Bool) && dim == 0 || throw(ArgumentError("A slice barcode family has no homological dimension selector."))
    payload = _interval_payload(_slice_barcode_at(result,index))
    return merge(payload,(;metadata=merge(payload.metadata,(;source=:slice_barcode_family,
        slice_index=index,endpoint_provenance=:supplied_slice_endpoints))))
end

function _interval_payload(result::FiberedSliceResult; dim=0,index=nothing)
    dim isa Integer && !(dim isa Bool) && dim == 0 || throw(ArgumentError("A fibered slice has no homological dimension selector."))
    index === nothing || throw(ArgumentError("A fibered slice is already one barcode; index is not applicable."))
    vals = slice_values(result)
    boundary = isempty(vals) ? nothing : extrema(vals)
    return (;records=_barcode_records(slice_barcode(result);boundary),order=:sublevel,
        endpoint_semantics=:half_open_restriction,essential_status=:not_inferred,
        metadata=(;source=:fibered_slice,restriction_window=boundary,ambient_coverage=:restricted,
            representative_available=false,representative_reason=:not_retained))
end

function _interval_payload(result::ProjectedBarcodesResult; dim=0,index=nothing)
    dim isa Integer && !(dim isa Bool) && dim == 0 || throw(ArgumentError("A projected barcode family has no homological dimension selector."))
    bars = projected_barcodes(result)
    chosen = _interval_family_index(bars,index,"ProjectedBarcodesResult")
    payload = _interval_payload(bars[chosen])
    return merge(payload,(;metadata=merge(payload.metadata,(;source=:projected_barcodes,
        projection_index=projection_indices(result)[chosen]))))
end

function _interval_family_index(bars,index,name)
    if index === nothing
        length(bars) == 1 || throw(ArgumentError("$name contains $(length(bars)) barcodes; pass index."))
        return first(eachindex(bars))
    end
    if index isa Tuple
        all(i -> i isa Integer && !(i isa Bool),index) || throw(ArgumentError("index must contain integers."))
        checkbounds(Bool,bars,index...) || throw(ArgumentError("index is outside the barcode family."))
        return CartesianIndex(index)
    end
    index isa Integer && !(index isa Bool) && checkbounds(Bool,bars,index) ||
        throw(ArgumentError("index must identify an existing barcode in the family."))
    return Int(index)
end

function _interval_payload(diag::OrdinaryPersistence.PersistenceDiagram;dim=0,index=nothing)
    index === nothing || throw(ArgumentError("Ordinary persistence selects its barcode by dim, not index."))
    # The ordinary recipe retains its established strict conversion contract.
    data = _ordinary_display_data(diag,dim)
    records = NamedTuple[]
    for (i,(b,d)) in enumerate(data.finite)
        lower,upper = data.order === :sublevel ? (b,d) : (d,b)
        push!(records,_interval_record(lower,upper,1;
            left_closed=data.order === :sublevel,right_closed=data.order === :superlevel,
            members=((;dim=Int(dim),kind=:finite,index=i),)))
    end
    for (i,b) in enumerate(data.essential)
        lower,upper = data.order === :sublevel ? (b,Inf) : (-Inf,b)
        push!(records,_interval_record(lower,upper,1;
            left_closed=data.order === :sublevel,right_closed=data.order === :superlevel,
            members=((;dim=Int(dim),kind=:essential,index=i),)))
    end
    retained = OrdinaryPersistence.persistence_diagram_summary(diag).representatives_available
    return (;records=_group_interval_records(records),order=data.order,
        endpoint_semantics=:ordinary_half_open,essential_status=:certified,
        metadata=(;source=:ordinary_persistence,homological_dimension=Int(dim),order=data.order,
            finite_intervals=copy(data.finite),essential_births=copy(data.essential),
            finite_count=length(data.finite),essential_count=length(data.essential),
            rounded_endpoint_count=data.rounded_endpoint_count,
            essential_direction=data.order === :sublevel ? 1 : -1,
            interval_convention=data.order === :sublevel ? "[birth, death)" : "(death, birth]",
            representative_available=retained,representative_reason=retained ? :retained : :not_retained))
end

function _interval_payload(result::InvariantResult;dim=0,index=nothing)
    result.which in (:slice_barcode,:slice_barcodes) ||
        throw(ArgumentError("This invariant result does not contain an interval decomposition."))
    return _interval_payload(invariant_value(result);dim,index)
end

function _interval_payload_spec(payload,kind;window=nothing,interval=nothing,max_intervals=200,
                                barcode_title="Barcode",diagram_title="Persistence diagram")
    kind in (:barcode,:persistence_diagram) || throw(ArgumentError("Interval views support :barcode or :persistence_diagram."))
    panels = _interval_panels(payload.records;window,interval,max_intervals,order=payload.order,
        barcode_title,diagram_title,endpoint_semantics=payload.endpoint_semantics,
        essential_status=payload.essential_status,metadata=payload.metadata)
    spec = panels[kind === :barcode ? 1 : 2]
    if get(payload.metadata,:source,nothing) === :ordinary_persistence
        lane = payload.order === :sublevel ? spec.metadata.infinity_lanes.positive : spec.metadata.infinity_lanes.negative
        subtitle = (payload.order === :sublevel ? "Sublevel" : "Superlevel") * " · " * spec.subtitle
        rounded = get(payload.metadata,:rounded_endpoint_count,0)
        rounded > 0 && (subtitle *= "\nFloat64 display rounds $rounded endpoints; exact values remain in metadata.")
        return VisualizationSpec(spec.kind;title=spec.title,subtitle,layers=spec.layers,axes=spec.axes,
            legend=spec.legend,interaction=spec.interaction,
            metadata=merge(spec.metadata,(;essential_display_coordinate=lane)))
    end
    return spec
end
