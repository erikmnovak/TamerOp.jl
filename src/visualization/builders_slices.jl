# Decorated window and whole-line slice views. Exact intervals remain in metadata;
# only the drawing coordinates below are converted to Float64.

function _inspection_slice_record_label(record)
    left = record.left_closed ? "[" : "("
    right = record.right_closed ? "]" : ")"
    cuts = record.left_clipped && record.right_clipped ? " (both window cuts)" :
        record.left_clipped ? " (left window cut)" : record.right_clipped ? " (right window cut)" : ""
    return "#$(record.id) " * left * _interval_value(record.birth) * ", " *
        _interval_value(record.death) * right * " x$(record.multiplicity)" * cuts
end

function _inspection_slice_view(result, interval=nothing)
    view = _interval_view(result.intervals; window=result.window, interval)
    line_points = if result.window === nothing
        ()
    else
        a, d = result.line.basepoint, result.line.direction
        Tuple(_drawing_point((a[1] + t*d[1], a[2] + t*d[2])) for t in result.window)
    end
    return merge(view, (; line_points, all_records=view.records, records=view.displayed_records))
end

# Intersect an actual interval with the parameter viewport before drawing. In
# particular, no infinite endpoint may enter a coordinate calculation as 0*Inf.
function _inspection_slice_segment(result, record)
    result.window === nothing && return nothing
    lo, hi = max(record.birth, result.window[1]), min(record.death, result.window[2])
    lo <= hi || return nothing
    if lo == hi
        lo == record.birth && !record.left_closed && return nothing
        hi == record.death && !record.right_closed && return nothing
    end
    base, d = result.line.basepoint, result.line.direction
    return Tuple(_drawing_point((base[1]+t*d[1], base[2]+t*d[2])) for t in (lo,hi))
end

function _inspection_slice_panels(result, interval=nothing; highlight=true)
    barcode, diagram = _interval_panels(result.intervals; window=result.window, interval, highlight,
        barcode_kind=:slice_barcode, diagram_kind=:slice_diagram,
        barcode_title="Slice barcode", diagram_title="Decorated slice diagram",
        endpoint_semantics=result.endpoint_semantics, essential_status=result.essential_status,
        metadata=(; slice_scope=get(result,:scope,:window)))
    if result.window === nothing && get(result,:scope,:window) === :window
        panels = map((barcode,diagram)) do panel
            VisualizationSpec(panel.kind; title=panel.title,
                subtitle="Slice misses the viewing box. " * panel.subtitle,
                layers=panel.layers, axes=panel.axes, legend=panel.legend,
                interaction=panel.interaction, metadata=panel.metadata)
        end
        return panels
    end
    return (barcode,diagram)
end

function _inspection_slice_snapshot(spec, result, interval)
    result === nothing && return spec
    view = _inspection_slice_view(result, interval)
    panels = VisualizationSpec[]
    for panel in spec.panels
        if panel.kind in (:regions, :region_labels, :query_overlay) && length(view.line_points) == 2
            a, b = view.line_points
            layers = copy(panel.layers)
            push!(layers, SegmentLayer([(a[1],a[2],b[1],b[2])], _VisualRole(:foreground), 0.9, 2.0, :dash))
            if interval !== nothing
                record = result.intervals[interval]
                segment = _inspection_slice_segment(result, record)
                if segment !== nothing
                    p, q = segment
                    push!(layers, SegmentLayer([(p[1],p[2],q[1],q[2])], _VisualRole(:selected), 1.0, 5.0))
                    p == q && push!(layers, PointLayer([p], _VisualRole(:selected), 1.0, 12.0))
                end
            end
            panel = VisualizationSpec(panel.kind; title=panel.title, subtitle=panel.subtitle,
                layers, axes=panel.axes, legend=panel.legend, interaction=panel.interaction,
                metadata=merge(panel.metadata, (; slice_line=result.line)))
        end
        push!(panels, panel)
    end
    append!(panels, _inspection_slice_panels(result, interval))
    a, d = result.line.basepoint, result.line.direction
    line_label = "q(t) = (" * join(_interval_value.(a), ", ") * ") + t (" *
        join(_interval_value.(d), ", ") * ")."
    scope_label = get(result,:scope,:window) === :global ?
        " Whole-line restriction; the parameter view clips its geometry." :
        " Results are restricted to the viewing box; window endpoints are censored."
    subtitle = spec.subtitle * "\nSlice: " * line_label * scope_label
    interval === nothing || (subtitle *= "\nSelected " * _inspection_slice_record_label(result.intervals[interval]))
    return VisualizationSpec(spec.kind; title=spec.title, subtitle, panels,
        axes=spec.axes, legend=spec.legend, interaction=spec.interaction,
        metadata=merge(spec.metadata, (; slice_result=result, slice_view=view,
            panel_columns=2, figure_size=(1800,max(1100,450*cld(length(panels),2))))))
end
