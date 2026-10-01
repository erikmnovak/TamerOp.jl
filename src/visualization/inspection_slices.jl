# Exact line restrictions for the linked inspector. A midpoint alone
# cannot retain closed deaths or singleton intervals: each event point is a
# separate stratum between its adjacent open intervals.

function _inspection_slice_capability(obj, geometry_available)
    geometry_available || return (false, "A slice requires a supported planar encoding.")
    pi = _inspection_classifier(encoding_map(obj))
    if pi isa GridEncodingMap && any(!=(1), pi.orientation)
        return (false, "Linked slices currently require positively oriented grid axes.")
    end
    return (true, "")
end

function _inspection_slice_config(value)
    value isa NamedTuple && length(value) == 2 && haskey(value, :basepoint) && haskey(value, :direction) ||
        throw(ArgumentError("slice must be (basepoint=(x,y), direction=(dx,dy)), or false to disable."))
    for name in (:basepoint, :direction)
        point = value[name]
        (point isa Tuple || point isa AbstractVector) && length(point) == 2 &&
            all(x -> x isa Real && !(x isa Bool) && isfinite(x), point) ||
            throw(ArgumentError("slice $name must contain two finite real coordinates."))
    end
    basepoint, direction = _visual_point(value.basepoint), _visual_point(value.direction)
    all(>=(0), direction) && any(>(0), direction) || throw(ArgumentError(
        "slice direction must be nonnegative and nonzero; it is not normalized."))
    _drawing_point(basepoint)
    _drawing_point(direction)
    return (; basepoint, direction)
end

function _inspection_slice_coordinates(obj, line, box)
    pi = _inspection_classifier(encoding_map(obj))
    pi isa PLEncodingMap || return line
    # The polyhedral public classifier verifies rational queries exactly.
    # Its generic Real conversion does not preserve irrational algebraic
    # values, so reject those rather than duplicate owner membership kernels.
    try
        for corner in box, coordinate in corner
            _VisualRational(coordinate)
        end
        return (; basepoint=Tuple(_VisualRational.(line.basepoint)),
            direction=Tuple(_VisualRational.(line.direction)))
    catch err
        err isa InexactError || rethrow()
        throw(ArgumentError("Polyhedral linked slices currently require rational line and viewing-box " *
            "coordinates; irrational algebraic coordinates are unsupported by the verified classifier."))
    end
end

function _inspection_slice_window(line, box)
    lower, upper = nothing, nothing
    for i in 1:2
        b, d = line.basepoint[i], line.direction[i]
        if iszero(d)
            box[1][i] <= b <= box[2][i] || return nothing
        else
            lo, hi = (box[1][i] - b) / d, (box[2][i] - b) / d
            lower = lower === nothing ? lo : max(lower, lo)
            upper = upper === nothing ? hi : min(upper, hi)
        end
    end
    return lower <= upper ? (lower, upper) : nothing
end

_inspection_slice_cross(a, b) = a[1] * b[2] - a[2] * b[1]

function _inspection_slice_events(geometry, line, window, limit)
    lo, hi = window
    events = Set{_VisualCoordinate}((lo, hi))
    d, b = line.direction, line.basepoint
    axis = iszero(d[1]) ? 2 : 1
    function retain(t)
        lo <= t <= hi || return nothing
        push!(events, t)
        2 * length(events) - 1 <= limit || throw(ArgumentError(
            "The slice exceeds slice_limit=$limit event strata. Choose a smaller viewing box " *
            "or reopen with a larger slice_limit; barcode rank work grows quadratically."))
        return nothing
    end
    # The prepared cells include their exact clipped faces and exceptional
    # lower-dimensional fibers. Intersections with every edge isolate all
    # changes of the actual classifier, including a line lying along a face.
    for component in geometry.components
        vertices = component.vertices
        for p in vertices
            relative = (p[1] - b[1], p[2] - b[2])
            iszero(_inspection_slice_cross(relative, d)) && retain(relative[axis] / d[axis])
        end
        pairs = component.dimension == 2 ?
            ((i, mod1(i + 1, length(vertices))) for i in eachindex(vertices)) :
            (component.dimension == 1 ? ((1, 2),) : ())
        for (i, j) in pairs
            p, q = vertices[i], vertices[j]
            edge = (q[1] - p[1], q[2] - p[2])
            denominator = _inspection_slice_cross(d, edge)
            iszero(denominator) && continue
            relative = (p[1] - b[1], p[2] - b[2])
            t = _inspection_slice_cross(relative, edge) / denominator
            u = _inspection_slice_cross(relative, d) / denominator
            0 <= u <= 1 && retain(t)
        end
    end
    return sort!(collect(events))
end

function _inspection_slice_global_events(pi, line, limit)
    events = Set{_VisualCoordinate}()
    function retain(t)
        push!(events, t)
        2 * length(events) + 1 <= limit || throw(ArgumentError(
            "The global slice exceeds slice_limit=$limit event strata. Reopen with a " *
            "larger slice_limit or use slice_scope=:window; barcode rank work grows quadratically."))
    end
    if pi isa PLEncodingMap
        # Every membership predicate is constant between these hyperplanes,
        # even if its intersection lies outside the drawn viewing box. Strict
        # stored bounds restore the original halfspace before intersecting.
        for region in pi.regions, i in axes(region.A, 1)
            slope = sum(region.A[i,j] * line.direction[j] for j in 1:2)
            iszero(slope) && continue
            bound = region.b[i] + (region.strict_mask[i] ? region.strict_eps : zero(region.strict_eps))
            origin = sum(region.A[i,j] * line.basepoint[j] for j in 1:2)
            retain((bound - origin) / slope)
        end
    else
        splits = pi isa GridEncodingMap ? encoding_axes(pi) : critical_coordinates(pi)
        for i in 1:2
            iszero(line.direction[i]) && continue
            for coordinate in splits[i]
                boundary = _visual_exact_coordinate(coordinate)
                # Match the existing exact nearest-lattice extension. The
                # half-integer itself is a separate stratum, retaining ties.
                pi isa ZnEncodingMap && (boundary -= 1//2)
                retain((boundary - line.basepoint[i]) / line.direction[i])
            end
        end
    end
    return sort!(collect(events))
end

_inspection_slice_locate(pi, point) = _visual_locate(pi, point)
# Canonicalize exactly before the public verified path so even a rational
# AlgebraicReal event cannot enter the owner's approximate generic conversion.
_inspection_slice_locate(pi::PLEncodingMap, point) =
    locate(pi, Tuple(_VisualRational.(point)); mode=:verified)

# Zn's public classifier stores machine-integer slab coordinates. A global
# tail sample can lie arbitrarily far away. Replace it by a lattice point in
# the same outer slab before calling that classifier, rather than overflowing
# round(Int, ...) or changing which boundary owns a nearest-lattice tie.
function _inspection_slice_locate(pi::ZnEncodingMap, point)
    splits = critical_coordinates(pi)
    representative = ntuple(2) do i
        isempty(splits[i]) && return 0
        lower = BigInt(first(splits[i])) - 1
        upper = BigInt(last(splits[i]))
        value = clamp(round(BigInt, point[i]), lower, upper)
        typemin(Int) <= value <= typemax(Int) || throw(ArgumentError(
            "The nearest-lattice slice needs an outer-slab representative outside the " *
            "classifier's machine-integer coordinate range; use interior coordinates."))
        return Int(value)
    end
    return locate(pi, representative)
end

function _inspection_slice_result(obj, prepared, line, limit; scope=:window)
    scope in (:window, :global) || throw(ArgumentError("slice_scope must be :window or :global."))
    geometry = prepared.geometry
    window = _inspection_slice_window(line, geometry.box)
    global_scope = scope === :global
    common = (; line, window, box=geometry.box, scope,
        domain=global_scope ? (-Inf, Inf) : window,
        endpoint_semantics=global_scope ? :decorated_global : :decorated_finite_window,
        essential_status=global_scope ? :certified : :not_inferred,
        exact_geometry=true, coefficient_field=_inspection_field_label(_inspection_field(obj)))
    !global_scope && window === nothing && return merge(common,
        (; intervals=(), events=(), chain=(), sample_parameters=()))
    pi = _inspection_classifier(encoding_map(obj))
    events = global_scope ? _inspection_slice_global_events(pi, line, limit) :
        _inspection_slice_events(geometry, line, window, limit)
    all(t -> isfinite(Float64(t)), events) || throw(ArgumentError(
        "Slice parameters must be representable as finite Float64 values for drawing; " *
        "rescale the line direction or basepoint."))
    parameters = _VisualCoordinate[]
    # These are representatives of proved-constant open tails, not finite
    # endpoints extrapolated from the final sample spacing.
    global_scope && push!(parameters, isempty(events) ? zero(_VisualRational) : first(events) - 1)
    for i in eachindex(events)
        push!(parameters, events[i])
        i < length(events) && push!(parameters, (events[i] + events[i+1]) / 2)
    end
    global_scope && !isempty(events) && push!(parameters, last(events) + 1)
    length(parameters) <= limit || throw(ArgumentError(
        "The slice exceeds slice_limit=$limit event strata; reopen with a larger slice_limit."))
    chain = Int[]
    for t in parameters
        point = ntuple(i -> line.basepoint[i] + t * line.direction[i], 2)
        label = _inspection_slice_locate(pi, point)
        label > 0 || throw(ArgumentError(global_scope ?
            "The global slice meets an unrepresented parameter at t=$t. Whole-line endpoints " *
            "cannot be certified; use slice_scope=:window with a fully represented viewing " *
            "box. Missing labels are not interpreted as zero spaces." :
            "The slice meets an unrepresented parameter at t=$t. Choose a fully represented " *
            "viewing box; missing labels are not interpreted as zero spaces."))
        push!(chain, label)
    end
    # Materialization is explicit slice work, never hover or interval selection.
    # Integer barcode endpoints index the alternating point/open strata. Only
    # proved-constant complete-line tails justify infinite real endpoints.
    M = Results.encoding_module(obj)
    bars = slice_barcode(M, chain; values=nothing, check_chain=true)
    records = NamedTuple[]
    for ((first_index, after_index), multiplicity) in bars
        last_index = after_index - 1
        if global_scope
            left_closed, right_closed = iseven(first_index), iseven(last_index)
            birth = first_index == 1 ? -Inf : events[div(first_index, 2)]
            death = last_index == length(parameters) ? Inf : events[div(last_index + 1, 2)]
        else
            left_closed, right_closed = isodd(first_index), isodd(last_index)
            birth = events[div(first_index, 2) + 1]
            left_closed || (birth = events[div(first_index, 2)])
            death = events[div(last_index, 2) + 1]
        end
        push!(records, (; birth, death, left_closed, right_closed, multiplicity,
            left_clipped=!global_scope && birth == window[1],
            right_clipped=!global_scope && death == window[2],
            singleton=birth == death))
    end
    sort!(records; by=r -> (r.birth, r.death, !r.left_closed, !r.right_closed))
    intervals = Tuple(merge((; id), record) for (id, record) in enumerate(records))
    return merge(common, (; intervals, events=Tuple(events), chain=Tuple(chain),
        sample_parameters=Tuple(parameters)))
end

function _inspection_slice_validate_interval(result, interval)
    interval === nothing && return nothing
    result === nothing && throw(ArgumentError("An interval selection requires an active slice."))
    interval <= length(result.intervals) || throw(ArgumentError(
        "interval must be a group ID in 1:$(length(result.intervals)), or 0 to clear."))
    return nothing
end
