# Exact physical-coordinate geometry for algebraic grid grades. Arrangement
# cells, their representatives and their endpoints retain exact coordinates.
@inline _algebraic_grid(pi) = hasproperty(pi, :coords) &&
    any(a -> eltype(a) <: AlgebraicReal, pi.coords)
@inline _algebraic_arrangement(arr) = eltype(arr.box[1]) <: AlgebraicReal

function _exact_grade_direction(dir, normalization)
    length(dir) == 2 || throw(ArgumentError("expected a two-dimensional direction"))
    d = AlgebraicReal.(collect(dir))
    any(!iszero, d) || throw(ArgumentError("zero direction is not a line"))
    if normalization === :L1
        d ./= sum(abs, d)
    elseif normalization === :Linf
        d ./= maximum(abs, d)
    elseif normalization !== :none
        throw(ArgumentError("normalize_dirs must be :none, :L1, or :Linf"))
    end
    return d
end

function _exact_arrangement_geometry(pi, box, normalization, include_axes)
    orientation = hasproperty(pi, :orientation) ? pi.orientation : (1,1)
    coordinates = [sort!(unique!(AlgebraicReal[orientation[i]*x for x in pi.coords[i] if isfinite(x)])) for i in 1:2]
    xs = [x for x in coordinates[1] if box[1][1] <= x <= box[2][1]]
    ys = [y for y in coordinates[2] if box[1][2] <= y <= box[2][2]]
    append!(xs,(box[1][1],box[2][1])); append!(ys,(box[1][2],box[2][2]))
    sort!(xs); unique!(xs); sort!(ys); unique!(ys)
    points = NTuple{2,AlgebraicReal}[(x,y) for x in xs for y in ys]
    sort!(points); unique!(points)
    oriented = sort!([(orientation[1]*p[1],orientation[2]*p[2]) for p in points])
    slopes = AlgebraicReal[]
    for i in eachindex(oriented), j in i+1:length(oriented)
        dx,dy = oriented[j][1]-oriented[i][1],oriented[j][2]-oriented[i][2]
        dx > 0 && dy > 0 && push!(slopes,dy/dx)
    end
    sort!(slopes); unique!(slopes)
    representatives = isempty(slopes) ? AlgebraicReal[1] :
        [slopes[1]/2; [(slopes[i]+slopes[i+1])/2 for i in 1:length(slopes)-1]; slopes[end]+1]
    directions = [_exact_grade_direction([one(AlgebraicReal),s],normalization) for s in representatives]
    if include_axes
        push!(directions,_exact_grade_direction([1,0],normalization))
        push!(directions,_exact_grade_direction([0,1],normalization))
    end
    for d in directions
        d[1] *= orientation[1]; d[2] *= orientation[2]
    end
    return points,slopes,directions,coordinates[1],coordinates[2]
end

function _line_order(points::Vector{NTuple{2,AlgebraicReal}}, dir::AbstractVector{AlgebraicReal};
                     strict::Bool=true, atol::Float64=0.0)
    return sort!(collect(eachindex(points)); by=i ->
        (-dir[2]*points[i][1]+dir[1]*points[i][2],points[i][1],points[i][2]))
end

function _unique_positions_for_order(points::Vector{NTuple{2,AlgebraicReal}}, order::Vector{Int},
                                     dir::AbstractVector{AlgebraicReal}; atol::Float64=0.0)
    positions = Int[]
    previous = nothing
    for (k,i) in enumerate(order)
        value = -dir[2]*points[i][1]+dir[1]*points[i][2]
        if previous === nothing || value != previous
            push!(positions,k)
            previous = value
        end
    end
    return positions
end

function _exact_grade_slice(pi, dir, offset, box; normalize_dirs=:L1, strict=true)
    d = _exact_grade_direction(dir, normalize_dirs)
    orientation = hasproperty(pi, :orientation) ? pi.orientation : (1, 1)
    all(i -> orientation[i]*d[i] >= 0, 1:2) || throw(ArgumentError(
        "slice direction must increase the oriented filtration coordinates"))
    n = (-d[2], d[1])
    c = AlgebraicReal(offset)/(d[1]^2+d[2]^2)
    x0 = (c*n[1], c*n[2])
    a, b = AlgebraicReal.(box[1]), AlgebraicReal.(box[2])
    all(a .<= b) || throw(ArgumentError("slice window endpoints must be ordered"))
    lower = nothing
    upper = nothing
    for i in 1:2
        if iszero(d[i])
            a[i] <= x0[i] <= b[i] || return Int[], AlgebraicReal[]
        else
            u, v = minmax((a[i]-x0[i])/d[i], (b[i]-x0[i])/d[i])
            lower = lower === nothing ? u : max(lower, u)
            upper = upper === nothing ? v : min(upper, v)
        end
    end
    lower < upper || return Int[], AlgebraicReal[]
    events = AlgebraicReal[lower, upper]
    for i in 1:2
        iszero(d[i]) && continue
        for cut in pi.coords[i]
            isfinite(cut) || continue
            t = (orientation[i]*AlgebraicReal(cut)-x0[i])/d[i]
            lower < t < upper && push!(events, t)
        end
    end
    sort!(events); unique!(events)
    labels = Int[]
    for j in 1:length(events)-1
        t = (events[j]+events[j+1])/2
        push!(labels, locate(pi, [x0[i]+t*d[i] for i in 1:2]))
    end
    return _chain_values_from_boundaries(labels, events; strict=strict, atol=0.0)
end

function _exact_grade_barcode(cache, dir, offset; values=:t, packed=false, tie_break=:up)
    tie_break in (:up, :down) || throw(ArgumentError("tie_break must be :up or :down"))
    values in (:t, :index) || throw(ArgumentError("values must be :t or :index"))
    arr = cache.arrangement
    d = _exact_grade_direction(dir,arr.normalize_dirs)
    _fibered_dir_cell_index(arr,d) # Enforce the arrangement's axis/orientation contract.
    chain, vals = _exact_grade_slice(arr.pi, d, offset, arr.input_box;
        normalize_dirs=:none, strict=arr.strict)
    index = isempty(chain) ? _empty_packed_index_barcode() :
        _index_packed_for_chain!(cache, _arr2d_chain_id!(arr, chain))
    result = if values === :index
        index
    else
        pairs = [EndpointPair{AlgebraicReal}(vals[p.b], vals[p.d]) for p in index.pairs]
        PackedBarcode{AlgebraicReal}(pairs, copy(index.mults))
    end
    return chain, vals, packed ? result : _barcode_from_packed(result)
end
