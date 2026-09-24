# Minimum-enclosing-ball radii for finite planar/linear simplices. Containment
# decisions use filtered predicates with exact fallback on the input Float64s;
# the final Euclidean radius is rounded to Float64, as in the Delaunay backend.

@inline function _diameter_ball_contains(a, b, p)
    # p is in the closed ball with diameter ab iff (p-a).(p-b) <= 0.
    value = 0.0
    permanent = 0.0
    for d in eachindex(a)
        product = (Float64(p[d]) - Float64(a[d])) * (Float64(p[d]) - Float64(b[d]))
        value += product
        permanent += abs(product)
    end
    if isfinite(permanent) && permanent >= floatmin(Float64) &&
       abs(value) > 64eps(Float64) * permanent
        return value < 0
    end
    exact = zero(Rational{BigInt})
    for d in eachindex(a)
        pq = Rational{BigInt}(Float64(p[d]))
        exact += (pq - Rational{BigInt}(Float64(a[d]))) *
                 (pq - Rational{BigInt}(Float64(b[d])))
    end
    return exact <= 0
end

@inline function _half_point_distance(a, b)
    radius = 0.0
    for d in eachindex(a)
        radius = hypot(radius, _half_coordinate_difference(a[d], b[d]))
    end
    if radius < floatmin(Float64)
        # Halving each subnormal component separately can erase a diagonal
        # whose final radius is representable. Normalize before halving here.
        scale = maximum(d -> abs(Float64(a[d]) - Float64(b[d])), eachindex(a))
        if scale > 0.0
            normalized = 0.0
            for d in eachindex(a)
                normalized = hypot(normalized, (Float64(a[d]) - Float64(b[d])) / scale)
            end
            radius = scale * (normalized / 2)
        end
    end
    isfinite(radius) || throw(ArgumentError("Ball radius exceeds the Float64 range; rescale the coordinates."))
    return radius
end

function _simplex_enclosing_radius(points, vertices)
    count = length(vertices)
    count > 0 || throw(ArgumentError("An enclosing ball requires a nonempty simplex."))
    count == 1 && return 0.0
    dimension = length(points[first(vertices)])
    dimension in (1, 2) || throw(ArgumentError("Enclosing-ball geometry supports ambient dimensions 1 and 2."))
    count == 2 && return _half_point_distance(points[vertices[1]], points[vertices[2]])
    if dimension == 1
        lo, hi = extrema(Float64(points[v][1]) for v in vertices)
        return _half_coordinate_difference(hi, lo)
    end
    # A minimum planar enclosing disk is supported by two or three points.
    # Test diameter disks first: this also handles repeated/collinear points.
    best = Inf
    lower = 0.0
    for i in 1:count-1, j in i+1:count
        a, b = points[vertices[i]], points[vertices[j]]
        radius = _half_point_distance(a, b)
        lower = max(lower, radius)
        if radius < best && all(v -> _diameter_ball_contains(a, b, points[v]), vertices)
            best = radius
        end
    end
    isfinite(best) && return max(best, lower)
    for i in 1:count-2, j in i+1:count-1, k in j+1:count
        a, b, c = points[vertices[i]], points[vertices[j]], points[vertices[k]]
        orientation = _delaunay_orient_sign(a, b, c)
        orientation == 0 && continue
        # Skip the defining vertices: their incircle determinant vanishes
        # identically and would otherwise trigger unnecessary exact arithmetic.
        contains = true
        for q in eachindex(vertices)
            (q == i || q == j || q == k) && continue
            side = _delaunay_incircle_sign(a, b, c, points[vertices[q]])
            if side != 0 && side != orientation
                contains = false
                break
            end
        end
        contains || continue
        best = min(best, _delaunay_circumradius(a, b, c))
    end
    isfinite(best) || throw(ArgumentError("Could not compute a finite enclosing radius; rescale the coordinates."))
    return max(best, lower)
end

# Roundoff must not put a coface before one of its faces. Taking face maxima
# changes no mathematical radius and limits any correction to rounding error.
function _enclosing_radii(points, simplices)
    radii = [Float64[] for _ in simplices]
    previous = Dict{Tuple{Vararg{Int}},Float64}()
    for d in eachindex(simplices)
        current = Dict{Tuple{Vararg{Int}},Float64}()
        for simplex in simplices[d]
            radius = _simplex_enclosing_radius(points, simplex)
            if d > 1
                for omit in eachindex(simplex)
                    face = Tuple(simplex[q] for q in eachindex(simplex) if q != omit)
                    radius = max(radius, previous[face])
                end
            end
            push!(radii[d], radius)
            current[Tuple(simplex)] = radius
        end
        previous = current
    end
    return radii
end
