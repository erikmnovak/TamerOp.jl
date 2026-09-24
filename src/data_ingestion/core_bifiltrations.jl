# Neighbor-distance grading for the core Cech and Delaunay-core bifiltrations.
# Private implementation fragment of DataIngestion.
#
# Blaser, Brun, Gardaa, Salbu, "Core Bifiltration", arXiv:2405.01214v4,
# Definitions 3.1--3.4 and Algorithm 2.  A point is its own first neighbor.
# The parameter order is increasing radius and decreasing neighbor count.

function _core_beta(beta)
    beta isa Real && !(beta isa Bool) ||
        throw(ArgumentError("core beta must be a finite positive real number."))
    value = Float64(beta)
    isfinite(value) && value > 0.0 ||
        throw(ArgumentError("core beta must be finite and positive (got $(beta))."))
    return value
end

function _core_k_values(k_values, n::Int)
    n > 0 || throw(ArgumentError("core filtrations require a nonempty point cloud."))
    k_values === nothing && return collect(1:n)
    k_values isa Union{AbstractVector,Tuple} ||
        throw(ArgumentError("core k_values must be a nonempty collection of integers in 1:$(n)."))
    isempty(k_values) && throw(ArgumentError("core k_values cannot be empty."))
    values = Vector{Int}(undef, length(k_values))
    for (i, k) in enumerate(k_values)
        k isa Integer && !(k isa Bool) && 1 <= k <= n ||
            throw(ArgumentError("core k_values must contain integers in 1:$(n); got $(k)."))
        values[i] = Int(k)
    end
    sort!(values)
    @inbounds for i in 2:length(values)
        values[i] != values[i - 1] ||
            throw(ArgumentError("core k_values must not contain duplicates."))
    end
    return values
end

"""
    _core_neighbor_distances(points, ks)

Return `distances[v,j] = d_ks[j](v)`, with the point itself counted among
its neighbors. `points` stores one point per row; `ks` is a validated sorted
selection from `1:size(points,1)`. Selection is exact on the computed Euclidean
distances; no approximate nearest-neighbor backend is used.

The result uses `n * length(ks)` numbers and one length-`n` scratch vector.
Selecting only some `ks` computes those density slices exactly. Extending their
births to unselected density values gives a step extension of the sampled
filtration, not generally the full core bifiltration at those values.
"""
function _core_neighbor_distances(points::AbstractMatrix{<:Real}, ks::Vector{Int})
    n, dimension = size(points)
    n > 0 && dimension > 0 ||
        throw(ArgumentError("core filtrations require a nonempty point cloud of positive ambient dimension."))
    isempty(ks) && throw(ArgumentError("core k_values cannot be empty."))
    previous = 0
    for k in ks
        previous < k <= n ||
            throw(ArgumentError("core neighbor distance columns require strictly increasing k_values in 1:$(n)."))
        previous = k
    end
    for coordinate in points
        isfinite(Float64(coordinate)) ||
            throw(ArgumentError("core point coordinates must be finite and representable as Float64."))
    end

    distances = Matrix{Float64}(undef, n, length(ks))
    scratch = Vector{Float64}(undef, n)
    @inbounds for v in 1:n
        for w in 1:n
            distance = 0.0
            for axis in 1:dimension
                # hypot avoids squaring overflow/underflow for finite
                # representable distances on clouds with extreme scales.
                delta = Float64(points[v, axis]) - Float64(points[w, axis])
                distance = hypot(distance, delta)
            end
            isfinite(distance) ||
                throw(ArgumentError("core pairwise distance exceeds the Float64 range."))
            scratch[w] = distance
        end
        partialsort!(scratch, 1:last(ks))
        for j in eachindex(ks)
            distances[v, j] = scratch[ks[j]]
        end
    end
    return distances
end

"""
    _core_simplex_multigrades(vertices, base_radius, distances, ks, beta)

Compute the minimal births of one simplex in order `(radius, neighbor_count)`
with orientation `(1,-1)`. The unpruned birth at `ks[j]` is
`(max(base_radius, beta * maximum(distances[v,j] for v in vertices)), ks[j])`.
The base radius is the minimum enclosing ball radius for the core Cech nerve,
or the alpha radius for the Delaunay-core nerve.
"""
function _core_simplex_multigrades(vertices, base_radius::Real,
                                    distances::AbstractMatrix{<:Real},
                                    ks::Vector{Int}, beta::Real)
    beta_value = _core_beta(beta)
    radius = Float64(base_radius)
    isfinite(radius) && radius >= 0.0 ||
        throw(ArgumentError("core simplex base radius must be finite and nonnegative."))
    isempty(vertices) && throw(ArgumentError("core simplex must contain at least one vertex."))
    n, count = size(distances)
    count == length(ks) ||
        throw(DimensionMismatch("core distance columns and k_values must have equal length."))
    count > 0 || throw(ArgumentError("core k_values cannot be empty."))
    for v in vertices
        v isa Integer && !(v isa Bool) && 1 <= v <= n ||
            throw(ArgumentError("core simplex vertex $(v) is outside 1:$(n)."))
    end
    return _core_simplex_multigrades_validated(vertices, radius, distances, ks, beta_value)
end

function _core_simplex_multigrades_validated(vertices, base_radius::Float64,
                                              distances::AbstractMatrix{<:Real},
                                              ks::Vector{Int}, beta::Float64)
    out = NTuple{2,Float64}[]
    sizehint!(out, length(ks))
    previous_k = 0
    previous_radius = -Inf
    @inbounds for j in eachindex(ks)
        k = ks[j]
        previous_k < k <= size(distances, 1) ||
            throw(ArgumentError("core simplex grading requires strictly increasing k_values in the point-cloud range."))
        max_distance = 0.0
        for v in vertices
            distance = Float64(distances[v, j])
            isfinite(distance) && distance >= 0.0 ||
                throw(ArgumentError("core neighbor distances must be finite and nonnegative."))
            max_distance = max(max_distance, distance)
        end
        radius = max(base_radius, beta * max_distance)
        isfinite(radius) || throw(ArgumentError("core birth radius exceeds the Float64 range."))
        radius >= previous_radius ||
            throw(ArgumentError("core neighbor distances must be nondecreasing with k."))

        # Since k increases, only equal-radius births can be dominated.
        # Larger k is earlier in the reverse density order and subsumes all
        # preceding births at the same radius. No tolerance is appropriate.
        if radius == previous_radius
            out[end] = (radius, Float64(k))
        else
            push!(out, (radius, Float64(k)))
        end
        previous_k = k
        previous_radius = radius
    end
    return out
end

function _core_multigrades(simplices_by_dim, radii_by_dim,
                             distances::AbstractMatrix{<:Real},
                             ks::Vector{Int}, beta::Real)
    length(simplices_by_dim) == length(radii_by_dim) ||
        throw(DimensionMismatch("core simplex and radius dimension counts must agree."))
    # Validate shared inputs once; each simplex still checks its own radius
    # and vertices before the unchecked vertex-indexing kernel runs.
    beta_value = _core_beta(beta)
    n, count = size(distances)
    count == length(ks) ||
        throw(DimensionMismatch("core distance columns and k_values must have equal length."))
    count > 0 || throw(ArgumentError("core k_values cannot be empty."))
    grades = Vector{Vector{NTuple{2,Float64}}}()
    sizehint!(grades, sum(length, simplices_by_dim))
    for dim in eachindex(simplices_by_dim)
        simplices = simplices_by_dim[dim]
        radii = radii_by_dim[dim]
        length(simplices) == length(radii) ||
            throw(DimensionMismatch("core simplex and radius counts differ in dimension $(dim - 1)."))
        for i in eachindex(simplices)
            simplex = simplices[i]
            isempty(simplex) && throw(ArgumentError("core simplex must contain at least one vertex."))
            for v in simplex
                v isa Integer && !(v isa Bool) && 1 <= v <= n ||
                    throw(ArgumentError("core simplex vertex $(v) is outside 1:$(n)."))
            end
            radius = Float64(radii[i])
            isfinite(radius) && radius >= 0.0 ||
                throw(ArgumentError("core simplex base radius must be finite and nonnegative."))
            push!(grades, _core_simplex_multigrades_validated(simplex, radius, distances, ks, beta_value))
        end
    end
    return grades
end
