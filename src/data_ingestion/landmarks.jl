# Landmark selection and coverage for finite Euclidean clouds or dense distances.

"""
    LandmarkSelection

A selected subset of the original points, in selection order, and its coverage.
Construct with [`select_landmarks`](@ref). Use `landmark_indices`,
`covering_radius`, `landmark_distances` and `describe` to inspect it. Distances
use Float64; a matrix is checked for symmetry and nonnegativity, not the triangle
inequality. The usual Rips bottleneck bound is conditional on a metric input.
"""
struct LandmarkSelection
    indices::Vector{Int}
    insertion_radii::Vector{Float64}
    nearest_distances::Vector{Float64}
    nearest_indices::Vector{Int}
    distances::Union{Nothing,Matrix{Float64}}
    geometry::Symbol
end

# A saved selection is the same mathematical/source record after reloading,
# even though its arrays are independent allocations.
Base.:(==)(a::LandmarkSelection,b::LandmarkSelection) =
    all(name -> getfield(a,name)==getfield(b,name),fieldnames(LandmarkSelection))
Base.isequal(a::LandmarkSelection,b::LandmarkSelection) =
    all(name -> isequal(getfield(a,name),getfield(b,name)),fieldnames(LandmarkSelection))
Base.hash(a::LandmarkSelection,h::UInt) =
    hash(Tuple(getfield(a,name) for name in fieldnames(LandmarkSelection)),h)


"""Return original, one-based point indices in landmark selection order."""
landmark_indices(s::LandmarkSelection) = copy(s.indices)
"""Return the maximum distance from any input point to its nearest landmark."""
covering_radius(s::LandmarkSelection) = maximum(s.nearest_distances; init=0.0)
"""
    landmark_distances(selection)

Return a copy of the landmark-by-original-point distance table. Request
`retain_distances=true` in `select_landmarks` to keep this O(m*n) table.
"""
function landmark_distances(s::LandmarkSelection)
    s.distances === nothing && throw(ArgumentError("landmark distances were not retained; use select_landmarks(...; retain_distances=true)."))
    return copy(s.distances)
end

function describe(s::LandmarkSelection)
    r = covering_radius(s)
    return (kind=:landmark_selection, original_points=length(s.nearest_distances),
        landmarks=length(s.indices), indices=landmark_indices(s),
        insertion_radii=copy(s.insertion_radii), covering_radius=r,
        nearest_landmark_indices=copy(s.nearest_indices),
        nearest_distances=copy(s.nearest_distances),
        distances_retained=s.distances !== nothing, geometry=s.geometry,
        conditional_bottleneck_bound=2r,
        bound_scope=:full_metric_rips_filtrations, triangle_inequality_checked=false)
end
function Base.show(io::IO,s::LandmarkSelection)
    print(io,"LandmarkSelection(",length(s.indices)," of ",length(s.nearest_distances),
        " points, covering radius=",covering_radius(s),")")
end

function _landmark_distance_input(data::PointCloud)
    points = data.points
    n = length(points)
    n > 0 || throw(ArgumentError("landmark selection needs a nonempty point cloud."))
    d = length(first(points))
    all(p -> length(p)==d && all(x -> isfinite(x) && isfinite(Float64(x)),p),points) ||
        throw(ArgumentError("points must have equal dimensions and finite Float64 coordinates."))
    return n, (i,j) -> _euclidean_distance(points[i],points[j]), :euclidean
end
function _landmark_distance_input(data::AbstractMatrix{<:Real})
    issparse(data) && throw(ArgumentError("landmark coverage requires complete dense distances; sparse omissions are absent edges, not known metric distances."))
    n = _validate_distance_matrix(data; require_finite=true)
    return n, (i,j) -> i==j ? 0.0 : max(0.0,Float64(data[min(i,j),max(i,j)])), :supplied_dissimilarities
end

"""
    select_landmarks(data; count=nothing, indices=nothing, retain_distances=false)

Select `count` points by farthest-point sampling, starting at original point 1,
with ties broken by original index. Alternatively, inspect the coverage of a
supplied ordered list `indices`. Give exactly one of `count` or `indices`.
`data` is a `PointCloud` (Euclidean distances) or a complete dense distance
matrix. Distinct selected indices are retained even when points coincide.

The result records insertion radii (the first is Inf), each point's nearest
landmark and distance, and the covering radius r. For a finite metric space,
the full Rips filtrations on the input and subset have bottleneck distance at
most 2r in diameter units. This is a conditional mathematical bound: no metric
certification is performed on matrices, and it does not certify neighbor-graph
approximations or essential bars censored at a cutoff. Float64 distance
rounding is separate from this bound.

Use `LandmarkRipsFiltration(landmarks=landmark_indices(selection), max_dim=2)`
to compute on the chosen subset. Set `retain_distances=true` only if the full
landmark-by-point table is needed; ordinary selection uses O(n+m) storage.
"""
function select_landmarks(data; count=nothing, indices=nothing, retain_distances=false)
    retain_distances isa Bool || throw(ArgumentError("retain_distances must be true or false."))
    (count === nothing) != (indices === nothing) ||
        throw(ArgumentError("supply exactly one of count or indices."))
    n, distance, geometry = _landmark_distance_input(data)
    if indices === nothing
        count isa Integer && !(count isa Bool) && 1 <= count <= n ||
            throw(ArgumentError("landmark count must be an integer in 1:$n."))
        m = Int(count)
        chosen = Int[]
    else
        indices isa AbstractVector && !isempty(indices) &&
            all(i -> i isa Integer && !(i isa Bool) && 1 <= i <= n,indices) ||
            throw(ArgumentError("landmarks must be a nonempty vector of indices in 1:$n."))
        chosen = Int.(indices)
        allunique(chosen) || throw(ArgumentError("landmark indices must be distinct."))
        m = length(chosen)
    end
    radii = Float64[]
    nearest = fill(Inf,n)
    assignment = zeros(Int,n)
    used = falses(n)
    table = retain_distances ? Matrix{Float64}(undef,m,n) : nothing
    candidate = 1
    for k in 1:m
        v = if indices === nothing
            push!(chosen,candidate)
            candidate
        else
            chosen[k]
        end
        push!(radii,nearest[v])
        used[v] = true
        next_candidate = 0
        next_distance = -Inf
        for i in 1:n
            d = Float64(distance(v,i))
            isfinite(d) && d >= 0 || throw(ArgumentError("landmark distances must be finite nonnegative Float64 values."))
            table === nothing || (table[k,i]=d)
            current = nearest[i]
            if d < current
                nearest[i] = d
                assignment[i] = v
                current = d
            end
            if indices === nothing && k < m && !used[i] && current > next_distance
                next_candidate = i
                next_distance = current
            end
        end
        candidate = next_candidate
    end
    return LandmarkSelection(chosen,radii,nearest,assignment,table,geometry)
end

# Prepare the selected metric once; reduction and explicit ingestion share it.
function _rips_landmark_input(data, spec::FiltrationSpec)
    construction = _construction_from_params(spec.params)
    selected = spec.kind === :landmark_rips
    greedy = construction.sparsify === :greedy_perm
    (selected || greedy) || return data,spec,nothing
    selected && greedy && throw(ArgumentError("use supplied landmarks or greedy selection, not both."))
    spec.kind in (:rips,:landmark_rips) || return data,spec,nothing
    n = data isa PointCloud ? length(data.points) : size(data,1)
    info = selected ? select_landmarks(data;indices=get(spec.params,:landmarks,nothing)) :
        select_landmarks(data;count=something(get(spec.params,:n_landmarks,nothing),_default_landmark_count(n)))
    idx = info.indices
    reduced = data isa PointCloud ? PointCloud([data.points[i] for i in idx]) : data[idx,idx]
    p = _filter_params(spec.params,[:landmarks,:n_landmarks])
    if greedy
        p = merge(p,(construction=ConstructionOptions(;sparsify=:none,
            collapse=construction.collapse,output_stage=construction.output_stage,budget=construction.budget),))
    end
    return reduced,FiltrationSpec(;kind=:rips,p...),info
end
