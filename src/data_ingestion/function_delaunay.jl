# Sublevel Delaunay-Cech bifiltrations, Alonso--Kerber--Lam--Lesnick (2024),
# https://arxiv.org/abs/2310.15902, Definition 2.1 and Section 3.
#
# A prefix triangulation alone does not define a filtration: insertion can
# remove simplices. Keep every prefix and cone each insertion conflict to the
# new vertex. The resulting incremental complex has simplices of dimension
# at most ambient dimension + 1. Its Cech grades are minimum enclosing-ball
# radii, not the circumsphere radii used to discover insertion conflicts.

function _function_delaunay_insert_faces!(simplices::Vector{Vector{Vector{Int}}},
                                         seen::Set{NTuple{4,Int}},
                                         vertices,
                                         max_dim::Int,
                                         spec::FiltrationSpec)
    ordered = sort!(collect(vertices))
    n = length(ordered)
    for mask in 1:((1 << n) - 1)
        count = count_ones(mask)
        count <= max_dim + 1 || continue
        key = (0, 0, 0, 0)
        pos = 1
        @inbounds for i in 1:n
            if (mask & (1 << (i - 1))) != 0
                key = Base.setindex(key, ordered[i], pos)
                pos += 1
            end
        end
        key in seen && continue
        _construction_check_max_simplices!(length(seen) + 1, count - 1, spec)
        count == 2 && _construction_check_max_edges!(length(simplices[2]) + 1, spec)
        push!(seen, key)
        push!(simplices[count], Int[key[i] for i in 1:count])
        if _construction_memory_budget(spec) !== nothing
            _construction_check_memory_budget!(
                _estimate_dense_bytes_from_cell_counts(BigInt[length(s) for s in simplices]), spec)
        end
    end
    return nothing
end

function _function_delaunay_interval_conflict(a, b, p, ambient_dim::Int)
    if ambient_dim == 2
        _delaunay_orient_sign(a, b, p) == 0 || return false
    end
    axis = ambient_dim == 1 || a[1] != b[1] ? 1 : 2
    return min(a[axis], b[axis]) < p[axis] < max(a[axis], b[axis])
end

function _graded_complex_from_point_cloud_function_delaunay(data::PointCloud,
                                                            spec::FiltrationSpec;
                                                            return_simplex_tree::Bool=false)
    _validate_geometric_filtration_request(data, spec)
    spec = _canonical_geometric_filtration_spec(spec)
    points = data.points
    ambient_dim = length(first(points))
    max_dim = Int(get(spec.params, :max_dim, 3))
    0 <= max_dim <= 3 || throw(ArgumentError("FunctionDelaunayFiltration requires 0 <= max_dim <= 3."))
    get(spec.params, :simplex_agg, :max) == :max ||
        throw(ArgumentError("FunctionDelaunayFiltration uses the maximum vertex function value on each simplex."))
    orientation = get(spec.params, :orientation, (1, 1))
    orientation == (1, 1) || throw(ArgumentError("FunctionDelaunayFiltration uses radius/function sublevels with orientation=(1,1)."))
    construction = _construction_from_params(spec.params)
    construction.sparsify == :none && construction.collapse == :none ||
        throw(ArgumentError("FunctionDelaunayFiltration requires sparsify=:none and collapse=:none."))
    n = length(points)
    _construction_check_max_simplices!(n, 0, spec)
    values = _point_vertex_values(points, spec)
    all(isfinite, values) || throw(ArgumentError("FunctionDelaunayFiltration requires finite vertex function values."))
    order = sortperm(1:n; by=i -> (values[i], i))
    ordered_points = points[order]
    simplices = [Vector{Vector{Int}}() for _ in 0:max_dim]
    seen = Set{NTuple{4,Int}}()
    for i in 1:n
        _function_delaunay_insert_faces!(simplices, seen, (i,), max_dim, spec)
    end

    if max_dim >= 1
        backend = ambient_dim == 2 ? _pointcloud_delaunay_backend(spec) : :naive
        # The naive geometry owner uses a packed quadratic edge-presence table.
        backend == :naive && ambient_dim == 2 &&
            _construction_check_memory_budget!(cld(big(n) * n, 8), spec)
        previous = _PackedDelaunay2D(NTuple{2,Int}[], Float64[], NTuple{3,Int}[], Float64[])
        for next in 1:n
            point = ordered_points[next]
            if isempty(previous.triangles)
                # Early prefixes can be collinear even in ambient dimension 2.
                # Subdividing a previous interval creates a triangular conflict
                # coface; retaining just the old/new edges would create a loop.
                for (i, j) in previous.edges
                    if _function_delaunay_interval_conflict(ordered_points[i], ordered_points[j], point, ambient_dim)
                        _function_delaunay_insert_faces!(simplices, seen,
                            (order[i], order[j], order[next]), max_dim, spec)
                    end
                end
            else
                for (i, j, k) in previous.triangles
                    a, b, c = ordered_points[i], ordered_points[j], ordered_points[k]
                    incircle = _delaunay_incircle_sign(a, b, c, point)
                    incircle == 0 && throw(ArgumentError(
                        "FunctionDelaunayFiltration encountered an exactly cocircular insertion at vertices " *
                        "$(order[i]), $(order[j]), $(order[k]), $(order[next]). " *
                        "The published incremental construction currently requires a noncocircular insertion order; " *
                        "use FunctionRipsFiltration for this degenerate point cloud."))
                    if incircle == _delaunay_orient_sign(a, b, c)
                        _function_delaunay_insert_faces!(simplices, seen,
                            (order[i], order[j], order[k], order[next]), max_dim, spec)
                    end
                end
            end
            # Bypass the identity cache: each prefix is a temporary view with a
            # different point set, and only the preceding triangulation is used.
            prefix = view(ordered_points, 1:next)
            current = ambient_dim == 1 ? _packed_delaunay_collinear(prefix) :
                      _packed_delaunay_simplices_2d(prefix; max_dim=2, backend=backend)
            for (i, j) in current.edges
                _function_delaunay_insert_faces!(simplices, seen, (order[i], order[j]), max_dim, spec)
            end
            for (i, j, k) in current.triangles
                _function_delaunay_insert_faces!(simplices, seen, (order[i], order[j], order[k]), max_dim, spec)
            end
            previous = current
        end
    end

    foreach(sort!, simplices)
    radii = _enclosing_radii(points, simplices)
    grades = Vector{NTuple{2,Float64}}()
    sizehint!(grades, length(seen))
    for dimension in eachindex(simplices)
        for (i, simplex) in enumerate(simplices[dimension])
            push!(grades, (radii[dimension][i], maximum(v -> values[v], simplex)))
        end
    end
    return _materialize_simplicial_output(simplices, grades, spec;
                                           return_simplex_tree=return_simplex_tree)
end
