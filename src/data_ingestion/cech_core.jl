# Published core nerves. Neighbor-distance grading lives in core_bifiltrations;
# this fragment owns Cech support and the full (possibly non-simplicial) planar
# Voronoi nerve, including all faces of cocircular Delaunay cells.

function _core_storage_preflight!(counts, n::Int, nk::Int, spec::FiltrationSpec)
    total = sum(counts)
    total <= typemax(Int) || throw(ArgumentError("Core nerve exceeds addressable storage; lower max_dim."))
    _construction_check_max_simplices!(total, length(counts) - 1, spec)
    length(counts) >= 2 && _construction_check_max_edges!(counts[2], spec)
    # Conservative packed grades + neighbor table + sparse boundary/simplex
    # storage estimate, separate from the standard dense-boundary guardrail.
    bytes = big(8) * n * (nk + 1) + big(16) * nk * total +
            sum(big(32) * d * counts[d] for d in eachindex(counts))
    _construction_check_memory_budget!(max(bytes, _estimate_dense_bytes_from_cell_counts(BigInt.(counts))), spec)
    return nothing
end

function _core_materialize(data, spec, simplices, radii; return_simplex_tree::Bool)
    n = length(data.points)
    ks = _core_k_values(get(spec.params, :k_values, nothing), n)
    _core_storage_preflight!(length.(simplices), n, length(ks), spec)
    distances = _core_neighbor_distances(point_matrix(data), ks)
    grades = _core_multigrades(simplices, radii, distances, ks, get(spec.params, :beta, 1.0))
    return _materialize_simplicial_output(simplices, grades, spec; return_simplex_tree=return_simplex_tree)
end

function _graded_complex_from_point_cloud_core(data::PointCloud, spec::FiltrationSpec;
                                                return_simplex_tree::Bool=false)
    _validate_geometric_filtration_request(data, spec)
    spec = _canonical_geometric_filtration_spec(spec)
    n = length(data.points)
    max_dim = min(Int(get(spec.params, :max_dim, 2)), n - 1)
    ks = _core_k_values(get(spec.params, :k_values, nothing), n)
    counts = BigInt[binomial(big(n), d + 1) for d in 0:max_dim]
    _core_storage_preflight!(counts, n, length(ks), spec)
    simplices = [_combinations(n, d + 1) for d in 0:max_dim]
    radii = _enclosing_radii(data.points, simplices)
    return _core_materialize(data, spec, simplices, radii; return_simplex_tree=return_simplex_tree)
end

# Connected triangles with the same empty circumcircle are one Delaunay cell.
# Recovering this cell from any triangulation makes the resulting Voronoi nerve
# independent of the backend's choice of cocircular diagonals.
function _delaunay_cells(points, packed::_PackedDelaunay2D)
    nt = length(packed.triangles)
    parents = collect(1:nt)
    function root(i)
        while parents[i] != i
            parents[i] = parents[parents[i]]
            i = parents[i]
        end
        return i
    end
    incident = Dict{NTuple{2,Int},Int}()
    for (t, triangle) in enumerate(packed.triangles)
        i, j, k = triangle
        for edge in ((i, j), (i, k), (j, k))
            other = get(incident, edge, 0)
            if other == 0
                incident[edge] = t
                continue
            end
            previous = packed.triangles[other]
            opposite = first(v for v in previous if v != edge[1] && v != edge[2])
            if _delaunay_incircle_sign(points[i], points[j], points[k], points[opposite]) == 0
                parents[root(t)] = root(other)
            end
        end
    end
    cells = Dict{Int,Vector{Int}}()
    radii = Dict{Int,Float64}()
    for t in 1:nt
        r = root(t)
        append!(get!(cells, r, Int[]), packed.triangles[t])
        # All these radii are mathematically equal; use one common numerical
        # value for the entire cell so its facets cannot get reordered.
        radii[r] = max(get(radii, r, 0.0), packed.tri_radius[t])
    end
    return [(sort!(unique!(cells[r])), radii[r]) for r in sort!(collect(keys(cells)))]
end

function _delaunay_nerve_simplices(points, spec::FiltrationSpec, max_dim::Int, nk::Int)
    n = length(points)
    _core_storage_preflight!([n], n, nk, spec)
    _validate_delaunay_points(points)
    max_dim == 0 && return [[[v] for v in 1:n]], [zeros(n)]
    if length(first(points)) == 2 && _pointcloud_delaunay_backend(spec) === :naive
        _construction_check_memory_budget!(cld(big(n) * n, 8), spec)
    end
    # Alpha edges need incident triangles even when output is truncated to 1D.
    entry = _packed_delaunay_entry(points, spec; max_dim=2)
    packed = entry.packed
    simplices = [Vector{Vector{Int}}() for _ in 0:max_dim]
    radii = [Float64[] for _ in 0:max_dim]
    indices = [Dict{Tuple{Vararg{Int}},Int}() for _ in 0:max_dim]
    function insert(vertices, radius)
        d = length(vertices)
        d <= max_dim + 1 || return nothing
        key = Tuple(vertices)
        prior = get(indices[d], key, 0)
        if prior != 0
            radii[d][prior] = min(radii[d][prior], radius)
            return nothing
        end
        counts = length.(simplices)
        counts[d] += 1
        _core_storage_preflight!(counts, n, nk, spec)
        push!(simplices[d], collect(vertices))
        push!(radii[d], radius)
        indices[d][key] = length(simplices[d])
        return nothing
    end
    for v in 1:n
        insert((v,), 0.0)
    end
    max_dim == 0 && return simplices, radii
    # The closed diameter test avoids the old absolute obtuseness tolerance.
    # Only an opposite vertex in an incident triangle can obstruct a Delaunay
    # edge's diameter disk; otherwise its midpoint is a Voronoi witness.
    edge_radius = copy(packed.edge_radius)
    edge_index = Dict(edge => i for (i, edge) in enumerate(packed.edges))
    incident_radius = fill(Inf, length(packed.edges))
    obstructed = falses(length(packed.edges))
    for (t, (i, j, k)) in enumerate(packed.triangles)
        for (edge, opposite) in (((i, j), k), ((i, k), j), ((j, k), i))
            index = edge_index[edge]
            incident_radius[index] = min(incident_radius[index], packed.tri_radius[t])
            # Boundary points on a diameter circle need no correction; either
            # radius is equal there. A nonpositive dot is enough for this gate.
            obstructed[index] |= _diameter_ball_contains(points[edge[1]], points[edge[2]], points[opposite])
        end
    end
    for i in eachindex(packed.edges)
        if obstructed[i]
            edge_radius[i] = max(edge_radius[i], incident_radius[i])
        end
        insert(packed.edges[i], edge_radius[i])
    end
    for (cell, radius) in _delaunay_cells(points, packed)
        # A cell with m cocircular vertices represents a common Voronoi
        # intersection, so all its subsets belong to the nerve.
        m = length(cell)
        for size in 2:min(m, max_dim + 1)
            # Avoid materializing a combinatorial batch before checking bounds.
            bound = binomial(big(m), size)
            max_new = length(simplices[size]) + bound
            # Existing edges/faces may be shared. The actual insertion checks
            # below enforce exact budgets, so this check is only for addressability.
            max_new <= typemax(Int) || throw(ArgumentError("Delaunay nerve exceeds addressable storage."))
            combination = collect(1:size)
            while true
                insert([cell[i] for i in combination], radius)
                _next_combination!(combination, m, size) || break
            end
        end
    end
    for d in eachindex(simplices)
        permutation = sortperm(simplices[d])
        simplices[d] = simplices[d][permutation]
        radii[d] = radii[d][permutation]
    end
    # Enforce face monotonicity only at floating-point rounding scale.
    for d in 2:length(simplices)
        faces = Dict(Tuple(s) => r for (s, r) in zip(simplices[d - 1], radii[d - 1]))
        for (i, simplex) in enumerate(simplices[d])
            for omit in eachindex(simplex)
                key = Tuple(simplex[q] for q in eachindex(simplex) if q != omit)
                radii[d][i] = max(radii[d][i], faces[key])
            end
        end
    end
    return simplices, radii
end

function _graded_complex_from_point_cloud_core_delaunay(data::PointCloud, spec::FiltrationSpec;
                                                         return_simplex_tree::Bool=false)
    _validate_geometric_filtration_request(data, spec)
    spec = _canonical_geometric_filtration_spec(spec)
    n = length(data.points)
    max_dim = min(Int(get(spec.params, :max_dim, 2)), n - 1)
    ks = _core_k_values(get(spec.params, :k_values, nothing), n)
    simplices, radii = _delaunay_nerve_simplices(data.points, spec, max_dim, length(ks))
    return _core_materialize(data, spec, simplices, radii; return_simplex_tree=return_simplex_tree)
end
