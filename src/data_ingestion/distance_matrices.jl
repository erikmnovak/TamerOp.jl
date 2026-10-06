# Distance-matrix Rips ingestion: supplied edges, checked distances, and shared
# budgeted flag expansion. Sparse storage distinguishes absent and zero edges.

function _validate_distance_matrix(data::AbstractMatrix{<:Real};
                                   symmetry_tol::Real=1.0e-10, require_finite::Bool=false)
    size(data, 1) == size(data, 2) ||
        throw(ArgumentError("distance-matrix Rips ingestion expects a square matrix; got size $(size(data))."))
    Base.require_one_based_indexing(data)
    n = size(data, 1)
    n > 0 || throw(ArgumentError("distance-matrix Rips ingestion expects at least one point."))
    tol = Float64(symmetry_tol)
    @inbounds for i in 1:n
        dii = Float64(data[i, i])
        isfinite(dii) || throw(ArgumentError("distance matrix diagonal entry ($i,$i) is not finite."))
        abs(dii) <= tol ||
            throw(ArgumentError("distance matrix diagonal entry ($i,$i) must be zero within tolerance $tol; got $dii."))
    end
    if data isa Matrix{Float64} || data isa Matrix{Float32}
        valid = true
        @inbounds for jb in 1:32:n, ib in 1:32:jb
            for j in jb:min(jb + 31, n)
                block_valid = true
                @simd for i in ib:min(ib + 31, j - 1)
                    a = Float64(data[i,j]); b = Float64(data[j,i])
                    block_valid &= (a >= -tol) & (b >= -tol) &
                        ((a == b) | (abs(a-b) <= tol)) &
                        (!require_finite | (isfinite(a) & isfinite(b)))
                end
                valid &= block_valid
            end
        end
        valid && return n
        # Reuse the general checked traversal for the precise invalid-input error.
    end
    # Visit symmetric tiles together. Both halves then stay in cache rather
    # than repeatedly walking an entire strided row of a large matrix.
    @inbounds for jb in 1:32:n, ib in 1:32:jb
        for j in jb:min(jb + 31, n), i in ib:min(ib + 31, j - 1)
            dij = Float64(data[i, j])
            dji = Float64(data[j, i])
            (isnan(dij) || isnan(dji) || (isfinite(data[i,j]) && !isfinite(dij)) ||
             (isfinite(data[j,i]) && !isfinite(dji))) &&
                throw(ArgumentError("distance matrix entries must be real distances representable as Float64, or +Inf for an absent edge."))
            require_finite && !(isfinite(dij) && isfinite(dji)) &&
                throw(ArgumentError("landmark coverage requires finite pairwise distances."))
            (dij >= -tol && dji >= -tol) ||
                throw(ArgumentError("distance matrix entries must be nonnegative within tolerance $tol."))
            (dij == dji || abs(dij - dji) <= tol) ||
                throw(ArgumentError("distance matrix is not symmetric within tolerance $tol at ($i,$j)."))
        end
    end
    return n
end

@inline _dm_packed_key(n::Int, i::Int, j::Int) = _packed_pair_index(n, i, j)

function _distance_matrix_edges_within_radius(data::AbstractMatrix{<:Real},
                                              radius::Float64)
    n = size(data, 1)
    edges = NTuple{2,Int}[]
    dists = Float64[]
    hint = min(max(0, 4n), div(n * max(n - 1, 0), 2))
    sizehint!(edges, hint)
    sizehint!(dists, hint)
    @inbounds for i in 1:(n - 1)
        for j in (i + 1):n
            d = max(0.0, Float64(data[i, j]))
            isfinite(d) && d <= radius || continue
            push!(edges, (i, j))
            push!(dists, d)
        end
    end
    return edges, dists
end

function _distance_matrix_knn_edges(data::AbstractMatrix{<:Real}, k::Int)
    n = size(data, 1)
    k > 0 || throw(ArgumentError("distance-matrix Rips knn sparsification expects knn > 0."))
    edges = Set{Int}()
    @inbounds for i in 1:n
        order = collect(1:n)
        sort!(order, by = j -> (j == i ? Inf :
            max(0.0, Float64(i < j ? data[i, j] : data[j, i])), j))
        for t in 1:min(k, n - 1)
            j = order[t]
            j == i && continue
            isfinite(data[i, j]) || continue
            a, b = i < j ? (i, j) : (j, i)
            push!(edges, _dm_packed_key(n, a, b))
        end
    end
    out_edges = NTuple{2,Int}[]
    out_dists = Float64[]
    sizehint!(out_edges, length(edges))
    sizehint!(out_dists, length(edges))
    @inbounds for i in 1:(n - 1)
        for j in (i + 1):n
            _dm_packed_key(n, i, j) in edges || continue
            push!(out_edges, (i, j))
            push!(out_dists, max(0.0, Float64(data[i, j])))
        end
    end
    return out_edges, out_dists
end

function _sparse_distance_matrix_edges(data::SparseMatrixCSC{<:Real}; symmetry_tol=1.0e-10)
    n = size(data, 1)
    n == size(data, 2) && n > 0 ||
        throw(ArgumentError("sparse distance input must be a nonempty square matrix."))
    length(data.colptr) == n + 1 && first(data.colptr) == 1 &&
        issorted(data.colptr) && last(data.colptr) == length(data.nzval) + 1 &&
        length(data.rowval) == length(data.nzval) ||
        throw(ArgumentError("sparse distance input has invalid compressed-column storage."))
    edges, dists = NTuple{2,Int}[], Float64[]
    slots = Dict{Tuple{Int,Int},Int}()
    for j in 1:n
        previous = 0
        for ptr in nzrange(data, j)
            i = data.rowval[ptr]
            previous < i <= n || throw(ArgumentError("sparse distance row indices must be distinct, sorted and within 1:n."))
            previous = i
            value = data.nzval[ptr]
            d = Float64(value)
            isnan(d) && throw(ArgumentError("sparse distances cannot be NaN."))
            isfinite(value) && !isfinite(d) && throw(ArgumentError("sparse distance cannot be represented as a finite Float64 grade."))
            if i == j
                isfinite(d) && abs(d) <= symmetry_tol ||
                    throw(ArgumentError("sparse distance diagonal entries must be zero within tolerance $symmetry_tol."))
                continue
            end
            d >= -symmetry_tol || throw(ArgumentError("sparse distances must be nonnegative within tolerance $symmetry_tol."))
            edge = minmax(i, j)
            slot = get(slots, edge, 0)
            if slot == 0
                push!(edges, edge); push!(dists, d)
                slots[edge] = length(edges)
            else
                old = dists[slot]
                (d == old || abs(d - old) <= symmetry_tol) ||
                    throw(ArgumentError("conflicting sparse distances for pair $edge."))
                if i < j
                    dists[slot] = d
                end
            end
        end
    end
    # CSC iteration visits lower/upper entries in different orders. Return the
    # same canonical order as dense input and retain explicitly stored zeros.
    permutation = sortperm(edges)
    return n, edges[permutation], max.(0.0, dists[permutation])
end

function _sparse_distance_knn_indices(n, edges, dists, k)
    k > 0 || throw(ArgumentError("distance-matrix Rips knn sparsification expects knn > 0."))
    incident = [Int[] for _ in 1:n]
    for (slot, (u,v)) in enumerate(edges)
        isfinite(dists[slot]) || continue
        push!(incident[u], slot); push!(incident[v], slot)
    end
    keep = falses(length(edges))
    for v in 1:n
        neighbors = incident[v]
        sort!(neighbors; by=i -> (dists[i], edges[i][1] == v ? edges[i][2] : edges[i][1]))
        for j in 1:min(k, length(neighbors))
            keep[neighbors[j]] = true
        end
    end
    return findall(keep)
end

function _distance_matrix_rips_edges(data, spec; check_budget=true)
    spec.kind === :rips || throw(ArgumentError("distance-matrix ingestion requires RipsFiltration / kind=:rips."))
    _validate_rips_params(spec.params)
    construction = _construction_from_params(spec.params)
    radius_raw = get(spec.params, :radius, nothing)
    radius = radius_raw === nothing ? Inf : Float64(radius_raw)
    construction.sparsify === :radius && !isfinite(radius) &&
        throw(ArgumentError("construction.sparsify=:radius requires a finite radius."))
    construction.sparsify in (:none, :radius, :knn) ||
        throw(ArgumentError("distance-matrix Rips supports construction.sparsify=:none, :radius, or :knn."))
    sparse_input = issparse(data)
    if sparse_input
        # sparse() preserves stored zeros for sparse transpose/adjoint wrappers.
        stored = data isa SparseMatrixCSC ? data : sparse(data)
        n, edges, dists = _sparse_distance_matrix_edges(stored)
    else
        n = _validate_distance_matrix(data)
        edges, dists = NTuple{2,Int}[], Float64[]
    end
    max_dim = Int(get(spec.params, :max_dim, 1))
    max_dim == 0 && return n, NTuple{2,Int}[], Float64[]
    if construction.sparsify === :knn
        k = Int(something(get(spec.params, :knn, nothing), 8))
        if sparse_input
            indices = _sparse_distance_knn_indices(n, edges, dists, k)
            edges, dists = edges[indices], dists[indices]
        else
            edges, dists = _distance_matrix_knn_edges(data, k)
        end
    elseif !sparse_input
        edges, dists = _distance_matrix_edges_within_radius(data, radius)
    end
    keep = findall(d -> isfinite(d) && d <= radius, dists)
    edges, dists = edges[keep], dists[keep]
    check_budget && _construction_check_max_edges!(length(edges), spec)
    return n, edges, dists
end

function _graded_complex_from_distance_matrix(data::AbstractMatrix{<:Real}, spec::FiltrationSpec;
                                              return_simplex_tree::Bool=false)
    _validate_geometric_filtration_request(data, spec)
    _validate_construction_request(data, spec)
    spec.kind === :edge_weighted && return _graded_complex_from_weighted_graph(data,spec;return_simplex_tree)
    data,spec,_ = _rips_landmark_input(data,spec)
    n, edges, dists = _distance_matrix_rips_edges(data, spec)
    _construction_check_max_simplices!(n, 0, spec)
    return _materialize_flag_output(edges, fill((0.0,), n), [(d,) for d in dists], spec;
                                    return_simplex_tree)
end

function _estimate_distance_matrix_cell_counts(data, spec; warnings, strict)
    spec.kind === :edge_weighted && return _estimate_weighted_flag_counts(data,spec;warnings,strict)
    data,spec,_ = _rips_landmark_input(data,spec)
    n, edges, _ = _distance_matrix_rips_edges(data, spec; check_budget=false)
    max_dim = Int(get(spec.params, :max_dim, 1))
    counts = BigInt[big(n)]
    max_dim == 0 && return counts
    push!(counts, big(length(edges)))
    forward_counts = zeros(Int, n)
    for (u, _) in edges
        forward_counts[u] += 1
    end
    for d in 2:max_dim
        # Every d-simplex has a unique smallest vertex and d forward neighbors.
        push!(counts, sum(binomial(big(k), d) for k in forward_counts; init=big(0)))
    end
    (max_dim >= 2 || _construction_from_params(spec.params).collapse !== :none) &&
        _ingestion_warn!(warnings, "Distance-matrix simplex counts above edges are upper bounds before any certified collapse.", strict)
    return counts
end
