# Edelsbrunner--Osang, arXiv:2011.03617, Algorithm 1 and Theorem 5.
# Only earlier levels discover the subset vertices of a later weighted
# Delaunay mosaic. Exact lower-hull cells let us apply Lemma 2 directly, without
# triangulating the higher-generation cells or perturbing their geometry.
# This fragment expects Polyhedra and CDDLib imports in DataIngestion.

function _rhomboid_incremental_storage_check(pending::Int, current::Int, top_count::Int,
                                            dimension::Int, n::Int, spec, workspace_requirements=nothing;
                                            facet_count::Int=0)
    _construction_check_max_simplices!(pending + current + top_count, 0, spec)
    bytes = big(pending + current) * (64 + cld(n, 8)) +
            big(current) * (dimension + 2) * 64 +
            big(top_count) * (96 + 2cld(n, 8))
    if facet_count > 0
        # Each copied halfspace holds d+1 normal coefficients plus its offset;
        # include the vector/element overhead with the existing 64-byte scalar
        # estimate. Check before materializing rows from the native hull.
        bytes += big(facet_count) * (64 + (dimension + 2) * 64)
    end
    # Exact-integer limb storage and the CDD working hull are input-dependent;
    # this is the same explicit preallocation estimate as other build budgets.
    _construction_check_memory_budget!(bytes, spec)
    if workspace_requirements !== nothing
        workspace_requirements[1] = max(workspace_requirements[1], pending + current + top_count)
        workspace_requirements[2] = max(workspace_requirements[2], bytes)
    end
    return nothing
end

function _rhomboid_incremental_top!(visit!, X::Matrix{QQ}, dimension::Int,
                                    spec::FiltrationSpec, ::Type{M};
                                    max_depth::Int=size(X, 1),
                                    workspace_requirements::Union{Nothing,Vector{BigInt}}=nothing) where {M<:Integer}
    n, ambient = size(X)
    0 <= dimension <= 3 || throw(ArgumentError(
        "RhomboidFiltration backend=:incremental supports affine dimensions 0 through 3; use backend=:exhaustive in higher dimension."))
    max_depth == 0 && return nothing
    if dimension == 0
        # Native geometry accepts distinct sites, so its zero-dimensional
        # affine hull contains exactly one point.
        n == 1 || error("Rhomboid incremental enumeration requires distinct sites.")
        support = one(M)
        sphere = _rhomboid_sphere(X, support)
        visit!(_RhomboidCell(zero(M), support), sphere)
        return nothing
    end

    # An injective projection of the affine hull changes only horizontal
    # coordinates of the lifted hull. Keep the full original squared norm,
    # so this projection does not change the Euclidean distance function.
    differences = QQ[X[i, axis] - X[1, axis] for i in 2:n, axis in 1:ambient]
    _, coordinate_columns = FieldLinAlg.rref(QQField(), differences)
    length(coordinate_columns) == dimension || error("Inconsistent rhomboid affine dimension.")
    squared_norms = QQ[sum(X[i, axis]^2 for axis in 1:ambient; init=zero(QQ)) for i in 1:n]
    # A top rhomboid has d+1 on-sphere sites, hence anchor at most n-d-1.
    last_level = min(max_depth, n - dimension)
    vertices = [Set{M}() for _ in 1:last_level]
    for site in 1:n
        push!(vertices[1], one(M) << (site - 1))
    end
    pending = n
    _rhomboid_incremental_storage_check(pending, 0, 0, dimension, n, spec, workspace_requirements)

    upward_ray = zeros(QQ, 1, dimension + 1)
    upward_ray[1, end] = one(QQ)
    for level in 1:last_level
        masks = sort!(collect(vertices[level]))
        pending -= length(masks)
        empty!(vertices[level])
        isempty(masks) && error("Rhomboid incremental enumeration found no vertices at a required depth.")
        _rhomboid_incremental_storage_check(pending, length(masks), 0, dimension, n, spec, workspace_requirements)
        lifts = zeros(QQ, length(masks), dimension + 1)
        for (row, mask) in enumerate(masks)
            for site in _rhomboid_indices(mask, n)
                for axis in 1:dimension
                    lifts[row, axis] += X[site, coordinate_columns[axis]]
                end
                lifts[row, end] += squared_norms[site]
            end
            for axis in axes(lifts, 2)
                lifts[row, axis] /= level
            end
        end

        # The upward ray gives the epigraph of the lower hull. It also handles
        # simplex inputs whose lifts all lie on a single nonvertical plane:
        # that plane remains a genuine lower facet of a full-dimensional hull.
        # hrep and its iterators are lazy backend operations. Materialize the
        # owned Julia facets before releasing the shared CDD boundary; geometry
        # classification below needs no native object or execution lock.
        facets = _with_cdd_execution() do
            polyhedron = Polyhedra.polyhedron(Polyhedra.vrep(lifts, upward_ray),
                                               CDDLib.Library(:exact))
            hull = Polyhedra.hrep(polyhedron)
            isempty(Polyhedra.hyperplanes(hull)) || error(
                "Rhomboid weighted vertices unexpectedly fail to span the affine hull.")
            halfspaces = Polyhedra.halfspaces(hull)
            _rhomboid_incremental_storage_check(pending, length(masks), 0,
                dimension, n, spec, workspace_requirements; facet_count=length(halfspaces))
            collect(halfspaces)
        end
        top_cells = _RhomboidCell{M}[]
        for facet in facets
            facet.a[end] < 0 || continue
            rhs = getproperty(facet, Symbol("\u03b2"))
            inside, union_mask = zero(M), zero(M)
            incident_count = 0
            for row in eachindex(masks)
                dot(facet.a, view(lifts, row, :)) == rhs || continue
                mask = masks[row]
                inside = incident_count == 0 ? mask : inside & mask
                union_mask |= mask
                incident_count += 1
            end
            incident_count > 0 || error("An exact weighted lower facet has no incident site.")
            count_ones(inside) == level - 1 || continue
            on = union_mask & ~inside
            if count_ones(on) != dimension + 1 || incident_count != dimension + 1
                throw(ArgumentError(
                    "RhomboidFiltration encountered cospherical sites $(_rhomboid_indices(on, n)). " *
                    "Native rhomboid backends require general position; use backend=:subdivision_cech for an exact degenerate-input model."))
            end
            _rhomboid_incremental_storage_check(pending, length(masks), length(top_cells) + 1,
                                                dimension, n, spec, workspace_requirements;
                                                facet_count=length(facets))
            push!(top_cells, _RhomboidCell(inside, on))
        end
        sort!(top_cells; by=cell -> (cell.inside, cell.on))
        unique!(top_cells)
        for cell in top_cells
            sphere = _rhomboid_sphere(X, cell.on)
            sphere === nothing && error("A weighted first-generation cell has dependent original sites.")
            if sphere.inside != cell.inside || sphere.on != cell.on
                throw(ArgumentError(
                    "RhomboidFiltration encountered a degenerate weighted-Delaunay cell. " *
                    "Use backend=:subdivision_cech for an exact degenerate-input model."))
            end
            visit!(cell, sphere)
            on_sites = _rhomboid_indices(cell.on, n)
            # Include the top vertex (generation d+1) as well. This is needed
            # in dimension one and is a valid discovered vertex in every
            # dimension. No subsets of the full input are enumerated here.
            for generation in 2:length(on_sites)
                depth = level - 1 + generation
                depth <= last_level || break
                subset = collect(1:generation)
                while true
                    mask = cell.inside
                    for local_site in subset
                        mask |= one(M) << (on_sites[local_site] - 1)
                    end
                    bucket = vertices[depth]
                    if !(mask in bucket)
                        pending += 1
                        _rhomboid_incremental_storage_check(pending, length(masks), length(top_cells),
                                                            dimension, n, spec, workspace_requirements;
                                                            facet_count=length(facets))
                        push!(bucket, mask)
                    end
                    _next_combination!(subset, length(on_sites), generation) || break
                end
            end
        end
    end
    return nothing
end
