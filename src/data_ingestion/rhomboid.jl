# The unsliced rhomboid bifiltration (Corbet--Kerber--Lesnick--Osang,
# arXiv:2103.07823, Section 4.4). A cell is the cube (inside, on) in the
# sphere arrangement, with radius increasing and minimum depth decreasing.
# Geometry uses exact rational predicates and squared radii. Emitted physical
# radii are exact algebraic reals, retaining every distinct critical value.

struct _RhomboidCell{M<:Integer}
    inside::M
    on::M
end
Base.:(==)(a::_RhomboidCell, b::_RhomboidCell) = a.inside == b.inside && a.on == b.on
Base.isequal(a::_RhomboidCell, b::_RhomboidCell) = isequal(a.inside, b.inside) && isequal(a.on, b.on)
Base.hash(c::_RhomboidCell, seed::UInt) = hash(c.on, hash(c.inside, seed))

struct _RhomboidSphere{M<:Integer}
    radius2::QQ
    inside::M
    on::M
end

function _rhomboid_points(data::PointCloud; allow_coincident::Bool=false)
    points = point_matrix(data)
    n, d = size(points)
    n > 0 && d > 0 || throw(ArgumentError("RhomboidFiltration requires a nonempty point cloud of positive ambient dimension."))
    X = Matrix{QQ}(undef, n, d)
    seen = Set{Tuple{Vararg{QQ}}}()
    for i in 1:n
        for j in 1:d
            isfinite(points[i, j]) || throw(ArgumentError("RhomboidFiltration requires finite coordinates."))
            X[i, j] = QQ(points[i, j])
        end
        key = Tuple(view(X, i, :))
        !allow_coincident && key in seen && throw(ArgumentError("Native RhomboidFiltration backends require distinct points; use backend=:subdivision_cech for labeled repeated sites."))
        push!(seen, key)
    end
    return X
end

function _rhomboid_affine_dimension(X::Matrix{QQ})
    n, d = size(X)
    n == 1 && return 0
    differences = QQ[X[i, j] - X[1, j] for i in 2:n, j in 1:d]
    return FieldLinAlg.rank(QQField(), differences)
end

function _rhomboid_indices(mask::Integer, n::Int)
    out = Int[]
    for i in 1:n
        iszero(mask & (one(mask) << (i - 1))) || push!(out, i)
    end
    return out
end

# Circumsphere of an affinely independent support, centered in its affine
# hull. Its exact classification can be reused for all incident cells.
function _rhomboid_sphere(X::Matrix{QQ}, support::M) where {M<:Integer}
    n, d = size(X)
    ids = _rhomboid_indices(support, n)
    a = first(ids)
    m = length(ids) - 1
    center = collect(view(X, a, :))
    if m > 0
        V = QQ[X[ids[j + 1], axis] - X[a, axis] for axis in 1:d, j in 1:m]
        gram = transpose(V) * V
        factor = lu(gram; check=false)
        issuccess(factor) || return nothing
        coefficients = factor \ QQ[gram[j, j] / 2 for j in 1:m]
        center += V * coefficients
    end
    radius2 = sum((center[j] - X[a, j])^2 for j in 1:d; init=zero(QQ))
    inside, on = zero(M), zero(M)
    for i in 1:n
        distance2 = sum((center[j] - X[i, j])^2 for j in 1:d; init=zero(QQ))
        bit = one(M) << (i - 1)
        if distance2 < radius2
            inside |= bit
        elseif distance2 == radius2
            on |= bit
        end
    end
    return _RhomboidSphere(radius2, inside, on)
end

function _rhomboid_storage_check!(counts, spec::FiltrationSpec, n::Int)
    total = sum(counts)
    total <= typemax(Int) || throw(ArgumentError("Rhomboid cell count exceeds addressable storage before enumeration."))
    _construction_check_max_simplices!(total, length(counts) - 1, spec)
    length(counts) > 1 && _construction_check_max_edges!(counts[2], spec)
    # Keys, exact radius bounds, face incidence and coordinate masks. This is
    # a preallocation estimate, not a bound on arbitrary-size integer limbs.
    bytes = big(total) * (128 + 2cld(n, 8)) +
            sum(big(32) * (d - 1) * counts[d] for d in eachindex(counts))
    _construction_check_memory_budget!(bytes, spec)
    return nothing
end

function _rhomboid_insert_faces!(cells, indices, key::_RhomboidCell{M},
                                  dimension::Int, counts, spec, n,
                                  depth_range) where {M}
    if depth_range !== nothing
        lower, upper = depth_range
        anchor = count_ones(key.inside)
        # A positive-dimensional carrier anchored at the top cap contributes
        # only its anchor vertex. That vertex is reached through the retained
        # neighboring carriers, and has its own constrained radius.
        (anchor > upper || (anchor == upper && dimension > 0) ||
         anchor + dimension < lower) && return nothing
    end
    haskey(indices, key) && return nothing
    counts[dimension + 1] += 1
    _rhomboid_storage_check!(counts, spec, n)
    push!(cells[dimension + 1], key)
    indices[key] = length(cells[dimension + 1])
    dimension == 0 && return nothing
    for site in _rhomboid_indices(key.on, n)
        bit = one(M) << (site - 1)
        face_on = key.on & ~bit
        _rhomboid_insert_faces!(cells, indices, _RhomboidCell(key.inside, face_on),
                                dimension - 1, counts, spec, n, depth_range)
        _rhomboid_insert_faces!(cells, indices, _RhomboidCell(key.inside | bit, face_on),
                                dimension - 1, counts, spec, n, depth_range)
    end
    return nothing
end

function _rhomboid_geometry(X::Matrix{QQ}, dimension::Int, spec::FiltrationSpec,
                            ::Type{M}; depth_range=get(spec.params, :depth_range, nothing),
                            backend::Symbol=:exhaustive,
                            workspace_requirements::Union{Nothing,Vector{BigInt}}=nothing) where {M<:Integer}
    backend in (:exhaustive, :incremental) || throw(ArgumentError(
        "Native rhomboid geometry requires resolved backend=:exhaustive or :incremental."))
    n = size(X, 1)
    top = dimension + 1
    cells = [Vector{_RhomboidCell{M}}() for _ in 0:top]
    indices = Dict{_RhomboidCell{M},Int}()
    spheres = Dict{M,_RhomboidSphere{M}}()
    counts = zeros(Int, top + 1)
    if depth_range !== nothing && last(depth_range) == 0
        # The depth-zero multicover is the whole ambient affine space at every
        # radius. Its contractible model needs no sphere enumeration.
        cell = _RhomboidCell(zero(M), zero(M))
        _rhomboid_insert_faces!(cells, indices, cell, 0, counts, spec, n, depth_range)
        return cells, Dict(cell => zero(QQ))
    end

    function retain_top!(cell, sphere)
        if depth_range !== nothing
            lower, upper = depth_range
            anchor = count_ones(cell.inside)
            (anchor >= upper || anchor + count_ones(cell.on) < lower) && return nothing
        end
        spheres[cell.on] = sphere
        _rhomboid_insert_faces!(cells, indices, cell, top, counts, spec, n, depth_range)
        return nothing
    end

    if backend === :incremental
        max_depth = depth_range === nothing ? n : last(depth_range)
        _rhomboid_incremental_top!(retain_top!, X, dimension, spec, M; max_depth, workspace_requirements)
    else
        combination = collect(1:top)
        while true
            support = zero(M)
            for site in combination
                support |= one(M) << (site - 1)
            end
            sphere = _rhomboid_sphere(X, support)
            if sphere !== nothing
                relevant = if depth_range === nothing
                    true
                else
                    lower, upper = depth_range
                    anchor = count_ones(sphere.inside)
                    # A degenerate sphere may contain more on-sphere sites
                    # than the enumerated support. Its actual depth span
                    # decides relevance before the native-cell validation.
                    anchor < upper && anchor + count_ones(sphere.on) >= lower
                end
                if relevant
                    sphere.on == support || throw(ArgumentError(
                        "RhomboidFiltration encountered cospherical sites $(_rhomboid_indices(sphere.on, n)). " *
                        "Native rhomboid backends require general position in the computed region; " *
                        "use backend=:subdivision_cech for an exact degenerate-input model."))
                    retain_top!(_RhomboidCell(sphere.inside, support), sphere)
                end
            end
            _next_combination!(combination, n, top) || break
        end
    end
    isempty(indices) && error("RhomboidFiltration: affine dimension produced no full-dimensional carriers in the requested depth window.")
    foreach(group -> sort!(group; by=c -> (c.inside, c.on)), cells)

    # A lower cell's constrained radius minimum is either its unconstrained
    # circumsphere or the minimum on a coface where another constraint is tight.
    # Window pruning retains every coface of a nonvertex contributing carrier:
    # cofaces have no larger anchor and contain its depth interval. At a cap
    # vertex of positive depth, a minimizing sphere touches an inside site,
    # hence has a realizing coface with anchor strictly below the cap too.
    radii2 = Dict{_RhomboidCell{M},QQ}()
    for cell_dimension in top:-1:0
        for cell in cells[cell_dimension + 1]
            if cell_dimension > 0
                sphere = get!(spheres, cell.on) do
                    candidate = _rhomboid_sphere(X, cell.on)
                    candidate === nothing && error("Rhomboid face has dependent generators.")
                    candidate
                end
                if iszero(sphere.inside & ~cell.inside) &&
                   iszero(cell.inside & ~(sphere.inside | sphere.on))
                    prior = get(radii2, cell, nothing)
                    radii2[cell] = prior === nothing ? sphere.radius2 : min(prior, sphere.radius2)
                end
            elseif iszero(cell.inside)
                radii2[cell] = zero(QQ)
            end
            haskey(radii2, cell) || error("Rhomboid radius has no realizing coface.")
            radius2 = radii2[cell]
            for site in _rhomboid_indices(cell.on, n)
                bit = one(M) << (site - 1)
                face_on = cell.on & ~bit
                for inside in (cell.inside, cell.inside | bit)
                    face = _RhomboidCell(inside, face_on)
                    depth_range !== nothing && !haskey(indices, face) && continue
                    prior = get(radii2, face, nothing)
                    radii2[face] = prior === nothing ? radius2 : min(prior, radius2)
                end
            end
        end
    end
    return cells, radii2
end

function _rhomboid_radius(radius2::QQ)
    return sqrt(AlgebraicReal(radius2))
end

function _rhomboid_cellular_complex(cells::Vector{Vector{_RhomboidCell{M}}}, radii2, n::Int) where {M}
    grades = NTuple{2,AlgebraicReal}[]
    boundaries = SparseMatrixCSC{Int,Int}[]
    for (slot, group) in enumerate(cells)
        for cell in group
            depth = length(_rhomboid_indices(cell.inside, n))
            push!(grades, (_rhomboid_radius(radii2[cell]), AlgebraicReal(depth)))
        end
        slot == 1 && continue
        faces = Dict(cell => i for (i, cell) in enumerate(cells[slot - 1]))
        rows, cols, values = Int[], Int[], Int[]
        sizehint!(rows, 2(slot - 1) * length(group))
        sizehint!(cols, 2(slot - 1) * length(group))
        sizehint!(values, 2(slot - 1) * length(group))
        for (column, cell) in enumerate(group)
            for (position, site) in enumerate(_rhomboid_indices(cell.on, n))
                bit = one(M) << (site - 1)
                on = cell.on & ~bit
                sign = isodd(position) ? 1 : -1
                push!(rows, faces[_RhomboidCell(cell.inside | bit, on)])
                push!(cols, column)
                push!(values, sign)
                push!(rows, faces[_RhomboidCell(cell.inside, on)])
                push!(cols, column)
                push!(values, -sign)
            end
        end
        push!(boundaries, sparse(rows, cols, values, length(cells[slot - 1]), length(group)))
    end
    return GradedComplex([collect(eachindex(group)) for group in cells], boundaries, grades)
end

# A consistent Freudenthal triangulation of each cube. Global site ordering
# makes restrictions agree on common faces. This is explicit stage opt-in;
# canonical graded-complex/module paths retain the smaller cellular complex.
function _rhomboid_simplex_tree(cells::Vector{Vector{_RhomboidCell{M}}}, radii2, n::Int, spec) where {M}
    vertices = Dict(cell.inside => i for (i, cell) in enumerate(cells[1]))
    simplices = [Vector{Vector{Int}}() for _ in cells]
    births = [Dict{Tuple{Vararg{Int}},NTuple{2,AlgebraicReal}}() for _ in cells]
    counts = zeros(Int, length(cells))
    for (slot, group) in enumerate(cells), cell in group
        generators = _rhomboid_indices(cell.on, n)
        grade = (_rhomboid_radius(radii2[cell]), AlgebraicReal(length(_rhomboid_indices(cell.inside, n))))
        # All faces have already been processed in lower dimension, so only
        # new simplices need their grade recorded here.
        chain = Int[vertices[cell.inside]]
        function insert_chain!()
            size = length(chain)
            # Enumerate faces without bit shifts bounded by machine word size.
            face = Int[]
            function visit(start)
                for i in start:size
                    push!(face, chain[i])
                    key = Tuple(sort(face))
                    dimension = length(key)
                    if !haskey(births[dimension], key)
                        counts[dimension] += 1
                        _rhomboid_storage_check!(counts, spec, n)
                        births[dimension][key] = grade
                    end
                    visit(i + 1)
                    pop!(face)
                end
            end
            visit(1)
        end
        function permutations!(mask, remaining)
            if isempty(remaining)
                insert_chain!()
                return
            end
            for i in eachindex(remaining)
                site = remaining[i]
                next = mask | (one(M) << (site - 1))
                push!(chain, vertices[next])
                permutations!(next, [remaining[j] for j in eachindex(remaining) if i != j])
                pop!(chain)
            end
        end
        permutations!(cell.inside, generators)
    end
    grades = NTuple{2,AlgebraicReal}[]
    for slot in eachindex(cells)
        keys_sorted = sort!(collect(keys(births[slot])))
        simplices[slot] = [collect(key) for key in keys_sorted]
        append!(grades, (births[slot][key] for key in keys_sorted))
    end
    return _simplex_tree_multi_from_simplices(simplices, grades)
end

struct _CompiledRhomboidGeometry{M<:Integer}
    cells::Vector{Vector{_RhomboidCell{M}}}
    radii2::Dict{_RhomboidCell{M},QQ}
    backend::Symbol
    workspace_simplices::BigInt
    workspace_bytes::BigInt
end

# Compile and publish under the same lock used by clearing. No partially built
# artifact escapes, and concurrent first calls publish one object. These private
# exact arrays are only read after publication; output builders own their arrays.
function _rhomboid_cached_geometry(build, cache::Union{Nothing,EncodingCache}, key)
    cache === nothing && return build()
    Base.lock(cache.lock)
    try
        cached = get(cache.geometry, key, nothing)
        cached === nothing || return cached.value
        geometry = build()
        cache.geometry[key] = GeometryCachePayload(geometry)
        return geometry
    finally
        Base.unlock(cache.lock)
    end
end

function _compile_rhomboid_geometry(X::Matrix{QQ}, spec::FiltrationSpec)
    n = size(X, 1)
    dimension = _rhomboid_affine_dimension(X)
    depths = get(spec.params, :depth_range, nothing)
    backend = get(spec.params, :backend, :auto)
    if depths === nothing
        # A complete top carrier already has 3^(dimension+1) faces. Bounded
        # models use the actual pruned counts instead of this full-cube bound.
        minimum_counts = [binomial(big(dimension + 1), q) * big(2)^(dimension + 1 - q)
                          for q in 0:dimension+1]
        _rhomboid_storage_check!(minimum_counts, spec, n)
    end
    # Rational hull rebuilding pays for itself on measured moderate planar
    # depth-two windows. Full-depth hulls and small inputs favor exhaustive
    # supports; do not generalize this gate to unmeasured dimensions/depths.
    if backend === :auto
        backend = dimension == 2 && n >= 32 && depths !== nothing && last(depths) == 2 ?
                  :incremental : :exhaustive
    end
    mask_type = n <= 64 ? UInt64 : BigInt
    workspace_requirements = BigInt[0,0]
    cells, radii2 = _rhomboid_geometry(X, dimension, spec, mask_type; backend, workspace_requirements)
    return _CompiledRhomboidGeometry(cells, radii2, backend, workspace_requirements...)
end

function _compiled_rhomboid_geometry(X::Matrix{QQ}, spec::FiltrationSpec,
                                     cache::Union{Nothing,EncodingCache})
    # Exact contents and shape, not an object ID or a digest alone, identify the
    # Euclidean sites. Field, homology degree, radius cutoff, dimension, axes and
    # output representation do not change these carriers or constrained radii.
    key = cache === nothing ? nothing : (:rhomboid_geometry,
        _structural_cache_key((X,get(spec.params,:depth_range,nothing),get(spec.params,:backend,:auto))))
    geometry = _rhomboid_cached_geometry(cache,key) do
        _compile_rhomboid_geometry(X,spec)
    end
    # A smaller budget must not gain admission merely because an earlier caller
    # compiled with a larger one. Include the incremental hull workspace peaks.
    _rhomboid_storage_check!(length.(geometry.cells),spec,size(X,1))
    _construction_check_max_simplices!(geometry.workspace_simplices,0,spec)
    _construction_check_memory_budget!(geometry.workspace_bytes,spec)
    return geometry
end

function _graded_complex_from_point_cloud_rhomboid(data::PointCloud, spec::FiltrationSpec;
                                                   return_simplex_tree::Bool=false,
                                                   cache::Union{Nothing,EncodingCache}=nothing)
    _validate_geometric_filtration_request(data, spec)
    spec = _canonical_geometric_filtration_spec(spec)
    backend = get(spec.params, :backend, :auto)
    X = _rhomboid_points(data; allow_coincident=backend === :subdivision_cech)
    backend === :subdivision_cech && return _subdivision_cech_complex(X, spec; return_simplex_tree, cache)
    n = size(X,1)
    depths = get(spec.params,:depth_range,nothing)
    geometry = _compiled_rhomboid_geometry(X,spec,cache)
    radii2 = geometry.radii2
    backend = geometry.backend
    # Filtering/truncating a returned object must never change future queries.
    cutoff = get(spec.params, :radius, nothing)
    cells = cache === nothing ? geometry.cells :
            cutoff === nothing ? copy(geometry.cells) : copy.(geometry.cells)
    if cutoff !== nothing
        # Rationalize supplied floats before squaring; squaring a Float64 first
        # would round the cutoff. Algebraic cutoffs already support exact square.
        cutoff2 = cutoff isa AlgebraicReal ? cutoff^2 : QQ(cutoff)^2
        foreach(group -> filter!(cell -> radii2[cell] <= cutoff2, group), cells)
    end
    result = if depths === nothing
        requested = get(spec.params, :max_dim, nothing)
        requested === nothing || resize!(cells, min(length(cells) - 1, Int(requested)) + 1)
        return_simplex_tree ? _rhomboid_simplex_tree(cells, radii2, n, spec) :
                             _rhomboid_cellular_complex(cells, radii2, n)
    else
        # Slice original carriers before truncating dimension: a q-cell can
        # contribute an essential (q-1)-dimensional horizontal cap.
        _rhomboid_depth_complex(cells, radii2, X, spec; return_simplex_tree)
    end
    orientation = (1, -1)
    grades = return_simplex_tree ? result.grade_data : result.grades
    radius_axis = sort!(unique!(vcat(AlgebraicReal[0], first.(grades))))
    cutoff === nothing || sort!(unique!(push!(radius_axis, AlgebraicReal(cutoff))))
    lo, hi = depths === nothing ? (0, n) : depths
    depth_axis = AlgebraicReal[-k for k in hi:-1:lo]
    axes = get(spec.params, :axes, (radius_axis, depth_axis))
    _record_ingestion_backend(:multicover, backend)
    return result, axes, orientation
end
