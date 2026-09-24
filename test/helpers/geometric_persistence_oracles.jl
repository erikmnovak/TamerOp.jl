# Independent, deliberately small reference constructions for A72. No geometry,
# boundary, rank, nullspace, or quotient kernel from production is used here.

function _a72_rref(A)
    # Keep this independent elimination practical in focused --compile=min
    # harnesses too; the library's algebra is deliberately not reused here.
    Base.Experimental.@force_compile
    R = Matrix(A)
    pivots = Int[]
    row = 1
    numerical = eltype(R) <: AbstractFloat
    nonzero(x) = numerical ? abs(x) > 1e-10 : !iszero(x)
    for col in axes(R, 2)
        row > size(R, 1) && break
        candidates = [i for i in row:size(R, 1) if nonzero(R[i, col])]
        isempty(candidates) && continue
        pivot = numerical ? candidates[argmax(abs.(R[candidates, col]))] : first(candidates)
        R[row, :], R[pivot, :] = copy(R[pivot, :]), copy(R[row, :])
        @views R[row, :] ./= R[row, col]
        for i in axes(R, 1)
            i == row && continue
            multiplier = R[i, col]
            nonzero(multiplier) || continue
            @views R[i, :] .-= multiplier .* R[row, :]
        end
        push!(pivots, col)
        row += 1
    end
    return R, pivots
end

function _a72_solve(A, Y)
    if size(A, 2) == 0
        _a72_equal(Y, zero(Y)) || error("Reference right-hand side is outside the zero image")
        return zeros(eltype(A), 0, size(Y, 2))
    end
    R, pivots = _a72_rref(hcat(A, Y))
    pivots[1:size(A, 2)] == collect(1:size(A, 2)) || error("Reference solve is not injective")
    X = R[1:size(A, 2), size(A, 2)+1:end]
    _a72_equal(A * X, Y) || error("Reference right-hand side is outside the image")
    return X
end

function _a72_source_active(G, query, orientation)
    counts = DT.cell_counts(G)
    offsets = cumsum([0; counts])
    return [[j for j in 1:counts[slot] if
             any(g -> all(orientation[a]*g[a] <= orientation[a]*query[a] for a in 1:2),
                 DT.cell_grade_set(G, offsets[slot]+j))] for slot in eachindex(counts)]
end

function _a72_cech(X; maxdim=3)
    n = size(X, 1)
    simplices = [Tuple[] for _ in 0:maxdim]
    radii2 = Dict{Tuple,QQ}()
    for mask in 1:(1 << n)-1
        simplex = Tuple(i for i in 1:n if !iszero(mask & (1 << (i-1))))
        length(simplex) <= maxdim+1 || continue
        push!(simplices[length(simplex)], simplex)
        radii2[simplex] = _a72_minimum_ball_squared(X, mask)
    end
    foreach(sort!, simplices)
    return (; simplices, radii2, boundaries=_a72_boundaries(simplices))
end

function _a72_simplicial_comparison(tree, reference)
    source = [Tuple[] for _ in reference.simplices]
    for id in 1:DT.simplex_count(tree)
        push!(source[DT.simplex_dimension(tree, id)+1], Tuple(DT.simplex_vertices(tree, id)))
    end
    maps = SparseMatrixCSC{Int,Int}[]
    for slot in eachindex(source)
        rows = Dict(s => i for (i,s) in enumerate(reference.simplices[slot]))
        push!(maps, sparse([rows[s] for s in source[slot]], collect(eachindex(source[slot])),
                           ones(Int,length(source[slot])), length(rows), length(source[slot])))
    end
    return maps
end

_a72_equal(A, B) = eltype(A) <: AbstractFloat ?
    isapprox(A, B; atol=1e-9, rtol=1e-9) : A == B

function _a72_nullspace_from_rref(R, pivots)
    free = setdiff(collect(axes(R, 2)), pivots)
    Z = zeros(eltype(R), size(R, 2), length(free))
    for (j, col) in enumerate(free)
        Z[col, j] = one(eltype(R))
        for (i, pivot) in enumerate(pivots)
            Z[pivot, j] = -R[i, col]
        end
    end
    return Z
end

# Independent minimum-ball oracle: enumerate possible positive KKT supports,
# solve the bordered Gram system for barycentric weights, then check both
# primal containment and dual positivity in rational arithmetic. These are
# certificates of a global optimum of the convex minimum-enclosing-ball
# problem; no production circumsphere/minimum-ball predicate is called.
function _a72_minimum_ball_squared(X, selected::Int)
    selected == 0 && return zero(QQ)
    n, d = size(X)
    ids = [i for i in 1:n if !iszero(selected & (1 << (i-1)))]
    for support in 1:(1 << n)-1
        support & selected == support || continue
        vertices = [i for i in ids if !iszero(support & (1 << (i-1)))]
        m = length(vertices)
        m <= d+1 || continue
        B = zeros(QQ, m+1, m+1)
        y = zeros(QQ, m+1, 1)
        for i in 1:m
            y[i] = sum(abs2, X[vertices[i], :])
            B[i, m+1] = B[m+1, i] = one(QQ)
            for j in 1:m
                B[i, j] = 2sum(X[vertices[i], a]*X[vertices[j], a] for a in 1:d)
            end
        end
        y[end] = one(QQ)
        _, pivots = _a72_rref(B)
        length(pivots) == m+1 || continue
        weights = _a72_solve(B, y)[1:m, 1]
        all(>=(0), weights) || continue
        center = [sum(weights[j]*X[vertices[j], a] for j in 1:m) for a in 1:d]
        radius2 = sum((center[a]-X[first(vertices), a])^2 for a in 1:d)
        all(i -> sum((center[a]-X[i, a])^2 for a in 1:d) <= radius2, ids) || continue
        return radius2
    end
    error("No rational minimum-ball certificate found")
end

function _a72_boundaries(simplices)
    boundaries = SparseMatrixCSC{Int,Int}[]
    for q in 1:length(simplices)-1
        row = Dict(s => i for (i, s) in enumerate(simplices[q]))
        I, J, V = Int[], Int[], Int[]
        for (j, s) in enumerate(simplices[q+1]), omit in eachindex(s)
            face = Tuple(s[a] for a in eachindex(s) if a != omit)
            push!(I, row[face]); push!(J, j); push!(V, isodd(omit) ? 1 : -1)
        end
        push!(boundaries, sparse(I, J, V, length(row), length(simplices[q+1])))
    end
    return boundaries
end

function _a72_subdivision_cech(X; maxdim=3)
    n = size(X, 1)
    masks = collect(0:(1 << n)-1)
    radii2 = [_a72_minimum_ball_squared(X, mask) for mask in masks]
    simplices = [Tuple[] for _ in 0:maxdim]
    function extend(chain)
        push!(simplices[length(chain)], Tuple(chain))
        length(chain) > maxdim && return
        for next in masks
            next != last(chain) && next & last(chain) == last(chain) || continue
            extend([chain; next])
        end
    end
    foreach(mask -> extend([mask]), masks)
    foreach(sort!, simplices)
    return (; simplices, radii2, boundaries=_a72_boundaries(simplices))
end

# Oriented subdivision of a cubical cell. Site order is the cubical orientation;
# the sign of a maximal chain is its generator permutation's parity. This is a
# chain map into subdivision-Cech, and its restrictions commute strictly with
# both filtration parameters. Empty subsets supply the contractible k=0 level.
function _a72_rhomboid_comparison(cells, reference)
    maps = SparseMatrixCSC{Int,Int}[]
    for slot in eachindex(reference.simplices)
        rows = Dict(s => i for (i, s) in enumerate(reference.simplices[slot]))
        I, J, V = Int[], Int[], Int[]
        for (j, cell) in enumerate(cells[slot])
            generators = [i for i in 0:62 if !iszero(cell.on & (UInt64(1) << i))]
            function permutations(prefix, remaining)
                if isempty(remaining)
                    mask = Int(cell.inside)
                    chain = [mask]
                    for site in prefix
                        mask |= 1 << site
                        push!(chain, mask)
                    end
                    inversions = sum(prefix[a] > prefix[b] for a in eachindex(prefix) for b in a+1:length(prefix); init=0)
                    push!(I, rows[Tuple(chain)]); push!(J, j); push!(V, isodd(inversions) ? -1 : 1)
                else
                    for a in eachindex(remaining)
                        permutations([prefix; remaining[a]], [remaining[b] for b in eachindex(remaining) if b != a])
                    end
                end
            end
            permutations(Int[], generators)
        end
        push!(maps, sparse(I, J, V, length(rows), length(cells[slot])))
    end
    return maps
end

function _a72_reference_homology(reference, active, degree, field)
    K = CM.coeff_type(field)
    convert(A) = K[CM.coerce(field, x) for x in A]
    outgoing = degree == 0 ? zeros(K, 0, length(active[1])) :
        convert(reference.boundaries[degree][active[degree], active[degree+1]])
    incoming = convert(reference.boundaries[degree+1][active[degree+1], active[degree+2]])
    reduced_outgoing, outgoing_pivots = _a72_rref(outgoing)
    nullity = size(outgoing, 2) - length(outgoing_pivots)
    _, bpivots = _a72_rref(incoming)
    if length(bpivots) == nullity
        # Independent outgoing/incoming reductions give dim ker(d_q) =
        # dim im(d_{q+1}). The integer boundary-square check establishes
        # im(d_{q+1}) subset ker(d_q), hence equality over the chosen field.
        # Every subsequently verified cycle is therefore a boundary. No
        # nullspace matrix, [B,Z] reduction, or boundary-space inverse is
        # needed for this mathematically certified zero homology case.
        if degree > 0
            lower = reference.boundaries[degree][active[degree], active[degree+1]]
            upper = reference.boundaries[degree+1][active[degree+1], active[degree+2]]
            iszero(lower * upper) || error("Reference boundaries do not form a complex")
        end
        H = zeros(K, size(outgoing, 2), 0)
        return (; basis=H, cycles=nothing, nboundaries=length(bpivots), outgoing,
                coordinate_rows=Int[], coordinate_inverse=zeros(K, 0, 0))
    end
    Z = _a72_nullspace_from_rref(reduced_outgoing, outgoing_pivots)
    B = incoming[:, bpivots]
    _, pivots = _a72_rref(hcat(B, Z))
    hpivots = [j-size(B, 2) for j in pivots if j > size(B, 2)]
    H = Z[:, hpivots]
    cycles = hcat(B, H)
    # C=[B,H] has independent columns. Selecting independent rows gives a
    # square invertible minor, whose inverse supplies coordinates on im(C).
    # This reference-only factorization is reused for all incoming arrows;
    # every query below still verifies reconstruction in the entire chain
    # group, so an element outside im(C) cannot silently acquire coordinates.
    _, coordinate_rows = _a72_rref(transpose(cycles))
    ncycles = size(cycles, 2)
    length(coordinate_rows) == ncycles || error("Reference cycle basis is dependent")
    coordinate_inverse = _a72_solve(cycles[coordinate_rows, :],
                                   Matrix{K}(I, ncycles, ncycles))
    return (; basis=H, cycles, nboundaries=size(B, 2), outgoing, coordinate_rows, coordinate_inverse)
end

function _a72_coordinates(H, cycles)
    if size(H.basis, 2) == 0
        boundary = H.outgoing * cycles
        _a72_equal(boundary, zero(boundary)) || error("Reference input is not a cycle")
        return zeros(eltype(cycles), 0, size(cycles, 2))
    end
    coordinates = H.coordinate_inverse * cycles[H.coordinate_rows, :]
    _a72_equal(H.cycles * coordinates, cycles) ||
        error("Reference right-hand side is outside the cycle space")
    return coordinates[H.nboundaries+1:end, :]
end

function _a72_lift_homology(data, vertex)
    q = Matrix(data.q.comps[vertex])
    _, pivots = _a72_rref(q)
    h = size(q, 1)
    inverse = zeros(eltype(q), size(q, 2), h)
    h == 0 || (inverse[pivots, :] = _a72_solve(q[:, pivots], Matrix{eltype(q)}(I, h, h)))
    return data.iZ.comps[vertex] * inverse
end

# The public H0 fast path uses the least global vertex of each connected
# component, ordered increasingly. Derive those representatives independently
# by graph traversal (not the production union-find), so they can be compared
# with the unrelated elimination basis of cohomology_module_data.
function _a72_component_lifts(G, active, field)
    vertices = active[1]
    @test issorted(vertices)
    local_index = Dict(v => i for (i, v) in enumerate(vertices))
    adjacency = [Int[] for _ in vertices]
    if length(active) >= 2
        boundary = G.boundaries[1]
        for edge in active[2]
            rows, values = findnz(boundary[:, edge])
            @test length(rows) == 2 && sort(values) == [-1, 1]
            a, b = local_index[rows[1]], local_index[rows[2]]
            push!(adjacency[a], b)
            push!(adjacency[b], a)
        end
    end
    seen = falses(length(vertices))
    representatives = Int[]
    for vertex in eachindex(vertices)
        seen[vertex] && continue
        push!(representatives, vertex)
        seen[vertex] = true
        queue = [vertex]
        while !isempty(queue)
            for neighbor in adjacency[pop!(queue)]
                seen[neighbor] && continue
                seen[neighbor] = true
                push!(queue, neighbor)
            end
        end
    end
    K = CM.coeff_type(field)
    lifts = zeros(K, length(vertices), length(representatives))
    for (column, row) in enumerate(representatives)
        lifts[row, column] = one(K)
    end
    return lifts
end

function _a72_check_module_comparison(data, spec, G, reference, comparison,
                                       queries, active_reference, active_source,
                                       degree, expected, field; cache=:auto)
    # Encoding axes contain every actual birth. Query-only coarse axes would
    # floor-snap births and change the module whose comparison is tested.
    orientation = get(spec.params, :orientation, spec.kind === :function_delaunay ? (1,1) : (1,-1))
    critical_axes = DI._axes_from_complex_grades(G, orientation)
    exact_axes = ntuple(i -> sort!(unique!(vcat(critical_axes[i],
        [orientation[i]*p[i] for p in queries]))), 2)
    spec = OPT.FiltrationSpec(; kind=spec.kind, merge(spec.params,(axes=exact_axes,))...)
    if degree == 1 && CM.coeff_type(field) == QQ
        for i in eachindex(queries), slot in eachindex(comparison)
            rows, _, _ = findnz(comparison[slot][:, active_source[i][slot]])
            retained = Set(active_reference[i][slot])
            @test all(r -> r in retained, rows)
        end
    end
    enc = DI.encode(data, spec; degree, field, stage=:encoding_result, cache)
    M = RES.encoding_module(enc)
    complex = DI.encode(data, spec; degree, field, stage=:cochain, cache)
    homology = TamerOp.ModuleComplexes.cohomology_module_data(complex, -degree)
    labels = [EC.locate(enc.pi, collect(p)) for p in queries]
    K = CM.coeff_type(field)
    # Repeated query parameters can describe exactly the same stalk. Sharing
    # the independent elimination result does not share production geometry or
    # persistence algebra and keeps the all-field oracle practical.
    reference_cache = Dict{Any,Any}()
    targets = [get!(reference_cache, Tuple(Tuple(v) for v in a)) do
                   _a72_reference_homology(reference, a, degree, field)
               end for a in active_reference]
    maps = Matrix{K}[]
    quotient_identifications = Matrix{K}[]
    for (i, u) in enumerate(labels)
        @test MD.dim_at(M, u) == expected[i] == size(targets[i].basis, 2)
        @test MD.dim_at(homology.H, u) == expected[i]
        S = comparison[degree+1][active_reference[i][degree+1], active_source[i][degree+1]]
        chainmap = K[CM.coerce(field, x) for x in S]
        quotient_lifts = _a72_lift_homology(homology, u)
        public_lifts = degree == 0 ? _a72_component_lifts(G, active_source[i], field) : quotient_lifts
        @test size(public_lifts, 2) == expected[i]
        identification = homology.q.comps[u] * _a72_solve(homology.iZ.comps[u], public_lifts)
        push!(quotient_identifications, Matrix(identification))
        @test length(last(_a72_rref(identification))) == expected[i]
        push!(maps, _a72_coordinates(targets[i], chainmap * public_lifts))
        quotient_comparison = _a72_coordinates(targets[i], chainmap * quotient_lifts)
        @test _a72_equal(last(maps), quotient_comparison * identification)
        @test length(last(_a72_rref(last(maps)))) == expected[i]
    end
    orientation = get(spec.params, :orientation, spec.kind === :function_delaunay ? (1,1) : (1,-1))
    for i in eachindex(queries), j in eachindex(queries)
        all(orientation[a]*queries[i][a] <= orientation[a]*queries[j][a] for a in 1:2) || continue
        # Intertwine the public component basis with the explicit quotient
        # witness basis, then check the independent carrier comparison. These
        # two mathematically valid bases need not give identical raw matrices.
        A = MD.structure_map(M; source=labels[i], target=labels[j])
        Ah = MD.structure_map(homology.H; source=labels[i], target=labels[j])
        @test _a72_equal(quotient_identifications[j] * A, Ah * quotient_identifications[i])
        inclusion = zeros(K, length(active_reference[j][degree+1]), length(active_reference[i][degree+1]))
        rows = Dict(v => r for (r, v) in enumerate(active_reference[j][degree+1]))
        for (c, v) in enumerate(active_reference[i][degree+1])
            inclusion[rows[v], c] = one(K)
        end
        B = _a72_coordinates(targets[j], inclusion * targets[i].basis)
        @test _a72_equal(maps[j] * A, B * maps[i])
    end
    # Every oriented rectangle is tested, including nonzero radius transports
    # and the depth/level direction. Same target coordinates require equality.
    radius_values = sort!(unique(first.(queries)))
    level_values = sort!(unique(last.(queries)); rev=orientation[2] == -1)
    query_labels = Dict(queries[i] => labels[i] for i in eachindex(queries))
    for i in 1:length(radius_values)-1, k in i+1:length(radius_values),
        j in 1:length(level_values)-1, l in j+1:length(level_values)
        corners = ((radius_values[i], level_values[j]),
                   (radius_values[i], level_values[l]),
                   (radius_values[k], level_values[j]),
                   (radius_values[k], level_values[l]))
        all(p -> haskey(query_labels, p), corners) || continue
        a, b, c, d = map(p -> query_labels[p], corners)
        @test _a72_equal(MD.structure_map(M;source=b,target=d)*MD.structure_map(M;source=a,target=b),
                         MD.structure_map(M;source=c,target=d)*MD.structure_map(M;source=a,target=c))
    end
    return (; module_object=M, labels, comparison_maps=maps, reference_homology=targets)
end
