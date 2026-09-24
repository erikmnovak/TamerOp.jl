# Independent signed cellular subdivision followed by the carrier-to-Cech map.
# Requires geometric_persistence_oracles.jl for independently certified balls,
# simplicial boundaries and quotient coordinates. No production subdivision,
# geometry predicate, quotient, or incidence routine is used here.
function _a66_carrier_comparison(boundaries, unions, reference)
    length(unions) <= length(reference.simplices) || error("reference skeleton too short")
    comparison = SparseMatrixCSC{Int,Int}[]
    rows0 = Dict(s => i for (i, s) in enumerate(reference.simplices[1]))
    push!(comparison, sparse([rows0[(Int(mask),)] for mask in unions[1]],
        collect(eachindex(unions[1])), ones(Int, length(unions[1])),
        length(reference.simplices[1]), length(unions[1])))
    for dimension in 1:length(unions)-1
        row = Dict(s => i for (i, s) in enumerate(reference.simplices[dimension+1]))
        previous = last(comparison)
        boundary = boundaries[dimension]
        size(boundary) == (length(unions[dimension]), length(unions[dimension+1])) ||
            error("source cellular boundary has wrong shape")
        I, J, V = Int[], Int[], Int[]
        sign = isodd(dimension) ? -1 : 1
        for cell in eachindex(unions[dimension+1])
            apex = Int(unions[dimension+1][cell])
            for facet in eachindex(unions[dimension])
                incidence = boundary[facet, cell]
                iszero(incidence) && continue
                for stored in nzrange(previous, facet)
                    flag = reference.simplices[dimension][previous.rowval[stored]]
                    last(flag) & apex == last(flag) || error("carrier assignment is not monotone on faces")
                    # Degenerate simplices vanish in normalized simplicial chains.
                    last(flag) == apex && continue
                    image = (flag..., apex)
                    push!(I, row[image]); push!(J, cell)
                    push!(V, sign * incidence * previous.nzval[stored])
                end
            end
        end
        C = sparse(I, J, V, length(row), length(unions[dimension+1]))
        dropzeros!(C)
        push!(comparison, C)
        @test reference.boundaries[dimension] * C == previous * boundary
    end
    return comparison
end

# Certify grade preservation at every nonzero chain coefficient, using the
# independent exact minimum-ball radii attached to the reference subset flags.
function _a66_check_carrier_grades(G, reference, comparison)
    offsets = cumsum([0; DT.cell_counts(G)])
    for slot in eachindex(comparison)
        rows, columns, values = findnz(comparison[slot])
        for entry in eachindex(values)
            flag = reference.simplices[slot][rows[entry]]
            radius2 = reference.radii2[last(flag)+1]
            depth = count_ones(first(flag))
            grade = only(DT.cell_grade_set(G, offsets[slot]+columns[entry]))
            @test radius2 <= grade[1]^2
            @test depth >= grade[2]
        end
    end
    return nothing
end

# A source simplex gives an ordered flag of carrier subsets. Repeated carrier
# labels vanish in normalized chains; the vertex-order sign is explicit. This
# handles both barycentric sliced cells and the direct subdivision-Cech backend.
function _a66_flag_carrier_comparison(source_flags, reference)
    maps = SparseMatrixCSC{Int,Int}[]
    for slot in eachindex(source_flags)
        row = Dict(s => i for (i, s) in enumerate(reference.simplices[slot]))
        I, J, V = Int[], Int[], Int[]
        for (column, source) in enumerate(source_flags[slot])
            ordered = sort!(Int.(collect(source)); by=count_ones)
            length(unique(ordered)) == length(ordered) || continue
            inversions = sum((source[i] > source[j] for i in 1:length(source) for j in i+1:length(source)); init=0)
            push!(I, row[Tuple(ordered)]); push!(J, column)
            push!(V, isodd(inversions) ? -1 : 1)
        end
        push!(maps, sparse(I, J, V, length(row), length(source_flags[slot])))
    end
    return maps
end

# Full and capped models compare through their explicit chain-induced
# isomorphisms to the same independently constructed persistence module.
function _a66_compare_through_reference(full, capped, queries)
    forward = [_a72_solve(capped.comparison_maps[i], full.comparison_maps[i]) for i in eachindex(queries)]
    backward = [_a72_solve(full.comparison_maps[i], capped.comparison_maps[i]) for i in eachindex(queries)]
    K = eltype(first(forward))
    for i in eachindex(queries)
        @test _a72_equal(backward[i] * forward[i], Matrix{K}(I, size(forward[i], 2), size(forward[i], 2)))
        @test _a72_equal(forward[i] * backward[i], Matrix{K}(I, size(forward[i], 1), size(forward[i], 1)))
    end
    for i in eachindex(queries), j in eachindex(queries)
        queries[i][1] <= queries[j][1] && queries[i][2] >= queries[j][2] || continue
        A = MD.map_leq(full.module_object, full.labels[i], full.labels[j])
        B = MD.map_leq(capped.module_object, capped.labels[i], capped.labels[j])
        @test _a72_equal(forward[j] * A, B * forward[i])
    end
    return nothing
end

function _a66_native_fixture(X, spec)
    dimension = DI._rhomboid_affine_dimension(X)
    cells, radii2 = DI._rhomboid_geometry(X, dimension, spec, UInt64;
                                        backend=get(spec.params, :backend, :exhaustive))
    cutoff = get(spec.params, :radius, nothing)
    cutoff === nothing || foreach(group -> filter!(cell -> radii2[cell] <= cutoff^2, group), cells)
    depth_range = get(spec.params, :depth_range, nothing)
    if depth_range === nothing
        requested = get(spec.params, :max_dim, nothing)
        requested === nothing || resize!(cells, min(length(cells)-1, requested)+1)
        G = DI._rhomboid_cellular_complex(cells, radii2, size(X, 1))
        unions = [[cell.inside | cell.on for cell in group] for group in cells]
    else
        groups, G = DI._rhomboid_depth_model(cells, radii2, size(X, 1), spec)
        unions = [[cell.carrier.inside | cell.carrier.on for cell in group] for group in groups]
    end
    actual = DI.encode(DT.PointCloud(X), spec; stage=:graded_complex)
    @test DT.cell_counts(actual) == DT.cell_counts(G)
    @test actual.grades == G.grades
    @test actual.boundaries == G.boundaries
    return G, unions
end

function _a66_active_reference(reference, queries)
    return [[[i for (i, s) in enumerate(group)
              if reference.radii2[last(s)+1] <= r^2 && count_ones(first(s)) >= k]
             for group in reference.simplices] for (r, k) in queries]
end

# A flag of nonempty subsets maps to an ordinary Cech simplex by choosing the
# least site of each subset. Every chosen site lies in the largest subset, so
# the resulting simplex is present by the same independent ball certificate.
# This compact comparison is used at depth one for the larger hexagon fixture.
function _a68_ordinary_cech_comparison(source_flags, reference)
    maps = SparseMatrixCSC{Int,Int}[]
    for slot in eachindex(source_flags)
        row = Dict(s => i for (i, s) in enumerate(reference.simplices[slot]))
        I, J, V = Int[], Int[], Int[]
        for (column, flag) in enumerate(source_flags[slot])
            all(x -> !iszero(x), flag) || error("ordinary Cech comparison requires positive depth")
            sites = [trailing_zeros(mask) + 1 for mask in flag]
            allunique(sites) || continue
            inversions = sum((sites[i] > sites[j] for i in 1:length(sites) for j in i+1:length(sites)); init=0)
            push!(I, row[Tuple(sort(sites))]); push!(J, column)
            push!(V, isodd(inversions) ? -1 : 1)
        end
        push!(maps, sparse(I, J, V, length(row), length(source_flags[slot])))
    end
    return maps
end

function _a68_flag_fixture(X, spec)
    data = DT.PointCloud(X)
    tree = DI.encode(data, spec; stage=:simplex_tree)
    G = DI.encode(data, spec; stage=:graded_complex)
    masks, radii2 = DI._subdivision_cech_vertices(X, spec)
    flags = [Tuple[] for _ in DT.cell_counts(G)]
    for id in 1:DT.simplex_count(tree)
        vertices = DT.simplex_vertices(tree, id)
        push!(flags[DT.simplex_dimension(tree, id)+1], Tuple(Int(masks[v]) for v in vertices))
    end
    @test length.(flags) == DT.cell_counts(G)
    @test G.boundaries == DI._graded_complex_from_simplex_tree(tree).boundaries
    return G, flags, masks, radii2
end
