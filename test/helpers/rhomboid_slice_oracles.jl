# Independent polytope-facet oracle for integer cube slices. Expected faces
# come from supporting hyperplanes of binary vertices, and signs from oriented
# determinants. No production incidence or subdivision formula is reused.

function _a76_slice_vertices(inside::Integer, on::Integer, level::Int, slab::Bool, n::Int)
    sites = [i for i in 1:n if !iszero(on & (one(on) << (i-1)))]
    vertices = Int[]
    for subset in 0:(1 << length(sites))-1
        vertex = Int(inside)
        for (j, site) in enumerate(sites)
            iszero(subset & (1 << (j-1))) || (vertex |= 1 << (site-1))
        end
        depth = count_ones(vertex)
        (slab ? level <= depth <= level+1 : depth == level) && push!(vertices, vertex)
    end
    return sort!(vertices)
end

function _a76_vertex_matrix(vertices, n)
    return QQ[iszero(vertex & (1 << (i-1))) ? 0 : 1 for i in 1:n, vertex in vertices]
end

# Translate the documented orientation convention into an ambient basis,
# recovering the minimal cube carrier directly from the polytope vertices.
function _a76_slice_basis(vertices, n)
    X = _a76_vertex_matrix(vertices, n)
    free = [i for i in 1:n if any(!=(X[i,1]), X[i,:])]
    horizontal = all(v -> count_ones(v) == count_ones(first(vertices)), vertices)
    dimension = isempty(free) ? 0 : length(free) - Int(horizontal)
    B = zeros(QQ, n, dimension)
    for j in 1:dimension
        B[free[j],j] = 1
        horizontal && (B[last(free),j] = -1)
    end
    return B
end

function _a76_determinant_sign(A)
    R = copy(A)
    orientation = 1
    for j in axes(R, 2)
        pivot = findfirst(i -> !iszero(R[i,j]), j:size(R,1))
        pivot === nothing && error("Independent facet orientation is singular")
        row = j + pivot - 1
        if row != j
            R[j,:], R[row,:] = copy(R[row,:]), copy(R[j,:])
            orientation = -orientation
        end
        orientation *= sign(R[j,j])
        for i in j+1:size(R,1)
            multiplier = R[i,j] / R[j,j]
            R[i,:] -= multiplier * R[j,:]
        end
    end
    return Int(orientation)
end

function _a76_geometric_facets(vertices, n)
    X = _a76_vertex_matrix(vertices, n)
    B = _a76_slice_basis(vertices, n)
    dimension = size(B,2)
    expected = Dict{Tuple,Int}()
    dimension == 0 && return expected
    _, coordinate_rows = _a72_rref(transpose(B))
    center = vec(sum(X; dims=2)) / length(vertices)
    candidates = Vector{Int}[]
    # These are all defining inequalities: 0 <= x_i <= 1 and the
    # lower/upper weight bounds. Redundant and coincident facets are removed.
    for i in 1:n, value in (0,1)
        push!(candidates, [v for v in vertices if Int(!iszero(v & (1 << (i-1)))) == value])
    end
    for depth in extrema(count_ones.(vertices))
        push!(candidates, [v for v in vertices if count_ones(v) == depth])
    end
    for face in candidates
        isempty(face) && continue
        F = _a76_vertex_matrix(face, n)
        _, pivots = _a72_rref(F .- F[:,1])
        length(pivots) == dimension-1 || continue
        tangent = _a76_slice_basis(face, n)
        @test size(tangent,2) == dimension-1
        outward = vec(sum(F; dims=2)) / length(face) - center
        coordinates = _a72_solve(B[coordinate_rows,:], hcat(outward,tangent)[coordinate_rows,:])
        coefficient = _a76_determinant_sign(coordinates)
        signature = Tuple(face)
        haskey(expected,signature) && @test expected[signature] == coefficient
        expected[signature] = coefficient
    end
    return expected
end

function _a76_check_slice_facets()
    total_cells, total_facets = 0, 0
    for q in 1:5, translated in (false,true)
        # Nonconsecutive labels and a nonzero anchor verify that the signs
        # use the ordered free coordinates, not raw site indices or depth.
        n = translated ? 2q+2 : q
        sites = translated ? collect(2:2:2q) : collect(1:q)
        inside = translated ? UInt64(1) | (UInt64(1) << (2q)) : UInt64(0)
        on = sum(UInt64(1) << (site-1) for site in sites)
        anchor = Int(count_ones(inside))
        carrier = DI._RhomboidCell(inside,on)
        for slab in (false,true), offset in (slab ? (0:q-1) : (1:q-1))
            level = anchor + offset
            cell = DI._RhomboidDepthCell(carrier,level,slab)
            vertices = _a76_slice_vertices(inside,on,level,slab,n)
            expected = _a76_geometric_facets(vertices,n)
            actual = Dict{Tuple,Int}()
            for (face,coefficient) in DI._rhomboid_depth_facets(cell,n)
                signature = Tuple(_a76_slice_vertices(face.carrier.inside,face.carrier.on,
                                                     face.level,face.slab,n))
                @test !haskey(actual,signature)
                actual[signature] = coefficient
            end
            @test actual == expected
            @test DI._rhomboid_depth_dimension(cell) == size(_a76_slice_basis(vertices,n),2)
            total_cells += 1
            total_facets += length(expected)
        end
    end
    @test total_cells == 50
    @test total_facets == 312
    println("A76 independent sliced-cell coverage: cells=",total_cells," oriented_facets=",total_facets)
    return nothing
end

function _a76_check_depth_boundaries(groups, complex, n)
    vertex_groups = [[Tuple(_a76_slice_vertices(cell.carrier.inside,cell.carrier.on,
                        cell.level,cell.slab,n)) for cell in group] for group in groups]
    for slot in 2:length(groups)
        rows = Dict(vertices => i for (i,vertices) in enumerate(vertex_groups[slot-1]))
        for (column,vertices) in enumerate(vertex_groups[slot])
            expected = zeros(Int,length(rows))
            for (face,coefficient) in _a76_geometric_facets(collect(vertices),n)
                @test haskey(rows,face)
                expected[rows[face]] = coefficient
            end
            @test collect(complex.boundaries[slot-1][:,column]) == expected
        end
    end
    return nothing
end
