using JSON3

# Independent ordinary-persistence oracles. The cubical reference below forms
# tensor products of interval/circle chain complexes directly; it uses neither
# DataIngestion geometry nor the production persistence reduction or field algebra.

function _a41_check_retained_cycle(G, representative)
    @test representative.available
    @test representative.field == CM.F2()
    @test representative.choice === :noncanonical_f2_column_reduction
    @test representative.source_geometry === :not_asserted
    dim = representative.dimension
    chain = representative.cycle
    counts = length.(G.cells_by_dim)
    z = zeros(Int, counts[dim + 1])
    z[collect(chain.cell_indices)] .= collect(chain.coefficients)
    @test chain.dimension == dim
    @test chain.cell_ids == Tuple(G.cells_by_dim[dim + 1][collect(chain.cell_indices)])
    @test all(==(1), chain.coefficients)
    dim == 0 || @test all(iseven, G.boundaries[dim] * z)
    birth, death = representative.interval
    order = representative.order
    @test all(g -> order === :sublevel ? g <= birth : g >= birth, chain.cell_grades)
    # Direct ranks in each source subcomplex independently certify that this
    # same cycle is nonzero before death, and is a boundary from death onward.
    for level in unique(first.(G.grades))
        born = order === :sublevel ? birth <= level : birth >= level
        born || continue
        active_next = dim + 2 <= length(counts) ?
            findall(g -> order === :sublevel ? g[1] <= level : g[1] >= level,
                    G.grades[sum(counts[1:(dim + 1)]) .+ (1:counts[dim + 2])]) : Int[]
        B = dim + 2 <= length(counts) ? Matrix(G.boundaries[dim + 1][:, active_next]) : zeros(Int, length(z), 0)
        alive = order === :sublevel ? level < death : level > death
        @test _a21_binary_rank(hcat(B, z)) - _a21_binary_rank(B) == Int(alive)
    end
    if representative.kind === :finite
        filling = representative.bounding_chain
        @test filling !== nothing
        @test filling.dimension == dim + 1
        c = zeros(Int, counts[dim + 2])
        c[collect(filling.cell_indices)] .= collect(filling.coefficients)
        @test mod.(G.boundaries[dim + 1] * c, 2) == z
        @test all(g -> order === :sublevel ? g <= death : g >= death, filling.cell_grades)
    else
        @test representative.bounding_chain === nothing
    end
    return z
end

function _a21_binary_rref(A::AbstractMatrix)
    R = isodd.(Int.(A))
    pivots = Int[]
    row = 1
    for col in axes(R, 2)
        row > size(R, 1) && break
        found = findfirst(i -> R[i, col], row:size(R, 1))
        found === nothing && continue
        pivot = row + found - 1
        R[row, :], R[pivot, :] = copy(R[pivot, :]), copy(R[row, :])
        for other in axes(R, 1)
            other == row && continue
            R[other, col] && (R[other, :] .= xor.(R[other, :], R[row, :]))
        end
        push!(pivots, col)
        row += 1
    end
    return R, pivots
end

_a21_binary_rank(A::AbstractMatrix) = length(last(_a21_binary_rref(A)))

function _a21_binary_cycles(A::AbstractMatrix)
    R, pivots = _a21_binary_rref(A)
    free = setdiff(collect(axes(A, 2)), pivots)
    Z = falses(size(A, 2), length(free))
    for (j, f) in enumerate(free)
        Z[f, j] = true
        for (i, p) in enumerate(pivots)
            Z[p, j] = R[i, f]
        end
    end
    return Z
end

function _a21_cubical_oracle(values::AbstractArray{T,N}, periodic,
                            input::Symbol, order::Symbol; signed::Bool=false) where {T,N}
    # A positive axis label is a vertex; a negative label is an edge. Even for
    # a one-vertex circle, both endpoints are retained before F2 cancellation.
    nv = ntuple(i -> size(values, i) + (input === :top_cells && !periodic[i]), N)
    ne = ntuple(i -> nv[i] - !periodic[i], N)
    factors = ntuple(i -> vcat(collect(1:nv[i]), -collect(1:ne[i])), N)
    cells = [NTuple{N,Int}[] for _ in 0:N]
    for cell in Iterators.product(factors...)
        push!(cells[count(x -> x < 0, cell) + 1], cell)
    end
    index = [Dict(c => i for (i, c) in enumerate(cs)) for cs in cells]
    function faces(cell)
        out = NTuple{N,Int}[]
        for axis in 1:N
            cell[axis] < 0 || continue
            lo = -cell[axis]
            hi = lo == nv[axis] ? 1 : lo + 1
            push!(out, ntuple(i -> i == axis ? lo : cell[i], N))
            push!(out, ntuple(i -> i == axis ? hi : cell[i], N))
        end
        return out
    end
    boundaries = [zeros(Int, length(cells[d]), length(cells[d + 1])) for d in 1:N]
    for d in 1:N, (j, cell) in enumerate(cells[d + 1]), (f, face) in enumerate(faces(cell))
        i = index[d][face]
        # Tensor boundary: (-1)^(preceding edge factors) * (high - low).
        coefficient = isodd(cld(f, 2)) ? (isodd(f) ? -1 : 1) : (isodd(f) ? 1 : -1)
        boundaries[d][i, j] += signed ? coefficient : 1
    end
    cell_values = Dict{NTuple{N,Int},T}()
    if input === :top_cells
        best = order === :sublevel ? min : max
        for cell in cells[end]
            cell_values[cell] = values[map(abs, cell)...]
        end
        for d in N:-1:1, cell in cells[d + 1], face in faces(cell)
            value = cell_values[cell]
            cell_values[face] = haskey(cell_values, face) ?
                best(cell_values[face], value) : value
        end
    else
        aggregate = order === :sublevel ? maximum : minimum
        for cs in cells, cell in cs
            corners = ntuple(i -> cell[i] > 0 ? (cell[i],) :
                (-cell[i], -cell[i] == nv[i] ? 1 : -cell[i] + 1), N)
            cell_values[cell] = aggregate(values[corner...] for corner in Iterators.product(corners...))
        end
    end
    grades = [[cell_values[c] for c in cs] for cs in cells]
    return (; boundaries, grades, cells)
end

function _a21_homology_map_rank(reference, dim, source, target, order)
    active(level) = [findall(g -> order === :sublevel ? g <= level : g >= level, gs)
                     for gs in reference.grades]
    src = active(source)
    dst = active(target)
    degree = dim + 1
    ds = dim == 0 ? falses(0, length(src[degree])) :
        reference.boundaries[dim][src[degree - 1], src[degree]]
    cycles = _a21_binary_cycles(ds)
    embedded = falses(length(dst[degree]), size(cycles, 2))
    dst_index = Dict(c => i for (i, c) in enumerate(dst[degree]))
    for (i, cell) in enumerate(src[degree])
        embedded[dst_index[cell], :] = cycles[i, :]
    end
    boundaries = degree == length(src) ? falses(length(dst[degree]), 0) :
        reference.boundaries[degree][dst[degree], dst[degree + 1]]
    # Image H_d(source)->H_d(target) is (Z_source+B_target)/B_target.
    return _a21_binary_rank(hcat(embedded, boundaries)) - _a21_binary_rank(boundaries)
end

function _a21_barcode_map_rank(diagram, dim, source, target, order)
    finite = OP.finite_intervals(diagram; dim=dim)
    essential = OP.essential_births(diagram; dim=dim)
    if order === :sublevel
        return count(b -> b[1] <= source && target < b[2], finite) +
               count(b -> b <= source, essential)
    end
    return count(b -> b[1] >= source && target > b[2], finite) +
           count(b -> b >= source, essential)
end

function _a21_test_cubical_maps(values, periodic, input, order)
    reference = _a21_cubical_oracle(values, periodic, input, order)
    diagram = OP.cubical_persistence(values; periodic=periodic, input=input,
                                     order=order, field=CM.F2())
    for d in 2:length(reference.boundaries)
        @test all(iseven, Int.(reference.boundaries[d - 1]) * Int.(reference.boundaries[d]))
    end
    events = sort!(unique(vec(values)))
    levels = sort!(unique(vcat(events, [first(events) - 1, last(events) + 1],
        [(events[i] + events[i + 1]) / 2 for i in 1:(length(events) - 1)])))
    order === :superlevel && reverse!(levels)
    for dim in 0:(length(reference.grades) - 1), i in eachindex(levels), j in i:length(levels)
        expected = _a21_homology_map_rank(reference, dim, levels[i], levels[j], order)
        @test _a21_barcode_map_rank(diagram, dim, levels[i], levels[j], order) == expected
    end
    @test OP.check_persistence_diagram(diagram; throw=false).valid
    return diagram
end

@testset "A21 independent periodic cubical persistence maps" begin
    for shape in ((1, 1), (1, 4), (2, 2), (3, 3)),
        periodic in ((false, false), (true, false), (false, true), (true, true)),
        input in (:top_cells, :vertices), order in (:sublevel, :superlevel)
        values = [mod(3i + 2j + i*j, 4)//1 for i in 1:shape[1], j in 1:shape[2]]
        @testset "$shape $periodic $input $order" begin
            _a21_test_cubical_maps(values, periodic, input, order)
        end
    end
    # Three-dimensional vertex cubes exercise genuine tensor boundaries too.
    for periodic in ((false, false, false), (true, false, true)), order in (:sublevel, :superlevel)
        values = reshape([0, 1, 1, 2, 2, 1, 1, 0], 2, 2, 2)
        _a21_test_cubical_maps(values, periodic, :vertices, order)
    end
end

@testset "A21 known cubical bars and interval conventions" begin
    for input in (:top_cells, :vertices), shape in ((1, 1), (1, 4), (2, 2)),
        (periodic, betti) in (((false, false), (1, 0, 0)),
                             ((true, false), (1, 1, 0)),
                             ((false, true), (1, 1, 0)),
                             ((true, true), (1, 2, 1)))
        diagram = OP.cubical_persistence(fill(2//3, shape); periodic=periodic, input=input)
        @test Tuple(length(OP.essential_births(diagram; dim=d)) for d in 0:2) == betti
        @test all(isempty(OP.finite_intervals(diagram; dim=d)) for d in 0:2)
        @test all(all(==(2//3), OP.essential_births(diagram; dim=d)) for d in 0:2)
        if all(periodic)
            @test OP.check_torus_persistence(diagram; check_h2=true, throw=false).valid
        else
            @test !OP.check_torus_persistence(diagram; check_h2=true, throw=false).valid
            @test_throws ArgumentError OP.check_torus_persistence(diagram; check_h2=true, throw=true)
        end
    end
    for input in (:top_cells, :vertices), order in (:sublevel, :superlevel)
        values = zeros(Int, 3, 3)
        values[2, 2] = 5
        order === :superlevel && (values .= 5 .- values)
        diagram = _a21_test_cubical_maps(values, (false, false), input, order)
        expected = order === :sublevel ? [(0, 5)] : [(5, 0)]
        @test OP.finite_intervals(diagram; dim=1) == expected
        @test isempty(OP.essential_births(diagram; dim=1))
        @test OP.persistence_intervals(diagram; dim=0) ==
              (order === :sublevel ? [(0, Inf)] : [(5, -Inf)])
        # Birth is included and death excluded, including reversed filtration order.
        b, d = only(expected)
        @test _a21_barcode_map_rank(diagram, 1, b, b, order) == 1
        @test _a21_barcode_map_rank(diagram, 1, d, d, order) == 0
    end
end

@testset "A21 exact ordinary grades and F2 reduction" begin
    a = BigInt(1)//BigInt(1)
    b = a + BigInt(1)//(BigInt(2)^70)
    @test a < b && Float64(a) == Float64(b)
    for input in (:top_cells, :vertices), order in (:sublevel, :superlevel)
        values = fill(order === :sublevel ? a : b, 3, 3)
        values[2, 2] = order === :sublevel ? b : a
        diagram = _a21_test_cubical_maps(values, (false, false), input, order)
        @test OP.finite_intervals(diagram; dim=1) ==
              (order === :sublevel ? [(a, b)] : [(b, a)])
        @test eltype(OP.finite_intervals(diagram; dim=1)) == Tuple{typeof(a),typeof(a)}
        @test eltype(OP.essential_births(diagram; dim=0)) == typeof(a)
    end
    # A triangle filled one step after its boundary: two H0 deaths and one H1 death.
    d1 = sparse([-1 -1 0; 1 0 -1; 0 1 1])
    d2 = sparse(reshape([1, -1, 1], 3, 1))
    G = DT.GradedComplex([[1, 2, 3], [1, 2, 3], [1]], [d1, d2],
                          [(a,), (a,), (a,), (b,), (b,), (b,), (b + 1,)])
    diagram = OP.persistence_diagram(G; field=CM.F2())
    @test OP.finite_intervals(diagram; dim=0) == [(a, b), (a, b)]
    @test OP.essential_births(diagram; dim=0) == [a]
    @test OP.finite_intervals(diagram; dim=1) == [(b, b + 1)]
    @test isempty(OP.persistence_intervals(diagram; dim=2))
    # Permuting tied vertices/edges and changing orientations preserves the object.
    perm = [3, 1, 2]
    H = DT.GradedComplex([[1, 2, 3], [1, 2, 3], [1]],
        [-d1[perm, perm], -d2[perm, :]], copy(G.grades))
    other = OP.persistence_diagram(H)
    @test all(OP.persistence_intervals(diagram; dim=d) == OP.persistence_intervals(other; dim=d) for d in 0:2)
    # The RP2 cellular boundary is multiplication by 2, hence zero over F2.
    rp2 = DT.GradedComplex([[1], [1], [1]],
        [spzeros(Int, 1, 1), sparse(reshape([2], 1, 1))], [(0,), (1,), (2,)])
    rp2_diagram = OP.persistence_diagram(rp2)
    @test [OP.essential_births(rp2_diagram; dim=d) for d in 0:2] == [[0], [1], [2]]
    @test all(isempty(OP.finite_intervals(rp2_diagram; dim=d)) for d in 0:2)
    for field in (:f2, :F2, CM.QQField(), CM.RealField(Float64))
        @test_throws ArgumentError OP.persistence_diagram(G; field=field)
        @test_throws ArgumentError OP.cubical_persistence(zeros(2, 2); field=field)
    end
    # The typed ingestion convenience route obeys the same interval/field contract.
    distance = [0.0 2.0; 2.0 0.0]
    rips = OP.persistence_diagram(distance, TamerOp.DataIngestion.RipsFiltration(max_dim=1);
                                  field=CM.F2())
    @test OP.finite_intervals(rips; dim=0) == [(0.0, 2.0)]
    @test OP.essential_births(rips; dim=0) == [0.0]
    # The ordinary image/typed-filtration path must preserve the same exact
    # upper-star values, including a cache populated by another invocation.
    values = fill(b, 3, 3)
    values[2, 2] = a
    cache = CM.EncodingCache()
    image = DT.ImageNd(values)
    filtration = TamerOp.DataIngestion.CubicalFiltration()
    for _ in 1:2
        typed = OP.persistence_diagram(image, filtration; order=:superlevel, cache=cache)
        @test OP.finite_intervals(typed; dim=1) == [(b, a)]
        @test OP.essential_births(typed; dim=0) == [b]
    end
    # Topology and ordinal ranks agree, but grades belong to this invocation.
    # A topology-cache hit must not return the previous exact endpoints.
    changed_values = fill(b + 7, 3, 3)
    changed_values[2, 2] = a + 3
    changed = OP.persistence_diagram(DT.ImageNd(changed_values), filtration;
                                     order=:superlevel, cache=cache)
    @test OP.finite_intervals(changed; dim=1) == [(b + 7, a + 3)]
    @test OP.essential_births(changed; dim=0) == [b + 7]
    restored = OP.persistence_diagram(image, filtration; order=:superlevel, cache=cache)
    @test OP.finite_intervals(restored; dim=1) == [(b, a)]
    # Explicit construction contracts still apply before a topology-cache hit.
    for selected_cache in (nothing, cache), construction in (
        OPT.ConstructionOptions(budget=OPT.ConstructionBudget(max_edges=0)),
        OPT.ConstructionOptions(sparsify=:knn),
    )
        restricted = TamerOp.DataIngestion.CubicalFiltration(; construction)
        @test_throws ArgumentError OP.persistence_diagram(image, restricted;
            order=:superlevel, cache=selected_cache)
    end
    allowed = TamerOp.DataIngestion.CubicalFiltration(
        construction=OPT.ConstructionOptions(budget=OPT.ConstructionBudget(max_edges=12)))
    allowed_diagram = OP.persistence_diagram(image, allowed; order=:superlevel, cache=cache)
    @test OP.finite_intervals(allowed_diagram; dim=1) == [(b, a)]
    channel = TamerOp.DataIngestion.CubicalFiltration(channels=[values])
    typed = OP.persistence_diagram(DT.ImageNd(zeros(3, 3)), channel; order=:superlevel)
    @test OP.finite_intervals(typed; dim=1) == [(b, a)]
    @test_throws ArgumentError OP.persistence_diagram(image,
        TamerOp.DataIngestion.CubicalFiltration(channels=[values, values]))
    @test_throws ArgumentError OP.persistence_diagram(image,
        TamerOp.DataIngestion.CubicalFiltration(channels=[zeros(2, 2)]))
    # Superlevels compare in descending order; negation would overflow for
    # unsigned grades and typemin(Int), or coerce them to an inexact type.
    for (low, high) in ((UInt(0), UInt(2)), (typemin(Int), typemin(Int) + 2))
        sub = OP.cubical_persistence([low, high, low]; input=:vertices)
        super = OP.cubical_persistence([high, low, high]; input=:vertices, order=:superlevel)
        @test OP.finite_intervals(sub; dim=0) == [(low, high)]
        @test OP.finite_intervals(super; dim=0) == [(high, low)]
        @test OP.essential_births(sub; dim=0) == [low]
        @test OP.essential_births(super; dim=0) == [high]
    end
    # Signed zero has one mathematical grade, even if hash keys distinguish it.
    for input in (:top_cells, :vertices), order in (:sublevel, :superlevel)
        zero_diagram = OP.cubical_persistence([-0.0 0.0; 0.0 -0.0]; input, order)
        @test all(isempty(OP.finite_intervals(zero_diagram; dim=d)) for d in 0:2)
        @test [OP.essential_births(zero_diagram; dim=d) for d in 0:2] == [[0.0], Float64[], Float64[]]
    end
end

@testset "A21 BigFloat grades retain stored precision" begin
    low, high = setprecision(BigFloat, 512) do
        (BigFloat(1), BigFloat(1) + BigFloat(2)^(-300))
    end
    setprecision(BigFloat, 256) do
        @test precision(BigFloat) == 256
        @test low < high
        @test BigFloat(high) == low # The scalar constructor would lose this event.
        boundary = sparse(reshape([1, -1], 2, 1))
        # Both public grade-storage constructors must preserve existing values.
        for grades in ([(low,), (low,), (high,)], [[low], [low], [high]])
            complex = DT.GradedComplex([[1, 2], [1]], [boundary], grades)
            @test complex.grades == [(low,), (low,), (high,)]
            @test all(g -> precision(g[1]) == 512, complex.grades)
            diagram = OP.persistence_diagram(complex)
            @test OP.finite_intervals(diagram; dim=0) == [(low, high)]
            @test OP.essential_births(diagram; dim=0) == [low]
            @test all(bar -> all(x -> precision(x) == 512, bar),
                      OP.finite_intervals(diagram; dim=0))
        end
        for input in (:top_cells, :vertices), order in (:sublevel, :superlevel)
            birth, death = order === :sublevel ? (low, high) : (high, low)
            values = fill(birth, 3, 3)
            values[2, 2] = death
            diagram = OP.cubical_persistence(values; input, order)
            @test OP.finite_intervals(diagram; dim=1) == [(birth, death)]
            @test all(bar -> all(x -> precision(x) == 512, bar),
                      OP.finite_intervals(diagram; dim=1))
            @test OP.essential_births(diagram; dim=0) == [birth]
            @test all(x -> precision(x) == 512, OP.essential_births(diagram; dim=0))
            @test OP.persistence_intervals(diagram; dim=1) == [(birth, death)]
        end
    end
end

@testset "A21 ordinary malformed complexes and input contracts" begin
    empty = DT.GradedComplex([Int[], Int[], Int[]],
                             [spzeros(Int, 0, 0), spzeros(Int, 0, 0)], NTuple{1,Int}[])
    diagram = OP.persistence_diagram(empty)
    @test all(isempty(OP.persistence_intervals(diagram; dim=d)) for d in 0:2)
    @test OP.check_persistence_diagram(diagram; throw=false).valid
    @test !OP.check_torus_persistence(diagram; check_h2=true, throw=false).valid
    @test_throws ArgumentError OP.persistence_diagram(empty; order=:increasing)
    @test_throws ArgumentError OP.cubical_persistence(zeros(0, 2))
    @test_throws ArgumentError OP.cubical_persistence(zeros(2, 0); input=:vertices)
    @test_throws ArgumentError OP.cubical_persistence(zeros(2, 2); input=:pixels)
    @test_throws ArgumentError OP.cubical_persistence(zeros(2, 2, 2); input=:top_cells)
    for periodic in ((true,), (true, true, true), (1, 0), [true, 1], :periodic)
        @test_throws ArgumentError OP.cubical_persistence(zeros(2, 2); periodic=periodic)
        @test_throws ArgumentError OP.cubical_persistence(zeros(2, 2); periodic=periodic, input=:vertices)
    end
    for value in (NaN, Inf, -Inf)
        @test_throws ArgumentError OP.cubical_persistence(fill(value, 2, 2))
        G = DT.GradedComplex([[1]], SparseMatrixCSC{Int,Int}[], [(value,)])
        @test_throws ArgumentError OP.persistence_diagram(G)
    end
    multiparameter = DT.GradedComplex([[1]], SparseMatrixCSC{Int,Int}[], [(0, 0)])
    @test_throws ArgumentError OP.persistence_diagram(multiparameter)
    incompatible = DT.GradedComplex([[1, 2], [1]], [sparse(reshape([1, -1], 2, 1))],
                                    [(1,), (0,), (0,)])
    @test_throws ArgumentError OP.persistence_diagram(incompatible)
    nonchain = DT.GradedComplex([[1], [1], [1]],
        [sparse(reshape([1], 1, 1)), sparse(reshape([1], 1, 1))], [(0,), (1,), (2,)])
    @test_throws ArgumentError OP.persistence_diagram(nonchain)
    # Hand-built storage must fail at the contract boundary, before reduction.
    raw_bad = (
        DT.GradedComplex{1,Int}([1], [1, 0, 2], [spzeros(Int, 0, 1)], [(0,)]),
        DT.GradedComplex{1,Int}([1, 1], [1, 2, 3], SparseMatrixCSC{Int,Int}[], [(0,), (1,)]),
        DT.GradedComplex{1,Int}([1, 1], [1, 2, 3], [spzeros(Int, 2, 1)], [(0,), (1,)]),
        DT.GradedComplex{1,Int}([1], [1, 2], SparseMatrixCSC{Int,Int}[], NTuple{1,Int}[]),
    )
    for bad in raw_bad
        @test_throws ArgumentError OP.persistence_diagram(bad)
    end
    for corruption in (:row_range, :column_pointer, :row_order, :duplicate_row)
        boundary = sparse(reshape([1, -1], 2, 1))
        if corruption === :row_range
            boundary.rowval[1] = 3
        elseif corruption === :column_pointer
            boundary.colptr[2] = 100
        elseif corruption === :row_order
            reverse!(boundary.rowval)
        else
            boundary.rowval[2] = 1
        end
        bad = DT.GradedComplex([[1, 2], [1]], [boundary], [(0,), (0,), (1,)])
        @test_throws ArgumentError OP.persistence_diagram(bad)
    end
end

@testset "A21 ordinary result summaries validation and provenance" begin
    diagram = OP.PersistenceDiagram([[(0//1, 2//1)], Tuple{Rational{Int},Rational{Int}}[]],
                                    [[0//1], Rational{Int}[]]; field=CM.F2())
    @test OP.finite_intervals(diagram; dim=0) == [(0//1, 2//1)]
    @test OP.essential_births(diagram; dim=0) == [0//1]
    @test OP.check_persistence_diagram(diagram; throw=false).valid
    summary = OP.persistence_diagram_summary(diagram)
    @test summary == CC.describe(diagram)
    @test summary.finite_counts == (1, 0)
    @test summary.essential_counts == (1, 0)
    @test summary.field == CM.F2()
    @test occursin("PersistenceDiagram", sprint(show, diagram))
    @test occursin("PersistenceDiagram", sprint(show, MIME"text/plain"(), diagram))
    @test occursin("PersistenceValidationSummary", sprint(show,
        OP.persistence_validation_summary(OP.check_persistence_diagram(diagram; throw=false))))
    p = RES.provenance(diagram)
    @test p.field == CM.F2()
    @test p.degree_convention === :homological
    @test p.backend === :not_recorded
    @test p.approximation === :not_recorded
    @test p.discretization === :not_recorded
    @test RES.result_summary(diagram) == summary
    @test FF.field(diagram) == CM.F2()
    @test OP.filtration_order(diagram) === :sublevel
    detached = OP.finite_intervals(diagram; dim=0)
    push!(detached, (7//1, 8//1))
    @test OP.finite_intervals(diagram; dim=0) == [(0//1, 2//1)]
    @test_throws ArgumentError OP.persistence_intervals(diagram; dim=-1)
    @test_throws ArgumentError OP.persistence_intervals(diagram; dim=true)
    @test isempty(OP.persistence_intervals(diagram; dim=2))
    @test_throws ArgumentError OP.PersistenceDiagram([[(2, 1)]], [Int[]])
    @test_throws ArgumentError OP.PersistenceDiagram([[(1, 1)]], [Int[]])
    @test_throws ArgumentError OP.PersistenceDiagram([[(0.0, Inf)]], [Float64[]])
    @test_throws ArgumentError OP.PersistenceDiagram([Tuple{Int,Int}[]], [[0]]; order=:descending)
    @test_throws ArgumentError OP.PersistenceDiagram([Tuple{Int,Int}[]], [Int[], Int[]])
    @test_throws ArgumentError OP.PersistenceDiagram([Tuple{Float64,Float64}[]], [[NaN]])
    # A malformed user-edited result must produce a useful invalid report.
    push!(diagram.finite_by_dim[1], (3//1, 2//1))
    report = OP.check_persistence_diagram(diagram; throw=false)
    @test !report.valid
    @test !isempty(report.issues)
    @test_throws ArgumentError OP.check_persistence_diagram(diagram; throw=true)
end

@testset "A21 curated ordinary public boundary" begin
    for name in (:PersistenceDiagram, :persistence_diagram, :cubical_persistence,
                 :persistence_intervals, :finite_intervals, :essential_births,
                 :persistence_diagram_summary, :check_torus_persistence)
        @test getproperty(TamerOp, name) === getproperty(OP, name)
        @test getproperty(TOA, name) === getproperty(OP, name)
    end
    for name in (:PersistenceValidationSummary, :check_persistence_diagram,
                 :persistence_validation_summary, :filtration_order)
        @test getproperty(TOA, name) === getproperty(OP, name)
    end
    @test TamerOp.describe === CC.describe
    @test TOA.describe === CC.describe
    @test TamerOp.provenance === RES.provenance
    @test TOA.provenance === RES.provenance
    ring = zeros(Int, 3, 3)
    ring[2, 2] = 5
    diagram = TamerOp.cubical_persistence(ring)
    @test TamerOp.finite_intervals(diagram; dim=1) == [(0, 5)]
    @test TamerOp.essential_births(diagram; dim=0) == [0]
    @test TamerOp.describe(diagram) == OP.persistence_diagram_summary(diagram)
    @test TamerOp.provenance(diagram) == RES.provenance(diagram)
    @test TamerOp.provenance(diagram).backend === :f2_clearing
    @test TamerOp.provenance(diagram).approximation === :none_in_reduction
    @test TamerOp.provenance(diagram).discretization === :none
    @test TamerOp.result_summary(diagram) == OP.persistence_diagram_summary(diagram)
    @test TOA.check_persistence_diagram(diagram; throw=true).valid
    @test TOA.filtration_order(diagram) === :sublevel
end

@testset "A41 retained ordinary interval representatives" begin
    d1 = sparse([-1 -1 0; 1 0 -1; 0 1 1])
    d2 = sparse(reshape([1, -1, 1], 3, 1))
    for order in (:sublevel, :superlevel)
        grade = order === :sublevel ? identity : (x -> 3 - x)
        G = DT.GradedComplex([[11, 12, 13], [21, 22, 23], [31]], [d1, d2],
            [(grade(0),), (grade(0),), (grade(0),),
             (grade(1),), (grade(1),), (grade(1),), (grade(2),)])
        plain = OP.persistence_diagram(G; order)
        retained = OP.persistence_diagram(G; order, representatives=true)
        @test OP.check_persistence_diagram(retained).valid
        @test OP.persistence_diagram_summary(retained).representatives_available
        @test !OP.persistence_diagram_summary(plain).representatives_available
        @test OP.provenance(retained).representatives === :retained_reduction_cycles
        @test all(OP.persistence_intervals(plain; dim=d) == OP.persistence_intervals(retained; dim=d) for d in 0:2)
        for dim in 0:2, kind in (:finite, :essential)
            count = length(kind === :finite ? OP.finite_intervals(retained; dim) : OP.essential_births(retained; dim))
            for index in 1:count
                rep = OP.persistence_representative(retained; dim, kind, index)
                _a41_check_retained_cycle(G, rep)
                @test rep.birth_included && !rep.death_included
                @test rep.index == index
                absent = OP.persistence_representative(plain; dim, kind, index)
                @test !absent.available && absent.reason === :not_retained
                @test absent.cycle === absent.bounding_chain === nothing
            end
        end
        loop = OP.persistence_representative(retained; dim=1)
        @test loop.cycle.cell_indices == (1, 2, 3)
        @test loop.cycle.cell_ids == (21, 22, 23)
        @test loop.bounding_chain.cell_indices == (1,)
        @test loop.bounding_chain.cell_ids == (31,)
        # Equal H0 intervals must retain two distinguishable, independent
        # members; matching an endpoint pair alone cannot choose their cycles.
        a = OP.persistence_representative(retained; dim=0, index=1)
        b = OP.persistence_representative(retained; dim=0, index=2)
        @test a.interval == b.interval
        @test a.cycle != b.cycle
        Za = _a41_check_retained_cycle(G, a)
        Zb = _a41_check_retained_cycle(G, b)
        @test _a21_binary_rank(hcat(Za, Zb)) == 2

        # Without the filling face, the nontrivial essential H1 representative
        # requires accumulated change-of-basis columns, not the last edge alone.
        unfilled = DT.GradedComplex([[11, 12, 13], [21, 22, 23]], [d1], G.grades[1:6])
        ring = OP.persistence_diagram(unfilled; order, representatives=true)
        essential = OP.persistence_representative(ring; dim=1, kind=:essential)
        _a41_check_retained_cycle(unfilled, essential)
        @test essential.cycle.cell_indices == (1, 2, 3)
        @test essential.interval == (grade(1), order === :sublevel ? Inf : -Inf)
    end
    # Exact endpoint distinctions survive retention even when display floats
    # cannot separate them. Even incidence coefficients vanish modulo two.
    a = BigInt(1)//BigInt(1)
    b = a + BigInt(1)//(BigInt(2)^70)
    exact = DT.GradedComplex([[1], [2], [3]], [spzeros(Int, 1, 1), sparse(reshape([2], 1, 1))],
        [(a,), (b,), (b + 1,)])
    diagram = OP.persistence_diagram(exact; representatives=true)
    for dim in 0:2
        rep = OP.persistence_representative(diagram; dim, kind=:essential)
        _a41_check_retained_cycle(exact, rep)
        @test rep.cycle.cell_grades == (exact.grades[dim + 1][1],)
    end
end

@testset "A41 ordinary representative contracts and ingestion" begin
    G = DT.GradedComplex([[1, 2], [3]], [sparse(reshape([-1, 1], 2, 1))], [(0,), (0,), (1,)])
    diagram = OP.persistence_diagram(G; representatives=true)
    for dim in (-1, true), index in (0, 1)
        @test_throws ArgumentError OP.persistence_representative(diagram; dim, index)
    end
    for index in (0, -1, true, 2)
        @test_throws ArgumentError OP.persistence_representative(diagram; dim=0, index)
    end
    @test_throws ArgumentError OP.persistence_representative(diagram; dim=1)
    @test_throws ArgumentError OP.persistence_representative(diagram; dim=0, kind=:cycle)
    @test_throws ArgumentError OP.persistence_diagram(G; representatives=:yes)
    @test_throws ArgumentError OP.cubical_persistence(zeros(2, 2); representatives=1)
    handbuilt = OP.PersistenceDiagram([[(0, 1)]], [[0]];
        meta=(representatives=:retained_reduction_cycles, backend=:f2_column_reduction))
    @test !OP.persistence_representative(handbuilt; dim=0).available
    @test OP.provenance(handbuilt).representatives === :not_retained
    # Mutation must not silently select an unrelated retained cycle.
    diagram.finite_by_dim[1][1] = (0, 2)
    @test !OP.check_persistence_diagram(diagram).valid
    @test_throws ArgumentError OP.persistence_representative(diagram; dim=0)

    ring = zeros(Int, 3, 3)
    ring[2, 2] = 5
    for input in (:top_cells, :vertices), order in (:sublevel, :superlevel)
        values = order === :sublevel ? ring : 5 .- ring
        cube = OP.cubical_persistence(values; input, order, representatives=true)
        @test OP.check_persistence_diagram(cube).valid
        @test OP.persistence_diagram_summary(cube).representatives_available
        for dim in 0:2, kind in (:finite, :essential)
            members = kind === :finite ? OP.finite_intervals(cube; dim) : OP.essential_births(cube; dim)
            for index in eachindex(members)
                @test OP.persistence_representative(cube; dim, kind, index).available
            end
        end
    end
    image = DT.ImageNd(Float64.(ring))
    typed = OP.persistence_diagram(image, TamerOp.DataIngestion.CubicalFiltration(); representatives=true)
    @test OP.persistence_diagram_summary(typed).representatives_available
    @test OP.check_persistence_diagram(typed).valid
    rips = OP.persistence_diagram([0.0 2.0; 2.0 0.0],
        TamerOp.DataIngestion.RipsFiltration(max_dim=1); representatives=true)
    @test OP.persistence_representative(rips; dim=0).available
    @test OP.persistence_representative(rips; dim=0, kind=:essential).available
    @test OP.check_persistence_diagram(rips).valid
    @test TOA.persistence_representative === OP.persistence_representative
end

@testset "Ordinary persistence scratch columns preserve symmetric differences" begin
    rng = MersenneTwister(0xf20103)
    scratch = Int[]
    for n in (0, 1, 8, 65, 200), _ in 1:40
        rank = randperm(rng, n)
        a = sort!(findall(rand(rng, Bool, n)); by=i -> rank[i])
        b = sort!(findall(rand(rng, Bool, n)); by=i -> rank[i])
        saved_a, saved_b = copy(a), copy(b)
        expected = sort!(collect(symdiff(Set(a), Set(b))); by=i -> rank[i])
        @test OP._xor_columns!(scratch, a, b, rank) == expected
        @test a == saved_a && b == saved_b
        @test isempty(OP._xor_columns!(scratch, a, a, rank))
        @test OP._xor_columns!(scratch, Int[], b, rank) == b
    end
end

@testset "Ordinary clearing agrees with retained reductions" begin
    rng = MersenneTwister(0xc1ea)
    for shape in ((2, 3), (3, 4), (2, 2, 2)), order in (:sublevel, :superlevel), trial in 1:8
        values = reshape(rand(rng, -2:2, prod(shape)), shape)
        periodic = ntuple(_ -> rand(rng, Bool), length(shape))
        plain = OP.cubical_persistence(values; periodic, order, input=:vertices)
        retained = OP.cubical_persistence(values; periodic, order, input=:vertices, representatives=true)
        @test plain.finite_by_dim == retained.finite_by_dim
        @test plain.essential_by_dim == retained.essential_by_dim
        @test plain.meta.backend === :f2_clearing
        @test retained.meta.backend === :f2_column_reduction
    end
end

@testset "Ordinary active parity columns" begin
    rng = MersenneTwister(0xb17)
    for n in (0, 1, 63, 64, 65, 4096, 4097, 270000)
        column = OP._ParityColumn(n)
        expected = Set{Int}()
        for _ in 1:400
            n == 0 && break
            i = rand(rng, 1:n)
            OP._toggle_entry!(column, i)
            i in expected ? delete!(expected, i) : push!(expected, i)
            @test OP._column_pivot(column) == maximum(expected; init=0)
        end
        while !isempty(expected)
            i = maximum(expected)
            @test OP._column_pivot(column) == i
            OP._toggle_entry!(column, i)
            delete!(expected, i)
        end
        @test OP._column_pivot(column) == 0
        @test all(words -> all(iszero, words), column.levels)
    end
end

@testset "Ordinary exact sparse and packed chain validation" begin
    rng = MersenneTwister(0xd020)
    for rows in (0, 1, 63, 64, 65, 129), k in (1, 12, 48)
        C = rand(rng, 0:1, rows, k)
        rows > 0 && (C[1, 1] = 1)
        X = rand(rng, 0:1, k, 20)
        A = sparse(hcat(C, C)); B = sparse(vcat(X, X))
        @test OP._double_boundary_zero_sparse(A, B)
        @test OP._double_boundary_zero_packed(A, B)
        @test all(iseven, Matrix(A) * Matrix(B))
        if rows > 0
            B[1, 1] = 1 - B[1, 1]
            @test !OP._double_boundary_zero_sparse(A, B)
            @test !OP._double_boundary_zero_packed(A, B)
            @test any(isodd, Matrix(A) * Matrix(B))
        end
        # Signed odd entries and extreme even integers must use parity only.
        A.nzval .= [isodd(i) ? typemax(Int) : -3 for i in eachindex(A.nzval)]
        B.nzval .= [iszero(x) ? typemin(Int) : -5 for x in B.nzval]
        expected = all(iseven, Int.(isodd.(Matrix(A))) * Int.(isodd.(Matrix(B))))
        @test OP._double_boundary_zero_sparse(A, B) == expected
        @test OP._double_boundary_zero_packed(A, B) == expected
    end
    C = ones(Int, 129, 48); X = ones(Int, 48, 20)
    G = DT.GradedComplex([collect(1:129), collect(1:96), collect(1:20)],
        [sparse(hcat(C,C)), sparse(vcat(X,X))],
        vcat(fill((0,),129), fill((1,),96), fill((2,),20)))
    diagram = OP.persistence_diagram(G)
    retained = OP.persistence_diagram(G; representatives=true)
    @test diagram.finite_by_dim == retained.finite_by_dim
    @test diagram.essential_by_dim == retained.essential_by_dim
    G.boundaries[2][1,1] = 0
    @test_throws ArgumentError OP.persistence_diagram(G)
end

@testset "Ordinary graph barcodes preserve filtration and algebraic fallback" begin
    rng = MersenneTwister(0x9aaf)
    for nv in (1, 2, 9, 32), order in (:sublevel, :superlevel), trial in 1:8
        ne = 3nv
        grades = rand(rng, -3:3, nv)
        B = zeros(Int, nv, ne)
        edge_grades = Int[]
        for j in 1:ne
            a, b = rand(rng, 1:nv, 2)
            B[a,j] += 1; B[b,j] -= 1
            push!(edge_grades, max(grades[a],grades[b]) + rand(rng,0:2))
        end
        gs = vcat(grades,edge_grades)
        order === :superlevel && (gs = -gs)
        G = DT.GradedComplex([collect(1:nv), collect(1:ne)], [sparse(B)], [(x,) for x in gs])
        plain = OP.persistence_diagram(G; order)
        retained = OP.persistence_diagram(G; order, representatives=true)
        @test plain.meta.backend === :f2_graph_union_find
        @test plain.finite_by_dim == retained.finite_by_dim
        @test plain.essential_by_dim == retained.essential_by_dim
        # Independent H0 ranks at every filtration event.
        for level in unique(gs)
            vertices = findall(x -> order === :sublevel ? x <= level : x >= level, gs[1:nv])
            edges = findall(x -> order === :sublevel ? x <= level : x >= level, gs[nv+1:end])
            r = _a21_binary_rank(B[vertices, edges])
            @test _a21_barcode_map_rank(plain,0,level,level,order) == length(vertices)-r
            @test _a21_barcode_map_rank(plain,1,level,level,order) == length(edges)-r
        end
    end
    one_endpoint = DT.GradedComplex([[1],[1]], [sparse(reshape([3],1,1))], [(0,), (2,)])
    diagram = OP.persistence_diagram(one_endpoint)
    @test diagram.meta.backend === :f2_clearing
    @test OP.finite_intervals(diagram; dim=0) == [(0,2)]
    @test isempty(OP.essential_births(diagram; dim=0))
    signed = DT.GradedComplex([[1,2],[1,2]], [sparse([3 typemin(Int); -5 0])], [(0,),(0,),(1,),(2,)])
    diagram = OP.persistence_diagram(signed)
    @test diagram.meta.backend === :f2_graph_union_find
    @test OP.finite_intervals(diagram; dim=0) == [(0,1)]
    @test OP.essential_births(diagram; dim=1) == [2]
end

@testset "Ordinary stored binary columns preserve exact supports" begin
    rng = MersenneTwister(0x50484154)
    for n in (1, 63, 64, 65, 4096, 4097, 270000)
        for mode in (:sparse, :dense), repetition in 1:8
            selected = if mode === :sparse
                sort!(unique(rand(rng, 1:n, min(n, 8))); rev=true)
            else
                first = rand(rng, 1:n)
                sort!(unique(vcat(collect(first:min(n, first + 160)),
                                 collect(max(1, n - 140):n))); rev=true)
            end
            stored = OP._BarcodeColumns(1)
            OP._store_barcode_column!(stored, 1, selected)
            active = OP._ParityColumn(n)
            expected = Set(rand(rng, 1:n, min(n, 20)))
            for row in expected
                OP._toggle_entry!(active, row)
            end
            previous = max(maximum(expected; init=0), maximum(selected; init=0))
            OP._add_barcode_column!(active, stored, 1)
            symdiff!(expected, selected)
            @test OP._column_pivot(active) == maximum(expected; init=0)
            @test OP._column_pivot(active, previous) == maximum(expected; init=0)
            # Drain every coefficient, not just the largest one.
            actual = Int[]
            while (row = OP._column_pivot(active)) != 0
                push!(actual, row)
                OP._toggle_entry!(active, row)
            end
            @test actual == sort!(collect(expected); rev=true)
            @test all(words -> all(iszero, words), active.levels)
            # Repeated addition cancels, including all occupancy levels.
            OP._add_barcode_column!(active, stored, 1)
            OP._add_barcode_column!(active, stored, 1)
            @test all(words -> all(iszero, words), active.levels)
        end
    end
    stored = OP._BarcodeColumns(2)
    OP._store_barcode_column!(stored, 1, [193, 129, 65, 1])
    OP._store_barcode_column!(stored, 2, collect(192:-1:1))
    @test stored.spans[1] == (1, 4)
    @test stored.spans[2] == (1, -3)
    @test sum(count_ones(bits) for (_, bits) in stored.words) == 192
    # Growth of either payload must preserve every previously saved span.
    active = OP._ParityColumn(193)
    for slot in (1, 2, 1, 2)
        OP._add_barcode_column!(active, stored, slot)
    end
    @test OP._column_pivot(active) == 0
end

@testset "Ordinary integer filtration sorting retains ties and extreme grades" begin
    for T in (Int, UInt, Int128, BigInt)
        low = T === BigInt ? -big(2)^200 : typemin(T)
        high = T === BigInt ? big(2)^200 : typemax(T)
        # Both vertices are born together; the same second vertex dies at
        # the edge. Its cycle/filling convention must match in either order.
        for order in (:sublevel, :superlevel)
            birth, death = order === :sublevel ? (low, high) : (high, low)
            g = DT.GradedComplex([[1, 2], [3]],
                [sparse([1, 2], [1, 1], [1, 1], 2, 1)], [(birth,), (birth,), (death,)])
            diagram = OP.persistence_diagram(g; order, representatives=true)
            @test OP.finite_intervals(diagram; dim=0) == [(birth, death)]
            @test OP.essential_births(diagram; dim=0) == [birth]
            rep = OP.persistence_representative(diagram; dim=0)
            @test rep.cycle.cell_indices == (1, 2)
            @test rep.bounding_chain.cell_indices == (1,)
            essential = OP.persistence_representative(diagram; dim=0, kind=:essential)
            @test essential.cycle.cell_indices == (1,)
            plain = OP.persistence_diagram(g; order)
            @test plain.finite_by_dim == diagram.finite_by_dim
            @test plain.essential_by_dim == diagram.essential_by_dim
        end
    end
end

@testset "Ordinary finished columns drain and reuse every hierarchy level" begin
    rng = MersenneTwister(0x64726169)
    for n in (0, 1, 63, 64, 65, 4096, 4097, 270000)
        active = OP._ParityColumn(n)
        out = [999]
        for trial in 1:12
            support = n == 0 ? Int[] : unique(rand(rng, 1:n, min(n, trial * 91)))
            for row in support
                OP._toggle_entry!(active, row)
            end
            @test OP._drain_column!(out, active) === out
            @test out == sort(support; rev=true)
            @test all(words -> all(iszero, words), active.levels)
            @test OP._column_pivot(active) == 0
            OP._drain_column!(out, active)
            @test isempty(out)
        end
    end
end

@testset "Ordinary pooled columns survive interleaved payload growth" begin
    rng = MersenneTwister(0x706f6f6c)
    n = 8193
    stored = OP._BarcodeColumns(128)
    supports = Vector{Int}[]
    for slot in 1:128
        support = if isodd(slot)
            first = rand(rng, 1:n-255)
            collect(first:first+255)
        else
            unique(rand(rng, 1:n, 9))
        end
        push!(supports, sort!(support; rev=true))
        OP._store_barcode_column!(stored, slot, supports[end])
    end
    active = OP._ParityColumn(n)
    expected = Set{Int}()
    for slot in shuffle(rng, 1:128)
        OP._add_barcode_column!(active, stored, slot)
        symdiff!(expected, supports[slot])
        @test OP._column_pivot(active) == maximum(expected; init=0)
    end
    actual = Int[]
    OP._drain_column!(actual, active)
    @test actual == sort!(collect(expected); rev=true)
    @test all(words -> all(iszero, words), active.levels)
    for _ in 1:2, slot in 1:128
        OP._add_barcode_column!(active, stored, slot)
    end
    @test all(words -> all(iszero, words), active.levels)
end

# Appended to the owner suite after selecting a production candidate. The rank
# oracle above is independent binary Gaussian elimination on persistent maps.
@testset "Ordinary mixed-dimensional component reduction" begin
    rng = MersenneTwister(0x683066)
    for nv in (1, 3, 7), order in (:sublevel, :superlevel), trial in 1:6
        ne = 2nv + 2
        B1 = zeros(Int, nv, ne)
        vg = rand(rng, 0:2, nv)
        eg = Int[]
        for j in 1:ne
            a, b = rand(rng, 1:nv, 2)
            B1[a,j] += 3; B1[b,j] -= 5
            push!(eg, max(vg[a], vg[b]) + rand(rng, 0:2))
        end
        Z = _a21_binary_cycles(B1)
        B2 = mod.(Int.(Z) * rand(rng, 0:1, size(Z,2), 4), 2)
        fg = [maximum(eg[findall(isodd, B2[:,j])];init=0) + rand(rng,0:2) for j in 1:4]
        gs = [vg, eg, fg]
        order === :superlevel && (gs = [-v for v in gs])
        G = DT.GradedComplex([collect(1:nv),collect(1:ne),collect(1:4)],
            [sparse(B1),sparse(B2)], [(x,) for x in vcat(gs...)])
        diag = OP.persistence_diagram(G;order)
        retained = OP.persistence_diagram(G;order,representatives=true)
        @test diag.finite_by_dim == retained.finite_by_dim
        @test diag.essential_by_dim == retained.essential_by_dim
        reference = (;boundaries=[B1,B2],grades=gs)
        levels = sort!(unique(vcat(gs...));rev=order===:superlevel)
        for h in 0:2, (i,s) in enumerate(levels), t in levels[i:end]
            @test _a21_barcode_map_rank(diag,h,s,t,order) ==
                  _a21_homology_map_rank(reference,h,s,t,order)
        end
    end
    # A one-ended algebraic edge kills H0 outright; it is not a graph edge.
    # The independent 2-cell must not enable the graph shortcut by dimension.
    G = DT.GradedComplex([[1],[1],[1]], [sparse(reshape([3],1,1)),spzeros(Int,1,1)],
        [(0,), (2,), (3,)])
    diag = OP.persistence_diagram(G)
    @test OP.finite_intervals(diag;dim=0) == [(0,2)]
    @test isempty(OP.essential_births(diag;dim=0))
    @test OP.essential_births(diag;dim=2) == [3]
end


@testset "Ordinary dual top boundary preserves persistent maps" begin
    rng = MersenneTwister(0x6475616c)
    for nr in (0,1,3,7), nc in (0,1,2,5), order in (:sublevel,:superlevel), trial in 1:3
        B = zeros(Int,nr,nc)
        eg = rand(rng,0:2,nr)
        for i in 1:nr
            endpoints = randperm(rng,nc)[1:min(nc,rand(rng,0:2))]
            for j in endpoints;B[i,j]=rand(rng,(-5,3));end
        end
        fg=[maximum(eg[findall(isodd,B[:,j])];init=0)+rand(rng,0:2) for j in 1:nc]
        gs=[Int[0],eg,fg]
        order===:superlevel && (gs=[-v for v in gs])
        G=DT.GradedComplex([[1],collect(1:nr),collect(1:nc)],
            [spzeros(Int,1,nr),sparse(B)],[(x,) for x in vcat(gs...)])
        diag=OP.persistence_diagram(G;order)
        retained=OP.persistence_diagram(G;order,representatives=true)
        @test diag.finite_by_dim==retained.finite_by_dim
        @test diag.essential_by_dim==retained.essential_by_dim
        reference=(;boundaries=[zeros(Int,1,nr),B],grades=gs)
        levels=sort!(unique(vcat(gs...));rev=order===:superlevel)
        for h in 0:2,(i,s) in enumerate(levels),t in levels[i:end]
            @test _a21_barcode_map_rank(diag,h,s,t,order)==
                  _a21_homology_map_rank(reference,h,s,t,order)
        end
    end
    # Two equal top boundaries create an essential top class when the later
    # cell arrives. A missing exterior component must not delete that class.
    for T in (Int,BigInt,Rational{BigInt}), order in (:sublevel,:superlevel)
        levels=T===Int ? [0,1,2,3] : T.([big(2)^70+i for i in 0:3])
        order===:superlevel && reverse!(levels)
        G=DT.GradedComplex([[1],[1],[1,2]], [spzeros(Int,1,1),sparse(reshape([3,-5],1,2))],
            [(x,) for x in levels])
        diag=OP.persistence_diagram(G;order)
        @test OP.finite_intervals(diag;dim=1)==[(levels[2],levels[3])]
        @test OP.essential_births(diag;dim=2)==[levels[4]]
        @test OP.essential_births(diag;dim=0)==[levels[1]]
    end
    # Three cofacets have no graph interpretation: retain general reduction.
    G=DT.GradedComplex([[1],[1],[1,2,3]],
        [spzeros(Int,1,1),sparse(reshape([1,3,-5],1,3))],[(0,),(1,),(2,),(3,),(4,)])
    diag=OP.persistence_diagram(G)
    @test OP.finite_intervals(diag;dim=1)==[(1,2)]
    @test OP.essential_births(diag;dim=2)==[3,4]
    offsets=G.dim_offsets;values=first.(G.grades);perm=sortperm(values);rank=invperm(perm)
    finite=[Tuple{Int,Int}[] for _ in 1:3];essential=[Int[] for _ in 1:3];cleared=falses(5)
    @test !OP._dual_top_barcode!(finite,essential,G,offsets,values,perm,rank,cleared,2)
    @test all(isempty,finite) && all(isempty,essential) && !any(cleared)
end

# A112 reference: dense BigInt modular row elimination, independent of both the
# persistence column algorithm and TamerOp's coefficient/linear-algebra kernels.
function _a112_rref(A, p)
    R = mod.(BigInt.(A), p)
    pivots = Int[]
    row = 1
    for col in axes(R, 2)
        row > size(R, 1) && break
        found = findfirst(i -> !iszero(R[i, col]), row:size(R, 1))
        found === nothing && continue
        k = row + found - 1
        R[row, :], R[k, :] = copy(R[k, :]), copy(R[row, :])
        R[row, :] = mod.(R[row, :] .* invmod(R[row, col], big(p)), p)
        for i in axes(R, 1)
            i == row && continue
            R[i, :] = mod.(R[i, :] .- R[i, col] .* R[row, :], p)
        end
        push!(pivots, col)
        row += 1
    end
    return R, pivots
end
_a112_rank(A, p) = length(last(_a112_rref(A, p)))
function _a112_cycles(A, p)
    R, pivots = _a112_rref(A, p)
    free = setdiff(collect(axes(A, 2)), pivots)
    Z = zeros(BigInt, size(A, 2), length(free))
    for (j, f) in enumerate(free)
        Z[f, j] = 1
        for (i, pivot) in enumerate(pivots)
            Z[pivot, j] = mod(-R[i, f], p)
        end
    end
    return Z
end
function _a112_map_rank(reference, dim, source, target, order, p)
    active(level) = [findall(g -> order === :sublevel ? g <= level : g >= level, gs)
                     for gs in reference.grades]
    src, dst = active(source), active(target)
    degree = dim + 1
    D = dim == 0 ? zeros(Int, 0, length(src[degree])) :
        reference.boundaries[dim][src[degree - 1], src[degree]]
    Z = _a112_cycles(D, p)
    embedded = zeros(BigInt, length(dst[degree]), size(Z, 2))
    positions = Dict(cell => i for (i, cell) in enumerate(dst[degree]))
    for (i, cell) in enumerate(src[degree])
        embedded[positions[cell], :] = Z[i, :]
    end
    B = degree == length(src) ? zeros(Int, length(dst[degree]), 0) :
        reference.boundaries[degree][dst[degree], dst[degree + 1]]
    return _a112_rank(hcat(embedded, B), p) - _a112_rank(B, p)
end
function _a112_check_maps(diagram, reference, p)
    order = OP.filtration_order(diagram)
    events = sort!(unique(vcat(reference.grades...)))
    isempty(events) && return
    # Rational interpolation avoids rounding when endpoints exceed 2^53.
    levels = sort!(unique(vcat(events, [first(events)-1, last(events)+1],
        [(big(events[i]) + big(events[i+1]))//2 for i in 1:length(events)-1]));
        rev=order === :superlevel)
    for dim in 0:length(reference.grades)-1, i in eachindex(levels), j in i:length(levels)
        @test _a21_barcode_map_rank(diagram, dim, levels[i], levels[j], order) ==
              _a112_map_rank(reference, dim, levels[i], levels[j], order, p)
    end
    @test OP.check_persistence_diagram(diagram; throw=true).valid
end
function _a112_reference(G)
    offsets = G.dim_offsets
    return (; boundaries=Matrix.(G.boundaries),
        grades=[first.(G.grades[offsets[d]:offsets[d+1]-1]) for d in 1:length(offsets)-1])
end
function _a112_check_representatives(G, diagram, p)
    reference = _a112_reference(G)
    counts = length.(reference.grades)
    order = diagram.order
    cycles_by_dim = [Tuple{Any,Vector{BigInt}}[] for _ in counts]
    for dim in 0:length(counts)-1, kind in (:finite, :essential)
        bars = kind === :finite ? OP.finite_intervals(diagram; dim) : OP.essential_births(diagram; dim)
        for index in eachindex(bars)
            rep = OP.persistence_representative(diagram; dim, kind, index)
            @test rep.available && rep.field == CM.Fp(p)
            @test rep.choice === (p == 2 ? :noncanonical_f2_column_reduction : :noncanonical_prime_column_reduction)
            z = zeros(BigInt, counts[dim+1])
            z[collect(rep.cycle.cell_indices)] = collect(rep.cycle.coefficients)
            push!(cycles_by_dim[dim+1], (rep.interval,z))
            @test all(c -> 0 < c < p, rep.cycle.coefficients)
            @test rep.cycle.cell_ids == Tuple(G.cells_by_dim[dim+1][collect(rep.cycle.cell_indices)])
            @test all(g -> order === :sublevel ? g <= rep.interval[1] : g >= rep.interval[1], rep.cycle.cell_grades)
            dim == 0 || @test all(iszero, mod.(G.boundaries[dim] * z, p))
            for t in unique(vcat(reference.grades...))
                (order === :sublevel ? t >= rep.interval[1] : t <= rep.interval[1]) || continue
                next = dim+2 <= length(counts) ?
                    findall(g -> order === :sublevel ? g <= t : g >= t, reference.grades[dim+2]) : Int[]
                B = dim+2 <= length(counts) ? reference.boundaries[dim+1][:, next] : zeros(Int, length(z), 0)
                alive = order === :sublevel ? t < rep.interval[2] : t > rep.interval[2]
                @test _a112_rank(hcat(B, z), p) - _a112_rank(B, p) == Int(alive)
            end
            if kind === :finite
                c = zeros(BigInt, counts[dim+2])
                filling = rep.bounding_chain
                c[collect(filling.cell_indices)] = collect(filling.coefficients)
                @test all(c -> 0 < c < p, filling.coefficients)
                @test mod.(G.boundaries[dim+1] * c, p) == z
                @test all(g -> order === :sublevel ? g <= rep.interval[2] : g >= rep.interval[2], filling.cell_grades)
            else
                @test rep.bounding_chain === nothing
            end
        end
    end
    # All simultaneously living representatives must form a complete homology
    # basis, not merely be individually nonzero classes.
    for dim in 0:length(counts)-1, t in unique(vcat(reference.grades...))
        vectors = [z for ((b,d),z) in cycles_by_dim[dim+1]
                   if order === :sublevel ? b <= t < d : d < t <= b]
        Z = isempty(vectors) ? zeros(BigInt,counts[dim+1],0) : hcat(vectors...)
        next = dim+2 <= length(counts) ?
            findall(g -> order === :sublevel ? g <= t : g >= t, reference.grades[dim+2]) : Int[]
        B = dim+2 <= length(counts) ? reference.boundaries[dim+1][:,next] : zeros(Int,counts[dim+1],0)
        @test _a112_rank(hcat(B,Z),p)-_a112_rank(B,p) == length(vectors) ==
              _a112_map_rank(reference,dim,t,t,order,p)
    end
end

@testset "A112 characteristic-dependent barcodes and retained coefficients" begin
    # Cellular Moore spaces: d_2 = q, d_1 = 0. For p|q, H1 and H2 persist;
    # otherwise d_2 kills H1 at grade 2 and creates no H2 class.
    for p in (2, 3, 5, 101, 2^61-1), q in (2, 3, 101, typemax(Int)), order in (:sublevel, :superlevel)
        F = CM.Fp(p)
        grades = order === :sublevel ? [(0//1,), (1//1,), (2//1,)] : [(2//1,), (1//1,), (0//1,)]
        G = DT.GradedComplex([[17], [17], [17]], [spzeros(Int,1,1), sparse(reshape([q],1,1))], grades)
        plain = TamerOp.persistence_diagram(G; field=F, order)
        D = TamerOp.persistence_diagram(G; field=F, order, representatives=true)
        b, d = grades[2][1], grades[3][1]
        @test OP.finite_intervals(D; dim=1) == (q % p == 0 ? Tuple{typeof(b),typeof(b)}[] : [(b,d)])
        @test OP.essential_births(D; dim=1) == (q % p == 0 ? [b] : typeof(b)[])
        @test OP.essential_births(D; dim=2) == (q % p == 0 ? [d] : typeof(d)[])
        @test all(OP.persistence_intervals(D; dim=k) == OP.persistence_intervals(plain; dim=k) for k in 0:2)
        @test OP.field(D) == F == TamerOp.provenance(D).field
        @test OP.provenance(D).backend == (p == 2 ? :f2_column_reduction : :prime_column_reduction)
        _a112_check_maps(D, _a112_reference(G), p)
        _a112_check_representatives(G, D, p)
    end
end

# Independent oriented simplex boundaries, then filtration-preserving integral
# basis changes. Unlike changing field labels, this exercises cancellation,
# nonunit pivots, tied events, and representations with the same exact maps.
function _a112_simplicial_fixture(rng, order)
    simplices = [[Int[v] for v in 1:5],
        [[i,j] for i in 1:5 for j in i+1:5],
        [[i,j,k] for i in 1:5 for j in i+1:5 for k in j+1:5],
        [[i,j,k,l] for i in 1:5 for j in i+1:5 for k in j+1:5 for l in k+1:5]]
    counts = length.(simplices)
    boundaries = Matrix{Int}[]
    for dim in 1:3
        B = zeros(Int, counts[dim], counts[dim+1])
        for (j, s) in enumerate(simplices[dim+1]), k in eachindex(s)
            face = s[setdiff(eachindex(s), [k])]
            i = findfirst(==(face), simplices[dim])
            B[i,j] = isodd(k) ? 1 : -1
        end
        push!(boundaries, B)
    end
    grades = [sort!(rand(rng, (2dim):(2dim+1), count)) for (dim,count) in enumerate(counts)]
    bases, inverses = Matrix{Int}[], Matrix{Int}[]
    for n in counts
        S, Si = Matrix{Int}(I,n,n), Matrix{Int}(I,n,n)
        for _ in 1:8
            i = rand(rng, 1:n-1); j = rand(rng, i+1:n); a = rand(rng, (-3,-2,2,3))
            S[:,j] += a * S[:,i]
            Si[i,:] -= a * Si[j,:]
        end
        @test Si*S == Matrix{Int}(I,n,n)
        push!(bases,S); push!(inverses,Si)
    end
    transformed = [inverses[d]*boundaries[d]*bases[d+1] for d in 1:3]
    @test all(iszero, transformed[1]*transformed[2])
    @test all(iszero, transformed[2]*transformed[3])
    order === :superlevel && (grades = [-g for g in grades])
    G = DT.GradedComplex([fill(7,n) for n in counts], sparse.(transformed), [(g,) for gs in grades for g in gs])
    return G, (; boundaries, grades)
end

@testset "A112 independent persistent maps and basis changes" begin
    rng = MersenneTwister(112)
    for seed in 1:3, order in (:sublevel, :superlevel)
        G, reference = _a112_simplicial_fixture(rng, order)
        for p in (2,3,101)
            D = OP.persistence_diagram(G; field=CM.Fp(p), order, representatives=true)
            plain = OP.persistence_diagram(G; field=CM.Fp(p), order)
            @test all(OP.persistence_intervals(D; dim=d) == OP.persistence_intervals(plain; dim=d) for d in 0:3)
            _a112_check_maps(D, reference, p)
            _a112_check_representatives(G,D,p)
        end
    end
end

@testset "A112 oriented periodic cubical persistent maps" begin
    for p in (2,3,101), order in (:sublevel, :superlevel), input in (:top_cells, :vertices),
        shape in ((1,1), (1,3), (3,3)), periodic in ((false,false), (true,false), (true,true))
        values = reshape([mod(3i+div(i,3),4) for i in 1:prod(shape)],shape)
        reference = _a21_cubical_oracle(values,periodic,input,order; signed=true)
        @test all(iszero, reference.boundaries[1]*reference.boundaries[2])
        D = TamerOp.cubical_persistence(values; field=CM.Fp(p),order,input,periodic,representatives=true)
        _a112_check_maps(D,reference,p)
        G = input === :top_cells ? OP._top_cell_complex_2d(values,periodic,order) :
            OP._vertex_cubical_complex(values,periodic,order,DI.ConstructionOptions(),nothing)
        _a112_check_representatives(G,D,p)
    end
    for p in (2,3,101), order in (:sublevel,:superlevel), periodic in ((false,false,true),(true,true,true))
        values=reshape([0,1,2,1,2,0,1,2],2,2,2)
        D=OP.cubical_persistence(values;field=CM.Fp(p),input=:vertices,order,periodic)
        reference=_a21_cubical_oracle(values,periodic,:vertices,order;signed=true)
        @test all(iszero,reference.boundaries[1]*reference.boundaries[2])
        @test all(iszero,reference.boundaries[2]*reference.boundaries[3])
        _a112_check_maps(D,reference,p)
        torus=OP.cubical_persistence(fill(2//3,1,1);periodic=true,field=CM.Fp(p),order)
        @test OP.check_torus_persistence(torus;throw=true).valid
    end
end

@testset "A112 field-aware input and public result contracts" begin
    for p in (3,101,2^61-1)
        F=CM.Fp(p)
        # Cancellation must not overflow at machine-sized coefficients/moduli.
        m=typemax(Int)
        G=DT.GradedComplex([[1],[1,2],[1]],
            [sparse(reshape([m,m],1,2)),sparse(reshape([m,-m],2,1))],[(0,),(1,),(1,),(2,)])
        D=OP.persistence_diagram(G;field=F,representatives=true)
        _a112_check_maps(D,_a112_reference(G),p)
        _a112_check_representatives(G,D,p)
        # Entries zero in the selected field impose no filtration constraint.
        zero_boundary=DT.GradedComplex([[1],[1]],[sparse(reshape([p],1,1))],[(5,),(0,)])
        @test OP.essential_births(OP.persistence_diagram(zero_boundary;field=F);dim=1)==[0]
        @test_throws ArgumentError OP.persistence_diagram(zero_boundary;field=CM.F2())
        empty=DT.GradedComplex([Int[],Int[]],[spzeros(Int,0,0)],NTuple{1,Int}[])
        @test OP.check_persistence_diagram(OP.persistence_diagram(empty;field=F,representatives=true)).valid
        stored=OP.PersistenceDiagram([[(1//3,2//3)]],[Rational{Int}[]];field=F)
        @test OP.field(stored)==F && OP.check_persistence_diagram(stored).valid
        @test !OP.persistence_representative(stored;dim=0).available
        typed=OP.persistence_diagram(DT.ImageNd([0 1;2 0]), DI.CubicalFiltration();field=F)
        direct=OP.cubical_persistence([0 1;2 0];input=:vertices,field=F)
        @test all(OP.persistence_intervals(typed;dim=d)==OP.persistence_intervals(direct;dim=d) for d in 0:2)
        rips=OP.persistence_diagram([0 1 1;1 0 1;1 1 0],DI.RipsFiltration(max_dim=2);field=F,representatives=true)
        @test OP.finite_intervals(rips;dim=0)==[(0.0,1.0),(0.0,1.0)]
        @test isempty(OP.persistence_intervals(rips;dim=1))
        @test OP.essential_births(rips;dim=0)==[0.0]
    end
    mod3=DT.GradedComplex([[1],[1],[1]], [sparse(reshape([3],1,1)),sparse(reshape([1],1,1))],[(0,),(1,),(2,)])
    @test OP.check_persistence_diagram(OP.persistence_diagram(mod3;field=CM.F3())).valid
    @test_throws ArgumentError OP.persistence_diagram(mod3;field=CM.F2())
    sign_error=DT.GradedComplex([[1,2,3],[1,2,3],[1]],
        [sparse([1 0 1;1 1 0;0 1 1]),sparse(ones(Int,3,1))],[(0,),(0,),(0,),(1,),(1,),(1,),(2,)])
    @test OP.check_persistence_diagram(OP.persistence_diagram(sign_error;field=CM.F2())).valid
    @test_throws ArgumentError OP.persistence_diagram(sign_error;field=CM.F3())
    for corruption in (:row_range,:column_pointer,:row_order,:duplicate_row)
        B=sparse(reshape([1,-1],2,1))
        corruption === :row_range && (B.rowval[1]=3)
        corruption === :column_pointer && (B.colptr[2]=100)
        corruption === :row_order && reverse!(B.rowval)
        corruption === :duplicate_row && (B.rowval[2]=1)
        G=DT.GradedComplex([[1,2],[1]],[B],[(0,),(0,),(1,)])
        @test_throws ArgumentError OP.persistence_diagram(G;field=CM.F3())
    end
    # Malformed retained coefficients must be reported, not silently coerced.
    G=DT.GradedComplex([[1],[1]],[sparse(reshape([2],1,1))],[(0,),(1,)])
    D=OP.persistence_diagram(G;field=CM.F3(),representatives=true)
    D.retained_representatives.finite[1][1].cycle.coefficients[1]=3
    @test !OP.check_persistence_diagram(D).valid
end

@testset "A112 exact grades and representative basis contract" begin
    pairs = ((typemin(Int),typemin(Int)+1), (typemax(Int)-1,typemax(Int)),
             (big(2)^90,big(2)^90+1), ((big(2)^90)//7,(big(2)^90+1)//7),
             (1.0,nextfloat(1.0)))
    for p in (3,101), (a,b) in pairs, order in (:sublevel,:superlevel)
        birth,death = order === :sublevel ? (a,b) : (b,a)
        B=sparse(reshape([2,-2],2,1))
        G=DT.GradedComplex([[1,1],[1]],[B],[(birth,),(birth,),(death,)])
        before=copy(B)
        for keep in (false,true)
            D=OP.persistence_diagram(G;field=CM.Fp(p),order,representatives=keep)
            @test OP.finite_intervals(D;dim=0)==[(birth,death)]
            @test OP.essential_births(D;dim=0)==[birth]
            @test typeof(only(OP.finite_intervals(D;dim=0))[1])==typeof(a)
            @test OP.check_persistence_diagram(D;throw=true).valid
            if keep
                rep=OP.persistence_representative(D;dim=0)
                @test rep.cycle.cell_grades==(birth,birth)
                @test rep.bounding_chain.cell_grades==(death,)
            end
        end
        @test B==before
    end
    a,b=setprecision(256) do
        a=BigFloat(1); (a,nextfloat(a))
    end
    setprecision(64) do
        G=DT.GradedComplex([[1],[1]],[sparse(reshape([2],1,1))],[(a,),(b,)])
        D=OP.persistence_diagram(G;field=CM.F3(),representatives=true)
        @test only(OP.finite_intervals(D;dim=0))==(a,b)
        @test all(x->precision(x)==256, only(OP.finite_intervals(D;dim=0)))
        @test precision(only(OP.persistence_representative(D;dim=0).cycle.cell_grades))==256
    end
end

@testset "A112 prime-field interval inspection" begin
    V = TamerOp.Visualization
    for p in (2,3,101)
        values=zeros(Int,3,3); values[2,2]=5
        D=TamerOp.cubical_persistence(values;field=CM.Fp(p),representatives=true)
        @test OP.finite_intervals(D;dim=1)==[(0,5)]
        for kind in (:barcode,:persistence_diagram)
            spec=V.visual_spec(D;kind,dim=1)
            @test V.check_visual_spec(spec).valid
            @test spec.metadata.finite_intervals==[(0,5)]
        end
        session=TamerOp.inspection_session(D;dim=1)
        try
            TOA.select_inspection!(session;interval=1,representative=true)
            snapshot=TOA.inspection_snapshot(session)
            rep=snapshot.metadata.selected_representative
            @test rep.field==CM.Fp(p) && rep.available
            @test all(c -> 0<c<p,rep.cycle.coefficients)
            lines=only(snapshot.panels[3].layers).labels
            @test any(contains("F$p reduction representative"),lines)
        finally
            TOA.close_inspection!(session)
        end
    end
end

# Independent Rips reference: enumerate vertex subsets and signed facets,
# without using the ingestion clique builder or its boundary constructors.
function _rips_contract_reference(distances::Matrix{Int}, max_dim; radius=typemax(Int))
    n=size(distances,1)
    simplices=[Vector{Int}[] for _ in 0:max_dim]
    grades=[Int[] for _ in 0:max_dim]
    for mask in 1:(1<<n)-1
        vertices=[v for v in 1:n if !iszero(mask & (1<<(v-1)))]
        length(vertices)<=max_dim+1 || continue
        pairs=[distances[vertices[i],vertices[j]] for i in 1:length(vertices) for j in i+1:length(vertices)]
        any(<(0),pairs) && continue # -1 denotes an absent edge, not zero.
        grade=maximum(pairs;init=0)
        grade<=radius || continue
        push!(simplices[length(vertices)],vertices)
        push!(grades[length(vertices)],grade)
    end
    boundaries=Matrix{Int}[]
    for slot in 2:length(simplices)
        rows=Dict(Tuple(v)=>i for (i,v) in enumerate(simplices[slot-1]))
        B=zeros(Int,length(simplices[slot-1]),length(simplices[slot]))
        for (j,v) in enumerate(simplices[slot]), k in eachindex(v)
            face=Tuple(v[t] for t in eachindex(v) if t!=k)
            B[rows[face],j]=isodd(k) ? 1 : -1
        end
        push!(boundaries,B)
    end
    return (;boundaries,grades)
end

function _rips_contract_inputs(distances)
    n=size(distances,1)
    rows,cols,values=Int[],Int[],Float64[]
    for i in 1:n, j in i+1:n
        distances[i,j]<0 && continue
        push!(rows,i);push!(cols,j);push!(values,Float64(distances[i,j]))
    end
    # sparse() retains explicit zero entries; matrix size retains isolated sites.
    upper=sparse(rows,cols,values,n,n)
    lower=copy(transpose(upper))
    # Addition may drop explicit zeros on some sparse implementations; create
    # mirrored entries directly so this fixture makes their meaning explicit.
    symmetric=sparse(vcat(rows,cols),vcat(cols,rows),vcat(values,values),n,n)
    dense=map(x->x<0 ? Inf : Float64(x),distances)
    return (upper,lower,symmetric,transpose(upper),dense)
end

@testset "Rips contracts: sparse edges and independent persistence maps" begin
    cycle=[0 1 -1 1;1 0 1 -1;-1 1 0 1;1 -1 1 0]
    filled=copy(cycle);filled[1,3]=filled[3,1]=2
    zeroedge=[0 0 -1 -1;0 0 2 -1;-1 2 0 -1;-1 -1 -1 0]
    octahedron=ones(Int,6,6)
    for i in 1:6
        octahedron[i,i]=0
    end
    for (i,j) in ((1,2),(3,4),(5,6))
        octahedron[i,j]=octahedron[j,i]=2
    end
    fixtures=[(cycle,2),(filled,2),(zeroedge,2),(octahedron,3)]
    rng=MersenneTwister(21129)
    for _ in 1:4
        A=zeros(Int,5,5)
        for i in 1:5, j in i+1:5
            A[i,j]=A[j,i]=rand(rng,-1:3)
        end
        push!(fixtures,(A,3))
    end
    for (distances,max_dim) in fixtures, cutoff in (1,3), p in (2,3,101)
        reference=_rips_contract_reference(distances,max_dim;radius=cutoff)
        filtration=DI.RipsFiltration(;max_dim,radius=cutoff)
        diagrams=[]
        for input in _rips_contract_inputs(distances)
            G=DI.graded_complex(DI.build_graded_complex(input,filtration))
            diagram=OP.persistence_diagram(input,filtration;field=CM.Fp(p),representatives=true)
            _a112_check_maps(diagram,reference,p)
            _a112_check_representatives(G,diagram,p)
            push!(diagrams,diagram)
        end
        for diagram in diagrams[2:end], dim in 0:max_dim
            @test OP.finite_intervals(diagram;dim)==OP.finite_intervals(first(diagrams);dim)
            @test OP.essential_births(diagram;dim)==OP.essential_births(first(diagrams);dim)
        end
    end
    for p in (2,3,101)
        sparse_filled=first(_rips_contract_inputs(filled))
        D=OP.persistence_diagram(sparse_filled,DI.RipsFiltration(max_dim=2);field=CM.Fp(p),max_homology_dim=1,representatives=true)
        @test OP.finite_intervals(D;dim=1)==[(1.0,2.0)]
        @test OP.persistence_representative(D;dim=1,index=1).available
        @test OP.check_persistence_diagram(D;throw=true).valid
        @test isempty(OP.essential_births(D;dim=1))
        @test_throws ArgumentError OP.finite_intervals(D;dim=2)
        @test_throws ArgumentError OP.essential_births(D;dim=2)
        @test_throws ArgumentError OP.persistence_representative(D;dim=2,index=1)
        truncated=OP.persistence_diagram(sparse_filled,DI.RipsFiltration(max_dim=2,radius=1);field=CM.Fp(p),max_homology_dim=1)
        @test OP.essential_births(truncated;dim=1)==[1.0]
        @test OP.provenance(truncated).window==(0.0,1.0)
        @test OP.provenance(truncated).essential_interpretation==:survives_radius_cutoff
        for kind in (:barcode,:persistence_diagram)
            spec=TamerOp.Visualization.visual_spec(truncated;kind,dim=1)
            @test spec.metadata.essential_status==:survives_radius_cutoff
            @test spec.metadata.restriction_window==(0.0,1.0)
            @test occursin("known through radius 1.0",spec.subtitle)
        end
        @test OP.provenance(D).rips.complete_homology_through==1
        @test OP.provenance(D).rips.stored_zero_distance==:edge_at_zero
        D2=OP.persistence_diagram(first(_rips_contract_inputs(octahedron)),DI.RipsFiltration(max_dim=3);field=CM.Fp(p),max_homology_dim=2)
        @test OP.finite_intervals(D2;dim=2)==[(1.0,2.0)]
        @test isempty(OP.essential_births(D2;dim=2))
    end
end

@testset "Rips contracts: degree cutoff and sparse input validation" begin
    input=sparse([1],[2],[0.0],3,3)
    before=(copy(input.colptr),copy(input.rowval),copy(input.nzval))
    D=OP.persistence_diagram(input,DI.RipsFiltration(max_dim=1);max_homology_dim=0)
    @test OP.essential_births(D;dim=0)==[0.0,0.0]
    @test isempty(OP.finite_intervals(D;dim=0))
    @test before==(input.colptr,input.rowval,input.nzval)
    for bad in (-1,true,1.5,typemax(Int),big(2)^90)
        @test_throws ArgumentError DI.RipsFiltration(max_dim=bad)
        @test_throws ArgumentError OP.persistence_diagram(input,DI.RipsFiltration(max_dim=2);max_homology_dim=bad)
    end
    for radius in (-1,NaN,-Inf,true,big(10)^400)
        @test_throws ArgumentError DI.RipsFiltration(;radius)
    end
    @test_throws ArgumentError OP.persistence_diagram(input,DI.RipsFiltration(max_dim=1);max_homology_dim=1)
    @test_throws ArgumentError OP.persistence_diagram(input,DI.RipsFiltration(max_dim=2);max_homology_dim=2)
    @test_throws ArgumentError OP.persistence_diagram(input,DI.RipsFiltration(max_dim=2);order=:superlevel)
    @test_throws ArgumentError OP.persistence_diagram(DT.PointCloud([[0.0],[1.0]]),DI.AlphaFiltration();max_homology_dim=1)
    @test !DI.check_filtration(DI.RipsFiltration((;max_dim=-1))).valid
    @test_throws ArgumentError DI.build_graded_complex(input,DI.RipsFiltration((;max_dim=-1)))
    for bad in (spzeros(0,0),spzeros(2,3),sparse([1],[1],[1.0],2,2),
                sparse([1],[2],[NaN],2,2),sparse([1],[2],[-1.0],2,2),
                sparse([1,2],[2,1],[1.0,2.0],2,2),
                sparse([1,2],[2,1],[Inf,1.0],2,2),
                sparse([1],[2],[big(10)^400],2,2))
        @test_throws ArgumentError DI.build_graded_complex(bad,DI.RipsFiltration(max_dim=2))
        @test !DI.check_data_filtration(bad,DI.RipsFiltration(max_dim=2)).valid
    end
    malformed=copy(input);malformed.rowval[1]=0
    @test_throws ArgumentError DI.build_graded_complex(malformed,DI.RipsFiltration())
    malformed=copy(input);malformed.colptr[end]=3
    @test_throws ArgumentError DI.build_graded_complex(malformed,DI.RipsFiltration())
    duplicate=SparseMatrixCSC(2,2,[1,1,3],[1,1],[1.0,1.0])
    @test_throws ArgumentError DI.build_graded_complex(duplicate,DI.RipsFiltration())
    near=sparse([1,2],[2,1],[1.0,1.0+1e-12],2,2)
    @test OP.finite_intervals(OP.persistence_diagram(near,DI.RipsFiltration());dim=0)==[(0.0,1.0)]
    absent=sparse([1],[2],[Inf],3,3)
    @test length(OP.essential_births(OP.persistence_diagram(absent,DI.RipsFiltration());dim=0))==3
    # A skeleton remains available explicitly, but is not certified in its top degree.
    triangle=ones(3,3)-Matrix{Float64}(I,3,3)
    skeleton=OP.persistence_diagram(triangle,DI.RipsFiltration(max_dim=1))
    @test OP.essential_births(skeleton;dim=1)==[1.0]
    @test OP.provenance(skeleton).rips.complete_homology_through==0
end

@testset "Rips contracts: shared budgets radius and clique construction" begin
    for mode in (:none,:radius,:knn), max_dim in 0:3
        input=first(_rips_contract_inputs([0 1 2 3;1 0 1 2;2 1 0 1;3 2 1 0]))
        opts=OPT.ConstructionOptions(sparsify=mode)
        filtration=DI.RipsFiltration(;max_dim,radius=1,knn=2,construction=opts)
        G=DI.graded_complex(DI.build_graded_complex(input,filtration))
        @test DT.cell_counts(G)==vcat([4],max_dim>=1 ? [3] : Int[],zeros(Int,max(0,max_dim-1)))
        @test maximum(first.(G.grades))<=1
        @test DI.check_data_filtration(input,filtration).valid
        est=DI.estimate_ingestion(input,filtration)
        @test all(DI.cell_counts_by_dim(est).>=DT.cell_counts(G))
        st=TamerOp.encode(input,filtration;stage=:simplex_tree)
        @test DT.cell_counts(st)==DT.cell_counts(G)
        dense=last(_rips_contract_inputs([0 1 2 3;1 0 1 2;2 1 0 1;3 2 1 0]))
        @test DT.cell_counts(DI.graded_complex(DI.build_graded_complex(dense,filtration)))==DT.cell_counts(G)
        points=DT.PointCloud([[0.0],[1.0],[2.0],[3.0]])
        @test DT.cell_counts(DI.graded_complex(DI.build_graded_complex(points,filtration)))==DT.cell_counts(G)
    end
    # Accepted symmetry-rounding differences must not change nearest-neighbor
    # choices between sparse and dense storage. Canonical upper entries and
    # vertex-index tie breaking determine the same actual boundary maps.
    rounded=[0.0 1.0 1.0;1.0+1e-12 0.0 1.0;1.0+1e-12 1.0 0.0]
    f=DI.RipsFiltration(max_dim=2,knn=1,construction=OPT.ConstructionOptions(sparsify=:knn))
    denseG=DI.graded_complex(DI.build_graded_complex(rounded,f))
    sparseG=DI.graded_complex(DI.build_graded_complex(sparse(triu(rounded,1)),f))
    @test denseG.boundaries==sparseG.boundaries
    @test denseG.grades==sparseG.grades
    square=DT.PointCloud([[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]])
    for mode in (:none,:radius), p in (2,3,101)
        opts=OPT.ConstructionOptions(sparsify=mode)
        f=DI.RipsFiltration(max_dim=2,radius=2,construction=opts)
        D=OP.persistence_diagram(square,f;field=CM.Fp(p),max_homology_dim=1)
        @test OP.finite_intervals(D;dim=1)==[(1.0,sqrt(2.0))]
        @test isempty(OP.essential_births(D;dim=1))
    end
    triangle=sparse([1,1,2],[2,3,3],[1.0,1,1],3,3)
    for budget in ((max_edges=2,), (max_simplices=6,), (memory_budget_bytes=71,))
        opts=OPT.ConstructionOptions(;budget)
        @test_throws ArgumentError DI.build_graded_complex(triangle,DI.RipsFiltration(max_dim=2,construction=opts))
    end
    opts=OPT.ConstructionOptions(budget=(max_edges=3,max_simplices=7,memory_budget_bytes=72))
    @test DT.cell_counts(DI.graded_complex(DI.build_graded_complex(triangle,DI.RipsFiltration(max_dim=2,construction=opts))))==[3,3,1]
    collapse=OPT.ConstructionOptions(collapse=:dominated_edges,budget=(max_edges=3,max_simplices=5,))
    @test DT.cell_counts(DI.graded_complex(DI.build_graded_complex(triangle,DI.RipsFiltration(max_dim=2,construction=collapse))))==[3,2,0]
    too_few=OPT.ConstructionOptions(collapse=:dominated_edges,budget=(max_edges=2,))
    @test_throws ArgumentError DI.build_graded_complex(triangle,DI.RipsFiltration(max_dim=2,construction=too_few))
    # A large path exercises sparse storage with a budget that rejects the
    # accidental complete graph caused by interpreting omitted entries as zero.
    n=1000;path=sparse(collect(1:n-1),collect(2:n),ones(n-1),n,n)
    opts=OPT.ConstructionOptions(budget=(max_edges=n-1,max_simplices=2n-1))
    G=DI.graded_complex(DI.build_graded_complex(path,DI.RipsFiltration(max_dim=3,construction=opts)))
    @test DT.cell_counts(G)==[n,n-1,0,0]
    @test nnz(G.boundaries[1])==2(n-1)
    @test length(OP.finite_intervals(OP.persistence_diagram(G);dim=0))==n-1
end

@testset "A115: implicit Rips independent persistent maps" begin
    rng=MersenneTwister(115)
    fixtures=Matrix{Int}[
        [0 1 -1 1;1 0 1 -1;-1 1 0 1;1 -1 1 0],
        [0 1 2 1;1 0 1 -1;2 1 0 1;1 -1 1 0],
        [0 0 -1;0 0 2;-1 2 0],
        zeros(Int,4,4)]
    for n in 1:7, trial in 1:3
        A=zeros(Int,n,n)
        for i in 1:n,j in i+1:n
            A[i,j]=A[j,i]=rand(rng,-1:4)
        end
        push!(fixtures,A)
    end
    for A in fixtures, p in (2,3,101), q in 0:2
        cutoff=2
        reference=_rips_contract_reference(A,q+1;radius=cutoff)
        input=first(_rips_contract_inputs(A))
        f=DI.RipsFiltration(max_dim=q+1,radius=cutoff)
        implicit=OP.persistence_diagram(input,f;field=CM.Fp(p),max_homology_dim=q,method=:implicit)
        explicit=OP.persistence_diagram(input,f;field=CM.Fp(p),max_homology_dim=q,method=:explicit)
        @test implicit.finite_by_dim==explicit.finite_by_dim
        @test implicit.essential_by_dim==explicit.essential_by_dim
        @test OP.check_persistence_diagram(implicit;throw=true).valid
        for d in 0:q,s in (-1,0,1,2,3),t in s:3
            @test _a21_barcode_map_rank(implicit,d,s,t,:sublevel)==
                _a112_map_rank(reference,d,s,t,:sublevel,p)
        end
    end
    # Relabeling vertices changes tie refinements, not interval multiplicities.
    for p in (2,3,101,2305843009213693951), q in (1,2)
        A=ones(Int,6,6)-Matrix{Int}(I,6,6)
        for (u,v) in ((1,2),(3,4),(5,6));A[u,v]=A[v,u]=2;end
        f=DI.RipsFiltration(max_dim=q+1)
        for input in _rips_contract_inputs(A)
            D=OP.persistence_diagram(input,f;field=CM.Fp(p),max_homology_dim=q)
            @test OP.finite_intervals(D;dim=1)==Tuple{Float64,Float64}[]
            q==2 && @test OP.finite_intervals(D;dim=2)==[(1.0,2.0)]
        end
        permutation=randperm(rng,6)
        D=OP.persistence_diagram(Float64.(A[permutation,permutation]),f;field=CM.Fp(p),max_homology_dim=q)
        q==2 && @test OP.finite_intervals(D;dim=2)==[(1.0,2.0)]
    end
end

@testset "A115: signed cofaces and characteristic-changing flag complex" begin
    # Barycentric subdivision of the six-vertex RP2 triangulation is flag.
    # Its H1 and H2 over F2 are one-dimensional; both vanish over odd primes.
    facets=[(1,2,3),(1,2,6),(1,3,5),(1,4,5),(1,4,6),
            (2,3,4),(2,4,5),(2,5,6),(3,4,6),(3,5,6)]
    faces=Set{Tuple}()
    for facet in facets, mask in 1:7
        push!(faces,Tuple(facet[i] for i in 1:3 if !iszero(mask & (1<<(i-1)))))
    end
    faces=sort!(collect(faces);by=x->(length(x),x))
    n=length(faces)
    A=fill(2,n,n)
    for i in 1:n
        A[i,i]=0
        for j in i+1:n
            if issubset(faces[i],faces[j]) || issubset(faces[j],faces[i])
                A[i,j]=A[j,i]=1
            end
        end
    end
    @test n==31
    for p in (2,3,101), radius in (1,2)
        f=DI.RipsFiltration(max_dim=3;radius)
        D=OP.persistence_diagram(Float64.(A),f;field=CM.Fp(p),max_homology_dim=2)
        E=OP.persistence_diagram(Float64.(A),f;field=CM.Fp(p),max_homology_dim=2,method=:explicit)
        @test D.finite_by_dim==E.finite_by_dim
        @test D.essential_by_dim==E.essential_by_dim
        for d in (1,2)
            @test OP.finite_intervals(D;dim=d)==(p==2 && radius==2 ? [(1.0,2.0)] : Tuple{Float64,Float64}[])
            @test OP.essential_births(D;dim=d)==(p==2 && radius==1 ? [1.0] : Float64[])
        end
    end
    # Verify each generated coefficient against an independently signed facet,
    # then verify delta^2=0 with BigInt arithmetic, not production field ops.
    input=[0.0 1 2 1;1 0 1 2;2 1 0 1;1 2 1 0]
    payload=DI._rips_persistence_graph(input,DI.RipsFiltration(max_dim=3),3)
    graph=OP._rips_graph(payload,3)
    for p in (2,3,101), d in 0:2
        K=CM.coeff_type(CM.Fp(p))
        sources=OP._rips_catalog(graph,Val(d+1),Set{Int}())
        targets=OP._rips_catalog(graph,Val(d+2),Set{Int}())
        for s in sources
            heap=OP._RipsTerm{K}[]
            OP._rips_coboundary!(heap,graph,s,one(K),Dict{Int,Int}(),OP._RipsReductionStats())
            got=Dict(t.index=>Int(t.coefficient.val) for t in heap)
            expected=Dict{Int,Int}()
            for t in targets,j in eachindex(t.vertices)
                face=Tuple(t.vertices[k] for k in eachindex(t.vertices) if k!=j)
                face==s.vertices && (expected[t.index]=mod(isodd(j) ? 1 : -1,p))
            end
            @test got==expected
            if d<=1
                twice=Dict{Int,BigInt}()
                for t in targets
                    haskey(got,t.index) || continue
                    empty!(heap)
                    OP._rips_coboundary!(heap,graph,t,one(K),Dict{Int,Int}(),OP._RipsReductionStats())
                    for u in heap
                        twice[u.index]=mod(get(twice,u.index,big(0))+big(got[t.index])*Int(u.coefficient.val),p)
                    end
                end
                @test all(iszero,values(twice))
            end
        end
    end
end

@testset "A115: routing budgets landmarks and sparse controls" begin
    # The documented root entrypoint returns the finite loop, not graph H1.
    S=sparse([1,2,3,1,1],[2,3,4,4,3],[1.0,1,1,1,2],4,4)
    public_diagram=TamerOp.persistence_diagram(S,TamerOp.RipsFiltration(max_dim=2);max_homology_dim=1)
    @test TamerOp.finite_intervals(public_diagram;dim=1)==[(1.0,2.0)]
    @test TamerOp.provenance(public_diagram).backend==:implicit_rips_cohomology
    @test TamerOp.provenance(public_diagram).construction==(requested=:rips,effective=:rips_flag_graph,substitution=:none)
    explicit_diagram=TamerOp.persistence_diagram(S,TamerOp.RipsFiltration(max_dim=2);max_homology_dim=1,method=:explicit)
    @test TamerOp.provenance(explicit_diagram).construction==(requested=:rips,effective=:not_recorded,substitution=:not_recorded)
    @test TamerOp.Advanced.check_persistence_diagram(public_diagram;throw=true).valid
    points=DT.PointCloud([[cos(t),sin(t)] for t in range(0,2pi;length=9)[1:8]])
    for mode in (:none,:radius,:knn,:greedy_perm),collapse in (:none,:dominated_edges),p in (2,3,101)
        f=DI.RipsFiltration(max_dim=3,radius=1.9,knn=3,n_landmarks=5,
            construction=OPT.ConstructionOptions(sparsify=mode,collapse=collapse))
        cache=CM.EncodingCache()
        a=OP.persistence_diagram(points,f;max_homology_dim=2,field=CM.Fp(p),cache)
        b=OP.persistence_diagram(points,f;max_homology_dim=2,field=CM.Fp(p),method=:explicit)
        c=OP.persistence_diagram(points,f;max_homology_dim=2,field=CM.Fp(p),cache)
        @test a.finite_by_dim==b.finite_by_dim==c.finite_by_dim
        @test a.essential_by_dim==b.essential_by_dim==c.essential_by_dim
        @test OP.provenance(a).computation.cross_call_mathematical_cache===false
        @test OP.provenance(a).computation.boundary_matrix_materialized===false
        mode==:greedy_perm && @test length(OP.provenance(a).input_selection.source_indices)==5
    end
    for radius in (nothing,1.5),collapse in (:none,:dominated_edges)
        f=DI.LandmarkRipsFiltration(max_dim=2,landmarks=[1,3,5,7];radius,
            construction=OPT.ConstructionOptions(;collapse))
        a=OP.persistence_diagram(points,f;max_homology_dim=1)
        b=OP.persistence_diagram(points,f;max_homology_dim=1,method=:explicit)
        @test a.finite_by_dim==b.finite_by_dim
        @test a.essential_by_dim==b.essential_by_dim
        @test OP.provenance(a).input_selection.source_indices==[1,3,5,7]
    end
    A=ones(4,4)-Matrix{Float64}(I,4,4)
    for max_dim in 0:3
        f=DI.RipsFiltration(;max_dim)
        a=OP.persistence_diagram(A,f)
        b=OP.persistence_diagram(A,f;method=:explicit)
        @test a.finite_by_dim==b.finite_by_dim
        @test a.essential_by_dim==b.essential_by_dim
        @test (OP.provenance(a).backend==:implicit_rips_cohomology)==(max_dim<=2)
    end
    f=DI.RipsFiltration(max_dim=2)
    D=OP.persistence_diagram(A,f;max_homology_dim=1,representatives=true)
    @test OP.provenance(D).representatives==:retained_reduction_cycles
    @test_throws ArgumentError OP.persistence_diagram(A,f;method=:imaginary)
    @test_throws ArgumentError OP.persistence_diagram(A,f;method=:implicit,representatives=true)
    @test_throws ArgumentError OP.persistence_diagram(A,DI.RipsFiltration(max_dim=4);max_homology_dim=3,method=:implicit)
    @test_throws ArgumentError OP.persistence_diagram(points,DI.AlphaFiltration();method=:implicit)
    @test_throws ArgumentError OP.persistence_diagram(A,f;cache=:invalid)
    for invalid in (NaN, Inf, -Inf), radius in (0.0, 1.0, Inf)
        bad=DT.PointCloud([[0.0,0.0],[invalid,1.0]])
        @test_throws ArgumentError OP.persistence_diagram(bad,DI.RipsFiltration(max_dim=2;radius);max_homology_dim=1)
    end
    triangle=ones(3,3)-Matrix{Float64}(I,3,3)
    for budget in ((max_edges=2,),(max_simplices=6,),(memory_budget_bytes=71,))
        f=DI.RipsFiltration(max_dim=2,construction=OPT.ConstructionOptions(;budget))
        @test_throws ArgumentError OP.persistence_diagram(triangle,f;max_homology_dim=1)
    end
    f=DI.RipsFiltration(max_dim=2,construction=OPT.ConstructionOptions(budget=(max_simplices=7,memory_budget_bytes=72)))
    @test isempty(OP.finite_intervals(OP.persistence_diagram(triangle,f;max_homology_dim=1);dim=1))
    f=DI.RipsFiltration(max_dim=2,construction=OPT.ConstructionOptions(collapse=:dominated_edges,budget=(max_edges=3,max_simplices=5)))
    @test OP.provenance(OP.persistence_diagram(triangle,f;max_homology_dim=1)).budget_counts==[3,2,0]
    # Reject unrepresentable simplex indices before allocating a graph/table.
    @test_throws ArgumentError OP._rips_graph((n=200000,),3)
    n=1000;path=sparse(collect(1:n-1),collect(2:n),ones(n-1),n,n)
    f=DI.RipsFiltration(max_dim=3,construction=OPT.ConstructionOptions(budget=(max_edges=n-1,max_simplices=2n-1)))
    D=OP.persistence_diagram(path,f;max_homology_dim=2)
    @test length(OP.finite_intervals(D;dim=0))==n-1
    @test OP.essential_births(D;dim=0)==[0.0]
    @test isempty(OP.persistence_intervals(D;dim=1))
    @test isempty(OP.persistence_intervals(D;dim=2))
    @test OP.provenance(D).input_selection.retained_edges==n-1
    # H0 and H1 share the edge catalog, even when all edges are cleared by H0.
    @test OP.provenance(D).reduction_stats.largest_column_catalog==n-1
end

# A117: dense BigInt cochain algebra, independent of reverse column reduction.
function _a117_check_cocycles(G, D, p; simplex_vertices=nothing, source_indices=nothing)
    reference = _a112_reference(G)
    counts = length.(reference.grades)
    nd = length(D.finite_by_dim)
    events = sort!(unique(vcat(reference.grades...));rev=D.order===:superlevel)
    isempty(events) && return
    levels = sort!(unique(vcat(events, [(events[i] isa AbstractFloat ? (events[i]+events[i+1])/2 : (big(events[i])+big(events[i+1]))//2)
        for i in 1:length(events)-1]));rev=D.order===:superlevel)
    active(g,t) = D.order===:sublevel ? g<=t : g>=t
    alive(b,d,t) = active(b,t) && (d===nothing || !active(d,t))
    function vector_at(dim,kind,index,t)
        r = OP.persistence_cocycle(D;dim,kind,index,scale=t)
        @test r.available && r.field==CM.Fp(p) && r.scale==t
        @test r.variance===:contravariant
        @test all(g->active(g,t),r.cochain.cell_grades)
        z = zeros(BigInt,counts[dim+1])
        for j in eachindex(r.cochain.coefficients)
            c = r.cochain.coefficients[j]
            @test 0<c<p
            if simplex_vertices===nothing
                cell = r.cochain.cell_indices[j]
                @test r.cochain.cell_ids[j]==G.cells_by_dim[dim+1][cell]
            else
                verts = collect(r.source_vertices[j])
                local_vertices = source_indices===nothing ? verts :
                    [findfirst(==(v),source_indices) for v in verts]
                inversions = sum((local_vertices[i]>local_vertices[k] for i in eachindex(local_vertices)
                    for k in i+1:length(local_vertices));init=0)
                c *= isodd(inversions) ? -1 : 1
                cell = findfirst(==(sort(local_vertices)),simplex_vertices[dim+1])
                @test cell !== nothing
            end
            z[cell] = mod(c,p)
            @test r.cochain.cell_grades[j]==reference.grades[dim+1][cell]
        end
        return z
    end
    for dim in 0:nd-1
        bars = [(kind,i, kind===:finite ? value[1] : value,
                          kind===:finite ? value[2] : nothing)
            for kind in (:finite,:essential)
            for (i,value) in enumerate(kind===:finite ? OP.finite_intervals(D;dim) : OP.essential_births(D;dim))]
        matrices = Matrix{BigInt}[]
        active_cells = Vector{Vector{Int}}()
        coboundaries = Matrix{BigInt}[]
        for t in levels
            selected=[findall(g->active(g,t),gs) for gs in reference.grades]
            cells=selected[dim+1]
            delta = dim+1<length(counts) ? transpose(reference.boundaries[dim+1][cells,selected[dim+2]]) :
                zeros(Int,0,length(cells))
            boundaries = dim>0 ? BigInt.(transpose(reference.boundaries[dim][selected[dim],cells])) :
                zeros(BigInt,length(cells),0)
            columns=[vector_at(dim,kind,i,t) for (kind,i,b,d) in bars if alive(b,d,t)]
            full = isempty(columns) ? zeros(BigInt,counts[dim+1],0) : hcat(columns...)
            C=full[cells,:]
            @test all(iszero,mod.(delta*C,p))
            @test _a112_rank(hcat(boundaries,C),p)-_a112_rank(boundaries,p)==size(C,2)
            @test size(C,2)==length(cells)-_a112_rank(delta,p)-_a112_rank(boundaries,p)
            push!(matrices,full);push!(active_cells,cells);push!(coboundaries,boundaries)
        end
        for a in eachindex(levels), b in a:length(levels)
            cells=active_cells[a]; B=coboundaries[a]
            restricted=matrices[b][cells,:]
            @test _a112_rank(hcat(B,restricted),p)-_a112_rank(B,p)==
                _a112_map_rank(reference,dim,levels[a],levels[b],D.order,p)
            for (kind,i,birth,death) in bars
                if alive(birth,death,levels[b])
                    later=vector_at(dim,kind,i,levels[b])[cells]
                    if alive(birth,death,levels[a])
                        @test later==vector_at(dim,kind,i,levels[a])[cells]
                    else
                        @test all(iszero,later) # restriction before birth
                    end
                elseif alive(birth,death,levels[a]) && death!==nothing && active(death,levels[b])
                    earlier=vector_at(dim,kind,i,levels[a])[cells]
                    # The dying class cannot extend through this inclusion,
                    # even after adding a coboundary or changing its representative.
                    @test _a112_rank(hcat(B,restricted,earlier),p)==_a112_rank(hcat(B,restricted),p)+1
                end
            end
        end
    end
    @test OP.check_persistence_diagram(D;throw=true).valid
end

function _a117_flag_fixture(A,maxdim)
    ref = _rips_contract_reference(A,maxdim)
    simplices=[Vector{Int}[] for _ in 0:maxdim]
    for mask in 1:(1<<size(A,1))-1
        v=[i for i in axes(A,1) if !iszero(mask & (1<<(i-1)))]
        length(v)<=maxdim+1 || continue
        all(A[v[i],v[j]]>=0 for i in eachindex(v) for j in i+1:length(v)) || continue
        push!(simplices[length(v)],v)
    end
    G=DT.GradedComplex([collect(1:length(v)) for v in simplices],sparse.(ref.boundaries),
        [(g,) for gs in ref.grades for g in gs])
    return G,simplices
end

@testset "A117 cohomology equations bases restrictions and duality" begin
    rng=MersenneTwister(117)
    for order in (:sublevel,:superlevel), p in (2,3,101)
        G,_=_a112_simplicial_fixture(rng,order)
        D=OP.persistence_diagram(G;order,field=CM.Fp(p),cocycles=true)
        H=OP.persistence_diagram(G;order,field=CM.Fp(p),representatives=true,cocycles=true)
        @test D.finite_by_dim==H.finite_by_dim && D.essential_by_dim==H.essential_by_dim
        @test OP.describe(H).representatives_available && OP.describe(H).cocycles_available
        _a117_check_cocycles(G,D,p)
        _a112_check_representatives(G,H,p)
    end
    for p in (2,3,101,2^61-1), q in (2,3,typemax(Int)), order in (:sublevel,:superlevel)
        grades=order===:sublevel ? [(0//1,),(1//1,),(2//1,)] : [(2//1,),(1//1,),(0//1,)]
        G=DT.GradedComplex([[17],[17],[17]],[spzeros(Int,1,1),sparse(reshape([q],1,1))],grades)
        D=OP.persistence_diagram(G;order,field=CM.Fp(p),cocycles=true)
        H=OP.persistence_diagram(G;order,field=CM.Fp(p))
        @test D.finite_by_dim==H.finite_by_dim && D.essential_by_dim==H.essential_by_dim
        _a117_check_cocycles(G,D,p)
    end
end

@testset "A117 implicit source simplices and persistent cohomology" begin
    rng=MersenneTwister(1117)
    fixtures=Matrix{Int}[[0 1 2 1;1 0 1 2;2 1 0 1;1 2 1 0],
        [i==j ? 0 : cld(i,2)==cld(j,2) ? 2 : 1 for i in 1:6,j in 1:6]]
    for n in 1:6
        A=zeros(Int,n,n)
        for i in 1:n,j in i+1:n
            A[i,j]=A[j,i]=rand(rng,(-1,0,1,2,3))
        end
        push!(fixtures,A)
    end
    for A in fixtures,p in (2,3,101)
        G,simplices=_a117_flag_fixture(A,3)
        for input in _rips_contract_inputs(A)
            D=OP.persistence_diagram(input,DI.RipsFiltration(max_dim=3);field=CM.Fp(p),
                max_homology_dim=2,cocycles=true,method=:implicit)
            H=OP.persistence_diagram(G;field=CM.Fp(p))
            @test D.finite_by_dim==H.finite_by_dim[1:3] && D.essential_by_dim==H.essential_by_dim[1:3]
            _a117_check_cocycles(G,D,p;simplex_vertices=simplices)
        end
    end
    A=fixtures[1];G,simplices=_a117_flag_fixture(A,1)
    D=OP.persistence_diagram(Float64.(A),DI.RipsFiltration(max_dim=1);cocycles=true)
    _a117_check_cocycles(G,D,2;simplex_vertices=simplices)
end

@testset "A117 cubical routes and public retention contracts" begin
    values=[0 0 0;0 5 0;0 0 0]
    for input in (:top_cells,:vertices), order in (:sublevel,:superlevel),periodic in (false,true),p in (2,3,101)
        vals=order===:sublevel ? values : 5 .- values
        G=input===:top_cells ? OP._top_cell_complex_2d(vals,(periodic,periodic),order) :
            OP._vertex_cubical_complex(vals,(periodic,periodic),order,
                DI.construction_mode(DI.CubicalFiltration()),nothing)
        D=TamerOp.cubical_persistence(vals;input,order,periodic,field=CM.Fp(p),cocycles=true)
        _a117_check_cocycles(G,D,p)
        H=TamerOp.cubical_persistence(vals;input,order,periodic,field=CM.Fp(p))
        @test D.finite_by_dim==H.finite_by_dim && D.essential_by_dim==H.essential_by_dim
    end
    D=TamerOp.cubical_persistence(values;cocycles=true)
    plain=TamerOp.cubical_persistence(values)
    @test TOA.persistence_cocycle===OP.persistence_cocycle
    @test OP.provenance(D).cocycles===:retained_restriction_cochains
    @test !OP.persistence_cocycle(plain;dim=1,scale=1).available
    @test !OP.persistence_representative(D;dim=1).available
    @test !OP.describe(plain).cocycles_available
    for scale in (true,NaN,Inf,-1,5,6,"1")
        @test_throws ArgumentError OP.persistence_cocycle(D;dim=1,scale)
    end
    for index in (true,0,2), dim in (-1,0,1,2)
        @test_throws ArgumentError OP.persistence_cocycle(D;dim,scale=1,index)
    end
    @test_throws ArgumentError OP.persistence_cocycle(D;dim=1,scale=1,kind=:cycle)
    @test_throws ArgumentError OP.cubical_persistence(values;cocycles=:yes)
    @test_throws ArgumentError OP.persistence_diagram(values,DI.RipsFiltration();cocycles=:yes)
    image=DT.ImageNd(values)
    C=OP.persistence_diagram(image,DI.CubicalFiltration();cocycles=true)
    @test OP.describe(C).cocycles_available
    A=[0.0 1 2 1;1 0 1 2;2 1 0 1;1 2 1 0]
    E=OP.persistence_diagram(A,DI.RipsFiltration(max_dim=2);cocycles=true,representatives=true,max_homology_dim=1)
    @test OP.describe(E).cocycles_available && OP.describe(E).representatives_available
    @test OP.check_persistence_diagram(E;throw=true).valid
    W=OP.persistence_diagram(A,DI.RipsFiltration(max_dim=2,radius=1);cocycles=true,max_homology_dim=1)
    @test OP.persistence_cocycle(W;dim=1,kind=:essential,scale=1).available
    @test_throws ArgumentError OP.persistence_cocycle(W;dim=1,kind=:essential,scale=1.1)
    @test_throws ArgumentError OP.persistence_cocycle(W;dim=2,scale=1)
    @test_throws ArgumentError OP.persistence_diagram(A,DI.RipsFiltration(max_dim=2,construction=OPT.ConstructionOptions(collapse=:dominated_edges));cocycles=true)
    G=DT.GradedComplex([Int[]],SparseMatrixCSC{Int,Int}[],Tuple{Int}[])
    @test OP.check_persistence_diagram(OP.persistence_diagram(G;cocycles=true)).valid
    push!(D.finite_by_dim[2],(0,4))
    @test !OP.check_persistence_diagram(D).valid
end

@testset "A117 landmark orientation and exact-scale edge cases" begin
    points=DT.PointCloud([[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0],[10.0,10.0]])
    chosen=[4,2,1,3]
    f=DI.LandmarkRipsFiltration(max_dim=2,landmarks=chosen)
    A=[norm(points.points[i]-points.points[j]) for i in chosen,j in chosen]
    # Build independent signed facets using an integer order-equivalent metric,
    # then restore the actual edge-length grades.
    integer_A=round.(Int,2 .* A)
    G,simplices=_a117_flag_fixture(integer_A,2)
    grades=[(maximum((A[v[i],v[j]] for i in eachindex(v) for j in i+1:length(v));init=0.0),)
        for cells in simplices for v in cells]
    G=DT.GradedComplex(G.cells_by_dim,G.boundaries,grades)
    for p in (2,3,101)
        D=OP.persistence_diagram(points,f;field=CM.Fp(p),cocycles=true,max_homology_dim=1)
        _a117_check_cocycles(G,D,p;simplex_vertices=simplices,source_indices=chosen)
        r=OP.persistence_cocycle(D;dim=1,scale=1)
        @test all(v->all(in(chosen),v),r.source_vertices)
        repeated=OP.persistence_diagram(points,f;field=CM.Fp(p),cocycles=true,max_homology_dim=1)
        @test OP.persistence_cocycle(repeated;dim=1,scale=1)==r
    end
    huge=big(2)^80
    G=DT.GradedComplex([[1],[2],[3]],[spzeros(Int,1,1),sparse(reshape([1],1,1))],
        [(huge,),(huge+1,),(huge+2,)])
    D=OP.persistence_diagram(G;cocycles=true,field=CM.Fp(101))
    r=OP.persistence_cocycle(D;dim=1,scale=(2huge+3)//2)
    @test r.interval==(huge+1,huge+2)
    @test eltype(r.cochain.cell_grades)===BigInt
    @test_throws ArgumentError OP.persistence_cocycle(D;dim=1,scale=huge+2)
    @test OP.persistence_cocycle(D;dim=0,kind=:essential,scale=huge+10).available
    D.retained_cocycles.finite[2][1].cochain.coefficients[1]=101
    @test !OP.check_persistence_diagram(D).valid
end

# Independent subset enumeration for weighted flag complexes. No production
# graph preparation, clique enumeration, grade aggregation or boundary builder.
function _weighted_flag_reference(A,maxdim;threshold=Inf)
    n=size(A,1)
    simplices=[Vector{Int}[] for _ in 0:maxdim]
    grades=[Int[] for _ in 0:maxdim]
    for mask in 1:(1<<n)-1
        v=[i for i in 1:n if !iszero(mask & (1<<(i-1)))]
        length(v)<=maxdim+1 || continue
        values=vcat([A[i,i] for i in v],[A[v[i],v[j]] for i in eachindex(v) for j in i+1:length(v)])
        all(isfinite,values) || continue
        grade=maximum(values)
        grade<=threshold || continue
        push!(simplices[length(v)],v);push!(grades[length(v)],Int(grade))
    end
    boundaries=Matrix{Int}[]
    for k in 2:length(simplices)
        B=zeros(Int,length(simplices[k-1]),length(simplices[k]))
        for (j,v) in enumerate(simplices[k]),i in eachindex(v)
            row=findfirst(==([v[t] for t in eachindex(v) if t!=i]),simplices[k-1])
            B[row,j]=isodd(i) ? 1 : -1
        end
        push!(boundaries,B)
    end
    G=DT.GradedComplex([collect(1:length(v)) for v in simplices],sparse.(boundaries),
        [(g,) for gs in grades for g in gs])
    return G,simplices,(;boundaries,grades)
end

@testset "Rips inputs: landmark order coverage and distances" begin
    DI=TamerOp.DataIngestion
    x=[0.,1,2,3,4]
    cloud=DT.PointCloud([[v] for v in x]); A=abs.(x .- x')
    for input in (cloud,A)
        s=TamerOp.select_landmarks(input;count=3,retain_distances=true)
        @test TamerOp.landmark_indices(s)==[1,5,3]
        @test TamerOp.covering_radius(s)==1
        @test TamerOp.landmark_distances(s)==A[[1,5,3],:]
        summary=TamerOp.describe(s)
        @test summary.insertion_radii==[Inf,4,2]
        @test summary.nearest_landmark_indices==[1,1,3,5,5]
        @test summary.nearest_distances==[0,1,0,1,0]
        @test summary.conditional_bottleneck_bound==2
        @test summary.bound_scope===:full_metric_rips_filtrations
        @test occursin("3 of 5",sprint(show,s))
        copied=TamerOp.landmark_indices(s);copied[1]=2
        @test TamerOp.landmark_indices(s)[1]==1
        for m in 1:5
            subset=DI.select_landmarks(input;count=m)
            idx=DI.landmark_indices(subset)
            @test length(idx)==m && allunique(idx)
            ds=A[idx,:]
            @test DI.covering_radius(subset)==maximum(minimum(ds;dims=1))
            # Nearest-point map induces the usual additive 2r Rips shift.
            nearest=DI.describe(subset).nearest_landmark_indices
            r=DI.covering_radius(subset)
            @test all(A[nearest[i],nearest[j]] <= A[i,j]+2r for i in 1:5 for j in 1:5)
            @test_throws ArgumentError DI.landmark_distances(subset)
        end
        supplied=DI.select_landmarks(input;indices=[5,1],retain_distances=true)
        @test DI.landmark_indices(supplied)==[5,1]
        @test DI.covering_radius(supplied)==2
        @test DI.landmark_distances(supplied)==A[[5,1],:]
    end
    @test DI.landmark_indices(DI.select_landmarks(zeros(5,5);count=5))==collect(1:5)
    @test DI.landmark_indices(DI.select_landmarks(DT.PointCloud([[0.],[0.],[1.]]);count=3))==[1,3,2]
    for count in (0,6,-1,true,1.5)
        @test_throws ArgumentError DI.select_landmarks(cloud;count)
    end
    for indices in (Int[],[0],[6],[1,1],[true],[1.5])
        @test_throws ArgumentError DI.select_landmarks(cloud;indices)
    end
    @test_throws ArgumentError DI.select_landmarks(cloud)
    @test_throws ArgumentError DI.select_landmarks(cloud;count=2,indices=[1])
    @test_throws ArgumentError DI.select_landmarks(sparse(A);count=2)
    @test_throws ArgumentError DI.select_landmarks([0. Inf;Inf 0];count=1)
    @test_throws ArgumentError DI.select_landmarks(DT.PointCloud([[NaN],[1.]]);count=1)
end

@testset "Rips inputs: landmark persistence pipeline and provenance" begin
    DI=TamerOp.DataIngestion
    x=[0.,1,2,3,4];A=abs.(x .- x');cloud=DT.PointCloud([[v] for v in x])
    f=DI.RipsFiltration(max_dim=2,n_landmarks=3,
        construction=TamerOp.ConstructionOptions(sparsify=:greedy_perm))
    lm=DI.LandmarkRipsFiltration(max_dim=2,landmarks=[5,1,3])
    for input in (cloud,A),filtration in (f,lm), method in (:implicit,:explicit),p in (2,3)
        d=OP.persistence_diagram(input,filtration;max_homology_dim=1,method,field=CM.Fp(p),cocycles=true)
        info=TamerOp.landmark_selection(d)
        idx=DI.landmark_indices(info)
        expected=OP.persistence_diagram(A[idx,idx],DI.RipsFiltration(max_dim=2);max_homology_dim=1,field=CM.Fp(p))
        @test d.finite_by_dim==expected.finite_by_dim
        @test d.essential_by_dim==expected.essential_by_dim
        @test DI.covering_radius(info)==1
        @test DI.describe(info).distances_retained==false
        build=DI.build_graded_complex(input,filtration)
        explicit=OP.persistence_diagram(DI.graded_complex(build);field=CM.Fp(p))
        @test OP.finite_intervals(explicit;dim=0)==OP.finite_intervals(d;dim=0)
        @test TamerOp.encode(input,filtration;stage=:graded_complex) isa DT.GradedComplex
    end
    @test TamerOp.landmark_selection(OP.persistence_diagram(A,DI.RipsFiltration()))===nothing
    # Existing automatic greedy input must not lose coincident selected vertices.
    d=OP.persistence_diagram(DT.PointCloud([[0.],[0.],[0.]]),DI.RipsFiltration(max_dim=0,n_landmarks=3,
        construction=TamerOp.ConstructionOptions(sparsify=:greedy_perm)))
    @test length(OP.essential_births(d;dim=0))==3
end

@testset "Rips inputs: weighted vertices independent maps and cocycles" begin
    DI=TamerOp.DataIngestion
    fixtures=Matrix{Float64}[]
    # A loop is born at 1 and filled at 4; a disconnected vertex arrives at 5.
    A=[-2. 0 4 -1 Inf;0 0 1 4 Inf;4 1 1 1 Inf;-1 4 1 -1 Inf;Inf Inf Inf Inf 5]
    push!(fixtures,A)
    # The octahedral sphere appears at 2 and fills when opposite edges enter at 4.
    sphere=[i==j ? Float64(-i) : (cld(i,2)==cld(j,2) ? 4.0 : 2.0) for i in 1:6,j in 1:6]
    push!(fixtures,sphere)
    rng=MersenneTwister(20261004)
    for trial in 1:3
        n=6;births=rand(rng,-2:2,n);B=fill(Inf,n,n)
        for i in 1:n
            B[i,i]=births[i]
            for j in 1:i-1
                rand(rng)<0.7 || continue
                B[i,j]=B[j,i]=max(births[i],births[j])+rand(rng,0:2)
            end
        end
        push!(fixtures,B)
    end
    for A in fixtures,threshold in (Inf,1.0),p in (2,3,101)
        n=size(A,1)
        G,simplices,reference=_weighted_flag_reference(A,3;threshold)
        f=DI.EdgeWeightedFiltration(max_dim=3,threshold=threshold)
        rows,cols,vals=Int[],Int[],Float64[]
        for i in 1:n,j in i:n
            isfinite(A[i,j]) || continue
            push!(rows,i);push!(cols,j);push!(vals,A[i,j])
        end
        S=sparse(rows,cols,vals,n,n)
        edges=[(i,j) for i in 1:n for j in i+1:n if isfinite(A[i,j])]
        graph=DT.GraphData(n,edges;weights=[A[i,j] for (i,j) in edges])
        gf=DI.EdgeWeightedFiltration(vertex_births=diag(A),max_dim=3,threshold=threshold)
        for (input,filtration) in ((A,f),(S,f),(transpose(S),f),(graph,gf)),method in (:implicit,:explicit)
            d=OP.persistence_diagram(input,filtration;field=CM.Fp(p),max_homology_dim=2,method,cocycles=true)
            plain=OP.persistence_diagram(input,filtration;field=CM.Fp(p),max_homology_dim=2,method)
            @test d.finite_by_dim==plain.finite_by_dim
            @test d.essential_by_dim==plain.essential_by_dim
            oracle=OP.persistence_diagram(G;field=CM.Fp(p))
            @test d.finite_by_dim==oracle.finite_by_dim[1:3]
            @test d.essential_by_dim==oracle.essential_by_dim[1:3]
            events=sort(unique(vcat(reference.grades...)))
            for dim in 0:2,s in events,t in events
                s<=t || continue
                @test _a21_barcode_map_rank(d,dim,s,t,:sublevel)==_a112_map_rank(reference,dim,s,t,:sublevel,p)
            end
            if method===:implicit
                _a117_check_cocycles(G,d,p;simplex_vertices=simplices)
            else
                built=DI.graded_complex(DI.build_graded_complex(input,filtration))
                _a117_check_cocycles(built,d,p)
            end
            @test OP.check_persistence_diagram(d;throw=true).valid
            if A===sphere && threshold==Inf
                @test OP.finite_intervals(d;dim=2)==[(2.,4.)]
            end
            if A===first(fixtures) && threshold==Inf
                @test OP.finite_intervals(d;dim=1)==[(1.,4.)]
                @test OP.essential_births(d;dim=0)==[-2.,5.]
            end
        end
    end
end

@testset "Rips inputs: weighted higher-dimensional fallback" begin
    # The boundary of the four-dimensional cross-polytope is S^3.
    A=[i==j ? Float64(-i) : (cld(i,2)==cld(j,2) ? 4.0 : 2.0) for i in 1:8,j in 1:8]
    f=TamerOp.EdgeWeightedFiltration(max_dim=4)
    G,_,reference=_weighted_flag_reference(A,4)
    for p in (2,3,101),method in (:auto,:explicit)
        d=TamerOp.persistence_diagram(A,f;max_homology_dim=3,field=CM.Fp(p),method,cocycles=true)
        @test OP.finite_intervals(d;dim=3)==[(2.,4.)]
        @test _a112_map_rank(reference,3,2,3,:sublevel,p)==1
        @test _a112_map_rank(reference,3,2,4,:sublevel,p)==0
        built=TamerOp.encode(A,f;stage=:graded_complex)
        _a117_check_cocycles(built,d,p)
    end
    @test_throws ArgumentError TamerOp.persistence_diagram(A,f;max_homology_dim=3,method=:implicit)
end

@testset "Rips inputs: weighted contracts budgets and empty windows" begin
    DI=TamerOp.DataIngestion
    f=DI.EdgeWeightedFiltration(max_dim=2)
    for A in ([2. 1;1 0], [0. 1;2 0], [NaN 1;1 0], [0. -Inf;-Inf 0])
        @test_throws ArgumentError OP.persistence_diagram(A,f)
    end
    @test_throws ArgumentError OP.persistence_diagram([1. 2;2 0],DI.RipsFiltration())
    @test_throws ArgumentError OP.persistence_diagram([0. 1;1 0],DI.EdgeWeightedFiltration(vertex_births=[0,0]))
    @test_throws ArgumentError OP.persistence_diagram(DT.GraphData(2,[(1,2)];weights=[1.]),DI.EdgeWeightedFiltration(vertex_births=[0]))
    @test_throws ArgumentError OP.persistence_diagram(DT.GraphData(2,[(1,2),(2,1)];weights=[1.,2.]),f)
    for values in (1.0,[0.0 0.0])
        @test_throws ArgumentError OP.persistence_diagram(DT.GraphData(2,[(1,2)];weights=[1.]),DI.EdgeWeightedFiltration(vertex_births=values))
        @test_throws ArgumentError OP.persistence_diagram(DT.GraphData(2,[(1,2)]),DI.EdgeWeightedFiltration(edge_weights=values))
    end
    @test_throws ArgumentError OP.persistence_diagram(sparse([1,2],[2,1],[1.,2.],2,2),f)
    @test_throws ArgumentError DI.EdgeWeightedFiltration(max_dim=-1)
    @test_throws ArgumentError DI.EdgeWeightedFiltration(threshold=NaN)
    @test_throws ArgumentError DI.EdgeWeightedFiltration(construction=TamerOp.ConstructionOptions(collapse=:dominated_edges))
    A=[-2. 1 2;1 -1 2;2 2 0]
    for method in (:implicit,:explicit),cocycles in (false,true)
        d=OP.persistence_diagram(A,DI.EdgeWeightedFiltration(max_dim=2,threshold=-3);max_homology_dim=1,method,cocycles)
        @test all(isempty,d.finite_by_dim) && all(isempty,d.essential_by_dim)
        d=OP.persistence_diagram(A,DI.EdgeWeightedFiltration(max_dim=0);method,cocycles)
        @test OP.essential_births(d;dim=0)==[-2.,-1.,0.]
        d=OP.persistence_diagram(A,f;method,cocycles)
        @test OP.finite_intervals(d;dim=0)==[(-1.,1.),(0.,2.)]
        @test OP.essential_births(d;dim=0)==[-2.]
        # Every input grade is validated even when the cutoff excludes it.
        @test_throws ArgumentError OP.persistence_diagram([2. 1;1 0],DI.EdgeWeightedFiltration(threshold=-1);method,cocycles)
        for budget in (TamerOp.ConstructionBudget(max_edges=1),TamerOp.ConstructionBudget(max_simplices=3))
            @test_throws ArgumentError OP.persistence_diagram(A,DI.EdgeWeightedFiltration(max_dim=2,
                construction=TamerOp.ConstructionOptions(budget=budget));method,cocycles)
        end
    end
    limited=OP.persistence_diagram(A,DI.EdgeWeightedFiltration(max_dim=2,threshold=1))
    @test OP.provenance(limited).weighted_flag.source_vertex_indices==[1,2,3]
    @test OP.provenance(limited).essential_interpretation===:survives_threshold_cutoff
    @test occursin("through threshold 1.0",TamerOp.Advanced.visual_spec(limited;kind=:barcode).subtitle)
    spec=DI._filtration_spec(DI.EdgeWeightedFiltration(vertex_births=[-1.,0.],max_dim=2,threshold=3))
    @test DI.filtration_parameters(DI.to_filtration(spec))==spec.params
    @test DI.check_filtration_spec(spec;throw=true).valid
    graph=DT.GraphData(3,[(1,2),(2,3)];weights=[1.,2.])
    filter=DI.EdgeWeightedFiltration(vertex_births=[-2.,-1.,0.],max_dim=2)
    G=TamerOp.encode(graph,filter;stage=:graded_complex)
    @test G isa DT.GradedComplex
    @test TamerOp.encode(graph,filter;stage=:simplex_tree) isa DT.SimplexTreeMulti
    for p in (2,3)
        d=OP.persistence_diagram(graph,filter;field=CM.Fp(p),representatives=true)
        _a112_check_representatives(G,d,p)
        # The finite encoding retains the same H0 maps as the direct barcode.
        M=TamerOp.encode(graph,filter;stage=:module,degree=0,field=CM.Fp(p))
        levels=[-2.,-1.,0.,1.,2.]
        @test M.dims==[1,2,3,2,1]
        for i in eachindex(levels),j in i:length(levels)
            entries=map(x -> x.val,Matrix(TamerOp.Modules.map_leq(M,i,j)))
            @test _a112_rank(entries,p)==_a21_barcode_map_rank(d,0,levels[i],levels[j],:sublevel)
        end
    end
end

@testset "Rips optimization: distance validation across tiles" begin
    for n in (1,31,32,33,65)
        A=[Float64(abs(i-j)) for i in 1:n,j in 1:n]
        @test DI._validate_distance_matrix(A)==n
        @test DI._validate_distance_matrix(transpose(A))==n
        @test DI._validate_distance_matrix(view(A,:,:))==n
    end
    # Every entry, including both sides of all tile seams, must be checked.
    A=[Float64(abs(i-j)) for i in 1:65,j in 1:65]
    for j in 1:65,i in 1:65
        old=A[i,j];A[i,j]=NaN
        @test_throws ArgumentError DI._validate_distance_matrix(A)
        A[i,j]=old
    end
    for (i,j) in ((1,33),(32,33),(33,32),(33,65),(65,33),(64,65))
        old=A[i,j]
        for bad in (-1.0,Inf,old+1.0)
            A[i,j]=bad
            @test_throws ArgumentError DI._validate_distance_matrix(A)
        end
        A[i,j]=old
    end
    A[1,65]=A[65,1]=Inf
    @test DI._validate_distance_matrix(A)==65
    @test_throws ArgumentError DI.select_landmarks(A;count=2)
    @test DI._validate_distance_matrix([0.0 -1e-11;0.0 0.0])==2
    @test_throws ArgumentError DI._validate_distance_matrix([0.0 -1e-8;0.0 0.0])
    bigdist=big(2)^2000
    @test_throws ArgumentError DI._validate_distance_matrix([big(0) bigdist;bigdist big(0)])
end

@testset "Rips optimization: complete and sparse catalog order" begin
    rng=MersenneTwister(51005)
    for complete in (false,true), n in (1,2,6,9)
        edges=NTuple{2,Int}[];grades=Float64[]
        births=[-Float64(i%3) for i in 1:n]
        A=fill(Inf,n,n)
        for j in 2:n,i in 1:j-1
            (!complete && rand(rng,Bool)) && continue
            d=Float64(rand(rng,0:3));push!(edges,(i,j));push!(grades,d);A[i,j]=A[j,i]=d
        end
        g=OP._rips_graph((;n,edges,dists=grades,births),3)
        for j in 1:n,i in 1:n
            @test OP._rips_distance(g,i,j)==A[i,j]
        end
        for N in 1:min(n,4)
            expected=[]
            # Independent enumeration: all increasing tuples, no neighbor walk.
            for v in Iterators.product(ntuple(_->1:n,N)...)
                all(v[k]<v[k+1] for k in 1:N-1) || continue
                grade=maximum(vv->births[vv],v)
                for i in 1:N,j in i+1:N;grade=max(grade,A[v[i],v[j]]);end
                isfinite(grade) || continue
                id=1+sum(binomial(v[k]-1,k) for k in 1:N)
                push!(expected,(v,id,grade))
            end
            sort!(expected;by=s->(s[3],s[2]),rev=true)
            for cleared in (Set{Int}(),Set(2:3:30))
                actual=OP._rips_catalog(g,Val(N),cleared)
                @test [(s.vertices,s.index,s.grade) for s in actual]==[s for s in expected if !(s[2] in cleared)]
                if N==3
                    records=OP._rips_record_catalog(g,Val(N),cleared)
                    @test [(s.vertices,s.index,s.grade) for s in actual]==[(s.vertices,s.index,s.grade) for s in records]
                end
            end
        end
    end
end

@testset "Rips optimization: binary queue parity and ordering" begin
    rng=MersenneTwister(51006)
    K=CM.FpElem{2}
    for trial in 1:8
        ordinary=OP._RipsTerm{K}[]
        compact=OP._RipsBinaryQueue(80)
        coefficients=Dict{Int,Int}()
        grades=Float64.(rand(rng,-3:3,80))
        for batch in 1:10
            for _ in 1:100
                id=rand(rng,1:80);c=rand(rng,0:1)
                term=OP._RipsTerm(id,grades[id],K(c))
                OP._rips_heap_push!(ordinary,term);OP._rips_heap_push!(compact,term)
                coefficients[id]=mod(get(coefficients,id,0)+c,2)
            end
            # Partial drains followed by new additions also exercise re-entry
            # and coefficients that cancel before the corresponding pivot.
            for _ in 1:rand(rng,0:6)
                a=OP._rips_pop_pivot!(ordinary);b=OP._rips_pop_pivot!(compact)
                ids=sort!([i for (i,c) in coefficients if c==1];by=i->(grades[i],i))
                if isempty(ids)
                    @test a===b===nothing
                else
                    i=first(ids)
                    @test a.index==b.index==i
                    @test a.grade==b.grade==grades[i]
                    @test a.coefficient==b.coefficient==one(K)
                    coefficients[i]=0
                end
            end
        end
        ids=sort!([i for (i,c) in coefficients if c==1];by=i->(grades[i],i))
        for i in ids
            a=OP._rips_pop_pivot!(ordinary);b=OP._rips_pop_pivot!(compact)
            @test (a.index,a.grade,a.coefficient)==(b.index,b.grade,b.coefficient)==(i,grades[i],one(K))
        end
        @test OP._rips_pop_pivot!(ordinary)===nothing
        @test OP._rips_pop_pivot!(compact)===nothing
    end
end

@testset "Rips optimization: cone certificate preserves requested mathematics" begin
    # A nonmetric dissimilarity: vertex 1 cones off the entire filtration at 1.
    # The other edges appear at 3 and cannot change ordinary homology thereafter.
    A=fill(3.0,6,6)
    for i in 1:6;A[i,i]=0;end
    for i in 2:6;A[1,i]=A[i,1]=1;end
    for p in (2,3,101), q in (0,1,2), radius in (0.5,1.0,3.0,Inf)
        f=DI.RipsFiltration(max_dim=q+1;radius)
        D=OP.persistence_diagram(A,f;field=CM.Fp(p),max_homology_dim=q)
        E=OP.persistence_diagram(A,f;field=CM.Fp(p),max_homology_dim=q,method=:explicit)
        @test D.finite_by_dim==E.finite_by_dim
        @test D.essential_by_dim==E.essential_by_dim
        cert=OP.provenance(D).computation.terminal_radius
        if radius>=3
            @test cert.radius==1
            @test cert.apex==1
            @test OP.provenance(D).input_selection.retained_edges==5
        else
            @test cert===nothing
        end
    end
    # A graph-only request has top-dimensional essential cycles. It must not
    # inherit the cone certificate from a higher-dimensional flag completion.
    for p in (2,3,101)
        f=DI.RipsFiltration(max_dim=1)
        D=OP.persistence_diagram(A,f;field=CM.Fp(p))
        @test length(OP.essential_births(D;dim=1))==10
        @test OP.provenance(D).computation.terminal_radius===nothing
        retained=OP.persistence_diagram(A,DI.RipsFiltration(max_dim=2);
            field=CM.Fp(p),max_homology_dim=1,cocycles=true)
        @test OP.provenance(retained).computation.terminal_radius===nothing
    end
    # The certificate does not weaken the requested input's construction budget.
    f=DI.RipsFiltration(max_dim=2,construction=OPT.ConstructionOptions(budget=(max_simplices=12,)))
    @test_throws ArgumentError OP.persistence_diagram(A,f;max_homology_dim=1)
    A[1,6]=A[6,1]=Inf
    D=OP.persistence_diagram(A,DI.RipsFiltration(max_dim=3);max_homology_dim=2)
    @test OP.provenance(D).computation.terminal_radius===nothing
end

@testset "Rips optimization: apparent facets and compact exact order" begin
    rng=MersenneTwister(51007)
    for n in (4,7,10), trial in 1:3
        A=zeros(Float64,n,n)
        for j in 2:n,i in 1:j-1;A[i,j]=A[j,i]=rand(rng,0:3);end
        g=OP._rips_graph(DI._rips_persistence_graph(A,DI.RipsFiltration(max_dim=3),3),3)
        for N in 1:4
            columns=OP._rips_catalog(g,Val(N),Set{Int}())
            for s in columns
                @test OP._rips_vertices(g,s.index,Val(N))==s.vertices
                if N>1
                    # Independent face list, using direct matrix grades and a
                    # separately computed binomial index, checks the tie order.
                    faces=[]
                    for removed in 1:N
                        face=Tuple(s.vertices[j] for j in 1:N if j!=removed)
                        grade=maximum((A[u,v] for u in face for v in face);init=0.0)
                        id=1+sum(binomial(face[k]-1,k) for k in 1:N-1)
                        grade==s.grade && push!(faces,(id,face,removed))
                    end
                    expected=isempty(faces) ? nothing : maximum(faces)
                    found=OP._rips_youngest_facet(g,s.vertices,s.grade)
                    @test (found===nothing ? nothing : (first(found).index,first(found).vertices,last(found)))==expected
                end
            end
        end
    end
    # Packed ordering must retain negative grades, signed zero ordering,
    # adjacent Float64 values and the exact colex refinement of every tie.
    grades=[-floatmax(Float64),-1.0,-nextfloat(0.0),-0.0,0.0,nextfloat(0.0),1.0,nextfloat(1.0),floatmax(Float64)]
    records=[(g,i) for g in grades for i in (1,2,typemax(Int))]
    @test isequal(sort(records;by=x->OP._rips_catalog_key(x...)),sort(records))
    A=fill(-1.0,6,6)
    for i in 1:6;A[i,i]=-2;end
    for p in (2,3,101)
        f=DI.EdgeWeightedFiltration(max_dim=3)
        D=OP.persistence_diagram(A,f;field=CM.Fp(p),max_homology_dim=2,cocycles=true)
        E=OP.persistence_diagram(A,f;field=CM.Fp(p),max_homology_dim=2,method=:explicit)
        @test D.finite_by_dim==E.finite_by_dim
        @test D.essential_by_dim==E.essential_by_dim
        @test OP.provenance(D).reduction_stats.apparent_pairs>0
    end
end

@testset "Rips optimization: signed-zero apparent pairs preserve classes" begin
    # A square with a filled triangle glued to one edge has one persistent
    # H1 class. Mixing zero signs changes storage order, not this topology.
    edges=((1,2),(2,4),(4,5),(1,5),(3,4),(3,5))
    permutations=([1,2,3,4,5],[5,4,3,2,1],[3,1,5,2,4])
    for mask in 0:63, permutation in permutations, p in (2,3,101), keep in (false,true)
        A=fill(Inf,5,5)
        for i in 1:5;A[i,i]=-1.0;end
        for (k,(u,v)) in enumerate(edges)
            A[u,v]=A[v,u]=iszero(mask & (1<<(k-1))) ? -0.0 : 0.0
        end
        A=A[permutation,permutation]
        f=DI.EdgeWeightedFiltration(max_dim=3)
        D=OP.persistence_diagram(A,f;field=CM.Fp(p),max_homology_dim=2,cocycles=keep)
        E=OP.persistence_diagram(A,f;field=CM.Fp(p),max_homology_dim=2,method=:explicit)
        @test length(OP.finite_intervals(D;dim=0))==4
        @test OP.essential_births(D;dim=0)==[-1.0]
        @test isempty(OP.finite_intervals(D;dim=1))
        @test OP.essential_births(D;dim=1)==OP.essential_births(E;dim=1)==[0.0]
        @test isempty(OP.finite_intervals(D;dim=2))
        @test isempty(OP.essential_births(D;dim=2))
        if keep
            inverse=invperm(permutation)
            for scale in (0.0,1.0)
                w=OP.persistence_cocycle(D;dim=1,kind=:essential,index=1,scale)
                @test w.available
                terms=Dict(Tuple(v)=>c for (v,c) in zip(w.source_vertices,w.cochain.coefficients))
                edge_value(u,v)=begin
                    a,b=inverse[u],inverse[v]
                    get(terms,minmax(a,b),0)*(a<b ? 1 : -1)
                end
                # Direct evaluation on the triangle boundary and square cycle
                # certifies closure and a nonzero cohomology class over Fp.
                @test mod(edge_value(3,4)+edge_value(4,5)-edge_value(3,5),p)==0
                @test mod(edge_value(1,2)+edge_value(2,4)+edge_value(4,5)-edge_value(1,5),p)!=0
            end
        end
    end
end

@testset "Rips optimization: signed-zero sphere cocycles" begin
    rng=MersenneTwister(51008)
    for trial in 1:8, p in (2,3,101), keep in (false,true)
        A=fill(-1.0,6,6)
        for j in 2:6,i in 1:j-1
            A[i,j]=A[j,i]=cld(i,2)==cld(j,2) ? 1.0 : rand(rng,Bool) ? -0.0 : 0.0
        end
        # Before 1 this is the join of three two-point sets, an octahedral S2.
        D=OP.persistence_diagram(A,DI.EdgeWeightedFiltration(max_dim=3);
            field=CM.Fp(p),max_homology_dim=2,cocycles=keep)
        @test isempty(OP.finite_intervals(D;dim=1))
        @test isempty(OP.essential_births(D;dim=1))
        @test OP.finite_intervals(D;dim=2)==[(0.0,1.0)]
        @test isempty(OP.essential_births(D;dim=2))
        if keep
            w=OP.persistence_cocycle(D;dim=2,kind=:finite,index=1,scale=0.0)
            @test w.available
            terms=Dict(Tuple(v)=>c for (v,c) in zip(w.source_vertices,w.cochain.coefficients))
            evaluation=sum((-1)^(iseven(a)+iseven(b)+iseven(c))*get(terms,(a,b,c),0)
                for a in 1:2,b in 3:4,c in 5:6)
            @test mod(evaluation,p)!=0
        end
    end
end

@testset "Rips optimization: mixed-zero random flag parity" begin
    # Different tie refinements in adjacent degrees must not change clearing.
    canonical_grade(x)=iszero(x) ? 0.0 : x
    finite_bars(D)=[sort!([(canonical_grade(b),canonical_grade(d)) for (b,d) in bars]) for bars in D.finite_by_dim]
    essential_bars(D)=[sort!(canonical_grade.(bars)) for bars in D.essential_by_dim]
    rng=MersenneTwister(51009)
    for n in 4:8, trial in 1:12
        A=fill(-1.0,n,n)
        for j in 2:n,i in 1:j-1
            A[i,j]=A[j,i]=rand(rng,(-0.0,0.0,1.0,2.0,Inf))
        end
        for p in (2,3,101), keep in (false,true)
            f=DI.EdgeWeightedFiltration(max_dim=3)
            D=OP.persistence_diagram(A,f;field=CM.Fp(p),max_homology_dim=2,cocycles=keep)
            E=OP.persistence_diagram(A,f;field=CM.Fp(p),max_homology_dim=2,method=:explicit)
            @test finite_bars(D)==finite_bars(E)
            @test essential_bars(D)==essential_bars(E)
        end
    end
end


@testset "Rips optimization: priority queue agrees with modular dictionary" begin
    rng=MersenneTwister(81005)
    for prime in (2,3,101), compact in (false,true)
        compact && prime!=2 && continue
        K=CM.coeff_type(CM.Fp(prime))
        heap=compact ? OP._RipsBinaryQueue(80) : OP._RipsTerm{K}[]
        expected=Dict{Int,K}()
        grades=rand(rng,(-0.0,0.0,0.5,1.0,10.0,floatmax(Float64)),200)
        function pop_reference!()
            if isempty(expected)
                @test OP._rips_pop_pivot!(heap)===nothing
            else
                id=first(sort!(collect(keys(expected));lt=(a,b)->
                    grades[a]<grades[b] || (grades[a]==grades[b] && a<b)))
                actual=OP._rips_pop_pivot!(heap)
                @test actual!==nothing && actual.index==id && actual.coefficient==expected[id]
                delete!(expected,id)
            end
        end
        for k in 1:2000
            if isempty(heap) || rand(rng)<0.7
                id=rand(rng,1:200);c=K(rand(rng,0:prime-1))
                OP._rips_heap_push!(heap,OP._RipsTerm(id,grades[id],c))
                v=get(expected,id,zero(K))+c
                iszero(v) ? delete!(expected,id) : (expected[id]=v)
            else
                pop_reference!()
            end
        end
        while !isempty(heap);pop_reference!();end
        @test isempty(expected)
    end
end

@testset "Rips optimization: shuffled graph rows retain exact distances" begin
    rng=MersenneTwister(81006)
    for n in (1,2,9,33), complete in (false,true)
        A=fill(Inf,n,n);edges=NTuple{2,Int}[];grades=Float64[]
        for j in 2:n,i in 1:j-1
            !complete && rand(rng,Bool) && continue
            grade=rand(rng,(-0.0,0.0,0.5,2.0,4.0))
            push!(edges,rand(rng,Bool) ? (i,j) : (j,i));push!(grades,grade)
            A[i,j]=A[j,i]=grade
        end
        order=randperm(rng,length(edges))
        g=OP._rips_graph((n=n,edges=edges[order],dists=grades[order],births=fill(-1.0,n)),3)
        for i in 1:n
            @test g.neighbors[i]==[j for j in 1:n if isfinite(A[i,j])]
            @test isequal(g.distances[i],[A[i,j] for j in g.neighbors[i]])
            for j in 1:n
                @test isequal(OP._rips_distance(g,i,j),A[i,j])
            end
        end
    end
end

@testset "Rips optimization: landmark validation keeps complete input contract" begin
    for n in (1,3,31,32,33,65)
        A=[Float64(abs(i-j)) for i in 1:n,j in 1:n]
        expected=DI.select_landmarks(A;count=min(n,3),retain_distances=true)
        for input in (Float32.(A),BigFloat.(A),Rational{BigInt}.(A),transpose(A),view(A,:,:))
            actual=DI.select_landmarks(input;count=min(n,3),retain_distances=true)
            @test DI.landmark_indices(actual)==DI.landmark_indices(expected)
            @test DI.landmark_distances(actual)==DI.landmark_distances(expected)
            @test actual.insertion_radii==expected.insertion_radii
            @test actual.nearest_distances==expected.nearest_distances
            @test actual.nearest_indices==expected.nearest_indices
        end
    end
    A=[Float64(abs(i-j)) for i in 1:65,j in 1:65]
    for (i,j) in ((1,33),(32,33),(33,65),(64,65)), value in (Inf,-Inf,NaN,-1.0)
        B=copy(A);B[i,j]=B[j,i]=value
        # Even unselected points must satisfy the complete-distance contract.
        @test_throws ArgumentError DI.select_landmarks(B;indices=[1])
        if value==Inf
            @test DI._validate_distance_matrix(B)==65
        end
    end
    for value in (NaN,Inf,1.0)
        B=copy(A);B[33,33]=value
        @test_throws ArgumentError DI.select_landmarks(B;count=1)
    end
    B=copy(A);B[33,65]+=1
    @test_throws ArgumentError DI.select_landmarks(B;count=1)
    @test DI.landmark_distances(DI.select_landmarks([0. -1e-11;0. 0.];count=2,retain_distances=true))==zeros(2,2)
    huge=big(2)^2000
    @test_throws ArgumentError DI.select_landmarks([big(0) huge;huge big(0)];count=1)
end

@testset "Rips optimization: shortcut and complete signed coboundaries agree" begin
    rng=MersenneTwister(81007)
    for prime in (2,3,101), complete in (false,true), trial in 1:4
        K=CM.coeff_type(CM.Fp(prime));n=6
        edges=NTuple{2,Int}[];grades=Float64[];A=fill(Inf,n,n)
        for j in 2:n,i in 1:j-1
            !complete && rand(rng,Bool) && continue
            grade=rand(rng,(-0.0,0.0,1.0,3.0))
            push!(edges,(i,j));push!(grades,grade);A[i,j]=A[j,i]=grade
        end
        g=OP._rips_graph((;n,edges,dists=grades,births=fill(-1.0,n)),3)
        for N in 1:3,s in OP._rips_catalog(g,Val(N),Set{Int}())
            reference=Tuple{Int,Float64,K}[]
            for w in 1:n
                w in s.vertices && continue
                grade=s.grade
                for v in s.vertices;grade=max(grade,A[v,w]);end
                isfinite(grade) || continue
                vertices=sort!([s.vertices...,w]);pos=findfirst(==(w),vertices)
                id=1+sum(binomial(vertices[k]-1,k) for k in eachindex(vertices))
                push!(reference,(id,grade,isodd(pos) ? one(K) : -one(K)))
            end
            sort!(reference;lt=(a,b)->a[2]<b[2]||(a[2]==b[2]&&a[1]<b[1]))
            for claimed in (false,true)
                pivots=Dict{Int,Vector{Pair{Int,K}}}()
                if claimed
                    for (id,_,_) in reference;pivots[id]=Pair{Int,K}[];end
                end
                heap=OP._RipsTerm{K}[];stats=OP._RipsReductionStats()
                actual=OP._rips_coboundary!(heap,g,s,one(K),pivots,stats;shortcut=true)
                if !claimed && !isempty(reference) && first(reference)[2]==s.grade
                    @test actual!==nothing && isequal((actual.index,actual.grade,actual.coefficient),first(reference))
                    @test isempty(heap)
                else
                    @test actual===nothing
                    collected=Tuple{Int,Float64,K}[]
                    while !isempty(heap)
                        t=OP._rips_pop_pivot!(heap);push!(collected,(t.index,t.grade,t.coefficient))
                    end
                    @test isequal(collected,reference)
                end
            end
        end
    end
end

# Enumerate partial bijections directly, with unused points sent to the diagonal.
# This oracle shares neither a cost matrix nor an assignment solver with production.
function _a116_matching_oracle(A,B,p,q)
    ground(a,b) = q==Inf ? max(abs(a[1]-b[1]),abs(a[2]-b[2])) :
        (abs(a[1]-b[1])^q+abs(a[2]-b[2])^q)^(1/q)
    diagonal(a) = (a[2]-a[1]) * (q==Inf ? 0.5 : 2.0^(1/q-1))
    best=Ref(Inf)
    function visit(i,used,costs)
        if i>length(A)
            allcosts=vcat(costs,[diagonal(B[j]) for j in eachindex(B) if !(j in used)])
            value=p==Inf ? maximum(allcosts;init=0.0) : sum(x->x^p,allcosts;init=0.0)^(1/p)
            best[]=min(best[],value)
            return
        end
        visit(i+1,used,vcat(costs,diagonal(A[i])))
        for j in eachindex(B)
            j in used || visit(i+1,union(used,[j]),vcat(costs,ground(A[i],B[j])))
        end
    end
    visit(1,Int[],Float64[])
    return best[]
end

@testset "A116 diagram distances independent matching oracle" begin
    rng=MersenneTwister(116)
    for trial in 1:24
        A=[(b,b+rand(rng,1:5)) for b in rand(rng,0:4,rand(rng,0:3))]
        B=[(b,b+rand(rng,1:5)) for b in rand(rng,0:4,rand(rng,0:3))]
        for order in (:sublevel,:superlevel)
            sign=order===:sublevel ? 1 : -1
            da=OP.PersistenceDiagram([[(sign*b,sign*d) for (b,d) in A]],[Int[]];order)
            db=OP.PersistenceDiagram([[(sign*b,sign*d) for (b,d) in B]],[Int[]];order,field=CM.Fp(3))
            @test OP.bottleneck_distance(da,db;dim=0) == _a116_matching_oracle(A,B,Inf,Inf)
            for p in (1,2,3),q in (1,2,Inf),backend in (:hungarian,:auction,:auto)
                @test isapprox(OP.wasserstein_distance(da,db;dim=0,p,q,backend),_a116_matching_oracle(A,B,p,q);atol=1e-7)
            end
            @test OP.wasserstein_distance(da,db;dim=0,p=Inf) == OP.bottleneck_distance(da,db;dim=0)
            @test OP.analysis_barcode(da;dim=0) == A
        end
    end
    for n in (16,25), backend in (:auto,:auction,:hungarian)
        a=OP.PersistenceDiagram([fill((0,4),n)],[Int[]])
        b=OP.PersistenceDiagram([fill((1,5),n)],[Int[]])
        @test isapprox(OP.wasserstein_distance(a,b;dim=0,backend),sqrt(n);atol=1e-10)
    end
    # Essential bars pair by sorted births, never with finite bars/diagonal.
    a=OP.PersistenceDiagram([[(0,4),(0,4)]],[[6,0]])
    b=OP.PersistenceDiagram([[(1,5)]],[[1,9]])
    for p in (1,2,3),q in (1,2,Inf)
        finite=_a116_matching_oracle([(0,4),(0,4)],[(1,5)],p,q)
        @test isapprox(OP.wasserstein_distance(a,b;dim=0,p,q),(finite^p+1+3^p)^(1/p))
    end
    @test OP.bottleneck_distance(a,b;dim=0)==3
    match=OP.bottleneck_matching(a,b;dim=0)
    @test match.distance==3
    @test length(match.a_to_b)==4 && length(match.b_to_a)==3
    @test match.a_to_b[3:4]==[3,2]
    @test match.points_a[1:2]==[(0,4),(0,4)]
    other=OP.PersistenceDiagram([Tuple{Int,Int}[]],[[0]])
    @test OP.wasserstein_distance(a,other;dim=0)==Inf
    @test OP.bottleneck_distance(a,other;dim=0)==Inf
    @test OP.wasserstein_distance(other,other;dim=3)==0
    @test_throws ArgumentError OP.bottleneck_distance(a,OP.PersistenceDiagram([[(4,0)]],[Int[]];order=:superlevel);dim=0)
    for kwargs in ((p=0,), (q=3,), (p=Inf,q=2,), (backend=:unknown,))
        @test_throws ArgumentError OP.wasserstein_distance(a,b;dim=0,kwargs...)
    end
end

@testset "A116 features independent formulas and policies" begin
    bars=[(0,4),(0,4),(1,3)]
    diagram=OP.PersistenceDiagram([bars,[(2,3)]],[Int[],[0]])
    tg=collect(0.0:0.5:4.0)
    tents=[max(0,min(t-b,d-t)) for (b,d) in bars,t in tg]
    landscape=OP.persistence_landscape(diagram;dim=0,tgrid=tg,kmax=4)
    expected=vcat(reduce(hcat,[sort(tents[:,i];rev=true) for i in axes(tents,2)]),zeros(1,length(tg)))
    @test landscape.values==expected
    @test OP.persistence_silhouette(diagram;dim=0,tgrid=tg)==vec(sum([4,4,2].*tents;dims=1))./10
    @test OP.persistence_silhouette(diagram;dim=0,tgrid=tg,weighting=:none)==vec(sum(tents;dims=1))./3
    @test isapprox(OP.barcode_entropy(diagram;dim=0,normalize=false),-sum(w*log(w) for w in (0.4,0.4,0.2)))
    summary=OP.barcode_summary(diagram;dim=0)
    @test summary.n_intervals==3 && summary.total_persistence==10 && summary.max_persistence==4
    @test summary.mean_persistence==10/3 && summary.l2_persistence==6
    @test isapprox(summary.entropy,-sum(w*log(w) for w in (0.4,0.4,0.2))/log(3))
    xg=[0.,1.,2.]; yg=[0.,2.,4.]; sigma=0.7
    for coords in (:birth_persistence,:birth_death,:midlife_persistence),threads in (false,true),diff in (false,true)
        centers=[coords===:birth_persistence ? (b,d-b) : coords===:birth_death ? (b,d) : ((b+d)/2,d-b) for (b,d) in bars]
        expected=[sum((d-b)*exp(-((x-cx)^2+(y-cy)^2)/(2sigma^2)) for ((b,d),(cx,cy)) in zip(bars,centers)) for y in yg,x in xg]
        image=OP.persistence_image(diagram;dim=0,xgrid=xg,ygrid=yg,sigma,coords,threads,differentiable=diff)
        @test isapprox(image.values,expected)
    end
    # Degree isolation, repeated members, explicit essential handling and reflection.
    @test_throws ArgumentError OP.persistence_landscape(diagram;dim=1)
    @test OP.analysis_barcode(diagram;dim=1,essential=:keep)==[(2,3),(0,Inf)]
    @test OP.analysis_barcode(diagram;dim=1,essential=:cap,essential_cap=1)==[(2,3),(0,1)]
    super=OP.PersistenceDiagram([[(10,6),(10,6),(9,7)]],[[10]];order=:superlevel)
    @test OP.analysis_barcode(super;dim=0,essential=:drop,origin=10)==bars
    @test OP.analysis_barcode(super;dim=0,essential=:cap,essential_cap=5,origin=10)==vcat(bars,[(0,5)])
    @test OP.persistence_landscape(super;dim=0,essential=:drop,origin=10,tgrid=tg,kmax=4).values==landscape.values
    @test_throws ArgumentError OP.analysis_barcode(super;dim=0,essential=:cap,essential_cap=11)
    for kwargs in ((essential=:unknown,), (essential=:cap,), (essential_cap=8,), (scale=0,), (origin=Inf,), (dim=-1,), (dim=true,))
        @test_throws ArgumentError OP.analysis_barcode(diagram;dim=0,kwargs...)
    end
    for grid in ([0.], [1.,0.], [0.,0.], [0.,Inf], [NaN,1.])
        @test_throws ArgumentError OP.persistence_landscape(diagram;dim=0,tgrid=grid)
    end
    @test_throws ArgumentError OP.persistence_image(diagram;dim=0,xgrid=[1,1])
    @test_throws ArgumentError OP.persistence_image(diagram;dim=0,sigma=Inf)
    @test_throws ArgumentError OP.persistence_silhouette(diagram;dim=1,tgrid=tg,essential=:keep)
    @test_throws ArgumentError OP.barcode_entropy(diagram;dim=0,base=1)
    empty=OP.PersistenceDiagram([Tuple{Int,Int}[]],[Int[]])
    @test OP.barcode_summary(empty;dim=0).n_intervals==0
    @test OP.barcode_entropy(empty;dim=0)==0
    @test all(iszero,OP.persistence_landscape(empty;dim=0).values)
    @test all(iszero,OP.persistence_silhouette(empty;dim=0,tgrid=tg))
    @test all(iszero,OP.persistence_image(empty;dim=0).values)
end

@testset "A116 exact coordinates and extreme numerical scales" begin
    huge=big(2)^1100
    a=OP.PersistenceDiagram([[(huge,huge+4)]],[BigInt[]])
    b=OP.PersistenceDiagram([[(huge+1,huge+5)]],[BigInt[]])
    @test OP.bottleneck_distance(a,b;dim=0)==1
    @test OP.wasserstein_distance(a,b;dim=0)==1
    @test OP.analysis_barcode(a;dim=0,origin=huge,scale=2)==[(0,2)]
    @test OP.barcode_summary(a;dim=0).total_persistence==4
    @test_throws ArgumentError OP.persistence_landscape(a;dim=0)
    @test OP.persistence_landscape(a;dim=0,origin=huge,tgrid=[0.,2.,4.],kmax=1).values==[0. 2. 0.]
    tiny=OP.PersistenceDiagram([[(big(0)//1,1//huge)]],[Rational{BigInt}[]])
    zero=OP.PersistenceDiagram([Tuple{Rational{BigInt},Rational{BigInt}}[]],[Rational{BigInt}[]])
    @test_throws ArgumentError OP.wasserstein_distance(tiny,zero;dim=0)
    @test OP.wasserstein_distance(tiny,zero;dim=0,scale=1//huge)==0.5
    @test OP.analysis_barcode(OP.PersistenceDiagram([[(-0.0,0.1)]],[Float64[]]);dim=0)[1][2]==Rational{BigInt}(0.1)
end

@testset "A116 exact saved results and retained source data" begin
    mktempdir() do dir
        path=joinpath(dir,"diagram.json")
        root=sqrt(TamerOp.AlgebraicReal(2))
        for T in (Int,BigInt,Rational{Int},Rational{BigInt},Float16,Float32,Float64,BigFloat,TamerOp.AlgebraicReal),order in (:sublevel,:superlevel)
            sign=order===:sublevel ? 1 : -1
            b=T===TamerOp.AlgebraicReal ? root : T(-0.0)
            d=b+T(2sign)
            data=OP.PersistenceDiagram([[(b,d),(b,d)],Tuple{T,T}[]],[[b],T[]];order,field=CM.Fp(101),
                meta=(source=DT.PointCloud{Float64},note="repeated",nested=(x=missing,y=[1,2],z=Dict("a"=>1)),requested_max_homology_dim=1))
            for profile in (:compact,:debug)
                SER.save_persistence_diagram_json(path,data;profile)
                loaded=SER.load_persistence_diagram_json(path)
                @test isequal(OP.persistence_diagram_data(data),OP.persistence_diagram_data(loaded))
                @test isequal(RES.provenance(data),RES.provenance(loaded))
                @test SER.check_persistence_diagram_json(path).valid
                @test SER.artifact_kind(SER.persistence_diagram_json_summary(path))=="PersistenceDiagram"
                @test OP.bottleneck_distance(data,loaded;dim=0)==0
                @test_throws ArgumentError OP.analysis_barcode(loaded;dim=2)
            end
        end
        empty=OP.PersistenceDiagram(Vector{Tuple{Int,Int}}[],Vector{Int}[])
        SER.save_persistence_diagram_json(path,empty)
        @test isequal(OP.persistence_diagram_data(SER.load_persistence_diagram_json(path)),OP.persistence_diagram_data(empty))
        for order in (:sublevel,:superlevel),prime in (2,3,101)
            G,_=_a112_simplicial_fixture(MersenneTwister(116),order)
            original=OP.persistence_diagram(G;order,field=CM.Fp(prime),representatives=true,cocycles=true)
            SER.save_persistence_diagram_json(path,original)
            loaded=SER.load_persistence_diagram_json(path)
            @test isequal(OP.persistence_diagram_data(original),OP.persistence_diagram_data(loaded))
            _a112_check_representatives(G,loaded,prime)
            _a117_check_cocycles(G,loaded,prime)
        end
        cloud=DT.PointCloud([[0.,0.],[1.,0.],[1.,1.],[0.,1.],[9.,9.]])
        for filtration in (DI.RipsFiltration(max_dim=2),DI.LandmarkRipsFiltration(max_dim=2,landmarks=[4,1,3,2]))
            original=OP.persistence_diagram(cloud,filtration;field=CM.Fp(3),cocycles=true,max_homology_dim=1)
            SER.save_persistence_diagram_json(path,original)
            loaded=SER.load_persistence_diagram_json(path)
            @test OP.finite_intervals(loaded;dim=1)==OP.finite_intervals(original;dim=1)
            @test OP.persistence_cocycle(loaded;dim=1,scale=1)==OP.persistence_cocycle(original;dim=1,scale=1)
            @test OP.describe(loaded)==OP.describe(original)
            if filtration isa DI.LandmarkRipsFiltration
                @test DI.describe(OP.landmark_selection(loaded))==DI.describe(OP.landmark_selection(original))
                @test isequal(OP.landmark_selection(loaded),OP.landmark_selection(original))
                @test hash(OP.landmark_selection(loaded))==hash(OP.landmark_selection(original))
            end
        end
        # Save must reject unsupported metadata before overwriting the destination.
        write(path,"keep this")
        unsupported=OP.PersistenceDiagram([[(0,1)]],[Int[]];meta=(callback=identity,))
        @test_throws ArgumentError SER.save_persistence_diagram_json(path,unsupported)
        @test read(path,String)=="keep this"
    end
end

@testset "A116 strict saved-result rejection and root bindings" begin
    mktempdir() do dir
        path=joinpath(dir,"diagram.json")
        original=OP.PersistenceDiagram([[(0,2)]],[[0]])
        SER.save_persistence_diagram_json(path,original)
        source=read(path,String)
        for (key,value) in (("schema_version",2),("schema_version",true),("kind","Other"),
                ("field_characteristic",3),("field_characteristic",2.5),("finite_counts",[true]),
                ("essential_counts",[0]),("order","superlevel"),("extra",1),
                ("representatives_available",0),("cocycles_available",true))
            obj=JSON3.read(source,Dict{String,Any});obj[key]=value
            write(path,JSON3.write(obj))
            @test_throws ArgumentError SER.load_persistence_diagram_json(path)
            @test !SER.check_persistence_diagram_json(path).valid
            @test_throws ArgumentError SER.check_persistence_diagram_json(path;throw=true)
        end
        obj=JSON3.read(source,Dict{String,Any});obj["data"]["tag"]="eval"
        write(path,JSON3.write(obj))
        @test_throws ArgumentError SER.load_persistence_diagram_json(path)
        # Mutate supported tagged data after saving, so validation cannot be
        # bypassed by going through an otherwise consistent cheap header.
        G=DT.GradedComplex([[10,11],[12]],[sparse(reshape([-1,1],2,1))],[(0,),(0,),(2,)])
        retained=OP.persistence_diagram(G;field=CM.Fp(3),representatives=true,cocycles=true)
        SER.save_persistence_diagram_json(path,retained)
        saved=read(path,String)
        named(o,key)=o["values"][findfirst(==(key),o["names"])]
        for part in ("representatives","cocycles"), bad in ("0","3","-1")
            obj=JSON3.read(saved,Dict{String,Any})
            storage=named(obj["data"],part)
            member=named(storage,"finite")["values"][1]["values"][1]
            chain=named(member,part=="representatives" ? "cycle" : "cochain")
            named(chain,"coefficients")["values"][1]["value"]=bad
            write(path,JSON3.write(obj))
            @test_throws ArgumentError SER.load_persistence_diagram_json(path)
        end
        obj=JSON3.read(saved,Dict{String,Any})
        named(obj["data"],"grade_type")["name"]="Base.exit"
        write(path,JSON3.write(obj))
        @test_throws ArgumentError SER.load_persistence_diagram_json(path)
        for replacement in (Float64,AbstractFloat)
            data=OP.persistence_diagram_data(original)
            @test_throws ArgumentError OP.persistence_diagram_from_data(merge(data,(grade_type=replacement,)))
        end
    end
    for name in (:analysis_barcode,:bottleneck_distance,:bottleneck_matching,:wasserstein_distance,
                 :persistence_landscape,:persistence_image,:persistence_silhouette,:barcode_entropy,:barcode_summary)
        @test getfield(TamerOp,name)===getfield(OP,name)
    end
    for name in (:save_persistence_diagram_json,:load_persistence_diagram_json)
        @test getfield(TamerOp,name)===getfield(SER,name)
    end
end

@testset "A116 novice tutorial executes from public API" begin
    include(joinpath(@__DIR__,"..","docs","build_scripts","check_ordinary_analysis.jl"))
    @test OrdinaryAnalysisNotebook.check() == 15
end


@testset "A116 saved BigFloat precision and exact rational analysis" begin
    mktempdir() do dir
        path=joinpath(dir,"precise.json")
        for bits in (80,256,377,512)
            original,expected=setprecision(BigFloat,bits) do
                x=BigFloat(1)/3
                d=OP.PersistenceDiagram([[(-BigFloat(0),x)]],[[x]];meta=(value=x,))
                d,Rational{BigInt}(x)
            end
            SER.save_persistence_diagram_json(path,original)
            loaded=SER.load_persistence_diagram_json(path)
            @test isequal(OP.persistence_diagram_data(original),OP.persistence_diagram_data(loaded))
            @test precision(OP.finite_intervals(loaded;dim=0)[1][2])==bits
            @test signbit(OP.finite_intervals(loaded;dim=0)[1][1])
            @test OP.analysis_barcode(loaded;dim=0,essential=:drop)==[(0,expected)]
            x=OP.essential_births(original;dim=0)[1]
            @test OP.analysis_barcode(loaded;dim=0,essential=:drop,scale=x)==[(0,1)]
            @test OP.analysis_barcode(loaded;dim=0,essential=:drop,origin=x)==[(-expected,0)]
        end
        original=setprecision(BigFloat,377) do
            x=BigFloat(1)/3
            G=DT.GradedComplex([[1,2],[3]],[sparse(reshape([-1,1],2,1))],[(x,),(x,),(x+1,)])
            OP.persistence_diagram(G;field=CM.Fp(3),representatives=true,cocycles=true)
        end
        SER.save_persistence_diagram_json(path,original)
        @test isequal(OP.persistence_diagram_data(original),OP.persistence_diagram_data(SER.load_persistence_diagram_json(path)))
        huge=big(2)^1100
        integer=OP.PersistenceDiagram([[(huge,huge+1)]],[BigInt[]])
        SER.save_persistence_diagram_json(path,integer)
        @test isequal(OP.persistence_diagram_data(integer),OP.persistence_diagram_data(SER.load_persistence_diagram_json(path)))
    end
end

@testset "Rips followup: greedy landmarks match direct distance minima" begin
    rng=MersenneTwister(101005)
    for n in (1,2,7,17,33), trial in 1:4
        A=zeros(Float64,n,n)
        for j in 2:n,i in 1:j-1
            A[i,j]=A[j,i]=rand(rng,(0.0,0.0,1.0,2.0,7.0))
        end
        for input in (A,Float32.(A),view(A,:,:)), m in unique((1,min(3,n),n))
            # Recompute all distances to selected points, independently of the
            # incremental nearest-distance update used by the implementation.
            ids=Int[];radii=Float64[]
            for k in 1:m
                available=setdiff(1:n,ids)
                distances=[isempty(ids) ? Inf : minimum(A[v,i] for v in ids) for i in available]
                best=available[argmax(distances)]
                push!(radii,maximum(distances));push!(ids,best)
            end
            table=A[ids,:]
            nearest=[minimum(table[:,i]) for i in 1:n]
            assignment=[ids[argmin(table[:,i])] for i in 1:n]
            for retain in (false,true)
                s=DI.select_landmarks(input;count=m,retain_distances=retain)
                @test s.indices==ids
                @test s.insertion_radii==radii
                @test s.nearest_distances==nearest
                @test s.nearest_indices==assignment
                @test DI.covering_radius(s)==maximum(nearest)
                @test retain ? DI.landmark_distances(s)==table : s.distances===nothing
            end
            fixed=reverse(ids)
            s=DI.select_landmarks(input;indices=fixed,retain_distances=true)
            @test s.indices==fixed
            @test DI.landmark_distances(s)==A[fixed,:]
            @test s.nearest_indices==[fixed[argmin(A[fixed,i])] for i in 1:n]
        end
    end
    points=DI.PointCloud([[0.,0.],[0.,0.],[3.,0.],[0.,4.],[3.,4.]])
    A=[sqrt(sum((points.points[i].-points.points[j]).^2)) for i in 1:5,j in 1:5]
    a=DI.select_landmarks(points;count=5,retain_distances=true)
    b=DI.select_landmarks(A;count=5,retain_distances=true)
    @test a.indices==b.indices
    @test a.insertion_radii==b.insertion_radii
    @test a.nearest_indices==b.nearest_indices
    @test DI.landmark_distances(a)==DI.landmark_distances(b)
end

@testset "Rips followup: native-float validation preserves generic errors" begin
    for T in (Float32,Float64), n in (1,2,31,32,33,65), require_finite in (false,true)
        A=T[abs(i-j) for i in 1:n,j in 1:n]
        @test DI._validate_distance_matrix(A;require_finite)==n
        @test DI._validate_distance_matrix(view(A,:,:);require_finite)==n
        # A view selects the generic checked traversal. Match its error type
        # and diagnostic, including malformed entries across tile boundaries.
        inputs=[A]
        for i in unique((1,n)), value in (T(NaN),T(Inf),T(-Inf),T(1))
            B=copy(A);B[i,i]=value;push!(inputs,B)
        end
        if n>1
            for (i,j) in unique(((1,n),(max(1,n-1),n))), value in (T(NaN),T(Inf),T(-Inf),T(-1),T(-1e-11),T(0))
                B=copy(A);B[i,j]=B[j,i]=value;push!(inputs,B)
                C=copy(A);C[i,j]=value;push!(inputs,C)
            end
        end
        outcome(B)=try
            (:ok,DI._validate_distance_matrix(B;require_finite))
        catch e
            (typeof(e),sprint(showerror,e))
        end
        for B in inputs
            @test outcome(B)==outcome(view(B,:,:))
        end
    end
end

@testset "Rips followup: terminal neighbor filtering preserves every clique" begin
    rng=MersenneTwister(101006)
    for n in (4,7,11), trial in 1:4
        A=zeros(Float64,n,n)
        for j in 2:n,i in 1:j-1
            A[i,j]=A[j,i]=rand(rng,(-0.0,0.0,1.0,2.0,7.0))
        end
        payload=DI._rips_persistence_graph(A,DI.RipsFiltration(max_dim=3),3)
        original=OP._rips_graph(payload,3)
        for radius in (0.0,1.0,2.0,7.0)
            neighbors=[[v for v in original.neighbors[u] if A[u,v]<=radius] for u in 1:n]
            g=OP._RipsGraph{true}(original.neighbors,original.distances,original.edges,
                original.grades,original.binomial,original.births,radius,neighbors)
            # Enumerate combinations and read raw distances directly. Exact
            # signed-zero grades and colex ordering must survive pruning.
            for N in 2:3
                expected=[]
                combinations=N==2 ? [(i,j) for j in 2:n for i in 1:j-1] :
                    [(i,j,k) for k in 3:n for j in 2:k-1 for i in 1:j-1]
                for vertices in combinations
                    all(A[u,v]<=radius for u in vertices for v in vertices if u<v) || continue
                    grade=original.births[first(vertices)]
                    for i in 1:N,j in i+1:N;grade=max(grade,A[vertices[i],vertices[j]]);end
                    id=1+sum(binomial(vertices[k]-1,k) for k in 1:N)
                    push!(expected,(grade,id,vertices))
                end
                sort!(expected;by=x->(x[1],x[2]),rev=true)
                for clear in (Set{Int}(),Set(x[2] for x in expected[1:2:end]))
                    columns=OP._rips_catalog(g,Val(N),clear)
                    actual=[(x.grade,x.index,x.vertices) for x in columns]
                    @test isequal(actual,[x for x in expected if !(x[2] in clear)])
                end
            end
        end
        for radius in (0.0,1.0,2.0,7.0), prime in (2,3,101)
            neighbors=[[v for v in original.neighbors[u] if A[u,v]<=radius] for u in 1:n]
            g=OP._RipsGraph{true}(original.neighbors,original.distances,original.edges,
                original.grades,original.binomial,original.births,radius,neighbors)
            K=CM.coeff_type(CM.Fp(prime))
            for simplex in OP._rips_catalog(g,Val(2),Set{Int}())
                expected=Tuple{Int,Float64,K}[]
                for w in 1:n
                    w in simplex.vertices && continue
                    all(A[v,w]<=radius for v in simplex.vertices) || continue
                    vertices=sort!([simplex.vertices...,w]);position=findfirst(==(w),vertices)
                    grade=max(simplex.grade,maximum(A[v,w] for v in simplex.vertices))
                    id=1+sum(binomial(vertices[k]-1,k) for k in eachindex(vertices))
                    push!(expected,(id,grade,isodd(position) ? one(K) : -one(K)))
                end
                sort!(expected;lt=(a,b)->a[2]<b[2] || (a[2]==b[2] && a[1]<b[1]))
                heap=OP._RipsTerm{K}[];stats=OP._RipsReductionStats()
                @test OP._rips_coboundary!(heap,g,simplex,one(K),Dict{Int,Int}(),stats)===nothing
                actual=Tuple{Int,Float64,K}[]
                while !isempty(heap)
                    entry=OP._rips_pop_pivot!(heap)
                    push!(actual,(entry.index,entry.grade,entry.coefficient))
                end
                @test isequal(actual,expected)
            end
        end
        for prime in (2,3,101)
            field=CM.Fp(prime);f=DI.RipsFiltration(max_dim=3)
            D=OP.persistence_diagram(A,f;field,max_homology_dim=2)
            E=OP.persistence_diagram(A,f;field,max_homology_dim=2,method=:explicit)
            @test D.finite_by_dim==E.finite_by_dim
            @test D.essential_by_dim==E.essential_by_dim
            @test OP.provenance(D).computation.terminal_radius!==nothing
        end
    end
end

@testset "Rips followup: batched queue preserves modular cancellation" begin
    rng=MersenneTwister(101007)
    for hint in (0,1,7,160)
        prime=2;K=CM.coeff_type(CM.Fp(prime))
        queue=OP._RipsBinaryQueue(hint)
        expected=Dict{Int,K}()
        grades=rand(rng,(-0.0,0.0,nextfloat(0.0),0.5,1.0,floatmax(Float64)),160)
        function pop_expected!()
            actual=OP._rips_pop_pivot!(queue)
            if isempty(expected)
                @test actual===nothing
            else
                id=first(sort!(collect(keys(expected));lt=(a,b)->
                    grades[a]<grades[b] || (grades[a]==grades[b] && a<b)))
                @test actual!==nothing && actual.index==id && actual.coefficient==expected[id]
                @test isequal(actual.grade,grades[id])
                delete!(expected,id)
            end
        end
        for batch in 1:80
            for _ in 1:rand(rng,1:160)
                id=rand(rng,1:160);c=K(rand(rng,0:prime-1))
                OP._rips_heap_push!(queue,OP._RipsTerm(id,grades[id],c))
                v=get(expected,id,zero(K))+c
                iszero(v) ? delete!(expected,id) : (expected[id]=v)
            end
            # Leave older sorted runs partially consumed while introducing
            # new terms whose priorities lie before, among or after them.
            for _ in 1:rand(rng,1:12);pop_expected!();end
            if batch%17==0
                empty!(queue);empty!(expected)
                @test isempty(queue) && length(queue)==0
            end
        end
        while !isempty(queue);pop_expected!();end
        @test isempty(expected)
        empty!(queue)
        for id in 1:40
            OP._rips_heap_push!(queue,OP._RipsTerm(id,grades[id],one(K)))
            OP._rips_heap_push!(queue,OP._RipsTerm(id,grades[id],-one(K)))
        end
        @test OP._rips_pop_pivot!(queue)===nothing
        @test isempty(queue) && length(queue)==0
    end
end

@testset "Rips followup: pruned and unpruned queue routing" begin
    K=CM.coeff_type(CM.Fp(2))
    A=[0. 1. 2. 1. 1.5; 1. 0. 1. 2. 1.5; 2. 1. 0. 1. 1.5;
       1. 2. 1. 0. 1.5; 1.5 1.5 1.5 1.5 0.]
    f=DI.RipsFiltration(max_dim=2)
    g=OP._rips_graph(DI._rips_persistence_graph(A,f,2),2)
    active=[[v for v in g.neighbors[u] if A[u,v]<=1.5] for u in 1:5]
    clipped=OP._RipsGraph{true}(g.neighbors,g.distances,g.edges,g.grades,
        g.binomial,g.births,1.5,active)
    @test OP._rips_reduction_queue(g,K) isa OP._RipsBinaryQueue
    @test OP._rips_reduction_queue(clipped,K) isa Vector{OP._RipsBinaryTerm}
    @test OP._rips_reduction_queue(g,CM.coeff_type(CM.Fp(3))) isa Vector{OP._RipsTerm{CM.coeff_type(CM.Fp(3))}}
    for cocycles in (false,true),method in (:implicit,:explicit)
        D=OP.persistence_diagram(A,f;field=CM.Fp(2),max_homology_dim=1,cocycles,method)
        @test D.finite_by_dim[2]==[(1.0,1.5)]
        @test isempty(D.essential_by_dim[2])
    end
end
