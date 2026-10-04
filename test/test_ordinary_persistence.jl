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
                            input::Symbol, order::Symbol) where {T,N}
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
    boundaries = [falses(length(cells[d]), length(cells[d + 1])) for d in 1:N]
    for d in 1:N, (j, cell) in enumerate(cells[d + 1]), face in faces(cell)
        i = index[d][face]
        boundaries[d][i, j] = !boundaries[d][i, j]
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
    for field in (:f2, :F2, CM.QQField(), CM.F3(), CM.Fp(5), CM.RealField(Float64))
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
