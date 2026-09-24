# Independent ordinary-persistence oracles. The cubical reference below forms
# tensor products of interval/circle chain complexes directly; it uses neither
# DataIngestion geometry nor the production persistence reduction or field algebra.

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
    @test TamerOp.provenance(diagram).backend === :f2_column_reduction
    @test TamerOp.provenance(diagram).approximation === :none_in_reduction
    @test TamerOp.provenance(diagram).discretization === :none
    @test TamerOp.result_summary(diagram) == OP.persistence_diagram_summary(diagram)
    @test TOA.check_persistence_diagram(diagram; throw=true).valid
    @test TOA.filtration_order(diagram) === :sublevel
end
