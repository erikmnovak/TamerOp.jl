# Mathematical oracle for the indicator-presentations chapter, not a tutorial.
# Run from the package root:
#   julia --startup-file=no --project=. docs/build_scripts/check_indicator_presentations.jl
# The fixtures use public APIs and need no test helpers or local notebooks.

using Test
using LinearAlgebra: det
import TamerOp as OP
import TamerOp.Advanced as OA
import TamerOp.CoreModules: QQField

# Verify a direct sum of closed-square indicators in explicit summand bases.
# A basis matrix carries summand coordinates into the package's stalk basis.
function check_indicator_samples(enc, axis, lower_bounds, upper_bounds, basis_at)
    P = OP.encoding_poset(enc)
    classifier = OP.encoding_map(enc)
    M = OP.encoding_module(enc)
    dims = OP.dimensions(enc)
    @test enc isa OP.EncodingResult
    @test OP.provenance(enc).field == QQField()
    @test OP.provenance(enc).category == :finite_poset_representations
    @test OP.dimensions(M).stalks == dims
    @test length(dims) == OA.nvertices(P)
    @test OA.check_module(M; throw=false).valid

    points = [[x, y] for x in axis for y in axis]
    labels = [OA.locate(classifier, point) for point in points]
    active = [findall(i -> all(lower_bounds[i] .<= point .<= upper_bounds[i]),
                      eachindex(lower_bounds)) for point in points]
    @test all(label -> 1 <= label <= OA.nvertices(P), labels)
    # The grid includes representatives of every returned signature, without
    # prescribing numeric IDs or a particular number of labels.
    @test Set(labels) == Set(1:OA.nvertices(P))

    bases = [basis_at(point, summands) for (point, summands) in zip(points, active)]
    for i in eachindex(points)
        @test OA.dim_at(M, labels[i]) == length(active[i])
        @test size(bases[i]) == (length(active[i]), length(active[i]))
        @test isempty(active[i]) || !iszero(det(bases[i]))
    end

    n = length(points)
    comparable = falses(n, n)
    maps = Dict{Tuple{Int,Int},Matrix{OP.QQ}}()
    matches_summand_basis = true
    for i in 1:n, j in 1:n
        all(points[i] .<= points[j]) || continue
        comparable[i, j] = true
        @test OA.leq(P, labels[i], labels[j])
        actual = Matrix(OA.structure_map(M; source=labels[i], target=labels[j]))
        expected = OP.QQ[target == source for target in active[j], source in active[i]]
        @test size(actual) == size(expected)
        # Naturality verifies the whole linear map even if the package changes
        # stalk bases: actual * source_basis == target_basis * expected.
        @test actual * bases[i] == bases[j] * expected
        matches_summand_basis &= actual == expected
        maps[(i, j)] = actual
    end

    ncompositions = 0
    for i in 1:n, j in 1:n, k in 1:n
        comparable[i, j] && comparable[j, k] || continue
        @test maps[(j, k)] * maps[(i, j)] == maps[(i, k)]
        ncompositions += 1
    end
    println("  Returned poset: ", OA.nvertices(P), " labels; every label sampled.")
    println("  Verified ", n, " points, ", length(maps),
        " comparable-pair maps, and ", ncompositions, " compositions.")
    println("  Raw sampled matrices match the ordered summand bases: ", matches_summand_basis)
    return nothing
end

function check_indicator_presentations()
    @testset "Indicator-presentation documentation examples" begin
        field = QQField()
        options = OA.EncodingOptions(; backend=:pl_backend,
            poset_kind=:signature, field=field)

        @testset "One square: image of a one-by-one presentation" begin
            upsets = [OA.BoxUpset([0.0, 0.0])]
            downsets = [OA.BoxDownset([2.0, 2.0])]
            coefficient = reshape(OP.QQ[1], 1, 1)
            enc = OP.encode(upsets, downsets, coefficient, options)
            M = OP.encoding_module(enc)
            classifier = OP.encoding_map(enc)
            anchor = OA.locate(classifier, [0.0, 0.0])
            basis_at = function (point, summands)
                isempty(summands) && return zeros(OP.QQ, 0, 0)
                label = OA.locate(classifier, point)
                return Matrix(OA.structure_map(M; source=anchor, target=label))
            end
            println("One square: image of [1] over QQ; support [0,2]^2.")
            check_indicator_samples(enc, [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
                [0.0], [2.0], basis_at)
        end

        @testset "Two squares: image of a diagonal presentation" begin
            lower_bounds = [0.0, 1.0]
            upper_bounds = [2.0, 3.0]
            upsets = [OA.BoxUpset([0.0, 0.0]), OA.BoxUpset([1.0, 1.0])]
            downsets = [OA.BoxDownset([2.0, 2.0]), OA.BoxDownset([3.0, 3.0])]
            coefficient = OP.QQ[1 0; 0 1]
            enc = OP.encode(upsets, downsets, coefficient, options)
            M = OP.encoding_module(enc)
            classifier = OP.encoding_map(enc)

            # Choose a compatible summand basis through public structure maps.
            # The first summand starts at a; the overlap starts at b; only the
            # second summand survives at c. No package basis order is assumed.
            a = OA.locate(classifier, [0.0, 0.0])
            b = OA.locate(classifier, [1.0, 1.0])
            c = OA.locate(classifier, [3.0, 3.0])
            first_column = Matrix(OA.structure_map(M; source=a, target=b))
            surviving_row = Matrix(OA.structure_map(M; source=b, target=c))
            @test size(first_column) == (2, 1)
            @test size(surviving_row) == (1, 2)
            @test any(x -> !iszero(x), first_column)
            @test any(x -> !iszero(x), surviving_row)
            @test surviving_row * first_column == zeros(OP.QQ, 1, 1)
            pivot = something(findfirst(x -> !iszero(x), vec(surviving_row)))
            second_column = zeros(OP.QQ, 2, 1)
            second_column[pivot, 1] = inv(surviving_row[1, pivot])
            overlap_basis = hcat(first_column, second_column)
            @test !iszero(det(overlap_basis))
            @test surviving_row * overlap_basis == OP.QQ[0 1]

            basis_at = function (point, summands)
                isempty(summands) && return zeros(OP.QQ, 0, 0)
                label = OA.locate(classifier, point)
                if summands == [1]
                    return Matrix(OA.structure_map(M; source=a, target=label))
                end
                from_overlap = Matrix(OA.structure_map(M; source=b, target=label))
                return from_overlap * (summands == [2] ? second_column : overlap_basis)
            end

            println("Two squares: image of I2 over QQ; [0,2]^2 direct sum [1,3]^2.")
            check_indicator_samples(enc, [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
                lower_bounds, upper_bounds, basis_at)

            p, q, r = [0.5, 0.5], [1.5, 1.5], [2.5, 2.5]
            lp, lq, lr = [OA.locate(classifier, point) for point in (p, q, r)]
            @test [OA.dim_at(M, label) for label in (lp, lq, lr)] == [1, 2, 1]
            pq = Matrix(OA.structure_map(M; source=lp, target=lq))
            qr = Matrix(OA.structure_map(M; source=lq, target=lr))
            pr = Matrix(OA.structure_map(M; source=lp, target=lr))
            @test pq * basis_at(p, [1]) == basis_at(q, [1, 2]) * reshape(OP.QQ[1, 0], 2, 1)
            @test qr * basis_at(q, [1, 2]) == basis_at(r, [2]) * OP.QQ[0 1]
            @test any(x -> !iszero(x), pq)
            @test any(x -> !iszero(x), qr)
            @test pr == zeros(OP.QQ, 1, 1)
            @test qr * pq == pr
            @test OA.dim_at(M, OA.locate(classifier, [1.0, 1.0])) == 2
            @test OA.dim_at(M, OA.locate(classifier, [2.0, 2.0])) == 2

            # At this mixed point the source U1 and target D2 are active, but
            # the surviving 1-by-1 coefficient is zero; activity is not rank.
            mixed = [0.5, 2.5]
            active_columns = findall(i -> all(mixed .>= lower_bounds[i]), eachindex(lower_bounds))
            active_rows = findall(i -> all(mixed .<= upper_bounds[i]), eachindex(upper_bounds))
            @test active_columns == [1]
            @test active_rows == [2]
            @test coefficient[active_rows, active_columns] == zeros(OP.QQ, 1, 1)
            @test OA.dim_at(M, OA.locate(classifier, mixed)) == 0

            println("  p=(0.5,0.5), q=(1.5,1.5), r=(2.5,2.5): dimensions [1,2,1].")
            println("  Raw p->q matrix: ", pq)
            println("  Raw q->r matrix: ", qr)
            println("  Raw p->r matrix: ", pr)
            println("  Exact ranks: 1, 1, 0; q->r composed with p->q is zero.")
            println("  Both overlap boundaries (1,1) and (2,2) have dimension 2.")
            println("  Mixed point (0.5,2.5): active row [2], active column [1], coefficient [0], dimension 0.")
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("Julia ", VERSION)
    check_indicator_presentations()
end
