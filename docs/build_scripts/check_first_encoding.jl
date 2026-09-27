# Mathematical oracle for the first documentation examples, not a tutorial.
# Run from the package root:
#   julia --startup-file=no --project=. docs/build_scripts/check_first_encoding.jl
# This script constructs its own fixtures through public APIs. It does not load
# test helpers, local notebooks, or the ignored documentation planning files.

using Test
import TamerOp as OP
import TamerOp.Advanced as OA
import TamerOp.CoreModules: F2, QQField

function check_first_encoding()
    @testset "First documentation examples" begin
        @testset "Ring: direct ordinary persistence over F2" begin
            ring_values = [0 0 0; 0 5 0; 0 0 0]
            diagram = OP.cubical_persistence(ring_values;
                field=F2(), order=:sublevel, input=:top_cells, periodic=false)

            # At grade 0 the boundary squares enclose one hole. The central
            # square enters at grade 5, filling it; one component persists.
            @test OP.finite_intervals(diagram; dim=1) == [(0, 5)]
            @test isempty(OP.essential_births(diagram; dim=1))
            @test isempty(OP.finite_intervals(diagram; dim=0))
            @test OP.essential_births(diagram; dim=0) == [0]
            @test OP.persistence_intervals(diagram; dim=0) == [(0, Inf)]
            @test OP.provenance(diagram).field == F2()
            @test !(diagram isa OP.EncodingResult)

            # The proposed first exercise changes the filling time only.
            varied = OP.cubical_persistence([0 0 0; 0 3 0; 0 0 0];
                field=F2(), order=:sublevel, input=:top_cells, periodic=false)
            @test OP.finite_intervals(varied; dim=1) == [(0, 3)]
            @test OP.essential_births(varied; dim=0) == [0]
            println("Ring: F2, nonperiodic top-cell sublevels; H1 interval [0,5), essential H0 birth 0.")
            println("Ring variation: changing the center to 3 gives H1 interval [0,3).")
        end

        @testset "Square: finite encoding over QQ, including maps" begin
            field = QQField()
            births = [OA.BoxUpset([0.0, 0.0])]
            deaths = [OA.BoxDownset([2.0, 2.0])]
            coefficient = reshape(OP.QQ[1], 1, 1)
            options = OA.EncodingOptions(; backend=:pl_backend,
                poset_kind=:signature, field=field)
            enc = OP.encode(births, deaths, coefficient, options)

            P = OP.encoding_poset(enc)
            classifier = OP.encoding_map(enc)
            dims = OP.dimensions(enc)
            M = OP.encoding_module(enc)
            @test enc isa OP.EncodingResult
            @test OP.describe(enc).kind == :encoding_result
            @test OP.provenance(enc).field == field
            @test OP.provenance(enc).category == :finite_poset_representations
            @test OP.dimensions(M).stalks == dims
            @test length(dims) == OA.nvertices(P)
            @test OA.check_module(M; throw=false).valid

            # Hand model: im([1]) is QQ precisely where the birth and death
            # indicators are both active, namely on the CLOSED square [0,2]^2.
            # All displayed coordinates are exactly representable in Float64.
            axis = [-1.0, 0.0, 0.25, 1.0, 1.5, 2.0, 3.0]
            points = [[x, y] for x in axis for y in axis]
            labels = [OA.locate(classifier, point) for point in points]
            expected_dims = [Int(all(0.0 .<= point .<= 2.0)) for point in points]
            @test all(label -> 1 <= label <= OA.nvertices(P), labels)
            @test [OA.dim_at(M, label) for label in labels] == expected_dims

            # Recover the returned signature poset semantically, without
            # assuming a numeric label order or the prose's nine-label grid.
            # The second bit records the COMPLEMENT of the death indicator.
            signature_labels = Dict{Tuple{Bool,Bool},Int}()
            for (point, label) in zip(points, labels)
                signature = (all(point .>= 0.0), any(point .> 2.0))
                if haskey(signature_labels, signature)
                    @test signature_labels[signature] == label
                else
                    signature_labels[signature] = label
                end
            end
            @test length(signature_labels) == OA.nvertices(P)
            @test Set(values(signature_labels)) == Set(1:OA.nvertices(P))
            for (s, u) in signature_labels, (t, v) in signature_labels
                @test OA.leq(P, u, v) == ((!s[1] || t[1]) && (!s[2] || t[2]))
            end
            @test count(==(1), dims) == 1
            @test all(d -> d in (0, 1), dims)

            # Matrices have target rows and source columns. Only ORIGINAL
            # coordinatewise-comparable pairs carry a map of the R^2 module.
            n = length(points)
            comparable = falses(n, n)
            maps = Dict{Tuple{Int,Int},Matrix{OP.QQ}}()
            for i in 1:n, j in 1:n
                all(points[i] .<= points[j]) || continue
                comparable[i, j] = true
                @test OA.leq(P, labels[i], labels[j])
                actual = Matrix(OA.structure_map(M; source=labels[i], target=labels[j]))
                expected = zeros(OP.QQ, expected_dims[j], expected_dims[i])
                if expected_dims[i] == expected_dims[j] == 1
                    expected[1, 1] = 1
                end
                @test actual == expected
                maps[(i, j)] = actual
            end

            ncompositions = 0
            for i in 1:n, j in 1:n, k in 1:n
                comparable[i, j] && comparable[j, k] || continue
                @test maps[(j, k)] * maps[(i, j)] == maps[(i, k)]
                ncompositions += 1
            end

            # Same label does not imply that the original points are ordered.
            p = [0.25, 1.5]
            q = [1.5, 0.25]
            @test !all(p .<= q) && !all(q .<= p)
            @test OA.locate(classifier, p) == OA.locate(classifier, q)
            @test OA.leq(P, OA.locate(classifier, p), OA.locate(classifier, q))

            println("Square: QQ coefficient [1], support [0,2]^2, coordinatewise order on R^2.")
            println("Returned poset: ", OA.nvertices(P), " labels; order equals inclusion of the two signature bits.")
            for signature in sort!(collect(keys(signature_labels)))
                label = signature_labels[signature]
                println("  signature ", signature, " -> label ", label, "; dimension ", OA.dim_at(M, label))
            end
            println("Verified ", length(points), " points, ", length(maps),
                " comparable-pair maps, and ", ncompositions, " compositions.")
            println("Both boundaries 0 and 2 belong to the support; exterior spaces are zero.")
            println("Interior maps are [1]; other comparable maps are the unique correctly shaped zero matrices.")
            println("Incomparable interior points can share a label; label order does not imply original comparability.")
            println("The hand-written nine-label Cartesian model and the returned signature model are different encodings.")
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("Julia ", VERSION)
    check_first_encoding()
end
