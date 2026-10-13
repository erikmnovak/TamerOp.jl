# Execute the canonical task recipe and its stated mathematical conclusions.
using TamerOp, Test, SparseArrays
using TamerOp.CoreModules: F3
const _RELATIVE_RECIPE = joinpath(@__DIR__,"..","recipes","relative_pair.md")
for code in eachmatch(r"```julia\n(.*?)\n```"s,read(_RELATIVE_RECIPE,String))
    include_string(@__MODULE__,code.captures[1],_RELATIVE_RECIPE)
end
@testset "Relative-pair recipe" begin
    @test essential_births(diagram;dim=1)==[1//1]
    @test OA.dimensions(h)==[0,1]
    @test maps.connecting==reshape(TamerOp.CoreModules.coeff_type(F3()).([2,1]),2,1)
    @test OA.relative_cells(c;dim=1).cell_ids==(30,)
    @test size(OA.relative_boundary(c;dim=1))==(0,1)
    @test finite_intervals(d;dim=0)==[(0//1,1//1),(0//1,2//1)]
    @test finite_intervals(d;dim=1)==[(3//1,4//1)]
    @test size(OA.relative_structure_map(before,after;dim=1))==(0,1)
    # The schematic labels each cell's lifetime in the quotient coordinates.
    for (scale, expected) in ((0, [2,0]), (1, [2,1]), (2, [1,1]), (3, [0,1]), (4, [0,0]))
        snapshot = OA.relative_complex(changing; scale)
        @test OA.dimensions(snapshot) == expected
    end
    # The implementation account's explicit cone must give the same intervals.
    # Columns of D1 are e, s(v0), s(v1); the D2 column is s(e).
    D1 = [-1 1 0; 1 0 1]
    D2 = reshape([1,1,-1],3,1)
    @test iszero(D1 * D2)
    cone = OP.GradedComplex([[1,2],[3,4,5],[6]], [sparse(D1),sparse(D2)],
        [(0//1,), (0//1,), (1//1,), (2//1,), (3//1,), (4//1,)])
    cone_diagram = OP.persistence_diagram(cone; field=F3())
    for dim in 0:1
        @test OP.finite_intervals(cone_diagram; dim) == OP.finite_intervals(d; dim)
        @test OP.essential_births(cone_diagram; dim) == OP.essential_births(d; dim)
    end
    @test isempty(OP.finite_intervals(cone_diagram; dim=2))
    @test isempty(OP.essential_births(cone_diagram; dim=2))
    for kind in (:barcode,:persistence_diagram)
        spec = TamerOp.Visualization.visual_spec(d;kind,dim=1)
        @test TamerOp.Visualization.check_visual_spec(spec).valid
        @test spec.metadata.total_groups == 1
    end
end
