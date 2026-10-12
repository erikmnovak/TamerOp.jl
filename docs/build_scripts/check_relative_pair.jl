# Execute the canonical task recipe and its stated mathematical conclusions.
using TamerOp, Test, SparseArrays
using TamerOp.CoreModules: F3
const _RELATIVE_RECIPE = joinpath(@__DIR__,"..","recipes","relative_pair.md")
for code in eachmatch(r"```julia\n(.*?)\n```"s,read(_RELATIVE_RECIPE,String))
    include_string(@__MODULE__,code.captures[1],_RELATIVE_RECIPE)
end
@testset "Relative-pair recipe" begin
    @test essential_births(diagram;dim=1)==[1//1]
    @test A.dimensions(h)==[0,1]
    @test maps.connecting==reshape(TamerOp.CoreModules.coeff_type(F3()).([2,1]),2,1)
    @test A.relative_cells(c;dim=1).cell_ids==(30,)
    @test size(A.relative_boundary(c;dim=1))==(0,1)
    @test finite_intervals(d;dim=0)==[(0//1,1//1),(0//1,2//1)]
    @test finite_intervals(d;dim=1)==[(3//1,4//1)]
    @test size(A.relative_structure_map(before,after;dim=1))==(0,1)
    for kind in (:barcode,:persistence_diagram)
        spec = TamerOp.Visualization.visual_spec(d;kind,dim=1)
        @test TamerOp.Visualization.check_visual_spec(spec).valid
        @test spec.metadata.total_groups == 1
    end
end
