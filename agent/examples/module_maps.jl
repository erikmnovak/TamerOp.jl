import TamerOp as OP
import TamerOp.Advanced as OA

function run_module_maps_example()
    # 1 is below 2 and 3; both are below 4. Entries include transitive relations.
    order = Bool[1 1 1 1;
                 0 1 0 1;
                 0 0 1 1;
                 0 0 0 1]
    poset = OA.FinitePoset(order)
    maps = Dict(
        (1, 2) => reshape(OP.QQ[1, 0], 2, 1),
        (1, 3) => reshape(OP.QQ[0, 1], 2, 1),
        (2, 4) => reshape(OP.QQ[1, 0], 1, 2),
        (3, 4) => reshape(OP.QQ[0, 1], 1, 2),
    )
    module_ = OA.PModule(poset, [1, 2, 2, 1], maps;
        field=OP.CoreModules.QQField())
    @assert OA.check_module(module_; throw=true).valid
    @assert OA.check_module_functoriality(module_; throw=true).valid
    @assert OP.dimensions(module_).stalks == [1, 2, 2, 1]

    bottom_to_left = OA.structure_map(module_; source=1, target=2)
    left_to_top = OA.structure_map(module_; source=2, target=4)
    bottom_to_right = OA.structure_map(module_; source=1, target=3)
    right_to_top = OA.structure_map(module_; source=3, target=4)
    bottom_to_top = OA.structure_map(module_; source=1, target=4)
    @assert left_to_top * bottom_to_left == right_to_top * bottom_to_right == bottom_to_top
    @assert bottom_to_top == reshape(OP.QQ[1], 1, 1)

    # The first left-hand coordinate survives, while the second becomes zero.
    surviving = left_to_top * OP.QQ[1, 0]
    vanishing = left_to_top * OP.QQ[0, 1]
    @assert surviving == OP.QQ[1]
    @assert vanishing == OP.QQ[0]
    @assert OP.rank_map(module_, 1, 4) == 1

    return (; module_, bottom_to_top, surviving, vanishing)
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_module_maps_example()
    println("Dimensions: ", OP.dimensions(result.module_).stalks)
    println("Bottom to top: ", result.bottom_to_top)
    println("Images of the two left basis vectors: ", result.surviving, ", ", result.vanishing)
end
