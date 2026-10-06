# Run with julia --project=. docs/examples/generalized_rank.jl
using TamerOp

const GRExamples = let
    FF = TamerOp.FiniteFringe
    MD = TamerOp.Modules
    CM = TamerOp.CoreModules
    EC = TamerOp.EncodingCore
    A = TamerOp.Advanced

    # The two source lines agree at their common target in only one module.
    fork = FF.FinitePoset(Bool[1 0 1; 0 1 1; 0 0 1])
    parallel = MD.PModule{QQ}(fork, [1,1,2],
        Dict((1,3) => reshape(QQ[1,0],2,1), (2,3) => reshape(QQ[1,0],2,1));
        field=CM.QQField())
    transverse = MD.PModule{QQ}(fork, [1,1,2],
        Dict((1,3) => reshape(QQ[1,0],2,1), (2,3) => reshape(QQ[0,1],2,1));
        field=CM.QQField())
    @assert Dict(rank_invariant(parallel)) == Dict(rank_invariant(transverse))
    @assert generalized_rank(parallel; vertices=1:3) == 1
    @assert generalized_rank(transverse; vertices=1:3) == 0
    witness = generalized_rank(parallel; vertices=1:3, witnesses=true)
    @assert size(A.comparison_map(witness)) == (2,1)

    # The signed coefficient -1 corrects three overlapping selected intervals;
    # it is not a negative direct summand.
    star = FF.FinitePoset(Bool[1 0 0 1; 0 1 0 1; 0 0 1 1; 0 0 0 1])
    module3 = MD.PModule{QQ}(star, [1,1,1,2],
        Dict((1,4)=>reshape(QQ[1,0],2,1), (2,4)=>reshape(QQ[0,1],2,1),
             (3,4)=>reshape(QQ[1,1],2,1)); field=CM.QQField())
    signed = interval_rank_summary(module3; family=[[4],[1,4],[2,4],[3,4]])
    @assert A.interval_coefficients(signed) == [-1,1,1,1]
    @assert A.reconstruct_rank(signed; vertices=[4]) == 2

    # One finite label encodes the quadrant [0,infinity)^2, with zero below
    # either axis. An ell-worm centered at (4,6) fits until width 4/ell.
    gridposet = FF.ProductOfChainsPoset((1,1))
    quadrant = MD.PModule{QQ}(gridposet,[1],Dict{Tuple{Int,Int},Matrix{QQ}}();field=CM.QQField())
    grid = EC.GridEncodingMap(gridposet,([0],[0]))
    encoded = EncodingResult(gridposet,quadrant,grid)
    landscape = gril(encoded; centers=[(4,6)], levels=[1,2], lengths=[1,2,4])
    @assert A.gril_values(landscape) == reshape(QQ[4,0,2,0,1,0],1,2,3)
    (witness=witness, signed=signed, landscape=landscape)
end

println(describe(GRExamples.witness))
println(describe(GRExamples.signed))
println(describe(GRExamples.landscape))
