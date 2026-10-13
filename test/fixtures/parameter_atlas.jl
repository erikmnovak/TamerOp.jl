# Endpoint maps are checked independently of their pictures.
function _a125_triangle(;bigraded=false)
    G=TamerOp.DataTypes.GradedComplex([[11,12,13],[21,22,23],[31]],
        [sparse([-1 -1 0;1 0 -1;0 1 1]),sparse(reshape([1,-1,1],3,1))],
        bigraded ? [(0,0),(0,0),(0,0),(1,0),(1,0),(1,0),(2,1)] :
        [(0,),(0,),(0,),(1,),(1,),(1,),(2,)])
    p=[(0,0),(2,0),(0,2)]
    TamerOp.DataIngestion.filtered_complex(G;geometry=[[p[1]],[p[2]],[p[3]],p[[1,2]],p[[1,3]],p[[2,3]],p])
end
function _a125_zero_composite(field=TamerOp.CoreModules.QQField())
    K=TamerOp.CoreModules.coeff_type(field)
    P=TamerOp.FiniteFringe.FinitePoset(BitMatrix([i<=j for i in 1:3,j in 1:3]))
    TamerOp.Modules.PModule{K}(P,[1,2,1],Dict((1,2)=>reshape(K[1,0],2,1),(2,3)=>reshape(K[0,1],1,2));field)
end
