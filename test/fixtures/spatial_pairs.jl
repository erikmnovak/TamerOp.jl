function _a128_disk(;superlevel=false)
    D=TamerOp.DataTypes;I=TamerOp.DataIngestion
    grades=[0,0,0,1,1,1,2];superlevel && (grades=2 .- grades)
    G=D.GradedComplex([[11,12,13],[21,22,23],[31]],
        [sparse([-1 -1 0;1 0 -1;0 1 1]),sparse(reshape([1,-1,1],3,1))],[(g,) for g in grades])
    p=[(-1,0),(5,0),(-1,3)]
    I.filtered_complex(G;geometry=[[p[1]],[p[2]],[p[3]],p[[1,2]],p[[1,3]],p[[2,3]],p],
        faces=[Int[],Int[],Int[],[1,2],[1,3],[2,3],[4,5,6]],
        source_ids=[:a,:b,:c,:ab,:ac,:bc,:disk],orientation=(superlevel ? -1 : 1,))
end
