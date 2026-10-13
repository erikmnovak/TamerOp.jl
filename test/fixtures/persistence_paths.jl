function _a96_triangle(;superlevel=false)
    D=TamerOp.DataTypes;I=TamerOp.DataIngestion
    g=[0,1,2,3,4,5,7];superlevel && (g=10 .- g)
    G=D.GradedComplex([[11,12,13],[21,22,23],[31]],
        [sparse([-1 -1 0;1 0 -1;0 1 1]),sparse(reshape([1,-1,1],3,1))],[(v,) for v in g])
    p=[(-1,0),(5,0),(-1,3)]
    I.filtered_complex(G;geometry=[[p[1]],[p[2]],[p[3]],p[[1,2]],p[[1,3]],p[[2,3]],p],
        faces=[Int[],Int[],Int[],[1,2],[1,3],[2,3],[4,5,6]],orientation=(superlevel ? -1 : 1,))
end
function _a96_path(;degree=0,superlevel=false,field=TamerOp.CoreModules.F3(),representatives=true)
    g=[[2,0,1,4,6,3,8],[1,2,0,5,4,6,9]]
    superlevel && (g=[10 .- v for v in g])
    TamerOp.persistence_path(_a96_triangle(;superlevel);grades=g,parameters=[-2//3,1//7,5//2],degree,field,representatives)
end
function _a96_supplied(;matching=:bottleneck)
    O=TamerOp.OrdinaryPersistence
    diagrams=[O.PersistenceDiagram([[(0,4),(0,4),(1,2)]],[[0]]),
        O.PersistenceDiagram([[(1,5),(1,5)]],[[1]]),
        O.PersistenceDiagram([[(2,6)]],[[2]])]
    observations=[(;objective=length(O.finite_intervals(d;dim=0)),objective_label="Finite intervals",
        settings=(;iteration=i),invariants=(;essential=length(O.essential_births(d;dim=0)))) for (i,d) in enumerate(diagrams)]
    TamerOp.persistence_path(diagrams;parameters=[0,1//3,2],matching,observations)
end
