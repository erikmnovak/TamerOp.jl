# Oriented triangle with repeated labels across dimensions and nonuniform grades.
function _a130_complex(;source=true,order=:sublevel,tied=false)
    grades=tied ? fill((0//1,),7) : [(-2//1,),(-2//1,),(1//3,),(2//1,),(2//1,),(4//1,),(7//1,)]
    order===:superlevel && (grades=[(-g[1],) for g in grades])
    G=TamerOp.GradedComplex([[10,20,30],[10,20,30],[10]],
        [sparse([-1 -1 0;1 0 -1;0 1 1]),sparse(reshape([1,-1,1],3,1))],grades)
    source || return G
    points=[(-2.,1.),(3.,1.),(-0.5,4.)]
    geometry=[[points[i]] for i in 1:3]
    append!(geometry,[points[[1,2]],points[[1,3]],points[[2,3]],points])
    TamerOp.filtered_complex(G;geometry,orientation=(order===:sublevel ? 1 : -1,),construction="oriented triangle")
end
_a130_trace(;kwargs...)=TamerOp.persistence_reduction(_a130_complex(;kwargs...);order=get(kwargs,:order,:sublevel),field=TamerOp.CoreModules.F3())
_a130_session()=TamerOp.reduction_explorer(_a130_trace())
