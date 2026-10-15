# A repeated interval, two components, a changing target support, and identity runs.
function _a126_graphcode()
    G=TamerOp.Graphcodes
    rows=[[(1,4),(2,3)],[(1,4),(2,3)],[(1,4),(1,3)],[(1,4),(1,3)]]
    G.graphcode(rows,[Int[1 0;0 1],Int[1 0;0 1],Int[1 0;0 1]];
        x=[0//1,1//3,2//1,5//1],y=[8//1,4//1,1//1,0//1])
end
_a126_session()=TamerOp.Visualization.graphcode_explorer(_a126_graphcode();row=2,source_interval=2,target_interval=2,column=2)

function _a126_weighted_graphcode(field)
    C=TamerOp.CoreModules;K=C.coeff_type(field)
    a=field isa C.QQField ? K(-2//3) : K(2)
    TamerOp.Graphcodes.graphcode(fill([(1,3)],4),[reshape([v],1,1) for v in K[a,-1,1]];
        x=[0//1,1//3,8//1],y=[9,4,1,-2],field)
end
_a126_weighted_session(field)=TamerOp.Visualization.graphcode_explorer(_a126_weighted_graphcode(field);
    row=1,source_interval=1,target_interval=1,column=2)
