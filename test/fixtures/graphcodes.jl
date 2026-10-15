# A repeated interval, two components, a changing target support, and identity runs.
function _a126_graphcode()
    G=TamerOp.Graphcodes
    rows=[[(1,4),(2,3)],[(1,4),(2,3)],[(1,4),(1,3)],[(1,4),(1,3)]]
    G.graphcode(rows,[Int[1 0;0 1],Int[1 0;0 1],Int[1 0;0 1]];
        x=[0//1,1//3,2//1,5//1],y=[8//1,4//1,1//1,0//1])
end
_a126_session()=TamerOp.Visualization.graphcode_explorer(_a126_graphcode();row=2,source_interval=2,target_interval=2,column=2)
