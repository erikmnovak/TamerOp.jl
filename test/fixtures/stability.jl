# Unequal cell widths expose counting-measure and transposition mistakes.
function _a51_grid(;field=TamerOp.CoreModules.QQField())
    C=TamerOp.CoreModules;F=TamerOp.FiniteFringe;D=TamerOp.Modules
    K=C.coeff_type(field);Q=F.ProductOfChainsPoset((2,2))
    M=D.PModule{K}(Q,[2,1,1,0],Dict((1,2)=>K[1 0],(1,3)=>K[0 1],
        (2,4)=>zeros(K,0,1),(3,4)=>zeros(K,0,1));field)
    TamerOp.Stability.SkyscraperGrid(M,([-2,-1,1],[4,6,9]))
end
_a51_chambers()=TamerOp.Stability.skyscraper_chambers(_a51_grid())
_a51_session()=TamerOp.Visualization.stability_explorer(_a51_chambers();anchor=(-3//2,5))
function _a51_landscape()
    g=_a51_grid()
    points=[(x,y) for y in (9//2,11//2,7),x in (-7//4,-5//4,0)][:]
    TamerOp.Stability.filtered_landscape(g;points,thresholds=[0,1//5,1//2],layers=[1,2],atol=1//128)
end
