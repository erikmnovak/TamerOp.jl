function slice_class_triangle(field=TamerOp.CoreModules.F2())
    D=TamerOp.DataTypes;I=TamerOp.DataIngestion;F=TamerOp.FiniteFringe
    M=TamerOp.Modules;E=TamerOp.EncodingCore;K=TamerOp.CoreModules.coeff_type(field)
    G=D.GradedComplex([[11,12,13],[21,22,23],[31]],
        [sparse([-1 -1 0;1 0 -1;0 1 1]),sparse(reshape([1,-1,1],3,1))],
        [(0,0),(0,0),(0,0),(1,0),(1,0),(1,0),(2,2)])
    points=[(0,0),(2,0),(0,1)]
    source=I.filtered_complex(G;geometry=[[points[1]],[points[2]],[points[3]],
        points[[1,2]],points[[1,3]],points[[2,3]],points])
    P=F.ProductOfChainsPoset((4,4));pi=E.GridEncodingMap(P,([0,1,2,3],[0,1,2,3]))
    dims=[(i>=2 && !(i>=3 && j>=3)) ? 1 : 0 for j in 1:4 for i in 1:4]
    maps=Dict((a,b)=>ones(K,dims[b],dims[a]) for (a,b) in F.cover_edges(P))
    module_=M.PModule{K}(P,dims,maps;field)
    enc=TamerOp.Results.EncodingResult(P,module_,E.compile_encoding(P,pi))
    basis=I.source_homology_basis(source;degree=1,basis=(label,p)->
        dims[label]==0 ? zeros(K,3,0) : reshape(K[1,-1,1],3,1))
    (;enc,source,basis)
end
