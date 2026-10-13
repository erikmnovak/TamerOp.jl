# A half-open variant of Bjerkevik (2025), Example 4.16, as an actual quotient.
# The fixture is shared by mathematical, rendering and browser checks.
function _pruning_square_fixture(;field=TamerOp.CoreModules.F2(),closed=false)
    C=TamerOp.CoreModules;F=TamerOp.FiniteFringe;M=TamerOp.Modules
    A=TamerOp.AbelianCategories;I=TamerOp.Interleavings;K=C.coeff_type(field)
    cuts=([0,1,3,4],[0,1,3,4])
    # I1=[1,4)x[0,4), I2=[0,4)x[1,4), I3=[3,4)^2, epsilon=1/2.
    # The closed variant retains an additional naturality constraint at (4,4).
    axis=C.QQ[-1,0,1//2,1,2,3,7//2,4,5]
    points=vec(collect(Iterators.product(axis,axis)))
    upper(x)=closed ? x<=4 : x<4
    active=[findall((1<=x && 0<=y && upper(x)&&upper(y),0<=x && 1<=y && upper(x)&&upper(y))) for (x,y) in points]
    Q=F.GridPoset((collect(1:9),collect(1:9)))
    maps=Dict(e=>K[Int(a==b) for a in active[e[2]],b in active[e[1]]] for e in F.cover_edges(Q))
    S=M.PModule{K}(Q,length.(active),maps;field)
    dims=[Int(3<=x && 3<=y && upper(x)&&upper(y)) for (x,y) in points]
    J=M.PModule{K}(Q,dims,Dict((u,v)=>ones(K,dims[v],dims[u]) for (u,v) in F.cover_edges(Q));field)
    incl=M.PMorphism(J,S,Matrix{K}[ones(K,S.dims[v],dims[v]) for v in eachindex(points)])
    quotient,_=A.quotient_with_projection(incl)
    return I.RectilinearModule(quotient,cuts)
end

# Asymmetric rectangle with nonzero comparison maps at (1,1), epsilon=1/2.
function _pruning_box_fixture(;field=TamerOp.CoreModules.F2())
    C=TamerOp.CoreModules;F=TamerOp.FiniteFringe;M=TamerOp.Modules
    I=TamerOp.Interleavings;K=C.coeff_type(field)
    points=vec(collect(Iterators.product(C.QQ[-1,0,2,4,5],C.QQ[-2,-1,2,5,6])))
    dims=[Int(0<=x<4 && -1<=y<5) for (x,y) in points]
    Q=F.GridPoset((collect(1:5),collect(1:5)))
    maps=Dict((u,v)=>ones(K,dims[v],dims[u]) for (u,v) in F.cover_edges(Q))
    I.RectilinearModule(M.PModule{K}(Q,dims,maps;field),([0,4],[-1,5]))
end
