# Independent finite spread modules for end-curve and browser acceptance.
function _a127_module(shape,supports;field=TamerOp.CoreModules.F2(),orientation=(1,1))
    F=TamerOp.FiniteFringe;C=TamerOp.CoreModules;M=TamerOp.Modules
    m,n=shape;K=C.coeff_type(field);Q=F.ProductOfChainsPoset(shape;orientation)
    canonical(q)=begin
        i,j=mod1(q,m),cld(q,m)
        (orientation[1]==1 ? i-1 : m-i,orientation[2]==1 ? j-1 : n-j)
    end
    active=[findall(S->canonical(q) in S,supports) for q in 1:m*n]
    maps=Dict((u,v)=>K[a==b for a in active[v],b in active[u]] for (u,v) in F.cover_edges(Q))
    M.PModule{K}(Q,length.(active),maps;field)
end
_a127_rectangle(a,b)=[(i,j) for i in a[1]:b[1] for j in a[2]:b[2]]
function _a127_three_lines(;field=TamerOp.CoreModules.F2(),coincident=false)
    C=TamerOp.CoreModules;F=TamerOp.FiniteFringe;M=TamerOp.Modules;K=C.coeff_type(field)
    Q=F.ProductOfChainsPoset((3,3));dims=[0,0,1,0,1,2,1,2,2]
    bases=[zeros(K,2,0) for _ in 1:9]
    bases[3]=reshape(K[1,coincident ? 0 : 1],2,1)
    bases[5]=reshape(K[0,1],2,1);bases[7]=reshape(K[1,0],2,1)
    for q in (6,8,9);bases[q]=K[1 0;0 1];end
    maps=Dict((u,v)=>(dims[u]==0 || dims[v]==0 ? zeros(K,dims[v],dims[u]) :
        dims[v]==2 ? copy(bases[u]) : ones(K,1,1)) for (u,v) in F.cover_edges(Q))
    M.PModule{K}(Q,dims,maps;field)
end
function _a127_result()
    E=TamerOp.EndCurves
    M=_a127_module((3,2),[_a127_rectangle((0,0),(2,1)),[(0,1),(1,1),(1,0)],[(2,0)]])
    E.end_curves(M;origin=(-2,4),witnesses=true)
end
_a127_session()=TamerOp.Visualization.end_curve_explorer(_a127_result();kind=:birth,curve=1,point=(-2,4))
