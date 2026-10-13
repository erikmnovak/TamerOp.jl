# Independent interval formulas and exhaustive small vector-space oracles.
function _pr_interval(a,b;field=CM.QQField(),closed=false,domain=:real)
    K=CM.coeff_type(field);dims=domain===:real ? [0,1,1,Int(closed),0] : [0,1,0]
    IL.RectilinearModule(([a,b],),dims,Dict((v,v+1)=>ones(K,dims[v+1],dims[v]) for v in 1:length(dims)-1);field,domain)
end

@testset "A107 interval pruning and certificates" begin
    for field in (CM.QQField(),CM.F2(),CM.F3(),CM.Fp(101)), closed in (false,true)
        M=_pr_interval(0,4;field,closed)
        for e in (0,1//2,2,3)
            r=PR.epsilon_pruning(M,e)
            @test PR.check_pruning(r).valid
            @test PR.pruning_guarantee(r).bound==2e
            P=PR.pruned_module(r)
            for x in (-1,0,1//2,1,2,3,7//2,4,5)
                expected=e<=x && (closed ? x<=4-e : x<4-e)
                @test IL.ambient_dimension(P,[x])==Int(expected)
            end
            c=PR.pruning_certificate(r)
            @test IL.check_interleaving(c).valid
            @test c.problem.metadata.epsilon==2e
            @test PR.pruning_summary(r).max_stalk_dimension==1
            @test PR.pruning_maps(r).projection===r.projection
        end
    end
    for d in 0:2
        M=IL.RectilinearModule((Int[],),[d],Dict{Tuple{Int,Int},Matrix{QQ}}())
        r=PR.epsilon_pruning(M,1//3)
        @test IL.ambient_dimension(PR.pruned_module(r),[100])==d
        @test IL.check_interleaving(PR.pruning_certificate(r)).valid
    end
    r=PR.epsilon_pruning(_pr_interval(0,4;domain=:integer),1)
    @test [IL.ambient_dimension(r.pruned,[x]) for x in -1:5]==[0,0,1,1,0,0,0]
    @test PR.check_pruning(r).valid
    @test IL.check_interleaving(PR.pruning_certificate(r)).valid
    @test_throws ArgumentError PR.epsilon_pruning(_pr_interval(0,4;domain=:integer),1//2)
    for e in (-1,Inf,NaN,[1,1])
        @test_throws ArgumentError PR.epsilon_pruning(_pr_interval(0,4),e)
    end
    @test_throws ArgumentError PR.epsilon_pruning(_pr_interval(0,4),1;budget=IL.InterleavingBudget(max_hom_variables=1))
end

@testset "A107 asymmetric rectangle and nearby-module guarantee" begin
    for field in (CM.F2(),CM.F3(),CM.QQField())
        M=_pruning_box_fixture(;field)
        r=PR.epsilon_pruning(M,1//2)
        for x in (-1,0,1//2,1,7//2,4), y in (-1,-1//2,0,1,9//2,5)
            @test IL.ambient_dimension(r.pruned,(x,y))==Int(1//2<=x<7//2 && -1//2<=y<9//2)
        end
        c=PR.pruning_certificate(r)
        point=findfirst(==((QQ(1),QQ(1))),c.problem.points)
        eq=IL.interleaving_equations(c;vertex=point)
        @test eq.forward.lhs==eq.forward.rhs==ones(CM.coeff_type(field),1,1)
        @test eq.reverse.lhs==eq.reverse.rhs==ones(CM.coeff_type(field),1,1)
    end
    M=_pr_interval(0,4);N=IL.parameter_shift(M,-1//2)
    supplied=IL.find_interleaving(M,N,1//2).certificate
    @test supplied!==nothing
    r=PR.epsilon_pruning(M,1//2)
    g=PR.pruning_guarantee(r;against=supplied)
    @test g.bound==1 && g.reference===:source_and_compared_module
    @test g.pruning_distance_stability===:not_asserted
end

@testset "A107 independent pruning-pair oracle" begin
    # Enumerate ALL subspaces and ALL linear combinations of the supplied
    # matrices, then use the extremal defining conditions (no iteration).
    field=CM.F2();K=CM.coeff_type(field)
    spaces=[zeros(K,2,0),reshape(K[1,0],2,1),reshape(K[0,1],2,1),reshape(K[1,1],2,1),CM.eye(field,2)]
    contained(A,B)=FL.rank(field,hcat(A,B))==FL.rank(field,B)
    for code in 0:15, fcode in 0:15
        t=reshape(K[(code >> i)&1 for i in 0:3],2,2)
        f=reshape(K[(fcode >> i)&1 for i in 0:3],2,2)
        fs=[zeros(K,2,2),t,f,t+f]
        feasibleI=filter(I->all(g->contained(g*I,t*I),fs),spaces)
        expectedI=feasibleI[argmax(size.(feasibleI,2))]
        feasibleK=filter(spaces) do B
            contained(B,expectedI) || return false
            # Enumerating vectors checks the inverse-image condition directly.
            all(g->all(Iterators.product(0:1,0:1)) do v
                x=reshape(K[v...],2,1)
                !contained(x,expectedI) || !contained(t*x,g*B) || contained(x,B)
            end,fs)
        end
        expectedK=feasibleK[argmin(size.(feasibleK,2))]
        I,B,_,_=PR._pair(field,t,[t,f])
        @test contained(I,expectedI) && contained(expectedI,I)
        @test contained(B,expectedK) && contained(expectedK,B)
    end
end

@testset "A107 tamper checks and comparison provenance" begin
    M=_pr_interval(0,4)
    r=PR.epsilon_pruning(M,1)
    c=IL.find_interleaving(M,M,1).certificate
    @test PR.pruning_guarantee(r;against=c).reference===:source_and_compared_module
    @test_throws ArgumentError PR.pruning_guarantee(r;against=IL.find_interleaving(M,M,0).certificate)
    v=PR.check_pruning(r)
    @test v==PR.check_pruning(r)
    @test hash(v)==hash(PR.check_pruning(r))
    @test occursin("PruningResult",sprint(show,r))
    r.shift_map.comps[findfirst(!isempty,r.shift_map.comps)].*=0
    @test !PR.check_pruning(r).valid
    @test_throws ArgumentError PR.check_pruning(r;throw=true)
    @test_throws ArgumentError PR.pruning_certificate(r)
    @test TamerOp.epsilon_pruning===PR.epsilon_pruning
    @test TamerOp.Advanced.pruning_certificate===PR.pruning_certificate
    for name in (:epsilon_pruning,:pruned_module,:pruning_summary)
        @test name in names(TamerOp)
        @test getproperty(TamerOp.Advanced,name)===getproperty(PR,name)
    end
    Z=IL.RectilinearModule((Int[],),[0],Dict{Tuple{Int,Int},Matrix{QQ}}())
    zero_pruning=PR.epsilon_pruning(Z,1)
    small=_pr_interval(0,1)
    nearby=IL.find_interleaving(Z,small,1).certificate
    @test PR.pruning_guarantee(zero_pruning;against=nearby).bound==1
    @test_throws ArgumentError PR.epsilon_pruning(IL.encoding_module(M),1)
    @test_throws ArgumentError PR.epsilon_pruning(M,1;budget=IL.InterleavingBudget(max_matrix_entries=1))
end

@testset "A107 basis-equivariant retained subquotients" begin
    field=CM.F3();K=CM.coeff_type(field);M=_pruning_square_fixture(;field)
    A=IL.encoding_module(M)
    G=[d==2 ? K[1 1;0 1] : CM.eye(field,d) for d in A.dims]
    Gi=[FL.solve_fullcolumn(field,B,CM.eye(field,size(B,1))) for B in G]
    B=MD.PModule{K}(A.Q,copy(A.dims),Dict((u,v)=>G[v]*A.edge_maps[u,v]*Gi[u] for (u,v) in FF.cover_edges(A.Q));field)
    N=IL.RectilinearModule(B,EC.encoding_axes(M))
    r=PR.epsilon_pruning(M,1//2);s=PR.epsilon_pruning(N,1//2)
    rm=PR.pruning_maps(r);sm=PR.pruning_maps(s)
    fs=Matrix{K}[]
    for (v,x) in enumerate(PR._points(r.refined))
        change=G[EC.locate(M,x)]
        expected=change*rm.inclusion.comps[v]
        coords=FL.solve_fullcolumn(field,sm.inclusion.comps[v],expected)
        @test size(coords,1)==size(coords,2)==FL.rank(field,coords)
        expectedK=change*rm.inclusion.comps[v]*rm.kernel_inclusion.comps[v]
        actualK=sm.inclusion.comps[v]*sm.kernel_inclusion.comps[v]
        @test FL.rank(field,hcat(expectedK,actualK))==size(expectedK,2)==size(actualK,2)
        T=Matrix(transpose(FL.solve_fullcolumn(field,Matrix(transpose(rm.projection.comps[v])),
            Matrix(transpose(sm.projection.comps[v]*coords)))))
        @test FL.rank(field,T)==size(T,1)==size(T,2)
        push!(fs,T)
    end
    # Reidentify the equal finite grids only; retain the actual changed maps.
    Qr=IL.encoding_module(r.pruned);Qs=IL.encoding_module(s.pruned)
    target=MD.PModule{K}(Qr.Q,copy(Qs.dims),Dict(e=>Qs.edge_maps[e...] for e in FF.cover_edges(Qr.Q));field)
    @test MD.check_morphism(MD.PMorphism(Qr,target,fs)).valid
end

@testset "A107 a thick almost-decomposable module" begin
    for field in (CM.F2(),CM.F3(),CM.QQField())
        @info "A107 almost-decomposable fixture" field
        M=_pruning_square_fixture(;field)
        @test length(DF.basis(DF.Hom(IL.encoding_module(M),IL.encoding_module(M))))==1
        r=PR.epsilon_pruning(M,1//2)
        @test PR.check_pruning(r).valid
        @test r.hom_dimension>1
        # Independent formula on open cells: two copies of the square with its
        # upper-right corner removed. Probe off-diagonal points as well.
        for x in (-1,1//4,5//4,7//4,9//4,11//4,13//4,15//4,5),
            y in (-1,1//4,5//4,7//4,9//4,11//4,13//4,15//4,5)
            dim=3//2<x<7//2 && 3//2<y<7//2 && !(x>5//2 && y>5//2) ? 2 : 0
            @test IL.ambient_dimension(r.pruned,(x,y))==dim
        end
        @test IL.check_interleaving(PR.pruning_certificate(r)).valid
        # Explicit isomorphism to two copies of the same L support: transport
        # the two basis vectors from its least supported point, then verify
        # naturality and inverse identities on all stalks.
        P=IL.encoding_module(r.pruned);K=CM.coeff_type(field)
        root=EC.locate(r.pruned,(3//2,3//2))
        W=MD.PModule{K}(P.Q,copy(P.dims),Dict((u,v)=>(P.dims[u]==P.dims[v]==2 ? CM.eye(field,2) : zeros(K,P.dims[v],P.dims[u])) for (u,v) in FF.cover_edges(P.Q));field)
        fs=Matrix{K}[P.dims[v]==0 ? zeros(K,0,0) : Matrix(MD.map_leq(P,root,v)) for v in eachindex(P.dims)]
        @test MD.check_morphism(MD.PMorphism(W,P,fs)).valid
        for v in eachindex(P.dims)
            @test FL.solve_fullcolumn(field,fs[v],CM.eye(field,P.dims[v]))*fs[v]==CM.eye(field,P.dims[v])
        end
        # With closed terminal boundaries, every shifted Hom map must kill
        # the diagonal at (11/4,11/4), by naturality through (3,3)->(4,4).
        # Its pruning therefore retains a one-dimensional corner here.
        closed=PR.epsilon_pruning(_pruning_square_fixture(;field,closed=true),1//2)
        @test IL.ambient_dimension(closed.pruned,(11//4,11//4))==1
        # The actual pruned maps are two independent copies, not just matching
        # Hilbert values. Every nonzero-to-nonzero comparable map is invertible.
        for (u,v) in FF.cover_edges(IL.encoding_module(r.pruned).Q)
            Q=IL.encoding_module(r.pruned)
            Q.dims[u]==Q.dims[v]==2 || continue
            @test FL.rank(field,Q.edge_maps[u,v])==2
        end
    end
end
