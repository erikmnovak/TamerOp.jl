# Independent interval-sum and matrix oracles for selected stalk propagation.
function _a104_squares(field)
    K=CM.coeff_type(field);P=FF.ProductOfChainsPoset((4,4))
    points=[(x,y) for y in 0:3 for x in 0:3]
    active=[Int[i for (i,b) in enumerate((0<=x<2 && 1<=y<3,1<=x<3 && 0<=y<2)) if b] for (x,y) in points]
    dims=length.(active)
    maps=Dict{Tuple{Int,Int},Matrix{K}}((u,v)=>K[a==b for a in active[v],b in active[u]] for (u,v) in FF.cover_edges(P))
    M=MD.PModule{K}(P,dims,maps;field)
    enc=RES.EncodingResult(P,M,EC.GridEncodingMap(P,(collect(0:3),collect(0:3))))
    return (;P,M,enc,active)
end

@testset "A104 selected vectors, subspaces and independent interval oracles" begin
    AC=TamerOp.AbelianCategories;V=TamerOp.Visualization
    for field in (FIELD_QQ,CM.F2(),CM.F3(),CM.Fp(101),CM.RealField(Float64))
        K=CM.coeff_type(field);a=_a104_squares(field)
        for X in (reshape(K[1,0],2,1),reshape(K[0,1],2,1),reshape(K[1,1],2,1),
                  K[1 0;0 1],K[1 2 0;1 2 0],zeros(K,2,0),zeros(K,2,1))
            r=TamerOp.propagate(a.enc;source=6,vectors=X)
            @test AC.check_propagation(r).report.valid
            @test TamerOp.ambient_module(r) === a.M
            @test TamerOp.propagation_summary(r).encoding===a.enc
            @test TamerOp.describe(r).summand===:not_certified
            @test TamerOp.dimensions(r)==r.image_dimensions
            S=AC.generated_submodule(r);N=AC.submodule_object(S);i=AC.inclusion_map(S)
            Q,q=AC.quotient_with_projection(S)
            @test AC.check_submodule(S).valid
            @test MD.check_morphism(q).valid
            for t in 1:16
                expected=FF.leq(a.P,6,t) ? X[a.active[t],:] : nothing
                @test AC.propagation_image(r;target=t)==expected
                if expected===nothing
                    @test r.image_dimensions[t]===nothing
                    @test N.dims[t]==0
                else
                    rank=FL.rank(field,expected)
                    @test r.image_dimensions[t]==rank==N.dims[t]
                    @test FL.rank(field,hcat(i.comps[t],expected))==rank
                end
                @test Q.dims[t]+N.dims[t]==a.M.dims[t]
                @test iszero(q.comps[t]*i.comps[t])
            end
            for (u,v) in FF.cover_edges(a.P)
                A=MD.structure_map(a.M;source=u,target=v)
                @test i.comps[v]*MD.structure_map(N;source=u,target=v) == A*i.comps[u]
                @test MD.structure_map(Q;source=u,target=v)*q.comps[u] == q.comps[v]*A
            end
        end
        r=AC.propagate(a.M;source=6,vectors=K[1,1])
        @test AC.propagation_summary(r).nonzero_support==[6,7,10]
        # Explicit independent change of coordinates at every stalk.
        C=[d==2 ? K[1 1;0 1] : CM.eye(field,d) for d in a.M.dims]
        invC=[d==2 ? K[1 -1;0 1] : CM.eye(field,d) for d in a.M.dims]
        N=MD.PModule{K}(a.P,copy(a.M.dims),Dict(e=>C[e[2]]*a.M.edge_maps[e[1],e[2]]*invC[e[1]] for e in FF.cover_edges(a.P));field)
        changed=AC.propagate(N;source=6,vectors=C[6]*K[1,1])
        @test changed.image_dimensions==r.image_dimensions
        for t in 1:16
            r.images[t]===nothing && continue
            @test changed.images[t]==C[t]*r.images[t]
        end
        for target in (1,6,7,10,11)
            spec=V.visual_spec(r;target)
            @test V.check_visual_spec(spec).valid
            @test spec.metadata.selected.defined==FF.leq(a.P,6,target)
            @test spec.metadata.summand===:not_certified
            if spec.metadata.selected.defined
                @test spec.metadata.selected.matrix*r.vectors==spec.metadata.selected.images
            end
        end
        spec=V.visual_spec(TamerOp.propagate(a.enc;source=6,vectors=K[1,1]);target=7,box=((-1,-1),(4,4)))
        @test V.check_visual_spec(spec).valid
        @test only(filter(p->p.kind===:propagation_regions,spec.panels)).metadata.scope===:finite_submodule_pullback
    end
end

@testset "A104 fields, nonsplitting, arbitrary labels and invalid contracts" begin
    AC=TamerOp.AbelianCategories;V=TamerOp.Visualization
    for field in (FIELD_QQ,CM.F2(),CM.F3(),CM.Fp(101),CM.RealField(Float64))
        K=CM.coeff_type(field);P=chain_poset(2)
        M=MD.PModule{K}(P,[2,1],Dict((1,2)=>reshape(K[1,1],1,2));field)
        r=AC.propagate(M;source=1,vectors=[1,1])
        @test r.images[2]==reshape(K[2],1,1)
        @test r.image_dimensions==[1,field isa CM.PrimeField && field.p==2 ? 0 : 1]
        C=MD.PModule{K}(P,[1,1],Dict((1,2)=>ones(K,1,1));field)
        tail=AC.generated_submodule(AC.propagate(C;source=2,vectors=[1]))
        @test AC.submodule_object(tail).dims==[0,1]
        # Any retraction must be identity at 2 and zero at 1; its square fails.
        f=MD.PMorphism(C,AC.submodule_object(tail),[zeros(K,0,1),ones(K,1,1)])
        @test !MD.check_morphism(f).valid
        zero=MD.PModule{K}(P,[0,0],Dict((1,2)=>zeros(K,0,0));field)
        z=AC.propagate(zero;source=1,vectors=K[])
        @test size(z.images[2])==(0,1)
        @test AC.check_submodule(AC.generated_submodule(z)).valid
        # Vertex identifiers need not be a topological ordering.
        R=FF.FinitePoset(Bool[1 0 0;1 1 1;1 0 1])
        N=MD.PModule{K}(R,ones(Int,3),Dict((2,3)=>fill(K(2),1,1),(3,1)=>fill(K(3),1,1));field)
        s=AC.propagate(N;source=2,vectors=[1])
        @test s.images[1]==fill(K(6),1,1)
    end
    a=_a104_squares(FIELD_QQ);r=AC.propagate(a.M;source=6,vectors=[1,1])
    @test_throws ArgumentError AC.propagate(a.M;source=true,vectors=[1,1])
    @test_throws ArgumentError AC.propagate(a.M;source=0,vectors=[1,1])
    @test_throws DimensionMismatch AC.propagate(a.M;source=6,vectors=[1])
    @test_throws ArgumentError AC.propagate(a.M;source=6,vectors=1)
    @test_throws ArgumentError AC.propagation_image(r;target=17)
    @test_throws ArgumentError V.visual_spec(r;max_vertices=2)
    @test_throws ArgumentError V.visual_spec(r;target=0)
    @test_throws ArgumentError V.visual_spec(r;matrix_limit=(0,2))
    @test_throws ArgumentError V.visual_spec(r;box=((0,0),(4,4)))
    @test_throws ArgumentError V.visual_spec(r;unknown=1)
    saved=AC.propagation_image(r;target=6);saved[1]=999
    @test r.images[6][1]==1
    bad=deepcopy(r);bad.images[7][1]=3
    @test !AC.check_propagation(bad).report.valid
    @test_throws ErrorException AC.check_propagation(bad;throw=true)
    @test_throws ErrorException AC.generated_submodule(bad)
    real=_a104_squares(CM.RealField(Float64))
    @test_throws ArgumentError AC.propagate(real.M;source=6,vectors=[Inf,1])
    @test TOA.VectorPropagation===AC.VectorPropagation
    @test TOA.propagation_image===AC.propagation_image
end

@testset "A104 native selected-vector exports" begin
    if Base.find_package("CairoMakie")===nothing
        @test_skip false
    else
        @eval using CairoMakie
        a=_a104_squares(FIELD_QQ);r=TamerOp.propagate(a.enc;source=6,vectors=[1,1])
        mktempdir() do dir
            for ext in ("svg","png","pdf")
                path=joinpath(dir,"propagation."*ext)
                TamerOp.Visualization.save_visual(path,r;target=7,box=((-1,-1),(4,4)))
                @test filesize(path)>100
            end
        end
    end
end

@testset "A104 explicit retained source homology basis" begin
    I=TamerOp.DataIngestion;AC=TamerOp.AbelianCategories
    numerical=AC.propagate(_a104_squares(CM.RealField(Float64)).M;source=6,vectors=[1,1])
    @test_throws r"require exact" I.propagation_source_chain(numerical;target=6,basis=I.SourceChain[])
    for field in (FIELD_QQ,CM.F2(),CM.F3())
        K=CM.coeff_type(field)
        G=TamerOp.DataTypes.GradedComplex([[1,2],[1]],[sparse(reshape([-1,1],2,1))],[(0,),(0,),(1,)])
        x=I.filtered_complex(G;geometry=[[(0,0)],[(1,0)],[(0,0),(1,0)]])
        c=I.source_correspondence(x;field)
        b=[I.lift_source_chain(c;dim=0,coefficients=v,parameter=(0,)) for v in ([1,0],[0,1])]
        P=chain_poset(2);M=MD.PModule{K}(P,[2,1],Dict((1,2)=>reshape(K[1,1],1,2));field)
        r=AC.propagate(M;source=1,vectors=[1,-1])
        source=I.propagation_source_chain(r;target=1,basis=b)
        @test source.coefficients==K[1,-1]
        @test I.check_source_chain(source).valid
        @test source.metadata.basis_identification===:supplied_by_caller
        later=[I.lift_source_chain(c;dim=0,coefficients=[1,0],parameter=(1,))]
        @test iszero(I.propagation_source_chain(r;target=2,basis=later).coefficients)
        @test_throws ArgumentError I.propagation_source_chain(r;target=1,basis=[b[1],b[1]])
        @test_throws ArgumentError I.propagation_source_chain(r;target=1,basis=b,column=2)
        @test_throws DimensionMismatch I.propagation_source_chain(r;target=2,basis=b)
        @test_throws ArgumentError I.propagation_source_chain(r;target=1,basis=I.SourceChain[])
    end
end

@testset "A104 path certificates, one square and subspace basis changes" begin
    AC=TamerOp.AbelianCategories
    for field in (FIELD_QQ,CM.F2(),CM.F3(),CM.RealField(Float64))
        K=CM.coeff_type(field);P=FF.ProductOfChainsPoset((2,2))
        maps=Dict((u,v)=>ones(K,1,1) for (u,v) in FF.cover_edges(P))
        maps[(3,4)]=zeros(K,1,1)
        invalid=MD.PModule{K}(P,ones(Int,4),maps;field)
        @test_throws ArgumentError AC.propagate(invalid;source=1,vectors=[1])
        Q=FF.ProductOfChainsPoset((3,3));dims=[1,1,0,1,1,0,0,0,0]
        square=MD.PModule{K}(Q,dims,Dict((u,v)=>ones(K,dims[v],dims[u]) for (u,v) in FF.cover_edges(Q));field)
        one=AC.propagate(square;source=1,vectors=[1])
        @test AC.propagation_summary(one).nonzero_support==[1,2,4,5]
        @test AC.submodule_object(AC.generated_submodule(one)).dims==dims
        a=_a104_squares(field)
        for (X,Y) in ((K[1 0;0 1],K[1 1;0 1]),(reshape(K[1,1],2,1),K[1 0;1 0]))
            r=AC.propagate(a.M;source=6,vectors=X);s=AC.propagate(a.M;source=6,vectors=Y)
            @test r.image_dimensions==s.image_dimensions
            I=AC.inclusion_map(AC.generated_submodule(r));J=AC.inclusion_map(AC.generated_submodule(s))
            for q in 1:16
                @test FL.rank(field,hcat(I.comps[q],J.comps[q]))==I.dom.dims[q]==J.dom.dims[q]
            end
        end
    end
    F=CM.RealField(Float64;atol=1e-8,rtol=1e-8)
    P=FF.ProductOfChainsPoset((2,2));maps=Dict((u,v)=>ones(1,1) for (u,v) in FF.cover_edges(P))
    maps[(3,4)]=fill(1+1e-10,1,1)
    M=MD.PModule{Float64}(P,ones(Int,4),maps;field=F)
    @test AC.check_propagation(AC.propagate(M;source=1,vectors=[1])).report.valid
    maps[(3,4)]=fill(1.1,1,1)
    N=MD.PModule{Float64}(P,ones(Int,4),maps;field=F)
    @test_throws ArgumentError AC.propagate(N;source=1,vectors=[1])
end

@testset "A104 moderate chains and provenance" begin
    AC=TamerOp.AbelianCategories;rng=MersenneTwister(104)
    for field in (FIELD_QQ,CM.F2(),CM.F3(),CM.Fp(101),CM.RealField(Float64))
        K=CM.coeff_type(field);P=chain_poset(12);dims=rand(rng,0:4,12)
        maps=Dict((u,u+1)=>K.(rand(rng,-2:2,dims[u+1],dims[u])) for u in 1:11)
        M=MD.PModule{K}(P,dims,maps;field)
        for p in (1,5,12)
            B=K.(rand(rng,-2:2,dims[p],3));r=AC.propagate(M;source=p,vectors=B)
            Y=copy(B)
            for q in p:12
                q==p || (Y=maps[(q-1,q)]*Y)
                @test AC.propagation_image(r;target=q)==Y
            end
            @test AC.check_propagation(r).report.valid
            @test TamerOp.provenance(r).base_poset===P
            @test TamerOp.provenance(r).source_coordinates==B
        end
    end
    a=_a104_squares(FIELD_QQ);r=TamerOp.propagate(a.enc;source=6,vectors=[1,1])
    @test TamerOp.provenance(r).encoding==TamerOp.provenance(a.enc)
    wrong=AC.VectorPropagation(r.ambient,r.source,r.vectors,r.images,r.image_dimensions,
        (;encoding=_a104_squares(FIELD_QQ).enc,basis_convention=:ambient_stalk_coordinates))
    @test !AC.check_propagation(wrong).report.valid
end
