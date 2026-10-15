# Independent dense elimination and explicit topological/product oracles.
function _cup_oracle_rank(A)
    B=copy(A);row=1
    for col in axes(B,2)
        row>size(B,1) && break
        pivot=findfirst(i->!iszero(B[i,col]),row:size(B,1))
        pivot===nothing && continue
        pivot+=row-1;B[row,:],B[pivot,:]=copy(B[pivot,:]),copy(B[row,:])
        B[row,:].*=inv(B[row,col])
        for i in axes(B,1)
            i==row && continue
            B[i,:].-=B[i,col].*B[row,:]
        end
        row+=1
    end
    row-1
end

function _cup_torus()
    vertex(i,j)=mod(i,3)+3mod(j,3)+1
    triangles=Vector{Int}[]
    for i in 0:2,j in 0:2
        a=vertex(i,j);b=vertex(i+1,j);c=vertex(i,j+1);d=vertex(i+1,j+1)
        push!(triangles,[a,b,d],[a,c,d])
    end
    triangles
end
_cup_wedge()=[[1,2],[2,3],[1,3],[1,4],[4,5],[1,5],
    [1,6,7],[1,6,8],[1,7,8],[6,7,8]]
_cup_rp2()=[[1,2,3],[1,2,4],[1,3,5],[1,4,6],[1,5,6],
    [2,3,6],[2,4,5],[2,5,6],[3,4,5],[3,4,6]]

function _cup_exterior(F;zero_product=false)
    CP=TamerOp.CohomologyProducts;K=CM.coeff_type(F)
    tables=Dict{Tuple{Int,Int},Matrix{K}}()
    for (d,n) in enumerate([1,2,1])
        tables[(0,d-1)]=Matrix{K}(I,n,n);tables[(d-1,0)]=Matrix{K}(I,n,n)
    end
    tables[(1,1)]=zero_product ? zeros(K,1,4) : reshape(K[0,-1,1,0],1,4)
    CP.cohomology_algebra([1,2,1],tables;unit=[1],field=F)
end

@testset "A108 supplied graded algebras and exact contracts" begin
    CP=TamerOp.CohomologyProducts
    for F in (CM.QQField(),CM.F2(),CM.F3(),CM.Fp(5),CM.Fp(101))
        K=CM.coeff_type(F);A=_cup_exterior(F);B=_cup_exterior(F;zero_product=true)
        @test CP.dimensions(A)==CP.dimensions(B)==[1,2,1]
        @test CP.cup_length(A)==2 && CP.cup_length(B)==1
        @test CP.cup_product(A,[1,0],[0,1];left_degree=1,right_degree=1)==K[1]
        @test CP.cup_product(A,[0,1],[1,0];left_degree=1,right_degree=1)==K[-1]
        @test CP.cup_product(A,[1,1],[1,1];left_degree=1,right_degree=1)==K[0]
        @test CP.check_cohomology_algebra(A).valid
        @test CP.cup_unit(A)==K[1]
        @test CP.cup_product_matrix(A;left_degree=1,right_degree=1)==reshape(K[0,-1,1,0],1,4)
        @test isempty(CP.cup_product(A,[1],[1];left_degree=2,right_degree=2))
        @test_throws ArgumentError CP.cup_product(A,[1],[1];left_degree=1,right_degree=1)
        @test_throws ArgumentError CP.cup_product(A,[1.0,0.0],[1,0];left_degree=1,right_degree=1)
        @test_throws ArgumentError CP.cochain_cup(A,[1],[1];left_degree=0,right_degree=0)
        @test_throws ArgumentError CP.cohomology_algebra([1],Dict((0,0)=>ones(Int,1,1));unit=[0],field=F)
        @test_throws ArgumentError CP.cohomology_algebra([1],Dict((0,0)=>ones(Int,2,1));unit=[1],field=F)
        @test_throws ArgumentError CP.check_cohomology_algebra(A;budget=CP.CupProductBudget(max_products=0))
        damaged=deepcopy(A);damaged.unit[1]=zero(K)
        r=CP.check_cohomology_algebra(damaged)
        @test !r.valid && !isempty(r.issues)
        @test r==deepcopy(r) && hash(r)==hash(deepcopy(r))
        @test_throws ArgumentError CP.check_cohomology_algebra(damaged;throw=true)
        Z=CP.cohomology_algebra([0],Dict();unit=Int[],field=F)
        @test CP.cup_length(Z)==0 && CP.check_cohomology_algebra(Z).valid
        @test occursin("CohomologyAlgebra",sprint(show,A))
        @test CP.describe(A).source===:supplied_algebra
        if F!=CM.F2()
            @test_throws ArgumentError CP.cup_product(A,CM.coeff_type(CM.F2())[1,0],[0,1];left_degree=1,right_degree=1)
        end
    end
    # Graded commutativity is not sufficient: x*x=y, x*y=0, y*y=z
    # violates (x*x)*y=x*(x*y), with all classes in even degrees.
    ds=[1,0,1,0,1,0,0,0,1];tables=Dict{Tuple{Int,Int},Matrix{Int}}()
    for d in 0:8;tables[(0,d)]=Matrix{Int}(I,ds[d+1],ds[d+1]);tables[(d,0)]=copy(tables[(0,d)]);end
    tables[(2,2)]=ones(Int,1,1);tables[(4,4)]=ones(Int,1,1)
    @test_throws ArgumentError CP.cohomology_algebra(ds,tables;unit=[1])
    @test_throws ArgumentError CP.cohomology_algebra([1],Dict();unit=[1],field=CM.RealField(Float64))
    @test_throws ArgumentError CP.CupProductBudget(max_cells=-1)
end

@testset "A108 simplicial rings, signs, and independent cochain oracles" begin
    CP=TamerOp.CohomologyProducts
    for F in (CM.QQField(),CM.F2(),CM.F3(),CM.Fp(5))
        K=CM.coeff_type(F)
        A=CP.cohomology_algebra(_cup_torus();field=F)
        B=CP.cohomology_algebra(_cup_wedge();field=F)
        @test CP.dimensions(A)==CP.dimensions(B)==[1,2,1]
        @test CP.cup_length(A)==2 && CP.cup_length(B)==1
        @test _cup_oracle_rank(CP.cup_product_matrix(A;left_degree=1,right_degree=1))==1
        @test all(iszero,CP.cup_product_matrix(B;left_degree=1,right_degree=1))
        @test CP.check_cohomology_algebra(A).valid
        for X in (A,B),p in 0:2,q in 0:2-p
            L=CP.cohomology_representatives(X;degree=p)
            R=CP.cohomology_representatives(X;degree=q)
            O=CP.cohomology_representatives(X;degree=p+q)
            @test all(iszero,L.coboundary*L.representatives)
            @test _cup_oracle_rank(hcat(L.coboundaries,L.representatives))==size(L.coboundaries,2)+size(L.representatives,2)
            for i in axes(L.representatives,2),j in axes(R.representatives,2)
                x=L.representatives[:,i];y=R.representatives[:,j]
                actual=CP.cochain_cup(X,x,y;left_degree=p,right_degree=q)
                # Independently assemble from face pairs, not front/back lookup.
                oracle=zeros(K,length(O.cells))
                for (a,s) in enumerate(L.cells),(b,t) in enumerate(R.cells)
                    last(s)==first(t) || continue
                    merged=Tuple([collect(s);collect(t)[2:end]])
                    k=findfirst(==(merged),O.cells);k===nothing && continue
                    oracle[k]+=x[a]*y[b]
                end
                @test actual==oracle
                coord=CP.cohomology_coordinates(X,actual;degree=p+q)
                @test _cup_oracle_rank(hcat(O.coboundaries,actual-O.representatives*coord))==size(O.coboundaries,2)
                ex=zeros(K,CP.dimensions(X)[p+1]);ex[i]=one(K)
                ey=zeros(K,CP.dimensions(X)[q+1]);ey[j]=one(K)
                @test coord==CP.cup_product(X,ex,ey;left_degree=p,right_degree=q)
                # Products do not depend on adding coboundaries to either factor.
                xp=copy(x);yp=copy(y)
                size(L.coboundaries,2)>0 && (xp+=L.coboundaries[:,1])
                size(R.coboundaries,2)>0 && (yp+=R.coboundaries[:,end])
                @test CP.cohomology_coordinates(X,CP.cochain_cup(X,xp,yp;left_degree=p,right_degree=q);degree=p+q)==coord
            end
        end
        P=CP.cohomology_algebra(_cup_rp2();field=F)
        @test CP.dimensions(P)==(F==CM.F2() ? [1,1,1] : [1,0,0])
        @test CP.cup_length(P)==(F==CM.F2() ? 2 : 0)
        F==CM.F2() && @test CP.cup_product(P,[1],[1];left_degree=1,right_degree=1)==K[1]
        # Relabeling vertex order changes cochain choices, not ring invariants.
        relabeled=CP.cohomology_algebra([[10-v for v in s] for s in _cup_torus()];field=F)
        @test CP.dimensions(relabeled)==CP.dimensions(A)
        @test CP.cup_length(relabeled)==2
        E=CP.cohomology_algebra(Vector{Int}[];field=F)
        @test CP.dimensions(E)==[0] && CP.cup_length(E)==0
        pts=CP.cohomology_algebra([[1],[2],[3]];field=F)
        @test CP.dimensions(pts)==[3] && CP.cup_length(pts)==0
        disk=CP.cohomology_algebra([[1,2,3]];field=F)
        # Non-cocycles cannot be projected to a cohomology class.
        @test_throws ArgumentError CP.cohomology_coordinates(disk,K[1,0,0];degree=0)
        # Cochain Leibniz rule in nonbinary fields, before quotienting.
        L=CP.cohomology_representatives(disk;degree=0);R=CP.cohomology_representatives(disk;degree=1)
        x=K[1,2,3];y=K[2,1,4]
        z=CP.cochain_cup(disk,x,y;left_degree=0,right_degree=1)
        @test R.coboundary*z==CP.cochain_cup(disk,L.coboundary*x,y;left_degree=1,right_degree=1)+
            CP.cochain_cup(disk,x,R.coboundary*y;left_degree=0,right_degree=2)
    end
    @test_throws ArgumentError CP.cohomology_algebra([[1,1]])
    @test_throws ArgumentError CP.cohomology_algebra([[0,1]])
    @test_throws ArgumentError CP.cohomology_algebra([[1,2,3]];budget=CP.CupProductBudget(max_cells=6))
    @test_throws ArgumentError CP.cohomology_algebra([[1,2,3]];budget=CP.CupProductBudget(max_entries=2))
end

@testset "A108 ring persistence, interval cup-length and degree partitions" begin
    CP=TamerOp.CohomologyProducts
    for F in (CM.QQField(),CM.F2(),CM.F3(),CM.Fp(101)),order in (:sublevel,:superlevel)
        K=CM.coeff_type(F);A=_cup_exterior(F)
        identity=Dict((d-1)=>Matrix{K}(I,n,n) for (d,n) in enumerate(CP.dimensions(A)))
        # Endpoint cup-lengths equal 2, but the restriction kills positive degrees.
        killed=Dict(0=>ones(K,1,1),1=>zeros(K,2,2),2=>zeros(K,1,1))
        ts=order===:sublevel ? [0//1,1//3,7//2] : [7//2,1//3,0//1]
        R=CP.persistent_cup_products([A,A,A];parameters=ts,restrictions=[identity,killed],order)
        @test CP.check_persistent_cup_products(R).valid
        @test CP.cup_parameters(R)==ts
        @test CP.cup_length(R;source=1,target=1)==CP.cup_length(R;source=3,target=3)==2
        @test CP.cup_length(R;source=2,target=1)==2
        @test CP.cup_length(R;source=3,target=1)==0
        @test CP.cohomology_restriction(R;source=3,target=1,degree=1)==zeros(K,2,2)
        @test TamerOp.Modules.structure_map(CP.cohomology_module(R;degree=1);source=2,target=1)==Matrix{K}(I,2,2)
        V=CP.cup_module(R);P=CP.partition_module(R,[1,1])
        @test CP.dimensions(V)==CP.dimensions(P)==[1,1,1]
        @test CP.check_cup_module(V).valid && CP.check_cup_module(P).valid
        @test CP.cup_intervals(V)==CP.cup_intervals(P)
        bars=CP.cup_intervals(V)
        @test Set((b.first_stage,b.last_stage) for b in bars)==Set([(1,2),(3,3)])
        @test Set((b.birth,b.death) for b in bars)==Set([(ts[1],ts[3]),(ts[3],nothing)])
        @test all(b->b.birth_included && !b.death_included && b.order===order,bars)
        for factors in 1:4
            M=CP.cup_module(R;factors);bs=CP.cup_intervals(M)
            for s in 1:3,t in 1:s
                @test _cup_oracle_rank(CP.structure_map(M;source=s,target=t))==count(b->b.first_stage<=t<=s<=b.last_stage,bs)
            end
        end
        @test CP.dimensions(CP.cup_module(R;degree=1))==[0,0,0]
        @test CP.dimensions(CP.partition_module(R,[2]))==[1,1,1]
        @test CP.dimensions(CP.partition_module(R,[1,2]))==[0,0,0]
        @test_throws ArgumentError CP.cohomology_restriction(R;source=1,target=2,degree=1)
        @test_throws ArgumentError CP.cup_module(R;factors=0)
        @test_throws ArgumentError CP.partition_module(R,Int[])
        @test_throws ArgumentError CP.partition_module(R,[0,1])
        @test_throws ArgumentError CP.cup_module(R;degree=0)
        @test_throws ArgumentError CP.cup_length(R;source=true)
        @test_throws ArgumentError CP.persistent_cup_products([A,A];parameters=[0,0],restrictions=[identity])
        invalid=deepcopy(identity);invalid[2].=zero(K)
        @test_throws ArgumentError CP.persistent_cup_products([A,A];parameters=[0,1],restrictions=[invalid])
        @test_throws ArgumentError CP.persistent_cup_products([A,A];parameters=[0,1],restrictions=[Dict()])
        @test_throws ArgumentError CP.persistent_cup_products([A];parameters=[0],restrictions=[],budget=CP.CupProductBudget(max_stages=0))
        @test occursin("CupModule",sprint(show,V)) && occursin("PersistentCupProducts",sprint(show,R))
        @test CP.cup_product_summary(V).provenance.variance===:contravariant
        changed=CP.cohomology_algebra(R;stage=1);changed.unit[1]=zero(K)
        @test CP.check_persistent_cup_products(R).valid
    end
    # F2 polynomial algebra F2[x]/x^4 tests repeated factors and nonzero cubes.
    tables=Dict((p,q)=>ones(Int,1,1) for p in 0:3 for q in 0:3-p)
    A=CP.cohomology_algebra(ones(Int,4),tables;unit=[1])
    R=CP.persistent_cup_products([A];parameters=[2//3],restrictions=[])
    @test CP.cup_length(A)==CP.cup_length(R)==3
    @test [only(CP.dimensions(CP.cup_module(R;factors=k))) for k in 1:4]==[3,2,1,0]
    @test only(CP.dimensions(CP.partition_module(R,[1,1,1])))==1
    @test only(CP.dimensions(CP.partition_module(R,[1,2])))==1
    @test only(CP.dimensions(CP.cup_module(R;factors=2,degree=3)))==1
end

@testset "A108 filtered simplicial sources and additive reduction comparison" begin
    CP=TamerOp.CohomologyProducts;OP=TamerOp.OrdinaryPersistence;DT=TamerOp.DataTypes
    triangles=_cup_torus();cone=[[10;s] for s in triangles]
    edges=unique([sort(s[[i,j]]) for s in triangles for i in 1:3 for j in i+1:3])
    cells=[edges;triangles;cone];grades=[fill(0//1,length(edges));fill(1//3,length(triangles));fill(5//2,length(cone))]
    for F in (CM.F2(),CM.F3(),CM.Fp(5)),order in (:sublevel,:superlevel)
        gs=order===:sublevel ? grades : -grades
        R=CP.persistent_cup_products(cells;grades=gs,field=F,order)
        @test CP.cup_parameters(R)==(order===:sublevel ? [0//1,1//3,5//2] : [0//1,-1//3,-5//2])
        @test CP.check_persistent_cup_products(R).valid
        V=CP.cup_module(R);bar=only(CP.cup_intervals(V))
        @test CP.dimensions(V)==[0,1,0]
        @test (bar.birth,bar.death)==(order===:sublevel ? (1//3,5//2) : (-1//3,-5//2))
        @test CP.check_cup_module(V).valid
        @test [CP.cup_length(R;source=i) for i in 1:3]==[1,2,0]
        # Compare additive persistence to the independently reduced signed boundary complex.
        final=CP.cohomology_algebra(R;stage=3)
        reps=[CP.cohomology_representatives(final;degree=d) for d in 0:3]
        earliest(s)=order===:sublevel ? minimum(gs[i] for i in eachindex(cells) if issubset(s,cells[i])) :
            maximum(gs[i] for i in eachindex(cells) if issubset(s,cells[i]))
        cg=[(earliest(s),) for r in reps for s in r.cells]
        # Oriented integer boundary reconstructed independently from ordered faces.
        Bs=SparseMatrixCSC{Int,Int}[]
        for d in 1:3
            B=zeros(Int,length(reps[d].cells),length(reps[d+1].cells))
            for (j,s) in enumerate(reps[d+1].cells),(i,t) in enumerate(reps[d].cells)
                issubset(t,s) || continue
                k=findfirst(==(only(setdiff(s,t))),s);B[i,j]=(-1)^(k-1)
            end
            push!(Bs,sparse(B))
        end
        G=DT.GradedComplex([collect(1:length(r.cells)) for r in reps],Bs,cg)
        diag=OP.persistence_diagram(G;field=F,order,cocycles=true)
        for d in 1:3
            bs=CP.cup_intervals(CP.cup_module(R;factors=1,degree=d))
            @test sort([(b.birth,b.death) for b in bs if b.death!==nothing])==sort(OP.finite_intervals(diag;dim=d))
            @test sort([b.birth for b in bs if b.death===nothing])==sort(OP.essential_births(diag;dim=d))
        end
        # A117 representatives may be multiplied only after identifying source cells.
        a=CP.cohomology_algebra(R;stage=2);s=CP.cohomology_representatives(a;degree=1)
        persistent=Vector{CM.coeff_type(F)}[]
        for i in eachindex(OP.finite_intervals(diag;dim=1))
            b,death=OP.finite_intervals(diag;dim=1)[i]
            active=order===:sublevel ? b<=gs[length(edges)+1]<death : death<gs[length(edges)+1]<=b
            active || continue
            c=OP.persistence_cocycle(diag;dim=1,scale=gs[length(edges)+1],index=i).cochain
            v=zeros(CM.coeff_type(F),length(s.cells))
            for (j,k) in zip(c.cell_indices,c.coefficients)
                index=findfirst(==(reps[2].cells[j]),s.cells);index===nothing || (v[index]=CM.coerce(F,k))
            end
            push!(persistent,CP.cohomology_coordinates(a,v;degree=1))
        end
        @test length(persistent)==2 && _cup_oracle_rank(hcat(persistent...))==2
        @test any(!iszero,CP.cup_product(a,persistent[1],persistent[2];left_degree=1,right_degree=1))
    end
    @test_throws ArgumentError CP.persistent_cup_products([[1,2]];grades=[Inf])
    @test_throws ArgumentError CP.persistent_cup_products([[1,2]];grades=[1],order=:wrong)
    @test_throws ArgumentError CP.persistent_cup_products([[1],[2]];grades=[1,2],budget=CP.CupProductBudget(max_stages=1))
end

@testset "A108 curated API exposure" begin
    CP=TamerOp.CohomologyProducts
    for s in (:cohomology_algebra,:persistent_cup_products,:cup_product,:cup_module,:partition_module,:cup_length,:cup_intervals,:cup_product_summary)
        @test getproperty(TamerOp,s)===getproperty(CP,s)
    end
    for s in (:CohomologyAlgebra,:PersistentCupProducts,:CupModule,:CupProductBudget,
              :CupProductValidationSummary,:check_cohomology_algebra,:check_persistent_cup_products,
              :check_cup_module,:cup_unit,:cup_product_matrix,:cochain_cup,:cohomology_coordinates,
              :cohomology_representatives,:cohomology_restriction,:cohomology_module,:cup_parameters,:cup_poset,:cup_embedding,:cup_pmodule)
        @test getproperty(TOA,s)===getproperty(CP,s)
    end
end

@testset "A108 basis changes, product subspaces, and damaged results" begin
    CP=TamerOp.CohomologyProducts
    for F in (CM.QQField(),CM.F2(),CM.F3(),CM.Fp(101))
        K=CM.coeff_type(F);A=_cup_exterior(F)
        for a in 1:4
            # Deliberately change both H1 and H2 coordinates, including signs.
            Ts=[ones(K,1,1),K[1 a;0 1],reshape(K[-1],1,1)]
            Is=[ones(K,1,1),K[1 -a;0 1],reshape(K[-1],1,1)]
            tables=Dict((p,q)=>Is[p+q+1]*CP.cup_product_matrix(A;left_degree=p,right_degree=q)*kron(Ts[q+1],Ts[p+1])
                for p in 0:2 for q in 0:2-p)
            B=CP.cohomology_algebra(CP.dimensions(A),tables;unit=[1],field=F)
            R=CP.persistent_cup_products([A,B];parameters=[0,1],restrictions=[Dict(d=>Ts[d+1] for d in 0:2)])
            @test CP.cup_length(B)==2 && CP.cup_length(R;source=2,target=1)==2
            for V in (CP.cup_module(R),CP.cup_module(R;factors=1),CP.partition_module(R,[1,1]))
                @test CP.check_cup_module(V).valid
                @test _cup_oracle_rank(CP.structure_map(V;source=2,target=1))==first(CP.dimensions(V))
                E=CP.cup_embedding(V;stage=1);L=CP.cup_embedding(V;stage=2)
                ambient=zeros(K,4,4);ambient[1,1]=1;ambient[2:3,2:3]=Ts[2];ambient[4,4]=-one(K)
                @test E*CP.structure_map(V;source=2,target=1)==ambient*L
                @test all(b->b.first_stage==1 && b.last_stage==2,CP.cup_intervals(V))
                owned=CP.cup_pmodule(V)
                @test owned.dims==CP.dimensions(V)
                @test TamerOp.Modules.structure_map(owned;source=2,target=1)==CP.structure_map(V;source=2,target=1)
                owned.dims[1]+=1
                @test owned.dims!=CP.dimensions(V)
            end
        end
        R=CP.persistent_cup_products([A,A];parameters=[0,1],restrictions=[Dict(d=>Matrix{K}(I,n,n) for (d,n) in zip(0:2,[1,2,1]))])
        V=CP.cup_module(R);damaged=deepcopy(V);damaged.images[1][3].=zero(K)
        @test !CP.check_cup_module(damaged).valid
        @test_throws ArgumentError CP.check_cup_module(damaged;throw=true)
        badmaps=deepcopy(R);badmaps.restrictions[2,1][3].=zero(K)
        @test !CP.check_persistent_cup_products(badmaps).valid
        wrongshape=deepcopy(A);wrongshape.products[(1,1)]=zeros(K,2,2)
        @test !CP.check_cohomology_algebra(wrongshape).valid
        @test_throws ArgumentError CP.cup_module(R;budget=CP.CupProductBudget(max_entries=0))
        @test_throws ArgumentError CP.cup_length(A;budget=CP.CupProductBudget(max_entries=0))
    end
    # Different degree partitions of the same cup length need not agree.
    ds=[1,1,1,1];tables=Dict{Tuple{Int,Int},Matrix{Int}}()
    for d in 0:3;tables[(0,d)]=ones(Int,1,1);tables[(d,0)]=ones(Int,1,1);end
    tables[(1,2)]=ones(Int,1,1);tables[(2,1)]=ones(Int,1,1)
    A=CP.cohomology_algebra(ds,tables;unit=[1])
    R=CP.persistent_cup_products([A];parameters=[0],restrictions=[])
    @test CP.cup_length(A)==2
    @test CP.dimensions(CP.partition_module(R,[1,1]))==[0]
    @test CP.dimensions(CP.partition_module(R,[1,2]))==[1]
    @test CP.dimensions(CP.cup_module(R;factors=2))==[1]
end

@testset "A108 simplex-tree inputs and parameter boundaries" begin
    CP=TamerOp.CohomologyProducts;DT=TamerOp.DataTypes
    # Triangle boundary: three vertices at zero, edges at 1/3, 2/3, 7/3.
    offsets=[1,2,3,4,6,8,10];verts=[1,2,3,1,2,1,3,2,3]
    gs=[(0//1,),(0//1,),(0//1,),(1//3,),(2//3,),(7//3,)]
    tree=DT.SimplexTreeMulti(offsets,verts,[0,0,0,1,1,1],[1,4,7],collect(1:7),gs)
    R=CP.persistent_cup_products(tree)
    @test CP.check_persistent_cup_products(R).valid
    V=CP.cup_module(R;factors=1);b=only(CP.cup_intervals(V))
    @test b.birth==7//3 && b.death===nothing
    super=DT.SimplexTreeMulti(offsets,verts,[0,0,0,1,1,1],[1,4,7],collect(1:7),[(-g[1],) for g in gs])
    @test only(CP.cup_intervals(CP.cup_module(CP.persistent_cup_products(super;order=:superlevel);factors=1))).birth==-7//3
    malformed=deepcopy(tree);malformed.grade_data[1]=(9//1,)
    @test_throws ArgumentError CP.persistent_cup_products(malformed)
    duplicate=DT.SimplexTreeMulti([1,2,3,5],[1,1,1,2],[0,0,1],[1,3,4],[1,2,3,4],[(0,),(0,),(1,)])
    @test_throws ArgumentError CP.persistent_cup_products(duplicate)
    @test_throws ArgumentError CP.persistent_cup_products(tree;order=:superlevel)
    @test_throws ArgumentError CP.persistent_cup_products([[1]];grades=[true])
    @test_throws ArgumentError CP.persistent_cup_products([[1],[2]];grades=Real[2^60+1,0.5])
    # Repeated entries and cofaces use the earliest grade, with ties atomic.
    tied=CP.persistent_cup_products([[1,2],[2,1],[1,2,3]];grades=[2//3,1//3,1//3])
    @test CP.cup_parameters(tied)==[1//3]
    @test CP.dimensions(CP.cohomology_algebra(tied;stage=1))==[1,0,0]
    @test isempty(CP.cup_intervals(CP.cup_module(tied)))
    zerosigned=CP.persistent_cup_products([[1],[2]];grades=[-0.0,0.0])
    @test CP.cup_parameters(zerosigned)==[0.0]
    @test CP.check_persistent_cup_products(zerosigned).valid
    @test_throws ArgumentError CP.persistent_cup_products(CP.cohomology_module(R;degree=1))
end

@testset "A108 arbitrary posets, path relations, and cup images" begin
    CP=TamerOp.CohomologyProducts;FF=TamerOp.FiniteFringe;MM=TamerOp.Modules
    P=FF.FinitePoset(Bool[1 1 1 1;0 1 0 1;0 0 1 1;0 0 0 1])
    for F in (CM.QQField(),CM.F2(),CM.F3(),CM.Fp(5),CM.Fp(101))
        K=CM.coeff_type(F);A=_cup_exterior(F)
        id=Matrix{K}(I,2,2);projection=K[1 0;0 0]
        ringmap(B)=Dict(0=>ones(K,1,1),1=>copy(B),2=>reshape(K[B[1,1]*B[2,2]-B[1,2]*B[2,1]],1,1))
        maps=Dict((2,1)=>ringmap(id),(4,2)=>ringmap(projection),(3,1)=>ringmap(projection),(4,3)=>ringmap(id))
        R=CP.persistent_cup_products(P,fill(A,4);restrictions=maps)
        @test CP.check_persistent_cup_products(R).valid
        @test CP.cup_parameters(R)===nothing
        @test CP.cup_product_summary(R).provenance.indexing===:opposite_finite_poset
        @test CP.cup_product_summary(R).provenance.continuation===:finite_base
        @test CP.cohomology_restriction(R;source=4,target=1,degree=1)==projection
        @test CP.cup_length(R;source=4,target=1)==1
        @test CP.cup_length(R;source=4)==2
        V=CP.cup_module(R);W=CP.partition_module(R,[1,1])
        @test CP.dimensions(V)==CP.dimensions(W)==ones(Int,4)
        @test CP.check_cup_module(V).valid && CP.check_cup_module(W).valid
        for ((s,t),m) in maps
            @test CP.structure_map(V;source=s,target=t)==m[2]
            @test CP.structure_map(W;source=s,target=t)==m[2]
        end
        @test iszero(only(CP.structure_map(V;source=4,target=1)))
        for d in 0:2
            M=CP.cohomology_module(R;degree=d)
            @test MM.check_module_functoriality(M).valid
            @test all(FF.leq(M.Q,i,j)==FF.leq(P,j,i) for i in 1:4,j in 1:4)
            @test MM.structure_map(M;source=4,target=1)==CP.cohomology_restriction(R;source=4,target=1,degree=d)
        end
        @test_throws ArgumentError CP.cohomology_restriction(R;source=2,target=3,degree=1)
        @test_throws ArgumentError CP.structure_map(V;source=1,target=4)
        @test_throws ArgumentError CP.cup_length(R;source=2,target=3)
        @test_throws ArgumentError CP.cup_intervals(V)
        bad=deepcopy(maps);bad[4,2]=ringmap(id)
        @test_throws ArgumentError CP.persistent_cup_products(P,fill(A,4);restrictions=bad)
        broken=deepcopy(R);broken.restrictions[4,2][2].=id;broken.restrictions[4,2][3].=one(K)
        @test !CP.check_persistent_cup_products(broken).valid
        @test_throws ArgumentError CP.check_persistent_cup_products(broken;throw=true)
        incomplete=deepcopy(maps);delete!(incomplete,(4,2))
        @test_throws ArgumentError CP.persistent_cup_products(P,fill(A,4);restrictions=incomplete)
        extra=deepcopy(maps);extra[4,1]=ringmap(projection)
        @test_throws ArgumentError CP.persistent_cup_products(P,fill(A,4);restrictions=extra)
        @test_throws ArgumentError CP.persistent_cup_products(P,fill(A,4);restrictions=maps,budget=CP.CupProductBudget(max_stages=3))
        # Independent basis changes at every vertex, including orientation signs.
        Ts=[[ones(K,1,1),K[1 i;0 -1],reshape(K[-1],1,1)] for i in 1:4]
        Is=[[ones(K,1,1),K[1 i;0 -1],reshape(K[-1],1,1)] for i in 1:4]
        As=[CP.cohomology_algebra(A.dims,
            Dict((p,q)=>Is[i][p+q+1]*CP.cup_product_matrix(A;left_degree=p,right_degree=q)*kron(Ts[i][q+1],Ts[i][p+1])
                for p in 0:2 for q in 0:2-p);unit=[1],field=F) for i in 1:4]
        changed=Dict((s,t)=>Dict(d=>Is[t][d+1]*m[d]*Ts[s][d+1] for d in 0:2) for ((s,t),m) in maps)
        C=CP.persistent_cup_products(P,As;restrictions=changed)
        @test CP.check_persistent_cup_products(C).valid
        @test CP.cup_length(C;source=4,target=1)==1
        for Vc in (CP.cup_module(C),CP.partition_module(C,[1,1]),CP.cup_module(C;factors=1))
            @test CP.check_cup_module(Vc).valid
            for ((s,t),m) in changed
                ambient=zeros(K,4,4);ambient[1,1]=m[0][1,1];ambient[2:3,2:3]=m[1];ambient[4,4]=m[2][1,1]
                @test CP.cup_embedding(Vc;stage=t)*CP.structure_map(Vc;source=s,target=t)==ambient*CP.cup_embedding(Vc;stage=s)
            end
        end
        # Relabel the poset: neither its labels nor map keys are topologically sorted.
        perm=[4,2,1,3];inverse=invperm(perm)
        Q=FF.FinitePoset([FF.leq(P,perm[i],perm[j]) for i in 1:4,j in 1:4])
        S=CP.persistent_cup_products(Q,As[perm];restrictions=Dict((inverse[s],inverse[t])=>m for ((s,t),m) in changed))
        @test CP.check_cup_module(CP.cup_module(S)).valid
        @test CP.cohomology_restriction(S;source=inverse[4],target=inverse[1],degree=1)==CP.cohomology_restriction(C;source=4,target=1,degree=1)
        maps[2,1][1].=zero(K)
        @test CP.check_persistent_cup_products(R).valid # constructor owns its data
        copied=CP.cup_poset(R);copied._leq[1,4]=false
        @test FF.leq(CP.cup_poset(V),1,4)
        # A merge with paths of unequal length must also commute.
        L=Matrix{Bool}(I,5,5)
        for (u,v) in ((1,2),(2,5),(1,3),(3,4),(4,5));L[u,v]=true;end
        for k in 1:5,i in 1:5,j in 1:5;L[i,j]|=L[i,k]&&L[k,j];end
        Q=FF.FinitePoset(L);ms=Dict((v,u)=>ringmap(id) for (u,v) in FF.cover_edges(Q));ms[5,4]=ringmap(projection)
        @test_throws ArgumentError CP.persistent_cup_products(Q,fill(A,5);restrictions=ms)
        # Disconnected posets and empty algebras are allowed.
        Q=FF.FinitePoset(Matrix{Bool}(I,2,2));Z=CP.cohomology_algebra([0],Dict();unit=Int[],field=F)
        D=CP.persistent_cup_products(Q,[A,Z];restrictions=Dict())
        @test CP.dimensions(CP.cup_module(D))==[1,0]
        @test CP.check_cup_module(CP.cup_module(D)).valid
    end
end

@testset "A108 finite-poset and multiparameter simplicial sources" begin
    CP=TamerOp.CohomologyProducts;FF=TamerOp.FiniteFringe;DT=TamerOp.DataTypes;MM=TamerOp.Modules
    P=FF.FinitePoset(Bool[1 1 1 1;0 1 0 1;0 0 1 1;0 0 0 1])
    triangles=_cup_torus();edges=unique([sort(s[[i,j]]) for s in triangles for i in 1:3 for j in i+1:3])
    cone=[[10;s] for s in triangles]
    for F in (CM.QQField(),CM.F2(),CM.F3(),CM.Fp(5))
        # Skeleton, torus on one branch, cone on the other, and their union.
        R=CP.persistent_cup_products(P,[edges,triangles,cone,cone];field=F)
        @test CP.check_persistent_cup_products(R).valid
        @test CP.dimensions(CP.cup_module(R))==[0,1,0,0]
        @test CP.check_cup_module(CP.partition_module(R,[1,1])).valid
        @test_throws ArgumentError CP.persistent_cup_products(P,[triangles,edges,cone,cone];field=F)
        for order in (:sublevel,:superlevel)
            sign=order===:sublevel ? 1 : -1
            cells=[edges;triangles;cone]
            gs=[fill((0//1,0//1),length(edges));fill((2//3,0//1),length(triangles));fill((0//1,5//2),length(cone))]
            gs=[sign.*g for g in gs]
            S=CP.persistent_cup_products(cells;grades=gs,field=F,order)
            @test CP.cup_parameters(S)==[sign.*g for g in [(0//1,0//1),(2//3,0//1),(0//1,5//2),(2//3,5//2)]]
            @test CP.dimensions(CP.cup_module(S))==[0,1,0,0]
            @test CP.check_persistent_cup_products(S).valid
            @test CP.check_cup_module(CP.cup_module(S)).valid
            @test MM.check_module_functoriality(CP.cohomology_module(S;degree=1)).valid
            @test_throws ArgumentError CP.cup_intervals(CP.cup_module(S))
            # All induced maps agree with literal cochain restriction modulo boundaries.
            for (u,v) in FF.cover_edges(CP.cup_poset(S)),d in 0:3
                B=CP.cohomology_algebra(S;stage=u);A=CP.cohomology_algebra(S;stage=v)
                d<length(A.dims) && d<length(B.dims) || continue
                a=CP.cohomology_representatives(A;degree=d);b=CP.cohomology_representatives(B;degree=d)
                rows=[findfirst(==(c),a.cells) for c in b.cells]
                image=a.representatives[rows,:]
                difference=image-b.representatives*CP.cohomology_restriction(S;source=v,target=u,degree=d)
                @test _cup_oracle_rank(hcat(b.coboundaries,difference))==size(b.coboundaries,2)
            end
        end
    end
    # A multicritical vertex appears in both incomparable arms, not their meet.
    tree=DT.SimplexTreeMulti([1,2],[1],[0],[1,2],[1,3],[(0,2),(3,0)])
    S=CP.persistent_cup_products(tree)
    @test CP.cup_parameters(S)==[(0,0),(3,0),(0,2),(3,2)]
    @test CP.cohomology_module(S;degree=0).dims==[0,1,1,1]
    @test CP.check_persistent_cup_products(S).valid
    # The full grid also supports three parameters and budgeted joins.
    R=CP.persistent_cup_products([[1],[2]];grades=[(0,1,2),(3,0,0)])
    @test length(CP.cup_parameters(R))==8
    @test CP.check_persistent_cup_products(R).valid
    @test_throws ArgumentError CP.persistent_cup_products([[1],[2]];grades=[(0,1,2),(3,0,0)],budget=CP.CupProductBudget(max_stages=7))
    @test_throws ArgumentError CP.persistent_cup_products([[1],[2]];grades=[(0,1),(1,)])
    @test_throws ArgumentError CP.persistent_cup_products([[1],[2]];grades=[(0,1),(Inf,0)])
    # Full tree boundary closure must hold at every grade, even between samples.
    bad=DT.SimplexTreeMulti([1,2,3,5],[1,2,1,2],[0,0,1],[1,3,4],[1,2,3,4],[(2,2),(0,0),(1,1)])
    @test_throws ArgumentError CP.persistent_cup_products(bad)
end
