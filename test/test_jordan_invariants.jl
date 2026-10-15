# Independent small Gaussian elimination and explicit full-operator oracles.
function _ji_rank(A)
    R=Matrix(A);r=1
    for c in axes(R,2)
        p=findfirst(i->!iszero(R[i,c]),r:size(R,1));p===nothing && continue
        p+=r-1;R[r,:],R[p,:]=copy(R[p,:]),copy(R[r,:]);R[r,:]/=R[r,c]
        for i in r+1:size(R,1);R[i,:]-=R[i,c]*R[r,:];end
        r+=1;r>size(R,1) && break
    end
    return r-1
end
function _ji_chain(F,n,intervals)
    K=CM.coeff_type(F);Q=FF.ProductOfChainsPoset((n,))
    active=[[i for (i,(a,b)) in enumerate(intervals) if a<=v<=b] for v in 1:n]
    maps=Dict{Tuple{Int,Int},Matrix{K}}()
    for v in 1:n-1
        maps[v,v+1]=[K(i==j) for i in active[v+1],j in active[v]]
    end
    return MD.PModule{K}(Q,length.(active),maps;field=F)
end
function _ji_dense(M,slices)
    K=CM.coeff_type(M.field);labels=reduce(vcat,sort.(collect.(slices));init=Int[])
    ds=M.dims[labels];offset=cumsum(vcat(0,ds));lookup=Dict(v=>i for (i,v) in enumerate(labels))
    T=zeros(K,sum(ds),sum(ds))
    for i in 1:length(slices)-1,u in slices[i],v in slices[i+1]
        FF.leq(M.Q,u,v) || continue
        a=lookup[u];b=lookup[v]
        T[offset[b]+1:offset[b+1],offset[a]+1:offset[a+1]]=MD.map_leq(M,u,v)
    end
    return T
end
function _ji_assert(M,slices)
    r=JI.jordan_type(M;slices,witness=true)
    @test JI.check_jordan_type(r;source=M,throw=true).valid
    @test JI.check_jordan_type(JI.jordan_type(M;slices)).valid
    T=_ji_dense(M,slices);n=length(slices);d=size(T,1)
    @test Matrix(JI.jordan_operator(r))==T
    @test iszero(T^n)
    ranks=[_ji_rank(T^j) for j in 0:n]
    @test JI.jordan_power_ranks(r)==ranks
    @test sum(i*m for (i,m) in enumerate(JI.jordan_multiplicities(r)))==d
    return r
end
function _ji_basis_change(M,rng)
    K=CM.coeff_type(M.field);B=Matrix{K}[];Bi=Matrix{K}[]
    for d in M.dims
        X=Matrix{K}(I,d,d);Xi=copy(X)
        for _ in 1:4d
            d<2 && break
            a,b=randperm(rng,d)[1:2];c=K(rand(rng,-3:3))
            X[a,:]+=c*X[b,:];Xi[:,b]-=c*Xi[:,a]
        end
        push!(B,X);push!(Bi,Xi)
    end
    maps=Dict{Tuple{Int,Int},Matrix{K}}((u,v)=>B[v]*M.edge_maps[u,v]*Bi[u] for (u,v) in FF.cover_edges(M.Q))
    return MD.PModule{K}(M.Q,copy(M.dims),maps;field=M.field)
end

@testset "Jordan type and multirank" begin
    @testset "Known intervals and independent full operators" begin
        for F in (CM.QQField(),CM.F2(),CM.F3(),CM.Fp(101))
            rng=MersenneTwister(109)
            for n in 2:7
                intervals=[(a,b) for a in 1:n for b in a:n]
                M=_ji_chain(F,n,intervals);slices=[[i] for i in 1:n]
                r=_ji_assert(M,slices)
                @test JI.jordan_multiplicities(r)==collect(n:-1:1)
                @test JI.jordan_multiplicities(JI.jordan_type(_ji_basis_change(M,rng);slices))==JI.jordan_multiplicities(r)
                @test JI.jordan_multiplicities(JI.jordan_type(MD.direct_sum(M,M);slices))==2JI.jordan_multiplicities(r)
                @test JI.multirank(M;sources=[1],targets=[n])==1
                @test JI.multirank(M;sources=[n],targets=[1])==0
                # General subsets need not be antichains or follow the order.
                perm=randperm(rng,n);cuts=[perm[1:2:end],perm[2:2:end],Int[]]
                _ji_assert(M,cuts)
                _ji_assert(M,(Int[],collect(1:n),Int[]))
            end
        end
    end
    @testset "Classical-rank separation and characteristic effects" begin
        for F in (CM.QQField(),CM.F2(),CM.F3(),CM.Fp(101))
            K=CM.coeff_type(F);Q=FF.ProductOfChainsPoset((2,2))
            # Proposition 18's fork, with zero at the fourth grid vertex:
            # two independent functionals versus two equal functionals.
            maps=Dict((1,2)=>K[1 0],(1,3)=>K[0 1],(2,4)=>zeros(K,0,1),(3,4)=>zeros(K,0,1))
            X=MD.PModule{K}(Q,[2,1,1,0],maps;field=F)
            same=copy(maps);same[1,3]=K[1 0];Y=MD.PModule{K}(Q,[2,1,1,0],same;field=F)
            for u in 1:4,v in 1:4
                FF.leq(Q,u,v) || continue
                @test _ji_rank(MD.map_leq(X,u,v))==_ji_rank(MD.map_leq(Y,u,v))
            end
            @test JI.multirank(X;sources=[1],targets=[2,3])==2
            @test JI.multirank(Y;sources=[1],targets=[2,3])==1
            @test JI.jordan_multiplicities(_ji_assert(X,([1],[2,3],[4])))==[0,2,0]
            @test JI.jordan_multiplicities(_ji_assert(Y,([1],[2,3],[4])))==[2,1,0]
            # A commuting diamond has two paths in T^2; their sum vanishes in F2.
            constant=MD.PModule{K}(Q,ones(Int,4),Dict(e=>ones(K,1,1) for e in FF.cover_edges(Q));field=F)
            r=_ji_assert(constant,([1],[2,3],[4]))
            @test JI.jordan_multiplicities(r)==(F==CM.F2() ? [0,2,0] : [1,0,1])
            @test JI.multirank(constant;sources=[1],targets=[4])==1
            @test JI.jordan_multiplicities(JI.jordan_type(constant;slices=([1],[3,2],[4])))==JI.jordan_multiplicities(r)
            # A three-way merge cancels in characteristic three.
            leqmat=Matrix{Bool}(I,5,5);leqmat[1,:].=true;leqmat[:,5].=true
            P=FF.FinitePoset(leqmat)
            three=MD.PModule{K}(P,ones(Int,5),Dict(e=>ones(K,1,1) for e in FF.cover_edges(P));field=F)
            r3=_ji_assert(three,([1],[2,3,4],[5]))
            @test JI.jordan_multiplicities(r3)==(F==CM.F3() ? [1,2,0] : [2,0,1])
        end
    end
    @testset "Arbitrary maps, ownership, validation and budgets" begin
        for F in (CM.QQField(),CM.F2(),CM.F3(),CM.Fp(101))
            K=CM.coeff_type(F);rng=MersenneTwister(1109)
            leqmat=Matrix{Bool}(I,5,5);leqmat[1:3,4:5].=true;P=FF.FinitePoset(leqmat)
            for trial in 1:12
                ds=rand(rng,0:3,5)
                maps=Dict(e=>K.(rand(rng,-2:2,ds[e[2]],ds[e[1]])) for e in FF.cover_edges(P))
                M=MD.PModule{K}(P,ds,maps;field=F)
                for slices in (([1,2,3],[4,5]),([1,4],[2],[3,5]),([4,5],[1,2,3]))
                    r=_ji_assert(M,slices)
                    @test JI.jordan_multiplicities(JI.jordan_type(_ji_basis_change(M,rng);slices))==JI.jordan_multiplicities(r)
                end
            end
        end
        F=CM.QQField();M=_ji_chain(F,3,[(1,3),(1,1)]);slices=([1],[2],[3])
        r=TO.jordan_type(M;slices,witness=true)
        @test TO.jordan_multiplicities(r)==[1,0,1]
        @test TO.jordan_power_ranks(r)==[4,2,1,0]
        @test TO.jordan_summary(r).category==:finite_poset
        pi=EC.GridEncodingMap(M.Q,([0,1,2],))
        enc=TO.EncodingResult(M.Q,M,pi);cache=TO.SessionCache()
        @test TO.jordan_multiplicities(TO.jordan_type(enc;slices,cache))==[1,0,1]
        @test TO.multirank(enc;sources=[1],targets=[3],cache)==1
        inv=TO.invariant(enc;which=:jordan_type,slices,cache)
        @test TO.jordan_multiplicities(TO.invariant_value(inv))==[1,0,1]
        inv2=TO.invariant(enc;which=:multirank,sources=[1],targets=[3],cache)
        @test TO.invariant_value(inv2)==1
        @test JI.provenance(r).translated==false
        @test JI.field(r)==F
        @test JI.dimensions(r)==[2,1,1]
        @test occursin("JordanTypeResult",sprint(show,r))
        cs=JI.jordan_multiplicities(r);cs[1]=999
        sl=JI.jordan_slices(r);sl[1][1]=999
        bs=JI.jordan_subset_maps(r);bs[1].=0
        @test JI.check_jordan_type(r;source=M).valid
        M.edge_maps[1,2].=0
        @test !JI.check_jordan_type(r;source=M).valid
        @test JI.check_jordan_type(r).valid
        @test_throws ArgumentError JI.check_jordan_type(r;source=M,throw=true)
        bad=deepcopy(r);bad.multiplicities[1]+=1
        @test !JI.check_jordan_type(bad).valid
        bad=deepcopy(r);bad.blocks[1].=0
        @test !JI.check_jordan_type(bad).valid
        numeric=JI.jordan_type(M;slices)
        @test_throws ArgumentError JI.jordan_operator(numeric)
        @test_throws ArgumentError JI.jordan_subset_maps(numeric)
        @test_throws ArgumentError JI.jordan_operator(r;budget=JI.JordanBudget(max_entries=1))
        for invalid in (([1],[1]),([1,1],[2]),([0],[2]),([4],[2]),([true],[2]),([1],),:bad)
            @test !JI.check_jordan_query(M;slices=invalid).valid
            @test_throws ArgumentError JI.jordan_type(M;slices=invalid)
        end
        for budget in (JI.JordanBudget(max_vertices=1),JI.JordanBudget(max_slices=1),
                JI.JordanBudget(max_dimension=1),JI.JordanBudget(max_entries=1),JI.JordanBudget(max_products=0))
            @test !JI.check_jordan_query(M;slices,budget).valid
            @test_throws ArgumentError JI.jordan_type(M;slices,budget)
        end
        @test_throws ArgumentError JI.JordanBudget(max_vertices=-1)
        @test_throws ArgumentError JI.JordanBudget(max_entries=true)
        R=_ji_chain(CM.RealField(Float64),3,[(1,3)])
        @test_throws ArgumentError JI.jordan_type(R;slices)
        Q=FF.ProductOfChainsPoset((2,2));K=CM.coeff_type(F)
        maps=Dict(e=>ones(K,1,1) for e in FF.cover_edges(Q));maps[1,2].=0
        broken=MD.PModule{K}(Q,ones(Int,4),maps;field=F)
        @test_throws ArgumentError JI.jordan_type(broken;slices=([1],[2,3],[4]))
        Z=_ji_chain(F,2,Tuple{Int,Int}[])
        @test JI.jordan_multiplicities(_ji_assert(Z,([1],[2])))==[0,0]
        @test JI.multirank(Z;sources=Int[],targets=Int[])==0
        good=JI.check_jordan_type(r)
        @test good==deepcopy(good) && hash(good)==hash(deepcopy(good))
        @test JI.check_jordan_query(M;slices=(Set([1]),Set([2,3]))).valid
        # Independent concurrent queries retain deterministic exact outputs.
        jobs=[Threads.@spawn JI.jordan_type(Z;slices=([1],[2])) for _ in 1:8]
        @test all(x->JI.jordan_multiplicities(fetch(x))==[0,0],jobs)
    end
end

# Independent ambient fixtures and full-matrix oracles for translated families.
function _jt_interval(F,a,b;domain=:real,left=true,right=false)
    K=CM.coeff_type(F)
    dims=domain===:real ? [0,Int(left),1,Int(right),0] : [0,1,0]
    maps=Dict((i,i+1)=>ones(K,dims[i+1],dims[i]) for i in 1:length(dims)-1)
    TamerOp.Interleavings.RectilinearModule(([a,b],),dims,maps;field=F,domain)
end
function _jt_constant(F,d;parameters=1,domain=:real)
    K=CM.coeff_type(F);cuts=ntuple(_->Int[],parameters)
    TamerOp.Interleavings.RectilinearModule(cuts,[d],Dict{Tuple{Int,Int},Matrix{K}}();field=F,domain)
end
function _jt_operator(M,slices,x,y=x)
    points=reduce(vcat,collect.(slices));K=CM.coeff_type(M.finite.field)
    ds=[TamerOp.Interleavings.ambient_dimension(M,x.+z) for z in points]
    dt=[TamerOp.Interleavings.ambient_dimension(M,y.+z) for z in points]
    offsets=cumsum(vcat(0,ds));ends=cumsum(vcat(0,dt));T=zeros(K,sum(ds),sum(ds));D=zeros(K,sum(dt),sum(ds))
    slice_ids=reduce(vcat,[fill(i,length(s)) for (i,s) in enumerate(slices)])
    for i in eachindex(points),j in eachindex(points)
        if slice_ids[j]==slice_ids[i]+1 && all(points[i].<=points[j])
            T[offsets[j]+1:offsets[j+1],offsets[i]+1:offsets[i+1]]=
                TamerOp.Interleavings.structure_map(M;source=x.+points[i],target=x.+points[j])
        end
    end
    for (i,z) in enumerate(points)
        D[ends[i]+1:ends[i+1],offsets[i]+1:offsets[i+1]]=
            TamerOp.Interleavings.structure_map(M;source=x.+z,target=y.+z)
    end
    T,D
end
function _jt_oracle(M,slices,x,y)
    T,D=_jt_operator(M,slices,x,y)
    [_ji_rank(D*T^p) for p in 0:length(slices)]
end
function _jt_fork(F)
    K=CM.coeff_type(F);Q=FF.GridPoset((collect(1:4),collect(1:4)))
    ds=zeros(Int,16);ds[6]=2;ds[7]=1;ds[10]=1
    maps=Dict(e=>zeros(K,ds[e[2]],ds[e[1]]) for e in FF.cover_edges(Q))
    maps[6,7]=K[1 0];maps[6,10]=K[0 1]
    M=MD.PModule{K}(Q,ds,maps;field=F)
    TamerOp.Interleavings.RectilinearModule(M,([0,1,2],[0,1,2]);domain=:integer)
end

@testset "Translated Jordan families" begin
    @testset "Independent powers, transports and image modules" begin
        for F in (CM.QQField(),CM.F2(),CM.F3(),CM.Fp(101))
            M=_jt_interval(F,0,4);slices=([(0,)],[(1,)],[(2,)])
            f=JI.jordan_filtration(M;slices)
            for x in (-2,-1,0,1//2,1,2,3,4), y in (x,x+1//2,x+1,x+4)
                expected=_jt_oracle(M,slices,(x,),(y,))
                @test JI.jordan_filtered_rank(f;source=(x,),target=(y,))==expected
                for p in 0:3
                    @test JI.jordan_filtered_rank(f;source=(x,),target=(y,),power=p)==expected[p+1]
                end
            end
            for p in 0:3
                result=JI.jordan_module(f;power=p,witness=true)
                @test JI.check_jordan_module(result;throw=true).valid
                @test MD.check_module_functoriality(JI.encoding_module(result)).valid
                for x in (-1,0,1//2,1,2,7//2),y in (x,x+1//2,x+2)
                    T,D=_jt_operator(M,slices,(x,),(y,));B=JI.jordan_basis(f;at=(x,),power=p)
                    C=JI.jordan_basis(f;at=(y,),power=p)
                    @test _ji_rank(hcat(T^p,B))==_ji_rank(T^p)==size(B,2)
                    @test JI.jordan_basis(result;at=(x,))==B
                    H=JI.jordan_structure_map(f;source=(x,),target=(y,),power=p)
                    @test C*H==D*B
                    @test JI.structure_map(result;source=(x,),target=(y,))==H
                    @test _ji_rank(H)==_jt_oracle(M,slices,(x,),(y,))[p+1]
                    for q in 0:p
                        J=JI.jordan_structure_map(f;source=(x,),target=(y,),power=p,target_power=q)
                        @test JI.jordan_basis(f;at=(y,),power=q)*J==D*B
                    end
                end
            end
            # Composition in the translation parameter, including different powers.
            H=JI.jordan_structure_map(f;source=(0,),target=(1,),power=2,target_power=1)
            G=JI.jordan_structure_map(f;source=(1,),target=(2,),power=1,target_power=0)
            @test G*H==JI.jordan_structure_map(f;source=(0,),target=(2,),power=2,target_power=0)
            # Distinct points can have the SAME encoding label: keep both copies.
            short=JI.jordan_filtration(M;slices=([(0,)],[(1//10,)]))
            @test JI.jordan_filtered_rank(short;source=(1,),target=(1,))==[2,1,0]
        end
    end
    @testset "Branching, cancellations, basis changes and dimension" begin
        for F in (CM.QQField(),CM.F2(),CM.F3(),CM.Fp(101))
            for nmiddle in (2,3)
                points=[(i,nmiddle-i) for i in 1:nmiddle]
                slices=([(0,0)],points,[(nmiddle,nmiddle)])
                M=_jt_constant(F,1;parameters=2);f=JI.jordan_filtration(M;slices)
                oracle=_jt_oracle(M,slices,(0,0),(3,4))
                @test JI.jordan_filtered_rank(f;source=(0,0),target=(3,4))==oracle
                @test oracle[3]==(F==CM.Fp(nmiddle) ? 0 : 1)
                @test JI.encoding_axes(f)==(QQ[],QQ[])
                for p in 0:3
                    r=JI.jordan_module(f;power=p,witness=true)
                    @test only(JI.encoding_module(r).dims)==oracle[p+1]
                end
            end
            M=_jt_fork(F);slices=([(0,0)],[(1,0),(0,1)])
            f=JI.jordan_filtration(M;slices)
            @test JI.jordan_filtered_rank(f;source=(0,0),target=(0,0))==[4,2,0]
            changed=_ji_basis_change(M.finite,MersenneTwister(8109))
            N=TamerOp.Interleavings.RectilinearModule(changed,([0,1,2],[0,1,2]);domain=:integer)
            g=JI.jordan_filtration(N;slices)
            doubled=TamerOp.Interleavings.RectilinearModule(MD.direct_sum(M.finite,M.finite),([0,1,2],[0,1,2]);domain=:integer)
            h=JI.jordan_filtration(doubled;slices)
            for x in Iterators.product(-1:1,-1:1),delta in ((0,0),(0,1),(1,0),(1,1),(2,3))
                y=x.+delta;expected=_jt_oracle(M,slices,x,y)
                @test JI.jordan_filtered_rank(f;source=x,target=y)==expected
                @test JI.jordan_filtered_rank(g;source=x,target=y)==expected
                @test JI.jordan_filtered_rank(h;source=x,target=y)==2expected
            end
            r=JI.jordan_module(f;power=1,witness=true)
            @test JI.check_jordan_module(r).valid
            @test MD.check_module_functoriality(JI.encoding_module(r)).valid
            # Equality with the original finite selected-subset query at x=0.
            fixed=JI.jordan_type(M.finite;slices=([6],[7,10]))
            @test JI.jordan_power_ranks(fixed)==JI.jordan_filtered_rank(f;source=(0,0),target=(0,0))
            # Unordered slices, empty slices, and a third parameter are allowed.
            c=_jt_constant(F,2;parameters=3,domain=:integer)
            for pattern in (([(2,0,0)],[(0,0,0)],[(3,0,0)]),
                    ([(0,0,0)],Tuple{Int,Int,Int}[],[(1,1,1)]),
                    (Tuple{Int,Int,Int}[],Tuple{Int,Int,Int}[]))
                a=JI.jordan_filtration(c;slices=pattern)
                @test JI.jordan_filtered_rank(a;source=(0,0,0),target=(2,3,4))==_jt_oracle(c,pattern,(0,0,0),(2,3,4))
                @test JI.check_jordan_module(JI.jordan_module(a;power=1,witness=true)).valid
            end
        end
    end
    @testset "Critical strata, continuation, validation and workflows" begin
        F=CM.QQField();M=_jt_interval(F,0,3;left=false,right=true)
        slices=([(0,)],[(1//2,)])
        f=JI.jordan_filtration(M;slices)
        @test JI.encoding_axes(f)==(QQ[-1//2,0,5//2,3],)
        r=JI.jordan_module(f;power=1,witness=true)
        for x in (-1//2,0,1//100,5//2,251//100,3),y in (x,x+1//100,x+1)
            @test _ji_rank(JI.structure_map(r;source=(x,),target=(y,)))==_jt_oracle(M,slices,(x,),(y,))[2]
        end
        changed=TamerOp.Interleavings.refine_parameters(M,([-1,0,1,2,3,4],))
        g=JI.jordan_filtration(changed;slices)
        for x in (-1//2,0,1//2,5//2,3)
            @test JI.jordan_filtered_rank(f;source=(x,),target=(x+1//2,))==JI.jordan_filtered_rank(g;source=(x,),target=(x+1//2,))
        end
        copy_slices=JI.jordan_slices(f);copy_slices[1][1]=(QQ(999),)
        @test JI.jordan_slices(f)[1]==[(QQ(0),)]
        M.finite.dims[3]=99
        @test JI.check_jordan_filtration(f).valid
        @test occursin("JordanFiltration",sprint(show,f))
        @test occursin("JordanModuleResult",sprint(show,r))
        @test JI.provenance(f).translated && JI.provenance(r).scope==:global
        @test JI.jordan_summary(f).powers==0:2
        @test JI.jordan_power_ranks(f;at=(1,))==[2,1,0]
        @test JI.jordan_multiplicities(f;at=(1,))==[0,1]
        @test JI.check_jordan_query(f;source=(1,),target=(2,)).valid
        @test !JI.check_jordan_query(f;source=(NaN,),target=(2,)).valid
        @test_throws ArgumentError JI.check_jordan_query(f;source=(NaN,),target=(2,),throw=true)
        @test JI.jordan_summary(r).witness
        @test JI.dimensions(r)==JI.encoding_module(r).dims
        plain=JI.jordan_module(f;power=1)
        @test_throws ArgumentError JI.jordan_basis(plain;at=(1,))
        bad=deepcopy(r)
        # Pick a nonzero basis explicitly for the mutation check.
        i=findfirst(B->!isempty(B),bad.bases);bad.bases[i].*=0
        @test !JI.check_jordan_module(bad).valid
        @test_throws ArgumentError JI.check_jordan_module(bad;throw=true)
        broken=deepcopy(f);push!(broken.slices[2],broken.slices[1][1])
        @test !JI.check_jordan_filtration(broken).valid
        @test_throws ArgumentError JI.check_jordan_filtration(broken;throw=true)
        for pattern in (([(0,)],[(0,)]), ([(true,)],[(1,)]), ([(0,0)],[(1,1)]), ([(0,)],), :bad)
            @test_throws ArgumentError JI.jordan_filtration(f.source;slices=pattern)
        end
        for p in (-1,3,true,1//2)
            @test_throws ArgumentError JI.jordan_filtered_rank(f;source=(0,),target=(1,),power=p)
        end
        @test JI.jordan_filtered_rank(f;source=(1,),target=(0,),power=1)==Inf
        @test JI.jordan_filtered_rank(f;source=(1,),target=(0,))==fill(Inf,3)
        @test_throws ArgumentError JI.jordan_structure_map(f;source=(1,),target=(0,),power=0)
        @test_throws ArgumentError JI.jordan_structure_map(f;source=(0,),target=(1,),power=0,target_power=1)
        for budget in (JI.JordanBudget(max_dimension=0),JI.JordanBudget(max_vertices=0),JI.JordanBudget(max_entries=0),JI.JordanBudget(max_products=0))
            @test_throws ArgumentError JI.jordan_filtered_rank(f;source=(1,),target=(1,),budget)
        end
        @test_throws ArgumentError JI.jordan_module(f;power=1,budget=JI.JordanBudget(max_cells=1))
        @test_throws ArgumentError JI.encoding_axes(f;budget=JI.JordanBudget(max_events=1))
        @test JI.jordan_filtered_rank(f;source=(1,),target=(1,),budget=JI.JordanBudget(max_cells=1))==[2,1,0]
        finite=_ji_chain(F,3,[(1,3),(1,1)])
        pi=EC.GridEncodingMap(finite.Q,([0,1,2],));enc=TO.EncodingResult(finite.Q,finite,pi)
        @test_throws ArgumentError TO.jordan_filtration(finite;slices)
        @test_throws ArgumentError TO.jordan_filtration(enc;slices)
        a=TO.jordan_filtration(enc;slices,extension=:zero_below)
        b=TO.jordan_filtration(enc;slices,extension=:zero_below,cache=TO.SessionCache())
        @test TO.jordan_filtered_rank(a;source=(10,),target=(20,))==[2,1,0]
        @test TO.jordan_filtered_rank(a;source=(-2,),target=(-1,))==[0,0,0]
        @test TO.jordan_filtered_rank(b;source=(10,),target=(20,))==[2,1,0]
        @test TO.jordan_filtered_rank(enc;slices,extension=:zero_below,source=(10,),target=(20,))==[2,1,0]
        inv=TO.invariant(enc;which=:jordan_filtered_rank,slices,extension=:zero_below,source=(10,),target=(20,))
        @test TO.invariant_value(inv)==[2,1,0]
        @test TO.Advanced.JordanFiltration===JI.JordanFiltration
        @test TO.Advanced.jordan_interleaving===JI.jordan_interleaving
        integer=JI.jordan_filtration(_jt_interval(F,0,3;domain=:integer);slices=([(0,)],[(1,)]))
        @test_throws ArgumentError JI.jordan_filtered_rank(integer;source=(1//2,),target=(1,))
        @test_throws ArgumentError JI.jordan_filtration(integer.source;slices=([(0,)],[(1//2,)]))
        jobs=[Threads.@spawn JI.jordan_filtered_rank(f;source=(1,),target=(2,)) for _ in 1:8]
        @test all(t->fetch(t)==[2,1,0],jobs)
    end
end

@testset "Jordan landscapes, erosion and stability" begin
    for F in (CM.QQField(),CM.F2(),CM.F3())
        slices=([(0,)],[(1,)])
        f=JI.jordan_filtration(_jt_interval(F,0,4);slices)
        g=JI.jordan_filtration(_jt_interval(F,1,5);slices)
        # J^1 of [0,4) with this pattern is [0,3).
        for x in -1//2:1//2:4
            @test JI.jordan_landscape(f;at=(x,),power=1)==max(0,min(x,3-x))
            @test abs(JI.jordan_landscape(f;at=(x,),power=1)-JI.jordan_landscape(g;at=(x,),power=1))<=1
        end
        @test JI.jordan_landscape(f;at=(1,),power=1,layer=2)==0
        @test JI.jordan_landscape(f;at=(1,),power=2)==0
        @test JI.jordan_erosion_distance(f,f)==0
        @test JI.jordan_erosion_distance(f,g)==1
        @test JI.jordan_erosion_distance(g,f)==1
        @test JI.jordan_erosion_test(f,g;epsilon=1)
        @test !JI.jordan_erosion_test(f,g;epsilon=1//2)
        # Lift actual interleavings, including the zero terminal image module.
        problem=TamerOp.Interleavings.interleaving_problem(f.source,g.source,1)
        K=CM.coeff_type(F)
        forward=[ones(K,problem.targets[1].dims[i],problem.sources[1].dims[i]) for i in eachindex(problem.sources[1].dims)]
        reverse=[ones(K,problem.targets[2].dims[i],problem.sources[2].dims[i]) for i in eachindex(problem.sources[2].dims)]
        certificate=TamerOp.Interleavings.interleaving_certificate(problem,forward,reverse)
        for power in 0:2
            lifted=JI.jordan_interleaving(certificate;slices,power)
            @test TamerOp.Interleavings.check_interleaving(lifted).valid
            @test TamerOp.Interleavings.provenance(lifted).epsilon==1
        end
    end
    F=CM.QQField();slices=([(0,)],[(1,)])
    zero=JI.jordan_filtration(_jt_constant(F,0);slices)
    closed=JI.jordan_filtration(_jt_interval(F,0,4;right=true);slices)
    @test JI.jordan_erosion_distance(closed,zero)==2
    @test !JI.jordan_erosion_test(closed,zero;epsilon=2)
    @test JI.jordan_erosion_test(closed,zero;epsilon=201//100)
    @test JI.jordan_landscape(closed;at=(3//2,),power=1)==3//2
    constant=JI.jordan_filtration(_jt_constant(F,1);slices)
    @test JI.jordan_landscape(constant;at=(0,),power=1)==Inf
    @test JI.jordan_erosion_distance(constant,zero)==Inf
    # An isolated singleton has erosion distance zero from zero, not equality.
    K=CM.coeff_type(F)
    singleton=TamerOp.Interleavings.RectilinearModule(([0],),[0,1,0],Dict((1,2)=>zeros(K,1,0),(2,3)=>zeros(K,0,1));field=F)
    point=JI.jordan_filtration(singleton;slices)
    @test JI.jordan_erosion_distance(point,zero)==0
    @test !JI.jordan_erosion_test(point,zero;epsilon=0)
    @test JI.jordan_landscape(point;at=(0,))==0
    # Integer radii are maxima; test against enumeration, including boundary loss.
    for a in -1:1,b in a+1:a+5
        M=_jt_interval(F,a,b;domain=:integer);f=JI.jordan_filtration(M;slices)
        z=JI.jordan_filtration(_jt_constant(F,0;domain=:integer);slices)
        @test JI.jordan_erosion_distance(f,z)==cld(b-a,2)
        for x in a-1:b+1,p in 0:2,k in 1:2
            feasible=[t for t in 0:8 if _jt_oracle(M,slices,(x-t,),(x+t,))[p+1]>=k]
            @test JI.jordan_landscape(f;at=(x,),power=p,layer=k)==maximum(feasible;init=0)
        end
    end
    for epsilon in (-1,Inf,true)
        @test_throws ArgumentError JI.jordan_erosion_test(closed,zero;epsilon)
    end
    @test_throws ArgumentError JI.jordan_landscape(closed;at=(0,),layer=0)
    @test_throws ArgumentError JI.jordan_landscape(closed;at=(0,),layer=true)
    @test_throws ArgumentError JI.jordan_erosion_test(closed,zero;epsilon=0,budget=JI.JordanBudget(max_pairs=0))
    @test_throws ArgumentError JI.jordan_erosion_distance(closed,zero;budget=JI.JordanBudget(max_events=1))
    different=JI.jordan_filtration(_jt_constant(F,0);slices=([(0,)],[(2,)]))
    @test_throws ArgumentError JI.jordan_erosion_distance(closed,different)
    @test_throws ArgumentError JI.jordan_erosion_distance(closed,JI.jordan_filtration(_jt_constant(CM.F2(),0);slices))
end

@testset "Jordan global comparison oracles" begin
    # A two-dimensional image module with two path contributions. In F2 its
    # second image vanishes; the lifted maps must preserve that cancellation.
    for F in (CM.QQField(),CM.F2(),CM.F3())
        M=_jt_constant(F,1;parameters=2)
        pattern=([(0,0)],[(1,0),(0,1)],[(1,1)])
        K=CM.coeff_type(F);c=F==CM.F2() ? one(K) : K(2)
        p=IL.interleaving_problem(M,M,(1,2))
        cert=IL.interleaving_certificate(p,[fill(c,1,1)],[fill(inv(c),1,1)])
        for power in 0:3
            lifted=JI.jordan_interleaving(cert;slices=pattern,power)
            @test IL.check_interleaving(lifted).valid
            @test IL.provenance(lifted).translation==(1,2)
        end
    end
    # This grid contains all original/shifted critical positions and one point
    # between each pair, with outer endpoints beyond the support. Dense full
    # operators provide the ranks independently of the Jordan query engine.
    for domain in (:real,:integer)
        F=CM.F3();pattern=([(0,)],[(1,)])
        M=_jt_interval(F,0,3;domain,right=true)
        N=_jt_interval(F,1,4;domain,left=false)
        f=JI.jordan_filtration(M;slices=pattern);g=JI.jordan_filtration(N;slices=pattern)
        for epsilon in (domain===:real ? (0,1//2,1,3//2,2) : (0,1,2))
            grid=domain===:real ? collect(-3//1:1//4:5) : collect(-3:5)
            expected=true
            for x in grid,y in grid
                x<=y || continue
                mr=_jt_oracle(M,pattern,(x,),(y,));nr=_jt_oracle(N,pattern,(x,),(y,))
                me=_jt_oracle(M,pattern,(x-epsilon,),(y+epsilon,))
                ne=_jt_oracle(N,pattern,(x-epsilon,),(y+epsilon,))
                if !(all(me .<= nr) && all(ne .<= mr));expected=false;break;end
            end
            @test JI.jordan_erosion_test(f,g;epsilon)==expected
        end
    end
    # Nonconstant three-parameter source: the positive orthant with one stalk.
    F=CM.F2();K=CM.coeff_type(F);Q=FF.GridPoset((collect(1:3),collect(1:3),collect(1:3)))
    points=vec(collect(Iterators.product(-1:1,-1:1,-1:1)))
    dims=[Int(all(>=(0),x)) for x in points]
    maps=Dict(e=>ones(K,dims[e[2]],dims[e[1]]) for e in FF.cover_edges(Q))
    M=IL.RectilinearModule(MD.PModule{K}(Q,dims,maps;field=F),([0],[0],[0]))
    pattern=([(0,0,0)],[(1,1,1)])
    f=JI.jordan_filtration(M;slices=pattern)
    r=JI.jordan_module(f;power=1,witness=true)
    @test JI.check_jordan_module(r).valid
    for x in ((-1,-1,-1),(0,0,0),(1//2,0,2)),y in ((1,1,3),(2,2,4))
        @test JI.jordan_filtered_rank(f;source=x,target=y)==_jt_oracle(M,pattern,x,y)
        @test _ji_rank(JI.structure_map(r;source=x,target=y))==_jt_oracle(M,pattern,x,y)[2]
    end
    finite=_ji_chain(F,2,[(1,2)])
    prob=IL.interleaving_problem(finite,finite,[1,2])
    cert=IL.interleaving_certificate(prob,[ones(K,1,1),ones(K,1,1)],[ones(K,1,1),ones(K,1,1)])
    @test_throws ArgumentError JI.jordan_interleaving(cert;slices=([(0,)],[(1,)]),power=1)
end
