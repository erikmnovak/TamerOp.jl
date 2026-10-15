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
