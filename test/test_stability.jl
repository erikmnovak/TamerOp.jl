# Small independent field elimination, explicit flags and geometric masses.
function _hs_rank(A)
    R=copy(A);r=1
    for j in axes(R,2)
        p=findfirst(i->!iszero(R[i,j]),r:size(R,1));p===nothing && continue
        p+=r-1;R[r,:],R[p,:]=copy(R[p,:]),copy(R[r,:]);R[r,:]/=R[r,j]
        for i in r+1:size(R,1);R[i,:]-=R[i,j]*R[r,:];end
        r+=1;r>size(R,1) && break
    end
    r-1
end

function _hs_fork(F;coincident=false)
    K=CM.coeff_type(F);Q=FF.ProductOfChainsPoset((2,2))
    maps=Dict((1,2)=>K[1 0],(1,3)=>(coincident ? K[1 0] : K[0 1]),
        (2,4)=>zeros(K,0,1),(3,4)=>zeros(K,0,1))
    MD.PModule{K}(Q,[2,1,1,0],maps;field=F)
end

function _hs_change_basis(M,rng)
    K=CM.coeff_type(M.field);B=Matrix{K}[];Bi=Matrix{K}[]
    for d in M.dims
        X=Matrix{K}(I,d,d);Y=copy(X)
        for _ in 1:4d
            d<2 && break
            a,b=randperm(rng,d)[1:2];c=K(rand(rng,1:3))
            X[a,:]+=c*X[b,:];Y[:,b]-=c*Y[:,a]
        end
        push!(B,X);push!(Bi,Y)
    end
    maps=Dict((u,v)=>B[v]*M.edge_maps[u,v]*Bi[u] for (u,v) in FF.cover_edges(M.Q))
    MD.PModule{K}(M.Q,copy(M.dims),maps;field=M.field),B
end
@testset "A51 basis invariance and noninterval semistability" begin
    for F in (CM.QQField(),CM.F2(),CM.F3(),CM.Fp(5))
        K=CM.coeff_type(F)
        Q=FF.FinitePoset(Bool[1 1 1 1;0 1 0 0;0 0 1 0;0 0 0 1])
        M=MD.PModule{K}(Q,[2,1,1,1],Dict((1,2)=>K[1 0],(1,3)=>K[0 1],(1,4)=>K[1 1]);field=F)
        r=HS.skyscraper(M;anchor=1,weights=[3,1,1,1],witnesses=true)
        @test HS.skyscraper_slopes(r)==[2//9]
        @test only(HS.skyscraper_maps(r).factors).dims==[2,1,1,1]
        for seed in 1:3
            N,B=_hs_change_basis(M,MersenneTwister(seed))
            s=HS.skyscraper(N;anchor=1,weights=[3,1,1,1],witnesses=true)
            @test HS.skyscraper_slopes(s)==HS.skyscraper_slopes(r)
            @test HS.skyscraper_dimensions(s)==HS.skyscraper_dimensions(r)
            for i in eachindex(r.slopes),v in 1:4
                U=B[v]*r.witnesses.inclusions[i].comps[v];V=s.witnesses.inclusions[i].comps[v]
                @test _hs_rank(hcat(U,V))==size(U,2)
            end
        end
        # General weights with both positive and negative degrees, and zero module.
        thin=MD.PModule{K}(Q,ones(Int,4),Dict((1,v)=>ones(K,1,1) for v in 2:4);field=F)
        z=HS.StabilityWeights([1,2,3,4],[-2,4,3,0])
        h=HS.hn_filtration(thin,z;witnesses=true)
        @test HS.hn_slopes(h)==[2,1,0,-2]
        @test HS.hn_factors(h)==[[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0]]
        @test HS.check_hn_filtration(h;semistability=true).valid
        Z=MD.PModule{K}(Q,zeros(Int,4),Dict((1,v)=>zeros(K,0,0) for v in 2:4);field=F)
        @test isempty(HS.hn_slopes(HS.hn_filtration(Z,z)))
        @test isempty(HS.skyscraper_slopes(HS.skyscraper(Z;anchor=1)))
        # Every local slope equal proves semistability without enumerating subobjects.
        equal=HS.hn_filtration(M,HS.StabilityWeights([3,1,1,1],[6,2,2,2]);budget=HS.HNBudget(max_states=0,max_subspaces=0))
        @test HS.hn_slopes(equal)==[2]
    end
end

@testset "A51 exact chambers and volume refinement" begin
    for F in (CM.QQField(),CM.F2(),CM.F3())
        M=_hs_fork(F);g=HS.SkyscraperGrid(M,([0,1,3],[0,2,5]))
        c=HS.skyscraper_chambers(g;cells=[1])
        @test sum(HS._area(ch.polygon) for ch in c.chambers)==2
        # The crossing is 1+3x-2y=0, including a nonintegral wall point.
        for p in ((0,0),(0,1),(1//3,1),(3//4,7//4))
            at=HS.skyscraper_chamber_at(c;anchor=p,witnesses=true)
            x,y=p;m1=(3-x)*(2-y);m2=(1-x)*(5-y)
            expected=sort!(unique([CM.QQ(1)/m1,CM.QQ(1)/m2]);rev=true)
            @test HS.skyscraper_slopes(at.filtration)==expected
            @test at.boundary==(m1==m2)
        end
        @test_throws ArgumentError HS.skyscraper_chamber_at(c;anchor=(2,1))
        @test_throws ArgumentError HS.skyscraper_chambers(g;budget=HS.HNBudget(max_chambers=0))
        # Subdivide the first x cell; pull back maps along the floor classifier.
        Q=FF.ProductOfChainsPoset((3,2));labels=[1,1,2,3,3,4];K=CM.coeff_type(F)
        N=MD.PModule{K}(Q,M.dims[labels],Dict((u,v)=>Matrix(MD.map_leq(M,labels[u],labels[v])) for (u,v) in FF.cover_edges(Q));field=F)
        refined=HS.SkyscraperGrid(N,([0,1//2,1,3],[0,2,5]))
        for p in ((0,0),(1//4,1//3),(1//2,1),(3//4,7//4))
            @test HS.skyscraper_slopes(HS.skyscraper(g;anchor=p))==HS.skyscraper_slopes(HS.skyscraper(refined;anchor=p))
            for q in (p,(2,1+last(p)/2),(2,4)),t in (0,1//6,1//4,1)
                all(p .<= q) || continue
                @test HS.skyscraper_rank(g;source=p,target=q,threshold=t)==HS.skyscraper_rank(refined;source=p,target=q,threshold=t)
            end
        end
    end
end

@testset "A51 filtered landscape enclosures" begin
    F=CM.F2();K=CM.coeff_type(F);Q=FF.ProductOfChainsPoset((1,1))
    M=MD.PModule{K}(Q,[1],Dict{Tuple{Int,Int},Matrix{K}}();field=F)
    g=HS.SkyscraperGrid(M,([0,4],[0,6]))
    r=HS.filtered_landscape(g;points=[(2,3),(1,1),(4,1)],thresholds=[0,1//10,1//5],layers=[1,2],atol=1//256)
    # At (2,3), slope at the radius-r source is 1/((2+r)(3+r)).
    # For threshold 1/10 the crossing solves r^2+5r-4=0.
    exact=[2.0,(-5+sqrt(41))/2,0.0]
    for j in 1:3
        @test Float64(r.lower[1,j,1])<=exact[j]<=Float64(r.upper[1,j,1])
        @test r.upper[1,j,1]-r.lower[1,j,1]<=1//256
    end
    @test all(iszero,r.lower[:,:,2]) && all(iszero,r.upper[:,:,2])
    @test all(iszero,r.upper[3,:,:])
    d=HS.filtered_landscape_difference(r;threshold=1//10,baseline=0)
    @test d.lower[1,1]<=exact[2]-exact[1]<=d.upper[1,1]
    @test_throws ArgumentError HS.filtered_landscape(g;points=[(2,3)],atol=0)
    @test_throws ArgumentError HS.filtered_landscape(g;points=[(2,3)],atol=1//10^9,max_steps=1)
    @test_throws ArgumentError HS.FilteredLandscapeResult([(0,0)],[0],[1],fill(2,1,1,1),fill(1,1,1,1))
end
@testset "A51 exact finite stability" begin
    for F in (CM.F2(),CM.F3(),CM.Fp(5),CM.QQField())
        X=_hs_fork(F);Y=_hs_fork(F;coincident=true)
        for u in 1:4,v in 1:4
            FF.leq(X.Q,u,v) || continue
            @test _hs_rank(MD.map_leq(X,u,v))==_hs_rank(MD.map_leq(Y,u,v))
        end
        x=HS.skyscraper(X;anchor=1,witnesses=true)
        y=HS.skyscraper(Y;anchor=1,witnesses=true)
        @test HS.skyscraper_slopes(x)==[1//2]
        @test HS.skyscraper_slopes(y)==[1,1//3]
        @test HS.skyscraper_rank(x;target=1,threshold=3//4)==0
        @test HS.skyscraper_rank(y;target=1,threshold=3//4)==1
        @test HS.skyscraper_rank(y;target=2,threshold=1//3)==1
        @test HS.skyscraper_rank(y;target=2,threshold=1//3+1//100)==0
        for r in (x,y),i in HS.skyscraper_maps(r).inclusions
            @test MD.check_morphism(i).valid
            @test all(_hs_rank(i.comps[v])==i.dom.dims[v] for v in 1:4)
        end
        if F isa CM.PrimeField
            z=HS.StabilityWeights(ones(Int,4),[1,0,0,0])
            full=HS.hn_filtration(Y,z;witnesses=true)
            @test HS.hn_slopes(full)==[1,1//3]
            @test HS.check_hn_filtration(full;semistability=true).valid
            @test HS.hn_dimensions(full)==HS.skyscraper_dimensions(y)
        end
        g=HS.SkyscraperGrid(X,([0,1,3],[0,2,5]))
        r=HS.skyscraper(g;anchor=(0,0))
        # e1 occupies volume 2+4, e2 occupies 2+3; slopes 1/5 > 1/6.
        @test HS.skyscraper_slopes(r)==[1//5,1//6]
        @test HS.skyscraper_rank(g;source=(0,0),target=(0,0),threshold=1//5)==1
        @test HS.skyscraper_rank(g;source=(0,0),target=(3,5))==0
        @test_throws ArgumentError HS.skyscraper(g;anchor=(3,1))
    end
    @test_throws ArgumentError HS.StabilityWeights([0],[1])
    @test_throws ArgumentError HS.HNBudget(max_states=-1)
    @test_throws ArgumentError HS.skyscraper(_hs_fork(CM.F2());anchor=1,budget=HS.HNBudget(max_subspaces=2))
    @test_throws ArgumentError HS.skyscraper(_hs_fork(CM.F2());anchor=1,budget=HS.HNBudget(max_states=0))
end

# A separate finite-vector-set oracle: close under adding one vector, without
# RREF, nullspaces, or the production subspace enumerator.
function _hs_vector_spaces(F,d)
    K=CM.coeff_type(F)
    vectors=[Tuple(K.(x)) for x in Iterators.product(ntuple(_->0:F.p-1,d)...)]
    zero_space=Set([ntuple(_->zero(K),d)]);spaces=[zero_space];i=1
    while i<=length(spaces)
        U=spaces[i]
        for v in vectors
            v in U && continue
            S=Set(Tuple(u[j]+K(a)*v[j] for j in 1:d) for u in U for a in 0:F.p-1)
            any(==(S),spaces) || push!(spaces,S)
        end
        i+=1
    end
    spaces
end
function _hs_space_dimension(S,p)
    n=length(S);d=0
    while n>1;n=div(n,p);d+=1;end
    d
end
function _hs_semistable_oracle(M,z)
    F=M.field;choices=[_hs_vector_spaces(F,d) for d in M.dims]
    slope=sum(z.degree.*M.dims)/sum(z.mass.*M.dims)
    for candidate in Iterators.product(choices...)
        all(Tuple(M.edge_maps[u,v]*CM.coeff_type(F)[x...]) in candidate[v]
            for (u,v) in FF.cover_edges(M.Q) for x in candidate[u]) || continue
        dims=[_hs_space_dimension(S,F.p) for S in candidate]
        iszero(sum(dims)) && continue
        sum(z.degree.*dims)/sum(z.mass.*dims)<=slope || return false
    end
    true
end

@testset "A51 independent subobject and semistability oracles" begin
    for (F,d,expected_count) in ((CM.F2(),3,16),(CM.Fp(3),2,6),(CM.Fp(5),2,8))
        oracle=_hs_vector_spaces(F,d)
        production=HS._subspaces(F,d,HS.HNBudget(),HS._Work(0,0))
        @test length(oracle)==length(production)==expected_count
        for B in production
            image=Set(Tuple(B*CM.coeff_type(F)[x...]) for x in
                Iterators.product(ntuple(_->0:F.p-1,size(B,2))...))
            @test count(==(image),oracle)==1
        end
    end
    rng=MersenneTwister(5101)
    for F in (CM.F2(),CM.Fp(3)),trial in 1:8
        K=CM.coeff_type(F);Q=FF.ProductOfChainsPoset((2,))
        M=MD.PModule{K}(Q,[2,2],Dict((1,2)=>K.(rand(rng,0:F.p-1,2,2)));field=F)
        z=HS.StabilityWeights([2,3],trial%2==0 ? [3,-2] : [-2,3])
        r=HS.hn_filtration(M,z;witnesses=true)
        @test HS.check_hn_filtration(r;throw=true).valid
        @test all(N->_hs_semistable_oracle(N,z),HS.hn_maps(r).factors)
        @test all(diff(HS.hn_slopes(r)).<0)
    end
end

@testset "A51 checked splitting, public paths and validators" begin
    D=TamerOp.IndecomposableDecompositions
    for F in (CM.QQField(),CM.F2(),CM.Fp(3))
        M=_hs_fork(F)
        decomposition=D.indecomposable_decomposition(M;witnesses=true)
        z=HS.StabilityWeights([1,1,1,1],[1,0,0,0])
        r=HS.skyscraper(decomposition;anchor=1,witnesses=true)
        direct=HS.skyscraper(M;anchor=1,witnesses=true)
        lean=HS.skyscraper(decomposition;anchor=1)
        @test lean.witnesses===nothing && lean.slopes==r.slopes && lean.term_dimensions==r.term_dimensions
        @test HS.check_skyscraper(lean;semistability=true,throw=true).valid
        @test HS.skyscraper_slopes(r)==HS.skyscraper_slopes(direct)
        @test HS.skyscraper_dimensions(r)==HS.skyscraper_dimensions(direct)
        @test HS.check_skyscraper(r;semistability=true,throw=true).valid
        h=HS.hn_filtration(decomposition,z;witnesses=true)
        @test HS.hn_slopes(h)==[1//2]
        @test HS.check_hn_filtration(h;semistability=F isa CM.PrimeField,throw=true).valid
        @test TamerOp.hn_slopes(TamerOp.hn_filtration(decomposition,z))==[1//2]
        enc=TamerOp.Results.EncodingResult(M.Q,M,nothing)
        @test TamerOp.skyscraper_summary(TamerOp.skyscraper(enc;anchor=1)).slopes==[1//2]
        @test_throws ArgumentError HS.skyscraper(decomposition;anchor=1,budget=HS.HNBudget(max_subspaces=3))
        @test_throws ArgumentError HS.skyscraper(decomposition;anchor=1,weights=[1])
        # Malformed stored witnesses must fail validation rather than leak errors.
        r.witnesses.factor_projections[1].comps[1][:].=0
        @test !HS.check_skyscraper(r).valid
        @test_throws ArgumentError HS.check_skyscraper(r;throw=true)
        h.witnesses.factor_projections[1].comps[1][:].=0
        @test !HS.check_hn_filtration(h).valid
        D.decomposition_maps(decomposition).projections[1].comps[1][:].=0
        @test_throws ArgumentError HS.skyscraper(decomposition;anchor=1)
    end
    F=CM.QQField();N=_hs_fork(F);M=MD.direct_sum(N,N)
    d=D.indecomposable_decomposition(M;witnesses=true)
    direct=HS.skyscraper(M;anchor=1,witnesses=true)
    @test HS.check_skyscraper(direct;semistability=true,throw=true).valid
    r=HS.skyscraper(d;anchor=1,witnesses=true)
    @test HS.skyscraper_slopes(r)==[1//2]
    @test HS.skyscraper_dimensions(r)==[M.dims]
    @test direct.slopes==r.slopes && direct.term_dimensions==r.term_dimensions
    @test HS.check_skyscraper(r).valid
end

@testset "A51 retained filtration budgets" begin
    F=CM.QQField();K=CM.coeff_type(F)
    Q=FF.FinitePoset(Matrix{Bool}(I,8,8))
    M=MD.PModule{K}(Q,ones(Int,8),Dict{Tuple{Int,Int},Matrix{K}}();field=F)
    z=HS.StabilityWeights(ones(Int,8),1:8)
    @test_throws ArgumentError HS.hn_filtration(M,z;budget=HS.HNBudget(max_entries=64))
    @test length(HS.hn_slopes(HS.hn_filtration(M,z;budget=HS.HNBudget(max_entries=72))))==8
    @test_throws ArgumentError HS.hn_filtration(M,z;witnesses=true,budget=HS.HNBudget(max_entries=72))
end


@testset "A51 landscape correlation and supplied monotonicity" begin
    lo=reshape([0,0,7],1,3,1);hi=reshape([5,10,8],1,3,1)
    @test_throws ArgumentError HS.FilteredLandscapeResult([(0,0)],[0,1,2],[1],lo,hi)
    r=HS.FilteredLandscapeResult([(0,0)],[0],[1],reshape([1],1,1,1),reshape([2],1,1,1))
    d=HS.filtered_landscape_difference(r;threshold=0,baseline=0)
    @test all(iszero,d.lower) && all(iszero,d.upper)
end

# Intersect all maximizers in a separate vector-set search. This checks the
# minimum optimizer itself, including zero-weight stalks and closure along paths.
function _hs_weight_oracle(M,theta)
    F=M.field;choices=[_hs_vector_spaces(F,d) for d in M.dims]
    best=nothing;answer=nothing
    for candidate in Iterators.product(choices...)
        all(Tuple(M.edge_maps[u,v]*CM.coeff_type(F)[x...]) in candidate[v]
            for (u,v) in FF.cover_edges(M.Q) for x in candidate[u]) || continue
        value=sum(theta[v]*_hs_space_dimension(candidate[v],F.p) for v in eachindex(M.dims))
        if best===nothing || value>best
            best=value;answer=collect(copy.(candidate))
        elseif value==best
            answer=intersect.(answer,candidate)
        end
    end
    best,answer
end

@testset "A51 general matrix-space optimality" begin
    rng=MersenneTwister(5151)
    for F in (CM.F2(),CM.F3()),trial in 1:12
        K=CM.coeff_type(F)
        Q=FF.FinitePoset(Bool[1 1 1;0 1 0;0 0 1])
        M=MD.PModule{K}(Q,[2,2,1],Dict((1,2)=>K.(rand(rng,0:F.p-1,2,2)),
            (1,3)=>K.(rand(rng,0:F.p-1,1,2)));field=F)
        theta=CM.QQ.(rand(rng,-3:3,3))
        primitive=HS._integer_weights(theta,M.dims)
        value,spaces=_hs_weight_oracle(M,primitive)
        B,actual=HS._maximum_weight_submodule(M,theta,HS.HNBudget(),HS._Work(0,0))
        @test actual==value
        for v in 1:3
            vectors=Set(Tuple(B[v]*K[x...]) for x in Iterators.product(ntuple(_->0:F.p-1,size(B[v],2))...))
            @test vectors==spaces[v]
        end
        z=HS.StabilityWeights([1,2,3],theta)
        h=HS.hn_filtration(M,z;witnesses=true)
        @test HS.check_hn_filtration(h;throw=true).valid
        @test all(N->_hs_semistable_oracle(N,z),h.witnesses.factors)
    end
    # Odd alternating matrices have commutative rank two but noncommutative
    # rank three. A rank test at blowup one must not certify instability.
    for F in (CM.QQField(),CM.F2(),CM.F3(),CM.Fp(5))
        K=CM.coeff_type(F)
        A=K[0 1 0;-1 0 0;0 0 0];B=K[0 0 1;0 0 0;-1 0 0];C=K[0 0 0;0 0 1;0 -1 0]
        S=HS._WeightSpace(F,[3],[3],[1],[1],[(1,1,A),(1,1,B),(1,1,C)])
        @test_throws ArgumentError HS._minimum_shrunk(S,HS.HNBudget(max_blowup=1),HS._Work(0,0))
        w=HS._Work(0,0);U,defect=HS._minimum_shrunk(S,HS.HNBudget(),w)
        @test defect==0 && size(only(U),2)==0 && w.largest_blowup>=2
    end
    # Commuting diamond paths, with a negative vertex reached through
    # zero-weight intermediates. This checks the poset-path reduction itself.
    F=CM.F2();K=CM.coeff_type(F);Q=FF.ProductOfChainsPoset((2,2));A=K[1 1;0 0]
    M=MD.PModule{K}(Q,fill(2,4),Dict((1,2)=>Matrix{K}(I,2,2),(1,3)=>Matrix{K}(I,2,2),
        (2,4)=>A,(3,4)=>A);field=F)
    for theta in (CM.QQ[3,0,0,-2],CM.QQ[1,2,-3,-1],CM.QQ[-1,2,2,-2])
        value,spaces=_hs_weight_oracle(M,HS._integer_weights(theta,M.dims))
        B,actual=HS._maximum_weight_submodule(M,theta,HS.HNBudget(max_subspaces=0),HS._Work(0,0))
        @test value==actual
        for v in 1:4
            @test Set(Tuple(B[v]*K[x...]) for x in Iterators.product(ntuple(_->0:1,size(B[v],2))...))==spaces[v]
        end
    end
    # Thin closure uses exact capacities, rather than copies proportional to
    # coefficient size. A huge positive degree must still be inexpensive.
    Q=FF.ProductOfChainsPoset((3,));M=MD.PModule{K}(Q,[1,1,1],Dict((1,2)=>ones(K,1,1),(2,3)=>ones(K,1,1));field=F)
    theta=CM.QQ[big(10)^40,0,-1]
    B,value=HS._maximum_weight_submodule(M,theta,HS.HNBudget(max_trials=0),HS._Work(0,0))
    @test value==big(10)^40-1 && all(size(A,2)==1 for A in B)
    # The rank-two kernel family is complete over large prime fields too;
    # it must not enumerate all p+1 projective lines.
    F=CM.Fp(101);M=_hs_fork(F)
    r=HS.skyscraper(M;anchor=1,budget=HS.HNBudget(max_subspaces=4,max_trials=0))
    @test r.slopes==[1//2] && r.statistics.subspaces==4 && r.statistics.matrix_trials==0
    # Independent one-map pieces must not require forty simultaneous lucky
    # scalar draws. Their exact kernel formula also removes weight-size costs.
    for F in (CM.QQField(),CM.F2(),CM.F3())
        K=CM.coeff_type(F);count=40;n=2count
        order=Matrix{Bool}(I,n,n)
        for i in 1:count;order[i,count+i]=true;end
        Q=FF.FinitePoset(order)
        M=MD.PModule{K}(Q,fill(2,n),Dict((i,count+i)=>K[1 0;0 0] for i in 1:count);field=F)
        z=HS.StabilityWeights(ones(Int,n),[ones(Int,count);zeros(Int,count)])
        h=HS.hn_filtration(M,z;budget=HS.HNBudget(max_trials=0,max_entries=10000))
        @test h.slopes==[1,1//2,0]
        @test h.term_dimensions==[[ones(Int,count);zeros(Int,count)],
            [fill(2,count);ones(Int,count)],fill(2,n)]
        @test h.statistics.matrix_trials==0
        theta=CM.QQ[fill(big(10)^25,count);fill(-big(10)^25-1,count)]
        B,value=HS._maximum_weight_submodule(M,theta,HS.HNBudget(max_trials=0,max_entries=10000),HS._Work(0,0))
        @test value==count*big(10)^25
        @test [size(U,2) for U in B]==[ones(Int,count);zeros(Int,count)]
    end
end

@testset "A51 arbitrary rational thickness and budgets" begin
    F=CM.QQField();K=CM.coeff_type(F);Q=FF.ProductOfChainsPoset((4,))
    supports=[(1,4),(1,2),(2,4),(2,3),(3,3),(1,1)]
    modules=[MD.PModule{K}(Q,[Int(a<=v<=b) for v in 1:4],
        Dict((u,v)=>ones(K,Int(a<=v<=b),Int(a<=u<=b)) for (u,v) in FF.cover_edges(Q));field=F) for (a,b) in supports]
    M=reduce(MD.direct_sum,modules);z=HS.StabilityWeights([1,1,1,1],[3,2,1,0])
    slopes=[sum(z.degree[a:b])/sum(z.mass[a:b]) for (a,b) in supports]
    expected=sort(unique(slopes);rev=true)
    for seed in 1:3
        N,B=_hs_change_basis(M,MersenneTwister(seed))
        h=HS.hn_filtration(N,z;witnesses=true)
        @test h.slopes==expected
        @test h.term_dimensions==[sum(modules[i].dims for i in eachindex(modules) if slopes[i]>=s) for s in expected]
        @test HS.check_hn_filtration(h;semistability=true,throw=true).valid
        @test h.route==:certified_matrix_space
        for (j,s) in enumerate(expected),v in 1:4
            selected=findall(i->slopes[i]>=s,eachindex(modules))
            columns=Int[];offset=0
            for i in eachindex(modules)
                i in selected && append!(columns,offset+1:offset+modules[i].dims[v])
                offset+=modules[i].dims[v]
            end
            U=B[v][:,columns];V=h.witnesses.inclusions[j].comps[v]
            @test _hs_rank(hcat(U,V))==size(U,2)
        end
    end
    zscaled=HS.StabilityWeights(z.mass./17,z.degree./31)
    @test HS.hn_filtration(M,zscaled).slopes==expected.*(17//31)
    @test_throws ArgumentError HS.hn_filtration(M,z;budget=HS.HNBudget(max_trials=0))
    @test_throws ArgumentError HS.hn_filtration(M,z;budget=HS.HNBudget(max_blowup=0))
    @test_throws ArgumentError HS.HNBudget(max_trials=-1)
    @test_throws ArgumentError HS.HNBudget(max_blowup=true)
    huge=HS.StabilityWeights([1,1,1,1],[big(10)^50,1,-1,-2])
    @test_throws ArgumentError HS.hn_filtration(M,huge)
    # A rational, noninterval rank-three arrangement with every triple of
    # forms independent is semistable; no supplied decomposition is involved.
    P=FF.FinitePoset(Bool[1 1 1 1 1;0 1 0 0 0;0 0 1 0 0;0 0 0 1 0;0 0 0 0 1])
    N=MD.PModule{K}(P,[3,1,1,1,1],Dict((1,t+2)=>reshape(K[1,t,t*t],1,3) for t in 0:3);field=F)
    r=HS.skyscraper(N;anchor=1,witnesses=true)
    @test r.slopes==[3//7] && r.term_dimensions==[N.dims]
    @test HS.check_skyscraper(r;semistability=true,throw=true).valid
    @test HS.skyscraper(N;anchor=1).witnesses===nothing
    @test TamerOp.skyscraper(N;anchor=1).slopes==r.slopes
end

@testset "A51 general planar chambers and ambient queries" begin
    for F in (CM.QQField(),CM.F2())
        X=_hs_fork(F);M=MD.direct_sum(X,X)
        M,_=_hs_change_basis(M,MersenneTwister(517))
        g=HS.SkyscraperGrid(M,([0,1,3],[0,2,5]))
        chambers=HS.skyscraper_chambers(g;cells=[1])
        @test length(chambers.chambers)==2
        @test sum(HS._area(c.polygon) for c in chambers.chambers)==2
        @test chambers.statistics.matrix_trials>0
        for p in ((0,0),(0,1),(1//3,1),(3//4,7//4),(1//2,1//4))
            x,y=p;expected=sort(unique([inv(CM.QQ((3-x)*(2-y))),inv(CM.QQ((1-x)*(5-y)))]);rev=true)
            at=HS.skyscraper_chamber_at(chambers;anchor=p,witnesses=true)
            @test at.filtration.slopes==expected
            @test at.boundary==((3-x)*(2-y)==(1-x)*(5-y))
            @test HS.check_skyscraper(at.filtration;semistability=true,throw=true).valid
            @test at.filtration.statistics.matrix_trials==0
        end
        # Reuse a certified flag at an interior or exact wall coordinate.
        # These denominators would make cleared-weight matrix copies enormous.
        x=CM.QQ(1,big(10)^20+39)
        for y in (CM.QQ(1,4),(1+3x)/2)
            at=HS.skyscraper_chamber_at(chambers;anchor=(x,y),witnesses=true,
                budget=HS.HNBudget(max_trials=0,max_blowup=0))
            expected=sort(unique([inv((3-x)*(2-y)),inv((1-x)*(5-y))]);rev=true)
            @test at.filtration.slopes==expected
            @test at.boundary==(y==(1+3x)/2)
            @test HS.check_skyscraper(at.filtration;throw=true).valid
            @test at.filtration.context.route==:certified_chamber
        end
        @test_throws ArgumentError HS.skyscraper_chambers(g;cells=[1],budget=HS.HNBudget(max_subspaces=1))
        @test_throws ArgumentError HS.skyscraper_chambers(g;cells=[1],budget=HS.HNBudget(max_trials=0))
    end
    # A thick constant module on a three-parameter box: volume, not the
    # dimension of the ambient parameter space, determines its unique slope.
    F=CM.QQField();K=CM.coeff_type(F);Q=FF.ProductOfChainsPoset((1,1,1))
    M=MD.PModule{K}(Q,[3],Dict{Tuple{Int,Int},Matrix{K}}();field=F)
    g=HS.SkyscraperGrid(M,([0,4],[0,6],[0,8]))
    @test_throws ArgumentError HS.skyscraper_chambers(g)
    r=HS.skyscraper(g;anchor=(1//2,1,2))
    @test r.slopes==[1//105] && r.term_dimensions==[[3]]
    @test HS.skyscraper_rank(g;source=(1//2,1,2),target=(3,5,7),threshold=1//105)==3
    landscape=HS.filtered_landscape(g;points=[(2,3,4)],thresholds=[0],layers=[1,3,4],atol=1//64)
    @test landscape.lower[1,1,1]<=2<=landscape.upper[1,1,1]
    @test landscape.lower[1,1,3]==landscape.upper[1,1,3]==0
    # A thick planar landscape reuses certified flags even as bisection creates
    # fine rational coordinates; no matrix search is allowed during evaluation.
    X=_hs_fork(F);small=HS.SkyscraperGrid(X,([0,1,3],[0,2,5]))
    large=HS.SkyscraperGrid(MD.direct_sum(X,X),([0,1,3],[0,2,5]))
    prepared=HS.skyscraper_chambers(large)
    points=[(1//2,1),(3//2,1),(1//2,3)];thresholds=[0,1//8]
    expected=HS.filtered_landscape(small;points,thresholds,layers=[1],atol=1//1024)
    actual=HS.filtered_landscape(prepared;points,thresholds,layers=[1,2],atol=1//1024,
        budget=HS.HNBudget(max_trials=0,max_blowup=0))
    for k in 1:2
        @test actual.lower[:,:,k]==expected.lower[:,:,1]
        @test actual.upper[:,:,k]==expected.upper[:,:,1]
    end
    @test actual.context.query_route==:prepared_chambers
    partial=HS.skyscraper_chambers(large;cells=[1])
    @test_throws ArgumentError HS.filtered_landscape(partial;points=[(3//2,1)])
end


@testset "A51 weighted geometric cells and refinement" begin
    for F in (CM.QQField(),CM.F2(),CM.F3())
        M=_hs_fork(F)
        g=HS.SkyscraperGrid(M,([0,1,3],[0,2,5]);density=[2,3,5,7])
        chambers=HS.skyscraper_chambers(g;cells=[1])
        @test sum(HS._area(c.polygon) for c in chambers.chambers)==2
        for (x,y) in ((0,0),(1//2,3//4),(1//3,1//2),(3//4,1//4))
            m1=2*(1-x)*(2-y)+6*(2-y)
            m2=2*(1-x)*(2-y)+15*(1-x)
            expected=sort!(unique([inv(CM.QQ(m1)),inv(CM.QQ(m2))]);rev=true)
            r=HS.skyscraper(g;anchor=(x,y),witnesses=true)
            @test HS.skyscraper_slopes(r)==expected
            @test HS.check_skyscraper(r;semistability=true).valid
            at=HS.skyscraper_chamber_at(chambers;anchor=(x,y))
            @test HS.skyscraper_slopes(at.filtration)==expected
            @test at.boundary==(m1==m2)
            @test HS.provenance(r).measure==:piecewise_constant_density
        end
        # A constant module has mass equal to the weighted volume; refine cells
        # without changing either the module or the density function.
        K=CM.coeff_type(F);Q=FF.ProductOfChainsPoset((3,2))
        N=MD.PModule{K}(Q,ones(Int,6),Dict(e=>ones(K,1,1) for e in FF.cover_edges(Q));field=F)
        fine=HS.SkyscraperGrid(N,([0,1//2,1,3],[0,2,5]);density=[2,2,3,5,5,7])
        Q0=FF.ProductOfChainsPoset((2,2))
        N0=MD.PModule{K}(Q0,ones(Int,4),Dict(e=>ones(K,1,1) for e in FF.cover_edges(Q0));field=F)
        coarse=HS.SkyscraperGrid(N0,([0,1,3],[0,2,5]);density=[2,3,5,7])
        for p in ((0,0),(1//4,1//2),(1//2,2),(1,3))
            x,y=p
            expected=sum(rho*max(0,hi[1]-max(lo[1],x))*max(0,hi[2]-max(lo[2],y))
                for (rho,lo,hi) in ((2,(0,0),(1,2)),(3,(1,0),(3,2)),(5,(0,2),(1,5)),(7,(1,2),(3,5))))
            @test only(HS.skyscraper_slopes(HS.skyscraper(fine;anchor=p)))==1/CM.QQ(expected)
            @test HS.skyscraper_slopes(HS.skyscraper(fine;anchor=p))==HS.skyscraper_slopes(HS.skyscraper(coarse;anchor=p))
        end
        # Scaling the measure by c scales every slope by 1/c, including landscapes.
        plain=HS.SkyscraperGrid(M,([0,1,3],[0,2,5]))
        scaled=HS.SkyscraperGrid(M,([0,1,3],[0,2,5]);density=3)
        a=HS.filtered_landscape(plain;points=[(1//2,1)],thresholds=[0,1//5],atol=1//64)
        b=HS.filtered_landscape(scaled;points=[(1//2,1)],thresholds=[0,1//15],atol=1//64)
        @test HS.filtered_landscape_values(a)==HS.filtered_landscape_values(b)
        c=HS.filtered_landscape(HS.skyscraper_chambers(scaled);points=[(1//2,1)],thresholds=[0,1//15],atol=1//64)
        @test HS.filtered_landscape_values(c)==HS.filtered_landscape_values(b)
    end
    M=_hs_fork(CM.F2())
    @test_throws ArgumentError HS.SkyscraperGrid(M,([0,1,3],[0,2,5]);density=[1,0,1,1])
    @test_throws DimensionMismatch HS.SkyscraperGrid(M,([0,1,3],[0,2,5]);density=[1,2])
end

@testset "A51 encoded box and polyhedral geometric queries" begin
    for F in (CM.QQField(),CM.F2(),CM.F3())
        Q,H,pi=PLB.encode_fringe_boxes([PLB.BoxUpset([0,0])],[PLB.BoxDownset([3,2])],
            ones(Int,1,1),TamerOp.Options.EncodingOptions(field=F))
        M=TamerOp.IndicatorResolutions.pmodule_from_fringe(H)
        enc=RES.EncodingResult(Q,M,pi)
        g=HS.SkyscraperEncoding(enc;box=([-1,-1],[4,3]))
        r=HS.skyscraper(g;anchor=(1//3,1//2),witnesses=true)
        @test HS.skyscraper_slopes(r)==[1//4]
        @test HS.check_skyscraper(r;semistability=true).valid
        @test HS.skyscraper_rank(g;source=(1//3,1//2),target=(3,2),threshold=1//4)==1
        @test HS.skyscraper_rank(g;source=(1//3,1//2),target=(3,2),threshold=1//4+1//100)==0
        @test HS.skyscraper_rank(g;source=(1//3,1//2),target=(4,2))==0
        @test_throws ArgumentError HS.skyscraper(g;anchor=(3,2))
        @test_throws ArgumentError HS.skyscraper(g;anchor=(4,2))
        @test HS.skyscraper_slopes(TamerOp.skyscraper(enc;box=([-1,-1],[4,3]),density=2,anchor=(1//3,1//2)))==[1//8]
        @test TamerOp.SkyscraperEncoding===HS.SkyscraperEncoding
        @test HS.describe(g).measure==:lebesgue
        @test occursin("SkyscraperEncoding",sprint(show,g))
        @test_throws ArgumentError TamerOp.skyscraper(enc;density=2,anchor=1)
        @test_throws ArgumentError HS.SkyscraperEncoding(enc;box=([0,0],[0,1]))
        @test_throws ArgumentError HS.SkyscraperEncoding(enc;box=([0,0],[1,1]),budget=HS.HNBudget(max_states=0))
        weighted=HS.SkyscraperEncoding(enc;box=([-1,-1],[4,3]),density=fill(3,length(M.dims)))
        @test HS.skyscraper_slopes(HS.skyscraper(weighted;anchor=(1//3,1//2)))==[1//12]
        landscape=HS.filtered_landscape(g;points=[(3//2,1)],thresholds=[0],atol=1//64)
        @test only(landscape.lower)<=1<=only(landscape.upper)
        @test only(landscape.upper)-only(landscape.lower)<=1//64
    end
    # Nonrectangular support: a diagonal strip cut by the positive quadrant.
    for epsilon in (CM.QQ(1),big(1)//big(2)^80)
        U=PLP.PLUpset(PLP.poly_union(PLP.make_hpoly(CM.QQ[-1 -1],CM.QQ[0])))
        D=PLP.PLDownset(PLP.poly_union(PLP.make_hpoly(CM.QQ[1 1],CM.QQ[epsilon])))
        enc=TamerOp.encode(PLP.PLFringe([U],[D],ones(Int,1,1)),TamerOp.Options.EncodingOptions(backend=:pl))
        g=HS.SkyscraperEncoding(enc;box=((0,0),(2,2)),density=3)
        p=(epsilon/10,epsilon/5)
        r=HS.skyscraper(g;anchor=p,witnesses=true)
        @test only(HS.skyscraper_slopes(r))==200/(147epsilon^2)
        @test HS.check_skyscraper(r;semistability=true).valid
        @test HS.skyscraper_rank(g;source=p,target=(epsilon/2,epsilon/2))==1
        @test HS.skyscraper_rank(g;source=p,target=(epsilon,epsilon))==0
        @test_throws ArgumentError HS.skyscraper(g;anchor=(epsilon/2,epsilon/2))
    end
    # Exact simplex volume in three parameters, including a lower-dimensional
    # anchor stratum whose generated module nonetheless has positive mass.
    U=PLP.PLUpset(PLP.poly_union(PLP.make_hpoly(-Matrix{CM.QQ}(I,3,3),zeros(CM.QQ,3))))
    D=PLP.PLDownset(PLP.poly_union(PLP.make_hpoly(ones(CM.QQ,1,3),CM.QQ[1])))
    enc=TamerOp.encode(PLP.PLFringe([U],[D],ones(Int,1,1)),TamerOp.Options.EncodingOptions(backend=:pl))
    g=HS.SkyscraperEncoding(enc;box=((0,0,0),(2,2,2)))
    @test only(HS.skyscraper_slopes(HS.skyscraper(g;anchor=(0,0,0))))==6
    @test only(HS.skyscraper_slopes(HS.skyscraper(g;anchor=(1//10,1//5,1//5))))==48
    @test_throws ArgumentError HS.skyscraper(g;anchor=(0,0,0),budget=HS.HNBudget(max_states=1))
    @test_throws ArgumentError HS.skyscraper(g;anchor=(0,0,0),budget=HS.HNBudget(max_entries=1))
end

@testset "A51 finite-field extension certificates" begin
    for F in (CM.F2(),CM.F3(),CM.Fp(5))
        K=CM.coeff_type(F);b=HS.HNBudget();w=HS._Work(0,0)
        powers=HS._extension_basis(F,3,b,w)
        @test powers[1]==CM.eye(F,3)
        # Every nonzero extension element acts invertibly; products remain in
        # the same field algebra, including after a change of representation.
        matrices=[sum((K(c[i])*powers[i] for i in 1:3)) for c in Iterators.product(ntuple(_->0:F.p-1,3)...)]
        @test length(unique(vec.(matrices)))==F.p^3
        @test all(A->iszero(A) || _hs_rank(A)==3,matrices)
        @test all(A*B in matrices for A in matrices[1:min(end,12)],B in matrices[1:min(end,12)])
        @test HS._extension_basis(F,3,b,HS._Work(0,0))==powers
        A=K[0 1 0;-1 0 0;0 0 0];B=K[0 0 1;0 0 0;-1 0 0];C=K[0 0 0;0 0 1;0 -1 0]
        S=HS._WeightSpace(F,[3],[3],[1],[1],[(1,1,A),(1,1,B),(1,1,C)])
        state=HS._Work(0,0);U,defect=HS._minimum_shrunk(S,b,state)
        @test defect==0 && size(only(U),2)==0
        @test state.extension_trials>0 && state.largest_extension>1
        @test_throws ArgumentError HS._minimum_shrunk(S,HS.HNBudget(max_extension_degree=1),HS._Work(0,0))
        @test_throws ArgumentError HS._minimum_shrunk(S,HS.HNBudget(max_trials=4),HS._Work(0,0))
        @test_throws ArgumentError HS._extension_basis(F,3,HS.HNBudget(max_states=0),HS._Work(0,0))
        # Add a common kernel to the alternating space. Descent must recover
        # that actual subspace, not merely its dimension or a rank estimate.
        S2=HS._WeightSpace(F,[4],[3],[1],[1],[(i,j,hcat(T,zeros(K,3,1))) for (i,j,T) in S.maps])
        U2,defect2=HS._minimum_shrunk(S2,b,HS._Work(0,0))
        @test defect2==1 && _hs_rank(hcat(only(U2),K[0,0,0,1]))==1
    end
    @test_throws ArgumentError HS.HNBudget(max_extension_degree=-1)
end

@testset "A51 exact rational volume oracles" begin
    # Rationally sheared simplices and cubes, with independent determinant
    # formulas. Duplicate facets and their order must not duplicate volume.
    for n in 1:4
        B=Matrix{CM.QQ}(I,n,n)
        for i in 1:n
            B[i,i]=(i+1)//i
            for j in i+1:n;B[i,j]=1//(i+j);end
        end
        inverse=inv(B);boxlo=fill(CM.QQ(-5),n);boxhi=fill(CM.QQ(5),n)
        for simplex in (true,false)
            A=simplex ? vcat(-inverse,ones(CM.QQ,1,n)*inverse) : vcat(-inverse,inverse)
            b=simplex ? vcat(zeros(CM.QQ,n),CM.QQ[1]) : vcat(zeros(CM.QQ,n),ones(CM.QQ,n))
            expected=abs(det(B))/(simplex ? factorial(big(n)) : 1)
            h=PLP.make_hpoly(A,b)
            @test PLP._exact_poly_volume(h,boxlo,boxhi,Ref(0),100000,100000)==expected
            if n<=3
                order=vcat(reverse(1:size(A,1)),1,1)
                redundant=PLP.make_hpoly(A[order,:],b[order])
                @test PLP._exact_poly_volume(redundant,boxlo,boxhi,Ref(0),100000,100000)==expected
            end
        end
        flat=PLP.make_hpoly(vcat(Matrix{CM.QQ}(I,n,n),-Matrix{CM.QQ}(I,n,n)),zeros(CM.QQ,2n))
        @test iszero(PLP._exact_poly_volume(flat,boxlo,boxhi,Ref(0),100000,100000))
    end
    # Adding an inactive generator refines the encoding but changes neither
    # the module nor its geometric measure.
    F=CM.QQField();opts=TamerOp.Options.EncodingOptions(field=F)
    r1=PLB.encode_fringe_boxes([PLB.BoxUpset([0,0])],[PLB.BoxDownset([3,2])],ones(Int,1,1),opts)
    r2=PLB.encode_fringe_boxes([PLB.BoxUpset([0,0]),PLB.BoxUpset([1,1])],[PLB.BoxDownset([3,2])],reshape([1,0],1,2),opts)
    grids=map((r1,r2)) do (Q,H,pi)
        M=TamerOp.IndicatorResolutions.pmodule_from_fringe(H)
        HS.SkyscraperEncoding(M,pi;box=([0,0],[4,3]),density=2)
    end
    for p in ((0,0),(1//2,1//2),(1,1),(2,3//2))
        @test HS.skyscraper_slopes(HS.skyscraper(grids[1];anchor=p))==HS.skyscraper_slopes(HS.skyscraper(grids[2];anchor=p))
    end
    @test HS.check_skyscraper(grids[1];throw=true).valid
    @test_throws ArgumentError HS.skyscraper_chambers(grids[1])
    n=length(grids[1].finite.dims)
    wrong=MD.PModule{CM.QQ}(FF.FinitePoset(Matrix{Bool}(I,n,n)),copy(grids[1].finite.dims),
        Dict{Tuple{Int,Int},Matrix{CM.QQ}}();field=F)
    @test_throws ArgumentError HS.SkyscraperEncoding(wrong,grids[1].classifier;box=grids[1].box)
    @test HS.field(grids[1])==F
    @test HS.dimensions(grids[1])==grids[1].finite.dims
    broken=deepcopy(grids[1]);broken.density[1]=0
    @test !HS.check_skyscraper(broken).valid
    @test_throws ArgumentError HS.check_skyscraper(broken;throw=true)
end

@testset "A51 simultaneous small-field rank certificates" begin
    # Twelve independent coordinate projections share one source stalk. Over F2
    # an ordinary random proposal is full rank with probability 2^-12, whereas
    # extension sampling can certify the same known semistable module.
    F=CM.F2();K=CM.coeff_type(F);n=12
    relation=Matrix{Bool}(I,n+1,n+1);relation[1,:].=true
    Q=FF.FinitePoset(relation)
    maps=Dict((1,i+1)=>reshape(K[j==i for j in 1:n],1,n) for i in 1:n)
    M=MD.PModule{K}(Q,vcat(n,ones(Int,n)),maps;field=F)
    @test_throws ArgumentError HS.skyscraper(M;anchor=1,budget=HS.HNBudget(max_trials=4))
    r=HS.skyscraper(M;anchor=1,witnesses=true)
    @test r.slopes==[1//2]
    @test r.term_dimensions==[M.dims]
    @test r.statistics.extension_trials>0
    @test HS.check_skyscraper(r;semistability=true).valid
    # Every basis vector generates one of twelve disjoint intervals of mass 2.
    @test r.witnesses.terms[1].dims==M.dims
    @test all(MD.check_morphism(f).valid for f in r.witnesses.inclusions)
    @test_throws ArgumentError HS.skyscraper(M;anchor=1,budget=HS.HNBudget(max_trials=4,max_extension_degree=0))
end

@testset "A51 zero-measure anchor strata and nonconvex support" begin
    Q,H,pi=PLB.encode_fringe_boxes([PLB.BoxUpset([0,0]),PLB.BoxUpset([1,0])],
        [PLB.BoxDownset([3,2]),PLB.BoxDownset([1,2])],[1 0;0 0])
    M=TamerOp.IndicatorResolutions.pmodule_from_fringe(H)
    for thickness in (1,3)
        K=CM.QQ
        thick=MD.PModule{K}(Q,thickness.*M.dims,
            Dict(e=>kron(A,Matrix{K}(I,thickness,thickness)) for (e,A) in M.edge_maps);field=CM.QQField())
        g=HS.SkyscraperEncoding(thick,pi;box=([0,0],[4,3]))
        r=HS.skyscraper(g;anchor=(1,1),witnesses=true)
        @test iszero(r.weights[r.anchor])
        @test r.slopes==[1//2]
        @test HS.check_skyscraper(r;semistability=true).valid
        @test HS.skyscraper_rank(g;source=(1,1),target=(2,1),threshold=1//2)==thickness
    end
    # An L-shaped downset has area 3. Clipping at (1/2,1/2) leaves area 5/4.
    U=PLP.PLUpset(PLP.poly_union(PLP.make_hpoly(-Matrix{CM.QQ}(I,2,2),zeros(CM.QQ,2))))
    D=PLP.PLDownset(PLP.PolyUnion(2,[PLP.make_hpoly(Matrix{CM.QQ}(I,2,2),CM.QQ[2,1]),
                                    PLP.make_hpoly(Matrix{CM.QQ}(I,2,2),CM.QQ[1,2])]))
    enc=TamerOp.encode(PLP.PLFringe([U],[D],ones(Int,1,1)),TamerOp.Options.EncodingOptions(backend=:pl))
    g=HS.SkyscraperEncoding(enc;box=([0,0],[3,3]))
    @test only(HS.skyscraper_slopes(HS.skyscraper(g;anchor=(0,0))))==1//3
    @test only(HS.skyscraper_slopes(HS.skyscraper(g;anchor=(1//2,1//2))))==4//5
    @test HS.skyscraper_rank(g;source=(0,0),target=(3//2,3//2))==0
end
