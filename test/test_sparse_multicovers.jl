@testset "A69 public sparse multicover surface" begin
    S=TamerOp.SparseMulticovers
    @test TamerOp.Advanced.SparseMulticovers===S
    for name in (:SparseMulticoverBudget,:SparseMulticover,:SparseMulticoverValidationSummary,:check_sparse_multicover)
        @test getproperty(TamerOp.Advanced,name)===getproperty(S,name)
    end
    for name in (:sparse_multicover,:sparse_multicover_summary,:sparse_multicover_at,:sparse_covering,:sparse_ball_radii,:sparse_multicover_cells,:multicover_witness)
        @test getproperty(TamerOp,name)===getproperty(S,name)
    end
end

@testset "A69 grade intersection preserves open and closed events" begin
    S=TamerOp.SparseMulticovers;AR=TamerOp.AlgebraicReal;G=S.SparseMulticoverGrade
    a=[G(AR(0),false,1),G(AR(1),false,3),G(AR(1),true,5)]
    b=[G(AR(0),true,2),G(AR(1),false,4),G(AR(2),false,6)]
    expected=[G(AR(0),true,1),G(AR(1),false,3),G(AR(1),true,4),G(AR(2),false,5)]
    @test S._intersect_grades(a,b)==expected
    @test isempty(S._intersect_grades(a,G[]))
    @test isempty(S._intersect_grades(G[],b))
    # The Cartesian definition supplies an independent oracle for the merge.
    rng=MersenneTwister(69015)
    radii=[AR(0),AR(1)/2,AR(1),sqrt(AR(2)),AR(2)]
    for _ in 1:120
        a=S._antichain([G(rand(rng,radii),rand(rng,Bool),rand(rng,1:9)) for _ in 1:rand(rng,1:20)])
        b=S._antichain([G(rand(rng,radii),rand(rng,Bool),rand(rng,1:9)) for _ in 1:rand(rng,1:20)])
        result=S._intersect_grades(a,b)
        @test result==S._antichain([S._join(x,y) for x in a for y in b])
        @test result==S._intersect_grades(b,a)
        for r in radii,k in (1,5,9)
            @test any(g->S._reached(g,r,k),result)==
                (any(g->S._reached(g,r,k),a) && any(g->S._reached(g,r,k),b))
        end
    end
    X=reshape([0,1,10],:,1)
    r=S.sparse_multicover(X;epsilon=1)
    @test_throws ArgumentError S.sparse_multicover(X;epsilon=1,
        budget=S.SparseMulticoverBudget(max_grades=r.statistics.grades-1))
    bounded=S.sparse_multicover(X;epsilon=1,budget=S.SparseMulticoverBudget(max_grades=r.statistics.grades))
    @test bounded.simplex_grades==r.simplex_grades
end

@testset "A69 reused geometry has independent optimality certificates" begin
    S=TamerOp.SparseMulticovers;AR=TamerOp.AlgebraicReal
    # For the convex minimax problem, feasibility and a center in the convex
    # hull of active sites certify global optimality. This checks the returned
    # solution without reproducing the constructor's quadratic equations.
    function optimal_center(r,sites,radius,z)
        squared=radius^2;kappa=1/(3(1+r.internal_epsilon)^2)
        cap(i)=r.slowing_squared[i]===nothing ? squared :
            min(squared,kappa*squared+(1-kappa)*r.slowing_squared[i])
        distance(i)=sum((z[a]-r.points[i,a])^2 for a in eachindex(z))
        all(i->distance(i)<=cap(i),sites) || return false
        active=[i for i in sites if distance(i)==cap(i)]
        for mask in 1:(1<<length(active))-1
            count_ones(mask)<=length(z)+1 || continue
            basis=[active[j] for j in eachindex(active) if !iszero(mask & (1<<(j-1)))]
            p=r.points[first(basis),:]
            length(basis)==1 && z==p && return true
            length(basis)==1 && continue
            V=hcat([r.points[j,:]-p for j in basis[2:end]]...)
            G=transpose(V)*V
            iszero(det(G)) && continue
            weights=(G\transpose(V))*(z-p)
            all(w->w>=0,weights) && sum(weights)<=1 && p+V*weights==z && return true
        end
        false
    end
    fixtures=(reshape([0//1,1//3,2,11//2,13],:,1),
        [0 0;1 1;14 0;0 14], [0 0;4 0;0 4;4 4;1 2],
        [0 0 0;1 1 1;8 0 0;0 9 0;0 0 10],
        [0 0 0;1 2 -1;3 6 -3;9 18 -9])
    for X in fixtures,epsilon in (1//2,1//1)
        r=S.sparse_multicover(X;epsilon,max_dim=1)
        @test S.check_sparse_multicover(r).valid
        for (sites,radius,z) in zip(r.subsets,r.intersection_radii,r.intersection_centers)
            @test optimal_center(r,sites,radius,z)
        end
        # Also certify subsets later excluded by disappearance times. Query
        # them in both orders to check reuse independently of enumeration order.
        W=S._IntersectionWorkspace(r.points,r.slowing_squared,r.internal_epsilon)
        work=Ref(0);budget=S.SparseMulticoverBudget()
        subsets=[[i for i in axes(X,1) if !iszero(mask & (1<<(i-1)))] for mask in 1:(1<<size(X,1))-1]
        answers=[S._intersection!(W,sites,budget,work) for sites in reverse(subsets)]
        for (sites,(radius,z)) in zip(subsets,reverse(answers))
            @test optimal_center(r,sites,radius,z)
            @test S._intersection!(W,sites,budget,work)==(radius,z)
        end
        @test r.statistics.distinct_bases<r.statistics.active_bases
        @test all(r.intersection_centers[i]!==r.intersection_centers[j]
            for i in eachindex(r.subsets) for j in 1:i-1)
    end
    r=S.sparse_multicover(reshape([0,1,10],:,1);epsilon=1,max_dim=1)
    shifted=S.sparse_multicover(r.points.+7;epsilon=1,max_dim=1)
    @test shifted.intersection_radii==r.intersection_radii
    @test shifted.intersection_centers==[z.+7 for z in r.intersection_centers]
    @test shifted.simplex_grades==r.simplex_grades
    # Reuse never bypasses the declared search-work limit.
    @test_throws ArgumentError S.sparse_multicover(r.points;epsilon=1,max_dim=1,
        budget=S.SparseMulticoverBudget(max_bases=r.statistics.active_bases-1))
end

@testset "A69 exact net, weighted intersections and open grades" begin
    S=TamerOp.SparseMulticovers;AR=TamerOp.AlgebraicReal
    r=S.sparse_multicover(reshape([0,1,10],:,1);epsilon=1,max_dim=nothing)
    @test r.permutation==[1,3,2]
    @test r.insertion_squared==[nothing,1,100]
    @test r.covering_sequences==[[1],[2,1],[3,1]]
    @test S.sparse_covering(r;radius=8).mapping==[1,2,3]
    @test S.sparse_covering(r;radius=8+1//100).mapping==[1,1,3]
    @test S.sparse_covering(r;radius=80).mapping==[1,1,3]
    @test S.sparse_covering(r;radius=81).mapping==[1,1,1]
    root=findfirst(==([1]),r.subsets)
    @test [(g.radius,g.open,g.depth) for g in r.grades[root]]==[(0,false,1),(8,true,2),(80,true,3)]
    pair=findfirst(==([2,3]),r.subsets)
    k=3//16
    expected=(9-sqrt(AR(81-(1-k)*(81-(1-k)*16))))/(1-k)
    @test r.intersection_radii[pair]==expected
    @test 9//2<expected<5
    triple=findfirst(==([1,2,3]),r.subsets)
    @test r.intersection_radii[triple]==5
    @test S.check_sparse_multicover(r;recompute=true).valid
    @test S.sparse_multicover_summary(r).interleaving_factor==2
    @test S.sparse_multicover_summary(r).sparse_to_exact_factor==5//4
    # At a weight-change threshold the singleton does not acquire its new
    # depth until strictly afterwards, although old flags remain present.
    at=S.sparse_multicover_cells(r;radius=8,depth=2)
    after=S.sparse_multicover_cells(r;radius=8+1//100,depth=2)
    @test !any(c->c.subsets==[[1]],at)
    @test any(c->c.subsets==[[1]],after)
    @test all(c->any(d->c.dimension==d.dimension && c.index==d.index,after),at)
    # Independent symmetric three-active-ball oracle. The slowed center is
    # (1,1), the others (14,0),(0,14); stationarity forces center (t,t).
    planar=S.sparse_multicover([0 0;1 1;14 0;0 14];epsilon=1,max_dim=2)
    j=findfirst(==([2,3,4]),planar.subsets)
    t=(-5+sqrt(AR(6343)))/13
    expected_radius=sqrt(196-28t+2t^2)
    @test planar.intersection_radii[j]==expected_radius
    @test planar.intersection_centers[j]==[t,t]
    @test expected_radius>sqrt(AR(98))
end

@testset "A69 independent multicover inclusion witnesses" begin
    S=TamerOp.SparseMulticovers
    for X in (reshape([0,1,10],:,1),[0 0;2 0;0 2],[0 0 0;2 0 0;0 2 0;0 0 2])
        r=S.sparse_multicover(X;epsilon=1,max_dim=2)
        for radius in (0,1//2,1,3,8,9,81)
            weights=S.sparse_covering(r;radius).weights
            @test sum(weights)==size(X,1)
            for point in (vec(X[1,:]),fill(1//2,size(X,2)),fill(4,size(X,2)))
                q=S.sparse_multicover_at(r;point,radius)
                actual=count(i->sum((point[a]-X[i,a])^2 for a in axes(X,2))<=radius^2,axes(X,1))
                @test q.exact_depth==actual
                for depth in 1:actual
                    w=S.multicover_witness(r;point,radius,depth)
                    @test w.valid && w.target_radius==2radius
                    @test length(w.original_sites)>=depth
                end
                for depth in 1:q.sparse_depth
                    w=S.multicover_witness(r;point,radius,depth,direction=:to_exact)
                    @test w.valid && w.target_radius==5radius/4
                end
            end
        end
        @test S.check_sparse_multicover(r).valid
    end
end

@testset "A69 finite encoding and subdivision homology" begin
    S=TamerOp.SparseMulticovers;EC=TamerOp.EncodingCore
    r=S.sparse_multicover(reshape([0,1,10],:,1);epsilon=1,max_dim=2)
    for field in (CM.QQField(),CM.F2(),CM.F3())
        enc=TamerOp.encode(r;field,degree=0)
        M=TamerOp.Workflow.pmodule(enc)
        @test EC.check_encoding_map(enc.pi).valid
        @test TamerOp.provenance(enc).construction.effective===:sparse_subdivision_cech
        @test TamerOp.provenance(enc).approximation.interleaving_factor==2
        # Before any slowing, independent unions of intervals have 3,2,1
        # connected components. At depth two, the first overlap is at 1/2.
        for (radius,depth,expected) in ((0,1,3),(1//2,1,2),(5,1,1),
                (0,2,0),(1//2,2,1),(5,3,1),(81,3,1))
            v=EC.locate(enc.pi,[radius,-depth])
            @test v>0 && M.dims[v]==expected
        end
        @test TamerOp.rank_map(enc,(1//2,-2),(81,-2))==1
        @test MD.check_module_functoriality(M).valid
        @test EC.locate(enc.pi,[1,-4])==0
        @test_throws ArgumentError EC.locate(enc.pi,[1,2])
        @test_throws ArgumentError EC.locate(enc.pi,[-1,-1])
    end
    limited=S.sparse_multicover([0 0;1 0;0 1];epsilon=1,max_dim=0)
    @test_throws ArgumentError TamerOp.encode(limited;degree=0)
    triangle=S.sparse_multicover([0 0;2 0;1 2];epsilon=1,max_dim=2)
    enc=TamerOp.encode(triangle;field=CM.F2(),degree=1)
    M=TamerOp.Workflow.pmodule(enc)
    birth=sqrt(TamerOp.AlgebraicReal(5))/2
    for (radius,expected) in ((1,0),(birth,1),(6//5,1),(5//4,0))
        @test M.dims[EC.locate(enc.pi,[radius,-1])]==expected
    end
    @test TamerOp.rank_map(enc,[birth,-1],[6//5,-1])==1
    @test TamerOp.rank_map(enc,[birth,-1],[5//4,-1])==0
end

@testset "A69 degeneracies, budgets and owned queries" begin
    S=TamerOp.SparseMulticovers
    for X in ([0 0;1 0;2 0],[0 0;1 0;1 1;0 1],reshape([0],:,1))
        r=S.sparse_multicover(TamerOp.PointCloud(X);epsilon=1)
        @test S.check_sparse_multicover(r).valid
        @test r.points==X
        @test TamerOp.sparse_multicover===S.sparse_multicover
        @test occursin("SparseMulticover",sprint(show,r))
    end
    @test_throws ArgumentError S.sparse_multicover([0 0;0 0];epsilon=1)
    @test_throws ArgumentError S.sparse_multicover([0 0;1 0];epsilon=0)
    @test_throws ArgumentError S.sparse_multicover([0 0;1 0];epsilon=2)
    @test_throws ArgumentError S.sparse_multicover([0 NaN];epsilon=1)
    @test_throws ArgumentError S.sparse_multicover([0 0;1 0;0 1];epsilon=1,budget=S.SparseMulticoverBudget(max_subsets=2))
    @test_throws ArgumentError S.sparse_multicover([0 0;1 0;0 1];epsilon=1,budget=S.SparseMulticoverBudget(max_bases=1))
    @test_throws ArgumentError S.sparse_multicover([0 0;1 0;0 1];epsilon=1,budget=S.SparseMulticoverBudget(max_simplices=2))
    r=S.sparse_multicover(reshape([0,1],:,1);epsilon=1)
    @test_throws ArgumentError S.multicover_witness(r;point=[5],radius=0,depth=1)
    @test_throws ArgumentError S.multicover_witness(r;point=[0],radius=0,depth=0)
    @test_throws ArgumentError S.sparse_multicover_at(r;point=[0,0],radius=0)
    r.covering_sequences[2]=[2]
    @test !S.check_sparse_multicover(r).valid
    malformed=S.sparse_multicover(reshape([0,1],:,1);epsilon=1)
    malformed.simplices[2][1]=[1,2] # incomparable singleton subsets
    @test !S.check_sparse_multicover(malformed).valid
    @test_throws ArgumentError TamerOp.encode(malformed)
end
