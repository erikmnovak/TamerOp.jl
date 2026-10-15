# Exhaustive subset covering is independent of the ordered greedy sweep.
function _pc_brute(g,d,v)
    alive=[q.+v for q in g if !any(p->all(p.<=q.+v),d)]
    for k in 0:length(g),mask in 0:(1<<length(g))-1
        count_ones(mask)==k || continue
        chosen=[g[i] for i in eachindex(g) if !iszero(mask & (1<<(i-1)))]
        all(q->any(p->all(p.<=q),chosen),alive) && return k
    end
    error("oracle failed")
end
function _pc_rank(A)
    B=copy(A);r=0
    for j in axes(B,2)
        k=findfirst(i->!iszero(B[i,j]),r+1:size(B,1));k===nothing && continue
        k+=r;r+=1;B[r,:],B[k,:]=copy(B[k,:]),copy(B[r,:]);B[r,:]./=B[r,j]
        for i in r+1:size(B,1);B[i,:].-=B[i,j].*B[r,:];end
    end
    r
end

@testset "A106 contour laws and exact interval witnesses" begin
    rng=MersenneTwister(106)
    for direction in ((1,1),(2//3,5//2),(0,3),(2,0),(0,0))
        c=PC.LinearContour(direction);@test PC.check_contour(c;throw=true).valid
        for _ in 1:20
            x=Tuple(rand(rng,-5:5,2));y=x.+Tuple(rand(rng,0:5,2));s=rand(rng,0:8)//3;t=rand(rng,0:8)//5
            C(p,a)=PC.contour_point(c;point=p,scale=a)
            @test all(x.<=C(x,s))
            @test all(C(x,s).<=C(y,s+t))
            @test C(C(x,s),t)==C(x,s+t)
            @test C(x,0)==x
        end
    end
    for F in (CM.QQField(),CM.F2(),CM.Fp(3),CM.Fp(5))
        I=PC.BivariateInterval([(0,8),(4,6),(6,4),(8,2),(11,0)];field=F)
        r=PC.shift_dimension(I;shift=(4,4),witnesses=true)
        @test PC.shift_dimension_value(r)==2
        @test PC.check_shift_dimension(r;throw=true).optimality_checked
        @test length(PC.shift_generators(r))==2
        @test PC.shift_dimension(I;shift=(0,0)).value==5
        @test PC.shift_dimension(I;shift=(11,8)).value==1
        @test PC.describe(r).field==F
        r.witnesses.generators[1]=(999//1,999//1)
        @test !PC.check_shift_dimension(r).valid
        @test_throws ArgumentError PC.check_shift_dimension(r;throw=true)
    end
    # Removing a killed birth must retain relations where its upset meets the
    # surviving birth's upset: here the remaining interval is [2,inf)x[0,2).
    I=PC.BivariateInterval([(0,2),(2,0)],[(0,2)])
    @test I.generators==[(2,0)] && I.relations==[(2,2)]
    @test PC.shift_dimension(I;shift=(0,2)).value==0
    @test_throws ArgumentError PC.BivariateInterval([(0,2),(2,0)],[(2,2)])
    @test_throws ArgumentError PC.BivariateInterval([(1,1)],[(0,0)])
    @test_throws ArgumentError PC.LinearContour((-1,1))
    @test_throws ArgumentError PC.LinearContour((NaN,1))
    @test_throws ArgumentError PC.contour_point(PC.LinearContour((1,1));point=(0,0),scale=-1)
    @test_throws ArgumentError PC.ContourBudget(max_work=-1)
end

@testset "A106 interval sweep versus exhaustive covering" begin
    rng=MersenneTwister(10651)
    for trial in 1:100
        n=rand(rng,1:7);x=cumsum(rand(rng,1:4,n));y=reverse(cumsum(rand(rng,1:4,n)))
        g=collect(zip(x,y));d=[(last(x)+rand(rng,1:5),first(y)+rand(rng,1:5))]
        I=PC.BivariateInterval(g,d)
        v=(rand(rng,0:16)//2,rand(rng,0:16)//3)
        r=PC.shift_dimension(I;shift=v,witnesses=true)
        @test r.value==_pc_brute(g,d,v)
        @test PC.check_shift_dimension(r;throw=true).valid
        @test PC.shift_dimension(I;shift=v.+(1//2,2//3)).value<=r.value
        # Births and deaths are rational and may be translated together.
        offset=(-13//2,7//3)
        J=PC.BivariateInterval([p.+offset for p in g],[p.+offset for p in d])
        @test PC.shift_dimension(J;shift=v).value==r.value
    end
    # Death corners cut both arms while leaving a connected overlap.
    g=[(0,4),(2,2),(4,0)];d=[(0,7),(3,6),(6,3),(7,0)]
    I=PC.BivariateInterval(g,d)
    for a in 0:12,b in 0:12
        v=(a//2,b//2);r=PC.shift_dimension(I;shift=v,witnesses=true)
        @test r.value==_pc_brute(g,d,v)
        @test PC.check_shift_dimension(r;throw=true).valid
    end
end

@testset "A106 complete curves, exact samples and boundary conventions" begin
    I=PC.BivariateInterval([(0,0)],[(4,0),(0,6)])
    c=PC.LinearContour((2,1));r=PC.stable_rank(I;contour=c,witnesses=true)
    @test PC.stable_rank_values(r).scales==[0,2]
    @test r.values==[1,0] && r.scope===:all_scales
    @test PC.stable_rank_at(r;scale=2-1//100)==1
    @test PC.stable_rank_at(r;scale=2)==0
    @test PC.stable_rank_at(r;scale=100)==0
    @test PC.check_stable_rank(r;throw=true).optimality_checked
    samples=PC.stable_rank(I;contour=c,scales=[0,1,3])
    @test samples.values==[1,1,0] && samples.scope===:samples
    @test_throws ArgumentError PC.stable_rank_at(samples;scale=2)
    zero=PC.stable_rank(PC.BivariateInterval([],[]))
    @test zero.scales==[0] && zero.values==[0]
    free=PC.stable_rank(PC.BivariateInterval([(0,2),(2,0)]))
    @test free.scales==[0,2] && free.values==[2,1]
    identity=PC.stable_rank(I;contour=PC.LinearContour((0,0)))
    @test identity.scales==[0] && identity.values==[1]
    horizontal=PC.stable_rank(I;contour=PC.LinearContour((1,0)))
    @test horizontal.scales==[0,4] && horizontal.values==[1,0]
    @test_throws ArgumentError PC.stable_rank(I;scales=[1,0])
    @test_throws ArgumentError PC.stable_rank(I;scales=[0,0])
    @test_throws ArgumentError PC.stable_rank(I;budget=PC.ContourBudget(max_events=1))
    @test_throws ArgumentError PC.stable_rank(I;budget=PC.ContourBudget(max_work=1))
    @test_throws ArgumentError PC.shift_dimension(I;shift=(1,1),budget=PC.ContourBudget(max_work=1))
    @test_throws ArgumentError PC.shift_generators(PC.shift_dimension(I;shift=(1,1)))
    r.values[1]=99;@test !PC.check_stable_rank(r).valid
    rng=MersenneTwister(10602)
    for _ in 1:20
        n=rand(rng,2:6);xs=cumsum(rand(rng,1:3,n));ys=reverse(cumsum(rand(rng,1:3,n)))
        g=collect(zip(xs,ys));d=[(last(xs)+1,first(ys)+2)]
        interval=PC.BivariateInterval(g,d);contour=PC.LinearContour((rand(rng,1:3)//2,rand(rng,1:3)//2))
        curve=PC.stable_rank(interval;contour,witnesses=true)
        @test PC.check_stable_rank(curve;throw=true).valid
        @test all(diff(curve.values).<0)
        probes=sort!(unique([curve.scales;[(a+b)/2 for (a,b) in zip(curve.scales[1:end-1],curve.scales[2:end])];last(curve.scales)+1]))
        for t in probes
            @test PC.stable_rank_at(curve;scale=t)==_pc_brute(g,d,t.*contour.direction)
        end
    end
end

@testset "A106 finite encoding, one-parameter oracle and nonadditivity" begin
    for F in (CM.QQField(),CM.F2(),CM.Fp(3))
        K=CM.coeff_type(F);Q=FF.ProductOfChainsPoset((3,3))
        coords=collect(Iterators.product(0:2,0:2))[:]
        supports=[[(0<=x<2 && 1<=y<3),(1<=x<3 && 0<=y<2)] for (x,y) in coords]
        active=[findall(s) for s in supports]
        M=MD.PModule{K}(Q,length.(active),Dict((u,v)=>K[i==j for i in active[v],j in active[u]] for (u,v) in FF.cover_edges(Q));field=F)
        A=PC.BivariateInterval([(0,1)],[(2,1),(0,3)];field=F)
        B=PC.BivariateInterval([(1,0)],[(3,0),(1,2)];field=F)
        @test PC.shift_dimension(A;shift=(1,1)).value+PC.shift_dimension(B;shift=(1,1)).value==2
        # Exhaust every single homogeneous projective vector over F2/F3,
        # including mixed (1,1) at the overlap. In QQ this supplied witness
        # suffices: a nonzero shifted map rules out zero generators.
        shift_targets=[findfirst(==(p.+(1,1)),coords) for p in coords]
        required=[b===nothing ? nothing : MD.map_leq(M,a,b) for (a,b) in enumerate(shift_targets)]
        @test any(A->A!==nothing && _pc_rank(Matrix(A))>0,required)
        candidates=Tuple{Int,Vector{K}}[]
        for (a,d) in enumerate(M.dims)
            if F isa CM.PrimeField
                for v in Iterators.product(ntuple(_->0:F.p-1,d)...)
                    any(!iszero,v) && push!(candidates,(a,K[v...]))
                end
            elseif d==2
                push!(candidates,(a,ones(K,2)))
            end
        end
        valid=filter(candidates) do pair
            a,x=pair
            all(enumerate(shift_targets)) do (u,v)
                v===nothing && return true
                req=Matrix(required[u]);image=FF.leq(Q,a,v) ? reshape(MD.map_leq(M,a,v)*x,:,1) : zeros(K,M.dims[v],0)
                _pc_rank(hcat(image,req))==_pc_rank(image)
            end
        end
        @test !isempty(valid) # exact general-module answer 1, versus sum 2.
        @test_throws ArgumentError PC.BivariateInterval(M;axes=([0,1,2,3],[0,1,2,3]))
        @test_throws ArgumentError TamerOp.stable_rank([A,B])
        dims=Int[s[1] for s in supports]
        N=MD.PModule{K}(Q,dims,Dict((u,v)=>ones(K,dims[v],dims[u]) for (u,v) in FF.cover_edges(Q));field=F)
        enc=TamerOp.Results.EncodingResult(Q,N,nothing)
        curve=TamerOp.stable_rank(enc;axes=([0,1,2,3],[0,1,2,3]),contour=PC.LinearContour((1,0)))
        # One-parameter interval of length 2: one generator until death at 2.
        @test curve.scales==[0,2] && curve.values==[1,0]
        @test TamerOp.shift_dimension_value(TamerOp.shift_dimension(enc;axes=([0,1,2,3],[0,1,2,3]),shift=(2,0)))==0
    end
end
