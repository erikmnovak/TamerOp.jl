# Independent limit/colimit ranks, scrambled interval sums and chain oracles.
function _zz_rref(A)
    R=copy(A); pivots=Int[]; row=1
    for col in axes(R,2)
        pivot=findfirst(i->!iszero(R[i,col]),row:size(R,1))
        pivot===nothing && continue
        p=row+pivot-1
        R[row,:],R[p,:]=copy(R[p,:]),copy(R[row,:])
        R[row,:]/=R[row,col]
        for i in axes(R,1)
            i==row && continue
            R[i,:]-=R[i,col]*R[row,:]
        end
        push!(pivots,col); row+=1
        row>size(R,1) && break
    end
    R,pivots
end
_zz_rank(A)=length(last(_zz_rref(A)))
function _zz_null(A::Matrix{K}) where {K}
    R,pivots=_zz_rref(A); free=setdiff(collect(axes(A,2)),pivots)
    N=zeros(K,size(A,2),length(free))
    for (j,c) in enumerate(free)
        N[c,j]=one(K)
        for (i,p) in enumerate(pivots); N[p,j]=-R[i,c]; end
    end
    N
end

# Generalized rank from an independently assembled limit and colimit; no
# persistence, production rank/nullspace, or interval reduction is used.
function _zz_rank_on(z,a,b)
    K=CM.coeff_type(z.field); dims=z.dims[a:b]; offsets=cumsum([0;dims]); total=last(offsets)
    constraint_rows=sum((z.dims[z.directions[e]===:forward ? e+1 : e] for e in a:b-1);init=0)
    relation_cols=sum((z.dims[z.directions[e]===:forward ? e : e+1] for e in a:b-1);init=0)
    equations=zeros(K,constraint_rows,total); relations=zeros(K,total,relation_cols)
    row=0; col=0
    for e in a:b-1
        s,t=z.directions[e]===:forward ? (e,e+1) : (e+1,e)
        si=offsets[s-a+1]+1:offsets[s-a+2]; ti=offsets[t-a+1]+1:offsets[t-a+2]
        equations[row+1:row+z.dims[t],si]=z.maps[e]
        equations[row+1:row+z.dims[t],ti]-=Matrix{K}(I,z.dims[t],z.dims[t])
        relations[ti,col+1:col+z.dims[s]]=z.maps[e]
        relations[si,col+1:col+z.dims[s]]-=Matrix{K}(I,z.dims[s],z.dims[s])
        row+=z.dims[t]; col+=z.dims[s]
    end
    limit=_zz_null(equations); selected=zeros(K,total,size(limit,2))
    selected[1:dims[1],:]=limit[1:dims[1],:]
    _zz_rank(hcat(relations,selected))-_zz_rank(relations)
end

function _zz_oracle(z)
    n=length(z.dims); rho=zeros(Int,n+2,n+2)
    for a in 1:n,b in a:n; rho[a+1,b+1]=_zz_rank_on(z,a,b); end
    bars=Tuple{Int,Int}[]
    for a in 1:n,b in a:n
        m=rho[a+1,b+1]-rho[a,b+1]-rho[a+1,b+2]+rho[a,b+2]
        @test m>=0
        append!(bars,fill((a,b),m))
    end
    sort(bars)
end

function _zz_scrambled_sum(supports,dirs,F,rng)
    n=length(dirs)+1; K=CM.coeff_type(F)
    active=[findall(ab->ab[1]<=i<=ab[2],supports) for i in 1:n]
    bases=Matrix{K}[]; inverses=Matrix{K}[]
    for ids in active
        d=length(ids); U=Matrix{K}(I,d,d); V=copy(U)
        for _ in 1:3d
            d<2 && break
            a,b=randperm(rng,d)[1:2]; c=K(rand(rng,-2:2))
            U[a,:]+=c*U[b,:]; V[:,b]-=c*V[:,a]
        end
        @test U*V==Matrix{K}(I,d,d)
        push!(bases,U); push!(inverses,V)
    end
    maps=Matrix{K}[]
    for e in eachindex(dirs)
        s,t=dirs[e]===:forward ? (e,e+1) : (e+1,e)
        A=K[i==j for i in active[t],j in active[s]]
        push!(maps,bases[t]*A*inverses[s])
    end
    ZZ.ZigzagModule(length.(active),maps;directions=dirs,field=F)
end

function _zz_check_reps(r)
    z=r.diagram; K=CM.coeff_type(z.field); bars=ZZ.zigzag_intervals(r)
    for t in eachindex(z.dims)
        indices=findall(b->b.first<=t<=b.last,bars)
        B=zeros(K,z.dims[t],length(indices))
        for (j,index) in enumerate(indices); B[:,j]=ZZ.zigzag_representative(r;interval=index,vertex=t); end
        @test _zz_rank(B)==z.dims[t]
    end
    for (i,bar) in enumerate(bars), e in eachindex(z.maps)
        s,t=z.directions[e]===:forward ? (e,e+1) : (e+1,e)
        bar.first<=s<=bar.last || continue
        x=ZZ.zigzag_representative(r;interval=i,vertex=s)
        y=bar.first<=t<=bar.last ? ZZ.zigzag_representative(r;interval=i,vertex=t) : zeros(K,z.dims[t])
        @test z.maps[e]*x==y
    end
    @test ZZ.check_zigzag(r).valid
end

@testset "Zigzag right-filtration interval oracles" begin
    rng=MersenneTwister(120)
    for F in (CM.QQField(),CM.F2(),CM.F3(),CM.Fp(101))
        K=CM.coeff_type(F)
        for bits in 0:15
            dirs=[iszero(bits & (1<<(i-1))) ? :forward : :backward for i in 1:4]
            # All fifteen supports occur, with repeated members and arbitrary
            # bases. Covers endpoint, interior and direction-change behavior.
            supports=[(a,b) for a in 1:5 for b in a:5]
            append!(supports,[(2,4),(2,4)])
            z=_zz_scrambled_sum(supports,dirs,F,rng)
            r=ZZ.zigzag_persistence(z;representatives=true)
            @test r.supports==sort(supports)
            @test ZZ.zigzag_persistence(z).supports==r.supports
            _zz_check_reps(r)
            # Arbitrary maps are also checked against limit/colimit ranks,
            # rather than assuming every input was generated from known bars.
            dims=rand(rng,0:3,5)
            maps=[K.(rand(rng,-2:2,dirs[e]===:forward ? (dims[e+1],dims[e]) : (dims[e],dims[e+1]))) for e in 1:4]
            random=ZZ.ZigzagModule(dims,maps;directions=dirs,field=F)
            result=ZZ.zigzag_persistence(random;representatives=true)
            @test result.supports==_zz_oracle(random)
            _zz_check_reps(result)
        end
        for dims in ([0],[3],[0,0,0],[2,0,3])
            dirs=fill(:backward,length(dims)-1)
            z=ZZ.ZigzagModule(dims,[zeros(K,dims[e],dims[e+1]) for e in eachindex(dirs)];directions=dirs,field=F)
            r=ZZ.zigzag_persistence(z;representatives=true)
            @test r.supports==_zz_oracle(z)
            _zz_check_reps(r)
        end
    end
end

@testset "Zigzag event semantics and ordinary persistence" begin
    simplices=[[1],[2],[3],[1,2],[1,3],[2,3],[1,2,3]]
    for F in (CM.QQField(),CM.F2(),CM.F3(),CM.Fp(101))
        K=CM.coeff_type(F)
        events=[ZZ.ZigzagEvent(:add,s;time=i//3) for (i,s) in enumerate(simplices)]
        push!(events,ZZ.ZigzagEvent(:delete,[1,2,3];time=8//3),ZZ.ZigzagEvent(:delete,[2,3];time=3//1))
        r=ZZ.zigzag_persistence(events;degree=1,field=F,initial_time=0//1,representatives=true)
        @test r.supports==[(7,7),(9,9)]
        bars=ZZ.zigzag_intervals(r)
        @test (bars[1].left,bars[1].right,bars[1].left_closed,bars[1].right_closed)==(2//1,7//3,true,false)
        @test (bars[2].left,bars[2].right,bars[2].left_closed,bars[2].right_closed)==(7//3,8//3,false,true)
        _zz_check_reps(r)
        for (j,(a,b)) in enumerate(r.supports), t in a:b
            cycle=ZZ.zigzag_cycle(r;interval=j,vertex=t)
            @test all(iszero,r.source.complexes[t].boundaries[1]*cycle.coefficients)
            @test length(cycle.simplices)==length(cycle.coefficients)
        end
        w=ZZ.zigzag_chain_witness(r;interval=1,edge=7)
        @test any(!iszero,w.filling)
        @test r.source.complexes[w.target].boundaries[2]*w.filling==w.difference
        # Boundary on a cell can vanish only in a particular characteristic.
        C=ZZ.ZigzagComplex([1,1,1],[zeros(Int,1,1),[2;;]];field=F)
        q=ZZ.zigzag_persistence([C],[];degree=1,directions=Symbol[],representatives=true)
        @test q.supports==(F==CM.F2() ? [(1,1)] : Tuple{Int,Int}[])
        # All-forward simplex streams recover ordinary finite intervals and
        # terminal bars, with the latter censored rather than silently infinite.
        if F isa CM.PrimeField
            forward=events[1:7]
            d1=sparse([-1 -1 0;1 0 -1;0 1 1]); d2=sparse([1;-1;1;;])
            G=DT.GradedComplex([[1,2,3],[4,5,6],[7]],[d1,d2],[(i//3,) for i in 1:7])
            ordinary=OP.persistence_diagram(G;field=F)
            for degree in 0:1
                z=ZZ.zigzag_persistence(forward;degree,field=F,initial_time=0//1)
                ibars=ZZ.zigzag_intervals(z)
                @test sort([(b.left,b.right) for b in ibars if b.right_kind!==:terminal])==sort(OP.finite_intervals(ordinary;dim=degree))
                @test sort([b.left for b in ibars if b.right_kind===:terminal])==sort(OP.essential_births(ordinary;dim=degree))
            end
        end
    end
    # Identical time labels do not remove a nonempty event-index interval.
    events=[ZZ.ZigzagEvent(:add,[1];time=1),ZZ.ZigzagEvent(:delete,[1];time=1),ZZ.ZigzagEvent(:add,[1];time=1)]
    r=ZZ.zigzag_persistence(events;initial_time=1,representatives=true)
    @test r.supports==[(2,2),(4,4)]
    @test ZZ.zigzag_times(r)==[1,1,1,1]
    @test all(b->b.left==b.right==1,ZZ.zigzag_intervals(r))
    _zz_check_reps(r)
    @test ZZ.zigzag_persistence(ZZ.ZigzagEvent[]).supports==Tuple{Int,Int}[]
    @test ZZ.zigzag_persistence(ZZ.ZigzagEvent[];initial=[[1]]).supports==[(1,1)]
    # The four endpoint decorations come from the actual incident directions.
    for (dirs,ends,closed) in (([:forward,:forward],(20,30),(true,false)),
                               ([:forward,:backward],(20,20),(true,true)),
                               ([:backward,:forward],(10,30),(false,false)),
                               ([:backward,:backward],(10,20),(false,true)))
        maps=[zeros(Int,dirs[e]===:forward ? ([0,1,0][e+1],[0,1,0][e]) : ([0,1,0][e],[0,1,0][e+1])) for e in 1:2]
        r=ZZ.zigzag_persistence(ZZ.ZigzagModule([0,1,0],maps;directions=dirs,times=[10,20,30]))
        b=only(ZZ.zigzag_intervals(r))
        @test (b.left,b.right)==ends
        @test (b.left_closed,b.right_closed)==closed
        @test (b.first,b.last)==(2,2)
    end
end

@testset "Zigzag supplied chain maps and boundary witnesses" begin
    rng=MersenneTwister(1201)
    for F in (CM.QQField(),CM.F2(),CM.F3(),CM.Fp(101))
        K=CM.coeff_type(F)
        # One connected edge: the chosen vertex representative changes by an
        # actual boundary under a non-inclusion chain map swapping vertices.
        C=ZZ.ZigzagComplex([2,1],[reshape([-1,1],2,1)];field=F,labels=[[11,12],[21]])
        flip=[K[0 1;1 0],K[-1;;]]
        r=ZZ.zigzag_persistence([C,C],[flip];directions=[:backward],representatives=true)
        @test r.supports==[(1,2)]
        _zz_check_reps(r)
        w=ZZ.zigzag_chain_witness(r;interval=1,edge=1)
        @test any(!iszero,w.difference)
        @test C.boundaries[1]*w.filling==w.difference
        @test w.filling_labels==[21]
        @test ZZ.zigzag_cycle(r;interval=1,vertex=1).labels==[11,12]
        # A genuine zero map kills and births classes even with equal Betti
        # numbers, in either direction. Include a rational scaling over QQ.
        P=ZZ.ZigzagComplex([1],[];field=F)
        for dir in (:forward,:backward)
            q=ZZ.zigzag_persistence([P,P],[[zeros(K,1,1)]];directions=[dir],representatives=true)
            @test q.supports==[(1,1),(2,2)]
            _zz_check_reps(q)
            high=ZZ.zigzag_persistence([P,P],[[ones(K,1,1)]];directions=[dir],degree=4)
            @test isempty(high.supports)
        end
        @test_throws ArgumentError ZZ.zigzag_persistence([C,C],[[Matrix{K}(I,2,2),zeros(K,1,1)]];directions=[:forward])
        @test_throws ArgumentError ZZ.ZigzagComplex([1,1,1],[ones(K,1,1),ones(K,1,1)];field=F)
        # Attach contractible pairs on both sides of H1. The chain maps mix
        # homology cycles with boundaries, so equality only holds in quotient.
        dirs=[:backward,:forward,:backward,:forward]
        expected=[(1,5),(2,3),(4,5),(3,3)]
        module_data=_zz_scrambled_sum(expected,dirs,F,rng)
        complexes=ZZ.ZigzagComplex{K,typeof(F)}[]
        for h in module_data.dims
            d1=zeros(K,1,h+2); d1[1,end]=1
            d2=zeros(K,h+2,1); d2[h+1,1]=1
            push!(complexes,ZZ.ZigzagComplex([1,h+2,1],[d1,d2];field=F))
        end
        maps=Vector{Matrix{K}}[]
        for e in eachindex(dirs)
            s,t=dirs[e]===:forward ? (e,e+1) : (e+1,e)
            hs,ht=module_data.dims[s],module_data.dims[t]
            f1=zeros(K,ht+2,hs+2)
            f1[1:ht,1:hs]=module_data.maps[e]
            f1[ht+1,1:hs]=K.(rand(rng,-2:2,hs))
            f1[ht+1,hs+1]=1; f1[ht+2,hs+2]=1
            push!(maps,[ones(K,1,1),f1,ones(K,1,1)])
        end
        q=ZZ.zigzag_persistence(complexes,maps;directions=dirs,degree=1,representatives=true)
        @test q.supports==sort(expected)
        _zz_check_reps(q)
        for (j,(a,b)) in enumerate(q.supports),e in eachindex(dirs)
            s,t=dirs[e]===:forward ? (e,e+1) : (e+1,e)
            a<=s<=b || continue
            w=ZZ.zigzag_chain_witness(q;interval=j,edge=e)
            @test complexes[t].boundaries[2]*w.filling==w.difference
        end
        # Caller mutation cannot alter retained chains or chain maps.
        maps[1][2].=0
        @test any(!iszero,q.source.maps[1][2])
    end
end

@testset "Zigzag contracts ownership budgets and public surface" begin
    @test TamerOp.zigzag_persistence===ZZ.zigzag_persistence
    @test TOA.ZigzagBudget===ZZ.ZigzagBudget
    @test TOA.check_zigzag===ZZ.check_zigzag
    @test_throws ArgumentError ZZ.ZigzagModule([],[];directions=[])
    @test_throws ArgumentError ZZ.ZigzagModule([1,1],[[1;;]];directions=[:left])
    @test_throws ArgumentError ZZ.ZigzagModule([1,2],[ones(Int,1,2)];directions=[:forward])
    @test_throws ArgumentError ZZ.ZigzagModule([1,1],[[1.0;;]];directions=[:forward])
    @test_throws ArgumentError ZZ.ZigzagModule([1,1],[[1;;]];directions=[:forward],times=[2,1])
    @test_throws ArgumentError ZZ.ZigzagModule([1],[];directions=[],times=[Inf])
    @test_throws ArgumentError ZZ.ZigzagModule([true],[];directions=[])
    @test_throws ArgumentError ZZ.ZigzagModule([1],[];directions=[],field=CM.RealField(Float64))
    @test_throws ArgumentError ZZ.ZigzagModule([1,1],[[1//2;;]];directions=[:forward])
    @test_throws ArgumentError ZZ.ZigzagModule([1,1],[CM.coeff_type(CM.F3()).([1;;])];directions=[:forward])
    @test_throws ArgumentError ZZ.ZigzagBudget(max_vertices=-1)
    @test_throws ArgumentError ZZ.ZigzagBudget(max_vertices=true)
    @test_throws ArgumentError ZZ.ZigzagModule([2],[];directions=[],budget=ZZ.ZigzagBudget(max_dimension=1))
    # These must fail before allocating default labels or time arrays.
    @test_throws ArgumentError ZZ.ZigzagModule(1:10^9,[];directions=[])
    @test_throws ArgumentError ZZ.ZigzagComplex([10^9],[])
    @test_throws ArgumentError ZZ.ZigzagComplex(1:10^9,[])
    z=ZZ.ZigzagModule([1,1],[[1;;]];directions=[:backward])
    @test_throws ArgumentError ZZ.zigzag_persistence(z;representatives=true,budget=ZZ.ZigzagBudget(max_representative_entries=1))
    @test_throws ArgumentError ZZ.zigzag_persistence(z;budget=ZZ.ZigzagBudget(max_matrix_entries=1))
    @test_throws ArgumentError ZZ.zigzag_persistence(z;budget=ZZ.ZigzagBudget(max_vertices=1))
    r=ZZ.zigzag_persistence(z;representatives=true)
    z.maps[1][1]=0
    @test r.diagram.maps[1][1]==1
    a=ZZ.zigzag_map(r;edge=1); a[1]=0
    @test ZZ.zigzag_map(r;edge=1)[1]==1
    c=ZZ.zigzag_representative(r;interval=1,vertex=1); c[1]=0
    @test ZZ.check_zigzag(r).valid
    @test !ZZ.check_zigzag(ZZ.ZigzagResult(r.diagram,r.supports,[[[zero(eltype(c))],[zero(eltype(c))]]],nothing)).valid
    @test ZZ.describe(r).intervals==1
    @test occursin("ZigzagResult",sprint(show,r))
    @test ZZ.provenance(r).terminal===:censored
    plain=ZZ.zigzag_persistence(z)
    @test ZZ.check_zigzag(plain).valid
    @test_throws ArgumentError ZZ.zigzag_representative(plain;interval=1,vertex=1)
    @test_throws ArgumentError ZZ.zigzag_cycle(r;interval=1,vertex=1)
    @test_throws ArgumentError ZZ.zigzag_map(r;edge=0)
    @test_throws ArgumentError ZZ.zigzag_representative(r;interval=1,vertex=3)
    @test_throws ArgumentError ZZ.ZigzagEvent(:foo,[1];time=1)
    @test_throws ArgumentError ZZ.ZigzagEvent(:add,[1,1];time=1)
    @test_throws ArgumentError ZZ.ZigzagEvent(:add,[0];time=1)
    @test_throws ArgumentError ZZ.zigzag_persistence([ZZ.ZigzagEvent(:add,[1,2];time=1)])
    @test_throws ArgumentError ZZ.zigzag_persistence([ZZ.ZigzagEvent(:delete,[1];time=1)])
    @test_throws ArgumentError ZZ.zigzag_persistence([ZZ.ZigzagEvent(:add,[1];time=1)];initial=[[1]])
    @test_throws ArgumentError ZZ.zigzag_persistence([ZZ.ZigzagEvent(:delete,[1];time=1)];initial=[[1],[2],[1,2]])
    @test_throws ArgumentError ZZ.zigzag_persistence(ZZ.ZigzagEvent[];initial=[[1,2]])
    @test_throws ArgumentError ZZ.zigzag_persistence(ZZ.ZigzagEvent[];initial=[[1],[1]])
    @test_throws ArgumentError ZZ.zigzag_persistence([ZZ.ZigzagEvent(:add,[1];time=1)];initial_time=2)
    @test_throws ArgumentError ZZ.zigzag_persistence([ZZ.ZigzagEvent(:add,[1];time=1)];budget=ZZ.ZigzagBudget(max_cells=0))
    @test_throws ArgumentError ZZ.zigzag_persistence([ZZ.ZigzagEvent(:add,[1];time=1)];budget=ZZ.ZigzagBudget(max_matrix_entries=0))
    @test_throws ArgumentError ZZ.zigzag_persistence([ZZ.ZigzagComplex([1],[])],[];directions=[],degree=typemax(Int))
    # Every query owns its scratch; concurrent requests preserve the same bars.
    jobs=[Threads.@spawn ZZ.zigzag_persistence(r.diagram;representatives=true) for _ in 1:4]
    @test all(q->ZZ.check_zigzag(fetch(q)).valid,jobs)
end
