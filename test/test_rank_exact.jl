function _rx_rank(A)
    R=Matrix(A);r=1
    for c in axes(R,2)
        p=findfirst(i->!iszero(R[i,c]),r:size(R,1));p===nothing && continue
        p+=r-1;R[r,:],R[p,:]=copy(R[p,:]),copy(R[r,:]);R[r,:]/=R[r,c]
        for i in r+1:size(R,1);R[i,:]-=R[i,c]*R[r,:];end
        r+=1;r>size(R,1) && break
    end
    r-1
end
function _rx_indicator(Q,support,F)
    K=CM.coeff_type(F);dims=[Int(i in support) for i in 1:FF.nvertices(Q)]
    MD.PModule{K}(Q,dims,Dict{Tuple{Int,Int},Matrix{K}}(e=>ones(K,dims[e[2]],dims[e[1]]) for e in FF.cover_edges(Q));field=F)
end
function _rx_scramble(M,rng)
    K=CM.coeff_type(M.field);bs=Matrix{K}[];inv=Matrix{K}[]
    for d in M.dims
        B=Matrix{K}(I,d,d);Bi=copy(B)
        for _ in 1:3d
            d<2 && break
            a,b=randperm(rng,d)[1:2];c=K(rand(rng,-2:2))
            B[a,:]+=c*B[b,:];Bi[:,b]-=c*Bi[:,a]
        end
        push!(bs,B);push!(inv,Bi)
    end
    maps=Dict{Tuple{Int,Int},Matrix{K}}((u,v)=>bs[v]*M.edge_maps[u,v]*inv[u] for (u,v) in FF.cover_edges(M.Q))
    MD.PModule{K}(M.Q,copy(M.dims),maps;field=M.field)
end
function _rx_counts(r)
    B=RX.rank_exact_table(r);hs=RX.hook_labels(r)
    Dict((degree=i-1,birth=h.birth,death=h.death)=>B[i,j] for i in axes(B,1),(j,h) in enumerate(hs) if B[i,j]!=0)
end
function _rx_assert(r)
    @test RX.check_rank_exact(r;throw=true).valid
    b=RX.rank_exact_table(r);M=RX.rank_exact_module(r);Q=M.Q
    for u in 1:FF.nvertices(Q),v in 1:FF.nvertices(Q)
        FF.leq(Q,u,v) || continue
        @test RX.reconstructed_rank(r;source=u,target=v)==_rx_rank(MD.map_leq(M,u,v))
    end
    c=RX.hook_complex(r)
    @test c.augmentation.dom===c.terms[1]
    for (i,d) in enumerate(c.differentials)
        @test d.dom===c.terms[i+1] && d.cod===c.terms[i]
        prev=i==1 ? c.augmentation : c.differentials[i-1]
        for v in 1:FF.nvertices(Q)
            @test iszero(prev.comps[v]*d.comps[v])
            @test _rx_rank(prev.comps[v])+_rx_rank(d.comps[v])==d.cod.dims[v]
        end
    end
    # The numerical bridge is compared with the existing complete ordinary
    # Betti engine on ALL interval vertices, including diagonals and infinity.
    km,pairs,_=RX._kernel_diagram(M,RX.RankExactBudget())
    @test MD.check_module_functoriality(km).valid
    ordinary=DF.betti_table(km;opts=OPT.ResolutionOptions(maxlen=max(0,length(pairs)-1),minimal=true))
    off=findall(p->p[1]!=p[2],pairs);numeric=ordinary[:,off]
    lastrow=findlast(i->any(!iszero,numeric[i,:]),axes(numeric,1));lastrow===nothing && (lastrow=1)
    @test b==numeric[1:lastrow,:]
end

@testset "Rank-exact hook algebra" begin
    @testset "Hand Betti tables and actual minimal maps" begin
        for F in (CM.QQField(),CM.F2(),CM.F3(),CM.Fp(101))
            Q=FF.ProductOfChainsPoset((2,2));M=_rx_indicator(Q,[1],F)
            r=RX.rank_exact_resolution(M;threads=false)
            @test _rx_counts(r)==Dict((degree=0,birth=1,death=2)=>1,
                (degree=0,birth=1,death=3)=>1,(degree=1,birth=1,death=4)=>1)
            _rx_assert(r)
            @test RX.rank_exact_table(RX.rank_exact_betti(M))==RX.rank_exact_table(r)
            @test RX.rank_exact_summary(r).complete
            @test RX.rank_exact_summary(r).betti_totals==(2,1)
            ordinary_M=DF.betti_table(M;opts=OPT.ResolutionOptions(maxlen=3))
            @test Tuple(vec(sum(ordinary_M;dims=2)))==(1,2,1)
            @test Set(RX.hook_summands(r;degree=0))==Set([(birth=1,death=2),(birth=1,death=3)])
            @test only(RX.hook_summands(r;degree=1))==(birth=1,death=4)
            @test Dict((x.birth,x.death)=>x.coefficient for x in RX.signed_hook_decomposition(r))==Dict((1,2)=>1,(1,3)=>1,(1,4)=>-1)
            # A fork has no finite join: its syzygy is a principal upset.
            Qf=FF.FinitePoset(Bool[1 1 1;0 1 0;0 0 1]);Mf=_rx_indicator(Qf,[1],F)
            rf=RX.rank_exact_resolution(Mf)
            @test _rx_counts(rf)==Dict((degree=0,birth=1,death=2)=>1,
                (degree=0,birth=1,death=3)=>1,(degree=1,birth=1,death=nothing)=>1)
            _rx_assert(rf)
            # Three minimal outgoing directions: the nonzero relative Betti
            # multiplicities are the nonempty faces of a three-element set.
            Q3=FF.ProductOfChainsPoset((2,2,2));M3=_rx_indicator(Q3,[1],F)
            r3=RX.rank_exact_resolution(M3)
            @test _rx_counts(r3)==Dict((degree=0,birth=1,death=2)=>1,
                (degree=0,birth=1,death=3)=>1,(degree=0,birth=1,death=5)=>1,
                (degree=1,birth=1,death=4)=>1,(degree=1,birth=1,death=6)=>1,
                (degree=1,birth=1,death=7)=>1,(degree=2,birth=1,death=8)=>1)
            _rx_assert(r3)
            # Every interval on a chain is rank-projective, including the tail.
            C=FF.ProductOfChainsPoset((3,))
            for a in 1:3,b in a:3
                N=_rx_indicator(C,a:b,F);rn=RX.rank_exact_resolution(N)
                @test _rx_counts(rn)==Dict((degree=0,birth=a,death=b==3 ? nothing : b+1)=>1)
                @test RX.check_rank_exact(rn).valid
                @test RX.rank_exact_summary(RX.rank_exact_betti(N;maxdegree=0)).complete
            end
            # Every member of the declared finite hook family is projective
            # for this exact structure, even when it has an ordinary relation.
            for h in RX.hook_labels(r)
                H=RX.hook_module(Q;birth=h.birth,death=h.death,field=F)
                rh=RX.rank_exact_betti(H)
                @test _rx_counts(rh)==Dict((degree=0,birth=h.birth,death=h.death)=>1)
            end
            H=RX.hook_module(Q;birth=1,death=4,field=F)
            S=MD.direct_sum(M,MD.direct_sum(M,H));rs=RX.rank_exact_resolution(S)
            expected=Dict((degree=0,birth=1,death=2)=>2,(degree=0,birth=1,death=3)=>2,
                (degree=0,birth=1,death=4)=>1,(degree=1,birth=1,death=4)=>2)
            @test _rx_counts(rs)==expected
            scrambled=_rx_scramble(S,MersenneTwister(105))
            rr=RX.rank_exact_resolution(scrambled)
            @test _rx_counts(rr)==expected
            _rx_assert(rr)
            # Three distinct lines in a two-dimensional sink: an indecomposable
            # star representation which cannot be a sum of thin intervals.
            Qs=FF.FinitePoset(Bool[1 0 0 1;0 1 0 1;0 0 1 1;0 0 0 1]);K=CM.coeff_type(F)
            star=MD.PModule{K}(Qs,[1,1,1,2],Dict((1,4)=>reshape(K[1,0],2,1),
                (2,4)=>reshape(K[0,1],2,1),(3,4)=>reshape(K[1,1],2,1));field=F)
            sr=RX.rank_exact_resolution(_rx_scramble(star,MersenneTwister(1051)))
            @test _rx_counts(sr)==Dict((degree=0,birth=1,death=nothing)=>1,
                (degree=0,birth=2,death=nothing)=>1,(degree=0,birth=3,death=nothing)=>1,
                (degree=1,birth=4,death=nothing)=>1)
            _rx_assert(sr)
        end
    end
    @testset "Fields, truncation, relabeling and contracts" begin
        for F in (CM.QQField(),CM.F2(),CM.F3(),CM.Fp(101))
            K=CM.coeff_type(F);Q=FF.ProductOfChainsPoset((2,))
            M=MD.PModule{K}(Q,[2,2],Dict((1,2)=>K[1 1;1 -1]);field=F)
            r=RX.rank_exact_resolution(M)
            expected=F==CM.F2() ? Dict((degree=0,birth=1,death=2)=>1,
                (degree=0,birth=1,death=nothing)=>1,(degree=0,birth=2,death=nothing)=>1) :
                Dict((degree=0,birth=1,death=nothing)=>2)
            @test _rx_counts(r)==expected
            _rx_assert(r)
        end
        F=CM.QQField();Q=FF.ProductOfChainsPoset((2,2));M=_rx_indicator(Q,[1],F)
        prefix=RX.rank_exact_resolution(M;maxdegree=0)
        @test !RX.rank_exact_summary(prefix).complete
        @test RX.check_rank_exact(prefix).valid
        @test_throws ArgumentError RX.signed_hook_decomposition(prefix)
        @test_throws ArgumentError RX.reconstructed_rank(prefix;source=1,target=1)
        r=RX.rank_exact_resolution(M)
        @test TamerOp.rank_exact_resolution===RX.rank_exact_resolution
        @test TOA.hook_complex===RX.hook_complex
        @test CC.describe(r)==RX.rank_exact_summary(r)
        @test CC.dimensions(r)==[1,0,0,0]
        @test occursin("HookResolution",sprint(show,r))
        @test RES.provenance(r).degree_shift==0
        @test RES.provenance(r).infinity===:adjoined_zero_stalk
        @test RX.rank_exact_summary(RX.rank_exact_betti(M)).maps==false
        B=RX.rank_exact_table(r);B[1,1]=77
        @test RX.rank_exact_table(r)[1,1]!=77
        c=RX.hook_complex(r);c.augmentation.comps[1].*=0
        @test RX.check_rank_exact(r).valid
        damaged=deepcopy(r);damaged.augmentation.comps[1].*=0
        @test !RX.check_rank_exact(damaged).valid
        @test_throws ArgumentError RX.check_rank_exact(damaged;throw=true)
        damaged=deepcopy(r);damaged.coefficients[1][1,1]+=1
        @test !RX.check_rank_exact(damaged).valid
        damaged=deepcopy(r.betti);damaged.table[1,1]+=1
        @test !RX.check_rank_exact(damaged).valid
        # An ordinary projective resolution is exact but fails the additional
        # rank-exact requirement. Test that condition separately from Betti
        # recomputation, using the same legitimate lower-hook family.
        ordinary=DF.Resolutions.projective_resolution(M,OPT.ResolutionOptions(maxlen=2))
        labels=r.betti.hooks;indices=Dict(h=>i for (i,h) in enumerate(labels))
        ids=[[indices[(v,nothing)] for v in gs] for gs in ordinary.gens]
        bt=zeros(Int,3,length(labels))
        for i in 1:3,j in ids[i];bt[i,j]+=1;end
        fake=RX.HookResolution{QQ,typeof(M)}(RX.RankExactBetti(M,labels,bt,true,2),
            MD.PModule{QQ}[ordinary.Pmods...],ids,MD.PMorphism{QQ}[ordinary.d_mor...],ordinary.d_mat,ordinary.aug)
        @test_throws ArgumentError RX._validate_maps(fake,RX.RankExactBudget())
        for md in (-1,true,0.5)
            @test_throws ArgumentError RX.rank_exact_betti(M;maxdegree=md)
        end
        for b in (RX.RankExactBudget(max_vertices=1),RX.RankExactBudget(max_intervals=1),
                  RX.RankExactBudget(max_entries=1),RX.RankExactBudget(max_generators=0),RX.RankExactBudget(max_degrees=0))
            @test_throws ArgumentError RX.rank_exact_resolution(M;budget=b)
        end
        @test_throws ArgumentError RX.RankExactBudget(max_entries=-1)
        @test_throws ArgumentError RX.rank_exact_betti(_rx_indicator(Q,[1],CM.RealField(Float64)))
        @test_throws ArgumentError RX.hook_module(Q;birth=1,death=1)
        @test_throws ArgumentError RX.hook_module(Q;birth=2,death=3)
        @test_throws ArgumentError RX.hook_module(Q;birth=true)
        @test_throws ArgumentError RX.hook_summands(r;degree=10)
        @test_throws ArgumentError RX.reconstructed_rank(r;source=2,target=3)
        z=_rx_indicator(Q,Int[],F);rz=RX.rank_exact_resolution(z)
        @test RX.rank_exact_summary(rz).betti_totals==(0,)
        @test isempty(RX.signed_hook_decomposition(rz))
        @test RX.check_rank_exact(rz).valid
        Q0=FF.FinitePoset(falses(0,0));M0=_rx_indicator(Q0,Int[],F)
        r0=RX.rank_exact_resolution(M0)
        @test size(RX.rank_exact_table(r0))==(1,0)
        @test RX.check_rank_exact(r0).valid
        # The finite labels need not be a topological ordering.
        Qp=FF.FinitePoset(Bool[1 0 1;1 1 1;0 0 1]);Mp=_rx_indicator(Qp,[2,1],F)
        rp=RX.rank_exact_resolution(Mp)
        @test _rx_counts(rp)==Dict((degree=0,birth=2,death=3)=>1)
        _rx_assert(rp)
        out=Vector{Matrix{Int}}(undef,4)
        Threads.@threads for i in eachindex(out)
            out[i]=RX.rank_exact_table(RX.rank_exact_betti(M;threads=isodd(i)))
        end
        @test all(==(out[1]),out)
        # Snapshot the input instead of caching results on a mutable module.
        N=_rx_indicator(FF.ProductOfChainsPoset((2,)),[1,2],F)
        before=RX.rank_exact_betti(N);N.edge_maps[1,2][1,1]=0
        @test _rx_counts(before)!=_rx_counts(RX.rank_exact_betti(N))
        @test RX.check_rank_exact(before).valid
        bad=_rx_indicator(Q,1:4,F);bad.edge_maps[1,2][1,1]=0
        @test_throws ErrorException RX.rank_exact_betti(bad)
    end
end
