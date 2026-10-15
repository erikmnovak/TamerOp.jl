# Independent elementary elimination and the subspace-quotient definition of P_f.
function _mm_rref(A)
    R=copy(A);piv=Int[];r=1
    for c in axes(R,2)
        p=findfirst(i->!iszero(R[i,c]),r:size(R,1))
        p===nothing && continue
        p+=r-1;R[r,:],R[p,:]=copy(R[p,:]),copy(R[r,:]);R[r,:]/=R[r,c]
        for i in axes(R,1);i==r || (R[i,:]-=R[i,c]*R[r,:]);end
        push!(piv,c);r+=1;r>size(R,1) && break
    end
    R,piv
end
_mm_rank(A)=length(last(_mm_rref(A)))
_mm_col(A)=A[:,last(_mm_rref(A))]
function _mm_null(A)
    R,piv=_mm_rref(A);free=setdiff(collect(axes(A,2)),piv);N=zeros(eltype(A),size(A,2),length(free))
    for (j,c) in enumerate(free)
        N[c,j]=one(eltype(A))
        for (i,p) in enumerate(piv);N[p,j]=-R[i,c];end
    end
    N
end
_mm_intersect(A,B)=_mm_col(A*_mm_null(hcat(A,-B))[1:size(A,2),:])
function _mm_path(M,a,b)
    K=CM.coeff_type(M.field);n=length(M.dims)
    (a==0 || b==n+1) && return zeros(K,b==n+1 ? 0 : M.dims[b],a==0 ? 0 : M.dims[a])
    X=Matrix{K}(I,M.dims[a],M.dims[a])
    for t in a:b-1;X=Matrix(M.edge_maps[t,t+1])*X;end
    X
end
function _mm_quotient(f,a,b,c,d)
    c<=a<=d<=b || return 0
    A=f.dom;B=f.cod;t=a
    lo=_mm_path(A,a-1,t)
    V=hcat(_mm_null(_mm_path(A,t,b+1)),lo)
    Vm=hcat(_mm_null(_mm_path(A,t,b)),lo)
    loB=_mm_null(_mm_path(B,t,d))
    W=hcat(_mm_intersect(_mm_path(B,c,t),_mm_null(_mm_path(B,t,d+1))),loB)
    Wm=hcat(_mm_intersect(_mm_path(B,c-1,t),_mm_null(_mm_path(B,t,d+1))),loB)
    top=_mm_intersect(f.comps[t]*V,W)
    bottom=hcat(_mm_intersect(f.comps[t]*Vm,W),_mm_intersect(f.comps[t]*V,Wm))
    @test _mm_rank(hcat(top,bottom))==_mm_rank(top)
    _mm_rank(top)-_mm_rank(bottom)
end

function _mm_fixture(ss,tt,C,F;n=maximum(vcat(last.(ss),last.(tt));init=1),scramble=false,rng=MersenneTwister(111))
    K=CM.coeff_type(F);Q=FF.ProductOfChainsPoset((n,))
    function diagram(supports)
        active=[findall(s->s[1]<=t<=s[2],supports) for t in 1:n]
        Bs=Matrix{K}[];Is=Matrix{K}[]
        for ids in active
            d=length(ids);B=Matrix{K}(I,d,d);Bi=copy(B)
            if scramble && d>1
                for _ in 1:3d
                    a,b=randperm(rng,d)[1:2];c=K(rand(rng,-2:2))
                    B[a,:]+=c*B[b,:];Bi[:,b]-=c*Bi[:,a]
                end
            end
            push!(Bs,B);push!(Is,Bi)
        end
        maps=Dict{Tuple{Int,Int},Matrix{K}}((t,t+1)=>Bs[t+1]*K[i==j for i in active[t+1],j in active[t]]*Is[t] for t in 1:n-1)
        MD.PModule{K}(Q,length.(active),maps;field=F),active,Bs,Is
    end
    A,ai,ab,ax=diagram(ss);B,bi,bb,bx=diagram(tt)
    for j in eachindex(ss),i in eachindex(tt)
        a,b=ss[j];c,d=tt[i]
        @assert iszero(C[i,j]) || c<=a<=d<=b
    end
    MD.PMorphism(A,B,[bb[t]*K.(C[bi[t],ai[t]])*ax[t] for t in 1:n])
end
_mm_pairs(r)=Dict(((x.source.first,x.source.last),(x.target.first,x.target.last))=>x.multiplicity for x in MM.matched_intervals(r))
_mm_bars(r;side=:image)=Dict((x.interval.first,x.interval.last)=>x.multiplicity for x in MM.matching_intervals(r;side))
function _mm_oracles(f,r)
    n=length(f.dom.dims);got=_mm_pairs(r)
    for (a,b) in r.source_supports,(c,d) in r.target_supports
        @test get(got,((a,b),(c,d)),0)==_mm_quotient(f,a,b,c,d)
    end
    bars=_mm_bars(r)
    for a in 1:n,b in a:n
        @test sum(k for ((c,d),k) in bars if c<=a<=b<=d;init=0)==_mm_rank(_mm_path(f.cod,a,b)*f.comps[a])
    end
    @test all(vec(sum(r.counts;dims=2)).<=r.source_counts)
    @test all(vec(sum(r.counts;dims=1)).<=r.target_counts)
end

@testset "Additive morphism matching" begin
    @testset "Exact quotient, image and basis oracles" begin
        rng=MersenneTwister(111)
        for F in (CM.QQField(),CM.F2(),CM.F3(),CM.Fp(101))
            f=_mm_fixture([(2,2),(2,3)],[(1,2)],reshape([1,0],1,2),F;n=3)
            g=_mm_fixture([(2,2),(2,3)],[(1,2)],reshape([0,1],1,2),F;n=3)
            rf=MM.additive_matching(f;witness=true);rg=MM.additive_matching(g)
            @test _mm_pairs(rf)==Dict(((2,2),(1,2))=>1)
            @test _mm_pairs(rg)==Dict(((2,3),(1,2))=>1)
            @test _mm_bars(rf)==_mm_bars(rg)==Dict((2,2)=>1)
            @test MM.check_additive_matching(rf).valid
            @test MM.additive_matching_summary(rf).unmatched_source==1
            @test only(MM.unmatched_intervals(rf)).interval.last==3
            for (ss,tt,C,n) in (
                ([(1,4),(2,3)],[(2,2),(1,3)],[0 1;1 1],4),
                ([(1,3),(2,3)],[(1,1),(1,3)],[1 0;1 1],3),
                ([(1,1),(2,2)],[(1,1),(1,2)],[1 0;0 1],2),
                ([(1,3),(2,3),(2,3)],[(1,2),(1,3),(2,3)],[1 1 1;1 1 -1;0 1 1],3),
                ([(1,3),(1,3)],[(1,3),(1,3)],[1 1;1 -1],3))
                f=_mm_fixture(ss,tt,C,F;n);r=MM.additive_matching(f;witness=true)
                _mm_oracles(f,r)
                w=MM.additive_matching_witness(r)
                for t in 1:n
                    si=findall(s->s[1]<=t<=s[2],w.source_supports)
                    ti=findall(s->s[1]<=t<=s[2],w.target_supports)
                    @test f.comps[t]*w.source_bases[t]==w.target_bases[t]*w.coefficients[ti,si]
                    @test _mm_rank(w.source_bases[t])==f.dom.dims[t]
                    @test _mm_rank(w.target_bases[t])==f.cod.dims[t]
                end
                fs=_mm_fixture(ss,tt,C,F;n,scramble=true,rng)
                @test _mm_pairs(MM.additive_matching(fs))==_mm_pairs(r)
                if C==[0 1;1 1]
                    @test _mm_pairs(r)==Dict(((1,4),(1,3))=>1,((2,3),(2,2))=>1)
                end
            end
            supports=[(a,b) for a in 1:4 for b in a:4]
            for _ in 1:12
                ss=rand(rng,supports,6);tt=rand(rng,supports,6)
                C=[c<=a<=d<=b ? rand(rng,-2:2) : 0 for (c,d) in tt,(a,b) in ss]
                f=_mm_fixture(ss,tt,C,F;n=4);r=MM.additive_matching(f)
                _mm_oracles(f,r)
                @test _mm_pairs(r)==_mm_pairs(MM.additive_matching(_mm_fixture(ss,tt,C,F;n=4,scramble=true,rng)))
                # Direct sum with itself has twice every multiplicity.
                fd=_mm_fixture(vcat(ss,ss),vcat(tt,tt),[C zero(C);zero(C) C],F;n=4)
                @test _mm_pairs(MM.additive_matching(fd))==Dict(k=>2v for (k,v) in _mm_pairs(r))
            end
            ss=[(1,1),(1,3),(2,3),(2,3)]
            id=_mm_fixture(ss,ss,Matrix{Int}(I,4,4),F;n=3)
            @test _mm_pairs(MM.additive_matching(id))==Dict((s,s)=>count(==(s),ss) for s in ss)
            z=_mm_fixture(ss,ss,zeros(Int,4,4),F;n=3)
            @test isempty(MM.matched_intervals(MM.additive_matching(z)))
            # Composable nonzero maps of intervals can have zero composite.
            f=_mm_fixture([(2,3)],[(1,2)],ones(Int,1,1),F;n=3)
            g=_mm_fixture([(1,2)],[(1,1)],ones(Int,1,1),F;n=3)
            @test MM.additive_matching_summary(MM.additive_matching(f)).matched==1
            @test MM.additive_matching_summary(MM.additive_matching(g)).matched==1
            h=_mm_fixture([(2,3)],[(1,1)],zeros(Int,1,1),F;n=3)
            @test all(iszero(g.comps[t]*f.comps[t]) for t in 1:3)
            @test isempty(MM.matched_intervals(MM.additive_matching(h)))
        end
    end
    @testset "Contracts, chain restrictions and ownership" begin
        F=CM.QQField();K=QQ
        f=_mm_fixture([(1,3)],[(1,3)],ones(Int,1,1),F;n=3)
        r=MM.additive_matching(f;chain=[1,3],values=[1//3,7//3],witness=true)
        @test MM.matching_chain(r)==[1,3]
        @test MM.matching_parameters(r)==[1//3,7//3]
        @test only(MM.matched_intervals(r)).source.terminal===:censored
        @test only(MM.matched_intervals(r)).source.death===nothing
        @test RES.provenance(r).scope===:chain_restriction
        @test MM.check_additive_matching(r;throw=true).valid
        w=MM.additive_matching_witness(r);w.coefficients[1,1]=0
        @test MM.additive_matching_witness(r).coefficients[1,1]==1
        copied=MM.matching_morphism(r);copied.comps[1][1,1]=0
        @test MM.check_additive_matching(r).valid
        f.comps[1][1,1]=0
        @test MM.check_additive_matching(r).valid
        damaged=deepcopy(r);damaged.counts[1,1]=0
        @test !MM.check_additive_matching(damaged).valid
        @test_throws ArgumentError MM.check_additive_matching(damaged;throw=true)
        damaged=deepcopy(r);damaged.morphism.comps[1][1,1]=0
        @test !MM.check_additive_matching(damaged).valid
        @test CC.describe(r)==MM.additive_matching_summary(r)
        @test occursin("matched",sprint(show,r))
        @test TamerOp.additive_matching===MM.additive_matching
        @test TOA.check_additive_matching===MM.check_additive_matching
        @test CC.dimensions(r)==(source=[1,1],target=[1,1])
        f=_mm_fixture([(1,3)],[(1,3)],ones(Int,1,1),F;n=3)
        @test_throws ArgumentError MM.additive_matching_witness(MM.additive_matching(f))
        for chain in ([2,1],[1,1],[0],[4],[true])
            @test_throws ArgumentError MM.additive_matching(f;chain)
        end
        for values in ([1,2],[1,1,3],[1,2,Inf],1)
            @test_throws ArgumentError MM.additive_matching(f;values)
        end
        for b in (MM.AdditiveMatchingBudget(max_vertices=1),MM.AdditiveMatchingBudget(max_dimension=0),
                  MM.AdditiveMatchingBudget(max_entries=1),MM.AdditiveMatchingBudget(max_intervals=0))
            @test_throws ArgumentError MM.additive_matching(f;budget=b)
        end
        @test_throws ArgumentError MM.AdditiveMatchingBudget(max_entries=-1)
        @test_throws ArgumentError MM.additive_matching(_mm_fixture([(1,2)],[(1,2)],ones(Int,1,1),CM.RealField(Float64);n=2))
        @test_throws ArgumentError MM.matching_intervals(r;side=:wrong)
        @test_throws ArgumentError MM.unmatched_intervals(r;side=:wrong)
        @test MM.check_additive_matching(MM.additive_matching(f;chain=Int[])).valid
        for (ss,tt,C) in ((Tuple{Int,Int}[],[(1,3)],zeros(Int,1,0)), ([(1,3)],Tuple{Int,Int}[],zeros(Int,0,1)))
            @test isempty(MM.matched_intervals(MM.additive_matching(_mm_fixture(ss,tt,C,F;n=3))))
        end
        Q=FF.ProductOfChainsPoset((2,2));M=MD.PModule{K}(Q,ones(Int,4),Dict(e=>ones(K,1,1) for e in FF.cover_edges(Q));field=F)
        g=MD.PMorphism(M,M,[ones(K,1,1) for _ in 1:4])
        @test_throws ArgumentError MM.additive_matching(g)
        rr=MM.additive_matching(g;chain=[1,4])
        @test MM.additive_matching_summary(rr).matched==1
        @test MM.check_additive_matching(rr).valid
        @test_throws ArgumentError MM.additive_matching(g;chain=[2,3])
        Q0=FF.FinitePoset(falses(0,0));M0=MD.PModule{K}(Q0,Int[],Dict{Tuple{Int,Int},Matrix{K}}();field=F)
        @test MM.check_additive_matching(MM.additive_matching(MD.PMorphism(M0,M0,Matrix{K}[]))).valid
        # Relabel a chain: order is recovered from covers, not vertex numbering.
        L=Bool[1 0 1;1 1 1;0 0 1];Qp=FF.FinitePoset(L)
        Mp=MD.PModule{K}(Qp,ones(Int,3),Dict(e=>ones(K,1,1) for e in FF.cover_edges(Qp));field=F)
        @test MM.matching_chain(MM.additive_matching(MD.PMorphism(Mp,Mp,[ones(K,1,1) for _ in 1:3])))==[2,1,3]
        bad=deepcopy(f);bad.comps[1][1,1]=0
        @test_throws Exception MM.additive_matching(bad)
        out=Vector{Any}(undef,4)
        Threads.@threads for i in eachindex(out);out[i]=_mm_pairs(MM.additive_matching(f));end
        @test all(==(out[1]),out)
    end
end
