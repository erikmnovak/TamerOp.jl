# Brute-force vertex matrices are an independent oracle: no production Hom
# basis, shifted pullback or witness solver is used by this decision routine.
function _il_brute(M,N,t,p)
    n=length(M.dims); K=CM.coeff_type(M.field)
    function candidates(A,B)
        shapes=[(B.dims[t[i]],A.dims[i]) for i in 1:n]
        sizes=[prod(s) for s in shapes]; total=sum(sizes)
        out=Vector{Vector{Matrix{K}}}()
        for code in 0:p^total-1
            value=code; maps=Matrix{K}[]
            for shape in shapes
                X=zeros(K,shape)
                for k in eachindex(X); value,digit=divrem(value,p);X[k]=K(digit);end
                push!(maps,X)
            end
            all(e -> MD.map_leq(B,t[e[1]],t[e[2]])*maps[e[1]] ==
                     maps[e[2]]*MD.map_leq(A,e[1],e[2]),FF.cover_edges(M.Q)) && push!(out,maps)
        end
        return out
    end
    for f in candidates(M,N),g in candidates(N,M)
        if all(i -> g[t[i]]*f[i]==MD.map_leq(M,i,t[t[i]]) &&
                    f[t[i]]*g[i]==MD.map_leq(N,i,t[t[i]]),1:n)
            return true
        end
    end
    return false
end

function _il_interval(a,b;left=true,right=false,field=CM.QQField(),domain=:real)
    cuts=([a,b],);K=CM.coeff_type(field)
    dims=domain===:real ? [0,Int(left),1,Int(right),0] : [0,1,0]
    maps=Dict((i,i+1)=>ones(K,dims[i+1],dims[i]) for i in 1:length(dims)-1)
    return IL.RectilinearModule(cuts,dims,maps;field,domain)
end
function _il_constant(d;field=CM.QQField(),n=1,domain=:real)
    IL.RectilinearModule(ntuple(_->Int[],n),[d],Dict{Tuple{Int,Int},Matrix{CM.coeff_type(field)}}();field,domain)
end

@testset "A53 exact ambient shifts and boundaries" begin
    @info "A53: exact ambient shifts and boundaries"
    for field in (CM.QQField(),CM.F2(),CM.F3())
        M=_il_interval(0,2;field)
        zero=_il_constant(0;field)
        shifted=IL.parameter_shift(M,1//3)
        refined=IL.refine_parameters(M,([-1,0,1//3,1,2,3],))
        for x in (-10,-1//3,0,1//3,1,2,10)
            @test IL.ambient_dimension(shifted,[x])==IL.ambient_dimension(M,[x+1//3])
            @test IL.ambient_dimension(refined,[x])==IL.ambient_dimension(M,[x])
            for y in (x,3,20)
                y<x && continue
                @test MD.structure_map(shifted;source=[x],target=[y])==MD.structure_map(M;source=[x+1//3],target=[y+1//3])
                @test MD.structure_map(refined;source=[x],target=[y])==MD.structure_map(M;source=[x],target=[y])
            end
        end
        @test IL.find_interleaving(M,zero,1).status===:interleaved
        @test IL.find_interleaving(M,zero,99//100).status===:not_interleaved
        closed=_il_interval(0,2;field,right=true)
        @test IL.find_interleaving(closed,zero,1).status===:not_interleaved
        @test IL.find_interleaving(closed,zero,101//100).status===:interleaved
        @test IL.find_interleaving(M,M,0).status===:interleaved
        N=IL.parameter_shift(M,-1//2)
        @test IL.find_interleaving(M,N,1//2).status===:interleaved
        @test IL.find_interleaving(M,N,49//100).status===:not_interleaved
        r=IL.find_interleaving(M,N,1//2)
        @test IL.check_interleaving(IL.interleaving_witness(r)).valid
        @test IL.provenance(r.certificate).epsilon==1//2
        @test IL.provenance(r.certificate).confidence===nothing
        @test !isempty(sprint(show,r))
        @test IL.describe(M).provenance.window===nothing
    end
    M=_il_interval(0,3;domain=:integer)
    z=_il_constant(0;domain=:integer)
    @test IL.find_interleaving(M,z,1).status===:not_interleaved
    @test IL.find_interleaving(M,z,2).status===:interleaved
    @test_throws ArgumentError IL.parameter_shift(M,1//2)
    @test_throws ArgumentError IL.ambient_dimension(M,[1//2])
    @test_throws ArgumentError IL.refine_parameters(M,([0,2],))
    base=big(2)^54
    M=_il_interval(base,base+2)
    N=IL.parameter_shift(M,-1//2)
    @test IL.find_interleaving(M,N,1//2).status===:interleaved
    @test IL.find_interleaving(M,N,1//4).status===:not_interleaved
    @test IL.find_interleaving(_il_constant(1),_il_constant(0),10^6).status===:not_interleaved
end

@testset "A53 independent finite-field witness oracle" begin
    @info "A53: independent finite-field witness oracle"
    for field in (CM.F2(),CM.F3())
        K=CM.coeff_type(field);p=field.p
        Q=FF.GridPoset(([1,2],))
        modules=MD.PModule{K}[]
        for a in 0:1,b in 0:1, entry in 0:((a*b==0) ? 0 : p-1)
            push!(modules,MD.PModule{K}(Q,[a,b],Dict((1,2)=>fill(K(entry),b,a));field))
        end
        for M in modules,N in modules,t in ([1,2],[2,2])
            r=IL.find_interleaving(M,N,t)
            @test r.status!==:unknown
            @test (r.status===:interleaved)==_il_brute(M,N,t,p)
            if r.certificate!==nothing
                @test IL.check_interleaving(r.certificate).valid
                @test IL.provenance(r.certificate).epsilon===nothing
            end
        end
        # A genuine two-parameter square, all one-dimensional commuting maps.
        Q=FF.GridPoset(([1,2],[1,2])); edges=FF.cover_edges(Q)
        modules=MD.PModule{K}[]
        for entries in Iterators.product(ntuple(_->0:p-1,4)...)
            entries[1]*entries[4]%p==entries[2]*entries[3]%p || continue
            maps=Dict(e=>fill(K(entries[j]),1,1) for (j,e) in enumerate(edges))
            M=MD.PModule{K}(Q,ones(Int,4),maps;field)
            MD.check_module_functoriality(M).valid && push!(modules,M)
        end
        rng=MersenneTwister(53)
        for _ in 1:12
            M,N=rand(rng,modules,2);t=[2,2,4,4]
            r=IL.find_interleaving(M,N,t)
            @test (r.status===:interleaved)==_il_brute(M,N,t,p)
        end
    end
end

@testset "A53 general matrices and supplied certificates" begin
    @info "A53: general matrices and supplied certificates"
    for field in (CM.QQField(),CM.F2(),CM.F3())
        K=CM.coeff_type(field);Q=FF.GridPoset(([1,2,3],))
        A=K[1 1;0 1];B=K[1 0;1 1]
        M=MD.PModule{K}(Q,[2,2,2],Dict((1,2)=>A,(2,3)=>B);field)
        N=MD.PModule{K}(Q,[2,2,2],Dict((1,2)=>B,(2,3)=>A);field)
        problem=IL.interleaving_problem(M,N,[1,2,3])
        r=IL.find_interleaving(problem)
        @test r.status===:interleaved
        c=r.certificate
        @test IL.check_interleaving(c).valid
        for i in 1:3
            e=IL.interleaving_equations(c;vertex=i)
            @test e.forward.lhs==e.forward.rhs==Matrix{K}(I,2,2)
            @test e.reverse.lhs==e.reverse.rhs
        end
        @test IL.find_interleaving(problem;forward=c.forward).status===:interleaved
        @test IL.find_interleaving(problem;reverse=c.reverse).status===:interleaved
        @test_throws ArgumentError IL.find_interleaving(problem;forward=c.forward,reverse=c.reverse)
        z=[zeros(K,2,2) for _ in 1:3]
        @test IL.find_interleaving(problem;forward=z).reason===:fixed_map_obstruction
        @test IL.find_interleaving(problem;forward=z).status===:unknown
        @test_throws ArgumentError IL.interleaving_certificate(problem,z,c.reverse)
        bad=IL.InterleavingCertificate(problem,c.forward,MD.PMorphism(problem.sources[2],problem.targets[2],z))
        report=IL.check_interleaving(bad)
        @test !report.valid
        @test report.forward_failures==collect(1:3)
        @test report.reverse_failures==collect(1:3)
        @test_throws ArgumentError IL.check_interleaving(bad;throw=true)
        c.forward.comps[1][1,1]+=one(K)
        @test !IL.check_interleaving(c).valid
    end
    # Constant modules of unequal dimension: the linear relaxation is feasible
    # in characteristic two (trace obstruction disappears), but no isomorphism
    # exists. Exhaustive search must still certify nonexistence.
    Q=FF.GridPoset(([1],));K=CM.coeff_type(CM.F2())
    M=MD.PModule{K}(Q,[1],Dict{Tuple{Int,Int},Matrix{K}}();field=CM.F2())
    N=MD.PModule{K}(Q,[3],Dict{Tuple{Int,Int},Matrix{K}}();field=CM.F2())
    r=IL.find_interleaving(M,N,[1])
    @test r.status===:not_interleaved
    @test r.reason===:exhaustive_finite_field
    @test IL.find_interleaving(M,N,[1];budget=IL.InterleavingBudget(max_candidates=1)).status===:unknown
    M=_il_constant(2)
    @test IL.find_interleaving(M,M,0;budget=IL.InterleavingBudget(max_candidates=1)).status===:unknown
    # The rational solver must recover denominators in the second map, even
    # when the fixed first map has integral coefficients.
    p=IL.interleaving_problem(_il_constant(1),_il_constant(1),0)
    r=IL.find_interleaving(p;forward=[reshape(QQ[7],1,1)])
    @test only(r.certificate.reverse.comps[1])==1//7
    @test IL.check_interleaving_problem(p).valid
    p.canonical[1][1][1,1]=2
    @test !IL.check_interleaving_problem(p).valid
    @test !IL.check_interleaving(r.certificate).valid
    @test_throws ArgumentError IL.find_interleaving(p)
    # The first composition identity alone is insufficient.
    p=IL.interleaving_problem(_il_constant(1),_il_constant(2),0)
    f=MD.PMorphism(p.sources[1],p.targets[1],[reshape(QQ[1,0],2,1)])
    g=MD.PMorphism(p.sources[2],p.targets[2],[reshape(QQ[1,0],1,2)])
    report=IL.check_interleaving(IL.InterleavingCertificate(p,f,g))
    @test !report.valid && isempty(report.forward_failures) && report.reverse_failures==[1]
    # Inverse vertex matrices can still fail naturality.
    Q=FF.GridPoset(([1,2],))
    M=MD.PModule{QQ}(Q,[1,1],Dict((1,2)=>ones(QQ,1,1));field=CM.QQField())
    p=IL.interleaving_problem(M,M,[1,2])
    f=MD.PMorphism(p.sources[1],p.targets[1],[ones(QQ,1,1),fill(QQ(2),1,1)])
    g=MD.PMorphism(p.sources[2],p.targets[2],[ones(QQ,1,1),fill(QQ(1//2),1,1)])
    @test !IL.check_interleaving(IL.InterleavingCertificate(p,f,g)).valid
    # Three independent source lines at a two-dimensional sink form a
    # non-interval indecomposable; no barcode pairing can replace these maps.
    Q=FF.FinitePoset(Bool[1 0 0 1;0 1 0 1;0 0 1 1;0 0 0 1])
    for field in (CM.QQField(),CM.F2(),CM.F3(),CM.Fp(101))
        K=CM.coeff_type(field)
        maps=Dict((1,4)=>reshape(K[1,0],2,1),(2,4)=>reshape(K[0,1],2,1),(3,4)=>reshape(K[1,1],2,1))
        M=MD.PModule{K}(Q,[1,1,1,2],maps;field)
        U=K[1 1;0 1]
        N=MD.PModule{K}(Q,[1,1,1,2],Dict(e=>U*A for (e,A) in maps);field)
        r=IL.find_interleaving(M,N,collect(1:4))
        @test r.status===:interleaved
        @test IL.check_interleaving(r.certificate).valid
    end
    # Unequal Hom dimensions exercise solving for either direction. The
    # additional middle-vertex class vanishes under the double translation.
    Q=FF.GridPoset(([1,2,3],))
    for field in (CM.QQField(),CM.F2(),CM.F3())
        K=CM.coeff_type(field)
        M=MD.PModule{K}(Q,[1,1,1],Dict((1,2)=>ones(K,1,1),(2,3)=>ones(K,1,1));field)
        N=MD.PModule{K}(Q,[1,2,1],Dict((1,2)=>reshape(K[1,0],2,1),(2,3)=>reshape(K[1,0],1,2));field)
        t=[2,3,3]
        r=IL.find_interleaving(M,N,t)
        @test r.hom_dimensions==(2,1)
        @test r.status===:interleaved
        @test IL.find_interleaving(N,M,t).status===:interleaved
        field isa CM.PrimeField && @test _il_brute(M,N,t,field.p)
    end
end

@testset "A53 contracts and distance bounds" begin
    @info "A53: contracts and distance bounds"
    M=_il_interval(0,2);N=IL.parameter_shift(M,-1//2)
    r=IL.interleaving_distance_bounds(M,N;upper=1,tolerance=1//64)
    @test r.lower<=1//2<=r.upper
    @test r.upper-r.lower<=1//64
    @test IL.check_interleaving(r.certificate).valid
    r=IL.interleaving_distance_bounds(M,N;upper=1//4)
    @test r.upper===nothing && r.lower==1//4
    @test IL.interleaving_distance_bounds(M,M;upper=1).upper==0
    @test_throws ArgumentError IL.interleaving_distance_bounds(M,N;upper=-1)
    @test_throws ArgumentError IL.interleaving_distance_bounds(M,N;upper=1,tolerance=0)
    @test_throws ArgumentError IL.interleaving_problem(M,N,-1)
    @test_throws ArgumentError IL.interleaving_problem(M,N,[1,2])
    @test_throws ArgumentError IL.find_interleaving(M,_il_interval(0,2;field=CM.F2()),0)
    @test_throws ArgumentError IL.parameter_shift(M,Inf)
    @test_throws ArgumentError IL.RectilinearModule(([1,0],),[0,1,0,1,0],Dict())
    @test_throws ArgumentError IL.RectilinearModule(([0],),[1],Dict())
    @test_throws ArgumentError IL.InterleavingBudget(max_candidates=0)
    @test_throws ArgumentError IL.interleaving_problem(M,N,1;budget=IL.InterleavingBudget(max_vertices=2))
    @test_throws ArgumentError IL.find_interleaving(_il_constant(2),_il_constant(2),0;budget=IL.InterleavingBudget(max_hom_variables=1))
    @test_throws ArgumentError IL.find_interleaving(_il_constant(2),_il_constant(2),0;budget=IL.InterleavingBudget(max_matrix_entries=1))
    Q=FF.GridPoset(([1,2],));A=MD.PModule{QQ}(Q,[1,1],Dict((1,2)=>ones(QQ,1,1));field=CM.QQField())
    @test_throws ArgumentError IL.interleaving_problem(A,A,[2,1])
    @test_throws ArgumentError IL.interleaving_problem(A,A,[2,3])
    @test_throws ArgumentError IL.interleaving_problem(A,A,[1])
    @test_throws ArgumentError IL.RectilinearModule(([0],),[1,1,1],Dict((1,2)=>ones(1,1),(2,3)=>ones(1,1)))
    Q=FF.GridPoset(([1,2],[1,2]));K=QQ
    A=MD.PModule{K}(Q,ones(Int,4),Dict(e=>ones(K,1,1) for e in FF.cover_edges(Q));field=CM.QQField())
    @test_throws ArgumentError IL.interleaving_problem(A,A,[2,2,3,4])
    Q=FF.FinitePoset(falses(0,0))
    for field in (CM.QQField(),CM.F2(),CM.F3())
        K=CM.coeff_type(field)
        A=MD.PModule{K}(Q,Int[],Dict{Tuple{Int,Int},Matrix{K}}();field)
        @test IL.find_interleaving(A,A,Int[]).status===:interleaved
    end
    @test TamerOp.find_interleaving===IL.find_interleaving
    @test TOA.interleaving_certificate===IL.interleaving_certificate
end

@testset "A53 ambient encoding adapters" begin
    @info "A53: ambient encoding adapters"
    P,H,pi=PLB.encode_fringe_boxes([PLB.BoxUpset([0,0])],[PLB.BoxDownset([2,2])])
    M=IR.pmodule_from_fringe(H)
    A=IL.ambient_module(M,pi)
    for x in (-1,0,1,2,3),y in (-1,0,1,2,3)
        @test IL.ambient_dimension(A,[x,y])==Int(0<=x<=2 && 0<=y<=2)
    end
    @test IL.find_interleaving(A,A,[1,0]).status===:interleaved
    @test IL.provenance(IL.parameter_shift(A,[1,0])).origin.translation==(1,0)
    Q=FF.GridPoset(([0,2],));B=MD.PModule{QQ}(Q,[1,0],Dict((1,2)=>zeros(QQ,0,1));field=CM.QQField())
    grid=EC.GridEncodingMap(Q,([0,2],))
    @test_throws ArgumentError IL.ambient_module(B,grid)
    A=IL.ambient_module(B,grid;extension=:zero_below)
    @test [IL.ambient_dimension(A,[x]) for x in (-10,0,1,2,10)]==[0,1,1,0,0]
    @test IL.provenance(A).origin.extension===:zero_below
    @test_throws ArgumentError IL.ambient_module(B,grid;extension=:zero_window)
    @test_throws ArgumentError IL.ambient_module(B,nothing)
    # A simultaneous birth/death wall must retain its singleton support.
    enc=TO.encode([PLB.BoxUpset([0])],[PLB.BoxDownset([0])])
    point=IL.ambient_module(enc)
    @test IL.ambient_dimension(point,[0])==1
    @test IL.ambient_dimension(point,[-1//10])==0
    @test IL.find_interleaving(enc,enc,0).status===:interleaved
    @test IL.interleaving_distance_bounds(enc,enc;upper=1).upper==0
    @test IL.ambient_dimension(IL.parameter_shift(enc,1),[-1])==1
    @test IL.find_interleaving(point,_il_constant(0),0).status===:not_interleaved
    @test IL.find_interleaving(point,_il_constant(0),1//100).status===:interleaved
    # Native lattice encoding adapter, including its upper death+1 cut.
    tau=FZ.face(1,[false])
    flange=FZ.Flange{QQ}(1,[FZ.IndFlat(tau,[0])],[FZ.IndInj(tau,[2])],ones(QQ,1,1);field=CM.QQField())
    encz=TO.encode(flange)
    lattice=IL.ambient_module(encz)
    @test IL.provenance(lattice).domain===:integer
    @test [IL.ambient_dimension(lattice,[x]) for x in -1:4]==[0,1,1,1,0,0]
    @test IL.find_interleaving(encz,encz,1).status===:interleaved
end
