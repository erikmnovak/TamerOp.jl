# Independent interval-sum, chain-rank, and exhaustive finite-field split oracles.
function _a46_sum(P, supports, field; sparse_maps=false)
    K = CM.coeff_type(field)
    active = [findall(I -> v in I,supports) for v in 1:FF.nvertices(P)]
    maps = Dict((u,v) => K[i == j for i in active[v], j in active[u]]
                for (u,v) in FF.cover_edges(P))
    sparse_maps && return MD.PModule{K}(P,length.(active),Dict(e=>sparse(A) for (e,A) in maps);field)
    return MD.PModule{K}(P,length.(active),maps;field)
end

# Enumerate all subsets, test convexity by triples and connectivity by closure.
# Deliberately independent of production endpoint-antichain generation.
function _a46_intervals(P)
    n = FF.nvertices(P)
    result = Vector{Int}[]
    for mask in 1:2^n-1
        I = [v for v in 1:n if !iszero(mask & (1 << (v-1)))]
        all(!FF.leq(P,a,z) || !FF.leq(P,z,b) || z in I for a in I,b in I,z in 1:n) || continue
        connected = Set([first(I)])
        for _ in 1:n, a in I,b in I
            a in connected && (FF.leq(P,a,b) || FF.leq(P,b,a)) && push!(connected,b)
        end
        length(connected) == length(I) && push!(result,I)
    end
    return result
end

# Enumerate natural maps over a tiny prime field without Hom or rank routines.
function _a46_maps(A,B,p)
    K = CM.coeff_type(A.field)
    lengths = A.dims .* B.dims
    offsets = cumsum([0;lengths])
    out = Vector{Matrix{K}}[]
    for number in 0:p^offsets[end]-1
        coeffs = K.(digits(number;base=p,pad=offsets[end]))
        comps = [reshape(coeffs[offsets[v]+1:offsets[v+1]],B.dims[v],A.dims[v]) for v in eachindex(A.dims)]
        all(B.edge_maps[u,v]*comps[u] == comps[v]*A.edge_maps[u,v] for (u,v) in FF.cover_edges(A.Q)) && push!(out,comps)
    end
    return out
end

function _a46_has_split(M,I,p)
    A = _a46_sum(M.Q,[I],M.field)
    return any(all(g[v]*f[v] == CM.eye(M.field,A.dims[v]) for v in eachindex(M.dims))
        for f in _a46_maps(A,M,p),g in _a46_maps(M,A,p))
end

function _a46_check_witness(r)
    D = TamerOp.IntervalDecompositions
    w = D.decomposition_maps(r)
    M = D.source_module(r)
    @test MD.check_module_functoriality(w.residual).valid
    for f in (w.inclusion,w.projection,w.residual_inclusion,w.residual_projection)
        @test MD.check_morphism(f).valid
    end
    for v in eachindex(M.dims)
        i,p,j,q = w.inclusion.comps[v],w.projection.comps[v],w.residual_inclusion.comps[v],w.residual_projection.comps[v]
        @test p*i == CM.eye(M.field,w.summand.dims[v])
        @test q*j == CM.eye(M.field,w.residual.dims[v])
        @test all(iszero,p*j)
        @test all(iszero,q*i)
        @test i*p+j*q == CM.eye(M.field,M.dims[v])
    end
end

@testset "A46 exact multiplicities and exhaustive support oracles" begin
    D = TamerOp.IntervalDecompositions
    posets = (chain_poset(4), FF.ProductOfChainsPoset((2,2)),
        FF.FinitePoset(Bool[1 0 0 1;0 1 0 1;0 0 1 1;0 0 0 1]),
        FF.FinitePoset(Bool[1 0 0;1 1 1;1 0 1]),
        FF.FinitePoset(Matrix{Bool}(LinearAlgebra.I,3,3)))
    for field in (CM.QQField(),CM.F2(),CM.F3(),CM.Fp(5),CM.Fp(101)), P in posets
        family = _a46_intervals(P)
        supports = vcat(family[1:2:end],family[1:1])
        M = _a46_sum(P,supports,field;sparse_maps=true)
        # Change every stalk basis by an invertible triangular matrix.
        K = CM.coeff_type(field)
        C = [K[i == j || j == i+1 for i in 1:d,j in 1:d] for d in M.dims]
        Ci = [FL.solve_fullcolumn(field,A,CM.eye(field,size(A,1))) for A in C]
        N = MD.PModule{K}(P,copy(M.dims),Dict((u,v)=>C[v]*M.edge_maps[u,v]*Ci[u] for (u,v) in FF.cover_edges(P));field)
        candidates = D.interval_candidates(N)
        for labels in family
            expected = count(==(labels),supports)
            @test D.interval_multiplicity(N;vertices=labels) == expected
            expected > 0 && @test labels in candidates
        end
        for X in (M,N)
            r = D.interval_decomposition(X;witnesses=true)
            @test D.is_interval_decomposable(r) === true
            @test Dict(Tuple(I)=>m for (I,m) in zip(D.interval_family(r),D.interval_multiplicities(r))) ==
                Dict(Tuple(I)=>count(==(I),supports) for I in unique(supports))
            @test D.check_interval_decomposition(r;throw=true).splitting_checked
            _a46_check_witness(r)
        end
    end
end

@testset "A46 independent exhaustive finite-field splitting" begin
    D = TamerOp.IntervalDecompositions
    # All four maps F2^2 -> F2 and every one-dimensional source on a fork.
    P = FF.FinitePoset(Bool[1 0 1;0 1 1;0 0 1])
    field = CM.F2();K = CM.coeff_type(field)
    for a in 0:3,b in 0:3
        M = MD.PModule{K}(P,[1,1,2],Dict((1,3)=>reshape(K.(digits(a;base=2,pad=2)),2,1),
            (2,3)=>reshape(K.(digits(b;base=2,pad=2)),2,1));field)
        for I in _a46_intervals(P)
            @test (D.interval_multiplicity(M;vertices=I)>0) == _a46_has_split(M,I,2)
        end
        @test D.is_interval_decomposable(M) === true # type A3
    end
    # Three distinct lines in a 2D sink form a non-interval indecomposable.
    P = FF.FinitePoset(Bool[1 0 0 1;0 1 0 1;0 0 1 1;0 0 0 1])
    for field in (CM.QQField(),CM.F2(),CM.F3(),CM.Fp(101))
        K = CM.coeff_type(field)
        M = MD.PModule{K}(P,[1,1,1,2],Dict((1,4)=>reshape(K[1,0],2,1),
            (2,4)=>reshape(K[0,1],2,1),(3,4)=>reshape(K[1,1],2,1));field)
        @test !isempty(D.interval_candidates(M)) # endpoint tests are necessary, not sufficient
        r = D.interval_decomposition(M;witnesses=true)
        @test D.is_interval_decomposable(r) === false
        @test D.residual_dimensions(r) == M.dims
        @test D.check_interval_decomposition(r).valid
        _a46_check_witness(r)
        mixed = MD.direct_sum(M,_a46_sum(P,[[1,4],[1,4],[4]],field))
        r = D.interval_decomposition(mixed;witnesses=true)
        @test D.interval_multiplicities(r) == [2,1]
        @test D.residual_dimensions(r) == M.dims
        @test D.is_interval_decomposable(r) === false
        @test D.check_interval_decomposition(r).valid
        _a46_check_witness(r)
        if field == CM.F2()
            for I in _a46_intervals(P)
                @test !_a46_has_split(M,I,2)
            end
        end
    end
end

@testset "A46 chain rank differences, boundary effects and budgets" begin
    D = TamerOp.IntervalDecompositions
    rng = MersenneTwister(4612)
    for field in (CM.QQField(),CM.F2(),CM.F3(),CM.Fp(101))
        K = CM.coeff_type(field);P = chain_poset(5)
        for trial in 1:5
            dims = rand(rng,1:3,5)
            M = MD.PModule{K}(P,dims,Dict((u,v)=>K.(rand(rng,-2:2,dims[v],dims[u])) for (u,v) in FF.cover_edges(P));field)
            rk(a,b) = a<1 || b>5 ? 0 : FL.rank(field,MD.map_leq(M,a,b))
            for a in 1:5,b in a:5
                @test D.interval_multiplicity(M;vertices=a:b) == rk(a,b)-rk(a-1,b)-rk(a,b+1)+rk(a-1,b+1)
            end
            @test D.is_interval_decomposable(M) === true
        end
    end
    P = chain_poset(3);M = _a46_sum(P,[[1,2,3]],CM.QQField())
    @test TamerOp.generalized_rank(M;vertices=[2]) == 1
    @test TamerOp.interval_multiplicity(M;vertices=[2]) == 0
    for I in ([],[1,1],[1,2,3,4],[0],[4],[true],[1.0],[1,3],1)
        @test_throws ArgumentError D.interval_multiplicity(M;vertices=I)
    end
    @test_throws ArgumentError D.interval_decomposition(M;family=:unknown)
    @test_throws ArgumentError D.interval_decomposition(M;family=[[1],[1]])
    @test_throws ArgumentError D.interval_decomposition(M;budget=D.IntervalDecompositionBudget(max_vertices=1))
    @test_throws ArgumentError D.interval_decomposition(M;budget=D.IntervalDecompositionBudget(max_dimension=1))
    @test_throws ArgumentError D.interval_multiplicity(M;vertices=1:3,budget=D.IntervalDecompositionBudget(max_matrix_entries=1))
    for n in (0,-1,true,1.5)
        @test_throws ArgumentError D.IntervalDecompositionBudget(max_candidates=n)
    end
    anti = FF.FinitePoset(Matrix{Bool}(LinearAlgebra.I,5,5))
    N = _a46_sum(anti,[[i] for i in 1:5],CM.F2())
    @test_throws ArgumentError D.interval_candidates(N;budget=D.IntervalDecompositionBudget(max_antichains=2))
    @test_throws ArgumentError D.interval_candidates(N;budget=D.IntervalDecompositionBudget(max_candidates=2))
    @test_throws ArgumentError D.interval_decomposition(N;family=[[1],[2]],budget=D.IntervalDecompositionBudget(max_candidates=1))
    @test_throws ArgumentError D.interval_decomposition(_a46_sum(P,[[1,2]],CM.RealField(Float64)))
    point = _a46_sum(chain_poset(1),[[1],[1],[1]],CM.F2())
    @test_throws ArgumentError D.interval_decomposition(point;family=[],
        budget=D.IntervalDecompositionBudget(max_matrix_entries=4))
    # A large constant chain has one top/socle endpoint pair, not 2^n subsets.
    L = _a46_sum(chain_poset(80),[collect(1:80)],CM.F2())
    r = D.interval_decomposition(L;budget=D.IntervalDecompositionBudget(max_candidates=1,max_antichains=1))
    @test r.statistics == (scanned=1,candidates=1,tested=1)
    @test D.interval_multiplicities(r) == [1]
end

@testset "A46 thin-cycle field-dependent obstruction" begin
    D = TamerOp.IntervalDecompositions
    P = FF.FinitePoset(Bool[1 0 1 1;0 1 1 1;0 0 1 0;0 0 0 1])
    # Four nonzero edge scalars on a bipartite cycle. Thin connected support
    # forces any nonzero summand to use every vertex. Independent consistency
    # of basis rescalings is a*d == b*c, including reduction in characteristic 2.
    for field in (CM.QQField(),CM.F2(),CM.F3(),CM.Fp(5),CM.Fp(101))
        K = CM.coeff_type(field)
        for (a,b,c,d) in ((1,1,1,1),(1,1,1,-1),(1,-1,1,-1),(-1,1,-1,1))
            M = MD.PModule{K}(P,ones(Int,4),Dict((1,3)=>fill(K(a),1,1),
                (1,4)=>fill(K(b),1,1),(2,3)=>fill(K(c),1,1),(2,4)=>fill(K(d),1,1));field)
            expected = K(a)*K(d) == K(b)*K(c)
            @test D.interval_multiplicity(M;vertices=1:4) == Int(expected)
            for I in _a46_intervals(P)
                length(I)==4 && continue
                @test D.interval_multiplicity(M;vertices=I) == 0
            end
            r = D.interval_decomposition(M;witnesses=true)
            @test D.is_interval_decomposable(r) === expected
            @test D.check_interval_decomposition(r).valid
            _a46_check_witness(r)
        end
    end
end

@testset "A46 scope, witnesses, validation and encoding workflow" begin
    D = TamerOp.IntervalDecompositions
    P = FF.ProductOfChainsPoset((2,2));field=CM.QQField()
    M = _a46_sum(P,[[1,2,3,4],[2,4],[4],[4]],field)
    enc = RES.EncodingResult(P,M,EC.GridEncodingMap(P,([0,1],[0,1])))
    cache = CM.SessionCache()
    @test TamerOp.interval_multiplicity(enc;vertices=[4],cache) == 2
    r = TamerOp.interval_decomposition(enc;family=[[4]],cache)
    @test TamerOp.is_interval_decomposable(r) === missing
    @test TamerOp.is_interval_decomposable(enc;cache) === true
    @test D.residual_dimensions(r) == [1,2,1,2]
    @test_throws ArgumentError D.decomposition_maps(r)
    @test D.check_interval_decomposition(r).valid
    @test D.source_module(r) === M
    @test D.source_poset(r) === P
    @test isequal(TamerOp.describe(r),TamerOp.interval_decomposition_summary(r))
    @test TamerOp.provenance(r).ambient_identification === :not_asserted
    @test occursin("IntervalDecompositionResult",sprint(show,r))
    supports = D.interval_family(r);empty!(supports[1])
    counts = D.interval_multiplicities(r);counts[1]=99
    residual = D.residual_dimensions(r);residual[1]=99
    @test D.interval_family(r) == [[4]]
    @test D.interval_multiplicities(r) == [2]
    @test D.residual_dimensions(r) == [1,2,1,2]
    for family in ([],[[1]],[[4]],:all)
        s = D.interval_decomposition(M;family,witnesses=true)
        @test D.check_interval_decomposition(s).valid
        _a46_check_witness(s)
    end
    for P in (chain_poset(3),FF.FinitePoset(falses(0,0)))
        z = MD.zero_pmodule(P;field)
        r = D.interval_decomposition(z;witnesses=true)
        @test D.is_interval_decomposable(r) === true
        @test D.check_interval_decomposition(r).valid
        _a46_check_witness(r)
    end
    r = D.interval_decomposition(M;witnesses=true)
    for corrupt in (:counts,:support,:residual,:map,:shape,:blocks)
        bad = deepcopy(r)
        if corrupt === :counts; bad.multiplicities[1]+=1
        elseif corrupt === :support; push!(bad.supports[1],99)
        elseif corrupt === :residual; bad.residual_dims[1]+=1
        elseif corrupt === :map; bad.maps.inclusion.comps[1][1,1]+=1
        elseif corrupt === :shape; pop!(bad.maps.projection.comps)
        else; bad.maps.summand.edge_maps[1,2][1,1]=0
        end
        @test !D.check_interval_decomposition(bad).valid
        @test_throws ArgumentError D.check_interval_decomposition(bad;throw=true)
    end
    validation = D.check_interval_decomposition(r)
    @test validation == deepcopy(validation)
    @test isequal(validation,deepcopy(validation))
    @test hash(validation) == hash(deepcopy(validation))
    @test :valid in propertynames(validation)
    @test occursin("valid=true",sprint(show,validation))
    @test D.describe(validation) == D.interval_decomposition_summary(validation)
    # Shape-correct maps on a diamond need not define a functor.
    broken = _a46_sum(M.Q,[[1,2,3,4]],field)
    broken.edge_maps[1,2][1,1]=2
    @test_throws Exception D.interval_multiplicity(broken;vertices=[4])
    @test TOA.IntervalDecompositionResult === D.IntervalDecompositionResult
    @test TOA.interval_family(r) == D.interval_family(r)
end
