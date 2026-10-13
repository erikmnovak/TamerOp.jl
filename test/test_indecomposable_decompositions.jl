# Independent Krull-Schmidt examples, idempotent enumeration and gluing checks.

function _a50_star(m)
    order = Matrix{Bool}(I,m+1,m+1)
    order[1:m,m+1] .= true
    return FF.FinitePoset(order)
end

# Left regular multiplication in (a,b)_QQ on the basis 1,i,j,ij.
function _a50_quaternion(a,b)
    X = [0 a 0 0;1 0 0 0;0 0 0 a;0 0 1 0]
    Y = [0 0 b 0;0 0 0 -b;1 0 0 0;0 -1 0 0]
    return X,Y
end

function _a50_interval_sum(Q,supports,field)
    K = CM.coeff_type(field)
    active = [findall(I -> v in I,supports) for v in 1:FF.nvertices(Q)]
    return MD.PModule{K}(Q,length.(active),Dict((u,v) =>
        K[i == j for i in active[v],j in active[u]] for (u,v) in FF.cover_edges(Q));field)
end

# The common centralizer of Ts is End of this subspace representation: coordinate
# subspaces force diag(A,B), the diagonal forces A=B, and graphs force AT=TA.
function _a50_centralizer(field,Ts;Q=_a50_star(length(Ts)+3))
    K = CM.coeff_type(field); n = size(first(Ts),1)
    E,Z = CM.eye(field,n),zeros(K,n,n)
    maps = [vcat(E,Z),vcat(Z,E),vcat(E,E),[vcat(E,K.(T)) for T in Ts]...]
    return MD.PModule{K}(Q,[fill(n,length(maps));2n],
        Dict((j,length(maps)+1) => A for (j,A) in enumerate(maps));field)
end

function _a50_conjugate(M,rng=MersenneTwister(501))
    K = CM.coeff_type(M.field)
    Cs = [CM.eye(M.field,d) for d in M.dims]
    for C in Cs, _ in 1:2size(C,1)
        size(C,1) < 2 && continue
        i,j = randperm(rng,size(C,1))[1:2]
        C[i,:] .+= K(rand(rng,-2:2)).*C[j,:]
    end
    inverses = [FL.solve_fullcolumn(M.field,C,CM.eye(M.field,size(C,1))) for C in Cs]
    return MD.PModule{K}(M.Q,copy(M.dims),Dict((u,v) => sparse(Cs[v]*M.edge_maps[u,v]*inverses[u])
        for (u,v) in FF.cover_edges(M.Q));field=M.field)
end

function _a50_reconstruction(r)
    D = TamerOp.IndecomposableDecompositions
    M,w = D.source_module(r),D.decomposition_maps(r)
    @test sum(sum(N.dims) for N in w.summands;init=0) == sum(M.dims)
    for (k,N) in enumerate(w.summands)
        @test MD.check_module_functoriality(N).valid
        @test MD.check_morphism(w.inclusions[k]).valid
        @test MD.check_morphism(w.projections[k]).valid
    end
    for v in eachindex(M.dims)
        K = CM.coeff_type(M.field)
        total = zeros(K,M.dims[v],M.dims[v])
        for i in eachindex(w.summands),j in eachindex(w.summands)
            lhs = w.projections[i].comps[v]*w.inclusions[j].comps[v]
            @test lhs == (i == j ? CM.eye(M.field,w.summands[i].dims[v]) :
                zeros(K,w.summands[i].dims[v],w.summands[j].dims[v]))
        end
        for i in eachindex(w.summands)
            total .+= w.inclusions[i].comps[v]*w.projections[i].comps[v]
        end
        @test total == CM.eye(M.field,M.dims[v])
    end
end

@testset "A50 noninterval indecomposables and repeated summands" begin
    D = TamerOp.IndecomposableDecompositions
    for field in (CM.QQField(),CM.F2(),CM.F3(),CM.Fp(5),CM.Fp(101))
        K = CM.coeff_type(field); Q = _a50_star(3)
        N = MD.PModule{K}(Q,[1,1,1,2],Dict((1,4)=>reshape(K[1,0],2,1),
            (2,4)=>reshape(K[0,1],2,1),(3,4)=>reshape(K[1,1],2,1));field)
        @test D.is_indecomposable(N)
        @test !TamerOp.IntervalDecompositions.is_interval_decomposable(N)
        M = _a50_conjugate(MD.direct_sum(MD.direct_sum(N,N),
            _a50_interval_sum(Q,[[1],[2,4]],field)))
        r = D.indecomposable_decomposition(M;witnesses=true)
        @test sort(D.summand_dimensions(r)) == sort([[1,1,1,2],[1,1,1,2],[1,0,0,0],[0,1,0,1]])
        @test !D.is_indecomposable(r)
        @test D.check_indecomposable_decomposition(r;throw=true).splitting_checked
        _a50_reconstruction(r)
        for labels in ([1],[4],[1,4],[1,2,3,4])
            @test TamerOp.generalized_rank(M;vertices=labels) ==
                sum(TamerOp.generalized_rank(S;vertices=labels) for S in D.indecomposable_summands(r))
        end
        P = _a50_star(2)
        restricted = TamerOp.ChangeOfPosets.restriction(EN.EncodingMap(P,Q,[1,2,4]),N)
        @test sort(D.summand_dimensions(D.indecomposable_decomposition(restricted))) == sort([[1,0,1],[0,1,1]])
    end
end

@testset "A50 extension fields and nonsemisimple local algebras" begin
    D = TamerOp.IndecomposableDecompositions
    # Irreducible quadratic over each ORIGINAL ground field, not its closure.
    examples = ((CM.QQField(),[0 -1;1 0]),(CM.F2(),[0 1;1 1]),
        (CM.F3(),[0 -1;1 0]),(CM.Fp(5),[0 2;1 0]))
    for (field,T) in examples
        M = _a50_centralizer(field,[T])
        r = D.indecomposable_decomposition(M;witnesses=true)
        @test D.summand_dimensions(r) == [M.dims]
        c = only(D.describe(r).certificates)
        @test c.end_dimension == c.residue_dimension == 2
        @test c.radical_dimension == 0
        _a50_reconstruction(r)
        # Repeated thick indecomposables must be separated, despite identical
        # stalk dimensions and a nontrivial residue extension.
        M2 = _a50_conjugate(MD.direct_sum(M,M))
        r2 = D.indecomposable_decomposition(M2;witnesses=true)
        @test D.summand_dimensions(r2) == [M.dims,M.dims]
        _a50_reconstruction(r2)
        # A repeated irreducible factor has nilpotents without splitting over
        # the ground field: centralizer is k[t]/(q^2), residue degree two.
        K = CM.coeff_type(field)
        C = K.(T); J = [C CM.eye(field,2);zeros(K,2,2) C]
        thick = _a50_centralizer(field,[J])
        local_result = D.indecomposable_decomposition(thick;witnesses=true)
        local_proof = only(D.describe(local_result).certificates)
        @test (local_proof.end_dimension,local_proof.radical_dimension,
            local_proof.residue_dimension,local_proof.nilpotency_index) == (4,2,2,2)
        _a50_reconstruction(local_result)
    end
    for field in (CM.QQField(),CM.F2(),CM.F3(),CM.Fp(5))
        for n in (2,3)
            J = zeros(Int,n,n)
            for i in 1:n-1
                J[i,i+1] = 1
            end
            M = _a50_centralizer(field,[J])
            r = D.indecomposable_decomposition(M;witnesses=true)
            @test D.summand_dimensions(r) == [M.dims]
            c = only(D.describe(r).certificates)
            @test c.end_dimension == n
            @test c.radical_dimension == n-1
            @test c.residue_dimension == 1
            @test c.nilpotency_index == n
            @test D.check_indecomposable_decomposition(r).valid
        end
        # Left multiplication by x,y in the exterior algebra on two generators;
        # End is the opposite local algebra, noncommutative in odd characteristic.
        X = [0 0 0 0;1 0 0 0;0 0 0 0;0 0 1 0]
        Y = [0 0 0 0;0 0 0 0;1 0 0 0;0 -1 0 0]
        M = _a50_centralizer(field,[X,Y])
        r = D.indecomposable_decomposition(M)
        c = only(D.describe(r).certificates)
        @test (c.end_dimension,c.radical_dimension,c.residue_dimension,c.nilpotency_index) == (4,3,1,3)
    end
end

# Tiny, independent oracle: enumerate component matrices directly, imposing
# naturality and idempotence without Hom, polynomial factoring or local algebra.
function _a50_has_idempotent(M)
    p = M.field.p; K = CM.coeff_type(M.field)
    offsets = cumsum([0;M.dims.^2]); n = offsets[end]
    for number in 1:p^n-1
        cs = K.(digits(number;base=p,pad=n))
        X = [reshape(cs[offsets[v]+1:offsets[v+1]],d,d) for (v,d) in enumerate(M.dims)]
        all(X[v] == CM.eye(M.field,M.dims[v]) for v in eachindex(X)) && continue
        all(A*A == A for A in X) || continue
        all(X[v]*M.edge_maps[u,v] == M.edge_maps[u,v]*X[u] for (u,v) in FF.cover_edges(M.Q)) || continue
        return true
    end
    return false
end

@testset "A50 exhaustive independent idempotent oracles" begin
    D = TamerOp.IndecomposableDecompositions
    for field in (CM.F2(),CM.F3())
        K = CM.coeff_type(field); Q = _a50_star(2)
        for a in 0:field.p^2-1,b in 0:field.p^2-1
            M = MD.PModule{K}(Q,[1,1,2],Dict((1,3)=>reshape(K.(digits(a;base=field.p,pad=2)),2,1),
                (2,3)=>reshape(K.(digits(b;base=field.p,pad=2)),2,1));field)
            r = D.indecomposable_decomposition(M;witnesses=true)
            @test D.is_indecomposable(r) == !_a50_has_idempotent(M)
            @test all(!_a50_has_idempotent(N) for N in D.indecomposable_summands(r))
            _a50_reconstruction(r)
        end
    end
end

function _a50_bipath(na,nb)
    n = na+nb+2
    order = Matrix{Bool}(I,n,n)
    for chain in ([1;collect(2:na+1);n],[1;collect(na+2:n-1);n])
        for i in eachindex(chain),j in i:length(chain)
            order[chain[i],chain[j]] = true
        end
    end
    return FF.FinitePoset(order)
end

function _a50_all_intervals(Q)
    n = FF.nvertices(Q); supports = Vector{Int}[]
    for mask in 1:2^n-1
        S = [v for v in 1:n if !iszero(mask & (1 << (v-1)))]
        all(!FF.leq(Q,a,z) || !FF.leq(Q,z,b) || z in S for a in S,b in S,z in 1:n) || continue
        reached = Set([first(S)])
        for _ in 1:n,a in S,b in S
            a in reached && (FF.leq(Q,a,b) || FF.leq(Q,b,a)) && push!(reached,b)
        end
        length(reached) == length(S) && push!(supports,S)
    end
    return supports
end

@testset "A50 bipath families and endpoint gluing" begin
    D = TamerOp.IndecomposableDecompositions
    for field in (CM.QQField(),CM.F2(),CM.F3(),CM.Fp(5)), arms in ((1,1),(2,1))
        Q = _a50_bipath(arms...)
        family = _a50_all_intervals(Q)
        @test sort(D._bipath_family(Q,D.IndecomposableDecompositionBudget())) == sort(family)
        # Every interval family, plus multiplicity two, in a mixed stalk basis.
        M = _a50_conjugate(_a50_interval_sum(Q,[family;family[1:1]],field))
        r = D.bipath_decomposition(M;witnesses=true)
        expected = [[Int(v in S) for v in 1:FF.nvertices(Q)] for S in [family;family[1:1]]]
        @test sort(D.summand_dimensions(r)) == sort(expected)
        @test D.describe(r).route == :bipath_intervals
        @test D.check_indecomposable_decomposition(r;throw=true).valid
        _a50_reconstruction(r)
        perm = reverse(1:FF.nvertices(Q))
        R = FF.FinitePoset(Bool[FF.leq(Q,perm[u],perm[v]) for u in eachindex(perm),v in eachindex(perm)])
        K = CM.coeff_type(field)
        relabeled = MD.PModule{K}(R,M.dims[perm],Dict((u,v) => Matrix(M.edge_maps[perm[u],perm[v]])
            for (u,v) in FF.cover_edges(R));field)
        @test sort(D.summand_dimensions(D.bipath_decomposition(relabeled))) == sort([d[perm] for d in expected])
        # Equal arm barcodes: two surjections from bottom k^2 with either equal
        # or transverse kernels, zero top. Their glued decompositions differ.
        K = CM.coeff_type(field); P = _a50_bipath(1,1)
        common = Dict((1,2)=>reshape(K[1,0],1,2),(2,4)=>zeros(K,0,1),(3,4)=>zeros(K,0,1))
        A = MD.PModule{K}(P,[2,1,1,0],merge(common,Dict((1,3)=>reshape(K[1,0],1,2)));field)
        B = MD.PModule{K}(P,[2,1,1,0],merge(common,Dict((1,3)=>reshape(K[0,1],1,2)));field)
        @test sort(D.summand_dimensions(D.bipath_decomposition(A))) == sort([[1,1,1,0],[1,0,0,0]])
        @test sort(D.summand_dimensions(D.bipath_decomposition(B))) == sort([[1,1,0,0],[1,0,1,0]])
    end
end

@testset "A50 contracts workflows and honest bounded failures" begin
    D = TamerOp.IndecomposableDecompositions
    Q = FF.ProductOfChainsPoset((2,2)); field = CM.QQField()
    M = _a50_interval_sum(Q,[[1,2,3,4],[4]],field)
    enc = RES.EncodingResult(Q,M,EC.GridEncodingMap(Q,([0,1],[0,1])))
    cache = CM.SessionCache()
    r = TamerOp.indecomposable_decomposition(enc;cache,witnesses=true)
    @test D.source_module(r) === M
    @test D.source_poset(r) === Q
    @test TamerOp.summand_dimensions(r) == D.summand_dimensions(D.indecomposable_decomposition(M))
    @test !TamerOp.is_indecomposable(enc;cache)
    @test TamerOp.describe(r) == TamerOp.indecomposable_decomposition_summary(r)
    @test TamerOp.provenance(r).ambient_identification == :not_asserted
    @test TamerOp.provenance(r).stability == :not_asserted
    @test occursin("IndecomposableDecompositionResult",sprint(show,r))
    @test TOA.IndecomposableDecompositionResult === D.IndecomposableDecompositionResult
    @test TOA.decomposition_maps(r) === D.decomposition_maps(r)
    @test sort(D.summand_dimensions(TamerOp.bipath_decomposition(enc;cache))) == sort(r.dimensions)
    ds = D.summand_dimensions(r);ds[1][1] = 99
    @test D.summand_dimensions(r) != ds
    s = D.indecomposable_decomposition(M)
    @test_throws ArgumentError D.decomposition_maps(s)
    @test_throws ArgumentError D.indecomposable_summands(s)
    @test_throws ArgumentError TamerOp.summand_collection(s)
    @test D.check_indecomposable_decomposition(s).valid
    for val in (0,-1,true,1.5,big(typemax(Int))+1)
        @test_throws ArgumentError D.IndecomposableDecompositionBudget(max_search_steps=val)
    end
    for budget in (D.IndecomposableDecompositionBudget(max_vertices=1),
            D.IndecomposableDecompositionBudget(max_dimension=1),
            D.IndecomposableDecompositionBudget(max_matrix_entries=1),
            D.IndecomposableDecompositionBudget(max_endomorphism_dimension=1))
        @test_throws ArgumentError D.indecomposable_decomposition(M;budget)
    end
    @test_throws ArgumentError D.bipath_decomposition(M;budget=D.IndecomposableDecompositionBudget(max_bipath_intervals=1))
    @test_throws ArgumentError D.bipath_decomposition(_a50_interval_sum(chain_poset(4),[[1]],field))
    @test_throws ArgumentError D.bipath_decomposition(_a50_interval_sum(_a50_star(3),[[1]],field))
    @test_throws ArgumentError D.indecomposable_decomposition(_a50_interval_sum(Q,[[1]],CM.RealField(Float64)))
    for P in (Q,FF.FinitePoset(falses(0,0)))
        z = MD.zero_pmodule(P;field)
        rz = D.indecomposable_decomposition(z;witnesses=true)
        @test isempty(D.summand_dimensions(rz))
        @test !D.is_indecomposable(rz)
        @test D.check_indecomposable_decomposition(rz).valid
        _a50_reconstruction(rz)
    end
    # Hamilton's rational quaternion division algebra: the native field-quotient
    # certificate deliberately cannot assert completion for this example.
    X = [0 -1 0 0;1 0 0 0;0 0 0 -1;0 0 1 0]
    Y = [0 0 -1 0;0 0 0 1;1 0 0 0;0 -1 0 0]
    H = _a50_centralizer(field,[X,Y])
    @test_throws D.IndecomposableSearchExhausted D.indecomposable_decomposition(H;
        budget=D.IndecomposableDecompositionBudget(max_search_steps=12))
    @test occursin("No complete decomposition",sprint(showerror,D.IndecomposableSearchExhausted(12,H.dims,true)))
    point = _a50_interval_sum(chain_poset(1),[[1],[1]],CM.F2())
    @test_throws D.IndecomposableSearchExhausted D.indecomposable_decomposition(point;
        budget=D.IndecomposableDecompositionBudget(max_search_steps=1))
    # Mutated witnesses, summaries and nonfunctorial input are rejected.
    bad = deepcopy(r)
    vertex = findfirst(!iszero,bad.dimensions[1])
    bad.maps.inclusions[1].comps[vertex] .= 0
    @test !D.check_indecomposable_decomposition(bad).valid
    @test_throws ArgumentError D.check_indecomposable_decomposition(bad;throw=true)
    bad = deepcopy(r);bad.dimensions[1][1] = 99
    @test !D.check_indecomposable_decomposition(bad).valid
    invalid = _a50_interval_sum(Q,[[1,2,3,4]],field)
    invalid.edge_maps[1,2][1,1] = 2
    @test_throws ArgumentError D.indecomposable_decomposition(invalid)
    v = D.check_indecomposable_decomposition(r)
    @test v == deepcopy(v)
    @test isequal(v,deepcopy(v)) && hash(v) == hash(deepcopy(v))
    @test D.describe(v) == D.indecomposable_decomposition_summary(v)
    @test occursin("valid=true",sprint(show,v))
end

@testset "A50 existing summand inspector integration" begin
    D = TamerOp.IndecomposableDecompositions; V = TamerOp.Visualization
    field = CM.QQField(); Q = _a50_star(4)
    A = _a50_centralizer(field,[[0 2;1 0]];Q)
    B = _a50_centralizer(field,[[0 3;1 0]];Q)
    M = _a50_conjugate(MD.direct_sum(A,B))
    r = D.indecomposable_decomposition(M;witnesses=true)
    @test D.summand_dimensions(r) == [A.dims,B.dims]
    pieces = D.indecomposable_summands(r)
    # Same dimension vectors do not imply isomorphic indecomposables.
    @test TamerOp.DerivedFunctors.dim(TamerOp.DerivedFunctors.Hom(pieces[1],pieces[2])) == 0
    collection = TamerOp.summand_collection(r)
    @test V.check_summand_collection(collection).valid
    @test V.available_visuals(r) == (:summand_gallery,)
    spec = V.visual_spec(r)
    @test spec.metadata.certification == :checked_splitting
    session = V.summand_session(r)
    @test V.inspection_snapshot(session).metadata.certification == :checked_splitting
end

@testset "A50 rational backend contracts" begin
    D = TamerOp.IndecomposableDecompositions
    X,Y = _a50_quaternion(-1,-1)
    M = _a50_centralizer(CM.QQField(),[X,Y])
    @test_throws ArgumentError D.indecomposable_decomposition(M;backend=:auto)
    @test_throws ArgumentError D.indecomposable_decomposition(M;backend=:unknown)
    @test_throws ArgumentError D.indecomposable_decomposition(
        _a50_centralizer(CM.F3(),[X,Y]);backend=:hecke)
    if Base.get_extension(TamerOp,:TamerOpHeckeExt) === nothing
        err = try
            D.indecomposable_decomposition(M;backend=:hecke)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("using Hecke",sprint(showerror,err))
    end
    r = D.indecomposable_decomposition(_a50_interval_sum(chain_poset(1),[[1]],CM.QQField()))
    @test D.describe(r).backend == :native
    @test D.provenance(r).backend == :native
    bad = D.IndecomposableDecompositionResult(r.source,r.dimensions,r.certificates,
        r.route,:unknown,r.statistics,r.maps)
    @test !D.check_indecomposable_decomposition(bad).valid
end

@testset "A50 Hecke rational division quotients" begin
    if Base.find_package("Hecke") === nothing
        @test_skip false
    else
        @eval import Hecke
        @test Base.get_extension(TamerOp,:TamerOpHeckeExt) !== nothing
        D = TamerOp.IndecomposableDecompositions
        field = CM.QQField()
        E2,E4 = Matrix{Int}(I,2,2),Matrix{Int}(I,4,4)
        X,Y = _a50_quaternion(-1,-1)
        @test X*X == Y*Y == -E4
        @test X*Y == -Y*X
        H = _a50_centralizer(field,[X,Y])
        # Hamilton's norm is a sum of four rational squares, hence anisotropic.
        # These two presentations have a division End algebra of dimension four.
        for M in (H,_a50_conjugate(H))
            r = D.indecomposable_decomposition(M;backend=:hecke,witnesses=true)
            c = only(D.describe(r).certificates)
            @test (c.kind,c.end_dimension,c.radical_dimension,c.residue_dimension,
                c.nilpotency_index,c.center_degree,c.schur_index) ==
                (:rational_division_quotient,4,0,4,1,1,2)
            @test D.summand_dimensions(r) == [H.dims]
            @test D.is_indecomposable(r)
            @test D.check_indecomposable_decomposition(r;throw=true).splitting_checked
            @test D.describe(r).backend == D.provenance(r).backend == :hecke
            _a50_reconstruction(r)
        end
        # (2,3) splits over R but is division over QQ. A zero of its norm gives
        # x0^2+x1^2=0 mod 3, then x2^2+x3^2=0 mod 3: infinite descent.
        U,V = _a50_quaternion(2,3)
        indefinite = _a50_centralizer(field,[U,V];Q=H.Q)
        ri = D.indecomposable_decomposition(indefinite;backend=:hecke)
        @test D.is_indecomposable(ri)
        @test only(D.describe(ri).certificates).schur_index == 2

        # A degree-three cyclic division algebra, not quaternion-specific:
        # L=QQ(zeta_7+zeta_7^-1), sigma(t)=t^2-2, u^3=2, u*t=sigma(t)*u.
        # The defining cubic is irreducible modulo 2, so 2 is inert and its
        # valuation is not divisible by 3; 2 cannot be a norm from L.
        C = [0 0 1;1 0 2;0 1 -1]
        E3,Z3 = Matrix{Int}(I,3,3),zeros(Int,3,3)
        sigma = C^2-2*E3
        G = hcat(E3[:,1],sigma[:,1],(sigma^2)[:,1])
        U = [Z3 Z3 2G;G Z3 Z3;Z3 G Z3]
        V = kron(Matrix{Int}(I,3,3),C)
        @test C^3+C^2-2C-E3 == Z3
        @test G^3 == E3
        @test U^3 == 2Matrix{Int}(I,9,9)
        @test U*V == (V^2-2Matrix{Int}(I,9,9))*U
        cubic = _a50_centralizer(field,[U,V])
        rc = D.indecomposable_decomposition(cubic;backend=:hecke,witnesses=true)
        c = only(D.describe(rc).certificates)
        @test (c.end_dimension,c.residue_dimension,c.center_degree,c.schur_index) == (9,9,1,3)
        @test D.summand_dimensions(rc) == [cubic.dims]
        _a50_reconstruction(rc)

        # A nilpotent radical with the SAME noncommutative division quotient:
        # End is H[t]/(t^2), not a semisimple algebra itself.
        thick = _a50_centralizer(field,[kron(E2,X),kron(E2,Y),kron([0 1;0 0],E4)])
        rt = D.indecomposable_decomposition(thick;backend=:hecke,witnesses=true)
        c = only(D.describe(rt).certificates)
        @test (c.end_dimension,c.radical_dimension,c.residue_dimension,
            c.nilpotency_index,c.center_degree,c.schur_index) == (8,4,4,2,1,2)
        @test D.check_indecomposable_decomposition(rt).valid
        _a50_reconstruction(rt)

        # A nontrivial center: H tensor QQ(sqrt(2)) remains division at each real
        # embedding. H tensor QQ(i) instead splits as Mat_2(QQ(i)).
        for (a,expected_count,center_degree,index) in ((2,1,2,2),(-1,2,2,1))
            M = _a50_centralizer(field,[kron(E2,X),kron(E2,Y),kron([0 a;1 0],E4)])
            r = D.indecomposable_decomposition(M;backend=:hecke,witnesses=true)
            @test length(D.summand_dimensions(r)) == expected_count
            @test D.summand_dimensions(r) == [div.(M.dims,expected_count) for _ in 1:expected_count]
            @test all(c -> c.center_degree == center_degree && c.schur_index == index,
                D.describe(r).certificates)
            @test D.check_indecomposable_decomposition(r).valid
            _a50_reconstruction(r)
        end

        # A simple matrix algebra is not a division algebra. Exercise the adapter
        # directly as well as the public solver (which may split before calling it).
        S,T = _a50_quaternion(1,1)
        split = _a50_centralizer(field,[S,T])
        b = D.IndecomposableDecompositionBudget()
        for M in (split,MD.direct_sum(H,H),MD.direct_sum(H,indefinite))
            A = D._endomorphisms(M,b)
            L = D._multiplication_table(M,A,b)
            @test D._rational_division_certificate(L,b) === nothing
            r = D.indecomposable_decomposition(_a50_conjugate(M);backend=:hecke,witnesses=true)
            @test length(D.summand_dimensions(r)) == 2
            @test D.check_indecomposable_decomposition(r).valid
            _a50_reconstruction(r)
        end
        # Hecke conversion checks its own budget before allocation.
        A = D._endomorphisms(H,b)
        L = D._multiplication_table(H,A,b)
        @test_throws ArgumentError D._rational_division_certificate(L,
            D.IndecomposableDecompositionBudget(max_matrix_entries=100))
        @test_throws D.IndecomposableSearchExhausted D.indecomposable_decomposition(H;
            backend=:hecke,budget=D.IndecomposableDecompositionBudget(max_search_steps=3))
        r = D.indecomposable_decomposition(H;backend=:hecke,witnesses=true,
            budget=D.IndecomposableDecompositionBudget(max_search_steps=4))
        @test D.describe(r).statistics.search_steps == 4
        bad = deepcopy(r);bad.maps.inclusions[1].comps[1] .= 0
        @test !D.check_indecomposable_decomposition(bad).valid
        c = r.certificates[1]
        bad = deepcopy(r)
        bad.certificates[1] = D._Certificate(c.kind,c.end_dimension,c.radical_dimension,
            c.residue_dimension,c.nilpotency_index,c.center_degree,1)
        @test !D.check_indecomposable_decomposition(bad).valid
        # Existing consumers revalidate with the stored backend.
        @test TamerOp.Visualization.check_summand_collection(TamerOp.summand_collection(r)).valid
        enc = RES.EncodingResult(H.Q,H,EN.EncodingMap(H.Q,H.Q,collect(1:FF.nvertices(H.Q))))
        rw = TamerOp.indecomposable_decomposition(enc;backend=:hecke,cache=CM.SessionCache())
        @test TamerOp.is_indecomposable(enc;backend=:hecke)
        @test D.describe(rw).backend == :hecke
        @test D.summand_dimensions(rw) == [H.dims]
        @test D.check_indecomposable_decomposition(rw).valid
    end
end
