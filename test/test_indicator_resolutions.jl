using Test

using Test
using LinearAlgebra

const FL = PosetModules.FieldLinAlg

@testset "IndicatorResolutions internal invariants + Ext on A2" begin
    # Internal PModule should match fiber_dimension from the fringe.
    P = chain_poset(3)
    field = CM.QQField()
    K = CM.coeff_type(field)
    M = one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 2); scalar=one(K), field=field)
    PMM = IR.pmodule_from_fringe(M)
    for q in 1:P.n
        @test PMM.dims[q] == FF.fiber_dimension(M, q)
    end

    # Projective cover should be surjective on each vertex (full row rank).
    F0, pi0, _ = IR.projective_cover(PMM)
    for q in 1:P.n
        @test FL.rank(field, pi0.comps[q]) == PMM.dims[q]
    end

    # Kernel inclusion iota: K -> F0 should be injective and satisfy pi0 * iota = 0.
    K, iota = PM.kernel_with_inclusion(pi0)
    for q in 1:P.n
        @test FL.rank(field, iota.comps[q]) == K.dims[q]
        Z = pi0.comps[q] * iota.comps[q]
        @test Z == CM.zeros(field, size(Z,1), size(Z,2))
    end

    # Now test Ext dimensions on the A2 chain: 1 < 2
    _, S1, S2 = simple_modules_chain2()

    ext12 = DF.ext_dimensions_via_indicator_resolutions(S1, S2; maxlen=3)
    ext21 = DF.ext_dimensions_via_indicator_resolutions(S2, S1; maxlen=3)
    ext11 = DF.ext_dimensions_via_indicator_resolutions(S1, S1; maxlen=3)
    ext22 = DF.ext_dimensions_via_indicator_resolutions(S2, S2; maxlen=3)

    # Known quiver A2 facts:
    # Hom(S1,S2)=0, Ext^1(S1,S2)=1
    # Hom(S2,S1)=0, Ext^1(S2,S1)=0
    # Endomorphisms: Hom(Si,Si)=1, Ext^1(Si,Si)=0
    @test get(ext12, 0, 0) == 0
    @test get(ext12, 1, 0) == 1

    @test get(ext21, 0, 0) == 0
    @test get(ext21, 1, 0) == 0

    @test get(ext11, 0, 0) == 1
    @test get(ext11, 1, 0) == 0

    @test get(ext22, 0, 0) == 1
    @test get(ext22, 1, 0) == 0

    # Ext^0 should agree with FiniteFringe.hom_dimension on these simple cases.
    @test get(ext12, 0, 0) == FF.hom_dimension(S1, S2)
    @test get(ext21, 0, 0) == FF.hom_dimension(S2, S1)
    @test get(ext11, 0, 0) == FF.hom_dimension(S1, S1)
    @test get(ext22, 0, 0) == FF.hom_dimension(S2, S2)
end

@testset "IndicatorResolutions dense-id assembly parity + budgets" begin
    # Non-chain shape exercises the active-generator edge assembly logic.
    P = diamond_poset()
    field = CM.QQField()
    K = CM.coeff_type(field)

    U = [FF.principal_upset(P, 2), FF.principal_upset(P, 3)]
    D = [FF.principal_downset(P, 4)]
    Phi = spzeros(K, 1, 2)
    Phi[1, 1] = CM.coerce(field, 1)
    Phi[1, 2] = CM.coerce(field, 1)
    H = FF.FringeModule{K}(P, U, D, Phi; field=field)
    M = IR.pmodule_from_fringe(H)

    F0s, pi0s, _ = IR.projective_cover(M; threads=false)
    E0s, iotas, _ = IR._injective_hull(M; threads=false)

    if Threads.nthreads() > 1
        F0t, pi0t, _ = IR.projective_cover(M; threads=true)
        @test F0t.dims == F0s.dims
        @test pi0t.comps == pi0s.comps
        for (u, v) in FF.cover_edges(P)
            @test F0t.edge_maps[u, v] == F0s.edge_maps[u, v]
        end

        E0t, iotat, _ = IR._injective_hull(M; threads=true)
        @test E0t.dims == E0s.dims
        @test iotat.comps == iotas.comps
        for (u, v) in FF.cover_edges(P)
            @test E0t.edge_maps[u, v] == E0s.edge_maps[u, v]
        end
    end

    # Allocation guards: warm then measure on fixed fixture.
    IR.projective_cover(M; threads=false)
    alloc_proj_cover = @allocated IR.projective_cover(M; threads=false)
    @test alloc_proj_cover < 25_000_000

    IR._injective_hull(M; threads=false)
    alloc_inj_hull = @allocated IR._injective_hull(M; threads=false)
    @test alloc_inj_hull < 25_000_000
end

@testset "Cover-edge maps are label-consistent on non-chain posets" begin
    # Poset with relations: 1<3<4 and 2<4 (2 incomparable with 3)
    leq = falses(4,4)
    for i in 1:4
        leq[i,i] = true
    end
    leq[1,3] = true
    leq[3,4] = true
    leq[1,4] = true
    leq[2,4] = true
    P = FF.FinitePoset(leq)

    # A tiny fringe module that typically forces generators at vertices 2 and 3.
    U = [FF.principal_upset(P, 2), FF.principal_upset(P, 3)]
    D = [FF.principal_downset(P, 4)]
    field = CM.QQField()
    K = CM.coeff_type(field)
    Phi = spzeros(K, 1, 2)
    Phi[1,1] = CM.coerce(field, 1)
    Phi[1,2] = CM.coerce(field, 1)
    H = FF.FringeModule{K}(P, U, D, Phi; field=field)

    M = IR.pmodule_from_fringe(H)

    # Projective cover must be a P-module morphism.
    F0, pi0, _ = IR.projective_cover(M)
    C = FF.cover_edges(P)
    for (u, v) in C
        lhs = M.edge_maps[u, v] * pi0.comps[u]
        rhs = pi0.comps[v] * F0.edge_maps[u, v]
        @test lhs == rhs
    end

    # Injective hull inclusion must be a P-module morphism.
    E, iota, _ = IR._injective_hull(M)
    for (u, v) in C
        lhs = E.edge_maps[u, v] * iota.comps[u]
        rhs = iota.comps[v] * M.edge_maps[u, v]
        @test lhs == rhs
    end
end

@testset "verify_upset_resolution / verify_downset_resolution catch illegal monomial support" begin
    P = diamond_poset()
    field = CM.QQField()
    K = CM.coeff_type(field)

    # Deliberately invalid "resolution step": for upset resolutions, nonzero in delta (row i, col j)
    # requires U_row subset U_col. We violate that on the diamond poset.
    U2 = FF.principal_upset(P, 2)
    U3 = FF.principal_upset(P, 3)

    F0 = IR.UpsetPresentation{K}(P, [U2, U3], FF.Upset[], spzeros(K, 0, 2), nothing)
    F1 = IR.UpsetPresentation{K}(P, [U2],     FF.Upset[], spzeros(K, 0, 1), nothing)
    F_infer = IR.UpsetPresentation(P, [U2], FF.Upset[], spzeros(K, 0, 1), nothing)
    F_coerced = IR.UpsetPresentation(P, [U2], FF.Upset[], spzeros(Float64, 0, 1), nothing; field=field)
    @test eltype(F_infer.delta) == K
    @test eltype(F_coerced.delta) == K
    @test_throws MethodError IR.UpsetPresentation{K}(P, [U2], FF.Upset[], spzeros(K, 0, 1), nothing; field=field)

    # delta is |U1| x |U0| = 1 x 2. Put a nonzero at (U2 row, U3 col).
    # Since U2 is not a subset of U3, this must be rejected.
    delta_bad = spzeros(K, 1, 2)
    delta_bad[1, 2] = CM.coerce(field, 1)

    @test_throws ErrorException IR.verify_upset_resolution([F0, F1], [delta_bad];
        check_d2=false, check_exactness=false)

    # Dual check for downset copresentations: nonzero in rho (row i, col j) requires
    # D_row subset D_col, where rows come from the later stage.
    D2 = FF.principal_downset(P, 2)
    D3 = FF.principal_downset(P, 3)

    E0 = IR.DownsetCopresentation{K}(P, [D3], FF.Downset[], spzeros(K, 0, 1), nothing)
    E1 = IR.DownsetCopresentation{K}(P, [D2], FF.Downset[], spzeros(K, 0, 1), nothing)
    E_infer = IR.DownsetCopresentation(P, [D2], FF.Downset[], spzeros(K, 0, 1), nothing)
    E_coerced = IR.DownsetCopresentation(P, [D2], FF.Downset[], spzeros(Float64, 0, 1), nothing; field=field)
    @test eltype(E_infer.rho) == K
    @test eltype(E_coerced.rho) == K
    @test_throws MethodError IR.DownsetCopresentation{K}(P, [D2], FF.Downset[], spzeros(K, 0, 1), nothing; field=field)

    rho_bad = spzeros(K, 1, 1)
    rho_bad[1, 1] = CM.coerce(field, 1)  # D2 is not a subset of D3

    @test_throws ErrorException IR.verify_downset_resolution([E0, E1], [rho_bad];
        check_d2=false, check_exactness=false)
end


@testset "Indicator resolutions on the diamond poset (by-hand checks)" begin
    # -------------------------------------------------------------------------
    # The diamond poset (a.k.a. the rank-2 Boolean lattice) is the first place
    # where projective/injective resolutions can have length 2 (not just 1),
    # because there are two different length-2 chains from bottom to top:
    #   1 -> 2 -> 4
    #   1 -> 3 -> 4
    #
    # In the category algebra kP (representations of the poset), this produces
    # a genuine relation between the two composites, and it manifests as:
    #   Ext^2(S1, S4) = 1
    # while Ext^1(S1, S4) = 0 (since 1<4 is not a cover).
    #
    # These tests check:
    #  * the shape of the computed resolutions for S1 and S4 (length 2),
    #  * the "by hand" pattern of the differentials at the coefficient-matrix
    #    level (equal-entry and sum-to-zero patterns),
    #  * Ext^0/1/2 between ALL simples on the diamond against an interval-based
    #    computation (reduced cohomology in degrees -1 and 0).
    # -------------------------------------------------------------------------

    P = diamond_poset()
    field = CM.QQField()
    K = CM.coeff_type(field)

    # Sanity: cover edges should be exactly 1->2, 1->3, 2->4, 3->4.
    C = FF.cover_edges(P)
    @test C[1, 2] == true
    @test C[1, 3] == true
    @test C[2, 4] == true
    @test C[3, 4] == true
    @test C[1, 4] == false
    @test C[2, 3] == false
    @test C[3, 2] == false

    # Sanity: principal upsets/downsets on a non-chain poset.
    @test FF.principal_upset(P, 1).mask == BitVector([true,  true,  true,  true])
    @test FF.principal_upset(P, 2).mask == BitVector([false, true,  false, true])
    @test FF.principal_upset(P, 3).mask == BitVector([false, false, true,  true])
    @test FF.principal_upset(P, 4).mask == BitVector([false, false, false, true])

    @test FF.principal_downset(P, 1).mask == BitVector([true,  false, false, false])
    @test FF.principal_downset(P, 2).mask == BitVector([true,  true,  false, false])
    @test FF.principal_downset(P, 3).mask == BitVector([true,  false, true,  false])
    @test FF.principal_downset(P, 4).mask == BitVector([true,  true,  true,  true])

    # Simple module S_p at vertex p: support only at p, with all structure maps zero.
    simple_at(p::Int) = one_by_one_fringe(P, FF.principal_upset(P, p), FF.principal_downset(P, p); scalar=one(K), field=field)
    S = [simple_at(p) for p in 1:P.n]
    S1, S2, S3, S4 = S

    # -------------------------------------------------------------------------
    # (A) Resolution shape checks for the "interesting" case S1 and S4.
    # -------------------------------------------------------------------------

    # Projective (upset) resolution of S1 should have length 2:
    #   F2 -> F1 -> F0 -> S1 -> 0
    # with:
    #   U0 = [Up(1)]
    #   U1 = [Up(2), Up(3)]
    #   U2 = [Up(4)]
    F, dF = IR.upset_resolution(S1; maxlen=10)

    @test length(F) == 3
    @test length(dF) == 2

    # Verify structural correctness of the upset resolution (d^2=0 + exactness).
    @test IR.verify_upset_resolution(F, dF)

    U_by_a = [f.U0 for f in F]
    @test length(U_by_a[1]) == 1
    @test U_by_a[1][1].mask == FF.principal_upset(P, 1).mask

    @test length(U_by_a[2]) == 2
    @test U_by_a[2][1].mask == FF.principal_upset(P, 2).mask
    @test U_by_a[2][2].mask == FF.principal_upset(P, 3).mask

    @test length(U_by_a[3]) == 1
    @test U_by_a[3][1].mask == FF.principal_upset(P, 4).mask

    # Differential patterns, robust to a global nonzero scalar choice:
    #
    # delta0 : U1 -> U0 should send both branch generators (at 2 and 3)
    # to the unique generator at 1 with the SAME coefficient.
    #
    # delta1 : U2 -> U1 encodes the relation between the two branches at the top;
    # the two coefficients must sum to zero (so they are negatives of each other).
    delta0 = dF[1]
    delta1 = dF[2]

    @test size(delta0) == (2, 1)
    @test size(delta1) == (1, 2)

    a = delta0[1, 1]
    b = delta0[2, 1]
    @test a != 0 && b != 0
    @test a == b

    c = delta1[1, 1]
    d = delta1[1, 2]
    @test c != 0 && d != 0
    @test c + d == 0

    # Composition must be 0 (also checked by verify_upset_resolution, but this is "by hand").
    @test Matrix(delta1 * delta0) == CM.zeros(field, 1, 1)

    # Injective (downset) resolution of S4 should have length 2:
    #   0 -> S4 -> E0 -> E1 -> E2
    # with:
    #   D0 = [Down(4)]
    #   D1 = [Down(2), Down(3)]
    #   D2 = [Down(1)]
    E, dE = IR.downset_resolution(S4; maxlen=10)

    @test length(E) == 3
    @test length(dE) == 2

    # Verify structural correctness of the downset resolution.
    @test IR.verify_downset_resolution(E, dE)

    D_by_b = [e.D0 for e in E]
    @test length(D_by_b[1]) == 1
    @test D_by_b[1][1].mask == FF.principal_downset(P, 4).mask

    @test length(D_by_b[2]) == 2
    @test D_by_b[2][1].mask == FF.principal_downset(P, 2).mask
    @test D_by_b[2][2].mask == FF.principal_downset(P, 3).mask

    @test length(D_by_b[3]) == 1
    @test D_by_b[3][1].mask == FF.principal_downset(P, 1).mask

    rho0 = dE[1]
    rho1 = dE[2]

    @test size(rho0) == (2, 1)
    @test size(rho1) == (1, 2)

    r1 = rho0[1, 1]
    r2 = rho0[2, 1]
    @test r1 != 0 && r2 != 0
    @test r1 == r2

    s1 = rho1[1, 1]
    s2 = rho1[1, 2]
    @test s1 != 0 && s2 != 0
    @test s1 + s2 == 0

    @test Matrix(rho1 * rho0) == CM.zeros(field, 1, 1)

    # -------------------------------------------------------------------------
    # (B) Ext^0/1/2 between ALL simple modules on the diamond, computed "by hand"
    #     from the open interval (x,y).
    #
    # For degree 1 and 2 on simples, we only need:
    #   - Ext^1 corresponds to reduced H^{-1} of the order complex of (x,y),
    #     which is 1 iff (x,y) is empty (i.e. x<y is a cover), else 0.
    #   - Ext^2 corresponds to reduced H^0 of that order complex, i.e.
    #       (#connected components of the open interval) - 1,
    #     and is 0 for empty intervals.
    #
    # On the diamond: (1,4) = {2,3} has 2 components, so Ext^2(S1,S4)=1.
    # -------------------------------------------------------------------------

    function strict_interval(P::FF.FinitePoset, x::Int, y::Int)
        if x == y || !FF.leq(P, x, y)
            return Int[]
        end
        return [z for z in 1:P.n if z != x && z != y && FF.leq(P, x, z) && FF.leq(P, z, y)]
    end

    # Count connected components in the induced Hasse graph on 'verts' (undirected).
    function interval_components(P::FF.FinitePoset, verts::Vector{Int})
        isempty(verts) && return 0

        inV = falses(P.n)
        for v in verts
            inV[v] = true
        end

        C = FF.cover_edges(P)
        adj = [Int[] for _ in 1:P.n]
        for u in 1:P.n, v in 1:P.n
            if C[u, v] && inV[u] && inV[v]
                push!(adj[u], v)
                push!(adj[v], u)
            end
        end

        seen = falses(P.n)
        comps = 0
        for v in verts
            if seen[v]
                continue
            end
            comps += 1
            stack = [v]
            seen[v] = true
            while !isempty(stack)
                a = pop!(stack)
                for b in adj[a]
                    if !seen[b]
                        seen[b] = true
                        push!(stack, b)
                    end
                end
            end
        end
        return comps
    end

    function expected_ext_dims_simple(P::FF.FinitePoset, x::Int, y::Int)
        # Expected (Ext^0, Ext^1, Ext^2) for simples at x and y.
        if x == y
            return (ext0 = 1, ext1 = 0, ext2 = 0)
        end
        if !FF.leq(P, x, y)
            return (ext0 = 0, ext1 = 0, ext2 = 0)
        end

        verts = strict_interval(P, x, y)
        if isempty(verts)
            # Cover relation => reduced H^{-1}(empty) = k => Ext^1 = 1.
            return (ext0 = 0, ext1 = 1, ext2 = 0)
        end

        # Nonempty interval => Ext^1 = 0 and Ext^2 = reduced H^0 = components-1.
        c = interval_components(P, verts)
        return (ext0 = 0, ext1 = 0, ext2 = max(c - 1, 0))
    end

    for x in 1:P.n, y in 1:P.n
        ext_xy = DF.ext_dimensions_via_indicator_resolutions(S[x], S[y]; maxlen=6)
        exp = expected_ext_dims_simple(P, x, y)

        @test get(ext_xy, 0, 0) == exp.ext0
        @test get(ext_xy, 1, 0) == exp.ext1
        @test get(ext_xy, 2, 0) == exp.ext2

        # On this poset, the only possible nonzero higher group for simples would
        # come from higher reduced cohomology of intervals, which does not occur here.
        @test get(ext_xy, 3, 0) == 0
        @test get(ext_xy, 4, 0) == 0
    end

    # -------------------------------------------------------------------------
    # (C) First-page vs full Ext: S1 -> S4 is the motivating case.
    # The "one-step" data cannot see Ext^2, but the full resolution must.
    # -------------------------------------------------------------------------
    F1 = IR.upset_presentation_one_step(S1)
    E1 = IR.downset_copresentation_one_step(S4)
    hom0, ext1 = DF.hom_ext_first_page(F1, E1)

    @test hom0 == 0
    @test ext1 == 0

    ext_full = DF.ext_dimensions_via_indicator_resolutions(S1, S4; maxlen=6)
    @test get(ext_full, 2, 0) == 1
end


@testset "CoverCache memoization" begin
    P = chain_poset(5)

    # Clearing and recomputing should produce a fresh cache object.
    MD._clear_cover_cache!(P)
    ccA = MD._get_cover_cache(P)
    MD._clear_cover_cache!(P)
    ccB = MD._get_cover_cache(P)
    @test ccA !== ccB

    cc1 = MD._get_cover_cache(P)
    cc2 = MD._get_cover_cache(P)
    @test cc1 === cc2

    @test cc1.Q === P
    @test length(cc1.preds) == P.n
    @test length(cc1.succs) == P.n

    # On a chain, each vertex i>1 has exactly one cover predecessor i-1.
    @test isempty(cc1.preds[1])
    for i in 2:P.n
        @test cc1.preds[i] == [i - 1]
    end
end

@testset "Poset cache lifecycle" begin
    P = chain_poset(4)
    @test isdefined(P, :cache)
    @test P.cache.cover_edges === nothing
    @test P.cache.cover === nothing
    @test P.cache.upsets === nothing
    @test P.cache.downsets === nothing

    Ce = FF.cover_edges(P)
    @test P.cache.cover_edges === Ce

    cc = MD._get_cover_cache(P)
    @test P.cache.cover === cc
    @test P.cache.cover !== nothing

    u1 = FF.upset_indices(P, 1)
    d1 = FF.downset_indices(P, 1)
    @test P.cache.upsets !== nothing
    @test P.cache.downsets !== nothing
    @test collect(u1) == P.cache.upsets[1]
    @test collect(d1) == P.cache.downsets[1]

    MD._clear_cover_cache!(P)
    @test P.cache.cover_edges === nothing
    @test P.cache.cover === nothing
    @test P.cache.upsets === nothing
    @test P.cache.downsets === nothing

    cc2 = MD._get_cover_cache(P)
    @test cc2 !== cc
end


@testset "map_leq uses cached chosen-chain pointers" begin
    P = chain_poset(4)
    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)
    cc = MD._get_cover_cache(P)
    FF._clear_chain_parent_cache!(cc)
    n_before = sum(length.(cc.chain_parent))

    # Build a tiny 2-dimensional module on the chain 1<2<3<4.
    dims = [2, 2, 2, 2]
    edge = Dict{Tuple{Int,Int}, Matrix{K}}()
    for (u, v) in FF.cover_edges(P)
        edge[(u, v)] = CM.zeros(field, dims[v], dims[u])
    end
    edge[(1, 2)] = K[c(2) c(0); c(0) c(3)]
    edge[(2, 3)] = K[c(5) c(0); c(0) c(7)]
    edge[(3, 4)] = K[c(11) c(0); c(0) c(13)]
    M = MD.PModule{K}(P, dims, edge; field=field)
    for d in M.map_compose
        empty!(d)
    end
    m_before = sum(length.(M.map_compose))

    A14 = MD.map_leq(M, 1, 4; cache=cc)
    @test A14 == K[c(110) c(0); c(0) c(273)]

    # The chosen-chain cache should now contain at least the (1,4) entry.
    @test length(cc.chain_parent) >= 1
    n_after_first = sum(length.(cc.chain_parent))
    m_after_first = sum(length.(M.map_compose))
    @test m_after_first > m_before

    A14b = MD.map_leq(M, 1, 4; cache=cc)
    @test A14b == A14
    @test sum(length.(cc.chain_parent)) == n_after_first
    @test sum(length.(M.map_compose)) == m_after_first
end

@testset "map_leq tiny fast paths bypass compose memo" begin
    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)

    # 1x1 path: scalar composition path should not populate compose memo.
    P1 = chain_poset(4)
    dims1 = [1, 1, 1, 1]
    edge1 = Dict{Tuple{Int,Int}, Matrix{K}}(
        (1, 2) => reshape([c(2)], 1, 1),
        (2, 3) => reshape([c(3)], 1, 1),
        (3, 4) => reshape([c(5)], 1, 1),
    )
    M1 = MD.PModule{K}(P1, dims1, edge1; field=field)
    cc1 = MD._get_cover_cache(P1)
    FF._clear_chain_parent_cache!(cc1)
    for d in M1.map_compose
        empty!(d)
    end
    @test MD.map_leq(M1, 1, 4; cache=cc1) == reshape([c(30)], 1, 1)
    @test sum(length.(M1.map_compose)) == 0

    # Two-edge path with nontrivial dimensions should also skip compose memo.
    P2 = chain_poset(3)
    dims2 = [3, 2, 4]
    edge2 = Dict{Tuple{Int,Int}, Matrix{K}}(
        (1, 2) => K[
            c(1) c(2) c(0);
            c(0) c(3) c(4)
        ],
        (2, 3) => K[
            c(1) c(0);
            c(0) c(1);
            c(2) c(0);
            c(0) c(2)
        ],
    )
    M2 = MD.PModule{K}(P2, dims2, edge2; field=field)
    cc2 = MD._get_cover_cache(P2)
    FF._clear_chain_parent_cache!(cc2)
    for d in M2.map_compose
        empty!(d)
    end
    A13 = MD.map_leq(M2, 1, 3; cache=cc2)
    @test A13 == edge2[(2, 3)] * edge2[(1, 2)]
    @test sum(length.(M2.map_compose)) == 0
end


@testset "map_leq is path-independent on a functorial diamond" begin
    P = diamond_poset()
    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)
    cc = MD._get_cover_cache(P)
    FF._clear_chain_parent_cache!(cc)

    # A functorial module where the two length-2 paths 1->2->4 and 1->3->4 agree.
    dims = [1, 1, 1, 1]
    edge = Dict{Tuple{Int,Int}, Matrix{K}}()
    for (u, v) in FF.cover_edges(P)
        edge[(u, v)] = reshape([c(0)], 1, 1)
    end
    edge[(1, 2)] = reshape([c(2)], 1, 1)
    edge[(2, 4)] = reshape([c(5)], 1, 1)   # composite 10
    edge[(1, 3)] = reshape([c(1)], 1, 1)
    edge[(3, 4)] = reshape([c(10)], 1, 1)  # composite also 10
    M = MD.PModule{K}(P, dims, edge; field=field)
    for d in M.map_compose
        empty!(d)
    end

    A14 = MD.map_leq(M, 1, 4; cache=cc)
    @test A14 == reshape([c(10)], 1, 1)
end

@testset "map_leq compose memo is per-module" begin
    P = chain_poset(4)
    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)

    dims = [2, 2, 2, 2]
    edgeA = Dict{Tuple{Int,Int}, Matrix{K}}(
        (1, 2) => K[c(2) c(0); c(0) c(3)],
        (2, 3) => K[c(3) c(0); c(0) c(5)],
        (3, 4) => K[c(11) c(0); c(0) c(13)],
    )
    edgeB = Dict{Tuple{Int,Int}, Matrix{K}}(
        (1, 2) => K[c(5) c(0); c(0) c(7)],
        (2, 3) => K[c(7) c(0); c(0) c(11)],
        (3, 4) => K[c(13) c(0); c(0) c(17)],
    )
    M1 = MD.PModule{K}(P, dims, edgeA; field=field)
    M2 = MD.PModule{K}(P, dims, edgeB; field=field)

    cc = MD._get_cover_cache(P)
    FF._clear_chain_parent_cache!(cc)
    for d in M1.map_compose
        empty!(d)
    end
    for d in M2.map_compose
        empty!(d)
    end

    A1 = MD.map_leq(M1, 1, 4; cache=cc)
    A2 = MD.map_leq(M2, 1, 4; cache=cc)
    @test A1 == K[c(66) c(0); c(0) c(195)]
    @test A2 == K[c(455) c(0); c(0) c(1309)]
    @test A1 != A2
    @test sum(length.(M1.map_compose)) >= 1
    @test sum(length.(M2.map_compose)) >= 1
end

@testset "map_leq_many parity and preallocated output" begin
    P = chain_poset(4)
    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)

    dims = [1, 1, 1, 1]
    edge = Dict{Tuple{Int,Int}, Matrix{K}}(
        (1, 2) => reshape([c(2)], 1, 1),
        (2, 3) => reshape([c(3)], 1, 1),
        (3, 4) => reshape([c(5)], 1, 1),
    )
    M = MD.PModule{K}(P, dims, edge; field=field)
    cc = MD._get_cover_cache(P)
    FF._clear_chain_parent_cache!(cc)
    for d in M.map_compose
        empty!(d)
    end

    pairs = Tuple{Int,Int}[(1, 1), (1, 2), (1, 3), (1, 4), (2, 4), (4, 4)]
    batch = MD.map_leq_many(M, pairs; cache=cc)
    @test length(batch) == length(pairs)
    @test batch[1] == reshape([c(1)], 1, 1)
    @test batch[4] == reshape([c(30)], 1, 1)

    @inbounds for i in eachindex(pairs)
        u, v = pairs[i]
        @test batch[i] == MD.map_leq(M, u, v; cache=cc)
    end

    out = Vector{Matrix{K}}(undef, length(pairs))
    MD.map_leq_many!(out, M, pairs; cache=cc)
    @test out == batch
    @test sum(length.(M.map_compose)) == 0
end

@testset "map_leq_many cached plan reuse + signature invalidation" begin
    P = chain_poset(7)
    field = CM.F3()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)

    dims = fill(2, 7)
    edge = Dict{Tuple{Int,Int}, Matrix{K}}()
    @inbounds for u in 1:6
        edge[(u, u + 1)] = K[c(1) c(2); c(0) c(1)]
    end
    M = MD.PModule{K}(P, dims, edge; field=field)
    cc = MD._get_cover_cache(P)
    old_min = MD.MAP_LEQ_MANY_PLAN_MIN_LEN[]
    try
        MD.MAP_LEQ_MANY_PLAN_MIN_LEN[] = 1

        pairs = Tuple{Int,Int}[(1, 1), (1, 3), (2, 5), (1, 7), (3, 7), (4, 7)]
        b1 = MD.map_leq_many(M, pairs; cache=cc)
        b2 = MD.map_leq_many(M, pairs; cache=cc)
        @test b1 == b2
        @test any(!isempty(d) for d in M.map_many_plan)

        # Mutate the pair vector in place; the signature check should force a rebuild.
        pairs[1] = (2, 2)
        b3 = MD.map_leq_many(M, pairs; cache=cc)
        @test b3[1] == MD.map_leq(M, 2, 2; cache=cc)
        @test all(b3[i] == MD.map_leq(M, pairs[i][1], pairs[i][2]; cache=cc) for i in eachindex(pairs))
    finally
        MD.MAP_LEQ_MANY_PLAN_MIN_LEN[] = old_min
    end
end

@testset "Modules negative API contracts (map_leq / map_leq_many!)" begin
    P = diamond_poset()  # 2 and 3 are incomparable
    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)
    cc = MD._get_cover_cache(P)

    dims = [1, 1, 1, 1]
    edge = Dict{Tuple{Int,Int}, Matrix{K}}(
        (1, 2) => reshape([c(2)], 1, 1),
        (1, 3) => reshape([c(3)], 1, 1),
        (2, 4) => reshape([c(5)], 1, 1),
        (3, 4) => reshape([c(10)], 1, 1),
    )
    M = MD.PModule{K}(P, dims, edge; field=field)

    # map_leq contracts
    @test_throws ErrorException MD.map_leq(M, 0, 1; cache=cc)
    @test_throws ErrorException MD.map_leq(M, 1, 5; cache=cc)
    @test_throws ErrorException MD.map_leq(M, 2, 3; cache=cc)  # incomparable
    @test_throws ErrorException MD.map_leq(
        M, 1, 4;
        cache=cc,
        opts=CM.ModuleOptions(cache=cc),
    )

    # map_leq_many! contracts
    pairs = Tuple{Int,Int}[(1, 1), (1, 2), (1, 4)]
    @test_throws ErrorException MD.map_leq_many!(Vector{Matrix{K}}(undef, 2), M, pairs; cache=cc)
    @test_throws ErrorException MD.map_leq_many!(Vector{Matrix{K}}(undef, 1), M, Tuple{Int,Int}[(0, 1)]; cache=cc)
    @test_throws ErrorException MD.map_leq_many!(Vector{Matrix{K}}(undef, 1), M, Tuple{Int,Int}[(2, 3)]; cache=cc)
    @test_throws ErrorException MD.map_leq_many!(
        Vector{Matrix{K}}(undef, 1),
        M,
        Tuple{Int,Int}[(1, 4)];
        cache=cc,
        opts=CM.ModuleOptions(cache=cc),
    )
end

@testset "ModuleOptions contracts (cache + check_sizes)" begin
    P = chain_poset(3)
    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)
    cc = MD._get_cover_cache(P)

    dims = [1, 1, 1]
    edge = Dict{Tuple{Int,Int}, Matrix{K}}(
        (1, 2) => reshape([c(2)], 1, 1),
        (2, 3) => reshape([c(3)], 1, 1),
    )
    M = MD.PModule{K}(P, dims, edge; field=field)

    # opts.cache should be equivalent to cache keyword.
    A_kw = MD.map_leq(M, 1, 3; cache=cc)
    A_opt = MD.map_leq(M, 1, 3; opts=CM.ModuleOptions(cache=cc))
    @test A_kw == A_opt

    pairs = Tuple{Int,Int}[(1, 2), (1, 3), (2, 3)]
    B_kw = MD.map_leq_many(M, pairs; cache=cc)
    B_opt = MD.map_leq_many(M, pairs; opts=CM.ModuleOptions(cache=cc))
    @test B_kw == B_opt

    # check_sizes contract in constructor.
    bad_edge = Dict{Tuple{Int,Int}, Matrix{K}}(
        (1, 2) => reshape([c(1), c(2)], 2, 1),  # wrong row count for dims[2] == 1
        (2, 3) => reshape([c(1)], 1, 1),
    )
    @test_throws ErrorException MD.PModule{K}(P, dims, bad_edge; field=field)
    M_bad = MD.PModule{K}(P, dims, bad_edge; field=field, opts=CM.ModuleOptions(check_sizes=false))
    @test M_bad.dims == dims

    # Passing both explicit keyword and non-default opts is rejected.
    @test_throws ErrorException MD.PModule{K}(
        P, dims, edge;
        field=field,
        check_sizes=false,
        opts=CM.ModuleOptions(check_sizes=false),
    )
end

@testset "map_leq uses dense chain-parent cache on large finite posets" begin
    old_min = FF.CHAIN_PARENT_DENSE_MIN_ENTRIES[]
    old_per = FF.CHAIN_PARENT_DENSE_MAX_ENTRIES_PER_THREAD[]
    old_total = FF.CHAIN_PARENT_DENSE_MAX_TOTAL_ENTRIES[]
    try
        FF.CHAIN_PARENT_DENSE_MIN_ENTRIES[] = 1
        FF.CHAIN_PARENT_DENSE_MAX_ENTRIES_PER_THREAD[] = 1_000_000
        FF.CHAIN_PARENT_DENSE_MAX_TOTAL_ENTRIES[] = max(1, Threads.maxthreadid()) * 1_000_000

        P = chain_poset(40)
        cc = MD._get_cover_cache(P)
        @test cc.chain_parent_dense !== nothing
        FF._clear_chain_parent_cache!(cc)

        field = CM.QQField()
        K = CM.coeff_type(field)
        oneK = CM.coerce(field, 1)
        dims = ones(Int, 40)
        edge = Dict{Tuple{Int,Int}, Matrix{K}}()
        for (u, v) in FF.cover_edges(P)
            edge[(u, v)] = reshape([oneK], 1, 1)
        end
        M = MD.PModule{K}(P, dims, edge; field=field)
        for d in M.map_compose
            empty!(d)
        end

        pairs = Tuple{Int,Int}[(1, 40), (5, 40), (10, 35), (20, 39), (1, 30)]
        mats = MD.map_leq_many(M, pairs; cache=cc)
        @test all(A -> A == reshape([oneK], 1, 1), mats)
        @test sum(length.(cc.chain_parent)) == 0
        @test any(m -> any(m.seen), cc.chain_parent_dense)
    finally
        FF.CHAIN_PARENT_DENSE_MIN_ENTRIES[] = old_min
        FF.CHAIN_PARENT_DENSE_MAX_ENTRIES_PER_THREAD[] = old_per
        FF.CHAIN_PARENT_DENSE_MAX_TOTAL_ENTRIES[] = old_total
    end
end

@testset "direct_sum preserves sparse edge-map storage" begin
    P = chain_poset(2)
    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)

    dimsA = [70, 70]
    dimsB = [60, 60]
    edgeA = Dict{Tuple{Int,Int}, SparseMatrixCSC{K,Int}}(
        (1, 2) => sparse([1, 2], [3, 15], K[c(2), c(5)], dimsA[2], dimsA[1]),
    )
    edgeB = Dict{Tuple{Int,Int}, SparseMatrixCSC{K,Int}}(
        (1, 2) => sparse([4, 9], [5, 11], K[c(7), c(11)], dimsB[2], dimsB[1]),
    )

    A = MD.PModule{K}(P, dimsA, edgeA; field=field)
    B = MD.PModule{K}(P, dimsB, edgeB; field=field)
    S = MD.direct_sum(A, B)
    @test S.edge_maps.maps_from_pred[2][1] isa SparseMatrixCSC{K,Int}
    M12 = S.edge_maps[1, 2]
    @test nnz(M12) == 4
    @test M12[1, 3] == c(2)
    @test M12[2, 15] == c(5)
    @test M12[dimsA[2] + 4, dimsA[1] + 5] == c(7)
    @test M12[dimsA[2] + 9, dimsA[1] + 11] == c(11)
end

@testset "PModule stores concrete poset type + map_leq identity cache reuse" begin
    P = chain_poset(3)
    field = CM.QQField()
    K = CM.coeff_type(field)
    oneK = CM.coerce(field, 1)
    dims = [2, 2, 2]
    edge = Dict{Tuple{Int,Int}, Matrix{K}}(
        (1, 2) => reshape(K[oneK, oneK, oneK, oneK], 2, 2),
        (2, 3) => reshape(K[oneK, oneK, oneK, oneK], 2, 2),
    )
    M = MD.PModule{K}(P, dims, edge; field=field)
    cc = MD._get_cover_cache(P)

    @test M isa MD.PModule{K,typeof(field),Matrix{K},typeof(P)}

    A22 = MD.map_leq(M, 2, 2; cache=cc)
    B22 = MD.map_leq(M, 2, 2; cache=cc)
    @test A22 === B22

    batch = MD.map_leq_many(M, Tuple{Int,Int}[(2, 2), (2, 2), (1, 1)]; cache=cc)
    @test batch[1] === batch[2]
    @test batch[1] === A22
end

@testset "direct_sum route policy picks dense for high-density sparse inputs" begin
    P = chain_poset(2)
    field = CM.RealField(Float64; rtol=1e-10, atol=1e-12)
    K = CM.coeff_type(field)
    dims = [64, 64]

    low_vals = fill(one(K), 8)
    low_edgeA = Dict{Tuple{Int,Int}, SparseMatrixCSC{K,Int}}(
        (1, 2) => sparse(collect(1:8), collect(1:8), low_vals, dims[2], dims[1]),
    )
    low_edgeB = Dict{Tuple{Int,Int}, SparseMatrixCSC{K,Int}}(
        (1, 2) => sparse(collect(9:16), collect(9:16), low_vals, dims[2], dims[1]),
    )

    # Build a fully-populated sparse matrix (density ~= 1) to force dense output.
    Ii = Int[]
    Jj = Int[]
    Vv = K[]
    sizehint!(Ii, dims[1] * dims[2])
    sizehint!(Jj, dims[1] * dims[2])
    sizehint!(Vv, dims[1] * dims[2])
    @inbounds for j in 1:dims[1], i in 1:dims[2]
        push!(Ii, i)
        push!(Jj, j)
        push!(Vv, one(K))
    end
    high_dense_sparse = sparse(Ii, Jj, Vv, dims[2], dims[1])
    high_edgeA = Dict{Tuple{Int,Int}, SparseMatrixCSC{K,Int}}((1, 2) => high_dense_sparse)
    high_edgeB = Dict{Tuple{Int,Int}, SparseMatrixCSC{K,Int}}((1, 2) => high_dense_sparse)

    old_min = MD.DIRECT_SUM_SPARSE_MIN_TOTAL_ENTRIES[]
    try
        MD.DIRECT_SUM_SPARSE_MIN_TOTAL_ENTRIES[] = 1

        Slow = MD.direct_sum(
            MD.PModule{K}(P, dims, low_edgeA; field=field),
            MD.PModule{K}(P, dims, low_edgeB; field=field),
        )
        @test Slow.edge_maps[1, 2] isa SparseMatrixCSC{K,Int}

        Shigh = MD.direct_sum(
            MD.PModule{K}(P, dims, high_edgeA; field=field),
            MD.PModule{K}(P, dims, high_edgeB; field=field),
        )
        @test Shigh.edge_maps[1, 2] isa Matrix{K}
    finally
        MD.DIRECT_SUM_SPARSE_MIN_TOTAL_ENTRIES[] = old_min
    end
end

@testset "Modules direct oracle checks across non-QQ fields" begin
    nonqq_fields = (CM.F2(), CM.F3(), CM.Fp(5), CM.RealField(Float64; rtol=1e-10, atol=1e-12))

    for field in nonqq_fields
        K = CM.coeff_type(field)
        c(x) = CM.coerce(field, x)

        # map_leq and map_leq_many: exact oracle on a chain (unique path).
        P = chain_poset(4)
        dims = [2, 2, 2, 2]
        E12 = K[c(1) c(2); c(0) c(3)]
        E23 = K[c(2) c(0); c(4) c(1)]
        E34 = K[c(1) c(1); c(0) c(2)]
        edge = Dict{Tuple{Int,Int}, Matrix{K}}(
            (1, 2) => E12,
            (2, 3) => E23,
            (3, 4) => E34,
        )
        M = MD.PModule{K}(P, dims, edge; field=field)
        cc = MD._get_cover_cache(P)
        FF._clear_chain_parent_cache!(cc)
        for d in M.map_compose
            empty!(d)
        end

        oracle13 = E23 * E12
        oracle14 = E34 * oracle13
        @test MD.map_leq(M, 1, 3; cache=cc) == oracle13
        @test MD.map_leq(M, 1, 4; cache=cc) == oracle14

        pairs = Tuple{Int,Int}[(1, 1), (1, 2), (1, 3), (1, 4), (2, 4)]
        batch = MD.map_leq_many(M, pairs; cache=cc)
        @test batch[1] == CM.eye(field, dims[1])
        @test batch[3] == oracle13
        @test batch[4] == oracle14

        out = Vector{Matrix{K}}(undef, length(pairs))
        MD.map_leq_many!(out, M, pairs; cache=cc)
        @test out == batch

        # direct_sum: hand-assembled block oracle.
        Q2 = chain_poset(2)
        A = MD.PModule{K}(Q2, [2, 1], Dict((1, 2) => reshape(K[c(2), c(3)], 1, 2)); field=field)
        B = MD.PModule{K}(Q2, [1, 2], Dict((1, 2) => reshape(K[c(5), c(7)], 2, 1)); field=field)
        S = MD.direct_sum(A, B)
        M12 = S.edge_maps[1, 2]
        oracle12 = K[
            c(2) c(3) c(0);
            c(0) c(0) c(5);
            c(0) c(0) c(7)
        ]
        @test M12 == oracle12

        # canonical injections/projections are algebraically correct.
        _, iA, iB, pA, pB = MD.direct_sum_with_maps(A, B)
        @test pA.comps[1] * iA.comps[1] == CM.eye(field, A.dims[1])
        @test pB.comps[2] * iB.comps[2] == CM.eye(field, B.dims[2])
        @test pA.comps[1] * iB.comps[1] == CM.zeros(field, A.dims[1], B.dims[1])
    end
end

@testset "Modules randomized differential suite vs naive chain oracle" begin
    function rand_coeff(rng::Random.AbstractRNG, field)
        if field isa CM.RealField
            return CM.coerce(field, randn(rng))
        end
        v = rand(rng, -3:3)
        v == 0 && (v = 1)
        return CM.coerce(field, v)
    end

    function chain_oracle(M::MD.PModule{K}, u::Int, v::Int) where {K}
        u == v && return CM.eye(M.field, M.dims[v])
        A = M.edge_maps[u, u + 1]
        for k in (u + 1):(v - 1)
            A = FL._matmul(M.edge_maps[k, k + 1], A)
        end
        return A
    end

    function mat_eq_field(field, A, B)
        if field isa CM.RealField
            return isapprox(Matrix{Float64}(A), Matrix{Float64}(B); rtol=1e-8, atol=1e-9)
        end
        return A == B
    end

    rng = Random.MersenneTwister(0x5A17)
    densities = (0.2, 0.5, 0.8)
    sizes = (4, 6, 8)
    trials = 2

    with_fields(FIELDS_FULL) do field
        K = CM.coeff_type(field)
        for n in sizes
            P = chain_poset(n)
            cc = MD._get_cover_cache(P)
            for dens in densities
                for t in 1:trials
                    dims = [rand(rng, 1:4) for _ in 1:n]
                    edge = Dict{Tuple{Int,Int}, Matrix{K}}()
                    for u in 1:(n - 1)
                        A = CM.zeros(field, dims[u + 1], dims[u])
                        @inbounds for i in 1:dims[u + 1], j in 1:dims[u]
                            rand(rng) < dens || continue
                            A[i, j] = rand_coeff(rng, field)
                        end
                        edge[(u, u + 1)] = A
                    end

                    M = MD.PModule{K}(P, dims, edge; field=field)
                    FF._clear_chain_parent_cache!(cc)
                    for d in M.map_compose
                        empty!(d)
                    end

                    pairs = Tuple{Int,Int}[]
                    sizehint!(pairs, 32)
                    for _ in 1:32
                        u = rand(rng, 1:n)
                        v = rand(rng, u:n)
                        push!(pairs, (u, v))
                    end

                    batch = MD.map_leq_many(M, pairs; cache=cc)
                    out = Vector{Matrix{K}}(undef, length(pairs))
                    MD.map_leq_many!(out, M, pairs; cache=cc)
                    @test length(batch) == length(pairs)
                    @test length(out) == length(pairs)

                    @inbounds for i in eachindex(pairs)
                        u, v = pairs[i]
                        oracle = chain_oracle(M, u, v)
                        got = MD.map_leq(M, u, v; cache=cc)
                        @test mat_eq_field(field, got, oracle)
                        @test mat_eq_field(field, batch[i], oracle)
                        @test mat_eq_field(field, out[i], oracle)
                    end
                end
            end
        end
    end
end

@testset "Modules randomized branching-poset oracles across fields" begin
    function rand_nonzero_coeff(rng::Random.AbstractRNG, field)
        if field isa CM.RealField
            v = 0.0
            while v == 0.0
                v = randn(rng)
            end
            return CM.coerce(field, v)
        end
        v = zero(CM.coeff_type(field))
        while iszero(v)
            v = CM.coerce(field, rand(rng, -5:5))
        end
        return v
    end

    function random_branching_poset(rng::Random.AbstractRNG, n::Int)
        L = falses(n, n)
        @inbounds for i in 1:n
            L[i, i] = true
        end
        # Force a branching core 1<2,1<3,2<4,3<4.
        if n >= 4
            L[1, 2] = true
            L[1, 3] = true
            L[2, 4] = true
            L[3, 4] = true
        end
        for i in 1:n
            for j in (i + 1):n
                rand(rng) < 0.28 || continue
                L[i, j] = true
            end
        end
        # Transitive closure.
        for k in 1:n
            for i in 1:n
                L[i, k] || continue
                @inbounds for j in 1:n
                    L[k, j] || continue
                    L[i, j] = true
                end
            end
        end
        return FF.FinitePoset(L; check=false)
    end

    rng = Random.MersenneTwister(0xBADA55)
    trials = 4
    d = 3

    mat_eq_field(field, A, B) = field isa CM.RealField ?
        isapprox(Matrix{Float64}(A), Matrix{Float64}(B); rtol=1e-8, atol=1e-9) :
        (A == B)

    with_fields(FIELDS_FULL) do field
        K = CM.coeff_type(field)
        for _ in 1:trials
            P = random_branching_poset(rng, 7)
            n = FF.nvertices(P)
            scales = [[rand_nonzero_coeff(rng, field) for _ in 1:d] for _ in 1:n]
            dims = fill(d, n)
            edge = Dict{Tuple{Int,Int}, Matrix{K}}()
            for (u, v) in FF.cover_edges(P)
                A = CM.zeros(field, d, d)
                @inbounds for k in 1:d
                    A[k, k] = scales[v][k] / scales[u][k]
                end
                edge[(u, v)] = A
            end
            M = MD.PModule{K}(P, dims, edge; field=field)
            cc = MD._get_cover_cache(P)
            FF._clear_chain_parent_cache!(cc)
            for dct in M.map_compose
                empty!(dct)
            end

            pairs = Tuple{Int,Int}[]
            for _ in 1:24
                u = rand(rng, 1:n)
                v = rand(rng, 1:n)
                FF.leq(P, u, v) || continue
                push!(pairs, (u, v))
            end
            isempty(pairs) && continue

            batch = MD.map_leq_many(M, pairs; cache=cc)
            @inbounds for i in eachindex(pairs)
                u, v = pairs[i]
                oracle = CM.zeros(field, d, d)
                for k in 1:d
                    oracle[k, k] = scales[v][k] / scales[u][k]
                end
                @test mat_eq_field(field, batch[i], oracle)
                @test mat_eq_field(field, MD.map_leq(M, u, v; cache=cc), oracle)
            end
        end
    end
end

@testset "Modules threaded batch parity under contention" begin
    Threads.nthreads() > 1 || return

    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)
    rng = Random.MersenneTwister(0xC011EC7)

    P = chain_poset(18)
    dims = ones(Int, 18)
    edge = Dict{Tuple{Int,Int}, Matrix{K}}()
    for (u, v) in FF.cover_edges(P)
        edge[(u, v)] = reshape([c(rand(rng, 1:9))], 1, 1)
    end
    M = MD.PModule{K}(P, dims, edge; field=field)
    cc = MD._get_cover_cache(P)
    FF._clear_chain_parent_cache!(cc)
    for dct in M.map_compose
        empty!(dct)
    end

    base_pairs = Tuple{Int,Int}[]
    for _ in 1:64
        u = rand(rng, 1:18)
        v = rand(rng, u:18)
        push!(base_pairs, (u, v))
    end
    pairs = vcat(base_pairs, base_pairs, base_pairs, base_pairs)
    serial = MD.map_leq_many(M, pairs; cache=cc)

    nruns = max(8, 4 * Threads.nthreads())
    threaded = Vector{Vector{Matrix{K}}}(undef, nruns)
    Threads.@threads :static for r in 1:nruns
        threaded[r] = MD.map_leq_many(M, pairs; cache=cc)
    end
    for r in 1:nruns
        @test threaded[r] == serial
    end

    threaded_inplace = Vector{Vector{Matrix{K}}}(undef, nruns)
    Threads.@threads :static for r in 1:nruns
        out = Vector{Matrix{K}}(undef, length(pairs))
        MD.map_leq_many!(out, M, pairs; cache=cc)
        threaded_inplace[r] = out
    end
    for r in 1:nruns
        @test threaded_inplace[r] == serial
    end
end

@testset "Modules backendized map storage oracle (Nemo path)" begin
    FL._have_nemo() || return

    old_thr = FL.NEMO_THRESHOLD[]
    try
        FL.NEMO_THRESHOLD[] = 1
        field = CM.QQField()
        K = CM.coeff_type(field)
        c(x) = CM.coerce(field, x)

        P = chain_poset(3)
        dims = [4, 4, 4]
        E12 = K[c(1) c(2) c(0) c(0);
                c(0) c(1) c(3) c(0);
                c(0) c(0) c(1) c(4);
                c(0) c(0) c(0) c(1)]
        E23 = K[c(2) c(0) c(0) c(0);
                c(0) c(3) c(0) c(0);
                c(0) c(0) c(5) c(0);
                c(0) c(0) c(0) c(7)]
        M = MD.PModule{K}(P, dims, Dict((1, 2) => E12, (2, 3) => E23); field=field)
        @test M.edge_maps[1, 2] isa CM.BackendMatrix{K}
        @test M.edge_maps[2, 3] isa CM.BackendMatrix{K}

        cc = MD._get_cover_cache(P)
        for dct in M.map_compose
            empty!(dct)
        end
        oracle13 = E23 * E12
        @test Matrix{K}(MD.map_leq(M, 1, 3; cache=cc)) == oracle13
        out = MD.map_leq_many(M, Tuple{Int,Int}[(1, 2), (1, 3), (2, 3)]; cache=cc)
        @test Matrix{K}(out[2]) == oracle13
    finally
        FL.NEMO_THRESHOLD[] = old_thr
    end
end

@testset "Modules perf regression guard (map_leq_many memo path)" begin
    function old_map_leq_chain(M::MD.PModule{K}, preds::Vector{Vector{Int}}, u::Int, v::Int) where {K}
        u == v && return CM.eye(M.field, M.dims[v])
        uv = (u, v)
        if haskey(M.edge_maps, u, v)
            return M.edge_maps[u, v]
        end
        idx = findfirst(x -> x != u && FF.leq(M.Q, u, x), preds[v])
        w = (idx === nothing) ? u : preds[v][idx]
        return FL._matmul(M.edge_maps[w, v], old_map_leq_chain(M, preds, u, w))
    end

    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)
    P = chain_poset(12)
    dims = fill(2, 12)
    edge = Dict{Tuple{Int,Int}, Matrix{K}}()
    for u in 1:11
        edge[(u, u + 1)] = K[c(1) c(u % 3 + 1); c(0) c(1)]
    end
    M = MD.PModule{K}(P, dims, edge; field=field)
    cc = MD._get_cover_cache(P)
    FF._clear_chain_parent_cache!(cc)
    for dct in M.map_compose
        empty!(dct)
    end

    rng = Random.MersenneTwister(0xFEED)
    seed_pairs = Tuple{Int,Int}[]
    for _ in 1:32
        u = rand(rng, 1:12)
        v = rand(rng, u:12)
        push!(seed_pairs, (u, v))
    end
    pairs = vcat(seed_pairs, seed_pairs, seed_pairs, seed_pairs, seed_pairs)
    preds = cc.preds

    # warmup
    MD.map_leq_many(M, pairs; cache=cc)
    for (u, v) in pairs
        old_map_leq_chain(M, preds, u, v)
    end

    old_times = Float64[]
    new_times = Float64[]
    old_allocs = Int[]
    new_allocs = Int[]
    for _ in 1:4
        GC.gc()
        t_old = @timed begin
            for (u, v) in pairs
                old_map_leq_chain(M, preds, u, v)
            end
        end
        push!(old_times, t_old.time)
        push!(old_allocs, t_old.bytes)

        GC.gc()
        t_new = @timed MD.map_leq_many(M, pairs; cache=cc)
        push!(new_times, t_new.time)
        push!(new_allocs, t_new.bytes)
    end

    sort!(old_times); sort!(new_times)
    sort!(old_allocs); sort!(new_allocs)
    med_old_t = old_times[2]
    med_new_t = new_times[2]
    med_old_a = old_allocs[2]
    med_new_a = new_allocs[2]

    @test med_new_t <= 1.20 * med_old_t
    @test med_new_a <= med_old_a
end

@testset "CoverEdgeMapStore correctness" begin
    # simple chain 1 < 2 < 3
    leq = Bool[
        1 1 1;
        0 1 1;
        0 0 1
    ]
    Q = FF.FinitePoset(leq)
    field = CM.QQField()
    K = CM.coeff_type(field)
    c(x) = CM.coerce(field, x)

    dims = [1, 1, 1]

    # cover edges: (1,2), (2,3)
    edge_maps = Dict{Tuple{Int,Int}, Matrix{K}}()
    edge_maps[(1,2)] = reshape([c(1)], 1, 1)
    edge_maps[(2,3)] = reshape([c(2)], 1, 1)

    M = MD.PModule{K}(Q, dims, edge_maps; field=field)

    @test M.edge_maps[1, 2] == reshape([c(1)], 1, 1)
    @test M.edge_maps[2, 3] == reshape([c(2)], 1, 1)
    @test_throws KeyError M.edge_maps[1, 3]  # not a cover edge

    # lock in the stricter API (no tuple indexing)
    @test_throws MethodError M.edge_maps[(1,2)]

    # check predecessor alignment
    @test M.edge_maps.preds[2] == [1]
    @test M.edge_maps.preds[3] == [2]
    @test M.edge_maps.maps_from_pred[2][1] == reshape([c(1)], 1, 1)
    @test M.edge_maps.maps_from_pred[3][1] == reshape([c(2)], 1, 1)

    # iteration yields cover edges
    seen = Set{Tuple{Int,Int}}()
    for (e, A) in M.edge_maps
        push!(seen, e)
        u, v = e
        @test A == M.edge_maps[u, v]
    end
    @test seen == Set([(1,2), (2,3)])
end
