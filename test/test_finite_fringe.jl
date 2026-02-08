with_fields(FIELDS_FULL) do field
K = CM.coeff_type(field)
@inline c(x) = CM.coerce(field, x)

@testset "FiniteFringe basics" begin
    P = chain_poset(3)

    # cover edges on a 3-chain should be 1->2 and 2->3 only
    C = FF.cover_edges(P)
    @test C[1,2] == true
    @test C[2,3] == true
    @test C[1,3] == false

    # principal sets
    U2 = FF.principal_upset(P, 2)      # {2,3}
    D2 = FF.principal_downset(P, 2)    # {1,2}
    @test U2.mask == BitVector([false, true,  true])
    @test D2.mask == BitVector([true,  true,  false])

    # 1x1 interval module supported at {2}
    phi = spzeros(K, 1, 1)
    phi[1,1] = c(1)
    M = FF.FringeModule{K}(P, [U2], [D2], phi; field=field)

    @test FF.fiber_dimension(M, 1) == 0
    @test FF.fiber_dimension(M, 2) == 1
    @test FF.fiber_dimension(M, 3) == 0

    # Endomorphisms of a connected indicator should be 1-dimensional
    @test FF.hom_dimension(M, M) == 1

    # Monomial condition should reject nonzero entry when U cap D is empty
    U3 = FF.principal_upset(P, 3)      # {3}
    D1 = FF.principal_downset(P, 1)    # {1}
    phi_bad = spzeros(K, 1, 1)
    phi_bad[1,1] = c(1)
    @test_throws AssertionError FF.FringeModule{K}(P, [U3], [D1], phi_bad; field=field)


        @testset "FinitePoset validation errors" begin
        # Not reflexive.
        leq1 = trues(2, 2)
        leq1[1, 1] = false
        @test_throws ErrorException FF.FinitePoset(leq1)

        # Violates antisymmetry: 1<=2 and 2<=1.
        leq2 = trues(2, 2)
        @test_throws ErrorException FF.FinitePoset(leq2)

        # Violates transitivity: 1<=2, 2<=3, but not 1<=3.
        leq3 = falses(3, 3)
        for i in 1:3
            leq3[i, i] = true
        end
        leq3[1, 2] = true
        leq3[2, 3] = true
        @test_throws ErrorException FF.FinitePoset(leq3)
    end

    @testset "FinitePoset unchecked constructor" begin
        # Same leq3 as above: non-transitive but reflexive/antisymmetric.
        leq3 = falses(3, 3)
        for i in 1:3
            leq3[i, i] = true
        end
        leq3[1, 2] = true
        leq3[2, 3] = true

        # With check=false we should be able to build it (unsafe).
        P_bad = FF.FinitePoset(leq3; check=false)
        @test P_bad.n == 3
    end

    @testset "cover_edges correctness and caching" begin
        Random.seed!(12345)

        function transitive_closure!(R::BitMatrix)
            n = size(R, 1)
            @inbounds for k in 1:n
                for i in 1:n
                    if R[i, k]
                        for j in 1:n
                            R[i, j] |= R[k, j]
                        end
                    end
                end
            end
            return R
        end

        function naive_cover_edges(leq::BitMatrix)
            n = size(leq, 1)
            mat = falses(n, n)
            edges = Tuple{Int,Int}[]
            @inbounds for i in 1:n
                for j in 1:n
                    if i != j && leq[i, j]
                        covered = true
                        for k in 1:n
                            if k != i && k != j && leq[i, k] && leq[k, j]
                                covered = false
                                break
                            end
                        end
                        if covered
                            mat[i, j] = true
                            push!(edges, (i, j))
                        end
                    end
                end
            end
            return mat, edges
        end

        for n in (5, 8)
            # Random strict order edges (upper triangular), then take transitive closure.
            R = falses(n, n)
            @inbounds for i in 1:n
                R[i, i] = true
                for j in (i + 1):n
                    if rand() < 0.25
                        R[i, j] = true
                    end
                end
            end
            transitive_closure!(R)

            P = FF.FinitePoset(R; check=false)
            C1 = FF.cover_edges(P)

            # Regression tests: CoverEdges is iterable over edges, so BitMatrix(C) and
            # Matrix(C) must be explicitly specialized (otherwise Base tries the
            # generic iterator constructor and errors).
            @test BitMatrix(C1) === convert(BitMatrix, C1)
            @test Matrix(C1) == Matrix(BitMatrix(C1))

            C2 = FF.cover_edges(P)
            @test C1 === C2  # cached
            @test P.cache.cover_edges === C1

            # cached=false should force recomputation (new object) but same result.
            C3 = FF.cover_edges(P; cached=false)
            @test C3 !== C1
            @test BitMatrix(C3) == BitMatrix(C1)
            @test findall(C3) == findall(C1)

            FF.clear_cover_cache!(P)
            @test P.cache.cover_edges === nothing
            C4 = FF.cover_edges(P)
            @test C4 !== C1
            @test BitMatrix(C4) == BitMatrix(C1)

            mat_ref, edges_ref = naive_cover_edges(R)
            @test BitMatrix(C1) == mat_ref
            @test findall(C1) == edges_ref
        end
    end


    @testset "Upset/downset closure and generators" begin
        P4 = chain_poset(4)
        S = falses(P4.n)
        S[2] = true

        Ucl = FF.upset_closure(P4, S)
        Dcl = FF.downset_closure(P4, S)

        @test Ucl.mask == BitVector([false, true, true, true])
        @test Dcl.mask == BitVector([true, true, false, false])

        Pdia = diamond_poset()
        Ugen = FF.upset_from_generators(Pdia, [2, 3])
        Dgen = FF.downset_from_generators(Pdia, [2, 3])

        @test Ugen.mask == BitVector([false, true, true, true])  # {2,3,4}
        @test Dgen.mask == BitVector([true, true, true, false])  # {1,2,3}
    end

    @testset "cover_edges on a non-chain (diamond) poset" begin
        Pdia = diamond_poset()
        C = FF.cover_edges(Pdia)
        edges = Set([(I[1], I[2]) for I in findall(C)])
        @test edges == Set([(1, 2), (1, 3), (2, 4), (3, 4)])
    end


    @testset "Dense phi branch and dense_to_sparse_K" begin
        P3 = chain_poset(3)
        U2 = FF.principal_upset(P3, 2)
        D2 = FF.principal_downset(P3, 2)

        # Dense 1x1 phi exercises the non-sparse path in _check_monomial_condition.
        phi_dense = reshape(K[c(1)], 1, 1)
        Mdense = FF.FringeModule{K}(P3, [U2], [D2], phi_dense; field=field)
        @test FF.fiber_dimension(Mdense, 2) == 1

        # Converting dense -> sparse should preserve values.
        phi_sparse = FF.dense_to_sparse_K(phi_dense)
        @test phi_sparse isa SparseMatrixCSC{K, Int}
        @test Matrix(phi_sparse) == phi_dense

        # Sparse phi should behave the same.
        Msparse = FF.FringeModule{K}(P3, [U2], [D2], phi_sparse; field=field)
        @test FF.fiber_dimension(Msparse, 2) == 1

        # Dense constructor should reject nonzero phi when U cap D is empty.
        U3 = FF.principal_upset(P3, 3)
        D1 = FF.principal_downset(P3, 1)
        @test_throws AssertionError FF.FringeModule{K}(P3, [U3], [D1], reshape(K[c(1)], 1, 1); field=field)
    end
end

@testset "build_cache! two-phase warmup" begin
    old = FF.updown_cache_policy()
    try
        FF.set_updown_cache_policy!(mode=:always, finite_threshold=0, generic_threshold=0)
        P = chain_poset(6)
        @test P.cache.cover === nothing
        @test P.cache.upsets === nothing
        @test P.cache.downsets === nothing

        FF.build_cache!(P; cover=true, updown=true)
        @test P.cache.cover !== nothing
        @test P.cache.upsets !== nothing
        @test P.cache.downsets !== nothing

        cc1 = P.cache.cover
        FF.build_cache!(P; cover=true, updown=true)
        @test P.cache.cover === cc1
    finally
        FF.set_updown_cache_policy!(;
            mode=old.mode,
            finite_threshold=old.finite_threshold,
            generic_threshold=old.generic_threshold,
        )
    end
end

@testset "HomExt pi0_count (Prop 3.10 sanity)" begin
    # Disjoint union of two chains: intersection has two connected components.
    P = disjoint_two_chains_poset()

    U = FF.upset_from_generators(P, [2, 4])      # {2,4}
    D = FF.downset_from_generators(P, [2, 4])    # {1,2,3,4}

    @test HE.pi0_count(P, U, D) == 2

    # Connected case: chain
    Q = chain_poset(3)
    U2 = FF.principal_upset(Q, 2)
    D2 = FF.principal_downset(Q, 2)
    @test HE.pi0_count(Q, U2, D2) == 1
end


@testset "By-hand theory checks (Miller Prop 3.10 and chain intervals)" begin

    @testset "Prop 3.10: Hom for indicator modules uses pi0 on Hasse graph" begin
        # Disjoint union of two chains: 1<2 and 3<4 (no relations between components).
        P = disjoint_two_chains_poset()

        allU = FF.upset_closure(P, trues(P.n))
        allD = FF.downset_closure(P, trues(P.n))

        # U = upset generated by {2,4} is just {2,4}.
        # D = downset generated by {2,4} is the whole poset.
        U = FF.upset_from_generators(P, [2, 4])
        D = FF.downset_from_generators(P, [2, 4])

        # k[U] as a fringe image: k[U] -> k[Q] (death = all).
        kU = one_by_one_fringe(P, U, allD; field=field)

        # k[D] as a fringe image: k[Q] -> k[D] (birth = all).
        kD = one_by_one_fringe(P, allU, D; field=field)

        # Prop 3.10(1): dim Hom_Q(k[U], k[D]) = number of connected components of U cap D.
        # Here U cap D = U = {2,4}, which has 2 components.
        @test HE.pi0_count(P, U, D) == 2
        @test FF.hom_dimension(kU, kD) == 2
        @test FF.hom_dimension(kU, kD) == HE.pi0_count(P, U, D)

        # Prop 3.10(2): For upsets U',U, dim Hom(k[U'], k[U]) = number of components of U'
        # that lie inside U.
        Uprime = FF.upset_from_generators(P, [2, 4])  # {2,4} (2 components)
        Usmall = FF.upset_from_generators(P, [2])     # {2} (1 component)
        kUprime = one_by_one_fringe(P, Uprime, allD; field=field)
        kUsmall = one_by_one_fringe(P, Usmall, allD; field=field)

        @test FF.hom_dimension(kUprime, kUsmall) == 1
        @test FF.hom_dimension(kUsmall, kUprime) == 1

        # End(k[Uprime]) has one scalar per connected component of Uprime.
        @test FF.hom_dimension(kUprime, kUprime) == 2

        # Prop 3.10(3): For downsets D,D', dim Hom(k[D], k[D']) = number of components of D'
        # that lie inside D (note the direction).
        Dall = allD
        Dsmall = FF.downset_from_generators(P, [2])   # {1,2}
        kDall = one_by_one_fringe(P, allU, Dall; field=field)
        kDsmall = one_by_one_fringe(P, allU, Dsmall; field=field)

        @test FF.hom_dimension(kDall, kDsmall) == 1
        @test FF.hom_dimension(kDsmall, kDall) == 1
        @test FF.hom_dimension(kDall, kDall) == 2
    end

    @testset "Chain intervals: Hom and Ext^1 formula for type A_n" begin
        # On a finite chain, an interval module I[a,b] is indecomposable.
        # For the equioriented A_n quiver (a chain poset), the category is hereditary,
        # so Ext^t = 0 for t >= 2. Ext^1 is 1 exactly for the "one-step overlap" pattern:
        # Ext^1(I[a,b], I[c,d]) = 1 iff a < c <= b+1 <= d (and b < n so b+1 exists).
        #
        # We realize I[a,b] as the image of k[U_a] -> k[D_b] with phi = 1, where:
        #   U_a = principal upset at a  (degrees >= a)
        #   D_b = principal downset at b (degrees <= b)
        n = 4
        P = chain_poset(n)

        function interval_module(a::Int, b::Int)
            @assert 1 <= a <= b <= n
            U = FF.principal_upset(P, a)
            D = FF.principal_downset(P, b)
            return one_by_one_fringe(P, U, D; field=field)
        end

        intervals = [(a, b) for a in 1:n for b in a:n]
        mods = Dict{Tuple{Int,Int}, FF.FringeModule{K}}()
        for (a, b) in intervals
            mods[(a, b)] = interval_module(a, b)
        end

        for (a, b) in intervals, (c, d) in intervals
            M = mods[(a, b)]
            N = mods[(c, d)]

            ext = DF.ext_dimensions_via_indicator_resolutions(M, N; maxlen=5)

            # Hom(I[a,b], I[c,d]) is 1 iff c <= a <= d <= b, else 0.
            expected_hom = (c <= a && a <= d && d <= b) ? 1 : 0

            # Ext^1(I[a,b], I[c,d]) is 1 iff a < c <= b+1 <= d, else 0.
            expected_ext1 = (b < n && a < c && c <= b + 1 && b + 1 <= d) ? 1 : 0

            @test get(ext, 0, 0) == expected_hom
            @test get(ext, 1, 0) == expected_ext1
            @test get(ext, 2, 0) == 0
        end

        # Sanity-check the total complex sign convention: d_{t+1} * d_t == 0.
        # Pick a pair where Ext^1 is expected nonzero to ensure differentials are not trivial.
        M = interval_module(1, 2)   # I[1,2]
        N = interval_module(2, 3)   # I[2,3]

        F, dF = IR.upset_resolution(M; maxlen=5)
        E, dE = IR.downset_resolution(N; maxlen=5)

        dimsCt, dts = HE.build_hom_tot_complex(F, dF, E, dE)
        for t in 1:(length(dts) - 1)
            @test nnz(dts[t+1] * dts[t]) == 0
        end
    end

    @testset "Zero module edge case" begin
        # A fringe module with empty intersection but zero phi should be the zero module.
        P = chain_poset(3)
        U = FF.principal_upset(P, 3)      # {3}
        D = FF.principal_downset(P, 1)    # {1}
        phi0 = spzeros(K, 1, 1)          # must be zero since U cap D is empty
        M0 = FF.FringeModule{K}(P, [U], [D], phi0; field=field)

        for q in 1:P.n
            @test FF.fiber_dimension(M0, q) == 0
        end

        # Hom from/to zero should be 0.
        U2 = FF.principal_upset(P, 2)
        D2 = FF.principal_downset(P, 2)
        M = one_by_one_fringe(P, U2, D2; field=field)

        @test FF.hom_dimension(M0, M) == 0
        @test FF.hom_dimension(M, M0) == 0

        ext = DF.ext_dimensions_via_indicator_resolutions(M0, M; maxlen=5)
        @test get(ext, 0, 0) == 0
        @test get(ext, 1, 0) == 0
    end
end

@testset "one_by_one_fringe accepts Bool masks" begin
    P = chain_poset(5)

    # Upset {3,4,5} and downset {1,2,3} given as membership masks.
    U_mask = BitVector([i >= 3 for i in 1:P.n])
    D_mask = BitVector([i <= 3 for i in 1:P.n])

    H_mask  = PM.one_by_one_fringe(P, U_mask, D_mask, c(1); field=field)
    H_typed = PM.one_by_one_fringe(P, FF.Upset(P, U_mask), FF.Downset(P, D_mask), c(1); field=field)

    for q in 1:P.n
        @test FF.fiber_dimension(H_mask, q) == FF.fiber_dimension(H_typed, q)
    end

    # Also accept plain Vector{Bool} (not just BitVector).
    H_vec = PM.one_by_one_fringe(P, collect(U_mask), collect(D_mask), c(1); field=field)
    for q in 1:P.n
        @test FF.fiber_dimension(H_vec, q) == FF.fiber_dimension(H_typed, q)
    end
end

end # with_fields

@testset "Up/down cache policy tuning" begin
    old = FF.updown_cache_policy()
    try
        P = chain_poset(5)
        FF.set_updown_cache_policy!(mode=:always, finite_threshold=0, generic_threshold=0)
        _ = FF.upset_indices(P, 1)
        @test P.cache.upsets !== nothing
        @test P.cache.downsets !== nothing

        FF.clear_cover_cache!(P)
        FF.set_updown_cache_policy!(mode=:never, finite_threshold=10_000, generic_threshold=10_000)
        _ = FF.upset_indices(P, 1)
        @test P.cache.upsets === nothing
        @test P.cache.downsets === nothing

        Pc = FF.ProductOfChainsPoset((4, 4))
        FF.set_updown_cache_policy!(mode=:auto, finite_threshold=10_000, generic_threshold=10_000)
        _ = FF.upset_indices(Pc, 1)
        @test Pc.cache.upsets === nothing
        @test Pc.cache.downsets === nothing
    finally
        FF.set_updown_cache_policy!(;
            mode=old.mode,
            finite_threshold=old.finite_threshold,
            generic_threshold=old.generic_threshold,
        )
    end
end
