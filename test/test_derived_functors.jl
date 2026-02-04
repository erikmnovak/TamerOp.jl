using Test

using LinearAlgebra
import Base.Threads

# This file assumes the helper constructors defined in runtests.jl:
# - chain_poset(n)
# - diamond_poset()
# - one_by_one_fringe(P, U, D)
# - boolean_lattice_B3_poset()

function _is_real_field(field)
    return field isa CM.RealField
end

function _field_tol(field)
    field isa CM.RealField || return 0.0
    return field.atol + field.rtol
end

@testset "Derived functors across fields (A2)" begin
    P = chain_poset(2)
    U1 = FF.principal_upset(P, 1)
    D1 = FF.principal_downset(P, 1)
    U2 = FF.principal_upset(P, 2)
    D2 = FF.principal_downset(P, 2)

    with_fields(FIELDS_FULL) do field
        K = CM.coeff_type(field)
        S1 = one_by_one_fringe(P, U1, D1; scalar=one(K), field=field)
        S2 = one_by_one_fringe(P, U2, D2; scalar=one(K), field=field)

        ext12 = DF.ext_dimensions_via_indicator_resolutions(S1, S2; maxlen=3)
        ext21 = DF.ext_dimensions_via_indicator_resolutions(S2, S1; maxlen=3)
        ext11 = DF.ext_dimensions_via_indicator_resolutions(S1, S1; maxlen=3)
        ext22 = DF.ext_dimensions_via_indicator_resolutions(S2, S2; maxlen=3)

        @test get(ext12, 0, 0) == 0
        @test get(ext12, 1, 0) == 1
        @test get(ext21, 0, 0) == 0
        @test get(ext21, 1, 0) == 0
        @test get(ext11, 0, 0) == 1
        @test get(ext11, 1, 0) == 0
        @test get(ext22, 0, 0) == 1
        @test get(ext22, 1, 0) == 0

        @test get(ext12, 0, 0) == FF.hom_dimension(S1, S2)
        @test get(ext21, 0, 0) == FF.hom_dimension(S2, S1)
        @test get(ext11, 0, 0) == FF.hom_dimension(S1, S1)
        @test get(ext22, 0, 0) == FF.hom_dimension(S2, S2)
    end
end

@testset "Resolution threading parity" begin
    if Threads.nthreads() > 1
        P = chain_poset(3)
        U = FF.principal_upset(P, 2)
        D = FF.principal_downset(P, 2)
        FQ = CM.QQField()
        H = one_by_one_fringe(P, U, D; scalar=CM.coerce(FQ, 1), field=FQ)
        M = IR.pmodule_from_fringe(H)
        res = PM.ResolutionOptions(maxlen=2)

        R_serial = PM.projective_resolution(M, res; threads=false)
        R_thread = PM.projective_resolution(M, res; threads=true)
        @test R_thread.gens == R_serial.gens
        @test R_thread.d_mat == R_serial.d_mat

        E_serial = PM.injective_resolution(M, res; threads=false)
        E_thread = PM.injective_resolution(M, res; threads=true)
        @test E_thread.gens == E_serial.gens
        @test E_thread.d_mor == E_serial.d_mor
    end
end

@testset "Characteristic-sensitive rank (A1)" begin
    P = chain_poset(1)
    U = FF.principal_upset(P, 1)
    D = FF.principal_downset(P, 1)

    FQ = CM.QQField()
    KQ = CM.coeff_type(FQ)
    @inline cq(x) = CM.coerce(FQ, x)
    Hqq = one_by_one_fringe(P, U, D; scalar=cq(2), field=FQ)
    Hf2 = one_by_one_fringe(P, U, D; scalar=2, field=CM.F2())

    Mqq = IR.pmodule_from_fringe(Hqq)
    Mf2 = IR.pmodule_from_fringe(Hf2)

    @test FF.fiber_dimension(Hqq, 1) == 1
    @test FF.fiber_dimension(Hf2, 1) == 0
    @test Mqq.dims[1] == 1
    @test Mf2.dims[1] == 0
end

@testset "Characteristic-sensitive Hom/Ext (A2)" begin
    P = chain_poset(2)
    U1 = FF.principal_upset(P, 1)
    D1 = FF.principal_downset(P, 1)
    U2 = FF.principal_upset(P, 2)
    D2 = FF.principal_downset(P, 2)

    # One-generator modules with scalar 2: nonzero over QQ, zero over F2.
    FQ = CM.QQField()
    KQ = CM.coeff_type(FQ)
    @inline cq(x) = CM.coerce(FQ, x)
    Hqq = one_by_one_fringe(P, U1, D1; scalar=cq(2), field=FQ)
    Hf2 = one_by_one_fringe(P, U1, D1; scalar=2, field=CM.F2())

    Mqq = IR.pmodule_from_fringe(Hqq)
    Mf2 = IR.pmodule_from_fringe(Hf2)

    @test Mqq.dims[1] == 1
    @test Mf2.dims[1] == 0

    extqq = DF.ext_dimensions_via_indicator_resolutions(Hqq, Hqq; maxlen=2)
    extf2 = DF.ext_dimensions_via_indicator_resolutions(Hf2, Hf2; maxlen=2)

    @test get(extqq, 0, 0) == 1
    @test get(extf2, 0, 0) == 0
end


# -----------------------------------------------------------------------------
# Small helper: direct sum of poset-modules and the split short exact sequence
# -----------------------------------------------------------------------------

function direct_sum_with_split_sequence(A::MD.PModule{K}, C::MD.PModule{K}) where {K}
    Q = A.Q
    field = A.field
    @test FF.poset_equal(Q, C.Q)
    A.field == C.field || error("direct_sum_with_split_sequence: field mismatch")

    dimsB = [A.dims[v] + C.dims[v] for v in 1:Q.n]

    # Block-diagonal structure maps on cover edges.
    edge_mapsB = Dict{Tuple{Int,Int}, Matrix{K}}()
    for (i, j) in FF.cover_edges(Q)
        # We are iterating over cover edges of Q. The module edge-map store
        # guarantees cover-edge maps exist (missing entries are filled with zeros
        # at construction), so we can index directly and avoid Base.get().
        Aij = A.edge_maps[i, j]
        Cij = C.edge_maps[i, j]

        top    = hcat(Aij, CM.zeros(field, size(Aij,1), size(Cij,2)))
        bottom = hcat(CM.zeros(field, size(Cij,1), size(Aij,2)), Cij)
        edge_mapsB[(i,j)] = vcat(top, bottom)
    end

    B = MD.PModule{K}(Q, dimsB, edge_mapsB; field=field)

    # Inclusion i: A -> A oplus C (first summand)
    comps_i = Vector{Matrix{K}}(undef, Q.n)
    for v in 1:Q.n
        comps_i[v] = vcat(CM.eye(field, A.dims[v]),
                          CM.zeros(field, C.dims[v], A.dims[v]))
    end
    i = MD.PMorphism(A, B, comps_i)

    # Projection p: A oplus C -> C (second summand)
    comps_p = Vector{Matrix{K}}(undef, Q.n)
    for v in 1:Q.n
        comps_p[v] = hcat(CM.zeros(field, C.dims[v], A.dims[v]),
                          CM.eye(field, C.dims[v]))
    end
    p = MD.PMorphism(B, C, comps_p)

    return B, i, p
end

@testset "Minimality diagnostics for projective/injective resolutions" begin
    with_fields(FIELDS_FULL) do field
        # Use a small poset where minimal resolutions are nontrivial.
        P = diamond_poset()
        K = CM.coeff_type(field)
        c(x) = CM.coerce(field, x)

        # Simple at vertex 1.
        S1 = IR.pmodule_from_fringe(
            one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 1); scalar=one(K), field=field)
        )

        # Projective resolution should be minimal (and certified minimal by the checker).
        resP = DF.projective_resolution(S1, PM.ResolutionOptions(maxlen=4))
        repP = PM.minimality_report(resP)
        @test repP.cover_ok
        @test repP.minimal
        @test isempty(repP.diagonal_violations)
        @test PM.is_minimal(resP)
        PM.assert_minimal(resP)

        # Passing ResolutionOptions(minimal=true, check=true) should succeed and return a minimal resolution.
        resPmin = DF.projective_resolution(S1, PM.ResolutionOptions(maxlen=4, minimal=true, check=true))
        @test PM.is_minimal(resPmin)

        # Corrupt the resolution by inserting a diagonal coefficient in d^1 (degree 1 -> 0),
        # which should violate minimality (a generator mapping to itself at the same vertex).
        if length(resP.gens) >= 2 && !isempty(resP.gens[1]) && !isempty(resP.d_mat)
            gens_bad = [copy(g) for g in resP.gens]
            d_mat_bad = copy(resP.d_mat)

            # Ensure we can reference a "same vertex" pair across cod/domain by duplicating
            # an existing vertex label in degree 1.
            push!(gens_bad[2], gens_bad[1][1])

            D = d_mat_bad[1]
            extra = spzeros(K, size(D, 1), 1)
            extra[1, 1] = c(1)   # explicit diagonal coefficient
            d_mat_bad[1] = hcat(D, extra)

            res_bad = PM.ProjectiveResolution(resP.M, resP.Pmods, gens_bad, resP.d_mor, d_mat_bad, resP.aug)
            rep_bad = PM.minimality_report(res_bad; check_cover=true)
            @test !rep_bad.minimal
            @test !isempty(rep_bad.diagonal_violations)
            @test !PM.is_minimal(res_bad)
            @test_throws ErrorException PM.assert_minimal(res_bad)
        end

        # Injective resolution: also expected to be minimal for these constructions.
        resI = DF.injective_resolution(S1, PM.ResolutionOptions(maxlen=4))
        repI = PM.minimality_report(resI)
        @test repI.hull_ok
        @test repI.minimal
        @test isempty(repI.diagonal_violations)
        @test PM.is_minimal(resI)
        PM.assert_minimal(resI)

        resImin = DF.injective_resolution(S1, PM.ResolutionOptions(maxlen=4, minimal=true, check=true))
        @test PM.is_minimal(resImin)
    end
end

@testset "Betti extraction from projective resolutions (diamond)" begin
    with_fields(FIELDS_FULL) do field
        P = diamond_poset()
        K = CM.coeff_type(field)

        # Simple modules S1..S4 as fringe modules, then as poset-modules.
        Sm = Vector{MD.PModule{K}}(undef, P.n)
        for v in 1:P.n
            Hv = one_by_one_fringe(P, FF.principal_upset(P, v), FF.principal_downset(P, v); scalar=one(K), field=field)
            Sm[v] = IR.pmodule_from_fringe(Hv)
        end
        S1, S2, S3, S4 = Sm

        # Minimal projective resolution of S1 on the diamond should have:
        # P0 = P1, P1 = P2 oplus P3, P2 = P4.
        res = DF.projective_resolution(S1, PM.ResolutionOptions(maxlen=2))
        b = DF.betti(res)

        @test length(b) == 4
        @test b[(0, 1)] == 1
        @test b[(1, 2)] == 1
        @test b[(1, 3)] == 1
        @test b[(2, 4)] == 1

        Btbl = DF.betti_table(res)
        @test Btbl[1,1] == 1
        @test Btbl[2,2] == 1
        @test Btbl[2,3] == 1
        @test Btbl[3,4] == 1
    end
end


@testset "Yoneda product (diamond: Ext^1 x Ext^1 -> Ext^2)" begin
    with_fields(FIELDS_FULL) do field
        P = diamond_poset()
        K = CM.coeff_type(field)
        c(x) = CM.coerce(field, x)

        Sm = Vector{MD.PModule{K}}(undef, P.n)
        for v in 1:P.n
            Hv = one_by_one_fringe(P, FF.principal_upset(P, v), FF.principal_downset(P, v); scalar=one(K), field=field)
            Sm[v] = IR.pmodule_from_fringe(Hv)
        end
        S1, S2, S3, S4 = Sm

        # Compute the target Ext space once so coordinates are comparable.
        E14 = DF.Ext(S1, S4, PM.DerivedFunctorOptions(maxdeg=2))
        @test PM.dim(E14, 2) == 1

        # Via the chain 1 -> 2 -> 4
        E24 = DF.Ext(S2, S4, PM.DerivedFunctorOptions(maxdeg=1))
        E12 = DF.Ext(S1, S2, PM.DerivedFunctorOptions(maxdeg=2))  # needs tmax >= 2 because p+q = 2
        @test PM.dim(E24, 1) == 1
        @test PM.dim(E12, 1) == 1

        beta = [c(1)]
        alpha = [c(1)]
        _, coords_2 = PM.DerivedFunctors.yoneda_product(E24, 1, beta, E12, 1, alpha; ELN=E14)
        @test coords_2[1] != 0

        # Via the chain 1 -> 3 -> 4
        E34 = DF.Ext(S3, S4, PM.DerivedFunctorOptions(maxdeg=1))
        E13 = DF.Ext(S1, S3, PM.DerivedFunctorOptions(maxdeg=2))
        _, coords_3 = PM.DerivedFunctors.yoneda_product(E34, 1, [c(1)], E13, 1, [c(1)]; ELN=E14)
        @test coords_3[1] != 0

        # In a 1-dimensional target, the two products must be proportional.
        # With our deterministic lifts/basis choices, they should agree up to sign.
        if _is_real_field(field)
            @test isapprox(coords_2[1], coords_3[1]; atol=_field_tol(field), rtol=field.rtol) ||
                  isapprox(coords_2[1], -coords_3[1]; atol=_field_tol(field), rtol=field.rtol)
        else
            @test coords_2[1] == coords_3[1] || coords_2[1] == -coords_3[1]
        end
    end
end


@testset "Yoneda associativity sanity check (B3, degree 3 is nonzero)" begin
    with_fields(FIELDS_FULL) do field
        P = boolean_lattice_B3_poset()
        K = CM.coeff_type(field)
        c(x) = CM.coerce(field, x)

        Sm = Vector{MD.PModule{K}}(undef, P.n)
        for v in 1:P.n
            Hv = one_by_one_fringe(P, FF.principal_upset(P, v), FF.principal_downset(P, v); scalar=one(K), field=field)
            Sm[v] = IR.pmodule_from_fringe(Hv)
        end

        # Element numbering is the bitmask order described in boolean_lattice_B3_poset().
        S0   = Sm[1]  # {}
        S1   = Sm[2]  # {1}
        S12  = Sm[5]  # {1,2}
        S123 = Sm[8]  # {1,2,3}

        # Target space: Ext^3(S0, S123) should be 1-dimensional for B3.
        E03 = DF.Ext(S0, S123, PM.DerivedFunctorOptions(maxdeg=3))
        @test PM.dim(E03, 3) == 1

        # Degree-1 generators on the cover chain:
        #   {} -> {1} -> {1,2} -> {1,2,3}
        E23 = DF.Ext(S12, S123, PM.DerivedFunctorOptions(maxdeg=3))
        E12 = DF.Ext(S1,  S12, PM.DerivedFunctorOptions(maxdeg=3))
        E01 = DF.Ext(S0,  S1,   PM.DerivedFunctorOptions(maxdeg=3))

        @test PM.dim(E23, 1) == 1
        @test PM.dim(E12, 1) == 1
        @test PM.dim(E01, 1) == 1

        # Intermediate targets in degree 2 (also 1-dimensional for this choice).
        E13 = DF.Ext(S1,  S123, PM.DerivedFunctorOptions(maxdeg=3))
        E02 = DF.Ext(S0,  S12, PM.DerivedFunctorOptions(maxdeg=3))
        @test PM.dim(E13, 2) == 1
        @test PM.dim(E02, 2) == 1

        # Left bracketing: (e23 * e12) * e01
        _, x = PM.DerivedFunctors.yoneda_product(E23, 1, [c(1)], E12, 1, [c(1)]; ELN=E13)  # x in Ext^2(S1,S123)
        _, left = PM.DerivedFunctors.yoneda_product(E13, 2, x, E01, 1, [c(1)]; ELN=E03)

        # Right bracketing: e23 * (e12 * e01)
        _, y = PM.DerivedFunctors.yoneda_product(E12, 1, [c(1)], E01, 1, [c(1)]; ELN=E02)  # y in Ext^2(S0,S12)
        _, right = PM.DerivedFunctors.yoneda_product(E23, 1, [c(1)], E02, 2, y; ELN=E03)

        # Nontriviality + associativity up to sign in a 1-dimensional target.
        @test left[1] != 0
        @test right[1] != 0
        if _is_real_field(field)
            @test isapprox(left[1], right[1]; atol=_field_tol(field), rtol=field.rtol) ||
                  isapprox(left[1], -right[1]; atol=_field_tol(field), rtol=field.rtol)
        else
            @test left[1] == right[1] || left[1] == -right[1]
        end
    end
end


@testset "Connecting homomorphisms: split exact sequences give zero maps" begin
    with_fields(FIELDS_FULL) do field
        Q = chain_poset(4)
        K = CM.coeff_type(field)

        Sm = Vector{MD.PModule{K}}(undef, Q.n)
        for v in 1:Q.n
            Hv = one_by_one_fringe(Q, FF.principal_upset(Q, v), FF.principal_downset(Q, v); scalar=one(K), field=field)
            Sm[v] = IR.pmodule_from_fringe(Hv)
        end
        S1, S2, S3, S4 = Sm

        # -------------------------------------------------------------------------
        # Second argument LES: 0 -> A -> A oplus C -> C -> 0
        # delta : Ext^t(M,C) -> Ext^{t+1}(M,A)
        # Choose M=S1, C=S3 (Ext^2), A=S4 (Ext^3), t=2.
        # -------------------------------------------------------------------------
        A = S4
        C = S3
        B, i, p = direct_sum_with_split_sequence(A, C)

        # We test delta^2 : Ext^2(M,C) -> Ext^3(M,A), so we need the resolution through degree t+1 = 3.
        t = 2
        resM = DF.projective_resolution(S1, PM.ResolutionOptions(maxlen=t+1))   # i.e. maxlen=3
        EMA = DF.Ext(resM, A)
        EMB = DF.Ext(resM, B)
        EMC = DF.Ext(resM, C)

        delta2 = PM.connecting_hom(EMA, EMB, EMC, i, p; t=2)
        if _is_real_field(field)
            @test norm(Matrix(delta2)) <= _field_tol(field)
        else
            @test all(delta2 .== 0)
        end

        # -------------------------------------------------------------------------
        # First argument LES: 0 -> A -> A oplus C -> C -> 0
        # delta : Ext^t(A,N) -> Ext^{t+1}(C,N)
        # Choose N=S4, A=S3 (Ext^1), C=S2 (Ext^2), t=1.
        # -------------------------------------------------------------------------
        A1 = S3
        C1 = S2
        B1, i1, p1 = direct_sum_with_split_sequence(A1, C1)

        resN = DF.injective_resolution(S4, PM.ResolutionOptions(maxlen=2))
        EA = PM.ExtInjective(A1, resN)
        EB = PM.ExtInjective(B1, resN)
        EC = PM.ExtInjective(C1, resN)

        delta1 = PM.connecting_hom_first(EA, EB, EC, i1, p1; t=1)
        if _is_real_field(field)
            @test norm(Matrix(delta1)) <= _field_tol(field)
        else
            @test all(delta1 .== 0)
        end
    end
end

@testset "Connecting homomorphisms: nonsplit extension gives nonzero delta" begin
    with_fields(FIELDS_FULL) do field
        # On the 2-element chain 1<2, Ext^1(S1,S2) has dimension 1.
        # The interval module k[1,2] (dims [1,1] and identity map along 1<2)
        # is a nonsplit extension 0 -> S2 -> k[1,2] -> S1 -> 0.
        P = chain_poset(2)
        K = CM.coeff_type(field)

        S1 = IR.pmodule_from_fringe(one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 1); scalar=one(K), field=field))
        S2 = IR.pmodule_from_fringe(one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 2); scalar=one(K), field=field))
        I12 = IR.pmodule_from_fringe(one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 2); scalar=one(K), field=field))

        # i: S2 -> I12 is inclusion on stalk 2.
        comps_i = [CM.zeros(field, I12.dims[v], S2.dims[v]) for v in 1:P.n]
        comps_i[2] = CM.eye(field, 1)
        i = MD.PMorphism(S2, I12, comps_i)

        # p: I12 -> S1 is projection on stalk 1.
        comps_p = [CM.zeros(field, S1.dims[v], I12.dims[v]) for v in 1:P.n]
        comps_p[1] = CM.eye(field, 1)
        p = MD.PMorphism(I12, S1, comps_p)

        # Fix M = S1. Then delta^0: Hom(S1,S1) -> Ext^1(S1,S2) sends id to the extension class,
        # so it must be nonzero (hence rank 1, since both sides are 1-dimensional).
        resM = DF.projective_resolution(S1, PM.ResolutionOptions(maxlen=2))
        EMA = DF.Ext(resM, S2)
        EMB = DF.Ext(resM, I12)
        EMC = DF.Ext(resM, S1)
        delta0 = PM.connecting_hom(EMA, EMB, EMC, i, p; t=0)
        @test PM.FieldLinAlg.rank(field, delta0) == 1

        # The packaged long exact sequence should expose the same delta^0.
        les = PM.ExtLongExactSequenceSecond(S1, S2, I12, S1, i, p, PM.DerivedFunctorOptions(maxdeg=0))
        @test PM.FieldLinAlg.rank(field, les.delta[1]) == 1
        if _is_real_field(field)
            @test norm(Matrix(les.delta[1]) - Matrix(delta0)) <= _field_tol(field)
        else
            @test Matrix(les.delta[1]) == Matrix(delta0)
        end
    end
end

@testset "ExtAlgebra: cached multiplication agrees with yoneda_product" begin
    with_fields(FIELDS_FULL) do field
        P = diamond_poset()
        K = CM.coeff_type(field)
        c(x) = CM.coerce(field, x)

        # Build simples S1..S4 as 1x1 fringe modules, then as poset-modules.
        Sm = Vector{MD.PModule{K}}(undef, P.n)
        for v in 1:P.n
            Hv = one_by_one_fringe(P, FF.principal_upset(P, v), FF.principal_downset(P, v); scalar=one(K), field=field)
            Sm[v] = IR.pmodule_from_fringe(Hv)
        end
        S1, S2, S3, S4 = Sm

        # We take a direct sum with enough structure to produce nontrivial Ext in degree 1
        # and allow products in degree 2 on the diamond.
        #
        # M = S1 oplus S2 oplus S4 is small but already contains:
        # - degree-1 extension classes among summands, and
        # - degree-2 composites (in the full Ext(M,M)).
        M12, _, _ = direct_sum_with_split_sequence(S1, S2)
        M, _, _ = direct_sum_with_split_sequence(M12, S4)

        A = PM.ExtAlgebra(M, PM.DerivedFunctorOptions(maxdeg=2))
        E = A.E

        # Sanity: dimensions agree with the underlying Ext space.
        for t in 0:E.tmax
            @test PM.dim(A, t) == PM.dim(E, t)
        end

        # The unit should act as both-sided identity on every homogeneous degree <= tmax.
        oneA = one(A)
        for t in 0:E.tmax
            dt = PM.dim(A, t)
            if dt == 0
                continue
            end

            # Deterministic "generic" element: (1,2,3,...,dt).
            x = PM.element(A, t, [c(i) for i in 1:dt])

            @test (oneA * x).coords == x.coords
            @test (x * oneA).coords == x.coords
        end

        # Cache behavior: after one multiplication in (p,q), the multiplication matrix should exist,
        # and repeated multiplication should not grow the cache.
        if E.tmax >= 2 && PM.dim(A, 1) > 0
            d1 = PM.dim(A, 1)
            x = PM.element(A, 1, [c(i) for i in 1:d1])
            y = PM.element(A, 1, [c(d1 - i + 1) for i in 1:d1])

            prod1 = x * y
            @test haskey(A.mult_cache, (1, 1))
            nkeys = length(A.mult_cache)

            prod2 = x * y
            @test length(A.mult_cache) == nkeys

            # Cached multiplication must match a direct call to the mathematical core (Yoneda product)
            # in the same Ext space and bases.
            _, coords_direct = PM.DerivedFunctors.yoneda_product(A.E, 1, x.coords, A.E, 1, y.coords; ELN=A.E)
            @test prod1.coords == coords_direct
            @test prod2.coords == coords_direct
        end

        # Associativity in the cached algebra (within truncation).
        #
        # On the diamond, Ext^3 is expected to vanish for many modules; we therefore test an
        # associativity instance that stays inside degrees <= 2 by including a degree-0 factor.
        if E.tmax >= 2 && PM.dim(A, 1) > 0
            d1 = PM.dim(A, 1)

            # First basis direction e_1
            a = PM.element(A, 1, [one(K); zeros(K, d1 - 1)])

            # Last basis direction e_{d1}
            b = PM.element(A, 1, [zeros(K, d1 - 1); one(K)])

            c0 = oneA

            @test ((a * b) * c0).coords == (a * (b * c0)).coords
            @test ((c0 * a) * b).coords == (c0 * (a * b)).coords
        end
    end
end

@testset "Sparse assembly replacements (dense->sparse removed)" begin
    with_fields(FIELDS_FULL) do field
        K = CM.coeff_type(field)
        c(x) = CM.coerce(field, x)

        # 0) _append_scaled_triplets! matches the old findnz(sparse(F)) pattern (up to matrix equality)
        let
            F = [c(1) c(0); c(2) c(3)]
            I1 = Int[]; J1 = Int[]; V1 = K[]
            PM.CoreModules._append_scaled_triplets!(I1, J1, V1, F, 10, 20; scale=c(2))

            S1 = sparse(I1, J1, V1, 12, 22)

            Ii, Ji, Vi = findnz(sparse(F))
            S2 = sparse(Ii .+ 10, Ji .+ 20, Vi .* c(2), 12, 22)

            @test S1 == S2
        end

        # Build a small poset and module with nontrivial (but diagonal) structure maps.
        P = chain_poset(3)
        dims = fill(2, 3)

        A12 = [c(1) c(0); c(0) c(2)]
        A23 = [c(3) c(0); c(0) c(5)]

        edge_maps = Dict{Tuple{Int,Int}, Matrix{K}}()
        edge_maps[(1,2)] = A12
        edge_maps[(2,3)] = A23

        M = MD.PModule{K}(P, dims, edge_maps; field=field)

        # 1) _coeff_matrix_upsets: new sparse assembly equals old dense->sparse reference
        let
            U1 = PM.FiniteFringe.principal_upset(P, 1)
            U2 = PM.FiniteFringe.principal_upset(P, 2)
            U3 = PM.FiniteFringe.principal_upset(P, 3)
            domU = [U1, U2, U3]
            codU = [U1, U2, U3]

            Cnew = DF._coeff_matrix_upsets(domU, codU)

            Cold_dense = CM.zeros(field, length(codU), length(domU))
            for i in 1:length(domU), j in 1:length(codU)
                cval = one(K)
                for v in codU[j]
                    if !(v in domU[i])
                        cval = zero(K)
                        break
                    end
                end
                Cold_dense[j,i] = cval
            end
            Cold = sparse(Cold_dense)

            @test Cnew == Cold
            @test Cnew isa SparseMatrixCSC{K,Int}
        end

        # 1b) _coeff_matrix_upsets(P, dom_bases, cod_bases, f) agrees with the top-vertex matrix
        #
        # On a chain poset, vertex n is a top element. At that vertex, all principal upsets
        # are active, so the morphism component matrix is the full coefficient matrix.
        let
            P = chain_poset(3)
            DF = PM.DerivedFunctors

            # A simple module with zero structure maps. This forces projective_cover to
            # pick generators at multiple vertices in a predictable way, producing a
            # nontrivial next-stage differential P1 -> P0.
            dims = fill(1, 3)
            edge_maps = Dict{Tuple{Int,Int}, Matrix{K}}()
            edge_maps[(1,2)] = CM.zeros(field, 1, 1)
            edge_maps[(2,3)] = CM.zeros(field, 1, 1)
            M = MD.PModule{K}(P, dims, edge_maps; field=field)

            P0, pi0, gens0 = IR.projective_cover(M)
            K, iota = PM.kernel_with_inclusion(pi0)
            P1, pi1, gens1 = IR.projective_cover(K)

            # Differential d : P1 -> P0
            d = DF.compose(iota, pi1)

            dom_bases = DF._flatten_gens_at(gens1)
            cod_bases = DF._flatten_gens_at(gens0)

            C = DF._coeff_matrix_upsets(P, dom_bases, cod_bases, d)

            # On the chain poset(3), vertex 3 is top, so all summands are active there.
            @test C == sparse(d.comps[3])
            @test C isa SparseMatrixCSC{K,Int}
        end


        # 2) _precompose_on_hom_cochains_from_projective_coeff: dense & sparse coeff match dense reference
        let
            dom_gens = [3,2,1]
            cod_gens = [3,2,1]

            dom_offsets = zeros(Int, length(dom_gens) + 1)
            cod_offsets = zeros(Int, length(cod_gens) + 1)
            for i in 1:length(dom_gens)
                dom_offsets[i+1] = dom_offsets[i] + M.dims[dom_gens[i]]
                cod_offsets[i+1] = cod_offsets[i] + M.dims[cod_gens[i]]
            end

            coeff_dense = [
                c(1) c(0) c(0);
                c(2) c(3) c(0);
                c(0) c(4) c(5)
            ]
            coeff_sparse = sparse(coeff_dense)

            # Old dense reference
            Fref_dense = CM.zeros(field, dom_offsets[end], cod_offsets[end])
            for i in 1:length(dom_gens), j in 1:length(cod_gens)
                c = coeff_dense[j,i]
                iszero(c) && continue
                ui = dom_gens[i]
                vj = cod_gens[j]
                A = MD.map_leq(M, vj, ui)
                rows = (dom_offsets[i] + 1):dom_offsets[i+1]
                cols = (cod_offsets[j] + 1):cod_offsets[j+1]
                Fref_dense[rows, cols] .+= c .* A
            end
            Fref = sparse(Fref_dense)

            F1 = DF._precompose_on_hom_cochains_from_projective_coeff(M, dom_gens, cod_gens, dom_offsets, cod_offsets, coeff_dense)
            F2 = DF._precompose_on_hom_cochains_from_projective_coeff(M, dom_gens, cod_gens, dom_offsets, cod_offsets, coeff_sparse)

            @test F1 == sparse(Fref_dense)
            @test F2 == sparse(Fref_dense)
            @test F1 isa SparseMatrixCSC{K,Int}
            @test F2 isa SparseMatrixCSC{K,Int}
        end

        # 3) _tensor_map_on_tor_chains_from_projective_coeff: dense & sparse coeff match dense reference
        let
            dom_bases = [1,2]
            cod_bases = [2,3]

            dom_offsets = zeros(Int, length(dom_bases) + 1)
            cod_offsets = zeros(Int, length(cod_bases) + 1)
            for i in 1:length(dom_bases)
                dom_offsets[i+1] = dom_offsets[i] + M.dims[dom_bases[i]]
            end
            for j in 1:length(cod_bases)
                cod_offsets[j+1] = cod_offsets[j] + M.dims[cod_bases[j]]
            end

            coeff_dense = [
                c(0) c(1);
                c(2) c(0)
            ]
            coeff_sparse = sparse(coeff_dense)

            Bref_dense = CM.zeros(field, cod_offsets[end], dom_offsets[end])
            for i in 1:length(dom_bases), j in 1:length(cod_bases)
                c = coeff_dense[j,i]
                iszero(c) && continue
                u = dom_bases[i]
                v = cod_bases[j]
                A = MD.map_leq(M, u, v)
                rows = (cod_offsets[j] + 1):cod_offsets[j+1]
                cols = (dom_offsets[i] + 1):dom_offsets[i+1]
                Bref_dense[rows, cols] = c .* A
            end
            Bref = sparse(Bref_dense)

            B1 = DF._tensor_map_on_tor_chains_from_projective_coeff(M, dom_bases, cod_bases, dom_offsets, cod_offsets, coeff_dense)
            B2 = DF._tensor_map_on_tor_chains_from_projective_coeff(M, dom_bases, cod_bases, dom_offsets, cod_offsets, coeff_sparse)

            @test B1 == Bref
            @test B2 == Bref
        end

        # 4) _tor_blockdiag_map_on_chains matches dense reference
        let
            f = IR.id_morphism(M)
            gens = [1,2,3]

            dom_offsets = zeros(Int, length(gens) + 1)
            cod_offsets = zeros(Int, length(gens) + 1)
            for i in 1:length(gens)
                dom_offsets[i+1] = dom_offsets[i] + f.dom.dims[gens[i]]
                cod_offsets[i+1] = cod_offsets[i] + f.cod.dims[gens[i]]
            end

            T = DF._tor_blockdiag_map_on_chains(f, gens, dom_offsets, cod_offsets)

            Tref_dense = CM.zeros(field, cod_offsets[end], dom_offsets[end])
            for i in 1:length(gens)
                u = gens[i]
                rows = (cod_offsets[i] + 1):cod_offsets[i+1]
                cols = (dom_offsets[i] + 1):dom_offsets[i+1]
                Tref_dense[rows, cols] = f.comps[u]
            end
            @test T == sparse(Tref_dense)
        end
    end
end

@testset "DerivedFunctors: downset postcompose coefficient solver" begin
    with_fields(FIELDS_FULL) do field
        # Minimal regression: 1-vertex poset, so all downsets are trivial and the fiberwise equation is
        # just C * F = G at the unique vertex.

        Q = FF.FinitePoset(trues(1, 1); check = false)
        K = CM.coeff_type(field)
        c(x) = CM.coerce(field, x)

        # X(u) has dimension 3, E(u) has 3 downset summands, Ep(u) has 2 downset summands.
        X  = MD.PModule{K}(Q, [3], Dict{Tuple{Int, Int}, Matrix{K}}(); field=field)
        E  = MD.PModule{K}(Q, [3], Dict{Tuple{Int, Int}, Matrix{K}}(); field=field)
        Ep = MD.PModule{K}(Q, [2], Dict{Tuple{Int, Int}, Matrix{K}}(); field=field)

        F1 = CM.eye(field, 3)
        G1 = [c(1) c(2) c(3);
              c(4) c(5) c(6)]

        f = MD.PMorphism(X, E,  [F1])
        g = MD.PMorphism(X, Ep, [G1])

        dom_bases = [1, 1, 1]
        cod_bases = [1, 1]
        act_dom = [collect(1:3)]
        act_cod = [collect(1:2)]

        C = DF._solve_downset_postcompose_coeff(f, g, dom_bases, cod_bases, act_dom, act_cod)
        @test C == G1
        @test C * F1 == G1

        # Inconsistent system: F = 0, G != 0 should throw.
        F0 = CM.zeros(field, 3, 3)
        f0 = MD.PMorphism(X, E, [F0])
        @test_throws ErrorException DF._solve_downset_postcompose_coeff(f0, g, dom_bases, cod_bases, act_dom, act_cod)
    end
end
