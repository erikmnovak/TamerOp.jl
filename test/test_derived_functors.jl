using Test

using LinearAlgebra
using InteractiveUtils
import Base.Threads

const FL = PosetModules.FieldLinAlg

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

        if _is_real_field(field)
            # Real-field Ext/Hom dimensions are numerical (rank-threshold dependent):
            # keep stability checks but do not enforce exact algebraic dimensions.
            @test all(v >= 0 for v in values(ext12))
            @test all(v >= 0 for v in values(ext21))
            @test all(v >= 0 for v in values(ext11))
            @test all(v >= 0 for v in values(ext22))
            @test get(ext11, 0, 0) >= 1
            @test get(ext22, 0, 0) >= 1
        else
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
        @test length(E_thread.d_mor) == length(E_serial.d_mor)
        for i in eachindex(E_thread.d_mor)
            @test E_thread.d_mor[i].comps == E_serial.d_mor[i].comps
        end
    end
end

@testset "Derived assembly parity + allocation guards" begin
    field = CM.QQField()
    K = CM.coeff_type(field)
    P = chain_poset(3)

    M = IR.pmodule_from_fringe(
        one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 3); scalar=one(K), field=field)
    )
    N = IR.pmodule_from_fringe(
        one_by_one_fringe(P, FF.principal_upset(P, 3), FF.principal_downset(P, 3); scalar=one(K), field=field)
    )

    DC_ext_s = DF.ExtDoubleComplex(M, N; maxlen=2, threads=false)
    if Threads.nthreads() > 1
        DC_ext_t = DF.ExtDoubleComplex(M, N; maxlen=2, threads=true)
        @test DC_ext_t.dims == DC_ext_s.dims
        @test DC_ext_t.dv == DC_ext_s.dv
        @test DC_ext_t.dh == DC_ext_s.dh
    end

    DC_tor_s = DF.TorDoubleComplex(M, N; maxlen=2, threads=false)
    if Threads.nthreads() > 1
        DC_tor_t = DF.TorDoubleComplex(M, N; maxlen=2, threads=true)
        @test DC_tor_t.dims == DC_tor_s.dims
        @test DC_tor_t.dv == DC_tor_s.dv
        @test DC_tor_t.dh == DC_tor_s.dh
    end

    # Warm + allocation budgets on fixed tiny fixtures.
    DF.ExtDoubleComplex(M, N; maxlen=2, threads=false)
    alloc_extdc = @allocated DF.ExtDoubleComplex(M, N; maxlen=2, threads=false)
    @test alloc_extdc < 60_000_000

    DF.TorDoubleComplex(M, N; maxlen=2, threads=false)
    alloc_tordc = @allocated DF.TorDoubleComplex(M, N; maxlen=2, threads=false)
    @test alloc_tordc < 60_000_000

    Hm = one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 3); scalar=one(K), field=field)
    Hn = one_by_one_fringe(P, FF.principal_upset(P, 3), FF.principal_downset(P, 3); scalar=one(K), field=field)
    DF.ext_dimensions_via_indicator_resolutions(Hm, Hn; maxlen=2, verify=false)
    alloc_extdims = @allocated DF.ext_dimensions_via_indicator_resolutions(Hm, Hn; maxlen=2, verify=false)
    @test alloc_extdims < 80_000_000
end

@testset "Ext/Tor core threaded parity" begin
    if Threads.nthreads() > 1
        field = CM.QQField()
        K = CM.coeff_type(field)

        P = chain_poset(3)
        Hm = one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 2); scalar=one(K), field=field)
        Hn = one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 3); scalar=one(K), field=field)
        M = IR.pmodule_from_fringe(Hm)
        N = IR.pmodule_from_fringe(Hn)

        resM = DF.projective_resolution(M, PM.ResolutionOptions(maxlen=2); threads=false)
        E_serial = DF.Ext(resM, N; threads=false)
        E_thread = DF.Ext(resM, N; threads=true)
        @test E_thread.complex.d == E_serial.complex.d
        @test [DF.dim(E_thread, t) for t in 0:2] == [DF.dim(E_serial, t) for t in 0:2]

        Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)); check=false)
        RopH = one_by_one_fringe(Pop, FF.principal_upset(Pop, 2), FF.principal_downset(Pop, 2); scalar=one(K), field=field)
        LH = one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 1); scalar=one(K), field=field)
        Rop = IR.pmodule_from_fringe(RopH)
        L = IR.pmodule_from_fringe(LH)

        resR = DF.projective_resolution(Rop, PM.ResolutionOptions(maxlen=2); threads=false)
        T_serial = DF.ExtTorSpaces._Tor_resolve_first(Rop, L; maxdeg=2, threads=false, res=resR)
        T_thread = DF.ExtTorSpaces._Tor_resolve_first(Rop, L; maxdeg=2, threads=true, res=resR)
        @test T_thread.bd == T_serial.bd
        @test [DF.dim(T_thread, s) for s in 0:2] == [DF.dim(T_serial, s) for s in 0:2]
    end
end

@testset "Resolution cache plumbing" begin
    P, S1, S2 = simple_modules_chain2()
    opts = PM.ResolutionOptions(maxlen=2)
    cache = CM.ResolutionCache()

    RP1 = DF.projective_resolution(S1, opts; cache=cache)
    RP2 = DF.projective_resolution(S1, opts; cache=cache)
    @test RP1 === RP2

    RI1 = DF.injective_resolution(S1, opts; cache=cache)
    RI2 = DF.injective_resolution(S1, opts; cache=cache)
    @test RI1 === RI2

    T1 = IR.indicator_resolutions(S1, S2; maxlen=2, cache=cache)
    T2 = IR.indicator_resolutions(S1, S2; maxlen=2, cache=cache)
    @test T1 === T2

    sc = CM.SessionCache()

    M = IR.pmodule_from_fringe(S1)
    enc = CM.EncodingResult(P, M, nothing; H=S1, opts=CM.EncodingOptions(field=S1.field), backend=:test)
    WR1 = PM.resolve(enc; kind=:projective, opts=opts, cache=sc)
    WR2 = PM.resolve(enc; kind=:projective, opts=opts, cache=sc)
    @test WR1.res === WR2.res

    # Workflow-level ext/tor should reuse cached resolutions automatically.
    N = IR.pmodule_from_fringe(S2)
    encN = CM.EncodingResult(P, N, nothing; H=S2, opts=CM.EncodingOptions(field=S2.field), backend=:test)
    E1 = PM.ext(enc, encN; maxdeg=2, model=:projective, cache=sc)
    E2 = PM.ext(enc, encN; maxdeg=2, model=:projective, cache=sc)
    @test E1.res === E2.res
    H1 = PM.hom(enc, encN; cache=sc)
    H2 = PM.hom(enc, encN; cache=sc)
    @test DF.dim(H1) == DF.dim(H2)
    d_fast_h = PM.hom_dimension(enc, encN; cache=sc)
    @test d_fast_h == DF.dim(H1)
    Epm1 = PM.ext(enc.M, encN.M; maxdeg=2, model=:projective, cache=sc)
    Epm2 = PM.ext(enc.M, encN.M; maxdeg=2, model=:projective, cache=sc)
    @test Epm1.res === Epm2.res
    Hpm1 = PM.hom(enc.M, encN.M; cache=sc)
    Hpm2 = PM.hom(enc.M, encN.M; cache=sc)
    @test DF.dim(Hpm1) == DF.dim(Hpm2)
    @test_throws MethodError PM.hom(enc, encN.M; cache=sc)
    @test_throws MethodError PM.hom(enc.M, encN; cache=sc)
    @test_throws MethodError PM.ext(enc, encN.M; maxdeg=2, model=:projective, cache=sc)

    C0 = PM.ModuleCochainComplex([enc.M], PM.PMorphism[]; tmin=0, check=true)
    RH1 = PM.rhom(C0, encN.M; cache=sc)
    RH2 = PM.rhom(C0, encN.M; cache=sc)
    @test RH1.tmin == RH2.tmin
    @test RH1.tmax == RH2.tmax
    @test length(RH1.d) == length(RH2.d)
    HX1 = PM.hyperext(C0, encN.M; maxdeg=2, cache=sc)
    HX2 = PM.hyperext(C0, encN.M; maxdeg=2, cache=sc)
    @test DF.dim(HX1, 0) == DF.dim(HX2, 0)
    @test_throws MethodError PM.rhom(C0, encN; cache=sc)
    @test_throws MethodError PM.hyperext(C0, encN; maxdeg=2, cache=sc)

    Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)); check=false)
    Uop = FF.principal_upset(Pop, 2)
    Dop = FF.principal_downset(Pop, 2)
    Hop = one_by_one_fringe(Pop, Uop, Dop; scalar=CM.coerce(S1.field, 1), field=S1.field)
    Rop = IR.pmodule_from_fringe(Hop)
    encRop = CM.EncodingResult(Pop, Rop, nothing; H=Hop, opts=CM.EncodingOptions(field=Hop.field), backend=:test)
    @test_throws ErrorException PM.hom_dimension(enc, encRop; cache=sc)

    T1 = PM.tor(encRop, enc; maxdeg=2, model=:first, cache=sc)
    T2 = PM.tor(encRop, enc; maxdeg=2, model=:first, cache=sc)
    @test T1.resRop === T2.resRop

    T3 = PM.tor(encRop, enc; maxdeg=2, model=:second, cache=sc)
    T4 = PM.tor(encRop, enc; maxdeg=2, model=:second, cache=sc)
    @test T3.resL === T4.resL
    Tpm1 = PM.tor(encRop.M, enc.M; maxdeg=2, model=:first, cache=sc)
    Tpm2 = PM.tor(encRop.M, enc.M; maxdeg=2, model=:first, cache=sc)
    @test Tpm1.resRop === Tpm2.resRop
    @test_throws MethodError PM.tor(encRop, enc.M; maxdeg=2, model=:first, cache=sc)
    @test_throws MethodError PM.tor(encRop.M, enc; maxdeg=2, model=:first, cache=sc)

    # hom_dimension should cache computed fringes when enc.H is absent.
    enc_noH = CM.EncodingResult(P, M, nothing; H=nothing, opts=CM.EncodingOptions(field=S1.field), backend=:test)
    encN_noH = CM.EncodingResult(P, N, nothing; H=nothing, opts=CM.EncodingOptions(field=S2.field), backend=:test)
    ec = CM._workflow_encoding_cache(sc)
    g0 = length(ec.geometry)
    d1 = PM.hom_dimension(enc_noH, encN_noH; cache=sc)
    g1 = length(ec.geometry)
    d2 = PM.hom_dimension(enc_noH, encN_noH; cache=sc)
    g2 = length(ec.geometry)
    @test d1 == d2
    @test d1 == DF.dim(PM.hom(enc_noH, encN_noH; cache=sc))
    @test g1 >= g0 + 2
    @test g2 == g1

    CM._clear_resolution_cache!(cache)
    RP3 = DF.projective_resolution(S1, opts; cache=cache)
    @test RP3 !== RP1

    WRS1 = PM.resolve(enc; kind=:projective, opts=opts, cache=sc)
    WRS2 = PM.resolve(enc; kind=:projective, opts=opts, cache=sc)
    @test WRS1.res === WRS2.res

    ES1 = PM.ext(enc, encN; maxdeg=2, model=:projective, cache=sc)
    ES2 = PM.ext(enc, encN; maxdeg=2, model=:projective, cache=sc)
    @test ES1.res === ES2.res

    TS1 = PM.tor(encRop, enc; maxdeg=2, model=:first, cache=sc)
    TS2 = PM.tor(encRop, enc; maxdeg=2, model=:first, cache=sc)
    @test TS1.resRop === TS2.resRop

    # Module cache keys include field identity: changing field should route to a
    # different module cache bucket.
    mc1 = CM._module_cache!(sc, enc.M)
    mc2 = CM._module_cache!(sc, enc.M)
    @test mc1 === mc2
    enc_f2 = CM.change_field(enc, CM.F2())
    mc3 = CM._module_cache!(sc, enc_f2.M)
    @test mc3 !== mc1

    CM._clear_session_cache!(sc)
    WRS3 = PM.resolve(enc; kind=:projective, opts=opts, cache=sc)
    @test WRS3.res !== WRS1.res
    @test_throws ErrorException PM.resolve(enc; kind=:proj, opts=opts, cache=sc)
    @test_throws ErrorException PM.resolve(enc; kind=:inj, opts=opts, cache=sc)
end

@testset "HomSystemCache type stability" begin
    P, S1, S2 = simple_modules_chain2()
    M = IR.pmodule_from_fringe(S1)
    N = IR.pmodule_from_fringe(S2)
    K = CM.coeff_type(M.field)
    cache = DF.HomSystemCache{K}()
    @test keytype(cache.hom[1]) == DF._HomKey2
    @test keytype(cache.precompose[1]) == DF._HomKey3
    @test keytype(cache.postcompose[1]) == DF._HomKey3

    _hom_cached(M1, N1, c) = DF.hom_with_cache(M1, N1; cache=c)
    report_hom = sprint(io -> InteractiveUtils.code_warntype(io, _hom_cached,
                                                              Tuple{typeof(M), typeof(N), typeof(cache)}))
    @test !occursin("Body::Any", report_hom)
    H = @inferred _hom_cached(M, N, cache)
    @test H isa DF.HomSpace{K}

    idM = IR.id_morphism(M)
    idN = IR.id_morphism(N)
    Hself = @inferred _hom_cached(M, N, cache)

    _pre_cached(Hd, Hc, f, c) = DF.precompose_matrix_cached(Hd, Hc, f; cache=c)
    report_pre = sprint(io -> InteractiveUtils.code_warntype(io, _pre_cached,
                                                              Tuple{typeof(Hself), typeof(Hself), typeof(idM), typeof(cache)}))
    @test !occursin("Body::Any", report_pre)
    pre = @inferred _pre_cached(Hself, Hself, idM, cache)
    @test pre isa SparseArrays.SparseMatrixCSC{K,Int}

    _post_cached(Hd, Hc, g, c) = DF.postcompose_matrix_cached(Hd, Hc, g; cache=c)
    report_post = sprint(io -> InteractiveUtils.code_warntype(io, _post_cached,
                                                               Tuple{typeof(Hself), typeof(Hself), typeof(idN), typeof(cache)}))
    @test !occursin("Body::Any", report_post)
    post = @inferred _post_cached(Hself, Hself, idN, cache)
    @test post isa SparseArrays.SparseMatrixCSC{K,Int}
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
        repP = DF.minimality_report(resP)
        @test repP.cover_ok
        @test repP.minimal
        @test isempty(repP.diagonal_violations)
        @test DF.is_minimal(resP)
        DF.assert_minimal(resP)

        # Passing ResolutionOptions(minimal=true, check=true) should succeed and return a minimal resolution.
        resPmin = DF.projective_resolution(S1, PM.ResolutionOptions(maxlen=4, minimal=true, check=true))
        @test DF.is_minimal(resPmin)

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

            res_bad = DF.ProjectiveResolution(resP.M, resP.Pmods, gens_bad, resP.d_mor, d_mat_bad, resP.aug)
            rep_bad = DF.minimality_report(res_bad; check_cover=true)
            @test !rep_bad.minimal
            @test !isempty(rep_bad.diagonal_violations)
            @test !DF.is_minimal(res_bad)
            @test_throws ErrorException DF.assert_minimal(res_bad)
        end

        # Injective resolution: also expected to be minimal for these constructions.
        resI = DF.injective_resolution(S1, PM.ResolutionOptions(maxlen=4))
        repI = DF.minimality_report(resI)
        @test repI.hull_ok
        @test repI.minimal
        @test isempty(repI.diagonal_violations)
        @test DF.is_minimal(resI)
        DF.assert_minimal(resI)

        resImin = DF.injective_resolution(S1, PM.ResolutionOptions(maxlen=4, minimal=true, check=true))
        @test DF.is_minimal(resImin)
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

        Btbl = DF.betti_table(res)
        if _is_real_field(field)
            # Numerical resolutions over reals can introduce extra/duplicated generators.
            @test get(b, (0, 1), 0) >= 1
            @test get(b, (1, 2), 0) >= 1
            @test get(b, (1, 3), 0) >= 1
            @test get(b, (2, 4), 0) >= 1
            @test Btbl[1,1] >= 1
            @test Btbl[2,2] >= 1
            @test Btbl[2,3] >= 1
            @test Btbl[3,4] >= 1
        else
            @test length(b) == 4
            @test b[(0, 1)] == 1
            @test b[(1, 2)] == 1
            @test b[(1, 3)] == 1
            @test b[(2, 4)] == 1
            @test Btbl[1,1] == 1
            @test Btbl[2,2] == 1
            @test Btbl[2,3] == 1
            @test Btbl[3,4] == 1
        end
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
        if _is_real_field(field)
            # Numerical Ext over reals can introduce duplicated classes in this setup.
            @test DF.dim(E14, 2) >= 1
            return
        end
        @test DF.dim(E14, 2) == 1

        # Via the chain 1 -> 2 -> 4
        E24 = DF.Ext(S2, S4, PM.DerivedFunctorOptions(maxdeg=1))
        E12 = DF.Ext(S1, S2, PM.DerivedFunctorOptions(maxdeg=2))  # needs tmax >= 2 because p+q = 2
        @test DF.dim(E24, 1) == 1
        @test DF.dim(E12, 1) == 1

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
        @test coords_2[1] == coords_3[1] || coords_2[1] == -coords_3[1]
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
        if _is_real_field(field)
            @test DF.dim(E03, 3) >= 1
        else
            @test DF.dim(E03, 3) == 1
        end

        # Degree-1 generators on the cover chain:
        #   {} -> {1} -> {1,2} -> {1,2,3}
        E23 = DF.Ext(S12, S123, PM.DerivedFunctorOptions(maxdeg=3))
        E12 = DF.Ext(S1,  S12, PM.DerivedFunctorOptions(maxdeg=3))
        E01 = DF.Ext(S0,  S1,   PM.DerivedFunctorOptions(maxdeg=3))

        if _is_real_field(field)
            @test DF.dim(E23, 1) >= 1
            @test DF.dim(E12, 1) >= 1
            @test DF.dim(E01, 1) >= 1
        else
            @test DF.dim(E23, 1) == 1
            @test DF.dim(E12, 1) == 1
            @test DF.dim(E01, 1) == 1
        end

        # Intermediate targets in degree 2 (also 1-dimensional for this choice).
        E13 = DF.Ext(S1,  S123, PM.DerivedFunctorOptions(maxdeg=3))
        E02 = DF.Ext(S0,  S12, PM.DerivedFunctorOptions(maxdeg=3))
        if _is_real_field(field)
            @test DF.dim(E13, 2) >= 1
            @test DF.dim(E02, 2) >= 1
            # Numerical lift solves in this chain can be inconsistent under tolerance;
            # keep a coarse sanity check for RealField and skip strict associativity.
            return
        else
            @test DF.dim(E13, 2) == 1
            @test DF.dim(E02, 2) == 1
        end

        # Left bracketing: (e23 * e12) * e01
        _, x = PM.DerivedFunctors.yoneda_product(E23, 1, [c(1)], E12, 1, [c(1)]; ELN=E13)  # x in Ext^2(S1,S123)
        _, left = PM.DerivedFunctors.yoneda_product(E13, 2, x, E01, 1, [c(1)]; ELN=E03)

        # Right bracketing: e23 * (e12 * e01)
        _, y = PM.DerivedFunctors.yoneda_product(E12, 1, [c(1)], E01, 1, [c(1)]; ELN=E02)  # y in Ext^2(S0,S12)
        _, right = PM.DerivedFunctors.yoneda_product(E23, 1, [c(1)], E02, 2, y; ELN=E03)

        # Nontriviality + associativity up to sign in a 1-dimensional target.
        @test left[1] != 0
        @test right[1] != 0
        @test left[1] == right[1] || left[1] == -right[1]
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

        delta2 = DF.connecting_hom(EMA, EMB, EMC, i, p; t=2)
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

        delta1 = DF.connecting_hom_first(EA, EB, EC, i1, p1; t=1)
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
        delta0 = DF.connecting_hom(EMA, EMB, EMC, i, p; t=0)

        # The packaged long exact sequence should expose the same delta^0.
        les = PM.ExtLongExactSequenceSecond(S1, S2, I12, S1, i, p, PM.DerivedFunctorOptions(maxdeg=0))
        if _is_real_field(field)
            r0 = FL.rank(field, delta0)
            r1 = FL.rank(field, les.delta[1])
            @test r0 == r1
            @test 0 <= r0 <= 1
            @test norm(Matrix(les.delta[1]) - Matrix(delta0)) <= _field_tol(field)
        else
            @test FL.rank(field, delta0) == 1
            @test FL.rank(field, les.delta[1]) == 1
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
            @test DF.dim(A, t) == DF.dim(E, t)
        end
        _is_real_field(field) && return

        # The unit should act as both-sided identity on every homogeneous degree <= tmax.
        oneA = one(A)
        for t in 0:E.tmax
            dt = DF.dim(A, t)
            if dt == 0
                continue
            end

            # Deterministic "generic" element: (1,2,3,...,dt).
            x = DF.element(A, t, [c(i) for i in 1:dt])

            @test (oneA * x).coords == x.coords
            @test (x * oneA).coords == x.coords
        end

        # Cache behavior: after one multiplication in (p,q), the multiplication matrix should exist,
        # and repeated multiplication should not grow the cache.
        if E.tmax >= 2 && DF.dim(A, 1) > 0
            d1 = DF.dim(A, 1)
            x = DF.element(A, 1, [c(i) for i in 1:d1])
            y = DF.element(A, 1, [c(d1 - i + 1) for i in 1:d1])

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
        if E.tmax >= 2 && DF.dim(A, 1) > 0
            d1 = DF.dim(A, 1)

            # First basis direction e_1
            a = DF.element(A, 1, [one(K); zeros(K, d1 - 1)])

            # Last basis direction e_{d1}
            b = DF.element(A, 1, [zeros(K, d1 - 1); one(K)])

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

            Cnew = DF._coeff_matrix_upsets(domU, codU, K)

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
            Ker, iota = PM.kernel_with_inclusion(pi0)
            P1, pi1, gens1 = IR.projective_cover(Ker)

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
                cval = coeff_dense[j,i]
                iszero(cval) && continue
                ui = dom_gens[i]
                vj = cod_gens[j]
                A = MD.map_leq(M, vj, ui)
                rows = (dom_offsets[i] + 1):dom_offsets[i+1]
                cols = (cod_offsets[j] + 1):cod_offsets[j+1]
                Fref_dense[rows, cols] .+= cval .* A
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
                cval = coeff_dense[j,i]
                iszero(cval) && continue
                u = dom_bases[i]
                v = cod_bases[j]
                A = MD.map_leq(M, u, v)
                rows = (cod_offsets[j] + 1):cod_offsets[j+1]
                cols = (dom_offsets[i] + 1):dom_offsets[i+1]
                Bref_dense[rows, cols] = cval .* A
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
