using Test
using Random
using SparseArrays

const FZ = PM.FlangeZn
const ZE = PM.ZnEncoding
const IR = PM.IndicatorResolutions
with_fields(FIELDS_FULL) do field
    K = CM.coeff_type(field)
    @inline c(x) = CM.coerce(field, x)

# ---------------------------------------------------------------------------
# Local helpers for the updated FlangeZn indicator API.
#
# The library intentionally removed the old Flat/Injective and flat/inj shims
# (and also disallows ambiguous reordered IndFlat/IndInj constructors) because
# both inputs are vectors and mixups were a frequent source of subtle bugs.
# ---------------------------------------------------------------------------
    mk_face(n, coords) = FZ.face(n, coords)
    mk_flat(b, coords; id=:F) = FZ.IndFlat(mk_face(length(b), coords), b; id=id)
    mk_inj(b, coords; id=:E) = FZ.IndInj(mk_face(length(b), coords), b; id=id)

    function _gather_projection_rows_naive(Bu::Matrix{K},
                                           rows_u::Vector{Int},
                                           rows_v::Vector{Int}) where {K}
        Ph = length(rows_v)
        du = size(Bu, 2)
        out = Matrix{K}(undef, Ph, du)
        Pg = length(rows_u)
        j = 1
        @inbounds for i in 1:Ph
            target = rows_v[i]
            while j <= Pg && rows_u[j] < target
                j += 1
            end
            if j > Pg || rows_u[j] != target
                error("naive box reference: projection mismatch; expected rows_v subset rows_u")
            end
            for c in 1:du
                out[i, c] = Bu[j, c]
            end
        end
        return out
    end

    function _naive_pmodule_on_box(FG::FZ.Flange{K},
                                   a::NTuple{N,Int},
                                   b::NTuple{N,Int},
                                   field::CM.AbstractCoeffField) where {K,N}
        Q, coords = ZE.grid_poset(a, b)
        ncoords = length(coords)
        rows_at = Vector{Vector{Int}}(undef, ncoords)
        B = Vector{Matrix{K}}(undef, ncoords)
        dims = Vector{Int}(undef, ncoords)

        @inbounds for u in 1:ncoords
            A, rows, cols = FZ.degree_matrix(FG, coords[u])
            rows_at[u] = rows
            if isempty(rows) || isempty(cols)
                B[u] = zeros(K, length(rows), 0)
                dims[u] = 0
            else
                Bu = PosetModules.FieldLinAlg.colspace(field, A)
                B[u] = Bu
                dims[u] = size(Bu, 2)
            end
        end

        edge_maps = Dict{Tuple{Int,Int}, Matrix{K}}()
        for (u, v) in FF.cover_edges(Q)
            du = dims[u]
            dv = dims[v]
            if du == 0 || dv == 0
                edge_maps[(u, v)] = zeros(K, dv, du)
                continue
            end
            Im = _gather_projection_rows_naive(B[u], rows_at[u], rows_at[v])
            edge_maps[(u, v)] = PosetModules.FieldLinAlg.solve_fullcolumn(field, B[v], Im)
        end

        return PM.PModule{K}(Q, dims, edge_maps; field=field)
    end

    function _sample_comparable_pairs(Q; rng::AbstractRNG, nsample::Int=64)
        pairs = Tuple{Int,Int}[]
        n = PM.nvertices(Q)
        for u in 1:n, v in 1:n
            FF.leq(Q, u, v) || continue
            push!(pairs, (u, v))
        end
        if length(pairs) <= nsample
            return pairs
        end
        order = collect(1:length(pairs))
        Random.shuffle!(rng, order)
        return [pairs[order[i]] for i in 1:nsample]
    end


    @testset "Zn wrappers: common encoding matches explicit encoding route" begin
    # Two 1D flanges representing interval modules [0,5] and [2,7].
    tau = FZ.face(1, [])  # no free coordinates for n=1 (i.e. an interval-type flange)

    F1 = FZ.IndFlat(tau, [0]; id=:F1)
    E1 = FZ.IndInj(tau, [5]; id=:E1)
    FG1 = FZ.Flange{K}(1, [F1], [E1], reshape([c(1)], 1, 1))

    F2 = FZ.IndFlat(tau, [2]; id=:F2)
    E2 = FZ.IndInj(tau, [7]; id=:E2)
    FG2 = FZ.Flange{K}(1, [F2], [E2], reshape([c(1)], 1, 1))

    enc = PM.EncodingOptions(backend=:zn, max_regions=50_000)
    df = PM.DerivedFunctorOptions(maxdeg=2)
    res = PM.ResolutionOptions(maxlen=3)

    enc_out = DF.encode_pmodules_from_flanges(FG1, FG2, enc)
    P = enc_out.P
    Ms = enc_out.Ms

    @test length(Ms) == 2
    @test Ms[1].Q === P
    @test Ms[2].Q === P

    E_explicit = DF.Ext(Ms[1], Ms[2], df)
    E_wrap = DF.ExtZn(FG1, FG2, enc, df)

    @test [PM.dim(E_explicit, t) for t in 0:2] == [PM.dim(E_wrap, t) for t in 0:2]

    # Resolution wrappers: compare against "encode + resolution" directly.
    enc1 = DF.encode_pmodule_from_flange(FG1, enc)
    res_wrap = DF.projective_resolution_Zn(FG1, enc, res; return_encoding=true)
    @test FF.poset_equal(res_wrap.P, enc1.P)
    @test DF.betti_table(res_wrap.res) == DF.betti_table(DF.projective_resolution(enc1.M, res))

    res_min = PM.ResolutionOptions(maxlen=3, minimal=true, check=true)

    bt_wrap = DF.betti(DF.projective_resolution_Zn(FG1, enc, res_min))
    bt_explicit = DF.betti(DF.projective_resolution(enc1.M, res_min))
    @test bt_wrap == bt_explicit
    end

    @testset "Wrappers for Z^n: injective resolutions and minimal Bass data" begin
    n = 1
    flats = [mk_flat([0],[false]), mk_flat([2],[false])]
    inj   = [mk_inj([4],[false])]
    Phi = [c(1); c(1)] |> x -> reshape(x, 1, 2)
    FG = FZ.Flange{K}(n, flats, inj, Phi)

    enc = PM.EncodingOptions(backend=:zn, max_regions=50_000)
    res = PM.ResolutionOptions(maxlen=3, check=true)

    P, M, pi = DF.encode_pmodule_from_flange(FG, enc)

    # Compare injective resolutions (wrapper vs explicit encode-then-resolve)
    resI = DF.injective_resolution(M, res)
    resI_Z = DF.injective_resolution_Zn(FG, enc, res)
    @test DF.bass_table(resI_Z) == DF.bass_table(resI)

    # Minimal injective resolutions and minimal Bass invariants
    res_min = PM.ResolutionOptions(maxlen=3, minimal=true, check=true)

    resMinI = DF.injective_resolution(M, res_min)
    resMinI_Z = DF.injective_resolution_Zn(FG, enc, res_min)
    @test DF.bass_table(resMinI_Z) == DF.bass_table(resMinI)

    @test DF.bass(resMinI_Z) == DF.bass(resMinI)
    end

    @testset "FlangeZn: IndFlat/IndInj constructors (new API)" begin
    tau = FZ.face(2, [2])

    # Canonical constructor order: (tau, b; id=...)
    F = FZ.IndFlat(tau, [1, 2])
    @test F isa FZ.IndFlat
    @test F.id == :F
    @test F.b == (1, 2)
    @test F.tau == tau

    E = FZ.IndInj(tau, [3, 4]; id=:E)
    @test E isa FZ.IndInj
    @test E.id == :E
    @test E.b == (3, 4)
    @test E.tau == tau

    # The library intentionally does NOT support reordered positional arguments
    # (both inputs are vectors, so such constructors were a source of subtle bugs).
    @test_throws MethodError FZ.IndFlat([1, 2], tau, :F2)
    @test_throws MethodError FZ.IndInj([3, 4], tau, :E2)
    end

    @testset "FlangeZn dim_at + minimize invariance" begin
    # n = 1 interval [b,c] via flat (>= b) and injective (<= c)
    n = 1
    tau0 = FZ.Face(n, [false])
    b = 1
    cmax = 3
    F1 = FZ.IndFlat(tau0, [b]; id=:F1)
    E1 = FZ.IndInj(tau0, [cmax]; id=:E1)
    Phi = reshape(K[c(1)], 1, 1)
    FG = FZ.Flange{K}(n, [F1], [E1], Phi)

    # dim is 1 on [b,c], 0 otherwise
    for g in (b-2):(cmax+2)
        d = FZ.dim_at(FG, [g]; rankfun=A -> PosetModules.FieldLinAlg.rank(field, A))
        expected = (b <= g <= cmax) ? 1 : 0
        @test d == expected
    end

    # intersects should detect empty intersection when b > c
    F_bad = FZ.IndFlat(tau0, [5]; id=:Fbad)
    E_bad = FZ.IndInj(tau0, [2]; id=:Ebad)
    @test FZ.intersects(F_bad, E_bad) == false

    # Minimize should merge proportional duplicate columns without changing dim_at
    F2 = FZ.IndFlat(tau0, [b]; id=:F2)
    Phi2 = reshape(K[c(1), c(2)], 1, 2)    # second column is 2x the first
    FG2 = FZ.Flange{K}(n, [F1, F2], [E1], Phi2)
    FG2m = FZ.minimize(FG2)

    for g in (b-1):(cmax+1)
        d1 = FZ.dim_at(FG2, [g];  rankfun=A -> PosetModules.FieldLinAlg.rank(field, A))
        d2 = FZ.dim_at(FG2m, [g]; rankfun=A -> PosetModules.FieldLinAlg.rank(field, A))
        @test d1 == d2
    end

        if field isa CM.QQField
        @testset "canonical_matrix / degree_matrix / bounding_box" begin
        # 1D: flat is x >= 1, inj is x <= 3.
        flats = [FZ.IndFlat(mk_face(length([1]), [false]), [1]; id=:F)]
        injectives = [FZ.IndInj(mk_face(length([3]), [false]), [3]; id=:E)]
        Phi = FZ.canonical_matrix(flats, injectives)
        @test Phi == reshape(K[1], 1, 1)

        # Non-intersecting pair: x >= 5 and x <= 3.
        flats_bad = [FZ.IndFlat(mk_face(length([5]), [false]), [5]; id=:Fbad)]
        injectives_bad = [FZ.IndInj(mk_face(length([3]), [false]), [3]; id=:Ebad)]
        Phibad = FZ.canonical_matrix(flats_bad, injectives_bad)
        @test Phibad == reshape(K[0], 1, 1)

        # degree_matrix should pick out exactly the active row/col at a given degree.
        Phi2 = reshape(K[2], 1, 1)
        H = FZ.Flange{K}(1, flats, injectives, Phi2)
        Phi_sub, rows, cols = FZ.degree_matrix(H, [2])
        @test rows == [1]
        @test cols == [1]
        @test Phi_sub == reshape(K[2], 1, 1)

        # Outside the intersection, there should be no active flats or injectives.
        Phi_sub2, rows2, cols2 = FZ.degree_matrix(H, [10])
        @test rows2 == Int[]
        @test cols2 == Int[]
        @test size(Phi_sub2) == (0, 0)

        # bounding_box in 1D with margin 1:
        #   flats force a >= (b_flat - margin) = 0
        #   injectives force b <= (b_inj + margin) = 4
        a_box, b_box = FZ.bounding_box(H; margin=1)
        @test a_box == [0]
        @test b_box == [4]
        end
        end

        @testset "minimize: do not merge different labels, but merge proportional duplicates" begin
        # Two proportional columns with different underlying flats must not be merged.
        F1 = FZ.IndFlat(mk_face(length([0]), [false]), [0]; id=:F1)
        F2 = FZ.IndFlat(mk_face(length([1]), [false]), [1]; id=:F2)  # different threshold => different upset
        E1 = FZ.IndInj(mk_face(length([2]), [false]), [2]; id=:E1)

        # Columns are proportional but flats differ.
        Phi = K[1 2]
        H = FZ.Flange{K}(1, [F1, F2], [E1], Phi)
        Hmin = FZ.minimize(H)
        @test length(Hmin.flats) == 2

        # Two proportional rows with identical injectives should be merged.
        Fin = [FZ.IndFlat(mk_face(length([0]), [false]), [0]; id=:F)]
        Einj1 = FZ.IndInj(mk_face(length([0]), [false]), [0]; id=:E)
        Einj2 = FZ.IndInj(mk_face(length([0]), [false]), [0]; id=:Edup) # same underlying downset as Einj1
        Phi_rows = reshape(K[1, 2], 2, 1)        # 2x1, proportional rows
        H2 = FZ.Flange{K}(1, Fin, [Einj1, Einj2], Phi_rows)
        H2min = FZ.minimize(H2)
        @test length(H2min.injectives) == 1

        # Rank at degree 0 should be unchanged by minimization.
        @test FZ.dim_at(H2, [0]) == FZ.dim_at(H2min, [0])
        end

        @testset "ZnEncoding: region encoding without enumerating a box" begin
        # FG, b, cmax are in scope here because this testset is nested.
        enc = PM.EncodingOptions(backend=:zn, max_regions=1000)
        P, Henc, pi = PM.encode_from_flange(FG, enc)

        # The encoding poset should be a 3-chain:
        #   left of b  <  between  <  right of cmax.
        @test PM.nvertices(P) == 3
        @test Set(FF.cover_edges(P)) == Set([(1,2),(2,3)])

        # In 1D, critical coordinates come from:
        #   flat threshold b
        #   complement(injective) threshold cmax+1
        @test pi.coords[1] == [b, cmax+1]

        # Spot-check the encoded fiber dimensions via locate.
        @test FF.fiber_dimension(Henc, PM.locate(pi, [b-5])) == 0
        @test FF.fiber_dimension(Henc, PM.locate(pi, [b])) == 1
        @test FF.fiber_dimension(Henc, PM.locate(pi, [cmax])) == 1
        @test FF.fiber_dimension(Henc, PM.locate(pi, [cmax+1])) == 0
        end

    @testset "ZnEncoding: direct flange -> fringe (Remark 6.14 bridge)" begin
        # Build the encoding only, then push FG down to a fringe presentation.
        enc = PM.EncodingOptions(backend=:zn, max_regions=1000)
        P, pi = PM.encode_poset_from_flanges(FG, enc)
        H = PM.fringe_from_flange(P, pi, FG)   # strict=true by default

        # Compare with the convenience wrapper.
        P2, H2, pi2 = PM.encode_from_flange(FG, enc)
        @test PM.nvertices(P) == PM.nvertices(P2)
        @test Set(FF.cover_edges(P)) == Set(FF.cover_edges(P2))

        # Fiber dimensions should match on all sampled degrees.
        for g in (b-5):(cmax+5)
            t  = PM.locate(pi,  [g])
            t2 = PM.locate(pi2, [g])
            @test t == t2
            if t != 0
                @test FF.fiber_dimension(H,  t)  == FF.fiber_dimension(H2, t2)
            end
        end

        # strictness: labels not present in the encoding must be rejected.
        F_extra = FZ.IndFlat(mk_face(length([b + 1]), [false]), [b + 1]; id=:Fextra)
        E = FZ.IndInj(mk_face(length([cmax]), [false]), [cmax]; id=:E)
        FG_extra = FZ.Flange{K}(1, [FZ.IndFlat(mk_face(length([b]), [false]), [b]; id=:F), F_extra], [E], K[1 1])

        @test_throws ErrorException PM.fringe_from_flange(P, pi, FG_extra)
    end
    end

@testset "CrossValidateFlangePL smoke test" begin
    n = 1
    tau0 = FZ.Face(n, [false])
    F1 = FZ.IndFlat(tau0, [1]; id=:F1)
    E1 = FZ.IndInj(tau0, [3]; id=:E1)
    Phi = reshape(K[c(1)], 1, 1)
    FG = FZ.Flange{K}(n, [F1], [E1], Phi)

    ok, report = FZ.cross_validate(FG; margin=1, rankfun=A -> PosetModules.FieldLinAlg.rank(field, A))
    @test ok == true
    @test haskey(report, "mismatches")
    @test isempty(report["mismatches"])
end

@testset "ZnEncoding 2D: free coordinate compression and correctness" begin
    # A 2D flange that depends only on coordinate 1; coordinate 2 is free everywhere.
    flats = [mk_flat([0, 0], [false, true])]
    inj   = [mk_inj([1, 0], [false, true])]
    Phi   = reshape(K[1], 1, 1)
    FG = FZ.Flange{K}(2, flats, inj, Phi)

    enc = PM.EncodingOptions(backend=:zn, max_regions=100)
    P, M, pi = DF.encode_pmodule_from_flange(FG, enc)

    # Expected: only 3 regions along coordinate 1 (below, inside, above), and 1 slab along coord 2.
    @test PM.nvertices(P) == 3

    # pi should ignore coordinate 2
    for g1 in -3:4
        u = PM.locate(pi, [g1, -10])
        v = PM.locate(pi, [g1,  10])
        @test u == v
    end

    # Monotonicity: g <= h implies pi(g) <= pi(h)
    for g1 in -3:3, h1 in g1:4
        ug = PM.locate(pi, [g1, 0])
        uh = PM.locate(pi, [h1, 0])
        @test FF.leq(P, ug, uh)
    end

    # Dimension consistency on a representative grid of lattice points
    for g1 in -3:4, g2 in (-2, 0, 7)
        g = [g1, g2]
        @test FZ.dim_at(FG, g) == M.dims[PM.locate(pi, g)]
    end
end

@testset "ZnEncoding 2D: common encoding for multiple flanges" begin
    # Two slabs along coordinate 1, coordinate 2 free.
    FG1 = FZ.Flange{K}(2,
        [mk_flat([0,0],[false,true])],
        [mk_inj([1,0],[false,true])],
        reshape(K[1], 1, 1)
    )
    FG2 = FZ.Flange{K}(2,
        [mk_flat([1,0],[false,true])],
        [mk_inj([2,0],[false,true])],
        reshape(K[1], 1, 1)
    )

    enc = PM.EncodingOptions(backend=:zn, max_regions=200)
    P, Ms, pi = DF.encode_pmodules_from_flanges(FG1, FG2, enc)
    M1, M2 = Ms

    # Critical coordinates along g1 are {0,1,2,3} giving <= 5 slabs => P.n <= 5.
    @test PM.nvertices(P) <= 5

    for g1 in -1:4, g2 in (-5, 0, 5)
        g = [g1, g2]
        u = PM.locate(pi, g)
        @test u != 0
        @test FZ.dim_at(FG1, g) == M1.dims[u]
        @test FZ.dim_at(FG2, g) == M2.dims[u]
    end
end

@testset "ZnEncoding advanced cache API: compile_zn_cache + locate_many + box cache" begin
    FG = FZ.Flange{K}(2,
        [mk_flat([0, 0], [false, true])],
        [mk_inj([1, 0], [false, true])],
        reshape(K[1], 1, 1)
    )

    enc = PM.EncodingOptions(backend=:zn, max_regions=200)
    P, M, pi = DF.encode_pmodule_from_flange(FG, enc)
    zcache = ZE.compile_zn_cache(P, pi)
    cenc = CM.compile_encoding(P, pi)

    Xint = Int[
        -2 -1 0 1 2 3
         7 -3 5 0 9 1
    ]

    # Integer batched locate parity.
    d1 = ZE.locate_many(pi, Xint; threaded=false)
    d2 = ZE.locate_many(zcache, Xint; threaded=true)
    d3 = ZE.locate_many(cenc, Xint; threaded=true)
    @test d1 == d2 == d3
    @test d1 == [PM.locate(pi, Xint[:, j]) for j in 1:size(Xint, 2)]
    @test pi.cell_to_region !== nothing

    # Fast locate path should not depend on signature dictionary lookups when the
    # slab-cell lookup table is available.
    sig_copy = copy(pi.sig_to_region)
    empty!(pi.sig_to_region)
    @test ZE.locate_many(pi, Xint; threaded=false) == d1
    merge!(pi.sig_to_region, sig_copy)

    # Float batched locate parity (round-to-nearest lattice point).
    Xflt = Float64.(Xint) .+ 0.24
    d4 = ZE.locate_many(pi, Xflt; threaded=true)
    @test d4 == [PM.locate(pi, Xflt[:, j]) for j in 1:size(Xflt, 2)]

    # In-place API with destination buffer.
    d5 = fill(-1, size(Xint, 2))
    ZE.locate_many!(d5, zcache, Xint; threaded=true)
    @test d5 == d1

    # Reusable pmodule_on_box basis cache.
    bcache = ZE.compile_zn_box_cache(FG)
    Mbox1 = ZE.pmodule_on_box(FG; a=(-2, -2), b=(3, 3), cache=bcache)
    nentries1 = sum(length(v) for v in values(bcache.basis_cache))
    ntrans1 = length(bcache.transition_cache)
    Mbox2 = ZE.pmodule_on_box(FG; a=(-2, -2), b=(3, 3), cache=bcache)
    nentries2 = sum(length(v) for v in values(bcache.basis_cache))
    ntrans2 = length(bcache.transition_cache)
    @test Mbox1.dims == Mbox2.dims
    @test nentries2 == nentries1
    @test ntrans2 == ntrans1

    st = ZE.box_cache_stats(bcache)
    @test st.basis_entries == nentries2
    @test st.transition_entries == ntrans2
    @test st.basis_cache_hits >= 0
    @test st.transition_cache_misses >= 0
end

@testset "ZnEncoding large oracle fixtures (all fields)" begin
    # Fixture 1 (1D): independent interval summands with diagonal Phi.
    # Oracle: dim(g) = number of intervals [b_i, c_i] containing g.
    bs = [0, 25, 50]
    cs = [40, 70, 100]
    flats = [mk_flat([bs[i]], [false]; id=Symbol(:CF, i)) for i in eachindex(bs)]
    injs  = [mk_inj([cs[i]], [false]; id=Symbol(:CE, i)) for i in eachindex(cs)]
    Phi = Matrix{K}(I, length(injs), length(flats))
    FG = FZ.Flange{K}(1, flats, injs, Phi)
    M = ZE.pmodule_on_box(FG; a=(-10,), b=(120,), cache=ZE.compile_zn_box_cache(FG))

    gvals = -10:120
    expected = [count(i -> (bs[i] <= g <= cs[i]), eachindex(bs)) for g in gvals]
    @test M.dims == expected

    # Fixture 2 (2D free-axis): dims are invariant in y for fixed x.
    FG2 = FZ.Flange{K}(2,
        [mk_flat([0, 0], [false, true]), mk_flat([30, 0], [false, true]), mk_flat([60, 0], [false, true])],
        [mk_inj([20, 0], [false, true]), mk_inj([50, 0], [false, true]), mk_inj([90, 0], [false, true])],
        Matrix{K}(I, 3, 3)
    )
    Mxy = ZE.pmodule_on_box(FG2; a=(-5, -40), b=(95, 40), cache=ZE.compile_zn_box_cache(FG2))
    lensx = 95 - (-5) + 1
    lensy = 40 - (-40) + 1
    @test lensx * lensy == length(Mxy.dims)
    for x in (1, div(lensx, 2), lensx)
        d0 = Mxy.dims[x]
        d1 = Mxy.dims[x + (lensy ÷ 2) * lensx]
        d2 = Mxy.dims[x + (lensy - 1) * lensx]
        @test d0 == d1 == d2
    end
end

@testset "ZnEncoding optimization-branch counters (all fields)" begin
    FG = FZ.Flange{K}(2,
        [mk_flat([0, 0], [false, true]), mk_flat([35, 0], [false, true]), mk_flat([70, 0], [false, true])],
        [mk_inj([25, 0], [false, true]), mk_inj([55, 0], [false, true]), mk_inj([95, 0], [false, true])],
        Matrix{K}(I, 3, 3)
    )
    bcache = ZE.compile_zn_box_cache(FG)
    M1 = ZE.pmodule_on_box(FG; a=(-10, -50), b=(110, 50), cache=bcache)

    @test bcache.basis_cache_misses > 0
    @test bcache.full_basis_recomputes > 0
    @test bcache.incremental_refinements > 0
    @test bcache.transition_cache_misses > 0
    @test length(bcache.transition_cache) > 0

    hits_before = bcache.basis_cache_hits
    thits_before = bcache.transition_cache_hits
    misses_before = bcache.transition_cache_misses
    M2 = ZE.pmodule_on_box(FG; a=(-10, -50), b=(110, 50), cache=bcache)

    @test M2.dims == M1.dims
    @test bcache.basis_cache_hits > hits_before
    @test bcache.transition_cache_hits > thits_before
    @test bcache.transition_cache_misses == misses_before

    nedges = length(collect(FF.cover_edges(M1.Q)))
    @test length(bcache.transition_cache) <= nedges
    @test bcache.full_basis_recomputes + bcache.incremental_refinements >= bcache.basis_cache_misses
end

@testset "ZnEncoding edge-map matrix oracle fixtures" begin
    # 1D hand-computable oracle with two overlapping intervals:
    # I1=[0,2], I2=[1,3], Phi=I2.
    FG = FZ.Flange{K}(1,
        [mk_flat([0], [false]; id=:F1), mk_flat([1], [false]; id=:F2)],
        [mk_inj([2], [false]; id=:E1), mk_inj([3], [false]; id=:E2)],
        Matrix{K}(I, 2, 2)
    )
    M = ZE.pmodule_on_box(FG; a=(0,), b=(3,), cache=ZE.compile_zn_box_cache(FG))
    @test M.dims == [1, 2, 2, 1]
    @test Set(FF.cover_edges(M.Q)) == Set([(1, 2), (2, 3), (3, 4)])

    A12 = M.edge_maps[1, 2]
    A23 = M.edge_maps[2, 3]
    A34 = M.edge_maps[3, 4]
    E12 = reshape(K[c(1), c(0)], 2, 1)
    E23 = Matrix{K}(I, 2, 2)
    E34 = reshape(K[c(0), c(1)], 1, 2)

    if field isa CM.RealField
        @test isapprox(A12, E12; rtol=1e-10, atol=1e-12)
        @test isapprox(A23, E23; rtol=1e-10, atol=1e-12)
        @test isapprox(A34, E34; rtol=1e-10, atol=1e-12)
    else
        @test A12 == E12
        @test A23 == E23
        @test A34 == E34
    end

    # Composition oracle: map 1->4 should be zero (rank 0).
    @test PM.rank_map(M, 1, 4) == 0
end

@testset "ZnEncoding randomized differential vs naive reference (medium grids)" begin
    function _assert_fast_vs_naive(FG::FZ.Flange{K},
                                   a::NTuple{2,Int},
                                   b::NTuple{2,Int},
                                   rng::AbstractRNG;
                                   nsample::Int=72) where {K}
        M_fast = ZE.pmodule_on_box(FG; a=a, b=b, cache=ZE.compile_zn_box_cache(FG))
        M_ref  = _naive_pmodule_on_box(FG, a, b, field)

        @test FF.poset_equal(M_fast.Q, M_ref.Q)
        @test M_fast.dims == M_ref.dims

        for (u, v) in FF.cover_edges(M_fast.Q)
            rf = PosetModules.FieldLinAlg.rank(field, M_fast.edge_maps[u, v])
            rr = PosetModules.FieldLinAlg.rank(field, M_ref.edge_maps[u, v])
            @test rf == rr
        end

        for (u, v) in _sample_comparable_pairs(M_fast.Q; rng=rng, nsample=nsample)
            @test PM.rank_map(M_fast, u, v) == PM.rank_map(M_ref, u, v)
        end
    end

    if field isa CM.RealField
        # Real-field differential checks are deterministic and numerically stable.
        rng = MersenneTwister(0xA11CE)

        FG1 = FZ.Flange{K}(2,
            [mk_flat([0, 0], [false, true]), mk_flat([20, 0], [false, true]), mk_flat([40, 0], [false, true])],
            [mk_inj([10, 0], [false, true]), mk_inj([30, 0], [false, true]), mk_inj([50, 0], [false, true])],
            Matrix{K}(I, 3, 3)
        )
        _assert_fast_vs_naive(FG1, (-5, -20), (55, 20), rng; nsample=144)

        FG2 = FZ.Flange{K}(2,
            [mk_flat([-3, -2], [false, false]; id=:RF1),
             mk_flat([ 0, -1], [false, false]; id=:RF2),
             mk_flat([ 2,  1], [false, false]; id=:RF3)],
            [mk_inj([ 4,  2], [false, false]; id=:RE1),
             mk_inj([ 5,  3], [false, false]; id=:RE2),
             mk_inj([ 6,  4], [false, false]; id=:RE3)],
            Matrix{K}(I, 3, 3)
        )
        _assert_fast_vs_naive(FG2, (-4, -4), (6, 6), rng; nsample=128)
    else
        seeds = (0xC0FFEE, 0xBAD5EED, 0x1234567)
        for seed in seeds
            rng = MersenneTwister(seed)

            for trial in 1:4
                n = 2
                m = 8
                r = 8

                flats = Vector{FZ.IndFlat{2}}(undef, m)
                injs  = Vector{FZ.IndInj{2}}(undef, r)
                for i in 1:m
                    bvec = [rand(rng, -6:6), rand(rng, -6:6)]
                    cvec = [rand(rng, Bool), rand(rng, Bool)]
                    flats[i] = mk_flat(bvec, cvec; id=Symbol("RF$(seed)_$(trial)_$i"))
                end
                for j in 1:r
                    bvec = [rand(rng, -6:6), rand(rng, -6:6)]
                    cvec = [rand(rng, Bool), rand(rng, Bool)]
                    injs[j] = mk_inj(bvec, cvec; id=Symbol("RE$(seed)_$(trial)_$j"))
                end

                Phi = Matrix{K}(undef, r, m)
                @inbounds for i in 1:r, j in 1:m
                    Phi[i, j] = c(rand(rng, -2:2))
                end
                FG = FZ.Flange{K}(n, flats, injs, Phi)
                _assert_fast_vs_naive(FG, (-5, -5), (5, 5), rng; nsample=128)
            end

            # Adversarial fixture A: repeated/near-repeated columns with shifted windows.
            flatsA = [mk_flat([i - 3, 0], [false, true]; id=Symbol(:AF, i)) for i in 1:8]
            injsA  = [mk_inj([i + 1, 0], [false, true]; id=Symbol(:AE, i)) for i in 1:8]
            PhiA = zeros(K, 8, 8)
            @inbounds for i in 1:8
                PhiA[i, i] = c(1)
                i < 8 && (PhiA[i, i + 1] = c(1))
            end
            FGA = FZ.Flange{K}(2, flatsA, injsA, PhiA)
            _assert_fast_vs_naive(FGA, (-6, -4), (8, 4), rng; nsample=144)

            # Adversarial fixture B: mixed free/locked coordinates creating abrupt signature flips.
            flatsB = [
                mk_flat([-4, -1], [false, false]; id=:BF1),
                mk_flat([-2,  0], [false, true];  id=:BF2),
                mk_flat([ 0, -2], [true,  false]; id=:BF3),
                mk_flat([ 2,  1], [false, false]; id=:BF4),
                mk_flat([ 4,  0], [false, true];  id=:BF5),
                mk_flat([ 1,  3], [true,  false]; id=:BF6),
            ]
            injsB = [
                mk_inj([ 3,  2], [false, false]; id=:BE1),
                mk_inj([ 1,  4], [false, true];  id=:BE2),
                mk_inj([ 5,  1], [true,  false]; id=:BE3),
                mk_inj([ 7,  3], [false, false]; id=:BE4),
                mk_inj([ 6,  0], [false, true];  id=:BE5),
                mk_inj([ 2,  5], [true,  false]; id=:BE6),
            ]
            PhiB = Matrix{K}(undef, 6, 6)
            @inbounds for i in 1:6, j in 1:6
                PhiB[i, j] = c((i == j) ? 1 : ((i + j) % 3 == 0 ? -1 : 0))
            end
            FGB = FZ.Flange{K}(2, flatsB, injsB, PhiB)
            _assert_fast_vs_naive(FGB, (-6, -6), (8, 8), rng; nsample=160)
        end
    end
end

if field isa CM.QQField
@testset "ZnEncoding large oracle fixtures + optimization-branch assertions" begin
    # Fixture 1 (1D, large): independent interval summands with diagonal Phi.
    # Oracle: dim(g) = number of intervals [b_i, c_i] containing g.
    n = 1
    bs = [0, 40, 80, 120]
    cs = [60, 100, 140, 180]
    flats = [mk_flat([bs[i]], [false]; id=Symbol(:F, i)) for i in eachindex(bs)]
    injs  = [mk_inj([cs[i]], [false]; id=Symbol(:E, i)) for i in eachindex(cs)]
    Phi = Matrix{K}(I, length(injs), length(flats))
    FG = FZ.Flange{K}(n, flats, injs, Phi)

    bcache = ZE.compile_zn_box_cache(FG)
    M = ZE.pmodule_on_box(FG; a=(-20,), b=(220,), cache=bcache)

    gvals = -20:220
    expected = [count(i -> (bs[i] <= g <= cs[i]), eachindex(bs)) for g in gvals]
    @test M.dims == expected

    # Branch assertions after first run.
    @test bcache.basis_cache_misses > 0
    @test bcache.full_basis_recomputes > 0
    @test bcache.incremental_refinements > 0
    @test bcache.transition_cache_misses > 0

    hits_before = bcache.basis_cache_hits
    thits_before = bcache.transition_cache_hits
    misses_before = bcache.transition_cache_misses
    M2 = ZE.pmodule_on_box(FG; a=(-20,), b=(220,), cache=bcache)
    @test M2.dims == expected
    @test bcache.basis_cache_hits > hits_before
    @test bcache.transition_cache_hits > thits_before
    @test bcache.transition_cache_misses == misses_before

    # Fixture 2 (2D free-axis): transition-slot reuse should collapse far below #edges.
    FG2 = FZ.Flange{K}(2,
        [mk_flat([0, 0], [false, true]), mk_flat([60, 0], [false, true]), mk_flat([120, 0], [false, true])],
        [mk_inj([40, 0], [false, true]), mk_inj([100, 0], [false, true]), mk_inj([160, 0], [false, true])],
        Matrix{K}(I, 3, 3)
    )
    bcache2 = ZE.compile_zn_box_cache(FG2)
    Mxy = ZE.pmodule_on_box(FG2; a=(-10, -60), b=(190, 60), cache=bcache2)
    nedges = length(collect(FF.cover_edges(Mxy.Q)))
    @test length(bcache2.transition_cache) < max(10, div(nedges, 25))
    @test bcache2.incremental_refinements > 0

    # Oracle: because axis 2 is free, dims are invariant along y for fixed x.
    lensx = 190 - (-10) + 1
    lensy = 60 - (-60) + 1
    @test lensx * lensy == length(Mxy.dims)
    for x in (1, div(lensx, 2), lensx)
        d0 = Mxy.dims[x]                            # y = 1
        d1 = Mxy.dims[x + (lensy ÷ 2) * lensx]     # y ~ middle
        d2 = Mxy.dims[x + (lensy - 1) * lensx]     # y = last
        @test d0 == d1 == d2
    end

    # Fixture 3 (2D, sparse-first solve): high-dimensional sparse identity-like regime.
    k = 96
    flats3 = [mk_flat([0, 0], [false, true]; id=Symbol(:SF, i)) for i in 1:k]
    injs3  = [mk_inj([180, 0], [false, true]; id=Symbol(:SE, i)) for i in 1:k]
    Phi3 = Matrix{K}(I, k, k)
    FG3 = FZ.Flange{K}(2, flats3, injs3, Phi3)
    bcache3 = ZE.compile_zn_box_cache(FG3)
    M3 = ZE.pmodule_on_box(FG3; a=(-5, -20), b=(185, 20), cache=bcache3)

    @test maximum(M3.dims) == k
    @test bcache3.sparse_transition_solves > 0
    @test bcache3.dense_transition_solves >= 0
end
end

@testset "ZnEncoding 2D: strict fringe_from_flange rejects missing generators" begin
    FG1 = FZ.Flange{K}(2,
        [mk_flat([0,0],[false,true])],
        [mk_inj([1,0],[false,true])],
        reshape(K[1], 1, 1)
    )
    FG2 = FZ.Flange{K}(2,
        [mk_flat([10,0],[false,true])],  # new flat label not present in FG1 encoding
        [mk_inj([11,0],[false,true])],
        reshape(K[1], 1, 1)
    )

    enc = PM.EncodingOptions(backend=:zn, max_regions=200)
    P, pi = ZE.encode_poset_from_flanges(FG1, enc)

    @test_throws ErrorException ZE.fringe_from_flange(P, pi, FG2; strict=true)

    # Non-strict mode should still push forward, dropping unmatched generators safely.
    H2 = ZE.fringe_from_flange(P, pi, FG2; strict=false)
    M2 = IR.pmodule_from_fringe(H2)

    # Sanity: the pushed module is defined on P
    @test length(M2.dims) == PM.nvertices(P)
end

@testset "ZnEncodingMap region_weights: exact methods agree" begin
    # Same 2D setup used in test_znencoding_2d_regions.jl:
    # - one flat at x1 >= 0 (x2 free)
    # - one injective at x1 <= 1 (x2 free)
    flats = [mk_flat([0, 0], [false, true])]
    injs  = [mk_inj([1, 0], [false, true])]
    Phi   = reshape(K[1], 1, 1)
    FG    = FZ.Flange{K}(2, flats, injs, Phi)

    enc = PM.EncodingOptions(backend=:zn, max_regions=100)
    P, Henc, pi = DF.encode_pmodule_from_flange(FG, enc)

    a = [-2, -3]
    b = [ 3,  4]
    len2 = b[2] - a[2] + 1  # length in free coordinate

    # Determine region indices by locating representative points.
    rid_left  = PM.locate(pi, [-1, 0])  # x1 < 0
    rid_mid   = PM.locate(pi, [ 0, 0])  # 0 <= x1 <= 1
    rid_right = PM.locate(pi, [ 2, 0])  # x1 > 1

    expected = zeros(Int, length(pi.sig_y))
    expected[rid_left]  = 2 * len2  # x1 = -2,-1
    expected[rid_mid]   = 2 * len2  # x1 = 0,1
    expected[rid_right] = 2 * len2  # x1 = 2,3

    w_cells   = PM.region_weights(pi; box=(a, b), method=:cells)
    w_points  = PM.region_weights(pi; box=(a, b), method=:points)
    w_auto    = PM.region_weights(pi; box=(a, b), method=:auto)

    @test w_cells == expected
    @test w_points == expected
    @test w_auto == expected
end

@testset "ZnEncodingMap region_weights: Monte Carlo sampling is close (reproducible)" begin
    flats = [mk_flat([0, 0], [false, true])]
    injs  = [mk_inj([1, 0], [false, true])]
    Phi   = reshape(K[1], 1, 1)
    FG    = FZ.Flange{K}(2, flats, injs, Phi)

    enc = PM.EncodingOptions(backend=:zn, max_regions=100)
    P, Henc, pi = DF.encode_pmodule_from_flange(FG, enc)

    a = [-50, -500]
    b = [ 49,  499]
    len2 = b[2] - a[2] + 1  # 1000

    rid_left  = PM.locate(pi, [-1, 0])
    rid_mid   = PM.locate(pi, [ 0, 0])
    rid_right = PM.locate(pi, [ 2, 0])

    expected = zeros(Int, length(pi.sig_y))
    expected[rid_left]  = 50 * len2
    expected[rid_mid]   =  2 * len2
    expected[rid_right] = 48 * len2

    # Monte Carlo estimate
    rng = MersenneTwister(0)
    info = PM.region_weights(pi; box=(a, b), method=:sample, nsamples=20_000, rng=rng, return_info=true)

    @test info.method == :sample
    @test eltype(info.weights) == Float64
    @test length(info.weights) == length(expected)
    @test info.stderr !== nothing
    @test length(info.stderr) == length(expected)

    # Total points in the box:
    total_points = (b[1] - a[1] + 1) * (b[2] - a[2] + 1)
    @test isapprox(sum(info.weights), total_points; rtol=0.03)

    # Each bin should be reasonably close. Tolerances chosen for test stability.
    for j in eachindex(expected)
        @test isapprox(info.weights[j], expected[j]; rtol=0.06, atol=5_000.0)
    end
end

@testset "ZnEncodingMap region_weights: auto can be forced to sampling" begin
    flats = [mk_flat([0, 0], [false, true])]
    injs  = [mk_inj([1, 0], [false, true])]
    Phi   = reshape(K[1], 1, 1)
    FG    = FZ.Flange{K}(2, flats, injs, Phi)

    enc = PM.EncodingOptions(backend=:zn, max_regions=100)
    P, Henc, pi = DF.encode_pmodule_from_flange(FG, enc)

    a = [-50, -500]
    b = [ 49,  499]

    rng = MersenneTwister(1)
    info = PM.region_weights(pi; box=(a, b), method=:auto, max_cells=0, max_points=0,
                             nsamples=1_000, rng=rng, return_info=true)

    @test info.method == :sample
    @test eltype(info.weights) == Float64
end

@testset "ZnEncodingMap region_weights: :auto count_type promotes to BigInt on overflow" begin
    flats = [mk_flat([0, 0], [false, true])]
    injs  = [mk_inj([1, 0], [false, true])]
    Phi   = reshape(K[1], 1, 1)
    FG    = FZ.Flange{K}(2, flats, injs, Phi)

    enc = PM.EncodingOptions(backend=:zn, max_regions=100)
    P, Henc, pi = DF.encode_pmodule_from_flange(FG, enc)

    # Choose an interval length > typemax(Int) while endpoints still fit in Int64.
    a = [-9_000_000_000_000_000_000, 0]
    b = [ 9_000_000_000_000_000_000, 0]

    rid_left  = PM.locate(pi, [-1, 0])
    rid_mid   = PM.locate(pi, [ 0, 0])
    rid_right = PM.locate(pi, [ 2, 0])

    expected = zeros(BigInt, length(pi.sig_y))
    expected[rid_left]  = BigInt(-1) - BigInt(a[1]) + 1                 # a1..-1
    expected[rid_mid]   = BigInt(2)                                     # 0..1
    expected[rid_right] = BigInt(b[1]) - BigInt(2) + 1                  # 2..b1

    w = PM.region_weights(pi; box=(a, b), method=:cells, count_type=:auto)
    @test eltype(w) == BigInt
    @test w == expected
    @test sum(w) == (BigInt(b[1]) - BigInt(a[1]) + 1) * (BigInt(b[2]) - BigInt(a[2]) + 1)
end

@testset "region_poset(pi) reconstructs the encoder region poset" begin
    # -------------------------------------------------------------------------
    # 1) ZnEncoding: pi has (sig_y, sig_z) but no pi.P; region_poset should match P
    # -------------------------------------------------------------------------
    let
        n = 1
        b = [0]
        c = [5]
        I = FZ.Face(n, Int[])
        flats = [FZ.IndFlat(I, b; id=:F)]
        injectives = [FZ.IndInj(I, c; id=:E)]
        Phi = spzeros(K, 1, 1); Phi[1, 1] = 1
        FG = FZ.Flange{K}(n, flats, injectives, Phi)

        enc = PM.EncodingOptions(backend=:zn, max_regions=1000)
        Penc, Henc, pi = PM.encode_from_flange(FG, enc)
        rpcache = CM.EncodingCache()

        Q = PM.Invariants.region_poset(pi; poset_kind = :signature, cache=rpcache)
        @test PM.nvertices(Q) == PM.nvertices(Penc)
        @test FF.poset_equal(Q, Penc)

        # Cached repeat call should return the exact same poset object.
        Q2 = PM.Invariants.region_poset(pi; poset_kind = :signature, cache=rpcache)
        @test Q2 === Q

        Qdense = PM.Invariants.region_poset(pi; poset_kind = :dense, cache=rpcache)
        @test FF.leq_matrix(Q) == FF.leq_matrix(Qdense)

        # Projected arrangement should work without requiring pi.P.
        arr = PM.projected_arrangement(pi; dirs=[[1.0]])
        @test FF.poset_equal(arr.Q, Penc)

        # And should accept a provided Q for maximum speed.
        arr2 = PM.projected_arrangement(pi; dirs=[[1.0]], Q=Penc)
        @test FF.poset_equal(arr2.Q, Penc)
    end

    # -------------------------------------------------------------------------
    # 2) PLPolyhedra: PLEncodingMap has (sig_y, sig_z) but no pi.P
    # -------------------------------------------------------------------------
    let
        PLP = PM.PLPolyhedra
        if !PLP.HAVE_POLY
            @test true
        else
            n = 1

            # PLPolyhedra uses H-polytopes { x : A*x <= b }.
            # Upset: x >= 0  <=>  (-x <= 0)
            Uhp = PLP.make_hpoly([-1.0], 0.0)

            # Downset: x <= 2 <=>  (x <= 2)
            Dhp = PLP.make_hpoly([1.0], 2.0)

            U = PLP.PLUpset(PLP.PolyUnion(n, [Uhp]))
            D = PLP.PLDownset(PLP.PolyUnion(n, [Dhp]))

            # PLFringe requires an explicit Phi of size (#Downs) x (#Ups).
            F1 = PLP.PLFringe([U], PLP.PLDownset[], zeros(K, 0, 1))
            F2 = PLP.PLFringe(PLP.PLUpset[], [D], zeros(K, 1, 0))

            enc = PM.EncodingOptions(backend=:pl, max_regions=10_000)
            Ppl, Hpl, pipl = PLP.encode_from_PL_fringes(F1, F2, enc; poset_kind = :signature)
            @test Ppl isa PM.ZnEncoding.SignaturePoset
            rpcache = CM.EncodingCache()

            Qpl = PM.Invariants.region_poset(pipl; poset_kind = :signature, cache=rpcache)
            @test PM.nvertices(Qpl) == PM.nvertices(Ppl)
            @test FF.poset_equal(Qpl, Ppl)

            Qpl2 = PM.Invariants.region_poset(pipl; poset_kind = :signature, cache=rpcache)
            @test Qpl2 === Qpl

            Qpl_dense = PM.Invariants.region_poset(pipl; poset_kind = :dense, cache=rpcache)
            @test FF.leq_matrix(Qpl) == FF.leq_matrix(Qpl_dense)

            arr = PM.projected_arrangement(pipl; dirs=[[1.0]])
            @test FF.poset_equal(arr.Q, Ppl)

            arr2 = PM.projected_arrangement(pipl; dirs=[[1.0]], Q=Ppl)
            @test FF.poset_equal(arr2.Q, Ppl)
        end
    end


    # -------------------------------------------------------------------------
    # 3) PLBackend: PLEncodingMapBoxes has (sig_y, sig_z) but no pi.P
    # -------------------------------------------------------------------------
    if isdefined(PM, :PLBackend)
        let
            PB = PM.PLBackend
            n = 1
            U = PB.BoxUpset([0.0])
            D = PB.BoxDownset([2.0])

            Ups = [U]
            Downs = [D]

            enc = PM.EncodingOptions(backend=:pl_backend, max_regions=10_000)
            Pbx, Hbx, pibx = PB.encode_fringe_boxes(Ups, Downs, enc; poset_kind = :signature)
            @test Pbx isa PM.ZnEncoding.SignaturePoset
            rpcache = CM.EncodingCache()

            Qbx = PM.Invariants.region_poset(pibx; poset_kind = :signature, cache=rpcache)
            @test PM.nvertices(Qbx) == PM.nvertices(Pbx)
            @test FF.poset_equal(Qbx, Pbx)

            Qbx2 = PM.Invariants.region_poset(pibx; poset_kind = :signature, cache=rpcache)
            @test Qbx2 === Qbx

            Qbx_dense = PM.Invariants.region_poset(pibx; poset_kind = :dense, cache=rpcache)
            @test FF.leq_matrix(Qbx) == FF.leq_matrix(Qbx_dense)

            arr = PM.projected_arrangement(pibx; dirs=[[1.0]])
            @test FF.poset_equal(arr.Q, Pbx)

            arr2 = PM.projected_arrangement(pibx; dirs=[[1.0]], Q=Pbx)
            @test FF.poset_equal(arr2.Q, Pbx)

            Mbx = IR.pmodule_from_fringe(Hbx)
            @test PM.rank_map(Mbx, 1, 1) >= 0
        end
    end
end

if field isa CM.QQField
@testset "encode poset_kind=:dense yields FinitePoset" begin
    if isdefined(PM, :PLBackend)
        PB = PM.PLBackend
        Ups = [PB.BoxUpset([0.0])]
        Downs = [PB.BoxDownset([2.0])]
        enc = PM.EncodingOptions(backend=:pl_backend, max_regions=10_000, poset_kind=:dense)
        P, _H, _pi = PM.encode(Ups, Downs, enc)
        @test P isa PM.FinitePoset
    end

    PLP = PM.PLPolyhedra
    if PLP.HAVE_POLY
        n = 1
        Uhp = PLP.make_hpoly([-1.0], 0.0)
        Dhp = PLP.make_hpoly([1.0], 2.0)
        U = PLP.PLUpset(PLP.PolyUnion(n, [Uhp]))
        D = PLP.PLDownset(PLP.PolyUnion(n, [Dhp]))
        F = PLP.PLFringe([U], [D], ones(K, 1, 1))
        enc = PM.EncodingOptions(backend=:pl, max_regions=10_000, poset_kind=:dense)
        P, _H, _pi = PM.encode(F, enc)
        @test P isa PM.FinitePoset
    end
end
end
end # with_fields
