using Test

# Included from test/runtests.jl; uses shared aliases (PM, PLP, PLB, FF, QQ, ...).

with_fields(FIELDS_FULL) do field
    K = CM.coeff_type(field)
    @inline c(x) = CM.coerce(field, x)
    if field isa CM.QQField

@testset "Rn wrappers: common encoding matches explicit encoding route" begin
    # This test checks that the one-line R^n wrappers:
    #   - build the same finite encoding poset P as an explicit "encode then compute" route, and
    #   - give the same answers as running the finite-poset homological algebra directly.
    #
    # IMPORTANT:
    # The PL -> finite encoding step requires Polyhedra/CDDLib for region enumeration.
    # If Polyhedra is not available, we only test that the wrapper fails loudly.

    # Build two simple box modules in R^2 using the current PLPolyhedra API.
    # PLUpset/PLDownset wrap PolyUnion(n, parts::Vector{HPoly}).
    # make_hpoly(A, b) builds an HPoly for inequalities A*x <= b.

    # F1: support approximately [0,1]^2
    U1_hp = PLP.make_hpoly(K[c(-1) c(0); c(0) c(-1)], K[c(0), c(0)])    # x >= 0, y >= 0
    D1_hp = PLP.make_hpoly(K[c(1) c(0); c(0) c(1)], K[c(1), c(1)])      # x <= 1, y <= 1
    U1 = PLP.PLUpset(PLP.PolyUnion(2, [U1_hp]))
    D1 = PLP.PLDownset(PLP.PolyUnion(2, [D1_hp]))
    F1 = PLP.PLFringe([U1], [D1], reshape(K[c(1)], 1, 1))

    # F2: support approximately [1,2]^2
    U2_hp = PLP.make_hpoly(K[c(-1) c(0); c(0) c(-1)], K[c(-1), c(-1)])  # x >= 1, y >= 1
    D2_hp = PLP.make_hpoly(K[c(1) c(0); c(0) c(1)], K[c(2), c(2)])      # x <= 2, y <= 2
    U2 = PLP.PLUpset(PLP.PolyUnion(2, [U2_hp]))
    D2 = PLP.PLDownset(PLP.PolyUnion(2, [D2_hp]))
    F2 = PLP.PLFringe([U2], [D2], reshape(K[c(1)], 1, 1))

    enc_pl = PM.EncodingOptions(backend=:pl, max_regions=50_000, strict_eps=PLP.STRICT_EPS_QQ)
    df2 = PM.DerivedFunctorOptions(maxdeg=2)
    res3 = PM.ResolutionOptions(maxlen=3)

    # If Polyhedra is missing, encoding is unavailable by design.
    if !PLP.HAVE_POLY
        @test_throws ErrorException PM.encode((F1, F2); enc=enc_pl)
        return
    end

    # Common-encode both PL presentations to the same finite poset P and modules Ms on P.
    enc = PM.encode((F1, F2); enc=enc_pl)
    P = enc[1].P
    Ms = [enc[1].M, enc[2].M]

    # Workflow-level auto-cache: encode() should attach an EncodingCache even
    # without an explicit SessionCache, and geometry calls should reuse it.
    enc_one = PM.encode(F1, PM.EncodingOptions(backend=:pl))
    @test enc_one.pi isa CM.CompiledEncoding
    @test enc_one.pi.meta isa CM.EncodingCache
    unit_box = ([0.0, 0.0], [1.0, 1.0])
    adj_1 = PM.RegionGeometry.region_adjacency(enc_one.pi; box=unit_box, strict=true, mode=:fast)
    adj_2 = PM.RegionGeometry.region_adjacency(enc_one.pi; box=unit_box, strict=true, mode=:fast)
    @test adj_2 == adj_1
    r_unit = CM.locate(enc_one.pi, [0.5, 0.5]; mode=:verified)
    @test r_unit != 0
    @test PM.RegionGeometry.region_bbox(enc_one.pi, r_unit; box=unit_box) == (Float64[0.0, 0.0], Float64[1.0, 1.0])
    @test isapprox(PM.RegionGeometry.region_diameter(enc_one.pi, r_unit; box=unit_box, method=:bbox, metric=:L2), sqrt(2.0); atol=1e-10)

    # Ext computed on P should match ExtRn wrapper (which internally does the same steps).
    E_explicit = DF.Ext(Ms[1], Ms[2], df2)
    E_wrap = DF.ExtRn(F1, F2, enc_pl, df2)
    @test [PM.dim(E_explicit, t) for t in 0:2] == [PM.dim(E_wrap, t) for t in 0:2]

    # Resolution wrapper should use the same encoded poset as explicit encoding.
    enc1 = PM.encode(F1, enc_pl)
    res_wrap = DF.projective_resolution_Rn(F1, enc_pl, res3; return_encoding=true)
    @test FF.poset_equal(res_wrap.P, enc1.P)
    @test DF.betti_table(res_wrap.res) == DF.betti_table(DF.projective_resolution(enc1.M, res3))

    # Minimal Betti data: obtain it by requesting a checked-minimal resolution.
    res_min = PM.ResolutionOptions(maxlen=3, minimal=true, check=true)

    bt_wrap = DF.betti(DF.projective_resolution_Rn(F1, enc_pl, res_min))
    bt_explicit = DF.betti(DF.projective_resolution(enc1.M, res_min))
    @test bt_wrap == bt_explicit
end




@testset "PL common encoding for multiple fringes" begin

    enc_pl_10k = PM.EncodingOptions(backend=:pl, max_regions=10_000)

    # We can build HPolys even without Polyhedra, but we can only encode if Polyhedra is available.
    if !PLP.HAVE_POLY
        U1 = PLP.PLUpset(PLP.PolyUnion(1, [PLP.make_hpoly([-1.0], [0.0])]))   # x >= 0
        D1 = PLP.PLDownset(PLP.PolyUnion(1, [PLP.make_hpoly([ 1.0], [2.0])])) # x <= 2
        F1 = PLP.PLFringe([U1], [D1], reshape(K[c(1)], 1, 1))

        @test_throws ErrorException PLP.encode_from_PL_fringes(F1, F1, enc_pl_10k)
    else
        # Two 1D modules: support [0,2] and support [1,3].
        U1 = PLP.PLUpset(PLP.PolyUnion(1, [PLP.make_hpoly([-1.0], [ 0.0])]))  # x >= 0
        D1 = PLP.PLDownset(PLP.PolyUnion(1, [PLP.make_hpoly([ 1.0], [ 2.0])]))# x <= 2

        U2 = PLP.PLUpset(PLP.PolyUnion(1, [PLP.make_hpoly([-1.0], [-1.0])]))  # x >= 1
        D2 = PLP.PLDownset(PLP.PolyUnion(1, [PLP.make_hpoly([ 1.0], [ 3.0])]))# x <= 3

        F1 = PLP.PLFringe([U1], [D1], reshape(K[c(1)], 1, 1))
        F2 = PLP.PLFringe([U2], [D2], reshape(K[c(1)], 1, 1))

        P, Hs, pi = PLP.encode_from_PL_fringes(F1, F2, enc_pl_10k)

        @test length(Hs) == 2
        H1, H2 = Hs[1], Hs[2]

        # Helper: insist locate returns a valid region.
        function fd(H, x)
            q = PLP.locate(pi, x)
            @test q != 0
            return FF.fiber_dimension(H, q)
        end

        @test fd(H1, [-1.0]) == 0
        @test fd(H2, [-1.0]) == 0

        @test fd(H1, [0.5]) == 1
        @test fd(H2, [0.5]) == 0

        @test fd(H1, [1.5]) == 1
        @test fd(H2, [1.5]) == 1

        @test fd(H1, [2.5]) == 0
        @test fd(H2, [2.5]) == 1

        @test fd(H1, [4.0]) == 0
        @test fd(H2, [4.0]) == 0
    end
end


@testset "PLPolyhedra optional backend" begin
    if PLP.HAVE_POLY
        # Unit square: 0 <= x <= 1, 0 <= y <= 1
        A = K[c(1)  c(0);
              c(0)  c(1);
              c(-1) c(0);
              c(0)  c(-1)]
        b = K[c(1), c(1), c(0), c(0)]
        h = PLP.make_hpoly(A, b)

        @test PLP._in_hpoly(h, [0, 0]) == true
        @test PLP._in_hpoly(h, [1, 1]) == true
        @test PLP._in_hpoly(h, [2, 0]) == false
        @test PLP._in_hpoly(h, [-1, 0]) == false
    else
        # Even without Polyhedra/CDDLib, make_hpoly should still build an HPoly
        # that supports exact membership tests via its stored A*x <= b data.
        A = reshape(K[c(1)], 1, 1)
        b = K[c(1)]
        h = PLP.make_hpoly(A, b)
        @test h.poly === nothing
        @test PLP._in_hpoly(h, [0]) == true
        @test PLP._in_hpoly(h, [1]) == true
        @test PLP._in_hpoly(h, [2]) == false
    end
end

@testset "PLPolyhedra hand-solvable 2D oracle fixtures" begin
    if !PLP.HAVE_POLY
        @test true
    else
        Arect = QQ[1 0; 0 1; -1 0; 0 -1]
        rect_hpoly(xl, xu, yl, yu) = PLP.make_hpoly(Arect, QQ[xu, yu, -xl, -yl])

        function sigbits(i::Int, k::Int)
            bv = BitVector(undef, k)
            x = i - 1
            @inbounds for b in 1:k
                bv[b] = ((x >>> (b - 1)) & 0x1) == 0x1
            end
            return bv
        end

        function run_fixture(rects, witnesses, box, expected; outside=Tuple{Float64,Float64}[])
            nreg = length(rects)
            @test length(witnesses) == nreg

            k = max(1, ceil(Int, log2(nreg + 1)))
            sigy = [sigbits(i, k) for i in 1:nreg]
            sigz = [falses(k) for _ in 1:nreg]
            hps = [rect_hpoly(r[1], r[2], r[3], r[4]) for r in rects]
            reps = [(w[1], w[2]) for w in witnesses]
            pi = PLP.PLEncodingMap(2, sigy, sigz, hps, reps)

            rid = Vector{Int}(undef, nreg)
            for i in 1:nreg
                rid[i] = CM.locate(pi, [witnesses[i][1], witnesses[i][2]]; mode=:verified)
                @test rid[i] != 0
            end
            @test length(unique(rid)) == nreg

            for p in outside
                @test CM.locate(pi, [p[1], p[2]]; mode=:verified) == 0
            end

            w = PM.RegionGeometry.region_weights(pi; box=box, method=:exact)
            for i in 1:nreg
                @test isapprox(w[rid[i]], expected[:weights][i]; atol=1e-10)
            end
            @test isapprox(sum(w), sum(expected[:weights]); atol=1e-10)

            for i in 1:nreg
                bb = PM.RegionGeometry.region_bbox(pi, rid[i]; box=box)
                @test bb !== nothing
                lo, hi = bb
                elo, ehi = expected[:bbox][i]
                @test all(isapprox.(lo, elo; atol=1e-10))
                @test all(isapprox.(hi, ehi; atol=1e-10))
            end

            for i in 1:nreg
                d = PM.RegionGeometry.region_diameter(pi, rid[i]; box=box, metric=:L2, method=:bbox)
                @test isapprox(d, expected[:diameters][i]; atol=1e-10)
            end

            for i in 1:nreg
                bm = PM.RegionGeometry.region_boundary_measure(pi, rid[i]; box=box, strict=true, mode=:verified)
                @test isapprox(bm, expected[:boundary][i]; atol=1e-8)
            end

            adj = PM.RegionGeometry.region_adjacency(pi; box=box, strict=true, mode=:verified)
            exp_adj = Dict{Tuple{Int,Int},Float64}()
            for ((i, j), m) in expected[:adj]
                ri = rid[i]
                rj = rid[j]
                key = ri < rj ? (ri, rj) : (rj, ri)
                exp_adj[key] = float(m)
            end
            @test Set(keys(adj)) == Set(keys(exp_adj))
            for (kpair, mexp) in exp_adj
                @test isapprox(adj[kpair], mexp; atol=1e-8)
            end
        end

        fixtures = [
            (
                rects=[(0.0, 1.0, 0.0, 1.0)],
                witnesses=[(0.5, 0.5)],
                box=(Float64[0.0, 0.0], Float64[1.0, 1.0]),
                expected=(
                    weights=[1.0],
                    bbox=[(Float64[0.0, 0.0], Float64[1.0, 1.0])],
                    diameters=[sqrt(2.0)],
                    boundary=[4.0],
                    adj=Dict{Tuple{Int,Int},Float64}(),
                ),
                outside=[(-0.1, 0.5), (1.1, 0.5)],
            ),
            (
                rects=[(0.0, 1.0, 0.0, 1.0), (1.0, 2.0, 0.0, 1.0)],
                witnesses=[(0.5, 0.5), (1.5, 0.5)],
                box=(Float64[0.0, 0.0], Float64[2.0, 1.0]),
                expected=(
                    weights=[1.0, 1.0],
                    bbox=[(Float64[0.0, 0.0], Float64[1.0, 1.0]),
                          (Float64[1.0, 0.0], Float64[2.0, 1.0])],
                    diameters=[sqrt(2.0), sqrt(2.0)],
                    boundary=[4.0, 4.0],
                    adj=Dict((1, 2) => 1.0),
                ),
                outside=[(-0.1, 0.5), (2.1, 0.5)],
            ),
            (
                rects=[(-1.0, 0.0, -1.0, 0.0), (0.0, 1.0, -1.0, 0.0), (-1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0)],
                witnesses=[(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)],
                box=(Float64[-1.0, -1.0], Float64[1.0, 1.0]),
                expected=(
                    weights=[1.0, 1.0, 1.0, 1.0],
                    bbox=[(Float64[-1.0, -1.0], Float64[0.0, 0.0]),
                          (Float64[0.0, -1.0], Float64[1.0, 0.0]),
                          (Float64[-1.0, 0.0], Float64[0.0, 1.0]),
                          (Float64[0.0, 0.0], Float64[1.0, 1.0])],
                    diameters=[sqrt(2.0), sqrt(2.0), sqrt(2.0), sqrt(2.0)],
                    boundary=[4.0, 4.0, 4.0, 4.0],
                    adj=Dict((1, 2) => 1.0, (1, 3) => 1.0, (2, 4) => 1.0, (3, 4) => 1.0),
                ),
                outside=[(-1.1, 0.0), (1.1, 0.0)],
            ),
            (
                rects=[(0.0, 1.0, 0.0, 2.0), (1.0, 3.0, 0.0, 2.0)],
                witnesses=[(0.5, 1.0), (2.0, 1.0)],
                box=(Float64[0.0, 0.0], Float64[3.0, 2.0]),
                expected=(
                    weights=[2.0, 4.0],
                    bbox=[(Float64[0.0, 0.0], Float64[1.0, 2.0]),
                          (Float64[1.0, 0.0], Float64[3.0, 2.0])],
                    diameters=[sqrt(5.0), sqrt(8.0)],
                    boundary=[6.0, 8.0],
                    adj=Dict((1, 2) => 2.0),
                ),
                outside=[(-0.1, 1.0), (3.1, 1.0)],
            ),
            (
                rects=[(0.0, 2.0, 0.0, 1.0), (0.0, 2.0, 1.0, 2.0), (0.0, 2.0, 2.0, 3.0)],
                witnesses=[(1.0, 0.5), (1.0, 1.5), (1.0, 2.5)],
                box=(Float64[0.0, 0.0], Float64[2.0, 3.0]),
                expected=(
                    weights=[2.0, 2.0, 2.0],
                    bbox=[(Float64[0.0, 0.0], Float64[2.0, 1.0]),
                          (Float64[0.0, 1.0], Float64[2.0, 2.0]),
                          (Float64[0.0, 2.0], Float64[2.0, 3.0])],
                    diameters=[sqrt(5.0), sqrt(5.0), sqrt(5.0)],
                    boundary=[6.0, 6.0, 6.0],
                    adj=Dict((1, 2) => 2.0, (2, 3) => 2.0),
                ),
                outside=[(-0.1, 0.5), (2.1, 0.5)],
            ),
            (
                rects=[(0.0, 1.0, 0.0, 1.0), (1.0, 3.0, 0.0, 1.0), (0.0, 1.0, 1.0, 2.0), (1.0, 3.0, 1.0, 2.0)],
                witnesses=[(0.5, 0.5), (2.0, 0.5), (0.5, 1.5), (2.0, 1.5)],
                box=(Float64[0.0, 0.0], Float64[3.0, 2.0]),
                expected=(
                    weights=[1.0, 2.0, 1.0, 2.0],
                    bbox=[(Float64[0.0, 0.0], Float64[1.0, 1.0]),
                          (Float64[1.0, 0.0], Float64[3.0, 1.0]),
                          (Float64[0.0, 1.0], Float64[1.0, 2.0]),
                          (Float64[1.0, 1.0], Float64[3.0, 2.0])],
                    diameters=[sqrt(2.0), sqrt(5.0), sqrt(2.0), sqrt(5.0)],
                    boundary=[4.0, 6.0, 4.0, 6.0],
                    adj=Dict((1, 2) => 1.0, (1, 3) => 1.0, (2, 4) => 2.0, (3, 4) => 1.0),
                ),
                outside=[(-0.1, 0.5), (3.1, 0.5)],
            ),
        ]

        for fx in fixtures
            run_fixture(fx.rects, fx.witnesses, fx.box, fx.expected; outside=fx.outside)
        end
    end
end

@testset "PLPolyhedra non-convex union oracle fixtures (2D, QQ)" begin
    if !PLP.HAVE_POLY
        @test true
    else
        Arect = QQ[1 0; 0 1; -1 0; 0 -1]
        rect_hpoly(xl, xu, yl, yu) = PLP.make_hpoly(Arect, QQ[xu, yu, -xl, -yl])

        # Non-convex geometry: two disjoint boxes with a gap in between.
        pi = PLP.PLEncodingMap(
            2,
            [BitVector((true, false)), BitVector((false, true))],
            [falses(2), falses(2)],
            [rect_hpoly(0.0, 1.0, 0.0, 1.0), rect_hpoly(2.0, 3.0, 0.0, 1.0)],
            [(0.5, 0.5), (2.5, 0.5)],
        )
        box = (Float64[0.0, 0.0], Float64[3.0, 1.0])

        t_left = CM.locate(pi, [0.5, 0.5]; mode=:verified)
        t_gap = CM.locate(pi, [1.5, 0.5]; mode=:verified)
        t_right = CM.locate(pi, [2.5, 0.5]; mode=:verified)
        @test t_left != 0 && t_right != 0
        @test t_gap == 0
        @test t_left != t_right
        @test CM.locate(pi, [1.5, 1.5]; mode=:verified) == 0

        w = PM.RegionGeometry.region_weights(pi; box=box, method=:exact)
        @test isapprox(w[t_left], 1.0; atol=1e-10)
        @test isapprox(w[t_right], 1.0; atol=1e-10)
        @test isapprox(sum(w), 2.0; atol=1e-10)

        bb_left = PM.RegionGeometry.region_bbox(pi, t_left; box=box)
        bb_right = PM.RegionGeometry.region_bbox(pi, t_right; box=box)
        @test bb_left == (Float64[0.0, 0.0], Float64[1.0, 1.0])
        @test bb_right == (Float64[2.0, 0.0], Float64[3.0, 1.0])

        @test isapprox(PM.RegionGeometry.region_diameter(pi, t_left; box=box, metric=:L2, method=:bbox), sqrt(2.0); atol=1e-10)
        @test isapprox(PM.RegionGeometry.region_diameter(pi, t_right; box=box, metric=:L2, method=:bbox), sqrt(2.0); atol=1e-10)

        @test isapprox(PM.RegionGeometry.region_boundary_measure(pi, t_left; box=box, strict=false, mode=:verified), 4.0; atol=1e-8)
        @test isapprox(PM.RegionGeometry.region_boundary_measure(pi, t_right; box=box, strict=false, mode=:verified), 4.0; atol=1e-8)

        adj = PM.RegionGeometry.region_adjacency(pi; box=box, strict=false, mode=:verified)
        @test isempty(adj)
    end
end

@testset "PLPolyhedra exact 3D oracle fixtures (QQ)" begin
    if !PLP.HAVE_POLY
        @test true
    else
        Acube = QQ[1 0 0; 0 1 0; 0 0 1; -1 0 0; 0 -1 0; 0 0 -1]
        box_hpoly(xl, xu, yl, yu, zl, zu) = PLP.make_hpoly(Acube, QQ[xu, yu, zu, -xl, -yl, -zl])

        function mk_pi3(hps, reps)
            n = length(hps)
            k = max(1, ceil(Int, log2(n + 1)))
            sigy = [BitVector(((i - 1) >>> (b - 1)) & 0x1 == 0x1 for b in 1:k) for i in 1:n]
            sigz = [falses(k) for _ in 1:n]
            return PLP.PLEncodingMap(3, sigy, sigz, hps, reps)
        end

        # Fixture A: single unit cube.
        hp = box_hpoly(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        pi = mk_pi3([hp], [(0.5, 0.5, 0.5)])
        box = (Float64[0.0, 0.0, 0.0], Float64[1.0, 1.0, 1.0])
        r = CM.locate(pi, [0.5, 0.5, 0.5]; mode=:verified)
        @test r != 0
        @test isapprox(PM.RegionGeometry.region_weights(pi; box=box, method=:exact)[r], 1.0; atol=1e-10)
        @test PM.RegionGeometry.region_bbox(pi, r; box=box) == (Float64[0.0, 0.0, 0.0], Float64[1.0, 1.0, 1.0])
        @test isapprox(PM.RegionGeometry.region_diameter(pi, r; box=box, metric=:L2, method=:bbox), sqrt(3.0); atol=1e-10)
        @test isapprox(PM.RegionGeometry.region_boundary_measure(pi, r; box=box, mode=:verified), 6.0; atol=1e-8)
        @test isempty(PM.RegionGeometry.region_adjacency(pi; box=box, strict=true, mode=:verified))

        # Fixture B: two adjacent unit cubes sharing one face.
        hp1 = box_hpoly(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        hp2 = box_hpoly(1.0, 2.0, 0.0, 1.0, 0.0, 1.0)
        pi2 = mk_pi3([hp1, hp2], [(0.5, 0.5, 0.5), (1.5, 0.5, 0.5)])
        box2 = (Float64[0.0, 0.0, 0.0], Float64[2.0, 1.0, 1.0])
        r1 = CM.locate(pi2, [0.5, 0.5, 0.5]; mode=:verified)
        r2 = CM.locate(pi2, [1.5, 0.5, 0.5]; mode=:verified)
        @test r1 != 0 && r2 != 0 && r1 != r2
        w2 = PM.RegionGeometry.region_weights(pi2; box=box2, method=:exact)
        @test isapprox(w2[r1], 1.0; atol=1e-10)
        @test isapprox(w2[r2], 1.0; atol=1e-10)
        @test isapprox(PM.RegionGeometry.region_boundary_measure(pi2, r1; box=box2, mode=:verified), 6.0; atol=1e-8)
        @test isapprox(PM.RegionGeometry.region_boundary_measure(pi2, r2; box=box2, mode=:verified), 6.0; atol=1e-8)
        adj2 = PM.RegionGeometry.region_adjacency(pi2; box=box2, strict=true, mode=:verified)
        key = r1 < r2 ? (r1, r2) : (r2, r1)
        @test haskey(adj2, key)
        @test isapprox(adj2[key], 1.0; atol=1e-8)

        # Fixture C: rectangular prism.
        hp3 = box_hpoly(0.0, 2.0, 0.0, 1.0, 0.0, 3.0)
        pi3 = mk_pi3([hp3], [(1.0, 0.5, 1.5)])
        box3 = (Float64[0.0, 0.0, 0.0], Float64[2.0, 1.0, 3.0])
        r3 = CM.locate(pi3, [1.0, 0.5, 1.5]; mode=:verified)
        @test r3 != 0
        @test isapprox(PM.RegionGeometry.region_weights(pi3; box=box3, method=:exact)[r3], 6.0; atol=1e-10)
        @test isapprox(PM.RegionGeometry.region_diameter(pi3, r3; box=box3, metric=:L2, method=:bbox), sqrt(14.0); atol=1e-10)
        @test isapprox(PM.RegionGeometry.region_boundary_measure(pi3, r3; box=box3, mode=:verified), 22.0; atol=1e-8)
    end
end

@testset "PLBackend axis encoding (axis-aligned boxes)" begin
    enc_axis = PM.EncodingOptions(backend=:pl_backend)
    enc_axis_small = PM.EncodingOptions(backend=:pl_backend, max_regions=2)

    # One-dimensional example:
    # Birth upset:  x >= 0
    # Death downset: x <= 2
    # phi = [1], so the represented module should be supported exactly in the middle region.
    Ups = [PLB.BoxUpset([0.0])]
    Downs = [PLB.BoxDownset([2.0])]
    Phi = reshape(K[c(1)], 1, 1)

    # Error guard: if max_regions is too small, the backend should refuse.
    @test_throws ErrorException PLB.encode_fringe_boxes(Ups, Downs, Phi, enc_axis_small)

    P, H, pi = PLB.encode_fringe_boxes(Ups, Downs, Phi, enc_axis)

    @test P.n == 3
    @test pi.coords[1] == [0.0, 2.0]

    # The encoded poset should be a 3-element chain.
    @test Set(FF.cover_edges(P)) == Set([(1, 2), (2, 3)])

    # Locate regions by sampling points in each cell and check fiber dimensions.
    @test FF.fiber_dimension(H, PLB.locate(pi, [-1.0])) == 0   # left of 0: outside upset
    @test FF.fiber_dimension(H, PLB.locate(pi, [ 1.0])) == 1   # between: in upset and downset
    @test FF.fiber_dimension(H, PLB.locate(pi, [ 3.0])) == 0   # right of 2: outside downset

        # Boundary points: should agree with inequality convention (>= ell, <= u).
    @test FF.fiber_dimension(H, PLB.locate(pi, [0.0])) == 1
    @test FF.fiber_dimension(H, PLB.locate(pi, [2.0])) == 1

    @testset "locate caches (O(1) point location)" begin
        # Dense cell map should be present for encoded maps.
        @test !isempty(pi.cell_to_region)
        @test length(pi.cell_to_region) == prod(map(c -> length(c) + 1, pi.coords))

        # Signature Dict should be present and cover every region.
        @test length(pi.sig_to_region) == length(pi.sig_y)

        # For each 1D cell, locate(rep) should equal the cached value.
        ci = pi.coords[1]
        reps = [ci[1] - 1.0, (ci[1] + ci[2]) / 2, ci[end] + 1.0]
        for (k, x) in enumerate(reps)
            @test PLB.locate(pi, [x]) == pi.cell_to_region[k]
        end

        # Warm-up compile, then ensure tuple input is allocation-free.
        # Use the generic exported `locate` via the constant module `PM` so the
        # call is fully inferred (otherwise a module-valued `PLB` can force
        # dynamic dispatch and show spurious allocations).
        CM.locate(pi, (1.0,))
        @test (@allocated CM.locate(pi, (1.0,))) == 0

        # Canonical reconstruction: encode_fringe_boxes is the supported constructor.
        _, _, pi2 = PLB.encode_fringe_boxes(Ups, Downs, Phi, enc_axis)
        @test PLB.locate(pi2, [1.0]) == PLB.locate(pi, [1.0])
        @test PLB.locate(pi2, [0.0]) == PLB.locate(pi, [0.0])
        @test PLB.locate(pi2, [2.0]) == PLB.locate(pi, [2.0])

        # lock in the removal of the old 7-arg constructor.
        @test_throws MethodError PLB.PLEncodingMapBoxes(pi.n, pi.coords, pi.sig_y, pi.sig_z, pi.reps, pi.Ups, pi.Downs)
    end

    @testset "locate caches in 2D" begin
        Ups2 = [PLB.BoxUpset([0.0, 0.0])]
        Downs2 = [PLB.BoxDownset([2.0, 2.0])]
        Phi2 = reshape(K[c(1)], 1, 1)

        P2, H2, pi2 = PLB.encode_fringe_boxes(Ups2, Downs2, Phi2, enc_axis)

        @test !isempty(pi2.cell_to_region)
        @test length(pi2.cell_to_region) == (length(pi2.coords[1]) + 1) * (length(pi2.coords[2]) + 1)

        r_in  = PLB.locate(pi2, [ 1.0,  1.0])  # in upset and in downset
        r_q   = PLB.locate(pi2, [ 3.0,  1.0])  # in upset, outside downset
        r_d   = PLB.locate(pi2, [-1.0,  1.0])  # in downset, outside upset
        r_out = PLB.locate(pi2, [-1.0,  3.0])  # outside both

        @test length(Set([r_in, r_q, r_d, r_out])) == 4

        # Boundary checks: x=2 and x=0 are inclusive as expected.
        @test PLB.locate(pi2, [2.0, 1.0]) == r_in   # on u boundary
        @test PLB.locate(pi2, [0.0, 1.0]) == r_in   # on ell boundary

        # Tuple input: allocation-free after warm-up.
        # Same note as in 1D: use the exported `locate` via `PM` to avoid
        # dynamic module-property dispatch.
        CM.locate(pi2, (1.0, 1.0))
        @test (@allocated CM.locate(pi2, (1.0, 1.0))) == 0
    end

    @testset "incremental signature traversal consistency" begin
        Ups3 = [PLB.BoxUpset([0.0, 0.0]), PLB.BoxUpset([2.0, -1.0])]
        Downs3 = [PLB.BoxDownset([3.0, 1.0]), PLB.BoxDownset([1.0, 2.0])]
        Phi3 = zeros(K, length(Downs3), length(Ups3))

        _, _, pi3 = PLB.encode_fringe_boxes(Ups3, Downs3, Phi3, enc_axis)
        MY = cld(length(Ups3), 64)
        MZ = cld(length(Downs3), 64)

        lin = 1
        for I in CartesianIndices(Tuple(pi3.cell_shape))
            idx0 = ntuple(j -> I[j] - 1, pi3.n)
            x = PLB._cell_rep_axis(pi3.coords, idx0)
            y, z = PLB._signature(x, Ups3, Downs3)
            key = PLB._sigkey_from_bitvectors(y, z, Val(MY), Val(MZ))
            rid = get(pi3.sig_to_region, key, 0)
            @test rid != 0
            @test pi3.cell_to_region[lin] == rid
            lin += 1
        end
    end

end

@testset "PLBackend near-boundary parity (fast vs verified)" begin
    Ups = [PLB.BoxUpset([0.0, 0.0])]
    Downs = [PLB.BoxDownset([2.0, 2.0])]
    enc = PM.encode(Ups, Downs; backend=:pl_backend, field=field, output=:result, cache=:auto)

    epss = (1e-4, 1e-7, 1e-10, 1e-12)
    probes = Vector{Tuple{Float64,Float64}}()
    for e in epss
        push!(probes, (0.0 + e, 1.0))
        push!(probes, (0.0 - e, 1.0))
        push!(probes, (2.0 + e, 1.0))
        push!(probes, (2.0 - e, 1.0))
        push!(probes, (1.0, 0.0 + e))
        push!(probes, (1.0, 0.0 - e))
        push!(probes, (1.0, 2.0 + e))
        push!(probes, (1.0, 2.0 - e))
    end
    # Exact boundary probes included too.
    append!(probes, ((0.0, 1.0), (2.0, 1.0), (1.0, 0.0), (1.0, 2.0), (0.0, 0.0), (2.0, 2.0)))

    for x in probes
        rf = CM.locate(enc.pi, x; mode=:fast)
        rv = CM.locate(enc.pi, x; mode=:verified)
        @test rf == rv
        @test rf != 0
    end
end

@testset "PL runtime budget guards (deterministic, QQ)" begin
    if field isa CM.QQField
        @inline function _median_elapsed(f::Function; reps::Int=5)
            ts = Vector{Float64}(undef, reps)
            for i in 1:reps
                ts[i] = @elapsed f()
            end
            return sort(ts)[cld(reps, 2)]
        end
        @inline _ns_per_item(t::Float64, n::Int) = (t * 1.0e9) / max(1, n)
        strict_ci = get(ENV, "TAMER_STRICT_PERF_CI", "1") == "1"

        # PLBackend hot locate loop: fixed workload + conservative budget.
        Ups = [PLB.BoxUpset([0.0])]
        Downs = [PLB.BoxDownset([2.0])]
        enc_axis = PM.encode(Ups, Downs; backend=:pl_backend, output=:result, cache=:auto)
        xs = range(-2.0, 7.0; length=20_000)
        CM.locate(enc_axis.pi, (0.5,); mode=:fast) # warmup
        CM.locate(enc_axis.pi, (0.5,); mode=:verified) # warmup
        axis_fast = _median_elapsed() do
            s = 0
            @inbounds for x in xs
                s += CM.locate(enc_axis.pi, (x,); mode=:fast)
            end
            @test s > 0
        end
        axis_verified = _median_elapsed() do
            s = 0
            @inbounds for x in xs
                s += CM.locate(enc_axis.pi, (x,); mode=:verified)
            end
            @test s > 0
        end
        # Platform-normalized envelope (ns/query): strict in CI, looser local fallback.
        axis_fast_ns = _ns_per_item(axis_fast, length(xs))
        axis_verified_ns = _ns_per_item(axis_verified, length(xs))
        if strict_ci
            @test axis_fast_ns <= 1.15 * axis_verified_ns + 10.0
        else
            @test axis_fast_ns <= 1.3 * axis_verified_ns + 20.0
        end

        if PLP.HAVE_POLY
            # PLPolyhedra cached-vs-uncached locate_many! guard on fixed points.
            A1 = QQ[1 0; 0 1; -1 0; 0 -1]
            b1 = QQ[1, 1, 0, 0]
            A2 = QQ[1 0; 0 1; -1 0; 0 -1]
            b2 = QQ[2, 1, -1, 0]
            hp1 = PLP.make_hpoly(A1, b1)
            hp2 = PLP.make_hpoly(A2, b2)
            pi = PLP.PLEncodingMap(2,
                                   [BitVector(), BitVector()],
                                   [BitVector(), BitVector()],
                                   [hp1, hp2],
                                   [(0.5, 0.5), (1.5, 0.5)])
            box = (Float64[0, 0], Float64[2, 1])
            cache = PLP.compile_geometry_cache(pi; box=box, closure=true)

            npts = 12_000
            X = Matrix{Float64}(undef, 2, npts)
            @inbounds for j in 1:npts
                # Deterministic pseudo-grid samples.
                X[1, j] = (j % 2000) / 1000
                X[2, j] = ((j * 7) % 1000) / 1000
            end
            dest_uncached = zeros(Int, npts)
            dest_cached = zeros(Int, npts)

            # Warmup
            PLP.locate_many!(dest_uncached, pi, X; threaded=false, mode=:fast)
            PLP.locate_many!(dest_cached, cache, X; threaded=false, mode=:fast)
            t_uncached = _median_elapsed() do
                PLP.locate_many!(dest_uncached, pi, X; threaded=false, mode=:fast)
            end
            t_cached = _median_elapsed() do
                PLP.locate_many!(dest_cached, cache, X; threaded=false, mode=:fast)
            end

            @test dest_cached == dest_uncached
            many_uncached_ns = _ns_per_item(t_uncached, npts)
            many_cached_ns = _ns_per_item(t_cached, npts)
            # Platform-normalized envelope: cache should be faster on per-query cost.
            if strict_ci
                @test many_cached_ns <= 1.10 * many_uncached_ns + 20.0
            else
                @test many_cached_ns <= 1.2 * many_uncached_ns + 30.0
            end
        end
    end
end

@testset "PLPolyhedra adversarial degenerate geometry fixtures (QQ)" begin
    if !PLP.HAVE_POLY || !(field isa CM.QQField)
        @test true
    else
        # Fixture A: near-coplanar thin slab in 3D.
        eps = 1.0e-6
        A3 = QQ[1 0 0; 0 1 0; 0 0 1; -1 0 0; 0 -1 0; 0 0 -1]
        b3 = QQ[1, 1, eps, 0, 0, 0]
        hp3 = PLP.make_hpoly(A3, b3)
        pi3 = PLP.PLEncodingMap(3, [BitVector([true])], [BitVector([false])], [hp3], [(0.5, 0.5, eps / 2)])
        box3 = (Float64[0.0, 0.0, 0.0], Float64[1.0, 1.0, eps])
        r3 = CM.locate(pi3, [0.5, 0.5, eps / 2]; mode=:verified)
        @test r3 != 0
        w3 = PM.RegionGeometry.region_weights(pi3; box=box3, method=:exact)
        @test isapprox(w3[r3], eps; atol=1e-11)
        @test PM.RegionGeometry.region_bbox(pi3, r3; box=box3) == (Float64[0.0, 0.0, 0.0], Float64[1.0, 1.0, eps])

        for d in (1.0e-7, 1.0e-10, 1.0e-12)
            @test CM.locate(pi3, [0.5, 0.5, eps - d]; mode=:fast) == CM.locate(pi3, [0.5, 0.5, eps - d]; mode=:verified)
            @test CM.locate(pi3, [0.5, 0.5, eps + d]; mode=:fast) == CM.locate(pi3, [0.5, 0.5, eps + d]; mode=:verified)
        end

        # Fixture B: high-dimensional thin orthotope in 4D.
        t = 1.0e-4
        A4 = QQ[
            1 0 0 0;
            0 1 0 0;
            0 0 1 0;
            0 0 0 1;
            -1 0 0 0;
            0 -1 0 0;
            0 0 -1 0;
            0 0 0 -1
        ]
        b4 = QQ[1, 1, t, t, 0, 0, 0, 0]
        hp4 = PLP.make_hpoly(A4, b4)
        pi4 = PLP.PLEncodingMap(4, [BitVector([true])], [BitVector([false])], [hp4], [(0.5, 0.5, t / 2, t / 2)])
        box4 = (Float64[0.0, 0.0, 0.0, 0.0], Float64[1.0, 1.0, t, t])
        r4 = CM.locate(pi4, [0.5, 0.5, t / 2, t / 2]; mode=:verified)
        @test r4 != 0
        @test PM.RegionGeometry.region_bbox(pi4, r4; box=box4) == (Float64[0.0, 0.0, 0.0, 0.0], Float64[1.0, 1.0, t, t])
        @test isapprox(
            PM.RegionGeometry.region_diameter(pi4, r4; box=box4, metric=:L2, method=:bbox),
            sqrt(2.0 + 2.0 * t * t);
            atol=1e-10,
        )

        probes4 = (
            [0.5, 0.5, 0.0, t / 2],
            [0.5, 0.5, t, t / 2],
            [0.5, 0.5, t + 1.0e-8, t / 2],
            [0.5, 0.5, t / 2, t + 1.0e-8],
        )
        for x in probes4
            @test CM.locate(pi4, x; mode=:fast) == CM.locate(pi4, x; mode=:verified)
        end
    end
end

    end # field isa CM.QQField

    if !(field isa CM.QQField)
@testset "PL non-QQ field parity ($(field))" begin
    # Axis-aligned backend with non-QQ target field: geometry should stay the same,
    # while encoded module coefficients are coerced to the requested field.
    Ups = [PLB.BoxUpset([0.0])]
    Downs = [PLB.BoxDownset([2.0])]
    enc = PM.encode(Ups, Downs; backend=:pl_backend, field=field, output=:result, cache=:auto)

    @test enc.M.field == field
    @test enc.H.field == field

    box = ([-2.0], [7.0])
    w = PM.RegionGeometry.region_weights(enc.pi; box=box)
    t_left = CM.locate(enc.pi, [-1.0])
    t_mid = CM.locate(enc.pi, [1.0])
    t_right = CM.locate(enc.pi, [3.0])

    @test FF.fiber_dimension(enc.H, t_left) == 0
    @test FF.fiber_dimension(enc.H, t_mid) == 1
    @test FF.fiber_dimension(enc.H, t_right) == 0

    @test isapprox(w[t_left], 2.0; atol=1e-12)
    @test isapprox(w[t_mid], 2.0; atol=1e-12)
    @test isapprox(w[t_right], 5.0; atol=1e-12)
    @test isapprox(sum(w), 9.0; atol=1e-12)
end

@testset "PLBackend mode parity for non-QQ fields ($(field))" begin
    Ups = [PLB.BoxUpset([0.0, 0.0])]
    Downs = [PLB.BoxDownset([2.0, 2.0])]
    enc = PM.encode(Ups, Downs; backend=:pl_backend, field=field, output=:result, cache=:auto)
    box = ([-1.0, -1.0], [3.0, 3.0])

    adj_fast = PM.RegionGeometry.region_adjacency(enc.pi; box=box, strict=true, mode=:fast)
    adj_verified = PM.RegionGeometry.region_adjacency(enc.pi; box=box, strict=true, mode=:verified)
    @test adj_fast == adj_verified

    # Boundary and interior points should classify identically in both modes.
    @test CM.locate(enc.pi, [0.0, 1.0]; mode=:fast) == CM.locate(enc.pi, [0.0, 1.0]; mode=:verified)
    @test CM.locate(enc.pi, [2.0, 1.0]; mode=:fast) == CM.locate(enc.pi, [2.0, 1.0]; mode=:verified)
    @test CM.locate(enc.pi, [1.0, 1.0]; mode=:fast) == CM.locate(enc.pi, [1.0, 1.0]; mode=:verified)
end

@testset "PLPolyhedra non-QQ coercion and geometry parity ($(field))" begin
    if !PLP.HAVE_POLY
        @test true
    else
        A_up = -Matrix{QQ}(I, 1, 1)
        b_up = QQ[0]
        A_down = Matrix{QQ}(I, 1, 1)
        b_down = QQ[2]
        hp_up = PLP.make_hpoly(A_up, b_up)
        hp_down = PLP.make_hpoly(A_down, b_down)
        F = PLP.PLFringe(
            [PLP.PLUpset(PLP.PolyUnion(1, [hp_up]))],
            [PLP.PLDownset(PLP.PolyUnion(1, [hp_down]))],
            reshape(QQ[1], 1, 1),
        )
        enc_opts = PM.EncodingOptions(backend=:pl, field=field)
        enc = PM.encode(F, enc_opts; output=:result, cache=:auto)

        @test enc.M.field == field
        @test enc.H.field == field

        box = ([-2.0], [7.0])
        w = PM.RegionGeometry.region_weights(enc.pi; box=box, method=:exact)
        t_left = CM.locate(enc.pi, [-1.0]; mode=:verified)
        t_mid = CM.locate(enc.pi, [1.0]; mode=:verified)
        t_right = CM.locate(enc.pi, [3.0]; mode=:verified)

        @test FF.fiber_dimension(enc.H, t_left) == 0
        @test FF.fiber_dimension(enc.H, t_mid) == 1
        @test FF.fiber_dimension(enc.H, t_right) == 0

        @test isapprox(w[t_left], 2.0; atol=1e-9)
        @test isapprox(w[t_mid], 2.0; atol=1e-9)
        @test isapprox(w[t_right], 5.0; atol=1e-9)
        @test isapprox(sum(w), 9.0; atol=1e-8)
    end
end

@testset "PLPolyhedra near-boundary parity (fast vs verified, $(field))" begin
    if !PLP.HAVE_POLY
        @test true
    else
        A = reshape(QQ[1, -1], 2, 1)
        hp1 = PLP.make_hpoly(A, QQ[1, 0])   # [0,1]
        hp2 = PLP.make_hpoly(A, QQ[2, -1])  # [1,2]
        pi = PLP.PLEncodingMap(1,
                               [BitVector([false]), BitVector([true])],
                               [BitVector([false]), BitVector([false])],
                               [hp1, hp2],
                               [(0.5,), (1.5,)])

        epss = (1e-4, 1e-7, 1e-10, 1e-12)
        probes = Float64[0.0, 1.0, 2.0]
        for e in epss
            append!(probes, (0.0 - e, 0.0 + e, 1.0 - e, 1.0 + e, 2.0 - e, 2.0 + e))
        end
        for x in probes
            rf = PLP.locate(pi, [x]; mode=:fast)
            rv = PLP.locate(pi, [x]; mode=:verified)
            @test rf == rv
        end
    end
end

@testset "PL non-QQ deep geometry hand-oracles ($(field))" begin
    if field isa CM.QQField || !PLP.HAVE_POLY
        @test true
    else
        # 2D unit square exact oracle.
        A2 = QQ[1 0; 0 1; -1 0; 0 -1]
        b2 = QQ[1, 1, 0, 0]
        hp2 = PLP.make_hpoly(A2, b2)
        pi2 = PLP.PLEncodingMap(2, [BitVector([true])], [BitVector([false])], [hp2], [(0.5, 0.5)])
        box2 = (Float64[0.0, 0.0], Float64[1.0, 1.0])
        r2 = CM.locate(pi2, [0.5, 0.5]; mode=:verified)
        @test r2 != 0
        w2 = PM.RegionGeometry.region_weights(pi2; box=box2, method=:exact)
        @test isapprox(w2[r2], 1.0; atol=1e-10)
        @test isapprox(sum(w2), 1.0; atol=1e-10)
        @test PM.RegionGeometry.region_bbox(pi2, r2; box=box2) == (Float64[0.0, 0.0], Float64[1.0, 1.0])
        @test isapprox(PM.RegionGeometry.region_diameter(pi2, r2; box=box2, metric=:L2, method=:bbox), sqrt(2.0); atol=1e-10)
        @test isapprox(PM.RegionGeometry.region_boundary_measure(pi2, r2; box=box2, mode=:verified), 4.0; atol=1e-8)
        @test isempty(PM.RegionGeometry.region_adjacency(pi2; box=box2, strict=true, mode=:verified))

        # 3D unit cube exact oracle.
        A3 = QQ[1 0 0; 0 1 0; 0 0 1; -1 0 0; 0 -1 0; 0 0 -1]
        b3 = QQ[1, 1, 1, 0, 0, 0]
        hp3 = PLP.make_hpoly(A3, b3)
        pi3 = PLP.PLEncodingMap(3, [BitVector([true])], [BitVector([false])], [hp3], [(0.5, 0.5, 0.5)])
        box3 = (Float64[0.0, 0.0, 0.0], Float64[1.0, 1.0, 1.0])
        r3 = CM.locate(pi3, [0.5, 0.5, 0.5]; mode=:verified)
        @test r3 != 0
        w3 = PM.RegionGeometry.region_weights(pi3; box=box3, method=:exact)
        @test isapprox(w3[r3], 1.0; atol=1e-10)
        @test isapprox(PM.RegionGeometry.region_boundary_measure(pi3, r3; box=box3, mode=:verified), 6.0; atol=1e-8)
        @test isapprox(PM.RegionGeometry.region_diameter(pi3, r3; box=box3, metric=:L2, method=:bbox), sqrt(3.0); atol=1e-10)
        @test isempty(PM.RegionGeometry.region_adjacency(pi3; box=box3, strict=true, mode=:verified))
    end
end
    end
end # with_fields
