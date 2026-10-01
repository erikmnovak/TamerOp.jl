# Independent complete-line oracles: no sampled endpoint is treated as a
# certificate. These fixtures specify their interval modules geometrically.

function _a41_global_square_encoding(field=CM.QQField())
    K = CM.coeff_type(field)
    return TamerOp.encode([TOA.BoxUpset([0.0,0.0]), TOA.BoxUpset([1.0,1.0])],
        [TOA.BoxDownset([2.0,2.0]), TOA.BoxDownset([3.0,3.0])],
        K[1 0;0 1], TOA.EncodingOptions(;backend=:pl_backend,poset_kind=:signature,field))
end

function _a41_global_band_encoding(field=CM.QQField())
    K = CM.coeff_type(field)
    # The three disjoint regions are x+y<0, 0<=x+y<2, and x+y>=2.
    regions = [PLP.HPoly(2,QQ[1 1],QQ[0],nothing,BitVector([true]),zero(QQ)),
        PLP.HPoly(2,QQ[-1 -1;1 1],QQ[0,2],nothing,BitVector([false,true]),zero(QQ)),
        PLP.HPoly(2,QQ[-1 -1],QQ[-2],nothing,BitVector([false]),zero(QQ))]
    pi = PLP.PLEncodingMap(2,[BitVector() for _ in regions],
        [BitVector() for _ in regions],regions,[(-1,0),(1,0),(3,0)])
    P = chain_poset(3)
    # e1 persists on the whole line; e2 dies at x+y=2; e3 starts at x+y=0.
    M = MD.PModule{K}(P,[2,3,2],Dict((1,2)=>K[1 0;0 1;0 0],
        (2,3)=>K[1 0 0;0 0 1]);field)
    return RES.EncodingResult(P,M,EC.compile_encoding(P,pi))
end

_a41_interval_contains(r,t) =
    (r.birth < t || (r.left_closed && r.birth == t)) &&
    (t < r.death || (r.right_closed && t == r.death))

@testset "A41 global slices preserve actual endpoints outside the viewport" begin
    V = TamerOp.Visualization
    enc = _a41_global_square_encoding()
    line = (basepoint=(0,0),direction=(1,1))
    s = V.inspection_session(enc;box=([3//2,3//2],[7//4,7//4]),slice=line,slice_scope=:global)
    away = V.inspection_session(enc;box=([10,10],[11,11]),slice=line,slice_scope=:global)
    try
        result = V.inspection_snapshot(s).metadata.slice_result
        @test result.scope === :global
        @test result.domain == (-Inf,Inf)
        @test result.window == (3//2,7//4)
        @test result.endpoint_semantics === :decorated_global
        @test result.essential_status === :certified
        @test Set((r.birth,r.death,r.left_closed,r.right_closed,r.multiplicity)
            for r in result.intervals) == Set(((0,2,true,true,1),(1,3,true,true,1)))
        @test all(r -> !r.left_clipped && !r.right_clipped,result.intervals)
        @test V.inspection_snapshot(away).metadata.slice_result.intervals == result.intervals
        @test V.check_visual_spec(V.inspection_snapshot(s)).valid
        @test V.check_visual_spec(V.inspection_snapshot(away)).valid

        V.select_inspection!(s;interval=2)
        V.select_inspection!(s;slice_scope=:window)
        restricted = V.inspection_snapshot(s).metadata.slice_result
        @test restricted.scope === :window
        @test restricted.domain == restricted.window
        @test restricted.essential_status === :not_inferred
        @test V.inspection_selection(s).interval === nothing
        r = only(restricted.intervals)
        @test (r.birth,r.death,r.multiplicity) == (3//2,7//4,2)
        @test r.left_clipped && r.right_clipped
        misses = V.inspection_summary(s).slice_cache_misses
        hits = V.inspection_summary(s).slice_cache_hits
        V.select_inspection!(s;slice_scope=:global)
        @test V.inspection_snapshot(s).metadata.slice_result === result
        @test V.inspection_summary(s).slice_cache_misses == misses
        @test V.inspection_summary(s).slice_cache_hits == hits + 1
        @test V.check_inspection_session(s).valid

        # A tangent has isolated closed stalks even when neither is in view.
        V.select_inspection!(away;slice=(basepoint=(0,2),direction=(1,1)))
        tangent = V.inspection_snapshot(away).metadata.slice_result
        @test tangent.window === nothing
        @test Set((r.birth,r.death,r.singleton) for r in tangent.intervals) ==
            Set(((0,0,true),(1,1,true)))
        @test V.check_visual_spec(V.inspection_snapshot(away)).valid
    finally
        foreach(V.close_inspection!, (s,away))
    end
end

@testset "A41 certified infinite tails preserve ranks and endpoint decoration" begin
    V = TamerOp.Visualization
    for field in FIELDS_FULL
        enc = _a41_global_band_encoding(field)
        s = V.inspection_session(enc;box=([1//4,1//4],[3//4,3//4]),
            slice=(basepoint=(0,0),direction=(1,1)),slice_scope=:global)
        try
            result = V.inspection_snapshot(s).metadata.slice_result
            @test Set((r.birth,r.death,r.left_closed,r.right_closed,r.multiplicity)
                for r in result.intervals) == Set(((-Inf,Inf,false,false,1),
                    (-Inf,1,false,false,1),(0,Inf,true,false,1)))
            @test all(r -> !r.left_clipped && !r.right_clipped,result.intervals)
            @test all(r -> !r.singleton,result.intervals)
            for (t,expected) in ((-100,2),(0,3),(1//2,3),(1,2),(100,2))
                @test sum(r.multiplicity for r in result.intervals
                    if _a41_interval_contains(r,t);init=0) == expected
                V.select_inspection!(s;point=(t,t))
                @test V.inspection_snapshot(s).metadata.inspection.dimension == expected
            end
            V.select_inspection!(s;parameter_pair=((-1,-1),(2,2)))
            @test V.inspection_snapshot(s).metadata.inspection.rank == 1
            @test V.check_visual_spec(V.inspection_snapshot(s)).valid
            @test V.check_inspection_session(s).valid
        finally
            V.close_inspection!(s)
        end
    end
end

@testset "A41 event-free whole lines and exact nearest-lattice tails" begin
    V = TamerOp.Visualization
    # An empty inequality list certifies a constant classifier on all R^2.
    hp = PLP.HPoly(2,zeros(QQ,0,2),QQ[],nothing,falses(0),zero(QQ))
    pi = PLP.PLEncodingMap(2,[BitVector()],[BitVector()],[hp],[(0,0)])
    P = chain_poset(1)
    for dim in (0,2)
        M = MD.PModule{QQ}(P,[dim],Dict{Tuple{Int,Int},Matrix{QQ}}();field=CM.QQField())
        enc = RES.EncodingResult(P,M,EC.compile_encoding(P,pi))
        s = V.inspection_session(enc;box=([-1,-1],[1,1]),slice_limit=1,
            slice=(basepoint=(5,0),direction=(0,1)),slice_scope=:global)
        try
            result = V.inspection_snapshot(s).metadata.slice_result
            @test result.window === nothing
            @test isempty(result.events)
            @test result.sample_parameters == (0,)
            @test result.chain == (1,)
            @test result.essential_status === :certified
            if dim == 0
                @test isempty(result.intervals)
            else
                r = only(result.intervals)
                @test (r.birth,r.death,r.multiplicity,r.left_closed,r.right_closed) ==
                    (-Inf,Inf,2,false,false)
            end
            @test V.check_visual_spec(V.inspection_snapshot(s)).valid
        finally
            V.close_inspection!(s)
        end
    end

    face = FZ.Face(2,[false,false])
    for (a,b,closed) in ((0,2,true),(1,1,false))
        flange = FZ.Flange(2,[FZ.IndFlat(face,(a,a);id=:U)],
            [FZ.IndInj(face,(b,b);id=:D)],reshape(QQ[1],1,1);field=CM.QQField())
        enc = TamerOp.encode(flange;backend=:zn)
        s = V.inspection_session(enc;box=([3//4,3//4],[5//4,5//4]),
            slice=(basepoint=(0,0),direction=(1,1)),slice_scope=:global)
        try
            r = only(V.inspection_snapshot(s).metadata.slice_result.intervals)
            @test (r.birth,r.death,r.left_closed,r.right_closed) ==
                (a-1//2,b+1//2,closed,closed)
            classifier = V._inspection_classifier(TamerOp.encoding_map(enc))
            # Huge tail representatives remain in the same classifier slabs;
            # their conversion cannot overflow a machine integer.
            huge = big(10)^100
            @test V._inspection_slice_locate(classifier,(huge,huge)) ==
                EC.locate(classifier,(b+2,b+2))
            @test V._inspection_slice_locate(classifier,(-huge,-huge)) ==
                EC.locate(classifier,(a-2,a-2))
            @test V.check_visual_spec(V.inspection_snapshot(s)).valid
        finally
            V.close_inspection!(s)
        end
    end
end

@testset "A41 uncovered domains and global budgets fail transactionally" begin
    V = TamerOp.Visualization
    P = FF.ProductOfChainsPoset((2,2))
    M = MD.PModule{QQ}(P,ones(Int,4),Dict(edge=>reshape(QQ[1],1,1)
        for edge in FF.cover_edges(P));field=CM.QQField())
    grid = EC.GridEncodingMap(P,([0,2],[0,2]))
    enc = RES.EncodingResult(P,M,EC.compile_encoding(P,grid))
    line = (basepoint=(0,0),direction=(1,1))
    s = V.inspection_session(enc;box=([1//2,1//2],[3//2,3//2]),slice=line)
    try
        # The grid certifies an upper tail, but leaves the lower tail unknown.
        # It cannot be promoted to a whole-line module by supplying zeroes.
        before = V.inspection_snapshot(s)
        selection = V.inspection_selection(s)
        summary = V.inspection_summary(s)
        err = try
            V.select_inspection!(s;slice_scope=:global)
            nothing
        catch caught
            caught
        end
        @test err isa ArgumentError
        @test occursin("unrepresented",sprint(showerror,err))
        @test occursin("slice_scope=:window",sprint(showerror,err))
        @test V.inspection_snapshot(s) === before
        @test V.inspection_selection(s) == selection
        @test V.inspection_summary(s).slice_cache_entries == summary.slice_cache_entries
        @test V.inspection_summary(s).slice_cache_misses == summary.slice_cache_misses
        @test only(before.metadata.slice_result.intervals).right_clipped
        for scope in (:unknown,"global",true)
            @test_throws ArgumentError V.select_inspection!(s;slice_scope=scope)
            @test V.inspection_snapshot(s) === before
        end
        @test V.check_inspection_session(s).valid
    finally
        V.close_inspection!(s)
    end
    @test_throws ArgumentError V.inspection_session(enc;box=([0,0],[1,1]),
        slice=line,slice_scope=:global)
    @test_throws ArgumentError V.inspection_session(enc;slice_scope=:unknown)

    # A small viewport needs three local strata; full-line certification needs
    # nine. Reject that request before materializing or changing the session.
    limited = V.inspection_session(_a41_global_square_encoding();slice_limit=3,
        box=([3//2,3//2],[7//4,7//4]),slice=line)
    try
        before = V.inspection_snapshot(limited)
        selection = V.inspection_selection(limited)
        @test_throws ArgumentError V.select_inspection!(limited;slice_scope=:global)
        @test V.inspection_snapshot(limited) === before
        @test V.inspection_selection(limited) == selection
        @test V.check_inspection_session(limited).valid
    finally
        V.close_inspection!(limited)
    end
end
