# Cache lifetime checks use known interval modules and independently recomputed
# answers. Weak references measure reachability, not allocator/RSS behavior.
function _lifecycle_interval(F, n=3)
    K = CM.coeff_type(F)
    P = FF.ProductOfChainsPoset((n,))
    dims = vcat(0, ones(Int,n-1))
    maps = Dict((i,i+1) => ones(K,dims[i+1],dims[i]) for i in 1:n-1)
    M = MD.PModule{K}(P,dims,maps;field=F)
    pi = TO.EncodingCore.GridEncodingMap(P,(collect(0:n-1),))
    return TO.EncodingResult(P,M,pi)
end

function _lifecycle_retained_input(session,n)
    enc = _lifecycle_interval(CM.QQField(),n)
    TO.resolve(enc;opts=OPT.ResolutionOptions(maxlen=1),cache=session)
    return WeakRef(enc.M.dims)
end

function _lifecycle_runtime_input(n)
    enc = _lifecycle_interval(CM.F2(),n)
    TamerOp.SliceInvariants.module_cache(enc.M)
    return WeakRef(enc.M.dims)
end

# Keep the observed object out of the caller's GC root frame.
Base.@noinline _lifecycle_alive(ref::WeakRef) = ref.value !== nothing

@testset "A30 cache ownership and reset" begin
    @testset "Session inspection, exact answers and retained results" begin
        for F in (CM.QQField(),CM.F2(),CM.F3())
            sc = TO.SessionCache(); other = TO.SessionCache()
            initial = TO.cache_stats(sc)
            @test TO.describe(sc) == initial
            @test initial.kind == :session_cache
            @test initial.buckets.module_buckets == 0
            @test initial.encoding.geometry == 0
            @test all(iszero,initial.resolutions)
            enc = _lifecycle_interval(F)
            r = TO.resolve(enc;opts=OPT.ResolutionOptions(maxlen=2),cache=sc)
            r2 = TO.resolve(enc;opts=OPT.ResolutionOptions(maxlen=2),cache=sc)
            @test TO.resolution_object(r) === TO.resolution_object(r2)
            b = TO.betti_table(r)
            @test b[1,:] == [0,1,0] && sum(b)==1
            @test DF.dim(TO.hom(enc,enc;cache=sc)) == 1
            @test DF.dim(TO.ext(enc,enc;maxdeg=1,cache=sc),1) == 0
            @test TO.jordan_multiplicities(TO.jordan_type(enc;slices=([1],[2],[3]),cache=sc)) == [0,1,0]
            ro = TO.resolve(enc;opts=OPT.ResolutionOptions(maxlen=2),cache=other)
            other_counts = TO.cache_stats(other)
            before = TO.cache_stats(sc)
            @test before.buckets.module_buckets > 0
            @test before.module_resolutions.projective > 0
            @test before.retained_module_keys > 0
            @test TO.describe(sc) == before
            @test occursin("SessionCache",sprint(show,sc))
            @test TO.cache_stats(sc) == before
            @test empty!(sc) === sc
            @test TO.cache_stats(sc) == initial
            @test TO.cache_stats(other) == other_counts
            @test TO.betti_table(r) == b == TO.betti_table(ro)
            rebuilt = TO.resolve(enc;opts=OPT.ResolutionOptions(maxlen=2),cache=sc)
            @test TO.resolution_object(rebuilt) !== TO.resolution_object(r)
            @test TO.betti_table(rebuilt) == TO.betti_table(TO.resolve(enc;opts=OPT.ResolutionOptions(maxlen=2),cache=nothing)) == b
            @test empty!(sc) === empty!(sc)
        end
    end

    @testset "Structural raw inputs and immutable mathematical inputs" begin
        sc = TO.SessionCache()
        points = [[0.0],[1.0]]
        data = TO.PointCloud(points)
        spec = TO.FiltrationSpec(kind=:rips,max_dim=1,axes=([0.0,1.0,2.0],))
        enc = TO.encode(data,spec;degree=0,cache=sc)
        dims = TO.restricted_hilbert(enc;cache=sc)
        @test dims == [2,1,1]
        @test TO.encode(data,spec;degree=0,cache=sc).P === enc.P
        data.points[2][1] = 2.0
        changed = TO.encode(data,spec;degree=0,cache=sc)
        @test TO.restricted_hilbert(changed;cache=sc) == [2,2,1]
        @test TO.restricted_hilbert(changed;cache=sc) == TO.restricted_hilbert(TO.encode(data,spec;degree=0,cache=nothing);cache=nothing)
        empty!(sc)
        @test TO.restricted_hilbert(enc;cache=nothing) == dims
        # Editing mathematics uses a new module, not in-place edits to a result.
        F=CM.QQField(); K=CM.coeff_type(F); old=_lifecycle_interval(F)
        maps=Dict(e=>copy(old.M.edge_maps[e...]) for e in FF.cover_edges(old.P))
        maps[2,3].=0
        new=MD.PModule{K}(old.P,copy(old.M.dims),maps;field=F)
        @test TO.multirank(old;sources=[2],targets=[3],cache=sc)==1
        @test TO.multirank(TO.EncodingResult(old.P,new,old.pi);sources=[2],targets=[3],cache=sc)==0
        @test TO.multirank(old;sources=[2],targets=[3],cache=sc)==1
        @test TO.cache_stats(sc).mutation == :read_only_mathematical_objects
    end

    @testset "Concurrent queries and idle reset" begin
        sc=TO.SessionCache(); enc=_lifecycle_interval(CM.F2())
        for P in (enc.P,FF.FinitePoset(Bool[1 1;0 1]))
            jobs=[Threads.@spawn TO.ChangeOfPosets.product_poset(P,P;session_cache=sc) for _ in 1:12]
            products=fetch.(jobs)
            @test all(p->p.P===first(products).P,products)
            @test FF.nvertices(first(products).P)==FF.nvertices(P)^2
            @test TO.ChangeOfPosets.check_monotone_map(first(products).pi1).valid
            @test TO.ChangeOfPosets.check_monotone_map(first(products).pi2).valid
        end
        jobs=[Threads.@spawn TamerOp.Workflow._hom_cache_from_session(nothing,sc,CM.coeff_type(enc.M.field)) for _ in 1:12]
        helpers=fetch.(jobs)
        @test all(h->h===first(helpers),helpers)
        jobs=[Threads.@spawn begin
            r=TO.resolve(enc;opts=OPT.ResolutionOptions(maxlen=1),cache=sc)
            TO.betti_table(r)
        end for _ in 1:8]
        @test all(b->b[1,:]==[0,1,0] && sum(b)==1,fetch.(jobs))
        empty!(sc) # All tasks have joined: this is the supported reset boundary.
        @test TO.cache_stats(sc)==TO.cache_stats(TO.SessionCache())
    end

    @testset "Reachability and distinct ownership scopes" begin
        CM.clear_runtime_caches!()
        @test all(x->x.entries==0,CM.runtime_cache_summary())
        sc=TO.SessionCache()
        refs=[_lifecycle_retained_input(sc,n) for n in (3,5,9)]
        Base.GC.gc(true)
        @test all(_lifecycle_alive,refs)
        @test TO.cache_stats(sc).retained_module_keys==3
        runtime_ref=_lifecycle_runtime_input(7)
        Base.GC.gc(true)
        @test _lifecycle_alive(runtime_ref)
        empty!(sc)
        @test _lifecycle_alive(runtime_ref) # A session reset must not reset global state.
        CM.clear_runtime_caches!()
        for _ in 1:3; Base.GC.gc(true); end
        @test !any(_lifecycle_alive,refs)
        @test !_lifecycle_alive(runtime_ref)
        @test all(x->x.entries==0,CM.runtime_cache_summary())
        # A runtime reset must not erase an explicitly retained session or result.
        enc=_lifecycle_interval(CM.QQField()); result=TO.resolve(enc;cache=sc)
        before=TO.cache_stats(sc); expected=TO.betti_table(result)
        slice_cache=TamerOp.SliceInvariants.module_cache(enc.M)
        CM.clear_runtime_caches!()
        @test TO.cache_stats(sc)==before
        @test TO.betti_table(result)==expected
        @test TamerOp.SliceInvariants.source_module(slice_cache)===enc.M
        @test TamerOp.SliceInvariants.module_cache(enc.M)!==slice_cache
        @test TO.Advanced.runtime_cache_summary()==CM.runtime_cache_summary()
        @test TO.Advanced.clear_runtime_caches!()===nothing
    end
end
