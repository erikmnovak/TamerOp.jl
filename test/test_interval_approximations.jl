# Build independent half-open rectangle sums on explicit singleton/open strata.
function _a103_rectangles(boxes;field=CM.QQField())
    cuts=ntuple(i->sort(unique([QQ(p[i]) for b in boxes for p in b])),2)
    reps=map(c->[c[1]-1;reduce(vcat,([c[i],i==length(c) ? c[i]+1 : (c[i]+c[i+1])/2] for i in eachindex(c)))],cuts)
    P=FF.GridPoset(map(a->collect(1:length(a)),reps))
    active=[[j for (j,b) in enumerate(boxes) if all(b[1] .<= p) && all(p .< b[2])] for p in vec(collect(Iterators.product(reps...)))]
    K=CM.coeff_type(field)
    maps=Dict((u,v)=>K[i==j for i in active[v],j in active[u]] for (u,v) in FF.cover_edges(P))
    TamerOp.RectilinearModule(MD.PModule{K}(P,length.(active),maps;field),cuts)
end

@testset "A103 public approximation surface" begin
    A=TamerOp.IntervalApproximations
    @test TamerOp.Advanced.IntervalApproximations===A
    for name in (:IntervalApproximationBudget,:IntervalApproximation,:ApproximationValidationSummary,:check_interval_approximation)
        @test getproperty(TamerOp.Advanced,name)===getproperty(A,name)
    end
    for name in (:interval_approximation,:approximation_summary,:approximation_at,:approximation_supports,:approximation_slices,:approximation_barcode,:approximation_error,:approximation_module)
        @test getproperty(TamerOp,name)===getproperty(A,name)
    end
end

@testset "A103 static exact support rendering" begin
    A=TamerOp.IntervalApproximations;V=TamerOp.Visualization
    M=_a103_rectangles([((-1,0),(3,2)),((1,1),(2,4))])
    r=A.interval_approximation(M;window=((-2,-1),(5,6)),resolution=1//2)
    spec=V.visual_spec(r;slices=true)
    @test V.check_visual_spec(spec).valid
    @test spec.metadata.provenance.global_error_bound===nothing
    @test spec.metadata.exact_supports==A.approximation_supports(r)
    @test V.available_visuals(r)==(:interval_approximation,)
    @test V.check_visual_spec(V.visual_spec(r;selected=Int[])).valid
    @test_throws ArgumentError V.visual_spec(r;selected=[0])
    @test_throws ArgumentError V.visual_spec(r;slices=1)
    for name in (:TamerOpCairoMakieExt,:TamerOpWGLMakieExt)
        ext=Base.get_extension(TamerOp,name)
        ext===nothing && continue
        backend=name===:TamerOpCairoMakieExt ? :cairomakie : :wglmakie
        fig=V.visualize(r;backend,slices=true)
        @test fig!==nothing
        Makie=name===:TamerOpCairoMakieExt ? ext.CairoMakie.Makie : ext.WGLMakie.Makie
        axis=only(filter(x->x isa Makie.Axis,fig.content))
        @test axis.limits[]==((-2.,5.),(-1.,6.))
        @test !axis.xreversed[] && !axis.yreversed[]
        mktempdir() do dir
            out=V.save_visual(joinpath(dir,"approximation.svg"),r;backend=:cairomakie,slices=true)
            @test isfile(out)
        end
        empty!(fig)
    end
    if haskey(ENV,"TAMEROP_A103_ARTIFACT_DIR")
        dir=ENV["TAMEROP_A103_ARTIFACT_DIR"];mkpath(dir)
        V.save_visual(joinpath(dir,"supports.png"),r;backend=:cairomakie,slices=true)
        V.save_visual(joinpath(dir,"supports.html"),r;backend=:wglmakie,slices=true)
    end
end

@testset "A103 exact source slices and candidate geometry" begin
    A=TamerOp.IntervalApproximations
    boxes=[((0,0),(3,2)),((1,1),(2,4)),((0,0),(3,2))]
    for field in (CM.QQField(),CM.F2(),CM.F3(),CM.Fp(101))
        M=_a103_rectangles(boxes;field)
        r=A.interval_approximation(M;window=((-1,-2),(5,5)),resolution=1//2)
        @test A.check_interval_approximation(r).valid
        @test A.approximation_summary(r).global_error_bound===nothing
        @test !A.approximation_summary(r).source_splitting
        slices=A.approximation_slices(r)
        for (o,bar) in zip(slices.offsets,slices.barcodes)
            expected=sort([(max(b[1][1],b[1][2]-o),min(b[2][1],b[2][2]-o)) for b in boxes if max(b[1][1],b[1][2]-o)<min(b[2][1],b[2][2]-o)])
            @test bar==expected
        end
        # Independently construct the union membership and chain ranks from
        # its rectangle products, not from endpoint reconstruction.
        R=A.approximation_module(r)
        for x in -1:1//4:4,y in -1:1//3:4
            expected=count(bs->any(b->b[1][1]<=x<b[2][1] && b[1][2]<=y<b[2][2],bs),r.rectangles)
            @test A.approximation_at(r,(x,y)).dimension==expected
            @test TamerOp.Advanced.ambient_dimension(R,(x,y))==expected
        end
        @test MD.check_module_functoriality(R.finite).valid
        e=A.approximation_error(r,M;offsets=[-3//4,1//3,2,10])
        @test length(e.distances)==4 && all(>=(0),e.distances)
        @test e.global_bound===nothing
        @test A.approximation_at(r,(5,5)).dimension==0
        @test TamerOp.interval_approximation===A.interval_approximation
    end
end

@testset "A103 multiplicity, tiny coordinates, clipping and failures" begin
    A=TamerOp.IntervalApproximations
    M=_a103_rectangles([((0,0),(2,2)),((0,0),(2,2))])
    r=A.interval_approximation(M;window=((0,0),(1,1)),resolution=1//2)
    @test A.approximation_barcode(r;offset=0)==[(0,1),(0,1)]
    @test A.approximation_at(r,(0,0)).dimension==2
    @test A.approximation_at(r,(1,1)).dimension==0
    @test A.approximation_error(r,M).sampled_max==0
    rs=A.interval_approximation(M;window=((0,0),(1,1)),resolution=1//2,matching=r.matches)
    @test A.approximation_supports(rs)==A.approximation_supports(r)
    @test_throws ArgumentError A.interval_approximation(M;window=((0,0),(1,1)),resolution=0)
    @test_throws ArgumentError A.interval_approximation(M;window=((0,0),(1,1)),resolution=1//100,budget=A.IntervalApproximationBudget(max_lines=2))
    @test_throws ArgumentError A.interval_approximation(M;window=((0,0),(1,1)),resolution=1//2,matching=[[1,1],[1,2]])
    @test_throws ArgumentError A.interval_approximation(M;window=((0,0),(1,1)),resolution=1//2,matching=:compatible)
    @test_throws ArgumentError A.approximation_at(r,(NaN,0))
    @test_throws ArgumentError A.approximation_module(r;budget=A.IntervalApproximationBudget(max_vertices=2))
    @test !A.check_interval_approximation(r;budget=A.IntervalApproximationBudget(max_lines=1)).valid
    push!(r.rectangles[1],((QQ(7),QQ(7)),(QQ(8),QQ(8))))
    @test !A.check_interval_approximation(r).valid
    eps=big(1)//big(10)^30
    small=_a103_rectangles([((1,1),(1+eps,1+eps))])
    tiny=A.interval_approximation(small;window=((1,1),(1+eps,1+eps)),resolution=eps/2)
    @test A.approximation_at(tiny,(1,1)).dimension==1
    @test A.approximation_at(tiny,(1+eps,1)).dimension==0
    @test A.approximation_error(tiny,small).sampled_max==0
    Z=TamerOp.RectilinearModule(MD.zero_pmodule(M.finite.Q;field=M.finite.field),M.cuts)
    zero_result=A.interval_approximation(Z;window=((0,0),(1,1)),resolution=1//2)
    @test isempty(A.approximation_supports(zero_result))
    @test all(iszero,A.approximation_module(zero_result).finite.dims)
    @test A.approximation_error(zero_result,Z).sampled_max==0
    # Deliberately incompatible supplied matching across incomparable supports
    # must split disconnected pieces, not call their union one interval.
    far=_a103_rectangles([((2,0),(5//2,1//2)),((0,2),(1//2,5//2))])
    free=A.interval_approximation(far;window=((0,0),(3,3)),resolution=2)
    @test length(free.rectangles)==2
    @test A.check_interval_approximation(free).valid
    qboxes=[((QQ(2),QQ(0)),(QQ(5//2),QQ(1//2))),((QQ(0),QQ(2)),(QQ(1//2),QQ(5//2)))]
    @test length(A._components(qboxes))==2
    copied=A.approximation_supports(free);empty!(copied)
    @test length(free.rectangles)==2
end

@testset "A103 independent order-connectivity oracle" begin
    A=TamerOp.IntervalApproximations;rng=MersenneTwister(103)
    for _ in 1:50
        births=unique([(QQ(rand(rng,-3:2)),QQ(rand(rng,-3:2))) for _ in 1:5])
        deaths=unique([(QQ(rand(rng,-2:3)),QQ(rand(rng,-2:3))) for _ in 1:5])
        boxes=[(a,b) for a in births for b in deaths if all(a .< b)]
        # Transitive closure of comparability between rectangle interiors.
        adjacency=[i==j || all(boxes[i][1].<boxes[j][2]) || all(boxes[j][1].<boxes[i][2]) for i in eachindex(boxes),j in eachindex(boxes)]
        for k in eachindex(boxes),i in eachindex(boxes),j in eachindex(boxes)
            adjacency[i,j] |= adjacency[i,k] && adjacency[k,j]
        end
        components=A._components(boxes)
        for i in eachindex(boxes),j in eachindex(boxes)
            @test any(c->i in c && j in c,components)==adjacency[i,j]
        end
    end
end
