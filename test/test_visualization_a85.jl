# A85 independent interval-cost and rectangle oracles, state/lifecycle contracts,
# and optional native rendering. Browser events live in matching.spec.mjs.
function _a85_rectangles(;field=FIELD_QQ)
    xs=[0//1,1//1,3//1,4//1]
    P=FF.ProductOfChainsPoset((4,4))
    pi=EC.GridEncodingMap(P,(xs,xs))
    rectangles=[((0,1),(3,4)),((1,0),(4,4)),((1,1),(3,3))]
    K=CM.coeff_type(field)
    modules=map(rectangles) do (lo,hi)
        active=[lo[1]<=x<hi[1] && lo[2]<=y<hi[2] for y in xs for x in xs]
        dims=Int.(active)
        MD.PModule{K}(P,dims,Dict((a,b)=>fill(one(K),dims[b],dims[a]) for (a,b) in FF.cover_edges(P));field)
    end
    M,N=MD.direct_sum(modules[1],modules[2]),modules[3]
    return RES.EncodingResult(P,M,pi),RES.EncodingResult(P,N,pi)
end

@testset "A85 actual bottleneck assignments and display contracts" begin
    V=TamerOp.Visualization;SI=TamerOp.SliceInvariants
    cases=[([(0.,2.),(0.,2.),(8.,10.)],[(0.,3.),(0.,3.)],1.),
        ([(0.,Inf)],[(2.,Inf)],2.), ([(0.,Inf)],Tuple{Float64,Float64}[],Inf),
        ([(-Inf,2.)],[(-Inf,3.)],1.),([(-Inf,Inf)],[(-Inf,Inf)],0.),
        (Tuple{Float64,Float64}[],Tuple{Float64,Float64}[],0.)]
    for (A,B,expected) in cases
        w=SI.bottleneck_matching(A,B)
        @test w.distance==expected
        spec=V.visual_spec(w)
        @test V.check_visual_spec(spec).valid
        records=spec.metadata.matching_records
        @test sort([r.a for r in records if r.a!=0])==collect(eachindex(A))
        @test sort([r.b for r in records if r.b!=0])==collect(eachindex(B))
        @test maximum((r.cost for r in records);init=0)==expected
        @test all(r->r.a==0 || w.a_to_b[r.a]==r.b,records)
        @test spec.metadata.uniqueness==:not_asserted
        session=V.inspection_session(w;max_pairs=1)
        for id in eachindex(records)
            V.select_inspection!(session;pair=id)
            shot=V.inspection_snapshot(session)
            @test shot.metadata.selected_pair==id
            @test id in shot.metadata.displayed_pair_ids
            @test V.inspection_summary(session).matching_queries==0
        end
        before=V.inspection_selection(session)
        @test_throws ArgumentError V.select_inspection!(session;pair=length(records)+1)
        @test V.inspection_selection(session)==before
        @test_throws ArgumentError V.select_inspection!(session;optimum=true)
        V.reset_inspection!(session)
        @test V.inspection_selection(session).pair===nothing
        @test V.check_inspection_session(session).valid
        snapshot=V.inspection_snapshot(session)
        V.close_inspection!(session)
        @test V.inspection_snapshot(session)===snapshot
        @test V.check_inspection_session(session).valid
        @test !V.check_visual_request(session).valid
    end
    w=SI.bottleneck_matching(Dict((0//1,2//1)=>2,(10//1,12//1)=>1),Dict((0//1,3//1)=>2))
    s=V.visual_spec(w)
    @test count(r->r.diagonal,s.metadata.matching_records)==1
    @test s.metadata.matching_records[1].multiplicity_a==2
    @test length(s.metadata.bottleneck_pair_ids)==3
    bad=merge(w,(;a_to_b=fill(99,length(w.a_to_b))))
    @test !V.check_visual_request(bad).valid
    @test_throws ArgumentError V.visual_spec(bad)
    for kwargs in ((;pair=true),(;pair=-1),(;pair=99),(;max_pairs=0),(;unknown=true))
        @test !V.check_visual_request(w;kwargs...).valid
    end
    @test TamerOp.Advanced.MatchingInspectionSession===V.MatchingInspectionSession
end

@testset "A85 ordinary degree and coordinate policies" begin
    V=TamerOp.Visualization;O=TamerOp.OrdinaryPersistence
    a=O.PersistenceDiagram([[(3.,1.)]],[[4.]];order=:superlevel)
    b=O.PersistenceDiagram([[(3.,0.)]],[[5.]];order=:superlevel)
    session=V.inspection_session(a,b;dim=0)
    @test V.inspection_summary(session).distance==1
    @test V.inspection_snapshot(session).metadata.context.ordinary.order==:superlevel
    @test session.payload.matching.points_a==[(-3//1,-1//1),(-4//1,Inf)]
    dropped=V.inspection_session(a,b;dim=0,essential=:drop,scale=2)
    @test V.inspection_summary(dropped).distance==0.5
    @test V.inspection_snapshot(dropped).metadata.context.ordinary.essential==:drop
    @test_throws ArgumentError V.inspection_session(a,b;dim=0,essential=:cap)
    @test_throws ArgumentError V.inspection_session(a,b;dim=true)
    V.close_inspection!(session);V.close_inspection!(dropped)
end

@testset "A85 exact optimizing slice retains the interior switch" begin
    V=TamerOp.Visualization;F=TamerOp.Fibered2D
    # On y=x+h, 0<=h<=1, the two possible matchings have costs
    # 1+h/2 and 3/2-h/2. Their minimum peaks at h=1/2 with value 5/4.
    # For every positive slope, the short bar is within endpoint cost 1 of
    # either large bar; the two large weighted lengths sum to at most 5.
    # Match the longer, delete the shorter: global cost <=5/4.
    for field in FIELDS_FULL
        a,b=_a85_rectangles(;field)
        opts=TamerOp.Options.InvariantOptions(box=([0,0],[4,4]),threads=false)
        for (normalization,weight) in ((:L1,:lesnick_l1),(:Linf,:lesnick_linf))
            result=TamerOp.matching_distance_exact_2d(a,b;opts,normalize_dirs=normalization,weight,witness=true)
            @test result isa F.MatchingDistanceResult2D
            @test TamerOp.describe(result).exact_distance==5//4
            @test result.status==:attained
            @test F.check_matching_result_2d(result).valid
            q=F.slice_query(result);w=F.matching_witness(result)
            @test q.weight*w.distance==5//4
            @test q.direction[1]==q.direction[2]
            @test q.basepoint[2]-q.basepoint[1]==1//2
            @test V.check_visual_spec(V.visual_spec(result)).valid
            @test F.working_box(result)==((0,0),(4,4))
        end
    end
    a,b=_a85_rectangles()
    opts=TamerOp.Options.InvariantOptions(box=([0,0],[4,4]),threads=false)
    @test TamerOp.matching_distance_exact_2d(a,b;opts)==1.25
    r=TamerOp.matching_distance_exact_2d(a,a;opts,witness=true)
    @test r.exact_distance==0 && r.status==:attained
    @test F.matching_witness(r).distance==0
    degenerate=TamerOp.matching_distance_exact_2d(a,b;
        opts=TamerOp.Options.InvariantOptions(box=([0,0],[0,4])),witness=true)
    @test degenerate.status==:degenerate_window
    @test F.slice_query(degenerate)===nothing
    @test V.check_visual_spec(V.visual_spec(degenerate)).valid
    @test_throws ArgumentError TamerOp.matching_distance_exact_2d(a,b;opts,witness=true,max_candidates=1)
    if Threads.nthreads()>1
        threaded=TamerOp.matching_distance_exact_2d(a,b;opts=TamerOp.Options.InvariantOptions(box=([0,0],[4,4]),threads=true),witness=true)
        @test threaded.exact_distance==5//4
        @test F.slice_query(threaded)==F.slice_query(TamerOp.matching_distance_exact_2d(a,b;opts,witness=true))
    end
end

@testset "A85 selected sampled and exact matching sessions" begin
    V=TamerOp.Visualization
    a,b=_a85_rectangles()
    opts=TamerOp.Options.InvariantOptions(box=([0,0],[4,4]),threads=false)
    lines=[(;basepoint=(0,h),direction=(1,1)) for h in (0,1)]
    session=V.inspection_session(a,b;opts,samples=lines)
    @test V.inspection_summary(session).matching_queries==3
    @test V.inspection_snapshot(session).metadata.context.scope==:selected_slice
    @test V.inspection_summary(session).weighted_distance==1
    V.select_inspection!(session;sample=2)
    @test V.inspection_summary(session).scope==:sampled_slice
    @test V.inspection_summary(session).weighted_distance==1
    V.select_inspection!(session;optimum=true)
    @test V.inspection_summary(session).scope==:certified_finite_window
    @test V.inspection_summary(session).weighted_distance==5//4
    count=V.inspection_summary(session).matching_queries
    V.select_inspection!(session;pair=1)
    V.select_inspection!(session;optimum=true)
    @test V.inspection_summary(session).matching_queries==count
    @test V.check_visual_spec(V.visual_spec(session)).valid
    V.select_inspection!(session;slice=(;basepoint=(0,1//2),direction=(1,1)))
    @test V.inspection_summary(session).scope==:selected_slice
    @test V.inspection_summary(session).weighted_distance==1.25
    for kwargs in ((;sample=0),(;sample=999),(;pair=999),(;sample=1,optimum=true),
        (;slice=(;basepoint=(0,0),direction=(0,1))),(;optimum=:yes))
        before=V.inspection_snapshot(session)
        @test !V.check_inspection_selection(session;kwargs...).valid
        @test_throws ArgumentError V.select_inspection!(session;kwargs...)
        @test before===V.inspection_snapshot(session)
    end
    token=V._on_inspection(session,_ -> error("controlled listener failure"))
    V.reset_inspection!(session)
    @test length(V.inspection_summary(session).listener_errors)==1
    V._off_inspection!(session,token)
    @test V.inspection_summary(session).scope==:selected_slice
    V.close_inspection!(session)
    @test V.check_inspection_session(session).valid
    @test session.inputs===nothing && isempty(session.samples)
end

@testset "A85 native matching renderers" begin
    V=TamerOp.Visualization
    if Base.find_package("CairoMakie")===nothing || Base.find_package("WGLMakie")===nothing
        @test_skip false
    else
        @eval import CairoMakie, WGLMakie
        w=TamerOp.SliceInvariants.bottleneck_matching([(0.,2.),(0.,2.),(4.,Inf)],[(0.,3.),(5.,Inf)])
        specs=[V.visual_spec(w),V.visual_spec(w;pair=2,max_pairs=2)]
        for backend in (:cairomakie,:wglmakie), spec in specs, palette in (:accessible,:grayscale)
            fig=V.render(spec;backend,style=V.VisualStyle(;palette))
            WGLMakie.Makie.update_state_before_display!(fig)
            @test !isempty(fig.content)
        end
        mktempdir() do dir
            for suffix in ("png","svg","pdf")
                path=joinpath(dir,"matching.$suffix")
                V.save_visual(path,first(specs);backend=:cairomakie)
                @test filesize(path)>1000
            end
        end
        session=V.inspection_session(w)
        app=V.visualize(session;backend=:wglmakie)
        ext=Base.get_extension(TamerOp,:TamerOpWGLMakieExt)
        ui=ext._inspection_ui(app)
        n=length(ui.observers)
        initial_figure=ui.figure[]
        V.select_inspection!(session;pair=2)
        @test isempty(V.inspection_summary(session).listener_errors)
        @test length(ui.observers)==n
        @test ui.rebuild_count[]==2
        @test ui.figure[] !== initial_figure && isempty(initial_figure.content)
        @test length(ui.pointer_observers)==2
        V.close_inspection!(session)
        @test ui.closed[] && isempty(ui.observers) && isempty(ui.pointer_observers)
    end
end
