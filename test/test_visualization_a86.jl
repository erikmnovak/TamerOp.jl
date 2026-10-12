# Exact signed sums are checked against independent module maps and cumulative
# functions. Plot budgets and clipping never alter the mathematical answer.
function _a86_cross(;field=FIELD_QQ)
    xs=[0,2,5];P=FF.ProductOfChainsPoset((3,3));pi=EC.GridEncodingMap(P,(xs,xs))
    K=CM.coeff_type(field)
    dims=[x<=2 || y<=2 ? 1 : 0 for y in xs for x in xs]
    M=MD.PModule{K}(P,dims,Dict((a,b)=>fill(one(K),dims[b],dims[a]) for (a,b) in FF.cover_edges(P));field)
    return M,pi,xs
end

@testset "A86 exact rank cancellation and smoothing" begin
    V=TamerOp.Visualization;S=TamerOp.SignedMeasures
    for field in (FIELD_QQ,CM.F2(),CM.F3())
        M,pi,xs=_a86_cross(;field)
        sb=S.rectangle_signed_barcode((p,q)->TamerOp.InvariantCore.rank_map(M,p[1]+3*(p[2]-1),q[1]+3*(q[2]-1)),([0,2,5],[0,2,5]);threads=false)
        @test sort(S.weights(sb))==[-1,1,1]
        direct=(p,q)->TamerOp.InvariantCore.rank_map(M,EC.locate(pi,collect(p)),EC.locate(pi,collect(q)))
        # Every comparable pair, not only the overlapping stalk.
        for px in xs,py in xs,qx in xs,qy in xs
            px<=qx && py<=qy || continue
            expected=(qx<=2 || qy<=2) ? 1 : 0
            sp=V.visual_spec(sb;kind=:signed_cancellation,query=(p=(px,py),q=(qx,qy)),direct,max_terms=1)
            @test sp.metadata.reconstructed==expected
            @test sp.metadata.direct_value==expected
            @test sp.metadata.agreement===:equal
            @test length(sp.metadata.displayed_term_ids)==1
            @test V.check_visual_spec(sp).valid
        end
        sp=V.visual_spec(sb;kind=:signed_cancellation,query=(p=(0,0),q=(2,2)),direct)
        @test (sp.metadata.positive,sp.metadata.negative,sp.metadata.reconstructed)==(2,-1,1)
        eroded=V.visual_spec(sb;kind=:signed_diagonals,smoothing=(2,0),query=(p=(0,0),q=(0,2)),direct,term=3,max_terms=1)
        @test eroded.metadata.reconstructed==1
        @test eroded.metadata.agreement===:equal
        @test eroded.metadata.displayed_term_ids==[3]
        @test eroded.metadata.records[3].upper!=eroded.metadata.records[3].smoothed_upper
        @test V.check_visual_spec(eroded).valid
    end
    M,pi,xs=_a86_cross();sb=S.rectangle_signed_barcode((p,q)->TamerOp.InvariantCore.rank_map(M,p[1]+3*(p[2]-1),q[1]+3*(q[2]-1)),([0,2,5],[0,2,5]);threads=false)
    bad=V.visual_spec(sb;kind=:signed_cancellation,query=(p=(0,0),q=(2,2)),direct=(p,q)->99)
    @test bad.metadata.agreement===:mismatch
    @test bad.metadata.reconstructed==1
    for kw in ((;query=(p=(3,0),q=(5,2))), (;query=(p=(5,0),q=(2,2))), (;query=(p=(0,0),q=(2,2)),smoothing=(1,0)),
        (;smoothing=(-1,0)),(;term=true),(;term=0.0),(;term=20),(;max_terms=0),(;boundary=:halfopen),(;observable=:hilbert),
        (;query=(0,0)),(;units=("x",)),(;window=((0,0),(0,1))), (;direct=(p,q)->1))
        @test !V.check_visual_request(sb;kind=:signed_cancellation,kw...).valid
    end
    @test_throws ArgumentError V.visual_spec(sb;kind=:signed_cancellation,query=(p=(0,0),q=(2,2)),direct=(p,q)->1.0)
    @test_throws ArgumentError V.visual_spec(sb;kind=:signed_cancellation,query=(p=(0,0),q=(2,2)),direct=(p,q)->nothing)
    @test !V.check_visual_request(sb;kind=:rectangles,smoothing=(1,1)).valid
end

@testset "A86 endpoint pairs and half-open boundaries" begin
    V=TamerOp.Visualization;S=TamerOp.SignedMeasures
    xs=[0//1,2//3,7//2];ds=Real[2//3,7//2,Inf]
    pm=S.PointSignedMeasure((xs,xs,ds,ds),[(1,1,3,1),(1,1,1,3),(1,1,1,1)],[1,1,-1])
    direct=(p,q)->q[1]<2//3 || q[2]<2//3 ? 1 : 0
    for x in (0,1//3,2//3,2,10),y in (0,1//3,2//3,2,10)
        sp=V.visual_spec(pm;query=(p=(0,0),q=(x,y)),direct,window=((-1,-1),(4,4)))
        @test sp.metadata.agreement===:equal
        @test sp.metadata.boundary===:halfopen
        @test V.check_visual_spec(sp).valid
    end
    sp=V.visual_spec(pm;kind=:signed_prominence,term=3,smoothing=(2//3,0),query=(p=(0,0),q=(0,0)),direct)
    @test sp.metadata.reconstructed==1
    @test count(r->!r.survives,sp.metadata.records)==2
    @test sp.metadata.records[1].upper==(Inf,2//3)
    @test only(sp.panels[1].metadata.vectors).vector==(Inf,2//3)
    # Inclusive zero-width rectangles survive at equality; half-open ones vanish.
    closed=S.RectSignedBarcode(([0,2],[0,2]),[S.Rect{2}((0,0),(2,2))],[1])
    z=V.visual_spec(closed;kind=:signed_cancellation,smoothing=(2,2),query=(p=(0,0),q=(0,0)))
    @test z.metadata.records[1].survives
    @test z.metadata.reconstructed==1
    @test_throws ArgumentError V.visual_spec(S.PointSignedMeasure((xs,xs,ds,ds),[(2,1,1,1)],[1]))
    @test_throws ArgumentError V.visual_spec(pm;boundary=:mystery)
    # The ordinary 2D cumulative interpretation is never applied to 4D ranks.
    @test !(:signed_atoms in V.available_visuals(pm))
    # The real workflow's endpoint adapter agrees with fresh module ranks.
    M,pi,_=_a86_cross();enc=RES.EncodingResult(M.Q,M,pi)
    rpm=TamerOp.rank_signed_measure(enc;threads=false)
    directM=(p,q)->TamerOp.InvariantCore.rank_map(M,EC.locate(pi,collect(p)),EC.locate(pi,collect(q)))
    for q in ((0,0),(2,2),(5,2),(5,5),(100,1))
        spec=V.visual_spec(rpm;query=(p=(0,0),q=q),direct=directM)
        @test spec.metadata.agreement===:equal
    end
end

@testset "A86 cumulative atoms and semantic backgrounds" begin
    V=TamerOp.Visualization;S=TamerOp.SignedMeasures
    xs=[0,2,7];ys=[-3,1,9]
    surface=[x<=2 || y<=1 ? 1 : 0 for x in xs,y in ys]
    pm=S.point_signed_measure(surface,(xs,ys))
    background=(;axes=(xs,ys),values=permutedims(surface),observable=:hilbert,source="cross module",units=("x units","y units"))
    for (i,x) in enumerate(xs),(j,y) in enumerate(ys)
        sp=V.visual_spec(pm;kind=:signed_cancellation,observable=:hilbert,query=(x,y),direct=p->(p[1]<=2 || p[2]<=1 ? 1 : 0),
            background,units=background.units)
        @test sp.metadata.reconstructed==surface[i,j]
        @test sp.metadata.agreement===:equal
        @test V.check_visual_spec(sp).valid
        @test sp.panels[1].metadata.background.source=="cross module"
    end
    endsp=V.visual_spec(pm;kind=:signed_cancellation,query=(7,9),observable=:euler)
    @test (endsp.metadata.positive,endsp.metadata.negative)==(1,-1)
    @test endsp.metadata.agreement===:not_checked
    @test_throws ArgumentError V.visual_spec(pm;kind=:signed_cancellation,query=(1,1))
    @test_throws ArgumentError V.visual_spec(pm;kind=:signed_cancellation,smoothing=(1,1))
    @test_throws ArgumentError V.visual_spec(pm;kind=:signed_cancellation,background)
    @test_throws ArgumentError V.visual_spec(pm;kind=:signed_cancellation,background=merge(background,(;axes=([0,1,7],ys))),units=background.units)
    @test_throws ArgumentError V.visual_spec(pm;kind=:signed_cancellation,background=merge(background,(;values=zeros(1,1))),units=background.units)
    huge=big(2)^70
    duplicates=S.PointSignedMeasure(([0],[0]),[(1,1),(1,1),(1,1)],[huge,-huge,big(1)])
    sp=V.visual_spec(duplicates;kind=:signed_cancellation,query=(0,0),max_terms=1,term=3)
    @test sp.metadata.reconstructed==1
    @test length(sp.metadata.records)==3
    @test sp.metadata.displayed_term_ids==[3]
    empty=S.PointSignedMeasure((Int[],Int[]),Tuple{Int,Int}[],Int[])
    @test V.check_visual_spec(V.visual_spec(empty;kind=:signed_cancellation)).valid
    malformed=S.PointSignedMeasure(([0],[0]),[(2,1)],[1])
    @test !V.check_visual_request(malformed;kind=:signed_cancellation).valid
    close_axes=[big(2)^60,big(2)^60+1]
    close_points=S.PointSignedMeasure((close_axes,[0]),[(1,1),(2,1)],[1,-1])
    precise=V.visual_spec(close_points;kind=:signed_cancellation,query=(last(close_axes),0),
        window=((big(2)^60-1000,-1),(big(2)^60+1000,1)))
    @test precise.metadata.reconstructed==0
    @test precise.metadata.display_collisions==1
    @test precise.metadata.records[2].point[1]-precise.metadata.records[1].point[1]==1
    @test_throws ArgumentError V.visual_spec(S.PointSignedMeasure(([0],[0]),[(1,1)],[1.0]);kind=:signed_cancellation)
end

@testset "A86 declared interval-family reconstruction" begin
    V=TamerOp.Visualization;I=TamerOp.Invariants
    P=FF.FinitePoset(Bool[1 0 0 1;0 1 0 1;0 0 1 1;0 0 0 1])
    K=CM.coeff_type(FIELD_QQ)
    M=MD.PModule{K}(P,[1,1,1,2],Dict((1,4)=>reshape(K[1,0],2,1),(2,4)=>reshape(K[0,1],2,1),(3,4)=>reshape(K[1,1],2,1));field=FIELD_QQ)
    family=[[4],[1,4],[2,4],[3,4]]
    summary=I.interval_rank_summary(M;family)
    for vertices in family
        sp=V.visual_spec(summary;query=(;vertices),direct=vs->I.generalized_rank(M;vertices=vs))
        @test sp.metadata.agreement===:equal
        @test sp.metadata.domain===:declared_family
        @test V.check_visual_spec(sp).valid
    end
    sp=V.visual_spec(summary;query=(vertices=[4],))
    @test (sp.metadata.positive,sp.metadata.negative,sp.metadata.reconstructed)==(3,-1,2)
    @test !V.check_visual_request(summary;query=(vertices=[1],)).valid
    @test !V.check_visual_request(summary;smoothing=(1,0)).valid
end

@testset "A86 linked state and direct-query reuse" begin
    V=TamerOp.Visualization;S=TamerOp.SignedMeasures
    M,pi,_=_a86_cross();sb=S.rectangle_signed_barcode((p,q)->TamerOp.InvariantCore.rank_map(M,p[1]+3*(p[2]-1),q[1]+3*(q[2]-1)),([0,2,5],[0,2,5]);threads=false)
    calls=Ref(0)
    direct=(p,q)->begin calls[]+=1; q[1]<=2 || q[2]<=2 ? 1 : 0;end
    @test V.check_visual_request(sb;kind=:signed_cancellation,query=(p=(0,0),q=(2,2)),direct).valid
    @test calls[]==0
    s=V.inspection_session(sb;query=(p=(0,0),q=(2,2)),direct,max_terms=1)
    @test calls[]==1
    @test TamerOp.Advanced.SignedInspectionSession===V.SignedInspectionSession
    token=V._on_inspection(s,_->error("expected listener failure"))
    for i in 1:3
        V.select_inspection!(s;term=i)
        @test V.inspection_snapshot(s).metadata.displayed_term_ids==[i]
        @test V.check_inspection_session(s).valid
    end
    V.select_inspection!(s;view=:signed_diagonals)
    @test calls[]==1
    @test length(V.inspection_summary(s).listener_errors)==4
    V._off_inspection!(s,token)
    @test V.check_visual_spec(V.visual_spec(s)).valid
    before=V.inspection_snapshot(s);state=V.inspection_selection(s)
    @test_throws ArgumentError V.select_inspection!(s;term=99)
    @test_throws ArgumentError V.select_inspection!(s;query=(p=(0,0),q=(-1,0)))
    @test V.inspection_snapshot(s)===before
    @test V.inspection_selection(s)==state
    @test calls[]==1
    V.select_inspection!(s;query=(p=(0,0),q=(5,5)))
    @test calls[]==2
    @test V.inspection_snapshot(s).metadata.reconstructed==0
    V.reset_inspection!(s)
    @test calls[]==3
    @test V.inspection_snapshot(s).metadata.reconstructed==1
    shot=V.inspection_snapshot(s)
    V.close_inspection!(s);V.close_inspection!(s)
    @test V.inspection_snapshot(s)===shot
    @test V.check_inspection_session(s).valid
    @test !V.check_visual_request(s).valid
    @test_throws ArgumentError V.select_inspection!(s;term=1)
end

@testset "A86 rendering, exports and live native controls" begin
    V=TamerOp.Visualization;S=TamerOp.SignedMeasures
    xs=[0,2,5];ds=[2.,5.,Inf]
    pm=S.PointSignedMeasure((xs,xs,ds,ds),[(1,1,3,2),(1,1,2,3),(1,1,2,2)],[1,1,-1])
    spec=V.visual_spec(pm;query=(p=(0,0),q=(2,2)),direct=(p,q)->(q[1]<5 || q[2]<5 ? 1 : 0),term=3)
    for backend in (:cairomakie,:wglmakie)
        name=backend===:cairomakie ? "CairoMakie" : "WGLMakie"
        if Base.find_package(name)===nothing;@test_skip false;continue;end
        if backend===:cairomakie;@eval import CairoMakie;mk=CairoMakie.Makie
        else;@eval import WGLMakie;mk=WGLMakie.Makie;end
        for style in (V.VisualStyle(),V.VisualStyle(palette=:grayscale))
            fig=V.render(spec;backend,style)
            mk.update_state_before_display!(fig)
            @test !isempty(fig.content)
            @test all(a->all(>(0),mk.widths(mk.viewport(a.scene)[])),filter(x->x isa mk.Axis,fig.content))
        end
        if backend===:cairomakie
            mktempdir() do dir
                for format in ("png","svg","pdf")
                    path=joinpath(dir,"signed.$format")
                    V.save_visual(path,spec;backend)
                    @test filesize(path)>1000
                end
            end
        else
            session=V.inspection_session(pm;query=(p=(0,0),q=(2,2)))
            app=V.visualize(session;backend)
            ext=Base.get_extension(TamerOp,:TamerOpWGLMakieExt);ui=ext._inspection_ui(app)
            ui.controls.term.value[]="3";ui.controls.apply_term.value[]+=1
            @test V.inspection_selection(session).term==3
            @test ui.error_text[]==""
            mk.update_state_before_display!(ui.figure[])
            ui.callbacks.hover(:signed_arithmetic,first(ui.targets[:signed_arithmetic]).point)
            @test !isempty(ui.hover_text[])
            ui.controls.query.value[]="0,0,5,5";ui.controls.apply_query.value[]+=1
            @test V.inspection_snapshot(session).metadata.reconstructed==0
            @test isempty(ui.hover_text[])
            ui.controls.smoothing.value[]="-1,0";ui.controls.apply_smoothing.value[]+=1
            @test occursin("nonnegative",ui.error_text[])
            @test V.inspection_selection(session).smoothing==(0,0)
            V.close_inspection!(session)
            @test ui.closed[]
            @test isempty(ui.observers) && isempty(ui.pointer_observers)
            @test V.check_inspection_session(session).valid
        end
    end
end
