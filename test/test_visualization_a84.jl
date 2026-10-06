# A84 ordinary pair-rank sections. Oracles use explicit maps and interval
# membership, independent of the section builder or sampled heatmaps.

function _a84_chain(field; composite_zero=true)
    K = CM.coeff_type(field)
    return MD.PModule{K}(chain_poset(3), [1,2,1],
        Dict((1,2)=>reshape(K[1,0],2,1), (2,3)=>(composite_zero ? K[0 1] : K[1 0])); field)
end

function _a84_grid(; orientation=(1,1), coords=([0,2],[0,2]))
    P = FF.ProductOfChainsPoset((length(coords[1]),length(coords[2])))
    M = MD.PModule{QQ}(P, ones(Int, FF.nvertices(P)),
        Dict(edge=>ones(QQ,1,1) for edge in FF.cover_edges(P)); field=FIELD_QQ)
    pi = EC.GridEncodingMap(P, coords; orientation)
    return RES.EncodingResult(P, M, EC.compile_encoding(P,pi))
end

@testset "A84 ranks distinguish maps from stalk dimensions" begin
    V = TamerOp.Visualization
    for field in FIELDS_FULL
        M, N = _a84_chain(field), _a84_chain(field; composite_zero=false)
        from = V.visual_spec(M; kind=:rank_section, source=1, vertex=3)
        to = V.visual_spec(M; kind=:rank_section, target=3, vertex=1)
        @test only(from.metadata.rank_sections).ranks == [1,1,0]
        @test only(to.metadata.rank_sections).ranks == [0,1,1]
        @test only(V.visual_spec(N; kind=:rank_section, source=1).metadata.rank_sections).ranks == [1,1,1]
        @test TamerOp.dimensions(M) == TamerOp.dimensions(N)
        for s in (from,to)
            @test V.check_visual_spec(s).valid
            @test s.metadata.inspection.defined
            @test s.metadata.inspection.rank == 0
            @test s.metadata.inspection.matrix == zeros(CM.coeff_type(field),1,1)
            @test s.metadata.inspection.source_dimension == s.metadata.inspection.target_dimension == 1
            @test s.metadata.rank_queries == 2
            @test !s.metadata.all_pairs_table
            @test only(s.metadata.rank_sections).retained_matrices == 0
        end
        multi = V.visual_spec(M; kind=:rank_section, source=[1,2,1], vertex=3)
        @test multi.metadata.rank_sections[1] === multi.metadata.rank_sections[3]
        @test multi.metadata.rank_sections[2].ranks == [nothing,2,1]
        @test multi.metadata.rank_queries == 3
        @test multi.metadata.rank_range == (0,2)
        @test V.check_visual_spec(multi).valid
        backward = V.visual_spec(M; kind=:rank_section, source=3, vertex=1)
        @test !backward.metadata.inspection.defined
        @test backward.metadata.inspection.matrix === nothing
        @test backward.metadata.inspection.rank === nothing
        @test only(backward.metadata.rank_sections).ranks == [nothing,nothing,1]
        K=CM.coeff_type(field)
        det2=MD.PModule{K}(chain_poset(2),[2,2],Dict((1,2)=>K[1 1; 1 -1]);field)
        r=V.visual_spec(det2;kind=:rank_section,source=1,vertex=2)
        @test only(r.metadata.rank_sections).ranks == [2, field isa CM.PrimeField && field.p == 2 ? 1 : 2]
        @test r.metadata.inspection.exact == !(field isa CM.RealField)
    end
end

@testset "A84 exact parameter masks and reversed axes" begin
    V = TamerOp.Visualization
    enc = _a84_grid()
    pi = V._inspection_classifier(RES.encoding_map(enc))
    box=((-1,-1),(4,4))
    for from in (true,false)
        options=from ? (;source=(1,1)) : (;target=(1,1))
        s=V.visual_spec(enc;kind=:rank_section,options...,point=(3//2,1//2),box)
        @test V.check_visual_spec(s).valid
        @test !s.metadata.inspection.defined
        @test s.metadata.inspection.source == s.metadata.inspection.target
        panel=s.panels[1]
        @test panel.metadata.fixed_coordinates == (from ? (:p1,:p2) : (:q1,:q2))
        @test panel.metadata.varying_coordinates == (from ? (:q1,:q2) : (:p1,:p2))
        for c in panel.metadata.geometry.components
            center=ntuple(i->sum(p[i] for p in c.vertices)/length(c.vertices),2)
            ordered=from ? all(center[i]>=1 for i in 1:2) : all(center[i]<=1 for i in 1:2)
            @test (c.region_id == -1) == !ordered
            for (point,included) in zip(c.vertices,c.vertex_included)
                ordered=from ? all(point[i]>=1 for i in 1:2) : all(point[i]<=1 for i in 1:2)
                expected=ordered ? V._visual_locate(pi,point) : -1
                @test included == (expected == c.region_id)
            end
        end
        @test any(c->c.region_id==-1,panel.metadata.geometry.components)
    end
    epsilon=1//big(2)^60
    anchor=(1+epsilon,1)
    s=V.visual_spec(enc;kind=:rank_section,source=[(1,1),anchor],point=(1,1),box)
    @test s.metadata.rank_sections[1] === s.metadata.rank_sections[2]
    @test s.metadata.inspections[1].defined && !s.metadata.inspections[2].defined
    @test s.metadata.anchors[2] == anchor
    @test !isempty(s.panels[3].metadata.query_collisions)
    @test !isempty(s.panels[3].metadata.warnings)
    @test any(c->any(p->p[1]==1+epsilon,c.vertices),s.panels[3].metadata.geometry.components)
    @test s.metadata.rank_queries == 3
    missing=V.visual_spec(enc;kind=:rank_section,source=(-1,-1),point=(1,1),box)
    @test all(isnothing,only(missing.metadata.rank_sections).ranks)
    @test !missing.metadata.inspection.defined
    @test missing.metadata.inspection.source_dimension === nothing
    reversed=_a84_grid(;orientation=(-1,1))
    s=V.visual_spec(reversed;kind=:rank_section,source=(-1,1),point=(-3,3),box=((-4,-1),(1,4)))
    @test s.metadata.inspection.defined && s.metadata.inspection.rank == 1
    @test s.panels[1].metadata.orientation == (-1,1)
    @test occursin("decreasing",s.panels[1].axes.xlabel)
    reverse_bad=V.visual_spec(reversed;kind=:rank_section,source=(-1,1),point=(0,2))
    @test !reverse_bad.metadata.inspection.defined
end

@testset "A84 request validation and bounded live reuse" begin
    V=TamerOp.Visualization
    M=_a84_chain(FIELD_QQ)
    enc=_a84_grid()
    for obj in (M,enc), options in ((;), (;source=1,target=2), (;source=true),
        (;source=0), (;source=999), (;source=[]), (;source=1,matrix_limit=(0,1)),
        (;source=1,point=(0,0)), (;source=1,box=((0,0),(2,2))))
        @test !V.check_visual_request(obj;kind=:rank_section,options...).valid
        @test_throws ArgumentError V.visual_spec(obj;kind=:rank_section,options...)
    end
    for options in ((;source=(0,NaN)),(;source=(0,Inf)),(;source=(1,1),vertex=1),
                    (;source=[(1,1),(2,)]))
        @test !V.check_visual_request(enc;kind=:rank_section,options...).valid
    end
    s=V.inspection_session(enc;view=:rank_from,box=((-1,-1),(4,4)),cache_limit=4)
    try
        @test isempty(V.inspection_snapshot(s).metadata.rank_sections)
        V.select_inspection!(s;point=(1,1))
        snap=V.inspection_snapshot(s)
        row=only(snap.metadata.rank_sections)
        V.select_inspection!(s;point=(3//2,3//2))
        moved=V.inspection_snapshot(s)
        @test only(moved.metadata.rank_sections) === row
        @test moved.metadata.anchors == [(3//2,3//2)]
        @test snap.panels[1].metadata.geometry != moved.panels[1].metadata.geometry
        V.select_inspection!(s;parameter_pair=((3//2,3//2),(3,3)))
        @test only(V.inspection_snapshot(s).metadata.rank_sections) === row
        @test V.inspection_snapshot(s).metadata.inspection.rank == 1
        V.select_inspection!(s;view=:rank_to)
        @test !V.inspection_snapshot(s).metadata.from
        @test V.inspection_snapshot(s).metadata.anchors == [(3,3)]
        previous=V.inspection_snapshot(s)
        stats=V.inspection_summary(s)
        @test_throws ArgumentError V.select_inspection!(s;point=(NaN,0))
        @test V.inspection_snapshot(s) === previous
        @test V.inspection_summary(s).cache_entries == stats.cache_entries
        @test V.inspection_summary(s).cache_entries <= 4
        V.reset_inspection!(s)
        @test isempty(V.inspection_snapshot(s).metadata.rank_sections)
        @test V.inspection_selection(s).view == :rank_to
    finally
        V.close_inspection!(s)
    end
    @test V.inspection_summary(s).closed
end

@testset "A84 square survival and zero rank over every field" begin
    V=TamerOp.Visualization
    for field in FIELDS_FULL
        K=CM.coeff_type(field)
        enc=TamerOp.encode([TOA.BoxUpset([0.0,0.0]),TOA.BoxUpset([1.0,1.0])],
            [TOA.BoxDownset([2.0,2.0]),TOA.BoxDownset([3.0,3.0])],
            K[1 0;0 1],TOA.EncodingOptions(;backend=:pl_backend,poset_kind=:signature,field))
        for (p,q,rank,ds,dt) in (((1//2,1//2),(3//2,3//2),1,1,2),
                                ((3//2,3//2),(5//2,5//2),1,2,1),
                                ((1//2,1//2),(5//2,5//2),0,1,1),
                                ((2,2),(2,2),2,2,2),
                                ((2,2),(2+1//big(2)^60,2),1,2,1))
            for options in ((;source=p,point=q),(;target=q,point=p))
                s=V.visual_spec(enc;kind=:rank_section,options...,box=((-1,-1),(4,4)))
                @test s.metadata.inspection.rank == rank
                @test s.metadata.inspection.source_dimension == ds
                @test s.metadata.inspection.target_dimension == dt
                @test V.check_visual_spec(s).valid
            end
        end
    end
end

@testset "A84 pointer anchors retain exact clipping geometry" begin
    V=TamerOp.Visualization
    enc=_a84_grid()
    p=(0.48514856722685296,0.5084334129555489)
    box=((-1,-1),(4,4))
    for view in (:rank_from,:rank_to)
        session=V.inspection_session(enc;view,box)
        try
            V.select_inspection!(session;point=p,input=:pointer)
            snapshot=V.inspection_snapshot(session)
            request=view===:rank_from ? (;source=p) : (;target=p)
            expected=V.visual_spec(enc;kind=:rank_section,request...,box)
            @test snapshot.metadata.coordinate_input===:approximate_pointer
            @test V.inspection_selection(session).query_points==(p,)
            @test snapshot.panels[1].metadata.geometry.components==expected.panels[1].metadata.geometry.components
            @test all(v->all(x->x isa Rational{BigInt},v),
                [v for c in snapshot.panels[1].metadata.geometry.components for v in c.vertices])
            @test V.check_visual_spec(snapshot).valid
        finally
            V.close_inspection!(session)
        end
    end
end

@testset "A84 native static figures and live view controls" begin
    V=TamerOp.Visualization
    if Base.find_package("CairoMakie") === nothing || Base.find_package("WGLMakie") === nothing
        @test_skip false
    else
        @eval import CairoMakie
        @eval import WGLMakie
        M=_a84_chain(FIELD_QQ)
        enc=_a84_grid()
        mktempdir() do dir
            for (i,spec) in enumerate((V.visual_spec(M;kind=:rank_section,source=[1,2],vertex=3),
                V.visual_spec(enc;kind=:rank_section,target=(3,3),point=(1,1))))
                for palette in (:accessible,:grayscale)
                    path=joinpath(dir,"rank_$(i)_$(palette).svg")
                    V.save_visual(path,spec;backend=:cairomakie,style=V.VisualStyle(;palette))
                    @test filesize(path)>1000
                end
            end
        end
        s=V.inspection_session(enc)
        app=V.visualize(s;backend=:wglmakie)
        E=Base.get_extension(TamerOp,:TamerOpWGLMakieExt)
        ui=E._inspection_ui(app)
        B=WGLMakie.Bonito
        client=B.Session(B.NoConnection();asset_server=B.NoServer())
        try
            html=sprint(show,MIME"text/html"(),B.session_dom(client,app;init=false))
            @test occursin("Rank from source",html)
            @test occursin("Rank to target",html)
            @test !occursin("Bonito.Dropdown(",html)
            ui.controls.view.value[]=:rank_from
            ui.controls.point_x.value[]="1"
            ui.controls.point_y.value[]="1"
            ui.controls.select_point.value[]=true
            @test isempty(ui.error_text[])
            @test isempty(V.inspection_summary(s).listener_errors)
            @test ui.rank_figure[] !== nothing
            WGLMakie.Makie.update_state_before_display!(ui.rank_figure[])
            @test WGLMakie.Makie.widths(WGLMakie.Makie.viewport(ui.rank_axis[].scene)[])[2] > 300
            before=V.inspection_summary(s)
            ui.callbacks.hover(;point=(1,1))
            @test V.inspection_summary(s).snapshot_builds==before.snapshot_builds
            ui.controls.view.value[]=:rank_to
            @test !V.inspection_snapshot(s).metadata.from
            @test isempty(V.inspection_summary(s).listener_errors)
            ui.controls.view.value[]=:module
            @test ui.rank_figure[] === nothing
            ui.controls.view.value[]=:rank_from
            ui.controls.reset.value[]=true
            @test ui.rank_figure[] === nothing
            ui.controls.close.value[]=true
            @test V.inspection_summary(s).closed
            @test ui.closed[]
            @test isempty(ui.rank_observers)
        finally
            ui.dispose()
            close(client)
            V.close_inspection!(s)
        end
    end
end

@testset "A84 grayscale ranks remain distinct from the order mask" begin
    V=TamerOp.Visualization
    if Base.find_package("CairoMakie") === nothing
        @test_skip false
    else
        @eval import CairoMakie
        luminance(color) = begin
            rgb=CairoMakie.Makie.to_color(color)
            0.2126*rgb.r + 0.7152*rgb.g + 0.0722*rgb.b
        end
        style=V.VisualStyle(;palette=:grayscale)
        mask=luminance(V._visual_color(style,V._VisualRole(:absent)))
        for rank in 1:6
            @test mask-luminance(V._rank_section_color(rank,6)) > 0.1
        end
    end
end

@testset "A84 finite order is independent of numeric labels" begin
    V=TamerOp.Visualization
    relation=BitMatrix([i==j for i in 1:4,j in 1:4])
    for (u,v) in ((4,2),(4,3),(2,1),(3,1),(4,1))
        relation[u,v]=true
    end
    P=FF.FinitePoset(relation)
    M=MD.PModule{QQ}(P,ones(Int,4),Dict(e=>ones(QQ,1,1) for e in FF.cover_edges(P));field=FIELD_QQ)
    s=V.visual_spec(M;kind=:rank_section,source=2,vertex=3)
    @test only(s.metadata.rank_sections).ranks == [1,1,nothing,nothing]
    @test !s.metadata.inspection.defined
    @test s.metadata.inspection.relation == :incomparable
    @test s.panels[1].metadata.layout == :schematic
    @test V.visual_spec(M;kind=:rank_section,target=1,vertex=4).metadata.inspection.rank == 1
end

@testset "A84 slanted strict faces, singletons and exact algebraic anchors" begin
    V=TamerOp.Visualization
    PL=TamerOp.PLPolyhedra
    regions=[PL.HPoly(2,QQ[1 1],QQ[0],nothing,BitVector([true]),zero(QQ)),
        PL.HPoly(2,QQ[-1 -1;1 1],QQ[0,2],nothing,BitVector([false,true]),zero(QQ)),
        PL.HPoly(2,QQ[-1 -1],QQ[-2],nothing,BitVector([false]),zero(QQ))]
    pi=PL.PLEncodingMap(2,[BitVector() for _ in regions],[BitVector() for _ in regions],
        regions,[(-1,0),(1//2,1//2),(2,1)])
    P=chain_poset(3)
    M=MD.PModule{QQ}(P,[0,1,0],Dict((1,2)=>zeros(QQ,1,0),(2,3)=>zeros(QQ,0,1));field=FIELD_QQ)
    enc=RES.EncodingResult(P,M,EC.compile_encoding(P,pi))
    for (q,expected) in (((1//2,1//2),1),((1,1),0),((1//4,7//4),0))
        s=V.visual_spec(enc;kind=:rank_section,source=(0,0),point=q,box=((-1,-1),(3,3)))
        @test s.metadata.inspection.rank == expected
        @test V.check_visual_spec(s).valid
        for c in s.panels[1].metadata.geometry.components, (point,included) in zip(c.vertices,c.vertex_included)
            label=point[1]<0 || point[2]<0 ? -1 : sum(point)<2 ? 2 : 3
            @test included == (c.region_id==label)
        end
    end
    singleton=TamerOp.encode([TOA.BoxUpset([0,0])],[TOA.BoxDownset([0,0])],ones(QQ,1,1),
        TOA.EncodingOptions(;backend=:pl,field=FIELD_QQ))
    s=V.visual_spec(singleton;kind=:rank_section,source=(0,0),point=(0,0),box=((-1,-1),(1,1)))
    row=only(s.metadata.rank_sections)
    @test s.metadata.inspection.rank == 1
    @test any(c->c.dimension==0 && c.region_id>0 && row.ranks[c.region_id]==1,
        s.panels[1].metadata.geometry.components)
    @test V.visual_spec(singleton;kind=:rank_section,source=(0,0),point=(1,1)).metadata.inspection.rank==0
    AR=TamerOp.ExactReals.AlgebraicReal
    a=sqrt(AR(2))
    algebraic=V.visual_spec(_a84_grid();kind=:rank_section,source=(a,1),point=(a,2),box=((0,0),(3,3)))
    @test algebraic.metadata.anchors == [(a,1)]
    @test algebraic.metadata.inspection.rank == 1
    @test any(c->any(p->p[1]==a,c.vertices),algebraic.panels[1].metadata.geometry.components)
end

@testset "A84 cache-free sessions and owned selected matrices" begin
    V=TamerOp.Visualization
    M=_a84_chain(FIELD_QQ)
    s=V.inspection_session(M;view=:rank_from,cache_limit=0)
    try
        V.select_inspection!(s;pair=(1,2))
        row=only(V.inspection_snapshot(s).metadata.rank_sections)
        V.inspection_snapshot(s).metadata.inspection.matrix[1,1]=0
        @test MD.structure_map(M;source=1,target=2)==reshape(QQ[1,0],2,1)
        V.select_inspection!(s;pair=(1,3))
        @test only(V.inspection_snapshot(s).metadata.rank_sections).ranks==row.ranks
        @test V.inspection_summary(s).cache_entries==0
        @test V.inspection_snapshot(s).metadata.inspection.matrix==zeros(QQ,1,1)
        @test V.check_inspection_session(s).valid
    finally
        V.close_inspection!(s)
    end
end

@testset "A84 rank sections coexist with retained slices and lazy encodings" begin
    V=TamerOp.Visualization
    s=V.inspection_session(_a84_grid();view=:rank_from,box=((0,0),(4,4)),
        slice=(basepoint=(0,0),direction=(1,1)))
    try
        before=V.inspection_snapshot(s).metadata.slice_result
        V.select_inspection!(s;parameter_pair=((1,1),(3,3)))
        after=V.inspection_snapshot(s)
        @test after.metadata.slice_result === before
        @test length(after.panels)==4
        @test length(after.metadata.panel_positions)==4
        @test V.check_visual_spec(after).valid
        @test first(after.panels).metadata.slice_line == before.line
    finally
        V.close_inspection!(s)
    end
    d1=sparse([-1 -1 0;1 0 -1;0 1 1])
    d2=sparse(reshape([1,-1,1],3,1))
    graded=DT.GradedComplex([Int[1,2,3],Int[1,2,3],Int[1]], [d1,d2],
        [(0.,0.),(0.,0.),(0.,0.),(1.,0.),(1.,0.),(1.,0.),(2.,1.)])
    filtration=OPT.FiltrationSpec(kind=:graded,axes=([0.,1.,2.],[0.,1.]))
    enc=TamerOp.encode(graded,filtration;degree=1,field=FIELD_QQ,stage=:encoding_result)
    @test !RES.result_summary(enc).materialized
    @test V.check_visual_request(enc;kind=:rank_section,source=2).valid
    @test !RES.result_summary(enc).materialized
    s=V.inspection_session(enc;view=:rank_from)
    try
        @test !RES.result_summary(enc).materialized
        V.select_inspection!(s;pair=(2,5))
        snap=V.inspection_snapshot(s)
        @test !snap.metadata.module_materialized_before
        @test snap.metadata.module_materialized_after
        @test snap.metadata.inspection.rank==1
        @test snap.metadata.inspection.source_dimension==1
        @test snap.metadata.inspection.target_dimension==1
    finally
        V.close_inspection!(s)
    end
end

@testset "A84 numerical rank uses the declared tolerance" begin
    V=TamerOp.Visualization
    for (atol,expected) in ((1e-6,1),(1e-12,2))
        field=CM.RealField(Float64;atol,rtol=0.0)
        M=MD.PModule{Float64}(chain_poset(2),[2,2],Dict((1,2)=>[1.0 0;0 1e-8]);field)
        spec=V.visual_spec(M;kind=:rank_section,source=1,vertex=2)
        @test only(spec.metadata.rank_sections).ranks == [2,expected]
        @test spec.metadata.inspection.rank == expected
        @test !spec.metadata.inspection.exact
        @test occursin("atol=",spec.subtitle)
    end
end
