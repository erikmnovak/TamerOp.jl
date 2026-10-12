# Independent finite-map oracles and explicit comparison certificates.
function _a88_square(;field=FIELD_QQ)
    CO=TamerOp.ChangeOfPosets;K=CM.coeff_type(field)
    P=FF.ProductOfChainsPoset((3,3));R=FF.ProductOfChainsPoset((2,2))
    id=EN.EncodingMap(P,P,collect(1:9));coarse=EN.EncodingMap(P,R,[1,1,2,1,1,2,3,3,4])
    joint=CO.joint_encoding(id,coarse)
    dims=[1,1,0,1,1,0,0,0,0]
    M=MD.PModule{K}(P,dims,Dict((a,b)=>ones(K,dims[b],dims[a]) for (a,b) in FF.cover_edges(P));field)
    ds=[1,0,0,0];N=MD.PModule{K}(R,ds,Dict((a,b)=>ones(K,ds[b],ds[a]) for (a,b) in FF.cover_edges(R));field)
    left=RES.EncodingResult(P,M,EC.GridEncodingMap(P,([0,1,2],[0,1,2])))
    right=RES.EncodingResult(R,N,EC.GridEncodingMap(R,([0,2],[0,2])))
    result=CO.encode_pmodules_to_common_poset(M,N,joint)
    modules=CO.translated_modules(result)
    f=MD.PMorphism(modules.left,modules.right,[CM.eye(field,d) for d in modules.left.dims])
    return (;P,R,id,coarse,joint,M,N,left,right,result,f)
end

@testset "A88 realized square and exact comparison matrices" begin
    V=TamerOp.Visualization;CO=TamerOp.ChangeOfPosets
    for field in (FIELD_QQ,CM.F2(),CM.F3(),CM.RealField(Float64))
        a=_a88_square(;field)
        spec=V.visual_spec(a.result;sources=(a.left,a.right),joint=a.joint,comparison=a.f,point=(1//2,1//2),box=((-1,-1),(3,3)))
        @test V.check_visual_spec(spec).valid
        @test spec.metadata.selection.query_labels==(1,1)
        @test spec.metadata.selection.vertex==1
        @test spec.metadata.comparison_validation.isomorphism
        @test spec.metadata.realization_scope===:supplied_finite_source
        @test spec.metadata.derived_invariance===:not_asserted
        regions=filter(p->p.kind===:transport_regions,spec.panels)
        @test length(regions)==2
        @test regions[1].metadata.selected_labels==[1]
        @test regions[2].metadata.selected_labels==[1]
        @test length(regions[1].metadata.geometry.components)>length(regions[2].metadata.geometry.components)
        labels=only(filter(p->p.kind===:joint_labels,spec.panels))
        @test length(labels.metadata.realized_pairs)==9
        @test labels.metadata.cartesian_count==36
        for p in 1:9,q in 1:9
            s=V.visual_spec(a.result;joint=a.joint,comparison=a.f,pair=(p,q),view=:square)
            @test V.check_visual_spec(s).valid
            @test s.metadata.naturality.defined==FF.leq(CO.common_poset(a.result),p,q)
            if FF.leq(CO.common_poset(a.result),p,q)
                e=s.metadata.naturality
                @test e.valid
                @test e.exact==!(field isa CM.RealField)
                @test e.left==e.right
                @test e.source_map==MD.structure_map(a.f.dom;source=p,target=q)
                @test e.target_component==MD.component(a.f,q)
            end
        end
        @test V.visual_spec(a.result;sources=(a.left,a.right),point=(-1,0)).metadata.selection.status===:no_common_label
        @test V.visual_spec(a.result;side=:right,label=1).metadata.selection.candidates==[1,2,4,5]
        @test V.visual_spec(a.joint).kind===:joint_encoding
        @test V.visual_spec(a.coarse).kind===:order_map
        @test :pushforward_overlay in V.available_visuals(a.coarse)
        @test V.check_visual_spec(V.visual_spec(a.coarse;kind=:pushforward_overlay)).valid
        restricted=TamerOp.restriction(a.coarse,a.right)
        rs=V.visual_spec(restricted;pair=(1,5),view=:square)
        @test rs.metadata.refinement===:restriction
        @test length(rs.metadata.selected_maps)==2
        @test rs.metadata.selected_maps[1].matrix==rs.metadata.selected_maps[2].matrix
    end
end

@testset "A88 abstract products, invalid comparisons and budgets" begin
    V=TamerOp.Visualization;CO=TamerOp.ChangeOfPosets;a=_a88_square()
    product=CO.encode_pmodules_to_common_poset(a.M,a.N)
    no_witness=V.visual_spec(product)
    @test length(no_witness.panels[1].metadata.vertex_ids)==36
    @test isempty(no_witness.panels[1].metadata.labeled_vertices)
    @test no_witness.metadata.refinement===:abstract_product
    @test no_witness.metadata.realization_scope===:not_certified
    witnessed=V.visual_spec(product;joint=a.joint,max_pairs=1)
    occupancy=only(filter(p->p.kind===:joint_labels,witnessed.panels))
    @test !occupancy.metadata.exhaustive
    @test length(occupancy.metadata.realized_pairs)==9
    @test length(occupancy.metadata.pairs)==36
    fake=FF.FinitePoset(Matrix{Bool}(I,9,9))
    fake_result=CO.CommonRefinementTranslationResult(fake,(nothing,nothing),
        EN.EncodingMap(fake,a.P,collect(1:9)),EN.EncodingMap(fake,a.R,[1,1,2,1,1,2,3,3,4]);refinement=:realized_joint_image)
    @test_throws ArgumentError V.visual_spec(fake_result;joint=a.joint)
    @test_throws ArgumentError V.visual_spec(a.result;max_vertices=2)
    for kw in ((;vertex=true),(;pair=(0,1)),(;pair=(1,2),vertex=1),(;max_pairs=0),
        (;side=:unknown),(;label=0),(;direction=:unknown),(;view=:none),(;matrix_limit=(0,2)),(;mystery=true))
        @test !V.check_visual_request(a.result;kw...).valid
    end
    @test_throws ArgumentError V.visual_spec(a.result;side=:left,label=100)
    @test_throws ArgumentError V.visual_spec(a.result;point=(0,0))
    @test_throws ArgumentError V.visual_spec(a.result;sources=(a.N,a.M))
    @test_throws ArgumentError V.visual_spec(a.result;comparison=MD.id_morphism(a.M))
    bad=MD.PMorphism(a.f.dom,a.f.cod,[copy(A) for A in a.f.comps]);bad.comps[1][1,1]=2
    sp=V.visual_spec(a.result;comparison=bad,pair=(1,2),view=:square)
    @test !sp.metadata.comparison_validation.valid
    @test !sp.metadata.naturality.valid
    # Same dimensions, different edge maps: do not accept a coordinate fiction.
    P=chain_poset(2);M=MD.PModule{QQ}(P,[1,1],Dict((1,2)=>ones(QQ,1,1));field=FIELD_QQ)
    N=MD.PModule{QQ}(P,[1,1],Dict((1,2)=>zeros(QQ,1,1));field=FIELD_QQ)
    pi=EN.EncodingMap(P,P,[1,2])
    @test_throws ArgumentError V.visual_spec(pi;sources=(M,N),comparison=MD.id_morphism(M))
    @test V.visual_spec(pi;sources=(M,N),comparison=MD.zero_morphism(M,N)).metadata.comparison_validation.valid
    @test !V.visual_spec(pi;sources=(M,N),comparison=MD.zero_morphism(M,N)).metadata.comparison_validation.isomorphism
    empty=FF.FinitePoset(falses(0,0));ep=EN.EncodingMap(empty,empty,Int[])
    @test V.check_visual_spec(V.visual_spec(ep)).valid
end

@testset "A88 finite category counterexample and Kan directions" begin
    V=TamerOp.Visualization;CO=TamerOp.ChangeOfPosets
    P=FF.FinitePoset(Bool[1 0 1 1;0 1 1 1;0 0 1 0;0 0 0 1]);R=chain_poset(1)
    M=MD.PModule{QQ}(P,ones(Int,4),Dict(e=>ones(QQ,1,1) for e in FF.cover_edges(P));field=FIELD_QQ)
    pi=EN.EncodingMap(P,R,ones(Int,4));enc=RES.EncodingResult(P,M,nothing)
    lan=TamerOp.pushforward_left(pi,enc);eta=CO.kan_unit(pi,M)
    sp=V.visual_spec(lan;comparison=eta,pair=(1,3),view=:square)
    @test sp.metadata.naturality.valid
    @test sp.metadata.comparison_validation.isomorphism
    @test sp.metadata.category===:finite_poset_representations
    @test sp.metadata.derived_invariance===:not_asserted
    opts=OPT.DerivedFunctorOptions(maxdeg=1,model=:projective)
    @test DF.dim(DF.Ext(M,M,opts),1)==1
    L=RES.translated_module(lan)
    @test DF.dim(DF.Ext(L,L,opts),1)==0
    ran=TamerOp.pushforward_right(pi,enc);eps=CO.kan_counit(pi,M;side=:right)
    @test V.visual_spec(ran;comparison=eps,direction=:right_to_left,pair=(1,4),view=:square).metadata.naturality.valid
end

@testset "A88 transactional linked selection and release" begin
    V=TamerOp.Visualization;a=_a88_square()
    s=V.inspection_session(a.result;sources=(a.left,a.right),joint=a.joint,comparison=a.f)
    @test s isa TamerOp.Advanced.TransportInspectionSession
    @test V.check_visual_spec(V.visual_spec(s)).valid
    @test V.check_inspection_session(s).valid
    calls=Ref(0);token=V._on_inspection(s,_ -> (calls[]+=1))
    V.select_inspection!(s;side=:right,label=1)
    @test V.inspection_summary(s).candidates==[1,2,4,5]
    @test V.inspection_summary(s).selection_status===:multiple_preimages
    V.select_inspection!(s;pair=(1,5),view=:square)
    snapshot=V.inspection_snapshot(s);revision=V.inspection_selection(s).revision
    V.select_inspection!(s;view=:overview)
    @test V.inspection_selection(s).pair==(1,5)
    @test V.inspection_snapshot(s).metadata.naturality.valid
    @test !V.check_inspection_selection(s;side=:right,label=100).valid
    before=V.inspection_snapshot(s)
    @test_throws ArgumentError V.select_inspection!(s;side=:right,label=100)
    @test V.inspection_snapshot(s)===before
    @test V.inspection_selection(s).revision==revision+1
    V.select_inspection!(s;point=(3//2,3//2))
    @test V.inspection_snapshot(s).metadata.selection.query_labels==(5,1)
    V.reset_inspection!(s)
    @test V.inspection_selection(s).pair===nothing
    @test calls[]==5
    V._off_inspection!(s,token)
    @test V.inspection_summary(s).listener_count==0
    final=V.inspection_snapshot(s);V.close_inspection!(s);V.close_inspection!(s)
    @test s.object===nothing && s.prepared===nothing
    @test V.inspection_snapshot(s)===final
    @test V.check_inspection_session(s).valid
    @test_throws ArgumentError V.select_inspection!(s;vertex=1)
end

@testset "A88 rendering exports and live controls" begin
    V=TamerOp.Visualization;a=_a88_square()
    for backend in (:cairomakie,:wglmakie)
        name=backend===:cairomakie ? "CairoMakie" : "WGLMakie"
        if Base.find_package(name)===nothing;@test_skip false;continue;end
        if backend===:cairomakie;@eval import CairoMakie;mk=CairoMakie.Makie
        else;@eval import WGLMakie;mk=WGLMakie.Makie;end
        for view in (:overview,:square),style in (V.VisualStyle(),V.VisualStyle(palette=:grayscale))
            spec=V.visual_spec(a.result;sources=(a.left,a.right),joint=a.joint,comparison=a.f,pair=(1,5),view)
            fig=V.render(spec;backend,style);mk.update_state_before_display!(fig)
            @test !isempty(fig.content)
            @test all(ax->all(>(0),mk.widths(mk.viewport(ax.scene)[])),filter(x->x isa mk.Axis,fig.content))
        end
        if backend===:cairomakie
            mktempdir() do dir
                for ext in ("png","svg","pdf")
                    path=joinpath(dir,"transport.$ext")
                    V.save_visual(path,a.result;comparison=a.f,pair=(1,5),view=:square,backend)
                    @test filesize(path)>1000
                end
            end
        else
            s=V.inspection_session(a.result;sources=(a.left,a.right),joint=a.joint,comparison=a.f)
            app=V.visualize(s;backend);ext=Base.get_extension(TamerOp,:TamerOpWGLMakieExt);ui=ext._inspection_ui(app)
            ui.controls.side.value[]="right";ui.controls.vertex.value[]="1";ui.controls.select.value[]=!ui.controls.select.value[]
            @test V.inspection_summary(s).candidates==[1,2,4,5]
            ui.controls.p.value[]="1";ui.controls.q.value[]="5";ui.controls.square.value[]=!ui.controls.square.value[]
            @test V.inspection_snapshot(s).metadata.naturality.valid
            @test ui.error_text[]==""
            ui.controls.overview.value[]=!ui.controls.overview.value[]
            @test V.inspection_selection(s).pair==(1,5)
            mk.update_state_before_display!(ui.figure[])
            target=first(ui.targets[]);count=V.inspection_summary(s).snapshot_builds
            ui.callbacks.hover(target.point,target.axis)
            @test !isempty(ui.hover_text[])
            @test V.inspection_summary(s).snapshot_builds==count
            ui.controls.vertex.value[]="999";ui.controls.select.value[]=!ui.controls.select.value[]
            @test !isempty(ui.error_text[])
            @test V.inspection_summary(s).snapshot_builds==count
            V.close_inspection!(s)
            @test ui.closed[]
            @test isempty(ui.observers) && isempty(ui.pointer_observers) && isempty(ui.targets[]) && isempty(ui.axes)
            @test V.check_inspection_session(s).valid
        end
    end
end

@testset "A88 nontrivial coordinates and numerical residuals" begin
    V=TamerOp.Visualization
    for field in (FIELD_QQ,CM.F2(),CM.F3(),CM.RealField(Float64;atol=1e-8,rtol=0.0))
        K=CM.coeff_type(field);P=chain_poset(2);pi=EN.EncodingMap(P,P,[1,2])
        A=K[1 2;0 1];B=K[1 1;1 2];F=K[1 1;0 1];G=K[1 0;1 1]
        M=MD.PModule{K}(P,[2,2],Dict((1,2)=>A);field)
        N=MD.PModule{K}(P,[2,2],Dict((1,2)=>B);field)
        f=MD.PMorphism(M,N,[F,G])
        spec=V.visual_spec(pi;sources=(M,N),comparison=f,pair=(1,2),view=:square,matrix_limit=(1,1))
        @test spec.metadata.comparison_validation.isomorphism
        e=spec.metadata.naturality
        @test e.source_map==A && e.target_map==B
        @test e.source_component==F && e.target_component==G
        @test e.left==e.right==K[1 2;1 3]
        @test size(e.left)==(2,2)
        @test all(p->size(only(p.layers).entries)==(1,1),filter(p->get(p.metadata,:panel_style,nothing)===:matrix,spec.panels))
        if field isa CM.RealField
            for (epsilon,expected) in ((1e-10,true),(1e-3,false))
                perturbed=MD.PMorphism(M,N,[F,G+[epsilon 0;0 0]])
                s=V.visual_spec(pi;sources=(M,N),comparison=perturbed,pair=(1,2),view=:square)
                @test s.metadata.naturality.valid==expected
                @test s.metadata.naturality.residual>0
                @test s.metadata.naturality.tolerance==1e-8
            end
        end
    end
end
