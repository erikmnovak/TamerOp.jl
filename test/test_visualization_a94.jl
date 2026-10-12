# Independent small integer/field and interval-membership comparison oracles.
function _a94_module(field;matrix=[1 1;1 -1],permuted=false)
    K=CM.coeff_type(field)
    P=permuted ? FF.FinitePoset(BitMatrix([true false;true true])) : chain_poset(2)
    MD.PModule{K}(P,[2,2],Dict((permuted ? (2,1) : (1,2))=>K.(matrix));field)
end
function _a94_encoding(coords;orientation=(1,1))
    P=FF.ProductOfChainsPoset((length(coords[1]),length(coords[2])))
    M=MD.PModule{QQ}(P,ones(Int,FF.nvertices(P)),Dict(e=>ones(QQ,1,1) for e in FF.cover_edges(P));field=FIELD_QQ)
    RES.EncodingResult(P,M,EC.compile_encoding(P,EC.GridEncodingMap(P,coords;orientation)))
end
_a94_values(s,name)=getproperty.(s.metadata.values[name],:value)

@testset "A94 native field ranks and shared signed scales" begin
    V=TamerOp.Visualization
    comparison=V.compare_visuals((f2=_a94_module(CM.F2()),f3=_a94_module(CM.F3()));
        vertices=(f2=[1,2],f3=[1,2]),correspondence="The same integral map reduced modulo each prime")
    spec=V.visual_spec(comparison;view=:rank,probe=2)
    # The source identity has rank two. The determinant -2 vanishes modulo
    # two, so the second rank is one in F2 and two in F3.
    expected=([2.;1.;;],[0.;0.;;],[2.;2.;;],[0.;1.;;])
    for backend in (:cairomakie,:wglmakie)
        package=backend===:cairomakie ? :CairoMakie : :WGLMakie
        if Base.find_package(string(package))===nothing
            @test_skip false
            continue
        end
        @eval import $package
        mk=getfield(@__MODULE__,package)
        fig=V.render(spec;backend)
        try
            mk.Makie.update_state_before_display!(fig)
            axes=filter(a->a isa mk.Axis,fig.content)
            @test length(axes)==4
            for (i,axis) in enumerate(axes)
                hm=only(filter(p->p isa mk.Heatmap,axis.scene.plots))
                @test hm[3][]==expected[i]
                @test collect(hm[1][])==[0.5,1.5,2.5]
                @test collect(hm[2][])==[0.,1.]
                @test Tuple(hm.colorrange[])==(isodd(i) ? (0.,2.) : (-1.,1.))
                @test all(>(0),mk.Makie.widths(mk.Makie.viewport(axis.scene)[]))
            end
            if backend===:cairomakie
                mktempdir() do dir
                    for extension in ("png","svg","pdf")
                        path=joinpath(dir,"field-ranks."*extension)
                        mk.save(path,fig)
                        @test filesize(path)>1000
                    end
                end
            end
        finally
            empty!(fig)
        end
    end
end

@testset "A94 characteristic and explicit correspondences" begin
    V=TamerOp.Visualization
    c=V.compare_visuals((f2=_a94_module(CM.F2()),f3=_a94_module(CM.F3();permuted=true));
        vertices=(f2=[1,2],f3=[2,1]),correspondence="Same integral matrix, corresponding source and target")
    h=V.visual_spec(c)
    @test _a94_values(h,:f2)==_a94_values(h,:f3)==[2,2]
    @test all(v->v.value==0,h.metadata.differences.f3)
    r=V.visual_spec(c;view=:rank,probe=2,matrix=true)
    @test _a94_values(r,:f2)==[2,1]
    @test _a94_values(r,:f3)==[2,2]
    @test getproperty.(r.metadata.differences.f3,:value)==[0,1]
    @test r.metadata.selected.f3.matrix==CM.coeff_type(CM.F3()).([1 1;1 -1])
    @test all(V.check_visual_spec(s).valid for s in (h,r))
    @test r.metadata.colorrange==(0.,2.)
    @test r.metadata.difference_range==(-1.,1.)
    @test V.describe(c).correspondence_status===:declared_not_certified
    @test V.available_visuals(c)==(:comparison,)
    @test occursin("f2",sprint(show,c))
    for name in (:ComparisonCase,:VisualComparison,:compare_visuals)
        @test getproperty(TamerOp,name)===getproperty(V,name)
    end
    @test TamerOp.Advanced.ComparisonInspectionSession===V.ComparisonInspectionSession
    rev=V.visual_spec(c;view=:rank,source=2)
    @test _a94_values(rev,:f2)==[nothing,2]
    @test rev.metadata.values.f2[1].status===:unordered
    for field in (CM.QQField(),CM.F2(),CM.F3(),CM.Fp(5))
        # Same map in changed source and target bases: B = T*A*S^-1.
        A=[1 0;0 0];B=[1 -1;1 -1]
        K=CM.coeff_type(field);S=K[1 1;0 1];T=K[1 0;1 1]
        @test K.(B)*S==T*K.(A)
        pair=V.compare_visuals((original=_a94_module(field;matrix=A),changed=_a94_module(field;matrix=B));
            vertices=(original=[1,2],changed=[1,2]),correspondence="Supplied bases with B*S=T*A")
        s=V.visual_spec(pair;view=:rank,probe=2,matrix=true)
        @test _a94_values(s,:original)==_a94_values(s,:changed)==[2,1]
        @test getproperty.(s.metadata.differences.changed,:value)==[0,0]
        @test s.metadata.selected.original.matrix!=s.metadata.selected.changed.matrix
    end
    for kw in ((;vertices=[1,2]),(;vertices=(f3=[1,2],f2=[1,2])),
            (;vertices=(f2=[1],f3=[1,2])),(;vertices=(f2=[0,2],f3=[1,2])),
            (;vertices=(f2=[1,2],f3=[1,2]),max_queries=3))
        @test_throws ArgumentError V.compare_visuals(c.cases;correspondence="test",kw...)
    end
    @test_throws ArgumentError V.compare_visuals(c.cases;vertices=(f2=[1],f3=[1]),correspondence="")
    @test_throws ArgumentError V.visual_spec(c;view=:barcode)
    @test_throws ArgumentError V.visual_spec(c;probe=true)
    @test_throws ArgumentError V.visual_spec(c;baseline=:unknown)
    @test_throws ArgumentError V.visual_spec(c;matrix=true)
    @test_throws ArgumentError V.ComparisonCase(c.cases.f2.object;degree=-1)
    @test_throws ArgumentError V.compare_visuals(c.cases;
        vertices=(f2=1:typemax(Int),f3=1:typemax(Int)),correspondence="Budget before materialization")
end

@testset "A94 exact parameter probes and numerical ambiguity" begin
    V=TamerOp.Visualization
    a=_a94_encoding(([0,2],[0,2]));b=_a94_encoding(([0,1,2],[0,1,2]))
    epsilon=1//big(2)^60
    qs=[(-1,0),(0,0),(1+epsilon,1),(1,1),(3,3)]
    c=V.compare_visuals((coarse=a,refined=b);points=qs,correspondence="Same parameter coordinates in constant modules")
    @test _a94_values(V.visual_spec(c),:coarse)==[nothing,1,1,1,1]
    r=V.visual_spec(c;view=:rank,source=3)
    @test _a94_values(r,:coarse)==[nothing,nothing,1,nothing,1]
    @test r.metadata.probes.coarse[3][1]==1+epsilon
    @test r.metadata.values.coarse[4].status===:unordered
    @test getproperty.(r.metadata.differences.refined,:value)==[nothing,nothing,0,nothing,0]
    reverse=_a94_encoding(([0,2],[0,2]);orientation=(-1,1))
    mapped=V.compare_visuals((forward=a,reversed=reverse);points=(forward=[(1,1),(3,3)],reversed=[(-1,1),(-3,3)]),
        correspondence="Reflect the first coordinate and its order")
    @test _a94_values(V.visual_spec(mapped;view=:rank),:reversed)==[1,1]
    @test_throws ArgumentError V.compare_visuals((a=V.ComparisonCase(a;units="m"),b=V.ComparisonCase(b;units="cm"));
        points=[(0,0)],correspondence="No implicit conversion")
    F=CM.RealField(Float64;atol=0.001,rtol=0.0)
    real=_a94_module(F;matrix=[1.0 0;0 0.0008])
    exact=_a94_module(CM.QQField();matrix=[1//1 0;0 8//10000])
    cc=V.compare_visuals((numerical=real,rational=exact);vertices=(numerical=[1,2],rational=[1,2]),correspondence="Declared rational reference")
    rr=V.visual_spec(cc;view=:rank,probe=2)
    @test rr.metadata.values.numerical[2].sensitivity.sensitive
    @test rr.metadata.values.numerical[2].sensitivity.ranks==(2,1)
    @test _a94_values(rr,:numerical)==[2,1]
    @test _a94_values(rr,:rational)==[2,2]
    @test rr.metadata.selected.numerical.matrix===nothing
    loose=_a94_module(CM.RealField(Float64;atol=2.0,rtol=0.0))
    identity=V.compare_visuals((a=loose,b=loose);vertices=(a=[1],b=[1]),correspondence="The identity of a stored two-dimensional space")
    idspec=V.visual_spec(identity;view=:rank,matrix=true)
    @test idspec.metadata.selected.a.value==only(idspec.metadata.values.a).value==2
    @test idspec.metadata.selected.a.sensitivity===nothing
end

@testset "A94 ordinary endpoints, degrees and coverage" begin
    V=TamerOp.Visualization;O=TamerOp.OrdinaryPersistence
    d=O.PersistenceDiagram([[(0//1,2//1),(0//1,2//1)]],[[1//1]])
    e=O.PersistenceDiagram([[(0//1,1//1)]],[[1//1]])
    qs=[-1//1,0//1,1//1,2//1,3//1]
    c=V.compare_visuals((first=d,second=e);parameters=qs,correspondence="Same scalar parameter",max_intervals=1)
    s=V.visual_spec(c;view=:barcode,probe=3)
    @test _a94_values(s,:first)==[0,2,3,1,1]
    @test _a94_values(s,:second)==[0,1,1,1,1]
    @test getproperty.(s.metadata.differences.second,:value)==[0,-1,-2,0,0]
    @test V.check_visual_spec(s).valid
    r=V.visual_spec(c;view=:rank,source=2)
    @test _a94_values(r,:first)==[nothing,2,2,0,0]
    @test s.panels[1].metadata.total_multiplicity==3
    @test s.panels[1].metadata.displayed_groups==1
    @test s.panels[1].metadata.window==s.panels[3].metadata.window
    sup=O.PersistenceDiagram([[(2,0)]],[[1]];order=:superlevel)
    cs=V.compare_visuals((a=sup,b=sup);parameters=[-1,0,1,2,3],correspondence="Superlevel original units")
    @test _a94_values(V.visual_spec(cs),:a)==[1,1,2,1,0]
    @test _a94_values(V.visual_spec(cs;view=:rank,source=3),:a)==[1,1,2,nothing,nothing]
    # Cellular differential multiplication by 2: survives in F2, dies in F3.
    G=DT.GradedComplex([[1],[2]],[sparse(reshape([2],1,1))],[(0,),(1,)])
    p2=O.persistence_diagram(G;field=CM.F2())
    p3=O.persistence_diagram(G;field=CM.F3())
    cc=V.compare_visuals((f2=p2,f3=p3);parameters=[0,1,2],correspondence="One integral cellular differential modulo the selected prime")
    @test _a94_values(V.visual_spec(cc),:f2)==[1,1,1]
    @test _a94_values(V.visual_spec(cc),:f3)==[1,0,0]
    @test V.describe(cc).context.f2.field==CM.F2()
    degrees=V.compare_visuals((H0=V.ComparisonCase(p2;degree=0),H1=V.ComparisonCase(p2;degree=1));
        parameters=[0,1,2],correspondence="Same parameters, different homology degrees")
    @test _a94_values(V.visual_spec(degrees),:H1)==[0,1,1]
    @test getproperty.(V.visual_spec(degrees).metadata.differences.H1,:value)==[-1,0,0]
    limited=O.PersistenceDiagram([Tuple{Int,Int}[]],[[0]];
        meta=(;window=(0,2),essential_interpretation=:survives_threshold_cutoff))
    coverage=V.compare_visuals((limited=limited,full=d);parameters=[-1,0,2,3],
        correspondence="Shared parameters with different query coverage",window=(-1,4))
    @test _a94_values(V.visual_spec(coverage),:limited)==[nothing,1,1,nothing]
    @test _a94_values(V.visual_spec(coverage;view=:rank,source=2),:limited)==[nothing,1,1,nothing]
    @test V.visual_spec(coverage;view=:barcode).metadata.window==(-1,4)
    limited_panel=V.visual_spec(coverage;view=:barcode).panels[1]
    @test only(limited_panel.metadata.records).death==2
    @test only(limited_panel.metadata.records).right_status===:censored
    @test only(limited_panel.metadata.stored_records).death==Inf
    @test limited_panel.axes.xlimits==V.visual_spec(coverage;view=:barcode).panels[3].axes.xlimits
    payload=(;metadata=(;restriction_window=(0,2)),order=:sublevel)
    @test V._comparison_covered(payload,2)
    @test !V._comparison_covered(payload,3)
    @test !V._comparison_covered(payload,-1)
end

@testset "A94 transactional linked selection and bounded reuse" begin
    V=TamerOp.Visualization
    c=V.compare_visuals((f2=_a94_module(CM.F2()),f3=_a94_module(CM.F3()));
        vertices=(f2=[1,2],f3=[1,2]),correspondence="Same integral map")
    s=V.inspection_session(c)
    @test V.check_inspection_session(s).valid
    @test V.check_visual_spec(V.visual_spec(s)).valid
    V.select_inspection!(s;view=:rank)
    @test V.inspection_summary(s).row_builds==2
    rows=s.rows;calls=Ref(0)
    token=V._on_inspection(s,_ -> (calls[]+=1))
    V.select_inspection!(s;probe=2,baseline=:f3)
    @test s.rows===rows && V.inspection_summary(s).row_builds==2
    @test calls[]==1
    @test getproperty.(V.inspection_snapshot(s).metadata.differences.f2,:value)==[0,-1]
    snapshot=V.inspection_snapshot(s);state=V.inspection_selection(s)
    @test !V.check_inspection_selection(s;probe=3).valid
    @test_throws ArgumentError V.select_inspection!(s;probe=3)
    @test V.inspection_snapshot(s)===snapshot && V.inspection_selection(s)==state
    V.select_inspection!(s;source=2)
    @test V.inspection_summary(s).row_builds==3
    V.reset_inspection!(s)
    @test V.inspection_selection(s).view===:hilbert
    V._off_inspection!(s,token)
    V.close_inspection!(s)
    @test s.object===nothing && s.rows===nothing
    @test V.check_inspection_session(s).valid
    @test_throws ArgumentError V.select_inspection!(s;probe=1)
    @test V.check_visual_spec(V.inspection_snapshot(s)).valid
end

@testset "A94 lazy encodings separate dimension and map costs" begin
    V=TamerOp.Visualization
    G=DT.GradedComplex([[1,2,3],[1,2,3],[1]],
        [sparse([-1 -1 0;1 0 -1;0 1 1]),sparse(reshape([1,-1,1],3,1))],
        [(0.,0.),(0.,0.),(0.,0.),(1.,0.),(1.,0.),(1.,0.),(2.,1.)])
    filtration=OPT.FiltrationSpec(kind=:graded,axes=([0.,1.,2.],[0.,1.]))
    enc=TamerOp.encode(G,filtration;degree=1,field=FIELD_QQ,stage=:encoding_result)
    c=V.compare_visuals((first=enc,second=enc);vertices=(first=[2,5],second=[2,5]),
        correspondence="Two views of the same lazily encoded module")
    @test !RES.result_summary(enc).materialized
    @test V.check_visual_spec(V.visual_spec(c)).valid
    @test !RES.result_summary(enc).materialized
    @test _a94_values(V.visual_spec(c;view=:rank),:first)==[1,1]
    @test RES.result_summary(enc).materialized
    @test_throws ArgumentError V.ComparisonCase(enc;degree=0)
end
