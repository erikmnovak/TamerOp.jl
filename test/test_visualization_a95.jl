# Independent fixed-grid statistics and ordinary-interval oracles.
@testset "A95 aligned summaries and negative contracts" begin
    F=TamerOp.Featurizers;V=TamerOp.Visualization
    args=(axes=([0,2,7],),sample_ids=["a","b","c"],statistic="supplied ranks",protocol="fixed repeats",population="fixtures")
    X=Rational{Int}[0 2 8;2 4 4;4 6 0]
    e=F.persistence_ensemble(X;args...)
    X[1,1]=99
    @test F.feature_values(e)[1,1]==0
    @test F.sample_ids(e)==["a","b","c"]
    s=F.summarize_ensemble(e;quantiles=(0.25,0.75),threshold=3)
    d=F.ensemble_values(s)
    @test d.mean==[2,4,4]
    @test d.median==[2,4,4]
    @test d.std==[2,2,4]
    @test d.lower==[1,3,2] && d.upper==[3,5,6]
    @test d.frequency==[1/3,2/3,2/3] && d.counts==[3,3,3]
    @test F.summarize_ensemble(e;band=:range).data.lower==[0,2,0]
    near=F.persistence_ensemble(reshape([1//1,1+1//big(10)^20],2,1);axes=([0],),sample_ids=["a","b"],statistic="exact",protocol="given",population="fixtures")
    @test F.ensemble_values(F.summarize_ensemble(near;threshold=1)).frequency==[0.5]
    @test F.check_ensemble(e).valid && F.check_ensemble(s).valid
    report=F.check_ensemble(e)
    @test report isa TamerOp.Advanced.EnsembleValidationSummary
    @test F.ensemble_summary(report).valid && TamerOp.describe(report).kind===:ensemble_validation
    @test occursin("valid=true",sprint(show,report))
    broken=deepcopy(e);pop!(broken.ids)
    @test !F.check_ensemble(broken).valid && !isempty(F.check_ensemble(broken).issues)
    @test_throws ArgumentError F.check_ensemble(broken;throw=true)
    @test TamerOp.persistence_ensemble===F.persistence_ensemble
    @test occursin("replicates",sprint(show,e)) && occursin("quantile",sprint(show,s))
    for kw in ((;sample_ids=["a","a","c"]),(;axes=([0,7,2],)),(;axis_labels=[""]),
            (;sampling=:paired),(;max_cells=8),(;degree=-1),(;layer=0),(;protocol=""))
        @test_throws ArgumentError F.persistence_ensemble(ones(3,3);merge(args,kw)...)
    end
    for v in (NaN,Inf,true)
        @test_throws ArgumentError F.persistence_ensemble(fill(v,3,3);args...)
    end
    @test_throws ArgumentError F.persistence_ensemble(ones(3,2);args...)
    @test_throws ArgumentError F.persistence_ensemble(ones(3,2);merge(args,(;axes=([big(2)^60,big(2)^60+1],)))...)
    for kw in ((;band=:unknown),(;level=1),(;quantiles=(0.9,0.1)),(;threshold=Inf),
            (;scope=:all),(;max_work=8),(;band=:bootstrap),(;band=:hoeffding,bounds=(0,10)),(;bounds=(0,10)))
        @test_throws ArgumentError F.summarize_ensemble(e;kw...)
    end
    missing_e=F.persistence_ensemble([1 missing missing;3 4 missing];axes=([0,1,2],),sample_ids=["a","b"],
        statistic="partial",protocol="given",population="fixtures",sampling=:iid)
    m=F.ensemble_values(F.summarize_ensemble(missing_e))
    @test m.counts==[2,1,0] && m.mean[1:2]==[2,4] && isnan(m.mean[3])
    @test isnan(m.std[2]) && isnan(m.frequency[3])
    @test_throws ArgumentError F.summarize_ensemble(missing_e;band=:bootstrap)
    @test V.check_visual_spec(V.visual_spec(F.summarize_ensemble(missing_e))).valid
end

@testset "A95 known bounds and whole-row bootstrap oracles" begin
    F=TamerOp.Featurizers
    X=[0. 2.;1. 4.;0. 2.;1. 4.]
    e=F.persistence_ensemble(X;axes=([0,3],),sample_ids=string.(1:4),statistic="paired Bernoulli values",protocol="IID sample",population="fair Bernoulli",sampling=:iid)
    s=F.summarize_ensemble(e;band=:hoeffding,bounds=([0.,2.],[1.,4.]),level=0.5)
    w=sqrt(log(8)/8)
    @test isapprox(s.data.lower,max.([0,2],[0.5,3].-[w,2w]))
    @test isapprox(s.data.upper,min.([1,4],[0.5,3].+[w,2w]))
    @test s.contract.coverage===:conditional_finite_sample
    point=F.summarize_ensemble(e;band=:hoeffding,bounds=([0.,2.],[1.,4.]),level=0.5,scope=:pointwise)
    @test all(point.data.lower.>=s.data.lower) && all(point.data.upper.<=s.data.upper)
    for kw in ((;bounds=nothing),(;bounds=(0,1)),(;bounds=(2,0)),(;bounds=(NaN,10)),(;bounds=([0.],[4.])))
        @test_throws ArgumentError F.summarize_ensemble(e;band=:hoeffding,kw...)
    end
    exact=F.persistence_ensemble(ones(2,1);axes=([0],),sample_ids=["a","b"],statistic="bounded values",protocol="IID",population="tiny interval",sampling=:iid)
    tiny_upper=1+1//big(10)^20
    outward=F.summarize_ensemble(exact;band=:hoeffding,bounds=(1,tiny_upper))
    @test outward.contract.bounds.upper[1]>=tiny_upper
    @test outward.data.upper[1]>1
    @test outward.contract.bounds.declared==(1,tiny_upper)
    constant=F.persistence_ensemble(fill(0.1,3,1);axes=([0],),sample_ids=["a","b","c"],statistic="constant",protocol="IID",population="constant",sampling=:iid)
    fixed=F.summarize_ensemble(constant;band=:hoeffding,bounds=(0.1,0.1))
    @test fixed.data.mean==fixed.data.lower==fixed.data.upper==[0.1]
    @test fixed.data.std==[0.]
    bootfixed=F.summarize_ensemble(constant;band=:bootstrap,bootstrap_samples=100)
    @test bootfixed.data.lower==bootfixed.data.upper==[0.1]
    # Independent exhaustive Bernoulli law: actual joint coverage exceeds the bound.
    n=8;alpha=0.2;width=sqrt(log(4/alpha)/(2n))
    covered=sum(binomial(n,k)/2.0^n for k in 0:n if abs(k/n-0.5)<=width)
    @test covered>=1-alpha
    b=F.summarize_ensemble(e;band=:bootstrap,seed=17,bootstrap_samples=256,scope=:pointwise,level=0.75)
    @test b.data==F.summarize_ensemble(e;band=:bootstrap,seed=17,bootstrap_samples=256,scope=:pointwise,level=0.75).data
    # The second coordinate is 2+2X; whole-row bootstrap preserves dependence.
    @test isapprox(b.data.upper[2]-3,2*(b.data.upper[1]-0.5))
    joint=F.summarize_ensemble(e;band=:bootstrap,seed=17,bootstrap_samples=256,level=0.75)
    @test isapprox(joint.data.upper[1]-0.5,joint.data.upper[2]-3)
    @test isapprox(joint.data.upper[1]-0.5,b.data.upper[2]-3)
    # Every possible bootstrap mean is k/4: the reported radius must lie on this grid.
    @test isapprox((b.data.upper[1]-0.5)*4,round((b.data.upper[1]-0.5)*4))
    @test joint.contract.coverage===:approximate && joint.contract.seed==17
    @test_throws ArgumentError F.summarize_ensemble(e;band=:bootstrap,bootstrap_samples=99)
    @test_throws ArgumentError F.summarize_ensemble(e;band=:bootstrap,max_work=100)
end

@testset "A95 ordinary diagrams and nonuniform scalar grid oracles" begin
    F=TamerOp.Featurizers;O=TamerOp.OrdinaryPersistence;V=TamerOp.Visualization
    a=O.PersistenceDiagram([[(0//1,4//1),(1//1,3//1)]],[Rational{Int}[]])
    b=O.PersistenceDiagram([[(0//1,2//1)]],[Rational{Int}[]])
    args=(grid=[0//1,1//1,2//1,3//1,4//1],sample_ids=["wide","short"],protocol="given",population="fixtures")
    e=F.persistence_ensemble([a,b];args...)
    @test e.values==[0 1 2 1 0;0 1 0 0 0]
    @test F.persistence_ensemble([a,b];args...,layer=2).values==[0 0 1 0 0;0 0 0 0 0]
    @test F.persistence_ensemble([a,b];args...,statistic=:betti).values==[1 2 2 1 0;1 1 0 0 0]
    reverse=O.PersistenceDiagram([[(4//1,0//1)]],[[3//1]];order=:superlevel)
    kw=(grid=[0,1,3,4],sample_ids=["down"],protocol="given",population="fixtures")
    @test_throws ArgumentError F.persistence_ensemble([reverse];kw...)
    @test F.persistence_ensemble([reverse];kw...,statistic=:betti,essential=:include).values==[1 2 2 1]
    @test F.persistence_ensemble([reverse];kw...,essential=:drop).values==[0 1 1 0]
    @test_throws ArgumentError F.persistence_ensemble([reverse];kw...,essential=:include)
    @test_throws ArgumentError F.persistence_ensemble([a,b];args...,max_work=2)
    @test V.check_visual_spec(V.visual_spec(F.summarize_ensemble(e);diagram_samples=(1,2))).valid
    axes=([0.,2.,9.],[-3.,5.]);A=[1. 7.;2. 8.;3. 9.]
    fields=[V.ScalarField(axes,A;id="one"),V.ScalarField(axes,3A;id="three")]
    ef=F.persistence_ensemble(fields;protocol="given",population="fixtures")
    sf=F.summarize_ensemble(ef;threshold=5)
    @test F.ensemble_values(sf).mean==2A
    @test F.ensemble_values(sf).frequency==[0 1;0.5 1;0.5 1]
    spec=V.visual_spec(sf;probe=5,sample="three")
    @test spec.metadata.selected.coordinates==(2.,5.) && spec.metadata.selected.value==24
    @test first(spec.panels[1].layers).values==permutedims(2A)
    @test spec.panels[2].metadata.pick_targets[5].label=="Probe 5: (2.0, 5.0); 24.0"
    @test V.check_visual_spec(spec).valid
end

@testset "A95 transactional linked selection and retained sources" begin
    F=TamerOp.Featurizers;V=TamerOp.Visualization
    e=F.persistence_ensemble([0//1 1//3 2//3;1//1 4//3 5//3];axes=([0,2,7],),sample_ids=["a","b"],statistic="given",protocol="given",population="fixtures")
    s=F.summarize_ensemble(e)
    source=V.VisualizationSpec(:source;title="Source A",layers=V.AbstractVisualizationLayer[V.PointLayer([(0.,0.)],:black,1.,8.)])
    session=V.inspection_session(s;samples=[(;source),(;)])
    @test V.check_visual_spec(V.visual_spec(session)).valid
    notifications=Ref(0);token=V._on_inspection(session,_ -> (notifications[]+=1))
    V.select_inspection!(session;sample="b",probe=2,view=:frequency)
    snap=V.inspection_snapshot(session)
    @test snap.metadata.selected.value===4//3
    @test notifications[]==1 && V.check_inspection_session(session).valid
    state=V.inspection_selection(session)
    @test !V.check_inspection_selection(session;probe=0).valid
    @test_throws ArgumentError V.select_inspection!(session;sample="absent")
    @test V.inspection_snapshot(session)===snap && V.inspection_selection(session)==state
    V.reset_inspection!(session)
    @test last(V.inspection_snapshot(session).panels).title=="a: Source A"
    V._off_inspection!(session,token)
    V.close_inspection!(session)
    @test session.object===nothing && session.initial.samples===nothing && session.state.samples===nothing
    @test V.check_inspection_session(session).valid
    @test_throws ArgumentError V.select_inspection!(session;sample=1)
end

@testset "A95 native asymmetric rendering and export" begin
    F=TamerOp.Featurizers;V=TamerOp.Visualization
    e=F.persistence_ensemble([0. 1. 4.;2. 3. 8.];axes=([0,2,9],),sample_ids=["A","B"],statistic="values",protocol="supplied",population="fixtures")
    s=F.summarize_ensemble(e;band=:range)
    for backend in (:cairomakie,:wglmakie)
        package=backend===:cairomakie ? :CairoMakie : :WGLMakie
        if Base.find_package(string(package))===nothing
            @test_skip false;continue
        end
        @eval import $package
        mk=getfield(@__MODULE__,package)
        fig=V.visualize(s;backend)
        mk.Makie.update_state_before_display!(fig)
        ax=only(filter(a->a isa mk.Axis,fig.content))
        @test ax.aspect[]===nothing
        pts=filter(p->p isa mk.Scatter,ax.scene.plots)
        @test Tuple.(first(pts)[1][])==[(0.,1.),(2.,2.),(9.,6.)]
        @test any(p->p isa mk.Poly,ax.scene.plots)
        mktempdir() do dir
            for ext in (backend===:cairomakie ? ("png","svg","pdf") : ("html",))
                path=joinpath(dir,"ensemble."*ext)
                V.save_visual(path,s;backend)
                @test filesize(path)>1000
            end
        end
        empty!(fig)
        A=[1. 7.;2. 8.;3. 9.]
        fields=[V.ScalarField(([0.,2.,9.],[-3.,5.]),A;id="one"),V.ScalarField(([0.,2.,9.],[-3.,5.]),3A;id="three")]
        spatial=F.summarize_ensemble(F.persistence_ensemble(fields;protocol="given",population="fixtures"))
        fig=V.visualize(spatial;backend,probe=5,sample=2)
        mk.Makie.update_state_before_display!(fig)
        axes=filter(a->a isa mk.Axis,fig.content)
        @test length(axes)==2
        hm=only(filter(p->p isa mk.Heatmap,axes[1].scene.plots))
        @test hm[3][]==2A
        @test collect(hm[1][])==[-1.,1.,5.5,12.5]
        @test collect(hm[2][])==[-7.,1.,9.]
        marker=only(filter(p->p isa mk.Scatter,axes[1].scene.plots))
        @test Tuple.(marker[1][])==[(2.,5.)]
        empty!(fig)
    end
end
