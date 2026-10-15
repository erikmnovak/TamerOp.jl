include("fixtures/temporal.jl")
@testset "A55 temporal views and transactional selection" begin
    V=TamerOp.Visualization;T=TamerOp.TemporalPersistence;r=temporal_cloud_fixture()
    @test T.check_temporal(r).valid
    @test r.barcodes[3].supports[2]==(1,5)
    @test V.available_visuals(r)==(:temporal_persistence,)
    s=V.temporal_explorer(r;occurrence=3,scale_index=3,interval=2)
    try
        snap=V.inspection_snapshot(s)
        @test V.check_visual_spec(snap).valid
        @test V.check_visual_spec(V.visual_spec(s)).valid
        @test snap.metadata.source_cycle!==nothing
        @test snap.metadata.directions==[:forward,:backward,:forward,:backward]
        @test snap.metadata.landscape.metric===:max_norm_on_integer_indices
        # Layers share a nonnegative scale, including an identically zero layer.
        @test snap.panels[4].metadata.colorrange==(0.,2.)
        @test V._temporal_landscape_panel(r,s.prepared.landscape,merge(s.state,(;layer=2))).metadata.colorrange==(0.,2.)
        old=s.prepared.landscape
        V.select_inspection!(s;occurrence=4)
        @test s.prepared.landscape===old
        @test V.inspection_summary(s).landscape_preparations==1
        V.select_inspection!(s;scale_index=1)
        @test V.inspection_selection(s).interval==0
        @test V.inspection_snapshot(s).metadata.source_cycle===nothing
        V.select_inspection!(s;interval=1)
        @test V.inspection_snapshot(s).metadata.source_cycle!==nothing
        before=V.inspection_snapshot(s);selection=V.inspection_selection(s)
        @test_throws ArgumentError V.select_inspection!(s;occurrence=99)
        @test_throws ArgumentError V.select_inspection!(s;layer=0)
        @test V.inspection_snapshot(s)===before
        @test V.inspection_selection(s)==selection
        @test !V.check_inspection_selection(s;scale_index=0).valid
        saved=V.visual_state(s)
        @test saved.request.family===:snapshot
        @test saved.snapshot.metadata.source_cycle!==nothing
        @test V.check_inspection_session(s).valid
        V.reset_inspection!(s)
        @test V.inspection_selection(s).occurrence==3
        @test V.inspection_selection(s).scale_index==3
        for backend in (:cairomakie,:wglmakie)
            ext=Base.get_extension(TamerOp,backend===:cairomakie ? :TamerOpCairoMakieExt : :TamerOpWGLMakieExt)
            ext===nothing && continue
            fig=V.render(V.inspection_snapshot(s);backend)
            M=backend===:cairomakie ? ext.CairoMakie.Makie : ext.WGLMakie.Makie
            axes=filter(x->x isa M.Axis,fig.content)
            @test length(axes)>=5
            heat=only(p for ax in axes for p in ax.scene.plots if p isa M.Heatmap)
            @test Tuple(heat.colorrange[])==(0.,2.)
            for panel in V.inspection_snapshot(s).panels[2:5]
                plan=ext._HANDLERS.plan(panel;style=V.VisualStyle())
                labels=filter(c->c.kind===:text,plan.commands)
                @test all(c->all(==((:center,:center)),c.kwargs.align),labels)
            end
            empty!(fig)
        end
        if Base.get_extension(TamerOp,:TamerOpCairoMakieExt)!==nothing
            mktempdir() do dir
                for suffix in ("png","svg","pdf")
                    path=joinpath(dir,"temporal.$suffix");V.save_visual(path,V.inspection_snapshot(s);backend=:cairomakie)
                    @test filesize(path)>1000
                end
            end
            if haskey(ENV,"TAMEROP_A55_HTML")
                current=V.visual_state(s);V.select_inspection!(s;occurrence=5,scale_index=5)
                V.save_visual_session(ENV["TAMEROP_A55_HTML"],(;selected=current,last=V.visual_state(s)))
            end
        end
    finally;V.close_inspection!(s);end
    @test s.object===s.prepared===nothing
    @test V.check_inspection_session(s).valid
    @test_throws ArgumentError V.select_inspection!(s;occurrence=1)
end

@testset "A55 live hover preserves exact controls" begin
    V=TamerOp.Visualization;ext=Base.get_extension(TamerOp,:TamerOpWGLMakieExt)
    if ext!==nothing
        s=V.temporal_explorer(temporal_cloud_fixture();occurrence=3,scale_index=3)
        ui=ext._inspection_ui(V.visualize(s;backend=:wglmakie))
        try
            figure=ui.figure[];axes=copy(ui.chart_axes[]);M=ext.WGLMakie.Makie
            @test length(ui.retained[].panels)==5
            @test M.widths(M.viewport(axes[3].scene)[])[1]>1.5M.widths(M.viewport(axes[4].scene)[])[1]
            before=V.inspection_selection(s)
            target=ui.callbacks.hover((3.,3.),4)
            @test target.selection==(;occurrence=3,scale_index=3)
            @test V.inspection_selection(s)==before
            ui.controls.occurrence.value[]="2"
            ui.controls.scale.value[]="1"
            ui.controls.interval.value[]="0"
            ui.controls.apply.value[]=true
            s.requests.task===nothing || wait(s.requests.task)
            @test isempty(ui.error_text[])
            @test V.inspection_selection(s).scale_index==1
            @test V.inspection_selection(s).occurrence==2
            @test isempty(V.inspection_summary(s).listener_errors)
            @test ui.figure[]===figure
            @test all(a===b for (a,b) in zip(ui.chart_axes[],axes))
        finally
            ui.dispose(;close_session=true)
        end
        @test isempty(ui.observers)
        @test isempty(ui.pointer_observers)
    end
end

@testset "A55 single and empty temporal geometry" begin
    V=TamerOp.Visualization;T=TamerOp.TemporalPersistence
    for cloud in ([(0.,0.)],zeros(0,2))
        r=T.temporal_persistence([TamerOp.PointCloud(cloud)];scales=[0],representatives=true)
        spec=V.visual_spec(r)
        @test V.check_visual_spec(spec).valid
        @test size(spec.panels[4].metadata.values)==(1,1)
        @test spec.panels[4].metadata.colorrange==(0.,1.)
        @test all(iszero,spec.metadata.landscape.values)
    end
end

@testset "A55 longer time paths retain maps without overlapping every label" begin
    V=TamerOp.Visualization;T=TamerOp.TemporalPersistence
    r=T.temporal_persistence([[(0.,0.)] for _ in 1:21];scales=[0])
    panel=V._temporal_time_panel(r,(;occurrence=21))
    @test length(panel.metadata.pick_targets)==41
    @test length(panel.metadata.directions)==40
    @test length(panel.axes.xticks[1])<=10
    labels=[text for layer in panel.layers if layer isa V.TextLayer for text in layer.labels]
    @test count(x->x in ("source","union"),labels)==1
end
