include("fixtures/stability.jl")

@testset "A51 exact inspector selections and rollback" begin
    V=TamerOp.Visualization;H=TamerOp.Stability
    chambers=_a51_chambers();s=V.stability_explorer(chambers;anchor=(-3//2,5))
    @test V.available_visuals(chambers)==(:stability,)
    @test V.check_inspection_session(s;throw=true).valid
    @test V.check_visual_spec(V.visual_spec(s)).valid
    snap=V.inspection_snapshot(s);geometry=snap.panels[1]
    @test geometry.metadata.encoding_axes==([-2,-1,1],[4,6,9])
    @test (-2//1,4//1) in first(geometry.metadata.polygons).polygon
    @test snap.metadata.slopes==[1//2,2//5]
    @test length(snap.panels)>=5
    generated=snap.panels[2].metadata.cells
    @test [c.vertex for c in generated]==[1,2,3]
    @test [c.dimension for c in generated]==[2,1,1]
    @test first(generated).polygon==[(-3//2,5),(-1,5),(-1,6),(-3//2,6)]
    original=V.inspection_selection(s);counter=Ref(0)
    token=V._on_inspection(s,_ -> counter[]+=1)
    for kwargs in ((;anchor=(1,5)),(;factor=99),(;vertex=5),(;threshold=NaN),(;view=:unknown))
        @test_throws ArgumentError V.select_inspection!(s;kwargs...)
        @test V.inspection_selection(s)==original
        @test V.inspection_snapshot(s)===snap
    end
    @test counter[]==0
    V.select_inspection!(s;anchor=(-3//2,21//4),threshold=1//3)
    @test V.inspection_snapshot(s).metadata.boundary
    @test V.inspection_snapshot(s).metadata.slopes==[8//15]
    @test counter[]==1
    V.reset_inspection!(s);@test V.inspection_selection(s).anchor==original.anchor
    V._off_inspection!(s,token);@test counter[]==2
    # The inspector owns its input, and exact query budget failure rolls back.
    empty!(chambers.chambers)
    V.select_inspection!(s;threshold=1//2)
    @test !isempty(s.object.chambers)
    old=V.inspection_snapshot(s);selection=V.inspection_selection(s);s.budget=H.HNBudget(max_states=0)
    @test_throws ArgumentError V.select_inspection!(s;anchor=(-7//4,5))
    @test V.inspection_snapshot(s)===old && V.inspection_selection(s)==selection
    V.close_inspection!(s);@test V.check_inspection_session(s).valid
    @test s.object===nothing && isempty(s.listeners)
    @test_throws ArgumentError V.select_inspection!(s;threshold=0)
    Q=TamerOp.FiniteFringe.ProductOfChainsPoset((1,));K=TamerOp.CoreModules.QQ
    Z=TamerOp.Modules.PModule{K}(Q,[0],Dict{Tuple{Int,Int},Matrix{K}}();field=TamerOp.CoreModules.QQField())
    empty_hn=H.hn_filtration(Z,H.StabilityWeights([1],[0]);witnesses=true)
    empty_view=V.stability_explorer(empty_hn)
    @test V.check_inspection_session(empty_view;throw=true).valid
    @test isempty(V.inspection_snapshot(empty_view).metadata.slopes)
    V.close_inspection!(empty_view)
    @test_throws ArgumentError V.stability_explorer(empty_hn;factor=1)
end

@testset "A51 landscape scale and bounded sample contracts" begin
    V=TamerOp.Visualization;H=TamerOp.Stability
    r=_a51_landscape();s=V.stability_explorer(r)
    first_panel=V.inspection_snapshot(s).panels[1]
    @test first_panel.layers[1].x==[-1.75,-1.25,0.]
    @test first_panel.layers[1].y==[4.5,5.5,7.]
    range=first_panel.metadata.colorrange
    for t in r.thresholds,k in r.layers
        V.select_inspection!(s;threshold=t,layer=k)
        p=V.inspection_snapshot(s).panels[1]
        @test p.metadata.colorrange==range
        j=findfirst(==(t),r.thresholds);l=findfirst(==(k),r.layers)
        @test p.metadata.lower==r.lower[:,j,l]
        @test p.metadata.upper==r.upper[:,j,l]
    end
    V.select_inspection!(s;view=:difference,threshold=1//2,baseline=0,layer=1,vertex=4)
    p=V.inspection_snapshot(s).panels[1];d=H.filtered_landscape_difference(r;threshold=1//2,baseline=0)
    @test p.metadata.lower==d.lower[:,1] && p.metadata.upper==d.upper[:,1]
    @test p.metadata.colorrange[1]==-p.metadata.colorrange[2]
    @test_throws ArgumentError V.select_inspection!(s;threshold=1//7)
    @test_throws ArgumentError V.visual_spec(r;budget=H.HNBudget(max_entries=1))
    V.close_inspection!(s)
end

@testset "A51 native geometry, linked lifecycle and portable states" begin
    V=TamerOp.Visualization;H=TamerOp.Stability
    s=_a51_session();snap=V.inspection_snapshot(s)
    landscape=_a51_landscape()
    for (backend,extname) in ((:cairomakie,:TamerOpCairoMakieExt),(:wglmakie,:TamerOpWGLMakieExt))
        ext=Base.get_extension(TamerOp,extname)
        if ext===nothing;@test_skip false;continue;end
        fig=V.render(snap;backend);M=backend===:cairomakie ? ext.CairoMakie.Makie : ext.WGLMakie.Makie
        axes=filter(a->a isa M.Axis,fig.content)
        @test axes[1].xlabel[]=="Parameter x"
        points=[Tuple(p) for plot in axes[1].scene.plots if plot isa M.Scatter for p in plot[1][]]
        @test (-1.5,5.) in points
        generated_points=[Tuple(p) for plot in axes[2].scene.plots if plot isa M.Scatter for p in plot[1][]]
        @test (-1.5,5.) in generated_points
        empty!(fig)
        fig=V.render(V.visual_spec(landscape);backend)
        heatmaps=[p for a in filter(a->a isa M.Axis,fig.content) for p in a.scene.plots if p isa M.Heatmap]
        @test length(heatmaps)==1
        @test Tuple(only(heatmaps).colorrange[])==(0.,Float64(maximum(landscape.upper)))
        expected=V.visual_spec(landscape).panels[1].layers[1]
        @test isequal(Matrix(only(heatmaps)[3][]),permutedims(expected.values))
        empty!(fig)
        if backend===:wglmakie
            ui=ext._inspection_ui(V.visualize(s;backend));linked=ext._inspection_ui(V.visualize(s;backend))
            original=ui.figure[];rev=V.inspection_selection(s).revision
            t=first(ui.targets[]);ui.callbacks.hover(t.point,t.axis_index)
            @test V.inspection_selection(s).revision==rev
            ui.controls.x.value[]="-3//2";ui.controls.y.value[]="21//4"
            ui.controls.apply.value[]=true;wait(s.requests.task)
            @test V.inspection_snapshot(s).metadata.boundary
            @test linked.controls.y.value[]=="21//4"
            ui.controls.factor.value[]="99";ui.controls.apply.value[]=true;wait(s.requests.task)
            @test occursin("factor",ui.error_text[])
            V.reset_inspection!(s)
            @test ui.figure[]===original && ui.update_count[]>0
            linked.dispose();ui.dispose()
            @test isempty(ui.observers) && isempty(ui.pointer_observers)
            # A different quantity changes the scale and its label, not the
            # canvas, coordinate frame or colorbar's connection to its heatmap.
            ls=V.stability_explorer(landscape)
            lui=ext._inspection_ui(V.visualize(ls;backend))
            original=lui.figure[];ax=only(lui.chart_axes[])
            heat=only(p for p in ax.scene.plots if p isa M.Heatmap)
            bar=only(b for b in original.content if b isa M.Colorbar)
            for view in (:difference,:landscape)
                V.select_inspection!(ls;view,threshold=1//2,layer=2,vertex=4)
                expected=V.inspection_snapshot(ls).panels[1]
                @test lui.figure[]===original && only(lui.chart_axes[])===ax
                @test only(p for p in ax.scene.plots if p isa M.Heatmap)===heat
                @test only(b for b in original.content if b isa M.Colorbar)===bar
                @test bar.label[]==expected.layers[1].colorbar_label
                @test Tuple(bar.limits[])==expected.metadata.colorrange
                @test isequal(Matrix(heat[3][]),permutedims(expected.layers[1].values))
            end
            V.reset_inspection!(ls)
            @test lui.figure[]===original && lui.rebuild_count[]==1
            V.close_inspection!(ls)
            @test isempty(lui.observers) && isempty(lui.pointer_observers)
        end
    end
    if Base.get_extension(TamerOp,:TamerOpCairoMakieExt)!==nothing
        mktempdir() do dir
            for suffix in ("png","svg","pdf")
                path=joinpath(dir,"hn.$suffix");V.save_visual(path,snap;backend=:cairomakie);@test filesize(path)>1000
            end
            file=get(ENV,"TAMEROP_STABILITY_HTML",joinpath(dir,"hn.html"));mkpath(dirname(file))
            chambers=V.visual_state(s);V.select_inspection!(s;anchor=(-3//2,21//4));wall=V.visual_state(s)
            values=V.visual_state(V.stability_explorer(landscape))
            difference=V.visual_state(V.stability_explorer(landscape;view=:difference,threshold=1//2))
            V.save_visual_session(file,(;chambers,wall,values,difference));@test filesize(file)>1000
        end
    end
    V.close_inspection!(s)
end
