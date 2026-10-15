include("fixtures/cohomological_coordinates.jl")
@testset "A97 source and circular coordinate views" begin
    V=TamerOp.Visualization;C=TamerOp.CohomologicalCoordinates
    source,a=coordinate_torus_fixture();r=C.toroidal_coordinates(source;scale=0,cocycles=a)
    s=V.coordinate_explorer(r;vertex=7,coordinate=2)
    try
        snap=V.inspection_snapshot(s)
        @test V.check_visual_spec(snap).valid
        @test V.check_visual_spec(V.visual_spec(s)).valid
        @test length(snap.panels)==4
        @test snap.panels[1].metadata.values==r.values[:,2]
        @test snap.metadata.lifts==r.lifts
        @test snap.panels[1].metadata.source_axes==(1,2)
        @test all(p -> all(isapprox.(p[1],p[2])),zip(snap.panels[2].layers[2].points,[(cospi(2v),sinpi(2v)) for v in r.values[:,2]]))
        V.select_inspection!(s;phase=0.25,coordinate=1,source_axes=(3,1))
        @test V.inspection_snapshot(s).panels[1].metadata.values==mod.(r.values[:,1].+0.25,1.)
        before=V.inspection_snapshot(s);selected=V.inspection_selection(s)
        @test_throws ArgumentError V.select_inspection!(s;vertex=999)
        @test_throws ArgumentError V.select_inspection!(s;source_axes=(1,1))
        @test_throws ArgumentError V.select_inspection!(s;phase=Inf)
        @test V.inspection_snapshot(s)===before
        @test V.inspection_selection(s)==selected
        @test !V.check_inspection_selection(s;coordinate=0).valid
        @test V.visual_state(s).request.family===:snapshot
        for backend in (:cairomakie,:wglmakie)
            ext=Base.get_extension(TamerOp,backend===:cairomakie ? :TamerOpCairoMakieExt : :TamerOpWGLMakieExt)
            ext===nothing && continue
            fig=V.render(V.inspection_snapshot(s);backend)
            M=backend===:cairomakie ? ext.CairoMakie.Makie : ext.WGLMakie.Makie
            axes=filter(x->x isa M.Axis,fig.content)
            @test length(axes)>=3
            colored=[p for ax in axes for p in ax.scene.plots if p isa M.Scatter && p.color[] isa AbstractVector && eltype(p.color[])<:Real]
            @test length(colored)>=3
            @test all(p->Tuple(p.colorrange[])==(0.,1.),colored)
            readout_lines=only(last(V.inspection_snapshot(s).panels).layers).labels
            labels=filter(b->b isa M.Label && b.text[] in readout_lines,fig.content)
            @test length(labels)==length(readout_lines)
            @test all(b->minimum(b.layoutobservables.computedbbox[])[2]>=minimum(M.viewport(fig.scene)[])[2],labels)
            empty!(fig)
        end
        if Base.get_extension(TamerOp,:TamerOpCairoMakieExt)!==nothing
            mktempdir() do dir
                for suffix in ("png","svg","pdf")
                    path=joinpath(dir,"coordinates.$suffix");V.save_visual(path,V.inspection_snapshot(s);backend=:cairomakie)
                    @test filesize(path)>1000
                end
            end
            if haskey(ENV,"TAMEROP_A97_HTML")
                firstview=V.visual_state(s);V.select_inspection!(s;vertex=12,coordinate=2,phase=0.5)
                V.save_visual_session(ENV["TAMEROP_A97_HTML"],(;selected=firstview,rotated=V.visual_state(s)))
            end
        end
        V.reset_inspection!(s)
        @test V.inspection_selection(s).vertex==7
        @test V.check_inspection_session(s).valid
    finally;V.close_inspection!(s);end
    @test s.object===nothing
    @test V.check_inspection_session(s).valid
    @test_throws ArgumentError V.select_inspection!(s;vertex=1)
end

@testset "A97 live coordinate selection lifecycle" begin
    V=TamerOp.Visualization;ext=Base.get_extension(TamerOp,:TamerOpWGLMakieExt)
    if ext!==nothing
        r=TamerOp.circular_coordinates(coordinate_ring_fixture();scale=0,kind=:essential)
        s=V.coordinate_explorer(r)
        ui=ext._inspection_ui(V.visualize(s;backend=:wglmakie))
        try
            old=V.inspection_selection(s);fig=ui.figure[]
            target=first(ui.targets[])
            ui.callbacks.hover(target.point,target.axis_index)
            @test V.inspection_selection(s)==old
            ui.controls.vertex.value[]="4";ui.controls.phase.value[]="0.125"
            ui.controls.apply.value[]=true
            s.requests.task===nothing || wait(s.requests.task)
            @test isempty(ui.error_text[])
            @test V.inspection_selection(s).vertex==4
            @test V.inspection_selection(s).phase==0.125
            @test ui.figure[]===fig
            @test isempty(V.inspection_summary(s).listener_errors)
        finally;ui.dispose(;close_session=true);end
        @test isempty(ui.observers)
        @test isempty(ui.pointer_observers)
    end
end
