include("fixtures/source_laboratory.jl")
@testset "A124 fresh construction and independent rectangle oracle" begin
    V=TamerOp.Visualization;O=TamerOp.OrdinaryPersistence;D=TamerOp.DataIngestion
    @test TamerOp.source_laboratory===V.source_laboratory
    @test TOA.SourceLaboratorySession===V.SourceLaboratorySession
    for field in (TamerOp.CoreModules.F2(),TamerOp.CoreModules.F3())
        s=_a124_session(;field)
        @test O.finite_intervals(s.diagrams[1];dim=1)==[(2.0,sqrt(5.0))]
        @test isempty(O.essential_births(s.diagrams[1];dim=1))
        @test all(g->g[2]==sqrt(0.5),s.current.source.complex.grades)
        @test V.check_inspection_session(s).valid
        @test V.check_visual_spec(V.visual_spec(s)).valid
        @test V.inspection_snapshot(s).panels[1].axes==V.inspection_snapshot(s).panels[2].axes
        @test occursin("SourceLaboratorySession",sprint(show,s))
        @test V.describe(s).generation==1
        function fresh()
            d=V.source_draft(s)
            spec=D.RipsCodensityFiltration(max_dim=2,radius=d.max_diameter,dtm_mass=d.dtm_mass,nn_backend=:bruteforce)
            x=D.filtered_complex(TamerOp.DataTypes.PointCloud(d.points),spec)
            actual=V.source_object(s)
            @test actual.complex.grades==x.complex.grades
            @test actual.complex.boundaries==x.complex.boundaries
            @test actual.geometry==x.geometry
            @test actual.provenance.laboratory_input_ids==d.ids
            # Independent scalar filtration on all vertices at a high threshold.
            G=x.complex
            one=TamerOp.DataTypes.GradedComplex(G.cells_by_dim,G.boundaries,[(g[1],) for g in G.grades])
            diagram=O.persistence_diagram(one;field)
            @test O.finite_intervals(s.diagrams[2];dim=1)==O.finite_intervals(diagram;dim=1)
            @test O.essential_births(s.diagrams[2];dim=1)==O.essential_births(diagram;dim=1)
        end
        for kw in ((;move=(3,(2.4,1.3))),(;add=(0.3,0.7)),(;remove=2),(;dtm_mass=0.65),(;max_diameter=1.4))
            old=s.current;generation=s.state.generation
            V.edit_source!(s;kw...);@test s.current===old
            @test V.inspection_summary(s).dirty
            @test isempty(V.inspection_snapshot(s).panels[2].metadata.active_cells)
            V.rebuild_source!(s);@test s.current!==old
            @test s.state.generation==generation+1
            @test !V.inspection_summary(s).dirty
            fresh()
        end
        # Accessors cannot mutate committed inputs or history.
        V.source_draft(s).points[1,1]=999
        @test s.draft.points[1,1]==0.0
        V.source_object(s).complex.grades[1]=(99.,99.)
        @test s.current.source.complex.grades[1]!=(99.,99.)
        history=V.source_history(s);history[1].ids[1]=999
        @test s.history[1].ids[1]==1
        retained=s.diagrams;V.select_inspection!(s;diameter=0.4)
        @test s.diagrams[1]===retained[1] && s.diagrams[2]===retained[2]
        V.select_inspection!(s;codensity=0.0)
        @test isempty(O.finite_intervals(s.diagrams[2];dim=1))
        @test isempty(V.inspection_snapshot(s).panels[1].metadata.active_cells)
        @test isempty(V.inspection_snapshot(s).panels[1].metadata.active_source_ids)
        V.reset_inspection!(s);@test V.inspection_summary(s).dirty
        V.rebuild_source!(s);@test s.current.input.points==_a124_cloud().coords
        V.close_inspection!(s);@test V.check_inspection_session(s).valid
        @test_throws ArgumentError V.source_draft(s)
    end
end
@testset "A124 codensity boundary and duplicate source inputs" begin
    V=TamerOp.Visualization;O=TamerOp.OrdinaryPersistence;P=TamerOp.DataTypes.PointCloud
    s=V.source_laboratory(P([0.0 0.0;1.0 0.0;4.0 0.0]);degree=0,dtm_mass=0.4,max_diameter=5.,codensity=1.)
    @test V.inspection_snapshot(s).panels[1].metadata.active_source_ids==[1,2]
    @test O.finite_intervals(s.diagrams[2];dim=0)==[(0.,1.)]
    @test O.essential_births(s.diagrams[2];dim=0)==[0.]
    @test [s.current.source.complex.grades[i][2] for i in 1:3]==sqrt.([0.5,0.5,4.5])
    V.select_inspection!(s;codensity=sqrt(0.5));@test length(O.essential_births(s.diagrams[2];dim=0))==1
    V.select_inspection!(s;codensity=prevfloat(sqrt(0.5)));@test isempty(O.essential_births(s.diagrams[2];dim=0))
    V.edit_source!(s;add=(0.,0.));V.rebuild_source!(s);V.select_inspection!(s;codensity=0.)
    @test O.essential_births(s.diagrams[2];dim=0)==[0.]
    @test length(s.draft.ids)==4 && allunique(s.draft.ids)
    @test V.check_inspection_session(s).valid
    V.close_inspection!(s)
    @test_throws ArgumentError V.source_laboratory(P(zeros(3,3)))
    @test_throws ArgumentError V.source_laboratory(P(zeros(1,2)))
    @test_throws ArgumentError V.source_laboratory(P([0. 0.;1. 1.]);field=TamerOp.CoreModules.QQField())
end
@testset "A124 budgets history and failed obsolete cancelled rebuilds" begin
    V=TamerOp.Visualization;s=_a124_session(;history_limit=2)
    V.edit_source!(s;add=(0.2,0.3));id=last(s.draft.ids)
    V.undo_source_edit!(s);V.edit_source!(s;add=(0.4,0.5));@test last(s.draft.ids)>id
    V.undo_source_edit!(s);V.undo_source_edit!(s;redo=true);@test length(s.draft.ids)==5
    for i in 1:4;V.edit_source!(s;dtm_mass=0.3+i/100);end
    @test length(V.source_history(s))==3
    @test_throws ArgumentError V.source_laboratory(_a124_cloud();degree=2)
    @test_throws ArgumentError V.source_laboratory(_a124_cloud();max_points=3)
    @test_throws ArgumentError V.source_laboratory(_a124_cloud();max_matrix_entries=1)
    for kw in ((;dtm_mass=1),(;max_diameter=Inf),(;move=(100,(0,1))),(;add=(NaN,1)),(;remove=true),(;move=(1,(0,0)),add=(1,1)))
        old=V.inspection_selection(s);@test_throws ArgumentError V.edit_source!(s;kw...)
        @test V.inspection_selection(s)==old
    end
    @test !V.check_inspection_selection(s;codensity=-1).valid
    @test_throws ArgumentError V.check_inspection_selection(s;diameter=-1,throw=true)
    tiny=V.source_laboratory(_a124_cloud();max_diameter=0.1,max_cells=4)
    V.edit_source!(tiny;max_diameter=3.0);old=tiny.current
    @test_throws ArgumentError V.rebuild_source!(tiny)
    @test tiny.current===old && V.inspection_summary(tiny).dirty
    @test !tiny.busy
    V.undo_source_edit!(tiny);V.rebuild_source!(tiny);@test !V.inspection_summary(tiny).dirty
    # Queue an edit for the worker's next cooperative yield. Even a completed
    # reduction is obsolete if its draft changed before publication.
    owner=Ref(nothing);old=s.current;generation=s.state.generation
    V._request_inspection!(s,owner;rebuild=true)
    worker=s.requests.task
    changed=@async begin
        while s.requests.active===nothing && !istaskdone(worker);yield();end
        V.edit_source!(s;move=(1,(-1,0)))
    end
    wait(worker);wait(changed)
    @test s.current===old && s.state.generation==generation
    @test s.requests.committed==0 && s.requests.discarded>=1
    status=Symbol[]
    V._request_inspection!(s,owner;rebuild=true,status=(stage,error)->push!(status,stage))
    V._cancel_inspection_requests!(s,owner);task=s.requests.task;task===nothing||wait(task)
    @test s.requests.committed==0
    V._request_inspection!(s,owner;diameter=-1,status=(stage,error)->push!(status,stage));wait(s.requests.task)
    @test last(status)===:error && s.requests.committed==0
    V._request_inspection!(s,owner;rebuild=true,status=(stage,error)->push!(status,stage))
    V._request_inspection!(s,owner;rebuild=true,diameter=1.2,status=(stage,error)->push!(status,stage));wait(s.requests.task)
    @test last(status)===:committed && s.requests.committed==1
    @test s.state.diameter==1.2 && !V.inspection_summary(s).dirty
    @test s.requests.discarded>=1
    @test V.check_inspection_session(s).valid
    V.close_inspection!(s);V.close_inspection!(tiny)
end
@testset "A124 native coordinates exports and lifecycle" begin
    V=TamerOp.Visualization;s=_a124_session();spec=V.inspection_snapshot(s)
    original=V.visual_state(s)
    for (backend,name) in ((:cairomakie,:TamerOpCairoMakieExt),(:wglmakie,:TamerOpWGLMakieExt))
        ext=Base.get_extension(TamerOp,name)
        if ext===nothing;@test_skip false;continue;end
        fig=V.render(spec;backend)
        M=backend===:cairomakie ? ext.CairoMakie.Makie : ext.WGLMakie.Makie
        axes=filter(x->x isa M.Axis,fig.content)
        @test length(axes)==4
        for i in (1,2)
            @test Tuple(axes[i].limits[][1:2])==(spec.panels[i].axes.xlimits,spec.panels[i].axes.ylimits)
            points=[Tuple(q) for plot in axes[i].scene.plots if plot isa M.Scatter for q in plot[1][]]
            @test all(p->p in points,[(0.,0.),(2.,0.),(2.,1.),(0.,1.)])
        end
        @test axes[3].limits[]==axes[4].limits[]
        # The actual native line data include the independent rectangle barcode.
        points=[Tuple(q) for plot in axes[3].scene.plots if plot isa M.LineSegments for q in plot[1][]]
        @test any(p->isapprox(p[1],2.;atol=1e-6),points)
        @test any(p->isapprox(p[1],sqrt(5.);atol=1e-6),points)
        empty!(fig)
        if backend===:cairomakie
            mktempdir() do dir
                for suffix in ("svg","png","pdf")
                    path=joinpath(dir,"comparison.$suffix");V.save_visual(path,spec;backend);@test filesize(path)>1000
                end
            end
        else
            app=V.visualize(s;backend);ui=ext._inspection_ui(app)
            ui.controls.x.value[]="0.4";ui.controls.y.value[]="0.2";ui.controls.move.value[]=true
            @test V.inspection_summary(s).dirty
            ui.controls.rebuild.value[]=true;@test !V.inspection_summary(s).dirty
            @test ui.update_count[]>0
            ui.controls.mass.value[]="1";ui.controls.settings.value[]=true;@test occursin("strictly",ui.error_text[])
            ui.dispose();@test isempty(ui.observers) && isempty(ui.pointer_observers)
            closing=_a124_session();view=ext._inspection_ui(V.visualize(closing;backend))
            entered=Channel{Nothing}(1);release=Channel{Nothing}(1)
            V._on_inspection(closing,z->begin
                if !V.inspection_summary(z).closed;put!(entered,nothing);take!(release);end
            end)
            update=@async V.rebuild_source!(closing);take!(entered)
            view.controls.close.value[]=true;@test !V.inspection_summary(closing).closed
            put!(release,nothing);wait(update);wait(view.close_task[])
            @test view.closed[] && V.inspection_summary(closing).closed
            @test isempty(view.observers) && isempty(view.pointer_observers)
            @test isempty(V.inspection_summary(closing).listener_errors)
        end
    end
    V.edit_source!(s;move=(3,(2.3,1.4)));V.rebuild_source!(s)
    if Base.get_extension(TamerOp,:TamerOpCairoMakieExt)!==nothing
        mktempdir() do tmp
            file=get(ENV,"TAMEROP_SOURCE_LAB_HTML",joinpath(tmp,"comparison.html"));mkpath(dirname(file))
            edited=V.visual_state(s)
            V.save_visual_session(file,(;original,edited);title="Source and construction comparison")
            @test filesize(file)>1000
            json=joinpath(tmp,"comparison.json");V.save_visual_session(json,(;original,edited))
            saved=V.load_visual_session(json)
            @test V.inspection_snapshot(saved).metadata.draft==original.snapshot.metadata.draft
        end
    end
    V.close_inspection!(s)
end
@testset "A124 cancelling an active rebuild and closing" begin
    V=TamerOp.Visualization
    for operation in (:cancel,:close)
        s=_a124_session();old=s.current;generation=s.state.generation;owner=Ref(nothing)
        V.edit_source!(s;move=(1,(-0.4,0.2)))
        V._request_inspection!(s,owner;rebuild=true);worker=s.requests.task
        observed=Ref(false)
        interrupt=@async begin
            while s.requests.active===nothing && !istaskdone(worker);yield();end
            observed[]=s.requests.active!==nothing
            operation===:cancel ? V._cancel_inspection_requests!(s,owner) : V.close_inspection!(s)
        end
        wait(worker);wait(interrupt)
        @test observed[] && s.requests.discarded==1 && s.requests.committed==0
        @test s.state.generation==generation
        if operation===:cancel
            @test s.current===old && V.inspection_summary(s).dirty
            V.rebuild_source!(s);@test !V.inspection_summary(s).dirty
            V.close_inspection!(s)
        else
            @test V.inspection_summary(s).closed && s.current===nothing
        end
        @test V.check_inspection_session(s).valid
    end
end
