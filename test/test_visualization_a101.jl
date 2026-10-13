@testset "A101 exact certificate views and lifecycle" begin
    V=TamerOp.Visualization
    for field in (CM.QQField(),CM.F2(),CM.F3())
        M=_pruning_square_fixture(;field)
        r=PR.epsilon_pruning(M,1//2)
        session=V.inspection_session(r;point=(2,2))
        @test V.check_visual_spec(V.visual_spec(session)).valid
        @test V.check_inspection_session(session).valid
        initial=V.inspection_selection(session)
        for view in (:composition,:naturality,:domains,:pruning),side in (:forward,:reverse)
            V.select_inspection!(session;view,side)
            s=V.inspection_snapshot(session)
            @test V.check_visual_spec(s).valid
            @test s.metadata.verified
            e=s.metadata.equations
            @test e.forward.lhs==e.forward.rhs
            @test e.reverse.lhs==e.reverse.rhs
            if view===:domains
                domains=filter(p->p.kind===:certificate_domain,s.panels)
                @test length(domains)==4
                for (panel,M) in zip(domains,(r.source,r.pruned,IL.parameter_shift(r.source,2),IL.parameter_shift(r.pruned,2)))
                    for cell in panel.metadata.cells
                        @test cell.dimension==IL.ambient_dimension(M,cell.point)
                    end
                    for cell in panel.metadata.boundary
                        @test cell.dimension==IL.ambient_dimension(M,cell.point)
                    end
                end
                @test all(p->p.axes.xlimits==domains[1].axes.xlimits && p.axes.ylimits==domains[1].axes.ylimits,domains)
            end
        end
        before=V.inspection_selection(session);snapshot=V.inspection_snapshot(session)
        for opts in ((;vertex=0),(;point=(NaN,0)),(;point=(0,)),(;edge=99999),(;side=:both),(;matrix_limit=(0,2)),(;unknown=true))
            @test_throws ArgumentError V.select_inspection!(session;opts...)
            @test !V.check_inspection_selection(session;opts...).valid
            @test V.inspection_selection(session)==before
            @test V.inspection_snapshot(session)===snapshot
        end
        # The session owns its checked mathematical input.
        r.shift_map.comps[findfirst(!isempty,r.shift_map.comps)].*=0
        V.select_inspection!(session;view=:composition,point=(5//2,7//2))
        @test V.inspection_snapshot(session).metadata.verified
        V.reset_inspection!(session)
        @test V.inspection_selection(session).vertex==initial.vertex
        V.close_inspection!(session)
        @test V.check_inspection_session(session).valid
        @test session.prepared===nothing && isempty(session.listeners)
        @test_throws ArgumentError V.select_inspection!(session;vertex=1)
        @test V.check_visual_spec(V.inspection_snapshot(session)).valid
    end
end

@testset "A101 finite translation and certificate rejection" begin
    V=TamerOp.Visualization;Q=FF.GridPoset(([1,2],))
    M=MD.PModule{QQ}(Q,[1,1],Dict((1,2)=>ones(QQ,1,1));field=CM.QQField())
    c=IL.find_interleaving(M,M,[2,2]).certificate
    s=V.inspection_session(c)
    @test V.inspection_snapshot(s).metadata.provenance.epsilon===nothing
    @test V.inspection_snapshot(s).panels[1].metadata.coordinate_space===:schematic
    @test_throws ArgumentError V.select_inspection!(s;view=:domains)
    @test_throws ArgumentError V.select_inspection!(s;view=:pruning)
    @test_throws ArgumentError V.select_inspection!(s;point=(1,))
    @test !V.check_visual_request(c;view=:domains).valid
    @test !V.check_visual_request(c;view=:pruning).valid
    @test !V.check_visual_request(c;point=(1,)).valid
    @test !V.check_visual_request(c;matrix_limit=(0,2)).valid
    V.select_inspection!(s;view=:naturality)
    @test V.inspection_snapshot(s).metadata.selected.valid
    c.forward.comps[1].*=0
    @test_throws ArgumentError V.visual_spec(c)
    V.close_inspection!(s)
end

@testset "A101 exact display limits and all parameter coordinates" begin
    V=TamerOp.Visualization
    constant=IL.RectilinearModule((Int[],),[1],Dict{Tuple{Int,Int},Matrix{QQ}}())
    identity=IL.find_interleaving(constant,constant,0).certificate
    chart=V.visual_spec(identity).panels[1]
    @test only(filter(layer->layer isa V.TextLayer,chart.layers)).labels==["x = x+d = x+2d"]
    A=IL.RectilinearModule((Int[],Int[],Int[]),[1],Dict{Tuple{Int,Int},Matrix{QQ}}())
    c=IL.find_interleaving(A,A,0).certificate
    s=V.inspection_session(c)
    @test V.inspection_snapshot(s).panels[1].metadata.coordinate_space===:schematic
    @test length(c.problem.points[V.inspection_selection(s).vertex])==3
    @test !V.check_inspection_selection(s;view=:domains).valid
    @test_throws ArgumentError V.select_inspection!(s;view=:domains)
    V.close_inspection!(s)
    a=big(10)^20;dims=[0,1,1,0,0]
    M=IL.RectilinearModule(([a,a+1],),dims,Dict((v,v+1)=>ones(QQ,dims[v+1],dims[v]) for v in 1:4))
    c=IL.find_interleaving(M,M,0).certificate
    s=V.inspection_session(c)
    @test V.inspection_snapshot(s).panels[1].metadata.coordinate_space===:schematic
    @test any(!iszero,V.inspection_snapshot(s).metadata.selected.first)
    @test_throws ArgumentError V.select_inspection!(s;view=:domains)
    @test V.inspection_snapshot(s).metadata.selection.view===:composition
    V.close_inspection!(s)
end

@testset "A101 static and live native rendering" begin
    V=TamerOp.Visualization
    s=V.inspection_session(PR.epsilon_pruning(_pruning_box_fixture(),1//2);point=(1,1))
    @test V.inspection_snapshot(s).metadata.selected.lhs==ones(CM.coeff_type(CM.F2()),1,1)
    for package in (:CairoMakie,:WGLMakie)
        if Base.find_package(string(package))===nothing
            @test_skip false
            continue
        end
        @eval import $package
        @info "A101 native rendering" package
        mk=getfield(@__MODULE__,package);backend=package===:CairoMakie ? :cairomakie : :wglmakie
        frames=Pair{Symbol,V.VisualSessionState}[]
        for view in (:composition,:naturality,:domains,:pruning)
            V.select_inspection!(s;view)
            fig=V.render(V.inspection_snapshot(s);backend,size=(1600,1400))
            try
                mk.Makie.update_state_before_display!(fig)
                axes=filter(x->x isa mk.Axis,fig.content)
                @test !isempty(axes)
                @test axes[1].xlabel[]=="parameter 1"
                @test all(a->all(>(0),mk.Makie.widths(mk.Makie.viewport(a.scene)[])),axes)
                if package===:CairoMakie
                    dir=get(ENV,"TAMEROP_A101_OUTPUT",joinpath(@__DIR__,"browser","artifacts","certificates"));mkpath(dir)
                    for extension in ("png","svg","pdf")
                        path=joinpath(dir,"certificate-$(view).$(extension)");mk.save(path,fig)
                        @test filesize(path)>1000
                    end
                    push!(frames,view=>V.visual_state(s))
                end
            finally;empty!(fig);end
        end
        if package===:CairoMakie
            dir=get(ENV,"TAMEROP_A101_OUTPUT",joinpath(@__DIR__,"browser","artifacts","certificates"))
            V.save_visual_session(joinpath(dir,"certificates.html"),(;frames...);title="Exact certificates",
                caption="Four retained views of a verified certificate. These controls select saved views; they do not run Julia queries.",size=(1600,1400))
            @test filesize(joinpath(dir,"certificates.html"))>1000
        end
        if package===:WGLMakie
            app=V.visualize(s;backend)
            ext=Base.get_extension(TamerOp,:TamerOpWGLMakieExt);ui=ext._inspection_ui(app)
            before=V.inspection_selection(s)
            ui.callbacks.hover(first(ui.targets[]).point)
            @test V.inspection_selection(s)==before
            ui.dispose()
            @test ui.closed[] && isempty(ui.observers) && isempty(ui.pointer_observers)
        end
    end
    V.close_inspection!(s)
end
