include("fixtures/parameter_atlas.jl")

@testset "A125 sampled homology and exact event witnesses" begin
    V=TamerOp.Visualization;C=TamerOp.CoreModules
    @test TamerOp.parameter_atlas===V.parameter_atlas
    for field in (C.QQField(),C.F2(),C.F3())
        a=TamerOp.parameter_atlas(_a125_triangle();parameters=[-1,0,1,2],degree=1,field,witnesses=true)
        @test V.check_parameter_atlas(a).valid
        @test getproperty.(V.atlas_states(a),:dimension)==[0,0,1,0]
        @test V.atlas_summary(a).sampling===:sampled_path
        @test !V.atlas_summary(a).exhaustive
        born=V.atlas_transition(a;source=2,target=3)
        @test [r.index for r in born.entries]==[4,5,6]
        @test born.cokernel_dimension==1 && born.kernel_dimension==0
        @test DI.check_source_chain(only(born.born)).valid
        death=V.atlas_transition(a;source=3,target=4)
        @test death.defined && size(death.matrix)==(0,1)
        @test death.kernel_dimension==1 && death.cokernel_dimension==0
        w=only(death.killed)
        @test w.filling.dimension==2 && w.cycle.dimension==1
        @test a.object.complex.boundaries[2]*w.filling.coefficients==w.cycle.coefficients
        @test DI.check_source_chain(w.cycle).valid && DI.check_source_chain(w.filling).valid
        endpoint=V.atlas_transition(a;source=2,target=4)
        @test endpoint.kernel_dimension==endpoint.cokernel_dimension==0
        @test length(endpoint.entries)==4 # the intervening class is invisible to this endpoint map
        @test !V.atlas_transition(a;source=4,target=3).defined
        for event in (:summary,:entries,:kernel)
            spec=V.visual_spec(a;source=3,target=4,event)
            @test V.check_visual_spec(spec).valid
            @test all(p->p.axes==a.states[1].view.axes,spec.panels[1:4])
            @test spec.metadata.transition.matrix==death.matrix
        end
    end
    h0=TamerOp.parameter_atlas(_a125_triangle();parameters=[0,1,2],degree=0,witnesses=true)
    @test getproperty.(V.atlas_states(h0),:dimension)==[3,1,1]
    @test V.atlas_transition(h0;source=1,target=2).kernel_dimension==2
    @test V.check_parameter_atlas(h0).valid
    @test occursin("ParameterAtlas",sprint(show,h0))
    grid=TamerOp.parameter_atlas(_a125_triangle(bigraded=true);grid=([0,1,2],[0,1]),degree=1)
    @test V.atlas_summary(grid).grid_shape==(2,3)
    @test V.atlas_states(grid)[1].parameter==(0,1)
    incomparable=V.atlas_transition(grid;source=1,target=5)
    @test incomparable.relation===:incomparable && incomparable.matrix===nothing
    @test V.check_visual_spec(V.visual_spec(grid;source=1,target=5)).valid
end

@testset "A125 encoding order precedes labels and unavailable stalks" begin
    V=TamerOp.Visualization;PL=TamerOp.PLPolyhedra;Q=Rational{BigInt}
    region=PL.HPoly(2,Q[-1 0;0 -1;1 0;0 1],Q[0,0,2,2],nothing,falses(4),zero(Q))
    pi=PL.PLEncodingMap(2,[BitVector()],[BitVector()],[region],[(1,1)])
    P=TamerOp.FiniteFringe.FinitePoset(trues(1,1))
    M=TamerOp.Modules.PModule{Q}(P,[1],Dict{Tuple{Int,Int},Matrix{Q}}();field=TamerOp.CoreModules.QQField())
    enc=TamerOp.Results.EncodingResult(P,M,TamerOp.EncodingCore.compile_encoding(P,pi))
    a=TamerOp.parameter_atlas(enc;parameters=[(0,1),(1,0),(2,2),(3,3)],witnesses=true)
    @test getproperty.(V.atlas_states(a),:label)==[1,1,1,0]
    @test V.atlas_states(a)[4].dimension===nothing
    @test V.atlas_transition(a;source=1,target=2).relation===:incomparable
    @test V.atlas_transition(a;source=1,target=2).matrix===nothing
    @test V.atlas_transition(a;source=1,target=3).matrix==ones(Q,1,1)
    @test V.atlas_transition(a;source=1,target=4).relation===:outside
    @test !V.atlas_transition(a;source=4,target=4).defined
    @test V.check_parameter_atlas(a).valid
    @test V.check_visual_spec(V.visual_spec(a;source=1,target=2)).valid
    @test_throws ArgumentError TamerOp.parameter_atlas(M;parameters=[1],field=TamerOp.CoreModules.F2())
end

@testset "A125 module maps and budget/selection contracts" begin
    V=TamerOp.Visualization;C=TamerOp.CoreModules
    for field in (C.QQField(),C.F2())
        a=TamerOp.parameter_atlas(_a125_zero_composite(field);parameters=1:3,witnesses=true)
        @test V.atlas_transition(a;source=1,target=2).rank==1
        @test V.atlas_transition(a;source=1,target=2).cokernel==reshape(TamerOp.CoreModules.coeff_type(field)[0,1],2,1)
        @test V.atlas_transition(a;source=2,target=3).rank==1
        t=V.atlas_transition(a;source=1,target=3)
        @test t.rank==0 && t.defined && all(iszero,t.matrix)
        @test t.kernel_dimension==t.cokernel_dimension==1
        @test V.check_parameter_atlas(a).valid
        @test V.check_visual_spec(V.visual_spec(a;source=1,target=3,event=:kernel)).valid
    end
    x=_a125_triangle()
    for kw in ((;parameters=[]),(;parameters=[0,0]),(;parameters=[NaN]),(;parameters=[0],grid=([0],[0])),
        (;parameters=[0,1],max_states=1),(;parameters=[0,1],max_transitions=1),
        (;parameters=[0],max_cells=6),(;parameters=[0],max_matrix_entries=1),
        (;parameters=[0],degree=-1),(;parameters=[0],witnesses=1),(;parameters=[0],field=C.RealField(Float64)))
        @test_throws ArgumentError TamerOp.parameter_atlas(x;kw...)
    end
    a=TamerOp.parameter_atlas(x;parameters=[0,1,2],degree=1,witnesses=true)
    s=V.inspection_session(a;source=2,target=3)
    @test V.check_inspection_session(s).valid
    @test V.check_visual_spec(V.visual_spec(s)).valid
    V.select_inspection!(s;event=:kernel)
    @test V.inspection_selection(s).event===:kernel
    V.select_inspection!(s;witness=1)
    @test V.inspection_selection(s).event===:kernel
    old=V.inspection_selection(s)
    @test_throws ArgumentError V.select_inspection!(s;event=:kernel,witness=2)
    @test V.inspection_selection(s)==old
    V.select_inspection!(s;source=3,target=2)
    @test !V.inspection_summary(s).transition.defined
    @test !V.check_inspection_selection(s;event=:entries).valid
    V.reset_inspection!(s)
    @test V.inspection_selection(s).source==2
    V.close_inspection!(s)
    @test V.check_inspection_session(s).valid
    @test V.inspection_snapshot(s).kind===:parameter_atlas
    @test_throws ArgumentError V.select_inspection!(s;source=1)
end

@testset "A125 multicritical superlevel order and retained source identities" begin
    V=TamerOp.Visualization
    G=DT.MultiCriticalGradedComplex([[1,2],[3]],[sparse(reshape([-1,1],2,1))],
        [[(0,4)],[(0,4)],[(1,3),(3,1)]])
    x=DI.filtered_complex(G;geometry=[[(0,0)],[(1,0)],[(0,0),(1,0)]],orientation=(1,-1))
    a=TamerOp.parameter_atlas(x;parameters=[(0,4),(1,3+1//100),(1,3),(3,1)],witnesses=true)
    @test getproperty.(V.atlas_states(a),:dimension)==[2,2,1,1]
    t=V.atlas_transition(a;source=2,target=3)
    @test length(t.entries)==1 && only(t.entries).index==3
    @test t.kernel_dimension==1
    @test V.atlas_transition(a;source=3,target=4).rank==1
    @test V.check_parameter_atlas(a).valid
    @test V.atlas_summary(a).sampling===:sampled_path
    # The drawing is a skeleton; the actual 3-cell kills its boundary sphere.
    cube=DI.filtered_complex(zeros(2,2,2);input=:vertices)
    counts=DT.cell_counts(cube.complex)
    grades=[((i==sum(counts) ? 1 : 0),) for i in 1:sum(counts)]
    G3=DT.GradedComplex(DT.cells(cube.complex),cube.complex.boundaries,grades)
    shell=DI.filtered_complex(G3;geometry=cube.geometry)
    sphere=TamerOp.parameter_atlas(shell;parameters=[0,1],degree=2,witnesses=true)
    filling=only(V.atlas_transition(sphere;source=1,target=2).killed).filling
    @test filling.dimension==3 && DI.check_source_chain(filling).valid
    @test getproperty.(V.atlas_states(sphere),:dimension)==[1,0]
    @test V.check_visual_spec(V.visual_spec(sphere;target=2,event=:kernel)).valid
end

@testset "A125 native renderers, linked viewers and saved atlas exports" begin
    V=TamerOp.Visualization
    cairo=Base.get_extension(TamerOp,:TamerOpCairoMakieExt)
    wgl=Base.get_extension(TamerOp,:TamerOpWGLMakieExt)
    a=TamerOp.parameter_atlas(_a125_triangle();parameters=[0,1,2],degree=1,witnesses=true)
    s=V.inspection_session(a;source=2,target=3,event=:kernel)
    frame=V.visual_state(s)
    mktempdir() do directory
        path=joinpath(directory,"atlas.json")
        V.save_visual_session(path,(;contact=V.visual_state(V.visual_spec(a)),filling=frame);selected=:filling)
        saved=V.load_visual_session(path)
        snapshot=V.inspection_snapshot(saved)
        @test snapshot.metadata.transition.matrix==V.atlas_transition(a;source=2,target=3).matrix
        @test only(snapshot.metadata.transition.killed).cycle.coefficients==DI.source_chain_data(only(V.atlas_transition(a;source=2,target=3).killed).cycle).coefficients
        @test V.check_visual_session(saved).valid
        @test_throws ArgumentError V.inspection_session(saved)
        for (backend,extension) in ((:cairomakie,cairo),(:wglmakie,wgl))
            if extension===nothing
                @test_skip false
            else
                fig=V.visualize(a;backend,target=3,source=2,event=:kernel)
                M=backend===:cairomakie ? cairo.CairoMakie.Makie : wgl.WGLMakie.Makie
                axes=filter(ax->ax isa M.Axis,fig.content)
                @test length(axes)==5 # three states, source cycle, target filling
                expected=(a.states[1].view.axes.xlimits,a.states[1].view.axes.ylimits)
                @test all(ax->ax.limits[]==expected,axes)
                @test V.check_visual_spec(V.inspection_snapshot(s)).valid
                empty!(fig)
            end
        end
        if cairo!==nothing
            for extension in ("png","svg","pdf")
                output=joinpath(directory,"atlas."*extension)
                V.save_visual(output,saved)
                @test filesize(output)>400
            end
            html=get(ENV,"TAMEROP_ATLAS_HTML",joinpath(directory,"atlas.html"))
            mkpath(dirname(html))
            V.save_visual_session(html,(;contact=V.visual_state(V.visual_spec(a)),filling=frame);
                title="A sampled triangle atlas",caption="A cycle and its filling explain the selected endpoint map.")
            @test occursin("A sampled triangle atlas",read(html,String))
            png=get(ENV,"TAMEROP_ATLAS_PNG","")
            isempty(png) || (mkpath(dirname(png));V.save_visual(png,saved))
        end
    end
    if wgl===nothing
        @test_skip false
    else
        app=V.visualize(s;backend=:wglmakie);app2=V.visualize(s;backend=:wglmakie)
        u=wgl._inspection_ui(app);v=wgl._inspection_ui(app2);fig=u.figure
        V.select_inspection!(s;source=1,target=2,event=:entries)
        @test u.controls.source.value[]==v.controls.source.value[]=="1"
        @test u.figure===fig
        @test isempty(u.error_text[])
        u.dispose();@test V.inspection_summary(s).listener_count==1
        V.select_inspection!(s;source=2,target=3,event=:kernel)
        @test v.controls.source.value[]=="2"
        V.close_inspection!(s)
        @test v.closed[] && isempty(v.observers) && isempty(v.pointer_observers)
        @test V.check_inspection_session(s).valid
    end
end
