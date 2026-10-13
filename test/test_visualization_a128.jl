include("fixtures/spatial_pairs.jl")

@testset "A128 source closure and exact relative oracles" begin
    O=TamerOp.OrdinaryPersistence;V=TamerOp.Visualization;C=TamerOp.CoreModules
    @test TamerOp.spatial_pair===O.spatial_pair
    for F in (C.F2(),C.F3(),C.Fp(5)), superlevel in (false,true)
        x=_a128_disk(;superlevel);scale=superlevel ? 0 : 2
        ring=O.spatial_pair(x;source_ids=[:ab,:ac,:bc],field=F)
        @test ring.selected==collect(1:6) && ring.added==[1,2,3]
        @test O.check_spatial_pair(ring).valid
        @test O.check_filtered_pair(O.filtered_pair(ring)).valid
        h=O.relative_homology(ring;scale)
        @test O.dimensions(h;part=:ambient)==[1,0,0]
        @test O.dimensions(h;part=:subcomplex)==[1,1,0]
        @test O.dimensions(h)==[0,0,1]
        delta=O.relative_maps(h;dim=2).connecting
        @test size(delta)==(1,1) && !iszero(only(delta))
        @test O.essential_births(O.persistence_diagram(ring);dim=2)==[scale]
        empty=O.spatial_pair(x;field=F)
        full=O.spatial_pair(x;cells=[7],field=F)
        @test full.added==collect(1:6) && full.selected==collect(1:7)
        @test O.dimensions(O.relative_homology(full;scale))==[0,0,0]
        @test O.dimensions(O.relative_homology(empty;scale))==[1,0,0]
        @test O.spatial_pair(x;cells=[1,2],rule=:induced,field=F).selected==[1,2,4]
        @test O.spatial_pair(x;cells=[1,2,3],rule=:induced,field=F).selected==collect(1:7)
        for part in (:ambient,:subcomplex,:relative)
            diagram=O.persistence_diagram(ring;part)
            @test O.check_persistence_diagram(diagram).valid
            for t in (superlevel ? [3,2,1,0,-1] : [-1,0,1,2,3])
                ht=O.relative_homology(ring;scale=t)
                @test [V._relative_count(diagram,d,t) for d in 0:2]==O.dimensions(ht;part)
            end
        end
        view=V.visual_spec(ring;scale,degree=2,details=:representative)
        @test V.check_visual_spec(view).valid
        @test view.metadata.counts==(ambient=0,subcomplex=0,relative=1)
        @test view.metadata.representative.boundary==C.coeff_type(F).([1,-1,1])
        @test view.metadata.representative.connecting_coordinates==vec(delta)
        @test view.metadata.representative.boundary_cell_indices==[4,5,6]
    end
end

@testset "A128 incidence cancellation, entry grades and brush boundaries" begin
    O=TamerOp.OrdinaryPersistence;V=TamerOp.Visualization;C=TamerOp.CoreModules
    x=_a128_disk()
    brush=(;dims=(1,2),lower=(-1,0),upper=(5,0))
    @test O.spatial_pair(x;brush,rule=:induced).seeds==[1,2]
    @test O.spatial_pair(x;brush,rule=:induced).selected==[1,2,4]
    @test O.spatial_pair(x;brush,rule=:face_closure).seeds==[1,2,4]
    @test O.spatial_pair(x;brush=(;dims=(2,1),lower=(0,-1),upper=(0,5)),rule=:induced).selected==[1,2,4]
    @test O.spatial_pair(x;brush=(;dims=(1,2),lower=(1,1),upper=(1,1))).seeds==[7]
    @test isempty(O.spatial_pair(x;brush=(;dims=(1,2),lower=(100,100),upper=(101,101))).selected)
    delayed=O.spatial_pair(x;cells=[4,5,6],grades=Dict(i=>3 for i in 1:6))
    @test all(r->r.subcomplex_grade==3,filter(r->r.selected,O.spatial_cells(delayed)))
    @test O.dimensions(O.relative_homology(delayed;scale=2))==[1,0,0]
    @test O.dimensions(O.relative_homology(delayed;scale=3))==[0,0,1]
    torus=DI.filtered_complex(zeros(Int,1,1);input=:vertices,periodic=true)
    @test all(iszero,torus.complex.boundaries)
    @test sort.(DI.face_incidence(torus))==[Int[],[1],[1],[2,3]]
    @test O.spatial_pair(torus;cells=[4]).selected==[1,2,3,4]
    @test O.spatial_pair(torus;cells=[4],rule=:chain_closure).selected==[4]
    @test O.spatial_pair(torus;cells=[1],rule=:induced).selected==[1,2,3,4]
    @test_throws ArgumentError O.spatial_pair(torus;cells=[4],grades=Dict(1=>2))
    G=DT.GradedComplex([[1],[2],[3]],[spzeros(Int,1,1),sparse(reshape([2],1,1))],[(0,),(0,),(0,)])
    cw=DI.filtered_complex(G;geometry=[[(0,0)],[(0,0),(1,0)],[(0,0),(1,0),(0,1)]],faces=[Int[],[1],[2]])
    @test O.spatial_pair(cw;cells=[3],field=C.F2()).selected==[1,2,3]
    @test O.spatial_pair(cw;cells=[3],rule=:chain_closure,field=C.F2()).selected==[3]
    @test O.spatial_pair(cw;cells=[3],rule=:chain_closure,field=C.F3()).selected==[2,3]
    matrix=O.spatial_pair(G;cells=[3],rule=:chain_closure)
    @test O.spatial_pair_summary(matrix).geometry===:not_asserted
    @test V.check_visual_spec(V.visual_spec(matrix)).valid
    @test_throws ArgumentError O.spatial_pair(G;cells=[3])
    missing=DI.filtered_complex(x.complex;geometry=x.geometry)
    @test_throws ArgumentError O.spatial_pair(missing;cells=[7])
    for kw in ((;cells=[8]),(;cells=[true]),(;cells=[4,4]),(;cells=[4],rule=:induced),
        (;cells=[4],source_ids=[:a]),(;source_ids=[:unknown]),(;source_ids=:a),(;rule=:clip),
        (;brush=(;dims=(1,1),lower=(0,0),upper=(1,1))),
        (;brush=(;dims=(1,2),lower=(1,1),upper=(0,0))),
        (;cells=[7],grades=Dict(1=>10)),(;field=C.QQField()),(;order=:superlevel))
        @test_throws ArgumentError O.spatial_pair(x;kw...)
    end
    @test_throws ArgumentError DI.filtered_complex(x.complex;geometry=x.geometry,faces=[Int[] for _ in 1:7])
    # Exact integer centroid arithmetic does not round the brush classification.
    large=big(2)^60
    Gp=DT.GradedComplex([[1]],SparseMatrixCSC{Int,Int}[],[(0,)])
    xp=DI.filtered_complex(Gp;geometry=[[(large+1,0)]],faces=[Int[]])
    @test isempty(O.spatial_pair(xp;rule=:induced,brush=(;dims=(1,2),lower=(large,0),upper=(large,0))).selected)
end

@testset "A128 atomic queries, reuse, validation and snapshots" begin
    O=TamerOp.OrdinaryPersistence;V=TamerOp.Visualization
    p=O.spatial_pair(_a128_disk();cells=[4,5,6]);s=V.inspection_session(p;degree=2)
    cached=s.diagrams
    V.select_inspection!(s;scale=1)
    @test s.diagrams===cached
    @test V.inspection_summary(s).counts.relative==0
    before=V.inspection_selection(s)
    @test_throws ArgumentError V.select_inspection!(s;details=:representative)
    @test V.inspection_selection(s)==before
    V.select_inspection!(s;scale=2,details=:representative)
    @test V.check_inspection_session(s).valid && V.check_visual_spec(V.visual_spec(s)).valid
    @test !V.check_inspection_selection(s;witness=2).valid
    V.select_inspection!(s;details=:summary,entry=3)
    @test V.inspection_summary(s).counts.relative==0
    V.select_inspection!(s;scale=3)
    @test V.inspection_summary(s).counts.relative==1
    V.reset_inspection!(s)
    @test V.inspection_selection(s).scale==2
    V.select_inspection!(s;cells=Int[])
    @test isempty(V.inspection_summary(s).selected)
    V.select_inspection!(s;brush=(;dims=(1,2),lower=(-1,0),upper=(5,0)),rule=:induced)
    @test V.inspection_summary(s).selected==[1,2,4]
    snap=V.visual_state(s)
    @test snap!==nothing
    V.close_inspection!(s)
    @test V.check_inspection_session(s).valid
    @test V.inspection_snapshot(s).kind===:relative_explorer
    @test_throws ArgumentError V.select_inspection!(s;scale=0)
    @test_throws ArgumentError V.inspection_session(p;max_algebra_cells=6)
    @test_throws ArgumentError V.visual_spec(p;details=:maps,max_matrix_entries=1)
end

@testset "A128 native figures, linked controls and saved views" begin
    V=TamerOp.Visualization;O=TamerOp.OrdinaryPersistence
    p=O.spatial_pair(_a128_disk();cells=[4,5,6])
    s=V.inspection_session(p;degree=2,details=:representative)
    cairo=Base.get_extension(TamerOp,:TamerOpCairoMakieExt)
    wgl=Base.get_extension(TamerOp,:TamerOpWGLMakieExt)
    for (backend,extension) in ((:cairomakie,cairo),(:wglmakie,wgl))
        if extension===nothing
            @test_skip false
        else
            fig=V.visualize(p;backend,degree=2)
            M=backend===:cairomakie ? cairo.CairoMakie.Makie : wgl.WGLMakie.Makie
            axes=filter(ax->ax isa M.Axis,fig.content)
            @test length(axes)==5
            @test axes[1].limits[]==axes[2].limits[]
            @test axes[1].limits[][1][1]<-1 && axes[1].limits[][1][2]>5
            @test axes[1].limits[][2][1]<0 && axes[1].limits[][2][2]>3
            @test !isempty(axes[1].scene.plots)
            empty!(fig)
        end
    end
    mktempdir() do directory
        path=joinpath(directory,"relative.json")
        V.save_visual_session(path,V.visual_state(s))
        saved=V.load_visual_session(path)
        @test V.check_visual_session(saved).valid
        @test V.inspection_snapshot(saved).metadata.representative.connecting_coordinates==V.inspection_snapshot(s).metadata.representative.connecting_coordinates
        @test_throws ArgumentError V.inspection_session(saved)
        if cairo!==nothing
            for ext in ("svg","png","pdf")
                path=joinpath(directory,"pair.$ext");V.save_visual(path,saved)
                @test filesize(path)>500
            end
            html=get(ENV,"TAMEROP_RELATIVE_HTML",joinpath(directory,"relative.html"))
            mkpath(dirname(html))
            V.save_visual_session(html,(;ring=V.visual_state(s),empty=V.visual_state(V.visual_spec(O.spatial_pair(_a128_disk());degree=2)));
                title="A disk relative to its boundary",caption="Precomputed answers; geometric selection and new relative queries require live Julia.")
            @test occursin("A disk relative to its boundary",read(html,String))
        end
    end
    if wgl===nothing
        @test_skip false
    else
        a=V.visualize(s;backend=:wglmakie);b=V.visualize(s;backend=:wglmakie)
        ua=wgl._inspection_ui(a);ub=wgl._inspection_ui(b)
        @test V.inspection_summary(s).listener_count==2
        V.select_inspection!(s;cells=Int[],details=:summary)
        @test occursin("selected",ua.status_text[]) && ua.status_text[]==ub.status_text[]
        @test V.check_inspection_session(s).valid
        V.close_inspection!(s)
        @test ua.closed[] && ub.closed[]
        @test isempty(ua.observers) && isempty(ua.pointer_observers)
    end
    V.close_inspection!(s)
end


@testset "A128 queued pair ownership and automatic retained incidence" begin
    O=TamerOp.OrdinaryPersistence;V=TamerOp.Visualization
    s=V.inspection_session(O.spatial_pair(_a128_disk();cells=[4,5,6]);degree=2)
    V._request_inspection!(s,:test;cells=[7])
    while V.inspection_summary(s).requests.pending || V.inspection_summary(s).requests.running
        yield();sleep(0.01)
    end
    @test s.object.selected==collect(1:7)
    @test isempty(O.essential_births(s.diagrams.relative;dim=2))
    @test O.dimensions(O.relative_homology(s.object;scale=2))==[0,0,0]
    @test V.check_inspection_session(s).valid
    V.close_inspection!(s)
    for x in (DI.filtered_complex(DT.PointCloud([0. 0.;2. 0.;0. 3.]),DI.RipsFiltration(max_dim=2)),
              DI.filtered_complex(zeros(Int,2,3);input=:vertices),
              DI.filtered_complex(zeros(Int,1);input=:vertices,periodic=true))
        @test DI.face_incidence(x)!==nothing
        @test DI.check_filtered_complex(x).valid
        p=O.spatial_pair(x;cells=findall(==(0),x.complex.cell_dims),rule=:induced)
        @test p.selected==collect(eachindex(x.geometry))
        @test O.check_spatial_pair(p).valid
    end
    # Two large Int endpoints have an exact midpoint without machine overflow.
    G=DT.GradedComplex([[1,2],[3]],[sparse(reshape([-1,1],2,1))],[(0,),(0,),(1,)])
    a=typemax(Int)-4;b=typemax(Int)-2
    x=DI.filtered_complex(G;geometry=[[(a,0)],[(b,0)],[(a,0),(b,0)]],faces=[Int[],Int[],[1,2]])
    p=O.spatial_pair(x;brush=(;dims=(1,2),lower=(a+1,0),upper=(a+1,0)))
    @test p.seeds==[3] && p.added==[1,2]
end

@testset "A128 three-dimensional and projected brush semantics" begin
    O=TamerOp.OrdinaryPersistence;V=TamerOp.Visualization
    x=DI.filtered_complex(zeros(Int,2,3,4);input=:vertices)
    p=O.spatial_pair(x;rule=:induced,brush=(;dims=(1,2,3),lower=(1,1,1),upper=(2,2,1)))
    @test length(p.seeds)==4 && length(p.selected)==9
    @test O.spatial_pair_summary(p).subcomplex_cells==(4,4,1,0)
    extruded=O.spatial_pair(x;rule=:induced,brush=(;dims=(1,2),lower=(1,1),upper=(2,2)))
    @test length(extruded.selected)==63
    @test O.spatial_pair_summary(extruded).subcomplex_cells==(16,28,16,3)
    snapshot=V.visual_spec(p;view_dims=(3,1,2))
    @test V.check_visual_spec(snapshot).valid
    @test snapshot.panels[1].axes.xlabel=="source x3"
    @test snapshot.panels[1].axes.zlabel=="source x2"
    @test V.visual_spec(extruded;view_dims=(2,1)).panels[1].metadata.projection
    for name in (:TamerOpCairoMakieExt,:TamerOpWGLMakieExt)
        extension=Base.get_extension(TamerOp,name)
        if extension===nothing
            @test_skip false
        else
            M=name===:TamerOpCairoMakieExt ? extension.CairoMakie.Makie : extension.WGLMakie.Makie
            figure=V.visualize(p;backend=name===:TamerOpCairoMakieExt ? :cairomakie : :wglmakie,view_dims=(3,1,2))
            axes=filter(ax->ax isa M.Axis3,figure.content)
            @test length(axes)==2
            @test axes[1].limits[]==axes[2].limits[]
            @test axes[1].limits[][1][2]>4
            empty!(figure)
        end
    end
end


@testset "A128 display budgets retain selected source priority" begin
    O=TamerOp.OrdinaryPersistence;V=TamerOp.Visualization
    p=O.spatial_pair(_a128_disk();cells=[7])
    view=V.visual_spec(p;max_cells=1)
    @test view.panels[1].metadata.displayed_indices==[7]
    @test view.metadata.selected==collect(1:7)
    @test length(view.metadata.table)==7
    @test V.check_visual_spec(view).valid
    panel=V._relative_matrix(zeros(Int,0,20),"Zero map")
    @test panel.metadata.matrix_size==(0,20) && panel.metadata.truncated
end

@testset "A128 witness captions describe coefficients and support" begin
    O=TamerOp.OrdinaryPersistence;V=TamerOp.Visualization
    view=V.visual_spec(O.spatial_pair(_a128_disk();cells=[4,5,6]);degree=2,details=:representative)
    text=join(only(last(view.panels).layers).labels,"\n")
    @test occursin("cell 7: 1",text)
    @test !occursin("Dict{",text)
    panel=view.panels[end-1]
    @test only(panel.legend.entries).label=="witness support"
    @test V.check_visual_spec(view).valid
end
