include("fixtures/density_clustering.jl")
@testset "A124-F1 independent radius graphs and multicritical faces" begin
    D=TamerOp.DataIngestion;V=TamerOp.Visualization;DT=TamerOp.DataTypes
    for X in ([0. 0.;1. 0.;2. 0.], [0. 0.;0. 0.;1. 0.;1. 1.], [0. 0.], [0. 0.;1. 0.;0. 1.;1. 1.]),cap in (0.,1.,2.)
        n=size(X,1);f=TamerOp.RadiusDegreeRipsFiltration(max_dim=2,radius=cap)
        x=TamerOp.filtered_complex(TamerOp.PointCloud(X),f)
        @test D.check_filtered_complex(x).valid
        @test x.orientation==(1,-1)
        @test D.filtration_arity(f)==2
        @test D.filtration_kind(D.to_filtration(D._filtration_spec(f)))==:radius_degree_rips
        distances=[sqrt(sum((X[i,:]-X[j,:]).^2)) for i in 1:n,j in 1:n]
        for r in unique(vcat(0.,cap,filter(<=(cap),vec(distances)))),k in 0:n
            degrees=[count(j->j!=i && distances[i,j]<=r,1:n) for i in 1:n]
            for id in eachindex(x.sources)
                vertices=x.sources[id]
                expected=all(v->degrees[v]>=k,vertices) && all(distances[u,v]<=r for u in vertices,v in vertices)
                @test D._source_active(x,id,(r,k))==expected
            end
        end
        @test V._filtration_controls(x,(0,0))[2].label=="Minimum neighbors at this diameter k"
    end
    cloud=TamerOp.PointCloud([0. 0.;1. 0.;2. 0.])
    @test_throws ArgumentError TamerOp.RadiusDegreeRipsFiltration(radius=-1)
    @test_throws ArgumentError TamerOp.RadiusDegreeRipsFiltration(max_dim=true)
    @test_throws ArgumentError TamerOp.RadiusDegreeRipsFiltration(construction=TamerOp.ConstructionOptions(sparsify=:knn))
    @test_throws ArgumentError TamerOp.encode(cloud,TamerOp.RadiusDegreeRipsFiltration();stage=:simplex_tree)
    G=TamerOp.encode(cloud,TamerOp.RadiusDegreeRipsFiltration(radius=2.);stage=:graded_complex)
    @test G!==nothing
    @test_throws ArgumentError TamerOp.encode(cloud,TamerOp.RadiusDegreeRipsFiltration();pipeline=TamerOp.PipelineOptions(orientation=(1,1)),stage=:graded_complex)
    raw=TamerOp.FiltrationSpec(kind=:radius_degree_rips,max_dim=1,radius=2.)
    @test TamerOp.encode(cloud,raw;stage=:graded_complex) isa DT.MultiCriticalGradedComplex
    bounded=TamerOp.RadiusDegreeRipsFiltration(radius=2.,construction=TamerOp.ConstructionOptions(sparsify=:radius))
    @test TamerOp.encode(cloud,bounded;stage=:graded_complex).grades==G.grades
    fractional=TamerOp.filtered_complex(TamerOp.PointCloud([0. 0.;0.1 0.]),TamerOp.RadiusDegreeRipsFiltration(radius=1//10))
    @test DT.cell_counts(fractional.complex)==[2,1]
    @test D._source_active(fractional,3,(0.1,1))
    x=TamerOp.filtered_complex(cloud,TamerOp.RadiusDegreeRipsFiltration(radius=2.))
    for field in (TamerOp.CoreModules.F2(),TamerOp.CoreModules.F3())
        encoded=TamerOp.encode(cloud,TamerOp.RadiusDegreeRipsFiltration(radius=2.);degree=0,field)
        module_=TamerOp.Workflow.pmodule(encoded)
        for radius in 0:2,k in 0:2
            vertex=TamerOp.EncodingCore.locate(TamerOp.Results.encoding_map(encoded),(radius,k))
            actual=vertex==0 ? 0 : TamerOp.Modules.dim_at(module_,vertex)
            @test actual==(radius==0 ? (k==0 ? 3 : 0) : 1)
        end
        atlas=TamerOp.parameter_atlas(x;parameters=[(0.,0),(1.,0),(1.,2),(2.,0)],degree=0,field)
        @test getproperty.(V.atlas_states(atlas),:dimension)==[3,1,1,1]
        @test V.atlas_transition(atlas;source=1,target=2).kernel_dimension==2
        @test V.atlas_transition(atlas;source=3,target=2).rank==1
        @test V.atlas_transition(atlas;source=3,target=4).rank==1
        @test !V.atlas_transition(atlas;source=1,target=3).defined
    end
end
@testset "A129 reproducible clustering and independent components" begin
    O=TamerOp.OrdinaryPersistence;a=O.clustering_analysis(_a129_cloud();max_diameter=5.)
    @test O.density_clustering===TamerOp.density_clustering
    r=TamerOp.density_clustering(a;diameter=0.3,neighbors=1)
    @test TamerOp.cluster_labels(r)==[1,1,1,2,2,2,0]
    @test r.noise==(7,)
    @test TamerOp.density_clustering(_a129_cloud();max_diameter=5.,diameter=0.3,neighbors=1).labels==r.labels
    @test TamerOp.density_clustering(a;r.parameters...).labels==r.labels
    @test length(r.clusters)==2
    @test TamerOp.cluster_labels(TamerOp.density_clustering(a;diameter=0.,neighbors=1))==zeros(Int,7)
    @test all(iszero,TamerOp.cluster_labels(TamerOp.density_clustering(a;diameter=0.3,neighbors=1,min_size=4)))
    @test all(iszero,TamerOp.cluster_labels(TamerOp.density_clustering(a;diameter=0.3,neighbors=1,min_persistence=6)))
    labels=TamerOp.cluster_labels(r);labels[1]=99;@test r.labels[1]==1
    for (i,t) in enumerate(a.diameters),k in 0:a.npoints-1
        X=_a129_cloud().coords;n=size(X,1);dist=[sqrt(sum((X[u,:]-X[v,:]).^2)) for u in 1:n,v in 1:n]
        active=[count(v->v!=u && dist[u,v]<=t,1:n)>=k for u in 1:n]
        parts=Vector{Int}[];seen=falses(n)
        for u in 1:n
            (!active[u] || seen[u]) && continue
            todo=[u];seen[u]=true;part=Int[]
            while !isempty(todo)
                v=pop!(todo);push!(part,v)
                for w in 1:n
                    if active[w] && !seen[w] && dist[v,w]<=t;push!(todo,w);seen[w]=true;end
                end
            end
            push!(parts,sort(part))
        end
        @test a.counts[i,k+1]==length(parts)
        result=TamerOp.density_clustering(a;diameter=t,neighbors=k)
        @test [collect(c.source_ids) for c in result.clusters]==parts
    end
    dup=O.clustering_analysis(TamerOp.PointCloud([0. 0.;0. 0.]);max_diameter=0.)
    @test TamerOp.cluster_labels(TamerOp.density_clustering(dup;diameter=0,neighbors=1))==[1,1]
    @test isempty(O.clustering_gaps(dup;neighbors=1))
    # At k=0 the three elder bars have observed lifetimes 5, 1 and 3.
    line=O.clustering_analysis(TamerOp.PointCloud([0. 0.;1. 0.;4. 0.]);max_diameter=5.)
    @test TamerOp.cluster_labels(TamerOp.density_clustering(line;diameter=0,neighbors=0,min_persistence=2))==[1,0,2]
    @test TamerOp.cluster_labels(TamerOp.density_clustering(line;diameter=1,neighbors=0,min_persistence=3))==[1,1,2]
    @test TamerOp.cluster_labels(TamerOp.density_clustering(line;diameter=1,neighbors=0,min_persistence=3.1))==[1,1,0]
    @test getproperty.(O.clustering_gaps(line;neighbors=0),:threshold)==[2,4]
    survivor=only(TamerOp.cluster_memberships(TamerOp.density_clustering(line;diameter=5,neighbors=0)))
    @test survivor.censored && survivor.observed_lifetime==5
    @test_throws ArgumentError TamerOp.density_clustering(a;diameter=6,neighbors=1)
    @test_throws ArgumentError TamerOp.density_clustering(a;diameter=1,neighbors=7)
    @test_throws ArgumentError O.clustering_analysis(_a129_cloud();max_diameter=5.,max_grid_entries=2)
end
@testset "A129 native and linked clustering contract" begin
    V=TamerOp.Visualization;s=_a129_session()
    @test V.check_inspection_session(s).valid
    @test V.check_visual_spec(V.visual_spec(s)).valid
    plane=V.inspection_snapshot(s).panels[2]
    heat=only(filter(l->l isa V.HeatmapLayer,plane.layers))
    @test heat.x[1]==0 && heat.x[end]==5 && heat.y==collect(-0.5:1:6.5)
    @test heat.values[1,1]==7 && all(iszero,heat.values[2:end,1])
    @test V.clustering_result(s).labels==(1,1,1,2,2,2,0)
    @test !V.check_inspection_selection(s;neighbors=-1).valid
    old=V.inspection_selection(s);@test_throws ArgumentError V.select_inspection!(s;neighbors=-1)
    @test V.inspection_selection(s)==old
    V.select_inspection!(s;neighbors=0);@test last(V.clustering_result(s).labels)==3
    V.reset_inspection!(s);@test V.clustering_result(s).labels==(1,1,1,2,2,2,0)
    owner=Ref(nothing)
    V._request_inspection!(s,owner;neighbors=0);V._request_inspection!(s,owner;neighbors=1)
    wait(s.requests.task);@test s.state.neighbors==1
    @test s.requests.committed==1
    for (backend,extname) in ((:cairomakie,:TamerOpCairoMakieExt),(:wglmakie,:TamerOpWGLMakieExt))
        ext=Base.get_extension(TamerOp,extname)
        if ext===nothing;@test_skip false;continue;end
        fig=V.render(V.inspection_snapshot(s);backend)
        M=backend===:cairomakie ? ext.CairoMakie.Makie : ext.WGLMakie.Makie
        axes=filter(a->a isa M.Axis,fig.content);@test length(axes)==3
        points=[Tuple(p) for plot in axes[1].scene.plots if plot isa M.Scatter for p in plot[1][]]
        @test length(points)==size(_a129_cloud().coords,1)
        @test all(p->any(q->all(isapprox.(q,p;atol=1e-6,rtol=0)),points),Tuple.(eachrow(_a129_cloud().coords)))
        empty!(fig)
        if backend===:wglmakie
            ui=ext._inspection_ui(V.visualize(s;backend))
            original_figure=ui.figure[]
            ui.controls.neighbors.value[]="0";ui.controls.apply.value[]=true;wait(s.requests.task)
            @test s.state.neighbors==0
            ui.controls.neighbors.value[]="-1";ui.controls.apply.value[]=true;wait(s.requests.task)
            @test occursin("neighbors",ui.error_text[])
            @test s.state.neighbors==0
            V.reset_inspection!(s)
            @test ui.figure[]===original_figure && ui.rebuild_count[]==1
            @test ui.update_count[]>=2
            ui.dispose();@test isempty(ui.observers) && isempty(ui.pointer_observers)
        end
    end
    if Base.get_extension(TamerOp,:TamerOpCairoMakieExt)!==nothing
        mktempdir() do dir
            for ext in ("svg","png","pdf")
                path=joinpath(dir,"clustering.$ext");V.save_visual(path,V.inspection_snapshot(s);backend=:cairomakie);@test filesize(path)>1000
            end
            file=get(ENV,"TAMEROP_CLUSTERING_HTML",joinpath(dir,"clustering.html"));mkpath(dirname(file))
            original=V.visual_state(s);V.select_inspection!(s;neighbors=0);other=V.visual_state(s)
            V.save_visual_session(file,(;original,other));@test filesize(file)>1000
        end
    end
    V.close_inspection!(s);@test V.check_inspection_session(s).valid
    @test s.object===nothing
end
