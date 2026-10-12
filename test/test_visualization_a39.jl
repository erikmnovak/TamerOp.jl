@testset "A39 exact neighbor oracles and display-only budgets" begin
    V=TamerOp.Visualization;D=TamerOp.DataTypes
    A=[0.0 0.0 9.;0. 2. 9.;3. 0. 9.;7. 5. 9.]
    pc=D.PointCloud(A)
    for backend in (:bruteforce,)
        s=V.visual_spec(pc;kind=:knn_graph,k=1,nn_backend=backend,dims=(2,1),labels=["a","b","c","d"])
        @test Set(s.metadata.source_edges)==Set([(1,2),(1,3),(3,4)])
        @test s.metadata.construction===:reconstructed_metric_preview
        @test s.metadata.graph_backend===backend
        edges=only(l for l in s.layers if l isa V.SegmentLayer)
        @test Set(edges.segments)==Set([(0.,0.,2.,0.),(0.,0.,0.,3.),(0.,3.,5.,7.)])
        limited=V.visual_spec(pc;kind=:knn_graph,k=1,nn_backend=backend,dims=(2,1),max_points=2,max_edges=1,max_labels=1,labels=["a","b","c","d"])
        @test limited.metadata.point_ids==[1,4]
        @test limited.metadata.total_edges==3
        @test isempty(limited.metadata.source_edges)
        @test limited.metadata.label_ids==[1]
        @test only(l for l in limited.layers if l isa V.TextLayer).labels==["a"]
        @test limited.axes==s.axes
        s=V.visual_spec(pc;kind=:radius_graph,radius=3.,nn_backend=backend)
        @test Set(s.metadata.source_edges)==Set([(1,2),(1,3)])
        @test s.metadata.total_edges==2
    end
    duplicate=D.PointCloud([1. 2.;1. 2.;5. 8.])
    @test V.visual_spec(duplicate;kind=:radius_graph,radius=0.).metadata.source_edges==[(1,2)]
    for n in (0,1)
        cloud=D.PointCloud(zeros(n,2))
        @test isempty(V.visual_spec(cloud;kind=:knn_graph).metadata.source_edges)
        @test isempty(V.visual_spec(cloud;kind=:radius_graph).metadata.source_edges)
    end
    hidden=V.visual_spec(pc;kind=:points_3d,max_points=0)
    @test isempty(only(hidden.layers).points)
    density=V.visual_spec(pc;kind=:point_density,max_points=1)
    @test sum(only(l for l in density.layers if l isa V.HeatmapLayer).values)==4
    @test density.metadata.density_population===:all_source_points
    for kw in ((;max_points=-1),(;max_edges=true),(;max_labels=1.2),
        (;nn_backend=:approx),(;k=true),(;radius=Inf))
        kind=haskey(kw,:radius) ? :radius_graph : :knn_graph
        @test !V.check_visual_request(pc;kind,kw...).valid
        @test_throws ArgumentError V.visual_spec(pc;kind,kw...)
    end
    @test_throws ArgumentError V.visual_spec(pc;kind=:knn_graph,k=1,nn_backend=:bruteforce,max_distance_pairs=1)
    @test_throws ArgumentError V.visual_spec(pc;kind=:knn_graph,k=1,max_graph_edges=1)
    @test_throws ArgumentError V.visual_spec(pc;kind=:points_2d,cache=:bad)
    @test_throws ArgumentError V.visual_spec(D.PointCloud([NaN 0.]);kind=:points_2d)
end

@testset "A39 retained graph identities weights polylines and cache lifetime" begin
    V=TamerOp.Visualization;D=TamerOp.DataTypes
    coords=[0. 0. 2.;1. 4. 3.;2. 1. 7.;4. 8. 9.;9. 3. 11.]
    graph=D.GraphData(5,[(1,3),(3,5),(1,5),(2,4)];coords,weights=[0.,2.,10.,4.])
    sc=TamerOp.SessionCache()
    s=V.visual_spec(graph;kind=:weighted_graph,max_points=3,max_edges=2,max_labels=2,
        labels=string.(1:5),dims=(2,1),cache=sc)
    @test s.metadata.point_ids==[1,3,5]
    @test s.metadata.edge_ids==[1,2]
    @test s.metadata.source_edges==[(1,3),(3,5)]
    @test s.metadata.weight_range==(0.,10.)
    @test s.metadata.omitted_edges==2
    @test s.metadata.construction===:supplied_graph
    @test !s.metadata.geometry_reused
    @test V.visual_spec(graph;kind=:graph,dims=(2,1),cache=sc).metadata.geometry_reused
    coords[1,1]=20
    s2=V.visual_spec(graph;kind=:graph,dims=(2,1),cache=sc)
    @test !s2.metadata.geometry_reused
    @test first(only(l for l in s2.layers if l isa V.PointLayer).points)==(0.,20.)
    graph.edge_u[1]=2
    @test first(V.visual_spec(graph;cache=sc,kind=:graph).metadata.source_edges)==(2,3)
    @test !V.visual_spec(graph;kind=:graph,cache=nothing).metadata.geometry_reused
    @test V.visual_spec(D.GraphData(0,Tuple{Int,Int}[];coords=zeros(0,2),weights=Float64[]);kind=:weighted_graph).metadata.total_points==0
    @test_throws ArgumentError V.visual_spec(D.GraphData(2,[(1,3)]);kind=:graph)
    @test !V.check_visual_request(graph;kind=:graph,labels=["x"]).valid
    exact=D.GraphData(2,[(1,2)];coords=[2^53 0;2^53+1 1],T=Int)
    rounded=V.visual_spec(exact;kind=:graph)
    @test rounded.metadata.drawing_rounding
    @test occursin("rounded",rounded.subtitle)
    poly=D.EmbeddedPlanarGraph2D([[0.,0.],[1.,2.],[4.,0.]],[(1,2),(2,3)];
        polylines=[[(0.,0.),(.3,4.),(1.,2.)],[(1.,2.),(2.,-1.),(4.,0.)]])
    ps=V.visual_spec(poly;max_edges=1,max_labels=1,labels=["first","middle","last"])
    @test ps.metadata.edge_ids==[1]
    @test_throws ArgumentError V.visual_spec(poly;max_path_points=2)
    @test only(l for l in ps.layers if l isa V.PolylineLayer).paths==[[(0.,0.),(.3,4.),(1.,2.)]]
end

@testset "A39 shared ingestion graph cache and independent oracle" begin
    V=TamerOp.Visualization;D=TamerOp.DataTypes;DI=TamerOp.DataIngestion;C=TamerOp.CoreModules
    rng=MersenneTwister(39011)
    A=rand(rng,48,3);pc=D.PointCloud(A);sc=TamerOp.SessionCache()
    opts=TamerOp.ConstructionOptions(sparsify=:knn)
    filtration=TamerOp.RipsFiltration(max_dim=1,knn=3,nn_backend=:bruteforce,construction=opts)
    G=TamerOp.encode(pc,filtration;stage=:graded_complex,cache=sc)
    s=V.visual_spec(pc;kind=:knn_graph,k=3,nn_backend=:bruteforce,cache=sc,dims=(1,3))
    @test s.metadata.graph_reused
    # Reusing a preview graph must still record the actual backend in ingestion.
    trace=DI._IngestionExecutionTrace()
    Base.ScopedValues.with(DI._INGESTION_EXECUTION_TRACE=>trace) do
        DI._prepared_point_graph(A,:knn,3;backend=:bruteforce,cache=C._workflow_encoding_cache(sc))
    end
    @test trace.backends==[:knn_graph=>:bruteforce]
    expected=Set{Tuple{Int,Int}}()
    for i in axes(A,1)
        neighbors=sort([j for j in axes(A,1) if i!=j];by=j->(sum(abs2,A[i,:]-A[j,:]),j))[1:3]
        for j in neighbors;push!(expected,minmax(i,j));end
    end
    @test Set(s.metadata.source_edges)==expected
    B=D.boundary_map(G;dim=1)
    @test Set(Tuple(findall(!iszero,B[:,j])) for j in axes(B,2))==expected
    @test all(j->sort(nonzeros(B[:,j]))==[-1,1],axes(B,2))
    # Display axes, labels and budgets cannot change the mathematical graph.
    t=V.visual_spec(pc;kind=:knn_graph,k=3,nn_backend=:bruteforce,cache=sc,dims=(3,1),max_edges=4)
    @test t.metadata.graph_reused
    @test t.metadata.total_edges==length(expected)
    @test t.metadata.source_edges==sort!(collect(expected))[1:4]
    # Returned metadata owns its arrays; consumers cannot corrupt preparation.
    s.metadata.source_edges[1]=(1,1)
    @test Set(V.visual_spec(pc;kind=:knn_graph,k=3,cache=sc,nn_backend=:bruteforce).metadata.source_edges)==expected
    A[1,:].=[20.,30.,40.]
    @test !V.visual_spec(pc;kind=:knn_graph,k=3,cache=sc,nn_backend=:bruteforce).metadata.graph_reused
    @test !V.visual_spec(pc;kind=:radius_graph,radius=.3,cache=sc,nn_backend=:bruteforce).metadata.graph_reused
    for k in 1:8;V.visual_spec(pc;kind=:knn_graph,k,cache=sc,nn_backend=:bruteforce);end
    bucket=C._workflow_encoding_cache(sc)
    @test length(bucket.geometry[(:prepared_point_graphs,)].value)<=4
    @test length(bucket.geometry[(:source_preview_projections,)].value)<=4
    C._clear_encoding_cache!(bucket)
    @test !V.visual_spec(pc;kind=:knn_graph,k=3,cache=sc,nn_backend=:bruteforce).metadata.graph_reused
    C._clear_encoding_cache!(bucket)
    tasks=[Threads.@spawn V.visual_spec(pc;kind=:knn_graph,k=3,cache=sc,nn_backend=:bruteforce) for _ in 1:8]
    results=fetch.(tasks)
    @test count(x->!x.metadata.graph_reused,results)==1
    @test all(x->x.metadata.source_edges==results[1].metadata.source_edges,results)
end

@testset "A39 exact optional neighbor backend and bounded radius output" begin
    V=TamerOp.Visualization;D=TamerOp.DataTypes
    if Base.find_package("NearestNeighbors")===nothing
        @test_skip false
    else
        @eval import NearestNeighbors
        rng=MersenneTwister(39012);pc=D.PointCloud(rand(rng,100,4))
        for kind in (:knn_graph,:radius_graph)
            options=kind===:knn_graph ? (;k=4) : (;radius=.45)
            brute=V.visual_spec(pc;kind,options...,nn_backend=:bruteforce)
            tree=V.visual_spec(pc;kind,options...,nn_backend=:nearestneighbors)
            @test Set(brute.metadata.source_edges)==Set(tree.metadata.source_edges)
            @test V.visual_spec(pc;kind,options...).metadata.graph_backend===:nearestneighbors
        end
        duplicates=D.PointCloud(zeros(20,2))
        @test_throws ArgumentError V.visual_spec(duplicates;kind=:radius_graph,radius=0.,max_graph_edges=10)
        # The sparse radius result is admitted even when its all-pairs upper
        # bound exceeds the actual edge budget.
        separated=D.PointCloud(hcat(collect(1.:100.),zeros(100)))
        empty=V.visual_spec(separated;kind=:radius_graph,radius=.2,max_graph_edges=1)
        @test empty.metadata.total_edges==0
    end
end

@testset "A39 native coordinate fidelity and actual exports" begin
    V=TamerOp.Visualization;D=TamerOp.DataTypes
    pc=D.PointCloud([0. 0. 1.;1. 4. 2.;3. 2. 5.;8. 9. 7.])
    specs=[V.visual_spec(pc;kind=:knn_graph,k=1,nn_backend=:bruteforce,dims=(2,1),max_edges=2,max_labels=2,labels=["a","b","c","d"]),
        V.visual_spec(pc;kind=:points_3d,dims=(3,2,1),max_points=2)]
    @test specs[1].metadata.source_edges==[(1,2),(2,3)]
    @test only(l for l in specs[1].layers if l isa V.SegmentLayer).segments==[(0.,0.,4.,1.),(4.,1.,2.,3.)]
    for (backend,name) in ((:cairomakie,"CairoMakie"),(:wglmakie,"WGLMakie"))
        if Base.find_package(name)===nothing;@test_skip false;continue;end
        backend===:cairomakie ? (@eval import CairoMakie) : (@eval import WGLMakie)
        M=backend===:cairomakie ? CairoMakie.Makie : WGLMakie.Makie
        for spec in specs
            fig=V.render(spec;backend,size=(900,640))
            ax=only(b for b in fig.content if b isa M.Axis || b isa M.Axis3)
            points=only(l for l in spec.layers if l isa V.PointLayer || l isa V.Point3Layer)
            plot=only(p for p in ax.scene.plots if p isa M.Scatter)
            @test Tuple.(plot[1][])==points.points
            if spec===specs[1]
                p=only(p for p in ax.scene.plots if p isa M.LineSegments)
                @test Tuple.(p[1][])==[(x,y) for (a,b,c,d) in only(l for l in spec.layers if l isa V.SegmentLayer).segments for (x,y) in ((a,b),(c,d))]
                @test only(p for p in ax.scene.plots if p isa M.Text).input_text[]==["a","b"]
            end
            empty!(fig)
        end
        mktempdir() do dir
            for ext in (backend===:cairomakie ? ("png","svg","pdf") : ("html",))
                file=joinpath(dir,"preview.$ext")
                V.save_visual(file,specs[1];backend)
                @test filesize(file)>1000
                if ext=="html" && haskey(ENV,"TAMEROP_SOURCE_PREVIEW_HTML")
                    target=ENV["TAMEROP_SOURCE_PREVIEW_HTML"];mkpath(dirname(target));cp(file,target;force=true)
                end
                dest=get(ENV,"TAMEROP_A39_GALLERY","")
                if !isempty(dest);mkpath(dest);cp(file,joinpath(dest,"preview.$ext");force=true);end
            end
        end
    end
end
