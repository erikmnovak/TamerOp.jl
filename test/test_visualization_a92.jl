# Geometry is tested against actual cellular boundaries, not a plotted preview.
function _a92_triangle(;multi=false,ambient=2)
    B1=sparse([-1 -1 0;1 0 -1;0 1 1]);B2=sparse(reshape([1,-1,1],3,1))
    grades=[(0//1,),(0//1,),(0//1,),(1//1,),(1//1,),(1//1,),(2//1,)]
    G=if multi
        DT.MultiCriticalGradedComplex([[11,12,13],[21,22,23],[31]],[B1,B2],
            [[(0,0)] for _ in 1:7])
    else
        DT.GradedComplex([[11,12,13],[21,22,23],[31]],[B1,B2],grades)
    end
    P=[(0//1,0//1),(2//1,0//1),(0//1,2//1)]
    ambient>2 && (P=[(p...,ntuple(_->0//1,ambient-2)...) for p in P])
    geometry=[[P[1]],[P[2]],[P[3]],P[[1,2]],P[[1,3]],P[[2,3]],P]
    DI.filtered_complex(G;geometry)
end

@testset "A92 exact cellular activity and selection" begin
    V=TamerOp.Visualization
    x=_a92_triangle();G=DI.graded_complex(x)
    @test DI.check_filtered_complex(x).valid
    @test TamerOp.filtered_complex===DI.filtered_complex
    @test V.available_visuals(x)==(:filtered_complex,:chain_inspector)
    @test occursin("7 cells",sprint(show,x))
    for (p,counts) in ((-1,[0,0,0]),(0,[3,0,0]),(1,[3,3,0]),(2,[3,3,1]))
        s=V.visual_spec(x;parameter=p)
        @test s.metadata.active_counts==counts
        @test s.metadata.drawn_counts==counts
        @test V.check_visual_spec(s).valid
        # Euler characteristic distinguishes the empty state, loop and filling.
        @test sum((-1)^(d-1)*counts[d] for d in eachindex(counts))==sum((-1)^G.cell_dims[i] for i in eachindex(s.metadata.active) if s.metadata.active[i];init=0)
    end
    p=2-big(1)//big(2)^60
    @test V.visual_spec(x;parameter=p).metadata.active_counts==[3,3,0]
    @test V.visual_spec(x;parameter=2).metadata.active_counts==[3,3,1]
    @test TamerOp.OrdinaryPersistence.finite_intervals(TamerOp.persistence_diagram(G);dim=1)==[(1,2)]
    s=V.visual_spec(x;parameter=2,cell=7)
    @test s.metadata.selected.cell_id==31
    @test s.metadata.selected.source==(2,31)
    @test [f.index for f in s.metadata.selected.boundary]==[4,5,6]
    @test [f.coefficient for f in s.metadata.selected.boundary]==[1,-1,1]
    @test G.boundaries[1]*[f.coefficient for f in s.metadata.selected.boundary]==zeros(Int,3)
    budget=V.visual_spec(x;parameter=2,cell=7,max_cells=2)
    @test first(budget.metadata.displayed_indices)==7
    @test budget.metadata.omitted_count==5
    @test budget.metadata.active_counts==[3,3,1]
    @test budget.metadata.selected.boundary==s.metadata.selected.boundary
    edges=V.visual_spec(x;parameter=2,cell=7,dimensions=(0,1))
    @test edges.metadata.drawn_counts==[3,3,0]
    @test edges.metadata.skeleton
    inactive=V.visual_spec(x;parameter=0,activity=:all,cell=7)
    @test inactive.metadata.drawn_counts==[3,3,1]
    @test !inactive.metadata.selected.active
    for kw in ((;parameter=NaN),(;parameter=(1,2)),(;cell=0),(;cell=true),
        (;dimensions=(3,)),(;dimensions=(1,1)),(;activity=:guess),(;alpha=Inf),
        (;alpha=-0.1),(;max_cells=0),(;max_labels=-1),(;view_dims=(1,1)),(;field_axis=2))
        @test !V.check_visual_request(x;kw...).valid
        @test_throws ArgumentError V.visual_spec(x;kw...)
    end
    @test_throws ArgumentError DI.filtered_complex(G;geometry=x.geometry,orientation=(-1,))
    broken=DT.GradedComplex(DT.cells(G),[G.boundaries[1],sparse(ones(Int,3,1))],G.grades)
    @test_throws ArgumentError DI.filtered_complex(broken;geometry=x.geometry)
    @test_throws ArgumentError DI.filtered_complex(G;geometry=x.geometry[1:6])
    x4=_a92_triangle(ambient=4)
    @test_throws ArgumentError V.visual_spec(x4)
    @test V.visual_spec(x4;view_dims=(1,2)).metadata.projection
    @test !V.visual_spec(_a92_triangle(ambient=3)).metadata.projection
    @test V.check_visual_spec(V.visual_spec(_a92_triangle(ambient=3);parameter=-1)).valid
    # Display rounding must not imply that distinct source vertices are equal.
    close_geometry=deepcopy(x.geometry)
    tiny=big(1)//big(2)^60
    close_geometry[1]=[(big(1)//1,0//1)]
    close_geometry[2]=[(big(1)//1+tiny,0//1)]
    rounded=DI.filtered_complex(G;geometry=close_geometry)
    @test any(contains("drawing precision"),V.visual_spec(rounded;parameter=0).panels[1].metadata.warnings)
end

@testset "A92 multicritical union and exact orientation" begin
    V=TamerOp.Visualization
    G=DT.MultiCriticalGradedComplex([[10,20],[30]],[sparse(reshape([-1,1],2,1))],
        [[(0,0)],[(0,0)],[(1,3),(3,1)]])
    x=DI.filtered_complex(G;geometry=[[(0,0)],[(1,0)],[(0,0),(1,0)]])
    for (p,edge) in (((2,2),0),((1,3),1),((3,1),1),((0,4),0),((4,4),1))
        @test V.visual_spec(x;parameter=p).metadata.active_counts==[2,edge]
    end
    @test !V._complex_active(x,3,(1,1)) # not the bounding-box minimum
    # A disconnected family of appearances remains a union, not a convex hull.
    G2=DT.MultiCriticalGradedComplex([[10,20],[30]],[G.boundaries[1]],
        [[(0,5)],[(0,5)],[(1,2),(3,4)]])
    y=DI.filtered_complex(G2;geometry=x.geometry,orientation=(1,-1))
    for (p,e) in (((2,3),0),((1,2),1),((3,4),1),((4,5),0))
        @test V.visual_spec(y;parameter=p).metadata.active_counts==[2,e]
    end
    bad=DT.MultiCriticalGradedComplex([[10,20],[30]],[G.boundaries[1]],
        [[(0,0)],[(4,4)],[(1,3),(3,1)]])
    @test_throws ArgumentError DI.filtered_complex(bad;geometry=x.geometry)
end

@testset "A92 constructed source geometry and image events" begin
    V=TamerOp.Visualization
    pc=DT.PointCloud([0.0 0.0;1.0 0.0;0.0 1.0])
    x=DI.filtered_complex(pc,DI.RipsFiltration(max_dim=2))
    @test V.visual_spec(x;parameter=1).metadata.active_counts==[3,2,0]
    @test V.visual_spec(x;parameter=2).metadata.active_counts==[3,3,1]
    @test DI.cell_source(x,7)==(1,2,3)
    @test x.provenance.grade_scale==:diameter
    tree=DI._simplex_tree_multi_from_complex(DI.graded_complex(x))
    retained=DI.filtered_complex(tree;coordinates=DT.point_matrix(pc))
    @test retained.geometry==x.geometry
    @test retained.sources==x.sources
    reduced=DI.filtered_complex(pc,DI.RipsFiltration(max_dim=1,n_landmarks=2,
        construction=DI.ConstructionOptions(sparsify=:greedy_perm)))
    @test reduced.provenance.sampled
    @test [only(DI.cell_source(reduced,i)) for i in 1:2]==reduced.provenance.source_vertex_ids
    # The square uses vertex values: its ring has 8 vertices and 8 edges.
    values=zeros(3,3);values[2,2]=2
    ring=DI.filtered_complex(DT.ImageNd(values),DI.CubicalFiltration())
    @test V.visual_spec(ring;parameter=0).metadata.active_counts==[8,8,0]
    @test V.visual_spec(ring;parameter=2).metadata.active_counts==[9,12,4]
    @test DI.cell_source(ring,22)==(origin=(1,1),mask=(1,1))
    @test DI.cell_geometry(ring,22)==[(1,1),(2,1),(2,2),(1,2)]
    for N in (2,3), periodic in (false,true), orientation in ((1,),(-1,))
        a=reshape(collect(1.0:2^N),ntuple(_->2,N))
        f=DI.CubicalFiltration(orientation=orientation,periodic=periodic)
        im=DI.filtered_complex(DT.ImageNd(a),f)
        @test DI.check_filtered_complex(im).valid
        for t in (0,1,3,2^N,2^N+1)
            s=V.visual_spec(im;parameter=t)
            expected=Bool[]
            for i in eachindex(im.geometry)
                id=DI.cell_source(im,i);vary=findall(==(1),collect(id.mask))
                vals=[a[ntuple(k->mod1(id.origin[k]+(k in vary ? (b>>(findfirst(==(k),vary)-1))&1 : 0),2),N)...] for b in 0:2^length(vary)-1]
                push!(expected,orientation[1]==1 ? maximum(vals)<=t : minimum(vals)>=t)
            end
            @test s.metadata.active==expected
            @test V.check_visual_spec(s).valid
        end
    end
    periodic=DI.filtered_complex(DT.ImageNd(zeros(2,2)),DI.CubicalFiltration(periodic=true))
    channel=[1.0 2.0;3.0 4.0]
    mixed=DI.filtered_complex(DT.ImageNd(channel),DI.CubicalFiltration(channels=[channel,channel],orientation=(1,-1)))
    @test DI.graded_complex(mixed).grades[end]==(4.0,1.0)
    @test V.visual_spec(mixed;parameter=(4,1)).metadata.active_counts==[4,4,1]
    @test V.visual_spec(mixed;parameter=(3,2)).metadata.active_counts==[2,0,0]
    @test V.visual_spec(periodic;parameter=0).metadata.active_counts==[4,8,4]
    @test maximum(p[1] for cell in periodic.geometry for p in cell)==3
    @test periodic.provenance.domain==((1,3),(1,3))
    @test occursin("opposite faces",join(V.visual_spec(periodic).panels[2].layers[1].labels," "))
    field=V.visual_spec(ring;parameter=2,field_axis=1)
    colored=filter(l->l isa V.PointLayer && l.color isa Vector,field.panels[1].layers)
    @test only(colored).color==vec(values)
    graph=DT.GraphData(3,[(1,2),(2,3)];coords=[0.0 0.0;1.0 0.0;1.0 1.0])
    gx=DI.filtered_complex(graph,DI.GraphLowerStarFiltration(vertex_values=[0.0,1.0,2.0]))
    @test V.visual_spec(gx;parameter=1).metadata.active_counts==[2,1]
end

@testset "A92 ordinary image identity and exact grades" begin
    V=TamerOp.Visualization;O=TamerOp.OrdinaryPersistence
    values=[0 0 0;0 5 0;0 0 0]
    x=DI.filtered_complex(values)
    @test V.visual_spec(x;parameter=0).metadata.active_counts==[16,24,8]
    @test V.visual_spec(x;parameter=5).metadata.active_counts==[16,24,9]
    @test DI.cell_source(x,45)==(origin=(2,2),mask=(1,1))
    @test DI.graded_complex(x).boundaries==O._top_cell_complex_2d(values,(false,false),:sublevel).boundaries
    for input in (:top_cells,:vertices),order in (:sublevel,:superlevel),per in ((false,false),(true,false),(true,true))
        y=DI.filtered_complex(values;input,order,periodic=per)
        @test DI.check_filtered_complex(y).valid
        original=O.cubical_persistence(values;input,order,periodic=per)
        retained=O.persistence_diagram(DI.graded_complex(y);order)
        @test O.finite_intervals(original;dim=1)==O.finite_intervals(retained;dim=1)
        @test O.essential_births(original;dim=1)==O.essential_births(retained;dim=1)
    end
    torus=DI.filtered_complex(zeros(1,1);periodic=true)
    @test V.visual_spec(torus;cell=4).metadata.active_counts==[1,2,1]
    @test isempty(V.visual_spec(torus;cell=4).metadata.selected.boundary)
    tiny=big(1)//big(2)^60
    exact=DI.filtered_complex([big(1)//1 big(1)//1+tiny];input=:top_cells)
    @test V.visual_spec(exact;parameter=1).metadata.active_counts[end]==1
    @test V.visual_spec(exact;parameter=1+tiny).metadata.active_counts[end]==2
    # The top-cell adapter also supports voxel volumes.
    volume=DI.filtered_complex(ones(2,2,2);input=:top_cells)
    @test DI.check_filtered_complex(volume).valid
    @test DT.cell_counts(DI.graded_complex(volume))==[27,54,36,8]
    @test_throws ArgumentError DI.filtered_complex(values;input=:guess)
end

@testset "A92 linked session and optional native rendering" begin
    V=TamerOp.Visualization;x=_a92_triangle();s=V.inspection_session(x)
    calls=Ref(0);token=V._on_inspection(s,_ -> calls[]+=1)
    V.select_inspection!(s;parameter=1,cell=4)
    @test V.inspection_summary(s).active_counts==[3,3,0]
    @test calls[]==1
    old=V.inspection_snapshot(s);state=V.inspection_selection(s)
    @test_throws ArgumentError V.select_inspection!(s;parameter=NaN)
    @test V.inspection_snapshot(s)===old && V.inspection_selection(s)==state
    V.reset_inspection!(s);@test V.inspection_selection(s).cell===nothing
    @test V.check_inspection_session(s).valid
    cairo=Base.get_extension(TamerOp,:TamerOpCairoMakieExt)
    wgl=Base.get_extension(TamerOp,:TamerOpWGLMakieExt)
    if cairo!==nothing && wgl!==nothing
        mktempdir() do dir
            for obj in (x,_a92_triangle(ambient=3),DI.filtered_complex(zeros(2,2,2);input=:vertices)),backend in (:cairomakie,:wglmakie)
                figure=V.visualize(obj;parameter=2,backend)
                @test figure!==nothing
                M=cairo.CairoMakie.Makie
                M.update_state_before_display!(figure)
                for axis in filter(a->a isa M.Axis3,figure.content)
                    # All retained vertices lie inside our padded source box.
                    # Late aspect changes used to leave inherited clipping planes
                    # cutting off cube corners and horizontal edges/faces.
                    for plot in filter(p->p isa M.Scatter,axis.scene.plots)
                        world=M.apply_transform_and_model(plot,plot[1][])
                        @test all(p->!M.is_clipped(plot.clip_planes[],p),world)
                    end
                end
            end
            for format in ("png","svg","pdf")
                path=joinpath(dir,"complex."*format)
                V.save_visual(path,x;parameter=2,backend=:cairomakie)
                @test filesize(path)>100
            end
            @test V.visualize(x;parameter=2,cell=7,style=V.VisualStyle(palette=:grayscale),backend=:cairomakie)!==nothing
            app=V.visualize(s;backend=:wglmakie)
            ui=wgl._inspection_ui(app)
            @test ui!==nothing
            V.select_inspection!(s;parameter=2,cell=7)
            @test occursin("7",ui.controls.cell.value[])
            @test isempty(ui.error_text[])
            ui.dispose()
        end
    else
        @test_skip false
    end
    V.close_inspection!(s)
    @test s.object===nothing && isempty(s.listeners)
    @test V.check_inspection_session(s).valid
    @test_throws ArgumentError V.select_inspection!(s;parameter=1)
end
