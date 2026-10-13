# Independent source geometry, known topological events, and live state tests.
function _a93_cloud(;square=false,repeated=false,kwargs...)
    points=DT.PointCloud(square ? [-1 -1;1 -1;1 1;-1 1] : repeated ? [0 0;0 0;2 0] : [0 0;2 0])
    backend=square || repeated ? :subdivision_cech : :exhaustive
    filtration=DI.RhomboidFiltration(;backend,
        construction=TamerOp.ConstructionOptions(budget=TamerOp.ConstructionBudget(max_simplices=20000)),kwargs...)
    return DI.filtered_complex(points,filtration)
end

@testset "A93 native source coordinates and static exports" begin
    V=TamerOp.Visualization
    x=_a93_cloud(depth_range=(1,1))
    spec=V.visual_spec(x;kind=:geometric_filtration,parameter=(1,1),cell=3,
        view_dims=(1,2),sample_resolution=5)
    # Two closed unit disks centered at (0,0) and (2,0). At y=0 every
    # sampled x in -1:3 lies in their union; away from that row only x=0,2 do.
    plotted=[0. 0. 1. 0. 0.;1. 1. 1. 1. 1.;0. 0. 1. 0. 0.;
             1. 1. 1. 1. 1.;0. 0. 1. 0. 0.]
    for backend in (:cairomakie,:wglmakie)
        package=backend===:cairomakie ? :CairoMakie : :WGLMakie
        if Base.find_package(string(package))===nothing
            @test_skip false
            continue
        end
        @eval import $package
        mk=getfield(@__MODULE__,package)
        fig=V.render(spec;backend)
        try
            mk.Makie.update_state_before_display!(fig)
            axes=filter(a->a isa mk.Axis,fig.content)
            source=only(filter(a->a.xlabel[]=="source x1",axes))
            hm=only(filter(p->p isa mk.Heatmap,source.scene.plots))
            @test collect(hm[1][])==collect(-1.5:1:3.5)
            @test collect(hm[2][])==collect(-1.25:0.5:1.25)
            @test hm[3][]==plotted
            points=filter(p->p isa mk.Scatter,source.scene.plots)
            @test all(p->Tuple.(p[1][])==[(0.,0.),(2.,0.)],points)
            @test length(points)==2 # Indexed sites and the selected cap's sites.
            @test all(a->all(>(0),mk.Makie.widths(mk.Makie.viewport(a.scene)[])),axes)
            @test Tuple(mk.Makie.widths(mk.Makie.viewport(fig.scene)[]))==(1250,1250)
            if backend===:cairomakie
                mktempdir() do dir
                    for extension in ("png","svg","pdf")
                        path=joinpath(dir,"geometric."*extension)
                        mk.save(path,fig)
                        @test filesize(path)>1000
                    end
                end
            end
        finally
            empty!(fig)
        end
    end
end

@testset "A93 exact geometry and terminal caps" begin
    V=TamerOp.Visualization
    x=_a93_cloud(depth_range=(1,1));G=DI.graded_complex(x)
    @test DI.check_filtered_complex(x).valid
    @test :geometric_filtration in V.available_visuals(x)
    @test DT.cell_counts(G)==[2,1,0]
    @test DI.cell_geometry(x,3)==[(0//1,0//1,1//1),(2//1,0//1,1//1)]
    @test DI.cell_source(x,3).section==:horizontal
    @test DI.cell_source(x,3).radius_squared==1
    @test DI.coverage_at(x,(1,0);parameter=(1,1)).count==2
    @test !DI.coverage_at(x,(1,0);parameter=(1-big(1)//2^60,1)).member
    for (r,h) in ((0,2),(1-big(1)//2^60,2),(1,1),(2,1))
        s=V.visual_spec(x;kind=:geometric_filtration,parameter=(r,1),sample_resolution=3)
        @test s.metadata.selected_betti==h
        @test s.metadata.betti==[h,0,0]
        @test V.check_visual_spec(s).valid
        @test !s.metadata.projection
        for (i,px) in enumerate(s.panels[2].metadata.xs),(j,py) in enumerate(s.panels[2].metadata.ys)
            r2=TamerOp.CoreModules.QQ(r)^2
            @test s.panels[2].metadata.coverage_counts[j,i]==sum((TamerOp.CoreModules.QQ(px)-a)^2+TamerOp.CoreModules.QQ(py)^2<=r2 for a in (0,2))
        end
    end
    s=V.visual_spec(x;kind=:geometric_filtration,parameter=(1,1),view_dims=(1,2),sample_resolution=3,
        invariant=:barcode,cell=3)
    @test s.metadata.projection
    @test TamerOp.OrdinaryPersistence.finite_intervals(s.metadata.diagram;dim=0)==[(0,1)]
    @test occursin("squared birth radius",join(s.panels[end-1].layers[1].labels))
    @test TamerOp.Advanced.radius_slice===DI.radius_slice
    @test DT.parameter_dim(DI.radius_slice(x;depth=1))==1
    cache=TamerOp.SessionCache();p=DT.PointCloud([0 0;2 0]);f=DI.RhomboidFiltration(depth_range=(1,1))
    a=DI.filtered_complex(p,f;cache);b=DI.filtered_complex(p,f;cache)
    @test a.geometry==b.geometry
    a.geometry[1][1]=(99//1,0//1,1//1)
    @test DI.cell_geometry(b,1)!=DI.cell_geometry(a,1)
    @test_throws ArgumentError DI.filtered_complex(p,f;cache=:invalid)
    depth_zero=_a93_cloud(depth_range=(0,0))
    zero_view=V.visual_spec(depth_zero;kind=:geometric_filtration,sample_resolution=3)
    @test zero_view.metadata.parameter==(0,0)
    @test zero_view.metadata.selected_betti==1
    @test all(zero_view.panels[2].metadata.membership)
    native_empty=_a93_cloud(depth_range=(2,2),radius=0)
    @test isempty(DI.radius_slice(native_empty;depth=2).grades)
    @test V.visual_spec(native_empty;kind=:geometric_filtration,sample_resolution=3).metadata.selected_betti==0
    @test_throws ArgumentError V.visual_spec(native_empty;kind=:geometric_filtration,parameter=(1,2))
end

@testset "A93 square flags repeated sites and empty windows" begin
    V=TamerOp.Visualization;O=TamerOp.OrdinaryPersistence
    x=_a93_cloud(square=true,depth_range=(1,4));m=DI.geometric_data(x)
    @test m.representation===:schematic_flags
    @test DI.check_filtered_complex(x).valid
    sq2=sqrt(TamerOp.ExactReals.AlgebraicReal(2))
    @test V.visual_spec(x;kind=:geometric_filtration,sample_resolution=3).panels[1].axes.zlabel=="subset size"
    capped_flags=_a93_cloud(square=true,depth_range=(1,1))
    @test maximum(last(p) for cell in capped_flags.geometry for p in cell)==4
    @test all(g->g[2]==1,DI.graded_complex(capped_flags).grades)
    for field in (TamerOp.CoreModules.F2(),TamerOp.CoreModules.PrimeField(3),TamerOp.CoreModules.QQField()),(r,h1) in ((0,0),(1,1),(sq2,0))
        s=V.visual_spec(x;kind=:geometric_filtration,parameter=(r,1),field,degree=1,sample_resolution=3,max_cells=4)
        @test s.metadata.selected_betti==h1
        @test V.check_visual_spec(s).valid
        @test s.metadata.omitted_count == max(0,s.metadata.eligible_count-4)
    end
    S=DI.radius_slice(x;depth=1);bars=O.persistence_diagram(S)
    @test O.finite_intervals(bars;dim=1)==[(1,sq2)]
    # Different subset labels can occupy the same barycentre, including pairs
    # on opposite diagonals. Their vertices and incidence must remain distinct.
    n=DT.cell_counts(DI.graded_complex(x))[1]
    @test length(unique(first.(x.geometry[1:n])))<n
    @test length(unique(x.sources[1:n]))==n
    repeated=_a93_cloud(repeated=true,depth_range=(1,3))
    @test DI.coverage_at(repeated,(0,0);parameter=(0,2)).covered_sites==[1,2]
    @test DI.coverage_at(repeated,(0,0);parameter=(0,2)).member
    @test !DI.coverage_at(repeated,(0,0);parameter=(0,3)).member
    @test V.visual_spec(repeated;kind=:geometric_filtration,parameter=(0,2),sample_resolution=3).metadata.selected_betti==1
    empty=_a93_cloud(square=true,depth_range=(4,4),radius=0)
    @test sum(DT.cell_counts(DI.graded_complex(empty)))==0
    @test DI.check_filtered_complex(empty).valid
    @test isempty(DI.radius_slice(empty;depth=4).grades)
    @test V.check_visual_spec(V.visual_spec(empty;kind=:geometric_filtration,sample_resolution=3)).valid
    @test V.visual_spec(empty;kind=:geometric_filtration,sample_resolution=3).metadata.selected_betti==0
    capped=_a93_cloud(depth_range=(1,1),radius=0)
    c=V.visual_spec(capped;kind=:geometric_filtration,invariant=:barcode,sample_resolution=3)
    @test all(r->r.right_status===:censored,c.panels[4].metadata.records)
    @test all(r->r.singleton && r.right_closed,c.panels[4].metadata.records)
end

@testset "A93 lifted cells source sections and comparisons" begin
    V=TamerOp.Visualization
    p=DT.PointCloud([0 0;2 0;0 2])
    f=DI.RhomboidFiltration();x=DI.filtered_complex(p,f)
    @test DI.check_filtered_complex(x).valid
    @test DT.max_dim(DI.graded_complex(x))==3
    @test maximum(last(v) for cell in x.geometry for v in cell)==3
    for i in eachindex(x.geometry)
        src=DI.cell_source(x,i)
        @test all(v->length(src.inside)<=last(v)<=length(src.inside)+length(src.on),x.geometry[i])
    end
    sliced=DI.filtered_complex(p,DI.RhomboidFiltration(depth_range=(1,2)))
    @test DI.check_filtered_complex(sliced).valid
    s=V.visual_spec(x;kind=:geometric_filtration,parameter=(2,1),compare=sliced,sample_resolution=3)
    @test s.metadata.skeleton
    @test s.metadata.comparison.relation===:same_multicover_on_common_window
    # Source coverage in a section z=2 consists of radius sqrt(5-4)=1 disks.
    space=DI.filtered_complex(DT.PointCloud([0 0 0;2 0 0]),DI.RhomboidFiltration(depth_range=(1,1)))
    @test_throws ArgumentError V.visual_spec(space;kind=:geometric_filtration)
    s=V.visual_spec(space;kind=:geometric_filtration,view_dims=(1,2,4),parameter=(sqrt(TamerOp.ExactReals.AlgebraicReal(5)),1),
        source_anchor=(0,0,2),sample_resolution=3)
    @test s.panels[2].metadata.source_anchor==(0,0,2)
    @test DI.coverage_at(space,(1,0,2);parameter=(sqrt(TamerOp.ExactReals.AlgebraicReal(5)),1)).count==2
    @test DI.coverage_at(space,(1,0,2);parameter=(2,1)).count==0
    @test_throws ArgumentError V.visual_spec(x;kind=:geometric_filtration,compare=_a93_cloud(square=true),sample_resolution=3)
    incremental=DI.filtered_complex(p,DI.RhomboidFiltration(backend=:incremental))
    @test incremental.sources==x.sources && incremental.geometry==x.geometry
    @test DI.graded_complex(incremental).boundaries==DI.graded_complex(x).boundaries
    flags=DI.filtered_complex(p,DI.RhomboidFiltration(backend=:subdivision_cech,
        construction=TamerOp.ConstructionOptions(budget=TamerOp.ConstructionBudget(max_simplices=10000))))
    O=TamerOp.OrdinaryPersistence
    for k in 0:3,degree in 0:2
        a=O.persistence_diagram(DI.radius_slice(x;depth=k))
        b=O.persistence_diagram(DI.radius_slice(flags;depth=k))
        @test O.finite_intervals(a;dim=degree)==O.finite_intervals(b;dim=degree)
        @test O.essential_births(a;dim=degree)==O.essential_births(b;dim=degree)
    end
end

@testset "A93 core regions contracts and session rollback" begin
    V=TamerOp.Visualization
    p=DT.PointCloud([0 0;2 0;0 3])
    x=DI.filtered_complex(p,DI.CoreFiltration(max_dim=2,k_values=[1,2,3]))
    y=DI.filtered_complex(p,DI.CoreDelaunayFiltration(max_dim=2,k_values=[1,2,3]))
    @test DI.geometric_data(x).grade_precision===:floating_point
    @test DI.geometric_data(y).backend===:core_delaunay
    @test isempty(DI.coverage_at(x,(0,0);parameter=(1,2)).eligible_sites)
    @test DI.coverage_at(x,(0,0);parameter=(2,2)).eligible_sites==[1,2]
    # Site 3 is nearest but ineligible. Restricting to the full-cloud Voronoi
    # cells differs from recomputing Voronoi cells on the eligible sites.
    @test DI.coverage_at(x,(0,2);parameter=(2,2)).member
    @test !DI.coverage_at(y,(0,2);parameter=(2,2)).member
    @test DI.coverage_at(y,(1,0);parameter=(2,2)).contributing_sites==[1,2]
    for r in (0,1,2,3),k in (1,2,3),z in (x,y)
        s=V.visual_spec(z;kind=:geometric_filtration,parameter=(r,k),sample_resolution=3)
        @test V.check_visual_spec(s).valid
        G=DI.graded_complex(z);slice=DI.radius_slice(z;depth=k)
        @test count(g->first(g)<=r,slice.grades)==sum(s.metadata.active_counts)
    end
    @test V.visual_spec(x;kind=:geometric_filtration,compare=y,parameter=(2,2),sample_resolution=3).metadata.comparison.relation===:geometric_comparison_only
    a=_a93_cloud(depth_range=(1,1))
    for kw in ((;parameter=(-1,1)),(;parameter=(1,1.5)),(;parameter=(1,2)),(;source_dims=(1,1)),
        (;source_anchor=(0,)),(;sample_resolution=1),(;sample_resolution=129),(;degree=-1),
        (;invariant=:unknown),(;invariant=:barcode,field=TamerOp.CoreModules.QQField()),(;max_matrix_entries=0))
        @test !V.check_visual_request(a;kind=:geometric_filtration,kw...).valid
        @test_throws ArgumentError V.visual_spec(a;kind=:geometric_filtration,kw...)
    end
    s=V.inspection_session(a;kind=:geometric_filtration,sample_resolution=3,max_algebra_cells=2)
    @test V.inspection_snapshot(s).metadata.selected_betti==2
    before=V.inspection_snapshot(s)
    @test_throws ArgumentError V.select_inspection!(s;parameter=(1,1))
    @test V.inspection_snapshot(s)===before
    @test V.inspection_summary(s).revision==0
    @test !V.check_inspection_selection(s;parameter=(1,2)).valid
    V.close_inspection!(s)
    @test V.check_inspection_session(s).valid
    s=V.inspection_session(a;kind=:geometric_filtration,sample_resolution=3)
    V.select_inspection!(s;parameter=(1,1),cell=3)
    @test V.inspection_snapshot(s).metadata.selected_betti==1
    @test V.inspection_summary(s).revision==1
    V.reset_inspection!(s)
    @test V.inspection_snapshot(s).metadata.selected_betti==2
    V.close_inspection!(s)
    @test V.inspection_snapshot(s).kind===:geometric_filtration
end

@testset "A123 geometric queries reuse only the fixed-depth barcode" begin
    V=TamerOp.Visualization
    x=_a93_cloud(depth_range=(1,2))
    s=V.inspection_session(x;kind=:geometric_filtration,parameter=(3//4,1),
        view_dims=(1,2),invariant=:barcode,sample_resolution=5)
    diagram=s.snapshot.metadata.diagram
    V.select_inspection!(s;parameter=(1,1))
    @test s.snapshot.metadata.diagram===diagram
    @test s.snapshot.metadata.barcode_reused
    @test s.snapshot.metadata.betti[1]==1
    V.select_inspection!(s;parameter=(1,2))
    @test !s.snapshot.metadata.barcode_reused
    @test s.snapshot.metadata.geometric_options.complex.parameter==(1,2)
    controls=V._filtration_controls(x,(1,2))
    @test controls[1].label=="Physical radius"
    @test controls[2].discrete && (controls[2].lower,controls[2].upper)==(1,2)
    @test_throws ArgumentError V._filtration_control_parameter(controls,(1,3))
    @test_throws ArgumentError V._filtration_control_parameter(controls,(-1,1))
    capped=_a93_cloud(depth_range=(2,2),radius=0)
    fixed=V._filtration_controls(capped,(0,2))
    @test fixed[1].lower==fixed[1].upper==0
    @test_throws ArgumentError V._filtration_control_parameter(fixed,(1//100,2))
    V.close_inspection!(s)
end
