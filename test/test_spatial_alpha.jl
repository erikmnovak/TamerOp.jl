using JSON3
@testset "A114 exact spatial alpha geometry" begin
    I=TamerOp.DataIngestion; O=TamerOp.OrdinaryPersistence
    # Regular tetrahedron: edge^2/4=2, face circumradius^2=8/3,
    # tetrahedron circumradius^2=3. Its shell has one H2 interval.
    X=TamerOp.PointCloud([1 1 1;1 -1 -1;-1 1 -1;-1 -1 1])
    f=TamerOp.AlphaFiltration(delaunay_backend=:naive)
    @test TamerOp.AlphaFiltration().params.max_dim == 3
    G,_,_=I._graded_complex_from_spatial_alpha(X,I._filtration_spec(f))
    @test length.(G.cells_by_dim)==[4,6,4,1]
    @test first.(G.grades)==[zeros(Int,4);fill(2,6);fill(8//3,4);3]
    for F in (CM.F2(),CM.F3(),CM.Fp(101))
        d=O.persistence_diagram(X,f;field=F)
        @test O.finite_intervals(d;dim=2)==[(8//3,3//1)]
        @test length(O.finite_intervals(d;dim=1))==3
        @test isempty(O.essential_births(d;dim=2))
        @test TamerOp.provenance(d).construction.grade_arithmetic===:exact_rational_squared_radius
    end
    # An obtuse triangular face inherits the tetrahedron's empty-sphere
    # radius, rather than its own nonempty circumcircle.
    Y=TamerOp.PointCloud([0 0 0;4 0 0;1 1 0;1 0 1])
    x=I.filtered_complex(Y,f)
    @test I.check_filtered_complex(x).valid
    G2=I.graded_complex(x)
    @test all(isfinite,first.(G2.grades))
    for k in 1:length(G2.boundaries),j in axes(G2.boundaries[k],2)
        offs=cumsum([0;length.(G2.cells_by_dim)])
        for ptr in nzrange(G2.boundaries[k],j)
            i=G2.boundaries[k].rowval[ptr]
            @test G2.grades[offs[k]+i][1]<=G2.grades[offs[k+1]+j][1]
        end
    end
    # Spatial embedding of a planar square: no metric-changing projection.
    square=TamerOp.PointCloud([0 0 0;1 0 1;1 1 1;0 1 0])
    d=O.persistence_diagram(square,f)
    @test O.finite_intervals(d;dim=1)==[(1//2,3//4)]
    @test isempty(O.essential_births(d;dim=1))
    line=TamerOp.PointCloud([0 0 0;1 2 2;2 4 4])
    @test O.finite_intervals(O.persistence_diagram(line,f);dim=0)==[(0//1,9//4),(0//1,9//4)]
    singleton=TamerOp.PointCloud([2 3 4])
    @test O.essential_births(O.persistence_diagram(singleton,f);dim=0)==[0]
    @test_throws ArgumentError O.persistence_diagram(TamerOp.PointCloud([0 0 0;0 0 0]),f)
    @test_throws ArgumentError TamerOp.AlphaFiltration(max_dim=-1)
    @test_throws ArgumentError O.persistence_diagram(TamerOp.PointCloud(zeros(4,4)),f)
    for md in 0:3
        fm=TamerOp.AlphaFiltration(max_dim=md,delaunay_backend=:naive)
        g,_,_=I._graded_complex_from_spatial_alpha(X,I._filtration_spec(fm))
        @test length(g.cells_by_dim)==md+1
        @test g.grades==G.grades[1:length(g.grades)]
    end
    limited=TamerOp.AlphaFiltration(construction=TamerOp.ConstructionOptions(budget=TamerOp.ConstructionBudget(max_simplices=10)))
    @test_throws ArgumentError O.persistence_diagram(X,limited)
    build=I.build_graded_complex(X,f)
    @test TamerOp.provenance(build).construction.affine_dimension==3
    @test TamerOp.provenance(build).backend==(:spatial_alpha=>:native_exact,)
    retained=O.persistence_diagram(X,f;source=true)
    @test TamerOp.provenance(retained).construction.effective===:alpha
end

@testset "A114 independent GUDHI exact geometry oracle" begin
    O=TamerOp.OrdinaryPersistence
    data=JSON3.read(read(joinpath(@__DIR__,"fixtures","spatial_alpha_gudhi.json"),String))
    for case in data.cases
        cloud=TamerOp.PointCloud([Float64.(p) for p in case.points])
        for backend in (:naive,:auto)
            f=TamerOp.AlphaFiltration(delaunay_backend=backend)
            d=O.persistence_diagram(cloud,f;field=CM.F3())
            if startswith(String(case.name),"generic")
                source=TamerOp.DataIngestion.filtered_complex(cloud,f)
                gotgrades=Dict(source.sources[i]=>Float64(only(source.complex.grades[i])) for i in eachindex(source.sources))
                expectedgrades=Dict(Tuple(Int.(cell[1]).+1)=>Float64(cell[2]) for cell in case.filtration)
                @test keys(gotgrades)==keys(expectedgrades)
                @test all(isapprox(gotgrades[key],value;rtol=1e-12,atol=1e-12) for (key,value) in expectedgrades)
            end
            for dim in 0:2
                got=sort([(Float64(a),Float64(b)) for (a,b) in O.finite_intervals(d;dim)])
                expected=sort([(Float64(b[1]),Float64(b[2])) for b in case.bars[dim+1] if b[2]!==nothing])
                @test length(got)==length(expected)
                @test all(isapprox.(first.(got),first.(expected);rtol=1e-12,atol=1e-12))
                @test all(isapprox.(last.(got),last.(expected);rtol=1e-12,atol=1e-12))
                @test length(O.essential_births(d;dim))==count(b->b[2]===nothing,case.bars[dim+1])
            end
        end
    end
end

@testset "A114 cospherical refinement and optional geometry" begin
    I=TamerOp.DataIngestion;O=TamerOp.OrdinaryPersistence
    cube=TamerOp.PointCloud([Float64[x,y,z] for x in (-1,1), y in (-1,1), z in (-1,1)][:])
    f=TamerOp.AlphaFiltration(delaunay_backend=:naive)
    g,_,_=I._graded_complex_from_spatial_alpha(cube,I._filtration_spec(f))
    @test length.(g.cells_by_dim)==[8,19,18,6]
    d=O.persistence_diagram(cube,f;field=CM.F3())
    @test O.finite_intervals(d;dim=2)==[(2//1,3//1)]
    @test O.essential_births(d;dim=0)==[0]
    @test isempty(O.essential_births(d;dim=1))
    @test isempty(O.essential_births(d;dim=2))
    st,_,_=I._graded_complex_from_spatial_alpha(cube,I._filtration_spec(f);return_simplex_tree=true)
    ds=O.persistence_diagram(I.filtered_complex(st;coordinates=I.point_matrix(cube));field=CM.F3())
    @test ds.finite_by_dim==d.finite_by_dim
    if I._SPATIAL_DELAUNAY_IMPL[]!==nothing
        for cloud in (cube,TamerOp.PointCloud(rand(MersenneTwister(114),10,3)))
            slow=O.persistence_diagram(cloud,f;field=CM.F3())
            fast=O.persistence_diagram(cloud,TamerOp.AlphaFiltration(delaunay_backend=:fast);field=CM.F3())
            @test fast.finite_by_dim==slow.finite_by_dim
            @test fast.essential_by_dim==slow.essential_by_dim
        end
        backend=I._SPATIAL_DELAUNAY_IMPL[]
        try
            I._SPATIAL_DELAUNAY_IMPL[]=nothing
            @test_throws ArgumentError O.persistence_diagram(cube,TamerOp.AlphaFiltration(delaunay_backend=:fast))
            @test O.persistence_diagram(cube,TamerOp.AlphaFiltration()).finite_by_dim==d.finite_by_dim
        finally
            I._SPATIAL_DELAUNAY_IMPL[]=backend
        end
    else
        @test_throws ArgumentError O.persistence_diagram(cube,TamerOp.AlphaFiltration(delaunay_backend=:fast))
    end
    enc=TamerOp.encode(cube,f;degree=2)
    p=TamerOp.provenance(enc)
    @test p.construction.effective===:alpha
    @test p.construction.substitution===:none
    @test p.construction.affine_dimension==3
    @test p.approximation.grade_arithmetic===:exact_rational_squared_radius
    X=I._spatial_alpha_points(cube)
    @test_throws ArgumentError I._alpha_check_tetrahedra(X,[[1,2,3,8]])
    @test_throws ArgumentError I._alpha_check_tetrahedra(X,[[1,2,3,4]])
    # Two valid five-tetrahedron cube triangulations have complementary hull
    # diagonals. Their union passes local empty-ball/face checks but covers
    # the hull twice; exact volume must reject that numerical candidate.
    even=[[1,4,6,7],[1,2,4,6],[1,3,4,7],[1,5,6,7],[4,6,7,8]]
    odd=[[2,3,5,8],[1,2,3,5],[2,3,4,8],[2,5,6,8],[3,5,7,8]]
    @test I._alpha_check_tetrahedra(X,even)===nothing
    @test I._alpha_check_tetrahedra(X,odd)===nothing
    @test_throws ArgumentError I._alpha_check_tetrahedra(X,vcat(even,odd))
end
