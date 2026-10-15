include("fixtures/cohomological_coordinates.jl")
@testset "A97 circular coordinates and gauge" begin
    C=TamerOp.CohomologicalCoordinates;O=TamerOp.OrdinaryPersistence
    source=coordinate_ring_fixture();n=12
    for field in (CM.F2(),CM.F3(),CM.Fp(41))
        r=C.circular_coordinates(source;scale=0,kind=:essential,field)
        @test C.check_coordinates(r).valid
        @test r.vertices==collect(1:n)
        @test size(C.coordinate_values(r))==(n,1)
        @test C.coordinate_summary(r).components==1
        @test TamerOp.provenance(r).field==field
        @test isapprox(C.coordinate_summary(r).energy,[1/n])
        z=exp.(2pi*im*r.values[:,1]);steps=z[[2:n;1]]./z
        @test all(isapprox.(steps,fill(first(steps),n);atol=1e-9))
        @test isapprox(abs(angle(first(steps))),2pi/n)
        alt=C.circular_coordinates(source;scale=0,kind=:essential,field,anchors=[5])
        @test isapprox(exp.(2pi*im*(alt.values[:,1]-r.values[:,1])),fill(exp(-2pi*im*r.values[5,1]),n))
        @test alt.values[5,1]==0
        diagram=O.persistence_diagram(source;field,cocycles=true)
        reused=C.circular_coordinates(source;scale=0,kind=:essential,field,diagram)
        @test isapprox(reused.values,r.values)
        @test isapprox(C.circular_coordinates(diagram;scale=0,kind=:essential).values,r.values)
        @test_throws ArgumentError C.circular_coordinates(source;scale=0,kind=:essential,field,anchors=[1,2])
    end
    @test_throws ArgumentError C.circular_coordinates(source;scale=-1,kind=:essential)
    @test_throws ArgumentError C.circular_coordinates(source;scale=0,kind=:finite)
    @test_throws ArgumentError C.circular_coordinates(source;scale=0,kind=:essential,edge_weights=zeros(n))
    @test_throws ArgumentError C.circular_coordinates(source;scale=0,kind=:essential,field=CM.QQField())
    @test_throws ArgumentError C.circular_coordinates(source;scale=0,kind=:essential,budget=C.CoordinateBudget(max_vertices=2))
    @test_throws ArgumentError C.toroidal_coordinates(source;scale=0,classes=[(;kind=:essential,index=1),(;kind=:essential,index=1)])
    @test TamerOp.circular_coordinates===C.circular_coordinates
    @test TamerOp.Advanced.check_coordinates===C.check_coordinates
    weights=Float64.(1:n)
    weighted=C.circular_coordinates(source;scale=0,kind=:essential,edge_weights=weights)
    @test isapprox(vec(abs.(weighted.harmonic)),1 ./ (weights.*sum(1 ./ weights)))
    @test isapprox(only(C.coordinate_summary(weighted).energy),1/sum(1 ./ weights))
    G=source.complex
    disjoint=DT.GradedComplex([collect(1:n+1),collect(1:n)],
        [vcat(G.boundaries[1],spzeros(Int,1,n))],fill((0,),2n+1))
    geometry=[source.geometry[1:n];[[(4.,3.)]];source.geometry[n+1:end]]
    ds=TamerOp.DataIngestion.filtered_complex(disjoint;geometry)
    dr=C.circular_coordinates(ds;scale=0,kind=:essential)
    @test dr.anchors==[1,n+1]
    @test dr.values[end,1]==0
    @test C.coordinate_summary(dr).components==2
    @test_throws ArgumentError C.circular_coordinates(ds;scale=0,kind=:essential,anchors=[1])
    cloud=TamerOp.PointCloud([[cospi(2i/8),sinpi(2i/8)] for i in 0:7])
    filtration=TamerOp.RipsFiltration(max_dim=2)
    finite=C.circular_coordinates(cloud,filtration;scale=1.)
    @test C.check_coordinates(finite).valid
    @test only(finite.classes).kind===:finite
    @test_throws ArgumentError C.circular_coordinates(cloud,filtration;scale=2.)
    super=DI.filtered_complex(source.complex;geometry=source.geometry,orientation=(-1,))
    sr=C.circular_coordinates(super;scale=0,kind=:essential)
    @test C.check_coordinates(sr).valid
    @test sr.provenance.source_order===:superlevel
    square=TamerOp.PointCloud([0. 0.;1. 0.;1. 1.;0. 1.])
    collapsed=TamerOp.RipsFiltration(max_dim=2,construction=TamerOp.ConstructionOptions(collapse=:dominated_edges))
    dc=O.persistence_diagram(square,collapsed;source=true,cocycles=true,field=CM.F3())
    pulled=C.circular_coordinates(dc;scale=1.)
    @test C.check_coordinates(pulled).valid
    @test pulled.source===dc.retained_source.source
    @test isapprox(only(C.coordinate_summary(pulled).energy),1/4)
end

@testset "A97 integer lifting and obstruction" begin
    C=TamerOp.CohomologicalCoordinates;I=TamerOp.DataIngestion
    simplices=[[[i] for i in 1:4],[[1,2],[1,3],[2,3],[2,4],[3,4]],[[1,2,3]]]
    G=I._materialize_simplicial_output(simplices,[(0,) for _ in 1:10],I.FiltrationSpec(kind=:rips))[1]
    pts=[(0.,0.),(1.,0.),(0.,1.),(1.,1.)]
    source=I.filtered_complex(G;geometry=[[pts[v] for v in s] for ss in simplices for s in ss])
    a=reshape([1,-1,1,0,0],:,1)
    r=C.circular_coordinates(source;scale=0,cocycles=a,field=CM.F3())
    @test r.provenance.lift===:smith_normal_form_correction
    @test iszero(r.delta1*r.lifts)
    @test all(iszero,mod.(r.lifts-a,3))
    @test C.check_coordinates(r).valid
    @test_throws ArgumentError C.circular_coordinates(source;scale=0,cocycles=a,field=CM.Fp(5))
    @test_throws ArgumentError C.circular_coordinates(source;scale=0,cocycles=a,field=CM.F3(),budget=C.CoordinateBudget(max_integer_entries=1))
    # Moore space: one loop with a 3-fold attaching map. Its F3 class has
    # Bockstein obstruction; a real-valued pseudo-lift must not be accepted.
    G=DT.GradedComplex([[1],[1],[1]],[spzeros(Int,1,1),sparse(reshape([3],1,1))],[(0,),(0,),(0,)])
    moore=I.filtered_complex(G;geometry=[[(0,0)],[(0,0),(0,0)],[(0,0),(0,0),(0,0)]])
    @test_throws ArgumentError C.circular_coordinates(moore;scale=0,cocycles=ones(Int,1,1),field=CM.F3())
end

@testset "A97 torus class basis and independent circle factors" begin
    C=TamerOp.CohomologicalCoordinates
    source,a=coordinate_torus_fixture()
    r=C.toroidal_coordinates(source;scale=0,cocycles=a)
    @test size(r.values)==(20,2)
    @test C.check_coordinates(r).valid
    @test iszero(r.delta1*r.lifts)
    @test rank(r.harmonic)==2
    # A unimodular class change is a torus automorphism, after the same gauge.
    U=[1 2;0 1];q=C.toroidal_coordinates(source;scale=0,cocycles=a*U)
    @test isapprox(exp.(2pi*im*q.values),exp.(2pi*im*(r.values*U)))
    @test isapprox(q.harmonic,r.harmonic*U)
    @test_throws ArgumentError C.toroidal_coordinates(source;scale=0,cocycles=hcat(a[:,1],a[:,1]))
    r.values[1,1]=2
    @test !C.check_coordinates(r).valid
    @test_throws ArgumentError C.check_coordinates(r;throw=true)
end
