@testset "A55 actual union zigzags and source representatives" begin
    T=TamerOp.TemporalPersistence;Z=TamerOp.ZigzagPersistence
    for F in (CM.F2(),CM.F3(),CM.QQField())
        # Two observations of the same separated sites, with actual shared IDs.
        clouds=[TamerOp.PointCloud([0. 0.;2. 0.]),TamerOp.PointCloud([0. 0.;2. 0.])]
        r=T.temporal_persistence(clouds;point_ids=[[1,2],[1,2]],scales=[0,1,2],field=F,representatives=true)
        @test T.check_temporal(r).valid
        @test T.temporal_axes(r).directions==[:forward,:backward]
        @test T.temporal_axes(r).times==[1,3//2,2]
        @test r.module_object.dims==[2,2,2,2,2,2,1,1,1]
        @test length(Z.zigzag_intervals(T.temporal_barcode(r;scale_index=1)))==2
        @test T.temporal_barcode(r;scale_index=3).supports==[(1,3)]
        @test size(T.temporal_map(r;source=(3,1),target=(2,3)))==(1,2)
        @test_throws ArgumentError T.temporal_map(r;source=(2,1),target=(1,1))
        c=T.temporal_cycle(r;scale_index=1,interval=1,occurrence=2)
        @test TamerOp.DataIngestion.check_source_chain(c).valid
        @test count(!iszero,c.coefficients)==1
        # The spatial map merges two classes. A temporal barcode-only landscape
        # would miss that constraint on its closed two-dimensional square.
        l=T.spatiotemporal_landscape(r;layers=1:3,boundary=:zero)
        @test l.values[2,2,:]==[1,0,0]
        @test !any(l.censored)
        @test all(iszero,l.values[1,:,:])
        censored=T.spatiotemporal_landscape(r;layers=1:2)
        @test censored.censored[2,2,1]
        @test !censored.censored[2,2,2]
        @test censored.censored[1,1,2]
        @test T.temporal_landscape_values(l).values==l.values
        @test TamerOp.temporal_persistence===T.temporal_persistence
        @test TamerOp.Advanced.TemporalBudget===T.TemporalBudget
    end
    # Disjoint snapshots acquire a shared connected component only at the union.
    r=T.temporal_persistence([[(0,0)],[(1,0)]];scales=[0,1],representatives=true)
    @test sort(T.temporal_barcode(r;scale_index=1).supports)==[(1,2),(2,3)]
    @test T.temporal_barcode(r;scale_index=2).supports==[(1,3)]
    @test r.provenance.occurrence_kinds==[:source,:union,:source]
    @test r.sources[2].sources==[(1,),(2,),(1,2)]
end

@testset "A55 delay windows, repeated labels, empty clouds and bounds" begin
    T=TamerOp.TemporalPersistence
    c=T.delay_clouds([0,1,0,-1,0,1];dimension=2,delay=1,window=3,stride=1)
    @test length(c.clouds)==3
    @test c.point_ids==[[1,2,3],[2,3,4],[3,4,5]]
    @test c.clouds[1].coords==[0. 1.;1. 0.;0. -1.]
    r=T.temporal_persistence(c;scales=[0,1,2],degree=1,representatives=true)
    @test T.check_temporal(r).valid
    @test length(r.sources)==5
    @test r.provenance.embedding.construction===:delay_windows
    empty=TamerOp.PointCloud(zeros(0,2))
    r=T.temporal_persistence([empty,empty];scales=[0,1],times=[7,7])
    @test all(iszero,r.module_object.dims)
    @test r.times==[7,7,7]
    @test isempty(T.temporal_barcode(r;scale_index=1).supports)
    @test all(iszero,T.spatiotemporal_landscape(r).values)
    @test_throws ArgumentError T.temporal_cycle(r;scale_index=1,interval=1,occurrence=1)
    @test_throws ArgumentError T.delay_clouds([1,2];dimension=3,window=2)
    @test_throws ArgumentError T.temporal_persistence([[(0,0)],[(1,0)]];scales=[0],point_ids=[[1],[1]])
    @test_throws ArgumentError T.temporal_persistence([[(0,0)]];scales=[0,0])
    @test_throws ArgumentError T.temporal_persistence([[(0,0)]];scales=[-1,0])
    @test_throws ArgumentError T.temporal_persistence([[(0,0)],[(1,0)]];scales=[0],budget=T.TemporalBudget(max_occurrences=2))
    @test_throws ArgumentError T.temporal_persistence([[(0,0),(1,0)]];scales=[0],budget=T.TemporalBudget(max_points=1))
    @test_throws ArgumentError T.temporal_persistence([[(0,0),(1,0)]];scales=[0],budget=T.TemporalBudget(max_matrix_entries=1))
end

@testset "A55 supplied filtered chain maps and genuine two-dimensional ranks" begin
    T=TamerOp.TemporalPersistence;I=TamerOp.DataIngestion
    # A single degree-zero generator in each source. The first horizontal map
    # is zero at every scale. Dimensions alone cannot supply this landscape.
    G=DT.GradedComplex([[1]],SparseMatrixCSC{Int,Int}[],[(0,)])
    x=I.filtered_complex(G;geometry=[[(0,0)]])
    F=CM.F3();K=CM.coeff_type(F)
    r=T.temporal_persistence([x,x,x];scales=[0,1,2],directions=[:forward,:backward],
        chain_maps=[[zeros(K,1,1)],[ones(K,1,1)]],field=F,representatives=true)
    @test T.check_temporal(r).valid
    l=T.spatiotemporal_landscape(r;layers=[1],boundary=:zero)
    @test l.values[2,2,1]==0
    @test all(==(1),r.module_object.dims)
    @test_throws ArgumentError T.spatiotemporal_landscape(r;layers=[0])
    @test_throws ArgumentError T.spatiotemporal_landscape(r;boundary=:truncate)
    @test_throws ArgumentError T.spatiotemporal_landscape(r;budget=TamerOp.Invariants.GeneralizedRankBudget(max_queries=1))
    late=I.filtered_complex(DT.GradedComplex([[1]],SparseMatrixCSC{Int,Int}[],[(2,)]);geometry=[[(0,0)]])
    @test_throws ArgumentError T.temporal_persistence([x,late];scales=[0,1,2],directions=[:forward],chain_maps=[[ones(Int,1,1)]])
end

@testset "A55 H1 disappears at union bridges and returns in a later state" begin
    T=TamerOp.TemporalPersistence;I=TamerOp.DataIngestion
    square=[(0.,0.),(1.,0.),(1.,1.),(0.,1.)]
    r=T.temporal_persistence([square,[square;[(0.5,0.5)]],square];point_ids=[[1,2,3,4],[1,2,3,4,5],[1,2,3,4]],scales=[0,1,2],degree=1,representatives=true)
    @test T.check_temporal(r).valid
    @test r.barcodes[2].supports==[(1,1),(5,5)]
    @test all(iszero,r.module_object.dims[11:15])
    for (interval,occurrence) in ((1,1),(2,5))
        c=T.temporal_cycle(r;scale_index=2,interval,occurrence)
        @test I.check_source_chain(c).valid
        @test count(!iszero,c.coefficients)==4
    end
    x=r.sources[1];counts=DT.cell_counts(x.complex);K=CM.coeff_type(CM.F3())
    maps=[CM.eye(CM.F3(),c) for c in counts];maps[1][1,1]=K(2)
    @test_throws ArgumentError T.temporal_persistence([x,x];scales=[0,1,2],directions=[:forward],chain_maps=[maps],field=CM.F3())
end
