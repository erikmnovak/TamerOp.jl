# Independent source boundary/activity oracles, including characteristic != 2.
function _a43_triangle(;N=1,filled=true,death=2)
    D=TamerOp.DataTypes;I=TamerOp.DataIngestion
    B1=sparse([-1 -1 0;1 0 -1;0 1 1])
    B2=filled ? sparse(reshape([1,-1,1],3,1)) : spzeros(Int,3,0)
    G=D.GradedComplex([[1,2,3],[1,2,3],filled ? [1] : Int[]],[B1,B2],
        [ntuple(_->g,N) for g in vcat(zeros(Int,3),ones(Int,3),filled ? [death] : Int[])])
    v=[(0,0),(2,0),(1,2)]
    geometry=vcat([[x] for x in v],[[v[1],v[2]],[v[1],v[3]],[v[2],v[3]]],filled ? [v] : Vector{Tuple{Int,Int}}[])
    return I.filtered_complex(G;geometry)
end

@testset "A43 source cycles, cocycles and death fillings" begin
    I=TamerOp.DataIngestion;O=TamerOp.OrdinaryPersistence;V=TamerOp.Visualization
    x=_a43_triangle()
    for F in (CM.F2(),CM.F3(),CM.Fp(5),CM.Fp(101))
        K=CM.coeff_type(F)
        d=O.persistence_diagram(x;field=F,representatives=true,cocycles=true)
        @test O.finite_intervals(d;dim=1)==[(1,2)]
        @test O.check_persistence_diagram(d).valid
        r=O.persistence_representative(d;dim=1,scale=3//2)
        z=r.source_cycle;f=r.source_bounding_chain
        @test I.check_source_chain(z).valid
        @test z.parameter==(3//2,)
        @test f.parameter==(2,)
        @test iszero(K.(x.complex.boundaries[1])*z.coefficients)
        @test K.(x.complex.boundaries[2])*f.coefficients==z.coefficients
        @test all(i->iszero(z.coefficients[i]) || x.complex.grades[3+i][1]<=3//2,1:3)
        coc=O.persistence_cocycle(d;dim=1,scale=3//2).source_cochain
        @test I.check_source_chain(coc).valid
        @test any(!iszero,coc.coefficients)
        @test !iszero(sum(z.coefficients .* coc.coefficients))
        @test_throws ArgumentError O.persistence_representative(d;dim=1,scale=2)
        @test_throws ArgumentError O.persistence_cocycle(d;dim=1,scale=2)
        for chain in (z,f,coc)
            spec=V.visual_spec(chain)
            @test V.check_visual_spec(spec).valid
            @test spec.metadata.source_chain.field==F
            @test spec.metadata.source_chain.coefficients==I.source_chain_data(chain).coefficients
            @test spec.metadata.optimization==:not_run_by_plot
        end
        budget=V.visual_spec(z;max_cells=1,max_labels=0)
        @test length(budget.metadata.displayed_support)==1
        @test budget.metadata.omitted_support==2
        s=V.inspection_session(d;dim=1)
        @test length(V.inspection_snapshot(s).panels)==3
        V.select_inspection!(s;interval=1,member=1,representative=true)
        @test length(V.inspection_snapshot(s).panels)==5
        @test V.inspection_snapshot(s).metadata.selected_representative.source_geometry==:verified_source_cells
        V.reset_inspection!(s)
        @test length(V.inspection_snapshot(s).panels)==3
        V.close_inspection!(s)
        restored=O.persistence_diagram_from_data(O.persistence_diagram_data(d))
        @test O.check_persistence_diagram(restored).valid
        @test O.persistence_representative(restored;dim=1).source_cycle.coefficients==z.coefficients
    end
    plain=O.persistence_diagram(x.complex;representatives=true)
    @test O.persistence_representative(plain;dim=1).source_cycle===nothing
    cheap=O.persistence_diagram(x)
    @test cheap.retained_representatives===nothing
    @test cheap.retained_cocycles===nothing
end

@testset "A43 exact filtered contraction and contravariance" begin
    I=TamerOp.DataIngestion;D=TamerOp.DataTypes
    # A triangle collapses along edge (1,3) to its two-edge path. The removed
    # edge retracts to a signed two-edge chain, essential over odd fields.
    original=_a43_triangle(;death=1)
    G=D.GradedComplex([[1,2,3],[1,3],Int[]],[original.complex.boundaries[1][:,[1,3]],spzeros(Int,2,0)],[(0,),(0,),(0,),(1,),(1,)])
    reduced=I.filtered_complex(G;geometry=original.geometry[[1,2,3,4,6]],source_ids=original.sources[[1,2,3,4,6]],model=:reduced)
    for F in (CM.F2(),CM.F3(),CM.Fp(5))
        K=CM.coeff_type(F);c=I.source_correspondence(reduced,original;field=F)
        @test I.check_source_correspondence(c).valid
        @test c.retraction[2][:,2]==K.([1,1])
        @test c.retraction[2]*c.lift[2]==K.([1 0;0 1])
        @test K.(G.boundaries[1])*c.retraction[2]==c.retraction[1]*K.(original.complex.boundaries[1])
        co=I.lift_source_chain(c;dim=1,coefficients=[1,0],parameter=1,kind=:cocycle)
        @test co.coefficients==K.([1,1,0])
        @test iszero(transpose(K.(original.complex.boundaries[2]))*co.coefficients)
        @test_throws ArgumentError I.lift_source_chain(c;dim=1,coefficients=[1,0],parameter=0,kind=:cocycle)
        @test_throws ArgumentError I.lift_source_chain(c;dim=1,coefficients=[1,0],parameter=1,kind=:cycle)
        bad=deepcopy(c.lift);bad[2][1,1]=-one(K)
        F.p==2 || @test_throws ArgumentError I.source_correspondence(reduced,original;field=F,lift=bad)
    end
    @test_throws ArgumentError I.source_correspondence(reduced,_a43_triangle(;death=2))
    # Genuine multiparameter supplied chain map with no assumed retraction.
    multi=_a43_triangle(;N=2,filled=false)
    c=I.source_correspondence(multi;field=CM.F3())
    z=I.lift_source_chain(c;dim=1,coefficients=[1,-1,1],parameter=(1,2))
    @test I.check_source_chain(z).valid
    @test_throws ArgumentError I.lift_source_chain(c;dim=1,coefficients=[1,-1,1],parameter=(0,2))
    bad=deepcopy(c);bad.lift[2][1,1]=CM.coeff_type(CM.F3())(0)
    @test !I.check_source_correspondence(bad).valid
    # An explicit map is allowed to send a class to zero; it cannot masquerade
    # as class-preserving ordinary retention without a splitting certificate.
    x=_a43_triangle();n=I.source_correspondence(x;field=CM.F3())
    maps=[2M for M in n.lift]
    split=I.source_correspondence(x;field=CM.F3(),lift=maps,retraction=maps)
    @test I.check_source_correspondence(split).valid
    general=I.source_correspondence(x;field=CM.F3(),lift=maps)
    @test_throws ArgumentError I.lift_source_chain(general;dim=1,coefficients=[1,0,0],parameter=1,kind=:cocycle)
    @test_throws ArgumentError TamerOp.persistence_diagram(general;representatives=true)
    # Multicritical union activity, including incomparable appearances.
    grades=[[(0,0)] for _ in 1:3]
    append!(grades,[[ (1,3),(3,1)] for _ in 1:3])
    G=D.MultiCriticalGradedComplex([collect(v) for v in multi.complex.cells_by_dim],multi.complex.boundaries,grades)
    x=I.filtered_complex(G;geometry=multi.geometry)
    c=I.source_correspondence(x)
    @test I.check_source_chain(I.lift_source_chain(c;dim=1,coefficients=[1,1,1],parameter=(3,1))).valid
    @test_throws ArgumentError I.lift_source_chain(c;dim=1,coefficients=[1,1,1],parameter=(2,2))
    # Direct finite/multiparameter algebra often uses QQ, even though ordinary
    # prime-field persistence and its finite optimization are separate routes.
    rational=I.source_correspondence(multi;field=CM.QQField())
    rz=I.lift_source_chain(rational;dim=1,coefficients=[1//2,-1//2,1//2],parameter=(1,1))
    @test I.check_source_chain(rz).valid
    @test I.source_chain_data(rz).coefficients==(1//2,-1//2,1//2)
    @test TamerOp.Visualization.check_visual_spec(TamerOp.Visualization.visual_spec(rz)).valid
    @test_throws ArgumentError I.optimize_representative(rz)
end

@testset "A43 bounded minimum weighted support" begin
    I=TamerOp.DataIngestion;D=TamerOp.DataTypes
    # Three parallel cellular loops with two homologizing 2-cells. All cycles
    # are homologous multiples, and the minimum is independently enumerable.
    G=D.GradedComplex([[1],[1,2,3],[1,2]],[spzeros(Int,1,3),sparse([-1 0;1 -1;0 1])],[(0,) for _ in 1:6])
    geom=[[(0,0)],[(0,0),(1,0)],[(0,0),(0,1)],[(0,0),(-1,0)],[(0,0),(1,0),(0,1)],[(0,0),(0,1),(-1,0)]]
    x=I.filtered_complex(G;geometry=geom)
    for F in (CM.F2(),CM.F3(),CM.Fp(5))
        K=CM.coeff_type(F);c=I.source_correspondence(x;field=F)
        z=I.lift_source_chain(c;dim=1,coefficients=[1,0,0],parameter=0)
        out=I.optimize_representative(z;weights=[5,2,1])
        @test out.coefficients==K.([0,0,1])
        @test out.metadata.objective==1
        @test out.metadata.candidates==F.p^2
        @test K.(G.boundaries[2])*K.(collect(out.metadata.boundary_adjustment))==out.coefficients-z.coefficients
        # Oracle ranges over ALL boundary chains, without using the chosen basis.
        candidates=[z.coefficients+K.(G.boundaries[2])*K.([a,b]) for a in 0:F.p-1,b in 0:F.p-1]
        cost(v)=sum([5,2,1][findall(!iszero,v)];init=0)
        @test cost(out.coefficients)==minimum(cost.(candidates))
        @test_throws ArgumentError I.optimize_representative(z;max_candidates=F.p^2-1)
        @test_throws ArgumentError I.optimize_representative(z;weights=[1,-1,1])
        zero_class=I.lift_source_chain(c;dim=1,coefficients=[-1,1,0],parameter=0)
        @test_throws ArgumentError I.optimize_representative(zero_class)
    end
end

@testset "A43 periodic cubical original boundary equations" begin
    I=TamerOp.DataIngestion;O=TamerOp.OrdinaryPersistence;V=TamerOp.Visualization
    for input in (:top_cells,:vertices),shape in ((1,1),(2,3)),order in (:sublevel,:superlevel),F in (CM.F2(),CM.F3())
        d=O.cubical_persistence(fill(2//3,shape);periodic=true,input,order,field=F,representatives=true,cocycles=true,source=true)
        @test O.check_torus_persistence(d).valid
        for dim in 0:2, index in eachindex(O.essential_births(d;dim))
            r=O.persistence_representative(d;dim,kind=:essential,index)
            @test I.check_source_chain(r.source_cycle).valid
            @test r.source_cycle.parameter==(2//3,)
            @test r.source_cycle.source.provenance.periodic==(true,true)
            @test V.check_visual_spec(V.visual_spec(r.source_cycle)).valid
            co=O.persistence_cocycle(d;dim,kind=:essential,index,scale=2//3)
            @test I.check_source_chain(co.source_cochain).valid
        end
    end
    ring=[0 0 0;0 2 0;0 0 0]
    d=O.cubical_persistence(ring;representatives=true,source=true)
    r=O.persistence_representative(d;dim=1)
    @test r.interval==(0,2)
    @test r.source_bounding_chain.parameter==(2,)
    @test all(s->haskey(s,:origin)&&haskey(s,:mask),I.source_chain_data(r.source_cycle).source_ids)
end

@testset "A43 ingestion source retention" begin
    I=TamerOp.DataIngestion;O=TamerOp.OrdinaryPersistence;D=TamerOp.DataTypes
    pc=D.PointCloud([0. 0.;1. 0.;1. 1.;0. 1.])
    f=I.RipsFiltration(;max_dim=2)
    d=O.persistence_diagram(pc,f;source=true,representatives=true,cocycles=true,field=CM.F3())
    @test O.check_persistence_diagram(d).valid
    r=O.persistence_representative(d;dim=1)
    @test r.source_geometry==:verified_source_cells
    @test all(s->all(v->v in 1:4,s),I.source_chain_data(r.source_cycle).source_ids)
    fc=I.RipsFiltration(;max_dim=2,construction=TamerOp.Options.ConstructionOptions(collapse=:dominated_edges))
    dc=O.persistence_diagram(pc,fc;source=true,representatives=true,cocycles=true,field=CM.F3())
    @test O.finite_intervals(dc;dim=1)==O.finite_intervals(d;dim=1)
    @test I.check_source_correspondence(dc.retained_source).valid
    @test O.check_persistence_diagram(dc).valid
    @test_throws ArgumentError O.persistence_diagram(pc,f;source=true,method=:implicit)
    @test O.persistence_diagram(pc,f).retained_source===nothing
    landmarks=I.LandmarkRipsFiltration(;landmarks=[4,2,1],max_dim=2)
    ld=O.persistence_diagram(pc,landmarks;source=true,representatives=true,cocycles=true)
    @test Set(s[1] for s in ld.retained_source.source.sources[1:3])==Set([4,2,1])
    for i in 1:3
        v=only(ld.retained_source.source.sources[i])
        @test only(ld.retained_source.source.geometry[i])==Tuple(D.point_matrix(pc)[v,:])
    end
    @test_throws ArgumentError O.persistence_diagram([0. 1.;1. 0.],f;source=true)
    graph=D.GraphData(4,[(2,4)];weights=[1.],coords=[0. 0.;2. 1.;4. 3.;6. 2.])
    weighted=I.EdgeWeightedFiltration(vertex_births=[10.,0.,10.,0.],threshold=2.,max_dim=1)
    wd=O.persistence_diagram(graph,weighted;source=true,representatives=true)
    @test wd.retained_source.source.sources==[(2,),(4,),(2,4)]
    @test wd.retained_source.source.geometry[1]==[(2.,1.)]
    @test wd.retained_source.source.geometry[2]==[(6.,2.)]
end

@testset "A43 source interchange and rejected corrupted maps" begin
    O=TamerOp.OrdinaryPersistence;I=TamerOp.DataIngestion;S=TamerOp.Serialization
    fixtures=[O.persistence_diagram(_a43_triangle();field=CM.F3(),representatives=true,cocycles=true),
        O.cubical_persistence(fill(1//3,1,1);periodic=true,source=true,representatives=true,cocycles=true),
        O.persistence_diagram(TamerOp.DataTypes.PointCloud([0. 0.;1. 0.;1. 1.;0. 1.]),
            I.RipsFiltration(max_dim=2);source=true,representatives=true,cocycles=true)]
    mktempdir() do dir
        for (i,d) in enumerate(fixtures)
            path=joinpath(dir,"source-$i.json")
            S.save_persistence_diagram_json(path,d)
            loaded=S.load_persistence_diagram_json(path)
            @test O.check_persistence_diagram(loaded).valid
            @test isequal(O.persistence_diagram_data(loaded),O.persistence_diagram_data(d))
        end
    end
    data=O.persistence_diagram_data(first(fixtures))
    corrupt=deepcopy(data)
    corrupt.source.lift[2].entries[3][1]=0
    @test_throws ArgumentError O.persistence_diagram_from_data(corrupt)
    corrupt=deepcopy(data)
    corrupt.representatives.finite[2][1].cycle.coefficients[1]=2
    @test_throws ArgumentError O.persistence_diagram_from_data(corrupt)
    # A general chain map cannot acquire ordinary class-preserving semantics
    # merely by passing through the interchange constructor.
    unsigned=merge(data,(;source=merge(data.source,(;retraction=nothing,
        certificate=:supplied_filtered_chain_map))))
    @test_throws ArgumentError O.persistence_diagram_from_data(unsigned)
    # A one-dimensional periodic vertex input is a genuine circle, drawn with
    # an explicit zero second coordinate rather than fabricated source labels.
    one=O.cubical_persistence([0];input=:vertices,periodic=true,source=true,representatives=true)
    r=O.persistence_representative(one;dim=1,kind=:essential)
    @test I.check_source_chain(r.source_cycle).valid
    @test only(I.source_chain_data(r.source_cycle).geometry)==[(1,0),(2,0)]
end

@testset "A113 source correspondence in higher-dimensional cubical input" begin
    I=TamerOp.DataIngestion;O=TamerOp.OrdinaryPersistence;S=TamerOp.Serialization
    for shape in ((3,), (2,1,2), (1,1,1,1)),order in (:sublevel,:superlevel),p in (2,3,101)
        N=length(shape);periodic=ntuple(isodd,N)
        values=reshape([mod(i,2) for i in 1:prod(shape)],shape)
        d=O.cubical_persistence(values;periodic,order,field=CM.Fp(p),source=true,
            representatives=true,cocycles=true)
        @test I.check_source_correspondence(d.retained_source;throw=true).valid
        model=d.retained_source.source
        @test model.provenance.image_values==:top_cells
        @test model.provenance.source_field_dimension==N
        @test model.provenance.periodic==periodic
        G=I.graded_complex(model)
        for (j,cell) in enumerate(model.sources)
            @test length(cell.origin)==length(cell.mask)==N
            if all(==(1),cell.mask)
                @test G.grades[j][1]==values[cell.origin...]
            end
        end
        for dim in 0:N,kind in (:finite,:essential)
            bars=kind==:finite ? O.finite_intervals(d;dim) : O.essential_births(d;dim)
            for index in eachindex(bars)
                r=O.persistence_representative(d;dim,kind,index)
                @test I.check_source_chain(r.source_cycle;throw=true).valid
                r.source_bounding_chain===nothing || @test I.check_source_chain(r.source_bounding_chain;throw=true).valid
                c=O.persistence_cocycle(d;dim,kind,index,scale=first(r.interval))
                @test I.check_source_chain(c.source_cochain;throw=true).valid
            end
        end
        mktempdir() do folder
            path=joinpath(folder,"volume.json")
            S.save_persistence_diagram_json(path,d)
            restored=S.load_persistence_diagram_json(path)
            @test I.check_source_correspondence(restored.retained_source;throw=true).valid
            @test restored.retained_source.source.sources==model.sources
            @test restored.retained_source.source.geometry==model.geometry
            @test all(O.persistence_intervals(restored;dim)==O.persistence_intervals(d;dim) for dim in 0:N)
        end
    end
end
