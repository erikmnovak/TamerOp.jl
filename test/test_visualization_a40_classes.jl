include("fixtures/slice_classes.jl")

@testset "A40 source classes and certified slice relations" begin
    V=TamerOp.Visualization;I=TamerOp.DataIngestion
    for field in (CM.QQField(),CM.F2(),CM.F3(),CM.PrimeField(101))
        f=slice_class_triangle(field);K=CM.coeff_type(field)
        s=V.inspection_session(f.enc;box=((0,0),(3,3)),slice=(basepoint=(0,0),direction=(1,1)),
            slice_classes=true,source_basis=f.basis,cache_limit=2)
        try
            r=s.slice_result
            @test [(x.birth,x.death,x.left_closed,x.right_closed) for x in r.intervals]==[(1,2,true,false)]
            V.select_inspection!(s;interval=1,class_parameter=3//2,pin_class=true)
            c=V.slice_class(s)
            @test c.point==(3//2,3//2)
            @test I.check_source_chain(c.source_cycle).valid
            @test c.source_cycle.coefficients==K[1,-1,1]
            @test c.source_filling.coefficients==K[1]
            @test f.source.complex.boundaries[2]*c.source_filling.coefficients==c.source_cycle.coefficients
            @test V.slice_correspondence(s).status===:unique_preimage
            misses=V.inspection_summary(s).slice_cache_misses
            V.select_inspection!(s;class_parameter=5//4)
            @test V.inspection_summary(s).slice_cache_misses==misses
            V.select_inspection!(s;slice=(basepoint=(1//4,0),direction=(1,1)))
            c=V.slice_correspondence(s)
            @test c.status===:unique_preimage
            @test c.target_images*c.coefficients==c.image
            @test I.check_source_chain(c.target_cycle).valid
            @test all(iszero,c.filling.coefficients)
            @test any(p->p.title=="Pinned source cycle",V.inspection_snapshot(s).panels)
            V.select_inspection!(s;class_parameter=5//2)
            @test V.slice_correspondence(s).status===:zero_at_join
            @test all(iszero,V.slice_correspondence(s).image)
            state=V.inspection_selection(s);snap=V.inspection_snapshot(s)
            @test_throws ArgumentError V.select_inspection!(s;class_parameter=100)
            @test V.inspection_selection(s)==state
            @test V.inspection_snapshot(s)===snap
            V.select_inspection!(s;clear_class=true,class_parameter=3//2)
            V.select_inspection!(s;slice=(basepoint=(0,0),direction=(1,1)),interval=1)
            @test V.inspection_summary(s).slice_cache_hits>=1
            @test s.slice_result===r
            @test_throws ArgumentError V.select_inspection!(s;member=2)
            @test_throws ArgumentError V.select_inspection!(s;class_parameter=2)
            @test V.check_inspection_session(s).valid
            V.reset_inspection!(s)
            @test V.slice_correspondence(s)===nothing
        finally
            V.close_inspection!(s)
        end
        bad=I.source_homology_basis(f.source;degree=1,basis=(label,p)->zeros(K,3,TamerOp.Results.encoding_module(f.enc).dims[label]))
        @test_throws ArgumentError V.inspection_session(f.enc;box=((0,0),(3,3)),
            slice=(basepoint=(0,0),direction=(1,1)),slice_classes=true,source_basis=bad)
    end
end

@testset "A40 exact translated-slice vineyards" begin
    V=TamerOp.Visualization;O=TamerOp.OrdinaryPersistence
    f=slice_class_triangle()
    path=V.slice_path(f.source;basepoints=[(0,0),(1,0),(0,1)],degree=1,representatives=true)
    @test V.check_persistence_path(path).valid
    @test path.mode===:certified
    for frame in V.path_frames(path)
        b=frame.context.basepoint
        @test frame.grades==[maximum(g[k]-b[k] for k in 1:2) for g in f.source.complex.grades]
        expected=(max(1-b[1],-b[2]),max(2-b[1],2-b[2]))
        @test [(r.birth,r.death) for r in frame.records if !r.zero]==[expected]
    end
    @test_throws ArgumentError V.slice_path(f.source;basepoints=[(0,0),(1,0)],direction=(1,0))
    @test_throws ArgumentError V.slice_path(f.source;basepoints=[(0,0),(1,0)],max_frames=1)
    @test_throws ArgumentError V.slice_path(f.source;basepoints=[(0,0),(1,0)],
        budget=TamerOp.Advanced.PersistenceUpdateBudget(max_entries=1))
    explorer=V.slice_explorer(f.source;degree=1)
    try
        @test [(r.birth,r.death) for r in explorer.slice_result.intervals]==[(1,2)]
        V.select_inspection!(explorer;interval=1,class_parameter=3//2)
        @test TamerOp.DataIngestion.check_source_chain(V.slice_class(explorer).source_cycle).valid
        @test V.check_inspection_session(explorer).valid
    finally;V.close_inspection!(explorer);end
end

@testset "A40 source rendering and saved selections" begin
    V=TamerOp.Visualization;f=slice_class_triangle(CM.F3())
    s=V.inspection_session(f.enc;box=((0,0),(3,3)),slice=(basepoint=(0,0),direction=(1,1)),
        slice_classes=true,source_basis=f.basis)
    try
        V.select_inspection!(s;interval=1,class_parameter=3//2,pin_class=true)
        snap=V.inspection_snapshot(s)
        @test count(p->p.kind===:source_chain_geometry,snap.panels)==3
        @test V.check_visual_spec(snap).valid
        saved=V.visual_state(s)
        @test saved.request.family===:snapshot
        @test saved.snapshot.metadata.selected_slice_class.source_cycle_data.coefficients==(1,2,1)
        for backend in (:cairomakie,:wglmakie)
            ext=Base.get_extension(TamerOp,backend===:cairomakie ? :TamerOpCairoMakieExt : :TamerOpWGLMakieExt)
            ext===nothing && continue
            fig=V.render(snap;backend)
            M=backend===:cairomakie ? ext.CairoMakie.Makie : ext.WGLMakie.Makie
            @test length(filter(x->x isa M.Axis,fig.content))>=3
            empty!(fig)
        end
        if Base.get_extension(TamerOp,:TamerOpCairoMakieExt)!==nothing
            mktempdir() do dir
                for suffix in ("png","svg","pdf")
                    path=joinpath(dir,"slice.$suffix");V.save_visual(path,snap;backend=:cairomakie)
                    @test filesize(path)>1000
                end
                path=joinpath(dir,"slice.html")
                V.save_visual_session(path,(;selected=saved))
                @test filesize(path)>1000
            end
            if haskey(ENV,"TAMEROP_A40_HTML")
                V.select_inspection!(s;slice=(basepoint=(1//4,0),direction=(1,1)))
                V.save_visual_session(ENV["TAMEROP_A40_HTML"],(;selected=saved,translated=V.visual_state(s)))
            end
        end
    finally;V.close_inspection!(s);end
end

@testset "A40 multiplicity, nonunique preimages and exact field contracts" begin
    V=TamerOp.Visualization
    for field in (CM.QQField(),CM.F3())
        K=CM.coeff_type(field);P=FF.ProductOfChainsPoset((2,2))
        # Two basis vectors become equal at the upper corner.
        ds=[2,2,2,1]
        maps=Dict((a,b)=>(b==4 ? K[1 1] : CM.eye(field,2)) for (a,b) in FF.cover_edges(P))
        M=MD.PModule{K}(P,ds,maps;field)
        enc=RES.EncodingResult(P,M,EC.GridEncodingMap(P,([0,1],[0,1])))
        s=V.inspection_session(enc;box=((0,0),(2,2)),slice=(basepoint=(0,0),direction=(1,0)),slice_classes=true)
        try
            @test only(s.slice_result.intervals).multiplicity==2
            V.select_inspection!(s;interval=1,member=2,class_parameter=3//2,pin_class=true)
            V.select_inspection!(s;slice=(basepoint=(0,0),direction=(0,1)))
            c=V.slice_correspondence(s)
            @test c.status===:ambiguous_preimages
            @test size(c.kernel)==(2,1)
            @test all(iszero,c.target_images*c.kernel)
            @test c.target_images*c.coefficients==c.image
            @test c.source_point==(3//2,0)
            @test c.target_point==(0,3//2)
            @test c.join_point==(3//2,3//2)
            @test V.check_visual_spec(V.inspection_snapshot(s)).valid
        finally;V.close_inspection!(s);end
    end
end

@testset "A40 explicit bases, multicritical scalarization and budgets" begin
    V=TamerOp.Visualization;I=TamerOp.DataIngestion;Z=TamerOp.ZigzagPersistence
    @test TamerOp.slice_explorer===V.slice_explorer
    @test TamerOp.slice_path===V.slice_path
    @test TamerOp.source_homology_basis===I.source_homology_basis
    f=slice_class_triangle(CM.F3());K=CM.coeff_type(CM.F3())
    @test_throws ArgumentError V.slice_explorer(f.source;class_budget=Z.ZigzagBudget(max_vertices=2))
    @test_throws ArgumentError V.inspection_session(f.enc;box=((0,0),(3,3)),class_budget=nothing)
    # A complete basis can still use an incorrect coordinate identification.
    bad=I.source_homology_basis(f.source;degree=1,basis=(label,p)->
        TamerOp.Results.encoding_module(f.enc).dims[label]==0 ? zeros(K,3,0) :
        reshape((p[1]<3//2 ? K(1) : K(2)).*K[1,-1,1],3,1))
    @test_throws ArgumentError V.inspection_session(f.enc;box=((0,0),(3,3)),
        slice=(basepoint=(0,0),direction=(1,1)),slice_classes=true,source_basis=bad)
    s=V.inspection_session(f.enc;box=((0,0),(3,3)))
    try
        @test_throws ArgumentError V.select_inspection!(s;member=1)
        @test_throws ArgumentError V.select_inspection!(s;pin_class=:yes)
    finally;V.close_inspection!(s);end
    grades=[[(0,0)] for _ in 1:3]
    append!(grades,[[(1,0),(0,2)] for _ in 1:3]);push!(grades,[(3,3)])
    G=DT.MultiCriticalGradedComplex([[11,12,13],[21,22,23],[31]],copy(f.source.complex.boundaries),grades)
    source=I.filtered_complex(G;geometry=f.source.geometry)
    p=V.slice_path(source;basepoints=[(0,0),(3,0)],degree=1)
    @test V.check_persistence_path(p).valid
    for frame in V.path_frames(p)
        b=frame.context.basepoint
        expected=[minimum(maximum(g[k]-b[k] for k in 1:2) for g in cell) for cell in grades]
        @test frame.grades==expected
    end
end

@testset "A40 absent preimages and observed window ends" begin
    V=TamerOp.Visualization;F=CM.F2();K=CM.coeff_type(F)
    P=FF.ProductOfChainsPoset((2,2));dims=[0,1,0,1]
    maps=Dict((a,b)=>ones(K,dims[b],dims[a]) for (a,b) in FF.cover_edges(P))
    M=MD.PModule{K}(P,dims,maps;field=F)
    enc=RES.EncodingResult(P,M,EC.GridEncodingMap(P,([0,1],[0,1])))
    s=V.inspection_session(enc;box=((0,0),(2,2)),slice=(basepoint=(0,0),direction=(1,0)),slice_classes=true)
    try
        V.select_inspection!(s;interval=1,class_parameter=3//2,pin_class=true)
        @test V.slice_class(s).source_filling===nothing
        V.select_inspection!(s;slice=(basepoint=(0,0),direction=(0,1)))
        c=V.slice_correspondence(s)
        @test c.status===:no_corresponding_class
        @test c.coefficients===nothing
        @test size(c.target_images)==(1,0)
        @test c.image==K[1]
    finally;V.close_inspection!(s);end
end
