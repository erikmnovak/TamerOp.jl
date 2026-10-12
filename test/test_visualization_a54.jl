# Region and boundary expectations come from explicit interval summands.
function _a54_visual_sum(nx,ny,supports;field=CM.QQField(),axes=(collect(0:nx-1),collect(0:ny-1)))
    P=FF.ProductOfChainsPoset((nx,ny));K=CM.coeff_type(field)
    active=[findall(s->q in s,supports) for q in 1:nx*ny]
    maps=Dict{Tuple{Int,Int},Matrix{K}}(e=>K[i==j for i in active[e[2]],j in active[e[1]]] for e in FF.cover_edges(P))
    M=MD.PModule{K}(P,length.(active),maps;field)
    grid=TamerOp.EncodingCore.GridEncodingMap(P,axes)
    TamerOp.EncodingResult(P,M,grid)
end
function _a54_run_bar_oracle(path,supports)
    bars=Dict{Tuple{Int,Int},Int}()
    for support in supports
        start=0
        for i in 1:length(path)+1
            active=i<=length(path) && path[i] in support
            if active && start==0;start=i
            elseif !active && start!=0
                bars[start,i-1]=get(bars,(start,i-1),0)+1;start=0
            end
        end
    end
    bars
end

@testset "A54 visual boundary barcode independent interval oracle" begin
    I=TamerOp.Invariants;EC=TamerOp.EncodingCore
    P=FF.ProductOfChainsPoset((2,3))
    constant=_a54_visual_sum(2,3,[collect(1:6)])
    M=TamerOp.Results.encoding_module(constant)
    regions=[findall(i->mask & (1<<(i-1)) != 0,1:6) for mask in 1:63]
    filter!(r->I.check_generalized_rank_region(M;vertices=r).valid,regions)
    supports=[[1,2,3,4,5,6],[2,4,6],[3,4,5,6],[3,5],[4]]
    for field in (CM.QQField(),CM.F2(),CM.F3(),CM.Fp(5))
        enc=_a54_visual_sum(2,3,supports;field)
        for region in regions
            result=I.boundary_zigzag(TamerOp.Results.encoding_module(enc),enc.pi;vertices=region)
            path=I.boundary_path(result)
            expected=_a54_run_bar_oracle(path.vertices,supports)
            @test Dict((a,b)=>m for (a,b,m) in I.boundary_bars(result))==expected
            @test I.generalized_rank(result)==count(s->issubset(region,s),supports)
            @test I.check_boundary_zigzag(result).valid
            @test all(i->path.arrows[i]==1 ? FF.leq(P,path.vertices[i],path.vertices[i+1]) :
                FF.leq(P,path.vertices[i+1],path.vertices[i]),1:length(path.arrows))
        end
    end
    enc=_a54_visual_sum(3,3,[collect(1:9),[2,3,5,6],[4,5,7,8]])
    result=I.boundary_zigzag(TamerOp.Results.encoding_module(enc),enc.pi;vertices=2:8)
    @test I.boundary_path(result).vertices==[2,5,4,8,5,6]
    @test I.boundary_path(result).arrows==[1,-1,1,-1,1]
    @test I.generalized_rank(result)==1
    @test TamerOp.generalized_rank(result)==1
    @test I.describe(result).occurrences==6
    @test I.describe(result).distinct_labels==5
    empty!(I.boundary_path(result).vertices)
    @test length(I.boundary_path(result).vertices)==6
    @test occursin("BoundaryZigzagResult",sprint(show,result))
    for region in (Int[],[1,9],[1,1])
        @test_throws ArgumentError I.boundary_zigzag(TamerOp.Results.encoding_module(enc),enc.pi;vertices=region)
    end
    for budget in (I.GeneralizedRankBudget(max_queries=2),I.GeneralizedRankBudget(max_vertices=4),
            I.GeneralizedRankBudget(max_dimension=1),I.GeneralizedRankBudget(max_matrix_entries=1))
        @test_throws ArgumentError I.boundary_zigzag(TamerOp.Results.encoding_module(enc),enc.pi;vertices=2:8,budget)
    end
    # Finite characteristic changes the intersection of the two source lines.
    for field in (CM.QQField(),CM.F2(),CM.F3())
        K=CM.coeff_type(field);p=field isa CM.QQField ? 3 : field.p
        P=FF.ProductOfChainsPoset((2,2))
        N=MD.PModule{K}(P,[0,1,1,2],Dict((1,2)=>zeros(K,1,0),(1,3)=>zeros(K,1,0),
            (2,4)=>reshape(K[1,0],2,1),(3,4)=>reshape(K[1,p],2,1));field)
        r=I.boundary_zigzag(N,EC.GridEncodingMap(P,([0,1],[0,1]));vertices=[2,3,4])
        @test I.generalized_rank(r)==(field isa CM.QQField ? 0 : 1)
    end
    # Three distinct lines in one plane give a non-interval-decomposable module.
    # Its cap is L1 -> V <- L2 -> V <- L3 -> V, with this hand-derived barcode.
    for field in (CM.QQField(),CM.F2(),CM.F3(),CM.Fp(5))
        K=CM.coeff_type(field);P=FF.ProductOfChainsPoset((3,3))
        dims=[0,0,1,0,1,2,1,2,2]
        maps=Dict(e=>zeros(K,dims[e[2]],dims[e[1]]) for e in FF.cover_edges(P))
        maps[3,6]=reshape(K[1,0],2,1)
        maps[5,6]=maps[5,8]=reshape(K[0,1],2,1)
        maps[7,8]=reshape(K[1,1],2,1)
        maps[6,9]=maps[8,9]=Matrix{K}(LinearAlgebra.I,2,2)
        M=MD.PModule{K}(P,dims,maps;field)
        witness=I.boundary_zigzag(M,EC.GridEncodingMap(P,([0,1,2],[0,1,2]));vertices=[3,5,6,7,8,9])
        @test I.boundary_path(witness).vertices==[3,6,5,8,7,9]
        @test I.boundary_bars(witness)==[(1,2,1),(2,4,1),(4,6,1),(6,6,1)]
        @test I.generalized_rank(witness)==0
    end

end

@testset "A54 visual finite regions witnesses and transactional selection" begin
    V=TamerOp.Visualization;I=TamerOp.Invariants
    enc=_a54_visual_sum(3,3,[collect(1:9),[2,3,5,6],[4,5,7,8]])
    s=V.generalized_rank_session(enc;vertices=2:8,boundary=true,witnesses=true)
    @test TamerOp.generalized_rank_session===V.generalized_rank_session
    @test TamerOp.Advanced.GeneralizedRankInspectionSession===V.GeneralizedRankInspectionSession
    @test TamerOp.Advanced.boundary_zigzag===I.boundary_zigzag
    @test V.inspection_summary(s).rank==1
    @test V.check_visual_spec(V.inspection_snapshot(s)).valid
    @test V.check_visual_spec(V.visual_spec(s)).valid
    @test V.check_inspection_session(s).valid
    before=V.inspection_summary(s)
    V.select_inspection!(s;occurrence=5,bar=1)
    @test V.inspection_summary(s).rank_queries==before.rank_queries
    @test V.inspection_summary(s).boundary_queries==before.boundary_queries
    old=V.inspection_snapshot(s)
    for changes in ((vertices=[1,9],),(occurrence=99,),(bar=99,),(width=1,),(vertices=[],))
        @test !V.check_inspection_selection(s;changes...).valid
        @test_throws ArgumentError V.select_inspection!(s;changes...)
        @test V.inspection_snapshot(s)===old
    end
    calls=Ref(0);token=V._on_inspection(s,_ -> calls[]+=1)
    V.select_inspection!(s;vertices=[5])
    @test calls[]==1 && V.inspection_selection(s).occurrence==0
    @test V.inspection_summary(s).rank==3
    @test V.inspection_snapshot(s).metadata.boundary.labels==[5,5]
    @test V.inspection_snapshot(s).metadata.boundary.bars==[(1,2,3)]
    selection=V.inspection_selection(s);empty!(selection.vertices)
    @test V.inspection_selection(s).vertices==[5]
    V.select_inspection!(s;boundary=false,witnesses=false)
    @test length(V.inspection_snapshot(s).panels)==2
    V.reset_inspection!(s)
    @test V.inspection_selection(s).vertices==collect(2:8)
    @test V.inspection_summary(s).rank==1
    V._off_inspection!(s,token)
    V.close_inspection!(s)
    @test s.object===nothing && isempty(s.data) && isempty(s.listeners)
    @test V.check_inspection_session(s).valid
    @test_throws ArgumentError V.select_inspection!(s;vertices=[1])
    @test V.check_visual_spec(V.inspection_snapshot(s)).valid
    M=TamerOp.Results.encoding_module(enc)
    @test V.visual_spec(M;kind=:generalized_rank,vertices=[5]).metadata.rank==3
    @test !V.check_visual_request(M;kind=:generalized_rank,vertices=[5],boundary=true).valid
    @test_throws ArgumentError V.generalized_rank_session(M;vertices=[5],max_vertices=2)
end

@testset "A54 visual exact worms layers and query reuse" begin
    V=TamerOp.Visualization;I=TamerOp.Invariants
    enc=_a54_visual_sum(1,1,[[1]];axes=([0],[0]))
    s=V.generalized_rank_session(enc;center=(4,6),width=4,levels=(1,2),lengths=(1,2,4))
    @test V.inspection_summary(s).rank==1
    @test I.gril_values(V.inspection_snapshot(s).metadata.landscape)==reshape(QQ[4,0,2,0,1,0],1,2,3)
    first_count=V.inspection_summary(s)
    V.select_inspection!(s;level=2)
    @test V.inspection_summary(s).rank_queries==first_count.rank_queries
    @test V.inspection_summary(s).landscape_queries==first_count.landscape_queries
    V.select_inspection!(s;width=401//100)
    @test V.inspection_summary(s).rank==0
    @test V.inspection_summary(s).landscape_queries==first_count.landscape_queries
    V.select_inspection!(s;width=1//2,length=2,level=1)
    poly=V.inspection_snapshot(s).panels[1].metadata.exact_polygon
    @test Set(poly)==Set([(3,6),(4,5),(5,5),(5,6),(4,7),(3,7)])
    @test all(q->abs(q[1]-4)<=1 && abs(q[2]-6)<=1 && abs(sum(q)-10)<=1,poly)
    old=V.inspection_snapshot(s)
    for changes in ((center=(NaN,1),),(width=-1,),(width=true,),(length=3,),(level=3,),(vertices=[1],))
        @test !V.check_inspection_selection(s;changes...).valid
        @test_throws ArgumentError V.select_inspection!(s;changes...)
        @test V.inspection_snapshot(s)===old
    end
    V.select_inspection!(s;center=(1//3,2//3),width=0)
    @test V.inspection_selection(s).center==(1//3,2//3)
    @test V.inspection_summary(s).landscape_queries==first_count.landscape_queries+1
    V.select_inspection!(s;center=(-1,1))
    @test V.inspection_summary(s).rank==0
    V.reset_inspection!(s)
    @test V.inspection_summary(s).rank==1
    @test V.check_inspection_session(s).valid
    @test V.check_visual_spec(V.visual_spec(V.inspection_snapshot(s).metadata.landscape)).valid
    V.close_inspection!(s)
    # Exact nearby coordinates may coincide in the floating drawing, never in the query.
    large=big(2)^60
    coarse=_a54_visual_sum(2,1,[[1,2]];axes=([large,large+1],[0]))
    collapsed=V.visual_spec(coarse;kind=:generalized_rank,center=(large,1),width=0,levels=(1,),lengths=(1,))
    @test !isempty(collapsed.panels[1].metadata.display_collisions)
    @test V.check_visual_spec(collapsed).valid
    @test collapsed.metadata.rank==1
    probes=I.gril(TamerOp.Results.encoding_module(coarse),coarse.pi;centers=[(large,1),(large+1,1)],levels=[1],lengths=[1])
    @test !isempty(V.visual_spec(probes).panels[1].metadata.display_collisions)
    @test_throws ArgumentError V.visual_spec(probes;center=true)
end

@testset "A54 visual native domain geometry and exports" begin
    V=TamerOp.Visualization
    enc=_a54_visual_sum(2,2,[[1,2,3,4]];axes=([0,3],[0,7]))
    worm=V.visual_spec(enc;kind=:generalized_rank,center=(2,5),width=1//2,length=2)
    boundary=TamerOp.Invariants.boundary_zigzag(TamerOp.Results.encoding_module(enc),enc.pi;vertices=[2,3,4])
    bspec=V.visual_spec(boundary)
    @test V.check_visual_spec(worm).valid && V.check_visual_spec(bspec).valid
    for package in (:CairoMakie,:WGLMakie)
        if Base.find_package(string(package))===nothing
            @test_skip false
            continue
        end
        @eval import $package
        mk=getfield(@__MODULE__,package);backend=package===:CairoMakie ? :cairomakie : :wglmakie
        for spec in (worm,bspec)
            fig=V.render(spec;backend)
            try
                mk.Makie.update_state_before_display!(fig)
                axes=filter(x->x isa mk.Axis,fig.content)
                @test !isempty(axes)
                @test all(a->all(>(0),mk.Makie.widths(mk.Makie.viewport(a.scene)[])),axes)
                if spec===worm
                    ax=only(filter(a->a.xlabel[]=="Parameter 1",axes))
                    points=filter(p->p isa mk.Scatter,ax.scene.plots)
                    @test Tuple.(points[1][1][])==[(0.,0.),(3.,0.),(0.,7.),(3.,7.)]
                    @test Tuple.(points[2][1][])==[(2.,5.)]
                end
                if package===:CairoMakie
                    mktempdir() do dir
                        for suffix in ("png","svg","pdf")
                            path=joinpath(dir,"region."*suffix);mk.save(path,fig)
                            @test filesize(path)>1000
                        end
                    end
                end
            finally;empty!(fig);end
        end
    end
end
