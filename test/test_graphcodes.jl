include("fixtures/graphcodes.jl")

# Direct support-wise matrices and independent elementary GF(2) elimination.
function _gc_rank(A)
    R=copy(A);r=1
    for j in axes(R,2)
        p=findfirst(i->!iszero(R[i,j]),r:size(R,1));p===nothing && continue;p+=r-1
        R[r,:],R[p,:]=copy(R[p,:]),copy(R[r,:])
        for i in axes(R,1);i!=r && !iszero(R[i,j]) && (R[i,:]+=R[r,:]);end
        r+=1;r>size(R,1) && break
    end
    r-1
end
function _gc_fixture(rows,coeffs,m;reverse=(false,false),scramble=false,rng=MersenneTwister(126))
    K=CM.coeff_type(CM.F2());n=length(rows);Q=FF.ProductOfChainsPoset((m,n);orientation=Tuple(t ? -1 : 1 for t in reverse))
    ids=[(reverse[1] ? m-i+1 : i)+(reverse[2] ? n-j : j-1)*m for i in 1:m,j in 1:n]
    active=[[k for k in eachindex(rows[j]) if rows[j][k][1]<=i<=rows[j][k][2]] for i in 1:m,j in 1:n]
    bases=Matrix{K}[];inverses=Matrix{K}[]
    for q in 1:m*n
        index=findfirst(==(q),ids);d=length(active[index]);B=Matrix{K}(I,d,d);Bi=copy(B)
        if scramble && d>1
            for _ in 1:3d
                a,b=randperm(rng,d)[1:2];B[a,:]+=B[b,:];Bi[:,b]+=Bi[:,a]
            end
        end
        push!(bases,B);push!(inverses,Bi)
    end
    maps=Dict{Tuple{Int,Int},Matrix{K}}();ds=zeros(Int,m*n)
    for j in 1:n,i in 1:m
        u=ids[i,j];ds[u]=length(active[i,j])
        if i<m
            v=ids[i+1,j];H=K[a==b for a in active[i+1,j],b in active[i,j]]
            maps[u,v]=bases[v]*H*inverses[u]
        end
        if j<n
            v=ids[i,j+1];C=K.(coeffs[j][active[i,j+1],active[i,j]])
            maps[u,v]=bases[v]*C*inverses[u]
        end
    end
    MD.PModule{K}(Q,ds,maps;field=CM.F2()),ids
end
function _gc_reconstruction(M,g)
    N=GC.graphcode_module(g);ids=GC.graphcode_vertices(g);m,n=size(ids)
    @test GC.check_graphcode(g).valid
    @test MD.check_module_functoriality(N).valid
    for j in 1:n,i in 1:m
        B=GC.graphcode_basis(g;row=j,column=i)
        @test size(B)==(M.dims[ids[i,j]],N.dims[i+(j-1)*m])
        @test _gc_rank(B)==size(B,1)
        for l in j:n,k in i:m
            C=GC.graphcode_basis(g;row=l,column=k)
            @test MD.map_leq(M,ids[i,j],ids[k,l])*B==C*MD.map_leq(N,i+(j-1)*m,k+(l-1)*m)
        end
    end
end

@testset "A126 exact graphcodes" begin
    K=CM.coeff_type(CM.F2());rng=MersenneTwister(126)
    rows=[[(1,4),(2,3)],[(1,4),(1,3)],[(1,2),(1,3),(2,3)]]
    coeffs=[Int[1 0;1 1],Int[1 1;0 1;0 0]]
    for reversed in ((false,false),(true,false),(false,true),(true,true)),scramble in (false,true)
        M,ids=_gc_fixture(rows,coeffs,4;reverse=reversed,scramble,rng)
        g=GC.graphcode(M);_gc_reconstruction(M,g)
        @test GC.graphcode_vertices(g)==ids
        @test [[(r.first,r.last) for r in row] for row in GC.graphcode_rows(g)]==sort.(rows)
        h=GC.graphcode(M;vertices=ids[[1,3,4],[1,3]],x=[1//7,3//2,9//1],y=[-4,8])
        _gc_reconstruction(M,h);@test h.scope==:grid_restriction
        @test GC.graphcode_axes(h)==([1//7,3//2,9//1],[-4,8])
        saved=GC.graphcode_map(g;row=1);fill!(saved,zero(K))
        @test GC.check_graphcode(g).valid
        M.dims[1]=99;@test GC.check_graphcode(g).valid # owns input
    end
    encoded_source=first(_gc_fixture(rows,coeffs,4))
    encoded=TamerOp.Results.EncodingResult(encoded_source.Q,encoded_source,nothing)
    @test GC.check_graphcode(GC.graphcode(encoded)).valid
    native_session=TamerOp.Visualization.graphcode_explorer(GC.graphcode(encoded_source))
    @test TamerOp.Visualization.check_inspection_session(native_session).valid
    TamerOp.Visualization.close_inspection!(native_session)
    # Same barcode rows can encode nonisomorphic modules via their vertical maps.
    bars=[[(1,3)],[(1,3)]]
    g0=GC.graphcode(bars,[zeros(Int,1,1)];x=[0,2,7],y=[0,1])
    g1=GC.graphcode(bars,[ones(Int,1,1)];x=[0,2,7],y=[0,1])
    @test GC.graphcode_rows(g0)==GC.graphcode_rows(g1)
    @test _gc_rank(MD.map_leq(GC.graphcode_module(g0),1,4))==0
    @test _gc_rank(MD.map_leq(GC.graphcode_module(g1),1,4))==1
    @test length(GC.graphcode_components(g0))==2
    @test length(GC.graphcode_components(g1))==1
    # Branching can be a connected graph of a decomposable module (invertible map).
    connected=GC.graphcode([[(1,2),(1,2)],[(1,2),(1,2)]],[Int[1 1;0 1]];x=[0,1],y=[0,1])
    @test length(GC.graphcode_components(connected))==1
    @test _gc_rank(GC.graphcode_map(connected;row=1))==2
    # Two nonzero adjacent matrices can have zero composite in characteristic 2.
    cancel=GC.graphcode([[(1,3)],[(1,3),(1,3)],[(1,3)]],[ones(Int,2,1),ones(Int,1,2)];x=[0,2,9],y=[0,3,8])
    N=GC.graphcode_module(cancel);@test iszero(MD.map_leq(N,1,7))
    _gc_reconstruction(N,GC.graphcode(N))
    for trial in 1:25
        m=rand(rng,1:5);n=rand(rng,1:4)
        rs=[sort!([(a=rand(rng,1:m);(a,rand(rng,a:m))) for _ in 1:rand(rng,0:5)]) for j in 1:n]
        cs=[Int[(rs[j+1][t][1]<=rs[j][s][1]<=rs[j+1][t][2]<=rs[j][s][2]) && rand(rng,Bool)
            for t in eachindex(rs[j+1]),s in eachindex(rs[j])] for j in 1:n-1]
        M,ids=_gc_fixture(rs,cs,m;scramble=true,rng);g=GC.graphcode(M);_gc_reconstruction(M,g)
        supplied=GC.graphcode(rs,cs;x=collect(1:m),y=collect(1:n));c=GC.compress_graphcode(supplied);e=GC.expand_graphcode(c)
        @test e.rows==supplied.rows && e.maps==supplied.maps
        @test GC.check_graphcode(c).valid
        @test length(c.members)<=sum(length,rs)
        # Component projections sum to the identity, and all maps commute.
        total=[zeros(K,d,d) for d in M.dims]
        for k in eachindex(GC.graphcode_components(g))
            S=GC.graphcode_summand(g;component=k)
            @test MD.check_morphism(S.inclusion).valid && MD.check_morphism(S.projection).valid
            for q in eachindex(total)
                @test S.projection.comps[q]*S.inclusion.comps[q]==Matrix{K}(I,S.module_object.dims[q],S.module_object.dims[q])
                total[q]+=S.inclusion.comps[q]*S.projection.comps[q]
            end
        end
        @test all(total[q]==Matrix{K}(I,M.dims[q],M.dims[q]) for q in eachindex(total))
    end
    chain=GC.graphcode(fill([(1,3)],5),fill(ones(Int,1,1),4);x=[0,2,8],y=[0,1,2,3,4])
    c=GC.compress_graphcode(chain);@test length(c.members)==2
    @test c.members==[[(1,1),(2,1),(3,1),(4,1)],[(5,1)]]
    @test c.edges==[(1,2)]
    @test GC.expand_graphcode(c).maps==chain.maps
    @test GC.graphcode_summary(c).compression==:superfluous_vertices
    terminal=only(last(GC.graphcode_rows(g1))) # last row has one interval
    @test terminal.death===nothing && terminal.terminal==:censored
    @test occursin("Graphcode",sprint(show,chain))
    @test GC.describe(chain)==GC.graphcode_summary(chain)
    @test GC.check_graphcode(chain)==GC.check_graphcode(chain)
    @test hash(GC.check_graphcode(chain))==hash(GC.check_graphcode(chain))
end

@testset "A126 invalid inputs and witnesses" begin
    g=_a126_graphcode();M=GC.graphcode_module(g)
    for bad in ([[(0,1)]],[[(2,1)]],[[(1,5)]],[[(true,1)]])
        @test_throws ArgumentError GC.graphcode(bad,Matrix{Int}[];x=[0,1,2,3],y=[0])
    end
    @test_throws ArgumentError GC.graphcode([[(1,2)],[(2,3)]],[ones(Int,1,1)];x=[0,1,2],y=[0,1])
    @test_throws ArgumentError GC.graphcode([[(1,2)],[(1,2)]],[ones(Float64,1,1)];x=[0,1],y=[0,1])
    @test_throws ArgumentError GC.graphcode([[(1,2)]],Matrix{Int}[];x=[0,0],y=[0])
    @test_throws ArgumentError GC.graphcode(M;vertices=[2 1;1 2])
    @test_throws ArgumentError GC.graphcode(M;budget=GC.GraphcodeBudget(max_entries=1))
    @test_throws ArgumentError GC.graphcode(M;budget=GC.GraphcodeBudget(max_vertices=2))
    @test_throws ArgumentError GC.graphcode(M;budget=GC.GraphcodeBudget(max_intervals=0))
    @test_throws ArgumentError GC.graphcode_rows(GC.graphcode([[(1,1)]],Matrix{Int}[];x=[NaN],y=[0]))
    @test_throws ArgumentError GC.GraphcodeBudget(max_entries=-1)
    @test_throws ArgumentError GC.graphcode_map(g;row=4)
    @test_throws ArgumentError GC.graphcode_basis(g;row=true,column=1)
    @test_throws ArgumentError GC.graphcode(MD.PModule{CM.coeff_type(CM.F3())}(FF.ProductOfChainsPoset((1,1)),[0],Dict{Tuple{Int,Int},Matrix{CM.coeff_type(CM.F3())}}();field=CM.F3()))
    h=GC.graphcode(M);fill!(h.bases[1][1],0)
    @test !GC.check_graphcode(h).valid
    @test_throws ArgumentError GC.check_graphcode(h;throw=true)
    c=GC.compress_graphcode(g);push!(c.members[1],first(c.members[1]))
    @test !GC.check_graphcode(c).valid
    # A noncommuting source square is rejected rather than choosing an arbitrary path.
    bad=MD.PModule{CM.coeff_type(CM.F2())}(FF.ProductOfChainsPoset((2,2)),ones(Int,4),Dict((1,2)=>ones(CM.coeff_type(CM.F2()),1,1),(1,3)=>ones(CM.coeff_type(CM.F2()),1,1),(2,4)=>ones(CM.coeff_type(CM.F2()),1,1),(3,4)=>zeros(CM.coeff_type(CM.F2()),1,1));field=CM.F2())
    @test_throws Exception GC.graphcode(bad)
end

@testset "A126 graphcode visualization" begin
    V=TamerOp.Visualization;g=_a126_graphcode();s=_a126_session()
    @test V.check_inspection_session(s).valid
    snap=V.inspection_snapshot(s)
    @test snap.metadata.coefficient==one(CM.coeff_type(CM.F2()))
    @test snap.metadata.hom_admissible
    @test snap.panels[1].metadata.original_axes==([0//1,1//3,2//1,5//1],[8//1,4//1,1//1,0//1])
    @test snap.metadata.stalk_map==CM.coeff_type(CM.F2()).([1 0;0 1])
    before=V.inspection_selection(s)
    @test !V.check_inspection_selection(s;source_interval=99).valid
    @test_throws ArgumentError V.select_inspection!(s;row=99)
    @test V.inspection_selection(s)==before
    @test_throws ArgumentError V.graphcode_explorer(g;max_edges=1)
    @test_throws ArgumentError V.graphcode_explorer(g;max_intervals=1)
    @test_throws ArgumentError V.graphcode_explorer(g;max_edges=true)
    @test_throws ArgumentError V.graphcode_explorer(g;max_rows=1)
    @test_throws ArgumentError V.visual_spec(g;max_rows=1)
    @test_throws ArgumentError V.visual_spec(GC.compress_graphcode(g);max_intervals=1)
    @test V.available_visuals(g)==(:graphcode,)
    @test V.check_visual_spec(V.visual_spec(g)).valid
    @test V.check_visual_spec(V.visual_spec(GC.compress_graphcode(g))).valid
    V.select_inspection!(s;row=4,source_interval=0,target_interval=0,column=4,compressed=true)
    @test size(V.inspection_snapshot(s).metadata.coefficients)==(0,2)
    @test V.inspection_snapshot(s).panels[1].metadata.compression==:superfluous_vertices
    V.reset_inspection!(s)
    # Descending and nonuniform coordinates remain physical x coordinates.
    rev=GC.graphcode(g.rows,g.maps;x=reverse(g.x),y=g.y)
    rs=V.visual_spec(rev);@test rs.panels[1].metadata.original_axes[1]==reverse(g.x)
    # Float64 cannot silently collapse distinct exact coordinates.
    closex=[big(2)^60+i for i in 1:4]
    @test_throws ArgumentError V.visual_spec(GC.graphcode(g.rows,g.maps;x=closex,y=g.y))
    for (backend,extname) in ((:cairomakie,:TamerOpCairoMakieExt),(:wglmakie,:TamerOpWGLMakieExt))
        ext=Base.get_extension(TamerOp,extname)
        if ext===nothing;@test_skip false;continue;end
        fig=V.render(snap;backend)
        M=backend===:cairomakie ? ext.CairoMakie.Makie : ext.WGLMakie.Makie
        ax=first(filter(a->a isa M.Axis,fig.content))
        points=[Tuple(p) for plot in ax.scene.plots if plot isa M.Scatter for p in plot[1][]]
        @test any(p->isapprox(p[1],1/3;atol=1e-6,rtol=0),points)
        @test any(p->isapprox(p[1],0;atol=1e-6,rtol=0),points)
        empty!(fig)
        if backend===:wglmakie
            ui=ext._inspection_ui(V.visualize(s;backend));original=ui.figure[]
            ui.controls.source.value[]="1";ui.controls.target.value[]="2";ui.controls.apply.value[]=true;wait(s.requests.task)
            @test V.inspection_snapshot(s).metadata.coefficient==zero(CM.coeff_type(CM.F2()))
            ui.controls.source.value[]="99";ui.controls.apply.value[]=true;wait(s.requests.task)
            @test occursin("source interval",ui.error_text[])
            @test V.inspection_selection(s).source_interval==1
            V.reset_inspection!(s);V.select_inspection!(s;compressed=true)
            @test ui.figure[]===original && ui.rebuild_count[]==1 && ui.update_count[]>=3
            ui.dispose();@test isempty(ui.observers) && isempty(ui.pointer_observers)
            V.reset_inspection!(s)
        end
    end
    if Base.get_extension(TamerOp,:TamerOpCairoMakieExt)!==nothing
        mktempdir() do dir
            for ext in ("svg","png","pdf")
                path=joinpath(dir,"graphcode.$ext");V.save_visual(path,snap;backend=:cairomakie);@test filesize(path)>1000
            end
            file=get(ENV,"TAMEROP_GRAPHCODE_HTML",joinpath(dir,"graphcode.html"));mkpath(dirname(file))
            original=V.visual_state(s);V.select_inspection!(s;compressed=true,column=3);compressed=V.visual_state(s)
            V.save_visual_session(file,(;original,compressed));@test filesize(file)>1000
        end
    end
    V.close_inspection!(s);@test V.check_inspection_session(s).valid
    @test s.object===nothing && s.compressed===nothing
    @test_throws ArgumentError V.select_inspection!(s;column=2)
end
