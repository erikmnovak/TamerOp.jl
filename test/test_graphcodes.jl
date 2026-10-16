include("fixtures/graphcodes.jl")

# Direct support-wise matrices and independent exact elementary elimination.
function _gc_rank(A)
    R=copy(A);r=1
    for j in axes(R,2)
        p=findfirst(i->!iszero(R[i,j]),r:size(R,1));p===nothing && continue;p+=r-1
        R[r,:],R[p,:]=copy(R[p,:]),copy(R[r,:])
        R[r,:].*=inv(R[r,j])
        for i in axes(R,1);i!=r && !iszero(R[i,j]) && (R[i,:]-=R[i,j].*R[r,:]);end
        r+=1;r>size(R,1) && break
    end
    r-1
end
function _gc_fixture(rows,coeffs,m;reverse=(false,false),scramble=false,rng=MersenneTwister(126),field=CM.F2())
    K=CM.coeff_type(field);n=length(rows);Q=FF.ProductOfChainsPoset((m,n);orientation=Tuple(t ? -1 : 1 for t in reverse))
    ids=[(reverse[1] ? m-i+1 : i)+(reverse[2] ? n-j : j-1)*m for i in 1:m,j in 1:n]
    active=[[k for k in eachindex(rows[j]) if rows[j][k][1]<=i<=rows[j][k][2]] for i in 1:m,j in 1:n]
    bases=Matrix{K}[];inverses=Matrix{K}[]
    for q in 1:m*n
        index=findfirst(==(q),ids);d=length(active[index]);B=Matrix{K}(I,d,d);Bi=copy(B)
        if scramble && d>1
            for _ in 1:3d
                a,b=randperm(rng,d)[1:2];c=K(rand(rng,1:4));B[a,:]+=c.*B[b,:];Bi[:,b]-=c.*Bi[:,a]
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
    MD.PModule{K}(Q,ds,maps;field),ids
end
function _gc_reconstruction(M,g)
    N=GRC.graphcode_module(g);ids=GRC.graphcode_vertices(g);m,n=size(ids)
    @test GRC.check_graphcode(g).valid
    @test MD.check_module_functoriality(N).valid
    for j in 1:n,i in 1:m
        B=GRC.graphcode_basis(g;row=j,column=i)
        @test size(B)==(M.dims[ids[i,j]],N.dims[i+(j-1)*m])
        @test _gc_rank(B)==size(B,1)
        for l in j:n,k in i:m
            C=GRC.graphcode_basis(g;row=l,column=k)
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
        g=GRC.graphcode(M);_gc_reconstruction(M,g)
        @test GRC.graphcode_vertices(g)==ids
        @test [[(r.first,r.last) for r in row] for row in GRC.graphcode_rows(g)]==sort.(rows)
        h=GRC.graphcode(M;vertices=ids[[1,3,4],[1,3]],x=[1//7,3//2,9//1],y=[-4,8])
        _gc_reconstruction(M,h);@test h.scope==:grid_restriction
        @test GRC.graphcode_axes(h)==([1//7,3//2,9//1],[-4,8])
        saved=GRC.graphcode_map(g;row=1);fill!(saved,zero(K))
        @test GRC.check_graphcode(g).valid
        M.dims[1]=99;@test GRC.check_graphcode(g).valid # owns input
    end
    encoded_source=first(_gc_fixture(rows,coeffs,4))
    encoded=TamerOp.Results.EncodingResult(encoded_source.Q,encoded_source,nothing)
    @test GRC.check_graphcode(GRC.graphcode(encoded)).valid
    native_session=TamerOp.Visualization.graphcode_explorer(GRC.graphcode(encoded_source))
    @test TamerOp.Visualization.check_inspection_session(native_session).valid
    TamerOp.Visualization.close_inspection!(native_session)
    # Same barcode rows can encode nonisomorphic modules via their vertical maps.
    bars=[[(1,3)],[(1,3)]]
    g0=GRC.graphcode(bars,[zeros(Int,1,1)];x=[0,2,7],y=[0,1])
    g1=GRC.graphcode(bars,[ones(Int,1,1)];x=[0,2,7],y=[0,1])
    @test GRC.graphcode_rows(g0)==GRC.graphcode_rows(g1)
    @test _gc_rank(MD.map_leq(GRC.graphcode_module(g0),1,4))==0
    @test _gc_rank(MD.map_leq(GRC.graphcode_module(g1),1,4))==1
    @test length(GRC.graphcode_components(g0))==2
    @test length(GRC.graphcode_components(g1))==1
    # Branching can be a connected graph of a decomposable module (invertible map).
    connected=GRC.graphcode([[(1,2),(1,2)],[(1,2),(1,2)]],[Int[1 1;0 1]];x=[0,1],y=[0,1])
    @test length(GRC.graphcode_components(connected))==1
    @test _gc_rank(GRC.graphcode_map(connected;row=1))==2
    # Two nonzero adjacent matrices can have zero composite in characteristic 2.
    cancel=GRC.graphcode([[(1,3)],[(1,3),(1,3)],[(1,3)]],[ones(Int,2,1),ones(Int,1,2)];x=[0,2,9],y=[0,3,8])
    N=GRC.graphcode_module(cancel);@test iszero(MD.map_leq(N,1,7))
    _gc_reconstruction(N,GRC.graphcode(N))
    for trial in 1:25
        m=rand(rng,1:5);n=rand(rng,1:4)
        rs=[sort!([(a=rand(rng,1:m);(a,rand(rng,a:m))) for _ in 1:rand(rng,0:5)]) for j in 1:n]
        cs=[Int[(rs[j+1][t][1]<=rs[j][s][1]<=rs[j+1][t][2]<=rs[j][s][2]) && rand(rng,Bool)
            for t in eachindex(rs[j+1]),s in eachindex(rs[j])] for j in 1:n-1]
        M,ids=_gc_fixture(rs,cs,m;scramble=true,rng);g=GRC.graphcode(M);_gc_reconstruction(M,g)
        supplied=GRC.graphcode(rs,cs;x=collect(1:m),y=collect(1:n));c=GRC.compress_graphcode(supplied);e=GRC.expand_graphcode(c)
        @test e.rows==supplied.rows && e.maps==supplied.maps
        @test GRC.check_graphcode(c).valid
        @test length(c.members)<=sum(length,rs)
        # Component projections sum to the identity, and all maps commute.
        total=[zeros(K,d,d) for d in M.dims]
        for k in eachindex(GRC.graphcode_components(g))
            S=GRC.graphcode_summand(g;component=k)
            @test MD.check_morphism(S.inclusion).valid && MD.check_morphism(S.projection).valid
            for q in eachindex(total)
                @test S.projection.comps[q]*S.inclusion.comps[q]==Matrix{K}(I,S.module_object.dims[q],S.module_object.dims[q])
                total[q]+=S.inclusion.comps[q]*S.projection.comps[q]
            end
        end
        @test all(total[q]==Matrix{K}(I,M.dims[q],M.dims[q]) for q in eachindex(total))
    end
    chain=GRC.graphcode(fill([(1,3)],5),fill(ones(Int,1,1),4);x=[0,2,8],y=[0,1,2,3,4])
    c=GRC.compress_graphcode(chain);@test length(c.members)==2
    @test c.members==[[(1,1),(2,1),(3,1),(4,1)],[(5,1)]]
    @test c.edges==[(1,2)]
    @test GRC.expand_graphcode(c).maps==chain.maps
    @test GRC.graphcode_summary(c).compression==:superfluous_vertices
    terminal=only(last(GRC.graphcode_rows(g1))) # last row has one interval
    @test terminal.death===nothing && terminal.terminal==:censored
    @test occursin("Graphcode",sprint(show,chain))
    @test GRC.describe(chain)==GRC.graphcode_summary(chain)
    @test GRC.check_graphcode(chain)==GRC.check_graphcode(chain)
    @test hash(GRC.check_graphcode(chain))==hash(GRC.check_graphcode(chain))
end

@testset "A126 invalid inputs and witnesses" begin
    g=_a126_graphcode();M=GRC.graphcode_module(g)
    for bad in ([[(0,1)]],[[(2,1)]],[[(1,5)]],[[(true,1)]])
        @test_throws ArgumentError GRC.graphcode(bad,Matrix{Int}[];x=[0,1,2,3],y=[0])
    end
    @test_throws ArgumentError GRC.graphcode([[(1,2)],[(2,3)]],[ones(Int,1,1)];x=[0,1,2],y=[0,1])
    @test_throws ArgumentError GRC.graphcode([[(1,2)],[(1,2)]],[ones(Float64,1,1)];x=[0,1],y=[0,1])
    @test_throws ArgumentError GRC.graphcode([[(1,2)]],Matrix{Int}[];x=[0,0],y=[0])
    @test_throws ArgumentError GRC.graphcode(M;vertices=[2 1;1 2])
    @test_throws ArgumentError GRC.graphcode(M;budget=GRC.GraphcodeBudget(max_entries=1))
    @test_throws ArgumentError GRC.graphcode(M;budget=GRC.GraphcodeBudget(max_vertices=2))
    @test_throws ArgumentError GRC.graphcode(M;budget=GRC.GraphcodeBudget(max_intervals=0))
    @test_throws ArgumentError GRC.graphcode_rows(GRC.graphcode([[(1,1)]],Matrix{Int}[];x=[NaN],y=[0]))
    @test_throws ArgumentError GRC.GraphcodeBudget(max_entries=-1)
    @test_throws ArgumentError GRC.graphcode_map(g;row=4)
    @test_throws ArgumentError GRC.graphcode_basis(g;row=true,column=1)
    @test GRC.field(GRC.graphcode(MD.PModule{CM.coeff_type(CM.F3())}(FF.ProductOfChainsPoset((1,1)),[0],Dict{Tuple{Int,Int},Matrix{CM.coeff_type(CM.F3())}}();field=CM.F3())))==CM.F3()
    h=GRC.graphcode(M);fill!(h.bases[1][1],0)
    @test !GRC.check_graphcode(h).valid
    @test_throws ArgumentError GRC.check_graphcode(h;throw=true)
    c=GRC.compress_graphcode(g);push!(c.members[1],first(c.members[1]))
    @test !GRC.check_graphcode(c).valid
    # A noncommuting source square is rejected rather than choosing an arbitrary path.
    bad=MD.PModule{CM.coeff_type(CM.F2())}(FF.ProductOfChainsPoset((2,2)),ones(Int,4),Dict((1,2)=>ones(CM.coeff_type(CM.F2()),1,1),(1,3)=>ones(CM.coeff_type(CM.F2()),1,1),(2,4)=>ones(CM.coeff_type(CM.F2()),1,1),(3,4)=>zeros(CM.coeff_type(CM.F2()),1,1));field=CM.F2())
    @test_throws Exception GRC.graphcode(bad)
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
    @test_throws ArgumentError V.visual_spec(GRC.compress_graphcode(g);max_intervals=1)
    @test V.available_visuals(g)==(:graphcode,)
    @test V.check_visual_spec(V.visual_spec(g)).valid
    @test V.check_visual_spec(V.visual_spec(GRC.compress_graphcode(g))).valid
    V.select_inspection!(s;row=4,source_interval=0,target_interval=0,column=4,compressed=true)
    @test size(V.inspection_snapshot(s).metadata.coefficients)==(0,2)
    @test V.inspection_snapshot(s).panels[1].metadata.compression==:superfluous_vertices
    V.reset_inspection!(s)
    # Descending and nonuniform coordinates remain physical x coordinates.
    rev=GRC.graphcode(g.rows,g.maps;x=reverse(g.x),y=g.y)
    rs=V.visual_spec(rev);@test rs.panels[1].metadata.original_axes[1]==reverse(g.x)
    # Float64 cannot silently collapse distinct exact coordinates.
    closex=[big(2)^60+i for i in 1:4]
    @test_throws ArgumentError V.visual_spec(GRC.graphcode(g.rows,g.maps;x=closex,y=g.y))
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

@testset "A126 coefficient-labelled graphcodes over exact fields" begin
    rng=MersenneTwister(12615)
    for F in (CM.QQField(),CM.F2(),CM.F3(),CM.Fp(5),CM.Fp(101))
        K=CM.coeff_type(F)
        rows=[[(1,4),(2,3)],[(1,4),(1,3)],[(1,2),(1,3),(2,3)]]
        coeffs=[Int[2 0;-1 3],Int[1 -2;0 4;0 0]]
        for reversed in ((false,false),(true,false),(false,true),(true,true))
            M,ids=_gc_fixture(rows,coeffs,4;field=F,reverse=reversed,scramble=true,rng)
            g=GRC.graphcode(M);_gc_reconstruction(M,g)
            @test GRC.field(g)==F==GRC.graphcode_summary(g).field
            @test GRC.graphcode_module(g).field==F
            @test GRC.field(copy(g))==F && GRC.check_graphcode(copy(g)).valid
            @test GRC.field(GRC.graphcode(TamerOp.Results.EncodingResult(M.Q,M,nothing)))==F
            restricted=GRC.graphcode(M;vertices=ids[[1,3,4],[1,3]],x=[9//2,1//3,-4//1],y=[-2,7])
            _gc_reconstruction(M,restricted)
        end
        # Cancellation is field arithmetic, not reachability of the drawn graph.
        cancel=GRC.graphcode([[(1,2)],[(1,2),(1,2)],[(1,2)]],
            [reshape([1,1],2,1),reshape([1,-1],1,2)];x=[-2,5],y=[1,3,9],field=F)
        @test iszero(MD.map_leq(GRC.graphcode_module(cancel),1,5))
        @test length(GRC.graphcode_components(cancel))==1
        for trial in 1:12
            m=rand(rng,1:4);n=rand(rng,1:5)
            rs=[sort!([(a=rand(rng,1:m);(a,rand(rng,a:m))) for _ in 1:rand(rng,0:4)]) for j in 1:n]
            cs=[K[(rs[j+1][t][1]<=rs[j][s][1]<=rs[j+1][t][2]<=rs[j][s][2]) ? rand(rng,-3:3) : 0
                for t in eachindex(rs[j+1]),s in eachindex(rs[j])] for j in 1:n-1]
            supplied=GRC.graphcode(rs,cs;x=collect(1:m),y=collect(1:n),field=F)
            c=GRC.compress_graphcode(supplied);e=GRC.expand_graphcode(c)
            @test c.field==e.field==F
            @test e.rows==supplied.rows && e.maps==supplied.maps
            @test GRC.check_graphcode(c).valid
            M,ids=_gc_fixture(rs,cs,m;field=F,scramble=true,rng)
            g=GRC.graphcode(M);_gc_reconstruction(M,g)
            N=GRC.graphcode_module(g);total=[zeros(K,d,d) for d in N.dims]
            for part in eachindex(GRC.graphcode_components(g))
                S=GRC.graphcode_summand(g;component=part)
                @test S.module_object.field==F
                @test MD.check_morphism(S.inclusion).valid && MD.check_morphism(S.projection).valid
                for q in eachindex(total)
                    @test S.projection.comps[q]*S.inclusion.comps[q]==Matrix{K}(I,S.module_object.dims[q],S.module_object.dims[q])
                    total[q]+=S.inclusion.comps[q]*S.projection.comps[q]
                end
            end
            @test all(total[q]==Matrix{K}(I,N.dims[q],N.dims[q]) for q in eachindex(total))
        end
        # Remove three internal vertices with nonidentity continuation scalars.
        values=F isa CM.QQField ? K[-2//3,7//2,-5,11//13] : K[-1,1,-1,1]
        g=GRC.graphcode(fill([(1,2)],5),[reshape([a],1,1) for a in values];x=[0,1],y=[0,2,3,7,9],field=F)
        c=GRC.compress_graphcode(g)
        @test length(c.members)==2
        @test only(GRC.graphcode_edges(c)).coefficient==last(values)
        spans=GRC.graphcode_continuations(c)
        @test spans[1].coefficients==values[1:3]
        @test GRC.expand_graphcode(c).maps==g.maps
        @test only(MD.map_leq(GRC.graphcode_module(c),1,9))==prod(values)
        spans[1].coefficients[1]=zero(K)
        @test GRC.check_graphcode(c).valid
        broken=deepcopy(c);broken.continuations[1][1]=zero(K)
        @test !GRC.check_graphcode(broken).valid
        @test_throws ArgumentError GRC.expand_graphcode(broken)
        broken=deepcopy(c);pop!(broken.coefficients)
        @test !GRC.check_graphcode(broken).valid
        @test TOA.graphcode_continuations===GRC.graphcode_continuations
    end
    rows=[[(1,2)],[(1,2)]]
    @test_throws ArgumentError GRC.graphcode(rows,[ones(Int,1,1)];x=[0,1],y=[0,1],field=CM.RealField(Float64))
    @test_throws ArgumentError GRC.graphcode(rows,[ones(CM.coeff_type(CM.F3()),1,1)];x=[0,1],y=[0,1],field=CM.Fp(5))
    @test_throws ArgumentError GRC.graphcode(rows,[ones(CM.coeff_type(CM.F3()),1,1)];x=[0,1],y=[0,1],field=CM.QQField())
    @test_throws ArgumentError GRC.graphcode(rows,[ones(Float64,1,1)];x=[0,1],y=[0,1],field=CM.QQField())
    @test_throws ArgumentError GRC.graphcode([[(1,2)],[(2,2)]],[reshape([-2//3],1,1)];x=[0,1],y=[0,1],field=CM.QQField())
end

@testset "A126 exact field labels and saved coefficients" begin
    V=TamerOp.Visualization
    for F in (CM.QQField(),CM.F3(),CM.Fp(5))
        K=CM.coeff_type(F);a=F isa CM.QQField ? K(-2//3) : K(2)
        g=GRC.graphcode(fill([(1,3)],4),[reshape([x],1,1) for x in K[a,-1,1]];
            x=[0//1,1//3,8//1],y=[9,4,1,-2],field=F)
        s=V.graphcode_explorer(g;row=1,source_interval=1,target_interval=1)
        snapshot=V.inspection_snapshot(s)
        @test snapshot.metadata.coefficient==a
        @test snapshot.metadata.field==V._inspection_field_label(F)
        edge=only(filter(t->t.selection.target_interval==1 && t.selection.row==1,snapshot.panels[1].metadata.pick_targets))
        @test occursin(V._matrix_coefficient_text(a),edge.label)
        @test !occursin("in F2",edge.label)
        @test snapshot.panels[1].metadata.continuations[1].coefficients==K[a,-1]
        for (backend,extname) in ((:cairomakie,:TamerOpCairoMakieExt),(:wglmakie,:TamerOpWGLMakieExt))
            ext=Base.get_extension(TamerOp,extname)
            if ext===nothing;@test_skip false;continue;end
            fig=V.render(snapshot;backend)
            M=backend===:cairomakie ? ext.CairoMakie.Makie : ext.WGLMakie.Makie
            labels=[t for block in fig.content if block isa M.Label
                for plot in block.blockscene.plots if plot isa M.Text for t in plot.input_text[]]
            @test V._matrix_coefficient_text(a) in labels
            @test any(block->block isa M.Label && occursin(V._inspection_field_label(F),string(block.text[])),fig.content)
            empty!(fig)
        end
        original=V.visual_state(s)
        V.select_inspection!(s;compressed=true)
        @test V.inspection_snapshot(s).metadata.coefficient==a
        @test V.check_inspection_session(s).valid
        spec=V.visual_spec(GRC.compress_graphcode(g))
        @test V.check_visual_spec(spec).valid
        @test spec.panels[1].metadata.compressed_edges[1].coefficient==one(K)
        compressed=V.visual_state(s)
        mktempdir() do dir
            path=joinpath(dir,"graphcode.json")
            V.save_visual_session(path,(;original,compressed))
            saved=V.load_visual_session(path)
            for name in (:original,:compressed)
                restored=V.inspection_snapshot(saved;state=name)
                @test restored.metadata.coefficient==a
                @test restored.metadata.coefficients==reshape([a],1,1)
                @test restored.metadata.field==V._inspection_field_label(F)
                @test restored.panels[1].metadata.continuations[1].coefficients==K[a,-1]
            end
            if F isa CM.QQField && Base.get_extension(TamerOp,:TamerOpCairoMakieExt)!==nothing
                file=get(ENV,"TAMEROP_GRAPHCODE_QQ_HTML",joinpath(dir,"graphcode-qq.html"));mkpath(dirname(file))
                V.save_visual_session(file,(;original,compressed))
                @test filesize(file)>1000
            end
        end
        V.select_inspection!(s;row=2)
        @test V.inspection_snapshot(s).metadata.coefficient==-one(K)
        V.close_inspection!(s)
    end
end
