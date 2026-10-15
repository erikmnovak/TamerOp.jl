include("fixtures/end_curves.jl")

function _a127_binary_rank(A)
    B=Bool[!iszero(x) for x in A];r=0
    for j in axes(B,2)
        i=findfirst(k->B[k,j],r+1:size(B,1));i===nothing && continue
        i+=r;r+=1;B[r,:],B[i,:]=copy(B[i,:]),copy(B[r,:])
        for k in axes(B,1)
            k!=r && B[k,j] && (B[k,:].=xor.(B[k,:],B[r,:]))
        end
    end
    r
end
function _a127_basis_change(M)
    F=M.field;K=TamerOp.CoreModules.coeff_type(F)
    bases=[Matrix{K}(I,d,d) for d in M.dims]
    for S in bases,i in 1:size(S,1)-1;S[i,i+1]=one(K);end
    inverses=[TamerOp.FieldLinAlg.solve_fullcolumn(F,S,Matrix{K}(I,size(S,1),size(S,1))) for S in bases]
    maps=Dict((u,v)=>bases[v]*TamerOp.Modules.map_leq(M,u,v)*inverses[u] for (u,v) in TamerOp.FiniteFringe.cover_edges(M.Q))
    TamerOp.Modules.PModule{K}(M.Q,copy(M.dims),maps;field=F)
end
_a127_supports(cs)=sort([(Tuple(ENDC.curve_vertices(c)),ENDC.curve_multiplicity(c)) for c in cs])

@testset "A127 exact counts and end-curves" begin
    for F in (TamerOp.CoreModules.QQField(),TamerOp.CoreModules.F2(),TamerOp.CoreModules.Fp(3))
        M=_a127_module((3,2),[_a127_rectangle((0,0),(2,1))];field=F)
        r=ENDC.end_curves(M;witnesses=true)
        @test ENDC.two_parameter_count(M)==ENDC.two_parameter_count(r)==1
        @test _a127_supports(ENDC.birth_curves(r))==[(Tuple([(0,1),(0,0),(1,0),(2,0)]),1)]
        @test _a127_supports(ENDC.death_curves(r))==[(Tuple([(1,2),(2,2),(3,2),(3,1)]),1)]
        b=ENDC.end_curve_betti(r)
        expected=zeros(Int,4,3,3);expected[1,1,1]=1;expected[4,1,2]=1;expected[1,3,2]=1;expected[4,3,3]=1
        @test b==expected
        @test ENDC.check_end_curves(r;throw=true).witnesses_checked
        for kind in (:birth,:death)
            N=ENDC.end_curve_module(r;kind)
            @test TamerOp.Modules.check_module_functoriality(N;throw=true).valid
            for (u,v) in TamerOp.FiniteFringe.cover_edges(N.Q)
                @test size(TamerOp.Modules.map_leq(N,u,v))==(N.dims[v],N.dims[u])
            end
        end
        @test ENDC.end_curve_at(r;point=(3,2)).betti==(0,0,1)
        @test ENDC.end_curve_at(r;point=(3,2)).zero_halo
        @test ENDC.end_curve_summary(r).boundary_components===:not_computed
        @test occursin("zero extension",sprint(show,r))
        copy_b=ENDC.birth_curves(r);empty!(copy_b[1].vertices)
        @test !isempty(ENDC.birth_curves(r)[1].vertices)
        b[:].=99;@test ENDC.end_curve_betti(r)==expected
    end
    for orientation in ((1,1),(1,-1),(-1,1),(-1,-1))
        M=_a127_module((3,2),[_a127_rectangle((0,0),(2,1))];orientation)
        r=ENDC.end_curves(M;origin=(-3,4))
        @test ENDC.two_parameter_count(r)==1
        @test ENDC.birth_curves(r)[1].vertices==[(-3,5),(-3,4),(-2,4),(-1,4)]
        @test ENDC.death_curves(r)[1].vertices==[(-2,6),(-1,6),(0,6),(0,5)]
        @test ENDC.end_curve_summary(r).source_orientation==orientation
    end
    for shape in ((1,1),(1,3),(3,1),(2,2))
        r=ENDC.end_curves(_a127_module(shape,Vector{Tuple{Int,Int}}[]))
        @test ENDC.two_parameter_count(r)==0
        @test isempty(ENDC.birth_curves(r)) && isempty(ENDC.death_curves(r))
        @test all(iszero,ENDC.end_curve_betti(r))
        s=ENDC.end_curves(_a127_module(shape,[_a127_rectangle((0,0),shape.-1)]))
        @test ENDC.two_parameter_count(s)==1
    end
    # Direct sums and stalk-basis changes retain exact supports and multiplicities.
    supports=[_a127_rectangle((0,0),(2,1)),[(0,1),(1,1),(1,0)],[(2,0)],[(2,0)]]
    for F in (TamerOp.CoreModules.QQField(),TamerOp.CoreModules.F2(),TamerOp.CoreModules.Fp(5))
        M=_a127_module((3,2),supports;field=F);N=_a127_basis_change(M)
        a=ENDC.end_curves(M);b=ENDC.end_curves(N;witnesses=true)
        @test ENDC.two_parameter_count(a)==4
        @test ENDC.birth_curves(a)==ENDC.birth_curves(b)
        @test ENDC.death_curves(a)==ENDC.death_curves(b)
        @test ENDC.end_curve_betti(a)==ENDC.end_curve_betti(b)
        @test ENDC.check_end_curves(b;throw=true).valid
        @test sum(c.multiplicity for c in a.births if length(c.vertices)==1)==2
        singles=[ENDC.end_curves(_a127_module((3,2),[S];field=F)) for S in supports]
        @test ENDC.end_curve_betti(a)==sum(ENDC.end_curve_betti(s) for s in singles)
        @test _a127_supports(a.births)==_a127_supports(ENDC.EndCurveResult(vcat([s.births for s in singles]...),vcat([s.deaths for s in singles]...);shape=(3,2),field=F).births)
    end
    # A finite diagonal map is tested independently by elementary binary elimination.
    rng=MersenneTwister(127)
    for trial in 1:12
        m,n=rand(rng,1:3),rand(rng,1:3);supports=Vector{Tuple{Int,Int}}[]
        for _ in 1:rand(rng,0:3)
            x=sort(rand(rng,0:m-1,2));y=sort(rand(rng,0:n-1,2))
            push!(supports,_a127_rectangle((x[1],y[1]),(x[2],y[2])))
        end
        M=_a127_module((m,n),supports);q(i,j)=i+(j-1)*m
        rr=sum(M.dims)
        for j in 1:n,i in 1:m
            i<m && (rr-=_a127_binary_rank(TamerOp.Modules.map_leq(M,q(i,j),q(i+1,j))))
            j<n && (rr-=_a127_binary_rank(TamerOp.Modules.map_leq(M,q(i,j),q(i,j+1))))
            i<m && j<n && (rr+=_a127_binary_rank(TamerOp.Modules.map_leq(M,q(i,j),q(i+1,j+1))))
        end
        r=ENDC.end_curves(M)
        @test rr==length(supports)==ENDC.two_parameter_count(M)==ENDC.two_parameter_count(r)
        @test ENDC.check_end_curves(r).valid
    end
end

@testset "A127 curves distinguish equal rank data" begin
    expected_birth=[([(0,2),(1,2),(2,2),(2,1),(2,0)],1), ([(1,2),(1,1),(2,1)],1)]
    expected_distinct=[([(1,3),(2,3),(3,3),(3,2)],1), ([(2,3),(3,3),(3,2),(3,1)],1)]
    expected_coincident=[([(1,3),(2,3),(3,3),(3,2),(3,1)],1), ([(2,3),(3,3),(3,2)],1)]
    canonical(xs)=sort([(Tuple(p),m) for (p,m) in xs])
    for F in (TamerOp.CoreModules.QQField(),TamerOp.CoreModules.F2(),TamerOp.CoreModules.Fp(3))
        M=_a127_three_lines(;field=F);N=_a127_three_lines(;field=F,coincident=true)
        a=ENDC.end_curves(M;witnesses=true);b=ENDC.end_curves(N)
        @test ENDC.two_parameter_count(a)==ENDC.two_parameter_count(b)==2
        @test M.dims==N.dims
        for u in 1:9,v in 1:9
            TamerOp.FiniteFringe.leq(M.Q,u,v) || continue
            @test TamerOp.FieldLinAlg.rank(F,TamerOp.Modules.map_leq(M,u,v))==TamerOp.FieldLinAlg.rank(F,TamerOp.Modules.map_leq(N,u,v))
        end
        @test ENDC.end_curve_betti(a)==ENDC.end_curve_betti(b)
        @test _a127_supports(a.births)==_a127_supports(b.births)==canonical(expected_birth)
        @test _a127_supports(a.deaths)==canonical(expected_distinct)
        @test _a127_supports(b.deaths)==canonical(expected_coincident)
        @test ENDC.check_end_curves(a;throw=true).valid
        @test ENDC.death_curves(ENDC.end_curves(_a127_basis_change(M)))==ENDC.death_curves(a)
    end
end

@testset "A127 contracts and lattice scope" begin
    C=ENDC.EndCurve
    @test_throws ArgumentError C(:other,[(0,0)])
    @test_throws ArgumentError C(:birth,Tuple{Int,Int}[])
    @test_throws ArgumentError C(:birth,[(0,0),(1,1)])
    @test_throws ArgumentError C(:birth,[(0,0),(2,0)])
    @test_throws ArgumentError C(:birth,[(0,0),(0,0)])
    @test_throws ArgumentError C(:birth,[(0.0,0)])
    @test_throws ArgumentError C(:birth,[(0,0)];multiplicity=false)
    @test_throws ArgumentError ENDC.EndCurveResult([C(:birth,[(0,0)])],C[];shape=(1,1),field=TamerOp.CoreModules.F2())
    @test_throws ArgumentError ENDC.EndCurveResult([C(:birth,[(1,0)])],[C(:death,[(1,1)])];shape=(1,1),field=TamerOp.CoreModules.F2())
    r=ENDC.end_curves(_a127_module((1,1),[[(0,0)]]))
    encoded_source=_a127_module((1,1),[[(0,0)]])
    encoded=TamerOp.Results.EncodingResult(encoded_source.Q,encoded_source,nothing)
    @test TamerOp.two_parameter_count(encoded)==1
    @test TamerOp.birth_curves(TamerOp.end_curves(encoded))==ENDC.birth_curves(r)
    @test_throws ArgumentError ENDC.end_curve_witnesses(r)
    @test_throws ArgumentError ENDC.end_curve_at(r;point=(2,0))
    @test_throws ArgumentError ENDC.end_curve_at(r;point=(false,0))
    @test_throws ArgumentError ENDC.end_curve_module(r;kind=:boundary)
    @test_throws ArgumentError ENDC.end_curves(_a127_module((2,2),[]);budget=ENDC.EndCurveBudget(max_vertices=8))
    @test_throws ArgumentError ENDC.end_curves(_a127_module((2,2),[_a127_rectangle((0,0),(1,1))]);budget=ENDC.EndCurveBudget(max_dimension=2))
    @test_throws ArgumentError ENDC.end_curves(_a127_module((2,2),[]);origin=(typemax(Int),0))
    extreme=ENDC.end_curves(_a127_module((1,1),[[(0,0)]]);origin=(typemax(Int)-1,typemin(Int)))
    @test ENDC.check_end_curves(extreme).valid
    @test only(ENDC.death_curves(extreme)).vertices==[(typemax(Int),typemin(Int)+1)]
    @test_throws ArgumentError ENDC.EndCurveResult(C[],C[];shape=(typemax(Int),1),field=TamerOp.CoreModules.F2())
    @test_throws ArgumentError ENDC.end_curve_at(r;point=(typemin(Int),typemax(Int)))
    @test_throws ArgumentError ENDC.end_curve_module(r;kind=:birth,budget=ENDC.EndCurveBudget(max_vertices=1))
    # Physical GridPoset coordinates are accepted only for a complete integer lattice.
    K=TamerOp.CoreModules.coeff_type(TamerOp.CoreModules.F2())
    for (coords,valid) in ((([-2,-1],[4,5]),true),(([-2,0],[4,5]),false))
        Q=TamerOp.FiniteFringe.GridPoset(coords)
        M=TamerOp.Modules.PModule{K}(Q,ones(Int,4),Dict(e=>ones(K,1,1) for e in TamerOp.FiniteFringe.cover_edges(Q));field=TamerOp.CoreModules.F2())
        if valid
            @test ENDC.end_curves(M).origin==(-2,4)
        else
            @test_throws ArgumentError ENDC.end_curves(M)
        end
    end
    s=ENDC.EndCurveResult(r.births,r.deaths;shape=r.shape,field=TamerOp.CoreModules.F2())
    @test ENDC.check_end_curves(s).valid
    @test ENDC.end_curve_summary(s).scope===:supplied
    @test_throws ArgumentError ENDC.end_curve_betti(s)
    @test ENDC.end_curve_at(s;point=(0,0)).betti===nothing
    @test isequal(ENDC.check_end_curves(s),ENDC.check_end_curves(s))
    @test hash(ENDC.check_end_curves(s))==hash(ENDC.check_end_curves(s))
    push!(s.births[1].vertices,(2,2))
    @test !ENDC.check_end_curves(s).valid
    @test_throws ArgumentError ENDC.check_end_curves(s;throw=true)
end

@testset "A127 end-curve visualization" begin
    V=TamerOp.Visualization;r=_a127_result();s=V.end_curve_explorer(r;curve=1,point=(-2,4))
    snap=V.inspection_snapshot(s)
    @test V.check_visual_spec(snap).valid
    @test snap.metadata.count==3
    @test snap.panels[1].metadata.curves[1].vertices==r.births[1].vertices
    @test snap.panels[2].metadata.curves[1].vertices==r.deaths[1].vertices
    death_labels=[label for layer in snap.panels[2].layers if layer isa V.TextLayer for label in layer.labels]
    @test "D1, D2" in death_labels
    @test snap.panels[1].axes.xlimits==(-2.6,1.6)
    @test snap.panels[1].axes.ylimits==(3.4,6.6)
    @test V.available_visuals(r)==(:end_curves,)
    before=V.inspection_selection(s)
    @test_throws ArgumentError V.select_inspection!(s;curve=99)
    @test V.inspection_selection(s)==before
    @test !V.check_inspection_selection(s;point=(-4,4)).valid
    @test_throws ArgumentError V.select_inspection!(s;betti_degree=3)
    @test_throws ArgumentError V.end_curve_explorer(r;max_points=1)
    @test_throws ArgumentError V.visual_spec(r;max_curves=1)
    @test_throws ArgumentError V.visual_spec(r;max_curves=true)
    far=ENDC.end_curves(_a127_module((1,1),[[(0,0)]]);origin=(2^60,0))
    @test_throws ArgumentError V.visual_spec(far)
    V.select_inspection!(s;kind=:death,curve=1,point=(1,6),betti_degree=2)
    @test V.inspection_snapshot(s).metadata.point.zero_halo
    @test V.inspection_snapshot(s).metadata.point.betti[3]>0
    V.reset_inspection!(s)
    # The inspector owns its visible values independently of the caller.
    old=copy(r.births[1].vertices);empty!(r.births[1].vertices)
    @test V.inspection_snapshot(s).panels[1].metadata.curves[1].vertices==old
    for (backend,extname) in ((:cairomakie,:TamerOpCairoMakieExt),(:wglmakie,:TamerOpWGLMakieExt))
        ext=Base.get_extension(TamerOp,extname)
        if ext===nothing;@test_skip false;continue;end
        fig=V.render(snap;backend);M=backend===:cairomakie ? ext.CairoMakie.Makie : ext.WGLMakie.Makie
        axes=filter(a->a isa M.Axis,fig.content)
        points=[Tuple(p) for plot in axes[1].scene.plots if plot isa M.Scatter for p in plot[1][]]
        @test (-2.,4.) in points
        @test (-1.,5.) in points
        @test axes[1].xlabel[]=="Lattice x (module order)"
        empty!(fig)
        if backend===:wglmakie
            ui=ext._inspection_ui(V.visualize(s;backend));original=ui.figure[]
            ui.controls.curve.value[]="0";ui.controls.x.value[]="1";ui.controls.y.value[]="6"
            ui.controls.apply.value[]=true;wait(s.requests.task)
            @test V.inspection_selection(s).point==(1,6)
            ui.controls.curve.value[]="99";ui.controls.apply.value[]=true;wait(s.requests.task)
            @test occursin("curve",ui.error_text[])
            @test V.inspection_selection(s).curve==0
            V.reset_inspection!(s);V.select_inspection!(s;kind=:death,curve=1)
            @test ui.figure[]===original && ui.rebuild_count[]==1 && ui.update_count[]>=3
            ui.dispose();@test isempty(ui.observers) && isempty(ui.pointer_observers)
            V.reset_inspection!(s)
        end
    end
    if Base.get_extension(TamerOp,:TamerOpCairoMakieExt)!==nothing
        mktempdir() do dir
            for ext in ("svg","png","pdf")
                path=joinpath(dir,"end-curves.$ext");V.save_visual(path,snap;backend=:cairomakie);@test filesize(path)>1000
            end
            file=get(ENV,"TAMEROP_END_CURVES_HTML",joinpath(dir,"end-curves.html"));mkpath(dirname(file))
            births=V.visual_state(s);V.select_inspection!(s;kind=:death,curve=1,point=(1,6),betti_degree=2);deaths=V.visual_state(s)
            V.save_visual_session(file,(;births,deaths));@test filesize(file)>1000
        end
    end
    V.close_inspection!(s);@test V.check_inspection_session(s).valid
    @test s.object===nothing
    @test_throws ArgumentError V.select_inspection!(s;curve=0)
end
