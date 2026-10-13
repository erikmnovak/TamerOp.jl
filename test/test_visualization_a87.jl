# Independent coordinate and direct-sum oracles for the summand consumer.
function _a87_fixture(;field=CM.QQField())
    P=FF.ProductOfChainsPoset((2,2));K=CM.coeff_type(field)
    supports=[[1,2,3,4],[2,4],[4],[4]]
    active=[findall(I->q in I,supports) for q in 1:4]
    maps=Dict(e=>K[i==j for i in active[e[2]],j in active[e[1]]] for e in FF.cover_edges(P))
    M=MD.PModule{K}(P,length.(active),maps;field)
    enc=TamerOp.EncodingResult(P,M,TamerOp.EncodingCore.GridEncodingMap(P,([-3,2],[1,5])))
    M,enc
end

@testset "A87 checked split and selected reconstruction" begin
    V=TamerOp.Visualization;D=TamerOp.IntervalDecompositions
    for field in (CM.QQField(),CM.F2(),CM.F3(),CM.Fp(5))
        M,enc=_a87_fixture(;field)
        result=D.interval_decomposition(M;witnesses=true)
        c=V.summand_collection(result)
        @test V.check_summand_collection(c).valid
        @test TamerOp.Advanced.source_module(c)===M
        @test V.summand_module(c;id=3)===c.pieces[3]
        @test V.summand_maps(c;id=3).projection===c.projections[3]
        @test V.summand_module(c;id=:residual)===c.residual
        @test V.summand_maps(c;id=:residual).inclusion===c.residual_inclusion
        @test_throws ArgumentError V.summand_module(c;id=true)
        @test V.describe(c).multiplicities==[1,1,2]
        @test TamerOp.summand_collection===V.summand_collection
        @test TamerOp.Advanced.SummandCollection===V.SummandCollection
        s=V.summand_session(c;encoding=enc,box=([-4,0],[4,8]),selected=[2,3],focus=3,vertex=4)
        snap=V.inspection_snapshot(s)
        @test snap.metadata.selected_dimensions==[0,1,0,3]
        @test snap.metadata.complement_dimensions==[1,1,1,1]
        @test snap.metadata.certification==:checked_splitting
        @test snap.metadata.geometry_status==:exact_classifier_fibers
        @test V.check_visual_spec(snap).valid
        @test V.check_inspection_session(s).valid
        @test V.check_visual_spec(V.visual_spec(s)).valid
        # Multiplicity-two block at vertex four: these are the actual source coordinates.
        @test c.projections[3].comps[4]==CM.coeff_type(field)[0 0 1 0;0 0 0 1]
        @test c.inclusions[3].comps[4]*c.projections[3].comps[4]+sum(c.inclusions[j].comps[4]*c.projections[j].comps[4] for j in 1:2)==CM.eye(field,4)
        V.select_inspection!(s;selected=nothing,sort_by=:multiplicity,limit=1)
        @test V.inspection_selection(s).ids==[3]
        @test V.inspection_snapshot(s).metadata.complement_dimensions==[1,2,1,2]
        V.select_inspection!(s;selected=nothing,type_filter=:thin,sort_by=:index,limit=8)
        @test V.inspection_selection(s).ids==[1,2]
        V.select_inspection!(s;selected=Int[])
        @test V.inspection_selection(s).focus==0
        @test V.inspection_snapshot(s).metadata.selected_dimensions==zeros(Int,4)
        before=V.inspection_selection(s);snapshot=V.inspection_snapshot(s)
        for opts in ((;selected=[99]),(;vertex=0),(;layout=:hull),(;type_filter=:indecomposable),(;limit=17),(;pair=(true,2)))
            @test !V.check_inspection_selection(s;opts...).valid
            @test_throws ArgumentError V.select_inspection!(s;opts...)
            @test V.inspection_selection(s)==before
            @test V.inspection_snapshot(s)===snapshot
        end
        V.reset_inspection!(s)
        @test V.inspection_selection(s).ids==[2,3]
        V.close_inspection!(s)
        @test V.check_inspection_session(s).valid
        @test s.collection===nothing && isempty(s.prepared)
        @test V.check_visual_spec(V.inspection_snapshot(s)).valid
        @test_throws ArgumentError V.select_inspection!(s;vertex=2)
        @test_throws ArgumentError V.summand_collection(D.interval_decomposition(M))
    end
end

@testset "A87 supplied general summands and certificate failures" begin
    V=TamerOp.Visualization;field=CM.QQField();K=Rational{BigInt}
    P=FF.FinitePoset(BitMatrix([i==j || i<4&&j==4 for i in 1:4,j in 1:4]))
    N=MD.PModule{K}(P,[1,1,1,2],Dict((1,4)=>reshape(K[1,0],2,1),(2,4)=>reshape(K[0,1],2,1),(3,4)=>reshape(K[1,1],2,1));field)
    # Identity gives a checked thick piece, with no indecomposability label inferred.
    c=V.summand_collection(N,[N];inclusions=[MD.id_morphism(N)],projections=[MD.id_morphism(N)],
        grades=[(;generators=[(0,1),(1,0)],relations=[(2,2)])])
    @test V.check_summand_collection(c).valid
    @test V.summand_summary(c).multiplicities==[1]
    bounded=V.summand_collection(N,[N];inclusions=[MD.id_morphism(N)],projections=[MD.id_morphism(N)],
        budget=TamerOp.IntervalDecompositions.IntervalDecompositionBudget(max_candidates=1))
    @test V.check_summand_collection(bounded).valid
    s=V.summand_session(c;type_filter=:thick)
    @test V.inspection_selection(s).ids==[1]
    @test V.inspection_snapshot(s).metadata.statistics[1].generators==2
    @test V.inspection_snapshot(s).metadata.statistics[1].interval==false
    V.select_inspection!(s;type_filter=:interval)
    @test isempty(V.inspection_selection(s).ids)
    # A dimension-compatible map that is not a retraction must never certify a split.
    bad=MD.PMorphism{K}(N,N,[zeros(K,d,d) for d in N.dims])
    @test_throws ArgumentError V.summand_collection(N,[N];inclusions=[MD.id_morphism(N)],projections=[bad])
    @test_throws ArgumentError V.summand_collection(N,MD.PModule[];inclusions=[],projections=[])
    @test_throws ArgumentError V.summand_collection(N,[N,N];inclusions=[MD.id_morphism(N),MD.id_morphism(N)],projections=[MD.id_morphism(N),MD.id_morphism(N)])
    @test_throws ArgumentError V.summand_collection(N,[N];inclusions=[MD.id_morphism(N)],projections=[MD.id_morphism(N)],budget=TamerOp.IntervalDecompositions.IntervalDecompositionBudget(max_matrix_entries=1))
    c.projections[1].comps[4][1,1]=0
    @test !V.check_summand_collection(c).valid
    @test_throws ArgumentError V.check_summand_collection(c;throw=true)
end

@testset "A87 exact support cells and unverified supplied drawings" begin
    V=TamerOp.Visualization;D=TamerOp.IntervalDecompositions
    M,enc=_a87_fixture();c=V.summand_collection(D.interval_decomposition(M;witnesses=true))
    s=V.summand_session(c;encoding=enc,box=([-4,0],[4,8]),layout=:overlay)
    @test V.check_visual_spec(V.inspection_snapshot(s)).valid
    geometry=s.prepared.geometry
    for cell in geometry.components
        cell.dimension==2 || continue
        x=sum(first,cell.vertices)/length(cell.vertices);y=sum(last,cell.vertices)/length(cell.vertices)
        expected=x< -3 || y<1 ? 0 : 1+(x>=2)+2*(y>=5)
        @test cell.region_id==expected
        for (p,included) in zip(cell.vertices,cell.vertex_included)
            label=p[1]< -3 || p[2]<1 ? 0 : 1+(p[1]>=2)+2*(p[2]>=5)
            @test included==(label==expected)
        end
    end
    V.select_inspection!(s;layout=:outline,selected=[2],scale=:log)
    @test V.inspection_snapshot(s).metadata.selected_dimensions==[0,1,0,1]
    @test V.check_visual_spec(V.inspection_snapshot(s)).valid
    V.select_inspection!(s;layout=:isolate,focus=2)
    @test V.inspection_selection(s).focus==2
    @test V._summand_dim_color(0,4,:log)==V._VisualRole(:background)
    @test V._summand_dim_color(1,4,:log)!=V._summand_dim_color(4,4,:log)
    # Geometry is allowed to be imperfect without weakening the algebraic status.
    drawing=V.SummandSupportDrawing([[((0,0),(2,1)),((1,1),(2,3))],[((1,0),(2,2))],[((1,1),(2,2))]];
        provenance="Supplied staircase rectangles",error_statement="No certified approximation bound.")
    sp=V.visual_spec(c;drawing,layout=:overlay)
    @test sp.metadata.certification==:checked_splitting
    @test sp.metadata.geometry_status==:unverified
    @test V.check_visual_spec(sp).valid
    @test any(p->p.title=="I1: stalk dimensions",sp.panels)
    overlay=only(filter(p->p.kind===:summand_overlay,sp.panels))
    @test length(overlay.legend.entries)==3
    @test V.describe(drawing).boundaries==:lower_closed_upper_open
    # A declared-family split keeps genuine algebra outside the gallery.
    partial=V.summand_collection(D.interval_decomposition(M;family=[[2,4]],witnesses=true))
    pv=V.visual_spec(partial)
    @test pv.metadata.selected_dimensions==[0,1,0,1]
    @test pv.metadata.residual_dimensions==[1,1,1,3]
    @test pv.metadata.complement_dimensions==pv.metadata.residual_dimensions
    @test V.summand_maps(partial;id=:residual).projection.comps[4]*
        V.summand_maps(partial;id=:residual).inclusion.comps[4]==CM.eye(M.field,3)
    @test !V.check_visual_request(D.interval_decomposition(M)).valid
    @test !V.check_visual_request(c;layout=:hull).valid
    @test_throws ArgumentError V.SummandSupportDrawing([[((1,0),(0,1))]];provenance="bad")
    @test_throws ArgumentError V.visual_spec(c;encoding=enc,drawing)
    @test_throws ArgumentError V.visual_spec(c;max_vertices=2)
    @test_throws ArgumentError V.visual_spec(c;encoding=_a87_fixture()[2])
    # Unlike a rectangular grid, a general classifier can leave its complement
    # implicit. A represented zero space must still differ from that complement.
    PL=TamerOp.PLPolyhedra;K=Rational{BigInt};P=FF.FinitePoset(trues(1,1))
    poly=PL.HPoly(2,K[-1 0;1 0;0 -1;0 1],K[0,2,0,2],nothing,falses(4),zero(K))
    pi=PL.PLEncodingMap(2,[BitVector()],[BitVector()],[poly],[(1,1)])
    Z=MD.zero_pmodule(P;field=M.field)
    z=V.summand_collection(Z,[Z];inclusions=[MD.id_morphism(Z)],projections=[MD.id_morphism(Z)])
    zv=V.visual_spec(z;encoding=TamerOp.EncodingResult(P,Z,pi),box=([-1,-1],[3,3]))
    full=first(zv.panels)
    @test full.layers[1] isa V.RectLayer
    @test full.layers[1].fill_color==V._VisualRole(:unrepresented)
    @test any(l->l isa V.PolygonLayer && l.fill_color==V._VisualRole(:background) && l.alpha==1.,full.layers)
end

@testset "A87 ordinary Hom table independent interval oracle" begin
    V=TamerOp.Visualization;D=TamerOp.IntervalDecompositions
    M,enc=_a87_fixture();c=V.summand_collection(D.interval_decomposition(M;witnesses=true))
    s=V.summand_session(c;hom=true,hom_pair=(3,1),vertex=4)
    # Intervals: full square A, right column B, maximal singleton C^2.
    # B -> A and C -> A,B are allowed; reverse directions hit an incoming boundary.
    @test V.inspection_snapshot(s).metadata.hom_dimensions==[1 1 2;0 1 2;0 0 4]
    @test size(V.inspection_snapshot(s).metadata.hom_component)==(1,2)
    h=s.hom_cache[3,1]
    V.select_inspection!(s;vertex=2)
    @test s.hom_cache[3,1]===h
    @test size(V.inspection_snapshot(s).metadata.hom_component)==(1,0)
    V.select_inspection!(s;hom_pair=(1,3))
    @test V.inspection_snapshot(s).metadata.hom_component===nothing
    before=V.inspection_snapshot(s)
    @test_throws ArgumentError V.select_inspection!(s;hom_pair=(3,1),basis_index=99)
    @test V.inspection_snapshot(s)===before
    @test_throws ArgumentError V.summand_session(c;hom=true,max_hom_pairs=4)
    @test_throws ArgumentError V.summand_session(c;hom=true,max_hom_entries=1)
    V.close_inspection!(s)
    @test isempty(s.hom_cache)
end

@testset "A87 native rendering and portable snapshots" begin
    V=TamerOp.Visualization;M,enc=_a87_fixture()
    result=TamerOp.IntervalDecompositions.interval_decomposition(M;witnesses=true)
    session=V.summand_session(result;encoding=enc,box=([-4,0],[4,8]),selected=[2,3],vertex=4)
    snapshot=V.inspection_snapshot(session)
    state=V.visual_state(session)
    @test V.describe(state).kind==:visual_session_state
    for package in (:CairoMakie,:WGLMakie)
        if Base.find_package(string(package))===nothing
            @test_skip false
            continue
        end
        @eval import $package
        mk=getfield(@__MODULE__,package);backend=package===:CairoMakie ? :cairomakie : :wglmakie
        fig=V.render(snapshot;backend)
        try
            mk.Makie.update_state_before_display!(fig)
            axes=filter(x->x isa mk.Axis,fig.content)
            @test length(axes)>=5
            @test all(a->all(>(0),mk.Makie.widths(mk.Makie.viewport(a.scene)[])),axes)
            ax=first(axes)
            @test ax.xlabel[]=="x1"
            @test Tuple(mk.Makie.minimum(ax.finallimits[]))==(-4.,0.)
            @test Tuple(mk.Makie.maximum(ax.finallimits[]))==(4.,8.)
            if package===:CairoMakie
                mktempdir() do dir
                    path=joinpath(dir,"summands.json")
                    V.save_visual_session(path,session)
                    saved=V.load_visual_session(path)
                    @test V.check_visual_session(saved).valid
                    html=get(ENV,"TAMEROP_A87_HTML",joinpath(dir,"summands.html"))
                    mkpath(dirname(html))
                    full=V.visual_state(session)
                    V.select_inspection!(session;selected=[3],focus=3)
                    V.save_visual_session(html,(;edge_and_sink=full,sink=V.visual_state(session));
                        title="Checked summands",caption="Two prepared selections; no live Julia queries in this file.")
                    @test filesize(html)>1000
                    V.reset_inspection!(session)
                    for suffix in ("png","svg","pdf")
                        path=joinpath(dir,"summands."*suffix);mk.save(path,fig)
                        @test filesize(path)>1000
                    end
                end
            end
        finally;empty!(fig);end
        if package===:WGLMakie
            app=V.visualize(session;backend)
            ext=Base.get_extension(TamerOp,:TamerOpWGLMakieExt);ui=ext._inspection_ui(app)
            @test !ui.closed[]
            old=V.inspection_selection(session)
            target=first(ui.targets[])
            ui.callbacks.hover(target.point,target.axis)
            @test V.inspection_selection(session)==old
            ui.dispose()
            @test ui.closed[] && isempty(ui.observers) && isempty(ui.pointer_observers)
        end
    end
    V.close_inspection!(session)
end

@testset "A87 changed coordinates preserve the supplied split" begin
    V=TamerOp.Visualization;D=TamerOp.IntervalDecompositions
    M,enc=_a87_fixture();c=V.summand_collection(D.interval_decomposition(M;witnesses=true))
    K=Rational{BigInt}
    T=[Matrix{K}(LinearAlgebra.I,d,d) for d in M.dims]
    for A in T, i in 1:size(A,1)-1;A[i,i+1]=2;end
    U=inv.(T)
    changed=MD.PModule{K}(M.Q,copy(M.dims),Dict((u,v)=>T[v]*M.edge_maps[u,v]*U[u] for (u,v) in FF.cover_edges(M.Q));field=M.field)
    ins=[MD.PMorphism{K}(S,changed,[T[q]*c.inclusions[j].comps[q] for q in 1:4]) for (j,S) in enumerate(c.pieces)]
    outs=[MD.PMorphism{K}(changed,S,[c.projections[j].comps[q]*U[q] for q in 1:4]) for (j,S) in enumerate(c.pieces)]
    supplied=V.summand_collection(changed,c.pieces;inclusions=ins,projections=outs)
    @test V.check_summand_collection(supplied).valid
    @test supplied.projections[3].comps[4]==K[0 0 1 -2;0 0 0 1]
    view=V.visual_spec(supplied;selected=[3],focus=3,vertex=4)
    @test view.metadata.selected_dimensions==[0,0,0,2]
    @test view.metadata.complement_dimensions==[1,2,1,2]
    matrix=only(filter(p->occursin("projection at 4",p.title),view.panels))
    @test matrix.layers[1].entries==["0" "0" "1" "-2";"0" "0" "0" "1"]
end

@testset "A87 non-square reversed classifier and degenerate base" begin
    V=TamerOp.Visualization;EC=TamerOp.EncodingCore
    field=CM.QQField();K=Rational{BigInt};P=FF.ProductOfChainsPoset((3,2))
    dims=[1,2,3,4,5,6]
    # Zero maps make a functor; the identity supplied split has known dimensions.
    M=MD.PModule{K}(P,dims,Dict((u,v)=>zeros(K,dims[v],dims[u]) for (u,v) in FF.cover_edges(P));field)
    c=V.summand_collection(M,[M];inclusions=[MD.id_morphism(M)],projections=[MD.id_morphism(M)])
    enc=TamerOp.EncodingResult(P,M,EC.GridEncodingMap(P,([-5,-1,3],[2,7]);orientation=(-1,1)))
    s=V.summand_session(c;encoding=enc,box=([-6,0],[7,9]),shared_scale=false,scale=:log)
    for cell in s.prepared.geometry.components
        cell.dimension==2 || continue
        x=sum(first,cell.vertices)/length(cell.vertices);y=sum(last,cell.vertices)/length(cell.vertices)
        ix=searchsortedlast([-5,-1,3],-x);iy=searchsortedlast([2,7],y)
        label=ix==0||iy==0 ? 0 : ix+3*(iy-1)
        @test cell.region_id==label
    end
    @test V.inspection_snapshot(s).panels[1].metadata.dimensions==dims
    Z=MD.zero_pmodule(FF.FinitePoset(falses(0,0));field)
    z=V.summand_collection(Z,MD.PModule[];inclusions=[],projections=[])
    zv=V.summand_session(z)
    @test V.inspection_selection(zv).vertex==0
    @test V.check_visual_spec(V.inspection_snapshot(zv)).valid
end
