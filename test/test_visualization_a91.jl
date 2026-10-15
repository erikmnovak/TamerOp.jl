function _a91_graph(births=[0,0,1,4],edges=[(1,2),(2,3)],grades=[2,3];orientation=1)
    n=length(births);m=length(edges)
    B=spzeros(Int,n,m)
    for (i,(a,b)) in enumerate(edges);B[a,i]-=1;B[b,i]+=1;end
    G=DT.GradedComplex([collect(11:10+n),collect(101:100+m)],[B],[(g,) for g in vcat(births,grades)])
    if n==0
        # An empty retained source has no coordinate from which the convenience
        # constructor could infer ambient dimension. Its face list is empty too.
        return DI.filtered_complex(G;geometry=Vector{NTuple{2,Float64}}[],
            faces=Vector{Int}[],orientation=(orientation,),provenance=(;ambient_dim=2))
    end
    points=[(Float64(i),Float64(isodd(i))) for i in 1:n]
    DI.filtered_complex(G;geometry=vcat([[p] for p in points],[[points[a],points[b]] for (a,b) in edges]),orientation=(orientation,))
end
function _a91_bfs(x,p)
    G=DI.graded_complex(x);n=DT.cell_counts(G)[1];B=DT.boundary_map(G;dim=1)
    active(i)=any(g->all(k->x.orientation[k]*(p[k]-g[k])>=0,eachindex(p)),DT.cell_grade_set(G,i))
    adj=[Int[] for _ in 1:n]
    for e in axes(B,2)
        active(n+e) || continue
        ids=findall(!iszero,B[:,e]);length(ids)==2 || continue
        a,b=ids;push!(adj[a],b);push!(adj[b],a)
    end
    todo=Set(filter(active,1:n));parts=Vector{Int}[]
    while !isempty(todo)
        seed=minimum(todo);delete!(todo,seed);q=[seed];part=Int[]
        while !isempty(q)
            i=pop!(q);push!(part,i)
            for j in adj[i]
                j in todo || continue
                delete!(todo,j);push!(q,j)
            end
        end
        push!(parts,sort!(part))
    end
    sort!(parts)
end

@testset "A91 core components, bars and exact events" begin
    O=TamerOp.OrdinaryPersistence;x=_a91_graph();h=O.component_history(x)
    @test O.check_component_history(h).valid
    @test TamerOp.component_history===O.component_history
    @test TamerOp.Advanced.ComponentHistory===O.ComponentHistory
    @test O.component_history_summary(h).essential==2
    @test occursin("4 vertices",sprint(show,h))
    @test [(r.component,r.birth,r.death) for r in O.component_intervals(h)]==[(1,0,nothing),(2,0,2),(3,1,3),(4,4,nothing)]
    @test [(e.edge,e.survivor,e.absorbed) for e in O.component_merges(h)]==[(1,1,2),(2,1,3)]
    for (t,n) in ((-1,0),(0,2),(1,3),(2,2),(3,1),(4,2))
        @test O.component_count(h;scale=t)==n
        @test sort([c.vertices for c in O.components_at(h;scale=t)])==_a91_bfs(x,(t,))
    end
    @test O.component_count(h;scale=2-big(1)//big(2)^70)==3
    @test O.component_members(h;scale=1,component=3).vertices==[3]
    @test O.component_members(h;scale=0,component=3).status==:not_born
    @test O.component_members(h;scale=2,component=2).status==:merged
    @test O.component_members(h;scale=2,component=2).vertices==[1,2]
    @test O.component_members(h;scale=3,component=2).vertices==[1,2,3]
    for f in (CM.F2(),CM.F3(),CM.Fp(5))
        d=O.persistence_diagram(h;field=f);oracle=O.persistence_diagram(DI.graded_complex(x);field=f)
        @test sort(O.finite_intervals(d;dim=0))==sort(O.finite_intervals(oracle;dim=0))
        @test sort(O.essential_births(d;dim=0))==sort(O.essential_births(oracle;dim=0))
    end
    @test_throws ArgumentError O.component_members(h;scale=0,component=0)
    @test_throws ArgumentError O.component_count(h;scale=Inf)
    @test_throws ArgumentError O.component_history(O.persistence_diagram(DI.graded_complex(x)))
    broken=deepcopy(h);broken.parent[1]=0
    @test !O.check_component_history(broken).valid
    @test_throws ArgumentError O.check_component_history(broken;throw=true)
end

@testset "A91 core simultaneous, superlevel, multicritical and source parity" begin
    O=TamerOp.OrdinaryPersistence
    ties=_a91_graph([0,0,0],[(1,2),(2,3),(3,1)],[0,1,1]);h=O.component_history(ties)
    @test O.component_count(h;scale=0)==2
    @test O.component_count(h;scale=1)==1
    @test O.component_history_summary(h).ephemeral==1
    @test length(O.component_intervals(h))==2
    @test length(O.component_intervals(h;include_ephemeral=true))==3
    @test length(O.component_merges(h))==2 # the cycle-closing edge is not a merge
    sup=_a91_graph([5,5,4],[(1,2),(2,3)],[3,2];orientation=-1);hs=O.component_history(sup)
    @test [(r.birth,r.death) for r in O.component_intervals(hs)]==[(5,nothing),(5,3),(4,2)]
    @test O.component_count(hs;scale=4)==3
    @test O.component_count(hs;scale=3)==2
    @test O.component_parameter(hs,3)==(3,)
    @test O.filtration_order(O.persistence_diagram(hs))==:superlevel
    @test sort(O.finite_intervals(O.persistence_diagram(hs);dim=0))==sort(O.finite_intervals(O.persistence_diagram(DI.graded_complex(sup);order=:superlevel);dim=0))
    G=DT.MultiCriticalGradedComplex([[1,2],[3]],[sparse(reshape([-1,1],2,1))],
        [[(0,0)],[(0,0)],[(1,3),(3,1)]])
    x=DI.filtered_complex(G;geometry=[[(0,0)],[(1,0)],[(0,0),(1,0)]])
    @test_throws ArgumentError O.component_history(x)
    a=O.component_history(x;slice=(base=(0,0),direction=(1,1)))
    b=O.component_history(x;slice=(base=(0,0),direction=(1,2)))
    @test only(O.component_merges(a)).scale==3
    @test only(O.component_merges(b)).scale==3//2
    @test O.component_parameter(b,3//2)==(3//2,3)
    @test O.component_count(a;scale=2)==2
    for t in (0,1,3//2,2,3)
        @test sort([c.vertices for c in O.components_at(b;scale=t)])==_a91_bfs(x,O.component_parameter(b,t))
    end
    @test_throws ArgumentError O.component_history(x;slice=(base=(0,0),direction=(0,1)))
    @test_throws ArgumentError O.component_history(x;slice=(base=(0,0),direction=(-1,1)))
    for values in ([0 2 0;0 2 1],),order in (:sublevel,:superlevel),periodic in (false,true)
        source=DI.filtered_complex(values;order,periodic)
        c=DI.source_correspondence(source;field=CM.F3())
        history=O.component_history(c)
        diagram=O.persistence_diagram(c;order)
        @test O.component_history(diagram).births==history.births
        for t in (-1,0,1,2,3)
            @test sort([p.vertices for p in O.components_at(history;scale=t)])==_a91_bfs(source,(t,))
        end
    end
    # A genuine reduction removes a same-grade vertex/edge pair. Membership
    # must still use the original two vertices, not the reduced point's basis.
    source=_a91_graph([0,0],[(1,2)],[0])
    reduced=_a91_graph([0],Tuple{Int,Int}[],Int[])
    c=DI.source_correspondence(reduced,source;field=CM.F3(),
        lift=[sparse(reshape([1,0],2,1)),spzeros(Int,1,0)],
        retraction=[sparse(reshape([1,1],1,2)),spzeros(Int,0,1)])
    @test O.component_members(O.component_history(c);scale=0,component=2).vertices==[1,2]
    @test O.component_members(O.component_history(O.persistence_diagram(c));scale=0,component=2).sources==[(0,11),(0,12)]
    extra=_a91_graph([0,0],Tuple{Int,Int}[],Int[])
    split=DI.source_correspondence(reduced,extra;field=CM.F3(),
        lift=[sparse(reshape([1,0],2,1)),spzeros(Int,0,0)],
        retraction=[sparse(reshape([1,0],1,2)),spzeros(Int,0,0)])
    @test_throws ArgumentError O.component_history(split) # a split map need not preserve H0
    empty=O.component_history(_a91_graph(Int[],Tuple{Int,Int}[],Int[]))
    @test isempty(O.components_at(empty;scale=0))
    @test isempty(O.component_intervals(empty))
    @test O.component_history_summary(empty).essential==0
    loops=O.component_history(_a91_graph([0],[(1,1)],[0]))
    @test O.component_count(loops;scale=0)==1
    @test isempty(O.component_merges(loops))
    bad=DT.GradedComplex([[1,2],[3]],[sparse(reshape([1,1],2,1))],[(0,),(0,),(1,)])
    @test_throws ArgumentError O.component_history(DI.filtered_complex(bad;geometry=[[(0,0)],[(1,0)],[(0,0),(1,0)]]))
end

@testset "A91 core randomized independent graph and reduction oracles" begin
    O=TamerOp.OrdinaryPersistence;rng=MersenneTwister(9106)
    for trial in 1:12
        n=rand(rng,3:12);births=rand(rng,0:3,n)
        edges=[(i,j) for i in 1:n for j in i+1:n if rand(rng)<0.3]
        grades=[max(births[i],births[j])+rand(rng,0:3) for (i,j) in edges]
        x=_a91_graph(births,edges,grades);h=O.component_history(x)
        for t in -1:7
            @test sort([c.vertices for c in O.components_at(h;scale=t)])==_a91_bfs(x,(t,))
        end
        oracle=O.persistence_diagram(DI.graded_complex(x);field=CM.Fp(5))
        @test sort(O.finite_intervals(O.persistence_diagram(h);dim=0))==sort(O.finite_intervals(oracle;dim=0))
        @test sort(O.essential_births(O.persistence_diagram(h);dim=0))==sort(O.essential_births(oracle;dim=0))
    end
end

@testset "A91 core mixed orientations and source coordinate dimensions" begin
    O,V=TamerOp.OrdinaryPersistence,TamerOp.Visualization
    G=DT.GradedComplex([[1,2],[3]],[sparse(reshape([-1,1],2,1))],[(0,4),(0,4),(2,2)])
    geom=[[(0,0,0)],[(1,1,1)],[(0,0,0),(1,1,1)]]
    x=DI.filtered_complex(G;geometry=geom,orientation=(1,-1))
    h=O.component_history(x;slice=(base=(0,4),direction=(1,-1)))
    @test O.component_count(h;scale=1)==2
    @test O.component_count(h;scale=2)==1
    @test O.component_parameter(h,2)==(2,2)
    @test V.check_visual_spec(V.visual_spec(h;scale=2)).valid
    @test V.check_visual_spec(V.visual_spec(h;scale=2,view_dims=(1,3))).valid
    @test_throws ArgumentError O.component_history(x;slice=(base=(0,4),direction=(1,1)))
    wide=DI.filtered_complex(G;geometry=[[tuple(p...,0) for p in cell] for cell in geom],orientation=(1,-1))
    hw=O.component_history(wide;slice=(base=(0,4),direction=(1,-1)))
    @test_throws ArgumentError V.visual_spec(hw)
    @test V.check_visual_spec(V.visual_spec(hw;view_dims=(1,4))).valid
end

@testset "A91 static specifications and session lifecycle" begin
    O,V=TamerOp.OrdinaryPersistence,TamerOp.Visualization
    h=O.component_history(_a91_graph());s=V.visual_spec(h;scale=2,component=2)
    @test V.check_visual_spec(s).valid
    @test s.metadata.selection.vertices==[1,2]
    @test s.metadata.selection.status==:merged
    @test s.metadata.parameter==(2,)
    @test length(s.panels)==4
    exact=V.visual_spec(h;scale=2-big(1)//big(2)^70)
    @test exact.panels[1].metadata.rounded_event_collision
    @test length(exact.metadata.partition)==3
    budget=V.visual_spec(h;scale=2,component=2,max_cells=1,max_branches=1,max_labels=0)
    @test budget.metadata.partition==s.metadata.partition
    @test budget.metadata.selection==s.metadata.selection
    @test budget.panels[1].metadata.records[1].component==2
    @test length(budget.panels[2].metadata.displayed_indices)==1
    @test length(budget.panels[3].metadata.displayed_nodes)<=1
    @test V.check_visual_spec(V.visual_spec(O.component_history(_a91_graph(Int[],Tuple{Int,Int}[],Int[])))).valid
    for kw in ((;scale=NaN),(;component=0),(;event=1),(;max_cells=0),(;max_branches=0),(;view_dims=(1,1)))
        @test !V.check_visual_request(h;kw...).valid
        @test_throws ArgumentError V.visual_spec(h;kw...)
    end
    session=V.inspection_session(h;scale=1,component=2)
    @test V.check_visual_spec(V.visual_spec(session)).valid
    V.select_inspection!(session;event=5)
    @test V.inspection_selection(session).scale==2
    @test V.inspection_snapshot(session).metadata.selection.vertices==[1,2]
    V.select_inspection!(session;scale=3)
    @test V.inspection_selection(session).event===nothing
    before=V.inspection_snapshot(session);revision=V.inspection_summary(session).revision
    @test_throws ArgumentError V.select_inspection!(session;component=100)
    @test V.inspection_snapshot(session)===before
    @test V.inspection_summary(session).revision==revision
    V.reset_inspection!(session)
    @test V.inspection_selection(session).scale==1
    V.close_inspection!(session)
    @test V.check_inspection_session(session).valid
    @test session.object===nothing
    @test_throws ArgumentError V.select_inspection!(session;scale=0)
end

@testset "A91 native Cairo WGL exports and controls" begin
    O,V=TamerOp.OrdinaryPersistence,TamerOp.Visualization
    if Base.find_package("CairoMakie")!==nothing && Base.find_package("WGLMakie")!==nothing
        @eval using CairoMakie,WGLMakie
        h=O.component_history(_a91_graph());spec=V.visual_spec(h;scale=2,component=2,event=5)
        mktempdir() do tmp
            for extension in ("png","svg","pdf")
                path=joinpath(tmp,"components.$extension");V.save_visual(path,spec;backend=:cairomakie)
                @test filesize(path)>100
            end
        end
        for palette in (:accessible,:grayscale),backend in (:cairomakie,:wglmakie)
            fig=V.render(spec;backend,style=V.VisualStyle(;palette))
            @test fig isa CairoMakie.Figure;empty!(fig)
        end
        session=V.inspection_session(h;scale=1)
        app=V.visualize(session;backend=:wglmakie)
        ext=Base.get_extension(TamerOp,:TamerOpWGLMakieExt);ui=ext._inspection_ui(app)
        ui.controls.component.value[]="2";ui.controls.select.value[]=true
        @test V.inspection_selection(session).component==2
        ui.controls.scale.value[]="2";ui.controls.apply_scale.value[]=true
        @test V.inspection_snapshot(session).metadata.selection.status==:merged
        @test isempty(V.inspection_summary(session).listener_errors)
        ui.controls.scale.value[]="bad";ui.controls.apply_scale.value[]=true
        @test !isempty(ui.error_text[])
        V.close_inspection!(session)
        @test ui.closed[] && isempty(ui.observers) && isempty(ui.pointer_observers)
        @test !haskey(ext._INSPECTION_UI_HANDLES,app)
    else;@test_skip false;end
end
