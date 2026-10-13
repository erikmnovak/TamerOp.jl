include("fixtures/persistence_paths.jl")

@testset "A96 certified paths and independent reductions" begin
    O=TamerOp.OrdinaryPersistence;V=TamerOp.Visualization;D=TamerOp.DataTypes;C=TamerOp.CoreModules
    @test TamerOp.persistence_path===V.persistence_path
    @test TOA.PersistencePath===V.PersistencePath
    @test TOA.path_transitions===V.path_transitions
    support=V.visual_spec(_a96_path();frame=1,track=2,representative=true,scale=2).panels[5]
    highlight=last(support.layers)
    @test highlight isa V.PointLayer && highlight.markersize==18.0
    @test highlight.color==V._VisualRole(:categorical,2)
    @test Set(highlight.points)==Set([(-1.0,0.0),(5.0,0.0)])
    for F in (C.F2(),C.F3(),C.Fp(5)),superlevel in (false,true),degree in (0,1)
        p=_a96_path(;degree,superlevel,field=F)
        @test V.check_persistence_path(p).valid
        @test p.mode===:certified && length(p.frames)>3
        @test all(t->t.kind===:certified,p.transitions)
        @test all(t->isempty(t.unmatched_before)&&isempty(t.unmatched_after),p.transitions)
        @test V.persistence_path_summary(p).representatives
        original=_a96_triangle(;superlevel).complex
        function fresh(values)
            g=D.GradedComplex(original.cells_by_dim,original.boundaries,[(v,) for v in values])
            d=O.persistence_diagram(g;field=F,order=p.order)
            sort!(vcat(collect(O.finite_intervals(d;dim=degree)),[(b,p.order===:sublevel ? Inf : -Inf) for b in O.essential_births(d;dim=degree)]))
        end
        for (i,f) in enumerate(p.frames)
            @test sort!([(r.birth,r.death) for r in f.records if !r.zero])==fresh(f.grades)
            if degree==1
                birth=superlevel ? minimum(f.grades[4:6]) : maximum(f.grades[4:6])
                @test [(r.birth,r.death) for r in f.records]==[(birth,f.grades[7])]
            end
            for r in f.records
                rep=r.representative;rep===nothing && continue
                K=C.coeff_type(F)
                function vector(chain,d)
                    ids=original.dim_offsets[d+1]:original.dim_offsets[d+1]+length(original.cells_by_dim[d+1])-1
                    K[get(Dict(chain),j,0) for j in ids]
                end
                cycle=vector(rep.cycle,degree)
                degree==0 || @test iszero(K.(Matrix(original.boundaries[degree]))*cycle)
                if rep.bounding_chain!==nothing
                    @test K.(Matrix(original.boundaries[degree+1]))*vector(rep.bounding_chain,degree+1)==cycle
                end
                @test all(k->p.order===:sublevel ? f.grades[first(k)]<=r.birth : f.grades[first(k)]>=r.birth,rep.cycle)
            end
            i==length(p.frames) && continue
            after=p.frames[i+1]
            f.parameter==after.parameter && continue
            # No missing crossing: interpolate interval endpoints halfway and
            # compare with a separate fresh reduction of the entire complex.
            later=Dict(r.id=>r for r in after.records)
            midpoint=[((r.birth+later[r.id].birth)/2,isfinite(r.death) ? (r.death+later[r.id].death)/2 : r.death) for r in f.records]
            @test sort!(filter(x->x[1]!=x[2],midpoint))==fresh((f.grades.+after.grades)./2)
        end
        spec=V.visual_spec(p;frame=length(p.frames),track=1,representative=true,scale=superlevel ? 4 : 6)
        @test V.check_visual_spec(spec).valid
        @test spec.metadata.selected_record.id==1
        @test spec.panels[3].axes.xlimits==spec.panels[4].axes.xlimits
        @test spec.panels[1].axes.ylimits==spec.panels[2].axes.ylimits
        @test V.check_visual_spec(V.visual_spec(p;view_dims=(2,1))).valid
        projected=V.visual_spec(p;view_dims=(2,1))
        @test projected.panels[5].metadata.view_dims==(2,1)
        @test V.path_frames(p)==p.frames
        transitions=V.path_transitions(p)
        @test transitions==p.transitions
        empty!(transitions[1].links);@test !isempty(p.transitions[1].links)
        copy=V.path_frames(p);empty!(copy[1].records);@test !isempty(p.frames[1].records)
        @test length(V.path_tracks(p))==V.persistence_path_summary(p).tracks
    end
end

@testset "A96 supplied matching, ambiguity and strict contracts" begin
    V=TamerOp.Visualization;O=TamerOp.OrdinaryPersistence;C=TamerOp.CoreModules;D=TamerOp.DataTypes
    p=_a96_supplied()
    @test V.check_persistence_path(p).valid
    @test all(t->t.kind===:matched&&t.ambiguous,p.transitions)
    @test p.transitions[1].distance==1
    objective=only(filter(s->s.kind===:path_objective,V.visual_spec(p).panels))
    @test getproperty.(objective.metadata.exact_values,:value)==[3,2,1]
    @test getproperty.(objective.metadata.exact_values,:parameter)==[0,1//3,2]
    @test V.visual_spec(p;frame=2).metadata.observation.settings.iteration==2
    @test length(p.transitions[1].unmatched_before)==1
    @test length(p.transitions[2].unmatched_before)==1
    @test length(unique(r.id for r in p.frames[1].records))==4
    badmatch=deepcopy(p)
    badmatch.transitions[1]=merge(badmatch.transitions[1],(;distance=99))
    @test !V.check_persistence_path(badmatch).valid
    @test V.check_visual_spec(V.visual_spec(p;track=1,max_tracks=1)).valid
    q=_a96_supplied(;matching=:none)
    @test isempty(intersect([r.id for r in q.frames[1].records],[r.id for r in q.frames[2].records]))
    @test all(isempty(t.links) for t in q.transitions)
    @test V.check_persistence_path(q).valid
    # Coincident zero pairs and simultaneous crossings remain recorded, even
    # though diagonal pairs do not contribute to an ordinary barcode.
    G=D.GradedComplex([[1,2],[3]],[sparse(reshape([-1,1],2,1))],[(0,),(0,),(0,)])
    z=V.persistence_path(G;grades=[[1,1,1]],representatives=true)
    @test any(r->r.zero,z.frames[1].records)
    @test length(V.visual_spec(z).panels[4].metadata.records)==1
    tied=D.GradedComplex([[1,2,3]],SparseMatrixCSC{Int,Int}[],[(0,),(1,),(2,)])
    ties=V.persistence_path(tied;grades=[[2,1,0]])
    @test any(t->t.ambiguous,ties.transitions)
    @test any(e->e.tie===:simultaneous,Iterators.flatten(t.events for t in ties.transitions))
    @test V.check_persistence_path(ties).valid
    empty=O.PersistenceDiagram([Tuple{Int,Int}[]],[Int[]])
    @test_throws ArgumentError V.persistence_path([empty];observations=[(;objective=Inf)])
    @test_throws ArgumentError V.persistence_path([empty];observations=[(;settings=sin)])
    @test_throws ArgumentError V.persistence_path([empty];observations=[])
    sources=[V.visual_spec(_a96_triangle();parameter=1)]
    supplied=V.persistence_path([empty];sources)
    @test V.check_visual_spec(V.visual_spec(supplied)).valid
    @test V.path_frames(supplied)[1].source!==sources[1]
    retained=O.persistence_diagram(_a96_triangle().complex;representatives=true)
    @test !V.persistence_path_summary(V.persistence_path([retained])).representatives
    @test V.persistence_path_summary(V.persistence_path([retained];representatives=true)).representatives
    @test V.check_visual_spec(V.visual_spec(V.persistence_path([empty]))).valid
    @test V.check_persistence_path(V.persistence_path([empty,empty];matching=:bottleneck)).valid
    essential=O.PersistenceDiagram([Tuple{Int,Int}[]],[[0]])
    truncated=O.PersistenceDiagram(essential.finite_by_dim,essential.essential_by_dim,essential.field,essential.order,
        (;essential_interpretation=:survives_radius_cutoff))
    @test occursin("construction cutoff",join(V.visual_spec(V.persistence_path([truncated])).panels[end].layers[1].labels," "))
    infinite=V.persistence_path([essential,empty];matching=:bottleneck)
    @test isinf(infinite.transitions[1].distance) && V.check_persistence_path(infinite).valid
    @test occursin("infinite cost",join(V.visual_spec(infinite;frame=2).panels[end].layers[1].labels," "))
    fractional=O.PersistenceDiagram([[(1//3,7//3)]],[Rational{Int}[]])
    @test V.check_persistence_path(V.persistence_path([fractional,fractional];matching=:bottleneck)).valid
    sup=O.PersistenceDiagram([[(7//3,1//3)]],[[4//1]];order=:superlevel)
    @test V.check_persistence_path(V.persistence_path([sup,sup];matching=:bottleneck)).valid
    other=O.PersistenceDiagram([[(0,1)]],[Int[]];field=C.F3())
    for kw in ((;matching=:certified),(;parameters=[0,0]),(;parameters=[0,Inf]),(;max_frames=1),
        (;max_records=1),(;degree=true),(;sources=[V.visual_spec(empty)]))
        @test_throws ArgumentError V.persistence_path([empty,O.PersistenceDiagram([[(0,1),(2,4)]],[Int[]])];kw...)
    end
    @test_throws ArgumentError V.persistence_path([empty,other])
    @test_throws ArgumentError V.persistence_path([other,other];matching=:bottleneck,max_matching_entries=1)
    for kw in ((;max_frames=3),(;max_records=1),(;order=:superlevel),(;field=C.QQField()),(;parameters=[0,0,1]))
        @test_throws ArgumentError V.persistence_path(_a96_triangle();grades=[[2,0,1,4,6,3,8],[1,2,0,5,4,6,9]],kw...)
    end
    @test_throws ArgumentError V.persistence_path(_a96_triangle();grades=[[1,2]])
    @test_throws ArgumentError V.persistence_path(_a96_triangle();grades=[[0,0,0,-1,1,1,2]])
    large=big(2)^60
    collision=V.persistence_path([O.PersistenceDiagram([[(large,large+1)]],[BigInt[]])])
    @test_throws ArgumentError V.visual_spec(collision)
    malformed=deepcopy(ties);malformed.frames[1].records[1]=merge(malformed.frames[1].records[1],(;birth=999))
    @test !V.check_persistence_path(malformed).valid
    @test_throws ArgumentError V.check_persistence_path(malformed;throw=true)
end

@testset "A96 atomic linked selections and saved state" begin
    V=TamerOp.Visualization;p=_a96_path();s=V.inspection_session(p)
    @test V.check_inspection_session(s).valid
    @test V.check_visual_spec(V.visual_spec(s)).valid
    V.select_inspection!(s;frame=length(p.frames),track=1,representative=true)
    @test V.inspection_snapshot(s).metadata.representative!==nothing
    last=V.inspection_selection(s)
    @test_throws ArgumentError V.select_inspection!(s;frame=10000)
    @test V.inspection_selection(s)==last
    @test !V.check_inspection_selection(s;track=10000).valid
    event=findfirst(f->f.keyframe==0,p.frames)
    V.select_inspection!(s;frame=event)
    @test V.inspection_snapshot(s).metadata.representative===nothing
    @test V.inspection_selection(s).track==1
    @test occursin("crossing state",join(V.inspection_snapshot(s).panels[end].layers[1].labels," "))
    V.reset_inspection!(s);@test V.inspection_selection(s).frame==1
    owner=Ref(nothing)
    for frame in 1:length(p.frames);V._request_inspection!(s,owner;frame);end
    wait(s.requests.task)
    @test V.inspection_selection(s).frame==length(p.frames)
    @test s.requests.discarded>0
    @test isempty(V.inspection_summary(s).listener_errors)
    frozen=V.visual_state(s)
    @test frozen.snapshot.kind===:persistence_path
    V.close_inspection!(s)
    @test V.check_inspection_session(s).valid && s.object===nothing
    @test_throws ArgumentError V.select_inspection!(s;frame=1)
    @test V.inspection_snapshot(s).metadata.parameter==p.frames[end].parameter
end

@testset "A96 native renderers and offline export" begin
    V=TamerOp.Visualization;p=_a96_path(;degree=1);s=V.inspection_session(p;track=1,representative=true,scale=6)
    spec=V.inspection_snapshot(s)
    for (backend,name) in ((:cairomakie,:TamerOpCairoMakieExt),(:wglmakie,:TamerOpWGLMakieExt))
        ext=Base.get_extension(TamerOp,name)
        if ext===nothing
            @test_skip false
            continue
        end
        fig=V.render(spec;backend)
        M=backend===:cairomakie ? ext.CairoMakie.Makie : ext.WGLMakie.Makie
        axes=filter(x->x isa M.Axis,fig.content)
        @test length(axes)==5
        @test Tuple(axes[1].limits[][1:2])==(spec.panels[1].axes.xlimits,spec.panels[1].axes.ylimits)
        @test !isempty(axes[1].scene.plots)
        for (axis,expected) in ((axes[1],(-2/3,5.0)),(axes[2],(-2/3,7.0)),(axes[3],(5.0,7.0)))
            positions=[Tuple(q) for plot in axis.scene.plots if plot isa M.Scatter for q in plot[1][]]
            @test any(q->isapprox(q[1],expected[1];atol=1e-6)&&isapprox(q[2],expected[2];atol=1e-6),positions)
        end
        empty!(fig)
        if backend===:wglmakie
            app=V.visualize(s;backend);ui=ext._inspection_ui(app)
            ui.callbacks.step(1);@test s.state.frame==2
            ui.callbacks.slide(Float64(length(p.frames)));@test s.state.frame==length(p.frames)
            ui.callbacks.slide(1.5);@test s.state.frame==length(p.frames)
            @test occursin("integer frame",ui.error_text[])
            @test ui.update_count[]>=1
            @test ui.rebuild_count[]==1
            ui.callbacks.start(-1);@test ui.playing[]
            ui.callbacks.stop();@test !ui.playing[] && ui.timer[]===nothing
            ui.dispose();@test isempty(ui.observers) && isempty(ui.pointer_observers)
            # Closing during an outstanding notification must stop playback
            # immediately and finish after the current update releases its view.
            closing=V.inspection_session(p)
            closing_ui=ext._inspection_ui(V.visualize(closing;backend))
            entered=Channel{Nothing}(1);release=Channel{Nothing}(1)
            V._on_inspection(closing, current -> begin
                if !V.inspection_summary(current).closed
                    put!(entered,nothing);take!(release)
                end
            end)
            closing_ui.callbacks.start(1)
            update=@async V.select_inspection!(closing;frame=2)
            take!(entered)
            closing_ui.controls.close.value[]=true
            @test !closing_ui.playing[] && closing_ui.timer[]===nothing
            @test !V.inspection_summary(closing).closed
            put!(release,nothing);wait(update);wait(closing_ui.close_task[])
            @test V.inspection_summary(closing).closed && closing_ui.closed[]
            @test isempty(V.inspection_summary(closing).listener_errors)
            @test isempty(closing_ui.observers) && isempty(closing_ui.pointer_observers)
        else
            mktempdir() do dir
                for suffix in ("svg","png","pdf")
                    file=joinpath(dir,"path.$suffix");V.save_visual(file,spec;backend)
                    @test filesize(file)>1000
                end
            end
        end
    end
    if Base.get_extension(TamerOp,:TamerOpCairoMakieExt)!==nothing
        V.reset_inspection!(s);first=V.visual_state(s)
        V.select_inspection!(s;frame=length(p.frames));last=V.visual_state(s)
        dir=get(ENV,"TAMEROP_PATH_HTML","")
        mktempdir() do tmp
            file=isempty(dir) ? joinpath(tmp,"path.html") : dir
            mkpath(dirname(file))
            V.save_visual_session(file,(;start=first,finish=last);selected=:start,title="A persistence path")
            @test isfile(file)
            json=joinpath(tmp,"path.json")
            V.save_visual_session(json,(;start=first,finish=last);selected=:start)
            loaded=V.load_visual_session(json)
            @test V.inspection_snapshot(loaded).metadata.mode===:certified
            @test V.inspection_snapshot(loaded).metadata.exact_frames==first.snapshot.metadata.exact_frames
        end
    end
    V.close_inspection!(s)
end
