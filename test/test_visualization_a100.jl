using JSON3

# Saved selected views must survive source/session closure, preserve exact
# readouts, and reject inconsistent sources without ever loading executable code.

function _a100_square()
    OP=TamerOp; A=OP.Advanced
    OP.encode([A.BoxUpset([0,0])], [A.BoxDownset([2,2])], reshape(OP.QQ[1],1,1),
        A.EncodingOptions(;backend=:pl_backend,poset_kind=:signature,field=OP.CoreModules.QQField()))
end

@testset "A100 finite selection freeze, replay and source identity" begin
    V=TamerOp.Visualization; S=TamerOp.Serialization; A=TamerOp.Advanced
    enc=_a100_square()
    s=V.inspection_session(enc;box=([-1,-1],[3,3]))
    x=(big(1)//2^60,1//2); y=(3//2,3//2)
    V.select_inspection!(s;parameter_pair=(x,y))
    inside=V.visual_state(s)
    V.select_inspection!(s;point=(2,1))
    boundary=V.visual_state(s)
    V.select_inspection!(s;point=(5//2,1))
    outside=V.visual_state(s)
    @test inside.selection.query_points == (x,y)
    @test boundary.selection.query_points == ((2,1),)
    @test outside.selection.query_points == ((5//2,1),)
    @test V.describe(inside).resume_family === :finite_module
    @test V.check_visual_spec(inside.snapshot).valid
    @test !inside.snapshot.interaction.requires_live_julia
    @test TamerOp.visual_state === V.visual_state
    @test TamerOp.save_visual_session === V.save_visual_session
    @test A.visual_session_states === V.visual_session_states
    mktempdir() do dir
        path=joinpath(dir,"view.json")
        style=V.VisualStyle(fontsize=19)
        V.save_visual_session(path,(inside=inside,boundary=boundary,outside=outside);
            object=enc,selected=:boundary,style,size=(1000,700),caption="Exact square",
            provenance=(;field=:QQ,category=:finite_poset_modules,grade_scale=:coordinates))
        V.close_inspection!(s)
        saved=V.load_visual_session(path)
        @test V.check_visual_session(saved).valid
        @test V.visual_session_states(saved) == (:inside,:boundary,:outside)
        @test V.inspection_selection(saved;state=:inside).query_points == (x,y)
        @test saved.style.fontsize == 19
        @test saved.size == (1000,700)
        @test saved.provenance.field === :QQ
        @test V.describe(saved).source_mode === :embedded
        @test V.check_visual_request(saved;state=:inside).valid
        @test !V.check_visual_request(saved;state=:missing).valid
        @test !V.check_visual_request(saved;irrelevant=true).valid
        @test V.visual_spec(saved;state=:inside) === V.inspection_snapshot(saved;state=:inside)
        @test_throws ArgumentError V.inspection_selection(saved;state=:missing)
        @test_throws ArgumentError V.visual_spec(saved;cache=nothing)
        live=V.inspection_session(saved;state=:inside)
        @test V.inspection_selection(live).query_points == (x,y)
        @test live.cache_hits == 0
        @test isempty(live.listeners)
        @test V.check_inspection_session(live).valid
        V.select_inspection!(live;point=(1//3,1//3))
        @test V.inspection_selection(live).query_points == ((1//3,1//3),)
        V.close_inspection!(live)
        @test_throws ArgumentError V.inspection_session(saved;object=TamerOp.encoding_module(enc))
        # Snapshot-only persistence does not silently attach a runtime object.
        simple=joinpath(dir,"snap.json")
        V.save_visual_session(simple,inside)
        static=V.load_visual_session(simple)
        @test V.describe(static).source_mode === :none
        @test_throws ArgumentError V.inspection_session(static)
        @test V.visual_state(s).request.family === :snapshot
        # Every failed save leaves the old complete artifact untouched.
        original=read(path)
        @test_throws ArgumentError V.save_visual_session(path,(;))
        @test_throws ArgumentError V.save_visual_session(path,inside;selected=:missing)
        @test_throws ArgumentError V.save_visual_session(path,inside;max_bytes=10)
        @test read(path) == original
        @test_throws ArgumentError V.save_visual_session(joinpath(dir,"view.jl"),inside)
        @test_throws ArgumentError V.load_visual_session(path;max_bytes=10)
        raw=JSON3.read(read(path,String),Dict{String,Any})
        raw["schema_version"]=99
        write(simple,JSON3.write(raw)); @test_throws ArgumentError V.load_visual_session(simple)
        raw["schema_version"]=1; raw["payload"] *= " "
        write(simple,JSON3.write(raw)); @test_throws ArgumentError V.load_visual_session(simple)
        write(simple,"{}"); @test_throws ArgumentError V.load_visual_session(simple)
    end
end

@testset "A100 portable relative sources and changed objects" begin
    V=TamerOp.Visualization; S=TamerOp.Serialization; C=TamerOp.CoreModules
    P=TamerOp.FiniteFringe.FinitePoset(Bool[1 1;0 1])
    M=TamerOp.Modules.PModule{TamerOp.QQ}(P,[2,1],Dict((1,2)=>TamerOp.QQ[1 3]);field=C.QQField())
    s=V.inspection_session(M)
    V.select_inspection!(s;pair=(1,2))
    frame=V.visual_state(s)
    mktempdir() do dir
        bundle=joinpath(dir,"original");mkpath(joinpath(bundle,"objects"))
        artifact=joinpath(bundle,"objects","module.json")
        write(artifact,S.visual_source_to_json(M))
        path=joinpath(bundle,"view.json")
        V.save_visual_session(path,frame;object=M,object_path="objects/module.json")
        saved=V.load_visual_session(path)
        @test V.describe(saved).source_mode === :relative
        moved=joinpath(dir,"moved");mv(bundle,moved)
        loaded=V.load_visual_session(joinpath(moved,"view.json"))
        live=V.inspection_session(loaded)
        @test TamerOp.Modules.map_leq(live.object,1,2) == TamerOp.QQ[1 3]
        V.close_inspection!(live)
        moved_artifact=joinpath(moved,"objects","module.json")
        before=read(moved_artifact,String)
        write(moved_artifact,before*" ")
        @test V.check_visual_session(V.load_visual_session(joinpath(moved,"view.json"))).valid
        @test_throws ArgumentError V.inspection_session(loaded)
        rm(moved_artifact)
        @test_throws ArgumentError V.inspection_session(loaded)
        supplied=V.inspection_session(loaded;object=M)
        V.close_inspection!(supplied)
        wrong=TamerOp.Modules.PModule{TamerOp.QQ}(P,[2,1],Dict((1,2)=>TamerOp.QQ[1 4]);field=C.QQField())
        @test_throws ArgumentError V.inspection_session(loaded;object=wrong)
        # Attaching a source that computes different selected matrix values fails.
        @test_throws ArgumentError V.save_visual_session(joinpath(moved,"wrong.json"),frame;object=wrong)
        outside=joinpath(dir,"outside.json");write(outside,before)
        @test_throws ArgumentError V.save_visual_session(joinpath(moved,"escape.json"),frame;object_path="../outside.json")
        symlink(outside,moved_artifact)
        @test_throws ArgumentError V.save_visual_session(joinpath(moved,"escape.json"),frame;object_path="objects/module.json")
    end
    V.close_inspection!(s)
end

@testset "A100 ordinary source members, captions and malformed requests" begin
    V=TamerOp.Visualization; S=TamerOp.Serialization; O=TamerOp.OrdinaryPersistence
    values=zeros(Int,3,3);values[2,2]=5
    diagram=O.cubical_persistence(values;input=:top_cells,source=true,representatives=true)
    s=V.inspection_session(diagram;dim=1,window=(-1,6))
    V.select_inspection!(s;interval=1,member=1,representative=true)
    frame=V.visual_state(s)
    @test frame.request.family === :ordinary_intervals
    @test V.check_visual_spec(frame.snapshot).valid
    @test any(occursin("source",o.path) for o in frame.omissions)
    mktempdir() do dir
        path=joinpath(dir,"ring.json")
        V.save_visual_session(path,s;object=diagram,caption="[0,5): retained source cycle")
        V.close_inspection!(s)
        saved=V.load_visual_session(path)
        live=V.inspection_session(saved)
        @test live.selected_representative.available
        @test live.selected_representative.source_cycle.dimension == 1
        @test O.finite_intervals(live.object;dim=1) == [(0,5)]
        @test live.state.representative
        V.close_inspection!(live)
        record=S.read_visual_session(path)
        badframe=merge(record.states.view,(;request=(;family=:eval,options=(;),selection=(;))))
        bad=merge(record,(;states=(;view=badframe)))
        write(path,S.visual_session_to_json(bad))
        @test_throws ArgumentError V.load_visual_session(path)
        # An ordinary artifact is a separately owned portable resource too.
        artifact=joinpath(dir,"diagram.json")
        S.save_persistence_diagram_json(artifact,diagram)
        V.save_visual_session(path,frame;object=diagram,object_path="diagram.json")
        live=V.inspection_session(V.load_visual_session(path))
        @test O.finite_intervals(live.object;dim=1) == [(0,5)]
        V.close_inspection!(live)
    end
end

@testset "A100 declarative metadata ownership and typed validation" begin
    V=TamerOp.Visualization
    exact=reshape(TamerOp.QQ[1//3,-2],1,2)
    spec=V.VisualizationSpec(:exact_table;layers=V.AbstractVisualizationLayer[
        V.MatrixLayer(string.(exact),["v"],["u1","u2"])],
        metadata=(;panel_style=:matrix,exact,convenience=(;callback=identity,retained=(1//3,2//7))))
    frame=V.visual_state(spec;selection=(;coefficient=1//3))
    exact[1]=99
    @test frame.snapshot.metadata.exact[1] == 1//3
    @test !haskey(frame.snapshot.metadata.convenience,:callback)
    @test frame.snapshot.metadata.convenience.retained == (1//3,2//7)
    @test only(frame.omissions).path == "snapshot.metadata.convenience.callback"
    @test V.check_visual_spec(frame.snapshot).valid
    @test !isempty(sprint(show,frame))
    mktempdir() do dir
        path=joinpath(dir,"matrix.json")
        V.save_visual_session(path,frame;caption="Exact coefficients")
        saved=V.load_visual_session(path)
        @test V.inspection_snapshot(saved).metadata.exact == TamerOp.QQ[1//3 -2]
        @test V.VisualizationValidationSummary(V.check_visual_session(saved)).report.valid
        @test !isempty(sprint(show,saved))
    end
end
