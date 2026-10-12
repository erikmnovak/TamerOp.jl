# Saved sessions must reject malformed declarations and inconsistent mathematical
# readouts even when an outer JSON envelope has a freshly recomputed digest.

function _a100_boundary_module(; last_entry=3)
    P = TamerOp.FiniteFringe.FinitePoset(Bool[1 1 1; 0 1 1; 0 0 1])
    A = TamerOp.QQ[1 0 0; 0 1 0; 0 0 1]
    B = TamerOp.QQ[1 0 0; 0 2 0; 0 0 last_entry]
    return TamerOp.Modules.PModule{TamerOp.QQ}(P, [3, 3, 3],
        Dict((1, 2)=>A, (2, 3)=>B); field=TamerOp.CoreModules.QQField())
end

@testset "A100 malformed typed session records fail at the load boundary" begin
    V = TamerOp.Visualization
    S = TamerOp.Serialization
    spec = V.VisualizationSpec(:exact_table;
        layers=V.AbstractVisualizationLayer[
            V.MatrixLayer(reshape(["1/3"], 1, 1), ["target"], ["source"])],
        metadata=(; panel_style=:matrix, exact=1//3))
    frame = V.visual_state(spec; selection=(; entry=(1, 1)))
    mktempdir() do directory
        path = joinpath(directory, "view.json")
        V.save_visual_session(path, frame)
        record = S.read_visual_session(path)
        @test V.check_visual_session(V.load_visual_session(path)).valid

        # All replacements are declarative values accepted by the scalar codec.
        # The envelope is therefore valid; these exercise session schema checks.
        malformed = (
            :states_array=>merge(record, (; states=Int[])),
            :empty_states=>merge(record, (; states=(;))),
            :selected_integer=>merge(record, (; selected=17)),
            :style_record=>merge(record, (; style=(; palette=:default))),
            :size_arity=>merge(record, (; size=(100, 100, 100))),
            :title_integer=>merge(record, (; title=17)),
            :caption_tuple=>merge(record, (; caption=(1, 2))),
            :provenance_array=>merge(record, (; provenance=Int[])),
            :source_empty=>merge(record, (; source=(;))),
        )
        for (name, bad) in malformed
            @testset "$name" begin
                write(path, S.visual_session_to_json(bad))
                @test_throws ArgumentError V.load_visual_session(path)
            end
        end

        bad_frames = (
            :snapshot_integer=>merge(record.states.view, (; snapshot=17)),
            :selection_array=>merge(record.states.view, (; selection=Int[])),
            :request_symbol=>merge(record.states.view, (; request=:untyped)),
            :omission_integer=>merge(record.states.view, (; omissions=[17])),
            :omission_missing_field=>merge(record.states.view,
                (; omissions=[(; path="snapshot.metadata", type="Function")])),
            :omission_wrong_type=>merge(record.states.view,
                (; omissions=[(; path=17, type="Function", reason="runtime")])),
            :omission_extra_field=>merge(record.states.view,
                (; omissions=[(; path="snapshot.metadata", type="Function",
                    reason="runtime", extra=true)])),
        )
        for (name, bad_frame) in bad_frames
            @testset "$name" begin
                bad = merge(record, (; states=(; view=bad_frame)))
                write(path, S.visual_session_to_json(bad))
                @test_throws ArgumentError V.load_visual_session(path)
            end
        end

        bad_source = merge(record, (; source=(; mode=:none, identity=repeat("0", 64))))
        write(path, S.visual_session_to_json(bad_source))
        @test_throws ArgumentError V.load_visual_session(path)

        request = (; family=:execute_source_text, options=(;), selection=(;),
            source_identity=repeat("0", 64))
        bad_frame = merge(record.states.view, (; request))
        write(path, S.visual_session_to_json(merge(record, (; states=(; view=bad_frame)))))
        @test_throws ArgumentError V.load_visual_session(path)

        request = merge(record.states.view.request, (; source_identity=repeat("0", 64)))
        bad_frame = merge(record.states.view, (; request))
        write(path, S.visual_session_to_json(merge(record, (; states=(; view=bad_frame)))))
        @test_throws ArgumentError V.load_visual_session(path)
    end
end

@testset "A100 nested specification metadata has a finite capture bound" begin
    V = TamerOp.Visualization
    tail = Dict{Symbol,Any}()
    layers = V.AbstractVisualizationLayer[
        V.TextLayer(["Retained value"], [(0.0, 0.0)], :black, 12.0)]
    nested = V.VisualizationSpec(:nested_data; layers, metadata=(; tail))
    for _ in 1:80
        nested = V.VisualizationSpec(:nested_data; layers, metadata=(; child=nested))
    end
    tail[:back_to_root] = nested
    @test V.check_visual_spec(nested).valid
    # This metadata graph crosses specifications as well as a dictionary. The
    # capture depth budget must be propagated across those owner boundaries.
    @test_throws ArgumentError V.visual_state(nested)
end

@testset "A100 replay verifies undisplayed exact data and figure layout" begin
    V = TamerOp.Visualization
    S = TamerOp.Serialization
    M = _a100_boundary_module()
    session = V.inspection_session(M; matrix_limit=(1, 1))
    V.select_inspection!(session; pair=(1, 2))
    frame = V.visual_state(session)
    @test frame.snapshot.metadata.inspection.truncated
    @test size(frame.snapshot.metadata.inspection.matrix) == (3, 3)
    @test size(only(frame.snapshot.panels[2].layers).entries) == (1, 1)
    try
        mktempdir() do directory
            path = joinpath(directory, "map.json")
            V.save_visual_session(path, frame; object=M)
            original = read(path)
            record = S.read_visual_session(path)
            live = V.inspection_session(V.load_visual_session(path))
            @test V.inspection_snapshot(live).metadata.inspection.matrix ==
                TamerOp.Modules.structure_map(M; source=1, target=2)
            V.close_inspection!(live)

            tampered = deepcopy(frame)
            tampered.snapshot.metadata.inspection.matrix[3, 3] = 2
            @test only(tampered.snapshot.panels[2].layers).entries ==
                only(frame.snapshot.panels[2].layers).entries
            @test V.check_visual_spec(tampered.snapshot).valid
            @test_throws ArgumentError V.save_visual_session(path, tampered; object=M)
            @test read(path) == original

            # Recompute the file digest to exercise mathematical replay rather
            # than merely the outer integrity checksum.
            bad_record = deepcopy(record)
            bad_record.states.view.snapshot.metadata.inspection.matrix[3, 3] = 2
            write(path, S.visual_session_to_json(bad_record))
            restored = V.load_visual_session(path)
            @test V.check_visual_session(restored).valid
            @test_throws ArgumentError V.inspection_session(restored)

            snapshot = frame.snapshot
            relaid = V.VisualizationSpec(snapshot.kind; title=snapshot.title,
                subtitle=snapshot.subtitle, layers=snapshot.layers, panels=snapshot.panels,
                axes=snapshot.axes, legend=snapshot.legend, interaction=snapshot.interaction,
                metadata=merge(snapshot.metadata, (; figure_size=(800, 600))))
            changed_layout = V.VisualSessionState(relaid, frame.selection, frame.request, frame.omissions)
            @test V.check_visual_spec(relaid).valid
            @test_throws ArgumentError V.save_visual_session(path, changed_layout; object=M)
        end
    finally
        V.close_inspection!(session)
    end
end

@testset "A100 source changes outside the selection invalidate its identity" begin
    V = TamerOp.Visualization
    S = TamerOp.Serialization
    M = _a100_boundary_module()
    changed = _a100_boundary_module(; last_entry=7)
    session = V.inspection_session(M)
    other = V.inspection_session(changed)
    try
        V.select_inspection!(session; pair=(1, 2))
        V.select_inspection!(other; pair=(1, 2))
        frame = V.visual_state(session)
        @test V.inspection_snapshot(session).metadata.inspection.matrix ==
            V.inspection_snapshot(other).metadata.inspection.matrix
        @test S.visual_source_identity(M) != S.visual_source_identity(changed)
        mktempdir() do directory
            path = joinpath(directory, "selected.json")
            @test_throws ArgumentError V.save_visual_session(path, frame; object=changed)
            @test !isfile(path)
            V.save_visual_session(path, frame; object=M)
            saved = V.load_visual_session(path)
            @test_throws ArgumentError V.inspection_session(saved; object=changed)
        end
    finally
        V.close_inspection!(session)
        V.close_inspection!(other)
    end
end

@testset "A100 clearing an ordinary member survives save and replay" begin
    V = TamerOp.Visualization
    O = TamerOp.OrdinaryPersistence
    values = zeros(Int, 3, 3)
    values[2, 2] = 5
    diagram = O.cubical_persistence(values; input=:top_cells, representatives=true)
    session = V.inspection_session(diagram; dim=1, window=(-1, 6))
    try
        V.select_inspection!(session; interval=1, member=1, representative=true)
        V.select_inspection!(session; member=0)
        no_member = V.visual_state(session)
        @test no_member.selection.interval == 1
        @test no_member.selection.member === nothing
        @test !no_member.selection.representative
        V.select_inspection!(session; interval=0)
        no_interval = V.visual_state(session)
        @test no_interval.selection.interval === nothing
        @test no_interval.selection.member === nothing
        @test !no_interval.selection.representative
        mktempdir() do directory
            path = joinpath(directory, "cleared.json")
            V.save_visual_session(path, (; no_member, no_interval); object=diagram)
            saved = V.load_visual_session(path)
            live = V.inspection_session(saved; state=:no_member)
            state = V.inspection_selection(live)
            @test state.interval == 1
            @test state.member === nothing
            @test !state.representative
            V.close_inspection!(live)
            live = V.inspection_session(saved; state=:no_interval)
            state = V.inspection_selection(live)
            @test state.interval === nothing
            @test state.member === nothing
            @test !state.representative
            @test O.finite_intervals(live.object; dim=1) == [(0, 5)]
            V.close_inspection!(live)
        end
    finally
        V.close_inspection!(session)
    end
end

@testset "A100 byte limits and lexical resource declarations" begin
    S=TamerOp.Serialization; V=TamerOp.Visualization
    for limit in (true,false,0,-1,:invalid,"100",big(typemax(Int))+1)
        @test_throws ArgumentError S.visual_session_to_json((;);max_bytes=limit)
        @test_throws ArgumentError S.visual_session_from_json("{}";max_bytes=limit)
        @test_throws ArgumentError S.visual_session_reference(;max_bytes=limit)
        @test_throws ArgumentError S.visual_session_source((;mode=:none,identity=nothing),pwd();max_bytes=limit)
    end
    @test_throws ArgumentError S.visual_session_source((;mode=:embedded,payload="oversized"),pwd();max_bytes=2)
    spec=V.VisualizationSpec(:resource_fixture;layers=V.AbstractVisualizationLayer[
        V.TextLayer(["Retained view"],[(0.0,0.0)],:black,12.0)])
    mktempdir() do dir
        path=joinpath(dir,"view.json")
        V.save_visual_session(path,V.visual_state(spec))
        record=S.read_visual_session(path)
        for relative in ("", "/absolute.json", "../outside.json", "a/../../outside.json", "a//b.json", "C:\\file.json", "nul\0.json")
            source=(;mode=:relative,identity=repeat("0",64),path=relative,sha256=repeat("0",64))
            write(path,S.visual_session_to_json(merge(record,(;source))))
            @test_throws ArgumentError V.load_visual_session(path)
        end
        source=(;mode=:relative,identity=repeat("0",64),path="missing/source.json",sha256=repeat("0",64))
        write(path,S.visual_session_to_json(merge(record,(;source))))
        @test V.check_visual_session(V.load_visual_session(path)).valid
    end
end
