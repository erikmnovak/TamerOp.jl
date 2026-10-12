# Captured correspondence records preserve exact displayed identities while
# explicitly distinguishing archived checks from live representative queries.
function _a100_composed_ring()
    V = TamerOp.Visualization
    O = TamerOp.OrdinaryPersistence
    values = zeros(Int, 3, 3)
    values[2, 2] = 5
    diagram = O.cubical_persistence(values; input=:top_cells,
        source=true, representatives=true, field=TamerOp.CoreModules.F2())
    cycle = O.persistence_representative(diagram; dim=1, scale=2).source_cycle
    note = V.correspondence_annotation(diagram, cycle; label="Retained square boundary", curve=0.12)
    spec = V.compose_visuals((source=V.visual_panel(cycle),
        barcode=V.visual_panel(diagram; kind=:barcode, dim=1)); annotations=(note,))
    return spec, cycle
end

@testset "A100 complete composed-view save and reopen" begin
    V = TamerOp.Visualization
    original, _ = _a100_composed_ring()
    frame = V.visual_state(original; selection=(; interval=(; dim=1, kind=:finite, index=1)))
    @test V.describe(frame).resume_family === :snapshot
    @test any(omission -> endswith(omission.path, "composition_object"), V.describe(frame).omissions)
    @test all(omission -> !isempty(omission.type) && !isempty(omission.reason), V.describe(frame).omissions)
    mktempdir() do directory
        path = joinpath(directory, "source-and-interval.json")
        V.save_visual_session(path, (; source_and_interval=frame);
            caption="A chosen boundary on the retained source and its interval", provenance=(; fixture=:cubical_ring))
        reopened = V.load_visual_session(path)
        snapshot = V.inspection_snapshot(reopened)
        @test V.check_visual_session(reopened).valid
        @test V.visual_session_states(reopened) == (:source_and_interval,)
        @test V.inspection_selection(reopened).interval == (; dim=1, kind=:finite, index=1)
        @test V.check_visual_spec(snapshot).valid
        @test snapshot.axes == frame.snapshot.axes
        @test snapshot.panels[1].axes == original.panels[1].axes
        note = only(snapshot.metadata.annotations)
        @test V.describe(note).meaning === :captured_computed_correspondence
        @test note.evidence.interval == (0, 5)
        @test note.evidence.support == only(frame.snapshot.metadata.annotations).evidence.support
        @test note.from.point == V._composition_anchor(original.panels[1], only(original.metadata.annotations).from).point
        @test !haskey(snapshot.panels[1].metadata, :composition_object)
        @test snapshot.interaction.requires_live_julia === false
        @test_throws ArgumentError V.inspection_session(reopened)
    end

    # Capture owns the selected coefficients rather than aliasing a mutable
    # original drawing matrix. It does not confuse labels with scalar values.
    entries = ["1//3" "-2"]
    matrix = V.VisualizationSpec(:coefficients; layers=V.AbstractVisualizationLayer[
        V.MatrixLayer(entries, ["target"], ["first", "second"])], metadata=(; panel_style=:matrix))
    captured = V.visual_state(matrix)
    entries[1, 1] = "changed original"
    @test only(captured.snapshot.layers).entries == ["1//3" "-2"]
end

function _a100_without_composition_objects(spec)
    V = TamerOp.Visualization
    metadata = (; (name => value for (name, value) in pairs(spec.metadata) if name !== :composition_object)...)
    haskey(metadata, :overlay_specs) && (metadata = merge(metadata,
        (; overlay_specs=map(_a100_without_composition_objects, metadata.overlay_specs))))
    return V._composition_copy(spec; metadata,
        panels=V.VisualizationSpec[_a100_without_composition_objects(panel) for panel in spec.panels])
end

@testset "A100 frozen composition identities and verification meaning" begin
    V = TamerOp.Visualization
    S = TamerOp.Serialization
    original, cycle = _a100_composed_ring()
    old = only(original.metadata.annotations)
    frozen = V._freeze_visual_composition(original)
    saved = _a100_without_composition_objects(frozen)
    note = only(saved.metadata.annotations)
    evidence = note.evidence
    @test V.check_visual_spec(saved).valid
    @test V.describe(old).meaning === :computed_correspondence
    @test V.describe(note).meaning === :captured_computed_correspondence
    @test evidence.mode === :frozen_correspondence
    @test evidence.verification === :checked_before_capture
    @test evidence.interval == (0, 5)
    @test evidence.parameter == (2,)
    @test evidence.degree == 1
    @test evidence.member == (; kind=:finite, index=1)
    @test evidence.field == TamerOp.CoreModules.F2()
    # The persistence reduction chooses a representative; its support need not
    # be the four shortest edges around the missing central square.
    @test evidence.support.cell_indices == Tuple(findall(!iszero, cycle.coefficients))
    @test !isempty(evidence.support.cell_indices)
    @test all(==(1), evidence.support.coefficients)
    @test note.from.target === nothing
    @test note.from.point == V._composition_anchor(original.panels[1], old.from).point
    @test note.to.target == (:interval, 1, :finite, 1)
    @test V._composition_anchor(saved.panels[2], note.to).point == (2.5, 1.0)
    @test length(saved.metadata.captured_source_anchors) == 1
    @test V.describe(only(saved.metadata.captured_source_anchors).original) == V.describe(old.from)
    @test !haskey(evidence, :diagram) && !haskey(evidence, :cycle)
    @test !haskey(saved.panels[1].metadata, :composition_object)
    @test only(original.metadata.annotations).evidence.cycle === cycle
    @test old.from.target[1] === :source_cell
    twice = V._freeze_visual_composition(saved)
    @test length(twice.metadata.captured_source_anchors) == 1
    @test V.check_visual_spec(twice).valid
    restored = S.visual_data_from_json(S.visual_data_to_json(note))
    @test isequal(V.describe(restored), V.describe(note))
    @test isequal(restored.evidence.support, evidence.support)

    function replaced(annotation)
        V._composition_copy(saved; metadata=merge(saved.metadata, (; annotations=(annotation,))))
    end
    moved = V.VisualAnnotation(V.VisualAnchor(:source; point=(99, 99)), note.to,
        note.kind, note.label, note.curve, evidence)
    @test !V.check_visual_spec(replaced(moved)).valid
    @test_throws ArgumentError V.check_visual_spec(replaced(moved); throw=true)
    for bad in (merge(evidence, (; verification=:computed_after_reopen)),
            merge(evidence, (; degree=-1)),
            merge(evidence, (; interval=(0, 6))),
            merge(evidence, (; member=(; kind=:finite, index=2))),
            merge(evidence, (; support=merge(evidence.support,
                (; coefficients=Tuple(0 for _ in evidence.support.coefficients))))))
        invalid = V.VisualAnnotation(note.from, note.to, note.kind, note.label, note.curve, bad)
        @test !V.check_visual_spec(replaced(invalid)).valid
    end
    # A live checked relation still requires the original runtime objects.
    @test !V.check_visual_spec(_a100_without_composition_objects(original)).valid
end

@testset "A100 authored and matrix anchors retain their separate meanings" begin
    V = TamerOp.Visualization
    original, _ = _a100_composed_ring()
    old = only(original.metadata.annotations)
    authored = V.VisualAnnotation(old.from, old.to; label="An authored comparison")
    figure = V.compose_visuals((source=original.panels[1], barcode=original.panels[2]); annotations=(authored,))
    frozen = _a100_without_composition_objects(V._freeze_visual_composition(figure))
    note = only(frozen.metadata.annotations)
    @test V.check_visual_spec(frozen).valid
    @test V.describe(note).meaning === :authored_explanation
    @test note.evidence === nothing
    @test frozen.metadata.correspondence_verification === :no_computed_correspondence
    @test note.from.point == V._composition_anchor(original.panels[1], old.from).point
    matrix = V.VisualizationSpec(:coefficients; layers=V.AbstractVisualizationLayer[
        V.MatrixLayer(["1//3" "-2"], ["target"], ["first", "second"])], metadata=(; panel_style=:matrix))
    link = V.VisualAnnotation(V.VisualAnchor(:matrix; target=(:matrix, 1, 2)), old.from;
        label="Coefficient and a source cell")
    composed = V.compose_visuals((matrix=matrix, source=original.panels[1]); annotations=(link,))
    captured = _a100_without_composition_objects(V._freeze_visual_composition(composed))
    @test V.check_visual_spec(captured).valid
    @test only(captured.metadata.annotations).from.target == (:matrix, 1, 2)
    @test only(captured.panels[1].layers).entries == ["1//3" "-2"]
end
