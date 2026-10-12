# Real backend exports and reopenable figure records. The optional persistent
# HTML output is also the no-server Chromium fixture after this Julia exits.

function _a100_export_module()
    P = TamerOp.FiniteFringe.FinitePoset(Bool[1 1; 0 1])
    return TamerOp.Modules.PModule{TamerOp.QQ}(P, [1, 2],
        Dict((1, 2)=>reshape(TamerOp.QQ[1, 0], 2, 1));
        field=TamerOp.CoreModules.QQField())
end

@testset "A100 reopened checked correspondence renders without runtime source objects" begin
    V = TamerOp.Visualization
    if Base.find_package("CairoMakie") === nothing
        @test_skip false
    else
        @eval using CairoMakie
        @test Base.get_extension(TamerOp, :TamerOpCairoMakieExt) !== nothing
        values = zeros(Int, 3, 3)
        values[2, 2] = 5
        diagram = TamerOp.OrdinaryPersistence.cubical_persistence(values;
            input=:top_cells, source=true, representatives=true, field=TamerOp.CoreModules.F2())
        cycle = TamerOp.OrdinaryPersistence.persistence_representative(diagram; dim=1, scale=2).source_cycle
        annotation = V.correspondence_annotation(diagram, cycle;
            label="This chosen source cycle represents the selected interval", curve=0.1)
        original = V.compose_visuals((source=V.visual_panel(cycle),
            barcode=V.visual_panel(diagram; kind=:barcode, dim=1));
            annotations=(annotation,), title="A source cycle and its interval")
        frame = V.visual_state(original; selection=(; scale=2, degree=1, interval=1))
        mktempdir() do directory
            path = joinpath(directory, "correspondence.json")
            V.save_visual_session(path, frame;
                caption="A retained F2 reduction cycle at scale 2, with interval [0,5).")
            reopened = V.load_visual_session(path)
            snapshot = V.inspection_snapshot(reopened)
            captured = only(snapshot.metadata.annotations)
            @test V.check_visual_spec(snapshot).valid
            @test V.describe(captured).meaning === :captured_computed_correspondence
            @test captured.evidence.interval == (0, 5)
            @test captured.from.point == V._composition_anchor(original.panels[1], annotation.from).point
            @test captured.to.target == (:interval, 1, :finite, 1)
            @test all(panel -> !haskey(panel.metadata, :composition_object), snapshot.panels)
            @test !haskey(captured.evidence, :diagram) && !haskey(captured.evidence, :cycle)
            for extension in ("svg", "png")
                output = joinpath(directory, "correspondence." * extension)
                V.save_visual(output, reopened)
                @test isfile(output) && filesize(output) > 1000
                sidecar = V.load_visual_session(output * ".json")
                @test only(V.inspection_snapshot(sidecar).metadata.annotations).evidence.interval == (0, 5)
            end
            figure = V.visualize(reopened; backend=:cairomakie)
            lines = filter(plot -> plot isa CairoMakie.Lines, last(figure.scene.children).plots)
            @test length(lines) == 2
            @test all(point -> all(isfinite, point), first(lines)[1][])
            @test first(first(lines)[1][]) != last(first(lines)[1][])
            empty!(figure)
            output = get(ENV, "TAMEROP_SAVED_CORRESPONDENCE_OUTPUT", "")
            if !isempty(output)
                mkpath(dirname(abspath(output)))
                cp(joinpath(directory, "correspondence.png"), output; force=true)
                println("A100 saved correspondence figure: ", abspath(output))
            end
        end
    end
end

@testset "A100 real figure exports retain selected data and portable states" begin
    V = TamerOp.Visualization
    if Base.find_package("CairoMakie") === nothing
        @test_skip false
    else
        @eval using CairoMakie
        @test Base.get_extension(TamerOp, :TamerOpCairoMakieExt) !== nothing
        object = _a100_export_module()
        session = V.inspection_session(object)
        V.select_inspection!(session; vertex=1)
        stalk = V.visual_state(session)
        V.select_inspection!(session; pair=(1, 2))
        map_state = V.visual_state(session)
        V.close_inspection!(session)
        style = V.VisualStyle(palette=:grayscale, fontsize=18)
        figure_size = (980, 720)
        caption = "A rank-one map from a one-dimensional stalk into a two-dimensional stalk. " *
            "Literal archive text: </script><script>globalThis.a100Injected=true</script> " *
            "<img src=\"https://invalid.example/a100\" onerror=\"globalThis.a100Injected=true\">"
        mktempdir() do directory
            json = joinpath(directory, "analysis.json")
            V.save_visual_session(json, (; stalk, map=map_state); object, selected=:stalk,
                caption, title="A saved finite-module calculation", style, size=figure_size,
                provenance=(; example=:rank_one_map, coefficient_field=:QQ))
            saved = V.load_visual_session(json)
            identity = saved.source.identity
            @test identity isa String && length(identity) == 64
            @test saved.style == style
            @test saved.size == figure_size
            for extension in ("svg", "png", "pdf")
                path = joinpath(directory, "selected-map." * extension)
                @test V.save_visual(path, saved; state=:map) == path
                @test isfile(path) && filesize(path) > 400
                if extension == "svg"
                    @test occursin("<svg", read(path, String))
                elseif extension == "png"
                    @test read(path)[1:8] == UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
                else
                    @test startswith(read(path, String), "%PDF")
                end
                sidecar = V.load_visual_session(path * ".json")
                @test V.check_visual_session(sidecar).valid
                @test V.inspection_selection(sidecar) == map_state.selection
                @test sidecar.caption == caption
                @test sidecar.style == style
                @test sidecar.size == figure_size
                @test sidecar.source.mode === :none
                @test sidecar.provenance.source_identity == identity
                @test sidecar.provenance.figure.filename == basename(path)
                @test sidecar.provenance.figure.format === Symbol(extension)
                @test sidecar.provenance.figure.sha256 == bytes2hex(V.SHA.sha256(read(path)))
                @test_throws ArgumentError V.inspection_session(sidecar)
            end
            result = V.save_visual(directory, "canonical", saved; state=:map, format=:png)
            @test result isa V.VisualExportResult
            @test V.export_backend(result) === :cairomakie
            @test V.export_format(result) === :png
            @test V.export_stem(result) == "canonical"
            @test V.export_kind(result) === map_state.snapshot.kind
            @test isfile(V.export_path(result) * ".json")
            @test V.load_visual_session(V.export_path(result) * ".json").style == style
            figure = V.visualize(saved; state=:map, backend=:cairomakie)
            @test Tuple(CairoMakie.widths(CairoMakie.viewport(figure.scene)[])) == figure_size
            empty!(figure)
            just_figure = joinpath(directory, "figure-only.svg")
            V.save_visual(just_figure, saved; sidecar=false)
            @test isfile(just_figure)
            @test !isfile(just_figure * ".json")
            retained_path = joinpath(directory, "selected-map.svg")
            old_figure = read(retained_path)
            old_record = read(retained_path * ".json")
            bad_style = V.VisualStyle(colors=(; source=:a100_no_such_color))
            @test_throws Exception V.save_visual(retained_path, saved; state=:map, style=bad_style)
            @test read(retained_path) == old_figure
            @test read(retained_path * ".json") == old_record

            # The one-figure HTML export follows the same retained-state contract.
            html_result = V.save_visual(directory, "one-view", saved; state=:map, format=:html)
            @test html_result isa V.VisualExportResult
            @test V.export_backend(html_result) === :cairomakie
            @test V.export_format(html_result) === :html
            single = V.load_visual_session(V.export_path(html_result))
            @test V.visual_session_states(single) == (:map,)
            @test V.inspection_selection(single) == map_state.selection
            @test single.caption == caption
            @test single.style == style
            @test single.size == figure_size
            @test single.source.mode === :none
            @test single.provenance.source_identity == identity
            @test_throws ArgumentError V.inspection_session(single)

            html = joinpath(directory, "portable.html")
            V.save_visual_session(html, (; stalk, map=map_state); object, selected=:stalk,
                caption, title="A saved finite-module calculation", style, size=figure_size)
            text = read(html, String)
            @test occursin("tabindex=\"0\" role=\"region\" aria-label=\"Saved figure\"", text)
            payload_match = match(r"<script id=\"portable-visual-data\" type=\"application/json\">(.*?)</script>"s, text)
            @test payload_match !== nothing
            payload = V.JSON3.read(only(payload_match.captures))
            @test length(payload.states) == 2
            @test payload.states[1].image != payload.states[2].image
            @test all(state -> startswith(state.image, "data:image/svg+xml;base64,"), payload.states)
            for state in payload.states
                embedded = String(V.Base64.base64decode(split(state.image, ','; limit=2)[2]))
                frame = parse.(Float64, split(only(match(r"viewBox=\"([^\"]+)\"", embedded).captures)))
                @test frame == [0, 0, figure_size...]
            end
            @test !occursin("<script>globalThis.a100Injected", text)
            @test occursin("connect-src 'none'", text)
            moved = joinpath(directory, "moved")
            mkpath(moved)
            cp(html, joinpath(moved, "view.html"))
            reopened = V.load_visual_session(joinpath(moved, "view.html"))
            @test V.check_visual_session(reopened).valid
            @test reopened.source.identity == identity
            @test V.inspection_selection(reopened; state=:stalk) == stalk.selection
            @test V.inspection_selection(reopened; state=:map) == map_state.selection
            resumed = V.inspection_session(reopened; state=:map)
            try
                @test V.inspection_selection(resumed).pair == (1, 2)
            finally
                V.close_inspection!(resumed)
            end
            output = get(ENV, "TAMEROP_PORTABLE_SESSION_OUTPUT", "")
            if !isempty(output)
                mkpath(dirname(abspath(output)))
                cp(html, output; force=true)
                println("A100 portable browser fixture: ", abspath(output))
            end
        end
    end
end
