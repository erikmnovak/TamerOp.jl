@testset "A44 image comparison ranges and full cells" begin
    V, D = TamerOp.Visualization, TamerOp.DataTypes
    values = [0 5 2; 4 1 3]
    spec = V.visual_spec(D.ImageNd(values); kind=:image, view_dims=(2,1),
        colorrange=(0,5), title="Appearance", colorbar_label="grade")
    @test V.visual_layers(spec)[1].values == Float64.(values)
    @test V.visual_axes(spec).xlimits == (0.5,3.5)
    @test V.visual_axes(spec).ylimits == (0.5,2.5)
    @test V.visual_axes(spec).aspect == :equal
    @test spec.title == "Appearance"
    @test V.visual_layers(spec)[1].colorbar_label == "grade"
    @test V.visual_metadata(spec).colorrange == (0.0,5.0)
    for limits in ((0,0), (1,0), (0,Inf), (NaN,1), (false,true), (0,1,2), "auto")
        @test_throws ArgumentError V.visual_spec(D.ImageNd(values); kind=:image, colorrange=limits)
    end
    @test_throws ArgumentError V.visual_spec(D.ImageNd(values); kind=:image, title=1)
    volume = D.ImageNd(cat(zeros(2,3), ones(2,3); dims=3))
    channels = V.visual_spec(volume; kind=:channels, colorrange=(0,1))
    @test all(p.metadata.colorrange == (0.0,1.0) for p in V.visual_panels(channels))
    viewer = V.visual_spec(volume; kind=:slice_viewer, colorrange=(0,1), colorbar_label="present")
    @test viewer.metadata.colorrange == (0.0,1.0)
    @test viewer.metadata.colorbar_label == "present"
    if Base.find_package("CairoMakie") === nothing
        @test_skip false
    else
        @eval import CairoMakie
        M = CairoMakie.Makie
        for value in (0,1)
            figure = V.visualize(D.ImageNd(fill(value,3,3)); kind=:image,
                colorrange=(0,1), backend=:cairomakie)
            M.update_state_before_display!(figure)
            axis = only(item for item in figure.content if item isa M.Axis)
            heat = only(p for p in axis.scene.plots if p isa M.Heatmap)
            @test Tuple(heat.colorrange[]) == (0.0,1.0)
            @test all(==(value), heat[3][])
        end
    end
end

@testset "A45 readable ring interval coordinates" begin
    V = TamerOp.Visualization
    diagram = TamerOp.cubical_persistence([0 0 0; 0 5 0; 0 0 0])
    for kind in (:barcode, :persistence_diagram)
        spec = V.visual_spec(diagram; kind, dim=1, window=(-1,6))
        @test spec.axes.xticks[1] == [-1,0,5,6]
        @test spec.axes.xticks[2] == ["-1","0","5","6"]
        @test spec.metadata.records[1].birth == 0
        @test spec.metadata.records[1].death == 5
    end
    essential = V.visual_spec(diagram; kind=:barcode, dim=0, window=(-1,6))
    @test 0 in essential.axes.xticks[1]
    @test last(essential.axes.xticks[2]) == "+Inf"
    crowded = V.visual_spec(Dict((i//100, (i+1)//100)=>1 for i in 1:20); kind=:barcode, window=(0,1))
    @test crowded.axes.xticks[1] == [0,1]
end

@testset "A45 informative defaults without presentation options" begin
    V, D = TamerOp.Visualization, TamerOp.DataTypes
    values = [0 0 0; 0 5 0; 0 0 0]
    diagram = TamerOp.cubical_persistence(values)
    bars = V.visual_spec(diagram; kind=:barcode, dim=1)
    points = V.visual_spec(diagram; kind=:persistence_diagram, dim=1)
    @test bars.metadata.interval_segments == ((0.0,1.0,5.0,1.0),)
    @test points.metadata.interval_points == ((0.0,5.0),)
    @test bars.axes.xticks == ([0.0,5.0],["0","5"])
    @test bars.axes.yticks == ([1.0],["#1"])
    @test bars.axes.xlimits[1] < 0 < 5 < bars.axes.xlimits[2]
    @test occursin("1 interval", bars.subtitle)
    @test occursin("Filled/open", bars.subtitle)
    @test !occursin("Filled/open", points.subtitle)
    @test occursin("Brackets", points.subtitle)
    @test all(!occursin(word,bars.subtitle) for word in ("censored","Inf","Float64","max_intervals"))
    @test bars.metadata.figure_size[2] < points.metadata.figure_size[2]
    essential = V.visual_spec(diagram; kind=:barcode, dim=0)
    @test occursin("continues indefinitely",essential.subtitle)
    @test last(essential.axes.xticks[2]) == "+Inf"
    @test only(essential.metadata.records).death == Inf
    clipped = V.visual_spec(Dict((0,5)=>2,(6,8)=>1,(9,10)=>1);
        kind=:barcode,window=(1,7),max_intervals=1,interval=3)
    for word in ("4 intervals", "3 groups", "outside window", "hidden by max_intervals",
                 "Selected #3", "finite endpoint beyond the view")
        @test occursin(word,clipped.subtitle)
    end
    @test clipped.axes.yticks[2] == ["#1 x2"]
    censored = first(V._interval_panels(V._group_interval_records([
        V._interval_record(0,1,1;right_status=:censored)])))
    @test occursin("continuation is unknown",censored.subtitle)
    @test !occursin("continues indefinitely",censored.subtitle)
    empty = V.visual_spec(Tuple{Int,Int}[];kind=:barcode)
    @test empty.subtitle == "0 intervals"
    @test any(layer -> layer isa V.TextLayer && "No nonzero intervals" in layer.labels,empty.layers)
    image = V.visual_spec(D.ImageNd(values))
    @test image.axes.xticks == ([1.0,2.0,3.0],["1","2","3"])
    @test image.axes.xlabel == "Column index" && image.axes.ylabel == "Row index"
    @test isempty(image.subtitle)
    @test image.layers[1].values == Float64.(values)
    if Base.find_package("CairoMakie") === nothing
        @test_skip false
    else
        @eval import CairoMakie
        M = CairoMakie.Makie
        # Exercise the public short calls, then verify the actual scene data.
        fig = TamerOp.visualize(diagram;kind=:barcode,dim=1)
        M.update_state_before_display!(fig)
        ax = only(item for item in fig.content if item isa M.Axis)
        @test Tuple(M.widths(M.viewport(fig.scene)[])) == bars.metadata.figure_size
        @test !ax.topspinevisible[] && !ax.rightspinevisible[]
        @test !ax.xgridvisible[] && !ax.ygridvisible[]
        @test ax.xticks[] == bars.axes.xticks
        @test M.widths(M.viewport(ax.scene)[])[1] > 0.75 * bars.metadata.figure_size[1]
        @test M.widths(M.viewport(ax.scene)[])[2] > 0.5 * bars.metadata.figure_size[2]
        custom = TamerOp.visualize(diagram;kind=:barcode,dim=1,
            style=TamerOp.VisualStyle(fontsize=18),size=(760,360))
        custom_axis = only(item for item in custom.content if item isa M.Axis)
        @test custom_axis.xlabelsize[] == 18
        again = TamerOp.visualize(diagram;kind=:barcode,dim=1)
        @test only(item for item in again.content if item isa M.Axis).xlabelsize[] == 16
    end
end

@testset "A16 native visualization activation and honest failures" begin
    VIZ = TamerOp.Visualization
    spec = VIZ.VisualizationSpec(:activation_test;
        layers=VIZ.AbstractVisualizationLayer[VIZ.PointLayer([(0.0, 1.0)], :black, 1.0, 5.0)])
    activated_before = Set(keys(VIZ._VISUAL_RENDERERS))
    for backend in (:cairomakie, :wglmakie)
        if !VIZ._visual_backend_available(backend)
            err = try
                VIZ.render(spec; backend=backend)
                nothing
            catch caught
                caught
            end
            @test err isa ArgumentError
            @test occursin(backend === :cairomakie ? "using CairoMakie" : "using WGLMakie", sprint(showerror, err))
        end
    end
    @test Set(keys(VIZ._VISUAL_RENDERERS)) == activated_before
    @test_throws ArgumentError VIZ.render(spec; backend=:not_a_backend)
    mktempdir() do dir
        @test_throws ArgumentError VIZ.save_visual(joinpath(dir, "wrong.html"), spec; backend=:cairomakie)
        @test_throws ArgumentError VIZ.save_visual(joinpath(dir, "wrong.html"), spec; backend=:not_a_backend)
        @test_throws ArgumentError VIZ.save_visual(joinpath(dir, "wrong.png"), spec; backend=:wglmakie)
        @test isempty(readdir(dir))
        if !VIZ._visual_save_available(:wglmakie)
            @test_throws ArgumentError VIZ.save_visual(joinpath(dir, "missing.html"), spec)
            @test !isfile(joinpath(dir, "missing.html"))
        end
        if isempty(activated_before)
            @test_throws ArgumentError VIZ.render(spec)
            @test_throws ArgumentError VIZ.save_visual(dir, "missing", spec)
            @test isempty(readdir(dir))
        end
        # A real backend error must propagate; it must never become a summary file.
        VIZ._register_visual_backend!(:a16_failing;
            render=(spec; kwargs...) -> error("A16 renderer failure"),
            save=(path, spec; kwargs...) -> error("A16 saver failure"))
        try
            @test_throws ErrorException VIZ.render(spec; backend=:a16_failing)
            @test_throws ErrorException VIZ.save_visual(joinpath(dir, "failure.png"), spec; backend=:a16_failing)
            @test !isfile(joinpath(dir, "failure.png"))
        finally
            delete!(VIZ._VISUAL_RENDERERS, :a16_failing)
            delete!(VIZ._VISUAL_SAVERS, :a16_failing)
        end
    end
end

function _a40_closed_square_encoding(field=CM.QQField(); repeated::Bool=false)
    K = CM.coeff_type(field)
    opts = TOA.EncodingOptions(; backend=:pl_backend, poset_kind=:signature, field)
    second_lo = repeated ? [0.0,0.0] : [1.0,1.0]
    second_hi = repeated ? [2.0,2.0] : [3.0,3.0]
    return TamerOp.encode([TOA.BoxUpset([0.0,0.0]), TOA.BoxUpset(second_lo)],
        [TOA.BoxDownset([2.0,2.0]), TOA.BoxDownset(second_hi)],
        K[1 0;0 1], opts)
end

@testset "A40 A41 exact closed-square slice intervals" begin
    V = TamerOp.Visualization
    enc = _a40_closed_square_encoding()
    s = V.inspection_session(enc; box=([-1,-1],[4,4]))
    try
        # These intervals come directly from intersecting each closed square
        # with x(t)=basepoint+t*direction. They do not use a computed chain.
        cases = (
            ((0//1,0//1),(1//1,1//1), ((0//1,2//1),(1//1,3//1))),
            ((0//1,1//1),(1//1,1//1), ((0//1,1//1),(1//1,2//1))),
            ((0//1,2//1),(1//1,1//1), ((0//1,0//1),(1//1,1//1))),
            ((0//1,3//1),(1//1,1//1), ()),
            ((0//1,0//1),(1//1,2//1), ((0//1,1//1),(1//1,3//2))),
            ((1//3,-1//5),(2//1,3//1), ((1//15,11//15),(2//5,16//15))),
            ((0//1,3//2),(1//1,0//1), ((0//1,2//1),(1//1,3//1))),
        )
        for (basepoint, direction, expected) in cases
            V.select_inspection!(s; slice=(;basepoint,direction))
            snap = V.inspection_snapshot(s)
            result = snap.metadata.slice_result
            @test result.endpoint_semantics == :decorated_finite_window
            @test result.essential_status == :not_inferred
            @test result.exact_geometry
            @test result.line.basepoint == basepoint
            @test result.line.direction == direction
            @test Set((r.birth,r.death) for r in result.intervals) == Set(expected)
            @test all(r -> r.left_closed && r.right_closed, result.intervals)
            @test all(r -> r.multiplicity == 1, result.intervals)
            @test all(r -> !r.left_clipped && !r.right_clipped, result.intervals)
            @test all(r -> r.singleton == (r.birth == r.death), result.intervals)
            @test all(r -> r.birth isa Rational && r.death isa Rational, result.intervals)
            @test V.check_visual_spec(snap).valid
            @test V.check_inspection_session(s).valid
        end

        # Closed deaths and simultaneous births are visible at the event itself.
        V.select_inspection!(s; slice=(basepoint=(0,0),direction=(1,1)))
        result = V.inspection_snapshot(s).metadata.slice_result
        samples = (-1//2,0//1,1//2,1//1,3//2,2//1,5//2,3//1,7//2)
        dimensions = (0,1,1,2,2,2,1,1,0)
        for (t, expected) in zip(samples,dimensions)
            @test sum(r.multiplicity for r in result.intervals
                if r.birth <= t <= r.death; init=0) == expected
            stalk = V.visual_spec(enc; kind=:module_inspector,point=(t,t))
            @test stalk.metadata.inspection.dimension == expected
        end
        # The same dimension profile is insufficient: adjacent maps are nonzero
        # while their composite vanishes, as the two interval summands require.
        for (source,target,expected_rank) in ((1//2,3//2,1),(3//2,5//2,1),(1//2,5//2,0))
            map = V.visual_spec(enc; kind=:module_inspector,
                parameter_pair=((source,source),(target,target)))
            @test map.metadata.inspection.defined
            @test map.metadata.inspection.rank == expected_rank
        end

        # A tangent encounters two isolated, nonzero stalks; neither may be
        # deleted as an empty half-open interval or merged into a long bar.
        V.select_inspection!(s; slice=(basepoint=(0,2),direction=(1,1)))
        for (t,expected) in ((0//1,1),(1//2,0),(1//1,1))
            stalk = V.visual_spec(enc; kind=:module_inspector,point=(t,2+t))
            @test stalk.metadata.inspection.dimension == expected
        end
        @test length(V.inspection_snapshot(s).metadata.slice_result.intervals) == 2
    finally
        V.close_inspection!(s)
    end
    square = TamerOp.encode([TOA.BoxUpset([0.0,0.0])], [TOA.BoxDownset([2.0,2.0])],
        reshape(QQ[1],1,1), TOA.EncodingOptions(;backend=:pl_backend,poset_kind=:signature,field=CM.QQField()))
    single = V.inspection_session(square;box=([-1,-1],[3,3]),
        slice=(basepoint=(0,0),direction=(1,1)))
    try
        r = only(V.inspection_snapshot(single).metadata.slice_result.intervals)
        @test (r.birth,r.death,r.multiplicity,r.left_closed,r.right_closed) == (0,2,1,true,true)
        V.select_inspection!(single;slice=(basepoint=(0,2),direction=(1,1)))
        r = only(V.inspection_snapshot(single).metadata.slice_result.intervals)
        @test (r.birth,r.death,r.singleton,r.multiplicity) == (0,0,true,1)
    finally
        V.close_inspection!(single)
    end
end

@testset "A40 A41 finite windows and interval multiplicity" begin
    V = TamerOp.Visualization
    enc = _a40_closed_square_encoding()
    line = (basepoint=(0,0),direction=(1,1))
    clipped = V.inspection_session(enc; box=([1//2,1//2],[5//2,5//2]),slice=line)
    interior = V.inspection_session(enc; box=([3//2,3//2],[7//4,7//4]),slice=line)
    repeated = V.inspection_session(_a40_closed_square_encoding(;repeated=true);
        box=([-1,-1],[4,4]),slice=line)
    corner = V.inspection_session(enc;box=([0,0],[2,2]),
        slice=(basepoint=(0,2),direction=(1,1)))
    try
        result = V.inspection_snapshot(clipped).metadata.slice_result
        @test result.window == (1//2,5//2)
        @test Set((r.birth,r.death,r.multiplicity,r.left_clipped,r.right_clipped)
            for r in result.intervals) == Set(((1//2,2//1,1,true,false),(1//1,5//2,1,false,true)))
        @test all(r -> r.left_closed && r.right_closed, result.intervals)
        @test result.essential_status == :not_inferred
        @test all(r -> isfinite(r.birth) && isfinite(r.death), result.intervals)

        # Both summands continue beyond this viewport. The displayed restricted
        # module is two copies of one finite censored interval, never an
        # essential bar and never an extrapolated death from sample spacing.
        middle = V.inspection_snapshot(interior).metadata.slice_result
        r = only(middle.intervals)
        @test (r.birth,r.death,r.multiplicity) == (3//2,7//4,2)
        @test r.left_clipped && r.right_clipped && r.left_closed && r.right_closed
        @test middle.essential_status == :not_inferred
        repeated_result = V.inspection_snapshot(repeated).metadata.slice_result
        r = only(repeated_result.intervals)
        @test (r.birth,r.death,r.multiplicity) == (0//1,2//1,2)
        @test !r.left_clipped && !r.right_clipped && !r.singleton
        corner_result = V.inspection_snapshot(corner).metadata.slice_result
        @test corner_result.window == (0,0)
        r = only(corner_result.intervals)
        @test (r.birth,r.death,r.multiplicity,r.singleton) == (0,0,1,true)
        @test r.left_closed && r.right_closed && r.left_clipped && r.right_clipped

        # A supported axis direction that misses the viewport gives an empty
        # restriction, without pretending that it observed the ambient module.
        V.select_inspection!(repeated; slice=(basepoint=(10,0),direction=(0,1)))
        missing = V.inspection_snapshot(repeated).metadata.slice_result
        @test missing.window === nothing
        @test isempty(missing.intervals)
        @test V.check_visual_spec(V.inspection_snapshot(repeated)).valid
    finally
        foreach(V.close_inspection!, (clipped,interior,repeated,corner))
    end
end

@testset "A40 A41 coincident diagram points retain distinct decorated groups" begin
    V = TamerOp.Visualization
    # These distinct interval summands have the same diagram location, but
    # their endpoint inclusion and multiplicity must remain separately readable.
    closed = (id=1,birth=0//1,death=1//1,left_closed=true,right_closed=true,
        multiplicity=1,left_clipped=false,right_clipped=false,singleton=false)
    half_open = merge(closed,(id=2,right_closed=false,multiplicity=2))
    records = (closed,half_open)
    result = (line=(basepoint=(0//1,0//1),direction=(1//1,1//1)),
        window=(-1//1,2//1),intervals=records,
        endpoint_semantics=:decorated_finite_window,essential_status=:not_inferred)
    for selected in (nothing,1,2)
        barcode, diagram = V._inspection_slice_panels(result,selected)
        @test barcode.metadata.records === diagram.metadata.records === records
        @test diagram.metadata.interval_ids == (1,2)
        @test diagram.metadata.selected_interval === selected
        @test diagram.metadata.total_multiplicity == 3
        @test diagram.metadata.interval_points == ((0.0,1.0),(0.0,1.0))
        @test barcode.metadata.interval_segments == ((0.0,1.0,1.0,1.0),(0.0,2.0,1.0,2.0))
        annotation = only(layer for layer in diagram.layers if layer isa V.TextLayer)
        @test annotation.positions == [(0.0,1.0)]
        @test Set(split(only(annotation.labels),'\n')) == Set(("#1 []","#2 [) x2"))
        @test V.check_visual_spec(barcode).valid && V.check_visual_spec(diagram).valid
    end
end

@testset "A40 A41 slice selection transactions and linked identity" begin
    V = TamerOp.Visualization
    enc = _a40_closed_square_encoding()
    line = (basepoint=(0,0),direction=(1,1))
    s = V.inspection_session(enc; box=([-1,-1],[4,4]),cache_limit=2,slice=line)
    try
        @test V.inspection_summary(s).slice_available
        V.select_inspection!(s; point=(1//2,1//2))
        V.select_inspection!(s; interval=1)
        selected = V.inspection_selection(s)
        @test selected.interval == 1
        @test selected.query_points == ((1//2,1//2),)
        snap = V.inspection_snapshot(s)
        result = snap.metadata.slice_result
        panels = [p for p in snap.panels if p.kind in (:slice_barcode,:slice_diagram)]
        @test Set(p.kind for p in panels) == Set((:slice_barcode,:slice_diagram))
        @test all(p -> p.metadata.records === result.intervals, panels)
        @test all(p -> p.metadata.interval_ids == Tuple(r.id for r in result.intervals), panels)
        @test all(p -> p.metadata.selected_interval == 1, panels)
        @test snap.metadata.slice_view.records === result.intervals
        V.select_inspection!(s; view=:presentation,upset=2,downset=1)
        @test V.inspection_selection(s).interval == 1
        @test V.inspection_selection(s).query_points == selected.query_points
        @test V.inspection_selection(s).slice == selected.slice

        # A changed line clears a bar identity, while retaining the separate
        # original-parameter stalk selection shared by the module panels.
        translated = (basepoint=(0,1),direction=(1,1))
        V.select_inspection!(s; slice=translated)
        @test V.inspection_selection(s).interval === nothing
        @test V.inspection_selection(s).query_points == selected.query_points
        @test Set((r.birth,r.death) for r in V.inspection_snapshot(s).metadata.slice_result.intervals) ==
            Set(((0//1,1//1),(1//1,2//1)))
        hits = V.inspection_summary(s).slice_cache_hits
        retained = V.inspection_snapshot(s).metadata.slice_result
        V.select_inspection!(s; slice=translated)
        @test V.inspection_snapshot(s).metadata.slice_result === retained
        @test V.inspection_summary(s).slice_cache_hits == hits
        V.select_inspection!(s; slice=line)
        @test V.inspection_summary(s).slice_cache_hits > hits
        @test V.inspection_summary(s).slice_cache_entries <= 2

        for options in ((; slice=(basepoint=(0,0),direction=(0,0))),
                        (; slice=(basepoint=(0,0),direction=(-1,1))),
                        (; slice=(basepoint=(NaN,0),direction=(1,1))),
                        (; slice=(basepoint=(0,0),direction=(Inf,1))),
                        (; slice=(basepoint=(0,),direction=(1,1))),
                        (; slice=(basepoint=(0,0),direction=(true,1))),
                        (; slice=(basepoint=(0,0),direction=(1,1),unknown=true)),
                        (; interval=-1), (; interval=1000), (; interval=true))
            before = V.inspection_selection(s)
            picture = V.inspection_snapshot(s)
            @test !V.check_inspection_selection(s; options...).valid
            @test_throws ArgumentError V.select_inspection!(s; options...)
            @test V.inspection_selection(s) == before
            @test V.inspection_snapshot(s) === picture
        end
        V.select_inspection!(s; interval=1)
        V.select_inspection!(s; interval=0)
        @test V.inspection_selection(s).interval === nothing
        V.select_inspection!(s; slice=false)
        @test V.inspection_selection(s).slice === nothing
        @test get(V.inspection_snapshot(s).metadata,:slice_result,nothing) === nothing
        @test !any(p -> p.kind in (:slice_barcode,:slice_diagram), V.inspection_snapshot(s).panels)
        @test V.inspection_selection(s).query_points == selected.query_points
        @test_throws ArgumentError V.select_inspection!(s; interval=1)
        V.select_inspection!(s; slice=line)
        final_snapshot = V.inspection_snapshot(s)
        V.close_inspection!(s)
        @test V.inspection_snapshot(s) === final_snapshot
        @test V.inspection_summary(s).slice_cache_entries == 0
        @test_throws ArgumentError V.select_inspection!(s; slice=translated)
    finally
        V.close_inspection!(s)
    end

    P = chain_poset(2)
    finite = MD.PModule{QQ}(P,[1,1],Dict((1,2)=>reshape(QQ[1],1,1));field=CM.QQField())
    @test_throws ArgumentError V.inspection_session(finite; slice=line)
    finite_session = V.inspection_session(finite)
    try
        @test !V.inspection_summary(finite_session).slice_available
        @test_throws ArgumentError V.select_inspection!(finite_session; slice=line)
    finally
        V.close_inspection!(finite_session)
    end
end

@testset "A40 A41 slice restrictions preserve coefficient fields" begin
    V = TamerOp.Visualization
    for field in FIELDS_FULL
        s = V.inspection_session(_a40_closed_square_encoding(field);
            box=([-1,-1],[4,4]),slice=(basepoint=(0,0),direction=(1,1)))
        try
            result = V.inspection_snapshot(s).metadata.slice_result
            if field isa CM.QQField
                @test result.coefficient_field == "QQ"
            elseif field isa CM.PrimeField
                @test result.coefficient_field == "F$(field.p)"
            else
                @test occursin("numerical rank",result.coefficient_field)
                @test occursin("atol=$(field.atol)",result.coefficient_field)
                @test occursin("rtol=$(field.rtol)",result.coefficient_field)
            end
            @test Set((r.birth,r.death,r.multiplicity,r.left_closed,r.right_closed)
                for r in result.intervals) == Set(((0//1,2//1,1,true,true),(1//1,3//1,1,true,true)))
            for (source,target,expected_rank) in ((1//2,3//2,1),(3//2,5//2,1),(1//2,5//2,0))
                V.select_inspection!(s; parameter_pair=((source,source),(target,target)))
                @test V.inspection_snapshot(s).metadata.inspection.rank == expected_rank
                @test V.inspection_snapshot(s).metadata.slice_result === result
            end
            @test V.check_inspection_session(s).valid
        finally
            V.close_inspection!(s)
        end
    end
end

@testset "A40 A41 open deaths exact collisions and computation limits" begin
    V = TamerOp.Visualization
    line = (basepoint=(0,0),direction=(1,1))
    # A one-vertex-supported product-grid module is nonzero on the half-open
    # middle square. Its right endpoint is open, unlike the closed box fixture.
    for (left,right) in ((0//1,2//1),(2//1,2//1+1//(big(2)^56)))
        P = FF.ProductOfChainsPoset((3,3))
        pi = EC.GridEncodingMap(P, ([-1//1,left,right],[-1//1,left,right]))
        dims = [0,0,0,0,1,0,0,0,0]
        maps = Dict(edge => zeros(QQ,dims[edge[2]],dims[edge[1]]) for edge in FF.cover_edges(P))
        module_on_grid = MD.PModule{QQ}(P,dims,maps;field=CM.QQField())
        enc = RES.EncodingResult(P,module_on_grid,EC.compile_encoding(P,pi))
        s = V.inspection_session(enc;box=([-1,-1],[3,3]),slice=line)
        try
            snap = V.inspection_snapshot(s)
            r = only(snap.metadata.slice_result.intervals)
            @test (r.birth,r.death,r.left_closed,r.right_closed,r.multiplicity) ==
                (left,right,true,false,1)
            @test !r.singleton && !r.left_clipped && !r.right_clipped
            @test V.visual_spec(enc;kind=:module_inspector,point=(right,right)).metadata.inspection.dimension == 0
            mid = (left+right)/2
            @test V.visual_spec(enc;kind=:module_inspector,point=(mid,mid)).metadata.inspection.dimension == 1
            diagram = only(p for p in snap.panels if p.kind == :slice_diagram)
            @test diagram.metadata.coordinate_collisions == (left == 0 ? () : (r.id,))
            @test only(diagram.metadata.interval_points) == (Float64(left),Float64(right))
            @test V.check_visual_spec(snap).valid
        finally
            V.close_inspection!(s)
        end
    end

    enc = _a40_closed_square_encoding()
    bounded = V.inspection_session(enc;box=([-1,-1],[4,4]),slice_limit=2)
    try
        before = V.inspection_snapshot(bounded)
        state = V.inspection_selection(bounded)
        @test_throws ArgumentError V.select_inspection!(bounded;slice=line)
        @test V.inspection_snapshot(bounded) === before
        @test V.inspection_selection(bounded) == state
        @test V.inspection_summary(bounded).slice_cache_entries == 0
    finally
        V.close_inspection!(bounded)
    end
    for bad_limit in (0,-1,true,1.5)
        @test_throws ArgumentError V.inspection_session(enc;slice_limit=bad_limit)
    end

    # Geometry outside a represented grid cannot silently become zero space.
    P = FF.ProductOfChainsPoset((2,2))
    constant_module = MD.PModule{QQ}(P,ones(Int,4),
        Dict(edge => reshape(QQ[1],1,1) for edge in FF.cover_edges(P));field=CM.QQField())
    pi = EC.GridEncodingMap(P,([0,2],[0,2]))
    grid_enc = RES.EncodingResult(P,constant_module,EC.compile_encoding(P,pi))
    unknown = V.inspection_session(grid_enc;box=([-1,-1],[3,3]))
    try
        before = V.inspection_snapshot(unknown)
        err = try
            V.select_inspection!(unknown;slice=line)
            nothing
        catch caught
            caught
        end
        @test err isa ArgumentError
        @test occursin("unrepresented",sprint(showerror,err))
        @test V.inspection_snapshot(unknown) === before
        @test V.inspection_summary(unknown).slice_cache_entries == 0
    finally
        V.close_inspection!(unknown)
    end
    reversed = EC.GridEncodingMap(P,([0,2],[0,2]);orientation=(-1,1))
    reversed_enc = RES.EncodingResult(P,constant_module,EC.compile_encoding(P,reversed))
    reversed_session = V.inspection_session(reversed_enc;box=([-1,-1],[3,3]))
    try
        @test !V.inspection_summary(reversed_session).slice_available
        @test_throws ArgumentError V.select_inspection!(reversed_session;slice=line)
    finally
        V.close_inspection!(reversed_session)
    end
end

@testset "A40 A41 general polyhedral slices retain exact boundary semantics" begin
    V = TamerOp.Visualization
    PL = TamerOp.PLPolyhedra
    AR = TamerOp.ExactReals.AlgebraicReal
    # This oblique band exercises the general polyhedral encoder, rather than
    # its axis-aligned box specialization. Along (t,t), 0 <= x+y <= 2 is [0,1].
    up = PL.PLUpset(PL.poly_union(PL.make_hpoly(QQ[-1 -1],QQ[0])))
    down = PL.PLDownset(PL.poly_union(PL.make_hpoly(QQ[1 1],QQ[2])))
    fringe = PL.PLFringe([up],[down],reshape(QQ[1],1,1))
    enc = TamerOp.encode(fringe,TOA.EncodingOptions(;backend=:pl,field=CM.QQField()))
    @test RES.result_summary(enc).backend == :pl
    s = V.inspection_session(enc;box=([-1,-1],[2,2]),
        slice=(basepoint=(0,0),direction=(1,1)))
    try
        r = only(V.inspection_snapshot(s).metadata.slice_result.intervals)
        @test (r.birth,r.death,r.left_closed,r.right_closed,r.multiplicity) == (0,1,true,true,1)
        @test !r.left_clipped && !r.right_clipped
        # x+y = 1/3+3t, so the exact interval is [-1/9,5/9].
        line = (basepoint=(AR(1)/3,AR(0)),direction=(AR(1),AR(2)))
        V.select_inspection!(s;slice=line)
        result = V.inspection_snapshot(s).metadata.slice_result
        @test result.line == (basepoint=(1//3,0//1),direction=(1//1,2//1))
        @test all(x -> x isa Rational,(result.line.basepoint...,result.line.direction...))
        r = only(result.intervals)
        @test (r.birth,r.death,r.left_closed,r.right_closed) == (-1//9,5//9,true,true)
        @test V.check_visual_spec(V.inspection_snapshot(s)).valid

        # General PL verified queries currently accept rational coordinates.
        # Unsupported irrational algebraic input must never be rounded silently.
        radius = sqrt(AR(2))
        for bad in ((basepoint=(radius,0),direction=(1,1)),
                    (basepoint=(0,0),direction=(1,radius)))
            state = V.inspection_selection(s)
            picture = V.inspection_snapshot(s)
            cache = V.inspection_summary(s).slice_cache_entries
            @test !V.check_inspection_selection(s;slice=bad).valid
            err = try
                V.select_inspection!(s;slice=bad)
                nothing
            catch caught
                caught
            end
            @test err isa ArgumentError
            @test occursin("rational",sprint(showerror,err))
            @test V.inspection_selection(s) == state
            @test V.inspection_snapshot(s) === picture
            @test V.inspection_summary(s).slice_cache_entries == cache
        end
    finally
        V.close_inspection!(s)
    end

    # A complete, disjoint three-region partition independently specifies
    # 0 <= x+y < 2. The strict facet belongs to the represented zero region.
    regions = [PL.HPoly(2,QQ[1 1],QQ[0],nothing,BitVector([true]),zero(QQ)),
               PL.HPoly(2,QQ[-1 -1;1 1],QQ[0,2],nothing,BitVector([false,true]),zero(QQ)),
               PL.HPoly(2,QQ[-1 -1],QQ[-2],nothing,BitVector([false]),zero(QQ))]
    pi = PL.PLEncodingMap(2,[BitVector() for _ in regions],
        [BitVector() for _ in regions],regions,[(-1,0),(1//2,1//2),(2,1)])
    P = chain_poset(3)
    M = MD.PModule{QQ}(P,[0,1,0],
        Dict((1,2)=>zeros(QQ,1,0),(2,3)=>zeros(QQ,0,1));field=CM.QQField())
    half_open = RES.EncodingResult(P,M,EC.compile_encoding(P,pi))
    strict = V.inspection_session(half_open;box=([-1,-1],[2,2]),
        slice=(basepoint=(0,0),direction=(1,1)))
    try
        r = only(V.inspection_snapshot(strict).metadata.slice_result.intervals)
        @test (r.birth,r.death,r.left_closed,r.right_closed,r.multiplicity) == (0,1,true,false,1)
        @test !r.left_clipped && !r.right_clipped
        @test EC.locate(pi,(0//1,0//1);mode=:verified) == 2
        @test EC.locate(pi,(1//1,1//1);mode=:verified) == 3
        @test V.check_visual_spec(V.inspection_snapshot(strict)).valid
    finally
        V.close_inspection!(strict)
    end
end

@testset "A40 A41 live slice controls share exact interval selection" begin
    if Base.find_package("WGLMakie") === nothing
        @test_skip false
    else
        @eval import WGLMakie
        V = TamerOp.Visualization
        E = Base.get_extension(TamerOp,:TamerOpWGLMakieExt)
        B, M = WGLMakie.Bonito, WGLMakie.Makie
        s = V.inspection_session(_a40_closed_square_encoding();box=([-1,-1],[4,4]))
        V.select_inspection!(s;point=(1//2,1//2))
        app = V.visualize(s;backend=:wglmakie,size=(1000,480))
        sibling = V.visualize(s;backend=:wglmakie,size=(900,440))
        ui, other = E._inspection_ui(app), E._inspection_ui(sibling)
        controls = ui.controls.slice
        client = B.Session(B.NoConnection();asset_server=B.NoServer())
        try
            @test controls !== nothing
            @test ui.slice.interval_disabled[]
            @test ui.slice.figure[] === nothing
            dom = B.session_dom(client,app;init=false)
            html = sprint(show,MIME"text/html"(),dom)
            for suffix in ("base-x","base-y","direction-x","direction-y","angle","offset","apply","disable","interval")
                @test occursin(ui.dom_id * "-slice-" * suffix,html)
            end
            @test !isempty(B.serialize_binary(client,B.fused_messages!(client)))
            controls.apply.value[] = true
            @test isempty(ui.error_text[])
            @test isempty(V.inspection_summary(s).listener_errors)
            state = V.inspection_selection(s)
            @test state.slice == (basepoint=(0,0),direction=(1,1))
            @test state.query_points == ((1//2,1//2),)
            @test !ui.slice.interval_disabled[]
            @test ui.slice.figure[] !== nothing && other.slice.figure[] !== nothing
            @test ui.slice.figure[] !== other.slice.figure[]
            @test ui.slice.line_points[] == other.slice.line_points[]
            M.update_state_before_display!(ui.slice.figure[])
            @test ui.slice.callbacks.pick_interval(:barcode,(0.5,1.0)) == 1
            @test ui.slice.callbacks.pick_interval(:diagram,(1.0,3.0)) == 2
            events = M.events(ui.slice.figure[].scene)
            function click_slice_data!(ax,point)
                pixel = M.project(ax.scene,M.Point2d(point)) + minimum(M.viewport(ax.scene)[])
                events.mouseposition[] = (Float64(pixel[1]),Float64(pixel[2]))
                @test M.is_mouseinside(ax.scene)
                events.mousebutton[] = M.MouseButtonEvent(M.Mouse.left,M.Mouse.press)
                events.mousebutton[] = M.MouseButtonEvent(M.Mouse.left,M.Mouse.release)
            end
            # Dispatch native chart events to verify hit testing and the wired
            # session callback, beyond exercising the picking helper directly.
            click_slice_data!(ui.slice.axes[].diagram,(1.0,3.0))
            @test V.inspection_selection(s).interval == 2
            @test ui.slice.selected_bar[][] == [M.Point2d(1,2),M.Point2d(3,2)]
            @test ui.slice.selected_point[][] == [M.Point2d(1,3)]
            @test ui.slice.selected_region[] == [M.Point2d(1,1),M.Point2d(3,3)]
            @test ui.slice.selected_bar[][] == other.slice.selected_bar[][]
            @test ui.slice.selected_point[][] == other.slice.selected_point[][]
            rebuilds = ui.slice.rebuild_count[]
            misses = V.inspection_summary(s).slice_cache_misses
            click_slice_data!(ui.slice.axes[].barcode,(0.5,1.0))
            @test V.inspection_selection(s).interval == 1
            @test ui.slice.selected_bar[][] == other.slice.selected_bar[][] == [M.Point2d(0,1),M.Point2d(2,1)]
            controls.interval.value[] = 2
            @test V.inspection_selection(s).interval == 2
            @test ui.slice.rebuild_count[] == rebuilds
            @test V.inspection_summary(s).slice_cache_misses == misses

            # Sliders only draft approximate decimal coordinates. Their event
            # callbacks do not run algebra or move the committed slice.
            before = V.inspection_snapshot(s)
            selection = V.inspection_selection(s)
            controls.angle.value[] = 30
            controls.offset.value[] = 20
            @test V.inspection_snapshot(s) === before
            @test V.inspection_selection(s) == selection
            @test ui.slice.rebuild_count[] == rebuilds
            @test occursin("Draft",ui.slice.draft_text[])
            controls.base_x.value[] = "abc"
            controls.apply.value[] = true
            @test !isempty(ui.error_text[])
            @test V.inspection_snapshot(s) === before
            @test V.inspection_selection(s) == selection

            for (control,text) in zip((controls.base_x,controls.base_y,controls.dir_x,controls.dir_y),
                                     ("0","2","1","1"))
                control.value[] = text
            end
            controls.apply.value[] = true
            @test isempty(ui.error_text[])
            @test V.inspection_selection(s).slice == (basepoint=(0,2),direction=(1,1))
            @test V.inspection_selection(s).interval === nothing
            @test V.inspection_selection(s).query_points == selection.query_points
            @test all(r -> r.singleton,V.inspection_snapshot(s).metadata.slice_result.intervals)
            @test ui.slice.callbacks.select_interval(1)
            @test ui.slice.selected_region_point[] == [M.Point2d(0,2)]
            @test ui.slice.selected_bar_point[][] == [M.Point2d(0,1)]
            @test ui.slice.selected_region_point[] == other.slice.selected_region_point[]
            ui.controls.reset.value[] = true
            @test V.inspection_selection(s).slice == (basepoint=(0,2),direction=(1,1))
            @test V.inspection_selection(s).interval === nothing
            @test isempty(V.inspection_selection(s).query_points)
            @test isempty(ui.slice.selected_region_point[])
            @test isempty(ui.slice.selected_bar_point[][])

            controls.disable.value[] = true
            @test V.inspection_selection(s).slice === nothing
            @test isempty(ui.slice.line_points[]) && isempty(ui.slice.selected_region_point[])
            @test ui.slice.figure[] === nothing && other.slice.figure[] === nothing
            @test isempty(ui.slice.chart_observers)
            @test ui.slice.interval_disabled[]
            controls.apply.value[] = true
            @test ui.slice.figure[] !== nothing
            V.close_inspection!(s)
            @test ui.closed[] && other.closed[]
            @test ui.slice.figure[] === nothing && ui.slice.axes[] === nothing
            @test isempty(ui.slice.chart_observers)
            @test ui.slice.interval_disabled[]
            @test V.inspection_summary(s).listener_count == 0
        finally
            close(client)
            V.close_inspection!(s)
        end
    end
end

using SparseArrays

@testset "A04 MPPI visualizations identify sampled tracks" begin
    VIZ = TamerOp.Visualization
    MI = TamerOp.MultiparameterImages
    omega = inv(sqrt(2.0))
    lines = [MI.MPPLineSpec([0.5,0.5], 0.0, [0.0,0.0], omega),
             MI.MPPLineSpec([0.5,0.5], 0.5, [-0.5,0.5], omega)]
    decomp = MI._mpp_decomposition_from_barcodes(lines, [(0.0,2.0),(3.0,5.0)],
        [1,1], ([0.0,0.0],[3.0,3.0]); q=0)
    @test MI.nsummands(decomp) == 2
    for layout in (:overlay, :summands)
        spec = VIZ.visual_spec(decomp; kind=:mpp_decomposition, layout=layout)
        @test spec.title == "Sampled MPPI tracks"
        @test VIZ.visual_metadata(spec).interpretation == :sampled_tracks
        @test VIZ.visual_metadata(spec).nsummands == 2
        @test VIZ.check_visual_spec(spec).valid
        if layout == :summands
            @test [panel.title for panel in VIZ.visual_panels(spec)] == ["Track 1", "Track 2"]
        end
    end
    img = MI.mpp_image(decomp; xgrid=[0.0,1.0], ygrid=[0.0,1.0], sigma=1, threads=false)
    spec = VIZ.visual_spec(img; kind=:mpp_image)
    @test VIZ.visual_metadata(spec).interpretation == :sampled_tracks
    @test occursin("sampled bottleneck tracks", spec.subtitle)
    @test isapprox(spec.layers[1].values, MI.image_values(img))
end

@testset "A03 fibered distance visualizations report sampled maxima" begin
    VIZ = TamerOp.Visualization
    F2D = TamerOp.Fibered2D
    CM = TamerOp.CoreModules
    FF = TamerOp.FiniteFringe
    MD = TamerOp.Modules
    EC = TamerOp.EncodingCore
    OPT = TamerOp.Options

    field = CM.QQField()
    K = CM.coeff_type(field)
    P = FF.FinitePoset(reshape(Bool[true], 1, 1))
    pi = EC.GridEncodingMap(P, ([0.0], [0.0]))
    M = MD.PModule{K}(P, [1], Dict{Tuple{Int,Int},Matrix{K}}(); field=field)
    Z = MD.zero_pmodule(P; field=field)
    opts = OPT.InvariantOptions(box=([0.0, 0.0], [1.0, 1.0]), threads=false)
    arr = F2D.fibered_arrangement_2d(pi, opts; precompute=:none, threads=false)
    cacheM = F2D.fibered_barcode_cache_2d(M, arr; precompute=:none, threads=false)
    cacheZ = F2D.fibered_barcode_cache_2d(Z, arr; precompute=:none, threads=false)
    fam = F2D.fibered_slice_family_2d(arr)

    # The unit square has exact distance 1/2 on the diagonal, while this
    # representative family attains 1/4. The plot must identify its statistic.
    @test F2D.matching_distance_exact_2d(cacheM, cacheZ; threads=false) == 0.5
    for owner in (fam, cacheM), kind in (:fibered_family_contributions, :fibered_distance_diagnostic)
        spec = VIZ.visual_spec(owner; kind=kind, caches=(cacheM, cacheZ))
        metadata = VIZ.visual_metadata(spec)
        @test isapprox(metadata.sampled_matching_distance, 0.25; atol=1e-12, rtol=0)
        @test !hasproperty(metadata, :matching_distance)
        cell_panel = first(VIZ.visual_panels(spec))
        @test VIZ.visual_metadata(cell_panel).sampled_matching_distance == metadata.sampled_matching_distance
        @test !hasproperty(VIZ.visual_metadata(cell_panel), :matching_distance)
        @test occursin("Sampled", cell_panel.title)
        if kind == :fibered_distance_diagnostic
            @test spec.title == "Sampled fibered distance diagnostic"
            @test occursin("sampled maximum", spec.subtitle)
            @test VIZ.visual_panels(spec)[2].title == "Maximizing representative slice"
            @test VIZ.visual_panels(spec)[3].title == "Top sampled contributions"
        end
    end
    @test_throws ArgumentError VIZ.visual_spec(fam; kind=:fibered_distance_diagnostic)
end

@testset "A79 rank heatmap coordinates and missing cells" begin
    VIZ = TamerOp.Visualization
    P = chain_poset(3)
    # k^2 --diag(1,0)--> k^2 --[0 1]--> k has two rank-one cover maps
    # but zero composite. Rows are source a; columns are target b.
    expected = [2.0 1.0 0.0; NaN 2.0 1.0; NaN NaN 1.0]
    rank_spec = nothing
    for field in FIELDS_FULL
        K = CM.coeff_type(field)
        M = MD.PModule{K}(P, [2, 2, 1],
            Dict((1, 2) => K[1 0; 0 0], (2, 3) => K[0 1]); field=field)
        for store_zeros in (false, true)
            result = Inv.rank_invariant(M, OPT.InvariantOptions(threads=false); store_zeros=store_zeros)
            @test haskey(result, (1, 3)) == store_zeros
            @test Inv.value_at(result, 1, 3) == 0
            rank_spec = VIZ.visual_spec(result; kind=:rank_heatmap)
            layer = only(rank_spec.layers)
            @test layer isa VIZ.HeatmapLayer
            @test layer.x == layer.y == [1.0, 2.0, 3.0]
            @test rank_spec.axes.xlabel == "target region b"
            @test rank_spec.axes.ylabel == "source region a"
            @test isequal(layer.values, expected)
            @test VIZ.check_visual_spec(rank_spec).valid
        end
    end

    # A non-square layer also distinguishes the backend's x/y layout from the
    # specification's row/column layout; square data alone can hide a transpose.
    rectangular = VIZ.VisualizationSpec(:coordinate_oracle;
        layers=VIZ.AbstractVisualizationLayer[
            VIZ.HeatmapLayer([-2.0, 0.0, 5.0], [1.0, 4.0],
                             [1.0 2.0 3.0; 4.0 5.0 6.0], :viridis, 1.0, "value")])
    @test VIZ.check_visual_spec(rectangular).valid
    if Base.find_package("CairoMakie") === nothing
        @test_skip false # Renderer coordinates require the optional CairoMakie backend.
    else
        @eval import CairoMakie
        for (spec, backend_values) in ((rank_spec, [2.0 NaN NaN; 1.0 2.0 NaN; 0.0 1.0 1.0]),
                                       (rectangular, [1.0 4.0; 2.0 5.0; 3.0 6.0]))
            fig = VIZ.render(spec; backend=:cairomakie)
            ax = only(block for block in fig.content if block isa CairoMakie.Axis)
            heatmap = only(plot for plot in ax.scene.plots if plot isa CairoMakie.Heatmap)
            # Makie indexes values by x first, y second. Check its actual plotted
            # data, including the invisible incomparable pairs of the rank plot.
            @test isequal(heatmap[3][], backend_values)
        end
    end
end

@testset "Visualization engine v1" begin
    TOA = TamerOp.Advanced
    VIZ = TamerOp.Visualization
    EC = TamerOp.EncodingCore
    ENC = TamerOp.Encoding
    RES = TamerOp.Results
    CO = TamerOp.ChangeOfPosets
    F2D = TamerOp.Fibered2D
    SMO = TamerOp.SignedMeasures
    MPI = TamerOp.MultiparameterImages
    FZ = TamerOp.FlangeZn
    CM = TamerOp.CoreModules
    FF = TamerOp.FiniteFringe
    PLB = TamerOp.PLBackend
    OPT = TamerOp.Options
    DT = TamerOp.DataTypes
    DI = TamerOp.DataIngestion
    SI = TamerOp.SliceInvariants

    Pgrid = FF.ProductOfChainsPoset((2, 2))
    grid_pi = EC.GridEncodingMap(Pgrid, ([0.0, 1.0], [0.0, 1.0]))
    compiled_grid = EC.compile_encoding(Pgrid, grid_pi)
    enc_result = RES.EncodingResult(Pgrid, nothing, compiled_grid)
    cdr_grid = RES.CohomologyDimsResult(Pgrid, [0, 1, 0, 1], compiled_grid; degree=1)
    Pbox, _, box_pi = PLB.encode_fringe_boxes(
        [PLB.BoxUpset([0.0, -10.0]), PLB.BoxUpset([1.0, -10.0])],
        PLB.BoxDownset[],
        TOA.EncodingOptions(),
    )
    compiled_box = EC.compile_encoding(Pbox, box_pi)
    r_left = EC.locate(box_pi, [0.5, 0.0])
    r_right = EC.locate(box_pi, [2.0, 0.0])
    Hbox = TOA.one_by_one_fringe(Pbox,
                                 FF.principal_upset(Pbox, r_left),
                                 FF.principal_downset(Pbox, r_right),
                                 1)
    Mbox = TOA.pmodule_from_fringe(Hbox)
    enc_box_result = RES.EncodingResult(Pbox, Mbox, compiled_box)
    cdr_box = RES.CohomologyDimsResult(Pbox, [0, 1, 2], compiled_box; degree=1)
    rank_inv = TamerOp.invariant(enc_box_result; which=:rank_invariant)
    hilbert_inv = TamerOp.invariant(enc_box_result; which=:restricted_hilbert)
    rank_raw = TamerOp.rank_invariant(Mbox)
    opts_box = OPT.InvariantOptions(box=([-1.0, -1.0], [2.0, 1.0]), strict=true)
    arr_box = F2D.fibered_arrangement_2d(box_pi, opts_box; normalize_dirs=:L1, include_axes=true, precompute=:cells, threads=false)
    Hbox_alt = TOA.one_by_one_fringe(Pbox,
                                     FF.principal_upset(Pbox, r_right),
                                     FF.principal_downset(Pbox, r_right),
                                     1)
    Nbox = TOA.pmodule_from_fringe(Hbox_alt)
    cache_box = F2D.fibered_barcode_cache_2d(Mbox, arr_box; precompute=:none, threads=false)
    cache_box_alt = F2D.fibered_barcode_cache_2d(Nbox, arr_box; precompute=:none, threads=false)
    fam_box = F2D.fibered_slice_family_2d(arr_box)

    line = MPI.MPPLineSpec([1.0, 1.0], 0.0, [0.0, 0.0], 0.5)
    line2 = MPI.MPPLineSpec([1.0, 0.5], 0.25, [0.0, 0.25], 0.4)
    line3 = MPI.MPPLineSpec([0.5, 1.0], -0.1, [0.1, 0.0], 0.4)
    decomp = MPI.MPPDecomposition(
        [line, line2, line3],
        [
            [([0.0, 0.0], [1.0, 1.0], 0.5)],
            [([0.0, 0.5], [1.0, 0.8], 0.4), ([0.2, 0.0], [1.0, 0.6], 0.7)],
            [([0.0, 0.2], [0.8, 1.0], 0.6)],
        ],
        [0.25, 0.75, 0.5],
        ([0.0, 0.0], [1.0, 1.0]),
    )
    img = MPI.MPPImage([0.0, 1.0], [0.0, 1.0], [1.0 2.0; 3.0 4.0], 0.25, decomp)
    L = MPI.MPLandscape(2,
                        [0.0, 0.5, 1.0],
                        reshape(Float64[0.1, 0.2, 0.3, 0.0, 0.1, 0.0,
                                        0.2, 0.1, 0.0, 0.3, 0.2, 0.1,
                                        0.0, 0.1, 0.2, 0.1, 0.0, 0.0,
                                        0.3, 0.1, 0.1, 0.2, 0.2, 0.2], 2, 2, 2, 3),
                        fill(0.25, 2, 2),
                        [[1.0, 0.0], [0.0, 1.0]],
                        [-0.5, 0.5])

    sb = SMO.RectSignedBarcode((collect(1:3), collect(1:3)),
                               [SMO.Rect{2}((1, 1), (2, 2)), SMO.Rect{2}((2, 1), (2, 2))],
                               [2, -1])
    pm = SMO.PointSignedMeasure((collect(1:3), collect(1:3)), [(1, 1), (3, 2)], [1, -2])
    smd = SMO.SignedMeasureDecomposition(rectangles=sb, euler_signed_measure=pm, mpp_image=img)

    slice_single = SI.SliceBarcodesResult([Dict((0.0, 1.0) => 1)], [1.0], [[1.0, 1.0]], [0.0])
    slice_bank = SI.SliceBarcodesResult(reshape([Dict((0.0, 1.0) => 1), Dict((0.5, 1.5) => 2)], 1, 2),
                                        reshape([0.5, 0.5], 1, 2), Any[], Any[])

    parr = F2D.projected_arrangement(Pgrid, [0.0, 1.0, 2.0, 3.0])
    parr_grid = F2D.projected_arrangement(compiled_grid; dirs=[(1.0, 0.0), (0.0, 1.0)], include_axes=true, threads=false)
    pdres = F2D.ProjectedDistancesResult([0.2, 0.4], [1, 2], [(1.0, 0.0), (0.0, 1.0)], :bottleneck)
    pbres = F2D.ProjectedBarcodesResult([Dict((0.0, 1.0) => 1), Dict((0.5, 1.5) => 1)], [1, 2], [(1.0, 0.0), (0.0, 1.0)])

    opts = OPT.InvariantOptions(box=([0.0, 0.0], [2.0, 2.0]))
    arr = F2D.fibered_arrangement_2d(grid_pi, opts; include_axes=true, precompute=:cells, threads=false)
    fam = F2D.fibered_slice_family_2d(arr)
    Hgrid = TOA.one_by_one_fringe(Pgrid,
                                  FF.principal_upset(Pgrid, 2),
                                  FF.principal_downset(Pgrid, 4),
                                  1)
    Mgrid = TOA.pmodule_from_fringe(Hgrid)
    cache_grid = F2D.fibered_barcode_cache_2d(Mgrid, arr; precompute=:none, threads=false)
    slice_res = F2D.fibered_slice(cache_grid, (1.0, 1.0), 0.0)

    face = FZ.Face(2, [false, false])
    qq = CM.QQField()
    FG = FZ.Flange(2,
                   [FZ.IndFlat(face, (0, 0); id=:U1)],
                   [FZ.IndInj(face, (1, 1); id=:D1)],
                   reshape([CM.coerce(qq, 1)], 1, 1);
                   field=qq)

    Qmap = FF.ProductOfChainsPoset((2, 2))
    Pmap = FF.ProductOfChainsPoset((2, 2))
    emap = ENC.EncodingMap(Qmap, Pmap, [1, 2, 3, 4])
    trans = RES.ModuleTranslationResult(:pushforward_left, nothing, emap)

    Pcommon = FF.ProductPoset(Pmap, Pmap)
    n1 = FF.nvertices(Pmap)
    pi_left = ENC.EncodingMap(Pcommon, Pmap, [((q - 1) % n1) + 1 for q in 1:FF.nvertices(Pcommon)])
    pi_right = ENC.EncodingMap(Pcommon, Pmap, [div(q - 1, n1) + 1 for q in 1:FF.nvertices(Pcommon)])
    cref = CO.CommonRefinementTranslationResult(Pcommon, (nothing, nothing), pi_left, pi_right)

    pc2 = DT.PointCloud([0.0 0.0; 1.0 0.5; 2.0 1.0; 3.0 1.5])
    pc3 = DT.PointCloud([0.0 0.0 0.0; 1.0 0.5 0.25; 2.0 1.0 0.5; 3.0 1.5 0.75])
    g3 = DT.GraphData(4, [(1, 2), (2, 3), (3, 4)];
                      coords=[0.0 0.0 0.0; 1.0 0.5 0.25; 2.0 1.0 0.5; 3.0 1.5 0.75],
                      weights=[0.2, 0.5, 0.9])
    epg = DT.EmbeddedPlanarGraph2D([[0.0, 0.0], [1.0, 0.75], [2.0, 0.0]],
                                   [(1, 2), (2, 3)];
                                   polylines=[[(0.0, 0.0), (1.0, 0.75)], [(1.0, 0.75), (2.0, 0.0)]],
                                   bbox=(0.0, 0.0, 2.0, 1.0))
    img2 = DT.ImageNd(reshape(collect(1.0:16.0), 4, 4))
    img3 = DT.ImageNd(rand(4, 4, 3))
    Bgc = SparseArrays.sparse(Int[1, 2], Int[1, 2], Int[1, 1], 2, 2)
    gc = DT.GradedComplex([[1, 2], [3, 4]], [Bgc],
                          [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (2.0, 1.0)])
    st = DT.SimplexTreeMulti([1, 2, 4, 7], [1, 1, 2, 1, 2, 3], [0, 1, 2],
                             [1, 2, 3, 4], [1, 2, 3, 4],
                             [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])
    est = DI.IngestionEstimate((n_cells_est=12,
                                cell_counts_by_dim=[3, 4, 5],
                                axis_sizes=[5, 6],
                                poset_size=30,
                                nnz_est=18,
                                dense_bytes_est=256,
                                warnings=String[]))
    plan = DI.IngestionPlan(pc2,
                            OPT.FiltrationSpec(kind=:rips),
                            OPT.FiltrationSpec(kind=:rips),
                            OPT.ConstructionOptions(output_stage=:graded_complex,
                                                    budget=(20, 12, 512)),
                            OPT.PipelineOptions(),
                            :graded_complex,
                            CM.QQField(),
                            nothing,
                            est,
                            :point_cloud_sparse,
                            :multi,
                            :first,
                            true)
    plan_no_pre = DI.IngestionPlan(pc2,
                                   OPT.FiltrationSpec(kind=:rips),
                                   OPT.FiltrationSpec(kind=:rips),
                                   OPT.ConstructionOptions(output_stage=:graded_complex,
                                                           budget=(nothing, nothing, nothing)),
                                   OPT.PipelineOptions(),
                                   :graded_complex,
                                   CM.QQField(),
                                   nothing,
                                   nothing,
                                   :point_cloud_sparse,
                                   :multi,
                                   :first,
                                   true)
    gcres = DI.GradedComplexBuildResult(gc, [[0.0, 1.0], [0.0, 1.0]], (:increasing, :increasing))
    codres = DI.point_codensity(pc2, DI.RipsCodensityFiltration(max_dim=1, knn=2, dtm_mass=0.5, nn_backend=:bruteforce))

    @test TamerOp.available_visuals(nothing) == ()
    @test TamerOp.available_visuals(grid_pi) == (:regions, :region_labels, :query_overlay)
    @test TamerOp.available_visuals(compiled_grid) == (:regions, :region_labels, :query_overlay)
    @test TamerOp.available_visuals(enc_result) == (:regions, :region_labels, :query_overlay, :hasse, :module_inspector)
    @test TamerOp.available_visuals(emap) == (:regions, :region_labels, :pushforward_overlay)
    @test TamerOp.available_visuals(cref) == (:common_refinement,)
    @test TamerOp.available_visuals(trans) == (:pushforward_overlay,)
    @test TamerOp.available_visuals(img) == (:mpp_image,)
    @test TamerOp.available_visuals(L) == (:mp_landscape, :landscape_slices)
    @test TamerOp.available_visuals(sb) == (:rectangles, :density_image)
    @test TamerOp.available_visuals(pm) == (:signed_atoms,)
    @test TamerOp.available_visuals(slice_single) == (:barcode, :persistence_diagram, :barcode_bank, :slice_family)
    @test TamerOp.available_visuals(arr) == (:fibered_arrangement, :fibered_query, :fibered_cell_highlight, :fibered_tie_break, :fibered_offset_intervals, :fibered_projected_comparison)
    @test TamerOp.available_visuals(fam) == (:fibered_family, :fibered_chain_cells, :fibered_family_contributions, :fibered_distance_diagnostic)
    @test TamerOp.available_visuals(slice_res) == (:fibered_slice, :fibered_slice_overlay, :barcode, :persistence_diagram)
    @test TamerOp.available_visuals(cache_grid) == (:fibered_arrangement, :fibered_query, :fibered_cell_highlight, :fibered_tie_break, :fibered_offset_intervals, :fibered_projected_comparison, :fibered_family, :fibered_chain_cells, :fibered_family_contributions, :fibered_distance_diagnostic, :fibered_query_barcode, :barcode, :persistence_diagram)
    @test isempty(TamerOp.available_visuals(parr))
    @test TamerOp.available_visuals(parr_grid) == (:projected_arrangement,)
    @test_throws ArgumentError TOA.visual_spec(parr; kind=:projected_arrangement)
    @test TamerOp.available_visuals(pdres) == (:projected_distances,)
    @test TamerOp.available_visuals(FG) == (:regions, :constant_subdivision)
    @test TamerOp.available_visuals(pc2) == (:points_2d, :point_density, :knn_graph, :radius_graph)
    @test TamerOp.available_visuals(pc3) == (:points_3d, :points_2d, :point_density, :knn_graph, :radius_graph)
    @test TamerOp.available_visuals(codres) == (:points_2d, :point_density, :knn_graph, :radius_graph, :codensity_radius_snapshots)
    @test TamerOp.available_visuals(g3) == (:graph_3d, :graph, :weighted_graph)
    @test TamerOp.available_visuals(epg) == (:embedded_planar_graph,)
    @test TamerOp.available_visuals(img2) == (:image,)
    @test TamerOp.available_visuals(img3) == (:image, :channels, :slice_viewer)
    @test TamerOp.available_visuals(est) == (:simplex_counts,)
    @test TamerOp.available_visuals(plan) == ()
    @test TamerOp.available_visuals(gcres) == ()
    @test TamerOp.available_visuals(gc) == ()
    @test TamerOp.available_visuals(st) == (:simplex_counts,)
    @test TamerOp.available_visuals(rank_raw) == (:rank_heatmap, :rank_rectangles)
    @test TamerOp.available_visuals(rank_inv) == (:rank_heatmap, :rank_rectangles, :rank_query_overlay)
    @test TamerOp.available_visuals(cdr_box) == (:cohomology_support_plane, :cohomology_support)
    @test TamerOp.available_visuals(hilbert_inv) == (:hilbert_heatmap, :hilbert_bars, :restricted_hilbert_curve)

    report_bad_kind = TOA.check_visual_request(grid_pi; kind=:fibered_arrangement, throw=false)
    @test !report_bad_kind.valid
    @test_throws ArgumentError TOA.check_visual_request(grid_pi; kind=:fibered_arrangement, throw=true)

    report_missing_query = TOA.check_visual_request(grid_pi; kind=:query_overlay, throw=false)
    @test !report_missing_query.valid
    @test_throws ArgumentError TOA.check_visual_request(grid_pi; kind=:query_overlay, throw=true)

    report_missing_landscape = TOA.check_visual_request(L; kind=:landscape_slices, throw=false)
    @test !report_missing_landscape.valid
    @test_throws ArgumentError TOA.check_visual_request(L; kind=:landscape_slices, throw=true)

    report_missing_tie = TOA.check_visual_request(arr; kind=:fibered_tie_break, throw=false)
    @test !report_missing_tie.valid
    @test_throws ArgumentError TOA.check_visual_request(arr; kind=:fibered_tie_break, throw=true)

    report_tie_ok = TOA.check_visual_request(arr_box; kind=:fibered_tie_break, dir=[1.0, 1.0], offset=0.0, throw=false)
    @test report_tie_ok.valid

    report_missing_slice_overlay = TOA.check_visual_request(slice_res; kind=:fibered_slice_overlay, throw=false)
    @test !report_missing_slice_overlay.valid
    @test_throws ArgumentError TOA.check_visual_request(slice_res; kind=:fibered_slice_overlay, throw=true)

    report_offset_ok = TOA.check_visual_request(arr_box; kind=:fibered_offset_intervals, dir=[1.0, 1.0], throw=false)
    @test report_offset_ok.valid

    report_cod_ok = TOA.check_visual_request(codres; kind=:codensity_radius_snapshots,
                                             radii=[0.1, 0.2], codensity_levels=:quantiles, throw=false)
    @test report_cod_ok.valid
    report_cod_bad = TOA.check_visual_request(codres; kind=:codensity_radius_snapshots,
                                              radii=[-0.1], codensity_levels=:bad, throw=false)
    @test !report_cod_bad.valid
    @test_throws ArgumentError TOA.check_visual_request(codres; kind=:codensity_radius_snapshots,
                                                        radii=[-0.1], codensity_levels=:bad, throw=true)

    report_missing_query_barcode = TOA.check_visual_request(cache_grid; kind=:fibered_query_barcode, throw=false)
    @test !report_missing_query_barcode.valid
    @test_throws ArgumentError TOA.check_visual_request(cache_grid; kind=:fibered_query_barcode, throw=true)

    report_family_missing = TOA.check_visual_request(fam_box; kind=:fibered_family_contributions, throw=false)
    @test !report_family_missing.valid
    @test_throws ArgumentError TOA.check_visual_request(fam_box; kind=:fibered_family_contributions, throw=true)

    report_family_ok = TOA.check_visual_request(fam_box; kind=:fibered_family_contributions,
                                                caches=(cache_box, cache_box_alt), throw=false)
    @test report_family_ok.valid

    report_projected_ok = TOA.check_visual_request(arr; kind=:fibered_projected_comparison,
                                                   projected=parr_grid, throw=false)
    @test report_projected_ok.valid

    report_removed_preflight = TOA.check_visual_request(plan; kind=:preflight_diagnostics, throw=false)
    @test !report_removed_preflight.valid
    @test_throws ArgumentError TOA.check_visual_request(plan; kind=:preflight_diagnostics, throw=true)

    report_removed_dashboard = TOA.check_visual_request(gcres; kind=:complex_dashboard, throw=false)
    @test !report_removed_dashboard.valid
    @test_throws ArgumentError TOA.check_visual_request(gcres; kind=:complex_dashboard, throw=true)

    report_bad_decomp_layout = TOA.check_visual_request(decomp; kind=:mpp_decomposition, layout=:bad, throw=false)
    @test !report_bad_decomp_layout.valid
    @test_throws ArgumentError TOA.check_visual_request(decomp; kind=:mpp_decomposition, layout=:bad, throw=true)

    report_missing_rank_pairs = TOA.check_visual_request(rank_inv; kind=:rank_query_overlay, throw=false)
    @test !report_missing_rank_pairs.valid
    @test_throws ArgumentError TOA.check_visual_request(rank_inv; kind=:rank_query_overlay, throw=true)

    spec_grid = TOA.visual_spec(grid_pi; kind=:query_overlay, points=[(0.25, 0.25), (0.75, 0.75)])
    @test TOA.visual_kind(spec_grid) == :query_overlay
    @test any(layer -> layer isa TOA.PolygonLayer, spec_grid.layers)
    @test spec_grid.metadata.region_ids == [0, 1, 2, 3, 4]
    @test length(spec_grid.metadata.query_readout) == 2
    @test all(q.region_id == EC.locate(grid_pi, collect(q.point)) for q in spec_grid.metadata.query_readout)
    @test TOA.check_visual_spec(spec_grid).valid

    spec_compiled = TOA.visual_spec(compiled_grid; kind=:regions)
    @test TOA.visual_kind(spec_compiled) == :regions

    spec_result = TOA.visual_spec(enc_result; kind=:region_labels)
    @test TOA.visual_kind(spec_result) == :region_labels

    spec_box = TOA.visual_spec(box_pi; kind=:region_labels)
    @test TOA.visual_kind(spec_box) == :region_labels
    @test all(isfinite, spec_box.axes.xlimits)
    @test all(isfinite, spec_box.axes.ylimits)
    @test TOA.visual_metadata(spec_box).figure_size == (860, 620)
    @test TOA.visual_metadata(spec_box).legend_position == :right
    spec_box_regions = TOA.visual_spec(box_pi; kind=:regions)
    @test spec_box_regions.legend.visible
    polygon_layers = [layer for layer in spec_box.layers if layer isa TOA.PolygonLayer]
    @test !isempty(polygon_layers)
    @test length(unique(layer.fill_color for layer in polygon_layers)) == 3
    @test sort(unique(c.region_id for c in spec_box.metadata.geometry.components)) == [1, 2, 3]
    label_layer = only(layer for layer in spec_box.layers if layer isa TOA.TextLayer)
    @test sort(unique(label_layer.labels)) == ["1", "2", "3"]
    for (p, label) in zip(label_layer.positions, label_layer.labels)
        @test EC.locate(box_pi, collect(p)) == parse(Int, label)
    end

    spec_rank_heat = TOA.visual_spec(rank_raw; kind=:rank_heatmap)
    @test spec_rank_heat.layers[1] isa TOA.HeatmapLayer
    @test size(spec_rank_heat.layers[1].values, 1) == FF.nvertices(Pbox)
    @test all(begin
                  if FF.leq(Pbox, a, b)
                      isapprox(spec_rank_heat.layers[1].values[a, b], float(TOA.value_at(rank_raw, a, b)); atol=1e-12)
                  else
                      isnan(spec_rank_heat.layers[1].values[a, b])
                  end
              end for a in 1:FF.nvertices(Pbox), b in 1:FF.nvertices(Pbox))
    @test TOA.visual_metadata(spec_rank_heat).figure_size == (720, 620)

    spec_rank_rect = TOA.visual_spec(rank_inv; kind=:rank_rectangles)
    @test any(layer -> layer isa TOA.RectLayer, spec_rank_rect.layers)
    @test spec_rank_rect.legend.visible
    @test TOA.visual_metadata(spec_rank_rect).legend_position == :right

    spec_rank_query = TOA.visual_spec(rank_inv; kind=:rank_query_overlay,
                                      pairs=[((0.5, -9.0), (2.0, -9.0))])
    @test TOA.visual_kind(spec_rank_query) == :rank_query_overlay
    @test any(layer -> layer isa TOA.SegmentLayer, spec_rank_query.layers)
    @test count(layer -> layer isa TOA.PointLayer, spec_rank_query.layers) >= 2
    rank_query_labels = [layer for layer in spec_rank_query.layers if layer isa TOA.TextLayer]
    @test any(lbl -> any(startswith(x, "r1 = ") for x in lbl.labels), rank_query_labels)

    spec_cdr_support_plane = TOA.visual_spec(cdr_box; kind=:cohomology_support_plane)
    @test TOA.visual_kind(spec_cdr_support_plane) == :cohomology_support_plane
    @test TOA.visual_metadata(spec_cdr_support_plane).degree == 1
    @test TOA.visual_metadata(spec_cdr_support_plane).support_count == 2
    @test TOA.visual_metadata(spec_cdr_support_plane).max_dim == 2
    @test spec_cdr_support_plane.legend.visible
    @test spec_cdr_support_plane.axes.aspect == :auto
    @test count(layer -> layer isa TOA.PolygonLayer, spec_cdr_support_plane.layers) == length(spec_cdr_support_plane.metadata.geometry.components)
    @test count(layer -> layer isa TOA.TextLayer, spec_cdr_support_plane.layers) == 0

    spec_cdr_support_plane_grid = TOA.visual_spec(cdr_grid; kind=:cohomology_support_plane)
    @test TOA.visual_kind(spec_cdr_support_plane_grid) == :cohomology_support_plane
    @test spec_cdr_support_plane_grid.axes.aspect == :auto
    @test spec_cdr_support_plane_grid.legend.visible
    @test any(layer -> layer isa TOA.PolygonLayer, spec_cdr_support_plane_grid.layers)
    @test spec_cdr_support_plane_grid.metadata.region_ids == [1, 2, 3, 4]
    @test spec_cdr_support_plane_grid.metadata.support_count == 2
    @test spec_cdr_support_plane_grid.metadata.max_dim == 1
    # The negative-parameter area is outside this finite grid, not dimension zero.
    @test 0 in spec_cdr_support_plane_grid.metadata.geometry.region_ids

    spec_cdr_support = TOA.visual_spec(cdr_box; kind=:cohomology_support)
    @test TOA.visual_kind(spec_cdr_support) == :cohomology_support
    @test TOA.visual_metadata(spec_cdr_support).degree == 1
    @test TOA.visual_metadata(spec_cdr_support).support_count == 2
    @test TOA.visual_metadata(spec_cdr_support).max_dim == 2
    @test spec_cdr_support.legend.visible
    @test count(layer -> layer isa TOA.PolygonLayer, spec_cdr_support.layers) == length(spec_cdr_support.metadata.geometry.components)
    @test any(layer -> layer isa TOA.SegmentLayer, spec_cdr_support.layers)
    cdr_labels = [layer for layer in spec_cdr_support.layers if layer isa TOA.TextLayer]
    @test length(cdr_labels) == 1
    @test sort(unique(only(cdr_labels).labels)) == ["1", "2"]

    spec_hilbert_heat = TOA.visual_spec(hilbert_inv; kind=:hilbert_heatmap)
    @test TOA.visual_kind(spec_hilbert_heat) == :hilbert_heatmap
    @test any(layer -> layer isa TOA.PolygonLayer, spec_hilbert_heat.layers)
    @test spec_hilbert_heat.legend.visible

    spec_hilbert_bars = TOA.visual_spec(hilbert_inv; kind=:hilbert_bars)
    @test spec_hilbert_bars.layers[1] isa TOA.RectLayer
    @test TOA.visual_metadata(spec_hilbert_bars).figure_size == (760, 460)

    spec_hilbert_curve = TOA.visual_spec(hilbert_inv; kind=:restricted_hilbert_curve)
    @test spec_hilbert_curve.layers[1] isa TOA.PolylineLayer
    @test spec_hilbert_curve.layers[2] isa TOA.PointLayer

    spec_map = TOA.visual_spec(emap; kind=:pushforward_overlay)
    @test spec_map.layers[1] isa TOA.SegmentLayer
    @test length(spec_map.layers[1].segments) == FF.nvertices(Qmap)

    spec_cref = TOA.visual_spec(cref; kind=:common_refinement)
    @test spec_cref.layers[1] isa TOA.PointLayer
    @test length(spec_cref.layers[1].points) == FF.nvertices(Pcommon)

    spec_bar = TOA.visual_spec(slice_single; kind=:barcode)
    @test spec_bar.layers[1] isa TOA.SegmentLayer
    @test spec_bar.metadata.total_groups == 1

    spec_bank = TOA.visual_spec(slice_bank; kind=:barcode_bank)
    @test spec_bank.layers[1] isa TOA.HeatmapLayer
    @test size(spec_bank.layers[1].values) == (1, 2)

    spec_farr = TOA.visual_spec(arr; kind=:fibered_arrangement)
    @test spec_farr.layers[1] isa TOA.PolylineLayer
    @test spec_farr.layers[2] isa TOA.PointLayer
    @test TOA.visual_metadata(spec_farr).figure_size == (780, 620)

    spec_fquery = TOA.visual_spec(arr; kind=:fibered_query, dir=[1.0, 1.0], offset=0.0)
    @test TOA.visual_kind(spec_fquery) == :fibered_query
    @test spec_fquery.axes.xlimits[1] < 0.0
    @test spec_fquery.axes.xlimits[2] > 2.0
    @test spec_fquery.axes.ylimits[1] < 0.0
    @test spec_fquery.axes.ylimits[2] > 2.0
    @test TOA.visual_metadata(spec_fquery).figure_size == (780, 620)

    spec_fcell = TOA.visual_spec(arr_box; kind=:fibered_cell_highlight, dir=[1.0, 1.0], offset=0.0)
    @test TOA.visual_kind(spec_fcell) == :fibered_cell_highlight
    @test count(layer -> layer isa TOA.PolylineLayer, spec_fcell.layers) >= 4
    @test TOA.visual_metadata(spec_fcell).figure_size == (860, 620)

    spec_ftie = TOA.visual_spec(arr_box; kind=:fibered_tie_break, dir=[1.0, 1.0], offset=0.0)
    @test TOA.visual_kind(spec_ftie) == :fibered_tie_break
    @test TOA.visual_metadata(spec_ftie).tie_break_relevant
    @test TOA.visual_metadata(spec_ftie).cell_up != TOA.visual_metadata(spec_ftie).cell_down
    @test count(layer -> layer isa TOA.PolylineLayer, spec_ftie.layers) >= 5

    spec_ffam = TOA.visual_spec(fam; kind=:fibered_family)
    @test spec_ffam.layers[2] isa TOA.PolylineLayer

    spec_fchain = TOA.visual_spec(fam_box; kind=:fibered_chain_cells)
    @test TOA.visual_kind(spec_fchain) == :fibered_chain_cells
    @test any(layer -> layer isa TOA.RectLayer, spec_fchain.layers)

    spec_fcontrib = TOA.visual_spec(fam_box; kind=:fibered_family_contributions, caches=(cache_box, cache_box_alt))
    @test TOA.visual_kind(spec_fcontrib) == :fibered_family_contributions
    @test length(TOA.visual_panels(spec_fcontrib)) == 2
    @test TOA.visual_metadata(spec_fcontrib).sampled_matching_distance >= 0.0

    spec_fdist = TOA.visual_spec(fam_box; kind=:fibered_distance_diagnostic, caches=(cache_box, cache_box_alt))
    @test TOA.visual_kind(spec_fdist) == :fibered_distance_diagnostic
    @test length(TOA.visual_panels(spec_fdist)) == 3
    @test TOA.visual_metadata(spec_fdist).argmax_index !== nothing

    spec_foffset = TOA.visual_spec(arr_box; kind=:fibered_offset_intervals, dir=[1.0, 1.0])
    @test TOA.visual_kind(spec_foffset) == :fibered_offset_intervals
    @test any(layer -> layer isa TOA.RectLayer, spec_foffset.layers)

    spec_slice = TOA.visual_spec(slice_res; kind=:fibered_slice)
    @test TOA.visual_kind(spec_slice) == :fibered_slice
    @test any(layer -> layer isa TOA.RectLayer, spec_slice.layers)
    @test any(layer -> layer isa TOA.SegmentLayer, spec_slice.layers)
    @test TOA.visual_metadata(spec_slice).chain_length == length(F2D.slice_chain(slice_res))
    @test TOA.visual_metadata(spec_slice).figure_size == (860, 520)
    spec_slice_overlay = TOA.visual_spec(slice_res; kind=:fibered_slice_overlay, arrangement=arr, dir=[1.0, 1.0], offset=0.0)
    @test TOA.visual_kind(spec_slice_overlay) == :fibered_slice_overlay
    @test any(layer -> layer isa TOA.SegmentLayer, spec_slice_overlay.layers)
    spec_slice_bar = TOA.visual_spec(slice_res; kind=:barcode)
    @test spec_slice_bar.layers[1] isa TOA.SegmentLayer

    spec_query_bar = TOA.visual_spec(cache_grid; kind=:fibered_query_barcode, dir=[1.0, 1.0], offset=0.0)
    @test TOA.visual_kind(spec_query_bar) == :fibered_query_barcode
    @test length(TOA.visual_panels(spec_query_bar)) == 2

    spec_compare = TOA.visual_spec(arr; kind=:fibered_projected_comparison, projected=parr_grid)
    @test TOA.visual_kind(spec_compare) == :fibered_projected_comparison
    @test length(TOA.visual_panels(spec_compare)) == 2

    spec_parr = TOA.visual_spec(parr_grid; kind=:projected_arrangement)
    @test spec_parr.layers[1] isa TOA.PointLayer

    spec_pd = TOA.visual_spec(pdres; kind=:projected_distances)
    @test spec_pd.layers[1] isa TOA.PolylineLayer
    @test length(spec_pd.layers[1].paths[1]) == 2

    spec_pb = TOA.visual_spec(pbres; kind=:barcode_bank)
    @test spec_pb.layers[1] isa TOA.HeatmapLayer
    @test size(spec_pb.layers[1].values) == (1, 2)

    spec_rect = TOA.visual_spec(sb; kind=:rectangles)
    @test any(layer -> layer isa TOA.RectLayer, spec_rect.layers)

    spec_atoms = TOA.visual_spec(pm; kind=:signed_atoms)
    @test spec_atoms.layers[1] isa TOA.PointLayer

    spec_decomp = TOA.visual_spec(smd; kind=:density_image)
    @test spec_decomp.layers[1] isa TOA.HeatmapLayer

    spec_line = TOA.visual_spec(line; kind=:mpp_line_spec)
    @test spec_line.layers[2] isa TOA.PolylineLayer

    spec_mpp_decomp = TOA.visual_spec(decomp; kind=:mpp_decomposition)
    @test TOA.visual_metadata(spec_mpp_decomp).layout == :overlay
    @test TOA.visual_metadata(spec_mpp_decomp).figure_size == (920, 620)
    @test TOA.visual_metadata(spec_mpp_decomp).legend_position == :right
    @test spec_mpp_decomp.layers[1] isa TOA.PolylineLayer
    decomp_layers = [layer for layer in spec_mpp_decomp.layers if layer isa TOA.PolylineLayer]
    @test length(decomp_layers) == 1 + MPI.nsummands(decomp)
    @test length(unique(layer.color for layer in decomp_layers[2:end])) == MPI.nsummands(decomp)
    @test decomp_layers[3].linewidth > decomp_layers[2].linewidth
    @test decomp_layers[3].alpha > decomp_layers[2].alpha

    spec_mpp_decomp_panels = TOA.visual_spec(decomp; kind=:mpp_decomposition, layout=:summands)
    @test TOA.visual_metadata(spec_mpp_decomp_panels).layout == :summands
    @test TOA.visual_metadata(spec_mpp_decomp_panels).figure_size == (1120, 560)
    @test isempty(TOA.visual_layers(spec_mpp_decomp_panels))
    @test length(TOA.visual_panels(spec_mpp_decomp_panels)) == MPI.nsummands(decomp)
    @test all(length(panel.layers) == 2 for panel in TOA.visual_panels(spec_mpp_decomp_panels))

    spec_img = TOA.visual_spec(img; kind=:mpp_image)
    @test spec_img.layers[1] isa TOA.HeatmapLayer
    @test size(spec_img.layers[1].values) == size(MPI.image_values(img))

    spec_land = TOA.visual_spec(L; kind=:mp_landscape)
    @test all(layer -> layer isa TOA.PolylineLayer, spec_land.layers)
    @test length(spec_land.layers) == MPI.landscape_layers(L)
    @test TOA.visual_metadata(spec_land).render_mode == :curves
    @test TOA.visual_metadata(spec_land).figure_size == (860, 520)

    spec_land_slice = TOA.visual_spec(L; kind=:landscape_slices, idir=1, ioff=1)
    @test spec_land_slice.layers[1] isa TOA.PolylineLayer
    @test length(spec_land_slice.layers[1].paths[1]) == length(MPI.landscape_grid(L))

    spec_flange_regions = TOA.visual_spec(FG; kind=:regions, box=([0.0, 0.0], [3.0, 3.0]))
    @test any(layer -> layer isa TOA.RectLayer, spec_flange_regions.layers)

    spec_flange_subdiv = TOA.visual_spec(FG; kind=:constant_subdivision, box=([0.0, 0.0], [3.0, 3.0]))
    @test spec_flange_subdiv.layers[1] isa TOA.HeatmapLayer
    @test size(spec_flange_subdiv.layers[1].values) == (3, 3)

    spec_pc2 = TOA.visual_spec(pc2; kind=:point_density, labels=["p1", "p2", "p3", "p4"])
    @test TOA.visual_kind(spec_pc2) == :point_density
    @test spec_pc2.layers[1] isa TOA.HeatmapLayer
    @test any(layer -> layer isa TOA.PointLayer, spec_pc2.layers)
    @test any(layer -> layer isa TOA.TextLayer, spec_pc2.layers)
    @test !spec_pc2.legend.visible
    @test spec_pc2.axes.xlabel == "x1"
    @test spec_pc2.axes.ylabel == "x2"
    @test TOA.visual_metadata(spec_pc2).figure_size == (760, 520)

    spec_cod_points = TOA.visual_spec(codres; kind=:points_2d)
    @test TOA.visual_kind(spec_cod_points) == :points_2d
    @test spec_cod_points.layers[1] isa TOA.PointLayer
    @test spec_cod_points.metadata.object == :point_codensity_result
    @test spec_cod_points.metadata.dtm_mass == 0.5
    @test spec_cod_points.metadata.neighbor_count == 3
    @test spec_cod_points.metadata.value_range == extrema(DI.codensity_values(codres))
    @test TOA.check_visual_spec(spec_cod_points).valid

    spec_cod_snap = TOA.visual_spec(codres; kind=:codensity_radius_snapshots,
                                    radii=[0.1, 0.2], codensity_levels=[1.0, 1.4])
    @test TOA.visual_kind(spec_cod_snap) == :codensity_radius_snapshots
    @test length(spec_cod_snap.panels) == 4
    @test spec_cod_snap.metadata.nlevels == 2
    @test spec_cod_snap.metadata.nradii == 2
    @test spec_cod_snap.metadata.panel_columns == 2
    @test spec_cod_snap.metadata.dtm_mass == 0.5
    @test spec_cod_snap.metadata.neighbor_count == 3
    @test spec_cod_snap.subtitle == "rows filter by codensity cutoff; columns increase radius"
    @test all(occursin("c <=", panel.title) for panel in spec_cod_snap.panels)
    @test all(occursin("retained points", panel.subtitle) for panel in spec_cod_snap.panels)
    @test all(length(panel.layers) == 3 for panel in spec_cod_snap.panels)
    @test all(panel.layers[2] isa TOA.PointLayer for panel in spec_cod_snap.panels)
    @test all(panel.layers[2].markerspace == :data for panel in spec_cod_snap.panels)
    @test all(isapprox(panel.layers[2].markersize, 2.0 * panel.metadata.radius) for panel in spec_cod_snap.panels)
    @test TOA.check_visual_spec(spec_cod_snap).valid

    spec_pc3 = TOA.visual_spec(pc3; kind=:points_3d)
    @test TOA.visual_kind(spec_pc3) == :points_3d
    @test spec_pc3.layers[1] isa TOA.Point3Layer
    @test TOA.check_visual_spec(spec_pc3).valid
    @test spec_pc3.axes.zlimits !== nothing
    @test !spec_pc3.legend.visible
    @test TOA.visual_metadata(spec_pc3).projected_dims == (1, 2, 3)

    spec_knn = TOA.visual_spec(pc2; kind=:knn_graph, k=2)
    @test any(layer -> layer isa TOA.SegmentLayer, spec_knn.layers)
    @test spec_knn.legend.visible
    @test TOA.visual_metadata(spec_knn).figure_size == (780, 560)
    @test TOA.visual_metadata(spec_knn).legend_position == :right
    @test isapprox((spec_knn.layers[1]::TOA.SegmentLayer).linewidth, 1.3)

    spec_graph = TOA.visual_spec(g3; kind=:weighted_graph)
    @test TOA.visual_kind(spec_graph) == :weighted_graph
    @test any(layer -> layer isa TOA.PointLayer, spec_graph.layers)
    @test count(layer -> layer isa TOA.SegmentLayer, spec_graph.layers) >= 1
    @test spec_graph.legend.visible

    spec_graph3 = TOA.visual_spec(g3; kind=:graph_3d)
    @test any(layer -> layer isa TOA.Point3Layer, spec_graph3.layers)
    @test any(layer -> layer isa TOA.Segment3Layer, spec_graph3.layers)

    spec_epg = TOA.visual_spec(epg; kind=:embedded_planar_graph)
    @test TOA.visual_kind(spec_epg) == :embedded_planar_graph
    @test any(layer -> layer isa TOA.PolylineLayer, spec_epg.layers)

    spec_img2 = TOA.visual_spec(img2; kind=:image)
    @test spec_img2.layers[1] isa TOA.HeatmapLayer
    @test spec_img2.layers[1].show_colorbar

    spec_img3 = TOA.visual_spec(img3; kind=:channels)
    @test length(TOA.visual_panels(spec_img3)) == 3
    @test all(panel -> panel.layers[1] isa TOA.HeatmapLayer, TOA.visual_panels(spec_img3))

    @test_throws ArgumentError TOA.visual_spec(plan; kind=:plan_dashboard)
    @test_throws ArgumentError TOA.visual_spec(est; kind=:preflight_diagnostics)
    @test_throws ArgumentError TOA.visual_spec(gc; kind=:complex_dashboard)
    @test_throws ArgumentError TOA.visual_spec(gcres; kind=:cell_histogram)
    @test_throws ArgumentError TOA.visual_spec(gcres; kind=:grade_ranges)
    @test_throws ArgumentError TOA.visual_spec(gc; kind=:cell_histogram)
    @test_throws ArgumentError TOA.visual_spec(gc; kind=:grade_ranges)
    @test_throws ArgumentError TOA.visual_spec(hilbert_inv; kind=:rank_heatmap)

    spec_st = TOA.visual_spec(st; kind=:simplex_counts)
    @test spec_st.layers[1] isa TOA.RectLayer

    desc = describe(spec_img)
    @test desc.kind == :visualization_spec
    @test TOA.visual_summary(spec_img) == desc
    @test TOA.visual_metadata(spec_img).image_shape == size(MPI.image_values(img))
    @test occursin("VisualizationSpec", repr(MIME("text/plain"), spec_img))
    @test occursin("<div", repr(MIME("text/html"), spec_img))

    bad_spec = TOA.VisualizationSpec(:bad;
                                     layers=TOA.AbstractVisualizationLayer[
                                         TOA.TextLayer(["a"], [(0.0, 0.0), (1.0, 1.0)], :black, 10.0),
                                     ])
    bad_report = TOA.check_visual_spec(bad_spec; throw=false)
    @test !bad_report.valid
    @test_throws ArgumentError TOA.check_visual_spec(bad_spec; throw=true)

    have_cairo = Base.find_package("CairoMakie") !== nothing
    if have_cairo
        @eval import CairoMakie
        @test Base.get_extension(TamerOp, :TamerOpCairoMakieExt) !== nothing
        @test VIZ._visual_backend_available(:cairomakie)
    end
    if have_cairo
        fig_box = TamerOp.visualize(box_pi; kind=:regions, backend=:cairomakie)
        @test fig_box !== nothing
        fig_rank = TamerOp.visualize(rank_inv; kind=:rank_heatmap, backend=:cairomakie)
        @test fig_rank !== nothing
        fig_slice = TamerOp.visualize(slice_res; kind=:fibered_slice, backend=:cairomakie)
        @test fig_slice !== nothing
        fig_slice_overlay = TamerOp.visualize(slice_res; kind=:fibered_slice_overlay,
                                                   arrangement=arr, dir=[1.0, 1.0], offset=0.0,
                                                   backend=:cairomakie)
        @test fig_slice_overlay !== nothing
        fig_tie = TamerOp.visualize(arr_box; kind=:fibered_tie_break, dir=[1.0, 1.0], offset=0.0, backend=:cairomakie)
        @test fig_tie !== nothing
        fig_query_bar = TamerOp.visualize(cache_grid; kind=:fibered_query_barcode,
                                               dir=[1.0, 1.0], offset=0.0, backend=:cairomakie)
        @test fig_query_bar !== nothing
        fig_family_diag = TamerOp.visualize(fam_box; kind=:fibered_distance_diagnostic,
                                                 caches=(cache_box, cache_box_alt), backend=:cairomakie)
        @test fig_family_diag !== nothing
        fig_compare = TamerOp.visualize(arr; kind=:fibered_projected_comparison,
                                             projected=parr_grid, backend=:cairomakie)
        @test fig_compare !== nothing
        fig = TamerOp.visualize(img; backend=:cairomakie)
        @test fig !== nothing
        fig_panels = TamerOp.visualize(decomp; kind=:mpp_decomposition, layout=:summands, backend=:cairomakie)
        @test fig_panels !== nothing
        png_path = tempname() * ".png"
        TamerOp.save_visual(png_path, img; backend=:cairomakie)
        @test isfile(png_path)
        @test filesize(png_path) > 0
        png_export = TamerOp.save_visual(mktempdir(), "mpp_image_static", img; prefer=:static)
        @test png_export isa TamerOp.VisualExportResult
        @test TamerOp.export_format(png_export) == :png
        @test TamerOp.export_backend(png_export) == :cairomakie
        @test isfile(TamerOp.export_path(png_export))
    end

    have_wgl = Base.find_package("WGLMakie") !== nothing
    if have_wgl
        @eval import WGLMakie
        @test Base.get_extension(TamerOp, :TamerOpWGLMakieExt) !== nothing
        @test VIZ._visual_backend_available(:wglmakie)
    end

    export_dir = mktempdir()
    if have_wgl
        html_path = tempname() * ".html"
        html_saved = TamerOp.save_visual(html_path, img)
        @test html_saved == html_path
        @test isfile(html_path)
        @test filesize(html_path) > 0
        html_text = read(html_path, String)
        @test !occursin("VisualizationSpec", html_text)

        export_dir = mktempdir()
        export_res = TamerOp.save_visual(export_dir, "mpp_image_export", img; format=:html)
        @test export_res isa TamerOp.VisualExportResult
        @test TamerOp.export_stem(export_res) == "mpp_image_export"
        @test TamerOp.export_kind(export_res) == :mpp_image
        @test TamerOp.export_format(export_res) == :html
        @test TamerOp.export_backend(export_res) == :wglmakie
        @test isfile(TamerOp.export_path(export_res))
        @test describe(export_res).kind == :visual_export_result
        @test occursin("VisualExportResult", repr(MIME("text/plain"), export_res))

        batch_dir = mktempdir()
        exports = TamerOp.save_visuals(batch_dir,
                                            [
                                                (; stem="mpp_image", obj=img, kind=:mpp_image),
                                                (; stem="mpp_decomposition", obj=decomp, kind=:mpp_decomposition, layout=:summands),
                                            ];
                                            format=:html)
        @test length(exports) == 2
        @test all(res -> res isa TamerOp.VisualExportResult, exports)
        @test [TamerOp.export_stem(res) for res in exports] == ["mpp_image", "mpp_decomposition"]
        @test all(res -> TamerOp.export_format(res) == :html, exports)
        @test all(res -> isfile(TamerOp.export_path(res)), exports)

    else
        missing_path = joinpath(export_dir, "requires_wgl.html")
        @test_throws ArgumentError TamerOp.save_visual(missing_path, img)
        @test !isfile(missing_path)
        @test_throws ArgumentError TamerOp.save_visual(export_dir, "requires_wgl", img; format=:html)
    end

    @test_throws ArgumentError TamerOp.save_visual(export_dir, "mpp_image_export.html", img)
    @test_throws ArgumentError TamerOp.save_visuals(export_dir, [(; obj=img)]; format=:html)

    save_doc = string(@doc TamerOp.save_visual)
    batch_doc = string(@doc TamerOp.save_visuals)
    @test occursin("save_visual(outdir, stem, obj", save_doc)
    @test occursin("save_visuals(outdir, requests", batch_doc)

    if have_wgl
        fig = TOA.render(spec_img; backend=:wglmakie)
        @test fig !== nothing
    end
end

@testset "A21 ordinary persistence visual oracles" begin
    VIZ = TamerOp.Visualization
    finite = [[(1//3, 5//3), (2//1, 3//1)], Tuple{Rational{Int},Rational{Int}}[]]
    essential = [[0//1, 4//1], [2//1]]
    diagram = OP.PersistenceDiagram(finite, essential; field=CM.F2())
    @test VIZ.available_visuals(diagram) == (:persistence_diagram, :barcode)
    @test_throws ArgumentError VIZ.visual_spec(diagram; typo=true)
    @test_throws ArgumentError VIZ.visual_spec(diagram; backend=:cairomakie)
    @test_throws ArgumentError VIZ.visual_spec(diagram; cache=:unused)
    spec = VIZ.visual_spec(diagram)
    @test VIZ.visual_kind(spec) === :persistence_diagram
    @test VIZ.check_visual_spec(spec).valid
    metadata = VIZ.visual_metadata(spec)
    @test metadata.finite_intervals == finite[1]
    @test eltype(metadata.finite_intervals) == Tuple{Rational{Int},Rational{Int}}
    @test metadata.essential_births == essential[1]
    @test metadata.rounded_endpoint_count == 2
    @test metadata.display_coordinates === :float64
    @test metadata.essential_direction == 1
    @test metadata.essential_display_coordinate > 4
    @test metadata.interval_convention == "[birth, death)"
    @test occursin("Float64 display", spec.subtitle)
    @test occursin("Inf: continues indefinitely", spec.subtitle)
    @test spec.layers[2] isa VIZ.PointLayer
    @test [p for (r,p) in zip(metadata.displayed_records,metadata.interval_points) if isfinite(r.death)] == [(1/3,5/3),(2.0,3.0)]
    @test [p for (r,p) in zip(metadata.displayed_records,metadata.interval_points) if !isfinite(r.death)] == [(0.0,metadata.essential_display_coordinate),(4.0,metadata.essential_display_coordinate)]
    @test "+Inf" in spec.axes.yticks[2]

    bars = VIZ.visual_spec(diagram; kind=:barcode)
    @test bars.axes.aspect === :auto
    @test bars.layers[1] isa VIZ.SegmentLayer
    @test [(seg[1],seg[3]) for (r,seg) in zip(bars.metadata.displayed_records,bars.metadata.interval_segments) if isfinite(r.death)] == [(1/3,5/3),(2.0,3.0)]
    @test [(seg[1],seg[3]) for (r,seg) in zip(bars.metadata.displayed_records,bars.metadata.interval_segments) if !isfinite(r.death)] == [(0.0,metadata.essential_display_coordinate),(4.0,metadata.essential_display_coordinate)]
    @test count(status -> last(status) === :essential,bars.metadata.endpoint_display) == 2
    @test all(seg -> seg[1] <= seg[3],bars.metadata.interval_segments)
    @test VIZ.visual_spec(diagram; dim=1).metadata.finite_count == 0
    @test VIZ.visual_spec(diagram; dim=1).metadata.essential_count == 1
    @test VIZ.visual_spec(diagram; dim=7).metadata.essential_count == 0
    @test VIZ.check_visual_spec(VIZ.visual_spec(diagram; dim=7)).valid
    @test !VIZ.check_visual_request(diagram; dim=-1).valid
    @test !VIZ.check_visual_request(diagram; dim=true).valid
    @test !VIZ.check_visual_request(diagram; dim=0.5).valid
    @test_throws ArgumentError VIZ.visual_spec(diagram; dim=-1)
    @test_throws ArgumentError VIZ.visual_spec(diagram; kind=:not_a_kind)
    malformed = deepcopy(diagram)
    push!(malformed.finite_by_dim[1], (2//1, 2//1))
    @test !VIZ.check_visual_request(malformed).valid
    @test_throws ArgumentError VIZ.visual_spec(malformed)

    super_finite = [[(5//3, 1//3), (3//1, 2//1)]]
    super_diagram = OP.PersistenceDiagram(super_finite, [[0//1, 4//1]]; order=:superlevel)
    super_spec = VIZ.visual_spec(super_diagram)
    super_meta = super_spec.metadata
    @test [p for (r,p) in zip(super_meta.displayed_records,super_meta.interval_points) if isfinite(r.birth)] == [(5/3,1/3),(3.0,2.0)]
    @test super_meta.finite_intervals == super_finite[1]
    @test super_meta.essential_display_coordinate < 0
    @test super_meta.essential_direction == -1
    @test super_meta.interval_convention == "(death, birth]"
    @test "-Inf" in super_spec.axes.yticks[2]
    super_bars = VIZ.visual_spec(super_diagram; kind=:barcode)
    @test count(status -> first(status) === :essential,super_bars.metadata.endpoint_display) == 2
    @test [(seg[1],seg[3]) for (r,seg) in zip(super_bars.metadata.displayed_records,super_bars.metadata.interval_segments) if isfinite(r.birth)] == [(1/3,5/3),(2.0,3.0)]

    signed_zeros = OP.PersistenceDiagram([Tuple{Float64,Float64}[]], [[-0.0, 0.0]])
    zero_spec = VIZ.visual_spec(signed_zeros)
    @test VIZ.check_visual_spec(zero_spec).valid
    @test zero_spec.metadata.essential_count == 2
    @test isequal(zero_spec.metadata.essential_births, [-0.0, 0.0])
    @test zero_spec.metadata.total_groups == 1
    @test only(zero_spec.metadata.records).multiplicity == 2
    zero_bars = VIZ.visual_spec(signed_zeros; kind=:barcode)
    @test length(zero_bars.metadata.interval_segments) == 1
    @test zero_bars.metadata.total_multiplicity == 2

    large = big(2)^60
    close_endpoints = OP.PersistenceDiagram([[(large, large + 1)]], [BigInt[]])
    @test_throws ArgumentError VIZ.visual_spec(close_endpoints)
    overflow = OP.PersistenceDiagram([[(big(10)^400, big(10)^401)]], [BigInt[]])
    @test_throws ArgumentError VIZ.visual_spec(overflow)
    @test OP.finite_intervals(close_endpoints; dim=0) == [(large, large + 1)]
    @test OP.finite_intervals(diagram; dim=0) == finite[1]

    if Base.find_package("CairoMakie") === nothing
        @test_skip false
    else
        @eval import CairoMakie
        @test Base.get_extension(TamerOp, :TamerOpCairoMakieExt) !== nothing
        @test VIZ._visual_backend_available(:cairomakie)
        barcode_figure = VIZ.visualize(super_diagram; kind=:barcode, backend=:cairomakie)
        CairoMakie.Makie.update_state_before_display!(barcode_figure)
        barcode_axis = only(filter(item -> item isa CairoMakie.Axis, barcode_figure.content))
        plot_size = CairoMakie.Makie.widths(CairoMakie.Makie.viewport(barcode_axis.scene)[])
        figure_size = CairoMakie.Makie.widths(CairoMakie.Makie.viewport(barcode_figure.scene)[])
        # The plot should occupy the canvas; a bottom legend must not determine
        # the axis column width or absorb half of the available row height.
        @test plot_size[1] > 0.6 * figure_size[1]
        @test plot_size[2] > 0.5 * figure_size[2]
        mktempdir() do dir
            path = VIZ.save_visual(joinpath(dir, "ordinary.svg"), diagram; backend=:cairomakie)
            svg = read(path, String)
            @test occursin("<svg", svg)
            @test occursin("<path", svg)
            @test !occursin("VisualizationSpec", svg)
            barcode_path = VIZ.save_visual(joinpath(dir, "ordinary_superlevel.png"), super_diagram;
                                           kind=:barcode, backend=:cairomakie)
            bytes = read(barcode_path)
            @test bytes[1:8] == UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
            @test length(bytes) > 1000
            # Public computation-to-figure path: a ring is born at 0 and dies at 5.
            image = zeros(Int, 3, 3)
            image[2, 2] = 5
            computed = TamerOp.cubical_persistence(image)
            @test TamerOp.finite_intervals(computed; dim=1) == [(0, 5)]
            computed_path = TamerOp.save_visual(joinpath(dir, "computed_ring.svg"), computed;
                                               kind=:barcode, dim=1, backend=:cairomakie)
            @test occursin("<svg", read(computed_path, String))
        end
    end
end

@testset "A16 WGL HTML export and backend switching" begin
    if Base.find_package("WGLMakie") === nothing
        @test_skip false
    else
        @eval import WGLMakie
        have_cairo = Base.find_package("CairoMakie") !== nothing
        have_cairo && (@eval import CairoMakie)
        @test Base.get_extension(TamerOp, :TamerOpWGLMakieExt) !== nothing
        @test TamerOp.Visualization._visual_save_available(:wglmakie)
        diagram = TamerOp.OrdinaryPersistence.PersistenceDiagram(
            [[(1//3, 5//3)]], [[0//1]])
        mktempdir() do dir
            exported = TamerOp.save_visual(dir, "ordinary_interactive", diagram;
                kind=:barcode, backend=:wglmakie, format=:html)
            @test TamerOp.export_backend(exported) === :wglmakie
            @test TamerOp.export_format(exported) === :html
            @test TamerOp.export_kind(exported) === :barcode
            html = read(TamerOp.export_path(exported), String)
            # Bonito serializes the WGL scene and its canvas into a standalone
            # document. Browser execution is a separate integration concern.
            @test occursin("<html", lowercase(html))
            @test occursin("<canvas", lowercase(html))
            @test occursin("<script", lowercase(html))
            @test occursin("Bonito", html)
            @test !occursin("VisualizationSpec", html)
            @test sizeof(html) > 10_000
            @test WGLMakie.Makie.current_backend() === WGLMakie
            if have_cairo
                # Import order and a previously active WGL renderer must not
                # override the explicit static export request.
                @test Base.get_extension(TamerOp, :TamerOpCairoMakieExt) !== nothing
                static = TamerOp.save_visual(dir, "ordinary_after_wgl", diagram;
                    kind=:barcode, backend=:cairomakie, format=:png)
                @test TamerOp.export_backend(static) === :cairomakie
                @test TamerOp.export_format(static) === :png
                bytes = read(TamerOp.export_path(static))
                @test bytes[1:8] == UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
                @test length(bytes) > 1000
                @test CairoMakie.Makie.current_backend() === CairoMakie
            else
                @test_skip false
            end
        end
    end
end

@testset "A81 signed density coordinate reconstruction" begin
    V = TamerOp.Visualization
    for axes in (([10, 20], [10, 20]), ([-7, -2, 10], [-11, 5]))
        xs, ys = axes
        rects = [SM.Rect{2}((first(xs), first(ys)), (last(xs), last(ys))),
                 SM.Rect{2}((last(xs), first(ys)), (last(xs), last(ys)))]
        sb = SM.RectSignedBarcode(axes, rects, [2, -1])
        @test SM.check_rect_signed_barcode(sb).valid
        spec = V.visual_spec(sb; kind=:density_image)
        heat = only(spec.layers)
        # Independent rectangle-membership oracle; no endpoint-to-index conversion.
        expected = [sum(w for (r, w) in zip(rects, [2, -1])
                        if r.lo[1] <= x <= r.hi[1] && r.lo[2] <= y <= r.hi[2])
                    for y in ys, x in xs]
        @test heat.values == expected
        @test heat.x == xs
        @test heat.y == ys
        drawing = V.visual_spec(sb; kind=:rectangles)
        @test drawing.axes.xlimits[1] <= first(xs) <= last(xs) <= drawing.axes.xlimits[2]
        @test drawing.axes.ylimits[1] <= first(ys) <= last(ys) <= drawing.axes.ylimits[2]
        out = SM.SignedMeasureDecomposition(rectangles=sb)
        @test V.visual_spec(out; kind=:density_image).layers[1].values == expected
        @test !V.check_visual_request(sb; kind=:density_image, box=([0,0], [1,1])).valid
    end
    sb3 = SM.RectSignedBarcode(([1, 2], [1, 2], [1, 2]),
        [SM.Rect{3}((1, 1, 1), (2, 2, 2))], [1])
    @test isempty(V.available_visuals(SM.SignedMeasureDecomposition(rectangles=sb3)))
end

@testset "A81 and A82 invariant windows and exact rank queries" begin
    V = TamerOp.Visualization
    # The closed square has a one-dimensional stalk on [0,2]^2 and zero outside.
    for field in FIELDS_FULL
        enc = TamerOp.encode([PLB.BoxUpset([0.0, 0.0])], [PLB.BoxDownset([2.0, 2.0])],
            reshape([CM.coerce(field, 1)], 1, 1),
            OPT.EncodingOptions(backend=:pl_backend, poset_kind=:signature, field=field))
        rank = TamerOp.invariant(enc; which=:rank_invariant)
        hilbert = TamerOp.invariant(enc; which=:restricted_hilbert)
        boundary = (2//1, 1//1)
        exterior = (2 + 1//(2^53), 1//1)
        box = ([3//2, 1//2], [5//2, 3//2])
        spec = V.visual_spec(rank; kind=:rank_query_overlay,
            pairs=[(boundary, boundary), (exterior, exterior)], box=box)
        @test [q.rank for q in spec.metadata.query_results] == [1, 0]
        @test spec.metadata.query_results[2].source == exterior
        @test spec.axes.xlimits == (1.5, 2.5)
        @test spec.axes.ylimits == (0.5, 1.5)
        heat = V.visual_spec(hilbert; kind=:hilbert_heatmap, box=box)
        @test heat.axes.xlimits == (1.5, 2.5)
        @test heat.axes.ylimits == (0.5, 1.5)
        @test !V.check_visual_request(hilbert; kind=:hilbert_bars, box=box).valid
    end
    P = FF.ProductOfChainsPoset((3, 2))
    pi = EC.GridEncodingMap(P, ([0, 2, 7], [-4, 1]); orientation=(1, -1))
    dims = [1, 2, 3, 4, 5, 6]
    res = RES.CohomologyDimsResult(P, dims, EC.compile_encoding(P, pi); degree=2)
    box = ([1, -3], [5, 0])
    for kind in V.available_visuals(res)
        spec = V.visual_spec(res; kind=kind, box=box)
        @test spec.axes.xlimits == (1.0, 5.0)
        @test spec.axes.ylimits == (-3.0, 0.0)
        @test spec.metadata.region_ids == sort(unique(c.region_id for c in spec.metadata.geometry.components if c.region_id > 0))
        for c in spec.metadata.geometry.components
            q = [sum(v[j] for v in c.vertices) / length(c.vertices) for j in 1:2]
            c.dimension == 2 && @test EC.locate(pi, q) == c.region_id
        end
    end
    P3 = FF.ProductOfChainsPoset((2, 2, 2))
    pi3 = EC.GridEncodingMap(P3, ([0,1], [0,1], [0,1]))
    c3 = RES.CohomologyDimsResult(P3, ones(Int, 8), EC.compile_encoding(P3, pi3); degree=0)
    @test isempty(V.available_visuals(c3))
    @test !V.check_visual_request(c3; kind=:cohomology_support).valid
end
@testset "A81 ingestion axes and effective previews" begin
    VIZ = TamerOp.Visualization
    DT = TamerOp.DataTypes

    # Entries identify the source axis coordinates independently of the renderer.
    A = [100i + 10j + k for i in 1:2, j in 1:3, k in 1:5]
    img = DT.ImageNd(A)
    default = VIZ.visual_spec(img; kind=:image)
    @test default.metadata.view_dims == (2, 1)
    @test default.metadata.fixed_indices == Dict(3 => 3)
    @test default.layers[1].values == Float64.(A[:, :, 3])
    @test default.axes.xlabel == "axis 2"
    @test default.axes.ylabel == "axis 1"
    @test !haskey(default.metadata, :volume)

    for xdim in 1:3, ydim in 1:3
        xdim == ydim && continue
        other = only(setdiff(1:3, (xdim, ydim)))
        fixed = Dict(other => 1)
        spec = VIZ.visual_spec(img; kind=:image, view_dims=(xdim, ydim),
                               slice_indices=fixed, colormap=:viridis)
        heatmap = only(spec.layers)
        @test heatmap.x == collect(1.0:size(A, xdim))
        @test heatmap.y == collect(1.0:size(A, ydim))
        @test size(heatmap.values) == (size(A, ydim), size(A, xdim))
        @test spec.axes.xlabel == "axis $xdim"
        @test spec.axes.ylabel == "axis $ydim"
        @test heatmap.colormap == :viridis
        for y in axes(heatmap.values, 1), x in axes(heatmap.values, 2)
            index = ones(Int, 3)
            index[xdim] = x
            index[ydim] = y
            @test heatmap.values[y, x] == A[index...]
        end
        @test VIZ.check_visual_spec(spec).valid

        live = VIZ.visual_spec(img; kind=:slice_viewer, view_dims=(xdim, ydim),
                               slice_indices=fixed)
        @test only(live.layers).values == heatmap.values
        @test live.metadata.requires_live_julia
        @test live.interaction.widgets == (:slice_index,)
        @test !live.interaction.hover
        # The renderer uses this same extraction after each slider update.
        last_fixed = Dict(other => size(A, other))
        last_spec = VIZ.visual_spec(img; kind=:image, view_dims=(xdim, ydim),
                                    slice_indices=last_fixed)
        @test VIZ._image_slice_values(A, (xdim, ydim), last_fixed) == only(last_spec.layers).values
    end

    # An explicit partial selection must retain defaults for every other fixed axis.
    A4 = [1000i + 100j + 10k + l for i in 1:2, j in 1:3, k in 1:4, l in 1:5]
    img4 = DT.ImageNd(A4)
    plane = VIZ.visual_spec(img4; kind=:image, view_dims=(4, 2), slice_indices=Dict(1 => 2))
    @test plane.metadata.fixed_indices == Dict(1 => 2, 3 => 2)
    @test only(plane.layers).values == [Float64(A4[2, y, 2, x]) for y in 1:3, x in 1:5]
    @test VIZ.check_visual_spec(plane).valid

    channels = DT.ImageNd(A[:, :, 1:4])
    panels = VIZ.visual_spec(channels; kind=:channels, view_dims=(1, 2))
    @test length(panels.panels) == 4
    for (c, panel) in enumerate(panels.panels)
        @test only(panel.layers).values == [Float64(A[x, y, c]) for y in 1:3, x in 1:2]
        @test panel.axes.xlabel == "axis 1"
        @test panel.axes.ylabel == "axis 2"
    end
    @test VIZ.check_visual_spec(panels).valid
    @test_throws ArgumentError VIZ.visual_spec(channels; kind=:channels, view_dims=(3, 1))
    @test_throws ArgumentError VIZ.visual_spec(channels; kind=:channels, slice_indices=Dict(3 => 2))
    @test_throws ArgumentError VIZ.visual_spec(img; kind=:image, view_dims=(1, 1))
    @test_throws ArgumentError VIZ.visual_spec(img; kind=:image, view_dims=(1, 4))
    @test_throws ArgumentError VIZ.visual_spec(img; kind=:image, slice_indices=Dict(1 => 1))
    @test_throws ArgumentError VIZ.visual_spec(img; kind=:image, slice_indices=Dict(3 => 0))
    @test_throws ArgumentError VIZ.visual_spec(img; kind=:image, slice_indices=Dict(3 => 6))
    @test_throws ArgumentError VIZ.visual_spec(img; kind=:image, view_dims=(1, 2, 3))
    @test VIZ.available_visuals(DT.ImageNd([1.0, 2.0])) == ()
    @test VIZ.available_visuals(DT.ImageNd(zeros(0, 2))) == ()

    # A request must produce its stated dimension, not silently fall back.
    pc = DT.PointCloud([1.0 2.0 3.0; 4.0 5.0 6.0])
    projected = VIZ.visual_spec(pc; kind=:points_2d, dims=(3, 1))
    @test only(projected.layers).points == [(3.0, 1.0), (6.0, 4.0)]
    @test projected.axes.xlabel == "x3"
    @test projected.axes.ylabel == "x1"
    @test_throws ArgumentError VIZ.visual_spec(pc; kind=:points_2d, dims=(1, 1))
    @test_throws ArgumentError VIZ.visual_spec(pc; kind=:points_3d, dims=(1, 2))
    @test_throws ArgumentError VIZ.visual_spec(pc; kind=:points_2d, dims=(1, 4))
    @test_throws ArgumentError VIZ.visual_spec(pc; kind=:points_3d, labels=["a", "b"])
    @test VIZ.available_visuals(DT.PointCloud(reshape([1.0, 2.0], 2, 1))) == ()
    @test VIZ.available_visuals(DT.GraphData(2, [(1, 2)]; coords=reshape([1.0, 2.0], 2, 1))) == ()
    unweighted = DT.GraphData(2, [(1, 2)])
    @test VIZ.available_visuals(unweighted) == (:graph,)
    @test_throws ArgumentError VIZ.visual_spec(unweighted; kind=:weighted_graph)
    @test_throws ArgumentError VIZ.visual_spec(unweighted; kind=:graph, dims=(2, 1))

    for points in (zeros(0, 2), [2.0 3.0])
        preview = VIZ.visual_spec(DT.PointCloud(points); kind=:radius_graph)
        @test VIZ.check_visual_spec(preview).valid
        @test isempty(preview.layers[1].segments)
    end
    density = VIZ.visual_spec(DT.PointCloud([2.0 3.0]); kind=:point_density)
    @test sum(density.layers[1].values) == 1.0
    @test all(diff(density.layers[1].x) .> 0)
    @test all(diff(density.layers[1].y) .> 0)

    # Colors must use the same thirds of the weight range that the legend states.
    weighted = DT.GraphData(4, [(1, 2), (2, 3), (3, 4)];
                           coords=[0.0 0.0; 1.0 0.0; 2.0 0.0; 3.0 0.0],
                           weights=[0.0, 0.4, 1.0])
    edge_spec = VIZ.visual_spec(weighted; kind=:weighted_graph)
    @test edge_spec.layers[1].segments == [(0.0, 0.0, 1.0, 0.0)]
    @test edge_spec.layers[2].segments == [(1.0, 0.0, 2.0, 0.0)]
    @test edge_spec.layers[3].segments == [(2.0, 0.0, 3.0, 0.0)]
end

@testset "A82 exact queries retain their mathematical coordinates" begin
    VIZ = TamerOp.Visualization
    PLB = TamerOp.PLBackend
    EC = TamerOp.EncodingCore
    _, _, pi = PLB.encode_fringe_boxes([PLB.BoxUpset([0.0, 0.0])],
        [PLB.BoxDownset([2.0, 2.0])], TamerOp.Advanced.EncodingOptions())
    epsilon = 1 // (big(1) << 53)
    boundary = (2//big(1), 1//big(1))
    exterior = (2 + epsilon, 1//big(1))
    spec = VIZ.visual_spec(pi; kind=:query_overlay, points=[boundary, exterior], box=([-1, -1], [3, 3]))
    q = spec.metadata.query_readout
    @test q[1].point == boundary
    @test q[2].point == exterior
    @test q[1].region_id != q[2].region_id
    @test pi.sig_y[q[1].region_id] == pi.sig_y[q[2].region_id] == BitVector([true])
    @test pi.sig_z[q[1].region_id] == BitVector([false]) # complement of the downset
    @test pi.sig_z[q[2].region_id] == BitVector([true])
    @test q[1].display_point == q[2].display_point == (2.0, 1.0)
    @test !q[1].display_rounded && q[2].display_rounded
    @test length(spec.metadata.query_collisions) == 1
    @test !isempty(spec.metadata.warnings)
    @test occursin("rounding", spec.subtitle)
    @test_throws ArgumentError VIZ.visual_spec(pi; kind=:query_overlay, point=(Inf, 1))
    @test_throws ArgumentError VIZ.visual_spec(pi; kind=:query_overlay, point=(0, NaN))
end

@testset "A82 grid fibers follow grades orientation and viewport" begin
    VIZ = TamerOp.Visualization
    EC = TamerOp.EncodingCore
    FF = TamerOp.FiniteFringe
    pi = EC.GridEncodingMap(FF.ProductOfChainsPoset((2, 2)), ([0, 2], [1, 4]); orientation=(-1, 1))
    spec = VIZ.visual_spec(pi; kind=:regions, box=([-5, 0], [1, 6]))
    geometry = spec.metadata.geometry
    @test geometry.region_ids == [0, 1, 2, 3, 4]
    @test geometry.has_unrepresented_area
    cell = only(filter(c -> c.region_id == 1, geometry.components))
    @test Set(cell.vertices) == Set([(-2, 1), (0, 1), (0, 4), (-2, 4)])
    # This fiber is (-2,0] x [1,4): the reflected axis reverses boundary ownership.
    @test cell.edge_included == BitVector([true, true, false, false])
    @test cell.vertex_included == BitVector([false, true, false, false])
    @test all(!, cell.edge_clipped)
    @test all(c -> all(p -> -5 <= p[1] <= 1 && 0 <= p[2] <= 6, c.vertices), geometry.components)
    queries = VIZ.visual_spec(pi; kind=:query_overlay, points=[(-2, 1), (0, 1), (0, 4), (1, 1)], box=([-5, 0], [1, 6]))
    @test [q.region_id for q in queries.metadata.query_readout] == [2, 1, 3, 0]
    clipped = VIZ.visual_spec(pi; kind=:region_labels, box=([-5, 9//2], [-3, 11//2]))
    @test clipped.metadata.region_ids == [4]
    @test keys(clipped.legend.entries) == (:R4,)
    @test clipped.legend.entries.R4 == spec.legend.entries.R4
    @test only(filter(l -> l isa VIZ.TextLayer, clipped.layers)).labels == ["4"]
    @test !clipped.metadata.geometry.has_unrepresented_area
end

@testset "A82 clipped polyhedra retain open edges and lower strata" begin
    VIZ = TamerOp.Visualization
    PLP = TamerOp.PLPolyhedra
    Q = Rational{BigInt}
    function make_map(A, b, witness; strict=falses(length(b)))
        hp = PLP.HPoly(2, Q.(A), Q.(b), nothing, strict, zero(Q))
        PLP.PLEncodingMap(2, [BitVector()], [BitVector()], [hp], [witness])
    end
    triangle = make_map([-1 0; 0 -1; 1 1], [0, 0, 1], (1//4, 1//4); strict=BitVector([false, false, true]))
    spec = VIZ.visual_spec(triangle; kind=:region_labels, box=([-1, -1], [2, 2]))
    c = only(spec.metadata.geometry.components)
    @test c.dimension == 2
    @test Set(c.vertices) == Set([(0, 0), (1, 0), (0, 1)])
    @test count(c.vertex_included) == 1
    @test only(c.vertices[c.vertex_included]) == (0, 0)
    @test count(c.edge_included) == 2
    @test any(l -> l isa VIZ.SegmentLayer && l.linestyle === :dash, spec.layers)
    @test spec.metadata.geometry.has_unrepresented_area
    @test any(l -> l isa VIZ.PolygonLayer, spec.layers)
    @test VIZ.check_visual_spec(spec).valid
    # x=1, 0<=y<=1 is a line-only fiber, rather than an empty 2D cell.
    line = make_map([1 0; -1 0; 0 -1; 0 1], [1, -1, 0, 1], (1, 1//2))
    line_spec = VIZ.visual_spec(line; kind=:regions, box=([0, -1], [2, 2]))
    line_cell = only(line_spec.metadata.geometry.components)
    @test line_cell.dimension == 1
    @test Set(line_cell.vertices) == Set([(1, 0), (1, 1)])
    @test all(line_cell.vertex_included) && all(line_cell.edge_included)
    @test any(l -> l isa VIZ.SegmentLayer && l.linewidth == 3.0, line_spec.layers)
    point = make_map([1 0; -1 0; 0 1; 0 -1], [1, -1, 2, -2], (1, 2))
    point_spec = VIZ.visual_spec(point; kind=:regions, box=([0, 0], [3, 3]))
    point_cell = only(point_spec.metadata.geometry.components)
    @test point_cell.dimension == 0
    @test point_cell.vertices == [(1, 2)]
    @test point_cell.vertex_included == BitVector([true])
    @test any(l -> l isa VIZ.PointLayer && l.points == [(1.0, 2.0)] && l.markersize == 11.0, point_spec.layers)
    # A window touching only the excluded hypotenuse contains no part of the fiber.
    empty_spec = VIZ.visual_spec(triangle; kind=:regions, box=([1, 0], [2, 1]))
    @test isempty(empty_spec.metadata.geometry.components)
    @test empty_spec.metadata.region_ids == [0]
end

@testset "A82 disconnected box fibers and colliding exact geometry" begin
    VIZ = TamerOp.Visualization
    PLB = TamerOp.PLBackend
    PLP = TamerOp.PLPolyhedra
    EC = TamerOp.EncodingCore
    _, _, pi = PLB.encode_fringe_boxes([PLB.BoxUpset([0.0, 0.0])],
        [PLB.BoxDownset([2.0, 2.0])], TamerOp.Advanced.EncodingOptions())
    r = EC.locate(pi, [-1, 3])
    @test EC.locate(pi, [3, -1]) == r
    @test pi.sig_y[r] == BitVector([false])
    @test pi.sig_z[r] == BitVector([true]) # outside both indicators
    spec = VIZ.visual_spec(pi; kind=:regions, box=([-2, -2], [4, 4]))
    pieces = filter(c -> c.region_id == r, spec.metadata.geometry.components)
    @test length(pieces) == 2
    @test Set(Set(c.vertices) for c in pieces) == Set([Set([(-2, 2), (0, 2), (0, 4), (-2, 4)]),
                                                     Set([(2, -2), (4, -2), (4, 0), (2, 0)])])
    Q = Rational{BigInt}
    epsilon = 1 // (big(1) << 54)
    hp = PLP.HPoly(2, Q[-1 0; 1 0; 0 -1; 0 1], Q[-1, 1+epsilon, 0, 1], nothing, falses(4), zero(Q))
    thin = PLP.PLEncodingMap(2, [BitVector()], [BitVector()], [hp], [(1+epsilon/2, 1//2)])
    thin_spec = VIZ.visual_spec(thin; kind=:regions, box=([0, -1], [2, 2]))
    tc = only(thin_spec.metadata.geometry.components)
    @test tc.dimension == 2
    @test maximum(p[1] for p in tc.vertices) - minimum(p[1] for p in tc.vertices) == epsilon
    @test !isempty(thin_spec.metadata.geometry.coordinate_collisions)
    @test !isempty(thin_spec.metadata.warnings)
end

@testset "A81 visualization request and renderer contracts" begin
    VIZ = TamerOp.Visualization
    DT = TamerOp.DataTypes
    pc = DT.PointCloud([0.0 0.0; 1.0 2.0; 3.0 1.0])
    report = VIZ.check_visual_request(pc; kind=:knn_graph, k=1)
    @test report.valid
    @test :k in report.supported_keywords
    @test report.construction_cost.work == :all_pairs_distances
    @test report.construction_cost.timing == :not_measured
    @test !report.rendering.hover
    @test !report.rendering.selection
    @test !VIZ.check_visual_request(pc; kind=:points_2d, k=1).valid
    @test !VIZ.check_visual_request(pc; kind=:knn_graph, k="one").valid
    @test !VIZ.check_visual_request(pc; kind=:points_2d, dims=(1, 1)).valid
    @test !VIZ.check_visual_request(pc; kind=:points_2d, color_values=[1.0]).valid
    @test !VIZ.check_visual_request(pc; kind=:points_2d, backend=:a81_missing).valid
    @test_throws ArgumentError VIZ.visual_spec(pc; kind=:points_2d, typo=1)
    @test_throws ArgumentError VIZ.visual_spec(pc; kind=:points_2d, backend=:cairomakie)

    spec = VIZ.visual_spec(pc; labels=["a", "b", "c"])
    @test VIZ.visual_spec(spec) === spec
    @test !spec.interaction.hover && !spec.interaction.clicks
    @test spec.interaction.labels
    @test spec.interaction.mode == :static
    @test !spec.interaction.requires_live_julia
    @test VIZ.visual_summary(spec).rendering.renderer_keywords == (:figure, :size, :style)
    @test_throws ArgumentError VIZ.visualize(spec; kind=:barcode)
    @test_throws ArgumentError VIZ.render(spec; linewidth=10)
    @test_throws ArgumentError VIZ.render(spec; size=(-1, 400))
    @test_throws ArgumentError VIZ.render(spec; size=(500, 400), figure=:existing)
    @test_throws ArgumentError VIZ.render(spec; display=:imaginary)

    # The receiver has no kwargs splat: recipe keywords must never leak to it.
    VIZ._register_visual_backend!(:a81_capture;
        render=(spec; display, figure, size, style) -> (; spec, display, figure, size, style))
    try
        rendered = VIZ.visualize(pc; kind=:knn_graph, k=1, backend=:a81_capture, size=(601, 407))
        @test rendered.size == (601, 407)
        @test rendered.spec.kind == :knn_graph
        @test VIZ.check_visual_request(pc; backend=:a81_capture).valid
        @test :a81_capture in VIZ.visual_summary(spec).rendering.activated_backends
        @test_throws ArgumentError VIZ.visualize(pc; kind=:points_2d, alpha=0.5, backend=:a81_capture)
    finally
        delete!(VIZ._VISUAL_RENDERERS, :a81_capture)
    end

    poly = VIZ.PolygonLayer([[(0.0, 0.0), (2.0, 0.0), (0.0, 1.0)]], :orange, :black, 0.5, 0.0)
    solid = VIZ.SegmentLayer([(0.0, 0.0, 2.0, 0.0)], :black, 1.0, 1.0)
    @test solid.linestyle == :solid
    dashed = VIZ.SegmentLayer([(2.0, 0.0, 0.0, 1.0)], :black, 1.0, 1.0, :dash)
    polygon_spec = VIZ.VisualizationSpec(:polygon_test;
        layers=VIZ.AbstractVisualizationLayer[poly, solid, dashed])
    @test VIZ.check_visual_spec(polygon_spec).valid
    malformed = VIZ.VisualizationSpec(:polygon_test; layers=VIZ.AbstractVisualizationLayer[
        VIZ.PolygonLayer([[(0.0, 0.0), (1.0, 0.0)]], :orange, :black, 0.5, 0.0)])
    @test !VIZ.check_visual_spec(malformed).valid
    false_hover = VIZ.VisualizationSpec(:hover_test;
        layers=spec.layers, interaction=(; hover=true))
    @test !VIZ.check_visual_spec(false_hover).valid

    A = [100i + 10j + k for i in 1:2, j in 1:3, k in 1:4]
    image = DT.ImageNd(A)
    @test !VIZ.check_visual_request(image; view_dims=(1, 1)).valid
    @test !VIZ.check_visual_request(image; view_dims=(1, 3), slice_indices=Dict(2 => 9)).valid
    widget = VIZ.visual_spec(image; kind=:slice_viewer, view_dims=(1, 3), slice_indices=Dict(2 => 1))
    @test widget.interaction.requires_live_julia
    @test !widget.interaction.offline_widgets
    mktempdir() do dir
        @test_throws ArgumentError VIZ.save_visual(joinpath(dir, "widget.html"), widget)
        @test !isfile(joinpath(dir, "widget.html"))
        @test_throws ArgumentError VIZ.save_visual(joinpath(dir, "unknown.png"), spec; imaginary=1)
    end

    if Base.find_package("CairoMakie") !== nothing
        @eval import CairoMakie
        fig = VIZ.render(polygon_spec; backend=:cairomakie, size=(601, 407))
        CairoMakie.Makie.update_state_before_display!(fig)
        @test Tuple(CairoMakie.Makie.widths(CairoMakie.Makie.viewport(fig.scene)[])) == (601, 407)
        static_widget = VIZ.render(widget; backend=:cairomakie)
        @test !any(item -> item isa CairoMakie.Slider, static_widget.content)
        ax = only(filter(item -> item isa CairoMakie.Axis, static_widget.content))
        hm = only(filter(plot -> plot isa CairoMakie.Makie.Heatmap, ax.scene.plots))
        @test hm[3][] == Float64.(A[:, 1, :])
    end
end

@testset "A81 live volume coordinates" begin
    if Base.find_package("WGLMakie") !== nothing
        @eval import WGLMakie
        VIZ = TamerOp.Visualization
        A = [100i + 10j + k for i in 1:2, j in 1:3, k in 1:4]
        img = TamerOp.DataTypes.ImageNd(A)
        for vd in ((1, 3), (3, 1))
            spec = VIZ.visual_spec(img; kind=:slice_viewer, view_dims=vd, slice_indices=Dict(2 => 1))
            fig = VIZ.render(spec; backend=:wglmakie, size=(600, 420))
            ax = only(filter(item -> item isa WGLMakie.Axis, fig.content))
            hm = only(filter(plot -> plot isa WGLMakie.Makie.Heatmap, ax.scene.plots))
            # Makie's first matrix axis follows displayed x.
            expected = vd == (1, 3) ? Float64.(A[:, 1, :]) : permutedims(Float64.(A[:, 1, :]))
            @test hm[3][] == expected
            slider = only(filter(item -> item isa WGLMakie.Makie.Slider, fig.content))
            slider.value[] = 3
            expected = vd == (1, 3) ? Float64.(A[:, 3, :]) : permutedims(Float64.(A[:, 3, :]))
            @test hm[3][] == expected
        end
    end
end

@testset "A82 viewport retains fibers only on its edges and corners" begin
    VIZ = TamerOp.Visualization
    EC = TamerOp.EncodingCore
    FF = TamerOp.FiniteFringe
    pi = EC.GridEncodingMap(FF.ProductOfChainsPoset((2, 2)), ([0, 2], [0, 2]))
    edge = VIZ.visual_spec(pi; kind=:regions, box=([0, 0], [2, 1])).metadata.geometry
    @test edge.region_ids == [1, 2]
    @test only(filter(c -> c.region_id == 2, edge.components)).dimension == 1
    @test only(filter(c -> c.region_id == 2, edge.components)).vertices == [(2, 0), (2, 1)]
    corner = VIZ.visual_spec(pi; kind=:regions, box=([0, 0], [2, 2])).metadata.geometry
    @test corner.region_ids == [1, 2, 3, 4]
    @test only(filter(c -> c.region_id == 4, corner.components)).dimension == 0
    @test only(filter(c -> c.region_id == 4, corner.components)).vertices == [(2, 2)]
    @test only(filter(c -> c.region_id == 2, corner.components)).dimension == 1
    @test only(filter(c -> c.region_id == 3, corner.components)).dimension == 1
end

@testset "A82 exact algebraic grid geometry" begin
    VIZ = TamerOp.Visualization
    EC = TamerOp.EncodingCore
    FF = TamerOp.FiniteFringe
    AR = TamerOp.ExactReals.AlgebraicReal
    radius = sqrt(AR(2))
    pi = EC.GridEncodingMap(FF.ProductOfChainsPoset((2, 2)), ([AR(0), radius], [AR(0), AR(1)]))
    spec = VIZ.visual_spec(pi; kind=:query_overlay, points=[(radius, AR(1)/2)], box=([0, 0], [2, 2]))
    @test only(spec.metadata.query_readout).region_id == 2
    @test only(spec.metadata.query_readout).point[1] == radius
    @test only(spec.metadata.query_readout).display_rounded
    first_region = only(filter(c -> c.region_id == 1, spec.metadata.geometry.components))
    @test maximum(p[1] for p in first_region.vertices) == radius
    @test first_region.vertices[2][1]^2 == 2
    @test !first_region.edge_included[2]
end

@testset "A81 and A82 integer query and scalar view contracts" begin
    V = TamerOp.Visualization
    ZE = TamerOp.ZnEncoding
    face = FZ.Face(2, [false, false])
    field = CM.QQField()
    flange = FZ.Flange(2, [FZ.IndFlat(face, (0,0); id=:U)],
        [FZ.IndInj(face, (2,2); id=:D)], reshape([CM.coerce(field, 1)], 1, 1); field=field)
    P, H, pi = ZE.encode_from_flange(flange)
    enc = RES.EncodingResult(P, TO.pmodule_from_fringe(H), EC.compile_encoding(P, pi))
    inv = TamerOp.invariant(enc; which=:rank_invariant)
    a, b = (5//2, 1//1), (5//2 + 1//(2^53), 1//1)
    spec = V.visual_spec(inv; kind=:rank_query_overlay, pairs=[(a, a), (b, b)])
    @test [r.rank for r in spec.metadata.query_results] == [1, 0]
    @test spec.metadata.exact_query_pairs == [(a, a), (b, b)]
    @test !V.check_visual_request(inv; kind=:rank_query_overlay, pair=(b, a)).valid
    @test V.check_visual_request(inv; kind=:rank_query_overlay, pair=(a, b)).valid
    hilbert = TamerOp.invariant(enc; which=:restricted_hilbert)
    heat = V.visual_spec(hilbert; kind=:hilbert_heatmap)
    @test heat.metadata.geometry.geometry_kind == :nearest_lattice_tiles
    @test occursin("nearest-lattice", heat.subtitle)

    # Unknown geometry and a known zero stalk have distinct displayed colors.
    Q = QQ
    hp = PLP.HPoly(2, Q[-1 0; 0 -1; 1 1], Q[0,0,1], nothing, falses(3), zero(Q))
    partial = PLP.PLEncodingMap(2, [BitVector()], [BitVector()], [hp], [(1//4,1//4)])
    cdr = RES.CohomologyDimsResult(chain_poset(1), [0], partial; degree=0)
    for kind in (:cohomology_support, :cohomology_support_plane)
        s = V.visual_spec(cdr; kind=kind, box=([-1,-1], [2,2]))
        background = first(s.layers)
        fill = only(filter(l -> l isa V.PolygonLayer, s.layers))
        @test background.fill_color == s.legend.entries.outside.color
        @test background.fill_color != fill.fill_color
        entry = kind === :cohomology_support ? s.legend.entries.unsupported : s.legend.entries.v1
        @test fill.fill_color == entry.color
    end
    MI = TamerOp.MultiparameterImages
    line = MI.MPPLineSpec([1.0,1.0], 0.0, [0.0,0.0], 0.5)
    view = V.visual_spec(line; box=([-5,-2], [3,7]))
    @test view.axes.xlimits == (-5.0, 3.0)
    @test view.axes.ylimits == (-2.0, 7.0)
end
@testset "A82 contours omit internal cells of the same fiber" begin
    VIZ = TamerOp.Visualization
    PLB = TamerOp.PLBackend
    _, _, pi = PLB.encode_fringe_boxes([PLB.BoxUpset([0.0, 0.0])],
        [PLB.BoxDownset([2.0, 2.0])], TamerOp.Advanced.EncodingOptions())
    spec = VIZ.visual_spec(pi; kind=:regions, box=([-1, -1], [3, 3]))
    segments = [seg for layer in spec.layers if layer isa VIZ.SegmentLayer for seg in layer.segments]
    canonical(seg) = (seg[1], seg[2]) <= (seg[3], seg[4]) ? seg : (seg[3], seg[4], seg[1], seg[2])
    boundaries = Set(canonical.(segments))
    # Below the square neither crossing x=0 nor crossing y=0 changes the
    # classifier signature. Above it x=2 and y=2 also leave the fiber unchanged.
    for internal in ((0.0, -1.0, 0.0, 0.0), (-1.0, 0.0, 0.0, 0.0),
                     (2.0, 2.0, 2.0, 3.0), (2.0, 2.0, 3.0, 2.0))
        @test !(canonical(internal) in boundaries)
    end
    # All four true support boundaries remain, as does the clipped outer box.
    for boundary in ((0.0, 0.0, 2.0, 0.0), (0.0, 0.0, 0.0, 2.0),
                     (0.0, 2.0, 2.0, 2.0), (2.0, 0.0, 2.0, 2.0))
        @test canonical(boundary) in boundaries
    end
    cuts = Set(canonical(seg) for layer in spec.layers if layer isa VIZ.SegmentLayer && layer.linestyle === :dot for seg in layer.segments)
    @test (-1.0, -1.0, 0.0, -1.0) in cuts
    # Rendering simplification must retain the exact cell subdivision for queries.
    @test count(c -> c.dimension == 2, spec.metadata.geometry.components) == 9
end
@testset "A42a finite-poset layout and selection semantics" begin
    V = TamerOp.Visualization
    # Numeric IDs deliberately disagree with topological order: min=4, max=1.
    relation = falses(4, 4)
    for i in 1:4
        relation[i, i] = true
    end
    expected_edges = Set([(4, 2), (4, 3), (2, 1), (3, 1)])
    for (u, v) in expected_edges
        relation[u, v] = true
    end
    relation[4, 1] = true
    P = FF.FinitePoset(relation)
    spec = V.visual_spec(P; kind=:hasse)
    @test :hasse in V.available_visuals(P)
    @test spec.metadata.vertex_ids == collect(1:4)
    @test Set(spec.metadata.cover_edges) == expected_edges
    @test !((4, 1) in spec.metadata.cover_edges)
    @test length(unique(spec.metadata.positions)) == 4
    @test all(spec.metadata.positions[u][2] < spec.metadata.positions[v][2] for (u, v) in expected_edges)
    @test spec.metadata.dimensions === nothing
    @test V.check_visual_spec(spec).valid
    @test V.visual_spec(P; kind=:hasse, pair=(4, 1)).metadata.relation == :comparable
    @test V.visual_spec(P; kind=:hasse, pair=(4, 2)).metadata.relation == :cover
    @test V.visual_spec(P; kind=:hasse, pair=(2, 2)).metadata.relation == :equal
    @test V.visual_spec(P; kind=:hasse, pair=(2, 3)).metadata.relation == :incomparable
    @test V.visual_spec(P; kind=:hasse, pair=(1, 4)).metadata.relation == :reverse_comparable
    @test_throws ArgumentError V.visual_spec(P; kind=:hasse, vertex=0)
    @test_throws ArgumentError V.visual_spec(P; kind=:hasse, pair=(1, 5))
    @test_throws ArgumentError V.visual_spec(P; kind=:hasse, vertex=true)
    @test_throws ArgumentError V.visual_spec(P; kind=:hasse, vertex=1, pair=(1, 2))

    disconnected = V.visual_spec(disjoint_two_chains_poset(); kind=:hasse)
    @test Set(disconnected.metadata.cover_edges) == Set([(1, 2), (3, 4)])
    @test length(unique(disconnected.metadata.positions)) == 4
    empty = V.visual_spec(chain_poset(0); kind=:hasse)
    @test isempty(empty.metadata.vertex_ids)
    @test isempty(empty.metadata.cover_edges)
    @test V.check_visual_spec(empty).valid
    product = V.visual_spec(FF.ProductOfChainsPoset((2, 2)); kind=:hasse)
    @test Set(product.metadata.cover_edges) == Set([(1, 2), (1, 3), (2, 4), (3, 4)])

    for field in FIELDS_FULL
        K = CM.coeff_type(field)
        M = MD.PModule{K}(P, ones(Int, 4),
            Dict(edge => reshape(K[1], 1, 1) for edge in expected_edges); field=field)
        diagram = V.visual_spec(M; kind=:hasse)
        @test diagram.metadata.dimensions == ones(Int, 4)
        direct = V.visual_spec(M; kind=:module_inspector, pair=(4, 1)).metadata.inspection
        @test direct.defined
        @test direct.matrix == reshape(K[1], 1, 1)
        @test direct.rank == 1
        for pair in ((2, 3), (1, 4))
            absent = V.visual_spec(M; kind=:module_inspector, pair).metadata.inspection
            @test !absent.defined
            @test absent.matrix === nothing
            @test absent.rank === nothing
        end
        overview = V.visual_spec(M; kind=:module_inspector).metadata.inspection
        @test overview.kind == :overview
        @test V.visual_spec(M; kind=:module_inspector, vertex=4).metadata.inspection.kind == :stalk
    end
end

@testset "A42a selected maps over the supported coefficient fields" begin
    V = TamerOp.Visualization
    P = chain_poset(3)
    for field in FIELDS_FULL
        K = CM.coeff_type(field)
        injection = reshape(K[1, 0], 2, 1)
        projection = K[0 1]
        M = MD.PModule{K}(P, [1, 2, 1],
            Dict((1, 2) => injection, (2, 3) => projection); field=field)
        @test V.visual_spec(M).kind == :hasse
        @test Set(V.available_visuals(M)) == Set((:hasse, :module_inspector))
        cases = [
            (pair=(1, 2), matrix=injection, rank=1, nullity=0),
            (pair=(2, 3), matrix=projection, rank=1, nullity=1),
            (pair=(1, 3), matrix=zeros(K, 1, 1), rank=0, nullity=1),
            (pair=(2, 2), matrix=K[1 0; 0 1], rank=2, nullity=0),
        ]
        for c in cases
            spec = V.visual_spec(M; kind=:module_inspector, pair=c.pair)
            selected = spec.metadata.inspection
            @test selected.kind == :map
            @test selected.defined
            @test (selected.source, selected.target) == c.pair
            @test selected.matrix == c.matrix
            @test eltype(selected.matrix) == K
            @test selected.rank == selected.image_dimension == c.rank
            @test selected.kernel_dimension == c.nullity
            @test selected.source_dimension == size(c.matrix, 2)
            @test selected.target_dimension == size(c.matrix, 1)
            @test selected.basis_convention == :module_coordinates
            @test selected.field == field
            @test selected.exact == !(field isa CM.RealField)
            table = only(l for l in last(spec.panels).layers if l isa V.MatrixLayer)
            @test size(table.entries) == size(c.matrix)
            @test length(table.row_labels) == size(c.matrix, 1)
            @test length(table.column_labels) == size(c.matrix, 2)
            @test V.check_visual_spec(spec).valid
        end
        # Two nonzero adjacent maps have zero composite: dimensions alone
        # cannot recover this distinction. The matrix is a safe snapshot.
        selected = V.visual_spec(M; kind=:module_inspector, pair=(1, 2)).metadata.inspection
        selected.matrix[1, 1] = zero(K)
        @test MD.structure_map(M; source=1, target=2) == injection

        A = K[1 1; 1 -1]
        N = MD.PModule{K}(chain_poset(2), [2, 2], Dict((1, 2) => A); field=field)
        # det(A)=-2: only characteristic 2 drops rank, independently of the
        # numerical/exact rank implementation used by the visualization.
        expected_rank = field isa CM.PrimeField && field.p == 2 ? 1 : 2
        char_spec = V.visual_spec(N; kind=:module_inspector, pair=(1, 2))
        @test char_spec.metadata.inspection.rank == expected_rank
        @test char_spec.metadata.inspection.kernel_dimension == 2 - expected_rank
        char_table = only(l for l in last(char_spec.panels).layers if l isa V.MatrixLayer)
        @test char_table.entries == (field isa CM.QQField ? string.(numerator.(A)) : string.(A))
    end

    # Exact rational coefficients remain exact in the retained matrix and text.
    field = CM.QQField()
    A = reshape(QQ[1//3], 1, 1)
    M = MD.PModule{QQ}(chain_poset(2), [1, 1], Dict((1, 2) => A); field=field)
    spec = V.visual_spec(M; kind=:module_inspector, pair=(1, 2))
    @test spec.metadata.inspection.matrix == A
    table = only(l for l in last(spec.panels).layers if l isa V.MatrixLayer)
    @test table.entries == reshape(["1/3"], 1, 1)

    # Rank follows the declared RealField tolerance, not Float64's default.
    for (atol, expected_rank) in ((1e-6, 1), (1e-10, 2))
        real_field = CM.RealField(Float64; rtol=0.0, atol)
        N = MD.PModule{Float64}(chain_poset(2), [2, 2],
            Dict((1, 2) => [1.0 0.0; 0.0 1e-8]); field=real_field)
        numerical = V.visual_spec(N; kind=:module_inspector, pair=(1, 2)).metadata.inspection
        @test numerical.rank == expected_rank
        @test numerical.kernel_dimension == 2 - expected_rank
        @test !numerical.exact
        @test numerical.atol == atol
        @test numerical.rtol == 0.0
    end
end

@testset "A42a empty maps and matrix table limits" begin
    V = TamerOp.Visualization
    for field in FIELDS_FULL
        K = CM.coeff_type(field)
        M = MD.PModule{K}(chain_poset(3), [0, 2, 0], Dict(
            (1, 2) => zeros(K, 2, 0), (2, 3) => zeros(K, 0, 2)); field=field)
        for (pair, shape, nullity) in (((1, 2), (2, 0), 0),
                                      ((2, 3), (0, 2), 2),
                                      ((1, 3), (0, 0), 0),
                                      ((1, 1), (0, 0), 0))
            spec = V.visual_spec(M; kind=:module_inspector, pair)
            selected = spec.metadata.inspection
            @test selected.defined
            @test size(selected.matrix) == shape
            @test selected.rank == 0
            @test selected.kernel_dimension == nullity
            table = only(l for l in last(spec.panels).layers if l isa V.MatrixLayer)
            @test size(table.entries) == shape
            @test V.check_visual_spec(spec).valid
        end
        empty = MD.PModule{K}(chain_poset(0), Int[], Dict{Tuple{Int,Int},Matrix{K}}(); field=field)
        @test isempty(V.visual_spec(empty; kind=:hasse).metadata.dimensions)
        @test V.visual_spec(empty; kind=:module_inspector).metadata.inspection.kind == :overview
    end
    A = zeros(QQ, 14, 15)
    for i in 1:14
        A[i, i] = 1
    end
    M = MD.PModule{QQ}(chain_poset(2), [15, 14], Dict((1, 2) => A); field=CM.QQField())
    spec = V.visual_spec(M; kind=:module_inspector, pair=(1, 2), matrix_limit=(3, 4))
    selected = spec.metadata.inspection
    @test selected.matrix == A
    @test selected.rank == 14
    @test selected.kernel_dimension == 1
    @test selected.displayed_rows == collect(1:3)
    @test selected.displayed_columns == collect(1:4)
    @test selected.truncated
    table = only(l for l in last(spec.panels).layers if l isa V.MatrixLayer)
    @test table.entries == string.(numerator.(A[1:3, 1:4]))
    @test_throws ArgumentError V.visual_spec(M; kind=:module_inspector, pair=(1, 2), matrix_limit=(0, 4))
    @test_throws ArgumentError V.visual_spec(M; kind=:hasse, matrix_limit=(3, 4))
end

@testset "A42a square parameters, stalks, and actual labels" begin
    V = TamerOp.Visualization
    opts = TOA.EncodingOptions(; backend=:pl_backend, poset_kind=:signature, field=CM.QQField())
    enc = TamerOp.encode([TOA.BoxUpset([0.0, 0.0])],
        [TOA.BoxDownset([2.0, 2.0])], reshape(QQ[1], 1, 1), opts)
    pi = TamerOp.encoding_map(enc)
    expected_dims = TamerOp.dimensions(enc)
    @test :hasse in V.available_visuals(enc)
    @test :module_inspector in V.available_visuals(enc)
    for point in ((0, 0), (2, 2), (0, 2), (2, 0), (-1, 1), (3, 1))
        spec = V.visual_spec(enc; kind=:module_inspector, point, box=([-1, -1], [3, 3]))
        selected = spec.metadata.inspection
        @test selected.kind == :stalk
        hasse = only(p for p in spec.panels if p.kind == :hasse)
        @test hasse.metadata.selected_vertex == EC.locate(pi, point)
        @test hasse.metadata.dimensions == expected_dims
        @test expected_dims[hasse.metadata.selected_vertex] == Int(all(0 .<= collect(point) .<= 2))
        @test first(spec.panels).axes.xlimits == (-1.0, 3.0)
        @test first(spec.panels).axes.ylimits == (-1.0, 3.0)
        @test V.check_visual_spec(spec).valid
    end
    cases = [
        (points=((1//4, 1//2), (1, 3//2)), matrix=reshape(QQ[1], 1, 1)),
        (points=((-1, 1), (0, 1)), matrix=zeros(QQ, 1, 0)),
        (points=((1, 3//2), (3, 3//2)), matrix=zeros(QQ, 0, 1)),
        (points=((-1, 1), (3, 1)), matrix=zeros(QQ, 0, 0)),
    ]
    for c in cases
        spec = V.visual_spec(enc; kind=:module_inspector, parameter_pair=c.points)
        selected = spec.metadata.inspection
        @test selected.defined
        @test selected.source == EC.locate(pi, collect(c.points[1]))
        @test selected.target == EC.locate(pi, collect(c.points[2]))
        @test selected.matrix == c.matrix
        @test size(selected.matrix) == size(c.matrix)
    end
    # A shared label does not turn incomparable ambient points into a map.
    a, b = (1//4, 3//2), (3//2, 1//4)
    label = EC.locate(pi, a)
    @test EC.locate(pi, b) == label
    incomparable = V.visual_spec(enc; kind=:module_inspector, parameter_pair=(a, b)).metadata.inspection
    @test !incomparable.defined
    @test incomparable.matrix === nothing
    @test V.visual_spec(enc; kind=:module_inspector, pair=(label, label)).metadata.inspection.matrix == reshape(QQ[1], 1, 1)
    @test_throws ArgumentError V.visual_spec(enc; kind=:module_inspector, point=a, pair=(label, label))
    @test_throws ArgumentError V.visual_spec(enc; kind=:module_inspector, point=(NaN, 0))
    @test !V.check_visual_request(enc; kind=:module_inspector, point=(big(10)^400, 0)).valid

    # Exterior points are zero-dimensional represented stalks, not missing
    # classifier regions. The outside status occurs only where locate is 0.
    P = FF.ProductOfChainsPoset((2, 2))
    grid = EC.GridEncodingMap(P, ([0, 2], [1, 4]); orientation=(-1, 1))
    constant_module = MD.PModule{QQ}(P, ones(Int, 4),
        Dict(edge => reshape(QQ[1], 1, 1) for edge in FF.cover_edges(P)); field=CM.QQField())
    grid_enc = RES.EncodingResult(P, constant_module, EC.compile_encoding(P, grid))
    @test V.visual_spec(grid_enc; kind=:module_inspector, point=(1, 1)).metadata.inspection.kind == :outside
    extreme = V.visual_spec(grid_enc; kind=:module_inspector, point=(typemin(Int), 1))
    @test extreme.metadata.inspection.vertex == 2
    @test extreme.metadata.inspection.dimension == 1
    @test V.visual_spec(grid_enc; kind=:module_inspector,
        point=(typemax(UInt), UInt(1))).metadata.inspection.kind == :outside
    # Ambient orientation (-1,+1): decreasing x, increasing y is forward.
    forward = V.visual_spec(grid_enc; kind=:module_inspector,
        parameter_pair=((0, 1), (-2, 4))).metadata.inspection
    @test forward.defined && forward.matrix == reshape(QQ[1], 1, 1)
    backward = V.visual_spec(grid_enc; kind=:module_inspector,
        parameter_pair=((-2, 4), (0, 1))).metadata.inspection
    @test !backward.defined && backward.matrix === nothing
end

@testset "A42a overlapping squares distinguish maps from dimensions" begin
    V = TamerOp.Visualization
    opts = TOA.EncodingOptions(; backend=:pl_backend, poset_kind=:signature, field=CM.QQField())
    enc = TamerOp.encode([TOA.BoxUpset([0.0, 0.0]), TOA.BoxUpset([1.0, 1.0])],
        [TOA.BoxDownset([2.0, 2.0]), TOA.BoxDownset([3.0, 3.0])], QQ[1 0; 0 1], opts)
    a, b, c = (1//2, 1//2), (3//2, 3//2), (5//2, 5//2)
    left = V.visual_spec(enc; kind=:module_inspector, parameter_pair=(a, b)).metadata.inspection
    right = V.visual_spec(enc; kind=:module_inspector, parameter_pair=(b, c)).metadata.inspection
    full = V.visual_spec(enc; kind=:module_inspector, parameter_pair=(a, c)).metadata.inspection
    @test (left.source_dimension, left.target_dimension, right.target_dimension) == (1, 2, 1)
    @test (left.rank, right.rank, full.rank) == (1, 1, 0)
    @test any(!iszero, left.matrix)
    @test any(!iszero, right.matrix)
    @test full.matrix == zeros(QQ, 1, 1)
    @test right.matrix * left.matrix == full.matrix
    mixed = V.visual_spec(enc; kind=:module_inspector, point=(1//2, 5//2))
    hasse = only(p for p in mixed.panels if p.kind == :hasse)
    @test hasse.metadata.dimensions[hasse.metadata.selected_vertex] == 0
end

@testset "A42a lazy module inspection delays map construction" begin
    V = TamerOp.Visualization
    # Three edges form a cycle at (1,0), filled by the triangle at (2,1).
    # H1 therefore has dimensions [0,1,1] at y=0 and [0,1,0] at y=1.
    boundary1 = sparse([-1 -1 0; 1 0 -1; 0 1 1])
    boundary2 = sparse(reshape([1, -1, 1], 3, 1))
    graded = DT.GradedComplex([Int[1, 2, 3], Int[1, 2, 3], Int[1]],
        [boundary1, boundary2],
        [(0., 0.), (0., 0.), (0., 0.), (1., 0.), (1., 0.), (1., 0.), (2., 1.)])
    filtration = OPT.FiltrationSpec(kind=:graded, axes=([0., 1., 2.], [0., 1.]))
    enc = TamerOp.encode(graded, filtration; degree=1, field=CM.QQField(), stage=:encoding_result)
    @test enc.M isa DI._LazyEncodedModule
    @test enc.M.cached_module === nothing && enc.M.dims === nothing
    @test :hasse in V.available_visuals(enc)
    @test V.check_visual_request(enc; kind=:hasse).valid
    @test V.check_visual_request(enc; kind=:module_inspector, pair=(2, 5)).valid
    @test enc.M.cached_module === nothing && enc.M.dims === nothing
    h = V.visual_spec(enc; kind=:hasse)
    @test h.metadata.dimensions == [0, 1, 1, 0, 1, 0]
    @test enc.M.cached_module === nothing
    @test V.visual_spec(enc; kind=:module_inspector).metadata.inspection.kind == :overview
    @test enc.M.cached_module === nothing
    @test V.visual_spec(enc; kind=:module_inspector, vertex=2).metadata.inspection.kind == :stalk
    @test enc.M.cached_module === nothing
    absent = V.visual_spec(enc; kind=:module_inspector, pair=(5, 2)).metadata.inspection
    @test !absent.defined && absent.matrix === nothing
    @test enc.M.cached_module === nothing
    selected = V.visual_spec(enc; kind=:module_inspector, pair=(2, 5)).metadata.inspection
    @test selected.defined && selected.rank == 1
    @test enc.M.cached_module !== nothing
end

@testset "A42a matrix layers and native inspection figures" begin
    V = TamerOp.Visualization
    @test TOA.MatrixLayer === V.MatrixLayer
    malformed = V.VisualizationSpec(:matrix_test; layers=V.AbstractVisualizationLayer[
        V.MatrixLayer(reshape(["1"], 1, 1), String[], ["source"] )])
    @test !V.check_visual_spec(malformed).valid
    mixed = V.VisualizationSpec(:matrix_test; layers=V.AbstractVisualizationLayer[
        V.MatrixLayer(reshape(["1"], 1, 1), ["target"], ["source"]),
        V.TextLayer(["ignored?"], [(0.0, 0.0)], :black, 12.0)])
    @test !V.check_visual_spec(mixed).valid
    for shape in ((-1, 1), (true, 1), (1,), (0, 1), [1, 1])
        bad_shape = V.VisualizationSpec(:matrix_test; layers=V.AbstractVisualizationLayer[
            V.MatrixLayer(reshape(["1"], 1, 1), ["target"], ["source"])],
            metadata=(; matrix_size=shape))
        @test !V.check_visual_spec(bad_shape).valid
    end
    M = MD.PModule{QQ}(chain_poset(2), [1, 1],
        Dict((1, 2) => reshape(QQ[1//3], 1, 1)); field=CM.QQField())
    inspector = V.visual_spec(M; kind=:module_inspector, pair=(1, 2))
    @test !inspector.interaction.hover && !inspector.interaction.clicks
    @test V.check_visual_request(M; kind=:hasse).construction_cost.map_queries == :none
    for backend in (:cairomakie, :wglmakie)
        name = backend === :cairomakie ? "CairoMakie" : "WGLMakie"
        if Base.find_package(name) === nothing
            @test_skip false
            continue
        end
        if backend === :cairomakie
            @eval import CairoMakie
            makie = CairoMakie.Makie
        else
            @eval import WGLMakie
            makie = WGLMakie.Makie
        end
        fig = V.render(inspector; backend, size=(1000, 600))
        makie.update_state_before_display!(fig)
        labels = [String(item.text[]) for item in fig.content if item isa makie.Label]
        @test "1/3" in labels
        @test "e1 @ 1 [source]" in labels
        @test "e1 @ 2 [target]" in labels
        @test "target / source" in labels
        ax = only(item for item in fig.content if item isa makie.Axis)
        @test !ax.xticklabelsvisible[] && !ax.yticklabelsvisible[]
        @test makie.widths(makie.viewport(ax.scene)[])[2] > 250
        @test Tuple(makie.widths(makie.viewport(fig.scene)[])) == (1000, 600)
        for shape in ((0, 0), (0, 2), (2, 0))
            empty = V.VisualizationSpec(:matrix_test; layers=V.AbstractVisualizationLayer[
                V.MatrixLayer(fill("0", shape...), ["t$i" for i in 1:shape[1]],
                    ["s$j" for j in 1:shape[2]])], metadata=(; matrix_size=shape))
            empty_fig = V.render(empty; backend)
            @test any(item -> item isa makie.Label &&
                occursin("Empty matrix ($(shape[1]) x $(shape[2]))", item.text[]), empty_fig.content)
        end
        mktempdir() do dir
            if backend === :cairomakie
                path = joinpath(dir, "inspector.svg")
                V.save_visual(path, inspector; backend)
                @test occursin("<svg", read(path, String))
                @test filesize(path) > 1000
            else
                path = joinpath(dir, "inspector.html")
                V.save_visual(path, inspector; backend)
                @test occursin("html", lowercase(read(path, String)))
                @test filesize(path) > 1000
            end
        end
    end
end

@testset "A83 retained witness discovery and presentation request contracts" begin
    help_text = sprint(show, MIME"text/plain"(), Base.Docs.doc(TamerOp.Visualization.visual_spec))
    @test occursin(":module_inspector", help_text)
    @test occursin(":presentation_inspector", help_text)
    V, IR = TamerOp.Visualization, TamerOp.IndicatorResolutions
    P = chain_poset(3)
    H = FF.FringeModule{QQ}(P,
        [FF.Upset(P, trues(3)), FF.Upset(P, BitVector([false, true, true]))],
        [FF.Downset(P, BitVector([true, true, false])), FF.Downset(P, trues(3))],
        QQ[1 0; 0 1]; field=CM.QQField())
    M = TOA.pmodule_from_fringe(H)
    enc = RES.EncodingResult(P, M, nothing; H)
    @test TamerOp.encoding_presentation(enc) === H
    @test V.available_visuals(H) == (:presentation_inspector,)
    @test V.available_visuals(enc) == (:hasse, :module_inspector, :presentation_inspector)
    @test !(:presentation_inspector in V.available_visuals(M))
    @test V.visual_spec(H).kind == :presentation_inspector
    before = (H.fiber_queries[], H.fiber_dims[])
    overview = V.visual_spec(enc; kind=:presentation_inspector)
    @test isempty(overview.metadata.stalks)
    @test overview.metadata.presentation_map === nothing
    @test (H.fiber_queries[], H.fiber_dims[]) == before
    @test overview.metadata.selected_fibers_only
    @test overview.metadata.basis_convention == :embedded_presentation_image
    @test !overview.interaction.clicks && !overview.interaction.hover
    @test V.check_visual_spec(overview).valid

    otherP = chain_poset(3)
    wrong_base = FF.FringeModule{QQ}(otherP, [FF.Upset(otherP, trues(3))],
        [FF.Downset(otherP, trues(3))], reshape(QQ[1], 1, 1); field=CM.QQField())
    wrong_field = FF.change_field(H, CM.PrimeField(3))
    for absent in (
        RES.EncodingResult(P, M, nothing),
        RES.EncodingResult(P, M, nothing; presentation=H),
        RES.EncodingResult(P, M, nothing; H=(historical=H,)),
        RES.EncodingResult(P, M, nothing; H=wrong_base),
        RES.EncodingResult(P, M, nothing; H=wrong_field),
        CM.change_field(enc, CM.PrimeField(3)),
    )
        @test TamerOp.encoding_presentation(absent) === nothing
        @test !(:presentation_inspector in V.available_visuals(absent))
        @test !V.check_visual_request(absent; kind=:presentation_inspector).valid
        @test_throws ArgumentError V.visual_spec(absent; kind=:presentation_inspector)
    end
    @test V.check_visual_request(H; vertex=2, basis=false).valid
    @test V.check_visual_request(H; vertex=2, basis=true, upset=2, downset=1).valid
    for options in (
        (; basis=true), (; basis=false), (; vertex=2, basis=1),
        (; pair=(1, 2), basis=true), (; pair=(1, 2), basis=false),
        (; vertex=0), (; vertex=true), (; vertex=4), (; pair=(1, 4)),
        (; vertex=1, pair=(1, 2)), (; upset=0), (; downset=3), (; upset=true),
        (; matrix_limit=(0, 2)), (; point=(0, 0)), (; box=([0, 0], [1, 1])),
        (; hover=true),
    )
        @test !V.check_visual_request(H; options...).valid
        @test_throws ArgumentError V.visual_spec(H; options...)
    end
end

@testset "A83 active zero coefficients and exact support membership" begin
    V, IR = TamerOp.Visualization, TamerOp.IndicatorResolutions
    opts = TOA.EncodingOptions(; backend=:pl_backend, poset_kind=:signature, field=CM.QQField())
    enc = TamerOp.encode([TOA.BoxUpset([0.0, 0.0]), TOA.BoxUpset([1.0, 1.0])],
        [TOA.BoxDownset([2.0, 2.0]), TOA.BoxDownset([3.0, 3.0])], QQ[1 0; 0 1], opts)
    pi = TamerOp.encoding_map(enc)
    @test V.available_visuals(enc) == (:regions, :region_labels, :query_overlay,
        :hasse, :module_inspector, :presentation_inspector)
    t = (1//2, 5//2)
    qt = EC.locate(pi, t)
    spec = V.visual_spec(enc; kind=:presentation_inspector, point=t,
        basis=true, upset=1, downset=2, box=([-1, -1], [4, 4]))
    s = only(spec.metadata.stalks)
    @test spec.metadata.relation == :stalk
    @test IR.active_rows(s) == [2]
    @test IR.active_columns(s) == [1]
    @test IR.presentation_matrix(s) == reshape(QQ[0], 1, 1)
    @test IR.presentation_summary(s).dimension == 0
    @test size(IR.image_basis(s)) == (1, 0)
    @test IR.presentation_vertex(s) == qt
    full = only(p for p in spec.panels if p.title == "Full coefficient matrix Phi")
    @test full.metadata.matrix == QQ[1 0; 0 1]
    @test only(full.layers).entries == ["1" "0"; "0" "1"]
    @test full.metadata.row_roles == [:inactive, :selected]
    @test full.metadata.column_roles == [:selected, :inactive]
    @test full.metadata.cell_roles == [:inactive :inactive; :selected :inactive]
    block = only(p for p in spec.panels if p.title == "Selected stalk: active block")
    @test only(block.layers).row_labels == ["D2"]
    @test only(block.layers).column_labels == ["U1"]
    @test only(block.layers).entries == reshape(["0"], 1, 1)
    basis_panel = only(p for p in spec.panels if p.title == "Selected stalk: embedded image basis")
    @test basis_panel.metadata.matrix_size == (1, 0)
    @test size(only(basis_panel.layers).entries) == (1, 0)
    supports = [p for p in spec.panels if p.kind == :presentation_support]
    @test length(supports) == 2
    @test [(p.metadata.family, p.metadata.support_id) for p in supports] == [(:upset, 1), (:downset, 2)]
    @test supports[1].metadata.geometry === supports[2].metadata.geometry
    for panel in supports
        @test panel.axes.xlimits == (-1.0, 4.0)
        @test panel.axes.ylimits == (-1.0, 4.0)
        @test panel.metadata.membership[qt]
        @test only(panel.metadata.query_readout).point == t
        @test only(panel.metadata.query_readout).region_id == qt
        # Independent geometric oracle, sampled at each displayed polygon's
        # interior centroid: U1 is x,y >= 0; D2 is x,y <= 3.
        for layer in panel.layers
            layer isa V.PolygonLayer || continue
            for polygon in layer.polygons
                x = sum(first, polygon) / length(polygon)
                y = sum(last, polygon) / length(polygon)
                member = panel.metadata.family === :upset ? x >= 0 && y >= 0 : x <= 3 && y <= 3
                @test layer.fill_color isa V._VisualRole
                @test layer.fill_color.kind == (member ? :support : :absent)
            end
        end
    end
    @test V.check_visual_spec(spec).valid
    rank_only = V.visual_spec(enc; kind=:presentation_inspector, vertex=qt)
    @test IR.image_basis(only(rank_only.metadata.stalks)) === nothing
    @test !IR.presentation_summary(only(rank_only.metadata.stalks)).basis_available
    other_supports = V.visual_spec(enc; kind=:presentation_inspector, point=t, upset=2, downset=1)
    @test IR.presentation_matrix(only(other_supports.metadata.stalks)) == reshape(QQ[0], 1, 1)
    @test !other_supports.panels[1].metadata.membership[qt]
    @test !other_supports.panels[2].metadata.membership[qt]

    # Classification retains exact boundary distinctions despite coincident
    # drawing coordinates. The second square remains present past x=2.
    for (point, dim) in (((2//1, 3//2), 2), ((2 + 1//(2^53), 3//2), 1))
        exact = V.visual_spec(enc; kind=:presentation_inspector, point)
        @test IR.presentation_summary(only(exact.metadata.stalks)).dimension == dim
        @test only(exact.panels[1].metadata.query_readout).point == point
    end
    rounded = V.visual_spec(enc; kind=:presentation_inspector, point=(2 + 1//(2^53), 3//2))
    @test !isempty(rounded.panels[1].metadata.warnings)
    @test !V.check_visual_request(enc; kind=:presentation_inspector, point=t, pair=(qt, qt)).valid
end

@testset "A83 induced maps, finite membership tables, and display limits" begin
    V, IR = TamerOp.Visualization, TamerOp.IndicatorResolutions
    P = chain_poset(3)
    H = FF.FringeModule{QQ}(P,
        [FF.Upset(P, trues(3)), FF.Upset(P, BitVector([false, true, true]))],
        [FF.Downset(P, BitVector([true, true, false])), FF.Downset(P, trues(3))],
        QQ[1 0; 0 1]; field=CM.QQField())
    expected = (((1, 2), reshape(QQ[1, 0], 2, 1)),
                ((2, 3), QQ[0 1]), ((1, 3), zeros(QQ, 1, 1)),
                ((2, 2), QQ[1 0; 0 1]))
    maps = Dict{Tuple{Int,Int},Any}()
    for (pair, matrix) in expected
        spec = V.visual_spec(H; pair)
        m = spec.metadata.presentation_map
        maps[pair] = m
        @test spec.metadata.defined
        @test IR.induced_map(m) == matrix
        @test IR.image_basis(IR.target_stalk(m)) * IR.induced_map(m) ==
            IR.ambient_projection(m) * IR.image_basis(IR.source_stalk(m))
        @test all(IR.presentation_summary(s).basis_available for s in spec.metadata.stalks)
        @test V.check_visual_spec(spec).valid
        @test all(p.kind != :presentation_support for p in spec.panels)
    end
    @test IR.induced_map(maps[(2, 3)]) * IR.induced_map(maps[(1, 2)]) == IR.induced_map(maps[(1, 3)])
    colored = V.visual_spec(H; pair=(1, 3))
    full = only(p for p in colored.panels if p.title == "Full coefficient matrix Phi")
    @test full.metadata.cell_roles == [:source :inactive; :both :target]
    @test full.metadata.row_roles == [:source, :both]
    @test full.metadata.column_roles == [:both, :target]

    limited = V.visual_spec(H; vertex=3, basis=true, upset=2, downset=1, matrix_limit=(1, 2))
    up, down = limited.panels[1:2]
    @test up.metadata.family == :upset && up.metadata.support_id == 2
    @test down.metadata.family == :downset && down.metadata.support_id == 1
    @test up.metadata.matrix == reshape([0, 1, 1], 1, 3)
    @test down.metadata.matrix == reshape([1, 1, 0], 1, 3)
    @test up.metadata.membership == [false, true, true]
    @test down.metadata.membership == [true, true, false]
    @test only(up.layers).column_labels == ["q1", "q2"]
    @test only(up.layers).entries == ["0" "1"]
    @test up.metadata.truncated && down.metadata.truncated
    @test up.metadata.matrix_size == (1, 3)
    @test up.metadata.displayed_columns == 1:2
    limited_full = only(p for p in limited.panels if p.title == "Full coefficient matrix Phi")
    @test size(limited_full.metadata.matrix) == (2, 2)
    @test size(only(limited_full.layers).entries) == (1, 2)
    @test size(limited_full.metadata.cell_roles) == (1, 2)
    @test length(limited_full.metadata.row_roles) == 1
    @test length(limited_full.metadata.column_roles) == 2
    active = only(p for p in limited.panels if p.title == "Selected stalk: active block")
    @test only(active.layers).row_labels == ["D2"]
    @test only(active.layers).entries == ["0" "1"]
    @test IR.presentation_summary(only(limited.metadata.stalks)).dimension == 1
    @test V.check_visual_spec(limited).valid

    empty = FF.FringeModule{QQ}(P, FF.Upset[], FF.Downset[], zeros(QQ, 0, 0); field=CM.QQField())
    empty_spec = V.visual_spec(empty; vertex=1, basis=true)
    @test IR.presentation_summary(only(empty_spec.metadata.stalks)).dimension == 0
    @test size(IR.image_basis(only(empty_spec.metadata.stalks))) == (0, 0)
    @test V.check_visual_spec(empty_spec).valid
    @test !V.check_visual_request(empty; vertex=1, upset=1).valid
    @test !V.check_visual_request(empty; vertex=1, downset=1).valid

    for field in FIELDS_FULL
        K = CM.coeff_type(field)
        scalar = reshape(K[-1], 1, 1)
        constant = FF.FringeModule{K}(P, [FF.Upset(P, trues(3))],
            [FF.Downset(P, trues(3))], scalar; field)
        coefficient_spec = V.visual_spec(constant; vertex=2)
        full = only(p for p in coefficient_spec.panels if p.title == "Full coefficient matrix Phi")
        expected_text = field isa CM.QQField ? "-1" : string(only(scalar))
        @test only(only(full.layers).entries) == expected_text
        @test eltype(full.metadata.matrix) == K
        @test IR.presentation_summary(only(coefficient_spec.metadata.stalks)).dimension == 1
    end
end

@testset "A83 no map for unordered or unrepresented parameters" begin
    V, IR = TamerOp.Visualization, TamerOp.IndicatorResolutions
    opts = TOA.EncodingOptions(; backend=:pl_backend, poset_kind=:signature, field=CM.QQField())
    enc = TamerOp.encode([TOA.BoxUpset([0.0, 0.0])],
        [TOA.BoxDownset([2.0, 2.0])], reshape(QQ[1], 1, 1), opts)
    a, b = (1//4, 3//2), (3//2, 1//4)
    q = EC.locate(TamerOp.encoding_map(enc), a)
    @test EC.locate(TamerOp.encoding_map(enc), b) == q
    absent = V.visual_spec(enc; kind=:presentation_inspector, parameter_pair=(a, b))
    @test absent.metadata.relation == :incomparable
    @test !absent.metadata.defined && absent.metadata.presentation_map === nothing
    @test all(IR.image_basis(s) === nothing for s in absent.metadata.stalks)
    finite_identity = V.visual_spec(enc; kind=:presentation_inspector, pair=(q, q))
    @test finite_identity.metadata.relation == :equal
    @test IR.induced_map(finite_identity.metadata.presentation_map) == reshape(QQ[1], 1, 1)

    P = FF.ProductOfChainsPoset((2, 2))
    grid = EC.GridEncodingMap(P, ([0, 2], [1, 4]); orientation=(-1, 1))
    H = FF.FringeModule{QQ}(P, [FF.Upset(P, trues(4))],
        [FF.Downset(P, trues(4))], reshape(QQ[1], 1, 1); field=CM.QQField())
    grid_enc = RES.EncodingResult(P, TOA.pmodule_from_fringe(H), EC.compile_encoding(P, grid); H)
    outside = V.visual_spec(grid_enc; kind=:presentation_inspector, point=(1, 1))
    @test outside.metadata.relation == :outside
    @test only(outside.metadata.stalks) === nothing
    @test only(outside.panels[1].metadata.query_readout).region_id == 0
    for (points, relation) in ((((1, 1), (0, 1)), :outside), (((-2, 4), (0, 1)), :reverse_comparable))
        missing = V.visual_spec(grid_enc; kind=:presentation_inspector, parameter_pair=points)
        @test missing.metadata.relation == relation
        @test !missing.metadata.defined && missing.metadata.presentation_map === nothing
        @test all(s === nothing || IR.image_basis(s) === nothing for s in missing.metadata.stalks)
        @test V.check_visual_spec(missing).valid
    end
    forward = V.visual_spec(grid_enc; kind=:presentation_inspector, parameter_pair=((0, 1), (-2, 4)))
    @test forward.metadata.defined
    @test IR.induced_map(forward.metadata.presentation_map) == reshape(QQ[1], 1, 1)
    @test forward.metadata.relation == :comparable
    unordered_labels = V.visual_spec(H; pair=(2, 3))
    @test unordered_labels.metadata.relation == :incomparable
    @test unordered_labels.metadata.presentation_map === nothing
end

@testset "A83 native presentation figures and active coefficient text" begin
    V = TamerOp.Visualization
    for metadata in ((; row_colors=[:red, :blue]), (; cell_colors=fill(:red, 2, 1)),
                     (; empty_matrix_reason=false), (; matrix_row_heading=1))
        malformed = V.VisualizationSpec(:matrix_test;
            layers=V.AbstractVisualizationLayer[V.MatrixLayer(reshape(["0"], 1, 1), ["D1"], ["U1"])],
            metadata)
        @test !V.check_visual_spec(malformed).valid
        @test_throws ArgumentError V.check_visual_spec(malformed; throw=true)
    end
    opts = TOA.EncodingOptions(; backend=:pl_backend, poset_kind=:signature, field=CM.QQField())
    enc = TamerOp.encode([TOA.BoxUpset([0.0, 0.0]), TOA.BoxUpset([1.0, 1.0])],
        [TOA.BoxDownset([2.0, 2.0]), TOA.BoxDownset([3.0, 3.0])], QQ[1 0; 0 1], opts)
    zero_view = V.visual_spec(enc; kind=:presentation_inspector, point=(1//2, 5//2),
        basis=true, upset=1, downset=2, box=([-1, -1], [4, 4]))
    map_view = V.visual_spec(enc; kind=:presentation_inspector,
        parameter_pair=((3//2, 3//2), (5//2, 5//2)), box=([-1, -1], [4, 4]))
    for backend in (:cairomakie, :wglmakie)
        name = backend === :cairomakie ? "CairoMakie" : "WGLMakie"
        if Base.find_package(name) === nothing
            @test_skip false
            continue
        end
        if backend === :cairomakie
            @eval import CairoMakie
            makie = CairoMakie.Makie
        else
            @eval import WGLMakie
            makie = WGLMakie.Makie
        end
        fig = V.render(zero_view; backend)
        makie.update_state_before_display!(fig)
        labels = [item for item in fig.content if item isa makie.Label]
        texts = [String(item.text[]) for item in labels]
        @test "downset / upset" in texts
        @test any(startswith("D2"), texts) && any(startswith("U1"), texts)
        @test any(occursin("Empty matrix (1 x 0): the image has no basis vectors"), texts)
        @test "Active downset coordinates: D2" in texts
        active_color = makie.to_color(V._visual_color(V.VisualStyle(), V._VisualRole(:selected)))
        @test any(item -> item.text[] == "0" &&
            makie.to_color(item.color[]) == active_color, labels)
        @test any(item -> startswith(item.text[], "D2") &&
            makie.to_color(item.color[]) == active_color, labels)
        axes = [item for item in fig.content if item isa makie.Axis]
        for ax in (item for item in fig.content if item isa makie.Axis)
            isempty(ax.subtitle[]) && continue
            subtitle_plot = only(p for p in ax.blockscene.plots if
                p isa makie.Text && (p.text[] == ax.subtitle[] || p.text[] == [ax.subtitle[]]))
            # Axis title/subtitle text uses markerspace=:data in a campixel blockscene.
            # Its bounding box therefore has the same figure-pixel x coordinates as
            # the panel grid's allocated (suggested) bounding box.
            text_box = makie.boundingbox(subtitle_plot, :data)
            panel_grid = ax.layoutobservables.gridcontent[].parent
            panel_box = panel_grid.layoutobservables.suggestedbbox[]
            tolerance = 2.0  # rounding/font metric tolerance in figure pixels
            @test text_box.origin[1] >= panel_box.origin[1] - tolerance
            @test text_box.origin[1] + text_box.widths[1] <=
                  panel_box.origin[1] + panel_box.widths[1] + tolerance
        end
        @test length(axes) == 2
        @test all(makie.widths(makie.viewport(ax.scene)[])[2] > 150 for ax in axes)
        @test Tuple(makie.widths(makie.viewport(fig.scene)[])) == zero_view.metadata.figure_size
        mapped = V.render(map_view; backend)
        makie.update_state_before_display!(mapped)
        map_texts = [String(item.text[]) for item in mapped.content if item isa makie.Label]
        @test "Induced map C" in map_texts
        @test "Ambient downset projection R" in map_texts
        @test "downset / image basis" in map_texts
        mktempdir() do dir
            ext = backend === :cairomakie ? "svg" : "html"
            path = joinpath(dir, "presentation.$ext")
            V.save_visual(path, zero_view; backend)
            @test filesize(path) > 1000
            @test occursin(ext == "svg" ? "<svg" : "html", lowercase(read(path, String)))
        end
    end
end

@testset "A83 canonical integer and general PL encodings retain inspectable witnesses" begin
    viz = TamerOp.Visualization
    field = CM.QQField()
    face = FZ.Face(2, [false, false])
    flange = FZ.Flange(2,
        [FZ.IndFlat(face, (0, 0); id=:U)],
        [FZ.IndInj(face, (2, 2); id=:D)],
        reshape(QQ[1], 1, 1); field)
    integer_enc = TamerOp.encode(flange; backend=:zn)

    # This band is genuinely non-axis-aligned, so the PL test cannot pass by
    # silently taking the box encoder: U={x+y>=0}, D={x+y<=2}.
    up = PLP.PLUpset(PLP.poly_union(PLP.make_hpoly(QQ[-1 -1], QQ[0])))
    down = PLP.PLDownset(PLP.poly_union(PLP.make_hpoly(QQ[1 1], QQ[2])))
    pl_fringe = PLP.PLFringe([up], [down], reshape(QQ[1], 1, 1))
    pl_enc = TamerOp.encode(pl_fringe,
        TOA.EncodingOptions(; backend=:pl, field))

    # Both input modules are one-dimensional at (0,0) and (1,1), and vanish
    # at (3,3). The PL endpoint (1,1) is on the closed death boundary.
    for (enc, backend) in ((integer_enc, :zn), (pl_enc, :pl))
        @test RES.result_summary(enc).backend == backend
        witness = TamerOp.encoding_presentation(enc)
        @test witness isa FF.FringeModule
        @test FF.ambient_poset(witness) === TamerOp.encoding_poset(enc)
        @test :presentation_inspector in TamerOp.available_visuals(enc)
        classifier = TamerOp.encoding_map(enc)
        source, target, outside_support = (0//1, 0//1), (1//1, 1//1), (3//1, 3//1)
        # Integral representatives obey both owners' low-level query contracts;
        # the visual requests below separately exercise exact rational points.
        source_id = EC.locate(classifier, [0, 0])
        target_id = EC.locate(classifier, [1, 1])
        outside_id = EC.locate(classifier, [3, 3])
        @test all(>(0), (source_id, target_id, outside_id))

        stalk_spec = TOA.visual_spec(enc; kind=:presentation_inspector,
            point=target, basis=true, box=([-1, -1], [4, 4]))
        stalk = only(stalk_spec.metadata.stalks)
        @test IR.presentation_vertex(stalk) == target_id
        @test IR.active_rows(stalk) == [1]
        @test IR.active_columns(stalk) == [1]
        @test IR.presentation_matrix(stalk) == reshape(QQ[1], 1, 1)
        @test IR.presentation_summary(stalk).dimension == 1
        @test IR.image_basis(stalk) == reshape(QQ[1], 1, 1)
        support_panels = filter(p -> p.kind == :presentation_support, stalk_spec.panels)
        @test length(support_panels) == 2
        @test all(p -> p.metadata.membership[target_id], support_panels)
        @test support_panels[1].metadata.membership[outside_id]
        @test !support_panels[2].metadata.membership[outside_id]
        @test all(p -> only(p.metadata.query_readout).region_id == target_id, support_panels)
        @test viz.check_visual_spec(stalk_spec).valid

        identity_spec = TOA.visual_spec(enc; kind=:presentation_inspector,
            parameter_pair=(source, target), box=([-1, -1], [4, 4]))
        identity = identity_spec.metadata.presentation_map
        @test identity_spec.metadata.defined
        @test IR.ambient_projection(identity) == reshape(QQ[1], 1, 1)
        @test IR.induced_map(identity) == reshape(QQ[1], 1, 1)
        @test IR.image_basis(IR.target_stalk(identity)) * IR.induced_map(identity) ==
              IR.ambient_projection(identity) * IR.image_basis(IR.source_stalk(identity))

        leaving_spec = TOA.visual_spec(enc; kind=:presentation_inspector,
            parameter_pair=(target, outside_support), box=([-1, -1], [4, 4]))
        leaving = leaving_spec.metadata.presentation_map
        @test leaving_spec.metadata.defined
        @test isempty(IR.active_rows(IR.target_stalk(leaving)))
        @test IR.active_columns(IR.target_stalk(leaving)) == [1]
        @test size(IR.presentation_matrix(IR.target_stalk(leaving))) == (0, 1)
        @test size(IR.image_basis(IR.target_stalk(leaving))) == (0, 0)
        @test IR.ambient_projection(leaving) == zeros(QQ, 0, 1)
        @test IR.induced_map(leaving) == zeros(QQ, 0, 1)
        @test viz.check_visual_spec(leaving_spec).valid

        if backend === :zn
            @test support_panels[1].metadata.geometry.geometry_kind == :nearest_lattice_tiles
            @test occursin("ties round-to-even", support_panels[1].subtitle)
            # The drawing's real-coordinate tiles retain the documented
            # nearest-lattice query convention: 2.5 rounds to 2, not 3.
            edge = TOA.visual_spec(enc; kind=:presentation_inspector,
                point=(5//2, 1//1), box=([-1, -1], [4, 4]))
            past_edge = TOA.visual_spec(enc; kind=:presentation_inspector,
                point=(5//2 + 1//(2^53), 1//1), box=([-1, -1], [4, 4]))
            @test IR.presentation_summary(only(edge.metadata.stalks)).dimension == 1
            @test IR.presentation_summary(only(past_edge.metadata.stalks)).dimension == 0
        end
    end
end

@testset "A83 three-parameter encodings retain finite-label presentation inspection" begin
    viz = TamerOp.Visualization
    # A singleton finite model is sufficient to check the geometry boundary:
    # a three-parameter classifier does not imply a planar support picture.
    P = FF.ProductOfChainsPoset((1, 1, 1))
    H = FF.FringeModule{QQ}(P, [FF.Upset(P, trues(1))],
        [FF.Downset(P, trues(1))], reshape(QQ[1], 1, 1); field=CM.QQField())
    M = TOA.pmodule_from_fringe(H)
    raw_classifier = EC.GridEncodingMap(P, ([0], [0], [0]))
    enc = RES.EncodingResult(P, M, EC.compile_encoding(P, raw_classifier); H)
    @test TamerOp.encoding_presentation(enc) === H
    @test :presentation_inspector in TamerOp.available_visuals(enc)
    spec = TOA.visual_spec(enc; kind=:presentation_inspector, vertex=1, basis=true)
    @test all(p -> p.kind != :presentation_support, spec.panels)
    for support_panel in spec.panels[1:2]
        @test support_panel.metadata.support_coordinate == :finite_vertex
        @test support_panel.metadata.membership == [true]
        @test only(support_panel.layers).column_labels == ["q1"]
    end
    stalk = only(spec.metadata.stalks)
    @test IR.presentation_matrix(stalk) == reshape(QQ[1], 1, 1)
    @test IR.image_basis(stalk) == reshape(QQ[1], 1, 1)
    @test viz.check_visual_spec(spec).valid
    @test !viz.check_visual_request(enc; kind=:presentation_inspector, point=(0, 0, 0)).valid
    @test_throws ArgumentError TOA.visual_spec(enc; kind=:presentation_inspector, point=(0, 0, 0))
end

# A function boundary keeps LLVM from optimizing five field implementations
# into one very large test thunk; the package uses ordinary compilation.
@noinline function _check_linked_inspection_field(field, ::Type{K}) where {K}
    V = TamerOp.Visualization
    P = chain_poset(3)
    # The two intervals [1,2] and [2,3] have dimensions 1,2,1.
    # Both adjacent maps have rank one, but the composite is zero.
    H = FF.FringeModule{K}(P,
        [FF.Upset(P, BitVector([1,1,1])), FF.Upset(P, BitVector([0,1,1]))],
        [FF.Downset(P, BitVector([1,1,0])), FF.Downset(P, BitVector([1,1,1]))],
        K[1 0; 0 1]; field)
    M = TOA.pmodule_from_fringe(H)
    enc = RES.EncodingResult(P, M, nothing; H)
    session = TamerOp.inspection_session(enc; cache_limit=2)
    @test session isa TOA.InspectionSession
    @test V.inspection_summary(session).supported_views == (:module, :presentation)
    @test !V.inspection_summary(session).geometry_available
    @test V.inspection_selection(session).selector == :none
    @test V.inspection_snapshot(session).metadata.inspection.kind == :overview
    @test V.check_inspection_session(session).valid
    @test V.describe(session).closed == false
    @test occursin("InspectionSession", sprint(show, session))
    @test occursin("InspectionSession", sprint(show, MIME"text/plain"(), session))
    for q in 1:3
        V.select_inspection!(session; vertex=q)
        @test V.inspection_selection(session).vertex == q
        @test V.inspection_snapshot(session).metadata.inspection.dimension == (1,2,1)[q]
    end
    expected = (((1,2), reshape(K[1,0], 2,1)),
                ((2,3), reshape(K[0,1], 1,2)),
                ((1,3), reshape(K[0], 1,1)))
    actual = Matrix{K}[]
    for (pair, matrix) in expected
        V.select_inspection!(session; pair)
        snap = V.inspection_snapshot(session)
        @test snap.metadata.inspection.defined
        @test snap.metadata.inspection.matrix == matrix
        @test V.check_visual_spec(snap).valid
        push!(actual, snap.metadata.inspection.matrix)
        # Switching interpretation preserves the actual selected labels.
        V.select_inspection!(session; view=:presentation)
        ps = V.inspection_snapshot(session)
        @test V.inspection_selection(session).pair == pair
        @test IR.induced_map(ps.metadata.presentation_map) == matrix
        @test only(filter(p -> p.kind == :hasse, ps.panels)).metadata.selected_pair == pair
        V.select_inspection!(session; view=:module)
    end
    @test actual[2] * actual[1] == actual[3] == zeros(K,1,1)
    V.select_inspection!(session; pair=(3,1))
    @test !V.inspection_snapshot(session).metadata.inspection.defined
    @test V.inspection_snapshot(session).metadata.inspection.matrix === nothing
    V.select_inspection!(session; pair=(2,2))
    @test V.inspection_snapshot(session).metadata.inspection.matrix == K[1 0; 0 1]
    # Repeating the selected query uses bounded retained algebra.
    hits = V.inspection_summary(session).cache_hits
    V.select_inspection!(session; pair=(2,2))
    @test V.inspection_summary(session).cache_hits > hits
    @test V.inspection_summary(session).cache_entries <= 2
    old = V.inspection_snapshot(session)
    V.close_inspection!(session)
    @test V.inspection_summary(session).closed
    @test V.inspection_snapshot(session) === old
    @test V.inspection_summary(session).cache_entries == 0
    @test_throws ArgumentError V.select_inspection!(session; vertex=1)
    @test V.close_inspection!(session) === session
end

@testset "A37 linked inspection keeps maps and field semantics" begin
    for field in FIELDS_FULL
        _check_linked_inspection_field(field, CM.coeff_type(field))
    end
end

@testset "A37 selection transactions and finite-only contracts" begin
    V = TamerOp.Visualization
    P = chain_poset(2)
    M = MD.PModule{QQ}(P, [1,1], Dict((1,2) => reshape(QQ[1],1,1)); field=CM.QQField())
    s = TamerOp.inspection_session(M)
    @test V.inspection_summary(s).supported_views == (:module,)
    @test TamerOp.available_visuals(s) == (:linked_inspector,)
    @test V.check_visual_request(s).valid
    @test V.check_visual_request(s).rendering.selection
    @test V.check_visual_request(s).rendering.hover
    @test !V.check_visual_request(s; backend=:cairomakie).valid
    @test !V.check_visual_request(s; vertex=1).valid
    for kwargs in ((; vertex=0), (; vertex=true), (; vertex=3), (; pair=(1,3)),
                   (; vertex=1,pair=(1,2)), (; point=(0,0)),
                   (; view=:presentation), (; view=:imaginary), (; basis=true),
                   (; input=:imaginary))
        before, picture = V.inspection_selection(s), V.inspection_snapshot(s)
        @test !V.check_inspection_selection(s; kwargs...).valid
        @test_throws ArgumentError V.select_inspection!(s; kwargs...)
        @test V.inspection_selection(s) == before
        @test V.inspection_snapshot(s) === picture
    end
    @test_throws ArgumentError TamerOp.inspection_session(M; box=([0,0],[1,1]))
    @test_throws ArgumentError TamerOp.inspection_session(M; cache_limit=-1)
    @test_throws ArgumentError TamerOp.inspection_session(M; matrix_limit=(0,3))
    calls = Tuple{Int,Symbol}[]
    token = V._on_inspection(s, changed -> push!(calls,
        (V.inspection_selection(changed).revision, V.inspection_selection(changed).selector)))
    V.select_inspection!(s; vertex=2)
    @test last(calls)[2] == :vertex
    V.reset_inspection!(s)
    @test last(calls)[2] == :none
    @test length(calls) == 2 && calls[2][1] > calls[1][1]
    V._off_inspection!(s, token)
    V.select_inspection!(s; pair=(1,2))
    @test length(calls) == 2
    # Failed listener work cannot roll back the successfully selected vertex.
    reentrant = V._on_inspection(s, changed -> V.reset_inspection!(changed))
    V.select_inspection!(s; vertex=1)
    @test V.inspection_selection(s).vertex == 1
    V._off_inspection!(s, reentrant)
    closed_seen = Ref(false)
    V._on_inspection(s, changed -> (closed_seen[] = V.inspection_summary(changed).closed))
    live = V.visual_spec(s)
    @test live.kind == :linked_inspector
    @test live.interaction.requires_live_julia && !live.interaction.offline_widgets
    @test V.check_visual_spec(live).valid
    @test_throws ArgumentError V.render(live; backend=:cairomakie)
    mktempdir() do dir
        @test_throws ArgumentError TamerOp.save_visual(joinpath(dir,"live.html"), live)
        @test !isfile(joinpath(dir,"live.html"))
    end
    V.close_inspection!(s)
    @test closed_seen[]
    @test !V.check_visual_spec(live).valid
    @test V.check_visual_spec(V.inspection_snapshot(s)).valid
end

@testset "A37 exact coordinate entry cannot execute code" begin
    V = TamerOp.Visualization
    for (text, number) in (("2",2//1), ("-3/4",-3//4), (" 5//2 ",5//2),
                           ("0.1",1//10), ("-1.25",-5//4), ("1e-3",1//1000))
        @test V._parse_inspection_coordinate(text) == number
    end
    for text in ("", "NaN", "Inf", "1/0", "1/2/3", "sqrt(2)",
                 "1; error(\"executed\")", "begin 1 end")
        @test_throws ArgumentError V._parse_inspection_coordinate(text)
    end
end

@testset "A37 linked parameter queries preserve boundaries and ambient order" begin
    V = TamerOp.Visualization
    opts = TOA.EncodingOptions(; backend=:pl_backend, poset_kind=:signature, field=CM.QQField())
    enc = TamerOp.encode([TOA.BoxUpset([0.0,0.0]), TOA.BoxUpset([1.0,1.0])],
        [TOA.BoxDownset([2.0,2.0]), TOA.BoxDownset([3.0,3.0])], QQ[1 0;0 1], opts)
    s = TamerOp.inspection_session(enc; box=([-1,-1],[4,4]), cache_limit=3)
    @test V.inspection_summary(s).geometry_available
    for (point, dimension) in (((1//2,1//2),1), ((3//2,3//2),2),
                               ((5//2,5//2),1), ((1//2,5//2),0))
        V.select_inspection!(s; point)
        state = V.inspection_selection(s)
        @test only(state.query_points) == point
        @test state.vertex == EC.locate(TamerOp.encoding_map(enc), point)
        @test V.inspection_snapshot(s).metadata.inspection.dimension == dimension
    end
    V.select_inspection!(s; view=:presentation, upset=1, downset=2)
    active_zero = V.inspection_snapshot(s)
    stalk = only(active_zero.metadata.stalks)
    @test IR.active_rows(stalk) == [2] && IR.active_columns(stalk) == [1]
    @test IR.presentation_matrix(stalk) == reshape(QQ[0],1,1)
    @test IR.image_basis(stalk) === nothing
    V.select_inspection!(s; basis=true)
    @test size(IR.image_basis(only(V.inspection_snapshot(s).metadata.stalks))) == (1,0)
    @test V.inspection_selection(s).downset == 2
    a, b = (5//4,7//4), (7//4,5//4)
    @test EC.locate(TamerOp.encoding_map(enc), a) == EC.locate(TamerOp.encoding_map(enc), b)
    V.select_inspection!(s; parameter_pair=((5//4,5//4),(7//4,7//4)), basis=false)
    @test V.inspection_snapshot(s).metadata.defined
    @test IR.induced_map(V.inspection_snapshot(s).metadata.presentation_map) == QQ[1 0;0 1]
    V.select_inspection!(s; parameter_pair=(a,b), basis=false)
    @test V.inspection_selection(s).parameter_relation == :incomparable
    @test !V.inspection_snapshot(s).metadata.defined
    @test V.inspection_snapshot(s).metadata.presentation_map === nothing
    q = EC.locate(TamerOp.encoding_map(enc), a)
    V.select_inspection!(s; pair=(q,q))
    @test V.inspection_snapshot(s).metadata.defined
    @test IR.induced_map(V.inspection_snapshot(s).metadata.presentation_map) == QQ[1 0;0 1]
    # Switching back to point input retains the supplied rational exactly.
    p = (1//1 + 1//(big(1)<<70),3//2)
    V.select_inspection!(s; point=p, view=:module)
    @test only(V.inspection_selection(s).query_points) == p
    @test only(V.inspection_selection(s).query_points)[1] > 1
    V.select_inspection!(s; point=(1.0,1.5), input=:pointer)
    @test V.inspection_selection(s).input == :pointer
    V.close_inspection!(s)
end

@testset "A37 native linked controls and static snapshots" begin
    V = TamerOp.Visualization
    if Base.find_package("WGLMakie") === nothing || Base.find_package("CairoMakie") === nothing
        @test_skip false
    else
        @eval import WGLMakie
        @eval import CairoMakie
        E = Base.get_extension(TamerOp, :TamerOpWGLMakieExt)
        @test E !== nothing
        opts = TOA.EncodingOptions(; backend=:pl_backend, poset_kind=:signature, field=CM.QQField())
        enc = TamerOp.encode([TOA.BoxUpset([0.0,0.0]), TOA.BoxUpset([1.0,1.0])],
            [TOA.BoxDownset([2.0,2.0]), TOA.BoxDownset([3.0,3.0])], QQ[1 0;0 1], opts)
        s = TamerOp.inspection_session(enc; box=([-1,-1],[4,4]))
        app = TamerOp.visualize(s; backend=:wglmakie)
        @test app isa WGLMakie.Bonito.App
        ui = E._inspection_ui(app)
        controls = ui.controls
        graph = ui.axes.hasse
        before = V.inspection_summary(s)
        hovered = ui.callbacks.hover(; point=(1.5,1.5))
        @test hovered.dimension == 2 && hovered.approximate
        @test V.inspection_summary(s).revision == before.revision
        @test V.inspection_summary(s).cache_misses == before.cache_misses
        controls.point_x.value[] = "1/2"
        controls.point_y.value[] = "5/2"
        controls.select_point.value[] = true
        @test isempty(ui.error_text[])
        @test only(V.inspection_selection(s).query_points) == (1//2,5//2)
        @test V.inspection_snapshot(s).metadata.inspection.dimension == 0
        controls.view.option_index[] = 2
        controls.downset.option_index[] = 2
        controls.basis.value[] = true
        @test isempty(ui.error_text[])
        @test size(IR.image_basis(only(V.inspection_snapshot(s).metadata.stalks))) == (1,0)
        @test !ui.basis_disabled[]
        @test length(ui.markers.hasse.stalk[]) == 1
        @test length(ui.markers.region.stalk[]) == 1
        @test ui.axes.hasse === graph
        selected = V.inspection_selection(s)
        controls.point_x.value[] = "error(\"not a number\")"
        controls.select_point.value[] = true
        @test !isempty(ui.error_text[])
        @test V.inspection_selection(s) == selected
        # These are the same live callbacks consumed by WGL mouse picking.
        controls.endpoint.option_index[] = 2
        @test ui.callbacks.choose(:point,(0.5,0.5); input=:pointer)
        controls.endpoint.option_index[] = 3
        @test ui.callbacks.choose(:point,(2.5,2.5); input=:pointer)
        @test isempty(ui.error_text[])
        @test V.inspection_selection(s).selector == :parameter_pair
        @test V.inspection_selection(s).input == :pointer
        @test V.inspection_snapshot(s).metadata.defined
        @test IR.induced_map(V.inspection_snapshot(s).metadata.presentation_map) == reshape(QQ[0],1,1)
        @test ui.basis_disabled[]
        @test length(ui.markers.hasse.source[]) == length(ui.markers.hasse.target[]) == 1
        @test length(ui.markers.region.source[]) == length(ui.markers.region.target[]) == 1
        @test ui.preparation_count == 1
        controls.source_x.value[] = "5/4"
        controls.source_y.value[] = "7/4"
        controls.target_x.value[] = "7/4"
        controls.target_y.value[] = "5/4"
        controls.select_points.value[] = true
        @test !V.inspection_snapshot(s).metadata.defined
        @test V.inspection_selection(s).parameter_relation == :incomparable
        @test isempty(ui.error_text[])
        controls.reset.value[] = true
        @test V.inspection_selection(s).selector == :none
        @test isempty(ui.markers.region.source[]) && isempty(ui.markers.region.target[])
        controls.endpoint.option_index[] = 1
        controls.vertex.value[] = "1"
        controls.select_vertex.value[] = true
        controls.next_vertex.value[] = true
        @test V.inspection_selection(s).vertex == 2
        @test isempty(V.inspection_selection(s).query_points)
        @test isempty(ui.markers.region.stalk[])
        @test isempty(ui.error_text[])
        @test isempty(V.inspection_summary(s).listener_errors)
        @test V.inspection_summary(s).geometry_preparations == 1
        @test V.inspection_summary(s).hasse_preparations == 1
        mktempdir() do dir
            # Direct static export of a live specification captures current
            # selection, and reports the kind that was actually saved.
            saved = TamerOp.save_visual(dir, "selected", V.visual_spec(s); format=:svg)
            @test TamerOp.export_kind(saved) == V.inspection_snapshot(s).kind
            @test filesize(TamerOp.export_path(saved)) > 1000
            @test occursin("<svg", read(TamerOp.export_path(saved), String))
        end
        final_snapshot = V.inspection_snapshot(s)
        controls.close.value[] = true
        @test V.inspection_summary(s).closed
        @test ui.closed[] && ui.disabled[]
        @test isempty(ui.observers)
        @test V.inspection_summary(s).listener_count == 0
        @test V.inspection_snapshot(s) === final_snapshot
        @test !haskey(E._INSPECTION_UI_HANDLES, app)
        @test !ui.callbacks.choose(:vertex,1)
        @test V.inspection_snapshot(s) === final_snapshot
    end
end

@testset "A37 Bonito serialization and independent viewer lifecycle" begin
    if Base.find_package("WGLMakie") === nothing
        @test_skip false
    else
        @eval import WGLMakie
        V, A = TamerOp.Visualization, TamerOp.Advanced
        B, M = WGLMakie.Bonito, WGLMakie.Makie
        E = Base.get_extension(TamerOp, :TamerOpWGLMakieExt)
        opts = A.EncodingOptions(; backend=:pl_backend, poset_kind=:signature,
            field=TamerOp.CoreModules.QQField())
        enc = TamerOp.encode([A.BoxUpset([0.0,0.0]), A.BoxUpset([1.0,1.0])],
            [A.BoxDownset([2.0,2.0]), A.BoxDownset([3.0,3.0])], TamerOp.QQ[1 0;0 1], opts)
        s = TamerOp.inspection_session(enc; box=([-1,-1],[4,4]))
        app1 = TamerOp.visualize(s; backend=:wglmakie)
        ui1 = E._inspection_ui(app1)
        client1 = B.Session(B.NoConnection(); asset_server=B.NoServer())
        rejected_client = B.Session(B.NoConnection(); asset_server=B.NoServer())
        client2 = B.Session(B.NoConnection(); asset_server=B.NoServer())
        try
            # Bonito's own tests use this no-connection path. It renders native
            # controls, serializes WGL scenes and registers lifecycle callbacks;
            # it does not execute JavaScript or emulate a browser click.
            dom1 = B.session_dom(client1, app1; init=false)
            html1 = sprint(show, MIME"text/html"(), dom1)
            @test occursin(ui1.dom_id * "-point-x", html1)
            @test occursin("<canvas", html1)
            @test occursin("Live Julia session required", html1)
            bytes1 = B.serialize_binary(client1, B.fused_messages!(client1))
            @test !isempty(bytes1)
            @test ui1.browser_client[] === client1

            refused = B.session_dom(rejected_client, app1; init=false)
            refused_html = sprint(show, MIME"text/html"(), refused)
            @test occursin("already has a browser client", refused_html)
            # Bonito may HTML-escape parentheses in ordinary text nodes.
            has_reopen_guidance = occursin(r"visualize(?:\(|&#40;)session(?:\)|&#41;)", refused_html)
            @test has_reopen_guidance
            @test !occursin("<canvas", refused_html)
            @test ui1.browser_client[] === client1
            @test app1.session[] === client1
            close(rejected_client)
            @test !ui1.closed[] && !V.inspection_summary(s).closed

            # An independent returned App has independent WGL scenes and still
            # follows the same core selection session.
            app2 = TamerOp.visualize(s; backend=:wglmakie)
            ui2 = E._inspection_ui(app2)
            dom2 = B.session_dom(client2, app2; init=false)
            @test occursin(ui2.dom_id * "-point-x", sprint(show, MIME"text/html"(), dom2))
            @test !isempty(B.serialize_binary(client2, B.fused_messages!(client2)))
            @test ui1.figure !== ui2.figure
            @test V.inspection_summary(s).listener_count == 2

            close(client1)
            @test ui1.closed[] && isempty(ui1.observers)
            @test isempty(ui1.figure.content)
            @test !haskey(E._INSPECTION_UI_HANDLES, app1)
            @test !ui2.closed[] && !M.isclosed(ui2.figure.scene)
            @test !V.inspection_summary(s).closed
            @test V.inspection_summary(s).listener_count == 1

            V.select_inspection!(s; point=(1//2,5//2), view=:presentation, basis=true)
            @test isempty(V.inspection_summary(s).listener_errors)
            @test length(ui2.markers.region.stalk[]) == 1
            @test occursin("presentation", ui2.selection_text[])
            @test !isempty(B.serialize_binary(client2, B.fused_messages!(client2)))
            initial_child_count = length(client2.children)
            for support_id in (2,1,2)
                old_figure = ui2.support_figure[]
                V.select_inspection!(s; downset=support_id)
                @test isempty(V.inspection_summary(s).listener_errors)
                @test old_figure !== ui2.support_figure[]
                @test isempty(old_figure.content)
                @test length(client2.children) == initial_child_count
                @test !isempty(B.serialize_binary(client2, B.fused_messages!(client2)))
            end

            snapshot = V.inspection_snapshot(s)
            V.close_inspection!(s)
            @test ui2.closed[] && isempty(ui2.observers)
            @test isempty(ui2.figure.content)
            @test ui2.support_figure[] === nothing
            @test V.inspection_snapshot(s) === snapshot
            @test V.inspection_summary(s).listener_count == 0
            @test isempty(V.inspection_summary(s).listener_errors)
            @test !haskey(E._INSPECTION_UI_HANDLES, app2)
        finally
            close(rejected_client)
            close(client1)
            close(client2)
            V.close_inspection!(s)
        end
    end
end

@testset "A37 server embedding renders native inspector controls" begin
    if Base.find_package("WGLMakie") === nothing
        @test_skip false
    else
        @eval import WGLMakie
        V, A = TamerOp.Visualization, TamerOp.Advanced
        B = WGLMakie.Bonito
        E = Base.get_extension(TamerOp, :TamerOpWGLMakieExt)
        opts = A.EncodingOptions(; backend=:pl_backend, poset_kind=:signature,
            field=TamerOp.CoreModules.QQField())
        enc = TamerOp.encode([A.BoxUpset([0.0,0.0]), A.BoxUpset([1.0,1.0])],
            [A.BoxDownset([2.0,2.0]), A.BoxDownset([3.0,3.0])], TamerOp.QQ[1 0;0 1], opts)
        s = TamerOp.inspection_session(enc; box=([-1,-1],[4,4]))
        viewers = B.App[]
        # A server route is reusable, but each inspector viewer has one client.
        # Expand the inner App so the outer render pass processes all its DOM.
        # Returning the App directly printed Dropdown objects and caused other
        # widgets to fall back to independent HTML displays in a real browser.
        wrapper = B.App() do browser_session
            V.inspection_summary(s).closed && return B.DOM.p("Inspection session closed.")
            viewer = TamerOp.visualize(s; backend=:wglmakie)
            push!(viewers, viewer)
            return B.jsrender(browser_session, viewer)
        end
        clients = [B.Session(B.NoConnection(); asset_server=B.NoServer()) for _ in 1:3]
        try
            for client in clients[1:2]
                dom = B.session_dom(client, wrapper; init=false, html_document=true)
                html = sprint(show, MIME"text/html"(), dom)
                viewer = last(viewers)
                ui = E._inspection_ui(viewer)
                @test !occursin("Bonito.Dropdown", html)
                @test length(collect(eachmatch(r"<html\b", html))) == 1
                native_selects = (("endpoint", ("Stalk", "Source", "Target")),
                        ("view", ("Module coordinates", "Presentation image coordinates")),
                        ("upset", ("U1", "U2")), ("downset", ("D1", "D2")),
                        ("slice-interval", ("No selected interval",)),
                        ("slice-scope", ("Viewing-window restriction", "Whole line (certified endpoints)")))
                @test length(collect(eachmatch(r"<select\b", html))) == length(native_selects)
                for (suffix, choices) in native_selects
                    element = match(Regex("<select\\b[^>]*\\bid=\"" * ui.dom_id *
                        "-" * suffix * "\"[^>]*>(.*?)</select>", "s"), html)
                    @test element !== nothing
                    if element !== nothing
                        @test occursin("class=", element.match)
                        for choice in choices
                            @test occursin("<option", element.captures[1])
                            # Bonito escapes parentheses in visible option text.
                            option_text = replace(element.captures[1], "&#40;" => "(", "&#41;" => ")")
                            @test occursin(choice, option_text)
                        end
                    end
                end
                for suffix in ("vertex", "label-source", "label-target", "point-x", "point-y",
                        "source-x", "source-y", "target-x", "target-y", "basis")
                    @test occursin(Regex("<input\\b[^>]*\\bid=\"" * ui.dom_id * "-" * suffix * "\""), html)
                end
                for suffix in ("select-vertex", "previous-vertex", "next-vertex", "select-labels",
                        "select-point", "select-points", "reset", "close")
                    @test occursin(Regex("<button\\b[^>]*\\bid=\"" * ui.dom_id * "-" * suffix * "\""), html)
                end
                @test occursin("<canvas", html)
                @test !isempty(B.serialize_binary(client, B.fused_messages!(client)))
                @test ui.browser_client[] === client
                @test viewer.session[] === client
            end
            first_ui, second_ui = E._inspection_ui.(viewers)
            @test first_ui.figure !== second_ui.figure
            @test first_ui.dom_id != second_ui.dom_id
            @test V.inspection_summary(s).listener_count == 2
            V.select_inspection!(s; parameter_pair=((1//2,1//2), (5//2,5//2)))
            @test occursin("Supplied", first_ui.selection_text[])
            @test first_ui.selection_text[] == second_ui.selection_text[]
            @test V.inspection_snapshot(s).metadata.inspection.matrix == zeros(QQ, 1, 1)
            close(clients[1])
            @test first_ui.closed[]
            @test !second_ui.closed[]
            @test !V.inspection_summary(s).closed
            @test V.inspection_summary(s).listener_count == 1
            V.close_inspection!(s)
            @test second_ui.closed[]
            @test V.inspection_summary(s).listener_count == 0
            closed_dom = B.session_dom(clients[3], wrapper; init=false)
            @test occursin("Inspection session closed", sprint(show, MIME"text/html"(), closed_dom))
            @test length(viewers) == 2
        finally
            V.close_inspection!(s)
            foreach(close, clients)
        end
    end
end

@testset "A37 oriented grid outside labels and equivalent selections" begin
    V = TamerOp.Visualization
    P = FF.ProductOfChainsPoset((2, 2))
    grid = EC.GridEncodingMap(P, ([0, 2], [1, 4]); orientation=(-1, 1))
    U, D = FF.Upset(P, trues(4)), FF.Downset(P, trues(4))
    H = FF.FringeModule{QQ}(P, [U, U], [D, D], QQ[1 0; 0 1]; field=CM.QQField())
    enc = RES.EncodingResult(P, TOA.pmodule_from_fringe(H), EC.compile_encoding(P, grid); H)
    s = TamerOp.inspection_session(enc; box=([-3, 0], [2, 5]), cache_limit=16)
    prepared = V._inspection_scene(s)
    geometry = prepared.region.metadata.geometry
    positions = prepared.hasse.metadata.positions
    for view in (:module, :presentation)
        V.select_inspection!(s; view, point=(1, 1))
        @test V.inspection_selection(s).vertex == 0
        outside = V.inspection_snapshot(s)
        if view === :module
            @test outside.metadata.inspection.kind == :outside
            @test outside.metadata.inspection.dimension === nothing
        else
            @test outside.metadata.relation == :outside
            @test only(outside.metadata.stalks) === nothing
        end
        region = first(p for p in outside.panels if haskey(p.metadata, :geometry))
        @test only(region.metadata.query_readout).region_id == 0
        @test region.metadata.geometry === geometry
        @test only(p for p in outside.panels if p.kind == :hasse).metadata.positions === positions

        # This pair is forward in the oriented parameter order but its source
        # is unrepresented. It must never be presented as a zero matrix.
        V.select_inspection!(s; parameter_pair=((1, 1), (0, 1)))
        missing = V.inspection_snapshot(s)
        @test V.inspection_selection(s).parameter_relation == :comparable
        @test first(V.inspection_selection(s).pair) == 0
        if view === :module
            @test !missing.metadata.inspection.defined
            @test missing.metadata.inspection.matrix === nothing
            @test missing.metadata.inspection.source_dimension === nothing
            @test missing.metadata.inspection.target_dimension == 2
        else
            @test !missing.metadata.defined && missing.metadata.presentation_map === nothing
            @test first(missing.metadata.stalks) === nothing
            @test IR.presentation_summary(last(missing.metadata.stalks)).dimension == 2
            @test IR.image_basis(last(missing.metadata.stalks)) === nothing
        end

        # Orientation (-1,+1) means decreasing x and increasing y is forward.
        V.select_inspection!(s; parameter_pair=((0, 1), (-2, 4)))
        forward = V.inspection_snapshot(s)
        @test V.inspection_selection(s).parameter_relation == :comparable
        if view === :module
            @test forward.metadata.inspection.defined
            @test forward.metadata.inspection.matrix == QQ[1 0; 0 1]
        else
            @test forward.metadata.defined
            @test IR.induced_map(forward.metadata.presentation_map) == QQ[1 0; 0 1]
        end
        V.select_inspection!(s; parameter_pair=((-2, 4), (0, 1)))
        reverse = V.inspection_snapshot(s)
        @test V.inspection_selection(s).parameter_relation == :reverse_comparable
        if view === :module
            @test !reverse.metadata.inspection.defined
            @test reverse.metadata.inspection.matrix === nothing
        else
            @test !reverse.metadata.defined && reverse.metadata.presentation_map === nothing
        end
    end

    # Exact query coordinates differ while the selected algebra is identical.
    a, b = (-1//4, 2//1), (-1//2, 3//1)
    @test EC.locate(grid, a) == EC.locate(grid, b)
    V.select_inspection!(s; point=a)
    stalk = only(V.inspection_snapshot(s).metadata.stalks)
    before = V.inspection_summary(s)
    V.select_inspection!(s; point=b)
    after = V.inspection_summary(s)
    @test after.cache_hits == before.cache_hits + 1
    @test after.cache_misses == before.cache_misses
    @test only(V.inspection_selection(s).query_points) == b
    @test only(V.inspection_snapshot(s).metadata.stalks) === stalk
    panel = first(p for p in V.inspection_snapshot(s).panels if haskey(p.metadata, :query_readout))
    @test only(panel.metadata.query_readout).point == b
    @test only(panel.metadata.query_readout).region_id == EC.locate(grid, b)
    V.select_inspection!(s; upset=2, downset=2)
    @test V.inspection_summary(s).cache_hits == after.cache_hits + 1
    @test V.inspection_summary(s).cache_misses == after.cache_misses
    @test only(V.inspection_snapshot(s).metadata.stalks) === stalk
    @test V.inspection_selection(s).upset == V.inspection_selection(s).downset == 2
    supports = filter(p -> p.kind == :presentation_support, V.inspection_snapshot(s).panels)
    @test length(supports) == 2 && all(p -> p.metadata.support_id == 2, supports)
    @test V.inspection_summary(s).geometry_preparations == 1
    @test V.inspection_summary(s).hasse_preparations == 1
    V.close_inspection!(s)
end

@testset "A37 failed listeners do not stop later listeners or cleanup" begin
    V = TamerOp.Visualization
    P = chain_poset(2)
    M = MD.PModule{QQ}(P, [1, 1], Dict((1,2) => reshape(QQ[1], 1, 1)); field=CM.QQField())
    s = TamerOp.inspection_session(M)
    observed = Int[]
    failed = V._on_inspection(s, changed -> V.reset_inspection!(changed))
    V._on_inspection(s, changed -> push!(observed, V.inspection_selection(changed).revision))
    V.select_inspection!(s; vertex=2)
    @test V.inspection_selection(s).vertex == 2
    @test observed == [V.inspection_selection(s).revision]
    @test !isempty(V.inspection_summary(s).listener_errors)
    @test !V.inspection_summary(s).updating
    @test V.check_inspection_session(s).valid
    V._off_inspection!(s, failed)
    snapshot = V.inspection_snapshot(s)
    V.close_inspection!(s)
    @test length(observed) == 2 # The surviving listener also sees closure.
    @test V.inspection_summary(s).listener_count == 0
    @test V.check_inspection_session(s).valid
    @test V.inspection_snapshot(s) === snapshot
end

@testset "A37 zero-capacity and evicted algebra caches" begin
    V = TamerOp.Visualization
    P = chain_poset(2)
    M = MD.PModule{QQ}(P, [1, 1], Dict((1,2) => reshape(QQ[1], 1, 1)); field=CM.QQField())
    uncached = TamerOp.inspection_session(M; cache_limit=0)
    for _ in 1:2
        V.select_inspection!(uncached; pair=(1, 2))
        @test V.inspection_snapshot(uncached).metadata.inspection.matrix == reshape(QQ[1], 1, 1)
    end
    @test V.inspection_summary(uncached).cache_entries == 0
    @test V.inspection_summary(uncached).cache_hits == 0
    @test V.inspection_summary(uncached).cache_misses == 2
    @test V.check_inspection_session(uncached).valid
    V.close_inspection!(uncached)

    bounded = TamerOp.inspection_session(M; cache_limit=1)
    for pair in ((1, 1), (1, 2), (1, 1))
        V.select_inspection!(bounded; pair)
        @test V.inspection_summary(bounded).cache_entries == 1
        @test V.inspection_snapshot(bounded).metadata.inspection.target == pair[2]
    end
    @test V.inspection_summary(bounded).cache_misses == 3
    @test V.inspection_summary(bounded).cache_hits == 0
    V.select_inspection!(bounded; pair=(1, 1))
    @test V.inspection_summary(bounded).cache_hits == 1
    @test V.inspection_summary(bounded).cache_misses == 3
    @test V.check_inspection_session(bounded).valid
    V.close_inspection!(bounded)
end

@testset "A37 native Makie pointer event routing" begin
    V = TamerOp.Visualization
    if Base.find_package("WGLMakie") === nothing
        @test_skip false
    else
        @eval import WGLMakie
        E = Base.get_extension(TamerOp, :TamerOpWGLMakieExt)
        M = WGLMakie.Makie
        opts = TOA.EncodingOptions(; backend=:pl_backend, poset_kind=:signature, field=CM.QQField())
        enc = TamerOp.encode([TOA.BoxUpset([0.0,0.0])],
            [TOA.BoxDownset([2.0,2.0])], reshape(QQ[1],1,1), opts)
        s = TamerOp.inspection_session(enc; box=([-1,-1],[4,4]))
        app = TamerOp.visualize(s; backend=:wglmakie)
        ui = E._inspection_ui(app)
        try
            # Exercise the event consumers, including viewport hit testing and
            # nearest-vertex picking. This does not execute browser JavaScript.
            M.update_state_before_display!(ui.figure)
            events = M.events(ui.figure.scene)
            region, hasse = ui.axes.region, ui.axes.hasse
            @test region !== nothing
            @test ui.controls.endpoint.value[] == "Stalk"
            function move_to_data!(ax, point)
                # Makie.project(scene, point) yields scene-local pixels; native
                # mouse events and is_mouseinside use figure-global pixels.
                local_pixel = M.project(ax.scene, M.Point2d(point))
                pixel = local_pixel + minimum(M.viewport(ax.scene)[])
                events.mouseposition[] = (Float64(pixel[1]), Float64(pixel[2]))
            end
            function left_click!()
                events.mousebutton[] = M.MouseButtonEvent(M.Mouse.left, M.Mouse.press)
                events.mousebutton[] = M.MouseButtonEvent(M.Mouse.left, M.Mouse.release)
            end
            hover_work(d) = (d.revision, d.cache_hits, d.cache_misses, d.cache_entries,
                d.snapshot_builds, d.module_materialized, d.geometry_preparations, d.hasse_preparations)

            inside = (1.0,1.0)
            inside_label = EC.locate(TamerOp.encoding_map(enc), inside)
            outside_label = EC.locate(TamerOp.encoding_map(enc), (3.0,1.0))
            dims = TamerOp.dimensions(enc)
            @test dims[inside_label] == 1 && dims[outside_label] == 0
            before_hover = hover_work(V.inspection_summary(s))
            move_to_data!(region, inside)
            @test M.is_mouseinside(region.scene)
            @test !M.is_mouseinside(hasse.scene)
            @test hover_work(V.inspection_summary(s)) == before_hover
            @test occursin("Vertex $inside_label; dimension 1.", ui.hover_text[])
            left_click!()
            selected = V.inspection_selection(s)
            @test selected.selector == :point && selected.input == :pointer
            @test selected.vertex == inside_label
            @test all(isapprox.(only(selected.query_points), inside; atol=1e-5))
            @test V.inspection_snapshot(s).metadata.inspection.dimension == 1
            @test length(ui.markers.region.stalk[]) == length(ui.markers.hasse.stalk[]) == 1

            # The label is obtained from the actual encoder, and only its
            # schematic position is taken from the prepared Hasse layout.
            position = V._inspection_scene(s).hasse.metadata.positions[outside_label]
            before_hover = hover_work(V.inspection_summary(s))
            move_to_data!(hasse, position)
            @test M.is_mouseinside(hasse.scene)
            @test !M.is_mouseinside(region.scene)
            @test hover_work(V.inspection_summary(s)) == before_hover
            @test occursin("Vertex $outside_label; dimension 0.", ui.hover_text[])
            left_click!()
            selected = V.inspection_selection(s)
            @test selected.selector == :vertex && selected.input == :pointer
            @test selected.vertex == outside_label && isempty(selected.query_points)
            @test V.inspection_snapshot(s).metadata.inspection.dimension == 0
            @test length(ui.markers.hasse.stalk[]) == 1
            @test isempty(ui.markers.region.stalk[])

            # Outside both viewports there is no selection or algebra work.
            before_hover = hover_work(V.inspection_summary(s))
            events.mouseposition[] = (-100.0,-100.0)
            left_click!()
            @test V.inspection_selection(s) == selected
            @test hover_work(V.inspection_summary(s)) == before_hover
            # Hasse picking must not overwrite the captured vertex TextField.
            ui.controls.vertex.value[] = string(inside_label)
            ui.controls.select_vertex.value[] = true
            @test V.inspection_selection(s).vertex == inside_label
            @test V.inspection_selection(s).input == :provided
            ui.controls.next_vertex.value[] = true
            @test V.inspection_selection(s).vertex == min(inside_label + 1, length(dims))
            @test isempty(ui.error_text[])
            @test isempty(V.inspection_summary(s).listener_errors)
        finally
            V.close_inspection!(s)
        end
    end
end

@testset "A37 pending endpoints survive display changes" begin
    if Base.find_package("WGLMakie") === nothing
        @test_skip false
    else
        @eval import WGLMakie
        V, A = TamerOp.Visualization, TamerOp.Advanced
        E = Base.get_extension(TamerOp, :TamerOpWGLMakieExt)
        opts = A.EncodingOptions(; backend=:pl_backend, poset_kind=:signature,
            field=TamerOp.CoreModules.QQField())
        enc = TamerOp.encode([A.BoxUpset([0.0,0.0]), A.BoxUpset([1.0,1.0])],
            [A.BoxDownset([2.0,2.0]), A.BoxDownset([3.0,3.0])], TamerOp.QQ[1 0;0 1], opts)
        s = TamerOp.inspection_session(enc; box=([-1,-1],[4,4]))
        app = TamerOp.visualize(s; backend=:wglmakie)
        ui = E._inspection_ui(app)
        c = ui.controls
        a, b, replacement = (1//2,1//2), (5//2,5//2), (3//4,3//4)
        qa = TamerOp.EncodingCore.locate(TamerOp.encoding_map(enc), a)
        qb = TamerOp.EncodingCore.locate(TamerOp.encoding_map(enc), b)
        function point_control!(role, point)
            c.endpoint.option_index[] = role
            c.point_x.value[] = string(point[1])
            c.point_y.value[] = string(point[2])
            c.select_point.value[] = true
        end
        function vertex_control!(role, q)
            c.endpoint.option_index[] = role
            c.vertex.value[] = string(q)
            c.select_vertex.value[] = true
        end
        try
            point_control!(2, a) # Source first.
            c.view.option_index[] = 2
            c.upset.option_index[] = 2
            c.downset.option_index[] = 2
            c.basis.value[] = true
            @test only(V.inspection_selection(s).query_points) == a
            point_control!(3, b)
            @test V.inspection_selection(s).selector == :parameter_pair
            @test V.inspection_selection(s).query_points == (a,b)
            @test !V.inspection_selection(s).basis

            c.reset.value[] = true
            point_control!(3, b) # Target first also retains its role.
            c.view.option_index[] = 1
            point_control!(2, a)
            @test V.inspection_selection(s).query_points == (a,b)

            c.reset.value[] = true
            vertex_control!(2, qa)
            c.view.option_index[] = 2
            c.downset.option_index[] = 1
            vertex_control!(3, qb)
            @test V.inspection_selection(s).selector == :pair
            @test V.inspection_selection(s).pair == (qa,qb)
            @test isempty(V.inspection_selection(s).query_points)

            c.reset.value[] = true
            point_control!(2, a)
            c.downset.option_index[] = 2
            before = V.inspection_selection(s)
            vertex_control!(3, qb) # Retention must not allow mixed domains.
            @test V.inspection_selection(s) == before
            @test occursin("both endpoints", ui.error_text[])
            point_control!(3, b)
            @test V.inspection_selection(s).query_points == (a,b)
            @test isempty(ui.error_text[])

            # Same finite label does not make a different ambient point the
            # same endpoint. External replacement clears the pending source.
            c.reset.value[] = true
            point_control!(2, a)
            @test TamerOp.EncodingCore.locate(TamerOp.encoding_map(enc), replacement) == qa
            V.select_inspection!(s; point=replacement)
            point_control!(3, b)
            @test V.inspection_selection(s).selector == :point
            @test V.inspection_selection(s).query_points == (b,)

            c.reset.value[] = true
            point_control!(2, a)
            V.select_inspection!(s; vertex=qa) # Same label, changed domain.
            point_control!(3, b)
            @test V.inspection_selection(s).selector == :point
            @test V.inspection_selection(s).query_points == (b,)

            c.reset.value[] = true
            point_control!(2, a)
            V.select_inspection!(s; point=a, input=:pointer) # Changed input origin.
            point_control!(3, b)
            @test V.inspection_selection(s).selector == :point
            @test V.inspection_selection(s).query_points == (b,)

            c.reset.value[] = true
            point_control!(2, a)
            c.reset.value[] = true
            point_control!(3, b)
            @test V.inspection_selection(s).selector == :point
            @test V.inspection_selection(s).query_points == (b,)
            @test isempty(ui.error_text[])
            @test isempty(V.inspection_summary(s).listener_errors)
        finally
            V.close_inspection!(s)
        end
    end
end

@testset "A35 style contracts and request propagation" begin
    V = TamerOp.Visualization
    style = TamerOp.VisualStyle(; fontsize=22, linewidth_scale=2,
        markersize_scale=1.5, gap=18, padding=24,
        colors=(source="#123456", target=:darkorange))
    gray = TamerOp.VisualStyle(; palette=:grayscale)
    @test TamerOp.Advanced.VisualStyle === TamerOp.VisualStyle
    @test !ismutabletype(typeof(style))
    @test TamerOp.describe(style).fontsize == 22
    @test TamerOp.describe(gray).palette == :grayscale
    @test style == TamerOp.VisualStyle(; fontsize=22, linewidth_scale=2,
        markersize_scale=1.5, gap=18, padding=24,
        colors=(source="#123456", target=:darkorange))
    @test occursin("VisualStyle", sprint(show, style))
    for options in ((; palette=:unknown), (; palette=missing), (; fontsize=0), (; fontsize=Inf),
                    (; fontsize=true), (; linewidth_scale=-1), (; markersize_scale=0),
                    (; gap=-1), (; padding=NaN), (; font=""), (; mono_font=""),
                    (; colors=(unknown=:blue,)), (; colors=(source="not a color",)))
        @test_throws ArgumentError TamerOp.VisualStyle(; options...)
    end

    pc = TamerOp.DataTypes.PointCloud([0.0 0.0; 1.0 2.0; 3.0 1.0])
    spec = V.visual_spec(pc; kind=:points_2d)
    @test !V.check_visual_request(pc; kind=:points_2d, style).valid
    @test_throws ArgumentError V.visual_spec(pc; kind=:points_2d, style)
    @test :style in V.visual_summary(spec).rendering.renderer_keywords
    for metadata in ((; row_roles=[:source,:target]), (; column_roles=[:unknown]),
                     (; cell_roles=fill(missing,1,1)))
        invalid = V.VisualizationSpec(:a35_invalid_roles;
            layers=V.AbstractVisualizationLayer[V.MatrixLayer(reshape(["1"],1,1), ["t1"], ["s1"])],
            metadata)
        @test !V.check_visual_spec(invalid).valid
        @test_throws ArgumentError V.check_visual_spec(invalid; throw=true)
    end
    captured = NamedTuple[]
    V._register_visual_backend!(:a35_capture;
        render=(spec; display, figure, size, style) -> begin
            push!(captured, (; operation=:render, spec, style))
            (; spec, style, size)
        end,
        save=(path, spec; figure, size, style) -> begin
            push!(captured, (; operation=:save, spec, style))
            write(path, "A35 renderer-option propagation fixture")
            path
        end)
    try
        rendered = V.visualize(pc; kind=:knn_graph, k=1, backend=:a35_capture, style)
        @test rendered.style === style
        @test rendered.spec.kind == :knn_graph
        @test V.render(spec; backend=:a35_capture, style=gray).style === gray
        @test V.render(spec; backend=:a35_capture).style == TamerOp.VisualStyle()
        # A bad renderer option is rejected before even dispatching the
        # intentionally unsupported mathematical object to a recipe.
        @test_throws ArgumentError V.visualize(nothing; backend=:a35_capture, style=:bad)
        @test_throws ArgumentError V.render(spec; backend=:a35_capture, style=:bad)
        mktempdir() do dir
            invalid_dir = joinpath(dir, "invalid")
            before = length(captured)
            @test_throws ArgumentError V.save_visual(joinpath(invalid_dir, "one.txt"), nothing;
                backend=:a35_capture, style=:bad)
            @test_throws ArgumentError V.save_visual(invalid_dir, "one", nothing;
                backend=:a35_capture, format=:txt, style=:bad)
            @test_throws ArgumentError V.save_visuals(invalid_dir,
                [(; stem="one", obj=nothing)]; backend=:a35_capture, format=:txt, style=:bad)
            @test !ispath(invalid_dir)
            @test length(captured) == before
            batch = V.save_visuals(dir, [
                (; stem="inherited", obj=spec),
                (; stem="overridden", obj=spec, style=gray),
            ]; backend=:a35_capture, format=:txt, style)
            @test length(batch) == 2
            @test all(r -> isfile(V.export_path(r)), batch)
            saved = filter(x -> x.operation === :save, captured)
            @test saved[end-1].style === style
            @test saved[end].style === gray
            @test saved[end].spec === spec
        end
    finally
        delete!(V._VISUAL_RENDERERS, :a35_capture)
        delete!(V._VISUAL_SAVERS, :a35_capture)
    end
end

@testset "A35 default text contrast" begin
    # Independent sRGB contrast calculation from WCAG 2.2 SC 1.4.3:
    # https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html
    # This checks default text colors on their two default surfaces only;
    # custom palettes and complete accessibility require separate review.
    linear_srgb(c) = c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055)^2.4
    function relative_luminance(hex_color)
        rgb = ntuple(i -> parse(Int, hex_color[(2*i):(2*i+1)]; base=16) / 255, 3)
        r, g, b = linear_srgb.(rgb)
        return 0.2126*r + 0.7152*g + 0.0722*b
    end
    function contrast_ratio(foreground, background)
        first_luminance, second_luminance = relative_luminance(foreground), relative_luminance(background)
        return (max(first_luminance,second_luminance) + 0.05) /
            (min(first_luminance,second_luminance) + 0.05)
    end
    @test relative_luminance("#000000") == 0.0
    @test isapprox(relative_luminance("#FFFFFF"),1.0)
    @test isapprox(contrast_ratio("#000000","#FFFFFF"),21.0)
    @test contrast_ratio("#FFFFFF","#FFFFFF") == 1.0
    @test contrast_ratio("#767676","#FFFFFF") >= 4.5
    @test contrast_ratio("#777777","#FFFFFF") < 4.5
    text_roles = (:source,:target,:both,:selected,:inactive,:foreground,:muted,:support,:error)
    for palette in (:accessible,:grayscale)
        colors = TamerOp.VisualStyle(; palette).colors
        for background in (:background,:surface), role in text_roles
            @test contrast_ratio(getproperty(colors,role),getproperty(colors,background)) >= 4.5
        end
    end
end

@testset "A35 two-square mathematics survives palette changes" begin
    V, A = TamerOp.Visualization, TamerOp.Advanced
    opts = A.EncodingOptions(; backend=:pl_backend, poset_kind=:signature, field=CM.QQField())
    enc = TamerOp.encode([A.BoxUpset([0.0,0.0]), A.BoxUpset([1.0,1.0])],
        [A.BoxDownset([2.0,2.0]), A.BoxDownset([3.0,3.0])], QQ[1 0;0 1], opts)
    a, b, c = (1//2,1//2), (3//2,3//2), (5//2,5//2)
    specs = [V.visual_spec(enc; kind=:module_inspector, parameter_pair=pair,
        box=([-1,-1],[4,4])) for pair in ((a,b), (b,c), (a,c))]
    maps = [copy(spec.metadata.inspection.matrix) for spec in specs]
    @test size.(maps) == [(2,1), (1,2), (1,1)]
    @test [spec.metadata.inspection.rank for spec in specs] == [1,1,0]
    @test maps[2] * maps[1] == maps[3] == reshape(QQ[0], 1, 1)
    collision = V.VisualStyle(; colors=(source=:black, target=:black, both=:black))
    styles = (V.VisualStyle(), V.VisualStyle(; palette=:grayscale), collision)
    @test V._visual_color(collision, V._VisualRole(:source)) ==
        V._visual_color(collision, V._VisualRole(:target))
    @test V._visual_marker(V._VisualRole(:source)) != V._visual_marker(V._VisualRole(:target))
    @test !(V._visual_marker(V._VisualRole(:both)) in
        (V._visual_marker(V._VisualRole(:source)), V._visual_marker(V._VisualRole(:target))))
    V._register_visual_backend!(:a35_math_capture;
        render=(spec; display, figure, size, style) -> (; spec, style))
    try
        for (i, spec) in enumerate(specs), style in styles
            inspection = spec.metadata.inspection
            table = only(layer for layer in last(spec.panels).layers if layer isa V.MatrixLayer)
            entries, rows, columns = copy(table.entries), copy(table.row_labels), copy(table.column_labels)
            rendered = V.render(spec; backend=:a35_math_capture, style)
            @test rendered.spec === spec
            @test spec.metadata.inspection === inspection
            @test inspection.matrix == maps[i]
            @test (table.entries, table.row_labels, table.column_labels) == (entries, rows, columns)
        end
        presentation = V.visual_spec(enc; kind=:presentation_inspector, parameter_pair=(a,c),
            box=([-1,-1],[4,4]))
        full = only(p for p in presentation.panels if p.title == "Full coefficient matrix Phi")
        roles = copy(full.metadata.cell_roles)
        @test :source in roles && :target in roles
        for style in styles
            V.render(presentation; backend=:a35_math_capture, style)
            @test full.metadata.cell_roles == roles
            @test full.metadata.matrix == QQ[1 0;0 1]
        end
        # Equal endpoints share one visible role without losing either exact
        # query. Equality is mathematical, not equality after drawing rounds.
        for kind in (:module_inspector, :presentation_inspector)
            same = V.visual_spec(enc; kind, parameter_pair=(a,a), box=([-1,-1],[4,4]))
            panels = filter(p -> p.kind in (:query_overlay,:presentation_support), same.panels)
            @test !isempty(panels)
            for panel in panels
                markers = [layer for layer in panel.layers if layer isa V.PointLayer &&
                    layer.color isa V._VisualRole && layer.color.kind in (:source,:target,:both)]
                @test length(markers) == 1
                @test only(markers).color.kind === :both
                @test only(markers).points == [(0.5,0.5)]
                @test length(panel.metadata.query_readout) == 2
                @test all(q -> q.point == a, panel.metadata.query_readout)
            end
        end
        epsilon = 1 // (big(1) << 56)
        nearby = (a[1]+epsilon,a[2]+epsilon)
        rounded = V.visual_spec(enc; kind=:module_inspector,
            parameter_pair=(a,nearby), box=([-1,-1],[4,4]))
        region = only(p for p in rounded.panels if p.kind === :query_overlay)
        markers = [layer for layer in region.layers if layer isa V.PointLayer &&
            layer.color isa V._VisualRole && layer.color.kind in (:source,:target,:both)]
        @test [layer.color.kind for layer in markers] == [:source,:target]
        @test markers[1].points == markers[2].points
        @test region.metadata.query_readout[1].point != region.metadata.query_readout[2].point
        @test !isempty(region.metadata.query_collisions)
    finally
        delete!(V._VISUAL_RENDERERS, :a35_math_capture)
    end
end

@testset "A35 native style attributes and SVG exports" begin
    if Base.find_package("CairoMakie") === nothing
        @test_skip false
    else
        @eval import CairoMakie
        V, M = TamerOp.Visualization, CairoMakie.Makie
        point_panel = V.VisualizationSpec(:a35_roles; title="Style panel",
            subtitle="Source and target remain distinguishable when their colors coincide.",
            layers=V.AbstractVisualizationLayer[
                V.PointLayer([(0.0,0.0)], V._VisualRole(:source), 1.0, 8.0),
                V.PointLayer([(1.0,1.0)], V._VisualRole(:target), 1.0, 8.0),
                V.SegmentLayer([(0.0,0.0,1.0,1.0)], V._VisualRole(:edge), 1.0, 2.0),
            ])
        matrix_panel = V.VisualizationSpec(:a35_matrix; title="Exact coefficients",
            layers=V.AbstractVisualizationLayer[V.MatrixLayer(reshape(["1/3"],1,1), ["t1"], ["s1"])],
            metadata=(; panel_style=:matrix, row_roles=[:target], column_roles=[:source]))
        spec = V.VisualizationSpec(:a35_panels; panels=[point_panel, matrix_panel])
        default = V.VisualStyle()
        enlarged = V.VisualStyle(; fontsize=24, linewidth_scale=2, markersize_scale=1.5,
            gap=18, padding=24, colors=(source=:black, target=:black))
        figures = [V.render(spec; backend=:cairomakie, style, size=(1000,650))
            for style in (default, enlarged)]
        for fig in figures
            M.update_state_before_display!(fig)
        end
        axes = [only(item for item in fig.content if item isa M.Axis) for fig in figures]
        @test isapprox(axes[2].xlabelsize[] / axes[1].xlabelsize[],1.5)
        scatters = [[p for p in ax.scene.plots if p isa M.Scatter] for ax in axes]
        @test length.(scatters) == [2,2]
        @test all(isapprox.(scatters[2][1].markersize[], 1.5 .* scatters[1][1].markersize[]))
        @test scatters[2][1].marker[] != scatters[2][2].marker[]
        @test M.to_color(scatters[2][1].color[]) == M.to_color(scatters[2][2].color[])
        lines = [only(p for p in ax.scene.plots if p isa M.Lines) for ax in axes]
        @test isapprox(lines[2].linewidth[] / lines[1].linewidth[],2)
        labels = [[item for item in fig.content if item isa M.Label] for fig in figures]
        entries = [only(item for item in blocks if item.text[] == "1/3") for blocks in labels]
        @test isapprox(entries[2].fontsize[] / entries[1].fontsize[],1.5)
        @test M.to_font(entries[2].font[]) == M.to_font(enlarged.mono_font)
        texts = [String(item.text[]) for item in labels[2]]
        @test any(t -> startswith(t,"s1") && occursin("source",t), texts)
        @test any(t -> startswith(t,"t1") && occursin("target",t), texts)
        title = only(item for item in labels[2] if item.text[] == "Style panel")
        @test title.word_wrap[]
        @test !title.tellwidth[]
        supplied = M.Figure(; size=(811,509), figure_padding=1, backgroundcolor=:black)
        supplied_spec = V.VisualizationSpec(:a35_supplied; title="Supplied figure",
            panels=spec.panels)
        @test V.render(supplied_spec; backend=:cairomakie, figure=supplied, style=enlarged) === supplied
        M.update_state_before_display!(supplied)
        @test Tuple(M.widths(M.viewport(supplied.scene)[])) == (811,509)
        @test M.to_color(supplied.scene.backgroundcolor[]) == M.to_color(enlarged.colors.background)
        padding = supplied.layout.alignmode[].padding
        @test all(value -> value == enlarged.padding,
            (padding.left,padding.right,padding.bottom,padding.top))
        @test !isempty(supplied.layout.addedrowgaps)
        @test !isempty(supplied.layout.addedcolgaps)
        @test all(gap -> gap.x == enlarged.gap, supplied.layout.addedrowgaps)
        @test all(gap -> gap.x == enlarged.gap, supplied.layout.addedcolgaps)
        gray = V.VisualStyle(; palette=:grayscale)
        grayfig = V.render(point_panel; backend=:cairomakie, style=gray)
        grayax = only(item for item in grayfig.content if item isa M.Axis)
        for plot in grayax.scene.plots
            plot isa M.Scatter || continue
            color = M.to_color(plot.color[])
            @test isapprox(color.r,color.g)
            @test isapprox(color.g,color.b)
        end
        # Rendering a custom style does not leak a global theme into the next
        # ordinary figure.
        next_default = V.render(point_panel; backend=:cairomakie)
        defaultax = only(item for item in next_default.content if item isa M.Axis)
        defaultscatter = first(p for p in defaultax.scene.plots if p isa M.Scatter)
        @test defaultax.xlabelsize[] == axes[1].xlabelsize[]
        @test defaultscatter.markersize[] == scatters[1][1].markersize[]
        @test M.to_color(defaultscatter.color[]) == M.to_color(scatters[1][1].color[])

        # A data-space disk is geometry, so changing the UI marker scale must
        # not change its radius or turn it into a source/target selection glyph.
        data_spec = V.VisualizationSpec(:a35_data_marker;
            layers=V.AbstractVisualizationLayer[V.PointLayer([(0.0,0.0)],
                V._VisualRole(:target), 1.0, 0.25, :viridis, :data)])
        datafig = V.render(data_spec; backend=:cairomakie, style=enlarged)
        dataax = only(item for item in datafig.content if item isa M.Axis)
        disk = only(p for p in dataax.scene.plots if p isa M.Scatter)
        @test disk.markerspace[] === :data
        @test all(value -> isapprox(value,0.25), disk.markersize[])
        @test disk.marker[] == scatters[1][1].marker[]

        # Missing cells and zero are different mathematical observations.
        # Numerical colormap overrides affect colors, never the plotted values.
        values = [0.0 NaN; 1.0 2.0]
        heatmap_spec = V.VisualizationSpec(:a35_heatmap;
            layers=V.AbstractVisualizationLayer[V.HeatmapLayer([0.0,1.0], [0.0,1.0],
                values, :viridis, 1.0, "dimension", false)])
        for (appearance, expected_map) in ((gray,:grays),
                (V.VisualStyle(; palette=:grayscale, colormap=:plasma),:plasma))
            heatfig = V.render(heatmap_spec; backend=:cairomakie, style=appearance)
            heatax = only(item for item in heatfig.content if item isa M.Axis)
            heat = only(p for p in heatax.scene.plots if p isa M.Heatmap)
            @test isequal(heat[3][],permutedims(values))
            @test isequal(only(heatmap_spec.layers).values,values)
            @test M.to_colormap(heat.colormap[]) == M.to_colormap(expected_map)
            @test M.to_color(heat.nan_color[]) != first(M.to_colormap(expected_map))
        end
        mktempdir() do dir
            path = joinpath(dir,"style.svg")
            @test V.save_visual(path, spec; backend=:cairomakie, style=enlarged) == path
            @test occursin("<svg", read(path,String))
            @test filesize(path) > 1000
            batch = V.save_visuals(dir, [
                (; stem="accessible", obj=point_panel),
                (; stem="grayscale", obj=point_panel, style=gray),
            ]; backend=:cairomakie, format=:svg, style=enlarged)
            @test length(batch) == 2
            @test all(r -> occursin("<svg", read(V.export_path(r),String)), batch)
        end
    end
end

@testset "A35 query annotations preserve mathematical coordinates" begin
    if Base.find_package("CairoMakie") === nothing
        @test_skip false
    else
        @eval import CairoMakie
        V, A, M = TamerOp.Visualization, TamerOp.Advanced, CairoMakie.Makie
        opts = A.EncodingOptions(; backend=:pl_backend, poset_kind=:signature, field=CM.QQField())
        enc = TamerOp.encode([A.BoxUpset([0.0,0.0]), A.BoxUpset([1.0,1.0])],
            [A.BoxDownset([2.0,2.0]), A.BoxDownset([3.0,3.0])], QQ[1 0;0 1], opts)
        point = (1//2,5//2)
        presentation = V.visual_spec(enc; kind=:presentation_inspector, point,
            upset=1, downset=2, box=([-1,-1],[4,4]))
        support = first(panel for panel in presentation.panels if panel.kind === :presentation_support)
        label_index = support.metadata.query_label_layer
        @test label_index isa Int
        query_layer = support.layers[label_index]
        @test query_layer isa V.TextLayer
        @test query_layer.positions == [(0.5,2.5)]
        membership_layer = only(layer for (i,layer) in enumerate(support.layers) if
            layer isa V.TextLayer && i != label_index)
        membership_positions = copy(membership_layer.positions)
        @test (0.5,2.5) in membership_positions
        @test only(support.metadata.query_readout).point == point
        @test only(support.metadata.query_readout).display_point == (0.5,2.5)
        full = only(panel for panel in presentation.panels if panel.title == "Full coefficient matrix Phi")
        coefficients = copy(full.metadata.matrix)
        styles = (V.VisualStyle(), V.VisualStyle(; fontsize=24,markersize_scale=2))
        query_offsets = Any[]
        membership_offsets = Any[]
        for style in styles
            figure = V.render(support; backend=:cairomakie, style, size=(720,560))
            axis = only(block for block in figure.content if block isa M.Axis)
            textplots = [plot for plot in axis.scene.plots if plot isa M.Text]
            @test length(textplots) == length(membership_layer.labels) + 1
            query = last(textplots)
            # Makie normalizes positional text arguments into input_text;
            # text is the separate keyword fallback, unused by this renderer.
            @test query.input_text[] == query_layer.labels
            @test query.space[] === :data
            @test query.markerspace[] === :pixel
            @test query.offset[][1] > 0 && query.offset[][2] > 0
            push!(query_offsets,query.offset[])
            # Support membership and query annotations occupy opposite sides
            # of their data anchors; their coordinates are never nudged.
            for membership in textplots[1:end-1]
                @test membership.space[] === :data
                @test membership.markerspace[] === :pixel
                @test membership.offset[][1] < 0 && membership.offset[][2] < 0
            end
            push!(membership_offsets,first(textplots).offset[])
            @test query_layer.positions == [(0.5,2.5)]
            @test membership_layer.positions == membership_positions
            @test only(support.metadata.query_readout).point == point
            @test full.metadata.matrix == coefficients == QQ[1 0;0 1]
        end
        @test all(isapprox.(query_offsets[2],2 .* query_offsets[1]))
        @test all(isapprox.(membership_offsets[2],2 .* membership_offsets[1]))
        ordinary = V.VisualizationSpec(:a35_unmarked_text;
            layers=V.AbstractVisualizationLayer[V.TextLayer(["ordinary annotation"],
                [(0.5,2.5)], V._VisualRole(:foreground), 10.0)])
        figure = V.render(ordinary; backend=:cairomakie, style=last(styles))
        axis = only(block for block in figure.content if block isa M.Axis)
        textplot = only(plot for plot in axis.scene.plots if plot isa M.Text)
        @test all(iszero,textplot.offset[])
        @test only(ordinary.layers).positions == [(0.5,2.5)]
    end
end

@testset "A35 live styles and native HTML controls" begin
    if Base.find_package("WGLMakie") === nothing
        @test_skip false
    else
        @eval import WGLMakie
        V, A = TamerOp.Visualization, TamerOp.Advanced
        E = Base.get_extension(TamerOp, :TamerOpWGLMakieExt)
        B, M = WGLMakie.Bonito, WGLMakie.Makie
        opts = A.EncodingOptions(; backend=:pl_backend, poset_kind=:signature, field=CM.QQField())
        enc = TamerOp.encode([A.BoxUpset([0.0,0.0]), A.BoxUpset([1.0,1.0])],
            [A.BoxDownset([2.0,2.0]), A.BoxDownset([3.0,3.0])], QQ[1 0;0 1], opts)
        s = V.inspection_session(enc; box=([-1,-1],[4,4]))
        V.select_inspection!(s; parameter_pair=((1//2,1//2),(5//2,5//2)), view=:presentation)
        before = V.inspection_selection(s)
        style = V.VisualStyle(; palette=:grayscale, fontsize=22, gap=18, padding=24,
            markersize_scale=1.5, colors=(source=:black,target=:black))
        app = V.visualize(s; backend=:wglmakie, style, size=(960,480))
        ui = E._inspection_ui(app)
        client = B.Session(B.NoConnection(); asset_server=B.NoServer())
        try
            @test ui.style === style
            @test ui.figure_size == (960,480)
            @test V.inspection_selection(s) == before
            M.update_state_before_display!(ui.figure)
            @test Tuple(M.widths(M.viewport(ui.figure.scene)[])) == (960,480)
            # Native DOM and WGL serialization exercise the actual controls;
            # this is deliberately not a claim about browser JavaScript events.
            dom = B.session_dom(client, app; init=false)
            html = sprint(show,MIME"text/html"(),dom)
            @test occursin("<canvas",html)
            @test occursin(ui.dom_id * "-point-x",html)
            @test occursin(r"font-size:\s*22(?:\.0)?px",html)
            @test occursin("DejaVu Sans",html)
            @test occursin("DejaVu Sans Mono",html)
            @test occursin("source",lowercase(html)) && occursin("target",lowercase(html))
            @test !isempty(B.serialize_binary(client,B.fused_messages!(client)))
            ui.controls.point_x.value[] = "1/2"
            ui.controls.point_y.value[] = "5/2"
            ui.controls.endpoint.option_index[] = 1
            ui.controls.select_point.value[] = true
            @test only(V.inspection_selection(s).query_points) == (1//2,5//2)
            @test isempty(ui.error_text[])
            @test isempty(V.inspection_summary(s).listener_errors)
            @test ui.style === style
            # Signed zero is the same mathematical parameter even though
            # isequal distinguishes its floating representations.
            V.select_inspection!(s; parameter_pair=((0.0,0.0),(-0.0,0.0)), input=:pointer)
            selected_points = V.inspection_selection(s).query_points
            @test selected_points[1] == selected_points[2]
            @test !isequal(selected_points[1],selected_points[2])
            @test length(ui.markers.region.both[]) == 1
            @test isempty(ui.markers.region.source[]) && isempty(ui.markers.region.target[])
            @test isempty(V.inspection_summary(s).listener_errors)
            supplied = M.Figure(; size=(845,455), figure_padding=1, backgroundcolor=:black)
            supplied_app = V.visualize(s; backend=:wglmakie, figure=supplied, style)
            supplied_ui = E._inspection_ui(supplied_app)
            M.update_state_before_display!(supplied)
            @test supplied_ui.figure === supplied
            @test supplied_ui.figure_size == (845,455)
            @test Tuple(M.widths(M.viewport(supplied.scene)[])) == (845,455)
            @test M.to_color(supplied.scene.backgroundcolor[]) == M.to_color(style.colors.background)
            padding = supplied.layout.alignmode[].padding
            @test all(value -> value == style.padding,
                (padding.left,padding.right,padding.bottom,padding.top))
            @test !isempty(supplied.layout.addedcolgaps)
            @test all(gap -> gap.x == style.gap, supplied.layout.addedcolgaps)
            @test isequal(V.inspection_selection(s).query_points,selected_points)
        finally
            close(client)
            V.close_inspection!(s)
        end
    end
end
