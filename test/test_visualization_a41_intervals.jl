# Exact interval adapter and display semantics, independent of a rendering backend.
@testset "A41 interval records, multiplicities and display bounds" begin
    V = TamerOp.Visualization
    bars = Dict((0//1,2//1)=>3,(1//1,4//1)=>2,(8//1,9//1)=>1)
    spec = V.visual_spec(bars;kind=:barcode,window=(1//2,3//1),interval=2)
    @test V.check_visual_spec(spec).valid
    @test spec.metadata.total_groups == 3
    @test spec.metadata.displayed_groups == 2
    @test spec.metadata.offscreen_groups == 1
    @test spec.metadata.total_multiplicity == 6
    @test spec.metadata.displayed_multiplicity == 5
    @test spec.metadata.records[1].birth === 0//1
    @test spec.metadata.records[2].death === 4//1
    @test spec.metadata.endpoint_display == ((:offscreen,:finite),(:finite,:offscreen))
    @test spec.metadata.selected_record.id == 2
    @test spec.metadata.interval_segments == ((0.5,1.0,2.0,1.0),(1.0,2.0,3.0,2.0))
    diagram = V.visual_spec(bars;kind=:persistence_diagram,window=(1//2,3//1))
    @test diagram.metadata.interval_points == ((0.5,2.0),(1.0,3.0))
    @test diagram.metadata.records == spec.metadata.records
    @test all(r -> r.right_status === :finite,diagram.metadata.records)
    # The selected stable ID gets a row even when it is beyond the display budget.
    limited = V.visual_spec(bars;kind=:barcode,max_intervals=1,interval=3)
    @test limited.metadata.interval_ids == (3,)
    @test limited.metadata.omitted_groups == 2
    @test limited.metadata.interval_segments == ((8.0,1.0,9.0,1.0),)
    @test limited.metadata.all_interval_ids == (1,2,3)
    @test isempty(V.visual_spec(bars;window=(10,11)).metadata.interval_ids)
    @test V.visual_spec(bars;window=(10,11),interval=1).metadata.selected_interval == 1
    @test V.visual_spec([(1,2),(1,2),(3,3)]).metadata.total_multiplicity == 2
    @test V.visual_spec([(1,2),(1,2)]).metadata.total_groups == 1
    @test V.check_visual_spec(V.visual_spec(Tuple{Int,Int}[])).valid
    for kwargs in ((;max_intervals=0),(;max_intervals=true),(;interval=true),(;interval=7),
                   (;window=(3,2)),(;window=(0,Inf)),(;window=(NaN,1)))
        @test_throws ArgumentError V.visual_spec(bars;kwargs...)
    end
    @test_throws ArgumentError V.visual_spec(Dict((2,1)=>1))
    @test_throws ArgumentError V.visual_spec(Dict((1,2)=>-1))
    @test_throws ArgumentError V.visual_spec([(NaN,2.0)])
end

@testset "A41 barcode endpoint bounds and exact anchors across renderers" begin
    V = TamerOp.Visualization
    records = V._group_interval_records([
        V._interval_record(-Inf,1.0,1;left_closed=false),
        V._interval_record(-Inf,Inf,1;left_closed=false),
        V._interval_record(0.0,Inf,1),
        V._interval_record(-1//1,2//1,1),
        V._interval_record(-1//1,2//1,1;left_status=:censored,right_status=:censored)])
    spec = first(V._interval_panels(records;window=(0//1,1//1)))
    layers = [layer for layer in spec.layers if layer isa V.TextLayer]
    @test count(layer -> only(layer.labels) == "< -Inf",layers) == 2
    @test count(layer -> only(layer.labels) == "> +Inf",layers) == 2
    @test count(layer -> only(layer.labels) == "?",layers) == 2
    anchors = [only(layer.positions) for layer in layers]
    exact_records = spec.metadata.records
    exact_segments = spec.metadata.interval_segments
    for (backend,package) in ((:cairomakie,"CairoMakie"),(:wglmakie,"WGLMakie"))
        if Base.find_package(package) === nothing
            @test_skip false
            continue
        end
        if backend === :cairomakie
            @eval import CairoMakie
            M = CairoMakie.Makie
        else
            @eval import WGLMakie
            M = WGLMakie.Makie
        end
        for fontsize in (18,24)
            style = V.VisualStyle(;fontsize,linewidth_scale=fontsize == 24 ? 2 : 1)
            fig = V.render(spec;backend,size=(760,560),style)
            try
                ax = only(block for block in fig.content if block isa M.Axis)
                texts = [plot for plot in ax.scene.plots if plot isa M.Text]
                @test length(texts) == length(layers)
                selected_linewidth = 6 * style.linewidth_scale
                # Live selection adds a wider stroke after the annotation plots.
                # Reproduce that overlay independently of pointer event routing.
                for segment in exact_segments
                    M.lines!(ax,[segment[1],segment[3]],[segment[2],segment[4]];
                        color=:black,linewidth=selected_linewidth,depth_shift=-0.002)
                end
                for dimensions in ((760,560),(600,560),(420,560))
                    M.resize!(fig,dimensions...)
                    M.update_state_before_display!(fig)
                    viewport = M.widths(M.viewport(ax.scene)[])
                    for (plot,layer,anchor) in zip(texts,layers,anchors)
                        @test plot.input_text[] == layer.labels
                        rendered_anchor = only(plot.positions[])
                        # Makie stores display positions as Float32; compare to
                        # that conversion while the specification stays exact.
                        @test Tuple(rendered_anchor)[1:2] == map(eltype(rendered_anchor),anchor)
                        @test plot.offset[][1] == 0
                        @test plot.offset[][2] > selected_linewidth/2
                        # These are measured glyph bounds in axis-local pixels,
                        # including font metrics, alignment and final projection.
                        box = M.boundingbox(plot,:pixel)
                        lo,hi = minimum(box)[1:2],maximum(box)[1:2]
                        @test all(isfinite,lo) && all(isfinite,hi)
                        @test minimum(lo) >= -0.5
                        @test maximum(hi .- viewport) <= 0.5
                        bar_y = M.project(ax.scene,M.Point2d(anchor))[2]
                        @test lo[2] > bar_y + selected_linewidth/2
                    end
                    @test [only(layer.positions) for layer in layers] == anchors
                    @test spec.metadata.records === exact_records
                    @test spec.metadata.interval_segments === exact_segments
                end
            finally
                empty!(fig)
            end
        end
    end
end

@testset "A41 diagram tick glyphs stay separate after resizing" begin
    V = TamerOp.Visualization
    records = V._group_interval_records([
        V._interval_record(-Inf,1.0,1;left_closed=false),
        V._interval_record(-Inf,Inf,1;left_closed=false),
        V._interval_record(0.0,Inf,1)])
    specs = (last(V._interval_panels(records;window=(-1,2))),
        V.visual_spec(TamerOp.OrdinaryPersistence.PersistenceDiagram(
            [[(2,1),(2,1)]],[[2]];order=:superlevel)))
    for (backend,package) in ((:cairomakie,"CairoMakie"),(:wglmakie,"WGLMakie"))
        if Base.find_package(package) === nothing
            @test_skip false
            continue
        end
        if backend === :cairomakie
            @eval import CairoMakie
            M = CairoMakie.Makie
        else
            @eval import WGLMakie
            M = WGLMakie.Makie
        end
        for (fixture,spec) in enumerate(specs), fontsize in (18,24)
            ticks = (spec.axes.xticks,spec.axes.yticks)
            exact_records,points = spec.metadata.records,spec.metadata.interval_points
            fig = V.render(spec;backend,size=(760,640),style=V.VisualStyle(;fontsize))
            try
                ax = only(block for block in fig.content if block isa M.Axis)
                # The ordinary diagram has an extra order/infinity subtitle.
                # At narrow widths give that wrapped heading its normal panel
                # height; arbitrary tiny canvases cannot fit every decoration.
                sizes = fixture == 1 ? ((760,640),(420,420),(760,640),(480,460)) :
                    ((760,640),(420,560),(760,640),(480,560))
                for dimensions in sizes
                    M.resize!(fig,dimensions...)
                    M.update_state_before_display!(fig)
                    for (axis,expected_ticks) in zip((ax.xaxis,ax.yaxis),ticks)
                        @test axis.tickvalues[] == eltype(axis.tickvalues[]).(expected_ticks[1])
                        @test axis.ticklabels[] == expected_ticks[2]
                        # Bounds include final font metrics, rotation, alignment
                        # and display positions. Test both x and y decorations.
                        boxes = M.string_boundingboxes(axis.elements[:ticklabels])
                        @test length(boxes) == length(expected_ticks[2])
                        for i in eachindex(boxes), j in (i+1):length(boxes)
                            gap = max.(minimum(boxes[i])[1:2],minimum(boxes[j])[1:2]) .-
                                min.(maximum(boxes[i])[1:2],maximum(boxes[j])[1:2])
                            @test max(gap...) >= 1
                        end
                    end
                    if fixture == 1 && fontsize == 24
                        dimensions == (420,420) && (@test isapprox(ax.xticklabelrotation[], pi/2))
                        dimensions == (760,640) && (@test ax.xticklabelrotation[] == 0)
                    end
                    rotation = ax.xticklabelrotation[]
                    M.update_state_before_display!(fig)
                    @test ax.xticklabelrotation[] == rotation
                    @test ax.xticks[] == ticks[1] && ax.yticks[] == ticks[2]
                    @test spec.metadata.records === exact_records
                    @test spec.metadata.interval_points === points
                end
            finally
                empty!(fig)
            end
        end
    end
end

@testset "A41 infinite, censored and decorated endpoint evidence" begin
    V = TamerOp.Visualization
    spec = V.visual_spec([(-Inf,1.0),(0.0,Inf),(0.0,10.0)];window=(-1,2))
    @test spec.metadata.endpoint_display == ((:essential,:finite),(:finite,:offscreen),(:finite,:essential))
    @test spec.metadata.records[2].death == 10
    @test spec.metadata.records[3].death == Inf
    @test spec.metadata.interval_segments[2][3] == 2
    @test spec.metadata.interval_segments[3][3] > 2
    @test "-Inf" in spec.axes.xticks[2]
    @test "+Inf" in spec.axes.xticks[2]
    @test V.check_visual_spec(spec).valid
    records = V._group_interval_records([
        V._interval_record(0//1,1//1,1;right_closed=true),
        V._interval_record(0//1,1//1,2),
        V._interval_record(1//1,1//1,1;right_closed=true),
        V._interval_record(-1//1,2//1,1;left_status=:censored,right_status=:censored)])
    barcode,diagram = V._interval_panels(records;window=(-1,2))
    @test barcode.metadata.total_groups == 4
    @test barcode.metadata.total_multiplicity == 5
    @test barcode.metadata.endpoint_display[1] == (:censored,:censored)
    @test count(r -> r.singleton,barcode.metadata.records) == 1
    @test count(p -> p == (0.0,1.0),diagram.metadata.interval_points) == 2
    collision_labels = [label for layer in diagram.layers if layer isa V.TextLayer for label in layer.labels if occursin("#2",label)]
    @test any(label -> occursin("#3",label) && occursin("\n",label),collision_labels)
    tiny = big(1)//big(2)^60
    exact = V._group_interval_records([V._interval_record(1//1,1+tiny,1)])
    @test V._interval_view(exact).coordinate_collisions == (1,)
    @test only(V._interval_view(exact).records).death == 1+tiny
    # The exact endpoint remains available even when an explicit window avoids overflow.
    huge = big(10)^400
    @test only(V.visual_spec([(0//1,huge//1)];window=(0,1)).metadata.records).death == huge
end

@testset "A41 ordinary and packed family adapters" begin
    V = TamerOp.Visualization
    O = TamerOp.OrdinaryPersistence
    IC = TamerOp.InvariantCore
    packed = IC.PackedBarcode([IC.EndpointPair(0//1,2//1),IC.EndpointPair(0//1,2//1)], [2,3])
    @test V.visual_spec(packed).metadata.total_multiplicity == 5
    @test V.visual_spec(packed;kind=:persistence_diagram).metadata.total_groups == 1
    raw = Dict((0//1,2//1)=>1)
    family = TamerOp.SliceInvariants.SliceBarcodesResult(reshape([raw,raw],1,2),ones(1,2),[(1,1)],[(0,0),(0,1)])
    @test V.visual_spec(family;index=(1,2)).metadata.total_groups == 1
    @test V.visual_spec(family;kind=:persistence_diagram,index=2).metadata.slice_index == 2
    @test_throws ArgumentError V.visual_spec(family)
    @test_throws ArgumentError V.visual_spec(family;index=(3,2))
    fibered = TamerOp.Fibered2D.FiberedSliceResult([1,2],[0//1,1//1,2//1],raw)
    @test only(V.visual_spec(fibered;kind=:barcode).metadata.records).left_status === :censored
    @test only(V.visual_spec(fibered;kind=:barcode).metadata.records).right_status === :censored
    projected = TamerOp.Fibered2D.ProjectedBarcodesResult([raw],[7],[(1,1)])
    @test V.visual_spec(projected;kind=:persistence_diagram).metadata.projection_index == 7
    diagram = O.PersistenceDiagram([[(0//1,2//1),(0//1,2//1)]],[[1//1]])
    spec = V.visual_spec(diagram)
    @test spec.metadata.total_groups == 2
    @test spec.metadata.total_multiplicity == 3
    @test spec.metadata.records[1].members == ((dim=0,kind=:finite,index=1),(dim=0,kind=:finite,index=2))
    @test spec.metadata.records[2].members == ((dim=0,kind=:essential,index=1),)
    @test spec.metadata.records[2].death == Inf
    super = O.PersistenceDiagram([[(2//1,0//1)]],[[1//1]];order=:superlevel)
    super_spec = V.visual_spec(super)
    @test super_spec.metadata.order === :superlevel
    @test super_spec.metadata.records[1].birth == -Inf
    @test super_spec.metadata.interval_points[2] == (2.0,0.0)
    @test super_spec.metadata.records[2].left_closed == false
    @test super_spec.metadata.records[2].right_closed == true
    @test "-Inf" in super_spec.axes.yticks[2]
    zeros = O.PersistenceDiagram([Tuple{Float64,Float64}[]],[[-0.0,0.0]])
    @test V.visual_spec(zeros).metadata.total_groups == 1
    @test V.visual_spec(zeros).metadata.total_multiplicity == 2
    @test isequal(V.visual_spec(zeros).metadata.essential_births,[-0.0,0.0])
end

@testset "A41 diagram annotation bounds and separation across renderers" begin
    V = TamerOp.Visualization
    O = TamerOp.OrdinaryPersistence
    # These reproduce right-edge superlevel labels, adjacent infinite/finite
    # display lanes, and coincident multiline labels at the upper edge.
    ordinary = V.visual_spec(O.PersistenceDiagram([[(2,1),(2,1)]],[[2]];order=:superlevel))
    tails = V._group_interval_records([
        V._interval_record(-Inf,1.0,1;left_closed=false),
        V._interval_record(-Inf,Inf,1;left_closed=false),
        V._interval_record(0.0,Inf,1)])
    tail_spec = last(V._interval_panels(tails;window=(1//4,3//4)))
    squares = V._group_interval_records([
        V._interval_record(0,2,1;right_closed=true),
        V._interval_record(1,3,1;right_closed=true)])
    square_spec = last(V._interval_panels(squares;window=(3//2,7//4)))
    for (backend,package) in ((:cairomakie,"CairoMakie"),(:wglmakie,"WGLMakie"))
        if Base.find_package(package) === nothing
            @test_skip false
            continue
        end
        if backend === :cairomakie
            @eval import CairoMakie
            M = CairoMakie.Makie
        else
            @eval import WGLMakie
            M = WGLMakie.Makie
        end
        for (fixture,spec) in ((:ordinary_superlevel,ordinary),(:certified_tails,tail_spec),(:square_global,square_spec)), fontsize in (16,24)
            exact_points = spec.metadata.interval_points
            exact_records = spec.metadata.records
            style = V.VisualStyle(;fontsize,palette=fontsize == 16 ? :accessible : :grayscale)
            fig = V.render(spec;backend,size=(600,560),style)
            try
                ax = only(filter(item -> item isa M.Axis,fig.content))
                annotations = filter(plot -> plot isa M.Annotation,ax.scene.plots)
                @test length(annotations) == 2
                @test count(plot -> plot.visible[],annotations) == 1
                annotation = only(filter(plot -> plot.visible[],annotations))
                optimizer = only(filter(plot -> !plot.visible[],annotations))
                for dimensions in ((600,560),(760,640))
                    M.resize!(fig,dimensions...)
                    M.update_state_before_display!(fig)
                    @test all(plot -> !plot.visible[],optimizer.plots)
                    texts = [plot for parent in annotations for plot in parent.plots if plot isa M.Text]
                    @test count(plot -> plot.visible[],texts) == 1
                    @test annotation.target_positions[] == M.Point2d.(collect(unique(exact_points)))
                    # Native text boxes include multiline glyph extents. Check the
                    # actual final pixel positions, not guessed character widths.
                    boxes = annotation.text_bbs[] .+ annotation.offsets[]
                    viewport = M.widths(M.viewport(ax.scene)[])
                    if any(box -> any(minimum(box) .< -0.5) || any(maximum(box) .> viewport .+ 0.5),boxes)
                        println(stderr,repr((;backend,fixture,fontsize,viewport,
                            anchors=annotation.screenpoints_target[],offsets=annotation.offsets[],boxes,
                            labels=annotation.text[])))
                        flush(stderr)
                    end
                    for box in boxes
                        lo,hi = minimum(box),maximum(box)
                        @test all(isfinite,lo) && all(isfinite,hi)
                        @test minimum(lo) >= -0.5
                        @test maximum(hi .- viewport) <= 0.5
                    end
                    for i in eachindex(boxes), j in (i+1):length(boxes)
                        overlap = min.(maximum(boxes[i]),maximum(boxes[j])) .-
                            max.(minimum(boxes[i]),minimum(boxes[j]))
                        @test min(overlap...) <= 0.5
                    end
                    offsets = copy(annotation.offsets[])
                    M.update_state_before_display!(fig)
                    @test annotation.offsets[] == offsets
                    @test spec.metadata.interval_points === exact_points
                    @test spec.metadata.records === exact_records
                end
            finally
                empty!(fig)
            end
        end
    end
end
