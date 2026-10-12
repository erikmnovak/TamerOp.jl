include("visualization_makie_batches.jl")
include("visualization_makie_annotations.jl")
include("visualization_makie_retained.jl")
include("visualization_makie_composition.jl")

function _visual_makie_handlers(TO, MakieMod; allow_save::Bool=true)
    Viz = TO.Visualization

    Point2 = if isdefined(MakieMod, :Point2d)
        MakieMod.Point2d
    else
        nothing
    end

    to_point2(p) = Point2 === nothing ? p : Point2(float(p[1]), float(p[2]))

    # Resolve appearance only here: the specification retains its original
    # coordinates, coefficient strings and semantic roles.
    function _color(style, color)
        resolved = Viz._visual_color(style, color)
        (style.palette === :grayscale && !(color isa Viz._VisualRole)) || return resolved
        rgba = MakieMod.to_color(resolved)
        gray = 0.2126 * rgba.r + 0.7152 * rgba.g + 0.0722 * rgba.b
        return MakieMod.RGBAf(gray, gray, gray, rgba.alpha)
    end

    _role_color(style, role::Symbol) = _color(style, Viz._VisualRole(role))
    _textsize(style, base) = base * style.fontsize / 16

    function _style_grid!(grid, style)
        MakieMod.rowgap!(grid, style.gap)
        MakieMod.colgap!(grid, style.gap)
        return grid
    end

    function _style_figure!(fig, style)
        # A supplied figure retains its identity and size, while its outer
        # appearance follows the same style as the newly rendered panels.
        fig.scene.backgroundcolor[] = MakieMod.to_color(_role_color(style, :background))
        fig.layout.alignmode[] = MakieMod.Outside(style.padding)
        _style_grid!(fig.layout, style)
        return fig
    end

    function _panel_heading!(grid, spec, style)
        row = 1
        if !isempty(spec.title)
            MakieMod.Label(grid[row, 1], spec.title; font=style.font,
                fontsize=_textsize(style, 18), color=_role_color(style, :foreground),
                tellwidth=false, halign=:left, word_wrap=true)
            row += 1
        end
        if !isempty(spec.subtitle)
            MakieMod.Label(grid[row, 1], spec.subtitle; font=style.font,
                fontsize=_textsize(style, 12), color=_role_color(style, :muted),
                tellwidth=false, halign=:left, word_wrap=true)
            row += 1
        end
        return row
    end

    _uses_axis3(spec) = get(Viz.visual_metadata(spec), :source_view_dimension, 0)==3 || any(layer -> layer isa Viz.Point3Layer || layer isa Viz.Segment3Layer || layer isa Viz.Polygon3Layer,
                            Viz.visual_layers(spec))

    function _legend_entries(spec)
        legend = Viz.visual_legend(spec)
        entries = get(legend, :entries, NamedTuple())
        entries isa AbstractVector && return entries
        entries isa Tuple && return collect(entries)
        if entries isa NamedTuple
            isempty(keys(entries)) && return NamedTuple[]
            out = NamedTuple[]
            for (label, entry) in pairs(entries)
                if entry isa NamedTuple
                    push!(out, merge((; label=String(label)), entry))
                else
                    push!(out, (; label=String(label), color=entry, style=:patch))
                end
            end
            return out
        end
        return NamedTuple[]
    end

    function _colorbar_layers(layers)
        return [layer for layer in layers if layer isa Viz.HeatmapLayer && layer.show_colorbar]
    end

    function _apply_axis_limits!(ax, spec)
        spec.axes.xlimits === nothing || MakieMod.xlims!(ax, spec.axes.xlimits...)
        spec.axes.ylimits === nothing || MakieMod.ylims!(ax, spec.axes.ylimits...)
        if _uses_axis3(spec)
            get(spec.axes, :zlimits, nothing) === nothing || MakieMod.zlims!(ax, spec.axes.zlimits...)
        end
        get(spec.axes, :xticks, nothing) === nothing || (ax.xticks = spec.axes.xticks)
        get(spec.axes, :yticks, nothing) === nothing || (ax.yticks = spec.axes.yticks)
        if !_uses_axis3(spec) && spec.axes.aspect in (:equal, :data)
            ax.aspect = MakieMod.DataAspect()
        end
        return ax
    end

    function _render_text_panel!(fig, grid, spec, style)
        # Compact readouts must not constrain neighboring plot heights.
        grid.tellheight[] = false
        grid.valign[] = :top
        row = _panel_heading!(grid, spec, style)
        for layer in Viz.visual_layers(spec)
            layer isa Viz.TextLayer || continue
            for line in layer.labels
                MakieMod.Label(grid[row, 1], line;
                               font=style.font, fontsize=_textsize(style, layer.textsize),
                               color=_color(style, layer.color),
                               tellwidth=false, halign=:left, word_wrap=true)
                row += 1
            end
        end
        return nothing
    end

    function _render_matrix_panel!(grid, spec, style; composition_cells=nothing)
        grid.tellheight[] = false
        grid.valign[] = :top
        layers = Viz.visual_layers(spec)
        length(layers) == 1 && only(layers) isa Viz.MatrixLayer ||
            throw(ArgumentError("A matrix panel must contain exactly one MatrixLayer and no other layers."))
        row = _panel_heading!(grid, spec, style)
        for layer in layers
            nr, nc = Base.size(layer.entries)
            length(layer.row_labels) == nr && length(layer.column_labels) == nc ||
                throw(ArgumentError("MatrixLayer label lengths must match its displayed matrix dimensions."))
            if nr == 0 || nc == 0
                shape = get(Viz.visual_metadata(spec), :matrix_size, (nr, nc))
                reason = if shape[1] == 0 && shape[2] == 0
                    "both source and target are zero-dimensional"
                elseif shape[1] == 0
                    "the target is zero-dimensional"
                elseif shape[2] == 0
                    "the source is zero-dimensional"
                else
                    "the displayed submatrix has no entries"
                end
                reason = get(Viz.visual_metadata(spec), :empty_matrix_reason, reason)
                MakieMod.Label(grid[row, 1], "Empty matrix ($(shape[1]) x $(shape[2])): $reason.";
                               font=style.font, fontsize=_textsize(style, 14),
                               color=_role_color(style, :foreground),
                               tellwidth=false, halign=:left, word_wrap=true)
                row += 1
                for (name, labels) in ((get(Viz.visual_metadata(spec), :matrix_row_heading, "Target basis"), layer.row_labels),
                                       ("Source basis", layer.column_labels))
                    isempty(labels) && continue
                    MakieMod.Label(grid[row, 1], "$name: " * join(labels, ", ");
                                   font=style.font, fontsize=_textsize(style, 12),
                                   color=_role_color(style, :foreground),
                                   tellwidth=false, halign=:left, word_wrap=true)
                    row += 1
                end
                continue
            end
            # Each coefficient remains text, including finite-field residues and
            # exact fractions. Layout performs no algebra or coefficient scaling.
            _visual_matrix_table!(MakieMod, Viz, grid[row,1], layer, spec.metadata, style;
                color=c->_color(style,c), role_color=r->_role_color(style,r),
                textsize=s->_textsize(style,s), composition_cells)
            row += 1
        end
        return nothing
    end

    function _polygons!(ax, polygons, layer, style, draw)
        isempty(polygons) && return nothing
        options = (;color=(_color(style,layer.fill_color),layer.alpha),
            strokecolor=_color(style,layer.stroke_color),
            strokewidth=layer.linewidth * style.linewidth_scale)
        # Makie's vector poly recipe paints all fills before all outlines. For
        # overlapping stroked polygons that would uncover an earlier outline.
        # Preserve the supplied order in that case; disjoint fibers batch fully.
        if options.strokewidth > 0 && _visual_polygon_overlap(polygons)
            for polygon in polygons
                draw(:poly,ax, polygon; options...)
            end
        else
            draw(:poly,ax, polygons; options...)
        end
        return nothing
    end

    function _draw_interval_bars!(ax, spec, style, draw)
        spec.kind in (:barcode,:slice_barcode) || return false
        haskey(spec.metadata,:interval_segments) || return false
        layers = spec.layers
        # Only the canonical horizontal strokes qualify. A guide that appends a
        # comparison line keeps ordinary layer ordering instead of losing it.
        segments = [s for l in layers if l isa Viz.SegmentLayer for s in l.segments]
        Tuple(segments) == spec.metadata.interval_segments || return false
        all(l -> l isa Union{Viz.SegmentLayer,Viz.PointLayer,Viz.TextLayer}, layers) || return false
        all(l -> !(l isa Viz.SegmentLayer) || l.linestyle===:solid, layers) || return false
        all(l -> !(l isa Viz.PointLayer) || (l.markerspace===:pixel && !(l.color isa AbstractVector)), layers) || return false
        endpoints = get(spec.metadata,:barcode_endpoint_layers,(left=(),right=()))
        linepoints = MakieMod.Point2d[]; linecolors = MakieMod.RGBAf[]; widths = Float64[]
        points = MakieMod.Point2d[]; colors = MakieMod.RGBAf[]; sizes = Float64[]; markers = Symbol[]
        labels = String[]; anchors = MakieMod.Point2d[]; textcolors = MakieMod.RGBAf[]
        fontsizes = Float64[]; aligns = Tuple{Symbol,Symbol}[]; offsets = MakieMod.Vec2d[]
        for (index,layer) in enumerate(layers)
            if layer isa Viz.SegmentLayer
                for s in layer.segments
                    append!(linepoints,(to_point2((s[1],s[2])),to_point2((s[3],s[4]))))
                    c = MakieMod.to_color((_color(style,layer.color),layer.alpha))
                    w = layer.linewidth * style.linewidth_scale
                    append!(linecolors,(c,c)); append!(widths,(w,w))
                end
            elseif layer isa Viz.PointLayer
                for p in layer.points
                    push!(points,to_point2(p));push!(colors,MakieMod.to_color((_color(style,layer.color),layer.alpha)))
                    push!(sizes,layer.markersize*style.markersize_scale);push!(markers,Viz._visual_marker(layer.color))
                end
            else
                horizontal = index in endpoints.left ? :left : index in endpoints.right ? :right : :center
                alignment = horizontal===:center ? (:center,:center) : (horizontal,:bottom)
                offset = horizontal===:center ? (0.,0.) : (0.,3style.linewidth_scale+_textsize(style,2))
                for (label,anchor) in zip(layer.labels,layer.positions)
                    push!(labels,label);push!(anchors,to_point2(anchor));push!(textcolors,MakieMod.to_color(_color(style,layer.color)))
                    push!(fontsizes,_textsize(style,layer.textsize));push!(aligns,alignment);push!(offsets,MakieMod.Vec2d(offset))
                end
            end
        end
        # Endpoint marks are foreground decorations. Their order (outer disk,
        # open center, selected mark) is retained within the scatter buffer.
        isempty(linepoints) || draw(:linesegments,ax,linepoints;color=linecolors,linewidth=widths)
        isempty(points) || draw(:scatter,ax,points;color=colors,markersize=sizes,marker=markers)
        isempty(labels) || draw(:text,ax,anchors;text=labels,color=textcolors,font=style.font,
            fontsize=fontsizes,align=aligns,offset=offsets)
        return true
    end

    function _draw_layers!(ax, spec; style=Viz.VisualStyle(), text_halo=get(spec.metadata,:text_halo,false), sink=nothing)
        function draw(kind, parent, args...; kwargs...)
            if sink === nothing
                return getproperty(MakieMod, Symbol(kind, :!))(parent,args...;kwargs...)
            end
            push!(sink,(;kind,args,kwargs=(;kwargs...)))
            return length(sink)
        end
        colorbars = NamedTuple[]
        if spec.kind === :overlay
            for child in spec.metadata.overlay_specs
                append!(colorbars,_draw_layers!(ax,child;style,text_halo=true,sink))
            end
            sink === nothing && _apply_axis_limits!(ax,spec)
            return colorbars
        end
        if !text_halo && _draw_interval_bars!(ax,spec,style,draw)
            sink === nothing && _apply_axis_limits!(ax,spec)
            return colorbars
        end
        layers = Viz.visual_layers(spec)
        query_label_layer = get(Viz.visual_metadata(spec), :query_label_layer, nothing)
        annotation_scale = max(style.fontsize / 16, style.markersize_scale)
        interval_annotation_layers = get(Viz.visual_metadata(spec), :interval_annotation_layers, ())
        barcode_endpoint_layers = get(Viz.visual_metadata(spec), :barcode_endpoint_layers, (left=(),right=()))
        interval_labels = String[]
        interval_anchors = NTuple{2,Float64}[]
        skip_until = 0
        for (layer_index, layer) in enumerate(layers)
            layer_index <= skip_until && continue
            if layer isa Viz.HeatmapLayer
                hm = draw(:heatmap,ax, layer.x, layer.y, permutedims(layer.values);
                                       colormap=Viz._visual_colormap(style, layer.colormap),
                                       colorrange=something(get(spec.metadata, :colorrange, nothing), MakieMod.Makie.automatic),
                                       nan_color=_role_color(style, :missing),
                                       alpha=layer.alpha)
                layer.show_colorbar && push!(colorbars, (; plot=hm, label=layer.colorbar_label))
            elseif layer isa Viz.RectLayer
                polygons = [[to_point2((r[1], r[2])), to_point2((r[3], r[2])),
                             to_point2((r[3], r[4])), to_point2((r[1], r[4]))] for r in layer.rects]
                _polygons!(ax, polygons, layer, style, draw)
            elseif layer isa Viz.PolygonLayer
                _polygons!(ax, [to_point2.(p) for p in layer.polygons], layer, style, draw)
            elseif layer isa Viz.Polygon3Layer
                points = MakieMod.Point3d[]
                triangles = MakieMod.GeometryBasics.TriangleFace{Int}[]
                for polygon in layer.polygons
                    isempty(polygon) && continue
                    # Triangulate in the plane with greatest projected area. Dropping
                    # z unconditionally would erase vertical cubical faces.
                    normal=zeros(3)
                    origin=first(polygon)
                    for i in 2:length(polygon)-1
                        a=polygon[i].-origin;b=polygon[i+1].-origin
                        normal .+= (a[2]*b[3]-a[3]*b[2],a[3]*b[1]-a[1]*b[3],a[1]*b[2]-a[2]*b[1])
                    end
                    all(iszero,normal) && continue
                    keep=filter(!=(argmax(abs.(normal))),1:3)
                    projected=[MakieMod.Point2d(p[keep[1]],p[keep[2]]) for p in polygon]
                    faces=MakieMod.GeometryBasics.earcut_triangulate([projected])
                    offset = length(points)
                    append!(points, MakieMod.Point3d.(polygon))
                    append!(triangles, [MakieMod.GeometryBasics.TriangleFace{Int}(Tuple(convert.(Int,f) .+ offset)) for f in faces])
                end
                (isempty(triangles) && sink === nothing) || draw(:mesh,ax,
                    MakieMod.GeometryBasics.Mesh(points, triangles);
                    color=(_color(style,layer.color),layer.alpha))
            elseif layer isa Viz.SegmentLayer
                points = [to_point2(p) for s in layer.segments for p in ((s[1],s[2]),(s[3],s[4]))]
                (isempty(points) && sink === nothing) || draw(:linesegments,ax, points;
                    color=(_color(style, layer.color), layer.alpha),
                    linewidth=layer.linewidth * style.linewidth_scale, linestyle=layer.linestyle)
            elseif layer isa Viz.Segment3Layer
                points = [MakieMod.Point3d(p) for s in layer.segments for p in ((s[1],s[2],s[3]),(s[4],s[5],s[6]))]
                (isempty(points) && sink === nothing) || draw(:linesegments,ax, points;
                    color=(_color(style, layer.color), layer.alpha),
                    linewidth=layer.linewidth * style.linewidth_scale)
            elseif layer isa Viz.PolylineLayer
                points = MakieMod.Point2d[]
                closing = MakieMod.Point2d[]
                for path in layer.paths
                    isempty(path) && continue
                    isempty(points) || push!(points, MakieMod.Point2d(NaN, NaN))
                    append!(points, to_point2.(path))
                    layer.closed && append!(closing, (to_point2(last(path)), to_point2(first(path))))
                end
                options = (;color=(_color(style, layer.color), layer.alpha),
                           linewidth=layer.linewidth * style.linewidth_scale)
                (isempty(points) && sink === nothing) || draw(:lines,ax, points; options...)
                (isempty(closing) && sink === nothing) || draw(:linesegments,ax, closing; options...)
            elseif layer isa Viz.PointLayer
                if spec.kind in (:persistence_diagram,:slice_diagram) &&
                        layer.markerspace===:pixel && !(layer.color isa AbstractVector)
                    # Consecutive canonical interval markers retain their exact
                    # paint order in one scatter, including selected overlays.
                    stop = layer_index
                    while stop < length(layers)
                        next = layers[stop+1]
                        next isa Viz.PointLayer && next.markerspace===:pixel &&
                            !(next.color isa AbstractVector) || break
                        stop += 1
                    end
                    points = MakieMod.Point2d[]; colors = MakieMod.RGBAf[]
                    sizes = Float64[]; markers = Symbol[]
                    for index in layer_index:stop
                        item = layers[index]
                        for point in item.points
                            push!(points,to_point2(point))
                            push!(colors,MakieMod.to_color((_color(style,item.color),item.alpha)))
                            push!(sizes,item.markersize*style.markersize_scale)
                            push!(markers,Viz._visual_marker(item.color))
                        end
                    end
                    (isempty(points) && sink === nothing) || draw(:scatter,ax,points;color=colors,markersize=sizes,marker=markers)
                    skip_until = stop
                    continue
                end
                (isempty(layer.points) && sink === nothing) || begin
                    # Data-space markers can encode geometric radii. Scaling
                    # their appearance would change the represented set.
                    markersize = layer.markersize * (layer.markerspace === :data ? 1 : style.markersize_scale)
                    kwargs = layer.color isa AbstractVector ?
                             (; color=layer.color, colormap=Viz._visual_colormap(style, layer.colormap),
                                colorrange=something(get(spec.metadata, :point_colorrange, nothing), MakieMod.Makie.automatic),
                                nan_color=_role_color(style, :missing), alpha=layer.alpha,
                                markersize, markerspace=layer.markerspace) :
                             (; color=(_color(style, layer.color), layer.alpha), markersize,
                                markerspace=layer.markerspace,
                                marker=layer.markerspace === :data ? :circle : Viz._visual_marker(layer.color))
                    plot = draw(:scatter,ax,
                                      [p[1] for p in layer.points],
                                      [p[2] for p in layer.points];
                                      kwargs...)
                    label = get(spec.metadata, :point_colorbar_label, "")
                    layer.color isa AbstractVector && !isempty(label) &&
                        push!(colorbars, (; plot, label))
                end
            elseif layer isa Viz.Point3Layer
                (isempty(layer.points) && sink === nothing) || begin
                    kwargs = layer.color isa AbstractVector ?
                             (; color=layer.color, colormap=Viz._visual_colormap(style, layer.colormap),
                                colorrange=something(get(spec.metadata, :point_colorrange, nothing), MakieMod.Makie.automatic),
                                nan_color=_role_color(style, :missing), alpha=layer.alpha,
                                markersize=layer.markersize * style.markersize_scale) :
                             (; color=(_color(style, layer.color), layer.alpha),
                                markersize=layer.markersize * style.markersize_scale,
                                marker=Viz._visual_marker(layer.color))
                    plot = draw(:scatter,ax,
                                      [p[1] for p in layer.points],
                                      [p[2] for p in layer.points],
                                      [p[3] for p in layer.points];
                                      kwargs...)
                    label = get(spec.metadata, :point_colorbar_label, "")
                    layer.color isa AbstractVector && !isempty(label) && push!(colorbars, (;plot,label))
                end
            elseif layer isa Viz.TextLayer
                if layer_index in interval_annotation_layers
                    append!(interval_labels, layer.labels)
                    append!(interval_anchors, layer.positions)
                    continue
                end
                isempty(layer.labels) && sink === nothing && continue
                alignments = Tuple{Symbol,Symbol}[]
                offsets = MakieMod.Vec2d[]
                for pos in layer.positions
                    # Pixel offsets separate query annotations from their markers
                    # without changing the mathematical positions in the spec.
                    text_options = if layer_index == query_label_layer
                        (; align=(:left, :bottom), offset=(12, 16) .* annotation_scale)
                    elseif spec.kind === :presentation_support
                        # A fixed lower-left label also leaves room for live
                        # selection markers without rebuilding the support plot.
                        (; align=(:right, :top), offset=(-12, -12) .* annotation_scale)
                    elseif spec.kind === :hasse
                        # Center multiline labels beside schematic vertices.
                        (; align=(:left, :center))
                    elseif spec.kind in (:source_chain_geometry, :component_geometry)
                        # Source panels also appear beside coefficient tables.
                        # Grow labels into the viewport without moving cells.
                        middle = ((sum(spec.axes.xlimits)/2), (sum(spec.axes.ylimits)/2))
                        horizontal = pos[1] > middle[1] ? :right : pos[1] < middle[1] ? :left : :center
                        vertical = pos[2] > middle[2] ? :top : pos[2] < middle[2] ? :bottom : spec.kind === :component_geometry ? :bottom : :center
                        gap = spec.kind === :component_geometry ? 10 : 5
                        (; align=(horizontal,vertical),
                            offset=((horizontal===:right ? -gap : horizontal===:left ? gap : 0),
                                    (vertical===:top ? -gap : vertical===:bottom ? gap : 0)) .* annotation_scale)
                    elseif spec.kind in (:slice_diagram, :persistence_diagram, :collection_matrix)
                        (; align=(:left, :bottom), offset=(10, 10) .* annotation_scale)
                    elseif spec.kind in (:slice_barcode, :barcode)
                        # Endpoint labels grow inward and above the bar, leaving
                        # room for the live selection stroke (6 scaled pixels).
                        # The offset is in pixels; data anchors stay unchanged.
                        horizontal = layer_index in barcode_endpoint_layers.left ? :left :
                            layer_index in barcode_endpoint_layers.right ? :right : :center
                        horizontal === :center ? (; align=(:center,:center)) :
                            (; align=(horizontal,:bottom),
                                offset=(0,3 * style.linewidth_scale + _textsize(style,2)))
                    else
                        NamedTuple()
                    end
                    push!(alignments, get(text_options, :align, (:left, :bottom)))
                    push!(offsets, MakieMod.Vec2d(get(text_options, :offset, (0,0))))
                end
                positions = to_point2.(layer.positions)
                if text_halo && nameof(MakieMod) === :WGLMakie
                    # WGL's shader accepts a uniform stroke width and paints its
                    # stroke outside each glyph. Per-string stroke-width arrays
                    # construct a scene but fail when WGL serializes it.
                    draw(:text,ax, positions; text=layer.labels,
                        color=_color(style,layer.color),
                        strokecolor=_role_color(style,:background), strokewidth=1.,
                        font=style.font, fontsize=_textsize(style,layer.textsize),
                        align=alignments, offset=offsets)
                elseif text_halo
                    # Interleave each halo and glyph, retaining painter order even
                    # for overlapping labels. No separate scene per string.
                    background = _role_color(style, :background)
                    draw(:text,ax, repeat(positions; inner=2);
                        text=repeat(layer.labels; inner=2),
                        color=repeat([background, _color(style,layer.color)], length(positions)),
                        strokecolor=background, strokewidth=repeat([1.,0.],length(positions)),
                        font=style.font, fontsize=_textsize(style,layer.textsize),
                        align=repeat(alignments;inner=2), offset=repeat(offsets;inner=2))
                else
                    draw(:text,ax, positions; text=layer.labels, color=_color(style,layer.color),
                        font=style.font, fontsize=_textsize(style,layer.textsize),
                        align=alignments, offset=offsets)
                end
            elseif layer isa Viz.BarcodeLayer
                points = MakieMod.Point2d[]
                y = layer.ystart
                for (iv, mult) in zip(layer.intervals, layer.multiplicities)
                    for _ in 1:max(mult, 1)
                        append!(points, (to_point2((iv[1],y)), to_point2((iv[2],y))))
                        y += layer.ystep
                    end
                end
                (isempty(points) && sink === nothing) || draw(:linesegments,ax, points;
                    color=_color(style,layer.color), linewidth=layer.linewidth * style.linewidth_scale)
            else
                error("Unsupported visualization layer $(typeof(layer)) for Makie rendering.")
            end
        end
        # Establish the final limits and tick layout before optimizing labels.
        # Updating them afterward repeats the same annotation work.
        sink === nothing && _apply_axis_limits!(ax, spec)
        if sink === nothing && spec.kind in (:slice_diagram, :persistence_diagram, :collection_matrix)
            # Infinity lanes can be closer than two horizontal labels are wide.
            # Measure the unrotated glyphs, so deciding their orientation never
            # depends on the rotation being updated. Keep every tick and its
            # mathematical coordinate; y labels are separated vertically already.
            tick_axis = ax.xaxis
            MakieMod.onany(ax.scene, tick_axis.tickpositions, tick_axis.ticklabels,
                          ax.xticklabelfont, ax.xticklabelsize; update=true) do positions, labels, font, fontsize
                length(positions) == length(labels) || return nothing
                widths = [MakieMod.widths(MakieMod.Makie.text_bb(label,MakieMod.to_font(font),fontsize))[1]
                          for label in labels]
                crowded = any(1:(length(positions)-1)) do i
                    abs(positions[i+1][1]-positions[i][1]) <
                        (widths[i]+widths[i+1])/2 + fontsize/6
                end
                rotation = crowded ? pi/2 : 0.0
                ax.xticklabelrotation[] == rotation || (ax.xticklabelrotation[] = rotation)
                return nothing
            end
        end
        if !isempty(interval_labels)
            anchors = to_point2.(interval_anchors)
            options = (; text=interval_labels, color=_role_color(style, :foreground),
                font=style.font, fontsize=_textsize(style, 12), align=(:left, :bottom),
                justification=:left, linewidth=style.linewidth_scale,
                shrink=(4, 8) .* annotation_scale)
            if sink === nothing
                _visual_interval_annotations!(MakieMod, ax, anchors; options...)
            else
                push!(sink,(;kind=:interval_annotations,args=(anchors,),kwargs=options))
            end
        end
        return colorbars
    end

    function _render_legend!(fig, slot, spec, appearance)
        legend = Viz.visual_legend(spec)
        get(legend, :visible, false) || return nothing
        entries = _legend_entries(spec)
        isempty(entries) && return nothing
        elements = Any[]
        labels = String[]
        for entry in entries
            style = get(entry, :style, :patch)
            color_ref = get(entry, :color, :black)
            color = _color(appearance, color_ref)
            marker = get(entry, :marker, Viz._visual_marker(color_ref))
            if style === :line && isdefined(MakieMod, :LineElement)
                push!(elements, MakieMod.LineElement(color=color,
                    linewidth=2 * appearance.linewidth_scale,
                    linestyle=get(entry, :linestyle, :solid)))
            elseif style === :marker && isdefined(MakieMod, :MarkerElement)
                push!(elements, MakieMod.MarkerElement(color=color, marker=marker,
                    markersize=12 * appearance.markersize_scale))
            elseif isdefined(MakieMod, :PolyElement)
                push!(elements, MakieMod.PolyElement(color=color))
            elseif isdefined(MakieMod, :MarkerElement)
                push!(elements, MakieMod.MarkerElement(color=color, marker=marker,
                    markersize=12 * appearance.markersize_scale))
            else
                continue
            end
            push!(labels, String(get(entry, :label, string(style))))
        end
        isempty(elements) && return nothing
        title = get(legend, :title, "")
        at_right = get(Viz.visual_metadata(spec), :legend_position, :bottom) === :right
        # Bottom legends constrain their row height, not the shared axis width.
        # A vertical legend's defaults do the reverse and shrink the plot.
        return MakieMod.Legend(slot, elements, labels; title=title,
                               labelfont=appearance.font, titlefont=appearance.font,
                               labelsize=_textsize(appearance, 12), titlesize=_textsize(appearance, 14),
                               labelcolor=_role_color(appearance, :foreground),
                               titlecolor=_role_color(appearance, :foreground),
                               backgroundcolor=_role_color(appearance, :background),
                               framecolor=_role_color(appearance, :border),
                               rowgap=appearance.gap / 2, colgap=appearance.gap,
                               orientation=at_right ? :vertical : :horizontal,
                               tellwidth=at_right, tellheight=!at_right)
    end

    function _make_axis(figslot, spec; style=Viz.VisualStyle(), headings::Bool=true)
        foreground = _role_color(style, :foreground)
        common = (; xlabel=spec.axes.xlabel, ylabel=spec.axes.ylabel,
            title=headings ? spec.title : "", titlefont=style.font,
            titlesize=_textsize(style, 18), titlecolor=foreground,
            xlabelfont=style.font, ylabelfont=style.font,
            xlabelsize=style.fontsize, ylabelsize=style.fontsize,
            xlabelcolor=foreground, ylabelcolor=foreground,
            xticklabelfont=style.font, yticklabelfont=style.font,
            xticklabelsize=_textsize(style, 12), yticklabelsize=_textsize(style, 12),
            xticklabelcolor=foreground, yticklabelcolor=foreground,
            backgroundcolor=_role_color(style, :background))
        if _uses_axis3(spec)
            # Plots inherit the scene's clipping planes when they are created.
            # Establish the final data box/aspect before adding cells: changing
            # the aspect afterward can leave inherited planes cutting the data.
            aspect_options=spec.axes.aspect in (:equal,:data) ? (;aspect=:data) : (;)
            return MakieMod.Axis3(figslot; common...,aspect_options...,
                limits=(spec.axes.xlimits,spec.axes.ylimits,get(spec.axes,:zlimits,nothing)),
                zlabel=get(spec.axes, :zlabel, "z"), zlabelfont=style.font,
                zlabelsize=style.fontsize, zlabelcolor=foreground,
                zticklabelfont=style.font, zticklabelsize=_textsize(style, 12),
                zticklabelcolor=foreground,
                xspinewidth=style.linewidth_scale, yspinewidth=style.linewidth_scale,
                zspinewidth=style.linewidth_scale)
        end
        ax = MakieMod.Axis(figslot; common...,
            subtitle=headings ? spec.subtitle : "", subtitlefont=style.font,
            subtitlesize=_textsize(style, 12), subtitlecolor=_role_color(style, :muted),
            spinewidth=style.linewidth_scale)
        if get(Viz.visual_metadata(spec), :minimal_axes, false)
            MakieMod.hidespines!(ax, :t, :r)
            ax.xgridvisible[] = false
            ax.ygridvisible[] = false
        end
        if get(Viz.visual_metadata(spec), :hide_decorations, false)
            MakieMod.hidedecorations!(ax)
            MakieMod.hidespines!(ax)
        end
        return ax
    end

    function _colorbar!(slot, plot, label, style)
        return MakieMod.Colorbar(slot, plot; label,
            labelfont=style.font, labelsize=_textsize(style, 14),
            labelcolor=_role_color(style, :foreground),
            ticklabelfont=style.font, ticklabelsize=_textsize(style, 12),
            ticklabelcolor=_role_color(style, :foreground), spinewidth=style.linewidth_scale)
    end

    function _render_spec_into_grid!(fig, grid, spec; style=Viz.VisualStyle(), composition_cells=nothing)
        panel_style = get(Viz.visual_metadata(spec), :panel_style, nothing)
        if panel_style === :matrix || any(layer -> layer isa Viz.MatrixLayer, Viz.visual_layers(spec))
            result = _render_matrix_panel!(grid, spec, style; composition_cells)
            _style_grid!(grid, style)
            return result
        elseif panel_style === :text_only
            result = _render_text_panel!(fig, grid, spec, style)
            _style_grid!(grid, style)
            return result
        end
        # Labels wrap to each panel's actual width, including after resizing.
        # Axis title text does not wrap and can intrude into adjacent panels.
        plot_row = _panel_heading!(grid, spec, style)
        ax = _make_axis(grid[plot_row, 1], spec; style, headings=false)
        colorbars = _draw_layers!(ax, spec; style)
        offset = 2
        for cb in colorbars
            _colorbar!(grid[plot_row, offset], cb.plot, cb.label, style)
            offset += 1
        end
        legend_position = get(Viz.visual_metadata(spec), :legend_position, :bottom)
        if legend_position === :right
            _render_legend!(fig, grid[plot_row, offset], spec, style)
        elseif legend_position !== :none
            _render_legend!(fig, grid[plot_row + 1, 1], spec, style)
        end
        _style_grid!(grid, style)
        return ax
    end

    function render_spec(spec; display::Symbol=:inline, figure=nothing, size=nothing,
                         style=Viz.VisualStyle())
        spec isa Viz.VisualizationSpec || throw(ArgumentError("render_spec expected a VisualizationSpec, got $(typeof(spec))."))
        Viz._check_visual_render_options(; display, figure, size, style)
        fig = if figure === nothing
            fig_size = size === nothing ? get(Viz.visual_metadata(spec), :figure_size, nothing) : size
            options = (; fontsize=style.fontsize, figure_padding=style.padding,
                backgroundcolor=_role_color(style, :background))
            fig_size === nothing ? MakieMod.Figure(; options...) : MakieMod.Figure(; options..., size=fig_size)
        else
            figure
        end
        panels = Viz.visual_panels(spec)
        positions = get(Viz.visual_metadata(spec), :panel_positions, nothing)
        ncols = if isempty(panels)
            1
        elseif positions === nothing
            Int(clamp(get(Viz.visual_metadata(spec), :panel_columns, min(3, length(panels))), 1, length(panels)))
        else
            maximum(position -> last(position[2]), positions)
        end
        if isempty(panels)
            grid = fig[1, 1] = MakieMod.GridLayout()
            _render_spec_into_grid!(fig, grid, spec; style)
            return _style_figure!(fig, style)
        end

        first_panel_row = 1
        if !isempty(spec.title)
            MakieMod.Label(fig[first_panel_row, 1:ncols], spec.title;
                font=style.font, fontsize=_textsize(style, 22),
                color=_role_color(style, :foreground), tellwidth=false, word_wrap=true)
            first_panel_row += 1
        end
        if !isempty(spec.subtitle)
            MakieMod.Label(fig[first_panel_row, 1:ncols], spec.subtitle;
                font=style.font, fontsize=_textsize(style, 14),
                color=_role_color(style, :muted), tellwidth=false, word_wrap=true)
            first_panel_row += 1
        end
        composition_handles = NamedTuple[]
        for (idx, panel) in enumerate(panels)
            rows, cols = if positions === nothing
                row, col = div(idx - 1, ncols) + 1, mod(idx - 1, ncols) + 1
                (row:row, col:col)
            else
                positions[idx]
            end
            figure_rows = (first_panel_row + first(rows) - 1):(first_panel_row + last(rows) - 1)
            grid = fig[figure_rows, cols] = MakieMod.GridLayout()
            cells = Dict{Tuple{Int,Int},Any}()
            axis = _render_spec_into_grid!(fig, grid, panel; style, composition_cells=cells)
            push!(composition_handles,(;axis,grid,cells))
        end
        weights = get(Viz.visual_metadata(spec), :panel_row_weights, nothing)
        if weights !== nothing
            for (idx, weight) in enumerate(weights)
                # Auto(false, weight) apportions only the height remaining
                # after the heading rows and gaps have taken their space.
                MakieMod.rowsize!(fig.layout, first_panel_row + idx - 1,
                                 MakieMod.Auto(false, weight))
            end
        end
        # A composite can share one key across its panels without reducing
        # just one plot's viewport to make room for a repeated legend.
        panel_rows = positions === nothing ? cld(length(panels), ncols) :
            maximum(position -> last(position[1]), positions)
        last_panel_row = first_panel_row + panel_rows - 1
        legend_position = get(Viz.visual_metadata(spec), :legend_position, :bottom)
        if legend_position === :right
            _render_legend!(fig, fig[first_panel_row:last_panel_row, ncols + 1], spec, style)
        elseif legend_position !== :none
            _render_legend!(fig, fig[last_panel_row + 1, 1:ncols], spec, style)
        end
        _style_figure!(fig, style)
        spec.kind === :composition && _render_visual_composition!(TO,MakieMod,fig,spec,composition_handles,style)
        return fig
    end

    save_spec = if allow_save
        function(path::AbstractString, spec; kwargs...)
            spec isa Viz.VisualizationSpec || throw(ArgumentError("save_spec expected a VisualizationSpec, got $(typeof(spec))."))
            if spec.kind === :composition && lowercase(splitext(path)[2]) == ".html"
                return _save_composition_html(TO,MakieMod,path,spec;kwargs...)
            end
            fig = render_spec(spec; kwargs...)
            MakieMod.save(path, fig)
            return path
        end
    else
        nothing
    end

    function plan(spec;style=Viz.VisualStyle())
        commands=NamedTuple[]
        colorbars=_draw_layers!(nothing,spec;style,sink=commands)
        return (;commands,colorbars)
    end

    return (; render=render_spec, save=save_spec, plan,
              heading=_panel_heading!, legend=_render_legend!, colorbar=_colorbar!,
              apply_limits=_apply_axis_limits!, style_grid=_style_grid!,
              make_axis=_make_axis, draw_layers=_draw_layers!,
              render_panel=_render_spec_into_grid!, style_figure=_style_figure!)
end
