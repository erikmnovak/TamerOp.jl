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

    _uses_axis3(spec) = any(layer -> layer isa Viz.Point3Layer || layer isa Viz.Segment3Layer,
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
        spec.axes.aspect === :equal && (ax.aspect = MakieMod.DataAspect())
        spec.axes.aspect === :data && (ax.aspect = MakieMod.DataAspect())
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

    function _render_matrix_panel!(grid, spec, style)
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
            table = MakieMod.GridLayout(grid[row, 1]; rowgap=style.gap * 2/3,
                                        colgap=style.gap * 7/6,
                                        halign=:center, valign=:top, tellwidth=false)
            meta = Viz.visual_metadata(spec)
            row_colors = get(meta, :row_colors, fill(Viz._VisualRole(:foreground), nr))
            column_colors = get(meta, :column_colors, fill(Viz._VisualRole(:foreground), nc))
            cell_colors = get(meta, :cell_colors, fill(Viz._VisualRole(:foreground), nr, nc))
            row_roles = get(meta, :row_roles, nothing)
            column_roles = get(meta, :column_roles, nothing)
            cell_roles = get(meta, :cell_roles, nothing)
            MakieMod.Label(table[1, 1], get(meta, :matrix_corner, "target / source");
                           font=style.font, fontsize=_textsize(style, 11),
                           color=_role_color(style, :muted))
            for j in 1:nc
                role = column_roles === nothing ? nothing : column_roles[j]
                label = layer.column_labels[j] * (role === nothing ? "" : Viz._visual_role_label(role))
                color = role === nothing ? _color(style, column_colors[j]) : _role_color(style, role)
                MakieMod.Label(table[1, j + 1], label;
                               font=style.font, fontsize=_textsize(style, 12), color)
            end
            entry_fontsize = _textsize(style, nc > 8 ? 12 : 14)
            for i in 1:nr
                role = row_roles === nothing ? nothing : row_roles[i]
                label = layer.row_labels[i] * (role === nothing ? "" : Viz._visual_role_label(role))
                color = role === nothing ? _color(style, row_colors[i]) : _role_color(style, role)
                MakieMod.Label(table[i + 1, 1], label;
                               font=style.font, fontsize=_textsize(style, 12), color)
                for j in 1:nc
                    color = cell_roles === nothing ? _color(style, cell_colors[i,j]) :
                        _role_color(style, cell_roles[i,j])
                    MakieMod.Label(table[i + 1, j + 1], layer.entries[i, j];
                                   font=style.mono_font, fontsize=entry_fontsize, color)
                end
            end
            row += 1
        end
        return nothing
    end

    function _draw_layers!(ax, spec; style=Viz.VisualStyle())
        colorbars = NamedTuple[]
        layers = Viz.visual_layers(spec)
        query_label_layer = get(Viz.visual_metadata(spec), :query_label_layer, nothing)
        annotation_scale = max(style.fontsize / 16, style.markersize_scale)
        interval_annotation_layers = get(Viz.visual_metadata(spec), :interval_annotation_layers, ())
        barcode_endpoint_layers = get(Viz.visual_metadata(spec), :barcode_endpoint_layers, (left=(),right=()))
        interval_labels = String[]
        interval_anchors = NTuple{2,Float64}[]
        for (layer_index, layer) in enumerate(layers)
            if layer isa Viz.HeatmapLayer
                hm = MakieMod.heatmap!(ax, layer.x, layer.y, permutedims(layer.values);
                                       colormap=Viz._visual_colormap(style, layer.colormap),
                                       colorrange=something(get(spec.metadata, :colorrange, nothing), MakieMod.Makie.automatic),
                                       nan_color=_role_color(style, :missing),
                                       alpha=layer.alpha)
                layer.show_colorbar && push!(colorbars, (; plot=hm, label=layer.colorbar_label))
            elseif layer isa Viz.RectLayer
                for rect in layer.rects
                    poly = [to_point2((rect[1], rect[2])), to_point2((rect[3], rect[2])),
                            to_point2((rect[3], rect[4])), to_point2((rect[1], rect[4]))]
                    MakieMod.poly!(ax, poly;
                                   color=(_color(style, layer.fill_color), layer.alpha),
                                   strokecolor=_color(style, layer.stroke_color),
                                   strokewidth=layer.linewidth * style.linewidth_scale)
                end
            elseif layer isa Viz.PolygonLayer
                for polygon in layer.polygons
                    MakieMod.poly!(ax, to_point2.(polygon);
                                   color=(_color(style, layer.fill_color), layer.alpha),
                                   strokecolor=_color(style, layer.stroke_color),
                                   strokewidth=layer.linewidth * style.linewidth_scale)
                end
            elseif layer isa Viz.SegmentLayer
                for seg in layer.segments
                    MakieMod.lines!(ax, [seg[1], seg[3]], [seg[2], seg[4]];
                                    color=(_color(style, layer.color), layer.alpha),
                                    linewidth=layer.linewidth * style.linewidth_scale,
                                    linestyle=layer.linestyle)
                end
            elseif layer isa Viz.Segment3Layer
                for seg in layer.segments
                    MakieMod.lines!(ax, [seg[1], seg[4]], [seg[2], seg[5]], [seg[3], seg[6]];
                                    color=(_color(style, layer.color), layer.alpha),
                                    linewidth=layer.linewidth * style.linewidth_scale)
                end
            elseif layer isa Viz.PolylineLayer
                for path in layer.paths
                    xs = [p[1] for p in path]
                    ys = [p[2] for p in path]
                    MakieMod.lines!(ax, xs, ys; color=(_color(style, layer.color), layer.alpha),
                                    linewidth=layer.linewidth * style.linewidth_scale)
                    if layer.closed && !isempty(path)
                        MakieMod.lines!(ax, [path[end][1], path[1][1]], [path[end][2], path[1][2]];
                                        color=(_color(style, layer.color), layer.alpha),
                                        linewidth=layer.linewidth * style.linewidth_scale)
                    end
                end
            elseif layer isa Viz.PointLayer
                isempty(layer.points) || begin
                    # Data-space markers can encode geometric radii. Scaling
                    # their appearance would change the represented set.
                    markersize = layer.markersize * (layer.markerspace === :data ? 1 : style.markersize_scale)
                    kwargs = layer.color isa AbstractVector ?
                             (; color=layer.color, colormap=Viz._visual_colormap(style, layer.colormap),
                                nan_color=_role_color(style, :missing), alpha=layer.alpha,
                                markersize, markerspace=layer.markerspace) :
                             (; color=(_color(style, layer.color), layer.alpha), markersize,
                                markerspace=layer.markerspace,
                                marker=layer.markerspace === :data ? :circle : Viz._visual_marker(layer.color))
                    MakieMod.scatter!(ax,
                                      [p[1] for p in layer.points],
                                      [p[2] for p in layer.points];
                                      kwargs...)
                end
            elseif layer isa Viz.Point3Layer
                isempty(layer.points) || begin
                    kwargs = layer.color isa AbstractVector ?
                             (; color=layer.color, colormap=Viz._visual_colormap(style, layer.colormap),
                                nan_color=_role_color(style, :missing), alpha=layer.alpha,
                                markersize=layer.markersize * style.markersize_scale) :
                             (; color=(_color(style, layer.color), layer.alpha),
                                markersize=layer.markersize * style.markersize_scale,
                                marker=Viz._visual_marker(layer.color))
                    MakieMod.scatter!(ax,
                                      [p[1] for p in layer.points],
                                      [p[2] for p in layer.points],
                                      [p[3] for p in layer.points];
                                      kwargs...)
                end
            elseif layer isa Viz.TextLayer
                if layer_index in interval_annotation_layers
                    append!(interval_labels, layer.labels)
                    append!(interval_anchors, layer.positions)
                    continue
                end
                for (lbl, pos) in zip(layer.labels, layer.positions)
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
                    elseif spec.kind in (:slice_diagram, :persistence_diagram)
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
                    MakieMod.text!(ax, lbl; position=(pos[1], pos[2]), color=_color(style, layer.color),
                                   font=style.font, fontsize=_textsize(style, layer.textsize), text_options...)
                end
            elseif layer isa Viz.BarcodeLayer
                y = layer.ystart
                for (iv, mult) in zip(layer.intervals, layer.multiplicities)
                    for _ in 1:max(mult, 1)
                        MakieMod.lines!(ax, [iv[1], iv[2]], [y, y];
                                        color=_color(style, layer.color), linewidth=layer.linewidth * style.linewidth_scale)
                        y += layer.ystep
                    end
                end
            else
                error("Unsupported visualization layer $(typeof(layer)) for Makie rendering.")
            end
        end
        if !isempty(interval_labels)
            anchors = to_point2.(interval_anchors)
            options = (; text=interval_labels, color=_role_color(style, :foreground),
                font=style.font, fontsize=_textsize(style, 12), align=(:left, :bottom),
                justification=:left, linewidth=style.linewidth_scale,
                shrink=(4, 8) .* annotation_scale)
            # Native layout supplies measured, separated labels. Its corner
            # correction can overwrite one axis when both need clipping; apply
            # the two pixel bounds independently before displaying the result.
            optimizer = MakieMod.annotation!(ax, anchors; options..., visible=false)
            offsets = MakieMod.lift(ax.scene, optimizer.offsets, optimizer.text_bbs,
                                   MakieMod.viewport(ax.scene); ignore_equal_values=true) do proposed, boxes, viewport
                extent = MakieMod.widths(viewport)
                map(proposed, boxes) do offset, box
                    lo, hi = minimum(box), maximum(box)
                    corrected = ntuple(2) do k
                        lower, upper = 2.0-lo[k], Float64(extent[k])-2.0-hi[k]
                        all(isfinite, (offset[k], lower, upper)) || return 0.0
                        # A not-yet-laid-out or impossibly narrow viewport is
                        # centered until the next resize supplies enough space.
                        lower <= upper ? clamp(Float64(offset[k]), lower, upper) : lower/2+upper/2
                    end
                    MakieMod.Vec2d(corrected)
                end
            end
            # Finite relative-pixel offsets use Makie's explicit placement path;
            # no second optimization or feedback loop changes these positions.
            # Targets, interval metadata, and coordinate-based picking stay exact.
            MakieMod.annotation!(ax, offsets, anchors; options..., labelspace=:relative_pixel)
        end
        _apply_axis_limits!(ax, spec)
        if spec.kind in (:slice_diagram, :persistence_diagram)
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
            return MakieMod.Axis3(figslot; common...,
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

    function _render_spec_into_grid!(fig, grid, spec; style=Viz.VisualStyle())
        panel_style = get(Viz.visual_metadata(spec), :panel_style, nothing)
        if panel_style === :matrix || any(layer -> layer isa Viz.MatrixLayer, Viz.visual_layers(spec))
            result = _render_matrix_panel!(grid, spec, style)
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

    function _render_slice_viewer(spec; figure=nothing, style=Viz.VisualStyle())
        volume = get(spec.metadata, :volume, nothing)
        volume === nothing && return nothing
        view_dims = Tuple(get(spec.metadata, :view_dims, (1, 2)))
        fixed = Dict{Int,Int}(get(spec.metadata, :fixed_indices, Dict{Int,Int}()))
        control_dims = sort!(collect(setdiff(1:ndims(volume), collect(view_dims))))
        isempty(control_dims) && return nothing
        fig = figure === nothing ? MakieMod.Figure(;
            fontsize=style.fontsize, figure_padding=style.padding,
            backgroundcolor=_role_color(style, :background)) : figure
        grid = _style_grid!(MakieMod.GridLayout(), style)
        fig[1, 1:3] = grid
        plot_row = _panel_heading!(grid, spec, style)
        ax = _make_axis(grid[plot_row, 1], spec; style, headings=false)
        current = copy(fixed)
        # The core stores rows=y, columns=x. Makie needs rows=x, columns=y;
        # use the same slice helper for initial state and every slider update.
        zobs = MakieMod.Observable(permutedims(Viz._image_slice_values(volume, view_dims, current)))
        hm = MakieMod.heatmap!(ax,
                               1:size(zobs[], 1),
                               1:size(zobs[], 2),
                               zobs;
                               colormap=Viz._visual_colormap(style, get(spec.metadata, :colormap, :magma)),
                               colorrange=something(get(spec.metadata, :colorrange, nothing), MakieMod.Makie.automatic),
                               nan_color=_role_color(style, :missing),
                               alpha=1.0)
        _apply_axis_limits!(ax, spec)
        colorbar_label = get(spec.metadata, :colorbar_label, "intensity")
        isempty(colorbar_label) || _colorbar!(grid[plot_row, 2], hm, colorbar_label, style)
        for (row, dim) in enumerate(control_dims)
            MakieMod.Label(fig[row + 1, 1], "slice dim $dim";
                font=style.font, fontsize=style.fontsize,
                color=_role_color(style, :foreground), tellwidth=false)
            slider = MakieMod.Makie.Slider(fig[row + 1, 2];
                                     range=1:size(volume, dim),
                                     startvalue=get(current, dim, cld(size(volume, dim), 2)))
            MakieMod.Label(fig[row + 1, 3], MakieMod.lift(v -> string(Int(round(v))), slider.value);
                font=style.mono_font, fontsize=style.fontsize,
                color=_role_color(style, :foreground))
            MakieMod.on(slider.value) do v
                current[dim] = Int(round(v))
                zobs[] = permutedims(Viz._image_slice_values(volume, view_dims, current))
            end
        end
        _style_grid!(grid, style)
        return _style_figure!(fig, style)
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
        nameof(MakieMod) === :WGLMakie &&
            get(Viz.visual_interaction(spec), :notebook, :summary_card) === :widget_viewer && begin
            widget_fig = _render_slice_viewer(spec; figure=fig, style)
            widget_fig === nothing || return widget_fig
        end
        panels = Viz.visual_panels(spec)
        ncols = isempty(panels) ? 1 : Int(clamp(get(Viz.visual_metadata(spec), :panel_columns, min(3, length(panels))), 1, max(length(panels), 1)))
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
        for (idx, panel) in enumerate(panels)
            row = first_panel_row + div(idx - 1, ncols)
            col = 1 + mod(idx - 1, ncols)
            grid = fig[row, col] = MakieMod.GridLayout()
            _render_spec_into_grid!(fig, grid, panel; style)
        end
        return _style_figure!(fig, style)
    end

    save_spec = if allow_save
        function(path::AbstractString, spec; kwargs...)
            spec isa Viz.VisualizationSpec || throw(ArgumentError("save_spec expected a VisualizationSpec, got $(typeof(spec))."))
            fig = render_spec(spec; kwargs...)
            MakieMod.save(path, fig)
            return path
        end
    else
        nothing
    end

    return (; render=render_spec, save=save_spec,
              make_axis=_make_axis, draw_layers=_draw_layers!,
              render_panel=_render_spec_into_grid!, style_figure=_style_figure!)
end
