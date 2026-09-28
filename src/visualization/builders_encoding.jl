# Encoding and translation visualization builders, including migrated flange views.

available_visuals(::Any) = ()

function visual_spec(obj; kind::Symbol=:auto, cache=:auto, kwargs...)
    haskey(kwargs, :backend) && throw(ArgumentError("backend belongs to visualize/render, not visual_spec"))
    cache === :auto || throw(ArgumentError("visual_spec does not support a cache override"))
    report = check_visual_request(obj; kind=kind, kwargs..., throw=true)
    chosen = report.requested_kind
    return _visual_spec(obj, chosen; kwargs...)
end

_visual_spec(obj, kind::Symbol; kwargs...) =
    throw(ArgumentError("No visualization builder is registered for $(nameof(typeof(obj))) with kind=$(kind)."))

function _as_points2(point)
    if point === nothing
        return NTuple{2,Float64}[]
    elseif point isa NTuple{2,<:Real}
        return [(float(point[1]), float(point[2]))]
    elseif point isa AbstractVector{<:Real}
        length(point) == 2 || throw(ArgumentError("expected a 2-vector point"))
        return [(float(point[1]), float(point[2]))]
    else
        throw(ArgumentError("unsupported point representation $(typeof(point))"))
    end
end

function _collect_query_points(; point=nothing, points=nothing)
    out = Tuple{Real,Real}[]
    function append_point(p)
        (p isa Tuple || p isa AbstractVector) && length(p) == 2 && all(x -> x isa Real && isfinite(x), p) ||
            throw(ArgumentError("query points must contain two finite real coordinates"))
        push!(out, (p[1], p[2]))
    end
    point === nothing || append_point(point)
    if points !== nothing
        if points isa AbstractMatrix
            size(points, 1) == 2 || throw(ArgumentError("points matrix must have 2 rows"))
            for j in axes(points, 2)
                append_point((points[1, j], points[2, j]))
            end
        else
            foreach(append_point, points)
        end
    end
    return out
end

function _query_geometry_readout(pi, points, box)
    return [(; point=p, region_id=_visual_locate(pi, p), display_point=_drawing_point(p),
               display_rounded=any(p[i] != _drawing_point(p)[i] for i in 1:2),
               inside_viewport=all(box[1][i] <= p[i] <= box[2][i] for i in 1:2)) for p in points]
end

function _text_layer_from_labels(points::AbstractVector{<:NTuple{2,<:Real}}, labels::AbstractVector{<:AbstractString}; color::Symbol=:black, textsize::Float64=10.0)
    return TextLayer(String[String(lbl) for lbl in labels],
                     NTuple{2,Float64}[(float(p[1]), float(p[2])) for p in points],
                     color,
                     textsize)
end

function _offset_label_positions(points::AbstractVector{<:NTuple{2,<:Real}}, axes::NamedTuple;
                                 dx_frac::Float64=0.014, dy_frac::Float64=0.010)
    isempty(points) && return NTuple{2,Float64}[]
    xlimits = get(axes, :xlimits, nothing)
    ylimits = get(axes, :ylimits, nothing)
    xspan = xlimits === nothing ? 1.0 : abs(float(xlimits[2]) - float(xlimits[1]))
    yspan = ylimits === nothing ? 1.0 : abs(float(ylimits[2]) - float(ylimits[1]))
    dx = max(0.03, dx_frac * xspan)
    dy = max(0.03, dy_frac * yspan)
    return NTuple{2,Float64}[(float(p[1]) + dx, float(p[2]) + dy) for p in points]
end

function _rect_text_centers(rects::Vector{NTuple{4,Float64}})
    return NTuple{2,Float64}[_midpoint(rect) for rect in rects]
end

const _BOX_REGION_COLORS = (
    :seagreen3,
    :cornflowerblue,
    :goldenrod2,
    :orchid3,
    :tomato2,
    :slateblue3,
    :darkkhaki,
    :cadetblue3,
)

@inline _box_region_color(i::Int) = _BOX_REGION_COLORS[1 + mod(i - 1, length(_BOX_REGION_COLORS))]

@inline function _segment_key(seg::NTuple{4,Float64})
    x1, y1, x2, y2 = seg
    return (x1 < x2 || (x1 == x2 && y1 <= y2)) ? seg : (x2, y2, x1, y1)
end

function _region_boundary_segments(rects::Vector{NTuple{4,Float64}})
    counts = Dict{NTuple{4,Float64},Int}()
    for rect in rects
        xlo, ylo, xhi, yhi = rect
        for edge in ((xlo, ylo, xhi, ylo),
                     (xhi, ylo, xhi, yhi),
                     (xhi, yhi, xlo, yhi),
                     (xlo, yhi, xlo, ylo))
            key = _segment_key(edge)
            counts[key] = get(counts, key, 0) + 1
        end
    end
    segments = NTuple{4,Float64}[]
    for (seg, count) in counts
        count == 1 && push!(segments, seg)
    end
    return segments
end

function _flange_default_box(FG::Flange)
    xs = Int[]
    ys = Int[]
    for F in flats(FG)
        push!(xs, F.b[1]); push!(ys, F.b[2])
    end
    for E in injectives(FG)
        push!(xs, E.b[1]); push!(ys, E.b[2])
    end
    isempty(xs) && return ([0.0, 0.0], [5.0, 5.0])
    lo = [float(minimum(xs) - 1), float(minimum(ys) - 1)]
    hi = [float(maximum(xs) + 2), float(maximum(ys) + 2)]
    return (lo, hi)
end

function _clip_rect_2d(ell::NTuple{2,Float64}, u::NTuple{2,Float64}; L1=-Inf, U1=Inf, L2=-Inf, U2=Inf)
    xlo = max(ell[1], L1); xhi = min(u[1], U1)
    ylo = max(ell[2], L2); yhi = min(u[2], U2)
    xlo <= xhi && ylo <= yhi || return nothing
    return (xlo, ylo, xhi, yhi)
end

available_visuals(FG::Flange) = FG.n == 2 ? (:regions, :constant_subdivision) : ()

function _visual_spec(FG::Flange, kind::Symbol; box=nothing, alpha_up::Real=0.25, alpha_dn::Real=0.25, kwargs...)
    FG.n == 2 || throw(ArgumentError("Visualization v1 only supports 2D flanges."))
    _ = kwargs
    b = box === nothing ? _flange_default_box(FG) : box
    ell = (float(b[1][1]), float(b[1][2]))
    u = (float(b[2][1]), float(b[2][2]))
    if kind === :regions
        up_rects = NTuple{4,Float64}[]
        dn_rects = NTuple{4,Float64}[]
        up_labels = String[]
        dn_labels = String[]
        for (j, F) in enumerate(flats(FG))
            L1 = F.tau.coords[1] ? -Inf : float(F.b[1])
            L2 = F.tau.coords[2] ? -Inf : float(F.b[2])
            rect = _clip_rect_2d(ell, u; L1=L1, L2=L2)
            rect === nothing && continue
            push!(up_rects, rect)
            push!(up_labels, "U$(j)")
        end
        for (i, E) in enumerate(injectives(FG))
            U1 = E.tau.coords[1] ? Inf : float(E.b[1])
            U2 = E.tau.coords[2] ? Inf : float(E.b[2])
            rect = _clip_rect_2d(ell, u; U1=U1, U2=U2)
            rect === nothing && continue
            push!(dn_rects, rect)
            push!(dn_labels, "D$(i)")
        end
        layers = AbstractVisualizationLayer[
            RectLayer(up_rects, :dodgerblue, :navy, float(alpha_up), 0.0),
            RectLayer(dn_rects, :crimson, :darkred, float(alpha_dn), 0.0),
            _text_layer_from_labels(_rect_text_centers(up_rects), up_labels; color=:navy, textsize=10.0),
            _text_layer_from_labels(_rect_text_centers(dn_rects), dn_labels; color=:darkred, textsize=10.0),
        ]
        return VisualizationSpec(:regions;
                                 title="Flange regions",
                                 subtitle="flats and injectives clipped to a viewing box",
                                 layers=layers,
                                 axes=_default_axes_2d(xlabel="x1", ylabel="x2", xlimits=(ell[1], u[1]), ylimits=(ell[2], u[2]), aspect=:equal),
                                 metadata=(; object=:flange, nflats=length(up_rects), ninjectives=length(dn_rects), box=b),
                                 legend=_default_legend(visible=true, entries=(; flats=:dodgerblue, injectives=:crimson)),
                                 interaction=_default_interaction(labels=true))
    elseif kind === :constant_subdivision
        xlo = ceil(Int, ell[1]); ylo = ceil(Int, ell[2])
        xhi = floor(Int, u[1]); yhi = floor(Int, u[2])
        nx = max(0, xhi - xlo)
        ny = max(0, yhi - ylo)
        nx > 0 && ny > 0 || throw(ArgumentError("constant_subdivision requires a box with at least one interior unit cell."))
        vals = Matrix{Float64}(undef, ny, nx)
        for iy in 1:ny, ix in 1:nx
            g = [xlo + ix - 1, ylo + iy - 1]
            cols = active_flats(FG, g)
            rows = active_injectives(FG, g)
            vals[iy, ix] = (isempty(cols) || isempty(rows)) ? 0.0 : float(rank_restricted(FG.field, FG.phi, rows, cols))
        end
        return VisualizationSpec(:constant_subdivision;
                                 title="Flange constant subdivision",
                                 subtitle="heatmap of fiber dimensions on unit cells",
                                 layers=AbstractVisualizationLayer[
                                     HeatmapLayer(Float64[xlo + i - 1 for i in 1:nx], Float64[ylo + j - 1 for j in 1:ny], vals, :viridis, 1.0, "dim"),
                                 ],
                                 axes=_default_axes_2d(xlabel="x1", ylabel="x2", xlimits=(xlo, xhi), ylimits=(ylo, yhi), aspect=:equal),
                                 metadata=(; object=:flange, matrix_size=size(FG.phi), box=b))
    end
    throw(ArgumentError("Unsupported flange visualization kind $(kind)."))
end

available_visuals(::GridEncodingMap{2}) = (:regions, :region_labels, :query_overlay)
available_visuals(pi::PLEncodingMapBoxes) = pi.n == 2 ? (:regions, :region_labels, :query_overlay) : ()
available_visuals(pi::PLEncodingMap) = pi.n == 2 ? (:regions, :region_labels, :query_overlay) : ()
available_visuals(pi::ZnEncodingMap) = pi.n == 2 ? (:regions, :region_labels, :query_overlay) : ()

function _visual_spec(pi::Union{GridEncodingMap{2},PLEncodingMapBoxes,PLEncodingMap,ZnEncodingMap},
                      kind::Symbol; point=nothing, points=nothing, box=nothing, kwargs...)
    kind in (:regions, :region_labels, :query_overlay) || throw(ArgumentError("unsupported encoding visualization kind $kind"))
    geometry = _region_geometry_2d(pi; box)
    exact_points = kind === :query_overlay ? _collect_query_points(; point, points) : Tuple{Real,Real}[]
    readout = _query_geometry_readout(pi, exact_points, geometry.box)
    pts = NTuple{2,Float64}[q.display_point for q in readout]
    point_labels = [string("q", i, " -> ", q.region_id, q.inside_viewport ? "" : " (outside view)") for (i,q) in enumerate(readout)]
    layers = _region_geometry_layers(geometry)
    if kind === :region_labels
        # Every visible component gets its actual classifier label, including
        # disconnected pieces. No renumbering after viewport filtering.
        positions = NTuple{2,Float64}[_drawing_point((sum(p[1] for p in c.vertices)/length(c.vertices),
                                                     sum(p[2] for p in c.vertices)/length(c.vertices))) for c in geometry.components]
        labels = [c.region_id == 0 ? "outside (0)" : string(c.region_id) for c in geometry.components]
        push!(layers, _text_layer_from_labels(positions, labels))
    end
    if !isempty(pts)
        push!(layers, PointLayer(pts, :orange3, 0.95, 14.0))
        push!(layers, _text_layer_from_labels(pts, point_labels))
    end
    query_collisions = _display_collisions(exact_points)
    warnings = String[]
    isempty(geometry.coordinate_collisions) || push!(warnings, "Distinct exact geometry coordinates coincide in this display; inspect metadata.geometry.coordinate_collisions.")
    isempty(geometry.dimension_collapses) || push!(warnings, "A region loses dimension at drawing precision; inspect metadata.geometry.dimension_collapses for its exact geometry.")
    isempty(query_collisions) || push!(warnings, "Distinct exact queries coincide in this display; inspect metadata.query_readout.")
    any(q.display_rounded for q in readout) && push!(warnings, "Query markers are rounded for drawing; labels use the original coordinates in metadata.query_readout.")
    object, title = pi isa GridEncodingMap ? (:grid_encoding_map, "Grid encoding") :
                    pi isa PLEncodingMapBoxes ? (:pl_boxes, "Box encoding") :
                    pi isa ZnEncodingMap ? (:zn_encoding_map, "Integer encoding") : (:pl_encoding_map, "Polyhedral encoding")
    subtitle = geometry.geometry_kind === :nearest_lattice_tiles ?
        "integer fibers shown as nearest-lattice tiles; ties use round-to-even" :
        "solid: included edge; dashed: excluded edge; dotted: viewing-window edge; open circles: excluded vertices"
    isempty(warnings) || (subtitle *= " | Exact-coordinate readout available; display rounding detected.")
    ids = geometry.region_ids
    return VisualizationSpec(kind; title, subtitle, layers, axes=geometry.axes,
        metadata=(; object, nregions=count(!iszero, ids), ncells=length(geometry.components),
                    query_count=length(pts), query_readout=readout, query_collisions, warnings,
                    geometry, region_ids=ids, box=geometry.box, figure_size=(860, 620), legend_position=:right,
                    outside_region_id=0, outside_meaning=:not_represented),
        legend=_default_legend(visible=true, entries=(; (Symbol(r == 0 ? "outside_0" : "R$r") =>
            (r == 0 ? :gray90 : _box_region_color(r)) for r in ids)...)),
        interaction=_default_interaction(labels=kind === :region_labels || !isempty(point_labels)))
end

function _visual_spec(enc::CompiledEncoding, kind::Symbol; kwargs...)
    return _visual_spec(encoding_map(enc), kind; kwargs...)
end

function available_visuals(enc::CompiledEncoding)
    return available_visuals(encoding_map(enc))
end

function _visual_spec(res::EncodingResult, kind::Symbol; kwargs...)
    return _visual_spec(encoding_map(res), kind; kwargs...)
end

available_visuals(res::EncodingResult) = available_visuals(encoding_map(res))

function _poset_coordinates_2d(P::ProductOfChainsPoset{2})
    pts = Vector{NTuple{2,Float64}}(undef, nvertices(P))
    idx = 1
    for j in 1:P.sizes[2], i in 1:P.sizes[1]
        pts[idx] = (float(i), float(j))
        idx += 1
    end
    return pts
end

function _poset_coordinates_2d(P::GridPoset{2})
    pts = Vector{NTuple{2,Float64}}(undef, nvertices(P))
    idx = 1
    for y in P.coords[2], x in P.coords[1]
        pts[idx] = (float(x), float(y))
        idx += 1
    end
    return pts
end

function _poset_coordinates_2d(P::ProductPoset)
    n1 = nvertices(P.P1)
    pts = Vector{NTuple{2,Float64}}(undef, nvertices(P))
    for idx in 1:nvertices(P)
        i1 = ((idx - 1) % n1) + 1
        i2 = div(idx - 1, n1) + 1
        pts[idx] = (float(i1), float(i2))
    end
    return pts
end

_poset_coordinates_2d(P::AbstractPoset) = nothing

_has_poset_coordinates_2d(::AbstractPoset) = false
_has_poset_coordinates_2d(::Union{ProductOfChainsPoset{2},GridPoset{2},ProductPoset}) = true

available_visuals(pi::EncodingMap) = if !_has_poset_coordinates_2d(target_poset(pi))
    ()
elseif !_has_poset_coordinates_2d(source_poset(pi))
    (:regions, :region_labels)
else
    (:regions, :region_labels, :pushforward_overlay)
end

function _visual_spec(pi::EncodingMap, kind::Symbol; kwargs...)
    _ = kwargs
    tgt_pts = _poset_coordinates_2d(target_poset(pi))
    tgt_pts === nothing && throw(ArgumentError("Visualization v1 needs a 2D-embeddable target poset for EncodingMap visuals."))
    if kind === :regions || kind === :region_labels
        counts = zeros(Int, length(tgt_pts))
        for p in region_map(pi)
            counts[p] += 1
        end
        point_layer = PointLayer(tgt_pts, :royalblue, 0.9, 12.0)
        layers = AbstractVisualizationLayer[point_layer]
        if kind === :region_labels
            labels = [string(i, ":", counts[i]) for i in eachindex(counts)]
            push!(layers, _text_layer_from_labels(tgt_pts, labels; color=:black, textsize=10.0))
        end
        bbox = _bbox_from_points(tgt_pts)
        return VisualizationSpec(kind;
                                 title="Finite encoding map",
                                 subtitle="target-region occupancy counts",
                                 layers=layers,
                                 axes=_default_axes_2d(xlabel="target x", ylabel="target y",
                                                       xlimits=bbox === nothing ? nothing : bbox[1],
                                                       ylimits=bbox === nothing ? nothing : bbox[2],
                                                       aspect=:equal),
                                 metadata=(; object=:encoding_map, nsource=nvertices(source_poset(pi)), ntarget=nvertices(target_poset(pi))))
    elseif kind === :pushforward_overlay
        src_pts = _poset_coordinates_2d(source_poset(pi))
        src_pts === nothing && throw(ArgumentError("pushforward_overlay requires a 2D-embeddable source poset."))
        segments = NTuple{4,Float64}[]
        for (q, p) in enumerate(region_map(pi))
            s = src_pts[q]
            t = tgt_pts[p]
            push!(segments, (s[1], s[2], t[1], t[2]))
        end
        bbox = _bbox_from_points(vcat(src_pts, tgt_pts))
        return VisualizationSpec(:pushforward_overlay;
                                 title="Finite map overlay",
                                 subtitle="segments join source vertices to their targets",
                                 layers=AbstractVisualizationLayer[
                                     SegmentLayer(segments, :gray50, 0.6, 1.0),
                                     PointLayer(src_pts, :royalblue, 0.95, 10.0),
                                     PointLayer(tgt_pts, :crimson, 0.95, 10.0),
                                 ],
                                 axes=_default_axes_2d(xlabel="x", ylabel="y",
                                                       xlimits=bbox === nothing ? nothing : bbox[1],
                                                       ylimits=bbox === nothing ? nothing : bbox[2],
                                                       aspect=:equal),
                                 metadata=(; object=:encoding_map, nsource=nvertices(source_poset(pi)), ntarget=nvertices(target_poset(pi))),
                                 legend=_default_legend(visible=true, entries=(; source=:royalblue, target=:crimson, map=:gray50)))
    end
    throw(ArgumentError("Unsupported EncodingMap visualization kind $(kind)."))
end

available_visuals(res::CommonRefinementTranslationResult) = _has_poset_coordinates_2d(common_poset(res)) ? (:common_refinement,) : ()

function _visual_spec(res::CommonRefinementTranslationResult, kind::Symbol; kwargs...)
    _ = kwargs
    kind === :common_refinement || throw(ArgumentError("Unsupported kind $(kind) for CommonRefinementTranslationResult."))
    P = common_poset(res)
    pts = _poset_coordinates_2d(P)
    pts === nothing && throw(ArgumentError("common_refinement visualization requires a 2D-embeddable common poset."))
    proj = projection_maps(res)
    left_map = region_map(proj.left)
    right_map = region_map(proj.right)
    labels = [string("(", left_map[i], ",", right_map[i], ")") for i in eachindex(left_map)]
    bbox = _bbox_from_points(pts)
    return VisualizationSpec(:common_refinement;
                             title="Common refinement",
                             subtitle="common regions labeled by left/right projection indices",
                             layers=AbstractVisualizationLayer[
                                 PointLayer(pts, :purple3, 0.95, 12.0),
                                 _text_layer_from_labels(pts, labels; color=:black, textsize=9.0),
                             ],
                             axes=_default_axes_2d(xlabel="common x", ylabel="common y",
                                                   xlimits=bbox === nothing ? nothing : bbox[1],
                                                   ylimits=bbox === nothing ? nothing : bbox[2],
                                                   aspect=:equal),
                             metadata=(; object=:common_refinement, ncommon=nvertices(P), left_target=nvertices(target_poset(proj.left)), right_target=nvertices(target_poset(proj.right))))
end

available_visuals(res::ModuleTranslationResult) = begin
    map = translation_map(res)
    map isa EncodingMap && _has_poset_coordinates_2d(source_poset(map)) && _has_poset_coordinates_2d(target_poset(map)) ?
        (:pushforward_overlay,) : ()
end

function _visual_spec(res::ModuleTranslationResult, kind::Symbol; kwargs...)
    kind === :pushforward_overlay || throw(ArgumentError("Unsupported kind $(kind) for ModuleTranslationResult."))
    return _visual_spec(translation_map(res), :pushforward_overlay; kwargs...)
end
