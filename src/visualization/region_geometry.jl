# Exact geometry for planar classifier fibers. Float64 is a drawing format only.
const _VisualRational = Rational{BigInt}
const _VisualCoordinate = Union{_VisualRational,AlgebraicReal}
const _VisualPoint = NTuple{2,_VisualCoordinate}

_visual_exact_coordinate(x::Real) = _VisualRational(x)
_visual_exact_coordinate(x::AlgebraicReal) = x
_visual_point(p) = (_visual_exact_coordinate(p[1]), _visual_exact_coordinate(p[2]))

function _drawing_point(p)
    q = (Float64(p[1]), Float64(p[2]))
    all(isfinite, q) || throw(ArgumentError("visualization coordinates must be representable as finite Float64 values; rescale the viewing window"))
    return q
end

function _visual_box_2d(box)
    length(box) == 2 && length(box[1]) == 2 && length(box[2]) == 2 ||
        throw(ArgumentError("box must contain two 2D corners"))
    all(isfinite, box[1]) && all(isfinite, box[2]) || throw(ArgumentError("box must have finite corners"))
    lo, hi = _visual_point(box[1]), _visual_point(box[2])
    all(lo[i] < hi[i] for i in 1:2) || throw(ArgumentError("box must have positive width on both axes"))
    _drawing_point(lo); _drawing_point(hi)
    return (lo, hi)
end

function _visual_default_box_2d(points)
    pts = [_visual_point(p) for p in points]
    isempty(pts) && return ((-1//big(1), -1//big(1)), (1//big(1), 1//big(1)))
    lo = ntuple(i -> minimum(p[i] for p in pts), 2)
    hi = ntuple(i -> maximum(p[i] for p in pts), 2)
    pad = ntuple(i -> max(one(_VisualRational), (hi[i] - lo[i]) / 10), 2)
    return (ntuple(i -> lo[i] - pad[i], 2), ntuple(i -> hi[i] + pad[i], 2))
end

function _geometry_axes_2d(box)
    lo, hi = _drawing_point(box[1]), _drawing_point(box[2])
    # A rational interval can be narrower than one display ulp. Keep the exact
    # window in metadata and expand only its plotted limits, with a warning.
    limits(i) = lo[i] < hi[i] ? (lo[i], hi[i]) : (prevfloat(lo[i]), nextfloat(hi[i]))
    return _default_axes_2d(xlabel="x1", ylabel="x2", xlimits=limits(1), ylimits=limits(2), aspect=:equal)
end

function _display_collisions(points)
    groups = Dict{NTuple{2,Float64},Vector{_VisualPoint}}()
    for p in points
        exact = _visual_point(p)
        group = get!(groups, _drawing_point(exact), _VisualPoint[])
        exact in group || push!(group, exact)
    end
    return [(; display_point=q, exact_points=ps) for (q, ps) in groups if length(ps) > 1]
end

# Sutherland-Hodgman clipping with rational arithmetic. Unlike polygon-only
# wrappers this retains a line or point when the intersection loses dimension.
function _clip_visual_halfspace(vertices::Vector{_VisualPoint}, a, b)
    isempty(vertices) && return vertices
    out = _VisualPoint[]
    previous = last(vertices)
    pv = a[1] * previous[1] + a[2] * previous[2] - b
    for current in vertices
        cv = a[1] * current[1] + a[2] * current[2] - b
        if (pv <= 0) != (cv <= 0)
            t = pv / (pv - cv)
            push!(out, (previous[1] + t * (current[1] - previous[1]),
                        previous[2] + t * (current[2] - previous[2])))
        end
        cv <= 0 && push!(out, current)
        previous, pv = current, cv
    end
    unique!(out)
    return out
end

function _visual_polygon_dimension(vertices)
    length(vertices) <= 1 && return length(vertices) - 1
    a, b = first(vertices), vertices[2]
    return any((b[1] - a[1]) * (p[2] - a[2]) != (b[2] - a[2]) * (p[1] - a[1]) for p in vertices) ? 2 : 1
end

function _visual_component(pi, region_id::Int, vertices::Vector{_VisualPoint}, box)
    dim = _visual_polygon_dimension(vertices)
    dim == 1 && (vertices = sort(vertices)[[1, end]])
    included = BitVector(_visual_locate(pi, p) == region_id for p in vertices)
    pairs = dim == 2 ? [(i, mod1(i + 1, length(vertices))) for i in eachindex(vertices)] :
            dim == 1 ? [(1, 2)] : Tuple{Int,Int}[]
    edge_included = BitVector()
    edge_clipped = BitVector()
    for (i, j) in pairs
        a, b = vertices[i], vertices[j]
        mid = ((a[1] + b[1]) / 2, (a[2] + b[2]) / 2)
        push!(edge_included, _visual_locate(pi, mid) == region_id)
        push!(edge_clipped, any(a[k] == b[k] && (a[k] == box[1][k] || a[k] == box[2][k]) for k in 1:2))
    end
    return (; region_id, vertices, dimension=dim, vertex_included=included, edge_included, edge_clipped)
end

_visual_locate(pi, p) = locate(pi, collect(p))
# Reversed axes negate coordinates in the grid classifier. Widen before that
# operation so typemin(Int) and unsigned coordinates retain their exact order.
_visual_locate(pi::GridEncodingMap{2}, p) = locate(pi, collect(_visual_point(p)))
# Zn's real convenience classifier rounds to the nearest lattice point. Match
# that rule exactly, without an intermediate floating conversion.
_visual_locate(pi::ZnEncodingMap, p) = locate(pi, (round(Int, p[1]), round(Int, p[2])))

function _geometry_result(components, box; geometry_kind=:classifier_fibers)
    area(c) = c.dimension == 2 ? abs(sum(c.vertices[i][1] * c.vertices[mod1(i+1, length(c.vertices))][2] -
        c.vertices[i][2] * c.vertices[mod1(i+1, length(c.vertices))][1] for i in eachindex(c.vertices))) / 2 : zero(_VisualRational)
    covered_area = sum((area(c) for c in components if c.region_id != 0); init=zero(_VisualRational))
    box_area = (box[2][1] - box[1][1]) * (box[2][2] - box[1][2])
    has_unrepresented_area = covered_area < box_area
    region_ids = sort!(unique([c.region_id for c in components]))
    has_unrepresented_area && !(0 in region_ids) && pushfirst!(region_ids, 0)
    vertices = _VisualPoint[p for c in components for p in c.vertices]
    append!(vertices, box)
    dimension_collapses = [(; region_id=c.region_id, exact_dimension=c.dimension,
        display_dimension=_visual_polygon_dimension(unique([_drawing_point(p) for p in c.vertices])), exact_vertices=c.vertices)
        for c in components if _visual_polygon_dimension(unique([_drawing_point(p) for p in c.vertices])) < c.dimension]
    return (; components, region_ids, box, has_unrepresented_area, dimension_collapses,
            axes=_geometry_axes_2d(box), coordinate_collisions=_display_collisions(vertices), geometry_kind)
end

function _rectangular_geometry_2d(pi, splits; box=nothing, geometry_kind=:classifier_fibers)
    default_points = if all(!isempty, splits)
        [ntuple(i -> minimum(splits[i]), 2), ntuple(i -> maximum(splits[i]), 2)]
    else
        collect(encoding_representatives(pi))
    end
    view = _visual_box_2d(box === nothing ? _visual_default_box_2d(default_points) : box)
    edges = ntuple(i -> sort!(unique(vcat([view[1][i]],
        [_visual_exact_coordinate(c) for c in splits[i] if view[1][i] < c < view[2][i]], [view[2][i]]))), 2)
    components = NamedTuple[]
    for j in 1:(length(edges[2]) - 1), i in 1:(length(edges[1]) - 1)
        xl, xr = edges[1][i], edges[1][i+1]
        yl, yr = edges[2][j], edges[2][j+1]
        r = _visual_locate(pi, ((xl + xr) / 2, (yl + yr) / 2))
        vertices = _VisualPoint[(xl, yl), (xr, yl), (xr, yr), (xl, yr)]
        push!(components, _visual_component(pi, r, vertices, view))
    end
    # The viewport can retain a fiber only along its boundary. Also retain an
    # exceptional classifier label on an internal face instead of assigning
    # that face to whichever area cell happens to be painted last.
    nx, ny = length(edges[1]) - 1, length(edges[2]) - 1
    cell_id(i, j) = components[(j - 1) * nx + i].region_id
    for j in eachindex(edges[2]), i in 1:nx
        a, b = (edges[1][i], edges[2][j]), (edges[1][i+1], edges[2][j])
        r = _visual_locate(pi, ((a[1] + b[1])/2, a[2]))
        carried = (j > 1 && cell_id(i, j-1) == r) || (j <= ny && cell_id(i, j) == r)
        carried || push!(components, _visual_component(pi, r, _VisualPoint[a, b], view))
    end
    for j in 1:ny, i in eachindex(edges[1])
        a, b = (edges[1][i], edges[2][j]), (edges[1][i], edges[2][j+1])
        r = _visual_locate(pi, (a[1], (a[2] + b[2])/2))
        carried = (i > 1 && cell_id(i-1, j) == r) || (i <= nx && cell_id(i, j) == r)
        carried || push!(components, _visual_component(pi, r, _VisualPoint[a, b], view))
    end
    represented_vertices = Set{Tuple{_VisualPoint,Int}}()
    for c in components, (p, included) in zip(c.vertices, c.vertex_included)
        included && push!(represented_vertices, (p, c.region_id))
    end
    for y in edges[2], x in edges[1]
        p = (x, y)
        r = _visual_locate(pi, p)
        (p, r) in represented_vertices || push!(components, _visual_component(pi, r, _VisualPoint[p], view))
    end
    return _geometry_result(components, view; geometry_kind)
end

function _region_geometry_2d(pi::GridEncodingMap{2}; box=nothing)
    splits = ntuple(i -> sort!([pi.orientation[i] * _visual_exact_coordinate(x) for x in pi.coords[i]]), 2)
    return _rectangular_geometry_2d(pi, splits; box)
end

function _region_geometry_2d(pi::PLEncodingMapBoxes{2}; box=nothing)
    return _rectangular_geometry_2d(pi, pi.coords; box)
end

function _region_geometry_2d(pi::ZnEncodingMap{2}; box=nothing)
    # A slab starting at integer c starts at c-1/2 in a unit-tile drawing.
    splits = ntuple(i -> [_visual_exact_coordinate(c) - 1//2 for c in pi.coords[i]], 2)
    return _rectangular_geometry_2d(pi, splits; box, geometry_kind=:nearest_lattice_tiles)
end

function _region_geometry_2d(pi::PLEncodingMap; box=nothing)
    pi.n == 2 || throw(ArgumentError("region geometry is implemented only for two parameters"))
    view = _visual_box_2d(box === nothing ? _visual_default_box_2d(encoding_representatives(pi)) : box)
    lo, hi = view
    components = NamedTuple[]
    for (r, hp) in enumerate(pi.regions)
        vertices = _VisualPoint[(lo[1], lo[2]), (hi[1], lo[2]), (hi[1], hi[2]), (lo[1], hi[2])]
        for i in axes(hp.A, 1)
            bound = hp.b[i] + (hp.strict_mask[i] ? hp.strict_eps : zero(hp.strict_eps))
            vertices = _clip_visual_halfspace(vertices, (hp.A[i,1], hp.A[i,2]), bound)
            isempty(vertices) && break
        end
        isempty(vertices) && continue
        # A relative-interior point tests whether the clipped closure actually
        # meets every strict inequality, including a box touching an open face.
        center = (sum(p[1] for p in vertices) / length(vertices), sum(p[2] for p in vertices) / length(vertices))
        _visual_locate(pi, center) == r || continue
        push!(components, _visual_component(pi, r, vertices, view))
    end
    return _geometry_result(components, view)
end

function _region_geometry_layers(geometry; colors=nothing)
    layers = AbstractVisualizationLayer[]
    if geometry.has_unrepresented_area
        lo, hi = _drawing_point(geometry.box[1]), _drawing_point(geometry.box[2])
        background = colors === nothing ? :gray90 : get(colors, 0, :gray90)
        push!(layers, RectLayer([(lo..., hi...)], background, background, 1.0, 0.0))
    end
    # Draw full-dimensional fills first so face-only regions stay visible.
    for c in geometry.components
        c.dimension == 2 || continue
        color = colors === nothing ? (c.region_id == 0 ? :gray90 : _box_region_color(c.region_id)) : get(colors, c.region_id, :gray90)
        points = NTuple{2,Float64}[_drawing_point(p) for p in c.vertices]
        # A mathematically nonzero cell can collapse at drawing precision.
        # Its exact geometry and visible warning survive; do not ask a backend
        # to triangulate a degenerate polygon. Boundary markers still render.
        _visual_polygon_dimension(unique(points)) == 2 || continue
        push!(layers, PolygonLayer([points], color, color, 0.28, 0.0))
    end
    # The cells are an exact subdivision of each classifier fiber. An edge
    # shared by two included cells of the same fiber is an internal mesh edge,
    # not a region boundary. Keep cuts, excluded faces, and lower-dimensional
    # components. Compare exact endpoints so drawing collisions cannot cancel
    # mathematically distinct boundaries.
    edge_key(c, i) = begin
        a, b = c.vertices[i], c.vertices[mod1(i + 1, length(c.vertices))]
        isless(b, a) ? (c.region_id, b, a) : (c.region_id, a, b)
    end
    included_edges = Dict{Tuple{Int,_VisualPoint,_VisualPoint},Int}()
    for c in geometry.components
        c.dimension == 2 || continue
        for i in eachindex(c.edge_included)
            c.edge_included[i] && !c.edge_clipped[i] || continue
            key = edge_key(c, i)
            included_edges[key] = get(included_edges, key, 0) + 1
        end
    end
    for style in (:dot, :dash, :solid), c in geometry.components
        color = colors === nothing ? (c.region_id == 0 ? :gray55 : _box_region_color(c.region_id)) : get(colors, c.region_id, :gray55)
        segs = NTuple{4,Float64}[]
        for i in eachindex(c.edge_included)
            edge_style = c.edge_clipped[i] ? :dot : c.edge_included[i] ? :solid : :dash
            style == edge_style || continue
            c.dimension == 2 && edge_style === :solid && get(included_edges, edge_key(c, i), 0) > 1 && continue
            a, b = _drawing_point(c.vertices[i]), _drawing_point(c.vertices[mod1(i+1, length(c.vertices))])
            push!(segs, (a..., b...))
        end
        isempty(segs) || push!(layers, SegmentLayer(segs, color, 0.95, c.dimension == 1 ? 3.0 : 1.6, style))
    end
    # Open markers go underneath included vertices, so a shared boundary never
    # paints over the point's actual owning region with a later open marker.
    for included in (false, true), c in geometry.components
        color = colors === nothing ? (c.region_id == 0 ? :gray55 : _box_region_color(c.region_id)) : get(colors, c.region_id, :gray55)
        points = NTuple{2,Float64}[_drawing_point(p) for (p, yes) in zip(c.vertices, c.vertex_included) if yes == included]
        isempty(points) && continue
        if included
            push!(layers, PointLayer(points, color, 1.0, c.dimension == 0 ? 11.0 : 4.0))
        else
            push!(layers, PointLayer(points, color, 0.95, 6.0))
            push!(layers, PointLayer(points, :white, 1.0, 3.0))
        end
    end
    return layers
end
