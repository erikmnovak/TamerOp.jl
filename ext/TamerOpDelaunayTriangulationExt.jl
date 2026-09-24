module TamerOpDelaunayTriangulationExt

import DelaunayTriangulation as DT
using Random: Xoshiro

import TamerOp

const TO = TamerOp

const DI = TO.DataIngestion

@inline function _points2d(points)
    n = length(points)
    coords = Vector{NTuple{2,Float64}}(undef, n)
    @inbounds for i in 1:n
        p = points[i]
        length(p) >= 2 || throw(ArgumentError("PointCloud point dimension mismatch in Delaunay backend."))
        coords[i] = (Float64(p[1]), Float64(p[2]))
    end
    return coords
end

function _packed_delaunay_2d(points; max_dim::Int=2)
    n = length(points)
    n == 0 && return DI._PackedDelaunay2D(NTuple{2,Int}[], Float64[], NTuple{3,Int}[], Float64[])
    coords = _points2d(points)
    # A local fixed seed makes the chosen triangulation of cocircular inputs
    # reproducible without reading or modifying the caller's random stream.
    tri = DT.triangulate(coords; rng=Xoshiro(0))

    edges = NTuple{2,Int}[]
    sizehint!(edges, max(0, 3n))
    @inbounds for e in DT.each_solid_edge(tri)
        i, j = e
        i == j && continue
        a, b = i < j ? (i, j) : (j, i)
        push!(edges, (a, b))
    end
    sort!(edges)
    edge_radius = Float64[]
    sizehint!(edge_radius, length(edges))
    @inbounds for (a, b) in edges
        push!(edge_radius, DI._half_point_distance(coords[a], coords[b]))
    end

    triangles = NTuple{3,Int}[]
    tri_radius = Float64[]
    if max_dim >= 2
        sizehint!(triangles, max(0, 2n))
        sizehint!(tri_radius, max(0, 2n))
        @inbounds for t in DT.each_solid_triangle(tri)
            i, j, k = t
            (i == j || i == k || j == k) && continue
            a, b, c = DI._sort_triplet(i, j, k)
            push!(triangles, (a, b, c))
        end
        sort!(triangles)
        for (a, b, c) in triangles
            push!(tri_radius, DI._delaunay_circumradius(coords[a], coords[b], coords[c]))
        end
    end

    return DI._PackedDelaunay2D(edges, edge_radius, triangles, tri_radius)
end

function __init__()
    DI._set_pointcloud_delaunay_2d_impl!((points; max_dim::Int=2) ->
        _packed_delaunay_2d(points; max_dim=max_dim))
    return nothing
end

end # module
