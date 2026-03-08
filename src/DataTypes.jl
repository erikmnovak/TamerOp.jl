# =============================================================================
# DataTypes.jl
#
# Shared typed ingestion/data containers used by ingestion, serialization, and
# workflow layers.
# =============================================================================
module DataTypes

using SparseArrays

"""
    PointCloud(points)

Minimal point cloud container. `points` is an n-by-d matrix (rows are points),
or a vector of coordinate vectors.
"""
struct PointCloud{T}
    points::Vector{Vector{T}}
end

function PointCloud(points::AbstractMatrix{T}) where {T}
    pts = [Vector{T}(points[i, :]) for i in 1:size(points, 1)]
    return PointCloud{T}(pts)
end

PointCloud(points::AbstractVector{<:AbstractVector{T}}) where {T} =
    PointCloud{T}([Vector{T}(p) for p in points])

"""
    ImageNd(data)

Minimal N-dim image/scalar field container. `data` is an N-dim array.
"""
struct ImageNd{T,N}
    data::Array{T,N}
end

ImageNd(data::Array{T,N}) where {T,N} = ImageNd{T,N}(data)

"""
    GraphData(n, edges; coords=nothing, weights=nothing)

Minimal graph container. `edges` is a vector of (u,v) pairs (1-based).
Optional `coords` can store embeddings, and `weights` can store edge weights.
"""
struct GraphData{T}
    n::Int
    edges::Vector{Tuple{Int,Int}}
    coords::Union{Nothing, Vector{Vector{T}}}
    weights::Union{Nothing, Vector{T}}
end

function GraphData(n::Integer, edges::AbstractVector{<:Tuple{Int,Int}};
                   coords::Union{Nothing, AbstractVector{<:AbstractVector}}=nothing,
                   weights::Union{Nothing, AbstractVector}=nothing,
                   T::Type=Float64)
    coords_vec = coords === nothing ? nothing : [Vector{T}(c) for c in coords]
    weights_vec = weights === nothing ? nothing : Vector{T}(weights)
    return GraphData{T}(Int(n), Vector{Tuple{Int,Int}}(edges), coords_vec, weights_vec)
end

"""
    EmbeddedPlanarGraph2D(vertices, edges; polylines=nothing, bbox=nothing)

Embedded planar graph container for 2D applications.
"""
struct EmbeddedPlanarGraph2D{T}
    vertices::Vector{Vector{T}}
    edges::Vector{Tuple{Int,Int}}
    polylines::Union{Nothing, Vector{Vector{Vector{T}}}}
    bbox::Union{Nothing, NTuple{4,T}}
end

function EmbeddedPlanarGraph2D(vertices::AbstractVector{<:AbstractVector{T}},
                               edges::AbstractVector{<:Tuple{Int,Int}};
                               polylines::Union{Nothing, AbstractVector}=nothing,
                               bbox::Union{Nothing, NTuple{4,T}}=nothing) where {T}
    verts = [Vector{T}(v) for v in vertices]
    polys = polylines === nothing ? nothing : [[Vector{T}(p) for p in poly] for poly in polylines]
    return EmbeddedPlanarGraph2D{T}(verts, Vector{Tuple{Int,Int}}(edges), polys, bbox)
end

"""
    GradedComplex(cells_by_dim, boundaries, grades; cell_dims=nothing)

Generic graded cell complex container.
"""
struct GradedComplex{N,T}
    cells_by_dim::Vector{Vector{Int}}
    boundaries::Vector{SparseMatrixCSC{Int,Int}}
    grades::Vector{NTuple{N,T}}
    cell_dims::Vector{Int}
end

function _cell_dims_from_cells(cells_by_dim::Vector{Vector{Int}})
    out = Int[]
    for (d, cells) in enumerate(cells_by_dim)
        for _ in cells
            push!(out, d - 1)
        end
    end
    return out
end

function GradedComplex(cells_by_dim::Vector{Vector{Int}},
                       boundaries::Vector{SparseMatrixCSC{Int,Int}},
                       grades::Vector{<:AbstractVector{T}};
                       cell_dims::Union{Nothing,Vector{Int}}=nothing) where {T}
    total = sum(length.(cells_by_dim))
    if cell_dims === nothing
        if length(grades) == total
            cell_dims = _cell_dims_from_cells(cells_by_dim)
        else
            cell_dims = fill(0, length(grades))
        end
    end
    N = length(grades[1])
    ng = Vector{NTuple{N,T}}(undef, length(grades))
    for i in eachindex(grades)
        length(grades[i]) == N || error("GradedComplex: grade $i has wrong length.")
        ng[i] = ntuple(j -> T(grades[i][j]), N)
    end
    return GradedComplex{N,T}(cells_by_dim, boundaries, ng, cell_dims)
end

function GradedComplex(cells_by_dim::Vector{Vector{Int}},
                       boundaries::Vector{SparseMatrixCSC{Int,Int}},
                       grades::Vector{<:Tuple};
                       cell_dims::Union{Nothing,Vector{Int}}=nothing)
    total = sum(length.(cells_by_dim))
    if cell_dims === nothing
        if length(grades) == total
            cell_dims = _cell_dims_from_cells(cells_by_dim)
        else
            cell_dims = fill(0, length(grades))
        end
    end
    N = length(grades[1])
    T = eltype(grades[1])
    ng = Vector{NTuple{N,T}}(undef, length(grades))
    for i in eachindex(grades)
        length(grades[i]) == N || error("GradedComplex: grade $i has wrong length.")
        ng[i] = ntuple(j -> T(grades[i][j]), N)
    end
    return GradedComplex{N,T}(cells_by_dim, boundaries, ng, cell_dims)
end

"""
    MultiCriticalGradedComplex(cells_by_dim, boundaries, grades; cell_dims=nothing)

Graded cell complex where each cell can carry multiple minimal grades.
"""
struct MultiCriticalGradedComplex{N,T}
    cells_by_dim::Vector{Vector{Int}}
    boundaries::Vector{SparseMatrixCSC{Int,Int}}
    grades::Vector{Vector{NTuple{N,T}}}
    cell_dims::Vector{Int}
end

function MultiCriticalGradedComplex(cells_by_dim::Vector{Vector{Int}},
                                    boundaries::Vector{SparseMatrixCSC{Int,Int}},
                                    grades::Vector{<:AbstractVector{<:Tuple}};
                                    cell_dims::Union{Nothing,Vector{Int}}=nothing)
    total = sum(length.(cells_by_dim))
    length(grades) == total || error("MultiCriticalGradedComplex: grades length mismatch.")
    cell_dims = cell_dims === nothing ? _cell_dims_from_cells(cells_by_dim) : cell_dims
    length(cell_dims) == total || error("MultiCriticalGradedComplex: cell_dims length mismatch.")

    first_cell = findfirst(!isempty, grades)
    first_cell === nothing && error("MultiCriticalGradedComplex: each cell must have at least one grade.")
    first_grade = grades[first_cell][1]
    N = length(first_grade)
    T = eltype(first_grade)

    ng = Vector{Vector{NTuple{N,T}}}(undef, length(grades))
    for i in eachindex(grades)
        gi = grades[i]
        isempty(gi) && error("MultiCriticalGradedComplex: cell $i has empty grade set.")
        out = Vector{NTuple{N,T}}(undef, length(gi))
        for j in eachindex(gi)
            g = gi[j]
            length(g) == N || error("MultiCriticalGradedComplex: grade length mismatch at cell $i.")
            out[j] = ntuple(k -> T(g[k]), N)
        end
        ng[i] = unique(out)
    end
    return MultiCriticalGradedComplex{N,T}(cells_by_dim, boundaries, ng, cell_dims)
end

function MultiCriticalGradedComplex(cells_by_dim::Vector{Vector{Int}},
                                    boundaries::Vector{SparseMatrixCSC{Int,Int}},
                                    grades::Vector{<:AbstractVector{<:AbstractVector{T}}};
                                    cell_dims::Union{Nothing,Vector{Int}}=nothing) where {T}
    total = sum(length.(cells_by_dim))
    length(grades) == total || error("MultiCriticalGradedComplex: grades length mismatch.")
    cell_dims = cell_dims === nothing ? _cell_dims_from_cells(cells_by_dim) : cell_dims
    length(cell_dims) == total || error("MultiCriticalGradedComplex: cell_dims length mismatch.")

    first_cell = findfirst(!isempty, grades)
    first_cell === nothing && error("MultiCriticalGradedComplex: each cell must have at least one grade.")
    N = length(grades[first_cell][1])
    ng = Vector{Vector{NTuple{N,T}}}(undef, length(grades))
    for i in eachindex(grades)
        gi = grades[i]
        isempty(gi) && error("MultiCriticalGradedComplex: cell $i has empty grade set.")
        out = Vector{NTuple{N,T}}(undef, length(gi))
        for j in eachindex(gi)
            g = gi[j]
            length(g) == N || error("MultiCriticalGradedComplex: grade length mismatch at cell $i.")
            out[j] = ntuple(k -> T(g[k]), N)
        end
        ng[i] = unique(out)
    end
    return MultiCriticalGradedComplex{N,T}(cells_by_dim, boundaries, ng, cell_dims)
end

"""
    SimplexTreeMulti(simplex_offsets, simplex_vertices, simplex_dims,
                     dim_offsets, grade_offsets, grade_data)

Compact simplicial multifiltration container with packed storage.
"""
struct SimplexTreeMulti{N,T}
    simplex_offsets::Vector{Int}
    simplex_vertices::Vector{Int}
    simplex_dims::Vector{Int}
    dim_offsets::Vector{Int}
    grade_offsets::Vector{Int}
    grade_data::Vector{NTuple{N,T}}
end

function SimplexTreeMulti(simplex_offsets::Vector{Int},
                          simplex_vertices::Vector{Int},
                          simplex_dims::Vector{Int},
                          dim_offsets::Vector{Int},
                          grade_offsets::Vector{Int},
                          grade_data::Vector{NTuple{N,T}}) where {N,T}
    ns = length(simplex_dims)
    length(simplex_offsets) == ns + 1 ||
        error("SimplexTreeMulti: simplex_offsets must have length nsimplices+1.")
    length(grade_offsets) == ns + 1 ||
        error("SimplexTreeMulti: grade_offsets must have length nsimplices+1.")
    !isempty(dim_offsets) || error("SimplexTreeMulti: dim_offsets cannot be empty.")
    first(simplex_offsets) == 1 || error("SimplexTreeMulti: simplex_offsets must start at 1.")
    first(grade_offsets) == 1 || error("SimplexTreeMulti: grade_offsets must start at 1.")
    last(simplex_offsets) == length(simplex_vertices) + 1 ||
        error("SimplexTreeMulti: simplex_offsets terminator mismatch.")
    last(grade_offsets) == length(grade_data) + 1 ||
        error("SimplexTreeMulti: grade_offsets terminator mismatch.")
    last(dim_offsets) == ns + 1 ||
        error("SimplexTreeMulti: dim_offsets terminator mismatch.")
    for i in 1:ns
        simplex_offsets[i] <= simplex_offsets[i + 1] ||
            error("SimplexTreeMulti: simplex_offsets must be nondecreasing.")
        grade_offsets[i] < grade_offsets[i + 1] ||
            error("SimplexTreeMulti: each simplex must have at least one grade.")
    end
    return SimplexTreeMulti{N,T}(simplex_offsets, simplex_vertices, simplex_dims,
                                 dim_offsets, grade_offsets, grade_data)
end

@inline simplex_count(ST::SimplexTreeMulti) = length(ST.simplex_dims)
@inline max_simplex_dim(ST::SimplexTreeMulti) = isempty(ST.simplex_dims) ? -1 : maximum(ST.simplex_dims)

@inline function simplex_vertices(ST::SimplexTreeMulti, i::Integer)
    ii = Int(i)
    1 <= ii <= simplex_count(ST) || throw(BoundsError(ST.simplex_dims, ii))
    lo = ST.simplex_offsets[ii]
    hi = ST.simplex_offsets[ii + 1] - 1
    return @view ST.simplex_vertices[lo:hi]
end

@inline function simplex_grades(ST::SimplexTreeMulti, i::Integer)
    ii = Int(i)
    1 <= ii <= simplex_count(ST) || throw(BoundsError(ST.simplex_dims, ii))
    lo = ST.grade_offsets[ii]
    hi = ST.grade_offsets[ii + 1] - 1
    return @view ST.grade_data[lo:hi]
end

end # module DataTypes
