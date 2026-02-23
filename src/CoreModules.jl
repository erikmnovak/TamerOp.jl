module CoreModules
# -----------------------------------------------------------------------------
# Core prelude for this project
#  - QQ = Rational{BigInt} is the canonical exact scalar type
#  - Optional feature flags (e.g. for optional PL axis backend)
#  - Thin wrappers for exact linear algebra backends (optional Nemo)
#  - Exact rational <-> string helpers for serialization
# -----------------------------------------------------------------------------

using LinearAlgebra, SparseArrays

# ----- canonical field of scalars used everywhere --------------------------------
"Exact rationals used throughout (Rational{BigInt})."
const QQ = Rational{BigInt}


# ----- coefficient field layer ---------------------------------------------------
module CoeffFields
using LinearAlgebra
using Random
using ..CoreModules: QQ


"Abstract supertype for coefficient fields."
abstract type AbstractCoeffField end

"Exact rationals (QQ)."
struct QQField <: AbstractCoeffField end

"Real (floating) field with tolerances."
struct RealField{T<:AbstractFloat} <: AbstractCoeffField
    rtol::T
    atol::T
end

function RealField(::Type{T}; rtol::T = sqrt(eps(T)), atol::T = zero(T)) where {T<:AbstractFloat}
    RealField{T}(rtol, atol)
end

"Prime field of characteristic p."
struct PrimeField <: AbstractCoeffField
    p::Int
    function PrimeField(p::Integer)
        p > 1 || throw(ArgumentError("prime field requires p > 1"))
        new(Int(p))
    end
end

Base.:(==)(::QQField, ::QQField) = true
Base.:(==)(a::RealField{T}, b::RealField{T}) where {T<:AbstractFloat} =
    (a.rtol == b.rtol) && (a.atol == b.atol)
Base.:(==)(a::PrimeField, b::PrimeField) = a.p == b.p

F2() = PrimeField(2)
F3() = PrimeField(3)
Fp(p::Integer) = PrimeField(p)

"Element type for prime fields, parameterized by modulus."
struct FpElem{p} <: Integer
    val::Int
    function FpElem{p}(x::Integer) where {p}
        x isa FpElem{p} && return x
        new{p}(mod(x, p))
    end
end

Base.show(io::IO, x::FpElem{p}) where {p} = print(io, x.val)
Base.zero(::Type{FpElem{p}}) where {p} = FpElem{p}(0)
Base.one(::Type{FpElem{p}}) where {p} = FpElem{p}(1)
Base.iszero(x::FpElem{p}) where {p} = x.val == 0
Base.conj(x::FpElem{p}) where {p} = x
Base.hash(x::FpElem{p}, h::UInt) where {p} = hash(x.val, h)

Base.convert(::Type{FpElem{p}}, x::Integer) where {p} = FpElem{p}(x)
Base.convert(::Type{FpElem{p}}, x::FpElem{p}) where {p} = x
Base.promote_rule(::Type{FpElem{p}}, ::Type{<:Integer}) where {p} = FpElem{p}

Base.:+(a::FpElem{p}, b::FpElem{p}) where {p} = FpElem{p}(a.val + b.val)
Base.:-(a::FpElem{p}, b::FpElem{p}) where {p} = FpElem{p}(a.val - b.val)
Base.:-(a::FpElem{p}) where {p} = FpElem{p}(-a.val)
Base.:*(a::FpElem{p}, b::FpElem{p}) where {p} = FpElem{p}(a.val * b.val)
Base.:(==)(a::FpElem{p}, b::FpElem{p}) where {p} = a.val == b.val

function Base.inv(a::FpElem{p}) where {p}
    a.val == 0 && throw(DomainError(a, "division by zero in Fp"))
    return FpElem{p}(invmod(a.val, p))
end

Base.:/(a::FpElem{p}, b::FpElem{p}) where {p} = a * inv(b)

"Return the scalar element type used for a given field."
coeff_type(::QQField) = QQ
coeff_type(::RealField{T}) where {T<:AbstractFloat} = T
coeff_type(F::PrimeField) = FpElem{F.p}

"Infer a coefficient field object from an element type."
field_from_eltype(::Type{QQ}) = QQField()
field_from_eltype(::Type{<:Rational}) = QQField()
field_from_eltype(::Type{T}) where {T<:AbstractFloat} = RealField(T)
field_from_eltype(::Type{FpElem{p}}) where {p} = PrimeField(p)
field_from_eltype(::Type{K}) where {K} =
    throw(ArgumentError("no field mapping for element type $(K)"))

"Field-aware zero/one."
Base.zero(F::AbstractCoeffField) = zero(coeff_type(F))
Base.one(F::AbstractCoeffField) = one(coeff_type(F))

"Coerce scalars into the given coefficient field."
coerce(::QQField, x::Integer) = QQ(x)
coerce(::QQField, x::Rational) = QQ(x)
coerce(::QQField, x::AbstractFloat) = rationalize(BigInt, x)
coerce(::QQField, x::QQ) = x
coerce(::QQField, x::FpElem{p}) where {p} = QQ(x.val)

coerce(::RealField{T}, x::Integer) where {T<:AbstractFloat} = T(x)
coerce(::RealField{T}, x::Rational) where {T<:AbstractFloat} =
    T(numerator(x)) / T(denominator(x))
coerce(::RealField{T}, x::AbstractFloat) where {T<:AbstractFloat} = T(x)

function coerce(F::PrimeField, x::Integer)
    K = coeff_type(F)
    return K(x)
end

function coerce(F::PrimeField, x::Rational)
    p = F.p
    den = denominator(x)
    g = gcd(den, p)
    g == 1 || g == one(g) || throw(ArgumentError("denominator not invertible mod $p"))
    num = numerator(x)
    num_mod = Int(mod(num, p))
    den_mod = Int(mod(den, p))
    inv_den = invmod(den_mod, p)
    return FpElem{p}(num_mod * inv_den)
end

function coerce(F::PrimeField, x::FpElem{p}) where {p}
    F.p == p || throw(ArgumentError("cannot coerce FpElem{$p} into Fp($(F.p))"))
    return x
end

coerce(F::PrimeField, x::AbstractFloat) =
    throw(ArgumentError("cannot coerce float into Fp($(F.p)) without an explicit rule"))

"Allocate a dense zeros matrix over the field."
function zeros(F::AbstractCoeffField, m::Integer, n::Integer)
    K = coeff_type(F)
    A = Matrix{K}(undef, m, n)
    fill!(A, zero(K))
    return A
end

"Allocate a dense ones matrix over the field."
function ones(F::AbstractCoeffField, m::Integer, n::Integer)
    K = coeff_type(F)
    A = Matrix{K}(undef, m, n)
    fill!(A, one(K))
    return A
end

"Allocate a dense identity matrix over the field."
function eye(F::AbstractCoeffField, n::Integer)
    K = coeff_type(F)
    A = Matrix{K}(undef, n, n)
    z = zero(K)
    o = one(K)
    @inbounds for j in 1:n
        for i in 1:n
            A[i, j] = (i == j) ? o : z
        end
    end
    return A
end

"Allocate a dense random matrix over the field."
function rand(F::AbstractCoeffField, m::Integer, n::Integer; density::Real=1.0)
    (0.0 <= density <= 1.0) || throw(ArgumentError("density must be in [0,1]"))
    K = coeff_type(F)
    A = Matrix{K}(undef, m, n)

    if density == 1.0
        @inbounds for j in 1:n, i in 1:m
            A[i, j] = _rand_scalar(F)
        end
        return A
    end

    z = zero(K)
    @inbounds for j in 1:n, i in 1:m
        A[i, j] = (Base.rand() <= density) ? _rand_scalar(F) : z
    end
    return A
end

_rand_scalar(::QQField) = QQ(Base.rand(-5:5))
_rand_scalar(::RealField{T}) where {T<:AbstractFloat} = Base.rand(T)
_rand_scalar(F::PrimeField) = FpElem{F.p}(Base.rand(0:F.p-1))

end # module CoeffFields

using .CoeffFields: AbstractCoeffField, QQField, RealField, PrimeField,
    F2, F3, Fp, coeff_type, coerce, FpElem, field_from_eltype,
    eye, zeros, ones, rand

"""
    BackendMatrix{K}

Dense matrix wrapper that can carry an optional backend-native payload
(e.g. a Nemo matrix) to avoid repeated conversion in hot paths.
"""
mutable struct BackendMatrix{K} <: AbstractMatrix{K}
    data::Matrix{K}
    backend::Symbol
    payload::Any
    function BackendMatrix{K}(data::Matrix{K};
                              backend::Symbol=:nemo,
                              payload::Any=nothing) where {K}
        new{K}(data, backend, payload)
    end
end

BackendMatrix(A::AbstractMatrix{K}; backend::Symbol=:nemo, payload::Any=nothing) where {K} =
    BackendMatrix{K}(Matrix{K}(A); backend=backend, payload=payload)

Base.size(A::BackendMatrix) = size(A.data)
Base.axes(A::BackendMatrix) = axes(A.data)
Base.IndexStyle(::Type{<:BackendMatrix}) = IndexCartesian()
Base.getindex(A::BackendMatrix, i::Int, j::Int) = @inbounds A.data[i, j]
function Base.setindex!(A::BackendMatrix, v, i::Int, j::Int)
    @inbounds A.data[i, j] = v
    # Keep cached backend payload coherent with dense storage.
    A.payload = nothing
    return v
end
Base.parent(A::BackendMatrix) = A.data
Base.Matrix(A::BackendMatrix{K}) where {K} = copy(A.data)
Base.copy(A::BackendMatrix{K}) where {K} =
    BackendMatrix{K}(copy(A.data); backend=A.backend, payload=nothing)
Base.convert(::Type{Matrix{K}}, A::BackendMatrix{K}) where {K} = copy(A.data)
Base.convert(::Type{BackendMatrix{K}}, A::AbstractMatrix{K}) where {K} =
    BackendMatrix{K}(Matrix{K}(A); backend=:nemo, payload=nothing)

Base.:*(A::BackendMatrix{K}, B::AbstractMatrix{K}) where {K} = A.data * B
Base.:*(A::AbstractMatrix{K}, B::BackendMatrix{K}) where {K} = A * B.data
Base.:*(A::BackendMatrix{K}, B::BackendMatrix{K}) where {K} = A.data * B.data

@inline _unwrap_backend_matrix(A::BackendMatrix) = A.data
@inline _backend_kind(A::BackendMatrix) = A.backend
@inline _backend_payload(A::BackendMatrix) = A.payload
@inline function _set_backend_payload!(A::BackendMatrix, payload)
    A.payload = payload
    return A
end


"""
    change_field(x, field)

Coerce a structure into the specified coefficient field.
"""
function change_field end


"""
    _append_scaled_triplets!(I, J, V, A, row_off, col_off; scale=one(eltype(A)))
    _append_scaled_triplets!(I, J, V, A, rows, cols; scale=one(eltype(A)))

Internal helper for assembling sparse matrices via (I,J,V) triplets.

This eliminates the common anti-pattern:

    D = zeros(QQ, m, n)
    ... fill a few blocks/entries of D ...
    return sparse(D)

which allocates an m-by-n dense matrix even when the result is structurally sparse.

Two variants:

1. Offset-based (fast):
   Treat `A` as a contiguous block placed at global rows
       row_off .+ (1:size(A,1))
   and global cols
       col_off .+ (1:size(A,2))

2. Indexed (general):
   Place `A` at explicit global row indices `rows` and col indices `cols`.
   This supports non-contiguous placements (e.g. direct sums with custom order).

In both cases we append only nonzero entries of `A` (after scaling by `scale`).
Duplicates are allowed; `sparse(I,J,V,...)` will sum them.

All indexing is 1-based.
"""
function _append_scaled_triplets!(I::Vector{Int}, J::Vector{Int}, V::Vector{K},
                                 A::AbstractMatrix{K},
                                 row_off::Int, col_off::Int;
                                 scale::K = one(K)) where {K}
    # Sparse fast-path
    if A isa SparseMatrixCSC{K,Int}
        Ii, Jj, Vv = findnz(A)
        @inbounds for k in eachindex(Ii)
            a = Vv[k]
            iszero(a) && continue
            push!(I, row_off + Ii[k])
            push!(J, col_off + Jj[k])
            push!(V, scale * a)
        end
        return nothing
    end

    # Transpose/adjoint of sparse: iterate parent nnz and swap indices
    if (A isa LinearAlgebra.Transpose{K,<:SparseMatrixCSC{K,Int}}) ||
       (A isa LinearAlgebra.Adjoint{K,<:SparseMatrixCSC{K,Int}})
        S = parent(A)
        Ii, Jj, Vv = findnz(S)
        @inbounds for k in eachindex(Ii)
            a = Vv[k]
            iszero(a) && continue
            push!(I, row_off + Jj[k])
            push!(J, col_off + Ii[k])
            push!(V, scale * a)
        end
        return nothing
    end

    # Dense / generic: scan entries, skip zeros
    m, n = size(A)
    @inbounds for j in 1:n
        gj = col_off + j
        for i in 1:m
            a = A[i, j]
            iszero(a) && continue
            push!(I, row_off + i)
            push!(J, gj)
            push!(V, scale * a)
        end
    end
    return nothing
end

function _append_scaled_triplets!(I::Vector{Int}, J::Vector{Int}, V::Vector{K},
                                 A::AbstractMatrix{K},
                                 rows::AbstractVector{<:Integer},
                                 cols::AbstractVector{<:Integer};
                                 scale::K = one(K)) where {K}
    @assert length(rows) == size(A, 1)
    @assert length(cols) == size(A, 2)

    if A isa SparseMatrixCSC{K,Int}
        Ii, Jj, Vv = findnz(A)
        @inbounds for k in eachindex(Ii)
            a = Vv[k]
            iszero(a) && continue
            push!(I, Int(rows[Ii[k]]))
            push!(J, Int(cols[Jj[k]]))
            push!(V, scale * a)
        end
        return nothing
    end

    if (A isa LinearAlgebra.Transpose{K,<:SparseMatrixCSC{K,Int}}) ||
       (A isa LinearAlgebra.Adjoint{K,<:SparseMatrixCSC{K,Int}})
        S = parent(A)
        Ii, Jj, Vv = findnz(S)
        @inbounds for k in eachindex(Ii)
            a = Vv[k]
            iszero(a) && continue
            push!(I, Int(rows[Jj[k]]))
            push!(J, Int(cols[Ii[k]]))
            push!(V, scale * a)
        end
        return nothing
    end

    m, n = size(A)
    @inbounds for j in 1:n
        gj = Int(cols[j])
        for i in 1:m
            a = A[i, j]
            iszero(a) && continue
            push!(I, Int(rows[i]))
            push!(J, gj)
            push!(V, scale * a)
        end
    end
    return nothing
end


# ----- feature flags --------------------------------------------------------------



# ----- exact rational <-> string (for JSON round-trips) --------------------------
"Encode a rational as \"num/den\" so it survives JSON round-trips exactly."
rational_to_string(x::QQ) = string(numerator(x), "/", denominator(x))

"Inverse of `rational_to_string`."
function string_to_rational(s::AbstractString)::QQ
    t = split(strip(s), "/")
    length(t) == 2 || error("bad QQ string: $s")
    parse(BigInt, t[1]) // parse(BigInt, t[2])
end

# ----- common abstract supertype for "encoding map" objects --------------------
"""
    AbstractPLikeEncodingMap

Abstract supertype for "encoding map" objects `pi` used throughout the finite-encoding
and PL backends.

This is a dispatch hook only: it imposes no required fields. Concrete encoding maps
should subtype this and implement at least:

  - `locate(pi, x)`

Optionally they may implement geometric hooks (see RegionGeometry) such as:

  - `region_weights(pi; box=...)`
  - `region_bbox(pi, r; box=...)`
  - `region_boundary_measure(pi, r; box=...)`, etc.

The name "PLike" is informal: it refers to encodings that behave like a
piecewise-constant classifier on a stratification of parameter space.
"""
abstract type AbstractPLikeEncodingMap end

"Convenience alias used in type signatures in higher-level code."
const PLikeEncodingMap = AbstractPLikeEncodingMap


# -----------------------------------------------------------------------------
# Lightweight ingestion types (shared across Workflow + Serialization)
# -----------------------------------------------------------------------------

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

Embedded planar graph container for 2D applications (e.g. wing veins).
`vertices` is a vector of 2D coordinate vectors, `edges` are vertex index pairs.
`polylines` can store per-edge piecewise-linear geometry.
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
    polys = polylines === nothing ? nothing : [ [Vector{T}(p) for p in poly] for poly in polylines ]
    return EmbeddedPlanarGraph2D{T}(verts, Vector{Tuple{Int,Int}}(edges), polys, bbox)
end

"""
    GradedComplex(cells_by_dim, boundaries, grades; cell_dims=nothing)

Generic graded cell complex container ("escape hatch").
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
`grades[i]` is a non-empty vector of `NTuple{N,T}` grades for cell `i`.
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

    # Find first non-empty grade set to infer (N,T)
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

- `simplex_offsets` / `simplex_vertices` encode simplex vertex lists in CSR form.
- `simplex_dims[i]` is the dimension of simplex `i`.
- `dim_offsets[d+1]:(dim_offsets[d+2]-1)` selects simplices of dimension `d`.
- `grade_offsets` / `grade_data` encode minimal grade sets per simplex.
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

"""
    FiltrationSpec(; kind, params...)

Lightweight filtration specification container. Stores a `kind` symbol and a
`params` named tuple.
"""
struct FiltrationSpec
    kind::Symbol
    params::NamedTuple
end

FiltrationSpec(; kind::Symbol, params...) = FiltrationSpec(kind, NamedTuple(params))

"""
    ConstructionBudget(; max_simplices=nothing, max_edges=nothing, memory_budget_bytes=nothing)

Budget controls for data-ingestion construction.
"""
struct ConstructionBudget
    max_simplices::Union{Nothing,Int}
    max_edges::Union{Nothing,Int}
    memory_budget_bytes::Union{Nothing,Int}
end

ConstructionBudget(; max_simplices::Union{Nothing,Integer}=nothing,
                   max_edges::Union{Nothing,Integer}=nothing,
                   memory_budget_bytes::Union{Nothing,Integer}=nothing) =
    ConstructionBudget(max_simplices === nothing ? nothing : Int(max_simplices),
                       max_edges === nothing ? nothing : Int(max_edges),
                       memory_budget_bytes === nothing ? nothing : Int(memory_budget_bytes))

"""
    ConstructionOptions(; sparsify=:none, collapse=:none, output_stage=:encoding_result,
                         budget=(nothing, nothing, nothing))

Canonical construction controls for data ingestion.
"""
struct ConstructionOptions
    sparsify::Symbol
    collapse::Symbol
    output_stage::Symbol
    budget::ConstructionBudget
end

function ConstructionOptions(; sparsify::Symbol=:none,
                             collapse::Symbol=:none,
                             output_stage::Symbol=:encoding_result,
                             budget=(nothing, nothing, nothing))
    sp = sparsify
    co = collapse
    os = output_stage
    (sp == :none || sp == :radius || sp == :knn || sp == :greedy_perm) ||
        error("ConstructionOptions: sparsify must be :none, :radius, :knn, or :greedy_perm.")
    (co == :none || co == :dominated_edges || co == :acyclic) ||
        error("ConstructionOptions: collapse must be :none, :dominated_edges, or :acyclic.")
    (os == :simplex_tree || os == :graded_complex || os == :cochain || os == :module || os == :fringe || os == :flange || os == :encoding_result) ||
        error("ConstructionOptions: output_stage must be :simplex_tree, :graded_complex, :cochain, :module, :fringe, :flange, or :encoding_result.")
    b = if budget isa ConstructionBudget
        budget
    elseif budget isa Tuple
        length(budget) == 3 || error("ConstructionOptions: budget tuple must be (max_simplices, max_edges, memory_budget_bytes).")
        ConstructionBudget(; max_simplices=budget[1], max_edges=budget[2], memory_budget_bytes=budget[3])
    elseif budget isa NamedTuple
        ConstructionBudget(; budget...)
    else
        error("ConstructionOptions: budget must be ConstructionBudget, 3-tuple, or NamedTuple.")
    end
    return ConstructionOptions(sp, co, os, b)
end

"""
    PipelineOptions(; orientation=nothing, axes_policy=:encoding, axis_kind=nothing,
                     eps=nothing, poset_kind=:signature, field=nothing, max_axis_len=nothing)

Structured pipeline controls that materially affect data-ingestion encodings and
their reproducibility in serialized pipeline artifacts.
"""
struct PipelineOptions{OrientationT,AxisKindT,EpsT,FieldT}
    orientation::OrientationT
    axes_policy::Symbol
    axis_kind::AxisKindT
    eps::EpsT
    poset_kind::Symbol
    field::FieldT
    max_axis_len::Union{Nothing,Int}
end

PipelineOptions(; orientation=nothing,
                axes_policy::Symbol=:encoding,
                axis_kind=nothing,
                eps=nothing,
                poset_kind::Symbol=:signature,
                field=nothing,
                max_axis_len::Union{Nothing,Int}=nothing) =
    PipelineOptions(orientation, axes_policy, axis_kind, eps, poset_kind, field, max_axis_len)

"""
    GridEncodingMap(P, coords; orientation=ntuple(_->1, N))

Axis-aligned grid encoding map for a product-of-chains poset.
"""
struct GridEncodingMap{N,T,P} <: AbstractPLikeEncodingMap
    P::P
    coords::NTuple{N,Vector{T}}
    orientation::NTuple{N,Int}
    sizes::NTuple{N,Int}
    strides::NTuple{N,Int}
end

function _grid_strides(sizes::NTuple{N,Int}) where {N}
    strides = Vector{Int}(undef, N)
    strides[1] = 1
    for i in 2:N
        strides[i] = strides[i-1] * sizes[i-1]
    end
    return ntuple(i -> strides[i], N)
end

function GridEncodingMap(P, coords::NTuple{N,Vector{T}};
                         orientation::NTuple{N,Int}=ntuple(_ -> 1, N)) where {N,T}
    sizes = ntuple(i -> length(coords[i]), N)
    for i in 1:N
        o = orientation[i]
        (o == 1 || o == -1) || error("GridEncodingMap: orientation[$i] must be +1 or -1.")
    end
    return GridEncodingMap{N,T,typeof(P)}(P, coords, orientation, sizes, _grid_strides(sizes))
end


# -----------------------------------------------------------------------------
# Compiled encoding wrappers (primary type for repeated queries)
# -----------------------------------------------------------------------------

"""
    CompiledEncoding(P, pi; axes=nothing, reps=nothing, meta=NamedTuple())

Primary wrapper for encoding maps. Stores the finite encoding poset `P` and
cached encoding metadata used by hot query paths.
"""
struct CompiledEncoding{PiType,PType,AxesType,RepsType,MetaType} <: AbstractPLikeEncodingMap
    P::PType
    pi::PiType
    axes::AxesType
    reps::RepsType
    meta::MetaType
end

function compile_encoding(P, pi; axes=nothing, reps=nothing, meta=NamedTuple())
    axes_val = axes
    reps_val = reps
    if axes_val === nothing && hasmethod(axes_from_encoding, (typeof(pi),))
        axes_val = axes_from_encoding(pi)
    end
    if reps_val === nothing && hasmethod(representatives, (typeof(pi),))
        reps_val = representatives(pi)
    end
    return CompiledEncoding(P, pi, axes_val, reps_val, meta)
end

compile_encoding(::Any, ::Nothing; kwargs...) = nothing

function Base.getproperty(enc::CompiledEncoding, name::Symbol)
    if name === :P || name === :pi || name === :axes || name === :reps || name === :meta
        return getfield(enc, name)
    end
    return getproperty(getfield(enc, :pi), name)
end

function Base.propertynames(enc::CompiledEncoding, private::Bool=false)
    return (fieldnames(typeof(enc))..., propertynames(enc.pi, private)...)
end



# ----- shared API hooks ----------------------------------------------------------

# ----- shared API hooks ----------------------------------------------------------

"""
    locate(pi, x::AbstractVector) -> Int
    locate(pi, x::NTuple{N,<:Real}) -> Int

Generic classifier hook for finite encodings.

Many parts of this codebase build a finite encoding poset `P` for some
parameter space `Q` (for example `Q = R^n` or `Q = Z^n`). The corresponding
classifier `pi : Q -> P` is represented in code by an "encoding map" object.

Required interface:
- Implement `locate(pi, x::AbstractVector)` for your encoding map type.
  The input vector `x` must have length `dimension(pi)`.

Optional (performance) interface:
- You may also implement `locate(pi, x::NTuple{N,<:Real})` for fast,
  allocation-free tuple dispatch. If you do not, a generic fallback method
  is provided which converts the tuple to a vector.

Return value convention:
- return an integer in 1:P.n identifying the region in the target finite poset, or
- return 0 when `x` does not lie in the encoded domain.

This convention matches the existing PL encoders.
"""
function locate end

# Default tuple fallback (for convenience). Backends that care about allocation-free
# tuple dispatch should implement a specialized tuple method.
function locate(pi::AbstractPLikeEncodingMap, x::NTuple{N,<:Real}) where {N}
    return locate(pi, collect(x))
end

function locate(pi::AbstractPLikeEncodingMap, x::NTuple{N,<:Real}; kwargs...) where {N}
    return locate(pi, collect(x); kwargs...)
end

locate(enc::CompiledEncoding, x) = locate(enc.pi, x)
locate(enc::CompiledEncoding, x::NTuple{N,<:Real}) where {N} = locate(enc.pi, x)
locate(enc::CompiledEncoding, x::AbstractVector) = locate(enc.pi, x)
locate(enc::CompiledEncoding, x; kwargs...) = locate(enc.pi, x; kwargs...)
locate(enc::CompiledEncoding, x::NTuple{N,<:Real}; kwargs...) where {N} = locate(enc.pi, x; kwargs...)
locate(enc::CompiledEncoding, x::AbstractVector; kwargs...) = locate(enc.pi, x; kwargs...)

"""
    dimension(pi) -> Int

Ambient dimension of the parameter space for an encoding map.

This should match the expected length of the coordinate vector accepted by
`locate(pi, x::AbstractVector)`.
"""
function dimension end

dimension(enc::CompiledEncoding) = dimension(enc.pi)

"""
    representatives(pi)

Return a finite collection of representative points for the regions of `pi`.

This is used for utilities that need a cheap summary of the encoding's extent in
parameter space (for example to infer a bounding box). A representative point
must be indexable and have length `dimension(pi)`.
"""
function representatives end

function representatives(enc::CompiledEncoding)
    enc.reps === nothing ? representatives(enc.pi) : enc.reps
end

"""
    axes_from_encoding(pi)

Return a tuple of coordinate axes derived from the encoding itself.

Each axis is a sorted vector of coordinates (typically breakpoints / critical
values). Many invariant computations evaluate on an axis-aligned grid; this
function provides a reasonable default grid derived from the encoding.

Backends without a natural grid may omit this method and require callers to pass
explicit `axes`.
"""
function axes_from_encoding end

function axes_from_encoding(enc::CompiledEncoding)
    enc.axes === nothing ? axes_from_encoding(enc.pi) : enc.axes
end


# =============================================================================
# Option structs (public API)

"""
    EncodingOptions(; backend=:auto, max_regions=nothing, strict_eps=nothing,
                    poset_kind=:signature, field=QQField())

Options controlling finite encodings. Used by:
* `ZnEncoding.encode_from_flange(s)` and related helpers (backend=:zn)
* `PLPolyhedra.encode_from_PL_fringe(s)` (backend=:pl)
* `PLBackend.encode_fringe_boxes` (backend=:pl_backend, optional)

Fields:
* `backend`: Symbol, one of `:auto`, `:zn`, `:pl`, `:pl_backend`.
* `max_regions`: Integer cap for region enumeration (backend-dependent defaults).
* `strict_eps`: QQ tolerance used by PL polyhedra backend (default backend constant).
* `poset_kind`: `:signature` (structured, default) or `:dense` (materialized `FinitePoset`).
* `field`: coefficient field used for module data (encoding itself may still use QQ).
"""
struct EncodingOptions
    backend::Symbol
    max_regions::Union{Nothing, Int}
    strict_eps::Any
    poset_kind::Symbol
    field::AbstractCoeffField
end
EncodingOptions(; backend::Symbol=:auto,
                max_regions=nothing,
                strict_eps=nothing,
                poset_kind::Symbol=:signature,
                field::AbstractCoeffField=QQField()) =
    EncodingOptions(backend,
                    max_regions === nothing ? nothing : Int(max_regions),
                    strict_eps,
                    poset_kind,
                    field)

"""
    ResolutionOptions(; maxlen=3, minimal=false, check=true)

Options controlling (co)resolutions.

Fields:
* `maxlen`: length of resolution
* `minimal`: if true, request minimal resolution when available
* `check`: whether to verify minimality conditions
"""
struct ResolutionOptions
    maxlen::Int
    minimal::Bool
    check::Bool
end
ResolutionOptions(; maxlen::Int=3, minimal::Bool=false, check::Bool=true) =
    ResolutionOptions(maxlen, minimal, check)

@inline function validate_pl_mode(mode::Symbol)::Symbol
    if mode === :fast
        return :fast
    elseif mode === :verified
        return :verified
    end
    throw(ArgumentError("pl_mode must be exactly :fast or :verified (got $(mode))"))
end

"""
    InvariantOptions(; axes=nothing, axes_policy=:encoding, max_axis_len=256,
                     box=nothing, threads=nothing, strict=nothing, pl_mode=:fast)

Options controlling invariant computations (Euler surface, MMA decomposition,
signed barcode, distances, etc). `pl_mode` controls PL geometry location policy:
`:fast` (default) or `:verified`.
"""
struct InvariantOptions
    axes::Any
    axes_policy::Symbol
    max_axis_len::Int
    box::Any
    threads::Union{Nothing,Bool}
    strict::Union{Nothing,Bool}
    pl_mode::Symbol
end
InvariantOptions(; axes=nothing,
                 axes_policy::Symbol=:encoding,
                 max_axis_len::Int=256,
                 box=nothing,
                 threads=nothing,
                 strict=nothing,
                 pl_mode::Symbol=:fast) =
    InvariantOptions(axes, axes_policy, max_axis_len, box, threads, strict, validate_pl_mode(pl_mode))

# Preserve positional construction sites while defaulting to :fast mode.
InvariantOptions(axes, axes_policy::Symbol, max_axis_len::Int, box, threads, strict) =
    InvariantOptions(axes, axes_policy, max_axis_len, box, threads, strict, :fast)

"""
    DerivedFunctorOptions(; maxdeg=3, model=:auto, canon=:auto)

Options controlling derived functor computations (Ext, Tor, etc).

Fields
------
- maxdeg: compute degrees 0..maxdeg (inclusive).
- model: selects the computational model. Its meaning depends on the derived functor:

  * Ext(M, N, df):
      :projective  - compute using a projective resolution of M.
      :injective   - compute using an injective resolution of N.
      :unified     - compute a model-independent ExtSpace containing both models and explicit
                     comparison isomorphisms.
      :auto        - alias for :projective.

  * Tor(Rop, L, df):
      :first   - resolve Rop (a P^op-module) and tensor with L.
      :second  - resolve L and tensor with Rop.
      :auto    - alias for :first.

- canon: only used when model == :unified for Ext. Chooses the canonical coordinate basis in the
  unified ExtSpace:
    :projective or :injective (or :auto as alias for :projective).
"""
struct DerivedFunctorOptions
    maxdeg::Int
    model::Symbol
    canon::Symbol
end
DerivedFunctorOptions(; maxdeg::Int=3, model::Symbol=:auto, canon::Symbol=:auto) =
    DerivedFunctorOptions(maxdeg, model, canon)

"""
    FiniteFringeOptions(; check=true, cached=true, store_sparse=false, scalar=1, poset_kind=:regions)

Options for FiniteFringe convenience entrypoints.
"""
struct FiniteFringeOptions
    check::Bool
    cached::Bool
    store_sparse::Bool
    scalar::Any
    poset_kind::Symbol
end
FiniteFringeOptions(; check::Bool=true,
                    cached::Bool=true,
                    store_sparse::Bool=false,
                    scalar=1,
                    poset_kind::Symbol=:regions) =
    FiniteFringeOptions(check, cached, store_sparse, scalar, poset_kind)

"""
    ModuleOptions(; check_sizes=true, cache=nothing)

Options for Modules convenience entrypoints.
"""
struct ModuleOptions
    check_sizes::Bool
    cache::Any
end
ModuleOptions(; check_sizes::Bool=true, cache=nothing) =
    ModuleOptions(check_sizes, cache)

"""
    ResolutionCache()

Thread-safe cache object for repeated resolution-oriented computations.

Stored entries:
- projective resolutions, keyed by `(objectid(module), maxlen)`
- injective resolutions, keyed by `(objectid(module), maxlen)`
- indicator resolution tuples, keyed by `(objectid(HM), objectid(HN), maxlen_or_neg1)`
"""
struct ResolutionKey2
    a::UInt
    maxlen::Int
end

struct ResolutionKey3
    a::UInt
    b::UInt
    maxlen::Int
end

@inline _resolution_key2(a, maxlen::Integer) = ResolutionKey2(UInt(objectid(a)), Int(maxlen))
@inline _resolution_key3(a, b, maxlen::Integer) = ResolutionKey3(UInt(objectid(a)), UInt(objectid(b)), Int(maxlen))

mutable struct ResolutionCache
    lock::Base.ReentrantLock
    projective::Dict{ResolutionKey2,Any}
    injective::Dict{ResolutionKey2,Any}
    indicator::Dict{ResolutionKey3,Any}
    # Thread-sharded memo stores for lock-free fast-path lookups/inserts.
    projective_shards::Vector{Dict{ResolutionKey2,Any}}
    injective_shards::Vector{Dict{ResolutionKey2,Any}}
    indicator_shards::Vector{Dict{ResolutionKey3,Any}}
end

function ResolutionCache()
    nshards = max(1, Base.Threads.maxthreadid())
    return ResolutionCache(
        Base.ReentrantLock(),
        Dict{ResolutionKey2,Any}(),
        Dict{ResolutionKey2,Any}(),
        Dict{ResolutionKey3,Any}(),
        [Dict{ResolutionKey2,Any}() for _ in 1:nshards],
        [Dict{ResolutionKey2,Any}() for _ in 1:nshards],
        [Dict{ResolutionKey3,Any}() for _ in 1:nshards],
    )
end

function _clear_resolution_cache!(cache::ResolutionCache)
    Base.lock(cache.lock)
    empty!(cache.projective)
    empty!(cache.injective)
    empty!(cache.indicator)
    for d in cache.projective_shards
        empty!(d)
    end
    for d in cache.injective_shards
        empty!(d)
    end
    for d in cache.indicator_shards
        empty!(d)
    end
    Base.unlock(cache.lock)
    return nothing
end

"""
    EncodingCache()

Per-encoding cache bucket for geometry/poset-derived artifacts.

Intended contents:
- `posets`: derived encoding posets (e.g. axes/orientation -> poset)
- `cubical`: cubical cell structures keyed by grid size
- `region_posets`: reconstructed region posets keyed by signature identities
"""
mutable struct EncodingCache
    lock::Base.ReentrantLock
    posets::Dict{Any,Any}
    cubical::Dict{Any,Any}
    region_posets::Dict{Tuple{UInt,UInt,Symbol},Any}
    geometry::Dict{Any,Any}
end

EncodingCache() = EncodingCache(Base.ReentrantLock(),
                                Dict{Any,Any}(),
                                Dict{Any,Any}(),
                                Dict{Tuple{UInt,UInt,Symbol},Any}(),
                                Dict{Any,Any}())

function _clear_encoding_cache!(cache::EncodingCache)
    Base.lock(cache.lock)
    try
        empty!(cache.posets)
        empty!(cache.cubical)
        empty!(cache.region_posets)
        empty!(cache.geometry)
    finally
        Base.unlock(cache.lock)
    end
    return nothing
end

"""
    ModuleCache(module_id, field_key)

Module-scoped cache bucket keyed by module identity + field identity.
"""
mutable struct ModuleCache
    module_id::UInt
    field_key::UInt
    resolution::ResolutionCache
    payload::Dict{Symbol,Any}
end

ModuleCache(module_id::UInt, field_key::UInt) =
    ModuleCache(module_id, field_key, ResolutionCache(), Dict{Symbol,Any}())

function _clear_module_cache!(cache::ModuleCache)
    _clear_resolution_cache!(cache.resolution)
    empty!(cache.payload)
    return nothing
end

"""
    SessionCache()

Cross-query cache root with explicit lifetime controlled by the caller.

Hierarchy:
- SessionCache (long-lived, workflow-level)
  - EncodingCache buckets keyed by poset identity
  - ModuleCache buckets keyed by `(objectid(module), _field_cache_key(module.field))`
  - Zn encoding artifacts `(P, pi)` keyed by `(encoding_fingerprint, poset_kind, max_regions)`
  - Zn pushforward plans keyed by `(encoding_fingerprint, flange_fingerprint)`
  - Zn pushed fringes keyed by `(encoding_fingerprint, poset_kind, flange_fingerprint, field_key)`
  - Zn pushed modules keyed by `(encoding_fingerprint, poset_kind, flange_fingerprint, field_key)`
"""
mutable struct SessionCache
    lock::Base.ReentrantLock
    encoding::Dict{UInt,EncodingCache}
    modules::Dict{Tuple{UInt,UInt},ModuleCache}
    resolution::ResolutionCache
    hom_system::Any
    slice_plan::Any
    zn_encoding_artifacts::Dict{Tuple{UInt64,Symbol,Int},Any}
    zn_pushforward_plan::Dict{Tuple{UInt64,UInt64},Any}
    zn_pushforward_fringe::Dict{Tuple{UInt64,Symbol,UInt64,UInt},Any}
    zn_pushforward_module::Dict{Tuple{UInt64,Symbol,UInt64,UInt},Any}
    product_dense::IdDict{Any,Any}
    product_obj::IdDict{Any,Any}
end

SessionCache() = SessionCache(Base.ReentrantLock(),
                              Dict{UInt,EncodingCache}(),
                              Dict{Tuple{UInt,UInt},ModuleCache}(),
                              ResolutionCache(),
                              nothing,
                              nothing,
                              Dict{Tuple{UInt64,Symbol,Int},Any}(),
                              Dict{Tuple{UInt64,UInt64},Any}(),
                              Dict{Tuple{UInt64,Symbol,UInt64,UInt},Any}(),
                              Dict{Tuple{UInt64,Symbol,UInt64,UInt},Any}(),
                              IdDict{Any,Any}(),
                              IdDict{Any,Any}())

@inline _field_cache_key(field)::UInt = UInt(hash((typeof(field), field)))
@inline _poset_cache_key(P)::UInt = UInt(objectid(P))
@inline function _module_cache_key(M)
    fid = hasproperty(M, :field) ? _field_cache_key(getproperty(M, :field)) : UInt(0)
    return (UInt(objectid(M)), fid)
end

function _encoding_cache!(session::SessionCache, key::UInt)
    Base.lock(session.lock)
    try
        return get!(session.encoding, key) do
            EncodingCache()
        end
    finally
        Base.unlock(session.lock)
    end
end

_encoding_cache!(session::SessionCache, P) = _encoding_cache!(session, _poset_cache_key(P))

function _module_cache!(session::SessionCache, key::Tuple{UInt,UInt})
    Base.lock(session.lock)
    try
        return get!(session.modules, key) do
            ModuleCache(key[1], key[2])
        end
    finally
        Base.unlock(session.lock)
    end
end

_module_cache!(session::SessionCache, M) = _module_cache!(session, _module_cache_key(M))

function _invalidate_encoding_cache!(session::SessionCache, key::UInt)
    Base.lock(session.lock)
    try
        cache = pop!(session.encoding, key, nothing)
        cache === nothing || _clear_encoding_cache!(cache)
    finally
        Base.unlock(session.lock)
    end
    return nothing
end

_invalidate_encoding_cache!(session::SessionCache, P) =
    _invalidate_encoding_cache!(session, _poset_cache_key(P))

function _invalidate_module_cache!(session::SessionCache, key::Tuple{UInt,UInt})
    Base.lock(session.lock)
    try
        cache = pop!(session.modules, key, nothing)
        cache === nothing || _clear_module_cache!(cache)
    finally
        Base.unlock(session.lock)
    end
    return nothing
end

_invalidate_module_cache!(session::SessionCache, M) =
    _invalidate_module_cache!(session, _module_cache_key(M))

@inline _session_resolution_cache(session::SessionCache) = session.resolution
@inline _session_resolution_cache(session::SessionCache, M) = _module_cache!(session, M).resolution

@inline _session_hom_cache(session::SessionCache) = session.hom_system
@inline function _set_session_hom_cache!(session::SessionCache, cache)
    session.hom_system = cache
    return cache
end

@inline _session_slice_plan_cache(session::SessionCache) = session.slice_plan
@inline function _set_session_slice_plan_cache!(session::SessionCache, cache)
    session.slice_plan = cache
    return cache
end

@inline function _session_get_zn_pushforward_plan(session::SessionCache,
                                                  encoding_fp::UInt64,
                                                  flange_fp::UInt64)
    key = (encoding_fp, flange_fp)
    Base.lock(session.lock)
    try
        return get(session.zn_pushforward_plan, key, nothing)
    finally
        Base.unlock(session.lock)
    end
end

@inline function _session_get_zn_encoding_artifact(session::SessionCache,
                                                   encoding_fp::UInt64,
                                                   poset_kind::Symbol,
                                                   max_regions::Int)
    key = (encoding_fp, poset_kind, max_regions)
    Base.lock(session.lock)
    try
        return get(session.zn_encoding_artifacts, key, nothing)
    finally
        Base.unlock(session.lock)
    end
end

@inline function _session_set_zn_encoding_artifact!(session::SessionCache,
                                                    encoding_fp::UInt64,
                                                    poset_kind::Symbol,
                                                    max_regions::Int,
                                                    artifact)
    key = (encoding_fp, poset_kind, max_regions)
    Base.lock(session.lock)
    try
        session.zn_encoding_artifacts[key] = artifact
    finally
        Base.unlock(session.lock)
    end
    return artifact
end

@inline function _session_set_zn_pushforward_plan!(session::SessionCache,
                                                   encoding_fp::UInt64,
                                                   flange_fp::UInt64,
                                                   plan)
    key = (encoding_fp, flange_fp)
    Base.lock(session.lock)
    try
        session.zn_pushforward_plan[key] = plan
    finally
        Base.unlock(session.lock)
    end
    return plan
end

@inline function _session_get_zn_pushforward_fringe(session::SessionCache,
                                                    encoding_fp::UInt64,
                                                    poset_kind::Symbol,
                                                    flange_fp::UInt64,
                                                    field_key::UInt)
    key = (encoding_fp, poset_kind, flange_fp, field_key)
    Base.lock(session.lock)
    try
        return get(session.zn_pushforward_fringe, key, nothing)
    finally
        Base.unlock(session.lock)
    end
end

@inline function _session_set_zn_pushforward_fringe!(session::SessionCache,
                                                     encoding_fp::UInt64,
                                                     poset_kind::Symbol,
                                                     flange_fp::UInt64,
                                                     field_key::UInt,
                                                     fringe)
    key = (encoding_fp, poset_kind, flange_fp, field_key)
    Base.lock(session.lock)
    try
        session.zn_pushforward_fringe[key] = fringe
    finally
        Base.unlock(session.lock)
    end
    return fringe
end

@inline function _session_get_zn_pushforward_module(session::SessionCache,
                                                    encoding_fp::UInt64,
                                                    poset_kind::Symbol,
                                                    flange_fp::UInt64,
                                                    field_key::UInt)
    key = (encoding_fp, poset_kind, flange_fp, field_key)
    Base.lock(session.lock)
    try
        return get(session.zn_pushforward_module, key, nothing)
    finally
        Base.unlock(session.lock)
    end
end

@inline function _session_set_zn_pushforward_module!(session::SessionCache,
                                                     encoding_fp::UInt64,
                                                     poset_kind::Symbol,
                                                     flange_fp::UInt64,
                                                     field_key::UInt,
                                                     mod)
    key = (encoding_fp, poset_kind, flange_fp, field_key)
    Base.lock(session.lock)
    try
        session.zn_pushforward_module[key] = mod
    finally
        Base.unlock(session.lock)
    end
    return mod
end

const _WORKFLOW_ENCODING_CACHE_KEY = typemax(UInt)

@inline function _resolve_workflow_session_cache(cache)
    if cache === :auto
        return SessionCache()
    elseif cache === nothing
        return nothing
    elseif cache isa SessionCache
        return cache
    end
    throw(ArgumentError("cache must be :auto, nothing, or SessionCache"))
end

@inline function _resolve_workflow_specialized_cache(cache, ::Type{T}) where {T}
    # Public workflow contract stays simple: only :auto|nothing|SessionCache.
    # Specialized caches are derived from the resolved SessionCache internally.
    return nothing, _resolve_workflow_session_cache(cache)
end

@inline function _workflow_encoding_cache(session_cache::Union{Nothing,SessionCache})
    session_cache === nothing && return nothing
    return _encoding_cache!(session_cache, _WORKFLOW_ENCODING_CACHE_KEY)
end

@inline function _compile_encoding_cached(P, pi, session_cache::Union{Nothing,SessionCache})
    if session_cache === nothing
        # Keep a per-encoding cache even without a SessionCache so repeated
        # geometry/invariant queries on one EncodingResult can reuse artifacts.
        ec = EncodingCache()
        return compile_encoding(P, pi; meta=(encoding_cache=ec,))
    end
    ec = _encoding_cache!(session_cache, P)
    return compile_encoding(P, pi; meta=(encoding_cache=ec))
end

@inline function _encoding_with_session_cache(enc,
                                              session_cache::Union{Nothing,SessionCache})
    session_cache === nothing && return enc
    raw_pi = enc.pi isa CompiledEncoding ? enc.pi.pi : enc.pi
    pi2 = _compile_encoding_cached(enc.P, raw_pi, session_cache)
    return EncodingResult(enc.P, enc.M, pi2;
                          H=enc.H,
                          presentation=enc.presentation,
                          opts=enc.opts,
                          backend=enc.backend,
                          meta=enc.meta)
end

@inline function _resolution_cache_from_session(cache::Union{Nothing,ResolutionCache},
                                                session_cache::Union{Nothing,SessionCache},
                                                M=nothing)
    cache !== nothing && return cache
    session_cache === nothing && return nothing
    return M === nothing ? _session_resolution_cache(session_cache) :
                           _session_resolution_cache(session_cache, M)
end

@inline function _slot_cache_from_session(cache,
                                          session_cache::Union{Nothing,SessionCache},
                                          getter::Function,
                                          setter::Function,
                                          expected::Type,
                                          ctor)
    cache !== nothing && return cache
    session_cache === nothing && return nothing
    slot = getter(session_cache)
    if !(slot isa expected)
        slot = ctor()
        setter(session_cache, slot)
    end
    return slot
end

function _clear_session_cache!(session::SessionCache)
    Base.lock(session.lock)
    try
        for c in values(session.encoding)
            _clear_encoding_cache!(c)
        end
        for c in values(session.modules)
            _clear_module_cache!(c)
        end
        empty!(session.encoding)
        empty!(session.modules)
        _clear_resolution_cache!(session.resolution)
        session.hom_system = nothing
        session.slice_plan = nothing
        empty!(session.zn_encoding_artifacts)
        empty!(session.zn_pushforward_plan)
        empty!(session.zn_pushforward_fringe)
        empty!(session.zn_pushforward_module)
        empty!(session.product_dense)
        empty!(session.product_obj)
    finally
        Base.unlock(session.lock)
    end
    return nothing
end


# -----------------------------------------------------------------------------
# Workflow / pipeline result objects

"""
    EncodingResult(P, M, pi; H=nothing, presentation=nothing,
                  opts=EncodingOptions(), backend=opts.backend, meta=NamedTuple())

Small workflow object representing the output of a user-facing `encode(...)`.

Fields
------
- P : finite encoding poset
- M : PModule on P
- pi: classifier map from the original domain to P
- H : optional FringeModule pushed down to P (kept to avoid recomputation)
- presentation : optional original presentation object (Flange / PLFringe / ...)
- opts : EncodingOptions used for the encoding
- backend : backend actually used (Symbol)
- meta : arbitrary metadata (NamedTuple recommended)
"""
struct EncodingResult{PType,MType,PiType,HType,PresType,MetaType}
    P::PType
    M::MType
    pi::PiType
    H::HType
    presentation::PresType
    opts::EncodingOptions
    backend::Symbol
    meta::MetaType
end

EncodingResult(P, M, pi;
               H=nothing,
               presentation=nothing,
               opts::EncodingOptions=EncodingOptions(),
               backend::Symbol=opts.backend,
               meta=NamedTuple()) =
    EncodingResult(P, M, pi, H, presentation, opts, backend, meta)

"""
    CohomologyDimsResult(P, dims, pi; degree=0, field=QQField(), meta=NamedTuple())

Workflow object for dims-only cohomology output on an encoding poset.

Fields
------
- P : finite encoding poset
- dims : dimensions of `H^degree` at encoding vertices
- pi : classifier map from original domain to P
- degree : cohomological degree
- field : coefficient field used for computation
- meta : arbitrary metadata (NamedTuple recommended)
"""
struct CohomologyDimsResult{PType,DType,PiType,FType,MetaType}
    P::PType
    dims::DType
    pi::PiType
    degree::Int
    field::FType
    meta::MetaType
end

CohomologyDimsResult(P, dims, pi;
                     degree::Int=0,
                     field::AbstractCoeffField=QQField(),
                     meta=NamedTuple()) =
    CohomologyDimsResult(P, dims, pi, degree, field, meta)

# Module materialization hook used by workflow objects.
# Default is identity; subsystem modules can extend this for lazy wrappers.
materialize_module(M) = M
module_dims(M) = M.dims

compile_encoding(enc::EncodingResult; kwargs...) =
    enc.pi isa CompiledEncoding ? enc.pi : compile_encoding(enc.P, enc.pi; kwargs...)

compile_encoding(enc::CohomologyDimsResult; kwargs...) =
    enc.pi isa CompiledEncoding ? enc.pi : compile_encoding(enc.P, enc.pi; kwargs...)

Base.length(::EncodingResult) = 3
Base.IteratorSize(::Type{<:EncodingResult}) = Base.HasLength()

function Base.iterate(enc::EncodingResult, state::Int=1)
    state == 1 && return (enc.P, 2)
    state == 2 && return (enc.M, 3)
    state == 3 && return (enc.pi, 4)
    return nothing
end

Base.length(::CohomologyDimsResult) = 3
Base.IteratorSize(::Type{<:CohomologyDimsResult}) = Base.HasLength()

function Base.iterate(enc::CohomologyDimsResult, state::Int=1)
    state == 1 && return (enc.P, 2)
    state == 2 && return (enc.dims, 3)
    state == 3 && return (enc.pi, 4)
    return nothing
end

@inline function _encoding_with_session_cache(enc::CohomologyDimsResult,
                                              session_cache::Union{Nothing,SessionCache})
    session_cache === nothing && return enc
    raw_pi = enc.pi isa CompiledEncoding ? enc.pi.pi : enc.pi
    pi2 = _compile_encoding_cached(enc.P, raw_pi, session_cache)
    return CohomologyDimsResult(enc.P, enc.dims, pi2;
                                degree=enc.degree,
                                field=enc.field,
                                meta=enc.meta)
end

"""
    change_field(enc, field)

Return an EncodingResult obtained by coercing stored modules into `field`.
"""
function change_field(enc::EncodingResult, field::AbstractCoeffField)
    M2 = change_field(enc.M, field)
    H2 = enc.H === nothing ? nothing : change_field(enc.H, field)
    pres2 = enc.presentation
    if pres2 !== nothing && hasmethod(change_field, (typeof(pres2), AbstractCoeffField))
        pres2 = change_field(pres2, field)
    end
    return EncodingResult(enc.P, M2, enc.pi;
                          H=H2,
                          presentation=pres2,
                          opts=enc.opts,
                          backend=enc.backend,
                          meta=enc.meta)
end

"""
    change_field(enc, field)

Return a CohomologyDimsResult with updated field metadata.
Stored dimension vectors are field-independent.
"""
function change_field(enc::CohomologyDimsResult, field::AbstractCoeffField)
    return CohomologyDimsResult(enc.P, copy(enc.dims), enc.pi;
                                degree=enc.degree,
                                field=field,
                                meta=enc.meta)
end

unwrap(enc::EncodingResult) = (enc.P, enc.M, enc.pi)
unwrap(enc::CohomologyDimsResult) = (enc.P, enc.dims, enc.pi)

"""
    ResolutionResult(res; enc=nothing, betti=nothing, minimality=nothing,
                     opts=ResolutionOptions(), meta=NamedTuple())

Workflow object storing a resolution computation (projective or injective) plus provenance.
"""
struct ResolutionResult{ResType,EncType,BettiType,MinType,MetaType}
    res::ResType
    enc::EncType
    betti::BettiType
    minimality::MinType
    opts::ResolutionOptions
    meta::MetaType
end

ResolutionResult(res;
                 enc=nothing,
                 betti=nothing,
                 minimality=nothing,
                 opts::ResolutionOptions=ResolutionOptions(),
                 meta=NamedTuple()) =
    ResolutionResult(res, enc, betti, minimality, opts, meta)

unwrap(res::ResolutionResult) = res.res

"""
    InvariantResult(enc, which, value; opts=InvariantOptions(), meta=NamedTuple())

Workflow object storing a computed invariant plus provenance.
"""
struct InvariantResult{EncType,WhichType,ValType,MetaType}
    enc::EncType
    which::WhichType
    value::ValType
    opts::InvariantOptions
    meta::MetaType
end

InvariantResult(enc, which, value;
                opts::InvariantOptions=InvariantOptions(),
                meta=NamedTuple()) =
    InvariantResult(enc, which, value, opts, meta)

unwrap(inv::InvariantResult) = inv.value

"""
    change_field(res, field)

Return a ResolutionResult with its stored resolution/encoding coerced into `field`
when possible.
"""
function change_field(res::ResolutionResult, field::AbstractCoeffField)
    enc2 = res.enc === nothing ? nothing : change_field(res.enc, field)
    res2 = res.res
    if res2 !== nothing && hasmethod(change_field, (typeof(res2), AbstractCoeffField))
        res2 = change_field(res2, field)
    end
    return ResolutionResult(res2;
                            enc=enc2,
                            betti=res.betti,
                            minimality=res.minimality,
                            opts=res.opts,
                            meta=res.meta)
end

"""
    change_field(inv, field)

Return an InvariantResult with its stored encoding coerced into `field`
when possible.
"""
function change_field(inv::InvariantResult, field::AbstractCoeffField)
    enc2 = change_field(inv.enc, field)
    return InvariantResult(enc2, inv.which, inv.value; opts=inv.opts, meta=inv.meta)
end


end # module

# =============================================================================
# Stats
#
# Merged from the former src/Stats.jl to reduce file count.
#
# Important: this remains a sibling module `PosetModules.Stats` (not nested under
# CoreModules) so that internal code can continue to write:
#     using ..Stats: _wilson_interval
# without any refactor churn.
# =============================================================================
module Stats

# -----------------------------------------------------------------------------
# Small statistics helpers (no external dependencies)
# -----------------------------------------------------------------------------
#
# We avoid pulling in StatsFuns/Distributions to keep the dependency footprint
# minimal while still supporting confidence intervals and z-scores.

# Approximate inverse CDF for a standard normal distribution.
# Based on Peter John Acklam's rational approximation (public domain).
# Ref: http://home.online.no/~pjacklam/notes/invnorm/
function _normal_quantile(p::Real)
    if p <= 0
        return -Inf
    elseif p >= 1
        return Inf
    end

    # Coefficients in rational approximations.
    a = (-3.969683028665376e+01,
          2.209460984245205e+02,
         -2.759285104469687e+02,
          1.383577518672690e+02,
         -3.066479806614716e+01,
          2.506628277459239e+00)

    b = (-5.447609879822406e+01,
          1.615858368580409e+02,
         -1.556989798598866e+02,
          6.680131188771972e+01,
         -1.328068155288572e+01)

    c = (-7.784894002430293e-03,
         -3.223964580411365e-01,
         -2.400758277161838e+00,
         -2.549732539343734e+00,
          4.374664141464968e+00,
          2.938163982698783e+00)

    d = ( 7.784695709041462e-03,
          3.224671290700398e-01,
          2.445134137142996e+00,
          3.754408661907416e+00)

    # Define break-points.
    plow  = 0.02425
    phigh = 1 - plow

    if p < plow
        # Rational approximation for lower region.
        q = sqrt(-2*log(p))
        return (((((c[1]*q + c[2])*q + c[3])*q + c[4])*q + c[5])*q + c[6]) /
               ((((d[1]*q + d[2])*q + d[3])*q + d[4])*q + 1)
    elseif p > phigh
        # Rational approximation for upper region.
        q = sqrt(-2*log(1 - p))
        return -(((((c[1]*q + c[2])*q + c[3])*q + c[4])*q + c[5])*q + c[6]) /
                ((((d[1]*q + d[2])*q + d[3])*q + d[4])*q + 1)
    else
        # Rational approximation for central region.
        q = p - 0.5
        r = q*q
        return (((((a[1]*r + a[2])*r + a[3])*r + a[4])*r + a[5])*r + a[6])*q /
               (((((b[1]*r + b[2])*r + b[3])*r + b[4])*r + b[5])*r + 1)
    end
end

# Wilson score interval for a binomial proportion.
@inline function _wilson_interval(x::Integer, n::Integer; alpha::Real=0.05)
    if n <= 0
        return (0.0, 1.0)
    end
    if x < 0 || x > n
        throw(ArgumentError("x must satisfy 0 <= x <= n"))
    end

    z = _normal_quantile(1 - float(alpha)/2)
    phat = x / n
    denom = 1 + z^2 / n
    center = (phat + z^2/(2n)) / denom
    half = (z/denom) * sqrt((phat*(1 - phat) + z^2/(4n)) / n)

    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (lo, hi)
end

end # module Stats
