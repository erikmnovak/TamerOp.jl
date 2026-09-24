# =============================================================================
# OrdinaryPersistence.jl
#
# One-parameter persistence of finite filtered chain complexes over F2.
# =============================================================================
module OrdinaryPersistence

using SparseArrays
using ..CoreModules: F2, PrimeField
using ..DataTypes: GradedComplex, ImageNd
import ..DataIngestion
import ..ChainComplexes: describe
import ..FiniteFringe: field
import ..Results: provenance, result_summary

"""
    PersistenceDiagram(finite_by_dim, essential_by_dim; field=F2(), order=:sublevel)

Ordinary persistent homology over `F2()`. Homological dimensions are zero-based.
Finite endpoints and essential births retain their original scalar type.
Essential deaths are stored separately, so exact grades never acquire a
floating-point approximation merely to represent infinity.

For sublevels, a finite pair `(b,d)` represents `[b,d)`; for superlevels, it
represents `(d,b]` in the original parameter units. Empty (zero-length) bars
are omitted. Use `finite_intervals`, `essential_births`, `persistence_intervals`,
`field`, `filtration_order`, and `provenance` to inspect a diagram.
"""
struct PersistenceDiagram{T<:Real,M<:NamedTuple}
    finite_by_dim::Vector{Vector{Tuple{T,T}}}
    essential_by_dim::Vector{Vector{T}}
    field::PrimeField
    order::Symbol
    meta::M
end

function PersistenceDiagram(finite_by_dim::AbstractVector{<:AbstractVector{Tuple{T,T}}},
                            essential_by_dim::AbstractVector{<:AbstractVector{T}};
                            field=F2(), order::Symbol=:sublevel,
                            meta::NamedTuple=NamedTuple()) where {T<:Real}
    isconcretetype(T) || throw(ArgumentError("persistence diagram grades need a concrete real type."))
    diag = PersistenceDiagram([collect(bars) for bars in finite_by_dim],
                              [collect(births) for births in essential_by_dim],
                              _normalize_field(field), _normalize_order(order), meta)
    check_persistence_diagram(diag; throw=true)
    return diag
end

@inline function _normalize_order(order::Symbol)
    order in (:sublevel, :superlevel) ||
        throw(ArgumentError("ordinary persistence order must be :sublevel or :superlevel."))
    return order
end

@inline function _normalize_field(F)
    F isa PrimeField && F == F2() ||
        throw(ArgumentError("ordinary persistence supports field=F2() only; coefficients are reduced modulo 2."))
    return F
end

@inline function _degree_slot(dim::Integer)
    dim isa Bool && throw(ArgumentError("homological dimension must be a nonnegative integer, not Bool."))
    0 <= dim < typemax(Int) || throw(ArgumentError("homological dimension must be a nonnegative integer fitting Int."))
    return Int(dim) + 1
end

"""
    finite_intervals(diag; dim)

Finite nonempty intervals in homological dimension `dim`, preserving exact
endpoint types. A nonnegative dimension above the stored range has no bars.
"""
function finite_intervals(diag::PersistenceDiagram{T}; dim::Integer) where {T}
    slot = _degree_slot(dim)
    return slot <= length(diag.finite_by_dim) ? copy(diag.finite_by_dim[slot]) : Tuple{T,T}[]
end

"""
    essential_births(diag; dim)

Birth values of essential homology classes, preserving their scalar type.
"""
function essential_births(diag::PersistenceDiagram{T}; dim::Integer) where {T}
    slot = _degree_slot(dim)
    return slot <= length(diag.essential_by_dim) ? copy(diag.essential_by_dim[slot]) : T[]
end

"""
    persistence_intervals(diag; dim)

All finite and essential intervals in degree `dim`. Essential deaths are `Inf`
for sublevels and `-Inf` for superlevels. Only this explicit infinity marker
uses Float64; births and finite deaths retain their original scalar type.
For strictly typed exact data, use `finite_intervals` and `essential_births`.
"""
function persistence_intervals(diag::PersistenceDiagram{T}; dim::Integer) where {T}
    bars = Tuple{T,Union{T,Float64}}[]
    append!(bars, finite_intervals(diag; dim=dim))
    infinity = diag.order === :sublevel ? Inf : -Inf
    for birth in essential_births(diag; dim=dim)
        push!(bars, (birth, infinity))
    end
    sort!(bars; by=first, rev=diag.order === :superlevel)
    return bars
end

"""Return the coefficient field actually used for ordinary persistence."""
field(diag::PersistenceDiagram) = diag.field

"""
    filtration_order(diag)

Return `:sublevel` or `:superlevel`, in the original grade coordinates.
"""
filtration_order(diag::PersistenceDiagram) = diag.order

function provenance(diag::PersistenceDiagram{T}) where {T}
    defaults = (construction=(requested=:stored_intervals, effective=:stored_intervals,
                               substitution=:none), source=:not_recorded,
                grade_arithmetic=:stored_values, geometry=:not_recorded,
                backend=:not_recorded, approximation=:not_recorded,
                discretization=:not_recorded)
    recorded = merge(defaults, diag.meta)
    return merge(recorded, (category=:one_parameter_persistence_modules,
        base_poset=diag.order === :sublevel ? :real_line : :opposite_real_line,
        field=diag.field, degree=:by_homological_dimension, degree_convention=:homological,
        orientation=diag.order === :sublevel ? (1,) : (-1,), order=diag.order,
        interval_convention=diag.order === :sublevel ? :left_closed_right_open : :left_open_right_closed,
        zero_length_intervals=:omitted, essential_death=diag.order === :sublevel ? Inf : -Inf,
        grade_type=T, window=:unrestricted))
end

function _diagram_summary(diag::PersistenceDiagram)
    return (kind=:persistence_diagram, field=diag.field, order=diag.order,
        dimensions=Tuple(0:(length(diag.finite_by_dim) - 1)),
        finite_counts=Tuple(length.(diag.finite_by_dim)),
        essential_counts=Tuple(length.(diag.essential_by_dim)),
        provenance=provenance(diag))
end

"""Inspect interval counts and the mathematical provenance of a diagram."""
describe(diag::PersistenceDiagram) = _diagram_summary(diag)

"""
    persistence_diagram_summary(diag)

Cheap owner-local summary of interval counts and provenance.
"""
persistence_diagram_summary(diag::PersistenceDiagram) = _diagram_summary(diag)
result_summary(diag::PersistenceDiagram) = _diagram_summary(diag)

function Base.show(io::IO, diag::PersistenceDiagram)
    d = describe(diag)
    print(io, "PersistenceDiagram(field=", d.field, ", order=", d.order,
          ", finite_counts=", d.finite_counts, ", essential_counts=", d.essential_counts, ")")
end

function Base.show(io::IO, ::MIME"text/plain", diag::PersistenceDiagram)
    show(io, diag)
    print(io, "\n  homological dimensions: ", describe(diag).dimensions,
          "\n  intervals: ", diag.order === :sublevel ? "[birth, death)" : "(death, birth]",
          "\n  essential deaths: ", diag.order === :sublevel ? "+Inf" : "-Inf",
          "\n  grade type: ", provenance(diag).grade_type)
end

"""
    PersistenceValidationSummary

Compact display wrapper for `check_persistence_diagram` and
`check_torus_persistence` reports. Access the structured report as `.report`.
"""
struct PersistenceValidationSummary{R<:NamedTuple}
    report::R
end

"""Wrap an ordinary-persistence validation report for notebook display."""
persistence_validation_summary(report::NamedTuple) = PersistenceValidationSummary(report)

function Base.show(io::IO, summary::PersistenceValidationSummary)
    r = summary.report
    print(io, "PersistenceValidationSummary(valid=", r.valid, ", issues=", length(r.issues), ")")
end

function Base.show(io::IO, ::MIME"text/plain", summary::PersistenceValidationSummary)
    show(io, summary)
    for issue in summary.report.issues
        print(io, "\n  ", issue)
    end
end

describe(summary::PersistenceValidationSummary) = summary.report

"""
    check_persistence_diagram(diag; throw=false)

Check finite endpoints, field, order, degree storage, and strictly positive
interval lengths in the chosen filtration order. This validates the stored
barcode contract; it does not reconstruct a source filtration.
"""
function check_persistence_diagram(diag::PersistenceDiagram; throw::Bool=false)
    issues = String[]
    diag.field == F2() || push!(issues, "ordinary persistence requires F2().")
    diag.order in (:sublevel, :superlevel) || push!(issues, "invalid filtration order.")
    length(diag.finite_by_dim) == length(diag.essential_by_dim) ||
        push!(issues, "finite and essential storage must cover the same homological dimensions.")
    for (slot, bars) in enumerate(diag.finite_by_dim), (b, d) in bars
        isfinite(b) && isfinite(d) || push!(issues, "dimension $(slot - 1): finite endpoints must be finite.")
        ordered = diag.order === :sublevel ? b < d : b > d
        ordered || push!(issues, "dimension $(slot - 1): interval must have positive length in filtration order.")
    end
    for (slot, births) in enumerate(diag.essential_by_dim), b in births
        isfinite(b) || push!(issues, "dimension $(slot - 1): essential birth must be finite.")
    end
    valid = isempty(issues)
    throw && !valid && Base.throw(ArgumentError(join(issues, " ")))
    return (kind=:persistence_diagram_validation, valid=valid, issues=issues)
end

function _cell_dimensions(offsets::Vector{Int})
    dims = Vector{Int}(undef, last(offsets) - 1)
    for slot in 1:(length(offsets) - 1), i in offsets[slot]:(offsets[slot + 1] - 1)
        dims[i] = slot - 1
    end
    return dims
end

# Validate mutable hand-built storage before any unchecked reduction access.
function _validate_complex(G::GradedComplex{N,T}, order::Symbol) where {N,T}
    N == 1 || throw(ArgumentError("ordinary persistence requires a one-parameter GradedComplex; got $N parameters."))
    T <: Real && isconcretetype(T) || throw(ArgumentError("ordinary persistence requires a concrete real grade type."))
    offsets = getfield(G, :dim_offsets)
    isempty(offsets) && throw(ArgumentError("GradedComplex dimension offsets must not be empty."))
    first(offsets) == 1 && issorted(offsets) && last(offsets) == length(G.grades) + 1 ||
        throw(ArgumentError("GradedComplex dimension offsets must start at 1, be nondecreasing and end after the last grade."))
    length(getfield(G, :cell_ids)) == length(G.grades) ||
        throw(ArgumentError("GradedComplex cell and grade counts disagree."))
    all(g -> isfinite(g[1]), G.grades) || throw(ArgumentError("ordinary persistence requires finite cell grades."))
    counts = diff(offsets)
    length(G.boundaries) == max(length(counts) - 1, 0) ||
        throw(ArgumentError("GradedComplex needs one boundary per adjacent pair of chain degrees."))
    for (d, B) in enumerate(G.boundaries)
        size(B) == (counts[d], counts[d + 1]) ||
            throw(ArgumentError("boundary $d has the wrong dimensions."))
        length(B.colptr) == size(B, 2) + 1 && first(B.colptr) == 1 &&
            issorted(B.colptr) && last(B.colptr) == length(B.nzval) + 1 &&
            length(B.rowval) == length(B.nzval) ||
            throw(ArgumentError("boundary $d has invalid sparse column storage."))
        all(r -> 1 <= r <= size(B, 1), B.rowval) ||
            throw(ArgumentError("boundary $d has a row index outside its chain degree."))
        for col in 1:size(B, 2)
            previous = 0
            for ptr in nzrange(B, col)
                row = B.rowval[ptr]
                row > previous || throw(ArgumentError("boundary $d sparse rows must be strictly increasing in each column."))
                previous = row
                isodd(B.nzval[ptr]) || continue
                face = G.grades[offsets[d] + row - 1][1]
                cell = G.grades[offsets[d + 1] + col - 1][1]
                (order === :sublevel ? face <= cell : face >= cell) ||
                    throw(ArgumentError("boundary $d is not compatible with the $order filtration."))
            end
        end
    end
    # Compute each double boundary modulo two directly, avoiding integer
    # overflow and sparse multiplication's coefficient/type conventions.
    for d in 2:length(G.boundaries)
        A, B = G.boundaries[d - 1], G.boundaries[d]
        parity = falses(size(A, 1))
        touched = Int[]
        for col in 1:size(B, 2)
            empty!(touched)
            for p in nzrange(B, col)
                isodd(B.nzval[p]) || continue
                for q in nzrange(A, B.rowval[p])
                    isodd(A.nzval[q]) || continue
                    row = A.rowval[q]
                    parity[row] = !parity[row]
                    push!(touched, row)
                end
            end
            any(row -> parity[row], touched) &&
                throw(ArgumentError("boundary squared is nonzero over F2 in chain degree $d."))
        end
    end
    return offsets
end

function _canonical_column!(col::Vector{Int}, rank::Vector{Int})
    isempty(col) && return col
    sort!(col; by = i -> rank[i])
    out = 1
    i = 1
    @inbounds while i <= length(col)
        c = col[i]
        j = i + 1
        while j <= length(col) && col[j] == c
            j += 1
        end
        isodd(j - i) && (col[out] = c; out += 1)
        i = j
    end
    resize!(col, out - 1)
    return col
end

function _xor_columns(a::Vector{Int}, b::Vector{Int}, rank::Vector{Int})
    out = Vector{Int}()
    sizehint!(out, max(length(a), length(b)))
    i = 1
    j = 1
    @inbounds while i <= length(a) && j <= length(b)
        ai = a[i]
        bj = b[j]
        ra = rank[ai]
        rb = rank[bj]
        if ra == rb
            i += 1
            j += 1
        elseif ra < rb
            push!(out, ai)
            i += 1
        else
            push!(out, bj)
            j += 1
        end
    end
    @inbounds while i <= length(a)
        push!(out, a[i])
        i += 1
    end
    @inbounds while j <= length(b)
        push!(out, b[j])
        j += 1
    end
    return out
end

function _boundary_column_f2(G::GradedComplex,
                             dims::Vector{Int},
                             offsets::Vector{Int},
                             rank::Vector{Int},
                             cell::Int)
    d = dims[cell]
    d == 0 && return Int[]
    B = G.boundaries[d]
    local_col = cell - offsets[d + 1] + 1
    prev_offset = offsets[d]
    lo = B.colptr[local_col]
    hi = B.colptr[local_col + 1] - 1
    col = Int[]
    sizehint!(col, max(0, hi - lo + 1))
    @inbounds for ptr in lo:hi
        isodd(B.nzval[ptr]) || continue
        row_global = prev_offset + B.rowval[ptr] - 1
        rank[row_global] < rank[cell] ||
            throw(ArgumentError("cell boundary is not filtration-compatible: boundary cell $row_global appears after coface $cell."))
        push!(col, row_global)
    end
    return _canonical_column!(col, rank)
end

@inline _next_coord(i::Int, n::Int, periodic::Bool) =
    (periodic && i == n) ? 1 : i + 1

@inline function _candidate_prev(curr::Int, n::Int, periodic::Bool)
    curr > 1 && return curr - 1
    return periodic ? n : 0
end

function _face_axis_candidates(curr::Int, n::Int, periodic::Bool)
    out = Int[]
    curr <= n && push!(out, curr)
    p = _candidate_prev(curr, n, periodic)
    p != 0 && p != curr && push!(out, p)
    return out
end

@inline _aggregate_incident(vals::AbstractVector, order::Symbol) =
    order === :sublevel ? minimum(vals) : maximum(vals)

function _top_cell_grades_2d(vals::AbstractMatrix{T},
                             periodic::NTuple{2,Bool},
                             order::Symbol) where {T<:Real}
    nx, ny = size(vals)
    px, py = periodic
    nvx = px ? nx : nx + 1
    nvy = py ? ny : ny + 1
    neh = nx * nvy
    nev = nvx * ny
    nf = nx * ny
    grades = Vector{NTuple{1,T}}(undef, nvx * nvy + neh + nev + nf)
    t = 1

    @inbounds for j in 1:nvy
        ys = _face_axis_candidates(j, ny, py)
        for i in 1:nvx
            xs = _face_axis_candidates(i, nx, px)
            inc = T[]
            sizehint!(inc, length(xs) * length(ys))
            for y in ys, x in xs
                push!(inc, vals[x, y])
            end
            grades[t] = (_aggregate_incident(inc, order),)
            t += 1
        end
    end
    @inbounds for j in 1:nvy
        ys = _face_axis_candidates(j, ny, py)
        for i in 1:nx
            inc = T[vals[i, y] for y in ys]
            grades[t] = (_aggregate_incident(inc, order),)
            t += 1
        end
    end
    @inbounds for j in 1:ny
        for i in 1:nvx
            xs = _face_axis_candidates(i, nx, px)
            inc = T[vals[x, j] for x in xs]
            grades[t] = (_aggregate_incident(inc, order),)
            t += 1
        end
    end
    @inbounds for j in 1:ny
        for i in 1:nx
            grades[t] = (vals[i, j],)
            t += 1
        end
    end
    return grades
end

function _top_cell_complex_2d(vals::AbstractMatrix{T},
                              periodic::NTuple{2,Bool},
                              order::Symbol) where {T<:Real}
    nx, ny = size(vals)
    nx > 0 && ny > 0 || throw(ArgumentError("cubical top-dimensional cells must be nonempty."))
    px, py = periodic
    nvx = px ? nx : nx + 1
    nvy = py ? ny : ny + 1
    nv = nvx * nvy
    neh = nx * nvy
    nev = nvx * ny
    ne = neh + nev
    nf = nx * ny

    @inline vid(i::Int, j::Int) = i + (j - 1) * nvx
    @inline hid(i::Int, j::Int) = i + (j - 1) * nx
    @inline vidx(i::Int, j::Int) = neh + i + (j - 1) * nvx
    @inline fid(i::Int, j::Int) = i + (j - 1) * nx

    I1 = Vector{Int}(undef, 2ne)
    J1 = Vector{Int}(undef, 2ne)
    V1 = Vector{Int}(undef, 2ne)
    t = 1
    @inbounds for j in 1:nvy
        for i in 1:nx
            col = hid(i, j)
            i2 = _next_coord(i, nx, px)
            I1[t] = vid(i, j); J1[t] = col; V1[t] = 1; t += 1
            I1[t] = vid(i2, j); J1[t] = col; V1[t] = -1; t += 1
        end
    end
    @inbounds for j in 1:ny
        for i in 1:nvx
            col = vidx(i, j)
            j2 = _next_coord(j, ny, py)
            I1[t] = vid(i, j); J1[t] = col; V1[t] = 1; t += 1
            I1[t] = vid(i, j2); J1[t] = col; V1[t] = -1; t += 1
        end
    end
    b1 = dropzeros!(sparse(I1, J1, V1, nv, ne))

    I2 = Vector{Int}(undef, 4nf)
    J2 = Vector{Int}(undef, 4nf)
    V2 = Vector{Int}(undef, 4nf)
    t = 1
    @inbounds for j in 1:ny
        for i in 1:nx
            col = fid(i, j)
            i2 = _next_coord(i, nx, px)
            j2 = _next_coord(j, ny, py)
            I2[t] = vidx(i, j); J2[t] = col; V2[t] = 1; t += 1
            I2[t] = vidx(i2, j); J2[t] = col; V2[t] = -1; t += 1
            I2[t] = hid(i, j); J2[t] = col; V2[t] = -1; t += 1
            I2[t] = hid(i, j2); J2[t] = col; V2[t] = 1; t += 1
        end
    end
    b2 = dropzeros!(sparse(I2, J2, V2, ne, nf))

    cells = [collect(1:nv), collect(1:ne), collect(1:nf)]
    grades = _top_cell_grades_2d(vals, periodic, order)
    return GradedComplex(cells, SparseMatrixCSC{Int,Int}[b1, b2], grades)
end

"""
    persistence_diagram(G::GradedComplex; order=:sublevel, field=F2())

Reduce a finite, one-parameter graded chain complex over `F2()`. Integer
boundary coefficients are read modulo two. The input must have finite real
grades, correctly shaped boundaries, zero double boundary modulo two, and
boundary maps compatible with the selected order. These conditions are checked
before reduction.

Sublevels use increasing grades. Superlevels use decreasing grades without
negating or converting the grade values. Births are included and deaths
excluded in filtration order; zero-length intervals are omitted.
"""
function persistence_diagram(G::GradedComplex{N,T};
                             order::Symbol=:sublevel, field=F2()) where {N,T}
    _normalize_order(order)
    _normalize_field(field)
    offsets = _validate_complex(G, order)
    dims = _cell_dimensions(offsets)
    total = length(G.grades)
    values = T[g[1] for g in G.grades]
    perm = collect(1:total)
    # Grade comparisons avoid negating unsigned integers and typemin(Int).
    # At equal grade, faces precede cofaces; cell index settles remaining ties.
    sort!(perm; lt=(a, b) -> values[a] == values[b] ?
        (dims[a] == dims[b] ? a < b : dims[a] < dims[b]) :
        (order === :sublevel ? values[a] < values[b] : values[a] > values[b]))
    rank = Vector{Int}(undef, total)
    for (r, cell) in enumerate(perm)
        rank[cell] = r
    end

    nd = length(offsets) - 1
    intervals = [Tuple{T,T}[] for _ in 1:nd]
    essential = [T[] for _ in 1:nd]
    reduced = Vector{Vector{Int}}(undef, total)
    has_reduced = falses(total)
    positive = falses(total)
    alive = falses(total)
    for cell in perm
        col = _boundary_column_f2(G, dims, offsets, rank, cell)
        while !isempty(col)
            pivot = col[end]
            has_reduced[pivot] || break
            col = _xor_columns(col, reduced[pivot], rank)
        end
        if isempty(col)
            positive[cell] = true
            alive[cell] = true
        else
            pivot = col[end]
            has_reduced[pivot] = true
            reduced[pivot] = col
            positive[pivot] || throw(ArgumentError("ordinary persistence internal inconsistency: pivot did not birth a class."))
            alive[pivot] = false
            values[pivot] == values[cell] ||
                push!(intervals[dims[pivot] + 1], (values[pivot], values[cell]))
        end
    end
    for cell in 1:total
        alive[cell] && push!(essential[dims[cell] + 1], values[cell])
    end
    for bars in intervals
        sort!(bars; rev=order === :superlevel)
    end
    for births in essential
        sort!(births; rev=order === :superlevel)
    end
    # The reducer established these invariants; avoid revalidating its output.
    meta = (construction=(requested=:graded_chain_complex, effective=:graded_chain_complex,
                          substitution=:none), source=:graded_complex,
            grade_arithmetic=:exact_stored_values, geometry=:not_recorded,
            chain_validation=:boundary_squared_zero_mod_two,
            backend=:f2_column_reduction, approximation=:none_in_reduction,
            discretization=:none)
    return PersistenceDiagram(intervals, essential, field, order, meta)
end

"""
    persistence_diagram(data, filtration; order=:sublevel, field=F2(), cache=nothing)

Build a one-parameter complex with `build_graded_complex` and compute ordinary
persistent homology. The filtration must actually be compatible with `order`;
changing this keyword does not turn a general lower-star construction into an
upper-star construction. Cubical vertex data have a dedicated upper-star route.
Grade precision on other ingestion routes is that of their constructed complex.
"""
function persistence_diagram(data, filtration::DataIngestion.AbstractFiltration;
                             order::Symbol=:sublevel, field=F2(), cache=nothing)
    _normalize_order(order)
    _normalize_field(field)
    build = DataIngestion.build_graded_complex(data, filtration; cache=cache)
    diag = persistence_diagram(DataIngestion.graded_complex(build); order=order, field=field)
    kind = DataIngestion.filtration_kind(filtration)
    # The public build result records grades and orientation, but not the
    # executed construction/backend. Do not infer an identity construction:
    # some supported requests deliberately substitute a different model.
    meta = merge(diag.meta, (construction=(requested=kind, effective=:not_recorded, substitution=:not_recorded),
                            source=typeof(data), geometry=:ingestion_contract))
    return PersistenceDiagram(diag.finite_by_dim, diag.essential_by_dim, diag.field, diag.order, meta)
end

function _periodic_tuple(periodic, ::Val{N}) where {N}
    periodic isa Bool && return ntuple(_ -> periodic, N)
    (periodic isa Tuple || periodic isa AbstractVector) ||
        throw(ArgumentError("periodic must be a Bool or tuple/vector of Bool."))
    length(periodic) == N && all(p -> p isa Bool, periodic) ||
        throw(ArgumentError("periodic must have $N Bool entries."))
    return ntuple(i -> periodic[i]::Bool, N)
end

function _validate_cubical_values(values::AbstractArray{T,N}) where {T<:Real,N}
    N >= 1 || throw(ArgumentError("cubical persistence requires at least one array dimension."))
    isconcretetype(T) || throw(ArgumentError("cubical persistence requires a concrete real element type."))
    isempty(values) && throw(ArgumentError("cubical persistence requires a nonempty array."))
    Base.require_one_based_indexing(values)
    all(isfinite, values) || throw(ArgumentError("cubical persistence requires finite values."))
    return nothing
end

function _vertex_cubical_complex(values::AbstractArray{T,N}, periodic::NTuple{N,Bool},
                                 order::Symbol, construction, cache) where {T<:Real,N}
    # Cubical topology is owned by DataIngestion. Its image-grade storage uses
    # Float64, so send order ranks through that public builder and restore the
    # original exact values afterward. Maximal ranks implement lower stars;
    # decreasing rank order implements upper stars without numeric negation.
    levels = sort!(unique(vec(values)); rev=order === :superlevel)
    length(levels) <= 2^53 || throw(ArgumentError("too many distinct vertex levels for cubical rank construction."))
    ranks = Dict(value => i for (i, value) in enumerate(levels))
    rank_values = map(v -> ranks[v], values)
    filtration = DataIngestion.CubicalFiltration(; periodic=periodic, construction=construction)
    build = DataIngestion.build_graded_complex(ImageNd(Array(rank_values)), filtration; cache=cache)
    ranked = DataIngestion.graded_complex(build)
    grades = NTuple{1,T}[(levels[Int(g[1])],) for g in ranked.grades]
    return GradedComplex(ranked.cells_by_dim, ranked.boundaries, grades)
end

"""
    cubical_persistence(values; periodic=false, order=:sublevel,
                        input=:top_cells, field=F2())

Ordinary cubical persistence. `input=:top_cells` supports two-dimensional arrays;
entries grade squares, and their faces receive the minimum incident value for
sublevels or maximum for superlevels. `input=:vertices` supports arbitrary
positive array dimension; cells receive the maximum vertex value for sublevels
or minimum for superlevels. Both routes preserve exact input grade values.

`periodic` is a Bool or one Bool per array axis. One periodic cell along an axis
has coincident endpoints with cancelling boundary coefficients, as required for
a circle. Two periodic axes give a torus, including a 1-by-1 torus.

Finite bars are `[birth,death)` for sublevels and `(death,birth]` for superlevels.
The combined interval accessor reports essential death at `Inf` or `-Inf`,
respectively. Zero-length intervals are omitted.
"""
function cubical_persistence(values::AbstractArray{<:Real}; periodic=false,
                             order::Symbol=:sublevel, input::Symbol=:top_cells,
                             field=F2())
    _normalize_order(order)
    _normalize_field(field)
    _validate_cubical_values(values)
    per = _periodic_tuple(periodic, Val(ndims(values)))
    if input === :vertices
        return _vertex_cubical_persistence(values, per, order, field,
            DataIngestion.construction_mode(DataIngestion.CubicalFiltration()), nothing)
    end
    input === :top_cells || throw(ArgumentError("cubical persistence input must be :top_cells or :vertices."))
    ndims(values) == 2 || throw(ArgumentError("cubical persistence input=:top_cells supports two-dimensional arrays."))
    G = _top_cell_complex_2d(values, per, order)
    diag = persistence_diagram(G; order=order, field=field)
    meta = merge(diag.meta, (construction=(requested=:cubical_top_cells, effective=:cubical_top_cells,
        substitution=:none), source=(shape=size(values), periodic=per),
        grade_arithmetic=:exact_input_values, geometry=:cubical_cells))
    return PersistenceDiagram(diag.finite_by_dim, diag.essential_by_dim, diag.field, diag.order, meta)
end

function _vertex_cubical_persistence(values, per, order, field, construction, cache)
    G = _vertex_cubical_complex(values, per, order, construction, cache)
    diag = persistence_diagram(G; order=order, field=field)
    meta = merge(diag.meta, (construction=(requested=:cubical_vertices, effective=:cubical_vertices,
        substitution=:none), source=(shape=size(values), periodic=per),
        grade_arithmetic=:exact_input_values, geometry=:cubical_cells))
    return PersistenceDiagram(diag.finite_by_dim, diag.essential_by_dim, diag.field, diag.order, meta)
end

function persistence_diagram(data::ImageNd, filtration::DataIngestion.CubicalFiltration;
                             order::Symbol=:sublevel, field=F2(), cache=nothing)
    _normalize_order(order)
    _normalize_field(field)
    params = DataIngestion.filtration_parameters(filtration)
    channels = get(params, :channels, nothing)
    values = if channels === nothing
        data.data
    else
        length(channels) == 1 || throw(ArgumentError("ordinary cubical persistence requires one channel."))
        only(channels)
    end
    values isa AbstractArray{<:Real} || throw(ArgumentError("cubical channel must be a real-valued array."))
    size(values) == size(data.data) || throw(ArgumentError("cubical channel shape must match the image."))
    _validate_cubical_values(values)
    per = _periodic_tuple(params.periodic, Val(ndims(values)))
    return _vertex_cubical_persistence(values, per, order, field,
                                      DataIngestion.construction_mode(filtration), cache)
end

"""
    check_torus_persistence(diag; check_h1=true, check_h2=true, throw=false)

Check essential homology counts `(1,2,1)` for a two-dimensional torus, in
addition to the diagram storage contract. Passing establishes these barcode
counts only, not that the unknown source is homeomorphic to a torus.
"""
function check_torus_persistence(diag::PersistenceDiagram;
                                 check_h1::Bool=true, check_h2::Bool=true,
                                 throw::Bool=false)
    issues = copy(check_persistence_diagram(diag).issues)
    counts = ntuple(d -> length(essential_births(diag; dim=d - 1)), 3)
    counts[1] == 1 || push!(issues, "expected one essential H0 class on T^2; got $(counts[1]).")
    !check_h1 || counts[2] == 2 || push!(issues, "expected two essential H1 classes on T^2; got $(counts[2]).")
    !check_h2 || counts[3] == 1 || push!(issues, "expected one essential H2 class on T^2; got $(counts[3]).")
    valid = isempty(issues)
    throw && !valid && Base.throw(ArgumentError(join(issues, " ")))
    return (kind=:torus_persistence_validation, valid=valid,
            essential_counts=(h0=counts[1], h1=counts[2], h2=counts[3]), issues=issues)
end

end # module OrdinaryPersistence
