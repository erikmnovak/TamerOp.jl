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

# Retained F2 chains use dimension-local indices, not potentially repeated cell
# labels. This storage is produced only by the reduction below; provenance text
# alone never enables representative selection.
struct _PersistenceChain{T}
    indices::Vector{Int}
    ids::Vector{Int}
    grades::Vector{T}
end

struct _PersistenceRepresentative{T}
    birth::T
    death::Union{Nothing,T}
    cycle::_PersistenceChain{T}
    bounding_chain::Union{Nothing,_PersistenceChain{T}}
end

struct _PersistenceRepresentatives{T}
    finite::Vector{Vector{_PersistenceRepresentative{T}}}
    essential::Vector{Vector{_PersistenceRepresentative{T}}}
end

"""
    PersistenceDiagram(finite_by_dim, essential_by_dim; field=F2(), order=:sublevel)

Ordinary persistent homology over `F2()`. Homological dimensions are zero-based.
Finite endpoints and essential births retain their original scalar type.
Essential deaths are stored separately, so exact grades never acquire a
floating-point approximation merely to represent infinity.

For sublevels, a finite pair `(b,d)` represents `[b,d)`; for superlevels, it
represents `(d,b]` in the original parameter units. Empty (zero-length) bars
are omitted. Use `finite_intervals`, `essential_births`, `persistence_intervals`,
`field`, `filtration_order`, and `provenance` to inspect a diagram. Computing with
`representatives=true` also retains reduction cycles inspected by
`persistence_representative`; hand-built endpoint diagrams have none.
"""
struct PersistenceDiagram{T<:Real,M<:NamedTuple}
    finite_by_dim::Vector{Vector{Tuple{T,T}}}
    essential_by_dim::Vector{Vector{T}}
    field::PrimeField
    order::Symbol
    meta::M
    retained_representatives::Union{Nothing,_PersistenceRepresentatives{T}}
end

PersistenceDiagram(finite::Vector{Vector{Tuple{T,T}}}, essential::Vector{Vector{T}},
                   field::PrimeField, order::Symbol, meta::NamedTuple) where {T<:Real} =
    PersistenceDiagram(finite, essential, field, order, meta, nothing)

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

function _representative_chain(G::GradedComplex{N,T}, offsets, dim, indices) where {N,T}
    ordered = sort(indices)
    return _PersistenceChain{T}(ordered .- (offsets[dim + 1] - 1),
        getfield(G, :cell_ids)[ordered], T[G.grades[i][1] for i in ordered])
end

function _representative_chain_record(chain::_PersistenceChain, dim::Int)
    return (; dimension=dim, cell_indices=Tuple(chain.indices), cell_ids=Tuple(chain.ids),
        cell_grades=Tuple(chain.grades), coefficients=Tuple(fill(1, length(chain.indices))))
end

"""
    persistence_representative(diag; dim, kind=:finite, index=1)

Inspect one original interval member's retained `F2` cycle. Opt in when computing
the diagram with `representatives=true`; the default interval-only computation
does not retain change-of-basis columns. `kind` is `:finite` or `:essential`, and
`index` indexes `finite_intervals(diag; dim)` or `essential_births(diag; dim)`.
Equal intervals have separate member indices and potentially different cycles.

The result reports `available` and `reason`. An unavailable result has no cycle;
setting a provenance flag on a hand-built diagram does not create one. Invalid
dimensions, kinds and indices are errors even when cycles were not retained.

An available `cycle` records dimension-local cell indices, original cell IDs,
exact cell grades, and coefficients modulo two. Its class is nonzero throughout
the returned interval, including birth and excluding finite death in filtration
order. A finite interval also has a `bounding_chain` whose boundary is this cycle
at death. Essential intervals have no such chain in the supplied finite complex.
These are deterministic reduction choices, not canonical, optimized, or geometric
representatives. Source-cell IDs do not assert an embedding in the original data.
The input diagram and its retained storage must be treated as read-only.
"""
function persistence_representative(diag::PersistenceDiagram; dim::Integer,
                                   kind::Symbol=:finite, index::Integer=1)
    slot = _degree_slot(dim)
    kind in (:finite, :essential) || throw(ArgumentError("kind must be :finite or :essential."))
    !(index isa Bool) && 1 <= index <= typemax(Int) ||
        throw(ArgumentError("index must be a positive interval member index fitting Int."))
    values = kind === :finite ? diag.finite_by_dim : diag.essential_by_dim
    slot <= length(values) && index <= length(values[slot]) ||
        throw(ArgumentError("index is outside the stored $kind intervals in dimension $dim."))
    birth = kind === :finite ? values[slot][index][1] : values[slot][index]
    death = kind === :finite ? values[slot][index][2] :
        (diag.order === :sublevel ? Inf : -Inf)
    retained = diag.retained_representatives
    context = (; dimension=Int(dim), kind, index=Int(index), interval=(birth, death),
        field=diag.field, order=diag.order, birth_included=true, death_included=false,
        valid_parameters=diag.order === :sublevel ? "birth <= t < death" : "death < t <= birth",
        choice=retained === nothing ? :not_available : :noncanonical_f2_column_reduction,
        source_geometry=:not_asserted)
    retained === nothing && return merge(context, (; available=false, reason=:not_retained,
        cycle=nothing, bounding_chain=nothing))
    records = kind === :finite ? retained.finite : retained.essential
    slot <= length(records) && index <= length(records[slot]) ||
        throw(ArgumentError("retained representatives disagree with the diagram's interval storage."))
    record = records[slot][index]
    record.birth == birth && (kind === :essential ? record.death === nothing : record.death == death) ||
        throw(ArgumentError("retained representative endpoints disagree with the diagram; recompute after mutation."))
    chain = record.bounding_chain
    return merge(context, (; available=true, reason=:retained,
        cycle=_representative_chain_record(record.cycle, Int(dim)),
        bounding_chain=chain === nothing ? nothing : _representative_chain_record(chain, Int(dim) + 1)))
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
        grade_type=T, window=:unrestricted,
        representatives=diag.retained_representatives === nothing ? :not_retained : :retained_reduction_cycles))
end

function _diagram_summary(diag::PersistenceDiagram)
    return (kind=:persistence_diagram, field=diag.field, order=diag.order,
        dimensions=Tuple(0:(length(diag.finite_by_dim) - 1)),
        finite_counts=Tuple(length.(diag.finite_by_dim)),
        essential_counts=Tuple(length.(diag.essential_by_dim)),
        representatives_available=diag.retained_representatives !== nothing,
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
    retained = diag.retained_representatives
    if retained !== nothing
        for (kind, intervals, records) in ((:finite, diag.finite_by_dim, retained.finite),
                                           (:essential, diag.essential_by_dim, retained.essential))
            if length(intervals) != length(records)
                push!(issues, "retained $kind representatives cover different homological dimensions.")
                continue
            end
            for slot in eachindex(intervals)
                if length(intervals[slot]) != length(records[slot])
                    push!(issues, "dimension $(slot - 1): retained $kind representative count disagrees with intervals.")
                    continue
                end
                for i in eachindex(records[slot])
                    r = records[slot][i]
                    expected = kind === :finite ? intervals[slot][i] : (intervals[slot][i], nothing)
                    (r.birth, r.death) == expected ||
                        push!(issues, "dimension $(slot - 1): retained $kind representative endpoints disagree with interval $i.")
                    (r.bounding_chain !== nothing) == (kind === :finite) ||
                        push!(issues, "dimension $(slot - 1): retained $kind representative has an invalid bounding-chain contract.")
                    for chain in (r.cycle, r.bounding_chain)
                        chain === nothing && continue
                        length(chain.indices) == length(chain.ids) == length(chain.grades) ||
                            push!(issues, "retained chain indices, cell IDs and grades have different lengths.")
                        !isempty(chain.indices) && issorted(chain.indices) && allunique(chain.indices) && all(>(0), chain.indices) ||
                            push!(issues, "retained chains require nonempty, distinct positive cell indices in increasing order.")
                        level = chain === r.cycle ? r.birth : r.death
                        level === nothing && continue
                        all(g -> isfinite(g) && (diag.order === :sublevel ? g <= level : g >= level), chain.grades) ||
                            push!(issues, "retained chain has a cell outside its birth/death filtration stage.")
                    end
                end
            end
        end
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

# Exact double-boundary checks share no reduction state with the barcode.
function _double_boundary_zero_sparse(A, B)
    parity = zeros(Bool, size(A, 1))
    for col in axes(B, 2)
        nonzero = 0
        @inbounds for p in nzrange(B, col)
            isodd(B.nzval[p]) || continue
            for q in nzrange(A, B.rowval[p])
                isodd(A.nzval[q]) || continue
                row = A.rowval[q]
                old = parity[row]
                nonzero += old ? -1 : 1
                parity[row] = !old
            end
        end
        nonzero == 0 || return false
        # A zero result leaves the accumulator empty for the next column.
    end
    return true
end

function _double_boundary_zero_packed(A, B)
    nwords = cld(size(A, 1), 64)
    packed = zeros(UInt64, nwords, size(A, 2))
    @inbounds for col in axes(A, 2), p in nzrange(A, col)
        isodd(A.nzval[p]) || continue
        row = A.rowval[p] - 1
        packed[(row >>> 6) + 1, col] ⊻= UInt64(1) << (row & 63)
    end
    accumulator = zeros(UInt64, nwords)
    for col in axes(B, 2)
        @inbounds for p in nzrange(B, col)
            isodd(B.nzval[p]) || continue
            source = B.rowval[p]
            for word in 1:nwords
                accumulator[word] ⊻= packed[word, source]
            end
        end
        all(iszero, accumulator) || return false
    end
    return true
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
        # Measured sparse/dense controls favor packing once columns contain
        # at least three entries per word and B reuses A columns four times
        # on average. Packing single-use columns loses to direct parity.
        # Packed storage stays below one third of sparse index storage.
        words = cld(size(A, 1), 64)
        packed = words > 0 && nnz(A) ÷ max(1, size(A, 2)) >= 3 * words &&
                 nnz(B) >= 4 * size(A, 2)
        valid = packed ? _double_boundary_zero_packed(A, B) : _double_boundary_zero_sparse(A, B)
        valid || throw(ArgumentError("boundary squared is nonzero over F2 in chain degree $d."))
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

# The destination is scratch storage, distinct from both sorted source columns.
function _xor_columns!(out::Vector{Int}, a::Vector{Int}, b::Vector{Int}, rank::Vector{Int})
    resize!(out, length(a) + length(b))
    k = 1
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
            out[k] = ai; k += 1
            i += 1
        else
            out[k] = bj; k += 1
            j += 1
        end
    end
    @inbounds while i <= length(a)
        out[k] = a[i]; k += 1
        i += 1
    end
    @inbounds while j <= length(b)
        out[k] = b[j]; k += 1
        j += 1
    end
    resize!(out, k - 1)
    return out
end

function _boundary_column_f2!(col::Vector{Int}, G::GradedComplex,
                             dims::Vector{Int},
                             offsets::Vector{Int},
                             rank::Vector{Int},
                             cell::Int)
    empty!(col)
    d = dims[cell]
    d == 0 && return col
    B = G.boundaries[d]
    local_col = cell - offsets[d + 1] + 1
    prev_offset = offsets[d]
    lo = B.colptr[local_col]
    hi = B.colptr[local_col + 1] - 1
    sizehint!(col, max(0, hi - lo + 1); shrink=false)
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

# One active binary column; each higher level marks nonempty words below it.
# Saved pivot columns use sparse indices or packed words. Finding the largest
# entry skips empty ranges without rescanning or copying the active destination
# on each addition.
struct _ParityColumn
    levels::Vector{Vector{UInt64}}
end

function _ParityColumn(n::Int)
    levels = Vector{UInt64}[]
    words = max(1, cld(n, 64))
    while true
        push!(levels, zeros(UInt64, words))
        words == 1 && break
        words = cld(words, 64)
    end
    return _ParityColumn(levels)
end

# Change a whole coefficient word, propagating occupancy only if needed.
@inline function _xor_word!(column::_ParityColumn, word::Int, mask::UInt64)
    @inbounds begin
        words = column.levels[1]
        before = words[word]
        after = before ⊻ mask
        words[word] = after
        iszero(before) == iszero(after) && return nothing
        index = word
        for level in 2:length(column.levels)
            words = column.levels[level]
            word = ((index - 1) >>> 6) + 1
            mask = UInt64(1) << ((index - 1) & 63)
            before = words[word]
            after = before ⊻ mask
            words[word] = after
            iszero(before) == iszero(after) && break
            index = word
        end
    end
    return nothing
end

@inline function _toggle_entry!(column::_ParityColumn, index::Int)
    _xor_word!(column, ((index - 1) >>> 6) + 1, UInt64(1) << ((index - 1) & 63))
end

@inline function _column_pivot(column::_ParityColumn)
    iszero(column.levels[end][1]) && return 0
    index = 1
    @inbounds for level in length(column.levels):-1:1
        word = column.levels[level][index]
        index = ((index - 1) << 6) + 64 - leading_zeros(word)
    end
    return index
end

# After canceling a pivot, no entry above it can have appeared. Usually the
# next pivot is in the same word, avoiding a walk down the entire hierarchy.
@inline function _column_pivot(column::_ParityColumn, previous::Int)
    word = ((previous - 1) >>> 6) + 1
    @inbounds bits = column.levels[1][word]
    return iszero(bits) ? _column_pivot(column) : ((word - 1) << 6) + 64 - leading_zeros(bits)
end

# A finished column is consumed as a whole. Visit each occupied hierarchy
# word once, instead of repairing occupancy after removing each coefficient.
function _drain_level!(out::Vector{Int}, levels::Vector{Vector{UInt64}}, level::Int, index::Int)
    @inbounds bits = levels[level][index]
    @inbounds levels[level][index] = 0
    offset = (index - 1) << 6
    while !iszero(bits)
        bit = 64 - leading_zeros(bits)
        bits ⊻= UInt64(1) << (bit - 1)
        if level == 1
            push!(out, offset + bit)
        else
            _drain_level!(out, levels, level - 1, offset + bit)
        end
    end
    return nothing
end

function _drain_column!(out::Vector{Int}, column::_ParityColumn)
    empty!(out)
    _drain_level!(out, column.levels, length(column.levels), 1)
    return out
end

# Saved columns share two append-only payloads. A positive span length denotes
# sparse indices; a negative length denotes packed (word, mask) pairs. Keeping
# the same density rule avoids making sparse additions pay for empty bits.
struct _BarcodeColumns
    spans::Vector{Tuple{Int,Int}}
    rows::Vector{Int}
    words::Vector{Tuple{Int,UInt64}}
end
_BarcodeColumns(n::Int) = _BarcodeColumns(Vector{Tuple{Int,Int}}(undef, n), Int[], Tuple{Int,UInt64}[])

function _store_barcode_column!(stored::_BarcodeColumns, pivot::Int, col::Vector{Int})
    nwords = 0
    previous = 0
    for row in col
        word = ((row - 1) >>> 6) + 1
        nwords += word != previous
        previous = word
    end
    if length(col) < 2 * nwords
        start = length(stored.rows) + 1
        append!(stored.rows, col)
        stored.spans[pivot] = (start, length(col))
    else
        start = length(stored.words) + 1
        previous = 0
        mask = UInt64(0)
        for row in col
            word = ((row - 1) >>> 6) + 1
            if word != previous
                previous > 0 && push!(stored.words, (previous, mask))
                previous = word
                mask = UInt64(0)
            end
            mask |= UInt64(1) << ((row - 1) & 63)
        end
        previous > 0 && push!(stored.words, (previous, mask))
        stored.spans[pivot] = (start, -nwords)
    end
    return nothing
end

@inline function _add_barcode_column!(active::_ParityColumn, stored::_BarcodeColumns, pivot::Int)
    @inbounds start, count = stored.spans[pivot]
    if count > 0
        @inbounds for i in start:(start + count - 1)
            _toggle_entry!(active, stored.rows[i])
        end
    else
        @inbounds for i in start:(start - count - 1)
            word, mask = stored.words[i]
            _xor_word!(active, word, mask)
        end
    end
    return nothing
end

@inline function _component_root!(parent, vertex)
    @inbounds while parent[vertex] != vertex
        parent[vertex] = parent[parent[vertex]]
        vertex = parent[vertex]
    end
    return vertex
end

# Graph incidence in degree one determines H0 even in higher-dimensional
# complexes. A clearing set from higher degrees identifies edges already paired
# with faces; only the remaining cycle edges represent essential H1 classes.
# A graph-only caller supplies nothing for that set.
function _graph_barcode!(intervals, essential, G, offsets, values, perm, rank,
                         cleared::Union{Nothing,BitVector})
    boundary = isempty(G.boundaries) ? nothing : G.boundaries[1]
    if boundary !== nothing
      for col in axes(boundary, 2)
        count = 0
        @inbounds for ptr in nzrange(boundary, col)
            count += isodd(boundary.nzval[ptr])
        end
        count in (0, 2) || return false
      end
    end
    nvertices = length(offsets) >= 2 ? offsets[2] - 1 : 0
    lastedge = length(offsets) >= 3 ? offsets[3] - 1 : nvertices
    parent = collect(1:nvertices)
    sizes = ones(Int, nvertices)
    elder = copy(parent)
    for position in eachindex(perm)
        cell = perm[position]
        nvertices < cell <= lastedge || continue
        cleared !== nothing && cleared[position] && continue
        col = cell - nvertices
        a = 0
        b = 0
        @inbounds for ptr in nzrange(boundary, col)
            isodd(boundary.nzval[ptr]) || continue
            a == 0 ? (a = boundary.rowval[ptr]) : (b = boundary.rowval[ptr])
        end
        if a == 0
            push!(essential[2], values[cell])
            continue
        end
        ra = _component_root!(parent, a)
        rb = _component_root!(parent, b)
        if ra == rb
            push!(essential[2], values[cell])
            continue
        end
        born_a, born_b = elder[ra], elder[rb]
        older, younger = rank[born_a] < rank[born_b] ? (born_a, born_b) : (born_b, born_a)
        values[younger] == values[cell] || push!(intervals[1], (values[younger], values[cell]))
        if sizes[ra] < sizes[rb]
            ra, rb = rb, ra
        end
        parent[rb] = ra
        sizes[ra] += sizes[rb]
        elder[ra] = older
    end
    for vertex in 1:nvertices
        parent[vertex] == vertex && push!(essential[1], values[elder[vertex]])
    end
    return true
end

# Reverse the top boundary: its cells are dual vertices and its rows are
# dual edges when each has at most two odd incidences. A one-ended edge
# meets a distinguished auxiliary vertex. It has no reported essential class.
function _dual_top_barcode!(intervals, essential, G, offsets, values, perm, rank,
                            cleared, dim::Int)
    dim >= 2 || return false
    boundary = G.boundaries[dim]
    nfaces, ntop = size(boundary)
    first = zeros(Int,nfaces)
    second = zeros(Int,nfaces)
    for col in axes(boundary,2)
        @inbounds for ptr in nzrange(boundary,col)
            isodd(boundary.nzval[ptr]) || continue
            row = boundary.rowval[ptr]
            if first[row] == 0
                first[row] = col
            elseif second[row] == 0
                second[row] = col
            else
                return false
            end
        end
    end
    # Eligibility is now established; no fallback can observe partial outputs.
    auxiliary = ntop + 1
    parent = collect(1:auxiliary)
    sizes = ones(Int,auxiliary)
    elder = copy(parent)
    topoffset = offsets[dim+1] - 1
    faceoffset = offsets[dim] - 1
    for position in reverse(eachindex(perm))
        face = perm[position]
        faceoffset < face <= faceoffset+nfaces || continue
        row = face - faceoffset
        a = first[row]
        a == 0 && continue
        b = second[row] == 0 ? auxiliary : second[row]
        ra = _component_root!(parent,a)
        rb = _component_root!(parent,b)
        ra == rb && continue
        ea, eb = elder[ra], elder[rb]
        # Older in reverse order means later in the original order. The
        # auxiliary vertex is older than every actual top-dimensional cell.
        aolder = ea == auxiliary || (eb != auxiliary && rank[topoffset+ea] > rank[topoffset+eb])
        older, younger = aolder ? (ea,eb) : (eb,ea)
        topcell = topoffset + younger
        values[face] == values[topcell] || push!(intervals[dim],(values[face],values[topcell]))
        cleared[position] = true
        if sizes[ra] < sizes[rb]
            ra, rb = rb, ra
        end
        parent[rb] = ra
        sizes[ra] += sizes[rb]
        elder[ra] = older
    end
    for vertex in 1:ntop
        parent[vertex] == vertex || continue
        survivor = elder[vertex]
        survivor == auxiliary || push!(essential[dim+1],values[topoffset+survivor])
    end
    return true
end

# Descending chain degrees allow a paired birth column to be cleared without
# reducing it to zero. This shortcut computes barcodes, not retained cycles.
function _reduce_barcode!(intervals, essential, G, dims, offsets, values, perm, rank)
    total = length(perm)
    cleared = falses(total)
    has_reduced = falses(total)
    reduced = _BarcodeColumns(total)
    col = Int[]
    active = _ParityColumn(total)
    # The entire reduction uses filtration positions. Cell identities are
    # translated only on input and when materializing barcode endpoints.
    for dim in (length(offsets) - 2):-1:0
        dim == length(offsets)-2 && _dual_top_barcode!(intervals, essential, G, offsets, values, perm, rank, cleared, dim) && continue
        dim == 1 && _graph_barcode!(intervals, essential, G, offsets, values, perm, rank, cleared) && return nothing
        for position in eachindex(perm)
            cell = perm[position]
            dims[cell] == dim && !cleared[position] || continue
            if dim != 0
                boundary = G.boundaries[dim]
                local_col = cell - offsets[dim + 1] + 1
                @inbounds for ptr in nzrange(boundary, local_col)
                    isodd(boundary.nzval[ptr]) || continue
                    _toggle_entry!(active, rank[offsets[dim] + boundary.rowval[ptr] - 1])
                end
            end
            pivot_rank = _column_pivot(active)
            while pivot_rank != 0 && has_reduced[pivot_rank]
                _add_barcode_column!(active, reduced, pivot_rank)
                pivot_rank = _column_pivot(active, pivot_rank)
            end
            if pivot_rank == 0
                push!(essential[dim + 1], values[cell])
            else
                pivot = perm[pivot_rank]
                _drain_column!(col, active)
                has_reduced[pivot_rank] = true
                _store_barcode_column!(reduced, pivot_rank, col)
                cleared[pivot_rank] = true
                values[pivot] == values[cell] ||
                    push!(intervals[dim], (values[pivot], values[cell]))
            end
        end
    end
    return nothing
end

"""
    persistence_diagram(G::GradedComplex; order=:sublevel, field=F2(), representatives=false)

Reduce a finite, one-parameter graded chain complex over `F2()`. Integer
boundary coefficients are read modulo two. The input must have finite real
grades, correctly shaped boundaries, zero double boundary modulo two, and
boundary maps compatible with the selected order. These conditions are checked
before reduction.

Sublevels use increasing grades. Superlevels use decreasing grades without
negating or converting the grade values. Births are included and deaths
excluded in filtration order; zero-length intervals are omitted.
Opt in with `representatives=true` to retain selected-interval cycles and finite
death bounding chains for [`persistence_representative`](@ref). This tracks sparse
change-of-basis columns during reduction and can substantially increase memory.
"""
function persistence_diagram(G::GradedComplex{N,T};
                             order::Symbol=:sublevel, field=F2(), representatives=false) where {N,T}
    _normalize_order(order)
    _normalize_field(field)
    representatives isa Bool || throw(ArgumentError("representatives must be true or false."))
    offsets = _validate_complex(G, order)
    dims = _cell_dimensions(offsets)
    total = length(G.grades)
    values = T[g[1] for g in G.grades]
    # Integer grades use Julia's stable permutation sort. Cell storage is
    # already grouped by dimension, so stable ties put faces before cofaces
    # and then preserve cell identity. Other real grades retain the comparator
    # below, including its equality convention for signed floating-point zero.
    perm = if T <: Integer
        sortperm(values; rev=order === :superlevel)
    else
        indices = collect(1:total)
        sort!(indices; lt=(a, b) -> values[a] == values[b] ?
            (dims[a] == dims[b] ? a < b : dims[a] < dims[b]) :
            (order === :sublevel ? values[a] < values[b] : values[a] > values[b]))
    end
    rank = Vector{Int}(undef, total)
    for (r, cell) in enumerate(perm)
        rank[cell] = r
    end

    nd = length(offsets) - 1
    intervals = [Tuple{T,T}[] for _ in 1:nd]
    essential = [T[] for _ in 1:nd]
    finite_reps = representatives ? [_PersistenceRepresentative{T}[] for _ in 1:nd] : nothing
    essential_reps = representatives ? [_PersistenceRepresentative{T}[] for _ in 1:nd] : nothing
    backend = :f2_column_reduction
    if !representatives
        if last(offsets) == offsets[min(3, length(offsets))] &&
           _graph_barcode!(intervals, essential, G, offsets, values, perm, rank, nothing)
            backend = :f2_graph_union_find
        else
            _reduce_barcode!(intervals, essential, G, dims, offsets, values, perm, rank)
            backend = :f2_clearing
        end
    else
        reduced = Vector{Vector{Int}}(undef, total)
        has_reduced = falses(total)
        positive = falses(total)
        alive = falses(total)
        changes = representatives ? Vector{Vector{Int}}(undef, total) : nothing
        col, column_scratch = Int[], Int[]
        change = representatives ? Int[] : nothing
        change_scratch = representatives ? Int[] : nothing
        for cell in perm
            _boundary_column_f2!(col, G, dims, offsets, rank, cell)
            if representatives
                empty!(change)
                push!(change, cell)
            end
            while !isempty(col)
                pivot = col[end]
                has_reduced[pivot] || break
                # Both work buffers remain reusable. Only completed columns are
                # copied into the pivot table or retained representative changes.
                _xor_columns!(column_scratch, col, reduced[pivot], rank)
                col, column_scratch = column_scratch, col
                if representatives
                    _xor_columns!(change_scratch, change, changes[pivot], rank)
                    change, change_scratch = change_scratch, change
                end
            end
            if isempty(col)
                positive[cell] = true
                alive[cell] = true
                representatives && (changes[cell] = copy(change))
            else
                pivot = col[end]
                has_reduced[pivot] = true
                reduced[pivot] = copy(col)
                positive[pivot] || throw(ArgumentError("ordinary persistence internal inconsistency: pivot did not birth a class."))
                alive[pivot] = false
                if representatives
                    # The reduced death column is a cycle already present at the
                    # pivot's birth. Its tracked column bounds it exactly at death.
                    changes[pivot] = copy(change)
                    if values[pivot] != values[cell]
                        dim = dims[pivot]
                        push!(finite_reps[dim + 1], _PersistenceRepresentative{T}(
                            values[pivot], values[cell],
                            _representative_chain(G, offsets, dim, col),
                            _representative_chain(G, offsets, dim + 1, change)))
                    end
                end
                values[pivot] == values[cell] ||
                    push!(intervals[dims[pivot] + 1], (values[pivot], values[cell]))
            end
        end
        for cell in 1:total
            if alive[cell]
                push!(essential[dims[cell] + 1], values[cell])
                representatives && push!(essential_reps[dims[cell] + 1], _PersistenceRepresentative{T}(
                    values[cell], nothing, _representative_chain(G, offsets, dims[cell], changes[cell]), nothing))
            end
        end
    end
    for (values_by_dim, reps_by_dim) in ((intervals, finite_reps), (essential, essential_reps))
        for slot in eachindex(values_by_dim)
            values_at_dim = values_by_dim[slot]
            if representatives
                permutation = sortperm(values_at_dim; rev=order === :superlevel)
                values_by_dim[slot] = values_at_dim[permutation]
                reps_by_dim[slot] = reps_by_dim[slot][permutation]
            else
                sort!(values_at_dim; rev=order === :superlevel)
            end
        end
    end
    # The reducer established these invariants; avoid revalidating its output.
    meta = (construction=(requested=:graded_chain_complex, effective=:graded_chain_complex,
                          substitution=:none), source=:graded_complex,
            grade_arithmetic=:exact_stored_values, geometry=:not_recorded,
            chain_validation=:boundary_squared_zero_mod_two,
            backend=backend, approximation=:none_in_reduction,
            discretization=:none)
    retained = representatives ? _PersistenceRepresentatives{T}(finite_reps, essential_reps) : nothing
    return PersistenceDiagram(intervals, essential, field, order, meta, retained)
end

"""
    persistence_diagram(data, filtration; order=:sublevel, field=F2(), cache=nothing,
                        representatives=false)

Build a one-parameter complex with `build_graded_complex` and compute ordinary
persistent homology. The filtration must actually be compatible with `order`;
changing this keyword does not turn a general lower-star construction into an
upper-star construction. Cubical vertex data have a dedicated upper-star route.
Grade precision on other ingestion routes is that of their constructed complex.
"""
function persistence_diagram(data, filtration::DataIngestion.AbstractFiltration;
                             order::Symbol=:sublevel, field=F2(), cache=nothing, representatives=false)
    _normalize_order(order)
    _normalize_field(field)
    representatives isa Bool || throw(ArgumentError("representatives must be true or false."))
    build = DataIngestion.build_graded_complex(data, filtration; cache=cache)
    diag = persistence_diagram(DataIngestion.graded_complex(build); order, field, representatives)
    kind = DataIngestion.filtration_kind(filtration)
    # The public build result records grades and orientation, but not the
    # executed construction/backend. Do not infer an identity construction:
    # some supported requests deliberately substitute a different model.
    meta = merge(diag.meta, (construction=(requested=kind, effective=:not_recorded, substitution=:not_recorded),
                            source=typeof(data), geometry=:ingestion_contract))
    return PersistenceDiagram(diag.finite_by_dim, diag.essential_by_dim, diag.field, diag.order,
                              meta, diag.retained_representatives)
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
                        input=:top_cells, field=F2(), representatives=false)

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
`representatives=true` retains cell-chain cycles and finite death bounding
chains; inspect them with [`persistence_representative`](@ref). Cell indices refer
to the constructed cubical complex, not pixel coordinates.
"""
function cubical_persistence(values::AbstractArray{<:Real}; periodic=false,
                             order::Symbol=:sublevel, input::Symbol=:top_cells,
                             field=F2(), representatives=false)
    _normalize_order(order)
    _normalize_field(field)
    representatives isa Bool || throw(ArgumentError("representatives must be true or false."))
    _validate_cubical_values(values)
    per = _periodic_tuple(periodic, Val(ndims(values)))
    if input === :vertices
        return _vertex_cubical_persistence(values, per, order, field,
            DataIngestion.construction_mode(DataIngestion.CubicalFiltration()), nothing, representatives)
    end
    input === :top_cells || throw(ArgumentError("cubical persistence input must be :top_cells or :vertices."))
    ndims(values) == 2 || throw(ArgumentError("cubical persistence input=:top_cells supports two-dimensional arrays."))
    G = _top_cell_complex_2d(values, per, order)
    diag = persistence_diagram(G; order, field, representatives)
    meta = merge(diag.meta, (construction=(requested=:cubical_top_cells, effective=:cubical_top_cells,
        substitution=:none), source=(shape=size(values), periodic=per),
        grade_arithmetic=:exact_input_values, geometry=:cubical_cells))
    return PersistenceDiagram(diag.finite_by_dim, diag.essential_by_dim, diag.field, diag.order,
                              meta, diag.retained_representatives)
end

function _vertex_cubical_persistence(values, per, order, field, construction, cache, representatives)
    G = _vertex_cubical_complex(values, per, order, construction, cache)
    diag = persistence_diagram(G; order, field, representatives)
    meta = merge(diag.meta, (construction=(requested=:cubical_vertices, effective=:cubical_vertices,
        substitution=:none), source=(shape=size(values), periodic=per),
        grade_arithmetic=:exact_input_values, geometry=:cubical_cells))
    return PersistenceDiagram(diag.finite_by_dim, diag.essential_by_dim, diag.field, diag.order,
                              meta, diag.retained_representatives)
end

function persistence_diagram(data::ImageNd, filtration::DataIngestion.CubicalFiltration;
                             order::Symbol=:sublevel, field=F2(), cache=nothing, representatives=false)
    _normalize_order(order)
    _normalize_field(field)
    representatives isa Bool || throw(ArgumentError("representatives must be true or false."))
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
                                      DataIngestion.construction_mode(filtration), cache, representatives)
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
