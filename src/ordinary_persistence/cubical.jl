# Exact top-cell cubical filtrations. Topology uses the ingestion owner's cell
# order and signed tensor boundary; grades use geometric incidence, including
# periodic faces whose two boundary coefficients cancel.

function _top_cell_preflight(values::AbstractArray{T,N}, periodic::NTuple{N,Bool},
                             construction::ConstructionOptions) where {T,N}
    N >= 1 || throw(ArgumentError("cubical persistence requires at least one array dimension."))
    all(>(0), size(values)) || throw(ArgumentError("cubical top-dimensional cells must be nonempty."))
    construction.sparsify === :none && construction.collapse === :none ||
        throw(ArgumentError("Cubical construction requires sparsify=:none and collapse=:none."))
    dims = ntuple(N) do i
        n = big(size(values, i)) + Int(!periodic[i])
        n <= typemax(Int) || throw(ArgumentError("Cubical vertex count exceeds machine indexing on axis $i."))
        Int(n)
    end
    counts = DataIngestion._estimate_cubical_cell_counts(dims; periodic)
    total = sum(counts)
    # Include uncancelled incidences: periodic singleton axes can cancel in the
    # final matrix. This conservatively covers both triplet and direct assembly.
    incidences = sum(2d * counts[d + 1] for d in 1:N)
    max(total, incidences) < typemax(Int) ||
        throw(ArgumentError("Cubical construction exceeds machine-indexable cell/incidence storage."))
    spec = DataIngestion.FiltrationSpec(; kind=:cubical, construction)
    DataIngestion._construction_check_max_simplices!(total, N, spec)
    DataIngestion._construction_check_max_edges!(counts[2], spec)
    # Budget estimate covers coordinates, orientation indices, cell IDs,
    # grades, CSC boundaries and conservative assembly workspace. Dynamic
    # scalar payloads are shared with the input; allocator/runtime overhead and
    # optional geometry, representatives and reduction storage are not RSS caps.
    bytes = total * ((2N + 20) * sizeof(Int) + Base.elsize(Vector{NTuple{1,T}})) +
            6sizeof(Int) * incidences + sizeof(Int) * (total + N)
    DataIngestion._construction_check_memory_budget!(bytes, spec)
    return dims
end

function _incident_top_cell_grade(values::AbstractArray{T,N}, origin::NTuple{N,Int},
                                   mask::NTuple{N,Int}, periodic::NTuple{N,Bool},
                                   order::Symbol) where {T,N}
    shape = size(values)
    first_cell = ntuple(i -> min(origin[i], shape[i]), N)
    other_cell = ntuple(N) do i
        mask[i] == 1 && return first_cell[i]
        previous = _candidate_prev(origin[i], shape[i], periodic[i])
        previous == 0 ? first_cell[i] : previous
    end
    ranges = ntuple(i -> 1:(first_cell[i] == other_cell[i] ? 1 : 2), N)
    @inbounds grade = values[first_cell...]
    for choice in CartesianIndices(ranges)
        coords = ntuple(i -> choice[i] == 1 ? first_cell[i] : other_cell[i], N)
        @inbounds candidate = values[coords...]
        if order === :sublevel ? candidate < grade : candidate > grade
            grade = candidate
        end
    end
    return grade
end

function _top_cell_complex(values::AbstractArray{T,N}, periodic::NTuple{N,Bool},
                            order::Symbol, construction::ConstructionOptions) where {T<:Real,N}
    dims = _top_cell_preflight(values, periodic, construction)
    _validate_cubical_values(values)
    # Preserve the established direct indexing and storage costs in 2D.
    N == 2 && return _top_cell_complex_2d(values, periodic, order)
    structure = DataIngestion._cubical_structure(dims; periodic)
    cells = [collect(1:length(group)) for group in structure.cells_by_dim]
    grades = Vector{NTuple{1,T}}(undef, sum(length, cells))
    j = 1
    for group in structure.cells_by_dim, (origin, mask) in group
        grades[j] = (_incident_top_cell_grade(values, origin, mask, periodic, order),)
        j += 1
    end
    return GradedComplex(cells, structure.boundaries, grades)
end
