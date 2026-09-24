# Exact integer-depth slices and closed slabs of the rhomboid cell complex.
# Native cellular incidence avoids flag-subdivision growth on ordinary builds.

struct _RhomboidDepthCell{M<:Integer}
    carrier::_RhomboidCell{M}
    level::Int
    slab::Bool
end
Base.:(==)(a::_RhomboidDepthCell, b::_RhomboidDepthCell) =
    a.carrier == b.carrier && a.level == b.level && a.slab == b.slab
Base.isequal(a::_RhomboidDepthCell, b::_RhomboidDepthCell) =
    isequal(a.carrier, b.carrier) && a.level == b.level && a.slab == b.slab
Base.hash(c::_RhomboidDepthCell, seed::UInt) = hash(c.slab, hash(c.level, hash(c.carrier, seed)))

@inline function _rhomboid_depth_dimension(cell::_RhomboidDepthCell)
    q = Int(count_ones(cell.carrier.on))
    return cell.slab ? q : max(0, q - 1)
end

# Normalize extreme horizontal sections to their true vertex carrier. This
# gives shared faces identical keys and their own (possibly earlier) radius.
function _rhomboid_horizontal_cell(carrier::_RhomboidCell{M}, level::Int) where {M}
    anchor, q = Int(count_ones(carrier.inside)), Int(count_ones(carrier.on))
    anchor <= level <= anchor + q || return nothing
    if level == anchor
        carrier = _RhomboidCell(carrier.inside, zero(M))
    elseif level == anchor + q
        carrier = _RhomboidCell(carrier.inside | carrier.on, zero(M))
    end
    return _RhomboidDepthCell(carrier, level, false)
end

# Orient a slab by its ordered cube coordinates. Orient a horizontal section
# by (e_1-e_q,...,e_(q-1)-e_q), with the same global site order. Coordinate
# facet signs are the usual cubical signs in either case. The slab's lower
# and upper horizontal facets have signs (-1)^q and (-1)^(q-1).
function _rhomboid_depth_facets(cell::_RhomboidDepthCell{M}, n::Int) where {M}
    carrier, level = cell.carrier, cell.level
    sites = _rhomboid_indices(carrier.on, n)
    q = length(sites)
    degree = _rhomboid_depth_dimension(cell)
    result = Tuple{_RhomboidDepthCell{M},Int}[]
    degree == 0 && return result
    sizehint!(result, 2q + 2)
    if cell.slab
        for (height, sign) in ((level, isodd(q) ? -1 : 1),
                               (level + 1, isodd(q) ? 1 : -1))
            facet = _rhomboid_horizontal_cell(carrier, height)
            _rhomboid_depth_dimension(facet) == degree - 1 && push!(result, (facet, sign))
        end
        q == 1 && return result  # coordinate endpoints coincide with the caps
    elseif q == 2
        # Each endpoint lies in two coordinate facets of the original square;
        # count it once with the orientation e_first-e_last.
        for (site, sign) in ((first(sites), 1), (last(sites), -1))
            vertex = _RhomboidCell(carrier.inside | (one(M) << (site - 1)), zero(M))
            push!(result, (_RhomboidDepthCell(vertex, level, false), sign))
        end
        return result
    end
    for (position, site) in enumerate(sites)
        bit = one(M) << (site - 1)
        for upper in (false, true)
            face = _RhomboidCell(upper ? carrier.inside | bit : carrier.inside, carrier.on & ~bit)
            facet = if cell.slab
                anchor = Int(count_ones(face.inside))
                anchor <= level < anchor + q - 1 ? _RhomboidDepthCell(face, level, true) : nothing
            else
                _rhomboid_horizontal_cell(face, level)
            end
            facet === nothing && continue
            _rhomboid_depth_dimension(facet) == degree - 1 || continue
            sign = isodd(position) ? -1 : 1
            push!(result, (facet, upper ? -sign : sign))
        end
    end
    return result
end

# Return the descriptors as well as the complex: the descriptors support the
# simplex stage and independent natural comparison maps in the owner tests.
function _rhomboid_depth_model(cells::Vector{Vector{_RhomboidCell{M}}}, radii2,
                               n::Int, spec::FiltrationSpec) where {M}
    lo, hi = spec.params[:depth_range]
    requested = get(spec.params, :max_dim, nothing)
    maxdegree = requested === nothing ? length(cells) - 1 : min(Int(requested), length(cells) - 1)
    groups = [Vector{_RhomboidDepthCell{M}}() for _ in 0:maxdegree]
    counts = zeros(Int, maxdegree + 1)
    function insert!(cell)
        degree = _rhomboid_depth_dimension(cell)
        degree <= maxdegree || return
        counts[degree + 1] += 1
        _rhomboid_storage_check!(counts, spec, n)
        push!(groups[degree + 1], cell)
    end
    for group in cells, carrier in group
        anchor, q = Int(count_ones(carrier.inside)), Int(count_ones(carrier.on))
        if q == 0
            lo <= anchor <= hi && insert!(_RhomboidDepthCell(carrier, anchor, false))
            continue
        end
        for level in max(lo, anchor + 1):min(hi, anchor + q - 1)
            insert!(_RhomboidDepthCell(carrier, level, false))
        end
        for level in max(lo, anchor):min(hi - 1, anchor + q - 1)
            insert!(_RhomboidDepthCell(carrier, level, true))
        end
    end
    foreach(group -> sort!(group; by=c -> (c.carrier.inside, c.carrier.on, c.level, c.slab)), groups)
    grades = NTuple{2,AlgebraicReal}[]
    sizehint!(grades, sum(counts))
    boundaries = SparseMatrixCSC{Int,Int}[]
    radii = Dict{_RhomboidCell{M},AlgebraicReal}()
    for (slot, group) in enumerate(groups)
        for cell in group
            radius = get!(radii, cell.carrier) do
                _rhomboid_radius(radii2[cell.carrier])
            end
            push!(grades, (radius, AlgebraicReal(cell.level)))
        end
        slot == 1 && continue
        index = Dict(cell => i for (i, cell) in enumerate(groups[slot - 1]))
        rows, cols, values = Int[], Int[], Int[]
        sizehint!(rows, 2slot * length(group))
        sizehint!(cols, 2slot * length(group))
        sizehint!(values, 2slot * length(group))
        for (column, cell) in enumerate(group)
            for (facet, coefficient) in _rhomboid_depth_facets(cell, n)
                row = get(index, facet, 0)
                row != 0 || error("Rhomboid depth restriction is missing a boundary face.")
                push!(rows, row)
                push!(cols, column)
                push!(values, coefficient)
            end
        end
        push!(boundaries, sparse(rows, cols, values, length(groups[slot-1]), length(group)))
    end
    for d in 2:length(boundaries)
        all(iszero, nonzeros(boundaries[d-1] * boundaries[d])) || error(
            "Rhomboid sliced-cell incidence does not square to zero over the integers.")
    end
    complex = GradedComplex([collect(eachindex(group)) for group in groups], boundaries, grades)
    return groups, complex
end

# Barycentric subdivision is explicit opt-in. A simplex is a strict face flag;
# its radius/depth is the grade of its largest cell. Face chains therefore
# restrict identically on shared faces at every radius and coverage depth.
function _rhomboid_depth_simplex_tree(groups, complex::GradedComplex, n::Int, spec::FiltrationSpec)
    counts = length.(groups)
    offsets = cumsum(vcat(0, counts))
    total = last(offsets)
    maxdegree = length(groups) - 1
    total == 0 && return SimplexTreeMulti(Int[1], Int[], Int[],
        ones(Int, length(groups) + 1), Int[1], NTuple{2,AlgebraicReal}[])
    workspace_bytes = big(total) * (128 + 16(maxdegree + 1))
    _construction_check_memory_budget!(workspace_bytes, spec)
    proper_faces = [Int[] for _ in 1:total]
    ways = [zeros(BigInt, maxdegree + 1) for _ in 1:total]
    flag_counts = zeros(BigInt, maxdegree + 1)
    face_entries = big(0)
    for (slot, group) in enumerate(groups), column in eachindex(group)
        id = offsets[slot] + column
        if slot > 1
            boundary = complex.boundaries[slot-1]
            faces = proper_faces[id]
            candidates = sum((1 + length(proper_faces[offsets[slot-1] + boundary.rowval[pointer]])
                              for pointer in nzrange(boundary, column)); init=big(0))
            _construction_check_memory_budget!(workspace_bytes + 8(face_entries + candidates), spec)
            for pointer in nzrange(boundary, column)
                face = offsets[slot-1] + boundary.rowval[pointer]
                push!(faces, face)
                append!(faces, proper_faces[face])
            end
            sort!(unique!(faces))
            face_entries += length(faces)
        end
        ways[id][1] = 1
        for face in proper_faces[id], degree in 1:maxdegree
            ways[id][degree+1] += ways[face][degree]
        end
        flag_counts .+= ways[id]
        _rhomboid_storage_check!(flag_counts, spec, n)
        output_bytes = sum((128 + 32degree) * flag_counts[degree+1] for degree in 0:maxdegree)
        _construction_check_memory_budget!(workspace_bytes + 8face_entries + output_bytes, spec)
    end
    simplices = [Vector{Vector{Int}}() for _ in groups]
    grades = [NTuple{2,AlgebraicReal}[] for _ in groups]
    for slot in eachindex(groups)
        sizehint!(simplices[slot], Int(flag_counts[slot]))
        sizehint!(grades[slot], Int(flag_counts[slot]))
    end
    chain = Int[]
    function flags!(cell, grade)
        push!(chain, cell)
        slot = length(chain)
        push!(simplices[slot], reverse(chain))
        push!(grades[slot], grade)
        for face in proper_faces[cell]
            flags!(face, grade)
        end
        pop!(chain)
    end
    for cell in 1:total
        flags!(cell, complex.grades[cell])
    end
    return _simplex_tree_multi_from_simplices(simplices, reduce(vcat, grades; init=NTuple{2,AlgebraicReal}[]))
end

function _rhomboid_depth_complex(cells, radii2, X::Matrix{QQ}, spec::FiltrationSpec;
                                 return_simplex_tree::Bool=false)
    n = size(X, 1)
    groups, complex = _rhomboid_depth_model(cells, radii2, n, spec)
    return return_simplex_tree ? _rhomboid_depth_simplex_tree(groups, complex, n, spec) : complex
end
