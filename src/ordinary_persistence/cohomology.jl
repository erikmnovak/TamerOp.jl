# Reverse-filtration coboundary reduction. Over a field, finite-dimensional
# absolute cohomology is the natural dual of homology. The anti-transpose
# reduction pairs have the same intervals; restrictions reverse the arrows.
# See de Silva, Morozov and Vejdemo-Johansson, Dualities in persistent (co)homology
# (2011), and Bauer, Ripser (2021), Section 3.3.

function _sort_cocycles!(finite, essential, retained, order)
    for (bars, records) in ((finite, retained.finite), (essential, retained.essential))
        for slot in eachindex(bars)
            permutation = sortperm(bars[slot]; rev=order === :superlevel)
            bars[slot] = bars[slot][permutation]
            records[slot] = records[slot][permutation]
        end
    end
end

function _reduce_cohomology(G::GradedComplex{N,T}, offsets, dims, values, perm, rank,
                            field, order) where {N,T}
    K = coeff_type(field)
    nd = length(offsets)-1
    finite = [Tuple{T,T}[] for _ in 1:nd]
    essential = [T[] for _ in 1:nd]
    retained = _PersistenceCocycles([_PersistenceCocycle{T}[] for _ in 1:nd],
        [_PersistenceCocycle{T}[] for _ in 1:nd], :dimension_local_input_cells)
    reverse_rank = length(rank) .+ 1 .- rank
    cleared = Set{Int}()
    for dim in 0:(nd-1)
        # Transpose only the current differential, not the whole complex.
        delta = dim+1 < nd ? sparse(transpose(G.boundaries[dim+1])) : nothing
        pivots = Dict{Int,Tuple{Vector{Pair{Int,K}},Vector{Pair{Int,K}}}}()
        next_cleared = Set{Int}()
        column, scratch = Pair{Int,K}[], Pair{Int,K}[]
        change, change_scratch = Pair{Int,K}[], Pair{Int,K}[]
        for cell in Iterators.reverse(perm)
            dims[cell] == dim && !(cell in cleared) || continue
            empty!(column); empty!(change)
            push!(change, cell => one(K))
            if delta !== nothing
                local_cell = cell - offsets[dim+1] + 1
                for ptr in nzrange(delta, local_cell)
                    c = K(delta.nzval[ptr])
                    iszero(c) || push!(column, (offsets[dim+2]+delta.rowval[ptr]-1) => c)
                end
                sort!(column; by=e -> reverse_rank[first(e)])
            end
            while !isempty(column) && haskey(pivots, first(last(column)))
                reduced, basis_change = pivots[first(last(column))]
                scale = -last(last(column))
                _prime_add_scaled!(scratch, column, reduced, scale, reverse_rank)
                column, scratch = scratch, column
                _prime_add_scaled!(change_scratch, change, basis_change, scale, reverse_rank)
                change, change_scratch = change_scratch, change
            end
            death = nothing
            if !isempty(column)
                pivot = first(last(column))
                factor = inv(last(last(column)))
                _prime_scale!(column, factor); _prime_scale!(change, factor)
                pivots[pivot] = (copy(column), copy(change))
                push!(next_cleared, pivot)
                death = values[pivot]
                values[cell] == death && continue
                push!(finite[dim+1], (values[cell], death))
            else
                push!(essential[dim+1], values[cell])
            end
            record = _PersistenceCocycle{T}(values[cell], death,
                _representative_chain(G, offsets, dim, change), nothing)
            push!(death === nothing ? retained.essential[dim+1] : retained.finite[dim+1], record)
        end
        cleared = next_cleared
    end
    _sort_cocycles!(finite, essential, retained, order)
    meta = (construction=(requested=:graded_chain_complex,effective=:graded_chain_complex,substitution=:none),
        source=:graded_complex, grade_arithmetic=:exact_stored_values, geometry=:not_recorded,
        backend=:prime_coboundary_reduction, approximation=:none_in_reduction, discretization=:none,
        chain_validation=:boundary_squared_zero_mod_prime)
    return PersistenceDiagram(finite, essential, field, order, meta, nothing, retained)
end

"""
    persistence_cocycle(diag; dim, scale, kind=:finite, index=1)

Return a retained cohomology representative at the specified finite `scale`.
Compute `diag` with `cocycles=true`. `kind` and `index` select a member of
`finite_intervals` or `essential_births`, including multiplicities. `scale` must
lie in that interval and in the recorded filtration window. Birth is included;
finite death is excluded. An ordinary endpoint-only diagram reports
`available=false`; invalid queries still raise an ArgumentError.

The returned `cochain` assigns coefficients to oriented cells. Omitted cells
have coefficient zero. Integer coefficients are canonical residues modulo
`field.p`. Its coboundary vanishes on active cofaces, and its class is nonzero
modulo coboundaries. Restricting the same member from a larger subcomplex to a
smaller one within its interval is exactly deletion of inactive cells. For
sublevels this is H^d(K_t) -> H^d(K_s) for s <= t; superlevels reverse numerical
inequalities. Cohomology has the same interval multiset as homology over the
supported fields, but these restriction maps are contravariant.

Supplied complexes use dimension-local indices, original cell IDs, grades and
their boundary orientations. Implicit Rips uses colex simplex indices in the
selected vertex set and additionally returns ordered `source_vertices` in
original point/distance-row indices. That tuple order defines orientation,
even when landmark indices are unsorted. A landmark/neighbor graph represents
the selected complex, not all original points/edges. Generic ingestion returns
constructed-cell identities and does not assert an original-data embedding.

These deterministic representatives are not canonical or optimized. Equal bars
are separate members. Homology cycles retained by `representatives=true` are
chosen independently, so equal member indices do not assert dual bases. No
integer lift, circular coordinates or cup products are computed. Returned
records are immutable copies; treat the diagram's storage as read-only.
"""
function persistence_cocycle(diag::PersistenceDiagram; dim::Integer, scale,
                            kind::Symbol=:finite, index::Integer=1)
    slot = _diagram_degree_slot(diag, dim)
    kind in (:finite, :essential) || throw(ArgumentError("kind must be :finite or :essential."))
    !(index isa Bool) && 1 <= index <= typemax(Int) ||
        throw(ArgumentError("index must be a positive interval member index fitting Int."))
    bars = kind === :finite ? diag.finite_by_dim : diag.essential_by_dim
    slot <= length(bars) && index <= length(bars[slot]) ||
        throw(ArgumentError("index is outside the stored $kind intervals in dimension $dim."))
    scale isa Real && !(scale isa Bool) && isfinite(scale) ||
        throw(ArgumentError("scale must be a finite real parameter."))
    birth = kind === :finite ? bars[slot][index][1] : bars[slot][index]
    death = kind === :finite ? bars[slot][index][2] : nothing
    active = diag.order === :sublevel ? birth <= scale && (death === nothing || scale < death) :
        scale <= birth && (death === nothing || death < scale)
    active || throw(ArgumentError("scale is outside the selected interval; birth is included and finite death excluded."))
    window = get(diag.meta, :window, :unrestricted)
    window isa Tuple && !(window[1] <= scale <= window[2]) &&
        throw(ArgumentError("scale lies outside this diagram's recorded filtration window."))
    retained = diag.retained_cocycles
    context = (; dimension=Int(dim), kind, index=Int(index), scale, field=diag.field,
        order=diag.order, interval=(birth, something(death, diag.order === :sublevel ? Inf : -Inf)),
        variance=:contravariant, choice=:noncanonical_coboundary_reduction)
    retained === nothing && return merge(context, (;available=false, reason=:not_retained,
        cochain=nothing, source_vertices=nothing, indexing=:not_available))
    records = kind === :finite ? retained.finite : retained.essential
    slot <= length(records) && index <= length(records[slot]) ||
        throw(ArgumentError("retained cocycles disagree with diagram intervals."))
    r = records[slot][index]
    (r.birth, r.death) == (birth, death) ||
        throw(ArgumentError("retained cocycle endpoints disagree with diagram; recompute after mutation."))
    c = r.cochain
    selected = findall(g -> diag.order === :sublevel ? g <= scale : g >= scale, c.grades)
    cochain = (; dimension=Int(dim), cell_indices=Tuple(c.indices[selected]),
        cell_ids=Tuple(c.ids[selected]), cell_grades=Tuple(c.grades[selected]),
        coefficients=Tuple(c.coefficients[selected]))
    vertices = r.vertices === nothing ? nothing : Tuple(Tuple(r.vertices[i]) for i in selected)
    return merge(context, (;available=true, reason=:retained, cochain,
        source_vertices=vertices, indexing=retained.indexing))
end

function _check_cocycle_storage!(issues, diag)
    retained = diag.retained_cocycles
    retained === nothing && return
    for (kind, bars, records) in ((:finite,diag.finite_by_dim,retained.finite),
                                  (:essential,diag.essential_by_dim,retained.essential))
        if length(bars) != length(records)
            push!(issues, "retained cocycles cover different degrees.")
            continue
        end
        for slot in eachindex(bars)
            if length(bars[slot]) != length(records[slot])
                push!(issues, "retained cocycle counts disagree with intervals.")
                continue
            end
            for (i,r) in enumerate(records[slot])
                expected = kind === :finite ? bars[slot][i] : (bars[slot][i],nothing)
                (r.birth,r.death) == expected || push!(issues,"retained cocycle endpoints disagree with intervals.")
                c = r.cochain
                length(c.indices) == length(c.ids) == length(c.grades) == length(c.coefficients) ||
                    push!(issues,"retained cochain arrays have different lengths.")
                !isempty(c.indices) && issorted(c.indices) && allunique(c.indices) && all(>(0),c.indices) ||
                    push!(issues,"retained cochains require distinct positive cell indices in increasing order.")
                all(x -> 0 < x < diag.field.p,c.coefficients) || push!(issues,"invalid cochain coefficients.")
                all(g -> isfinite(g) && (diag.order === :sublevel ? g >= r.birth : g <= r.birth),c.grades) ||
                    push!(issues,"retained cochain has a cell preceding its birth.")
                r.vertices === nothing || (length(r.vertices)==length(c.indices) &&
                    all(v -> length(v)==slot && allunique(v) && all(>(0),v),r.vertices)) ||
                    push!(issues,"invalid retained simplex vertex identities.")
            end
        end
    end
end
