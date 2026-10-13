# Selected finite-grid Betti numbers. The right simple at a grid vertex has the
# tensor-product resolution of the simples on its chain factors. Tensoring
# with M gives the predecessor cube below. This computes finite-poset Betti
# multiplicities, including boundary vertices. It also computes polynomial
# Betti numbers for the extension which is zero below the grid and constant
# above it, in oriented integer index coordinates; arbitrary encodings need
# not have this interpretation.
using SparseArrays: spzeros, findnz
import ..FiniteFringe
import ..IndicatorResolutions
using ..CoreModules: _foreach_workchunk

_query_poset(M::PModule) = M.Q
_query_poset(H::FiniteFringe.FringeModule) = H.P
_query_module(M::PModule, labels::Vector{Int}) = (M, labels)

function _query_subposet(P, labels::Vector{Int})
    labels == 1:FiniteFringe.nvertices(P) && return P
    return FiniteFringe.FinitePoset([FiniteFringe.leq(P, a, b) for a in labels, b in labels])
end

function _query_module(H::FiniteFringe.FringeModule{K}, labels::Vector{Int}) where {K}
    P = _query_subposet(H.P, labels)
    U = [FiniteFringe.Upset(P, BitVector(u.mask[labels])) for u in H.U]
    D = [FiniteFringe.Downset(P, BitVector(d.mask[labels])) for d in H.D]
    restricted = FiniteFringe.FringeModule{K}(P, U, D, H.phi; field=H.field)
    return IndicatorResolutions.pmodule_from_fringe(restricted), collect(eachindex(labels))
end

_grid_betti_shape(::FiniteFringe.AbstractPoset) = nothing
_grid_betti_shape(P::FiniteFringe.ProductOfChainsPoset) = (P.sizes, P.strides, P.orientation)
_grid_betti_shape(P::FiniteFringe.GridPoset) = (P.sizes, P.strides, map(_ -> 1, P.sizes))
_betti_may_be_nonzero(M, v::Int) = true

function _betti_vertices(P, vertices)
    vertices isa AbstractVector || vertices isa Tuple ||
        throw(ArgumentError("betti_table: vertices must be a vector or tuple of integer vertex IDs"))
    n = FiniteFringe.nvertices(P)
    all(v -> v isa Integer && !(v isa Bool) && 1 <= v <= n, vertices) ||
        throw(ArgumentError("betti_table: vertices must lie in 1:$n"))
    return Int[v for v in vertices]
end

function _selected_betti_columns(B, ids, pad_to)
    selected = B[:, ids]
    pad_to === nothing || return Resolutions._pad_or_truncate_rows(selected, pad_to + 1)
    isempty(ids) && return zeros(Int, 1, 0)
    return Resolutions._trim_trailing_zero_rows(selected)
end

struct _BettiStencil
    masks::Vector{UInt}
    labels::Vector{Int}
    degrees::Vector{Int}
end

function _betti_stencil(shape, v::Int, maxlen::Int)
    sizes, strides, orientation = shape
    steps = Int[]
    for i in eachindex(sizes)
        c = div(v - 1, strides[i]) % sizes[i] + 1
        (orientation[i] == 1 ? c > 1 : c < sizes[i]) && push!(steps, orientation[i] * strides[i])
    end
    # Refuse exponential local work before allocating it; the general solver
    # remains available in high dimension. This is a resource bound, not a
    # claim that a particular dimension is a performance crossover.
    length(steps) < 8 * sizeof(UInt) || return nothing
    masks = UInt[0]; labels = [v]; degrees = [0]
    for (i, step) in enumerate(steps)
        for j in 1:length(masks)
            degrees[j] <= maxlen || continue
            length(masks) < 4096 || return nothing
            push!(masks, masks[j] | (UInt(1) << (i - 1)))
            push!(labels, labels[j] - step)
            push!(degrees, degrees[j] + 1)
        end
    end
    return _BettiStencil(masks, labels, degrees)
end

function _koszul_betti(M::PModule{K}, stencil::_BettiStencil, ids::Vector{Int}, maxlen::Int, check::Bool) where {K}
    dmax = maximum(stencil.degrees)
    totals = zeros(Int, dmax + 1)
    offsets = zeros(Int, length(ids))
    for i in eachindex(ids)
        d = stencil.degrees[i] + 1
        offsets[i] = totals[d]
        totals[d] += M.dims[ids[i]]
    end
    slots = Dict(mask => i for (i, mask) in enumerate(stencil.masks))
    ranks = zeros(Int, dmax + 2)
    previous = spzeros(K, 0, totals[1])
    for d in 1:dmax
        rows = Int[]; cols = Int[]; vals = K[]
        for j in eachindex(ids)
            stencil.degrees[j] == d || continue
            mask = stencil.masks[j]; remaining = mask; sign = one(K)
            while !iszero(remaining)
                bit = remaining & -remaining
                i = slots[xor(mask, bit)]
                A = Modules.map_leq(M, ids[j], ids[i])
                ar, ac, av = findnz(sparse(A))
                for k in eachindex(av)
                    push!(rows, offsets[i] + ar[k]); push!(cols, offsets[j] + ac[k])
                    push!(vals, sign * av[k])
                end
                sign = -sign; remaining = xor(remaining, bit)
            end
        end
        differential = sparse(rows, cols, vals, totals[d], totals[d+1])
        check && !Modules._coefficient_product_zero(M.field, previous, differential) &&
            throw(ArgumentError("betti_table: local structure maps do not form a Koszul complex at the field tolerance"))
        ranks[d+1] = FieldLinAlg.rank(M.field, differential)
        previous = differential
    end
    result = zeros(Int, maxlen + 1)
    for d in 0:min(maxlen, dmax)
        result[d+1] = totals[d+1] - ranks[d+1] - ranks[d+2]
        result[d+1] >= 0 || throw(ArgumentError("betti_table: inconsistent numerical ranks in the local complex"))
    end
    return result
end

function _selected_betti_table(M, vertices, opts::ResolutionOptions, pad_to; threads::Bool)
    P = _query_poset(M)
    ids = _betti_vertices(P, vertices)
    opts.maxlen >= 0 || throw(ArgumentError("betti_table: maxlen must be nonnegative"))
    pad_to === nothing || pad_to >= 0 || throw(ArgumentError("betti_table: pad_to must be nonnegative"))
    isempty(ids) && return zeros(Int, pad_to === nothing ? 1 : pad_to+1, 0)
    shape = _grid_betti_shape(P)
    shape === nothing && return nothing
    maxlen = min(opts.maxlen, length(shape[1]))
    unique_ids = unique(ids)
    stencils = _BettiStencil[]; nonzero_ids = Int[]
    for v in unique_ids
        _betti_may_be_nonzero(M, v) || continue
        stencil = _betti_stencil(shape, v, maxlen)
        stencil === nothing && return nothing
        push!(stencils, stencil); push!(nonzero_ids, v)
    end
    answers = zeros(Int, maxlen+1, length(unique_ids))
    if !isempty(stencils)
        # reduce(vcat, ...) may return the original vector for one stencil;
        # preserve its mask-to-vertex order when sorting the shared labels.
        labels = sort!(unique(reduce(vcat, (s.labels for s in stencils))))
        # Bound the dense induced order used for sparse, separated selections.
        # Broad queries retain the existing resolution route instead of
        # allocating an unbounded square relation matrix.
        length(labels) > 4096 && !(M isa PModule) && labels != 1:FiniteFringe.nvertices(P) && return nothing
        local_module, local_labels = _query_module(M, labels)
        lookup = Dict(labels[i] => local_labels[i] for i in eachindex(labels))
        col = Dict(v => i for (i,v) in enumerate(unique_ids))
        # All source construction is sequential. Map composition is already
        # synchronized by Modules; each worker owns its matrices and output.
        Modules._get_cover_cache(local_module.Q)
        _foreach_workchunk(length(stencils); threads) do indices, _
            for j in indices
                stencil = stencils[j]
                local_ids = [lookup[v] for v in stencil.labels]
                answers[:,col[nonzero_ids[j]]] = _koszul_betti(local_module, stencil, local_ids, maxlen, opts.check)
            end
        end
    end
    columns = Dict(v => i for (i,v) in enumerate(unique_ids))
    return _selected_betti_columns(answers, [columns[v] for v in ids], pad_to)
end

"""
    betti_table(M::Union{PModule,FringeModule}; vertices=nothing,
                opts=ResolutionOptions(), pad_to=nothing, threads=true)

Return finite-poset Betti multiplicities, with homological degrees as rows.
`vertices` selects columns in the supplied order (including repetitions).
On products of chains, selected queries use local Koszul complexes; other
posets use a full resolution. `opts.maxlen` truncates homological degrees,
`opts.check` checks the local chain equations, and `pad_to` pads or truncates
rows after the computation. With no padding, trailing zero rows are removed
within the selected columns; an empty selection has shape `1 x 0`.

Fringe input retains only the selected stalks and their predecessor cubes.
This computes invariants of the finite poset. An ambient polynomial-ring
interpretation requires the grid extension hypotheses, not arbitrary labels.
"""
function betti_table(M::Union{PModule,FiniteFringe.FringeModule}; vertices=nothing,
                     opts::ResolutionOptions=ResolutionOptions(), pad_to::Union{Nothing,Int}=nothing,
                     threads::Bool=(Threads.nthreads() > 1))
    if vertices !== nothing
        result = _selected_betti_table(M, vertices, opts, pad_to; threads)
        result === nothing || return result
    end
    B = betti_table(projective_resolution(M, opts; threads); pad_to)
    return vertices === nothing ? B : _selected_betti_columns(B, _betti_vertices(_query_poset(M), vertices), pad_to)
end
