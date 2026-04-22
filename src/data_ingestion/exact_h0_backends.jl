# Private exact H0 backends for lazy ingestion-owned encodings.

function _lazy_rectangle_qcache_set!(cache::Dict{NTuple{N,Int},Vector{Int}},
                                     order::Vector{NTuple{N,Int}},
                                     key::NTuple{N,Int},
                                     value::Vector{Int}) where {N}
    max_entries = _EXACT_RECTANGLE_QCACHE_MAX[]
    max_entries > 0 || return value
    if !haskey(cache, key)
        if length(order) >= max_entries
            old = popfirst!(order)
            delete!(cache, old)
        end
        push!(order, key)
    end
    cache[key] = value
    return value
end

function _slice_weight_vector(offsets::AbstractVector, offset_weights)
    no = length(offsets)
    if offset_weights === nothing
        return ones(Float64, no)
    elseif offset_weights isa AbstractVector
        length(offset_weights) == no ||
            throw(ArgumentError("slice_barcodes: offset_weights length must equal number of offsets."))
        return Float64[Float64(w) for w in offset_weights]
    elseif offset_weights isa Function
        return Float64[Float64(offset_weights(x0)) for x0 in offsets]
    end
    throw(ArgumentError("slice_barcodes: offset_weights must be nothing, a vector, or a function."))
end

function _supports_exact_slice_barcodes(enc::EncodingResult{PType,MType};
                                        opts::InvariantOptions=InvariantOptions(),
                                        directions=:auto,
                                        offsets=:auto,
                                        normalize_dirs::Symbol=:none,
                                        values=nothing,
                                        packed::Bool=false,
                                        kwargs...) where {PType,MType<:_LazyEncodedModule}
    directions === :auto && return false
    offsets === :auto && return false
    packed && return false
    values === nothing || return false
    normalize_dirs === :none || return false
    opts.box === nothing || opts.box === :auto || return false
    opts.strict === nothing || opts.strict === false || return false

    pi0 = enc.pi isa CompiledEncoding ? enc.pi.pi : enc.pi
    pi0 isa GridEncodingMap || return false
    nd = length(pi0.coords)
    nd in (1, 2) || return false
    pi0.orientation == ntuple(_ -> 1, nd) || return false

    dirs = directions isa AbstractVector ? directions : nothing
    offs = offsets isa AbstractVector ? offsets : nothing
    dirs === nothing && return false
    offs === nothing && return false
    isempty(dirs) && return false
    isempty(offs) && return false

    for dir in dirs
        _normalize_line_direction(dir, nd) === nothing && return false
    end
    for x0 in offs
        _normalize_line_basepoint(x0, nd) === nothing && return false
    end
    L = enc.M.lazy
    enc.M.degree == 0 || return false
    L.multicritical === :union || return false
    grades0 = L.grades_by_dim[1]
    grades0 isa Vector || return false
    for g in grades0
        (g isa Tuple && length(g) == nd && all(x -> x isa Real, g)) || return false
    end
    if length(L.grades_by_dim) >= 2
        grades1 = L.grades_by_dim[2]
        grades1 isa Vector || return false
        for g in grades1
            (g isa Tuple && length(g) == nd && all(x -> x isa Real, g)) || return false
        end
    end
    return _edge_endpoints_from_boundary(length(L.boundaries) >= 1 ? L.boundaries[1] : spzeros(Int, 0, 0)) !== nothing
end

function _exact_slice_barcodes(enc::EncodingResult{PType,MType};
                               opts::InvariantOptions=InvariantOptions(),
                               directions,
                               offsets,
                               direction_weight::Union{Symbol,Function,Real}=:none,
                               offset_weights=nothing,
                               normalize_weights::Bool=true,
                               threads::Bool=(Threads.nthreads() > 1)) where {PType,MType<:_LazyEncodedModule}
    _supports_exact_slice_barcodes(enc;
        opts=opts,
        directions=directions,
        offsets=offsets,
        normalize_dirs=:none,
        values=nothing,
        packed=false) || return nothing

    pi0 = enc.pi isa CompiledEncoding ? enc.pi.pi : enc.pi
    nd = length(pi0.coords)
    dirs = [_normalize_line_direction(dir, nd)::Vector{Float64} for dir in directions]
    offs = [_normalize_line_basepoint(x0, nd)::Vector{Float64} for x0 in offsets]
    ndirs = length(dirs)
    noffs = length(offs)
    bars = Matrix{Dict{Tuple{Float64,Float64},Int}}(undef, ndirs, noffs)
    L = enc.M.lazy

    if threads && Threads.nthreads() > 1 && (ndirs * noffs) > 1
        Threads.@threads for k in 1:(ndirs * noffs)
            i = div(k - 1, noffs) + 1
            j = mod(k - 1, noffs) + 1
            bc = _lazy_h0_line_barcode(L, offs[j], dirs[i])
            bc === nothing && error("slice_barcodes exact line path failed to build a barcode.")
            bars[i, j] = bc
        end
    else
        for i in 1:ndirs, j in 1:noffs
            bc = _lazy_h0_line_barcode(L, offs[j], dirs[i])
            bc === nothing && return nothing
            bars[i, j] = bc
        end
    end

    wdir = Vector{Float64}(undef, ndirs)
    @inbounds for i in 1:ndirs
        wdir[i] = SliceInvariants.direction_weight(dirs[i], direction_weight)
    end
    woff = _slice_weight_vector(offs, offset_weights)
    W = wdir * woff'
    if normalize_weights
        s = sum(W)
        s > 0.0 || throw(ArgumentError("slice_barcodes: total slice weight is zero"))
        W ./= s
    end
    return SliceBarcodesResult(bars, W, dirs, offs)
end

function _supports_exact_rectangle_signed_barcode_lazy(enc::EncodingResult{PType,MType},
                                                       pi0::GridEncodingMap{N};
                                                       opts::InvariantOptions=InvariantOptions(),
                                                       keep_endpoints::Bool=true,
                                                       kwargs...) where {PType,MType<:_LazyEncodedModule,N}
    isempty(kwargs) || return false
    opts.box === nothing || opts.box === :auto || return false
    opts.strict === nothing || opts.strict === false || return false

    N in (1, 2) || return false
    enc.M.degree == 0 || return false

    L = enc.M.lazy
    length(L.orientation) == N || return false
    L.orientation == ntuple(_ -> 1, N) || return false
    pi0.orientation == ntuple(_ -> 1, N) || return false
    L.multicritical === :union || return false

    _, axes_idx = SignedMeasures._rectangle_signed_barcode_grid_axes(
        pi0, opts; keep_endpoints=keep_endpoints
    )
    isempty(axes_idx[1]) && return false
    N == 2 && isempty(axes_idx[2]) && return false

    return _lazy_h0_rectangle_payload(L, Val(N)) !== nothing
end

function _supports_exact_rectangle_signed_barcode(enc::EncodingResult{PType,MType};
                                                  opts::InvariantOptions=InvariantOptions(),
                                                  keep_endpoints::Bool=true,
                                                  kwargs...) where {PType,MType<:_LazyEncodedModule}
    pi0 = enc.pi isa CompiledEncoding ? enc.pi.pi : enc.pi
    pi0 isa GridEncodingMap || return false
    return _supports_exact_rectangle_signed_barcode_lazy(
        enc, pi0; opts=opts, keep_endpoints=keep_endpoints, kwargs...
    )
end

function _supports_exact_rank_query_table(enc::EncodingResult{PType,MType};
                                          opts::InvariantOptions=InvariantOptions(),
                                          keep_endpoints::Bool=true,
                                          threads::Bool=(Threads.nthreads() > 1),
                                          kwargs...) where {PType,MType<:_LazyEncodedModule}
    pi0 = enc.pi isa CompiledEncoding ? enc.pi.pi : enc.pi
    pi0 isa GridEncodingMap || return false
    return _supports_exact_rectangle_signed_barcode_lazy(
        enc, pi0; opts=opts, keep_endpoints=keep_endpoints, kwargs...
    )
end

function _supports_exact_rank_signed_measure(enc::EncodingResult{PType,MType};
                                             opts::InvariantOptions=InvariantOptions(),
                                             keep_endpoints::Bool=true,
                                             threads::Bool=(Threads.nthreads() > 1),
                                             kwargs...) where {PType,MType<:_LazyEncodedModule}
    pi0 = enc.pi isa CompiledEncoding ? enc.pi.pi : enc.pi
    pi0 isa GridEncodingMap || return false
    isempty(kwargs) || return false
    _supports_exact_rectangle_signed_barcode_lazy(
        enc, pi0; opts=opts, keep_endpoints=keep_endpoints
    ) || return false
    payload = _lazy_h0_rectangle_payload(enc.M.lazy, Val(length(pi0.coords)))
    payload === nothing && return false
    vertex_births, edge_endpoints, edge_births = payload
    _, axes_idx = SignedMeasures._rectangle_signed_barcode_grid_axes(
        pi0, opts; keep_endpoints=keep_endpoints
    )
    return _supports_exact_rank_signed_measure_payload(
        vertex_births,
        edge_endpoints,
        edge_births,
        axes_idx,
    )
end

@inline function _full_grid_birth_axes(pi0::GridEncodingMap{N}) where {N}
    return ntuple(i -> Float64[Float64(x) for x in pi0.coords[i]], N)
end

function _full_grid_death_axes(birth_axes::NTuple{N,Vector{Float64}}) where {N}
    return ntuple(i -> begin
        ax = birth_axes[i]
        n = length(ax)
        vals = Vector{Float64}(undef, n)
        @inbounds for j in 1:n
            vals[j] = j < n ? ax[j + 1] : Inf
        end
        vals
    end, N)
end

@inline _full_grid_axes_idx(pi0::GridEncodingMap{N}) where {N} =
    ntuple(i -> collect(1:length(pi0.coords[i])), N)

function _supports_exact_restricted_hilbert(enc::EncodingResult{PType,MType};
                                            opts::InvariantOptions=InvariantOptions(),
                                            threads::Bool=(Threads.nthreads() > 1),
                                            kwargs...) where {PType,MType<:_LazyEncodedModule}
    _ = opts
    _ = threads
    isempty(kwargs) || return false
    pi0 = enc.pi isa CompiledEncoding ? enc.pi.pi : enc.pi
    pi0 isa GridEncodingMap || return false
    N = length(pi0.coords)
    N in (1, 2) || return false
    enc.M.degree == 0 || return false

    L = enc.M.lazy
    length(L.orientation) == N || return false
    L.orientation == ntuple(_ -> 1, N) || return false
    pi0.orientation == ntuple(_ -> 1, N) || return false
    L.multicritical === :union || return false

    payload = _lazy_h0_rectangle_payload(L, Val(N))
    payload === nothing && return false
    vertex_births, edge_endpoints, edge_births = payload
    return _supports_exact_rank_signed_measure_payload(
        vertex_births,
        edge_endpoints,
        edge_births,
        _full_grid_axes_idx(pi0),
    )
end

function _restricted_hilbert_from_exact_rank_measure_1d(inds::AbstractVector,
                                                        wts::AbstractVector,
                                                        n::Int)
    delta = zeros(Int, n + 1)
    @inbounds for i in eachindex(wts)
        b, d = inds[i]
        (1 <= b <= n && 1 <= d <= n) || return nothing
        d < b && continue
        wt = Int(wts[i])
        delta[b] += wt
        d < n && (delta[d + 1] -= wt)
    end
    vals = Vector{Int}(undef, n)
    run = 0
    @inbounds for q in 1:n
        run += delta[q]
        vals[q] = run
    end
    return vals
end

function _restricted_hilbert_from_exact_rank_measure_2d(inds::AbstractVector,
                                                        wts::AbstractVector,
                                                        m1::Int,
                                                        m2::Int)
    delta = zeros(Int, m1 + 1, m2 + 1)
    @inbounds for i in eachindex(wts)
        b1, b2, d1, d2 = inds[i]
        (1 <= b1 <= m1 && 1 <= d1 <= m1 && 1 <= b2 <= m2 && 1 <= d2 <= m2) || return nothing
        (d1 < b1 || d2 < b2) && continue
        wt = Int(wts[i])
        delta[b1, b2] += wt
        d1 < m1 && (delta[d1 + 1, b2] -= wt)
        d2 < m2 && (delta[b1, d2 + 1] -= wt)
        (d1 < m1 && d2 < m2) && (delta[d1 + 1, d2 + 1] += wt)
    end
    vals = Vector{Int}(undef, m1 * m2)
    lin = 1
    @inbounds for q2 in 1:m2
        for q1 in 1:m1
            acc = delta[q1, q2]
            q1 > 1 && (acc += delta[q1 - 1, q2])
            q2 > 1 && (acc += delta[q1, q2 - 1])
            (q1 > 1 && q2 > 1) && (acc -= delta[q1 - 1, q2 - 1])
            delta[q1, q2] = acc
            vals[lin] = acc
            lin += 1
        end
    end
    return vals
end

function _exact_restricted_hilbert(enc::EncodingResult{PType,MType};
                                   opts::InvariantOptions=InvariantOptions(),
                                   threads::Bool=(Threads.nthreads() > 1),
                                   kwargs...) where {PType,MType<:_LazyEncodedModule}
    _ = opts
    isempty(kwargs) || return nothing
    dims0 = enc.M.dims
    dims0 === nothing || return [Int(d) for d in dims0]
    _supports_exact_restricted_hilbert(enc; opts=opts, threads=threads) || return nothing

    pi0 = enc.pi isa CompiledEncoding ? enc.pi.pi : enc.pi
    pi0 isa GridEncodingMap || return nothing
    N = length(pi0.coords)
    payload = _lazy_h0_rectangle_payload(enc.M.lazy, Val(N))
    payload === nothing && return nothing
    vertex_births, edge_endpoints, edge_births = payload
    birth_axes = _full_grid_birth_axes(pi0)
    death_axes = _full_grid_death_axes(birth_axes)
    axes_idx = _full_grid_axes_idx(pi0)

    pm = if N == 1
        _exact_rank_signed_measure_1d_lazy(
            vertex_births,
            edge_endpoints,
            edge_births,
            birth_axes,
            death_axes,
            axes_idx;
            drop_zeros=true,
            tol=0,
        )
    else
        _exact_rank_signed_measure_2d_chain_lazy(
            vertex_births,
            edge_endpoints,
            edge_births,
            birth_axes,
            death_axes,
            axes_idx;
            drop_zeros=true,
            tol=0,
            threads=threads,
        )
    end
    pm === nothing && return nothing

    dims = if N == 1
        _restricted_hilbert_from_exact_rank_measure_1d(pm.inds, pm.wts, length(birth_axes[1]))
    else
        _restricted_hilbert_from_exact_rank_measure_2d(
            pm.inds,
            pm.wts,
            length(birth_axes[1]),
            length(birth_axes[2]),
        )
    end
    dims === nothing && return nothing
    enc.M.dims = dims
    return copy(dims)
end

@inline _rank_coord_token(x::Real) = repr(Float64(x))

@inline function _axis_position_map(axis_idx::Vector{Int})
    return Dict{Int,Int}(axis_idx[i] => i for i in eachindex(axis_idx))
end

@inline function _supports_exact_rank_signed_measure_payload(vertex_births::Vector{NTuple{1,Int}},
                                                             edge_endpoints::Vector{NTuple{2,Int}},
                                                             edge_births::Vector{NTuple{1,Int}},
                                                             axes_idx::NTuple{1,Vector{Int}})
    isempty(vertex_births) && return true
    pos1 = _axis_position_map(axes_idx[1])
    @inbounds for b in vertex_births
        get(pos1, b[1], 0) != 0 || return false
    end
    @inbounds for e in eachindex(edge_endpoints)
        get(pos1, edge_births[e][1], 0) != 0 || return false
    end
    return true
end

@inline function _supports_exact_rank_signed_measure_payload(vertex_births::Vector{NTuple{2,Int}},
                                                             edge_endpoints::Vector{NTuple{2,Int}},
                                                             edge_births::Vector{NTuple{2,Int}},
                                                             axes_idx::NTuple{2,Vector{Int}})
    isempty(vertex_births) && return true
    pos1 = _axis_position_map(axes_idx[1])
    pos2 = _axis_position_map(axes_idx[2])
    x0 = vertex_births[1][1]
    @inbounds for b in vertex_births
        b[1] == x0 || return false
        get(pos1, b[1], 0) != 0 || return false
        get(pos2, b[2], 0) != 0 || return false
    end
    @inbounds for e in eachindex(edge_endpoints)
        get(pos1, edge_births[e][1], 0) != 0 || return false
        get(pos2, edge_births[e][2], 0) != 0 || return false
        a, b = edge_endpoints[e]
        edge_births[e][2] == max(vertex_births[a][2], vertex_births[b][2]) || return false
    end
    return true
end

function _exact_rank_signed_measure_1d_lazy(vertex_births::Vector{NTuple{1,Int}},
                                            edge_endpoints::Vector{NTuple{2,Int}},
                                            edge_births::Vector{NTuple{1,Int}},
                                            birth_axes::NTuple{1,Vector{Float64}},
                                            death_axes::NTuple{1,Vector{Float64}},
                                            axes_idx::NTuple{1,Vector{Int}};
                                            drop_zeros::Bool=true,
                                            tol::Int=0)
    pos1 = _axis_position_map(axes_idx[1])
    nv = length(vertex_births)
    vertex_birth_pos = Vector{Int}(undef, nv)
    @inbounds for v in 1:nv
        p = get(pos1, vertex_births[v][1], 0)
        p != 0 || return nothing
        vertex_birth_pos[v] = p
    end

    ne = length(edge_endpoints)
    edge_birth_pos = Vector{Int}(undef, ne)
    @inbounds for e in 1:ne
        p = get(pos1, edge_births[e][1], 0)
        p != 0 || return nothing
        edge_birth_pos[e] = p
    end
    edge_order = sortperm(edge_birth_pos)

    parent = collect(1:nv)
    sz = ones(Int, nv)
    root_birth = copy(vertex_birth_pos)
    root_rep = collect(1:nv)
    active = falses(nv)
    verts_at_birth = [Int[] for _ in eachindex(axes_idx[1])]
    @inbounds for v in 1:nv
        push!(verts_at_birth[vertex_birth_pos[v]], v)
    end
    weights = Dict{NTuple{2,Int},Int}()
    eptr = 1
    @inbounds for q in eachindex(axes_idx[1])
        for v in verts_at_birth[q]
            active[v] = true
            parent[v] = v
            sz[v] = 1
            root_birth[v] = vertex_birth_pos[v]
            root_rep[v] = v
        end
        while eptr <= ne
            e = edge_order[eptr]
            edge_birth_pos[e] == q || break
            a, b = edge_endpoints[e]
            (active[a] && active[b]) || return nothing
            ra = _uf_find!(parent, a)
            rb = _uf_find!(parent, b)
            if ra != rb
                ba = root_birth[ra]
                bb = root_birth[rb]
                if ba < bb || (ba == bb && root_rep[ra] <= root_rep[rb])
                    survive = ra
                    die = rb
                else
                    survive = rb
                    die = ra
                end
                d1 = q - 1
                if d1 >= 1
                    key = (root_birth[die], d1)
                    weights[key] = get(weights, key, 0) + 1
                end
                parent[die] = survive
                sz[survive] += sz[die]
                root_birth[survive] = min(root_birth[survive], root_birth[die])
                root_rep[survive] = min(root_rep[survive], root_rep[die])
            end
            eptr += 1
        end
    end

    seen = falses(nv)
    d1_inf = length(death_axes[1])
    @inbounds for v in 1:nv
        active[v] || continue
        r = _uf_find!(parent, v)
        seen[r] && continue
        seen[r] = true
        key = (root_birth[r], d1_inf)
        weights[key] = get(weights, key, 0) + 1
    end

    keys_sorted = sort!(collect(keys(weights)))
    inds = NTuple{2,Int}[]
    wts = Int[]
    sizehint!(inds, length(keys_sorted))
    sizehint!(wts, length(keys_sorted))
    @inbounds for key in keys_sorted
        wt = weights[key]
        (drop_zeros && abs(wt) <= tol) && continue
        push!(inds, key)
        push!(wts, wt)
    end
    return PointSignedMeasure((birth_axes[1], death_axes[1]), inds, wts)
end

function _exact_rank_slice_measure_chain2d(vertex_birth2_pos::Vector{Int},
                                           edge_endpoints::Vector{NTuple{2,Int}},
                                           edge_birth1_pos::Vector{Int},
                                           edge_activation2_pos::Vector{Int},
                                           vertex_buckets::Vector{Vector{Int}},
                                           edge_order::Vector{Int},
                                           q2::Int,
                                           death1_inf::Int)
    nv = length(vertex_birth2_pos)
    active = falses(nv)
    parent = collect(1:nv)
    sz = ones(Int, nv)
    root_birth2 = zeros(Int, nv)
    root_rep = collect(1:nv)

    @inbounds for j in 1:q2
        for v in vertex_buckets[j]
            active[v] = true
            parent[v] = v
            sz[v] = 1
            root_birth2[v] = vertex_birth2_pos[v]
            root_rep[v] = v
        end
    end

    weights = Dict{NTuple{2,Int},Int}()
    @inbounds for e in edge_order
        edge_activation2_pos[e] <= q2 || continue
        a, b = edge_endpoints[e]
        active[a] && active[b] || return nothing
        ra = _uf_find!(parent, a)
        rb = _uf_find!(parent, b)
        ra == rb && continue
        ba = root_birth2[ra]
        bb = root_birth2[rb]
        if ba < bb || (ba == bb && root_rep[ra] <= root_rep[rb])
            survive = ra
            die = rb
        else
            survive = rb
            die = ra
        end
        d1 = edge_birth1_pos[e] - 1
        if d1 >= 1
            key = (root_birth2[die], d1)
            weights[key] = get(weights, key, 0) + 1
        end
        parent[die] = survive
        sz[survive] += sz[die]
        root_birth2[survive] = min(root_birth2[survive], root_birth2[die])
        root_rep[survive] = min(root_rep[survive], root_rep[die])
    end

    seen = falses(nv)
    @inbounds for v in 1:nv
        active[v] || continue
        r = _uf_find!(parent, v)
        seen[r] && continue
        seen[r] = true
        key = (root_birth2[r], death1_inf)
        weights[key] = get(weights, key, 0) + 1
    end
    return weights
end

function _exact_rank_signed_measure_2d_chain_lazy(vertex_births::Vector{NTuple{2,Int}},
                                                  edge_endpoints::Vector{NTuple{2,Int}},
                                                  edge_births::Vector{NTuple{2,Int}},
                                                  birth_axes::NTuple{2,Vector{Float64}},
                                                  death_axes::NTuple{2,Vector{Float64}},
                                                  axes_idx::NTuple{2,Vector{Int}};
                                                  drop_zeros::Bool=true,
                                                  tol::Int=0,
                                                  threads::Bool=(Threads.nthreads() > 1))
    pos1 = _axis_position_map(axes_idx[1])
    pos2 = _axis_position_map(axes_idx[2])
    nv = length(vertex_births)
    nv == 0 && return PointSignedMeasure((birth_axes[1], birth_axes[2], death_axes[1], death_axes[2]), NTuple{4,Int}[], Int[])

    vertex_birth1_pos = Vector{Int}(undef, nv)
    vertex_birth2_pos = Vector{Int}(undef, nv)
    @inbounds for v in 1:nv
        p1 = get(pos1, vertex_births[v][1], 0)
        p2 = get(pos2, vertex_births[v][2], 0)
        (p1 != 0 && p2 != 0) || return nothing
        vertex_birth1_pos[v] = p1
        vertex_birth2_pos[v] = p2
    end
    b1 = vertex_birth1_pos[1]
    @inbounds for p1 in vertex_birth1_pos
        p1 == b1 || return nothing
    end

    ne = length(edge_endpoints)
    edge_birth1_pos = Vector{Int}(undef, ne)
    edge_activation2_pos = Vector{Int}(undef, ne)
    @inbounds for e in 1:ne
        p1 = get(pos1, edge_births[e][1], 0)
        p2 = get(pos2, edge_births[e][2], 0)
        (p1 != 0 && p2 != 0) || return nothing
        edge_birth1_pos[e] = p1
        edge_activation2_pos[e] = p2
        a, b = edge_endpoints[e]
        edge_births[e][2] == max(vertex_births[a][2], vertex_births[b][2]) || return nothing
    end

    m2 = length(axes_idx[2])
    vertex_buckets = [Int[] for _ in 1:m2]
    @inbounds for v in 1:nv
        push!(vertex_buckets[vertex_birth2_pos[v]], v)
    end
    edge_order = sortperm(edge_birth1_pos)
    death1_inf = length(death_axes[1])
    slices = Vector{Dict{NTuple{2,Int},Int}}(undef, m2)
    if threads && Threads.nthreads() > 1 && m2 > 1
        Threads.@threads for q2 in 1:m2
            local_terms = _exact_rank_slice_measure_chain2d(
                vertex_birth2_pos,
                edge_endpoints,
                edge_birth1_pos,
                edge_activation2_pos,
                vertex_buckets,
                edge_order,
                q2,
                death1_inf,
            )
            local_terms === nothing && error("direct rank_signed_measure chain slice encountered an inactive endpoint.")
            slices[q2] = local_terms
        end
    else
        for q2 in 1:m2
            local_terms = _exact_rank_slice_measure_chain2d(
                vertex_birth2_pos,
                edge_endpoints,
                edge_birth1_pos,
                edge_activation2_pos,
                vertex_buckets,
                edge_order,
                q2,
                death1_inf,
            )
            local_terms === nothing && return nothing
            slices[q2] = local_terms
        end
    end

    weights = Dict{NTuple{4,Int},Int}()
    @inbounds for q2 in 1:(m2 - 1)
        cur = slices[q2]
        nxt = slices[q2 + 1]
        for (key, wt) in cur
            full_key = (b1, key[1], key[2], q2)
            weights[full_key] = get(weights, full_key, 0) + wt
        end
        for (key, wt) in nxt
            full_key = (b1, key[1], key[2], q2)
            weights[full_key] = get(weights, full_key, 0) - wt
        end
    end
    for (key, wt) in slices[m2]
        full_key = (b1, key[1], key[2], m2)
        weights[full_key] = get(weights, full_key, 0) + wt
    end

    keys_sorted = sort!(collect(keys(weights)))
    inds = NTuple{4,Int}[]
    wts = Int[]
    sizehint!(inds, length(keys_sorted))
    sizehint!(wts, length(keys_sorted))
    @inbounds for key in keys_sorted
        wt = weights[key]
        (drop_zeros && abs(wt) <= tol) && continue
        push!(inds, key)
        push!(wts, wt)
    end
    return PointSignedMeasure((birth_axes[1], birth_axes[2], death_axes[1], death_axes[2]), inds, wts)
end

function _append_exact_rank_entry_1d!(io::IO,
                                      first_ref::Base.RefValue{Bool},
                                      axis::Vector{Float64},
                                      p::Int,
                                      q::Int,
                                      val::Int)
    first_ref[] || write(io, ';')
    first_ref[] = false
    print(io,
          _rank_coord_token(axis[p]), "||",
          _rank_coord_token(axis[q]), "=>",
          val)
    return nothing
end

function _append_exact_rank_entry_2d!(io::IO,
                                      first_ref::Base.RefValue{Bool},
                                      axes::NTuple{2,Vector{Float64}},
                                      p1::Int,
                                      p2::Int,
                                      q1::Int,
                                      q2::Int,
                                      val::Int)
    first_ref[] || write(io, ';')
    first_ref[] = false
    print(io,
          _rank_coord_token(axes[1][p1]), "|",
          _rank_coord_token(axes[2][p2]), "||",
          _rank_coord_token(axes[1][q1]), "|",
          _rank_coord_token(axes[2][q2]), "=>",
          val)
    return nothing
end

@inline function _query_axis_lower_bound(axis::Vector{Int}, birth::Int)
    idx = searchsortedfirst(axis, birth)
    return idx <= length(axis) ? idx : 0
end

function _exact_rank_query_table_1d_canonical(vertex_births::Vector{NTuple{1,Int}},
                                              edge_endpoints::Vector{NTuple{2,Int}},
                                              edge_births::Vector{NTuple{1,Int}},
                                              birth_axis::Vector{Float64},
                                              axis_idx::Vector{Int})
    nv = length(vertex_births)
    m = length(axis_idx)
    vertex_qstart = zeros(Int, nv)
    comp_lb = zeros(Int, nv)
    vertices_at_q = [Int[] for _ in 1:m]
    for v in 1:nv
        lb = _query_axis_lower_bound(axis_idx, vertex_births[v][1])
        vertex_qstart[v] = lb
        comp_lb[v] = lb
        lb == 0 || push!(vertices_at_q[lb], v)
    end

    edges_at_q = [Int[] for _ in 1:m]
    for e in eachindex(edge_births)
        q0 = _query_axis_lower_bound(axis_idx, edge_births[e][1])
        q0 == 0 || push!(edges_at_q[q0], e)
    end

    active = falses(nv)
    parent = collect(1:nv)
    sz = ones(Int, nv)
    seen = zeros(Int, nv)
    freq = zeros(Int, m)
    touched_pos = Int[]
    stamp = 0
    first_ref = Ref(true)
    nnz = 0
    abs_mass = 0
    io = IOBuffer()

    for q in 1:m
        for v in vertices_at_q[q]
            active[v] = true
            parent[v] = v
            sz[v] = 1
        end
        for e in edges_at_q[q]
            a, b = edge_endpoints[e]
            (active[a] && active[b]) || error("exact rank table encountered an active edge before one of its endpoints.")
            ra = _uf_find!(parent, a)
            rb = _uf_find!(parent, b)
            ra == rb && continue
            if sz[ra] < sz[rb]
                ra, rb = rb, ra
            end
            parent[rb] = ra
            sz[ra] += sz[rb]
            lbra = comp_lb[ra]
            lbrb = comp_lb[rb]
            if lbra == 0 || (lbrb != 0 && lbrb < lbra)
                comp_lb[ra] = lbrb
            end
        end

        empty!(touched_pos)
        stamp += 1
        for v in 1:nv
            active[v] || continue
            r = _uf_find!(parent, v)
            seen[r] == stamp && continue
            seen[r] = stamp
            lb = comp_lb[r]
            lb == 0 && continue
            freq[lb] == 0 && push!(touched_pos, lb)
            freq[lb] += 1
        end

        run = 0
        for p in 1:q
            run += freq[p]
            run == 0 && continue
            _append_exact_rank_entry_1d!(io, first_ref, birth_axis, p, q, run)
            nnz += 1
            abs_mass += abs(run)
        end

        for pos in touched_pos
            freq[pos] = 0
        end
    end

    return String(take!(io)), nnz, Float64(abs_mass)
end

function _exact_rank_query_table_2d_row_canonical(q1::Int,
                                                  vertex_births::Vector{NTuple{2,Int}},
                                                  edge_endpoints::Vector{NTuple{2,Int}},
                                                  edge_births::Vector{NTuple{2,Int}},
                                                  birth_axes::NTuple{2,Vector{Float64}},
                                                  axes_idx::NTuple{2,Vector{Int}},
                                                  vertices_by_lb1::Vector{Vector{Int}},
                                                  vertex_lb2::Vector{Int})
    nv = length(vertex_births)
    m2 = length(axes_idx[2])
    min_lb2 = zeros(Int, nv)
    count_by_min_lb2 = zeros(Int, m2)
    touched_roots = Int[]
    touched_pos = Int[]
    first_ref = Ref(true)
    nnz = 0
    abs_mass = 0
    io = IOBuffer()
    q1_enc = axes_idx[1][q1]

    for q2 in 1:m2
        q_enc = (q1_enc, axes_idx[2][q2])
        roots = _lazy_h0_component_roots_at_query(vertex_births, edge_endpoints, edge_births, q_enc)
        roots === nothing && error("exact rank table encountered an active edge before one of its endpoints.")

        for p1 in 1:q1
            for v in vertices_by_lb1[p1]
                r = roots[v]
                r == 0 && continue
                j = vertex_lb2[v]
                (j == 0 || j > q2) && continue
                old = min_lb2[r]
                if old == 0
                    push!(touched_roots, r)
                    count_by_min_lb2[j] == 0 && push!(touched_pos, j)
                    count_by_min_lb2[j] += 1
                    min_lb2[r] = j
                elseif j < old
                    count_by_min_lb2[old] -= 1
                    count_by_min_lb2[j] == 0 && push!(touched_pos, j)
                    count_by_min_lb2[j] += 1
                    min_lb2[r] = j
                end
            end

            run = 0
            for p2 in 1:q2
                run += count_by_min_lb2[p2]
                run == 0 && continue
                _append_exact_rank_entry_2d!(io, first_ref, birth_axes, p1, p2, q1, q2, run)
                nnz += 1
                abs_mass += abs(run)
            end
        end

        for r in touched_roots
            min_lb2[r] = 0
        end
        empty!(touched_roots)
        for pos in touched_pos
            count_by_min_lb2[pos] = 0
        end
        empty!(touched_pos)
    end

    return String(take!(io)), nnz, Float64(abs_mass)
end

function _exact_rank_query_table_2d_canonical(vertex_births::Vector{NTuple{2,Int}},
                                              edge_endpoints::Vector{NTuple{2,Int}},
                                              edge_births::Vector{NTuple{2,Int}},
                                              birth_axes::NTuple{2,Vector{Float64}},
                                              axes_idx::NTuple{2,Vector{Int}};
                                              threads::Bool=(Threads.nthreads() > 1))
    m1 = length(axes_idx[1])
    m2 = length(axes_idx[2])
    nv = length(vertex_births)
    vertex_lb1 = zeros(Int, nv)
    vertex_lb2 = zeros(Int, nv)
    vertices_by_lb1 = [Int[] for _ in 1:m1]
    for v in 1:nv
        lb1 = _query_axis_lower_bound(axes_idx[1], vertex_births[v][1])
        lb2 = _query_axis_lower_bound(axes_idx[2], vertex_births[v][2])
        vertex_lb1[v] = lb1
        vertex_lb2[v] = lb2
        lb1 == 0 || push!(vertices_by_lb1[lb1], v)
    end

    row_payloads = Vector{String}(undef, m1)
    row_nnz = zeros(Int, m1)
    row_abs_mass = zeros(Float64, m1)

    if threads && Threads.nthreads() > 1 && m1 > 1
        Threads.@threads for q1 in 1:m1
            payload, nnz, abs_mass = _exact_rank_query_table_2d_row_canonical(
                q1,
                vertex_births,
                edge_endpoints,
                edge_births,
                birth_axes,
                axes_idx,
                vertices_by_lb1,
                vertex_lb2,
            )
            row_payloads[q1] = payload
            row_nnz[q1] = nnz
            row_abs_mass[q1] = abs_mass
        end
    else
        for q1 in 1:m1
            payload, nnz, abs_mass = _exact_rank_query_table_2d_row_canonical(
                q1,
                vertex_births,
                edge_endpoints,
                edge_births,
                birth_axes,
                axes_idx,
                vertices_by_lb1,
                vertex_lb2,
            )
            row_payloads[q1] = payload
            row_nnz[q1] = nnz
            row_abs_mass[q1] = abs_mass
        end
    end

    return join(filter(!isempty, row_payloads), ";"), sum(row_nnz), sum(row_abs_mass)
end

function _exact_rank_query_table_lazy(enc::EncodingResult{PType,MType},
                                      pi0::GridEncodingMap{N};
                                      opts::InvariantOptions=InvariantOptions(),
                                      keep_endpoints::Bool=true,
                                      threads::Bool=(Threads.nthreads() > 1),
                                      kwargs...) where {PType,MType<:_LazyEncodedModule,N}
    isempty(kwargs) || return nothing
    _supports_exact_rank_query_table(
        enc; opts=opts, keep_endpoints=keep_endpoints
    ) || return nothing

    payload = _lazy_h0_rectangle_payload(enc.M.lazy, Val(N))
    payload === nothing && return nothing
    vertex_births, edge_endpoints, edge_births = payload
    birth_axes, axes_idx = SignedMeasures._rectangle_signed_barcode_grid_axes(
        pi0, opts; keep_endpoints=keep_endpoints
    )

    table_canonical, nnz, abs_mass = if N == 1
        _exact_rank_query_table_1d_canonical(
            vertex_births,
            edge_endpoints,
            edge_births,
            birth_axes[1],
            axes_idx[1],
        )
    else
        _exact_rank_query_table_2d_canonical(
            vertex_births,
            edge_endpoints,
            edge_births,
            birth_axes,
            axes_idx;
            threads=threads,
        )
    end

    return (
        ;
        rank_query_axes=birth_axes,
        rank_table_canonical=table_canonical,
        rank_nonzero_count=nnz,
        rank_abs_mass=abs_mass,
        direct_rank_table=true,
    )
end

function _exact_rank_query_table(enc::EncodingResult{PType,MType};
                                 opts::InvariantOptions=InvariantOptions(),
                                 keep_endpoints::Bool=true,
                                 threads::Bool=(Threads.nthreads() > 1),
                                 kwargs...) where {PType,MType<:_LazyEncodedModule}
    pi0 = enc.pi isa CompiledEncoding ? enc.pi.pi : enc.pi
    pi0 isa GridEncodingMap || return nothing
    return _exact_rank_query_table_lazy(
        enc, pi0;
        opts=opts,
        keep_endpoints=keep_endpoints,
        threads=threads,
        kwargs...,
    )
end

function _exact_rank_signed_measure(enc::EncodingResult{PType,MType};
                                    opts::InvariantOptions=InvariantOptions(),
                                    drop_zeros::Bool=true,
                                    tol::Int=0,
                                    max_span=nothing,
                                    keep_endpoints::Bool=true,
                                    method::Symbol=:auto,
                                    bulk_max_elems::Int=20_000_000,
                                    threads::Bool=(Threads.nthreads() > 1),
                                    kwargs...) where {PType,MType<:_LazyEncodedModule}
    pi0 = enc.pi isa CompiledEncoding ? enc.pi.pi : enc.pi
    pi0 isa GridEncodingMap || return nothing
    _supports_exact_rank_signed_measure(
        enc; opts=opts, keep_endpoints=keep_endpoints, kwargs...
    ) || return nothing
    payload = _lazy_h0_rectangle_payload(enc.M.lazy, Val(length(pi0.coords)))
    payload === nothing && return nothing
    vertex_births, edge_endpoints, edge_births = payload
    birth_axes, axes_idx = SignedMeasures._rectangle_signed_barcode_grid_axes(
        pi0, opts; keep_endpoints=keep_endpoints
    )
    death_axes = SignedMeasures._rectangle_signed_barcode_grid_semantic_axes(
        pi0, opts; keep_endpoints=keep_endpoints
    )[2]

    pm = if length(pi0.coords) == 1
        _exact_rank_signed_measure_1d_lazy(
            vertex_births,
            edge_endpoints,
            edge_births,
            birth_axes,
            death_axes,
            axes_idx;
            drop_zeros=drop_zeros,
            tol=tol,
        )
    else
        _exact_rank_signed_measure_2d_chain_lazy(
            vertex_births,
            edge_endpoints,
            edge_births,
            birth_axes,
            death_axes,
            axes_idx;
            drop_zeros=drop_zeros,
            tol=tol,
            threads=threads,
        )
    end
    pm === nothing && return nothing
    return (
        ;
        axes=pm.axes,
        inds=pm.inds,
        wts=pm.wts,
        direct_rank_measure=true,
    )
end

function _exact_rectangle_signed_barcode_lazy(enc::EncodingResult{PType,MType},
                                              pi0::GridEncodingMap{N};
                                              opts::InvariantOptions=InvariantOptions(),
                                              drop_zeros::Bool=true,
                                              tol::Int=0,
                                              max_span=nothing,
                                              keep_endpoints::Bool=true,
                                              method::Symbol=:auto,
                                              bulk_max_elems::Int=20_000_000,
                                              threads::Bool=(Threads.nthreads() > 1),
                                              kwargs...) where {PType,MType<:_LazyEncodedModule,N}
    isempty(kwargs) || return nothing
    _supports_exact_rectangle_signed_barcode_lazy(
        enc, pi0; opts=opts, keep_endpoints=keep_endpoints
    ) || return nothing

    payload = _lazy_h0_rectangle_payload(enc.M.lazy, Val(N))
    payload === nothing && return nothing
    vertex_births, edge_endpoints, edge_births = payload
    _, axes_idx = SignedMeasures._rectangle_signed_barcode_grid_axes(
        pi0, opts; keep_endpoints=keep_endpoints
    )

    q_cache = Dict{NTuple{N,Int},Vector{Int}}()
    q_order = NTuple{N,Int}[]
    stamps = zeros(Int, length(vertex_births))
    stamp_ref = Ref(0)

    function component_roots(q_enc::NTuple{N,Int})
        cached = get(q_cache, q_enc, nothing)
        cached !== nothing && return cached
        roots = _lazy_h0_component_roots_at_query(vertex_births, edge_endpoints, edge_births, q_enc)
        roots === nothing && error("rectangle_signed_barcode exact path encountered an active edge before one of its endpoints.")
        return _lazy_rectangle_qcache_set!(q_cache, q_order, q_enc, roots)
    end

    function rank_idx(p::NTuple{K,Int}, q::NTuple{K,Int}) where {K}
        p_enc = ntuple(i -> axes_idx[i][p[i]], N)
        q_enc = ntuple(i -> axes_idx[i][q[i]], N)
        roots = component_roots(q_enc)
        return _lazy_h0_rank_from_component_roots(vertex_births, roots, p_enc, stamps, stamp_ref)
    end

    return SignedMeasures.rectangle_signed_barcode(
        rank_idx,
        axes_idx;
        drop_zeros=drop_zeros,
        tol=tol,
        max_span=max_span,
        method=method,
        bulk_max_elems=bulk_max_elems,
        threads=threads,
    )
end

function _exact_rectangle_signed_barcode(enc::EncodingResult{PType,MType};
                                         opts::InvariantOptions=InvariantOptions(),
                                         drop_zeros::Bool=true,
                                         tol::Int=0,
                                         max_span=nothing,
                                         keep_endpoints::Bool=true,
                                         method::Symbol=:auto,
                                         bulk_max_elems::Int=20_000_000,
                                         threads::Bool=(Threads.nthreads() > 1),
                                         kwargs...) where {PType,MType<:_LazyEncodedModule}
    pi0 = enc.pi isa CompiledEncoding ? enc.pi.pi : enc.pi
    pi0 isa GridEncodingMap || return nothing
    return _exact_rectangle_signed_barcode_lazy(
        enc, pi0;
        opts=opts,
        drop_zeros=drop_zeros,
        tol=tol,
        max_span=max_span,
        keep_endpoints=keep_endpoints,
        method=method,
        bulk_max_elems=bulk_max_elems,
        threads=threads,
        kwargs...,
    )
end
