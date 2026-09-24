# algebras.jl -- Yoneda/ext/tor algebra structures

"""
Algebras: multiplicative structures (Yoneda products, Ext/Tor algebras, actions).

This submodule should contain:
- Yoneda product computation
- Ext algebra / Tor algebra structures
- precomputed multiplication tables/caches (when appropriate)
"""
module Algebras
    import ..DerivedFunctors: provenance, _validate_native_derived_options
    using LinearAlgebra
    using SparseArrays

    using ...CoreModules: AbstractCoeffField, RealField, field_from_eltype, coerce, coeff_type
    using ...Options: DerivedFunctorOptions
    using ...Modules: PModule, PMorphism, map_leq, map_leq_many, map_leq_many!, _prepare_map_leq_batch_owned,
                        _accum_map_leq_many_scaled_matvecs!, _accum_map_leq_many_scaled_sourcevec!
    using ...ChainComplexes
    using ...FieldLinAlg: solve_fullcolumn
    using ...FiniteFringe: FinitePoset, FringeModule, cover_edges, nvertices, leq, poset_equal
    using ...IndicatorResolutions: pmodule_from_fringe

    import ..Utils: compose
    import ..Resolutions: ProjectiveResolution, _same_projective_resolution_model
    import ..ExtTorSpaces:
        Ext, Tor,
        ExtSpaceProjective, ExtSpaceInjective,
        TorSpace, TorSpaceSecond,
        representative, coordinates, cycles, boundaries
    import ..Functoriality: _lift_cocycle_to_chainmap_coeff,
                             _precompose_on_hom_cochains_from_projective_coeff
    import ..Functoriality: _tensor_map_on_tor_chains_from_projective_coeff,
                             _FUNCTORIALITY_USE_DIRECT_COEFF_MATVECS,
                             _FUNCTORIALITY_USE_COEFF_PLAN_CACHE,
                             _FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ,
                             _FUNCTORIALITY_DIRECT_COCYCLE_MIN_NNZ,
                             _coeff_nnz, _coeff_plan_scales, _tensor_coeff_plan,
                             _cocycle_coeff_plan

    # Graded-space interface (shared function objects).
    import ..GradedSpaces: degree_range, dim, basis, representative, coordinates, cycles, boundaries

    import ..ExtTorSpaces: _cochain_vector_from_morphism, split_cochain, _Ext_projective
    import ..Resolutions: _same_poset
    import ..DerivedFunctors: nonzero_degrees, generator_degrees, algebra_field,
                              total_dimension, algebra_summary, parent_algebra,
                              element_degree, element_coordinates, check_ext_algebra,
                              check_tor_algebra, underlying_ext_space, underlying_tor_space,
                              cached_product_degrees, _derived_validation_report,
                              _throw_invalid_derived_functor

    @inline function _map_blocks_buffer(M::PModule{K,F,MatT}, n::Int) where {K,F,MatT<:AbstractMatrix{K}}
        return Vector{MatT}(undef, n)
    end

    const _EXT_ACTION_USE_DIRECT_STREAM = Ref(true)
    # Direct streamed Tor action only pays off once the lifted coefficient map has
    # enough nonzeros to amortize plan construction and repeated map application.
    const _EXT_ACTION_DIRECT_STREAM_MIN_NNZ = Ref(24)
    const _EXT_ACTION_DIRECT_STREAM_MIN_EST_WORK = Ref(96)
    const _EXT_ACTION_DIRECT_STREAM_USE_PRODUCT_WORK = Ref(true)

    @inline function _block_span(offsets::AbstractVector{<:Integer}, block_id::Integer)
        i = Int(block_id)
        return Int(offsets[i + 1] - offsets[i])
    end

    @inline function _ext_action_direct_stream_work(plan,
                                                    target_offsets::AbstractVector{<:Integer},
                                                    source_offsets::AbstractVector{<:Integer},
                                                    nrep_cols::Int)
        npairs = length(plan.row_block_ids)
        npairs == 0 && return 0
        if !_EXT_ACTION_DIRECT_STREAM_USE_PRODUCT_WORK[]
            total = 0
            @inbounds for k in 1:npairs
                total += max(_block_span(target_offsets, plan.row_block_ids[k]),
                             _block_span(source_offsets, plan.col_block_ids[k]))
            end
            avgdim = cld(total, npairs)
            return npairs * max(1, nrep_cols) * max(1, avgdim)
        end
        total = 0
        @inbounds for k in 1:npairs
            total += max(1,
                         _block_span(target_offsets, plan.row_block_ids[k]) *
                         _block_span(source_offsets, plan.col_block_ids[k]))
        end
        return max(1, nrep_cols) * total
    end

    @inline function _ext_action_use_direct_stream(plan,
                                                   coeff_nnz::Int,
                                                   target_offsets::AbstractVector{<:Integer},
                                                   source_offsets::AbstractVector{<:Integer},
                                                   nrep_cols::Int)
        _EXT_ACTION_USE_DIRECT_STREAM[] || return false
        coeff_nnz >= max(_EXT_ACTION_DIRECT_STREAM_MIN_NNZ[],
                         _FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[]) || return false
        return _ext_action_direct_stream_work(plan, target_offsets, source_offsets, nrep_cols) >=
               _EXT_ACTION_DIRECT_STREAM_MIN_EST_WORK[]
    end

    # =============================================================================
    # Yoneda product on Ext (projective-resolution model)
    # =============================================================================

    # -----------------------------------------------------------------------------
    # Poset/module compatibility checks (do not assume FinitePoset has a `covers` field)
    # -----------------------------------------------------------------------------

    function _assert_same_pmodule_structure(A::PModule{K}, B::PModule{K}, ctx::String) where {K}
        A.field == B.field || error("$ctx: modules use different coefficient fields.")
        if !_same_poset(A.Q, B.Q)
            error("$ctx: modules live on different posets.")
        end
        if A.dims != B.dims
            error("$ctx: modules have different fiber dimensions.")
        end

        # Compare cover-edge structure maps.
        # Fast path when the poset object is shared: compare store-aligned arrays in lockstep
        # (no tuple keys, no searching, no allocating default zeros).
        if A.Q === B.Q && (A.edge_maps.succs === B.edge_maps.succs)
            storeA = A.edge_maps
            storeB = B.edge_maps
            succs = storeA.succs
            mapsA = storeA.maps_to_succ
            mapsB = storeB.maps_to_succ

            @inbounds for u in 1:nvertices(A.Q)
                su = succs[u]
                Au = mapsA[u]
                Bu = mapsB[u]
                for j in eachindex(su)
                    v = su[j]
                    if Au[j] != Bu[j]
                        error("$ctx: modules have different structure maps on cover edge ($u,$v).")
                    end
                end
            end
        else
            # Safe path for structurally-equal but not pointer-equal posets.
            for (u, v) in cover_edges(A.Q)
                if A.edge_maps[u, v] != B.edge_maps[u, v]
                    error("$ctx: modules have different structure maps on cover edge ($u,$v).")
                end
            end
        end
        return nothing
    end





    # -----------------------------------------------------------------------------
    # Internal: compose a chain-map component into a cocycle for Hom(P_{p+q}(L), N).
    # -----------------------------------------------------------------------------

    function _compose_into_module_cocycle(resL::ProjectiveResolution{K},
                                        resM::ProjectiveResolution{K},
                                        N::PModule{K},
                                        p::Int,
                                        q::Int,
                                        Fp::AbstractMatrix{K},
                                        beta_cocycle::AbstractVector{K},
                                        E_MN::ExtSpaceProjective{K},
                                        E_LN::ExtSpaceProjective{K}) where {K}

        deg = p + q
        dom_bases = resL.gens[deg+1]   # summands in P_{p+q}(L)
        mid_bases = resM.gens[p+1]     # summands in P_p(M)

        # Split beta into its pieces on the summands of P_p(M).
        _, beta_parts = split_cochain(E_MN, p, beta_cocycle)

        offs = E_LN.offsets[deg+1]
        out = zeros(K, offs[end])

        @inline function _accum_scaled_matvec!(block::AbstractVector{K}, A::AbstractMatrix{K},
                                               x::AbstractVector{K}, c::K, tmp::AbstractVector{K}) where {K}
            mul!(tmp, A, x)
            @inbounds for t in eachindex(block)
                block[t] += c * tmp[t]
            end
            return block
        end

        if _FUNCTORIALITY_USE_DIRECT_COEFF_MATVECS[] && _coeff_nnz(Fp) >= _FUNCTORIALITY_DIRECT_COCYCLE_MIN_NNZ[]
            plan = _cocycle_coeff_plan(N, dom_bases, mid_bases, Fp)
            scales = _coeff_plan_scales(Fp, plan)
            _accum_map_leq_many_scaled_matvecs!(out, N, plan.batch,
                                                plan.row_block_ids, plan.col_block_ids,
                                                offs, beta_parts, scales)
            return out
        end

        # Fallback path for A/B parity: same mathematics, but it still materializes
        # the map blocks eagerly.
        # If Fp is sparse, iterate only over nonzeros in each column.
        if issparse(Fp)
            Fps = sparse(Fp)  # ensure SparseMatrixCSC so colptr/rowval/nzval exist
            pairs = Tuple{Int,Int}[]
            ptr_slots = Int[]
            sizehint!(pairs, length(Fps.nzval))
            sizehint!(ptr_slots, length(Fps.nzval))
            for i in 1:length(dom_bases)
                u = dom_bases[i]
                for ptr in Fps.colptr[i]:(Fps.colptr[i + 1] - 1)
                    c = Fps.nzval[ptr]
                    iszero(c) && continue
                    j = Fps.rowval[ptr]
                    v = mid_bases[j]
                    leq(resL.M.Q, v, u) || continue
                    push!(pairs, (v, u))
                    push!(ptr_slots, ptr)
                end
            end
            pair_batch = _prepare_map_leq_batch_owned(pairs)
            map_blocks = _map_blocks_buffer(N, length(pairs))
            map_leq_many!(map_blocks, N, pair_batch)
            map_idx_by_ptr = zeros(Int, length(Fps.nzval))
            @inbounds for idx in eachindex(ptr_slots)
                map_idx_by_ptr[ptr_slots[idx]] = idx
            end

            for i in 1:length(dom_bases)
                u = dom_bases[i]
                block = zeros(K, N.dims[u])
                tmp = similar(block)

                # Nonzeros in column i live in ptr range [colptr[i], colptr[i+1)-1].
                ptr_lo = Fps.colptr[i]
                ptr_hi = Fps.colptr[i+1] - 1

                for ptr in ptr_lo:ptr_hi
                    j = Fps.rowval[ptr]
                    c = Fps.nzval[ptr]
                    # c is guaranteed nonzero in SparseMatrixCSC, but keep it defensive.
                    if iszero(c)
                        continue
                    end

                    map_idx = map_idx_by_ptr[ptr]
                    if map_idx == 0
                        continue
                    end

                    A = map_blocks[map_idx]        # N_v -> N_u
                    _accum_scaled_matvec!(block, A, beta_parts[j], c, tmp)
                end

                out[(offs[i]+1):offs[i+1]] = block
            end

            return out
        end

        # Dense fallback: original behavior.
        pairs = Tuple{Int,Int}[]
        sizehint!(pairs, length(dom_bases) * length(mid_bases))
        map_idx = zeros(Int, length(mid_bases), length(dom_bases))
        for i in 1:length(dom_bases)
            u = dom_bases[i]
            for j in 1:length(mid_bases)
                c = Fp[j, i]
                iszero(c) && continue
                v = mid_bases[j]
                leq(resL.M.Q, v, u) || continue
                push!(pairs, (v, u))
                map_idx[j, i] = length(pairs)
            end
        end
        pair_batch = _prepare_map_leq_batch_owned(pairs)
        map_blocks = _map_blocks_buffer(N, length(pairs))
        map_leq_many!(map_blocks, N, pair_batch)

        for i in 1:length(dom_bases)
            u = dom_bases[i]
            block = zeros(K, N.dims[u])
            tmp = similar(block)

            for j in 1:length(mid_bases)
                idx = map_idx[j, i]
                if idx == 0
                    continue
                end

                c = Fp[j, i]
                A = map_blocks[idx]
                _accum_scaled_matvec!(block, A, beta_parts[j], c, tmp)
            end

            out[(offs[i]+1):offs[i+1]] = block
        end

        return out
    end


    # -----------------------------------------------------------------------------
    # Public API: Yoneda product
    # -----------------------------------------------------------------------------

    """
        yoneda_product(E_MN, p, beta_coords, E_LM, q, alpha_coords; ELN=nothing, return_cocycle=false)

    Compute the Yoneda product

        Ext^p(M, N) x Ext^q(L, M) -> Ext^{p+q}(L, N).

    Inputs:
    - `E_MN` is an `ExtSpaceProjective` for (M,N).
    - `E_LM` is an `ExtSpaceProjective` for (L,M).
    - `beta_coords` are coordinates of a class in Ext^p(M,N) in the basis used by `E_MN`.
    - `alpha_coords` are coordinates of a class in Ext^q(L,M) in the basis used by `E_LM`.

    Output:
    - `(E_LN, coords)` where `coords` are coordinates of the product class in Ext^{p+q}(L,N)
    in the basis used by `E_LN`.
    - If `return_cocycle=true`, returns `(E_LN, coords, cocycle)` where `cocycle` is an explicit
    representative in the cochain space Hom(P_{p+q}(L), N).

    Cheap-first workflow
    - Use `ExtAlgebra` and homogeneous element multiplication when you plan to multiply many
      classes in the same module.
    - Set `return_cocycle=true` only when you actually need an explicit chain-level
      representative of the product.

    Notes for mathematicians:
    - This implements the classical Yoneda product by constructing a comparison map
    between projective resolutions and composing at the chain level.
    - The result is well-defined in cohomology; chain-level representatives depend on
    deterministic but non-canonical lift choices (as always).
    - If `ELN` uses another projective resolution of L, an identity comparison map
      transports the product into that resolution before coordinates are taken.
      The optional returned cocycle belongs to the returned `E_LN` model.

    Technical requirements:
    - `E_MN` must have `tmax >= p`.
    - `E_LM` must have `tmax >= p+q` (because we need P_{p+q}(L)).
    - The "middle" module M used by `E_MN.res` and the second argument of `E_LM` must agree
    as poset-modules (same fibers and structure maps).
    """
    function yoneda_product(E_MN::ExtSpaceProjective{K},
                            p::Int,
                            beta_coords::AbstractVector{K},
                            E_LM::ExtSpaceProjective{K},
                            q::Int,
                            alpha_coords::AbstractVector{K};
                            ELN::Union{Nothing,ExtSpaceProjective{K}}=nothing,
                            return_cocycle::Bool=false) where {K}

        if p < 0 || q < 0
            error("yoneda_product: degrees p and q must be >= 0.")
        end
        if p > E_MN.tmax
            error("yoneda_product: E_MN.tmax is too small for p = $p.")
        end
        if (p + q) > E_LM.tmax
            error("yoneda_product: E_LM.tmax is too small for p+q = $(p+q).")
        end

        resM = E_MN.res
        resL = E_LM.res
        N = E_MN.N

        # Compatibility: the middle module in Ext^q(L,M) must match the resolved module M.
        _assert_same_pmodule_structure(E_LM.N, resM.M, "yoneda_product (middle module check)")

        # Build (or validate) the target Ext space Ext(L,N).
        if ELN === nothing
            ELN_use = Ext(resL, N)
        else
            ELN_use = ELN
            # Very conservative checks: same resolved L and same N.
            _assert_same_pmodule_structure(ELN_use.N, N, "yoneda_product (target N check)")
            _assert_same_pmodule_structure(ELN_use.res.M, resL.M, "yoneda_product (target L check)")
            if ELN_use.tmax < (p + q)
                error("yoneda_product: provided ELN has tmax < p+q.")
            end
        end

        # Convert coordinates to explicit cocycles.
        beta_cocycle  = representative(E_MN, p, beta_coords)
        alpha_cocycle = reshape(representative(E_LM, q, alpha_coords), :, 1)

        # Lift alpha to a degree-q chain map into the projective resolution of M, up to component p.
        F = _lift_cocycle_to_chainmap_coeff(resL, resM, E_LM, q, alpha_cocycle; upto=p)
        Fp = F[p+1]  # P_{p+q}(L) -> P_p(M)

        # The composition is initially a cochain on resL, even if the requested
        # target uses a differently based or nonminimal resolution of L.
        same_model = _same_projective_resolution_model(resL, ELN_use.res)
        source_model = same_model ? ELN_use : Ext(resL, N; maxdeg=p+q)
        cocycle = _compose_into_module_cocycle(resL, resM, N, p, q, Fp, beta_cocycle, E_MN, source_model)
        if !same_model
            # Ext is contravariant in the resolved module: lift id_L from the
            # requested target resolution to resL, then precompose the cocycle.
            # Encode the augmentation as id_L in degree zero. The cocycle
            # lift also supports equal indexed posets stored as distinct
            # objects, whereas a public PMorphism requires poset identity.
            identity_model = Ext(ELN_use.res, ELN_use.res.M; maxdeg=0)
            identity_cocycle = _cochain_vector_from_morphism(
                identity_model, 0, ELN_use.res.aug)
            comparison = _lift_cocycle_to_chainmap_coeff(
                ELN_use.res, resL, identity_model, 0, identity_cocycle; upto=p+q)
            transfer = _precompose_on_hom_cochains_from_projective_coeff(
                N, ELN_use.res.gens[p+q+1], resL.gens[p+q+1],
                ELN_use.offsets[p+q+1], source_model.offsets[p+q+1],
                comparison[p+q+1])
            cocycle = transfer * cocycle
        end

        coords = coordinates(ELN_use, p+q, cocycle)

        if return_cocycle
            return (ELN_use, coords, cocycle)
        else
            return (ELN_use, coords)
        end
    end

    # =============================================================================
    # Ext algebra: Ext^*(M,M) with cached Yoneda multiplication
    # =============================================================================

    """
        ExtAlgebra(M::PModule{K}; maxdeg::Int=3) -> ExtAlgebra{K}
        ExtAlgebra(M::FringeModule{K}; maxdeg::Int=3) -> ExtAlgebra{K}

    Construct the truncated graded Ext algebra Ext^*(M,M) up to degree `maxdeg`,
    with multiplication given by the Yoneda product.

    This wrapper is intentionally "mathematician-facing":

    - It chooses (once) a projective resolution and Ext bases via `Ext(M,M; maxdeg=...)`.
    - It exposes homogeneous elements as `ExtElement` objects.
    - It supports `*` for Ext multiplication and caches the structure constants.

    Caching model (key point):
    For each bidegree (p,q) with p+q <= tmax, we cache a matrix

        MU[p,q] : Ext^p(M,M) x Ext^q(M,M) -> Ext^{p+q}(M,M)

    in coordinate bases as a linear map on the Kronecker product coordinates.

    Column convention:
    If dim_p = dim Ext^p and dim_q = dim Ext^q, we index basis pairs (i,j) by

        col = (i-1)*dim_q + j

    and we use Julia's `kron(x, y)` to build the vector of coefficients x_i*y_j
    in the same ordering.  This makes multiplication a single matrix-vector product:

        coords(x * y) = MU[p,q] * kron(coords(x), coords(y)).

    Truncation:
    The product is only defined when deg(x) + deg(y) <= A.tmax.
    """
    mutable struct ExtAlgebra{K}
        lock::ReentrantLock
        E::ExtSpaceProjective{K}
        mult_cache::Dict{Tuple{Int,Int}, Matrix{K}}
        unit_coords::Union{Nothing, Vector{K}}
        tmin::Int
        tmax::Int
    end

    """
        degree_range(A::ExtAlgebra) -> UnitRange{Int}
    """
    degree_range(A::ExtAlgebra) = A.tmin:A.tmax

    @inline nonzero_degrees(A::ExtAlgebra) = [t for t in degree_range(A) if dim(A, t) != 0]

    @inline function generator_degrees(A::ExtAlgebra)
        out = Int[]
        for t in degree_range(A)
            append!(out, fill(t, dim(A, t)))
        end
        return out
    end

    @inline algebra_field(A::ExtAlgebra) = A.E.M.field
    @inline total_dimension(A::ExtAlgebra) = sum(dim(A, t) for t in degree_range(A))

    """
        algebra_summary(A::ExtAlgebra) -> NamedTuple

    Cheap-first summary of an `ExtAlgebra`.

    This reports the graded support, total graded dimension, and current
    multiplication-cache state without forcing any new Yoneda products.

    Use this before asking for `basis(A, t)`, products of homogeneous elements,
    or explicit `ExtElement`s when you are exploring an algebra interactively.
    """
    @inline function algebra_summary(A::ExtAlgebra)
        return (
            kind=:ext_algebra,
            provenance=provenance(A),
            field=algebra_field(A),
            degree_range=degree_range(A),
            nonzero_degrees=Tuple(nonzero_degrees(A)),
            total_dimension=total_dimension(A),
            cached_products=lock(() -> length(A.mult_cache), A.lock),
        )
    end

    """
        underlying_ext_space(A::ExtAlgebra)

    Return the underlying graded Ext-space model carried by `A`.

    This is the owner-level accessor to the canonical additive model used for
    coordinates, bases, and representatives. Use it only when you need the full
    graded-space interface; for ordinary inspection start with
    `algebra_summary(A)`.
    """
    @inline underlying_ext_space(A::ExtAlgebra) = A.E

    """
        cached_product_degrees(A::ExtAlgebra)

    Return the currently cached Yoneda-product degree pairs for `A`.

    The result is cheap and reflects only products already computed or
    precomputed. It does not trigger new multiplication work.
    """
    @inline cached_product_degrees(A::ExtAlgebra) = lock(() -> sort!(collect(keys(A.mult_cache))), A.lock)

    function Base.show(io::IO, A::ExtAlgebra)
        d = algebra_summary(A)
        print(io, "ExtAlgebra(nonzero_degrees=", repr(d.nonzero_degrees),
              ", cached_products=", d.cached_products, ")")
    end

    function Base.show(io::IO, ::MIME"text/plain", A::ExtAlgebra)
        d = algebra_summary(A)
        print(io, "ExtAlgebra",
              "\n  field: ", d.field,
              "\n  degree_range: ", repr(d.degree_range),
              "\n  nonzero_degrees: ", repr(d.nonzero_degrees),
              "\n  total_dimension: ", d.total_dimension,
              "\n  cached_product_degrees: ", repr(cached_product_degrees(A)))
    end

    """
        representative(A::ExtAlgebra, t::Int, coords::AbstractVector; model=:canonical)

    Delegate to the underlying ExtSpaceProjective model stored in `A.E`.
    """
    function representative(A::ExtAlgebra, t::Int, coords::AbstractVector; model::Symbol = :canonical)
        return representative(A.E, t, coords; model=model)
    end

    """
        coordinates(A::ExtAlgebra, t::Int, cocycle; model=:canonical)

    Delegate to the underlying ExtSpaceProjective model stored in `A.E`.
    """
    function coordinates(A::ExtAlgebra, t::Int, cocycle; model::Symbol = :canonical)
        return coordinates(A.E, t, cocycle; model=model)
    end


    """
        ExtElement(A::ExtAlgebra{K}, deg::Int, coords::Vector{K})

    A homogeneous element of Ext^deg(M,M), expressed in the basis chosen by `A.E`.

    This is deliberately lightweight: it is just (algebra handle, degree, coordinate vector).
    Use:
    - `element(A, deg, coords)` to construct,
    - `basis(A, deg)` or `A[deg, i]` for basis elements,
    - multiplication via `*`.
    """
    struct ExtElement{K}
        A::ExtAlgebra{K}
        deg::Int
        coords::Vector{K}
    end

    @inline function _element_summary(kind::Symbol, x)
        coords = element_coordinates(x)
        return (
            kind=kind,
            provenance=provenance(x),
            field=algebra_field(parent_algebra(x)),
            degree=element_degree(x),
            coordinate_length=length(coords),
            nonzero_coordinates=count(y -> !iszero(y), coords),
            is_zero=all(iszero, coords),
        )
    end

    """
        parent_algebra(x::ExtElement)

    Return the `ExtAlgebra` that owns `x`.
    """
    @inline parent_algebra(x::ExtElement) = x.A

    """
        element_degree(x::ExtElement)

    Return the homogeneous cohomological degree of `x`.
    """
    @inline element_degree(x::ExtElement) = x.deg

    """
        element_coordinates(x::ExtElement)

    Return the coordinate vector of `x` in the basis carried by its parent
    `ExtAlgebra`.

    This is the semantic accessor preferred over direct field access in user
    code. It is equivalent to `coordinates(x)`.
    """
    @inline element_coordinates(x::ExtElement) = x.coords

    function Base.show(io::IO, x::ExtElement)
        d = _element_summary(:ext_element, x)
        print(io, "ExtElement(degree=", d.degree,
              ", nnz=", d.nonzero_coordinates, ")")
    end

    function Base.show(io::IO, ::MIME"text/plain", x::ExtElement)
        d = _element_summary(:ext_element, x)
        print(io, "ExtElement",
              "\n  field: ", d.field,
              "\n  degree: ", d.degree,
              "\n  coordinate_length: ", d.coordinate_length,
              "\n  nonzero_coordinates: ", d.nonzero_coordinates,
              "\n  is_zero: ", d.is_zero)
    end

    # ----------------------------
    # Construction
    # ----------------------------

    """
        ExtAlgebra(M, df::DerivedFunctorOptions) -> ExtAlgebra{K}
        ExtAlgebra(M::FringeModule{K}, df::DerivedFunctorOptions) -> ExtAlgebra{K}

    Construct the (truncated) graded Ext algebra Ext^*(M,M) up to degree df.maxdeg.

    Internally this chooses (once) a projective resolution and computes Ext using the
    projective-resolution model. Multiplication is via Yoneda products.

    The returned ExtAlgebra caches multiplication matrices so repeated products are fast.
    Concurrent queries may compute a missing product independently; only complete
    matrices are published, and subsequent queries share the published result.

    Cheap-first workflow
    - Start with `algebra_summary(A)`, `nonzero_degrees(A)`, and
      `generator_degrees(A)`.
    - Ask for `basis(A, t)`, products of homogeneous elements, or explicit
      `ExtElement`s only when you need multiplicative coordinates.
    """
    function ExtAlgebra(M::PModule{K}, df::DerivedFunctorOptions) where {K}
        _validate_native_derived_options(df, "ExtAlgebra", :projective)
        E = _Ext_projective(M, M; maxdeg=df.maxdeg)
        return ExtAlgebra{K}(ReentrantLock(), E, Dict{Tuple{Int,Int}, Matrix{K}}(), nothing, E.tmin, E.tmax)
    end

    function ExtAlgebra(M::FringeModule{K}, df::DerivedFunctorOptions) where {K}
        return ExtAlgebra(pmodule_from_fringe(M), df)
    end




    # ----------------------------
    # Basic queries and constructors for elements
    # ----------------------------

    "Dimension of Ext^deg(M,M) in the basis chosen by the underlying Ext space."
    dim(A::ExtAlgebra{K}, deg::Int) where {K} = dim(A.E, deg)

    """
        element(A::ExtAlgebra{K}, deg::Int, coords::AbstractVector{K}) -> ExtElement{K}

    Construct a homogeneous Ext element in degree `deg` with the given coordinate vector.
    """
    function element(A::ExtAlgebra{K}, deg::Int, coords::AbstractVector{K}) where {K}
        if deg < 0 || deg > A.tmax
            error("element: degree must satisfy 0 <= deg <= tmax.")
        end
        d = dim(A, deg)
        if length(coords) != d
            error("element: expected coordinate vector of length $d in degree $deg, got length $(length(coords)).")
        end
        return ExtElement{K}(A, deg, Vector{K}(coords))
    end


    """
        basis(A::ExtAlgebra{K}, deg::Int) -> Vector{ExtElement{K}}

    Return the standard coordinate basis of Ext^deg(M,M) as ExtElement objects.
    """
    function basis(A::ExtAlgebra{K}, deg::Int) where {K}
        d = dim(A, deg)
        out = Vector{ExtElement{K}}(undef, d)
        for i in 1:d
            c = zeros(K, d)
            c[i] = one(K)
            out[i] = ExtElement{K}(A, deg, c)
        end
        return out
    end


    """
        A[deg, i] -> ExtElement

    Indexing convenience: the i-th basis element in degree `deg`.
    """
    function Base.getindex(A::ExtAlgebra{K}, deg::Int, i::Int) where {K}
        d = dim(A, deg)
        if i < 1 || i > d
            error("ExtAlgebra getindex: basis index i=$i out of range 1:$d in degree $deg.")
        end
        c = zeros(K, d)
        c[i] = one(K)
        return ExtElement{K}(A, deg, c)
    end


    "Return the coordinate vector of an ExtElement."
    coordinates(x::ExtElement{K}) where {K} = element_coordinates(x)

    """
        representative(x::ExtElement{K}) -> Vector{K}

    Return a cocycle representative (in the internal cochain model) for the Ext class.
    This is useful for debugging and for users who want explicit representatives.
    """
    representative(x::ExtElement{K}) where {K} = representative(x.A.E, x.deg, x.coords)


    # ----------------------------
    # Linear structure on ExtElement
    # ----------------------------

    function _assert_same_algebra(x::ExtElement{K}, y::ExtElement{K}, ctx::String) where {K}
        if x.A !== y.A
            error("$ctx: elements live in different ExtAlgebra objects.")
        end
        if x.deg != y.deg
            error("$ctx: degrees differ (deg(x)=$(x.deg), deg(y)=$(y.deg)).")
        end
        return nothing
    end

    Base.:+(x::ExtElement{K}, y::ExtElement{K}) where {K} = (_assert_same_algebra(x, y, "ExtElement +");
                                                    ExtElement{K}(x.A, x.deg, x.coords + y.coords))

    Base.:-(x::ExtElement{K}, y::ExtElement{K}) where {K} = (_assert_same_algebra(x, y, "ExtElement -");
                                                    ExtElement{K}(x.A, x.deg, x.coords - y.coords))

    Base.:-(x::ExtElement{K}) where {K} = ExtElement{K}(x.A, x.deg, -x.coords)

    Base.:*(c::K, x::ExtElement{K}) where {K} =
        ExtElement{K}(x.A, x.deg, c .* x.coords)
    Base.:*(x::ExtElement{K}, c::K) where {K} = c * x

    Base.:*(c::Integer, x::ExtElement{K}) where {K} =
        coerce(x.A.E.M.field, c) * x
    Base.:*(x::ExtElement{K}, c::Integer) where {K} = c * x

    Base.iszero(x::ExtElement{K}) where {K} = all(x.coords .== 0)


    # ----------------------------
    # Unit element in Ext^0(M,M)
    # ----------------------------


    """
        unit(A::ExtAlgebra{K}) -> ExtElement{K}

    Return the multiplicative identity in Ext^0(M,M).

    Mathematically, Ext^0(M,M) = Hom(M,M), and the unit is id_M.
    In the projective-resolution model
        ... -> P_1 -> P_0 -> M -> 0,
    the inclusion Hom(M,M) -> Hom(P_0,M) sends id_M to the augmentation map P_0 -> M.
    That augmentation is a cocycle in C^0 and represents the unit class in H^0.
    """
    function unit(A::ExtAlgebra{K}) where {K}
        coords = lock(() -> A.unit_coords, A.lock)
        if coords === nothing
            computed = if dim(A, 0) == 0
                # Zero module edge case: Ext^0(0,0) is 0 as a vector space.
                zeros(K, 0)
            else
                cocycle = _cochain_vector_from_morphism(A.E, 0, A.E.res.aug)
                coordinates(A.E, 0, cocycle)
            end
            coords = lock(A.lock) do
                A.unit_coords === nothing && (A.unit_coords = computed)
                A.unit_coords
            end
        end
        return ExtElement{K}(A, 0, copy(coords))
    end

    Base.one(A::ExtAlgebra{K}) where {K} = unit(A)

    """
        check_ext_algebra(A; throw=false) -> NamedTuple

    Validate the cheap structural contracts of an `ExtAlgebra`.

    This checks cached multiplication matrix sizes, unit-coordinate dimensions,
    and graded dimension bookkeeping without forcing any new multiplication work.
    """
    function check_ext_algebra(A::ExtAlgebra{K}; throw::Bool=false) where {K}
        return lock(A.lock) do
            issues = String[]
            for ((p, q), MU) in A.mult_cache
                expected = (dim(A, p + q), dim(A, p) * dim(A, q))
                size(MU) == expected ||
                    push!(issues, "cached multiplication matrix for degrees ($p,$q) has size $(size(MU)) but expected $expected.")
            end
            if A.unit_coords !== nothing
                length(A.unit_coords) == dim(A, 0) ||
                    push!(issues, "unit coordinates must have length $(dim(A, 0)).")
            end
            length(generator_degrees(A)) == total_dimension(A) ||
                push!(issues, "generator degree list must match the total graded dimension.")
            report = _derived_validation_report(
                :ext_algebra,
                isempty(issues);
                field=algebra_field(A),
                degree_range=degree_range(A),
                total_dimension=total_dimension(A),
                cached_products=length(A.mult_cache),
                issues=issues,
            )
            throw && !report.valid && _throw_invalid_derived_functor(:check_ext_algebra, issues)
            return report
        end
    end

    # ----------------------------
    # Cached multiplication: ExtElement * ExtElement
    # ----------------------------

    # Ensure the multiplication matrix MU[p,q] is present in the cache.
    function _ensure_mult_cache!(A::ExtAlgebra{K}, p::Int, q::Int) where {K}
        key = (p, q)
        cached = lock(() -> get(A.mult_cache, key, nothing), A.lock)
        cached === nothing || return cached

        if p < 0 || q < 0
            error("_ensure_mult_cache!: degrees must be nonnegative.")
        end
        if p + q > A.tmax
            error("_ensure_mult_cache!: requested product degree p+q=$(p+q) exceeds truncation tmax=$(A.tmax).")
        end

        dp = dim(A, p)
        dq = dim(A, q)
        dr = dim(A, p + q)

        MU = zeros(K, dr, dp * dq)

        # Cache even the trivial cases so repeated calls are O(1).
        if dp == 0 || dq == 0 || dr == 0
            return lock(() -> get!(A.mult_cache, key, MU), A.lock)
        end

        # Precompute all products of basis elements e_i in Ext^p and e_j in Ext^q.
        # Each product is computed by the trusted "mathematical core" `yoneda_product`,
        # then stored as a column of MU in the kron(x,y) ordering.
        ei = zeros(K, dp)
        ej = zeros(K, dq)

        for i in 1:dp
            fill!(ei, zero(K))
            ei[i] = one(K)
            for j in 1:dq
                fill!(ej, zero(K))
                ej[j] = one(K)

                # Multiply e_i (degree p) by e_j (degree q) in Ext(M,M).
                _, coords = yoneda_product(A.E, p, ei, A.E, q, ej; ELN=A.E)

                MU[:, (i - 1) * dq + j] = coords
            end
        end

        return lock(() -> get!(A.mult_cache, key, MU), A.lock)
    end


    """
        multiply(A::ExtAlgebra{K}, p::Int, x::AbstractVector{K}, q::Int, y::AbstractVector{K}) -> Vector{K}

    Multiply two homogeneous elements given by coordinate vectors x in Ext^p and y in Ext^q.
    Returns the coordinate vector in Ext^{p+q}.
    """
    function multiply(A::ExtAlgebra{K}, p::Int, x::AbstractVector{K}, q::Int, y::AbstractVector{K}) where {K}
        dp = dim(A, p)
        dq = dim(A, q)
        if length(x) != dp
            error("multiply: left coordinate vector has length $(length(x)) but dim Ext^$p = $dp.")
        end
        if length(y) != dq
            error("multiply: right coordinate vector has length $(length(y)) but dim Ext^$q = $dq.")
        end

        MU = _ensure_mult_cache!(A, p, q)

        # kron(x,y) uses exactly the ordering we used for MU columns.
        v = kron(Vector{K}(x), Vector{K}(y))
        out = MU * v
        return Vector{K}(out)
    end


    """
        precompute!(A::ExtAlgebra{K}) -> ExtAlgebra{K}

    Eagerly compute and cache all multiplication matrices MU[p,q] with p+q <= A.tmax.
    This is optional. Most users will rely on lazy caching via `*`.
    """
    function precompute!(A::ExtAlgebra{K}) where {K}
        for p in 0:A.tmax
            for q in 0:(A.tmax - p)
                _ensure_mult_cache!(A, p, q)
            end
        end
        return A
    end


    # The user-facing multiplication on homogeneous Ext elements.
    function Base.:*(x::ExtElement{K}, y::ExtElement{K}) where {K}
        if x.A !== y.A
            error("ExtElement *: elements live in different ExtAlgebra objects.")
        end
        A = x.A
        p = x.deg
        q = y.deg
        if p + q > A.tmax
            error("ExtElement *: degree p+q=$(p+q) exceeds truncation tmax=$(A.tmax).")
        end
        coords = multiply(A, p, x.coords, q, y.coords)
        return ExtElement{K}(A, p + q, coords)
    end

    # ----------------------------------------------------------------------
    # Ext action on Tor (cap/action flavor)
    # ----------------------------------------------------------------------

    """
        ext_action_on_tor(A, T, x; s)

    Given:
    - `A::ExtAlgebra` for a module L (so A computes Ext^*(L,L)),
    - `T::TorSpaceSecond` computing Tor_*(Rop, L) using the *same* projective resolution of L,
    - an `ExtElement` x in degree m,

    return the induced degree-lowering action matrix:
        x cap - : Tor_s(Rop, L) -> Tor_{s-m}(Rop, L)

    in the homology bases of `T`.

    Notes:
    - This is implemented by lifting a cocycle representative of x to a chain map of the resolution
    (via `_lift_cocycle_to_chainmap_coeff`) and then tensoring that chain map with Rop.
    - For s < m, the target degree is negative, so the action is the zero map.
    - The source degree must be computed by `T`, and `x` must belong to `A`.
      The resolution models must agree in their generator, differential and
      augmentation coordinates; construct `T` with `res=A.E.res` to share them.
    - With this Yoneda convention, `action(x*y, s)` equals
      `action(x, s-deg(y)) * action(y, s)` whenever all source degrees exist.
      The Ext unit acts as the identity. No Tor multiplication is required.
    """
    function ext_action_on_tor(A::ExtAlgebra{K}, T::TorSpaceSecond{K}, x::ExtElement{K}; s::Int) where {K}
        x.A === A || throw(ArgumentError("ext_action_on_tor: x must belong to the supplied Ext algebra."))
        s in degree_range(T) || throw(ArgumentError(
            "ext_action_on_tor: source degree s=$s is outside $(degree_range(T))."))
        m = x.deg
        m in degree_range(A) || throw(ArgumentError(
            "ext_action_on_tor: Ext degree m=$m is outside $(degree_range(A))."))
        length(x.coords) == dim(A, m) || throw(DimensionMismatch(
            "ext_action_on_tor: expected $(dim(A, m)) Ext coordinates, got $(length(x.coords))."))
        _same_projective_resolution_model(A.E.res, T.resL) || throw(ArgumentError(
            "ext_action_on_tor: the resolution models differ; construct Tor with res=A.E.res."))
        if s < m
            return zeros(K, 0, dim(T, s))
        end

        # Choose a cocycle representative alpha in cochain degree m.
        alpha = reshape(representative(A.E, m, x.coords), :, 1)

        # Lift to a chain map P_{m+k} -> P_k, for k up to s-m.
        coeffs = _lift_cocycle_to_chainmap_coeff(A.E.res, A.E.res, A.E, m, alpha; upto=(s - m))

        # We need the coefficient matrix at k = s-m, which maps P_s -> P_{s-m}.
        coeff = coeffs[(s - m) + 1]

        dom_bases = T.resL.gens[s + 1]
        cod_bases = T.resL.gens[(s - m) + 1]
        reps = T.homol[s + 1].Hrep
        target_offsets = T.offsets[(s - m) + 1]
        source_offsets = T.offsets[s + 1]
        coeff_nnz = _coeff_nnz(coeff)
        direct_plan = if _EXT_ACTION_USE_DIRECT_STREAM[] &&
                         coeff_nnz >= max(_EXT_ACTION_DIRECT_STREAM_MIN_NNZ[],
                                          _FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[])
            _tensor_coeff_plan(T.Rop, dom_bases, cod_bases, coeff)
        else
            nothing
        end

        images = if direct_plan !== nothing &&
                    _ext_action_use_direct_stream(direct_plan, coeff_nnz,
                                                  target_offsets, source_offsets,
                                                  size(reps, 2))
            scales = _coeff_plan_scales(coeff, direct_plan)
            out = zeros(K, target_offsets[end], size(reps, 2))
            @inbounds for j in 1:size(reps, 2)
                _accum_map_leq_many_scaled_sourcevec!(view(out, :, j), T.Rop, direct_plan.batch,
                                                      direct_plan.row_block_ids, direct_plan.col_block_ids,
                                                      target_offsets, source_offsets,
                                                      view(reps, :, j), scales)
            end
            out
        else
            F = _tensor_map_on_tor_chains_from_projective_coeff(
                T.Rop, dom_bases, cod_bases, source_offsets, target_offsets, coeff
            )
            F * reps
        end
        return ChainComplexes.homology_coordinates(T.homol[(s - m) + 1], images)
    end

    # Convenience: compute action matrices for s = 0..df.maxdeg.
    function ext_action_on_tor(A::ExtAlgebra{K}, T::TorSpaceSecond{K}, x::ExtElement{K}, df::DerivedFunctorOptions) where {K}
        _validate_native_derived_options(df, "ext_action_on_tor", :second)
        maxavail = last(degree_range(T))
        maxdeg = df.maxdeg
        maxdeg >= 0 || throw(ArgumentError("ext_action_on_tor: df.maxdeg must be nonnegative."))
        if maxdeg > maxavail
            throw(ArgumentError("ext_action_on_tor: df.maxdeg=$(maxdeg) exceeds available Tor degrees $(maxavail)"))
        end
        mats = Vector{Matrix{K}}(undef, maxdeg + 1)
        for s in 0:maxdeg
            mats[s + 1] = ext_action_on_tor(A, T, x; s=s)
        end
        return mats
    end


    # ============================================================
    # TorAlgebra with lazy mu_chain generation (generator + cache)
    # ============================================================

    """
        TorAlgebra(T; mu_chain=Dict(), mu_chain_gen=nothing, unit_coords=nothing)

    Equip a computed Tor space with explicitly supplied multiplication data.
    Arbitrary right and left modules do not determine a canonical Tor algebra.

    Mathematical input:
    - `T` is a Tor computation object (either TorSpace or TorSpaceSecond).
    - A chain-level product is given by matrices

        mu_chain[(p,q)] : C_p tensor C_q -> C_{p+q}

    in the chosen chain bases, with input coordinates `kron(x, y)` for
    `x in C_p` and `y in C_q` (the right factor varies fastest).
    Products must preserve cycles and boundaries. Every induced product checks
    these conditions before returning or caching a homology multiplication.

    Practical API:
    - You may supply all maps explicitly via `mu_chain`.
    - Or, supply a lazy generator `mu_chain_gen(p,q)` that returns the required sparse matrix.
    The result is cached in `A.mu_chain` automatically on first use.

    Cache publication is synchronized; product construction runs outside the cache
    lock. A generator must therefore support concurrent independent calls, including
    repeated calls for one degree pair, and return the same product until replaced
    with a setter. Mutating captured generator state is unsupported.
    Supplied and cached matrices are treated as
    read-only; use the setters below to replace products.

    Use `check_tor_algebra(A; algebraic=true)` to additionally check the chain
    Leibniz rule, associativity on homology, and both unit identities when
    `unit_coords` is supplied. A unit is an H_0 class, not a chosen strict chain
    unit. Associativity is checked through the computed Tor degree range; no
    assertion is made about uncomputed products. Leibniz checks also require
    chain products in the extra incoming-boundary degree `maxdeg + 1`.
    Missing products are not assumed zero unless their source or target chain
    group is zero. Without `unit_coords`, the algebra may be nonunital.
    """
    mutable struct TorAlgebra{K}
        lock::ReentrantLock
        generation::UInt
        T::Union{TorSpace{K},TorSpaceSecond{K}}
        mu_chain::Dict{Tuple{Int,Int}, SparseMatrixCSC{K, Int}}
        mu_chain_gen::Union{Nothing, Function}
        mu_H_cache::Dict{Tuple{Int,Int}, Matrix{K}}
        unit_coords::Union{Nothing, Vector{K}}
    end

    """
        TorAlgebra(T; mu_chain=Dict(), mu_chain_gen=nothing, unit_coords=nothing)

    Constructor with optional lazy generator.
    """
    function TorAlgebra(T::Union{TorSpace{K},TorSpaceSecond{K}};
        mu_chain::Dict{Tuple{Int,Int}, SparseMatrixCSC{K,Int}}=Dict{Tuple{Int,Int}, SparseMatrixCSC{K,Int}}(),
        mu_chain_gen::Union{Nothing,Function}=nothing,
        unit_coords::Union{Nothing,Vector{K}}=nothing) where {K}
        A = TorAlgebra{K}(ReentrantLock(), UInt(0), T, copy(mu_chain), mu_chain_gen,
                         Dict{Tuple{Int,Int}, Matrix{K}}(),
                         unit_coords === nothing ? nothing : copy(unit_coords))
        check_tor_algebra(A; throw=true)
        return A
    end

    function TorAlgebra(T::Any; kwargs...)
        error("TorAlgebra: expected TorSpace{K} or TorSpaceSecond{K}, got $(typeof(T))")
    end

    degree_range(A::TorAlgebra) = degree_range(A.T)
    dim(A::TorAlgebra, deg::Int) = dim(A.T, deg)
    @inline nonzero_degrees(A::TorAlgebra) = [t for t in degree_range(A) if dim(A.T, t) != 0]

    @inline function generator_degrees(A::TorAlgebra)
        out = Int[]
        for t in degree_range(A)
            append!(out, fill(t, dim(A.T, t)))
        end
        return out
    end

    @inline algebra_field(A::TorAlgebra) = A.T isa TorSpaceSecond ? A.T.resL.M.field : A.T.resRop.M.field
    @inline total_dimension(A::TorAlgebra) = sum(dim(A, t) for t in degree_range(A))

    """
        algebra_summary(A::TorAlgebra) -> NamedTuple

    Cheap-first summary of a `TorAlgebra`.

    This reports the graded support, total graded dimension, and the currently
    cached chain- and homology-level products without generating new
    multiplication data.

    Use this before asking for explicit `TorElement`s or multiplication
    matrices.
    """
    @inline function algebra_summary(A::TorAlgebra)
        return lock(A.lock) do
            return (
                kind=:tor_algebra,
                provenance=provenance(A),
                field=algebra_field(A),
                degree_range=degree_range(A),
                nonzero_degrees=Tuple(nonzero_degrees(A)),
                total_dimension=total_dimension(A),
                cached_chain_products=length(A.mu_chain),
                cached_homology_products=length(A.mu_H_cache),
                has_lazy_generator=A.mu_chain_gen !== nothing,
                has_unit=A.unit_coords !== nothing,
            )
        end
    end

    """
        underlying_tor_space(A::TorAlgebra)

    Return the graded Tor-space model carried by `A`.

    This is the additive model underlying Tor coordinates and representatives.
    Use it when you need chain-level or graded-space access; for ordinary
    inspection start with `algebra_summary(A)`.
    """
    @inline underlying_tor_space(A::TorAlgebra) = A.T

    """
        cached_product_degrees(A::TorAlgebra)

    Return the currently cached degree pairs for chain- or homology-level Tor
    products.

    The result is cheap and purely inspectable: it does not trigger generation
    of new chain products or induced homology products.
    """
    @inline function cached_product_degrees(A::TorAlgebra)
        return lock(A.lock) do
            sort!(collect(union(Set(keys(A.mu_chain)), Set(keys(A.mu_H_cache)))))
        end
    end

    function Base.show(io::IO, A::TorAlgebra)
        d = algebra_summary(A)
        print(io, "TorAlgebra(nonzero_degrees=", repr(d.nonzero_degrees),
              ", cached_products=", repr(cached_product_degrees(A)), ")")
    end

    function Base.show(io::IO, ::MIME"text/plain", A::TorAlgebra)
        d = algebra_summary(A)
        print(io, "TorAlgebra",
              "\n  field: ", d.field,
              "\n  degree_range: ", repr(d.degree_range),
              "\n  nonzero_degrees: ", repr(d.nonzero_degrees),
              "\n  total_dimension: ", d.total_dimension,
              "\n  cached_product_degrees: ", repr(cached_product_degrees(A)),
              "\n  has_lazy_generator: ", d.has_lazy_generator)
    end

    """
        check_tor_algebra(A; algebraic=false, throw=false) -> NamedTuple

    Validate the cheap structural contracts of a `TorAlgebra`.

    This checks cached chain-level and homology-level product sizes and the
    degree bookkeeping of the underlying Tor space without forcing any new
    product computations.

    With `algebraic=true`, use a consistent snapshot of the product data to
    check descent, the chain Leibniz rule through the extra incoming-boundary
    degree, associativity on homology through `last(degree_range(A))`, and
    both unit identities if a unit was supplied. This may generate every
    needed product and is intentionally opt-in. It does not fill the original
    algebra's product caches. Strict associativity or a strict unit on chains
    is not required. RealField identities use its `atol` and `rtol`.

    The report distinguishes `checks=:structural` from `checks=:algebraic`
    and gives `verified_through` only for a successful algebraic check.
    A structural success alone does not certify an algebra.
    """
    function check_tor_algebra(A::TorAlgebra{K}; algebraic::Bool=false, throw::Bool=false) where {K}
        snapshot = _tor_algebra_snapshot(A)
        issues = String[]
        T = snapshot.T
        n = last(degree_range(snapshot))
        original_pairs = Tuple(cached_product_degrees(snapshot))
        for ((p, q), MU) in snapshot.mu_H_cache
            if !_tor_product_degrees(p, q, n)
                push!(issues, "cached homology product ($p,$q) is outside degrees 0:$n.")
            else
                expected = (dim(snapshot, p + q), dim(snapshot, p) * dim(snapshot, q))
                size(MU) == expected || push!(issues, "cached homology product ($p,$q) has size $(size(MU)); expected $expected.")
                _tor_finite(MU) || push!(issues, "cached homology product ($p,$q) contains nonfinite entries.")
            end
        end
        for ((p, q), mu) in snapshot.mu_chain
            try
                _tor_check_chain_product(T, p, q, mu)
            catch err
                err isa ArgumentError || rethrow()
                push!(issues, sprint(showerror, err))
            end
        end
        u = snapshot.unit_coords
        if u !== nothing
            length(u) == dim(snapshot, 0) || push!(issues, "unit coordinates must have length $(dim(snapshot, 0)).")
            _tor_finite(u) || push!(issues, "unit coordinates contain nonfinite entries.")
        end
        if algebraic && isempty(issues)
            # Recompute from chain data, so stale/hand-built homology tables do
            # not conceal an invalid product in the underlying chain model.
            stored_products = copy(snapshot.mu_H_cache)
            empty!(snapshot.mu_H_cache)
            try
                _check_tor_algebraic!(issues, snapshot)
                for (pair, stored) in stored_products
                    _tor_equal(algebra_field(snapshot), stored, snapshot.mu_H_cache[pair]) ||
                        push!(issues, "cached homology product $pair disagrees with the supplied chain product.")
                end
            catch err
                (err isa ArgumentError || err isa ErrorException || err isa DimensionMismatch) || rethrow()
                push!(issues, sprint(showerror, err))
            end
        end
        report = _derived_validation_report(
            :tor_algebra, isempty(issues); field=algebra_field(snapshot),
            degree_range=degree_range(snapshot), total_dimension=total_dimension(snapshot),
            cached_product_degrees=original_pairs,
            checks=algebraic ? :algebraic : :structural,
            verified_through=algebraic && isempty(issues) ? n : nothing,
            unit_supplied=u !== nothing, issues=issues)
        throw && !report.valid && _throw_invalid_derived_functor(:check_tor_algebra, issues)
        return report
    end

    function _tor_algebra_snapshot(A::TorAlgebra{K}) where {K}
        return lock(A.lock) do
            TorAlgebra{K}(ReentrantLock(), A.generation, A.T, copy(A.mu_chain),
                          A.mu_chain_gen, copy(A.mu_H_cache),
                          A.unit_coords === nothing ? nothing : copy(A.unit_coords))
        end
    end

    @inline _tor_product_degrees(p::Int, q::Int, n::Int) = 0 <= p <= n && 0 <= q <= n - p
    @inline _tor_finite(A::AbstractArray{<:AbstractFloat}) = all(isfinite, A)
    @inline _tor_finite(A::AbstractArray) = true
    @inline _tor_equal(field, X, Y) = _tor_finite(X) && _tor_finite(Y) &&
        (field isa RealField ? isapprox(X, Y; atol=field.atol, rtol=field.rtol) : X == Y)

    function _tor_check_chain_product(T, p::Int, q::Int, mu)
        n = length(T.dims) - 1
        _tor_product_degrees(p, q, n) ||
            throw(ArgumentError("Tor chain product ($p,$q) is outside available chain degrees 0:$n."))
        expected = (T.dims[p + q + 1], T.dims[p + 1] * T.dims[q + 1])
        size(mu) == expected || throw(ArgumentError("Tor chain product ($p,$q) has size $(size(mu)); expected $expected."))
        _tor_finite(mu) || throw(ArgumentError("Tor chain product ($p,$q) contains nonfinite entries."))
        return nothing
    end

    # Stream columns instead of materializing a tensor product of basis matrices.
    function _tor_product_images(mu::AbstractMatrix{K}, X, Y) where {K}
        out = Matrix{K}(undef, size(mu, 1), size(X, 2) * size(Y, 2))
        col = 0
        for i in axes(X, 2), j in axes(Y, 2)
            col += 1
            out[:, col] = mu * kron(view(X, :, i), view(Y, :, j))
        end
        return out
    end

    function _tor_homology_coordinates(field, H, z)
        _tor_finite(z) || throw(ArgumentError("Tor product arithmetic produced nonfinite chain coordinates."))
        c = if field isa RealField
            # Retain the user's numerical contract rather than reconstructing
            # a default tolerance solely from the coefficient element type.
            alpha = solve_fullcolumn(field, H.Z, z)
            gamma = solve_fullcolumn(field, H.Bfull, alpha)
            gamma[H.dimB + 1:end, :]
        else
            ChainComplexes.homology_coordinates(H, z)
        end
        _tor_finite(c) || throw(ArgumentError("Tor product arithmetic produced nonfinite homology coordinates."))
        return c
    end

    function _tor_product_coordinates(A::TorAlgebra{K}, p::Int, q::Int, mu) where {K}
        _tor_check_chain_product(A.T, p, q, mu)
        Hp, Hq, Hr = A.T.homol[p + 1], A.T.homol[q + 1], A.T.homol[p + q + 1]
        field = algebra_field(A)
        # Z = B + span(Hrep): these two ideal conditions plus the Hrep products
        # certify every cycle product and independence of both representatives.
        for (X, Y) in ((Hp.B, Hq.Z), (Hp.Hrep, Hq.B))
            (size(X, 2) == 0 || size(Y, 2) == 0) && continue
            z = _tor_product_images(mu, X, Y)
            c = try
                _tor_homology_coordinates(field, Hr, z)
            catch err
                (err isa ErrorException || err isa ArgumentError) || rethrow()
                throw(ArgumentError("Tor product ($p,$q) does not preserve cycles: $(sprint(showerror, err))"))
            end
            _tor_equal(field, c, zeros(K, size(c))) ||
                throw(ArgumentError("Tor product ($p,$q) does not preserve boundaries; it is not independent of representatives."))
        end
        z = _tor_product_images(mu, Hp.Hrep, Hq.Hrep)
        return try
            _tor_homology_coordinates(field, Hr, z)
        catch err
            (err isa ErrorException || err isa ArgumentError) || rethrow()
            throw(ArgumentError("Tor product ($p,$q) does not preserve cycles: $(sprint(showerror, err))"))
        end
    end

    function _check_tor_algebraic!(issues, A::TorAlgebra{K}) where {K}
        T, field = A.T, algebra_field(A)
        n = last(degree_range(A))
        # One additional degree is essential: boundaries in H_n come from C_{n+1}.
        for s in 1:(n + 1), p in 0:s
            q = s - p
            mu, _ = _get_mu_chain(A, p, q)
            lhs = T.bd[s] * mu
            rhs = zeros(K, size(lhs))
            if p > 0
                lower, _ = _get_mu_chain(A, p - 1, q)
                Iq = spdiagm(0 => fill(one(K), T.dims[q + 1]))
                rhs .+= lower * kron(T.bd[p], Iq)
            end
            if q > 0
                lower, _ = _get_mu_chain(A, p, q - 1)
                Ip = spdiagm(0 => fill(one(K), T.dims[p + 1]))
                rhs .+= (isodd(p) ? -one(K) : one(K)) .* (lower * kron(Ip, T.bd[q]))
            end
            _tor_equal(field, lhs, rhs) || push!(issues, "chain Leibniz identity fails in degrees ($p,$q).")
        end
        for s in 0:n, p in 0:s
            multiplication_matrix(A, p, s - p)
        end
        for s in 0:n, p in 0:s, q in 0:(s - p)
            r = s - p - q
            pq = multiplication_matrix(A, p, q)
            qr = multiplication_matrix(A, q, r)
            Ip = Matrix{K}(I, dim(A, p), dim(A, p))
            Ir = Matrix{K}(I, dim(A, r), dim(A, r))
            lhs = multiplication_matrix(A, p + q, r) * kron(pq, Ir)
            rhs = multiplication_matrix(A, p, q + r) * kron(Ip, qr)
            _tor_equal(field, lhs, rhs) || push!(issues, "homology associativity fails in degrees ($p,$q,$r).")
        end
        _check_tor_unit!(issues, A)
        return nothing
    end

    function _check_tor_unit!(issues, A::TorAlgebra{K}) where {K}
        u = A.unit_coords
        u === nothing && return nothing
        field = algebra_field(A)
        for p in degree_range(A)
            Ip = Matrix{K}(I, dim(A, p), dim(A, p))
            lhs = multiplication_matrix(A, 0, p) * kron(u, Ip)
            rhs = multiplication_matrix(A, p, 0) * kron(Ip, u)
            _tor_equal(field, lhs, Ip) || push!(issues, "left unit identity fails in degree $p.")
            _tor_equal(field, rhs, Ip) || push!(issues, "right unit identity fails in degree $p.")
        end
        return nothing
    end

    # Internal: obtain a chain-level multiplication map, generating it if needed.
    function _get_mu_chain(A::TorAlgebra{K}, p::Int, q::Int) where {K}
        _tor_product_degrees(p, q, length(A.T.dims) - 1) ||
            throw(ArgumentError("Tor chain product degrees ($p,$q) are outside the available range."))
        key = (p,q)
        cached, generator, generation = lock(A.lock) do
            (get(A.mu_chain, key, nothing), A.mu_chain_gen, A.generation)
        end
        cached === nothing || return (cached, generation)
        rows, cols = A.T.dims[p + q + 1], A.T.dims[p + 1] * A.T.dims[q + 1]
        M = if rows == 0 || cols == 0
            spzeros(K, rows, cols)
        elseif generator === nothing
            throw(ArgumentError("TorAlgebra: supply a chain product or generator for degrees ($p,$q); a generic Tor space has no canonical multiplication."))
        else
            generator(p,q)
        end
        isa(M, SparseMatrixCSC{K,Int}) || throw(ArgumentError("mu_chain_gen must return SparseMatrixCSC{$K,Int}."))
        _tor_check_chain_product(A.T, p, q, M)
        return lock(A.lock) do
            # An overlapping setter invalidates publication, not the private
            # result already being computed by this invocation.
            A.generation == generation ? (get!(A.mu_chain, key, M), generation) : (M, generation)
        end
    end

    """
        TorElement(A, deg, coords)

    A Tor class in degree `deg`, expressed in the basis used by `A.T`.
    """
    struct TorElement{K}
        A::TorAlgebra{K}
        deg::Int
        coords::Vector{K}
    end

    """
        parent_algebra(x::TorElement)

    Return the `TorAlgebra` that owns `x`.
    """
    @inline parent_algebra(x::TorElement) = x.A

    """
        element_degree(x::TorElement)

    Return the homogeneous homological degree of `x`.
    """
    @inline element_degree(x::TorElement) = x.deg

    """
        element_coordinates(x::TorElement)

    Return the coordinate vector of `x` in the basis carried by its parent
    `TorAlgebra`.

    This is the semantic accessor preferred over direct field access in user
    code. It is equivalent to `coordinates(x)`.
    """
    @inline element_coordinates(x::TorElement) = x.coords

    "Return the coordinate vector of a TorElement."
    coordinates(x::TorElement{K}) where {K} = element_coordinates(x)

    function Base.show(io::IO, x::TorElement)
        d = _element_summary(:tor_element, x)
        print(io, "TorElement(degree=", d.degree,
              ", nnz=", d.nonzero_coordinates, ")")
    end

    function Base.show(io::IO, ::MIME"text/plain", x::TorElement)
        d = _element_summary(:tor_element, x)
        print(io, "TorElement",
              "\n  field: ", d.field,
              "\n  degree: ", d.degree,
              "\n  coordinate_length: ", d.coordinate_length,
              "\n  nonzero_coordinates: ", d.nonzero_coordinates,
              "\n  is_zero: ", d.is_zero)
    end

    function element(A::TorAlgebra{K}, deg::Int, coords::AbstractVector{K}) where {K}
        deg in degree_range(A) || throw(ArgumentError("Tor element degree $deg is outside $(degree_range(A))."))
        length(coords) == dim(A, deg) || throw(DimensionMismatch("Tor element expects $(dim(A, deg)) coordinates in degree $deg."))
        _tor_finite(coords) || throw(ArgumentError("Tor element contains nonfinite coordinates."))
        return TorElement{K}(A, deg, collect(coords))
    end

    """
        unit(A::TorAlgebra)
        one(A::TorAlgebra)

    Return the supplied H_0 unit after checking its left and right identities
    in every computed degree. No unit is inferred from the chain basis.
    Product data are checked on a consistent snapshot; returned coordinates
    are owned by the element and can be edited without changing the algebra.
    """
    function unit(A::TorAlgebra{K}) where {K}
        snapshot = _tor_algebra_snapshot(A)
        snapshot.unit_coords === nothing && throw(ArgumentError("TorAlgebra has no supplied unit_coords."))
        check_tor_algebra(snapshot; throw=true)
        issues = String[]
        _check_tor_unit!(issues, snapshot)
        isempty(issues) || _throw_invalid_derived_functor(:unit, issues)
        return TorElement{K}(A, 0, copy(snapshot.unit_coords))
    end
    Base.one(A::TorAlgebra) = unit(A)

    """
        set_chain_product!(A, p, q, mu)

    Set the chain-level product map for degrees (p,q).
    This overrides any lazily generated value.
    An overlapping query may finish with its previously captured product, but cannot
    restore that old result to the cache after this setter returns.
    """
    function set_chain_product!(A::TorAlgebra{K}, p::Int, q::Int, mu::SparseMatrixCSC{K,Int}) where {K}
        _tor_check_chain_product(A.T, p, q, mu)
        lock(A.lock) do
            A.mu_chain[(p,q)] = mu
            empty!(A.mu_H_cache)
            A.generation += UInt(1)
        end
        return A
    end

    """
        set_chain_product_generator!(A, gen)

    Attach a lazy generator `gen(p,q)` to supply mu_chain maps on demand.
    Clears caches.
    An overlapping query may finish with its previously captured generator, but its
    result cannot populate the replacement generator's caches.
    """
    function set_chain_product_generator!(A::TorAlgebra{K}, gen::Function) where {K}
        lock(A.lock) do
            A.mu_chain_gen = gen
            empty!(A.mu_chain)
            empty!(A.mu_H_cache)
            A.generation += UInt(1)
        end
        return A
    end

    """
        multiplication_matrix(A, p, q)

    Return the induced multiplication matrix on Tor homology:

        Tor_p x Tor_q -> Tor_{p+q}

    The returned matrix has size:
        dim(Tor_{p+q}) x (dim(Tor_p)*dim(Tor_q)),

    and its column ordering matches `kron(x.coords, y.coords)`, exactly as for
    the supplied chain product. Before publishing a newly computed matrix,
    all cycle products and both boundary-ideal conditions are checked. Thus
    invalid products cannot silently depend on the chosen representatives.
    Associativity and unit identities can be checked separately with
    `check_tor_algebra(A; algebraic=true)`.
    The returned matrix is cached and must be treated as read-only. Copy it before
    editing its entries.
    """
    function multiplication_matrix(A::TorAlgebra{K}, p::Int, q::Int) where {K}
        # Both supported models expose the same typed homology vector, so
        # checking the degree range does not allocate on cached queries.
        n = length(A.T.homol)
        _tor_product_degrees(p, q, n - 1) ||
            throw(ArgumentError("Tor product ($p,$q) is outside computed homology degrees $(degree_range(A))."))
        key = (p,q)
        cached = lock(() -> get(A.mu_H_cache, key, nothing), A.lock)
        cached === nothing || return cached

        # Get chain-level multiplication map (possibly generated lazily)
        mu, generation = _get_mu_chain(A, p, q)

        out = _tor_product_coordinates(A, p, q, mu)

        return lock(A.lock) do
            A.generation == generation ? get!(A.mu_H_cache, key, out) : out
        end
    end

    """
        multiply(A, x, y)

    Multiply Tor elements using the registered chain-level product.
    """
    function multiply(A::TorAlgebra{K}, x::TorElement{K}, y::TorElement{K}) where {K}
        x.A === A && y.A === A || throw(ArgumentError("Tor factors must belong to the supplied algebra."))
        p, q = x.deg, y.deg
        p in degree_range(A) && q in degree_range(A) || throw(ArgumentError("Tor factor degree is outside the computed range."))
        length(x.coords) == dim(A, p) && length(y.coords) == dim(A, q) ||
            throw(DimensionMismatch("Tor factor coordinate lengths do not match their degrees."))
        _tor_finite(x.coords) && _tor_finite(y.coords) || throw(ArgumentError("Tor factor contains nonfinite coordinates."))
        M = multiplication_matrix(A, p, q)
        out_coords = M * kron(x.coords, y.coords)
        _tor_finite(out_coords) || throw(ArgumentError("Tor multiplication produced nonfinite coordinates."))
        return TorElement{K}(A, p + q, out_coords)
    end

end
