# Shared fields, owner aliases and fixtures are provided by test/prelude.jl.

# Build an endomorphism of M that is the identity everywhere except at vertex u,
# where it is replaced by the matrix A (assumes A has size M.dims[u] x M.dims[u]).
function endo_at_vertex(M::MD.PModule{K}, u::Int, A::AbstractMatrix{K}) where {K}
    comps = Vector{Matrix{K}}(undef, M.Q.n)
    for v in 1:M.Q.n
        dv = M.dims[v]
        comps[v] = CM.eye(M.field, dv)
    end
    comps[u] = Matrix{K}(A)
    return MD.PMorphism(M, M, comps)
end

# Compose morphisms fiberwise: (g o f)_u = g_u * f_u.
function compose_morphism(g::MD.PMorphism{K}, f::MD.PMorphism{K}) where {K}
    @assert f.cod === g.dom
    n = f.dom.Q.n
    comps = [g.comps[u] * f.comps[u] for u in 1:n]
    return MD.PMorphism(f.dom, g.cod, comps)
end

# Scalar endomorphism s*id on each fiber.
function scalar_endo(M::MD.PModule{K}, s::K) where {K}
    comps = Vector{Matrix{K}}(undef, M.Q.n)
    for u in 1:M.Q.n
        d = M.dims[u]
        comps[u] = d == 0 ? CM.zeros(M.field, 0, 0) : s .* CM.eye(M.field, d)
    end
    return MD.PMorphism(M, M, comps)
end

# Helper: build a chain-poset module with a single cover edge map.
# We intentionally keep this tiny; it is enough to test connecting morphisms by hand.
function _chain_module(P, dims::Vector{Int}, edge_map::AbstractMatrix{K}, field::CM.AbstractCoeffField) where {K}
    CM.coeff_type(field) == K || error("_chain_module: coeff_type(field) != eltype(edge_map)")
    edges = FF.cover_edges(P)
    D = Dict{Tuple{Int, Int}, Matrix{K}}()
    for (u, v) in edges
        D[(u, v)] = CM.zeros(field, dims[v], dims[u])
    end
    @assert length(edges) == 1
    D[first(edges)] = edge_map
    return MD.PModule{K}(P, dims, D; field=field)
end

@testset "A74 rationally transported Tor products and natural maps" begin
    # This is the dual-number DGA C0=<u,v>, C1=<h,z>, d(h)=v,
    # d(z)=0, v^2=0, vh=hv=z, and C1*C1=0. Rational changes of basis
    # make its nonzero differential dense and nonintegral while retaining a
    # known one-dimensional H0 and H1 and well-separated nonzero singular value.
    fieldQ = CM.QQField()
    fieldR = CM.RealField(Float64; atol=1e-12, rtol=1e-10)
    B0 = QQ[1 1//3; 1//4 1]
    B1 = QQ[1 1//5; 1//7 1]
    D = B0 * QQ[0 0; 1 0] * inv(B1)
    P = chain_poset(2)
    Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)))
    RQ = _chain_module(Pop, [0, 1], zeros(QQ, 0, 1), fieldQ)
    RR = _chain_module(Pop, [0, 1], zeros(Float64, 0, 1), fieldR)
    LQ = _chain_module(P, [2, 2], D, fieldQ)
    LR = _chain_module(P, [2, 2], Float64.(D), fieldR)
    options = OPT.DerivedFunctorOptions(maxdeg=2, model=:first)
    TQ, TR = DF.Tor(RQ, LQ, options), DF.Tor(RR, LR, options)
    @test [DF.dim(TQ, s) for s in 0:2] == [1, 1, 0]
    @test [DF.dim(TR, s) for s in 0:2] == [1, 1, 0]
    # The simple right-module resolution can choose either orientation.
    # Determine it once over QQ and retain exactly the same chain coordinates.
    pivot = findfirst(x -> !iszero(x), D)
    sign = TQ.bd[1][pivot] / D[pivot]
    @test sign in (QQ(-1), QQ(1))
    bases = (B0, sign .* B1)
    dual_product = QQ[1 0 0 0; 0 1 1 0]
    productsQ = Dict((p, q) => sparse(bases[p+q+1] * dual_product *
        kron(inv(bases[p+1]), inv(bases[q+1]))) for (p, q) in ((0, 0), (0, 1), (1, 0)))
    productsR = Dict(pair => sparse(Float64.(product)) for (pair, product) in productsQ)
    u, v, h, z = B0[:, 1], B0[:, 2], bases[2][:, 1], bases[2][:, 2]
    AQ = DF.TorAlgebra(TQ; mu_chain=productsQ, unit_coords=DF.coordinates(TQ, 0, u))
    AR = DF.TorAlgebra(TR; mu_chain=productsR, unit_coords=DF.coordinates(TR, 0, Float64.(u)))
    residuals = Float64[]
    function numerical_identity(X, Y)
        residual, scale = norm(X - Y), max(norm(X), norm(Y))
        push!(residuals, residual / max(1.0, scale))
        @test isfinite(residual) && residual <= fieldR.atol + fieldR.rtol * scale
    end
    numerical_identity(TR.bd[1], Float64.(TQ.bd[1]))
    @test all(H -> H.field == fieldR, TR.homol)
    basis_conditions = [cond(Float64.(B)) for B in bases]
    @test maximum(basis_conditions) < 3
    singular_values = svdvals(Float64.(D))
    @test singular_values[1] > 0.5
    @test singular_values[2] < 1e-12
    @test FL.rank(fieldQ, D) == FL.rank(fieldR, sparse(Float64.(D))) == 1
    for (field, T, algebra, products) in ((fieldQ, TQ, AQ, productsQ),
                                        (fieldR, TR, AR, productsR))
        K = CM.coeff_type(field)
        same = (X, Y) -> field isa CM.RealField ? numerical_identity(X, Y) : (@test X == Y)
        unit_class = DF.coordinates(T, 0, K.(u))
        positive_class = DF.coordinates(T, 1, K.(z))
        @test DF.check_tor_algebra(algebra; algebraic=true).valid
        same(T.bd[1] * K.(h), K.(v))
        same(T.bd[1] * K.(z), zeros(K, 2))
        same(one(algebra).coords, unit_class)
        for (p, q) in ((0, 0), (0, 1), (1, 0))
            x, y = p == 0 ? unit_class : positive_class,
                   q == 0 ? unit_class : positive_class
            expected = p + q == 0 ? unit_class : positive_class
            same(DF.multiply(algebra, DF.element(algebra, p, x),
                             DF.element(algebra, q, y)).coords, expected)
        end
        # Check the odd-degree Leibniz cancellation directly in chain
        # coordinates and prove it involves two nonzero terms.
        left_term = products[(0, 1)] * kron(K.(v), K.(h))
        right_term = products[(1, 0)] * kron(K.(h), K.(v))
        same(left_term, K.(z))
        same(right_term, K.(z))
        same(left_term - right_term, zeros(K, 2))
        for alpha in QQ[0, 1//3, -2//5], beta in QQ[0, 2//7, -1//2]
            changed = products[(0, 0)] * kron(K.(u + alpha*v), K.(u + beta*v))
            same(DF.coordinates(T, 0, changed), unit_class)
        end
    end

    # Compare exact and numerical answers through explicit homology-coordinate
    # transports, rather than assuming the two elimination methods pick the
    # same normalized representatives.
    transport = [reshape(DF.coordinates(TR, s,
        Float64.(DF.representative(TQ, s, QQ[1]))), 1, 1) for s in 0:1]
    @test all(J -> abs(only(J)) > 1e-2, transport)
    for (p, q) in ((0, 0), (0, 1), (1, 0))
        numerical_identity(transport[p+q+1] * Float64.(DF.multiplication_matrix(AQ, p, q)),
            DF.multiplication_matrix(AR, p, q) * kron(transport[p+1], transport[q+1]))
    end
    for a in QQ[0, 2//3, -3//2]
        G0 = bases[1] * QQ[1 0; 0 a] * inv(bases[1])
        G1 = bases[2] * QQ[a 0; 0 a*a] * inv(bases[2])
        @test G0 * TQ.bd[1] == TQ.bd[1] * G1
        fQ = MD.PMorphism(LQ, LQ, [G1, G0])
        fR = MD.PMorphism(LR, LR, [Float64.(G1), Float64.(G0)])
        mapsQ = [DF.tor_map_second(TQ, TQ, fQ; s) for s in 0:1]
        mapsR = [DF.tor_map_second(TR, TR, fR; s) for s in 0:1]
        @test mapsQ[1] == ones(QQ, 1, 1)
        @test mapsQ[2] == fill(a*a, 1, 1)
        numerical_identity(mapsR[1], ones(1, 1))
        numerical_identity(mapsR[2], fill(Float64(a*a), 1, 1))
        for s in 0:1
            numerical_identity(mapsR[s+1] * transport[s+1], transport[s+1] * Float64.(mapsQ[s+1]))
        end
        numerical_identity(mapsR[1] * one(AR).coords, one(AR).coords)
        for (p, q) in ((0, 0), (0, 1), (1, 0))
            product = DF.multiplication_matrix(AR, p, q)
            numerical_identity(mapsR[p+q+1] * product, product * kron(mapsR[p+1], mapsR[q+1]))
        end
    end
    @info "A74 numerical Tor diagnostics" basis_conditions=repr(basis_conditions) singular_values=repr(singular_values) max_scaled_residual=maximum(residuals) atol=fieldR.atol rtol=fieldR.rtol
end


# A matrix wrapper supplies real yield/nesting/error points inside composition,
# without adding a production-only test hook to the algebra kernels.
struct _A10CallbackMatrix{K,F} <: AbstractMatrix{K}
    data::Matrix{K}
    callback::F
end
Base.size(A::_A10CallbackMatrix) = size(A.data)
Base.getindex(A::_A10CallbackMatrix, i::Int, j::Int) = (A.callback(); A.data[i, j])

function _a10_callback_morphism(M::MD.PModule{K,F}, matrices, callback) where {K,F}
    blocks = [_A10CallbackMatrix(Matrix(A), callback) for A in matrices]
    return MD.PMorphism{K,F,eltype(blocks)}(M, M, blocks)
end

function _a10_spawn(f, job::Int)
    if iseven(job) && Base.Threads.nthreads(:interactive) > 0
        return Base.Threads.@spawn :interactive f()
    end
    return Base.Threads.@spawn :default f()
end

@testset "A10 Hom leases isolate concurrent, nested and failing calls" begin
    functor = DF.Functoriality
    old_plan = functor._FUNCTORIALITY_USE_HOM_BASIS_SOLVE_PLAN_CACHE[]
    old_scratch = functor._FUNCTORIALITY_USE_HOM_SOLVE_WORKSPACE_CACHE[]
    try
        with_fields(FIELDS_FULL) do field
            K = CM.coeff_type(field)
            same(A, B) = field isa CM.RealField ?
                isapprox(A, B; atol=field.atol, rtol=field.rtol) : A == B
            P = chain_poset(1)
            M = MD.PModule{K}(P, [2], Dict{Tuple{Int,Int},Matrix{K}}(); field=field)
            cache = DF.HomSystemCache(K)
            # Tensor/complex consumers attach their memo stores to this owner
            # through weak keys; distinct empty caches must retain identities.
            weak_owners = WeakKeyDict{Any,Int}(cache => 1)
            other_cache = DF.HomSystemCache(K)
            weak_owners[other_cache] = 2
            @test weak_owners[cache] == 1
            @test weak_owners[other_cache] == 2
            @test length(weak_owners) == 2
            H = DF.hom_with_cache(M, M; cache=cache)
            @test DF.dim(H) == 4
            I2 = CM.eye(field, 2)
            values = [K[1 CM.coerce(field, j); CM.coerce(field, j + 1) 1] for j in 1:12]
            # Column vectorization gives vec(X*A)=(A^T tensor I)vec(X)
            # and vec(A*X)=(I tensor A)vec(X), independently of Hom basis choice.
            for (use_plan, use_scratch) in ((true, true), (false, true), (false, false))
                functor._FUNCTORIALITY_USE_HOM_BASIS_SOLVE_PLAN_CACHE[] = use_plan
                functor._FUNCTORIALITY_USE_HOM_SOLVE_WORKSPACE_CACHE[] = use_scratch
                functor._clear_functoriality_caches!()
                DF.clear_hom_system_cache!(cache)
                tasks = [_a10_spawn(j) do
                    A = values[j]
                    f = _a10_callback_morphism(M, [A], yield)
                    pre = DF.precompose_matrix_cached(H, H, f; cache=cache)
                    post = DF.postcompose_matrix_cached(H, H, f; cache=cache)
                    # Repeated hits must preserve the same cached result.
                    @assert DF.precompose_matrix_cached(H, H, f; cache=cache) === pre
                    @assert DF.postcompose_matrix_cached(H, H, f; cache=cache) === post
                    (pre, post)
                end for j in eachindex(values)]
                results = fetch.(tasks)
                for (j, (pre, post)) in enumerate(results)
                    @test same(H.basis_matrix * pre, kron(transpose(values[j]), I2) * H.basis_matrix)
                    @test same(H.basis_matrix * post, kron(I2, values[j]) * H.basis_matrix)
                end
            end

            functor._FUNCTORIALITY_USE_HOM_BASIS_SOLVE_PLAN_CACHE[] = true
            functor._FUNCTORIALITY_USE_HOM_SOLVE_WORKSPACE_CACHE[] = true
            functor._clear_functoriality_caches!()
            outer = values[1]
            inner = MD.PMorphism(M, M, [values[2]])
            for induce in (functor._precompose_matrix, functor._postcompose_matrix)
                expected = induce(H, H, MD.PMorphism(M, M, [outer]))
                entered = Ref(false)
                nested_value = Ref{Matrix{K}}()
                function nested()
                    if !entered[]
                        entered[] = true
                        nested_value[] = induce(H, H, inner)
                    end
                    yield()
                end
                nested_map = _a10_callback_morphism(M, [outer], nested)
                @test same(induce(H, H, nested_map), expected)
                @test same(nested_value[], induce(H, H, inner))
                # The failing invocation has already leased and initialized its
                # RHS. Both exceptional and normal exits must return that lease.
                failure = _a10_callback_morphism(M, [outer], () -> error("A10 injected coefficient failure"))
                pool = functor._hom_solve_workspace_entry(H)
                before = (length(pool.pre), length(pool.post))
                @test_throws ErrorException induce(H, H, failure)
                @test (length(pool.pre), length(pool.post)) == before
                @test same(induce(H, H, inner), induce(H, H, inner))
                @test same(expected, induce(H, H, MD.PMorphism(M, M, [outer])))
            end

            # A live lease remains private across clear; returning it does not
            # attach it to the fresh pool. Nested acquisition cannot alias it.
            for W in (functor._PrecomposeWorkspace{K}, functor._PostcomposeWorkspace{K})
                functor._with_hom_workspace(H, nothing, W) do held
                    held.rhs = fill(one(K), 3, 3)
                    functor._clear_functoriality_caches!()
                    functor._with_hom_workspace(H, nothing, W) do nested
                        @test nested !== held
                        nested.rhs = zeros(K, 3, 3)
                        yield()
                        @test held.rhs == fill(one(K), 3, 3)
                    end
                    pool = functor._hom_solve_workspace_entry(H)
                    @test all(ws -> ws !== held, functor._hom_idle_workspaces(pool, W))
                end
            end
            @test fieldnames(typeof(functor._hom_solve_plan_entry(H))) == (:plan,)
            DF.clear_hom_system_cache!(cache)
            @test isempty(cache.hom) && isempty(cache.precompose) && isempty(cache.postcompose)
            rebuilt = DF.hom_with_cache(M, M; cache=cache)
            @test rebuilt !== H
            @test DF.hom_with_cache(M, M; cache=cache) === rebuilt
        end
    finally
        functor._FUNCTORIALITY_USE_HOM_BASIS_SOLVE_PLAN_CACHE[] = old_plan
        functor._FUNCTORIALITY_USE_HOM_SOLVE_WORKSPACE_CACHE[] = old_scratch
        functor._clear_functoriality_caches!()
    end
end

@testset "A10 derived caches reject recycled identity and pattern keys" begin
    F = DF.Functoriality
    old_coeff = F._FUNCTORIALITY_USE_COEFF_PLAN_CACHE[]
    old_result = F._FUNCTORIALITY_USE_TENSOR_COEFF_RESULT_CACHE[]
    old_direct = F._FUNCTORIALITY_USE_DIRECT_COEFF_TRIPLETS[]
    old_min = F._FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[]
    try
        F._FUNCTORIALITY_USE_COEFF_PLAN_CACHE[] = true
        F._FUNCTORIALITY_USE_TENSOR_COEFF_RESULT_CACHE[] = true
        F._FUNCTORIALITY_USE_DIRECT_COEFF_TRIPLETS[] = true
        F._FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[] = 0
        with_fields(FIELDS_FULL) do field
            F._clear_functoriality_caches!()
            K = CM.coeff_type(field)
            same(A, B) = field isa CM.RealField ?
                isapprox(A, B; atol=field.atol, rtol=field.rtol) : A == B
            P = chain_poset(1)
            M = MD.PModule{K}(P, [2], Dict{Tuple{Int,Int},Matrix{K}}(); field=field)
            cache = DF.HomSystemCache(K)
            H = DF.hom_with_cache(M, M; cache=cache)
            identity = CM.eye(field, 2)
            swap = K[0 1; 1 0]
            f = MD.PMorphism(M, M, [identity])
            g = MD.PMorphism(M, M, [swap])
            # Simulate an object-ID collision by putting the old live entry
            # under the new key. No garbage-collector timing is involved.
            for (store, action, expected) in
                ((cache.precompose, DF.precompose_matrix_cached, kron(transpose(swap), identity)),
                 (cache.postcompose, DF.postcompose_matrix_cached, kron(identity, swap)))
                action(H, H, f; cache=cache)
                store[DF._cache_key3(H, H, g)] = store[DF._cache_key3(H, H, f)]
                actual = action(H, H, g; cache=cache)
                @test same(H.basis_matrix * actual, expected * H.basis_matrix)
                @test action(H, H, g; cache=cache) === actual
                @test store[DF._cache_key3(H, H, g)].value === actual
            end
            # Replacing a basis leaves the HomSpace object's identity intact,
            # but changes the coordinate matrix that the cache must return.
            H.basis_matrix = H.basis_matrix[:, [2,1,3,4]]
            actual = DF.precompose_matrix_cached(H, H, g; cache=cache)
            @test same(H.basis_matrix * actual, kron(transpose(swap), identity) * H.basis_matrix)

            A, B = copy(identity), copy(swap)
            first_plan = F._cached_particular_solve_plan(field, A, A, nothing)
            key_a = F._SupportSolvePlanKey(UInt(objectid(A)), UInt(0), 2, 2)
            key_b = F._SupportSolvePlanKey(UInt(objectid(B)), UInt(0), 2, 2)
            F._SUPPORT_SOLVE_PLAN_CACHE[key_b] = F._SUPPORT_SOLVE_PLAN_CACHE[key_a]
            second_plan = F._cached_particular_solve_plan(field, B, B, nothing)
            rhs = reshape(K[1,0], 2, 1)
            @test same(F._solve_particular(first_plan, rhs), rhs)
            @test same(F._solve_particular(second_plan, rhs), reshape(K[0,1], 2, 1))
            @test F._cached_particular_solve_plan(field, B, B, nothing) === second_plan
            # A support-hash collision must compare the actual selected columns.
            wide = K[1 0 1; 0 1 1]
            left, right = [1,2], [2,3]
            F._cached_particular_solve_plan(field, wide[:,left], wide, left)
            key_left = F._SupportSolvePlanKey(UInt(objectid(wide)), F._support_hash(left), 2, 2)
            key_right = F._SupportSolvePlanKey(UInt(objectid(wide)), F._support_hash(right), 2, 2)
            F._SUPPORT_SOLVE_PLAN_CACHE[key_right] = F._SUPPORT_SOLVE_PLAN_CACHE[key_left]
            right_plan = F._cached_particular_solve_plan(field, wide[:,right], wide, right)
            @test same(F._solve_particular(right_plan, rhs), reshape(K[-1,1], 2, 1))
            if field isa CM.RealField
                other_field = CM.RealField(Float64; rtol=1e-5, atol=1e-8)
                other_plan = F._cached_particular_solve_plan(other_field, B, B, nothing)
                @test other_plan !== second_plan
                @test other_plan.field === other_field
                @test same(F._solve_particular(other_plan, rhs), reshape(K[0,1], 2, 1))
            end

            bases, offsets = [1,1], [0,2,4]
            ca, cb = sparse(identity), sparse(swap)
            tensor(coeff; gens=bases) = F._tensor_map_on_tor_chains_from_projective_coeff(
                M, gens, gens, offsets, offsets, coeff; cache=cache)
            @test same(tensor(ca), kron(identity, identity))
            coeff_key_a = F._coeff_plan_key(M, bases, bases, ca, 0x02)
            tensor_key_a = F._tensor_coeff_cache_key(M, bases, bases, offsets, offsets, ca)
            # Force collisions through all three levels: result, owner-specific
            # tensor plan, and shared sparse-pattern plan.
            for gens in (bases, copy(bases))
                coeff_key_b = F._coeff_plan_key(M, gens, gens, cb, 0x02)
                tensor_key_b = F._tensor_coeff_cache_key(M, gens, gens, offsets, offsets, cb)
                F._COEFF_MAP_PLAN_CACHE[coeff_key_b] = F._COEFF_MAP_PLAN_CACHE[coeff_key_a]
                for store in (F._TENSOR_COEFF_PLAN_CACHES, F._TENSOR_COEFF_RESULT_CACHES)
                    store[cache][tensor_key_b] = store[cache][tensor_key_a]
                end
                result = tensor(cb; gens=gens)
                @test same(result, kron(swap, identity))
                @test tensor(cb; gens=gens) === result
            end
            # Equal sparsity hashes do not imply equal active coefficient
            # patterns when sparse storage contains explicit zeros.
            explicit_zero = copy(ca)
            explicit_zero.nzval[1] = zero(K)
            @test same(tensor(explicit_zero), kron(Matrix(explicit_zero), identity))
            # Distinct matrices with the same active pattern still share the
            # read-only pattern plan, preserving structural reuse.
            p1 = F._tensor_coeff_plan(M, bases, bases, cb)
            p2 = F._tensor_coeff_plan(M, bases, bases, copy(cb))
            @test p1 === p2
            # Immutable module wrappers may be reboxed at every function call.
            # Force collection while their mutable storage remains live: a hit
            # must keep returning the already-published matrix, not rebuild it.
            pre_gc = DF.precompose_matrix_cached(H, H, g; cache=cache)
            post_gc = DF.postcompose_matrix_cached(H, H, g; cache=cache)
            tensor_gc = tensor(cb)
            support_gc = F._cached_particular_solve_plan(field, B, B, nothing)
            GC.@preserve M H g cb B bases offsets begin
                GC.gc(true)
                @test DF.precompose_matrix_cached(H, H, g; cache=cache) === pre_gc
                @test DF.postcompose_matrix_cached(H, H, g; cache=cache) === post_gc
                @test tensor(cb) === tensor_gc
                @test F._tensor_coeff_plan(M, bases, bases, cb) === p1
                @test F._cached_particular_solve_plan(field, B, B, nothing) === support_gc
            end
            module_entry = DF._identity_cache_entry((M,), nothing, nothing)
            other_poset = chain_poset(1)
            changed_fields = map(fieldnames(typeof(M))) do name
                name === :Q ? other_poset : getfield(M, name)
            end
            changed_poset_module = typeof(M)(changed_fields...)
            @test changed_poset_module.map_compose === M.map_compose
            @test !DF._identity_cache_matches(module_entry, (changed_poset_module,))
            if field isa CM.RealField
                changed_fields = map(fieldnames(typeof(M))) do name
                    name === :field ? CM.RealField(Float64; rtol=1e-4, atol=1e-6) : getfield(M, name)
                end
                changed_field_module = typeof(M)(changed_fields...)
                @test changed_field_module.map_compose === M.map_compose
                @test !DF._identity_cache_matches(module_entry, (changed_field_module,))
            end
            # Matrix views are also immutable and share a parent; their exact
            # view identity, not just the parent's identity, distinguishes maps.
            parent = K[1 0 0 1; 0 1 1 0]
            left_view = @view parent[:,1:2]
            right_view = @view parent[:,3:4]
            left_tensor, right_tensor = tensor(left_view), tensor(right_view)
            @test same(left_tensor, kron(identity, identity))
            @test same(right_tensor, kron(swap, identity))
            GC.@preserve M parent left_view right_view begin
                GC.gc(true)
                @test tensor(left_view) === left_tensor
                @test tensor(right_view) === right_tensor
            end
            DF.clear_hom_system_cache!(cache)
            @test isempty(cache.precompose) && isempty(cache.postcompose)
        end
    finally
        F._FUNCTORIALITY_USE_COEFF_PLAN_CACHE[] = old_coeff
        F._FUNCTORIALITY_USE_TENSOR_COEFF_RESULT_CACHE[] = old_result
        F._FUNCTORIALITY_USE_DIRECT_COEFF_TRIPLETS[] = old_direct
        F._FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[] = old_min
        F._clear_functoriality_caches!()
    end
end

@testset "A10 Ext maps share Hom spaces with independent input scratch" begin
    with_fields(FIELDS_FULL) do field
        K = CM.coeff_type(field)
        same(A, B) = field isa CM.RealField ?
            isapprox(A, B; atol=field.atol, rtol=field.rtol) : A == B
        P = chain_poset(2)
        M = _chain_module(P, [2, 0], zeros(K, 0, 2), field)
        N = _chain_module(P, [0, 2], zeros(K, 2, 0), field)
        resolution_cache = CM.ResolutionCache()
        E = DF.ExtInjective(M, N, TO.DerivedFunctorOptions(maxdeg=1, model=:injective); cache=resolution_cache)
        @test DF.dim(E, 0) == 0
        @test DF.dim(E, 1) == 4
        values = [K[1 CM.coerce(field, j); CM.coerce(field, j + 1) 1] for j in 1:8]
        serial = [DF.ext_map_first(E, E, MD.PMorphism(M, M, [A, zeros(K, 0, 0)]); t=1) for A in values]
        tasks = [_a10_spawn(j) do
            f = _a10_callback_morphism(M, [values[j], zeros(K, 0, 0)], yield)
            DF.ext_map_first(E, E, f; t=1)
        end for j in eachindex(values)]
        actual = fetch.(tasks)
        for j in eachindex(values)
            @test same(actual[j], serial[j])
            # On this quiver Ext^1(S1^2,S2^2)=Hom(K^2,K^2), with
            # first-argument action X -> X*A. Check representatives directly.
            H = E.homs[2]
            reps = E.cohom[2].Hrep
            @test same(H.basis_matrix * reps * actual[j],
                       kron(transpose(values[j]), CM.eye(field, 2)) * H.basis_matrix * reps)
        end

        # The joint degree-zero/one lift must also work when its resolutions
        # are the same object. For S1^2 on 1<2, lifting the original augmentation
        # gives identity coefficient matrices and a nonzero d1 square.
        res = DF.projective_resolution(M, TO.ResolutionOptions(maxlen=1))
        active = [[findall(v -> FF.leq(P, v, u), res.gens[k]) for u in 1:2] for k in 1:2]
        alpha_parts = [Vector{K}(res.aug.comps[u][:, findfirst(==(i), active[1][u])])
                       for (i, u) in enumerate(res.gens[1])]
        F0, F1 = DF.Functoriality._solve_projective_q0_q1_joint_coeff(res, res, alpha_parts, active)
        @test same(Matrix(F0), CM.eye(field, 2))
        @test same(Matrix(F1), CM.eye(field, 2))
        for u in 1:2
            @test same(res.aug.comps[u] * F0[active[1][u], active[1][u]], res.aug.comps[u])
        end
        @test !iszero(res.d_mat[1])
        @test same(res.d_mat[1] * F1, F0 * res.d_mat[1])
    end
end

@testset "A11 derived caches and nested threaded constructors preserve oracles" begin
    with_fields(FIELDS_FULL) do field
        K = CM.coeff_type(field)
        P = chain_poset(2)
        Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)))
        sources = [_chain_module(P, [r, 0], zeros(K, 0, r), field) for r in 1:3]
        targets = [_chain_module(P, [0, r], zeros(K, r, 0), field) for r in 1:3]
        rights = [_chain_module(Pop, [0, r], zeros(K, 0, r), field) for r in 1:3]
        rc = CM.ResolutionCache()
        hc = DF.HomSystemCache(K)
        tasks = [_a10_spawn(j) do
            r = mod1(j, 3)
            M, N, R = sources[r], targets[r], rights[r]
            yield()
            H = DF.hom_with_cache(M, M; cache=hc)
            proj = DF.Ext(M, N, TO.DerivedFunctorOptions(maxdeg=1, model=:projective); cache=rc)
            inj = DF.ExtInjective(M, N, TO.DerivedFunctorOptions(maxdeg=1, model=:injective); cache=rc)
            tor = DF.Tor(R, M, TO.DerivedFunctorOptions(maxdeg=1); cache=rc)
            (DF.dim(H), DF.dim(proj, 0), DF.dim(proj, 1), DF.dim(inj, 1), DF.dim(tor, 1))
        end for j in 1:12]
        for (j, result) in enumerate(fetch.(tasks))
            r = mod1(j, 3)
            @test result == (r*r, 0, r*r, r*r, r*r)
        end
        DF.clear_hom_system_cache!(hc)
        @test isempty(hc.hom)
        @test DF.dim(DF.hom_with_cache(sources[3], sources[3]; cache=hc)) == 9
        # 70 diagonal resolution coefficients force the inner parallel Hom
        # differential assembly, invoked from an already spawned constructor.
        # Ext^1(S1^70,S2)=K^70; the other displayed degrees vanish.
        large_source = _chain_module(P, [70, 0], zeros(K, 0, 70), field)
        nested = fetch(_a10_spawn(2) do
            E = DF.Ext(large_source, targets[1],
                       TO.DerivedFunctorOptions(maxdeg=1, model=:projective); cache=rc)
            (DF.dim(E, 0), DF.dim(E, 1))
        end)
        @test nested == (0, 70)
    end
end

@testset "A11 algebra caches publish concurrent products and survive replacement" begin
    with_fields(FIELDS_FULL) do field
        K = CM.coeff_type(field)
        same(A, B) = field isa CM.RealField ?
            isapprox(A, B; atol=field.atol, rtol=field.rtol) : A == B

        # Algebra products share these lazy additive models. This quotient has
        # cycles (e1+e4,e2,e3), boundary e1+e4 and quotient basis (e2,e3).
        CC = TO.ChainComplexes
        Z = K[1 0 0; 0 1 0; 0 0 1; 1 0 0]
        boundary = Z[:, 1:1]
        coh = CC._cohomology_data_from_bases(K, 1, 4, Z, boundary; lazy_reps=true)
        hom = CC._homology_data_from_bases(K, 1, 4, Z, boundary; lazy_reps=true)
        @test getfield(coh, :_Hrep) === nothing
        @test getfield(hom, :_Hrep) === nothing
        model_tasks = [_a10_spawn(j) do
            a, b = CM.coerce(field, j), CM.coerce(field, j + 1)
            z = Z * K[CM.coerce(field, 3), a, b]
            yield()
            cohc = vec(CC.coordinates(coh, z))
            homc = vec(CC.coordinates(hom, z))
            fastc = CC._cohomology_coordinates_vector(coh, z)
            plan = field isa CM.QQField ? CC._cohomology_coord_plan(coh) : nothing
            (CC.basis(coh), CC.basis(hom), cohc, homc, fastc, K[a, b], plan)
        end for j in 1:12]
        for (cohreps, homreps, cohc, homc, fastc, expected, plan) in fetch.(model_tasks)
            @test same(cohreps, Z[:, 2:3])
            @test same(homreps, Z[:, 2:3])
            @test cohreps === CC.basis(coh)
            @test homreps === CC.basis(hom)
            @test same(cohc, expected)
            @test same(homc, expected)
            @test same(fastc, expected)
            if plan !== nothing
                @test plan.rows === CC._cohomology_coord_plan(coh).rows
                @test plan.proj === CC._cohomology_coord_plan(coh).proj
            end
        end

        P = chain_poset(2)
        M = _chain_module(P, [1, 1], zeros(K, 1, 1), field)
        A = DF.ExtAlgebra(M, TO.DerivedFunctorOptions(maxdeg=2))
        # M=S1+S2 on 1<2: Ext dimensions are (2,1,0). Multiples of
        # the unit act by scalar multiplication on the degree-one generator.
        @test Tuple(DF.dim(A, t) for t in 0:2) == (2, 1, 0)
        pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
        @test isempty(DF.cached_product_degrees(A))
        ext_tasks = [_a10_spawn(j) do
            p, q = pairs[mod1(j, length(pairs))]
            a, b = CM.coerce(field, j), CM.coerce(field, j + 1)
            u = DF.element_coordinates(one(A))
            x, y = p == 0 ? a .* u : K[a], q == 0 ? b .* u : K[b]
            yield()
            actual = DF.multiply(A, p, x, q, y)
            expected = p + q == 0 ? (a * b) .* u : p + q == 1 ? K[a * b] : K[]
            table = DF.Algebras._ensure_mult_cache!(A, p, q)
            valid = DF.check_ext_algebra(A).valid
            count = DF.algebra_summary(A).cached_products
            (p, q, actual, expected, table, valid, count)
        end for j in 1:12]
        for (p, q, actual, expected, table, valid, count) in fetch.(ext_tasks)
            @test same(actual, expected)
            @test table === DF.Algebras._ensure_mult_cache!(A, p, q)
            @test valid
            @test 1 <= count <= 4
        end
        @test DF.cached_product_degrees(A) == pairs
        # Returned unit coordinates belong to the element, not the memo entry.
        u = one(A)
        saved_unit = copy(DF.element_coordinates(u))
        fill!(DF.element_coordinates(u), zero(K))
        @test DF.element_coordinates(one(A)) == saved_unit

        # An independent noncommutative oracle: Ext^0(K^2,K^2)=Mat_2(K).
        # The Yoneda product is ordinary composition F*G, in any Ext basis.
        Q = chain_poset(1)
        V = MD.PModule{K}(Q, [2], Dict{Tuple{Int,Int},Matrix{K}}(); field=field)
        B = DF.ExtAlgebra(V, TO.DerivedFunctorOptions(maxdeg=0))
        function endomorphism_coordinates(F)
            aug = B.E.res.aug
            f = MD.PMorphism(aug.dom, V, [F * aug.comps[1]])
            cocycle = DF.ExtTorSpaces._cochain_vector_from_morphism(B.E, 0, f)
            return DF.coordinates(B.E, 0, cocycle)
        end
        matrix_inputs = map(1:8) do j
            F = K[1 CM.coerce(field, j); 1 0]
            G = K[0 1; CM.coerce(field, j + 1) 1]
            (endomorphism_coordinates(F), endomorphism_coordinates(G),
             endomorphism_coordinates(F * G))
        end
        matrix_tasks = [_a10_spawn(j) do
            x, y, expected = matrix_inputs[j]
            yield()
            (DF.multiply(B, 0, x, 0, y), expected)
        end for j in eachindex(matrix_inputs)]
        for (actual, expected) in fetch.(matrix_tasks)
            @test same(actual, expected)
        end
        @test DF.algebra_summary(B).cached_products == 1

        # Resolving S2 on the opposite chain against S1+S2 gives C0=C1=K,
        # C2=0 and zero differential. Supply K[epsilon]/(epsilon^2), deg(epsilon)=1.
        Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)))
        Rop = _chain_module(Pop, [0, 1], zeros(K, 0, 1), field)
        T = DF.Tor(Rop, M, TO.DerivedFunctorOptions(maxdeg=2))
        @test Tuple(DF.dim(T, t) for t in 0:2) == (1, 1, 0)
        @test Tuple(T.dims[1:3]) == (1, 1, 0)
        @test all(iszero, T.dims[3:end])
        function dual_number_product(p, q)
            yield()
            return sparse(ones(K, T.dims[p + q + 1], T.dims[p + 1] * T.dims[q + 1]))
        end
        C = DF.TorAlgebra(T; mu_chain_gen=dual_number_product)
        @test isempty(DF.cached_product_degrees(C))
        tor_tasks = [_a10_spawn(j) do
            p, q = pairs[mod1(j, length(pairs))]
            a, b = CM.coerce(field, j), CM.coerce(field, j + 1)
            table = DF.multiplication_matrix(C, p, q)
            actual = DF.element_coordinates(DF.multiply(C, DF.element(C, p, K[a]),
                                                          DF.element(C, q, K[b])))
            expected = p + q < 2 ? K[a * b] : K[]
            valid = DF.check_tor_algebra(C).valid
            summary = DF.algebra_summary(C)
            (p, q, table, actual, expected, valid, summary)
        end for j in 1:12]
        for (p, q, table, actual, expected, valid, summary) in fetch.(tor_tasks)
            @test same(actual, expected)
            @test table === DF.multiplication_matrix(C, p, q)
            @test valid
            @test 1 <= summary.cached_chain_products <= 4
            @test 1 <= summary.cached_homology_products <= 4
        end
        @test DF.cached_product_degrees(C) == pairs

        # Generators may query another degree, including from a child task.
        # Holding the algebra lock across the callback would deadlock this call.
        nested = DF.TorAlgebra(T)
        DF.set_chain_product_generator!(nested, (p, q) -> begin
            if (p, q) == (0, 0)
                inner = fetch(_a10_spawn(() -> DF.multiplication_matrix(nested, 0, 1), 2))
                @assert same(inner, ones(K, 1, 1))
            end
            dual_number_product(p, q)
        end)
        @test same(DF.multiplication_matrix(nested, 0, 0), ones(K, 1, 1))
        @test DF.cached_product_degrees(nested) == [(0, 0), (0, 1)]

        DF.set_chain_product_generator!(nested, (p, q) -> error("A11 injected Tor generator failure"))
        @test_throws ErrorException DF.multiplication_matrix(nested, 0, 0)
        @test isempty(DF.cached_product_degrees(nested))
        DF.set_chain_product_generator!(nested, dual_number_product)
        @test same(DF.multiplication_matrix(nested, 0, 0), ones(K, 1, 1))

        # Replacement while a generator is suspended: the old active query may
        # finish, but it must not repopulate either cache in the new generation.
        started, resume = Channel{Nothing}(1), Channel{Nothing}(1)
        changing = DF.TorAlgebra(T; mu_chain_gen=(p, q) -> begin
            put!(started, nothing)
            take!(resume)
            dual_number_product(p, q)
        end)
        old_query = _a10_spawn(() -> DF.multiplication_matrix(changing, 0, 0), 1)
        take!(started)
        two = CM.coerce(field, 2)
        DF.set_chain_product_generator!(changing, (p, q) -> two .* dual_number_product(p, q))
        replacement = DF.multiplication_matrix(changing, 0, 0)
        put!(resume, nothing)
        @test same(fetch(old_query), ones(K, 1, 1))
        @test same(replacement, fill(two, 1, 1))
        @test DF.multiplication_matrix(changing, 0, 0) === replacement
        @test same(changing.mu_chain[(0, 0)], fill(two, 1, 1))
        DF.set_chain_product!(changing, 0, 0, spzeros(K, 1, 1))
        replaced = DF.multiplication_matrix(changing, 0, 0)
        @test iszero(replaced)
        @test DF.multiplication_matrix(changing, 0, 0) === replaced

        # Each algebra owns its map dictionary, even if initialized from the
        # same input. Read-only matrix payloads may still be shared.
        maps = Dict((0, 0) => sparse(ones(K, 1, 1)))
        left, right = DF.TorAlgebra(T; mu_chain=maps), DF.TorAlgebra(T; mu_chain=maps)
        DF.set_chain_product!(left, 0, 0, spzeros(K, 1, 1))
        @test iszero(DF.multiplication_matrix(left, 0, 0))
        @test same(DF.multiplication_matrix(right, 0, 0), ones(K, 1, 1))
        @test same(maps[(0, 0)], ones(K, 1, 1))
    end
end

@testset "A70 Tor products descend, satisfy DGA identities and respect induced maps" begin
    with_fields(FIELDS_FULL) do field
        K = CM.coeff_type(field)
        same(A, B) = field isa CM.RealField ?
            isapprox(A, B; atol=field.atol, rtol=field.rtol) : A == B
        P = chain_poset(2)
        Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)))
        R = _chain_module(Pop, [0, 1], zeros(K, 0, 1), field)
        edge = K[1 0; -1 0]
        L = _chain_module(P, [2, 2], edge, field)
        T = DF.Tor(R, L, TO.DerivedFunctorOptions(model=:first, maxdeg=2))
        @test Tuple(T.dims[1:3]) == (2, 2, 0)
        @test Tuple(DF.dim(T, s) for s in 0:2) == (1, 1, 0)
        D = Matrix(T.bd[1])
        @test same(D, edge) || same(D, -edge)

        # An explicitly supplied DGA, not a purported canonical Tor product:
        # C0 = K[v]/(v^2), C1 its regular bimodule with basis h,z=v*h=h*v;
        # d(h)=v and d(z)=0. Products of degree-one elements vanish.
        # u=e2 and v=e1-e2 in C0; orient h so that d(h)=v.
        u, v = K[0, 1], K[1, -1]
        h, z = K[D[1, 1], 0], K[0, 1]
        bases = (hcat(u, v), hcat(h, z))
        inverse_bases = (K[1 1; 1 0], K[D[1, 1] 0; 0 1])
        @test same(D * h, v)
        @test same(D * z, zeros(K, 2))
        function product(p, q, x, y; wrong_sign=false)
            p + q > 1 && return zeros(K, T.dims[p + q + 1])
            a, b = inverse_bases[p + 1] * x
            c, d = inverse_bases[q + 1] * y
            # Reversing the right action of v leaves the homology product
            # unchanged, but violates Leibniz on (h,h) in odd characteristic.
            second = wrong_sign && (p, q) == (1, 0) ? -a*d + b*c : a*d + b*c
            return bases[p + q + 1] * K[a*c, second]
        end
        function product_matrix(p, q; wrong_sign=false)
            out = zeros(K, T.dims[p + q + 1], T.dims[p + 1] * T.dims[q + 1])
            for i in 1:T.dims[p + 1], j in 1:T.dims[q + 1]
                x, y = zeros(K, T.dims[p + 1]), zeros(K, T.dims[q + 1])
                x[i], y[j] = one(K), one(K)
                out[:, (i - 1) * length(y) + j] = product(p, q, x, y; wrong_sign)
            end
            return sparse(out)
        end
        maps = Dict((p, q) => product_matrix(p, q) for p in 0:2 for q in 0:(2-p))
        unit_coords = DF.coordinates(T, 0, u)
        A = DF.TorAlgebra(T; mu_chain=maps, unit_coords=unit_coords)
        @test DF.check_tor_algebra(A).valid
        report = DF.check_tor_algebra(A; algebraic=true)
        @test report.valid
        @test report.checks == :algebraic
        @test report.verified_through == 2
        @test same(DF.element_coordinates(one(A)), unit_coords)
        @test same(DF.element_coordinates(DF.unit(A)), unit_coords)
        for (p, q) in ((0, 0), (0, 1), (1, 0), (1, 1))
            expected = p + q < 2 ? ones(K, 1, 1) : zeros(K, 0, 1)
            @test same(DF.multiplication_matrix(A, p, q), expected)
        end

        # Check the supplied structure independently on every basis vector.
        # In particular d(h*h)=v*h-h*v=z-z checks a nonzero sign cancellation.
        for p in 0:1, q in 0:1, x in eachcol(bases[p + 1]), y in eachcol(bases[q + 1])
            lhs = p + q == 1 ? D * product(p, q, x, y) : zeros(K, 2)
            rhs = zeros(K, 2)
            p == 1 && (rhs += product(p - 1, q, D * x, y))
            q == 1 && (rhs += (isodd(p) ? -one(K) : one(K)) .* product(p, q - 1, x, D * y))
            @test same(lhs, rhs)
        end
        for p in 0:1, q in 0:1, r in 0:1
            p + q + r > 2 && continue
            for x in eachcol(bases[p + 1]), y in eachcol(bases[q + 1]), w in eachcol(bases[r + 1])
                lhs = product(p + q, r, product(p, q, x, y), w)
                rhs = product(p, q + r, x, product(q, r, y, w))
                @test same(lhs, rhs)
            end
        end
        for p in 0:1, x in eachcol(bases[p + 1])
            @test same(product(0, p, u, x), x)
            @test same(product(p, 0, x, u), x)
        end

        # Change both degree-zero representatives by independent boundaries.
        # Check mixed products as well as the nonzero degree-zero product.
        for a in (zero(K), one(K), -one(K)), b in (zero(K), one(K), -one(K))
            @test same(DF.coordinates(T, 0, product(0, 0, u + a*v, u + b*v)), unit_coords)
            @test same(DF.coordinates(T, 1, product(0, 1, u + a*v, z)), DF.coordinates(T, 1, z))
            @test same(DF.coordinates(T, 1, product(1, 0, z, u + b*v)), DF.coordinates(T, 1, z))
        end

        # Actual module morphisms inducing algebra maps: u->u, v->a*v,
        # h->a*h, z->a^2*z. Zero also gives a noninvertible algebra map.
        for a in (zero(K), -one(K), CM.coerce(field, 2))
            G0 = bases[1] * K[1 0; 0 a] * inverse_bases[1]
            G1 = bases[2] * K[a 0; 0 a*a] * inverse_bases[2]
            @test same(G0 * D, D * G1)
            g = MD.PMorphism(L, L, [G1, G0])
            induced = [DF.tor_map_second(T, T, g; s=s) for s in 0:2]
            @test same(induced[1], ones(K, 1, 1))
            @test same(induced[2], fill(a*a, 1, 1))
            @test same(induced[1] * unit_coords, unit_coords)
            for p in 0:1, q in 0:1
                mu = DF.multiplication_matrix(A, p, q)
                @test same(induced[p+q+1] * mu, mu * kron(induced[p+1], induced[q+1]))
            end
        end

        # A bad supplied unit must not be accepted merely because its size fits.
        bad_unit = DF.TorAlgebra(T; mu_chain=maps, unit_coords=zeros(K, 1))
        @test !DF.check_tor_algebra(bad_unit; algebraic=true).valid
        @test_throws ArgumentError DF.check_tor_algebra(bad_unit; algebraic=true, throw=true)
        @test_throws ArgumentError one(bad_unit)
        @test_throws ArgumentError one(DF.TorAlgebra(T; mu_chain=maps))

        if !iszero(CM.coerce(field, 2))
            bad_sign_maps = copy(maps)
            bad_sign_maps[(1, 0)] = product_matrix(1, 0; wrong_sign=true)
            bad_sign = DF.TorAlgebra(T; mu_chain=bad_sign_maps, unit_coords=unit_coords)
            # Descent succeeds, so only checking homology cannot detect this.
            @test same(DF.multiplication_matrix(bad_sign, 1, 0), ones(K, 1, 1))
            @test !DF.check_tor_algebra(bad_sign; algebraic=true).valid
            @test_throws ArgumentError DF.check_tor_algebra(bad_sign; algebraic=true, throw=true)
        end

        # Explicitly test each descent condition rather than relying on d^2=0.
        diagonal = sparse(K[1 0 0 0; 0 0 0 1])
        bad_diagonal = DF.TorAlgebra(T; mu_chain=Dict((0, 0) => diagonal))
        @test same(diagonal * kron(v, u), -u)
        @test !iszero(sum(diagonal * kron(v, u)))
        @test_throws ArgumentError DF.multiplication_matrix(bad_diagonal, 0, 0)
        @test isempty(bad_diagonal.mu_H_cache)
        bad_generated = DF.TorAlgebra(T; mu_chain_gen=(p, q) -> diagonal)
        @test_throws ArgumentError DF.multiplication_matrix(bad_generated, 0, 0)
        @test isempty(bad_generated.mu_H_cache)
        DF.set_chain_product!(A, 0, 0, diagonal)
        @test_throws ArgumentError DF.multiplication_matrix(A, 0, 0)
        DF.set_chain_product!(A, 0, 0, maps[(0, 0)])
        @test same(DF.multiplication_matrix(A, 0, 0), ones(K, 1, 1))
        # mu(x,y)=epsilon(x)*y_1*u kills B x Z but not Z x B.
        right_bad_map = sparse(u * transpose(kron(K[1, 1], K[1, 0])))
        @test same(right_bad_map * kron(v, u), zeros(K, 2))
        @test same(right_bad_map * kron(u, v), u)
        right_bad = DF.TorAlgebra(T; mu_chain=Dict((0, 0) => right_bad_map))
        @test_throws ArgumentError DF.multiplication_matrix(right_bad, 0, 0)
        # A product sending the cycles (u,z) to h fails cycle preservation.
        cycle_bad_map = sparse(h * transpose(kron(K[1, 1], K[0, 1])))
        @test same(cycle_bad_map * kron(u, z), h)
        cycle_bad = DF.TorAlgebra(T; mu_chain=Dict((0, 1) => cycle_bad_map))
        @test_throws ArgumentError DF.multiplication_matrix(cycle_bad, 0, 1)

        # Requesting only H0 still requires the incoming C1 boundary. This is
        # the originally reported counterexample with its guard degree intact.
        Lsmall = _chain_module(P, [1, 2], reshape(v, 2, 1), field)
        short = DF.Tor(R, Lsmall, TO.DerivedFunctorOptions(model=:first, maxdeg=0))
        @test DF.dim(short, 0) == 1
        @test short.dims[1:2] == [2, 1]
        @test_throws ArgumentError DF.multiplication_matrix(
            DF.TorAlgebra(short; mu_chain=Dict((0, 0) => diagonal)), 0, 0)
        truncated = DF.Tor(R, L, TO.DerivedFunctorOptions(model=:first, maxdeg=0))
        partial = DF.TorAlgebra(truncated; mu_chain=Dict((0, 0) => maps[(0, 0)]))
        @test same(DF.multiplication_matrix(partial, 0, 0), ones(K, 1, 1))
        @test !DF.check_tor_algebra(partial; algebraic=true).valid
        guarded = DF.TorAlgebra(truncated; mu_chain=Dict(key => maps[key] for key in ((0, 0), (0, 1), (1, 0))),
                               unit_coords=DF.coordinates(truncated, 0, u))
        @test DF.check_tor_algebra(guarded; algebraic=true).valid
    end
end

@testset "A70 Tor tensor order and strict algebra axioms on matrix algebras" begin
    with_fields(FIELDS_FULL) do field
        K = CM.coeff_type(field)
        same(A, B) = field isa CM.RealField ?
            isapprox(A, B; atol=field.atol, rtol=field.rtol) : A == B
        P = chain_poset(1)
        U = MD.PModule{K}(P, [1], Dict{Tuple{Int,Int},Matrix{K}}(); field)
        V = MD.PModule{K}(P, [4], Dict{Tuple{Int,Int},Matrix{K}}(); field)
        matrix_basis = [reshape(CM.eye(field, 4)[:, i], 2, 2) for i in 1:4]
        mu = zeros(K, 4, 16)
        for i in 1:4, j in 1:4
            mu[:, (i - 1)*4 + j] = vec(matrix_basis[i] * matrix_basis[j])
        end
        X, Y = K[1 2; 3 4], K[0 1; 1 1]
        @test !same(X*Y, Y*X)
        G, Ginv = K[1 1; 0 1], K[1 -1; 0 1]
        conjugation = kron(transpose(Ginv), G)
        g = MD.PMorphism(V, V, [conjugation])
        # Both arguments give actual Tor functorial maps, not arbitrary matrices
        # merely declared to act on homology. Use both resolution models too.
        for model in (:first, :second), argument in (:first, :second)
            R, L = argument == :first ? (V, U) : (U, V)
            T = DF.Tor(R, L, TO.DerivedFunctorOptions(; model, maxdeg=0))
            e = DF.coordinates(T, 0, vec(CM.eye(field, 2)))
            A = DF.TorAlgebra(T; mu_chain=Dict((0, 0) => sparse(mu)), unit_coords=e)
            @test DF.check_tor_algebra(A; algebraic=true).valid
            x, y = DF.element(A, 0, DF.coordinates(T, 0, vec(X))), DF.element(A, 0, DF.coordinates(T, 0, vec(Y)))
            @test same(DF.element_coordinates(DF.multiply(A, x, y)), DF.coordinates(T, 0, vec(X*Y)))
            @test same(DF.element_coordinates(DF.multiply(A, y, x)), DF.coordinates(T, 0, vec(Y*X)))
            @test same(DF.element_coordinates(DF.multiply(A, one(A), x)), DF.element_coordinates(x))
            @test same(DF.element_coordinates(DF.multiply(A, x, one(A))), DF.element_coordinates(x))
            F = argument == :first ? DF.tor_map_first(T, T, g; s=0) : DF.tor_map_second(T, T, g; s=0)
            M = DF.multiplication_matrix(A, 0, 0)
            @test same(F*M, M*kron(F, F))
            @test same(F*e, e)
            @test same(F*DF.element_coordinates(x), DF.coordinates(T, 0, vec(G*X*Ginv)))
        end

        # A bilinear operation always descends on a complex concentrated at 0,
        # but need not be associative: (e*e)*e=e, e*(e*e)=0.
        W = MD.PModule{K}(P, [2], Dict{Tuple{Int,Int},Matrix{K}}(); field)
        T = DF.Tor(U, W, TO.DerivedFunctorOptions(maxdeg=0))
        nonassoc = sparse(K[0 0 1 0; 1 0 0 0])
        A = DF.TorAlgebra(T; mu_chain=Dict((0, 0) => nonassoc))
        @test same(DF.multiplication_matrix(A, 0, 0), nonassoc)
        @test !DF.check_tor_algebra(A; algebraic=true).valid
        @test_throws ArgumentError DF.check_tor_algebra(A; algebraic=true, throw=true)
    end
end

@testset "A70 Tor product input contracts and configured numerical tolerance" begin
    @test !isdefined(DF, :trivial_tor_product_generator)
    with_fields(FIELDS_FULL) do field
        K = CM.coeff_type(field)
        P = chain_poset(1)
        U = MD.PModule{K}(P, [1], Dict{Tuple{Int,Int},Matrix{K}}(); field)
        T = DF.Tor(U, U, OPT.DerivedFunctorOptions(maxdeg=0))
        mu = sparse(ones(K, 1, 1))
        @test_throws ArgumentError DF.TorAlgebra(T; mu_chain=Dict((-1, 0) => mu))
        @test_throws ArgumentError DF.TorAlgebra(T; mu_chain=Dict((0, 0) => spzeros(K, 2, 1)))
        @test_throws ArgumentError DF.TorAlgebra(T; unit_coords=K[])
        A = DF.TorAlgebra(T; mu_chain=Dict((0, 0) => mu), unit_coords=K[1])
        before = DF.cached_product_degrees(A)
        @test DF.check_tor_algebra(A; algebraic=true).valid
        @test DF.cached_product_degrees(A) == before
        @test isempty(A.mu_H_cache)
        @test DF.check_tor_algebra(A).checks == :structural
        @test DF.check_tor_algebra(A).verified_through === nothing
        stale = DF.TorAlgebra(T; mu_chain=Dict((0, 0) => mu))
        stale.mu_H_cache[(0, 0)] = zeros(K, 1, 1)
        @test DF.check_tor_algebra(stale).valid
        @test !DF.check_tor_algebra(stale; algebraic=true).valid
        x = one(A)
        x.coords[1] = zero(K)
        @test DF.unit(A).coords == K[1]
        @test_throws ArgumentError DF.element(A, -1, K[1])
        @test_throws DimensionMismatch DF.element(A, 0, K[])
        @test_throws ArgumentError DF.multiplication_matrix(A, 1, 0)
        @test_throws ArgumentError DF.set_chain_product!(A, -1, 0, mu)
        @test_throws ArgumentError DF.set_chain_product!(A, 0, 0, spzeros(K, 2, 1))
        @test_throws ArgumentError DF.multiply(A, one(A), DF.element(DF.TorAlgebra(T), 0, K[1]))
        @test_throws ArgumentError DF.multiplication_matrix(DF.TorAlgebra(T), 0, 0)
        @test_throws ArgumentError DF.multiplication_matrix(
            DF.TorAlgebra(T; mu_chain_gen=(p,q) -> ones(K, 1, 1)), 0, 0)
        malformed = DF.TorAlgebra(T)
        malformed.mu_H_cache[(-1, 0)] = ones(K, 1, 1)
        @test !DF.check_tor_algebra(malformed).valid
        @test_throws ArgumentError DF.check_tor_algebra(malformed; throw=true)
    end

    # The same almost-cycle is accepted or rejected using the declared field
    # tolerance, not a default tolerance reconstructed from Float64.
    for (tol, accepted) in ((1e-14, false), (1e-8, true))
        field = CM.RealField(Float64; atol=tol, rtol=tol)
        P = chain_poset(2)
        Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)))
        R = _chain_module(Pop, [0, 1], zeros(0, 1), field)
        L = _chain_module(P, [2, 2], [1.0 0; -1 0], field)
        T = DF.Tor(R, L, OPT.DerivedFunctorOptions(maxdeg=1))
        z = [1e-11, 1.0]
        mu = sparse(z * transpose(kron([1.0, 1.0], [0.0, 1.0])))
        A = DF.TorAlgebra(T; mu_chain=Dict((0, 1) => mu))
        if accepted
            @test isapprox(DF.multiplication_matrix(A, 0, 1), ones(1, 1); atol=tol, rtol=tol)
        else
            @test_throws ArgumentError DF.multiplication_matrix(A, 0, 1)
        end
    end

    # Finite coefficients can still overflow during products or the comparison
    # of two associative bracketings. Equal infinities are not a certificate.
    field = CM.RealField(Float64)
    P = chain_poset(1)
    U = MD.PModule{Float64}(P, [1], Dict{Tuple{Int,Int},Matrix{Float64}}(); field)
    T = DF.Tor(U, U, OPT.DerivedFunctorOptions(maxdeg=0))
    A = DF.TorAlgebra(T; mu_chain=Dict((0, 0) => sparse(reshape([1e308], 1, 1))))
    @test DF.multiplication_matrix(A, 0, 0) == reshape([1e308], 1, 1)
    @test !DF.check_tor_algebra(A; algebraic=true).valid
    @test_throws ArgumentError DF.multiply(A, DF.element(A, 0, [2.0]), DF.element(A, 0, [2.0]))
    @test_throws ArgumentError DF.TorAlgebra(T; mu_chain=Dict((0, 0) => sparse(reshape([NaN], 1, 1))))
    @test_throws ArgumentError DF.element(A, 0, [Inf])
end

@testset "A58 particular solve consumers preserve field and residual contracts" begin
    with_fields(FIELDS_FULL) do field
        K = CM.coeff_type(field)
        same(A, B) = field isa CM.RealField ?
            isapprox(A, B; atol=field.atol, rtol=field.rtol) : A == B
        s = field isa CM.RealField ? K(0.01) : one(K)
        A = K[s 0; s s; 0 0]
        X = K[1 -1; -1 1]
        Y = A * X
        functor_plan = DF.Functoriality._particular_solve_plan(field, A)
        downset_plan = DF.Resolutions._downset_particular_solve_plan(field, A)
        for solve in (B -> DF.Utils.solve_particular(field, A, B),
                      B -> DF.Functoriality._solve_particular(functor_plan, B),
                      B -> DF.Resolutions._downset_solve_particular(downset_plan, B))
            @test same(solve(Y), X)
            @test same(solve(zeros(K, 3, 2)), zeros(K, 2, 2))
            @test size(solve(zeros(K, 3, 0))) == (2, 0)
            bad = copy(Y)
            bad[3, 2] = one(K)
            @test_throws ErrorException solve(bad)
        end

        # A free coefficient variable remains free even if a RHS has a larger
        # column norm. Both direct and cached-plan routes solve all RHS columns.
        U = K[s 1 0; 0 0 0]
        B = K[1 -1; 0 0]
        Xdirect = DF.Utils.solve_particular(field, U, B)
        Xplanned = DF.Functoriality._solve_particular(
            DF.Functoriality._particular_solve_plan(field, U), B)
        @test same(U * Xdirect, B)
        @test same(Xplanned, Xdirect)
        @test iszero(Xdirect[3, :])
        @test count(row -> !iszero(row), eachrow(Xdirect)) == 1
    end
end

with_fields(FIELDS_FULL) do field
K = CM.coeff_type(field)
c(x) = CM.coerce(field, x)
same_coefficients(X, Y) = field isa CM.RealField ?
    isapprox(X, Y; atol=field.atol, rtol=field.rtol) : X == Y

@testset "Ext functoriality (projective model) in both arguments" begin
    P = chain_poset(2)
    # Simple at 1 and simple at 2 on P.
    S1 = IR.pmodule_from_fringe(one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 1); scalar=one(K), field=field))
    S2 = IR.pmodule_from_fringe(one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 2); scalar=one(K), field=field))

    # Build M = S1 oplus S1 so End(M) is noncommutative (Mat_2).
    M = MD.direct_sum(S1, S1)

    # Build N = S2 oplus S2 so End(N) is noncommutative (Mat_2).
    N = MD.direct_sum(S2, S2)

    EMN = DF.Ext(M, N, TO.DerivedFunctorOptions(maxdeg=2))

    # Nonvacuous check: Ext^1 should be 4 = 2*2 times Ext^1(S1,S2) (which is 1 on this poset).
    @test [TO.dim(EMN, t) for t in 0:2] == [0, 4, 0]

    # Two noncommuting endomorphisms of M at vertex 1 (dims there are 2).
    A = [c(1) c(1); c(0) c(1)]
    B = [c(1) c(0); c(1) c(1)]
    fA = endo_at_vertex(M, 1, A)
    fB = endo_at_vertex(M, 1, B)

    # Contravariant functoriality in the first argument:
    # Ext(fB o fA) = Ext(fA) o Ext(fB).
    F_A = TO.ext_map_first(EMN, EMN, fA; t=1)
    F_B = TO.ext_map_first(EMN, EMN, fB; t=1)
    F_BA = TO.ext_map_first(EMN, EMN, compose_morphism(fB, fA); t=1)
    @test same_coefficients(F_BA, F_A * F_B)

    # Two noncommuting endomorphisms of N at vertex 2 (dims there are 2).
    C = [c(2) c(1); c(0) c(1)]
    D = [c(1) c(0); c(1) c(2)]
    gC = endo_at_vertex(N, 2, C)
    gD = endo_at_vertex(N, 2, D)

    # Covariant functoriality in the second argument:
    # Ext(gD o gC) = Ext(gD) o Ext(gC).
    G_C = TO.ext_map_second(EMN, EMN, gC; t=1)
    G_D = TO.ext_map_second(EMN, EMN, gD; t=1)
    G_DC = TO.ext_map_second(EMN, EMN, compose_morphism(gD, gC); t=1)
    @test same_coefficients(G_DC, G_D * G_C)
end


@testset "Tor functoriality in both arguments" begin
    P = chain_poset(2)
    Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)))

    # L = simple at 1 on P (as in existing Tor-by-hand test).
    L = IR.pmodule_from_fringe(one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 1); scalar=one(K), field=field))

    # Rop = simple at 2 on P^op (as in existing Tor-by-hand test).
    Rop = IR.pmodule_from_fringe(one_by_one_fringe(Pop, FF.principal_upset(Pop, 2), FF.principal_downset(Pop, 2); scalar=one(K), field=field))

    # Make direct sums to get noncommuting endomorphisms.
    R2 = MD.direct_sum(Rop, Rop)
    L2 = MD.direct_sum(L, L)

    # Tor should be additive in each argument, so Tor_1 doubles here.
    T_R2_L = DF.Tor(R2, L, TO.DerivedFunctorOptions(maxdeg=3))
    T_R2_L2 = DF.Tor(R2, L2, TO.DerivedFunctorOptions(maxdeg=3))

    @test TO.dim(T_R2_L, 1) == 2
    @test TO.dim(T_R2_L2, 1) == 4

    # Noncommuting endomorphisms of R2 at vertex 2 (dims there are 2).
    A = [c(1) c(1); c(0) c(1)]
    B = [c(1) c(0); c(1) c(1)]
    fA = endo_at_vertex(R2, 2, A)
    fB = endo_at_vertex(R2, 2, B)

    # Covariant functoriality in the first argument:
    # Tor(fB o fA) = Tor(fB) o Tor(fA).
    F_A = TO.tor_map_first(T_R2_L, T_R2_L, fA; s=1)
    F_B = TO.tor_map_first(T_R2_L, T_R2_L, fB; s=1)
    F_BA = TO.tor_map_first(T_R2_L, T_R2_L, compose_morphism(fB, fA); s=1)
    @test same_coefficients(F_BA, F_B * F_A)

    # Noncommuting endomorphisms of L2 at vertex 1 (dims there are 2).
    C = [c(2) c(1); c(0) c(1)]
    D = [c(1) c(0); c(1) c(2)]
    gC = endo_at_vertex(L2, 1, C)
    gD = endo_at_vertex(L2, 1, D)

    # Covariant functoriality in the second argument:
    # Tor(gD o gC) = Tor(gD) o Tor(gC).
    G_C = TO.tor_map_second(T_R2_L2, T_R2_L2, gC; s=1)
    G_D = TO.tor_map_second(T_R2_L2, T_R2_L2, gD; s=1)
    G_DC = TO.tor_map_second(T_R2_L2, T_R2_L2, compose_morphism(gD, gC); s=1)
    @test same_coefficients(G_DC, G_D * G_C)
end

# Helper: build a tiny chain poset and some simple modules.
# The existing tests already use chain_poset and one_by_one_fringe etc.
# We reuse that style for consistency.

@testset "TorLongExactSequenceSecond" begin
    P = chain_poset(3)
    # Choose a genuine short exact sequence 0 -> A -> B -> C -> 0:
    # A = [2,2], B = [1,2], C = [1,1] as interval modules on the chain.
    A = IR.pmodule_from_fringe(one_by_one_fringe(P,
            FF.principal_upset(P, 2), FF.principal_downset(P, 2); scalar=one(K), field=field))
    B = IR.pmodule_from_fringe(one_by_one_fringe(P,
            FF.principal_upset(P, 1), FF.principal_downset(P, 2); scalar=one(K), field=field))
    C = IR.pmodule_from_fringe(one_by_one_fringe(P,
            FF.principal_upset(P, 1), FF.principal_downset(P, 1); scalar=one(K), field=field))

    # Explicit inclusion i: A -> B and projection p: B -> C (components per vertex).
    # dims(A) = (0,1,0), dims(B) = (1,1,0), dims(C) = (1,0,0).
    i = MD.PMorphism(A, B, [CM.zeros(field, 1, 0), CM.ones(field, 1, 1),         CM.zeros(field, 0, 0)])
    p = MD.PMorphism(B, C, [CM.ones(field, 1, 1),         CM.zeros(field, 0, 1), CM.zeros(field, 0, 0)])

    # Opposite poset
    Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)))

    # Right module on P^op (simple at vertex 2).
    Rop = IR.pmodule_from_fringe(one_by_one_fringe(Pop,
            FF.principal_upset(Pop, 2), FF.principal_downset(Pop, 2); scalar=one(K), field=field))

    LES = DF.TorLongExactSequenceSecond(Rop, i, p, TO.DerivedFunctorOptions(maxdeg=2))

    TorRA = DF.Tor(Rop, A, TO.DerivedFunctorOptions(maxdeg=2))
    TorRB = DF.Tor(Rop, B, TO.DerivedFunctorOptions(maxdeg=2))
    TorRC = DF.Tor(Rop, C, TO.DerivedFunctorOptions(maxdeg=2))
    @test LES.maxdeg == 2
    @test length(LES.iH) == 3
    @test length(LES.pH) == 3
    @test length(LES.delta) == 3

    @test [TO.dim(LES.TorA, s) for s in 0:LES.maxdeg] == [TO.dim(TorRA, s) for s in 0:LES.maxdeg] == [1, 0, 0]
    @test [TO.dim(LES.TorB, s) for s in 0:LES.maxdeg] == [TO.dim(TorRB, s) for s in 0:LES.maxdeg] == [0, 0, 0]
    @test [TO.dim(LES.TorC, s) for s in 0:LES.maxdeg] == [TO.dim(TorRC, s) for s in 0:LES.maxdeg] == [0, 1, 0]

end

@testset "hyperTor_map_first/second: induced maps on Tor_n" begin
    P = chain_poset(2)
    Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)))

    # L = simple at vertex 1 on P.
    L = IR.pmodule_from_fringe(one_by_one_fringe(P,
            FF.principal_upset(P, 1), FF.principal_downset(P, 1); scalar=one(K), field=field))

    # Rop = simple at vertex 2 on P^op.
    Rop = IR.pmodule_from_fringe(one_by_one_fringe(Pop,
            FF.principal_upset(Pop, 2), FF.principal_downset(Pop, 2); scalar=one(K), field=field))

    # Complex concentrated in degree 0.
    C = TO.ModuleCochainComplex([L], MD.PMorphism{K}[]; tmin=0, check=true)

    HT = TO.hyperTor(Rop, C; maxlen=2)
    T  = DF.Tor(Rop, L, TO.DerivedFunctorOptions(maxdeg=2))

    # Tor_1 is known nonzero in this classical example.
    @test TO.dim(HT, 1) == TO.dim(T, 1)
    d1 = TO.dim(HT, 1)
    @test d1 == 1

    f2 = scalar_endo(Rop, c(2))
    f3 = scalar_endo(Rop, c(3))
    g2 = scalar_endo(L,   c(2))
    g3 = scalar_endo(L,   c(3))

    gC2 = TO.ModuleCochainMap(C, C, [g2]; check=true)
    gC3 = TO.ModuleCochainMap(C, C, [g3]; check=true)

    # --- Compare to Tor maps for degree-0 complexes ---
    F2h = TO.hyperTor_map_first(f2, HT, HT; n=1)
    F2t = TO.tor_map_first(f2, T, T; n=1)
    @test same_coefficients(F2h, F2t)
    @test same_coefficients(F2h, c(2) .* CM.eye(field, d1))
    F2full = TO.ModuleComplexes.induced_map_on_cohomology(
        TO.derived_tensor_map_first(f2, HT.T, HT.T; check=true),
        HT.cohom,
        HT.cohom,
        -1,
    )
    @test same_coefficients(F2h, F2full)
    hcache = DF.HomSystemCache(K)
    F2full_cached = TO.ModuleComplexes.induced_map_on_cohomology(
        TO.derived_tensor_map_first(f2, HT.T, HT.T; check=true, cache=hcache),
        HT.cohom,
        HT.cohom,
        -1,
    )
    @test same_coefficients(F2full_cached, F2full)
    F2map_cached1 = TO.derived_tensor_map_first(f2, HT.T, HT.T; check=true, cache=hcache)
    F2map_cached2 = TO.derived_tensor_map_first(f2, HT.T, HT.T; check=true, cache=hcache)
    @test F2map_cached1 === F2map_cached2
    @test TO.ChainComplexes.is_cochain_map(F2map_cached1)
    @test same_coefficients(TO.hyperTor_map_first(f2, HT, HT; n=1, check=true, cache=hcache), F2h)
    @test same_coefficients(TO.tor_map_first(T, T, f2; n=1, cache=hcache), F2t)

    G2h = TO.hyperTor_map_second(gC2, HT, HT; n=1)
    G2t = TO.tor_map_second(g2, T, T; n=1)
    @test same_coefficients(G2h, G2t)
    @test same_coefficients(G2h, c(2) .* CM.eye(field, d1))
    G2full = TO.ModuleComplexes.induced_map_on_cohomology(
        TO.derived_tensor_map_second(gC2, HT.T, HT.T; check=true),
        HT.cohom,
        HT.cohom,
        -1,
    )
    @test same_coefficients(G2full, G2h)
    G2full_cached = TO.ModuleComplexes.induced_map_on_cohomology(
        TO.derived_tensor_map_second(gC2, HT.T, HT.T; check=true, cache=hcache),
        HT.cohom,
        HT.cohom,
        -1,
    )
    @test same_coefficients(G2full_cached, G2full)
    G2map_cached1 = TO.derived_tensor_map_second(gC2, HT.T, HT.T; check=true, cache=hcache)
    G2map_cached2 = TO.derived_tensor_map_second(gC2, HT.T, HT.T; check=true, cache=hcache)
    @test G2map_cached1 === G2map_cached2
    @test same_coefficients(TO.tor_map_second(g2, T, T; n=1, cache=hcache), G2t)

    plan_cached = TO.ModuleComplexes._tensor_map_first_plan(HT.T, HT.T, hcache)
    idx = findfirst(d -> !isempty(d.adeg), plan_cached.degrees)
    @test idx !== nothing
    dplan = plan_cached.degrees[idx]
    upto_cached = maximum(dplan.adeg; init=0)
    coeffs_cached = TO.ModuleComplexes._lift_projective_chainmap_coeff_cached(
        f2, HT.T.resR, HT.T.resR; upto=upto_cached, cache=hcache
    )
    block1 = TO._tensor_map_on_tor_chains_from_projective_coeff(
        dplan.terms[1],
        dplan.dom_gens[1],
        dplan.cod_gens[1],
        dplan.dom_offsets[1],
        dplan.cod_offsets[1],
        coeffs_cached[dplan.adeg[1] + 1];
        cache=hcache,
    )
    block2 = TO._tensor_map_on_tor_chains_from_projective_coeff(
        dplan.terms[1],
        dplan.dom_gens[1],
        dplan.cod_gens[1],
        dplan.dom_offsets[1],
        dplan.cod_offsets[1],
        coeffs_cached[dplan.adeg[1] + 1];
        cache=hcache,
    )
    @test block1 === block2

    coeffs_scalar = TO.ModuleComplexes._lift_projective_chainmap_coeff_uncached(
        f2, HT.T.resR, HT.T.resR; upto=upto_cached
    )
    for a in 0:upto_cached
        n = length(HT.T.resR.gens[a + 1])
        idxs = collect(1:n)
        expect = sparse(idxs, idxs, fill(c(2), n), n, n)
        @test same_coefficients(coeffs_scalar[a + 1], expect)
    end

    # --- Identity behavior ---
    Fid = TO.hyperTor_map_first(IR.id_morphism(Rop), HT, HT; n=1)
    @test same_coefficients(Fid, CM.eye(field, d1))

    Gid = TO.hyperTor_map_second(TO.ModuleComplexes.idmap(C), HT, HT; n=1)
    @test same_coefficients(Gid, CM.eye(field, d1))

    # --- Functoriality in first variable ---
    f32 = compose_morphism(f3, f2)   # f3 o f2 = 6*id
    F32 = TO.hyperTor_map_first(f32, HT, HT; n=1)
    F3  = TO.hyperTor_map_first(f3,  HT, HT; n=1)
    @test same_coefficients(F32, F3 * F2h)

    # --- Functoriality in second variable ---
    g32 = compose_morphism(g3, g2)
    gC32 = TO.ModuleCochainMap(C, C, [g32]; check=true)
    G32 = TO.hyperTor_map_second(gC32, HT, HT; n=1)
    G3  = TO.hyperTor_map_second(gC3,  HT, HT; n=1)
    @test same_coefficients(G32, G3 * G2h)

    # --- Bifunctorial commutativity (natural in both vars) ---
    @test same_coefficients(G3 * F2h, F2h * G3)
end


@testset "Tor by hand on chain of length 2" begin
    P = chain_poset(2)
    Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)))

    # L = simple at 1 on P
    Lfr = one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 1); scalar=one(K), field=field)
    L = IR.pmodule_from_fringe(Lfr)

    # Rop = simple at 2 on Pop (= P^op)
    Rfr = one_by_one_fringe(Pop, FF.principal_upset(Pop, 2), FF.principal_downset(Pop, 2); scalar=one(K), field=field)
    Rop = IR.pmodule_from_fringe(Rfr)

    T = DF.Tor(Rop, L, TO.DerivedFunctorOptions(maxdeg=3))

    @test TO.dim(T, 0) == 0
    @test TO.dim(T, 1) == 1
    @test TO.dim(T, 2) == 0
    @test TO.dim(T, 3) == 0
end

@testset "Tor extra structure: LES, actions, bicomplex" begin
    # Poset: chain 1 < 2
    P = chain_poset(2)
    Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)))  # opposite

    # Left modules on P
    S1 = _chain_module(P, [1, 0], CM.zeros(field, 0, 1), field)
    S2 = _chain_module(P, [0, 1], CM.zeros(field, 1, 0), field)
    P1 = _chain_module(P, [1, 1], CM.ones(field, 1, 1), field)  # projective at 1

    # Right modules (as P^op-modules) on Pop
    S1op = _chain_module(Pop, [1, 0], CM.zeros(field, 1, 0), field)
    S2op = _chain_module(Pop, [0, 1], CM.zeros(field, 0, 1), field)
    P2op = _chain_module(Pop, [1, 1], CM.ones(field, 1, 1), field)  # projective at 2 (in Pop)

    # Short exact sequence in the second variable: 0 -> S2 -> P1 -> S1 -> 0
    i = MD.PMorphism(S2, P1, [CM.zeros(field, 1, 0), CM.ones(field, 1, 1)])
    p = MD.PMorphism(P1, S1, [CM.ones(field, 1, 1), CM.zeros(field, 0, 1)])

    les2 = DF.TorLongExactSequenceSecond(S2op, i, p, TO.DerivedFunctorOptions(maxdeg=1))

    # Connecting map delta: Tor_1(S2op, S1) -> Tor_0(S2op, S2)
    # In this toy example it is nonzero (this is the standard non-split SES).
    @test size(les2.delta[1], 1) == 0
    @test size(les2.delta[2]) == (TO.dim(les2.TorA, 0), TO.dim(les2.TorC, 1))
    @test FL.rank(field, les2.delta[2]) == 1

    # Short exact sequence in the first variable: 0 -> S1op -> P2op -> S2op -> 0
    i1 = MD.PMorphism(S1op, P2op, [CM.ones(field, 1, 1), CM.zeros(field, 1, 0)])
    p1 = MD.PMorphism(P2op, S2op, [CM.zeros(field, 0, 1), CM.ones(field, 1, 1)])

    les1 = TO.TorLongExactSequenceFirst(S1, i1, p1, TO.DerivedFunctorOptions(maxdeg=1))

    @test size(les1.delta[2]) == (TO.dim(les1.TorA, 0), TO.dim(les1.TorC, 1))
    @test FL.rank(field, les1.delta[2]) == 1

    # Ext action on Tor via the resolve-second model:
    # The Ext^0 unit should act as identity on Tor_1.
    EA = TO.ExtAlgebra(S1, TO.DerivedFunctorOptions(maxdeg=2))
    Tsec = DF.Tor(S2op, S1, TO.DerivedFunctorOptions(maxdeg=2, model=:second); res=EA.E.res)
    u = DF.unit(EA)
    direct_old = DF.Algebras._EXT_ACTION_USE_DIRECT_STREAM[]
    coeff_old = DF.Functoriality._FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[]
    action_old = DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_NNZ[]
    work_old = DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_EST_WORK[]
    product_old = DF.Algebras._EXT_ACTION_DIRECT_STREAM_USE_PRODUCT_WORK[]
    try
        DF.Algebras._EXT_ACTION_USE_DIRECT_STREAM[] = false
        DF.Functoriality._FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[] = typemax(Int)
        DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_NNZ[] = typemax(Int)
        DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_EST_WORK[] = typemax(Int)
        DF.Algebras._EXT_ACTION_DIRECT_STREAM_USE_PRODUCT_WORK[] = false
        act_off = DF.ext_action_on_tor(EA, Tsec, u; s=1)

        DF.Algebras._EXT_ACTION_USE_DIRECT_STREAM[] = true
        DF.Functoriality._FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[] = 0
        DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_NNZ[] = 0
        DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_EST_WORK[] = 0
        DF.Algebras._EXT_ACTION_DIRECT_STREAM_USE_PRODUCT_WORK[] = true
        act_on = DF.ext_action_on_tor(EA, Tsec, u; s=1)

        @test same_coefficients(act_on, act_off)
        @test size(act_on) == (TO.dim(Tsec, 1), TO.dim(Tsec, 1))
        @test same_coefficients(act_on[1, 1], c(1))
    finally
        DF.Algebras._EXT_ACTION_USE_DIRECT_STREAM[] = direct_old
        DF.Functoriality._FUNCTORIALITY_DIRECT_COEFF_MIN_NNZ[] = coeff_old
        DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_NNZ[] = action_old
        DF.Algebras._EXT_ACTION_DIRECT_STREAM_MIN_EST_WORK[] = work_old
        DF.Algebras._EXT_ACTION_DIRECT_STREAM_USE_PRODUCT_WORK[] = product_old
    end

    # Tor double complex total cohomology agrees with Tor groups (degree reindexing).
    # Using small lengths is enough for this example.
    DC = TO.TorDoubleComplex(S2op, S1; maxlen=1)
    Tot = CC.total_complex(DC)

    # Tor_n corresponds to H^{-n} of the total cochain complex.
    Tfirst = DF.Tor(S2op, S1, TO.DerivedFunctorOptions(maxdeg=2))
    @test CC.cohomology_data(Tot, 0).dimH == TO.dim(Tfirst, 0)
    @test CC.cohomology_data(Tot, -1).dimH == TO.dim(Tfirst, 1)
    @test CC.cohomology_data(Tot, -2).dimH == TO.dim(Tfirst, 2)

    # TorAlgebra infrastructure smoke test: a trivial degree-0 product.
    # Choose a projective right module so Tor_0 is 1-dim and higher Tor vanishes.
    T0 = DF.Tor(P2op, S2, TO.DerivedFunctorOptions(maxdeg=0))
    @test TO.dim(T0, 0) == 1

    Alg = DF.TorAlgebra(T0)
    DF.set_chain_product!(Alg, 0, 0, sparse(CM.ones(field, 1, 1)))
    M00 = DF.multiplication_matrix(Alg, 0, 0)
    @test size(M00) == (1, 1)
    @test same_coefficients(M00[1, 1], c(1))

    x = DF.element(Alg, 0, [c(1)])
    y = DF.multiply(Alg, x, x)
    @test y.deg == 0
    @test same_coefficients(y.coords[1], c(1))
end
end

@testset "A70 Ext action: nonzero products, units, naturality and model contracts" begin
    with_fields(FIELDS_FULL) do field
        K = CM.coeff_type(field)
        same(X, Y) = field isa CM.RealField ?
            isapprox(X, Y; atol=field.atol, rtol=field.rtol) : X == Y
        # The diamond's sum of all four simples has Ext dimensions (4,4,1).
        # Tensoring its minimal resolution with the right simple at the top
        # gives Tor dimensions (1,2,1). The two length-two paths give nonzero
        # Ext products and nonzero successive cap actions, so the composition
        # identity below cannot pass merely because every product vanishes.
        P = FF.FinitePoset(Bool[1 1 1 1; 0 1 0 1; 0 0 1 1; 0 0 0 1])
        Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)))
        M = MD.PModule{K}(P, ones(Int, 4),
            Dict((u, v) => zeros(K, 1, 1) for (u, v) in FF.cover_edges(P)); field=field)
        rdims = [0, 0, 0, 1]
        R = MD.PModule{K}(Pop, rdims,
            Dict((u, v) => zeros(K, rdims[v], rdims[u]) for (u, v) in FF.cover_edges(Pop)); field=field)
        A = DF.ExtAlgebra(M, TO.DerivedFunctorOptions(maxdeg=2))
        T = DF.Tor(R, M, TO.DerivedFunctorOptions(model=:second, maxdeg=2); res=A.E.res)
        @test [DF.dim(A, t) for t in 0:2] == [4, 4, 1]
        @test [DF.dim(T, s) for s in 0:2] == [1, 2, 1]
        for s in 0:2
            @test same(DF.ext_action_on_tor(A, T, one(A); s=s), CM.eye(field, DF.dim(T, s)))
        end
        xs = DF.basis(A, 1)
        action1 = [DF.ext_action_on_tor(A, T, x; s=1) for x in xs]
        action2 = [DF.ext_action_on_tor(A, T, x; s=2) for x in xs]
        @test count(F -> !iszero(F), action1) == 2
        @test count(F -> !iszero(F), action2) == 2
        nonzero_products = 0
        for (i, x) in enumerate(xs), (j, y) in enumerate(xs)
            xy = x * y
            nonzero_products += !iszero(xy)
            composite = action1[i] * action2[j]
            @test same(DF.ext_action_on_tor(A, T, xy; s=2), composite)
            @test iszero(xy) == iszero(composite)
        end
        @test nonzero_products == 2
        @test size(DF.ext_action_on_tor(A, T, first(xs); s=0)) == (0, 1)

        # Naturality in the right module uses a rectangular, nonzero diagonal
        # inclusion R -> R + R; no multiplicative hypothesis on R is needed.
        R2 = MD.direct_sum(R, R)
        f = MD.PMorphism(R, R2,
            [u == 4 ? reshape(K[1, 1], 2, 1) : zeros(K, 0, 0) for u in 1:4])
        T2 = DF.Tor(R2, M, TO.DerivedFunctorOptions(model=:second, maxdeg=2); res=A.E.res)
        @test [DF.dim(T2, s) for s in 0:2] == [2, 4, 2]
        induced = [DF.tor_map_first(T, T2, f; s=s) for s in 0:2]
        for x in xs, s in 1:2
            @test same(induced[s] * DF.ext_action_on_tor(A, T, x; s=s),
                       DF.ext_action_on_tor(A, T2, x; s=s) * induced[s+1])
        end

        # Independently held storage is allowed when all chain coordinates
        # agree, but equal generator labels alone are insufficient.
        res = A.E.res
        res_copy = DF.ProjectiveResolution{K}(res.M, copy(res.Pmods), deepcopy(res.gens),
            copy(res.d_mor), copy(res.d_mat), res.aug)
        Tcopy = DF.Tor(R, M, TO.DerivedFunctorOptions(model=:second, maxdeg=2); res=res_copy)
        @test same(DF.ext_action_on_tor(A, Tcopy, first(xs); s=2), first(action2))
        altered_d = copy(res.d_mat)
        altered_d[1] = copy(altered_d[1])
        altered_d[1][1, 1] += one(K)
        badres = DF.ProjectiveResolution{K}(res.M, res.Pmods, res.gens,
            res.d_mor, altered_d, res.aug)
        badT = DF.TorSpaceSecond{K}(badres, T.Rop, T.bd, T.dims, T.offsets, T.homol)
        @test_throws ArgumentError DF.ext_action_on_tor(A, badT, one(A); s=1)
        @test_throws ArgumentError DF.ext_action_on_tor(A, badT, first(xs); s=0)
        otherA = DF.ExtAlgebra(M, TO.DerivedFunctorOptions(maxdeg=2))
        @test_throws ArgumentError DF.ext_action_on_tor(A, T, one(otherA); s=1)
        @test_throws ArgumentError DF.ext_action_on_tor(A, T, first(DF.basis(otherA, 1)); s=0)
        @test_throws ArgumentError DF.ext_action_on_tor(A, T, one(A); s=-1)
        @test_throws ArgumentError DF.ext_action_on_tor(A, T, one(A); s=3)
        @test_throws ArgumentError DF.ext_action_on_tor(A, T, one(A), TO.DerivedFunctorOptions(maxdeg=-1))
        @test_throws ArgumentError DF.ext_action_on_tor(A, T, one(A), TO.DerivedFunctorOptions(maxdeg=3))
        batched_unit = DF.ext_action_on_tor(A, T, one(A), TO.DerivedFunctorOptions(maxdeg=2))
        @test all(same(batched_unit[s+1], CM.eye(field, DF.dim(T, s))) for s in 0:2)
        invalid_degree = DF.ExtElement{K}(A, -1, K[])
        @test_throws ArgumentError DF.ext_action_on_tor(A, T, invalid_degree; s=0)
        invalid_coords = DF.ExtElement{K}(A, 1, zeros(K, DF.dim(A, 1) + 1))
        @test_throws DimensionMismatch DF.ext_action_on_tor(A, T, invalid_coords; s=0)
    end
end

@testset "A75 lazy Ext maps preserve rectangular functoriality and comparisons" begin
    with_fields(FIELDS_FULL) do field
        K = CM.coeff_type(field)
        same = (A, B) -> field isa CM.RealField ?
            isapprox(A, B; atol=field.atol, rtol=field.rtol) : A == B
        P = chain_poset(2)
        simple_copies = function (vertex, multiplicity)
            dims = vertex == 1 ? [multiplicity, 0] : [0, multiplicity]
            MD.PModule{K}(P, dims, Dict((1, 2) => zeros(K, dims[2], dims[1])); field=field)
        end
        M, Mp = simple_copies(1, 2), simple_copies(1, 3)
        N, Np = simple_copies(2, 2), simple_copies(2, 1)
        fmatrix = K[1 0; -1 1; 0 1]
        gmatrix = K[1 -1]
        f = MD.PMorphism(M, Mp, [fmatrix, zeros(K, 0, 0)])
        g = MD.PMorphism(N, Np, [zeros(K, 0, 0), gmatrix])
        # On the arrow 1 -> 2, Ext^1(S1^m,S2^n) = Mat(n,m).
        # The induced maps are independently known: X |-> X*f and X |-> g*X.
        expected_first = kron(transpose(fmatrix), Matrix{K}(I, 2, 2))
        expected_second = kron(Matrix{K}(I, 2, 2), gmatrix)
        expected_both = kron(transpose(fmatrix), gmatrix)
        for canon in (:projective, :injective), reuse in (false, true)
            cache = reuse ? CM.ResolutionCache() : nothing
            opts = OPT.DerivedFunctorOptions(maxdeg=1, model=:unified, canon=canon)
            E = DF.Ext(M, N, opts; cache=cache)
            F = DF.Ext(Mp, N, opts; cache=cache)
            G = DF.Ext(M, Np, opts; cache=cache)
            H = DF.Ext(Mp, Np, opts; cache=cache)
            for space in (E, F, G, H)
                @test (canon === :projective ? space.Einj : space.Eproj) === nothing
            end
            # This must work before any explicit comparison or eager model
            # access. Prior code passed the absent realization as `nothing`.
            alternate = canon === :projective ? :injective : :projective
            first_map = DF.ext_map_first(E, F, f; t=1, backend=alternate)
            second_map = DF.ext_map_second(E, G, g; t=1, backend=alternate)
            @test size(first_map) == (4, 6)
            @test size(second_map) == (2, 4)
            @test same(first_map, DF.ext_map_first(E, F, f; t=1, backend=canon))
            @test same(second_map, DF.ext_map_second(E, G, g; t=1, backend=canon))
            # Explicit identification with matrix coordinates, using the
            # projective cocycles of the simple modules. No basis equality
            # between independently eliminated models is assumed.
            transports = map((E, F, G, H)) do space
                proj = DF.projective_model(space)
                dim = DF.dim(space, 1)
                @test proj.res.gens[2] == fill(2, space.M.dims[1])
                @test size(proj.complex.d[1], 1) == dim
                transport = hcat((DF.coordinates(space, 1,
                    Matrix{K}(I, dim, dim)[:, j]; model=:projective) for j in 1:dim)...)
                @test TamerOp.FieldLinAlg.rank(field, transport) == dim
                transport
            end
            JE, JF, JG, JH = transports
            @test same(first_map * JF, JE * expected_first)
            @test same(second_map * JE, JG * expected_second)
            left = DF.ext_map_second(F, H, g; t=1, backend=alternate)
            right = DF.ext_map_first(G, H, f; t=1, backend=alternate)
            @test same(right * left, second_map * first_map)
            @test same(right * left * JF, JG * expected_both)
            # Eager realization and cached repeated queries induce the same
            # maps, including a rectangular contravariant map.
            for space in (E, F, G, H)
                forward, backward = DF.comparison_isomorphisms(space)
                @test same(forward[2] * backward[2], Matrix{K}(I, DF.dim(space, 1), DF.dim(space, 1)))
                @test DF.projective_model(space).M.field === field
                @test DF.injective_model(space).M.field === field
            end
            @test same(DF.ext_map_first(E, F, f; t=1), first_map)
            @test same(DF.ext_map_second(E, G, g; t=1), second_map)
            @test DF.ext_map_first(E, F, f; t=0) == zeros(K, 0, 0)
            @test_throws ErrorException DF.ext_map_first(E, F, f; t=1, backend=:invalid)
            if reuse
                @test DF.Ext(M, N, opts; cache=cache) === E
            else
                @test DF.injective_model(E).res !== DF.injective_model(F).res
            end
        end
        # Change coordinates in a chosen injective resolution, including the
        # coaugmentation. This forces the no-cache comparison to be a real
        # chain map rather than an identification of equal-sized storage.
        base = DF.ExtInjective(M, N, OPT.DerivedFunctorOptions(maxdeg=1, model=:injective))
        res = base.res
        gauge0, gauge1 = K[1 0; 1 1], K[1 1; 0 1]
        inverses = (K[1 0; -1 1], K[1 -1; 0 1])
        changed_d = [MD.PMorphism(morphism.dom, morphism.cod,
            [(morphism.cod.dims[v] == 0 ? zeros(K, 0, 0) : gauge1) *
                morphism.comps[v] *
                (morphism.dom.dims[v] == 0 ? zeros(K, 0, 0) : inverses[min(t, 2)])
             for v in 1:P.n]) for (t, morphism) in enumerate(res.d_mor)]
        changed_iota = MD.PMorphism(N, first(res.Emods),
            [gauge0 * component for component in res.iota0.comps])
        changed_res = DF.InjectiveResolution{K}(N, res.Emods, res.gens, changed_d, changed_iota)
        @test DF.check_injective_resolution(changed_res).valid
        changed = DF.ExtInjective(Mp, changed_res; maxdeg=1)
        action = DF.ext_map_first(base, changed, f; t=1)
        Jbase = hcat((DF.coordinates(base, 1, Matrix{K}(I, 4, 4)[:, j]) for j in 1:4)...)
        Jchanged = hcat((DF.coordinates(changed, 1, Matrix{K}(I, 6, 6)[:, j]) for j in 1:6)...)
        @test same(action * Jchanged, Jbase * kron(transpose(fmatrix), inverses[2]))
    end
end

@testset "A75 independent projective resolution coordinates preserve Ext and Tor maps" begin
    with_fields(FIELDS_FULL) do field
        K = CM.coeff_type(field)
        same = (A, B) -> field isa CM.RealField ?
            isapprox(A, B; atol=field.atol, rtol=field.rtol) : A == B
        P = chain_poset(2)
        Pop = FF.FinitePoset(transpose(FF.leq_matrix(P)))
        simple_copies = function (Q, vertex, multiplicity)
            dims = vertex == 1 ? [multiplicity, 0] : [0, multiplicity]
            u, v = only(FF.cover_edges(Q))
            MD.PModule{K}(Q, dims, Dict((u, v) => zeros(K, dims[v], dims[u])); field=field)
        end
        M, N, Np = simple_copies(P, 1, 2), simple_copies(P, 2, 2), simple_copies(P, 2, 1)
        R, Rp = simple_copies(Pop, 2, 2), simple_copies(Pop, 2, 1)
        L, Lp = simple_copies(P, 1, 2), simple_copies(P, 1, 1)
        gauge0, gauge1 = K[1 0; 1 1], K[1 1; 0 1]
        inverse0, inverse1 = K[1 0; -1 1], K[1 -1; 0 1]
        regauge = function (res)
            gauges = [d == 0 ? gauge0 : gauge1 for d in 0:length(res.Pmods)-1]
            inverses = [d == 0 ? inverse0 : inverse1 for d in 0:length(res.Pmods)-1]
            maps = [MD.PMorphism(morphism.dom, morphism.cod,
                [(morphism.cod.dims[v] == 0 ? zeros(K, 0, 0) : gauges[t]) * morphism.comps[v] *
                 (morphism.dom.dims[v] == 0 ? zeros(K, 0, 0) : inverses[t+1])
                 for v in 1:P.n]) for (t, morphism) in enumerate(res.d_mor)]
            coefficients = [sparse((size(D, 1) == 0 ? zeros(K, 0, 0) : gauges[t]) * D *
                (size(D, 2) == 0 ? zeros(K, 0, 0) : inverses[t+1])) for (t, D) in enumerate(res.d_mat)]
            augmentation = MD.PMorphism(first(res.Pmods), res.M,
                [component * inverse0 for component in res.aug.comps])
            changed = DF.ProjectiveResolution{K}(res.M, res.Pmods, res.gens, maps, coefficients, augmentation)
            @test DF.check_projective_resolution(changed).valid
            changed
        end
        resolution = DF.projective_resolution(M, OPT.ResolutionOptions(maxlen=2))
        changed = regauge(resolution)
        E, Echanged = DF.Ext(resolution, N; maxdeg=1), DF.Ext(changed, Np; maxdeg=1)
        g = MD.PMorphism(N, Np, [zeros(K, 0, 0), K[1 -1]])
        J = hcat((DF.coordinates(E, 1, Matrix{K}(I, 4, 4)[:, j]) for j in 1:4)...)
        Jchanged = hcat((DF.coordinates(Echanged, 1, Matrix{K}(I, 2, 2)[:, j]) for j in 1:2)...)
        actual = DF.ext_map_second(E, Echanged, g; t=1)
        @test same(actual * J, Jchanged * kron(transpose(inverse1), K[1 -1]))
        @test_throws AssertionError DF.ext_map_second(E, Echanged, g; t=-1)
        @test_throws AssertionError DF.ext_map_second(E, Echanged, g; t=2)

        options = OPT.DerivedFunctorOptions(maxdeg=1, model=:second)
        T, Tchanged = DF.Tor(R, M, options; res=resolution), DF.Tor(Rp, M, options; res=changed)
        f = MD.PMorphism(R, Rp, [zeros(K, 0, 0), K[1 -1]])
        JT = hcat((DF.coordinates(T, 1, Matrix{K}(I, 4, 4)[:, j]) for j in 1:4)...)
        JTchanged = hcat((DF.coordinates(Tchanged, 1, Matrix{K}(I, 2, 2)[:, j]) for j in 1:2)...)
        @test same(DF.tor_map_first(T, Tchanged, f; s=1) * JT,
                   JTchanged * kron(gauge1, K[1 -1]))
        @test_throws AssertionError DF.tor_map_first(T, Tchanged, f; s=-1)
        @test_throws AssertionError DF.tor_map_first(T, Tchanged, f; s=2)

        right_resolution = DF.projective_resolution(R, OPT.ResolutionOptions(maxlen=2))
        right_changed = regauge(right_resolution)
        options = OPT.DerivedFunctorOptions(maxdeg=1, model=:first)
        U, Uchanged = DF.Tor(R, L, options; res=right_resolution), DF.Tor(R, Lp, options; res=right_changed)
        h = MD.PMorphism(L, Lp, [K[1 -1], zeros(K, 0, 0)])
        JU = hcat((DF.coordinates(U, 1, Matrix{K}(I, 4, 4)[:, j]) for j in 1:4)...)
        JUchanged = hcat((DF.coordinates(Uchanged, 1, Matrix{K}(I, 2, 2)[:, j]) for j in 1:2)...)
        @test same(DF.tor_map_second(U, Uchanged, h; s=1) * JU,
                   JUchanged * kron(gauge1, K[1 -1]))
        @test_throws AssertionError DF.tor_map_second(U, Uchanged, h; s=-1)
        @test_throws AssertionError DF.tor_map_second(U, Uchanged, h; s=2)
    end
end

@testset "A62 long-chain Hom and tensor coefficient maps preserve composition" begin
    F = DF.Functoriality
    for field in (CM.QQField(),CM.F2(),CM.F3(),CM.Fp(5),CM.RealField(Float64;atol=1e-12,rtol=1e-10))
        K = CM.coeff_type(field)
        same(A,B) = field isa CM.RealField ? isapprox(A,B;atol=1e-11,rtol=1e-10) : A==B
        for dimension in (2,6)
            P = chain_poset(5)
            edge = CM.eye(field,dimension)
            edge[1,dimension] = one(K)
            M = MD.PModule{K}(P,fill(dimension,5),
                Dict((i,i+1)=>copy(edge) for i in 1:4);field=field)
            # 32 independent projective summands activate the actual direct
            # coefficient assembly. N^2=0 gives (I+N)^r=I+rN by hand.
            generators(v) = fill(v,32)
            offsets = collect(0:dimension:32dimension)
            coefficients = sparse(CM.eye(field,32))
            power(r) = begin
                A = CM.eye(field,dimension)
                A[1,dimension] = CM.coerce(field,r)
                A
            end
            expected = kron(coefficients,sparse(power(4)))
            # Hom(P_1^32,M) -> Hom(P_5^32,M), induced by P_5^32 -> P_1^32.
            direct_hom = F._precompose_on_hom_cochains_from_projective_coeff(
                M,generators(5),generators(1),offsets,offsets,coefficients)
            @test same(direct_hom,expected)
            @test same(MD.map_leq(M,1,5),power(4)) # assembly must not pollute this memo
            first_hom = F._precompose_on_hom_cochains_from_projective_coeff(
                M,generators(3),generators(1),offsets,offsets,coefficients)
            second_hom = F._precompose_on_hom_cochains_from_projective_coeff(
                M,generators(5),generators(3),offsets,offsets,coefficients)
            @test same(second_hom*first_hom,direct_hom)
            MD._clear_map_leq_memo!(M)
            MD._clear_map_leq_many_plan_cache!(M)
            # The covariant tensor map has the same independently known blocks.
            direct_tensor = F._tensor_map_on_tor_chains_from_projective_coeff(
                M,generators(1),generators(5),offsets,offsets,coefficients)
            @test same(direct_tensor,expected)
            first_tensor = F._tensor_map_on_tor_chains_from_projective_coeff(
                M,generators(1),generators(3),offsets,offsets,coefficients)
            second_tensor = F._tensor_map_on_tor_chains_from_projective_coeff(
                M,generators(3),generators(5),offsets,offsets,coefficients)
            @test same(second_tensor*first_tensor,direct_tensor)
        end
    end
end
