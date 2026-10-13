# Primary splittings, bounded complete finite-field search and bipath routing.

function _split_primary(M::PModule{K}, comps, factors) where {K}
    inclusions = Vector{Matrix{K}}[]
    for (q,e) in factors
        push!(inclusions,Matrix{K}[FieldLinAlg.nullspace(M.field,
            _polynomial_matrix(M.field,q^e,X)) for X in comps])
    end
    inverse = Matrix{K}[FieldLinAlg.solve_fullcolumn(M.field,
        reduce(hcat,[inc[v] for inc in inclusions]),eye(M.field,M.dims[v]))
        for v in eachindex(M.dims)]
    offsets = zeros(Int,length(M.dims))
    children = Tuple{PModule{K},PMorphism{K},PMorphism{K}}[]
    for inc in inclusions
        dims = [size(X,2) for X in inc]
        proj = Matrix{K}[inverse[v][offsets[v]+1:offsets[v]+dims[v],:] for v in eachindex(dims)]
        offsets .+= dims
        N = PModule{K}(M.Q,dims,Dict((u,v) => proj[v]*M.edge_maps[u,v]*inc[u]
            for (u,v) in cover_edges(M.Q));field=M.field)
        push!(children,(N,PMorphism{K}(N,M,inc),PMorphism{K}(M,N,proj)))
    end
    return children
end

function _classify(M, budget, state, backend)
    A = _endomorphisms(M,budget)
    d = size(A.B,2)
    d == 1 && return _schur_certificate()
    K = coeff_type(M.field)
    records = Tuple{Vector{K},typeof(one(A.R))}[]
    function try_element(c)
        _bound((big(length(records))+1)*(d+sum(M.dims)+1),budget.max_matrix_entries,
            "retained polynomial certificates")
        state.tested += 1
        comps = _components(M,A,c)
        factors = _primary_factors(M.field,A.R,A.F,comps)
        if length(factors) > 1
            state.splits += 1
            return _split_primary(M,comps,factors)
        end
        push!(records,(c,first(first(factors))))
        return nothing
    end
    for i in 2:d
        _charge!(state,M,budget)
        c = zeros(K,d); c[i] = one(K)
        split = try_element(c)
        split === nothing || return split
    end
    # The identity is useful when testing a scalar residue field but needs no
    # polynomial factorization or charged search step.
    c = zeros(K,d); c[1] = one(K)
    push!(records,(c,Nemo.gen(A.R)-one(A.R)))
    L = _multiplication_table(M,A,budget)
    cert = _local_certificate(M,A,records,L)
    cert === nothing || return cert
    if backend == :hecke
        _charge!(state,M,budget)
        cert = _rational_division_certificate(L,budget)
        cert === nothing || return cert
    end

    function try_search(c)
        split = try_element(c)
        split === nothing || return split
        # Certification is not needed after every candidate. Its cost is cubic
        # in the algebra dimension; geometrically spaced checks bound repetition.
        if ispow2(length(records))
            cert = _local_certificate(M,A,records,L)
            cert === nothing || return cert
        end
        return nothing
    end
    m = d-1
    if M.field isa PrimeField
        p = M.field.p
        # Projective End(M)/(k*1): adding a scalar or multiplying by a nonzero
        # scalar preserves the existence of a primary splitting. Enumerate one
        # representative per line, including extension-field eigenvalues via
        # irreducible polynomial factorization over the original ground field.
        for pivot in 1:m
            for number in big(1):big(p)^(m-pivot)-1
                _charge!(state,M,budget)
                c = zeros(K,d); c[pivot+1] = one(K)
                value = number
                for j in pivot+2:d
                    value,digit = divrem(value,p)
                    c[j] = K(digit)
                end
                result = try_search(c)
                result === nothing || return result
            end
        end
        # Every endomorphism is primary, equivalently every nonunit nilpotent.
        # This is an exhaustive local-ring certificate, not a sampling test.
        return _Certificate(:exhaustive_prime_field,d,nothing,nothing,nothing)
    end

    # A rational projector has an integer representative modulo scalar identity
    # after rescaling. Enumerating primitive vectors is complete for finding a
    # split. Terminal rational pieces are certified above, either by a field
    # quotient or by the optional exact Schur-index computation over its center.
    radius = 1
    while true
        base = 2big(radius)+1
        for number in big(1):base^m-1
            _charge!(state,M,budget)
            value = number
            ints = zeros(Int,m)
            for j in 1:m
                value,digit = divrem(value,base)
                ints[j] = isodd(digit) ? Int(div(digit+1,2)) : -Int(div(digit,2))
            end
            maximum(abs,ints) == radius || continue
            ints[findfirst(!iszero,ints)] > 0 || continue
            foldl(gcd,ints;init=0) == 1 || continue
            count(!iszero,ints) == 1 && continue # basis vectors already tested
            result = try_search(vcat(zero(K),K.(ints)))
            result === nothing || return result
        end
        radius += 1
    end
end

function _solve!(M, inclusion, projection, pieces, ins, outs, certificates, budget, state, backend)
    all(iszero,M.dims) && return nothing
    answer = _classify(M,budget,state,backend)
    if answer isa _Certificate
        push!(pieces,M)
        if inclusion !== nothing
            push!(ins,inclusion); push!(outs,projection)
        end
        push!(certificates,answer)
        return nothing
    end
    K = coeff_type(M.field)
    for (N,i,p) in answer
        ni = inclusion === nothing ? nothing : PMorphism{K}(N,inclusion.cod,
            Matrix{K}[inclusion.comps[v]*i.comps[v] for v in eachindex(M.dims)])
        np = projection === nothing ? nothing : PMorphism{K}(projection.dom,N,
            Matrix{K}[p.comps[v]*projection.comps[v] for v in eachindex(M.dims)])
        _solve!(N,ni,np,pieces,ins,outs,certificates,budget,state,backend)
    end
    return nothing
end

"""
    indecomposable_decomposition(M; witnesses=false, backend=:native,
        budget=IndecomposableDecompositionBudget())

Compute a certified Krull-Schmidt decomposition of a finite-poset module over
QQ or a prime field. `summand_dimensions` and the summary are retained by default;
`witnesses=true` also retains every summand and its inclusion/projection. Repeated
isomorphic summands occur separately; equal dimension vectors need not mean
isomorphic summands. Summand order and bases are not canonical.

The native algorithm factors minimal polynomials of End(M) elements over the
original field. Terminal pieces require a proved local endomorphism algebra.
Prime-field search is exhaustive if the resource budget permits it. Rational
certification with `backend=:native` uses a nilpotent ideal with field quotient;
noncommutative rational division quotients can exhaust that search even for
indecomposable input. After installing and loading Hecke, `backend=:hecke` also
certifies these quotients using exact Schur indices over their centers. This
backend accepts QQ only and retains the native splitting search. Both backends
can exhaust finite resource budgets; exhaustion throws
`IndecomposableSearchExhausted`, never a partial success. Hecke's arithmetic has
no wall-time or coefficient-size bound imposed by this budget.

This result concerns the specified finite category, not a stable multiparameter
barcode or an encoding-independent ambient decomposition.
"""
function indecomposable_decomposition(M::PModule{K}; witnesses::Bool=false, backend::Symbol=:native,
        budget::IndecomposableDecompositionBudget=IndecomposableDecompositionBudget()) where {K}
    _check_backend(backend,M.field)
    _context(M,budget)
    pieces,ins,outs = PModule{K}[],PMorphism{K}[],PMorphism{K}[]
    certificates = _Certificate[]
    state = _SearchState(0,0,0)
    identity = witnesses ? id_morphism(M) : nothing
    _solve!(M,identity,identity,pieces,ins,outs,certificates,budget,state,backend)
    maps = witnesses ? (;summands=pieces,inclusions=ins,projections=outs) : nothing
    return IndecomposableDecompositionResult(M,[copy(N.dims) for N in pieces],certificates,
        :endomorphism_algebra,backend,(;search_steps=state.steps,endomorphisms_tested=state.tested,
        primary_splits=state.splits),maps)
end

function _bipath_arms(Q)
    n = nvertices(Q)
    n >= 4 || throw(ArgumentError("A bipath needs two nonempty chain interiors sharing only bottom and top."))
    succ = [Int[] for _ in 1:n]; pred = [Int[] for _ in 1:n]
    for (u,v) in cover_edges(Q)
        push!(succ[u],v); push!(pred[v],u)
    end
    bottoms,tops = findall(isempty,pred),findall(isempty,succ)
    length(bottoms) == length(tops) == 1 || throw(ArgumentError("A bipath needs a unique bottom and top."))
    bottom,top = only(bottoms),only(tops)
    length(succ[bottom]) == length(pred[top]) == 2 || throw(ArgumentError("The bipath endpoints must join exactly two chains."))
    interior = [v for v in 1:n if v != bottom && v != top]
    all(v -> length(pred[v]) == length(succ[v]) == 1,interior) ||
        throw(ArgumentError("The bipath interior must consist of two disjoint chains."))
    arms = Vector{Int}[]
    visited = Set{Int}()
    for first in sort(succ[bottom])
        arm = Int[]; v = first
        while v != top
            v in visited && throw(ArgumentError("Bipath chains must share only their endpoints."))
            push!(visited,v); push!(arm,v)
            v = only(succ[v])
        end
        isempty(arm) && throw(ArgumentError("Both bipath interiors must be nonempty."))
        push!(arms,arm)
    end
    length(visited) == n-2 || throw(ArgumentError("The bipath has additional components."))
    return bottom,top,arms[1],arms[2]
end

function _bipath_family(Q,budget)
    bottom,top,a,b = _bipath_arms(Q)
    na,nb = length(a),length(b)
    count = 1+2(big(na)+1)*(nb+1)+div(big(na)*(na+1),2)+div(big(nb)*(nb+1),2)
    _bound(count,budget.max_bipath_intervals,"bipath intervals")
    _bound(count*nvertices(Q),budget.max_matrix_entries,"bipath supports")
    family = [collect(1:nvertices(Q))]
    for i in 0:na, j in 0:nb
        push!(family,sort([bottom;a[1:i];b[1:j]]))
        push!(family,sort([top;a[i+1:end];b[j+1:end]]))
    end
    for arm in (a,b), i in eachindex(arm), j in i:length(arm)
        push!(family,sort(arm[i:j]))
    end
    return family
end

"""
    bipath_decomposition(M; witnesses=false, budget=IndecomposableDecompositionBudget())

Validate that the finite base consists of two chains with nonempty interiors
sharing exactly bottom and top, then compute its complete interval decomposition.
The full module and its commuting maps determine the splitting; separate arm
barcodes do not suffice. Each returned indecomposable is a single thin interval.

This bounded specialization enumerates the five bipath interval families and
uses exact Hom-composition multiplicities and simultaneous splitting maps from
`interval_decomposition`. It does not implement the arm-matrix reduction algorithm
of Aoki, Escolar and Tada (Bipath persistence, 2025).
"""
function bipath_decomposition(M::PModule{K}; witnesses::Bool=false,
        budget::IndecomposableDecompositionBudget=IndecomposableDecompositionBudget()) where {K}
    _context(M,budget)
    family = _bipath_family(M.Q,budget)
    ib = IntervalDecompositions.IntervalDecompositionBudget(max_vertices=budget.max_vertices,
        max_dimension=budget.max_dimension,max_matrix_entries=budget.max_matrix_entries,
        max_candidates=budget.max_bipath_intervals)
    r = IntervalDecompositions.interval_decomposition(M;family,witnesses,budget=ib)
    all(iszero,r.residual_dims) || error("Bipath interval extraction left a nonzero residual.")
    dims = Vector{Int}[]; pieces,ins,outs = PModule{K}[],PMorphism{K}[],PMorphism{K}[]
    offsets = zeros(Int,length(M.dims))
    for (support,multiplicity) in zip(r.supports,r.multiplicities)
        ds = [Int(v in support) for v in eachindex(M.dims)]
        for j in 1:multiplicity
            push!(dims,copy(ds))
            if witnesses
                N = PModule{K}(M.Q,copy(ds),Dict((u,v) => fill(one(K),ds[v],ds[u])
                    for (u,v) in cover_edges(M.Q));field=M.field)
                inc = Matrix{K}[ds[v] == 0 ? zeros(K,M.dims[v],0) :
                    r.maps.inclusion.comps[v][:,offsets[v]+j:offsets[v]+j] for v in eachindex(ds)]
                proj = Matrix{K}[ds[v] == 0 ? zeros(K,0,M.dims[v]) :
                    r.maps.projection.comps[v][offsets[v]+j:offsets[v]+j,:] for v in eachindex(ds)]
                push!(pieces,N); push!(ins,PMorphism{K}(N,M,inc));push!(outs,PMorphism{K}(M,N,proj))
            end
        end
        offsets .+= multiplicity.*ds
    end
    maps = witnesses ? (;summands=pieces,inclusions=ins,projections=outs) : nothing
    return IndecomposableDecompositionResult(M,dims,[_schur_certificate() for _ in dims],
        :bipath_intervals,:native,(;search_steps=0,endomorphisms_tested=0,primary_splits=0),maps)
end
