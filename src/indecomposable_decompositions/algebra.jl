# Exact endomorphism polynomials and independently checkable local-algebra tests.

"""
    IndecomposableDecompositionBudget(; max_vertices=512, max_dimension=512,
        max_matrix_entries=8_000_000, max_endomorphism_dimension=128,
        max_search_steps=10_000, max_bipath_intervals=100_000)

Limits checked before Hom, multiplication-table and bipath-family allocations.
Search steps count coefficient vectors considered across the entire recursive
decomposition, including vectors skipped as redundant, and Hecke division
decisions (one step each). Exhaustion throws
`IndecomposableSearchExhausted`; it is not a proof of indecomposability.
Matrix limits count scalar entries, not rational coefficient bit sizes.
They bound the adapter's dense workspace, not the internal allocations or running
time of optional Hecke number-field and maximal-order arithmetic.
"""
struct IndecomposableDecompositionBudget
    max_vertices::Int
    max_dimension::Int
    max_matrix_entries::Int
    max_endomorphism_dimension::Int
    max_search_steps::Int
    max_bipath_intervals::Int
    function IndecomposableDecompositionBudget(; max_vertices=512, max_dimension=512,
            max_matrix_entries=8_000_000, max_endomorphism_dimension=128,
            max_search_steps=10_000, max_bipath_intervals=100_000)
        xs = (max_vertices,max_dimension,max_matrix_entries,max_endomorphism_dimension,
              max_search_steps,max_bipath_intervals)
        all(x -> x isa Integer && !(x isa Bool) && 0 < x <= typemax(Int), xs) ||
            throw(ArgumentError("Decomposition budgets must be positive machine integers."))
        return new(Int.(xs)...)
    end
end

"""A bounded exact search stopped without a certified complete decomposition."""
struct IndecomposableSearchExhausted <: Exception
    steps::Int
    dimensions::Vector{Int}
    rational::Bool
end
function Base.showerror(io::IO, e::IndecomposableSearchExhausted)
    print(io,"Indecomposable decomposition exhausted its search budget after ",e.steps,
        " steps on dimensions ",e.dimensions,". No complete decomposition was certified.")
    e.rational && print(io," If using backend=:native, noncommutative division quotients require the optional backend=:hecke certificate.")
end

_bound(n, limit, what) = n <= limit ? nothing :
    throw(ArgumentError("Indecomposable decomposition: $what exceeds its resource budget ($limit)."))

function _context(M, budget)
    M.field isa Union{QQField,PrimeField} ||
        throw(ArgumentError("Indecomposable decomposition requires rational or prime-field coefficients."))
    n = nvertices(M.Q)
    _bound(n,budget.max_vertices,"vertices")
    _bound(big(n)^2,budget.max_matrix_entries,"finite-base validation")
    _bound(sum(big,M.dims;init=big(0)),budget.max_dimension,"total stalk dimension")
    _bound(sum(d -> big(d)^2,M.dims;init=big(0)),budget.max_matrix_entries,"stalk matrices")
    report = check_module_functoriality(M)
    report.valid || throw(ArgumentError("Invalid finite module: "*join(report.issues,"; ")))
    return nothing
end

mutable struct _SearchState
    steps::Int
    tested::Int
    splits::Int
end
function _charge!(state, M, budget)
    state.steps < budget.max_search_steps ||
        throw(IndecomposableSearchExhausted(state.steps,copy(M.dims),M.field isa QQField))
    state.steps += 1
end

# Compact certificates describe the proof used, not a probabilistic confidence.
struct _Certificate
    kind::Symbol
    end_dimension::Int
    radical_dimension::Union{Nothing,Int}
    residue_dimension::Union{Nothing,Int}
    nilpotency_index::Union{Nothing,Int}
    center_degree::Union{Nothing,Int}
    schur_index::Union{Nothing,Int}
end
_Certificate(kind,d,j,r,n) = _Certificate(kind,d,j,r,n,
    kind in (:endomorphism_field,:local_algebra) ? r : nothing,
    kind in (:endomorphism_field,:local_algebra) ? 1 : nothing)
_schur_certificate() = _Certificate(:endomorphism_field,1,0,1,1)
_certificate_summary(c::_Certificate) = (; kind=c.kind, end_dimension=c.end_dimension,
    radical_dimension=c.radical_dimension,residue_dimension=c.residue_dimension,
    nilpotency_index=c.nilpotency_index,center_degree=c.center_degree,schur_index=c.schur_index)

function _check_backend(backend, field)
    backend in (:native,:hecke) || throw(ArgumentError("Decomposition backend must be :native or :hecke."))
    if backend == :hecke
        field isa QQField || throw(ArgumentError("backend=:hecke requires rational coefficients; use :native for prime fields."))
        Base.get_extension(parentmodule(@__MODULE__),:TamerOpHeckeExt) === nothing &&
            throw(ArgumentError("backend=:hecke requires Hecke. Install it with Pkg.add(\"Hecke\") and run `using Hecke`."))
    end
    return nothing
end

# Implemented by the optional exact rational-algebra extension. Returning nothing
# means the quotient is not a division algebra, never an indecomposability proof.
function _rational_division_certificate end

_nemo_field(::QQField) = Nemo.QQ
_nemo_field(F::PrimeField) = Nemo.GF(F.p)
_nemo_matrix(::QQField, F, A) = Nemo.matrix(F,Matrix{QQ}(A))
_nemo_matrix(::PrimeField, F, A) = Nemo.matrix(F,map(x -> x.val,A))
_coefficient(::QQField, x) = QQ(BigInt(Nemo.numerator(x)),BigInt(Nemo.denominator(x)))
_coefficient(F::PrimeField, x) = coeff_type(F)(BigInt(Nemo.lift(Nemo.ZZ,x)))

function _polynomial_matrix(field, p, A::AbstractMatrix{K}) where {K}
    n = size(A,1)
    V = zeros(K,n,n)
    for k in Nemo.degree(p):-1:0
        V = V*A
        c = _coefficient(field,Nemo.coeff(p,k))
        for i in 1:n
            V[i,i] += c
        end
    end
    return V
end

function _primary_factors(field, R, F, components)
    m = one(R)
    for A in components
        isempty(A) && continue
        m = lcm(m,Nemo.minpoly(R,_nemo_matrix(field,F,A)))
    end
    fs = collect(Nemo.factor(m))
    # FLINT's factor-container iteration order is not a public ordering contract.
    sort!(fs;by=f -> (Nemo.degree(first(f)),string(first(f))))
    return fs
end

function _endomorphisms(M, budget)
    nv = sum(d -> big(d)^2,M.dims;init=big(0))
    _bound(nv^2,budget.max_matrix_entries,"Hom workspace")
    H = Hom(M,M)
    B = H.basis_matrix
    d = size(B,2)
    _bound(d,budget.max_endomorphism_dimension,"endomorphism dimension")
    # Identity followed by a complement makes affine/scalar-equivalent searches
    # unnecessary. A nontrivial idempotent always has a nonzero complement class.
    K = coeff_type(M.field)
    identity = reduce(vcat,[vec(eye(M.field,n)) for n in M.dims];init=K[])
    c = FieldLinAlg.solve_fullcolumn(M.field,B,identity)
    pivot = findfirst(!iszero,c)
    pivot === nothing && error("Nonzero module has zero identity endomorphism.")
    C = hcat(identity,B[:,[j for j in 1:d if j != pivot]])
    offsets = cumsum([0;M.dims.^2])
    F = _nemo_field(M.field)
    R,_ = Nemo.polynomial_ring(F,"t";cached=false)
    return (; B=C,offsets,F,R)
end

function _components(M::PModule{K}, A, c) where {K}
    flat = A.B*c
    return Matrix{K}[reshape(flat[A.offsets[v]+1:A.offsets[v+1]],n,n)
        for (v,n) in enumerate(M.dims)]
end

function _product_coordinates(M, A, X, Y)
    K = coeff_type(M.field)
    flat = reduce(vcat,[vec(X[v]*Y[v]) for v in eachindex(M.dims)];init=K[])
    return FieldLinAlg.solve_fullcolumn(M.field,A.B,flat)
end

# Left regular representation of End(M), built only when direct factorization
# has not already split M. Table bounds are checked before allocating it.
function _multiplication_table(M, A, budget)
    d = size(A.B,2)
    _bound(4big(d)^3+big(d)^2,budget.max_matrix_entries,"endomorphism multiplication workspace")
    K = coeff_type(M.field)
    es = eye(M.field,d)
    comps = [_components(M,A,es[:,j]) for j in 1:d]
    return [reduce(hcat,[_product_coordinates(M,A,comps[i],comps[j]) for j in 1:d]) for i in 1:d]
end

function _multiply(L, a, b)
    c = zeros(eltype(b),length(b))
    for i in eachindex(a)
        iszero(a[i]) || (c .+= a[i].*(L[i]*b))
    end
    return c
end

function _ideal(field, L, seeds)
    J = FieldLinAlg.colspace(field,seeds)
    d = length(L)
    E = eye(field,d)
    while 0 < size(J,2) < d
        products = [J]
        for i in 1:d
            push!(products,L[i]*J)
            push!(products,reduce(hcat,[_multiply(L,J[:,j],E[:,i]) for j in axes(J,2)]))
        end
        next = FieldLinAlg.colspace(field,reduce(hcat,products))
        size(next,2) == size(J,2) && return J
        J = next
    end
    return J
end

function _nilpotency_index(field, L, J)
    isempty(J) && return 1
    power = J
    for exponent in 2:length(L)+1
        products = reduce(hcat,vec([_multiply(L,power[:,i],J[:,j])
            for i in axes(power,2), j in axes(J,2)]))
        next = FieldLinAlg.colspace(field,products)
        size(next,2) == 0 && return exponent
        size(next,2) == size(power,2) && return nothing
        power = next
    end
    return nothing
end

function _local_certificate(M, A, records, L)
    K = coeff_type(M.field)
    d = size(A.B,2)
    seeds = zeros(K,d,0)
    for (c,q) in records
        comps = _components(M,A,c)
        flat = reduce(vcat,[vec(_polynomial_matrix(M.field,q,X)) for X in comps];init=K[])
        seed = FieldLinAlg.solve_fullcolumn(M.field,A.B,flat)
        all(iszero,seed) || (seeds = FieldLinAlg.colspace(M.field,hcat(seeds,seed)))
    end
    J = _ideal(M.field,L,seeds)
    nilindex = _nilpotency_index(M.field,L,J)
    nilindex === nothing && return nothing
    r = d-size(J,2)
    # Each recorded irreducible q annihilates its element modulo J. If deg(q)
    # equals dim(A/J), that element generates the entire quotient as a field.
    # A nilpotent ideal with division quotient is the Jacobson radical, so A is
    # local and M indecomposable (Fitting/Krull-Schmidt).
    any(record -> Nemo.degree(last(record)) == r,records) || return nothing
    return _Certificate(:local_algebra,d,size(J,2),r,nilindex)
end
