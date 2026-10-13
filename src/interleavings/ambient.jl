struct InterleavingBudget
    max_vertices::Int
    max_matrix_entries::Int
    max_hom_variables::Int
    max_candidates::Int
    rational_height::Int
    function InterleavingBudget(; max_vertices=10_000, max_matrix_entries=2_000_000,
                                max_hom_variables=1024, max_candidates=10_000,
                                rational_height=8)
        all(x -> x isa Integer && x > 0,
            (max_vertices, max_matrix_entries, max_hom_variables, max_candidates, rational_height)) ||
            throw(ArgumentError("Interleaving budgets must be positive integers."))
        new(max_vertices, max_matrix_entries, max_hom_variables, max_candidates, rational_height)
    end
end

function _limit(value, maximum, label)
    value <= maximum || throw(ArgumentError("Interleaving $label budget exceeded ($value > $maximum)."))
end

function _coordinate(x)
    x isa Union{Integer,Rational,AbstractFloat} && isfinite(x) ||
        throw(ArgumentError("Coordinates must be finite integers, rationals, or represented binary floats."))
    return QQ(x)
end

function _cuts(axes, domain)
    domain in (:real, :integer) || throw(ArgumentError("domain must be :real or :integer."))
    isempty(axes) && throw(ArgumentError("At least one parameter axis is required."))
    result = map(a -> _coordinate.(collect(a)), Tuple(axes))
    all(a -> issorted(a) && allunique(a), result) ||
        throw(ArgumentError("Axis cuts must be strictly increasing."))
    domain === :integer && !all(a -> all(isinteger, a), result) &&
        throw(ArgumentError("Integer-domain cuts must be integers."))
    return result
end

_shape(cuts, domain) = map(a -> domain === :real ? 2length(a)+1 : length(a)+1, cuts)

function _grid(cuts, domain, budget)
    shape = _shape(cuts, domain)
    _limit(prod(big.(shape)), budget.max_vertices, "refinement vertices")
    return GridPoset(map(n -> collect(1:n), shape))
end

function _module_check(M, budget)
    M.field isa Union{QQField,PrimeField} ||
        throw(ArgumentError("Interleaving certificates require QQ or a prime field."))
    _limit(nvertices(M.Q), budget.max_vertices, "vertices")
    _limit(sum(big(length(A)) for (_, A) in M.edge_maps; init=big(0)),
           budget.max_matrix_entries, "module matrices")
    check_module_functoriality(M; throw=true)
end

function _snapshot(M::PModule{K}) where K
    maps = Dict{Tuple{Int,Int},Matrix{K}}((u,v) => Matrix{K}(A) for ((u,v),A) in M.edge_maps)
    return PModule{K}(M.Q, copy(M.dims), maps; field=M.field)
end

function _exact_matrix(field,A)
    K=coeff_type(field)
    A isa AbstractMatrix && all(x -> x isa K || x isa Union{Integer,Rational},A) ||
        throw(ArgumentError("Matrices must have exact coefficients in the specified field."))
    return Matrix{K}(map(x -> coerce(field,x),A))
end

"""
    RectilinearModule(M, cuts; domain=:real)
    RectilinearModule(cuts, dimensions, edge_maps; field=QQField(), domain=:real)

A global module on R^n or Z^n. For a real axis with m cuts the finite grid has
2m+1 strata: the lower open ray, the first singleton, the next open interval,
and so on, including the upper open ray. Integer axes have m+1 half-open bins.
The first axis varies fastest. Every exterior stalk is supplied, never inferred
from a cropped window. M must use a GridPoset of exactly this shape.
"""
struct RectilinearModule{N,M}
    finite::M
    cuts::NTuple{N,Vector{QQ}}
    domain::Symbol
    origin::NamedTuple
    function RectilinearModule(M::PModule, cuts; domain=:real,
                               budget=InterleavingBudget(), origin=(; kind=:explicit_global))
        c = _cuts(cuts, domain)
        M.Q isa GridPoset && M.Q.sizes == _shape(c, domain) ||
            throw(ArgumentError("The finite module must use the full stratum GridPoset, including exterior bins."))
        _module_check(M, budget)
        owned = _snapshot(M)
        new{length(c),typeof(owned)}(owned, c, domain, origin)
    end
end

function RectilinearModule(cuts, dims, maps; field=QQField(), domain=:real,
                           budget=InterleavingBudget())
    c = _cuts(cuts, domain)
    Q = _grid(c, domain, budget)
    length(dims)==nvertices(Q) && all(x -> x isa Integer && x>=0,dims) ||
        throw(ArgumentError("Supply a nonnegative integer dimension for every stratum."))
    _limit(sum(big(dims[u])*dims[v] for (u,v) in cover_edges(Q);init=big(0)),
           budget.max_matrix_entries,"module matrices")
    K=coeff_type(field)
    exact=Dict{Tuple{Int,Int},Matrix{K}}(e=>_exact_matrix(field,A) for (e,A) in maps)
    M = PModule{K}(Q, Int[x for x in dims], exact; field)
    return RectilinearModule(M, c; domain, budget)
end

function _axis_label(c, x, domain)
    j = searchsortedlast(c, x)
    domain === :integer && return j+1
    return 2j+1 - (j > 0 && c[j] == x)
end

function _label(cuts, domain, x)
    length(x) == length(cuts) || throw(ArgumentError("Wrong parameter dimension."))
    label, stride = 1, 1
    for i in eachindex(cuts)
        xi = _coordinate(x[i])
        domain === :integer && !isinteger(xi) && throw(ArgumentError("Integer-domain queries must be integral."))
        label += (_axis_label(cuts[i], xi, domain)-1)*stride
        stride *= domain === :real ? 2length(cuts[i])+1 : length(cuts[i])+1
    end
    return label
end

function _axis_representatives(c, domain)
    isempty(c) && return QQ[0]
    domain === :integer && return [c[1]-1; c]
    r = QQ[c[1]-1]
    for i in eachindex(c)
        push!(r, c[i])
        push!(r, i == length(c) ? c[i]+1 : (c[i]+c[i+1])/2)
    end
    return r
end

function _points(cuts, domain)
    axes = map(c -> _axis_representatives(c, domain), cuts)
    return vec(collect(Iterators.product(axes...)))
end

ambient_dimension(M::RectilinearModule, x) = M.finite.dims[_label(M.cuts, M.domain, x)]
locate(M::RectilinearModule, x::AbstractVector) = _label(M.cuts, M.domain, x)
locate(M::RectilinearModule, x::NTuple{N,<:Real}) where N = _label(M.cuts, M.domain, x)
encoding_module(M::RectilinearModule) = M.finite
encoding_axes(M::RectilinearModule) = map(copy,M.cuts)
function structure_map(M::RectilinearModule; source, target)
    length(source) == length(target) == length(M.cuts) || throw(ArgumentError("Wrong parameter dimension."))
    all(source .<= target) || throw(ArgumentError("Parameters must be coordinatewise comparable."))
    return copy(map_leq(M.finite, _label(M.cuts, M.domain, source), _label(M.cuts, M.domain, target)))
end

function _vector_shift(delta, n, domain; nonnegative=false)
    d = delta isa Real ? ntuple(_ -> _coordinate(delta), n) : Tuple(_coordinate.(delta))
    length(d) == n || throw(ArgumentError("Wrong translation dimension."))
    nonnegative && any(x -> x < 0, d) && throw(ArgumentError("Interleaving translations must be nonnegative."))
    domain === :integer && !all(isinteger, d) && throw(ArgumentError("Integer modules require integral translations."))
    return d
end

"""Translate parameters: parameter_shift(M, d)(x) = M(x+d), on the same global domain."""
function parameter_shift(M::RectilinearModule, delta; budget=InterleavingBudget())
    d = _vector_shift(delta, length(M.cuts), M.domain)
    c = ntuple(i -> M.cuts[i] .- d[i], length(d))
    return RectilinearModule(M.finite, c; domain=M.domain, budget,
        origin=(; kind=:parameter_shift, translation=d, source=M.origin))
end

function parameter_shift(M::EncodingResult,delta;extension=:none,budget=InterleavingBudget())
    return parameter_shift(ambient_module(M;extension,budget),delta;budget)
end

function _pull(M::PModule, Q, labels, budget)
    all(i -> 1 <= i <= nvertices(M.Q), labels) ||
        throw(ArgumentError("Encoding does not cover the whole ambient domain; no zero extension was assumed."))
    entries = sum(big(M.dims[labels[u]])*M.dims[labels[v]] for (u,v) in cover_edges(Q); init=big(0))
    _limit(entries, budget.max_matrix_entries, "pulled-back matrices")
    return pullback(EncodingMap(Q, M.Q, labels), M)
end

function _pull(M::RectilinearModule, Q, points, delta, budget)
    labels = [_label(M.cuts, M.domain, x .+ delta) for x in points]
    return _pull(M.finite, Q, labels, budget)
end

"""Refine a global encoding without changing its module or its boundary conventions."""
function refine_parameters(M::RectilinearModule, cuts; budget=InterleavingBudget())
    c = _cuts(cuts, M.domain)
    length(c) == length(M.cuts) && all(i -> issubset(M.cuts[i], c[i]), eachindex(c)) ||
        throw(ArgumentError("A refinement must retain every original cut on every axis."))
    Q = _grid(c, M.domain, budget)
    pulled = _pull(M, Q, _points(c,M.domain), ntuple(_ -> QQ(0),length(c)), budget)
    return RectilinearModule(pulled, c; domain=M.domain, budget, origin=M.origin)
end

function _global_encoding(M, pi, domain, budget)
    c = _cuts(map(a -> sort!(unique(a)), axes_from_encoding(pi)), domain)
    Q = _grid(c, domain, budget)
    points = _points(c, domain)
    labels = [locate(pi, domain === :integer ? Tuple(BigInt.(x)) : x) for x in points]
    pulled = _pull(M, Q, labels, budget)
    return RectilinearModule(pulled, c; domain, budget,
                            origin=(; kind=:ambient_encoding, map_type=Symbol(nameof(typeof(pi)))))
end

"""
    ambient_module(enc; extension=:none)

Recover the global rectilinear module of a box or lattice encoding. A
GridEncodingMap needs the explicit extension=:zero_below declaration: its
out-of-domain lower ideal is zero and its last bins continue to infinity.
This constructs that extension; it does not certify an unknown uncropped module.
Other geometric encodings are rejected.
"""
function ambient_module(enc::EncodingResult; extension=:none, budget=InterleavingBudget())
    return ambient_module(encoding_module(enc), encoding_map(enc); extension, budget)
end
function ambient_module(M::PModule, pi::CompiledEncoding; kwargs...)
    M.Q === encoding_poset(pi) || throw(ArgumentError("Compiled encoding and module posets differ."))
    return ambient_module(M, encoding_map(pi); kwargs...)
end
function ambient_module(M::PModule, pi::Union{PLEncodingMapBoxes,ZnEncodingMap};
                        extension=:none, budget=InterleavingBudget())
    extension === :none || throw(ArgumentError("A global box/lattice encoding needs no extension."))
    _module_check(M,budget)
    return _global_encoding(M, pi, pi isa ZnEncodingMap ? :integer : :real, budget)
end
function ambient_module(M::PModule{K}, pi::GridEncodingMap;
                        extension=:none, budget=InterleavingBudget()) where K
    extension === :zero_below || throw(ArgumentError("Grid encodings require explicit extension=:zero_below; a window does not specify ambient stalks."))
    all(==(1),pi.orientation) || throw(ArgumentError("Ambient shifts currently require increasing grid axes."))
    M.Q === pi.P || throw(ArgumentError("Grid encoding and module posets differ."))
    _module_check(M,budget)
    c = _cuts(pi.coords, :real)
    Q = _grid(c,:real,budget)
    labels = [locate(pi,x) for x in _points(c,:real)]
    dims = [i == 0 ? 0 : M.dims[i] for i in labels]
    _limit(sum(big(dims[u])*dims[v] for (u,v) in cover_edges(Q);init=big(0)),budget.max_matrix_entries,"extension matrices")
    maps = Dict{Tuple{Int,Int},Matrix{K}}((u,v) => (labels[u] == 0 || labels[v] == 0 ? zeros(K,dims[v],dims[u]) :
                Matrix{K}(map_leq(M,labels[u],labels[v]))) for (u,v) in cover_edges(Q))
    return RectilinearModule(PModule{K}(Q,dims,maps;field=M.field),c;budget,
        origin=(; kind=:declared_extension, extension=:zero_below, upper=:constant_tail))
end
ambient_module(::PModule, pi; kwargs...) = throw(ArgumentError("Unsupported ambient encoding $(typeof(pi)); supply an exact global RectilinearModule."))

provenance(M::RectilinearModule) = (; domain=M.domain, field=M.finite.field,
    boundary=M.domain === :real ? :singleton_and_open_strata : :left_closed_integer_bins,
    window=nothing, exterior=:explicit, origin=M.origin)
describe(M::RectilinearModule) = (; kind=:rectilinear_module, parameters=length(M.cuts),
    vertices=nvertices(M.finite.Q), provenance=provenance(M))
Base.show(io::IO, M::RectilinearModule) = print(io,"RectilinearModule(",M.domain,", ",length(M.cuts)," parameters, ",nvertices(M.finite.Q)," strata)")
