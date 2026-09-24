# Exact coordinate storage is independent of the coefficient field. Ordinary
# Float64 arrays keep the existing columnar path; exact payloads never pass
# through JSON numbers or a floating approximation of an algebraic root.

Base.@kwdef mutable struct _AlgebraicCoordinateJSON
    polynomial::Vector{String} = String[] # ascending integer coefficients
    real_root_index::Int = 0             # increasing order, starting at one
end

Base.@kwdef mutable struct _ExactCoordinateRowsJSON
    scalar_kind::String = ""
    offsets::Vector{Int} = Int[]
    rational_values::Vector{String} = String[]
    algebraic_values::Vector{_AlgebraicCoordinateJSON} = _AlgebraicCoordinateJSON[]
end

JSON3.StructTypes.StructType(::Type{_AlgebraicCoordinateJSON}) = JSON3.StructTypes.Mutable()
JSON3.StructTypes.StructType(::Type{_ExactCoordinateRowsJSON}) = JSON3.StructTypes.Mutable()

const _COORDINATE_POLYNOMIAL_RING = first(Nemo.polynomial_ring(Nemo.ZZ, :x; cached=false))

function _algebraic_coordinate_obj(x::AlgebraicReal)
    p = Nemo.minpoly(_COORDINATE_POLYNOMIAL_RING, x.value)
    roots = sort!(filter(isreal, Nemo.roots(Nemo.QQBar, p)))
    index = findfirst(==(x.value), roots)
    index === nothing && error("exact coordinate has no real root in its minimal polynomial")
    return _AlgebraicCoordinateJSON(
        polynomial=[string(Nemo.coeff(p, i)) for i in 0:Nemo.degree(p)],
        real_root_index=index)
end

function _algebraic_coordinate_from_obj(obj::_AlgebraicCoordinateJSON)
    length(obj.polynomial) >= 2 || throw(ArgumentError("algebraic coordinate requires a nonconstant minimal polynomial"))
    coefficients = BigInt[]
    for text in obj.polynomial
        n = tryparse(BigInt, text)
        n !== nothing && string(n) == text ||
            throw(ArgumentError("algebraic polynomial coefficients must be canonical decimal integers"))
        push!(coefficients, n)
    end
    last(coefficients) > 0 || throw(ArgumentError("algebraic minimal polynomial must have positive leading coefficient"))
    foldl(gcd, coefficients; init=big(0)) == 1 ||
        throw(ArgumentError("algebraic minimal polynomial must be primitive"))
    p = _COORDINATE_POLYNOMIAL_RING(coefficients)
    roots = sort!(filter(isreal, Nemo.roots(Nemo.QQBar, p)))
    1 <= obj.real_root_index <= length(roots) ||
        throw(ArgumentError("algebraic coordinate real_root_index is outside the sorted real roots"))
    root = roots[obj.real_root_index]
    Nemo.minpoly(_COORDINATE_POLYNOMIAL_RING, root) == p ||
        throw(ArgumentError("algebraic coordinate polynomial must be the root's minimal polynomial"))
    return AlgebraicReal(root)
end

function _algebraic_coordinate_from_obj(obj)
    haskey(obj, "polynomial") && haskey(obj, "real_root_index") ||
        throw(ArgumentError("algebraic coordinate requires polynomial and real_root_index"))
    return _algebraic_coordinate_from_obj(_AlgebraicCoordinateJSON(
        polynomial=String.(obj["polynomial"]), real_root_index=Int(obj["real_root_index"])))
end

function _exact_rational_coordinate(text::AbstractString)
    q = try
        string_to_rational(text)
    catch
        throw(ArgumentError("invalid exact rational coordinate"))
    end
    isfinite(q) && rational_to_string(q) == text ||
        throw(ArgumentError("exact rational coordinates require canonical num/den strings"))
    return q
end

# Nothing means the ordinary numeric rows are already an exact representation
# of the requested Float64 values. Integer and other finite Real inputs are
# stored as rationals; algebraic values retain their real-root data.
function _exact_coordinate_rows(rows)
    all(row -> eltype(row) <: Union{Float16,Float32,Float64}, rows) && return nothing
    offsets = Int[1]
    algebraic = any(row -> eltype(row) <: AlgebraicReal || any(x -> x isa AlgebraicReal, row), rows)
    rationals = String[]
    algebraics = _AlgebraicCoordinateJSON[]
    for row in rows
        for x in row
            x isa Real && isfinite(x) || throw(ArgumentError("exact coordinates must be finite real numbers"))
            if algebraic
                push!(algebraics, _algebraic_coordinate_obj(AlgebraicReal(x)))
            else
                push!(rationals, rational_to_string(QQ(x)))
            end
        end
        push!(offsets, algebraic ? length(algebraics) + 1 : length(rationals) + 1)
    end
    return _ExactCoordinateRowsJSON(scalar_kind=algebraic ? "algebraic_real" : "rational",
        offsets=offsets, rational_values=rationals, algebraic_values=algebraics)
end

function _exact_coordinate_rows_from_obj(obj::_ExactCoordinateRowsJSON)
    values = if obj.scalar_kind == "rational"
        isempty(obj.algebraic_values) || throw(ArgumentError("rational coordinates cannot contain algebraic_values"))
        [_exact_rational_coordinate(x) for x in obj.rational_values]
    elseif obj.scalar_kind == "algebraic_real"
        isempty(obj.rational_values) || throw(ArgumentError("algebraic coordinates cannot contain rational_values"))
        [_algebraic_coordinate_from_obj(x) for x in obj.algebraic_values]
    else
        throw(ArgumentError("unknown exact coordinate scalar_kind: $(obj.scalar_kind)"))
    end
    offsets = obj.offsets
    !isempty(offsets) && first(offsets) == 1 && last(offsets) == length(values) + 1 && issorted(offsets) ||
        throw(ArgumentError("exact coordinate offsets must partition the stored values"))
    return [values[offsets[i]:(offsets[i + 1] - 1)] for i in 1:(length(offsets) - 1)]
end

function _exact_coordinate_rows_from_obj(obj)
    haskey(obj, "scalar_kind") && haskey(obj, "offsets") ||
        throw(ArgumentError("exact coordinates require scalar_kind and offsets"))
    return _exact_coordinate_rows_from_obj(_ExactCoordinateRowsJSON(
        scalar_kind=String(obj["scalar_kind"]), offsets=Int.(obj["offsets"]),
        rational_values=String.(get(obj, "rational_values", String[])),
        algebraic_values=[_AlgebraicCoordinateJSON(polynomial=String.(x["polynomial"]),
            real_root_index=Int(x["real_root_index"])) for x in get(obj, "algebraic_values", [])]))
end

function _coordinate_rows_from_obj(rows, exact)
    exact === nothing && return [Float64.(row) for row in rows]
    isempty(rows) || throw(ArgumentError("exact coordinates must not also contain numeric coordinate rows"))
    return _exact_coordinate_rows_from_obj(exact)
end

function _coordinate_vector_from_obj(values, exact)
    exact === nothing && return Vector{Float64}(values)
    rows = _coordinate_rows_from_obj(values, exact)
    length(rows) == 1 || throw(ArgumentError("exact vector payload must contain one coordinate row"))
    return only(rows)
end

function _store_coordinate_rows!(obj, key::String, rows)
    exact = _exact_coordinate_rows(rows)
    obj[key] = exact === nothing ? [Float64.(row) for row in rows] : []
    exact === nothing || (obj["exact_" * key] = exact)
    return obj
end

function _store_coordinate_vector!(obj, key::String, values)
    exact = _exact_coordinate_rows((values,))
    obj[key] = exact === nothing ? Float64.(values) : []
    exact === nothing || (obj["exact_" * key] = exact)
    return obj
end

# Filtration specs contain heterogeneous parameters. Tag exact scalars in place
# rather than coercing the full parameter tree to one numeric type.
function _coordinate_parameter_obj(x)
    if x isa AlgebraicReal
        value = _algebraic_coordinate_obj(x)
        return Dict("scalar_kind" => "algebraic_real", "polynomial" => value.polynomial,
                    "real_root_index" => value.real_root_index)
    elseif x isa Rational || x isa BigInt || x isa BigFloat
        return Dict("scalar_kind" => "rational", "value" => rational_to_string(QQ(x)))
    elseif x isa Union{Float16,Float32}
        return Float64(x)
    elseif x isa NamedTuple || x isa AbstractDict
        return Dict(string(k) => _coordinate_parameter_obj(v) for (k, v) in pairs(x))
    elseif x isa AbstractArray || x isa Tuple
        return map(_coordinate_parameter_obj, x)
    end
    return x
end

function _coordinate_parameter_from_obj(x)
    if x isa AbstractDict
        if haskey(x, "scalar_kind")
            kind = String(x["scalar_kind"])
            kind == "rational" && return _exact_rational_coordinate(String(x["value"]))
            kind == "algebraic_real" && return _algebraic_coordinate_from_obj(x)
            throw(ArgumentError("unknown exact parameter scalar_kind: $kind"))
        end
        return Dict(k => _coordinate_parameter_from_obj(v) for (k, v) in pairs(x))
    elseif x isa AbstractVector
        result = map(_coordinate_parameter_from_obj, x)
        if !isempty(result) && all(v -> v isa Real, result)
            T = any(v -> v isa AlgebraicReal, result) ? AlgebraicReal :
                any(v -> v isa Rational, result) ? QQ : promote_type(map(typeof, result)...)
            return T.(result)
        end
        return result
    end
    return x
end
