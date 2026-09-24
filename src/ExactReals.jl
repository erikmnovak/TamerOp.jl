"""
Exact real algebraic coordinate values for filtration grades and queries.

`AlgebraicReal` preserves the represented value of rational and floating-point
inputs and supports exact arithmetic, ordering, and square roots. Explicit
`Float64`/`BigFloat` conversion is reserved for numerical output and plotting.
"""
module ExactReals

import Nemo

"""
    AlgebraicReal(x::Real)

A finite real algebraic number with exact arithmetic and comparisons. Rational
inputs are retained exactly; finite floating-point inputs mean their represented
binary rational value. For example, `sqrt(AlgebraicReal(2))^2 == 2` exactly.

Rhomboid filtrations use these values for physical radius grades, so distinct
critical radii remain distinct even when their `Float64` displays coincide.
Converting an irrational value to a rational or integer throws `InexactError`.
`Float64(x)` and `BigFloat(x)` explicitly request numerical approximations.
"""
struct AlgebraicReal <: Real
    value::Nemo.QQBarFieldElem
    function AlgebraicReal(value::Nemo.QQBarFieldElem)
        isreal(value) || throw(DomainError(value, "AlgebraicReal requires a real value."))
        return new(value)
    end
end

AlgebraicReal(x::AlgebraicReal) = x
AlgebraicReal(x::Union{Integer,Rational}) = AlgebraicReal(Nemo.QQBarFieldElem(x))
function AlgebraicReal(x::AbstractFloat)
    isfinite(x) || throw(DomainError(x, "AlgebraicReal requires a finite value."))
    return AlgebraicReal(Rational{BigInt}(x))
end
function AlgebraicReal(x::Real)
    throw(ArgumentError("Cannot construct an exact AlgebraicReal from $(typeof(x)); supply an integer, rational, finite float, or an algebraic expression."))
end

Base.convert(::Type{AlgebraicReal}, x::Real) = AlgebraicReal(x)
Base.promote_rule(::Type{AlgebraicReal}, ::Type{T}) where {T<:Union{Integer,Rational,AbstractFloat}} = AlgebraicReal
Base.zero(::Type{AlgebraicReal}) = AlgebraicReal(0)
Base.one(::Type{AlgebraicReal}) = AlgebraicReal(1)
Base.zero(::AlgebraicReal) = zero(AlgebraicReal)
Base.one(::AlgebraicReal) = one(AlgebraicReal)
Base.iszero(x::AlgebraicReal) = iszero(x.value)
Base.isone(x::AlgebraicReal) = isone(x.value)
Base.isinteger(x::AlgebraicReal) = isinteger(x.value)
Base.isfinite(::AlgebraicReal) = true
Base.isinf(::AlgebraicReal) = false
Base.isnan(::AlgebraicReal) = false
Base.signbit(x::AlgebraicReal) = Nemo.sign_real(x.value) < 0
Base.sign(x::AlgebraicReal) = AlgebraicReal(Nemo.sign_real(x.value))
Base.abs(x::AlgebraicReal) = signbit(x) ? -x : x
Base.abs2(x::AlgebraicReal) = x * x
Base.real(x::AlgebraicReal) = x
Base.conj(x::AlgebraicReal) = x
Base.imag(::AlgebraicReal) = zero(AlgebraicReal)
Base.float(x::AlgebraicReal) = x
Base.float(::Type{AlgebraicReal}) = AlgebraicReal
Base.big(x::AlgebraicReal) = x
Base.big(::Type{AlgebraicReal}) = AlgebraicReal
Base.widen(::Type{AlgebraicReal}) = AlgebraicReal

Base.:+(x::AlgebraicReal) = x
Base.:-(x::AlgebraicReal) = AlgebraicReal(-x.value)
Base.:+(x::AlgebraicReal, y::AlgebraicReal) = AlgebraicReal(x.value + y.value)
Base.:-(x::AlgebraicReal, y::AlgebraicReal) = AlgebraicReal(x.value - y.value)
Base.:*(x::AlgebraicReal, y::AlgebraicReal) = AlgebraicReal(x.value * y.value)
Base.:/(x::AlgebraicReal, y::AlgebraicReal) = AlgebraicReal(Nemo.divexact(x.value, y.value))
Base.inv(x::AlgebraicReal) = AlgebraicReal(inv(x.value))
Base.:^(x::AlgebraicReal, n::Integer) = AlgebraicReal(x.value^n)
Base.:^(x::AlgebraicReal, n::Rational) = AlgebraicReal(x.value^n)
function Base.sqrt(x::AlgebraicReal)
    signbit(x) && throw(DomainError(x, "A real square root requires a nonnegative value."))
    return AlgebraicReal(sqrt(x.value))
end

Base.:(==)(x::AlgebraicReal, y::AlgebraicReal) = x.value == y.value
Base.:<(x::AlgebraicReal, y::AlgebraicReal) = isless(x.value, y.value)
Base.:<=(x::AlgebraicReal, y::AlgebraicReal) = !isless(y.value, x.value)
Base.isless(x::AlgebraicReal, y::AlgebraicReal) = x < y
Base.isequal(x::AlgebraicReal, y::AlgebraicReal) = x == y

# Mixed comparisons also support infinite endpoints and Julia's total ordering
# of NaN and signed zero, without admitting those values as algebraic grades.
const _RationalInput = Union{Integer,Rational}
for op in (:(==), :(<), :(<=), :isless, :isequal)
    @eval begin
        Base.$op(x::AlgebraicReal, y::_RationalInput) = Base.$op(x, AlgebraicReal(y))
        Base.$op(x::_RationalInput, y::AlgebraicReal) = Base.$op(AlgebraicReal(x), y)
    end
end
Base.:(==)(x::AlgebraicReal, y::AbstractFloat) = isfinite(y) && x == AlgebraicReal(y)
Base.:(==)(x::AbstractFloat, y::AlgebraicReal) = y == x
Base.:<(x::AlgebraicReal, y::AbstractFloat) = isfinite(y) ? x < AlgebraicReal(y) : y == Inf
Base.:<(x::AbstractFloat, y::AlgebraicReal) = isfinite(x) ? AlgebraicReal(x) < y : x == -Inf
Base.:<=(x::AlgebraicReal, y::AbstractFloat) = isfinite(y) ? x <= AlgebraicReal(y) : y == Inf
Base.:<=(x::AbstractFloat, y::AlgebraicReal) = isfinite(x) ? AlgebraicReal(x) <= y : x == -Inf
Base.isless(x::AlgebraicReal, y::AbstractFloat) = isnan(y) || x < y
Base.isless(x::AbstractFloat, y::AlgebraicReal) = x < y || (iszero(x) && signbit(x) && iszero(y))
Base.isequal(x::AlgebraicReal, y::AbstractFloat) = x == y && !(iszero(y) && signbit(y))
Base.isequal(x::AbstractFloat, y::AlgebraicReal) = isequal(y, x)
Base.min(x::AlgebraicReal, y::AbstractFloat) = isnan(y) || isless(y, x) ? y : x
Base.min(x::AbstractFloat, y::AlgebraicReal) = min(y, x)
Base.max(x::AlgebraicReal, y::AbstractFloat) = isnan(y) || isless(x, y) ? y : x
Base.max(x::AbstractFloat, y::AlgebraicReal) = max(y, x)
Base.minmax(x::AlgebraicReal, y::AbstractFloat) = (min(x, y), max(x, y))
Base.minmax(x::AbstractFloat, y::AlgebraicReal) = minmax(y, x)

function Base.isapprox(x::AlgebraicReal, y::AlgebraicReal;
                       atol::Real=0, rtol::Real=atol > 0 ? 0 : sqrt(eps(Float64)),
                       nans::Bool=false, norm=abs)
    return x == y || norm(x - y) <= max(atol, rtol * max(norm(x), norm(y)))
end
Base.isapprox(x::AlgebraicReal, y::Union{Integer,Rational,AbstractFloat}; kwargs...) =
    isfinite(y) && isapprox(x, AlgebraicReal(y); kwargs...)
Base.isapprox(x::Union{Integer,Rational,AbstractFloat}, y::AlgebraicReal; kwargs...) =
    isapprox(y, x; kwargs...)

function (::Type{Rational{T}})(x::AlgebraicReal) where {T<:Integer}
    Nemo.is_rational(x.value) || throw(InexactError(:convert, Rational{T}, x))
    return Rational{T}(x.value)
end
Base.Rational(x::AlgebraicReal) = Rational{BigInt}(x)
function (::Type{T})(x::AlgebraicReal) where {T<:Integer}
    isinteger(x) || throw(InexactError(:convert, T, x))
    return T(x.value)
end
Base.Float64(x::AlgebraicReal) = Float64(x.value)
Base.Float32(x::AlgebraicReal) = Float32(BigFloat(x; precision=64))
Base.Float16(x::AlgebraicReal) = Float16(BigFloat(x; precision=64))
function Base.BigFloat(x::AlgebraicReal, rounding::RoundingMode=RoundNearest;
                       precision::Integer=Base.precision(BigFloat))
    precision > 0 || throw(ArgumentError("BigFloat precision must be positive."))
    return setprecision(BigFloat, Int(precision)) do
        ball = Nemo.ArbField(Int(precision) + 32; cached=false)(x.value)
        BigFloat(ball, rounding)
    end
end
Base.convert(::Type{T}, x::AlgebraicReal) where {T<:Union{Integer,Rational,AbstractFloat}} = T(x)

function Base.hash(x::AlgebraicReal, seed::UInt)
    # Julia requires equal values to hash identically across numeric types.
    Nemo.is_rational(x.value) && return hash(Rational{BigInt}(x), seed)
    # Construct a local polynomial parent: Nemo's default qqbar hash enters
    # the global cached polynomial-ring registry, including during threaded
    # reads of immutable coordinate arrays. Conjugate roots may hash alike;
    # equality still distinguishes them exactly.
    ring, _ = Nemo.polynomial_ring(Nemo.ZZ, :x; cached=false)
    polynomial = Nemo.minpoly(ring, x.value)
    result = hash(:AlgebraicReal, seed)
    for coefficient in Nemo.coefficients(polynomial)
        result = hash(BigInt(coefficient), result)
    end
    return result
end

Base.floor(::Type{T}, x::AlgebraicReal) where {T<:Integer} = T(floor(Nemo.ZZRingElem, x.value))
Base.ceil(::Type{T}, x::AlgebraicReal) where {T<:Integer} = T(ceil(Nemo.ZZRingElem, x.value))
Base.trunc(::Type{T}, x::AlgebraicReal) where {T<:Integer} = signbit(x) ? ceil(T, x) : floor(T, x)
Base.round(::Type{T}, x::AlgebraicReal, mode::RoundingMode=RoundNearest) where {T<:Integer} =
    T(round(Nemo.ZZRingElem, x.value, mode))
Base.round(::Type{T}, x::AlgebraicReal, ::RoundingMode{:ToZero}) where {T<:Integer} = trunc(T, x)
Base.round(::Type{T}, x::AlgebraicReal, ::RoundingMode{:FromZero}) where {T<:Integer} =
    signbit(x) ? floor(T, x) : ceil(T, x)
Base.round(::Type{T}, x::AlgebraicReal, ::RoundingMode{:NearestTiesUp}) where {T<:Integer} =
    floor(T, x + 1//2)
Base.floor(x::AlgebraicReal) = AlgebraicReal(floor(BigInt, x))
Base.ceil(x::AlgebraicReal) = AlgebraicReal(ceil(BigInt, x))
Base.trunc(x::AlgebraicReal) = AlgebraicReal(trunc(BigInt, x))
Base.round(x::AlgebraicReal, mode::RoundingMode=RoundNearest) = AlgebraicReal(round(BigInt, x, mode))

function Base.show(io::IO, x::AlgebraicReal)
    if Nemo.is_rational(x.value)
        show(io, Rational{BigInt}(x))
    else
        square = x.value * x.value
        if Nemo.is_rational(square)
            signbit(x) && print(io, "-")
            print(io, "sqrt(")
            show(io, Rational{BigInt}(square))
            print(io, ")")
        else
            ring, _ = Nemo.polynomial_ring(Nemo.ZZ, :x; cached=false)
            polynomial = Nemo.minpoly(ring, x.value)
            roots = sort!(filter(isreal, Nemo.roots(Nemo.QQBar, polynomial)); lt=isless)
            index = findfirst(==(x.value), roots)
            print(io, "AlgebraicReal(root(")
            show(io, polynomial)
            print(io, ", ", index, "))")
        end
    end
end

end # module ExactReals
