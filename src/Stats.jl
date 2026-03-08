# =============================================================================
# Stats.jl
#
# Small statistics helpers shared by encoding and invariant layers.
# =============================================================================
module Stats

function _normal_quantile(p::Real)
    if p <= 0
        return -Inf
    elseif p >= 1
        return Inf
    end

    a = (-3.969683028665376e+01,
          2.209460984245205e+02,
         -2.759285104469687e+02,
          1.383577518672690e+02,
         -3.066479806614716e+01,
          2.506628277459239e+00)

    b = (-5.447609879822406e+01,
          1.615858368580409e+02,
         -1.556989798598866e+02,
          6.680131188771972e+01,
         -1.328068155288572e+01)

    c = (-7.784894002430293e-03,
         -3.223964580411365e-01,
         -2.400758277161838e+00,
         -2.549732539343734e+00,
          4.374664141464968e+00,
          2.938163982698783e+00)

    d = (7.784695709041462e-03,
         3.224671290700398e-01,
         2.445134137142996e+00,
         3.754408661907416e+00)

    plow = 0.02425
    phigh = 1 - plow

    if p < plow
        q = sqrt(-2 * log(p))
        return (((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) /
               ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1)
    elseif p > phigh
        q = sqrt(-2 * log(1 - p))
        return -(((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) /
                ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1)
    else
        q = p - 0.5
        r = q * q
        return (((((a[1] * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * r + a[6]) * q /
               (((((b[1] * r + b[2]) * r + b[3]) * r + b[4]) * r + b[5]) * r + 1)
    end
end

@inline function _wilson_interval(x::Integer, n::Integer; alpha::Real=0.05)
    if n <= 0
        return (0.0, 1.0)
    end
    if x < 0 || x > n
        throw(ArgumentError("x must satisfy 0 <= x <= n"))
    end

    z = _normal_quantile(1 - float(alpha) / 2)
    phat = x / n
    denom = 1 + z^2 / n
    center = (phat + z^2 / (2n)) / denom
    half = (z / denom) * sqrt((phat * (1 - phat) + z^2 / (4n)) / n)

    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (lo, hi)
end

end # module Stats
