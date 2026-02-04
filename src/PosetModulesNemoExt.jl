module PosetModulesNemoExt

import PosetModules
import Nemo

const QQ = PosetModules.CoreModules.QQ

# Flip the runtime flag (optional, but useful for debugging / telemetry).
PosetModules.FieldLinAlg._NEMO_ENABLED[] = true

# Conversions between Matrix{QQ} and Nemo.fmpq_mat.
function _to_fmpq_mat(A::AbstractMatrix{QQ})
    m, n = size(A)
    S = Nemo.MatrixSpace(Nemo.QQ, m, n)
    M = S()
    @inbounds for i in 1:m, j in 1:n
        M[i, j] = Nemo.QQ(A[i, j])
    end
    return M
end

# Conversions between Matrix{FpElem{p}} and Nemo matrices over GF(p).
function _to_nemo_fp_mat(A::AbstractMatrix{PosetModules.CoreModules.FpElem{p}}) where {p}
    m, n = size(A)
    Fp, _ = Nemo.GF(p)
    S = Nemo.MatrixSpace(Fp, m, n)
    M = S()
    @inbounds for i in 1:m, j in 1:n
        M[i, j] = Fp(A[i, j].val)
    end
    return M
end

function _from_nemo_fp_mat(M, ::Val{p}) where {p}
    m, n = size(M)
    A = Matrix{PosetModules.CoreModules.FpElem{p}}(undef, m, n)
    @inbounds for i in 1:m, j in 1:n
        x = M[i, j]
        A[i, j] = PosetModules.CoreModules.FpElem{p}(Int(x))
    end
    return A
end

function _from_fmpq_mat(M::Nemo.fmpq_mat)
    m, n = size(M)
    A = Matrix{QQ}(undef, m, n)
    @inbounds for i in 1:m, j in 1:n
        x = M[i, j]
        A[i, j] = QQ(BigInt(Nemo.numerator(x)), BigInt(Nemo.denominator(x)))
    end
    return A
end

# Nemo-backed implementations for FieldLinAlg (QQ only).
function PosetModules.FieldLinAlg._nemo_rref(::PosetModules.CoreModules.QQField,
                                             A::AbstractMatrix{QQ}; pivots::Bool=true)
    M = _to_fmpq_mat(A)
    _, R = Nemo.rref(M)
    Rq = _from_fmpq_mat(R)

    pivs = Int[]
    m, n = size(Rq)
    @inbounds for i in 1:m
        for j in 1:n
            if Rq[i, j] != 0
                push!(pivs, j)
                break
            end
        end
    end

    return pivots ? (Rq, Tuple(pivs)) : Rq
end

PosetModules.FieldLinAlg._nemo_rank(::PosetModules.CoreModules.QQField,
                                    A::AbstractMatrix{QQ}) = Nemo.rank(_to_fmpq_mat(A))

function PosetModules.FieldLinAlg._nemo_nullspace(::PosetModules.CoreModules.QQField,
                                                  A::AbstractMatrix{QQ})
    _, N = Nemo.nullspace(_to_fmpq_mat(A))
    return _from_fmpq_mat(N)
end

function PosetModules.FieldLinAlg._nemo_rref(F::PosetModules.CoreModules.PrimeField,
                                             A::AbstractMatrix{PosetModules.CoreModules.FpElem{p}};
                                             pivots::Bool=true) where {p}
    p > 3 || error("nemo_rref: only for p > 3")
    F.p == p || error("nemo_rref: field mismatch")
    M = _to_nemo_fp_mat(A)
    _, R = Nemo.rref(M)
    Rf = _from_nemo_fp_mat(R, Val(p))

    pivs = Int[]
    m, n = size(Rf)
    @inbounds for i in 1:m
        for j in 1:n
            if Rf[i, j].val != 0
                push!(pivs, j)
                break
            end
        end
    end
    return pivots ? (Rf, Tuple(pivs)) : Rf
end

PosetModules.FieldLinAlg._nemo_rank(F::PosetModules.CoreModules.PrimeField,
                                    A::AbstractMatrix{PosetModules.CoreModules.FpElem{p}}) where {p} =
    (p > 3 ? (F.p == p ? Nemo.rank(_to_nemo_fp_mat(A)) : error("nemo_rank: field mismatch")) :
     error("nemo_rank: only for p > 3"))

function PosetModules.FieldLinAlg._nemo_nullspace(F::PosetModules.CoreModules.PrimeField,
                                                  A::AbstractMatrix{PosetModules.CoreModules.FpElem{p}}) where {p}
    p > 3 || error("nemo_nullspace: only for p > 3")
    F.p == p || error("nemo_nullspace: field mismatch")
    _, N = Nemo.nullspace(_to_nemo_fp_mat(A))
    return _from_nemo_fp_mat(N, Val(p))
end

end # module
