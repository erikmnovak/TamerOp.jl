# Exact division-quotient certificates for rational endomorphism algebras.
module TamerOpHeckeExt

import Hecke
import TamerOp
import TamerOp.IndecomposableDecompositions as D
import TamerOp.CoreModules as CM

function D._rational_division_certificate(L::Vector{Matrix{CM.QQ}}, budget)
    d = length(L)
    D._bound(8big(d)^3+4big(d)^2,budget.max_matrix_entries,
        "Hecke algebra conversion workspace")
    # Our L[i][:,j] represents e_i*e_j, with e_1=1. Hecke's table uses [i,j,k].
    table = [Hecke.QQ(L[i][k,j]) for i in 1:d, j in 1:d, k in 1:d]
    identity = [Hecke.QQ(i == 1) for i in 1:d]
    A = Hecke.structure_constant_algebra(Hecke.QQ,table;one=identity,check=true)
    radical = Hecke.radical(A)
    rb = Hecke.basis_matrix(radical)
    J = CM.QQ[D._coefficient(CM.QQField(),rb[j,i]) for i in 1:d, j in 1:Hecke.nrows(rb)]
    # Check nilpotence in TamerOp's own multiplication table. The radical is
    # obtained by the regular trace pairing in characteristic zero.
    nilindex = D._nilpotency_index(CM.QQField(),L,J)
    nilindex === nothing && error("Hecke returned a nonnilpotent radical.")
    S,_ = Hecke.quo(A,radical)
    Hecke.is_simple(S) || return nothing
    z = Hecke.dimension_of_center(S)
    r = Hecke.dim(S)
    index = Hecke.schur_index_over_center(S)
    # For S = Mat_m(D) over its center Z, dim_Z(S) = (m*index)^2.
    # Simplicity alone would incorrectly certify every matrix algebra.
    r == z*big(index)^2 || return nothing
    return D._Certificate(:rational_division_quotient,d,size(J,2),r,nilindex,z,index)
end

end
