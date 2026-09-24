# utils.jl -- low-level utilities for DerivedFunctors

module Utils
    using LinearAlgebra
    using SparseArrays

    # Sibling modules under TamerOp (two levels up from this nested module).
    using ...CoreModules: AbstractCoeffField, RealField, coeff_type, field_from_eltype
    using ...FieldLinAlg
    using ...Modules: PModule, PMorphism, _get_cover_cache
    using ...FiniteFringe: nvertices

    # ----------------------------
    # Basic utilities: morphism composition (local, explicit, reliable)
    # ----------------------------

    function compose(g::PMorphism{K}, f::PMorphism{K}) where {K}
        @assert f.cod === g.dom
        Q = f.dom.Q
        comps = Vector{Matrix{K}}(undef, nvertices(Q))
        for i in 1:nvertices(Q)
            comps[i] = FieldLinAlg._matmul(g.comps[i], f.comps[i])
        end
        return PMorphism{K}(f.dom, g.cod, comps)
    end

    function is_zero_matrix(field::AbstractCoeffField, A::AbstractMatrix)
        if field isa RealField
            isempty(A) && return true
            maxabs = maximum(abs, A)
            tol = field.atol + field.rtol * maxabs
            return maxabs <= tol
        end
        return all(iszero, A)
    end

    # Choose numerical pivots from A alone. Pivoting [A B] can choose a
    # right-hand-side column. Use QR column selection for conditioning without
    # forming a reduced matrix; solve on those independent coefficient columns
    # and leave the other variables zero.
    function solve_particular(field::RealField, A::AbstractMatrix, B::AbstractMatrix)
        K = coeff_type(field)
        A0, B0 = Matrix{K}(A), Matrix{K}(B)
        m, n = size(A0)
        size(B0, 1) == m || throw(DimensionMismatch("solve_particular: right-hand-side row count must match A"))
        X = zeros(K, n, size(B0, 2))
        isempty(B0) && return X
        if isempty(A0)
            is_zero_matrix(field, B0) || error("solve_particular: inconsistent system")
            return X
        end
        independent, pivots = FieldLinAlg._colspace_with_pivots(field, A0)
        if isempty(pivots)
            is_zero_matrix(field, B0) || error("solve_particular: inconsistent system")
            return X
        end
        X[pivots, :] = FieldLinAlg.solve_fullcolumn(field, independent, B0)
        return X
    end

    # Solve A*X = B (particular solution, free vars set to 0).
    function solve_particular(field::AbstractCoeffField, A::AbstractMatrix, B::AbstractMatrix)
        A0 = Matrix(A)
        B0 = Matrix(B)
        m, n = size(A0)
        @assert size(B0, 1) == m
        Aug = hcat(A0, B0)
        R, pivs_all = FieldLinAlg.rref(field, Aug)
        rhs = size(B0, 2)
        for i in 1:m
            if all(R[i, 1:n] .== 0)
                if any(R[i, n+1:n+rhs] .!= 0)
                    error("solve_particular: inconsistent system")
                end
            end
        end
        pivs = Int[]
        for p in pivs_all
            p <= n && push!(pivs, p)
        end
        X = zeros(eltype(A0), n, rhs)
        for (row, pcol) in enumerate(pivs)
            X[pcol, :] = R[row, n+1:n+rhs]
        end
        return X
    end

end
