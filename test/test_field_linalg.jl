using Random
using LinearAlgebra
using SparseArrays

@testset "FieldLinAlg engines" begin
    FL = PosetModules.FieldLinAlg
    F2 = CM.F2()
    F2Elem = CM.FpElem{2}

    function f2mat(A::AbstractMatrix{<:Integer})
        return F2Elem.(A .% 2)
    end

    function f2mat(v::AbstractVector{<:Integer})
        return F2Elem.(v .% 2)
    end

    function f2rank_naive(A::Matrix{Int})
        M = copy(A .% 2)
        m, n = size(M)
        r = 0
        c = 1
        while r < m && c <= n
            piv = 0
            for i in r+1:m
                if M[i, c] == 1
                    piv = i
                    break
                end
            end
            if piv == 0
                c += 1
                continue
            end
            r += 1
            if piv != r
                M[r, :], M[piv, :] = M[piv, :], M[r, :]
            end
            for i in 1:m
                i == r && continue
                if M[i, c] == 1
                    @inbounds for j in c:n
                        M[i, j] = xor(M[i, j], M[r, j])
                    end
                end
            end
            c += 1
        end
        return r
    end

    @testset "F2 rank + rank_dim (dense)" begin
        Aint = [
            1 0 1 1 0;
            1 1 0 1 1;
            0 1 1 0 1;
            1 1 1 0 0
        ]
        A = f2mat(Aint)
        r1 = FL.rank(F2, A)
        r2 = FL.rank_dim(F2, A)
        @test r1 == r2
        @test r1 == f2rank_naive(Aint)
    end

    @testset "F2 rank + rank_dim (sparse)" begin
        Aint = [
            1 0 1 0 0;
            0 1 1 0 1;
            1 1 0 1 0;
            0 0 1 1 1
        ]
        A = sparse(f2mat(Aint))
        r1 = FL.rank(F2, A)
        r2 = FL.rank_dim(F2, A)
        @test r1 == r2
        @test r1 == f2rank_naive(Aint)
    end

    @testset "F2 nullspace + rref" begin
        Aint = [
            1 0 1 0;
            0 1 1 1;
            1 1 0 1
        ]
        A = f2mat(Aint)
        N = FL.nullspace(F2, A)
        Z = A * N
        @test all(x -> x.val == 0, Z)

        R, pivs = FL.rref(F2, A; pivots=true)
        @test length(pivs) == FL.rank(F2, A)
        @test all(p -> 1 <= p <= size(A, 2), pivs)
    end

    @testset "F2 solve_fullcolumn (dense + sparse)" begin
        n = 6
        m = 10
        B = vcat(f2mat(Matrix{Int}(I, n, n)), f2mat(rand(0:1, m - n, n)))
        X = f2mat(rand(0:1, n, 3))
        Y = B * X
        Xhat = FL.solve_fullcolumn(F2, B, Y)
        @test Xhat == X

        Bs = sparse(B)
        Y2 = Bs * X
        Xhat2 = FL.solve_fullcolumn(F2, Bs, Y2)
        @test Xhat2 == X
    end

    @testset "F2 rank_restricted + colspace sparse" begin
        Aint = [
            1 0 1 1 0;
            0 1 1 0 1;
            1 1 0 1 1;
            0 0 1 1 0
        ]
        A = sparse(f2mat(Aint))
        rows = [1, 3, 4]
        cols = [2, 3, 5]
        sub = Matrix{Int}(Aint[rows, cols])
        @test FL.rank_restricted(F2, A, rows, cols) == f2rank_naive(sub)

        C = FL.colspace(F2, A)
        @test C isa SparseMatrixCSC
    end

    @testset "F2 rank_restricted dense" begin
        Aint = [
            1 0 1 1 0;
            0 1 1 0 1;
            1 1 0 1 1;
            0 0 1 1 0
        ]
        A = f2mat(Aint)
        rows = [1, 3, 4]
        cols = [2, 3, 5]
        sub = Matrix{Int}(Aint[rows, cols])
        r = FL.rank(F2, A[rows, cols])
        @test r == f2rank_naive(sub)
    end

    @testset "F2 rref sparse pivot consistency" begin
        Aint = [
            1 0 1 0 1;
            0 1 1 1 0;
            1 1 0 1 1;
            0 0 1 1 0
        ]
        A = sparse(f2mat(Aint))
        R, pivs = FL.rref(F2, A; pivots=true)
        @test length(pivs) == FL.rank(F2, A)
        @test all(p -> 1 <= p <= size(A, 2), pivs)
        @test R == FL.rref(F2, Matrix(A); pivots=false)
    end

    @testset "F2 solve_fullcolumn RHS check" begin
        B = f2mat([1 0 0;
                   0 1 0;
                   1 1 0;
                   0 0 1])
        y_bad = f2mat([1, 0, 0, 0])
        @test_throws ErrorException FL.solve_fullcolumn(F2, B, y_bad; check_rhs=true)
        x = FL.solve_fullcolumn(F2, B, y_bad; check_rhs=false)
        @test B * x != y_bad
    end

    @testset "F2 solve_fullcolumn cache parity" begin
        B = vcat(f2mat(Matrix{Int}(I, 3, 3)), f2mat([1 1 0; 0 1 1]))
        X = f2mat([1 0; 0 1; 1 1])
        Y = B * X

        FL.clear_f2_fullcolumn_cache!()
        x1 = FL.solve_fullcolumn(F2, B, Y; cache=true)
        @test x1 == X
        @test haskey(FL._F2_FULLCOLUMN_FACTOR_CACHE, B)

        x2 = FL.solve_fullcolumn(F2, B, Y; cache=true)
        @test x2 == X

        # Factor reuse via explicit factor argument
        fact = FL._F2_FULLCOLUMN_FACTOR_CACHE[B]
        x3 = FL.solve_fullcolumn(F2, B, Y; cache=false, factor=fact)
        @test x3 == X
    end

    @testset "F2 edge cases" begin
        A0 = f2mat(zeros(Int, 0, 5))
        @test FL.rank(F2, A0) == 0
        N0 = FL.nullspace(F2, A0)
        @test size(N0) == (5, 5)

        A1 = f2mat(zeros(Int, 4, 0))
        @test FL.rank(F2, A1) == 0
        N1 = FL.nullspace(F2, A1)
        @test size(N1) == (0, 0)

        Z = f2mat(zeros(Int, 3, 4))
        N = FL.nullspace(F2, Z)
        @test size(N) == (4, 4)
        @test Z * N == f2mat(zeros(Int, 3, 4))
    end

    @testset "F2 randomized properties" begin
        rng = MersenneTwister(20240202)

        for _ in 1:20
            m = rand(rng, 2:8)
            n = rand(rng, 2:8)
            A = f2mat(rand(rng, 0:1, m, n))

            r = FL.rank(F2, A)
            N = FL.nullspace(F2, A)
            @test size(N, 1) == n
            @test size(N, 2) == n - r
            @test A * N == f2mat(zeros(Int, m, size(N, 2)))

            C = FL.colspace(F2, A)
            @test FL.rank(F2, C) == r
        end
    end

    @testset "F2 sparse nullspace agrees with dense" begin
        rng = MersenneTwister(4242)
        for _ in 1:10
            m = rand(rng, 3:8)
            n = rand(rng, 3:8)
            nnz_target = rand(rng, 3:(m * n))
            I = rand(rng, 1:m, nnz_target)
            J = rand(rng, 1:n, nnz_target)
            V = [F2Elem(rand(rng, 0:1)) for _ in 1:nnz_target]
            A = sparse(I, J, V, m, n)
            dropzeros!(A)

            Ns = FL.nullspace(F2, A)
            Nd = FL.nullspace(F2, Matrix(A))

            @test size(Ns, 1) == n
            @test size(Ns, 2) == size(Nd, 2)
            @test A * Ns == f2mat(zeros(Int, m, size(Ns, 2)))
            @test FL.rank(F2, Ns) == size(Ns, 2)
        end
    end

    @testset "FpElem scalar/coercion properties" begin
        @testset "F2 arithmetic" begin
            K2 = CM.FpElem{2}
            a = K2(1)
            b = K2(1)
            @test a + b == K2(0)
            @test a * b == K2(1)
            @test a / b == K2(1)
        end

        @testset "F3 arithmetic + inverses" begin
            K3 = CM.FpElem{3}
            a = K3(2)
            @test a + K3(2) == K3(1)
            @test a * K3(2) == K3(1)
            @test a / K3(2) == K3(1)
            @test inv(a) * a == K3(1)
        end

        @testset "Fp(5) coercion + inverses" begin
            F5 = CM.Fp(5)
            K5 = CM.FpElem{5}
            @test CM.coerce(F5, 7) == K5(2)
            @test CM.coerce(F5, -1) == K5(4)
            @test inv(K5(2)) * K5(2) == K5(1)
        end
    end

    @testset "change_field helpers" begin
        FQ = CM.QQField()
        F2 = CM.F2()
        K = CM.coeff_type(FQ)
        @inline c(x) = CM.coerce(FQ, x)

        # --- Flange ---
        flats = [FZ.IndFlat(FZ.face(1, [1]), [0])]
        inj = [FZ.IndInj(FZ.face(1, [1]), [0])]
        phi = reshape(K[c(2)], 1, 1)
        FG = FZ.Flange(1, flats, inj, phi; field=FQ)
        FG2 = CM.change_field(FG, F2)
        @test FG2.field == F2
        @test eltype(FG2.phi) == CM.FpElem{2}
        @test FG2.phi[1, 1] == CM.FpElem{2}(0)

        # --- FringeModule ---
        P = chain_poset(2)
        U = FF.principal_upset(P, 1)
        D = FF.principal_downset(P, 1)
        phiH = spzeros(K, 1, 1)
        phiH[1, 1] = c(1)
        H = FF.FringeModule{K}(P, [U], [D], phiH)
        H2 = CM.change_field(H, F2)
        @test H2.field == F2
        @test eltype(H2.phi) == CM.FpElem{2}
        @test H2.phi[1, 1] == CM.FpElem{2}(1)

        # --- PModule + PMorphism ---
        edge = Dict{Tuple{Int,Int}, Matrix{K}}()
        edge[(1, 2)] = reshape(K[c(2)], 1, 1)
        M = MD.PModule{K}(P, [1, 1], edge)
        M2 = CM.change_field(M, F2)
        @test M2.field == F2
        @test eltype(M2.edge_maps.maps_to_succ[1][1]) == CM.FpElem{2}
        @test M2.edge_maps.maps_to_succ[1][1][1, 1] == CM.FpElem{2}(0)

        comps = [reshape(K[c(1)], 1, 1), reshape(K[c(2)], 1, 1)]
        f = MD.PMorphism(M, M, comps)
        f2 = CM.change_field(f, F2)
        @test f2.dom.field == F2
        @test f2.cod.field == F2
        @test f2.comps[1][1, 1] == CM.FpElem{2}(1)
        @test f2.comps[2][1, 1] == CM.FpElem{2}(0)

        # --- EncodingResult ---
        enc = CM.EncodingResult(P, M, nothing; H=H, presentation=FG)
        enc2 = CM.change_field(enc, F2)
        @test enc2.M.field == F2
        @test enc2.H.field == F2
        @test enc2.presentation.field == F2

        # --- ResolutionResult ---
        res = CM.ResolutionResult(M; enc=enc)
        res2 = CM.change_field(res, F2)
        @test res2.enc.M.field == F2
        @test res2.res.field == F2

        # --- InvariantResult ---
        inv = CM.InvariantResult(enc, :dummy, 7)
        inv2 = CM.change_field(inv, F2)
        @test inv2.enc.M.field == F2
        @test inv2.value == 7
    end

    @testset "SparseRowAccumulator correctness" begin
        K = QQ
        acc = FL.SparseRowAccumulator{K}(12)
        row = FL.SparseRow{K}()

        FL.reset_sparse_row_accumulator!(acc)
        FL.push_sparse_row_entry!(acc, 4, QQ(2))
        FL.push_sparse_row_entry!(acc, 2, QQ(3))
        FL.push_sparse_row_entry!(acc, 4, QQ(-2)) # cancellation
        FL.push_sparse_row_entry!(acc, 7, QQ(5))
        FL.push_sparse_row_entry!(acc, 2, QQ(1))
        FL.materialize_sparse_row!(row, acc)
        @test row.idx == [2, 7]
        @test row.val == QQ[4, 5]

        FL.reset_sparse_row_accumulator!(acc)
        FL.push_sparse_row_entry!(acc, 1, QQ(1))
        FL.materialize_sparse_row!(row, acc)
        @test row.idx == [1]
        @test row.val == QQ[1]
    end

    @testset "backend matrix storage hooks" begin
        FQ = CM.QQField()
        K = CM.coeff_type(FQ)
        P = chain_poset(2)
        old_nemo = FL._NEMO_ENABLED[]
        try
            FL._NEMO_ENABLED[] = true
            A = Matrix{K}(I, 256, 256) # above default Nemo threshold (50_000 entries)
            edge = Dict{Tuple{Int,Int}, Matrix{K}}((1, 2) => A)
            M = MD.PModule{K}(P, [256, 256], edge; field=FQ)
            @test M.edge_maps[1, 2] isa CM.BackendMatrix{K}
            @test Matrix(M.edge_maps[1, 2]) == A

            comps = [Matrix{K}(I, 256, 256), Matrix{K}(I, 256, 256)]
            f = MD.PMorphism(M, M, comps)
            @test f.comps[1] isa CM.BackendMatrix{K}
            @test f.comps[2] isa CM.BackendMatrix{K}
            @test Matrix(f.comps[1]) == comps[1]
        finally
            FL._NEMO_ENABLED[] = old_nemo
        end
    end

    @testset "F3 rank + rank_dim (dense)" begin
        F3 = CM.F3()
        F3Elem = CM.FpElem{3}
        Aint = [
            1 2 0 1;
            2 1 1 0;
            1 1 2 2;
            0 2 1 1
        ]
        A = F3Elem.(Aint .% 3)
        r1 = FL.rank(F3, A)
        r2 = FL.rank_dim(F3, A)
        @test r1 == r2
        @test 0 <= r1 <= min(size(A)...)
    end

    @testset "F3 nullspace + rref" begin
        F3 = CM.F3()
        F3Elem = CM.FpElem{3}
        Aint = [
            1 2 0 1;
            0 1 1 2;
            2 1 1 0
        ]
        A = F3Elem.(Aint .% 3)
        N = FL.nullspace(F3, A)
        @test A * N == zeros(F3Elem, size(A, 1), size(N, 2))

        R, pivs = FL.rref(F3, A; pivots=true)
        @test length(pivs) == FL.rank(F3, A)
        @test all(p -> 1 <= p <= size(A, 2), pivs)
        @test size(R) == size(A)
    end

    @testset "F3 solve_fullcolumn (dense + sparse)" begin
        F3 = CM.F3()
        F3Elem = CM.FpElem{3}
        n = 4
        m = 7
        B = vcat(F3Elem.(Matrix{Int}(I, n, n)), F3Elem.(rand(0:2, m - n, n)))
        X = F3Elem.(rand(0:2, n, 2))
        Y = B * X
        Xhat = FL.solve_fullcolumn(F3, B, Y)
        @test Xhat == X

        Bs = sparse(B)
        Y2 = Bs * X
        Xhat2 = FL.solve_fullcolumn(F3, Bs, Y2)
        @test Xhat2 == X
    end

    @testset "F3 rank_restricted + colspace sparse" begin
        F3 = CM.F3()
        F3Elem = CM.FpElem{3}
        Aint = [
            1 2 0 1 2;
            2 1 1 0 1;
            1 1 2 2 0;
            0 2 1 1 2
        ]
        A = sparse(F3Elem.(Aint .% 3))
        rows = [1, 3, 4]
        cols = [2, 3, 5]
        r1 = FL.rank_restricted(F3, A, rows, cols)
        r2 = FL.rank(F3, Matrix(A)[rows, cols])
        @test r1 == r2

        C = FL.colspace(F3, A)
        @test size(C, 1) == size(A, 1)
        @test FL.rank(F3, C) == FL.rank(F3, A)
    end

    @testset "F3 edge cases" begin
        F3 = CM.F3()
        F3Elem = CM.FpElem{3}
        A0 = F3Elem.(zeros(Int, 0, 5))
        @test FL.rank(F3, A0) == 0
        N0 = FL.nullspace(F3, A0)
        @test size(N0) == (5, 5)

        A1 = F3Elem.(zeros(Int, 4, 0))
        @test FL.rank(F3, A1) == 0
        N1 = FL.nullspace(F3, A1)
        @test size(N1) == (0, 0)
    end

    @testset "F3 solve_fullcolumn cache parity" begin
        F3 = CM.F3()
        F3Elem = CM.FpElem{3}
        B = vcat(F3Elem.(Matrix{Int}(I, 3, 3)), F3Elem.([1 2 0; 2 1 1]))
        X = F3Elem.([1 0; 2 1; 1 2])
        Y = B * X

        FL.clear_f3_fullcolumn_cache!()
        x1 = FL.solve_fullcolumn(F3, B, Y; cache=true)
        @test x1 == X
        @test haskey(FL._F3_FULLCOLUMN_FACTOR_CACHE, B)

        x2 = FL.solve_fullcolumn(F3, B, Y; cache=true)
        @test x2 == X

        fact = FL._F3_FULLCOLUMN_FACTOR_CACHE[B]
        x3 = FL.solve_fullcolumn(F3, B, Y; cache=false, factor=fact)
        @test x3 == X
    end

    @testset "QQ vs F2 vs F3 cache parity (basic behavior)" begin
        FL.clear_fullcolumn_cache!()
        FL.clear_f2_fullcolumn_cache!()
        FL.clear_f3_fullcolumn_cache!()
        @test isempty(FL._FULLCOLUMN_FACTOR_CACHE)
        @test isempty(FL._F2_FULLCOLUMN_FACTOR_CACHE)
        @test isempty(FL._F3_FULLCOLUMN_FACTOR_CACHE)
    end

    @testset "F3 randomized properties" begin
        F3 = CM.F3()
        F3Elem = CM.FpElem{3}
        rng = MersenneTwister(20240203)

        for _ in 1:20
            m = rand(rng, 2:8)
            n = rand(rng, 2:8)
            A = F3Elem.(rand(rng, 0:2, m, n))

            r = FL.rank(F3, A)
            N = FL.nullspace(F3, A)
            @test size(N, 1) == n
            @test size(N, 2) == n - r
            @test A * N == zeros(F3Elem, m, size(N, 2))

            C = FL.colspace(F3, A)
            @test FL.rank(F3, C) == r
        end
    end

    @testset "F3 sparse nullspace agrees with dense" begin
        F3 = CM.F3()
        F3Elem = CM.FpElem{3}
        rng = MersenneTwister(4243)
        for _ in 1:10
            m = rand(rng, 3:8)
            n = rand(rng, 3:8)
            nnz_target = rand(rng, 3:(m * n))
            I = rand(rng, 1:m, nnz_target)
            J = rand(rng, 1:n, nnz_target)
            V = [F3Elem(rand(rng, 0:2)) for _ in 1:nnz_target]
            A = sparse(I, J, V, m, n)
            dropzeros!(A)

            Ns = FL.nullspace(F3, A)
            Nd = FL.nullspace(F3, Matrix(A))

            @test size(Ns, 1) == n
            @test size(Ns, 2) == size(Nd, 2)
            @test A * Ns == zeros(F3Elem, m, size(Ns, 2))
            @test FL.rank(F3, Ns) == size(Ns, 2)
        end
    end

    @testset "FieldLinAlg contract tests (all fields)" begin
        rng = MersenneTwister(20250202)

        function _is_real(field)
            return field isa CM.RealField
        end

        function _tol(field, A)
            field isa CM.RealField || return zero(eltype(A))
            return field.atol + field.rtol * opnorm(Matrix(A), 1)
        end

        with_fields(FIELDS_FULL) do field
            K = CM.coeff_type(field)
            @testset "contracts over $(field)" begin
                m = rand(rng, 4:9)
                n = rand(rng, 4:9)
                A = CM.rand(field, m, n)

                r = FL.rank(field, A)
                @test r == FL.rank_dim(field, A)

                N = FL.nullspace(field, A)
                @test size(N, 1) == n
                @test size(N, 2) == n - r
                if _is_real(field)
                    @test norm(Matrix(A) * N) <= _tol(field, A) + 1e-8
                else
                    @test A * N == zeros(K, m, size(N, 2))
                end

                C = FL.colspace(field, A)
                @test FL.rank(field, C) == r

                # solve_fullcolumn for full column rank
                ncol = rand(rng, 2:5)
                nrow = ncol + rand(rng, 1:5)
                B = vcat(CM.eye(field, ncol), CM.rand(field, nrow - ncol, ncol))
                X = CM.rand(field, ncol, 2)
                Y = B * X
                Xhat = FL.solve_fullcolumn(field, B, Y)
                if _is_real(field)
                    @test norm(Matrix(B) * Xhat - Matrix(Y)) <= _tol(field, B) + 1e-8
                else
                    @test B * Xhat == Y
                end
            end
        end
    end

    @testset "rank_restricted API parity (dense + sparse)" begin
        with_fields(FIELDS_FULL) do field
            @testset "restricted rank over $(field)" begin
                Aint = [
                    1 0 2 1 0 3 1;
                    0 1 1 0 2 1 0;
                    1 1 0 2 1 0 2;
                    0 0 1 1 1 2 0;
                    2 1 0 1 0 1 1;
                    1 0 1 0 1 0 1
                ]
                A = CM.coerce.(Ref(field), Aint)
                As = sparse(A)
                rows = [1, 3, 4, 6]
                cols = [1, 2, 5, 7]

                rd = FL.rank_restricted(field, A, rows, cols)
                rd_ref = FL.rank(field, A[rows, cols])
                @test rd == rd_ref

                rs = FL.rank_restricted(field, As, rows, cols)
                rs_ref = FL.rank(field, Matrix(As)[rows, cols])
                @test rs == rs_ref

                @test FL.rank_restricted(field, A, Int[], cols) == 0
                @test FL.rank_restricted(field, A, rows, Int[]) == 0
                @test FL.rank_restricted(field, As, Int[], cols) == 0
                @test FL.rank_restricted(field, As, rows, Int[]) == 0
            end
        end
    end

    @testset "restricted nullspace/solve API parity (dense + sparse)" begin
        rng = MersenneTwister(20260206)
        with_fields(FIELDS_FULL) do field
            K = CM.coeff_type(field)
            is_real = field isa CM.RealField
            tol = is_real ? (field.atol + field.rtol + 1e-8) : 0.0

            @testset "restricted nullspace/solve over $(field)" begin
                Aint = [
                    1 0 2 1 0;
                    0 1 1 0 2;
                    1 1 0 2 1;
                    0 0 1 1 1;
                    2 1 0 1 0
                ]
                A = CM.coerce.(Ref(field), Aint)
                As = sparse(A)
                rows = [1, 3, 4]
                cols = [1, 2, 5]

                Nd = FL.nullspace_restricted(field, A, rows, cols)
                Ns = FL.nullspace_restricted(field, As, rows, cols)
                Aref = A[rows, cols]
                if is_real
                    @test norm(Matrix(Aref) * Nd) <= tol
                    @test norm(Matrix(Aref) * Ns) <= tol
                else
                    @test Aref * Nd == zeros(K, length(rows), size(Nd, 2))
                    @test Aref * Ns == zeros(K, length(rows), size(Ns, 2))
                end
                @test size(Nd, 1) == length(cols)
                @test size(Ns, 1) == length(cols)

                Bn = 3
                Bm = 6
                Bfull = vcat(CM.eye(field, Bn), CM.rand(field, Bm - Bn, Bn))
                Xtrue = CM.rand(field, Bn, 2)
                Yfull = Bfull * Xtrue
                srows = [1, 2, 3, 5]
                scols = [1, 2, 3]
                Xd = FL.solve_fullcolumn_restricted(field, Bfull, srows, scols, Yfull)
                Xs = FL.solve_fullcolumn_restricted(field, sparse(Bfull), srows, scols, Yfull)
                if is_real
                    @test norm(Matrix(Bfull[srows, scols]) * Xd - Matrix(Yfull[srows, :])) <= tol
                    @test norm(Matrix(Bfull[srows, scols]) * Xs - Matrix(Yfull[srows, :])) <= tol
                else
                    @test Bfull[srows, scols] * Xd == Yfull[srows, :]
                    @test Bfull[srows, scols] * Xs == Yfull[srows, :]
                end

                @test size(FL.nullspace_restricted(field, A, Int[], cols), 2) == length(cols)
                @test size(FL.nullspace_restricted(field, As, rows, Int[])) == (0, 0)
            end
        end
    end

    @testset "Characteristic-sensitive rank (small matrix)" begin
        A = [2 0;
             0 0]

        rqq = FL.rank(CM.QQField(), QQ.(A))
        rf2 = FL.rank(CM.F2(), CM.FpElem{2}.(A))
        rf3 = FL.rank(CM.F3(), CM.FpElem{3}.(A))

        @test rqq == 1
        @test rf2 == 0
        @test rf3 == 1
    end

    let field = CM.QQField()
    @testset "QQ linear algebra (FieldLinAlg QQ engine)" begin
        # Rank test
        A = QQ[QQ(1) QQ(2);
               QQ(2) QQ(4)]
        @test FL.rankQQ(A) == 1
        @test FL.rank(CM.QQField(), A) == 1

        # Nullspace test: A * v = 0
        N = FL.nullspaceQQ(A)
        @test size(N, 1) == 2
        @test size(N, 2) == 1
        v = N[:, 1]
        @test A * v == zeros(QQ, 2)

        # Solve full column rank system B*x = y
        B = QQ[QQ(1) QQ(0);
               QQ(0) QQ(1);
               QQ(1) QQ(1)]
        x_true = QQ[QQ(1), QQ(2)]
        y = B * x_true
        x = FL.solve_fullcolumnQQ(B, y)
        @test B * x == y
        @test FL.solve_fullcolumn(CM.QQField(), B, y) == x

        # Multiple right-hand sides
        Y = hcat(y, QQ(2) .* y)
        X = FL.solve_fullcolumnQQ(B, Y)
        @test B * X == Y

        @testset "rrefQQ / colspaceQQ / solve_fullcolumnQQ edge cases" begin
            A = QQ[1 2 3;
                   2 4 6;
                   1 1 1]
            R, piv = FL.rrefQQ(A)

            @test piv == (1, 2)
            @test R == QQ[1 0 -1;
                          0 1  2;
                          0 0  0]

            C = FL.colspaceQQ(A)
            @test size(C) == (3, 2)
            @test C == A[:, collect(piv)]

            B = QQ[1 2;
                   2 4;
                   3 6]
            b = QQ[1, 2, 3]
            @test_throws ErrorException FL.solve_fullcolumnQQ(B, b)
        end
    end

    @testset "QQ sparse colspace agrees with dense" begin
        A = sparse(QQ[1 0 2;
                      0 1 3;
                      0 0 0])
        Cd = FL.colspaceQQ(Matrix{QQ}(A))
        Cs = FL.colspaceQQ(A)

        @test FL.rankQQ(Cd) == FL.rankQQ(Cs)
        @test FL.rankQQ(Cs) == FL.rankQQ(A)

        B = FL.colspaceQQ(A)
        for j in 1:size(Cs,2)
            x = FL.solve_fullcolumnQQ(B, Cs[:,j])
            @test B*x == Cs[:,j]
        end
    end

    @testset "QQ FullColumnFactor + cache correctness" begin
        B = QQ[1 0;
               0 1;
               1 1]
        y = B * QQ[2, 3]

        FL.clear_fullcolumn_cache!()
        x1 = FL.solve_fullcolumnQQ(B, y; cache=true)
        @test B*x1 == y
        @test haskey(FL._FULLCOLUMN_FACTOR_CACHE, B)
        x2 = FL.solve_fullcolumnQQ(B, y; cache=true)
        @test x2 == x1
        bady = QQ[1,0,0]
        @test_throws ErrorException FL.solve_fullcolumnQQ(B, bady; cache=true)

        # Nemo factor/cache path
        FL.clear_fullcolumn_cache!()
        xn = FL.solve_fullcolumn(CM.QQField(), B, y; backend=:nemo, cache=true)
        @test B * xn == y
        @test haskey(FL._NEMO_FULLCOLUMN_FACTOR_CACHE_QQ, B)
        xn2 = FL.solve_fullcolumn(CM.QQField(), B, y; backend=:nemo, cache=true)
        @test xn2 == xn
    end

    @testset "Nemo conversion counters" begin
        FL.reset_conversion_counters!()
        A = CM.BackendMatrix(QQ[1 2; 3 4]; backend=:nemo)
        r1 = FL.rank(CM.QQField(), A; backend=:nemo)
        @test r1 == 2
        c1 = FL.conversion_counters()
        @test c1.qq_to_nemo >= 1

        r2 = FL.rank(CM.QQField(), A; backend=:nemo)
        @test r2 == r1
        c2 = FL.conversion_counters()
        @test c2.qq_to_nemo_cache_hits >= c1.qq_to_nemo_cache_hits
    end

    @testset "QQ vs F2 cache parity (basic behavior)" begin
        @test isdefined(FL, :clear_f2_fullcolumn_cache!)
        FL.clear_fullcolumn_cache!()
        FL.clear_f2_fullcolumn_cache!()
        @test isempty(FL._FULLCOLUMN_FACTOR_CACHE)
        @test isempty(FL._F2_FULLCOLUMN_FACTOR_CACHE)
    end

    @testset "QQ rank_dim matches exact rank on small matrices" begin
        A = QQ[1 2 3;
               2 4 6;
               1 0 1]
        @test FL.rankQQ(A) == 2
        @test FL.rankQQ_dim(A; backend=:auto) == 2
        @test FL.rankQQ_dim(A; backend=:modular) == 2
        @test FL.rank_dim(CM.QQField(), A; backend=:auto) == 2
    end

    @testset "QQ modular nullspace + solve" begin
        A = QQ[1 2 3 4;
               2 4 6 8;
               1 0 1 1]
        N = FL.nullspace(CM.QQField(), A; backend=:modular)
        @test size(N, 1) == size(A, 2)
        @test A * N == zeros(QQ, size(A, 1), size(N, 2))

        B = QQ[1 0;
               0 1;
               1 1;
               2 3]
        Xtrue = QQ[2 1;
                   3 4]
        Y = B * Xtrue
        X = FL.solve_fullcolumn(CM.QQField(), B, Y; backend=:modular)
        @test B * X == Y
    end

    @testset "QQ sparse nullspace + rank_restricted" begin
        A = sparse([1, 1, 2, 3],
                   [1, 3, 2, 4],
                   QQ[1, 2, -1, 3],
                   3, 4)

        Ns = FL.nullspaceQQ(A)
        Nd = FL.nullspaceQQ(Matrix(A))
        @test size(Ns, 1) == 4
        @test size(Ns, 2) == size(Nd, 2)
        @test A * Ns == zeros(QQ, size(A, 1), size(Ns, 2))
        @test FL.rankQQ(Ns) == size(Ns, 2)

        rng = MersenneTwister(123456)
        m, n = 30, 40
        nnz_target = 180
        I = rand(rng, 1:m, nnz_target)
        J = rand(rng, 1:n, nnz_target)
        V = [QQ(rand(rng, -3:3)) for _ in 1:nnz_target]
        A2 = sparse(I, J, V, m, n)
        dropzeros!(A2)

        for _ in 1:25
            rows = sort!(unique(rand(rng, 1:m, rand(rng, 1:15))))
            cols = sort!(unique(rand(rng, 1:n, rand(rng, 1:18))))
            r1 = FL.rankQQ_restricted(A2, rows, cols)
            r2 = FL.rankQQ(A2[rows, cols])
            @test r1 == r2
        end

        @test FL.rankQQ_restricted(A2, Int[], collect(1:n)) == 0
        @test FL.rankQQ_restricted(A2, collect(1:m), Int[]) == 0
        @test FL.rankQQ_restricted(A2, 1:m, 1:n) == FL.rankQQ(A2)
    end

    @testset "QQ rank_restricted dense" begin
        A = QQ[1 2 3 4;
               2 4 6 8;
               0 1 1 0;
               1 0 1 1]
        rows = [1, 3, 4]
        cols = [2, 3, 4]
        r1 = FL.rankQQ_restricted(sparse(A), rows, cols)
        r2 = FL.rankQQ(A[rows, cols])
        @test r1 == r2
    end

    @testset "QQ rref pivots on rectangular matrices" begin
        A = QQ[1 2 3 4;
               0 1 1 0]
        R, pivs = FL.rrefQQ(A)
        @test pivs == (1, 2)
        @test R == QQ[1 0 1 4;
                      0 1 1 0]

        B = QQ[1 0;
               0 1;
               1 1;
               0 0]
        Rb, pivs_b = FL.rrefQQ(B)
        @test pivs_b == (1, 2)
        @test Rb == QQ[1 0;
                       0 1;
                       0 0;
                       0 0]
    end

    @testset "QQ nullspace edge cases" begin
        A = Matrix{QQ}(I, 3, 3)
        N = FL.nullspaceQQ(A)
        @test size(N, 2) == 0

        Z = zeros(QQ, 2, 4)
        Nz = FL.nullspaceQQ(Z)
        @test size(Nz) == (4, 4)
        @test Z * Nz == zeros(QQ, 2, 4)
        @test FL.rankQQ(Nz) == 4
    end
    end

    @testset "Fp (p>3) engine parity" begin
        F5 = CM.Fp(5)
        F5Elem = CM.FpElem{5}
        fpmat(A) = F5Elem.(A .% 5)

        @testset "rank + rank_dim (dense)" begin
            Aint = [
                1 2 3 4;
                2 4 1 0;
                3 1 2 3;
                4 0 3 1
            ]
            A = fpmat(Aint)
            r1 = FL.rank(F5, A)
            r2 = FL.rank_dim(F5, A)
            @test r1 == r2
            @test 0 <= r1 <= min(size(A)...)
        end

        @testset "nullspace + rref" begin
            Aint = [
                1 2 3 4 0;
                0 1 1 2 3;
                2 1 4 0 1
            ]
            A = fpmat(Aint)
            N = FL.nullspace(F5, A)
            @test A * N == zeros(F5Elem, size(A, 1), size(N, 2))

            R, pivs = FL.rref(F5, A; pivots=true)
            @test length(pivs) == FL.rank(F5, A)
            @test all(p -> 1 <= p <= size(A, 2), pivs)
            @test size(R) == size(A)
        end

        @testset "solve_fullcolumn (dense + sparse)" begin
            n = 4
            m = 7
            B = vcat(fpmat(Matrix{Int}(I, n, n)), fpmat(rand(0:4, m - n, n)))
            X = fpmat(rand(0:4, n, 2))
            Y = B * X
            Xhat = FL.solve_fullcolumn(F5, B, Y)
            @test Xhat == X

            Bs = sparse(B)
            Y2 = Bs * X
            Xhat2 = FL.solve_fullcolumn(F5, Bs, Y2)
            @test Xhat2 == X
        end

        @testset "rank_restricted + colspace sparse" begin
            Aint = [
                1 2 0 1 2;
                2 1 1 0 1;
                1 1 2 2 0;
                0 2 1 1 2
            ]
            A = sparse(fpmat(Aint))
            rows = [1, 3, 4]
            cols = [2, 3, 5]
            r1 = FL.rank_restricted(F5, A, rows, cols)
            r2 = FL.rank(F5, Matrix(A)[rows, cols])
            @test r1 == r2

            C = FL.colspace(F5, A)
            @test size(C, 1) == size(A, 1)
            @test FL.rank(F5, C) == FL.rank(F5, A)
        end

        @testset "edge cases" begin
            A0 = fpmat(zeros(Int, 0, 5))
            @test FL.rank(F5, A0) == 0
            N0 = FL.nullspace(F5, A0)
            @test size(N0) == (5, 5)

            A1 = fpmat(zeros(Int, 4, 0))
            @test FL.rank(F5, A1) == 0
            N1 = FL.nullspace(F5, A1)
            @test size(N1) == (0, 0)
        end

        @testset "randomized properties + sparse nullspace" begin
            rng = MersenneTwister(20240204)
            for _ in 1:20
                m = rand(rng, 2:8)
                n = rand(rng, 2:8)
                A = fpmat(rand(rng, 0:4, m, n))
                r = FL.rank(F5, A)
                N = FL.nullspace(F5, A)
                @test size(N, 1) == n
                @test size(N, 2) == n - r
                @test A * N == zeros(F5Elem, m, size(N, 2))
            end

            for _ in 1:10
                m = rand(rng, 3:8)
                n = rand(rng, 3:8)
                nnz_target = rand(rng, 3:(m * n))
                I = rand(rng, 1:m, nnz_target)
                J = rand(rng, 1:n, nnz_target)
                V = [F5Elem(rand(rng, 0:4)) for _ in 1:nnz_target]
                A = sparse(I, J, V, m, n)
                dropzeros!(A)

                Ns = FL.nullspace(F5, A)
                Nd = FL.nullspace(F5, Matrix(A))
                @test size(Ns, 1) == n
                @test size(Ns, 2) == size(Nd, 2)
                @test A * Ns == zeros(F5Elem, m, size(Ns, 2))
                @test FL.rank(F5, Ns) == size(Ns, 2)
            end
        end

        @testset "Nemo backend parity (p=5)" begin
            A = fpmat([1 2 3; 2 4 1; 3 1 2; 4 0 3])
            r = FL.rank(F5, A)
            rn = FL.rank(F5, A; backend=:nemo)
            @test r == rn
            Nn = FL.nullspace(F5, A; backend=:nemo)
            @test A * Nn == zeros(F5Elem, size(A, 1), size(Nn, 2))

            rd = FL.rank_dim(F5, A; backend=:nemo)
            @test rd == r

            B = fpmat([1 0;
                       0 1;
                       1 1;
                       2 3])
            Xtrue = fpmat([2 1;
                           3 4])
            Y = B * Xtrue
            Xn = FL.solve_fullcolumn(F5, B, Y; backend=:nemo, cache=true)
            @test Xn == Xtrue
            @test haskey(FL._NEMO_FULLCOLUMN_FACTOR_CACHE_FP, B)
        end
    end

    @testset "Real engine parity" begin
        F = CM.RealField(Float64; rtol=1e-10, atol=1e-12)
        A = [1.0 2.0 3.0;
             2.0 4.0 6.0;
             0.0 1.0 1.0]
        r = FL.rank(F, A)
        @test r == 2
        @test FL.rank_dim(F, A) == 2

        N = FL.nullspace(F, A)
        @test size(N, 1) == size(A, 2)
        @test norm(A * N) <= 1e-8

        B = [1.0 0.0;
             0.0 1.0;
             1.0 1.0]
        x_true = [1.0, 2.0]
        y = B * x_true
        x = FL.solve_fullcolumn(F, B, y)
        @test norm(B * x - y) <= 1e-10

        R, pivs = FL.rref(F, A; pivots=true)
        @test length(pivs) == r
        @test size(R) == size(A)

        C = FL.colspace(F, A)
        @test size(C, 1) == size(A, 1)
        @test FL.rank(F, C) == r
    end

    @testset "Tiny <=4x4 fast-path parity" begin
        for field in FIELDS_FULL
            K = CM.coeff_type(field)
            tol = field isa CM.RealField ? (field.atol + 10 * field.rtol) : 0.0

            cmat(A::AbstractMatrix{<:Integer}) = begin
                m, n = size(A)
                M = Matrix{K}(undef, m, n)
                @inbounds for i in 1:m, j in 1:n
                    M[i, j] = CM.coerce(field, A[i, j])
                end
                M
            end

            A = cmat([
                1 2 3;
                2 4 6;
                0 1 1
            ])
            r = FL.rank(field, A)
            R, pivs = FL.rref(field, A; pivots=true)
            @test r == length(pivs)
            @test size(R) == size(A)

            B = cmat([
                1 2;
                0 1;
                1 1
            ])
            Xtrue = cmat([
                1 0;
                2 1
            ])
            Y = B * Xtrue
            Xhat = FL.solve_fullcolumn(field, B, Y)
            if field isa CM.RealField
                @test norm(B * Xhat - Y) <= max(tol, 1e-10)
            else
                @test Xhat == Xtrue
            end

            yv = Y[:, 1]
            xv = FL.solve_fullcolumn(field, B, yv)
            if field isa CM.RealField
                @test norm(B * xv - yv) <= max(tol, 1e-10)
            else
                @test xv == Xtrue[:, 1]
            end

            L = cmat([1 2 0; 0 1 1])
            Rm = cmat([1 0; 2 1; 0 1])
            Mtiny = FL._matmul(L, Rm)
            if field isa CM.RealField
                @test norm(Mtiny - (L * Rm)) <= max(tol, 1e-10)
            else
                @test Mtiny == L * Rm
            end

            As = sparse(A)
            rows = [1, 3]
            cols = [1, 2, 3]
            rr = FL.rank_restricted(field, As, rows, cols)
            @test rr == FL.rank(field, Matrix(A)[rows, cols])
        end
    end
end
