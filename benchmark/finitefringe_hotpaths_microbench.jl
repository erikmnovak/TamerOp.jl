#!/usr/bin/env julia
# FiniteFringe hot-path microbenchmarks.
# This script benchmarks the production FiniteFringe kernels most affected by
# cache/layout changes: repeated fiber-dimension queries and repeated
# hom-dimension queries on fixed fringe-module pairs.

using Random
using SparseArrays

try
    using PosetModules
catch
    include(joinpath(@__DIR__, "..", "src", "PosetModules.jl"))
    using .PosetModules
end

const FF = PosetModules.FiniteFringe
const CM = PosetModules.CoreModules

function _parse_int_arg(args, key::String, default::Int)
    for arg in args
        startswith(arg, key * "=") || continue
        return parse(Int, split(arg, "=", limit=2)[2])
    end
    return default
end

function _bench(name::AbstractString, f::Function; reps::Int)
    GC.gc()
    f()
    GC.gc()
    times_ms = Vector{Float64}(undef, reps)
    alloc_kib = Vector{Float64}(undef, reps)
    for i in 1:reps
        m = @timed f()
        times_ms[i] = 1000.0 * m.time
        alloc_kib[i] = m.bytes / 1024.0
    end
    sort!(times_ms)
    sort!(alloc_kib)
    mid = cld(reps, 2)
    println(rpad(name, 34),
            " median_time=", round(times_ms[mid], digits=3), " ms",
            " median_alloc=", round(alloc_kib[mid], digits=1), " KiB")
    return (ms=times_ms[mid], kib=alloc_kib[mid])
end

function _random_poset(n::Int; p::Float64=0.035, seed::Int=0xFF01)
    rng = Random.MersenneTwister(seed)
    leq = falses(n, n)
    @inbounds for i in 1:n
        leq[i, i] = true
        for j in (i + 1):n
            leq[i, j] = rand(rng) < p
        end
    end
    @inbounds for k in 1:n, i in 1:n, j in 1:n
        leq[i, j] = leq[i, j] || (leq[i, k] && leq[k, j])
    end
    return FF.FinitePoset(leq; check=false)
end

function _random_fringe_module(P::FF.AbstractPoset, field::CM.AbstractCoeffField;
                               nu::Int, nd::Int, density::Float64, seed::Int)
    rng = Random.MersenneTwister(seed)
    K = CM.coeff_type(field)
    n = FF.nvertices(P)
    U = Vector{FF.Upset}(undef, nu)
    D = Vector{FF.Downset}(undef, nd)
    @inbounds for i in 1:nu
        U[i] = FF.upset_closure(P, BitVector(rand(rng, Bool, n)))
    end
    @inbounds for j in 1:nd
        D[j] = FF.downset_closure(P, BitVector(rand(rng, Bool, n)))
    end
    phi = spzeros(K, nd, nu)
    @inbounds for j in 1:nd, i in 1:nu
        FF.intersects(U[i], D[j]) || continue
        rand(rng) < density || continue
        v = rand(rng, -3:3)
        v == 0 && continue
        phi[j, i] = CM.coerce(field, v)
    end
    return FF.FringeModule{K}(P, U, D, phi; field=field)
end

function _fiber_scan_reference(M::FF.FringeModule, queries::Vector{Int})
    s = 0
    @inbounds for q in queries
        cols = findall(U -> U.mask[q], M.U)
        rows = findall(D -> D.mask[q], M.D)
        isempty(cols) || isempty(rows) && continue
        s += PosetModules.FieldLinAlg.rank_restricted(M.field, M.phi, rows, cols)
    end
    return s
end

function _fiber_current(M::FF.FringeModule, queries::Vector{Int})
    s = 0
    @inbounds for q in queries
        s += FF.fiber_dimension(M, q)
    end
    return s
end

function main(args=ARGS)
    reps = _parse_int_arg(args, "--reps", 5)
    n = _parse_int_arg(args, "--n", 80)
    nu = _parse_int_arg(args, "--nu", 18)
    nd = _parse_int_arg(args, "--nd", 18)
    fiber_queries = _parse_int_arg(args, "--fiber_queries", 800)

    println("FiniteFringe hot-path microbench")
    println("reps=$(reps), n=$(n), nu=$(nu), nd=$(nd), fiber_queries=$(fiber_queries)\n")

    field = CM.QQField()
    P = _random_poset(n; seed=Int(0xFF01))
    Mf = _random_fringe_module(P, field; nu=max(nu, 20), nd=max(nd, 20), density=0.18, seed=Int(0xFF11))
    rng = Random.MersenneTwister(Int(0xFF12))
    queries = rand(rng, 1:FF.nvertices(P), fiber_queries)

    expected_f = _fiber_scan_reference(Mf, queries)
    got_f = _fiber_current(Mf, queries)
    expected_f == got_f || error("fiber parity failed: expected $(expected_f), got $(got_f)")

    println("== fiber_dimension batch ==")
    b_f_scan = _bench("fiber scan baseline", () -> _fiber_scan_reference(Mf, queries); reps=reps)
    b_f_cold = _bench("fiber current fresh", () -> begin
        Mc = _random_fringe_module(P, field; nu=max(nu, 20), nd=max(nd, 20), density=0.18, seed=Int(0xFF11))
        _fiber_current(Mc, queries)
    end; reps=reps)
    _fiber_current(Mf, queries)
    b_f_warm = _bench("fiber current warm", () -> _fiber_current(Mf, queries); reps=reps)
    println("fiber speedup fresh/scan = ", round(b_f_scan.ms / b_f_cold.ms, digits=2), "x")
    println("fiber speedup warm/scan  = ", round(b_f_scan.ms / b_f_warm.ms, digits=2), "x")
    println()

    M = _random_fringe_module(P, field; nu=nu, nd=nd, density=0.16, seed=Int(0xFF21))
    N = _random_fringe_module(P, field; nu=nu, nd=nd, density=0.16, seed=Int(0xFF22))
    h_ref = FF._hom_dimension_with_path(M, N, :sparse_path)
    h_auto = FF.hom_dimension(M, N)
    h_ref == h_auto || error("hom parity failed: sparse=$(h_ref), auto=$(h_auto)")

    println("== hom_dimension repeated pair ==")
    b_h_fresh = _bench("hom auto fresh pair", () -> begin
        Mc = _random_fringe_module(P, field; nu=nu, nd=nd, density=0.16, seed=Int(0xFF21))
        Nc = _random_fringe_module(P, field; nu=nu, nd=nd, density=0.16, seed=Int(0xFF22))
        FF.hom_dimension(Mc, Nc)
    end; reps=reps)
    FF.hom_dimension(M, N)
    b_h_warm = _bench("hom auto warm pair", () -> FF.hom_dimension(M, N); reps=reps)
    b_h_sparse = _bench("hom sparse path warm", () -> FF._hom_dimension_with_path(M, N, :sparse_path); reps=reps)
    println("hom warmup gain = ", round(b_h_fresh.ms / b_h_warm.ms, digits=2), "x")
    println("hom auto/sparse = ", round(b_h_sparse.ms / b_h_warm.ms, digits=2), "x")

    sparse_plan = FF._ensure_sparse_hom_plan!(M, N)
    small = size(sparse_plan.S, 2) <= size(sparse_plan.T, 2) ? sparse_plan.S : sparse_plan.T
    small_cols = size(small, 2)
    union_call = if size(sparse_plan.S, 2) <= size(sparse_plan.T, 2)
        () -> FF._rank_hcat_signed_sparse_workspace_with_prefix_rank!(
            field, sparse_plan.hcat_buf,
            sparse_plan.T, sparse_plan.S, sparse_plan.nnzT, size(sparse_plan.T, 2)
        )
    else
        () -> FF._rank_hcat_signed_sparse_workspace_with_prefix_rank!(
            field, sparse_plan.hcat_buf_rev,
            sparse_plan.S, sparse_plan.T, sparse_plan.nnzS, size(sparse_plan.S, 2)
        )
    end
    println()
    println("== sparse-rank subcomponents ==")
    println("small rank backend = ",
            PosetModules.FieldLinAlg._choose_linalg_backend(field, small; op=:rank, backend=:auto),
            " (cols=", small_cols, ", nnz=", nnz(small), ")")
    _bench("hom sparse small rank", () -> PosetModules.FieldLinAlg.rank(field, small); reps=reps)
    _bench("hom sparse union prefix", union_call; reps=reps)
end

main()
