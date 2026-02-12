#!/usr/bin/env julia
#
# finitefringe_microbench.jl
#
# Purpose
# - Benchmark core FiniteFringe kernels where recent optimizations were applied:
#   1) Uptight region partitioning (`_uptight_regions`)
#   2) Model-independent Hom dimension assembly (`hom_dimension`)
#
# What this compares
# - For each kernel, this script runs:
#   - a naive/baseline implementation (local copy of the previous algorithmic shape),
#   - the current optimized library implementation.
#
# Bench methodology
# - Deterministic synthetic fixtures.
# - Warmup + repeated timed runs.
# - Median wall-time and median allocation reported.
# - Functional parity checks (old vs new) before timing.
#
# Usage
#   julia --project=. benchmark/finitefringe_microbench.jl
#   julia --project=. benchmark/finitefringe_microbench.jl --reps=9 --n=180 --y=96 --nu=24 --nd=24
#

using Random
using SparseArrays

try
    using PosetModules
catch
    include(joinpath(@__DIR__, "..", "src", "PosetModules.jl"))
    using .PosetModules
end

const PM = PosetModules.Advanced
const FF = PM.FiniteFringe
const EN = PM.Encoding
const CM = PM.CoreModules
const FL = PM.FieldLinAlg

function _parse_int_arg(args, key::String, default::Int)
    for a in args
        startswith(a, key * "=") || continue
        return max(1, parse(Int, split(a, "=", limit=2)[2]))
    end
    return default
end

function _bench(name::AbstractString, f::Function; reps::Int=7)
    GC.gc()
    f() # warmup
    GC.gc()
    times_ms = Vector{Float64}(undef, reps)
    bytes = Vector{Int}(undef, reps)
    for i in 1:reps
        m = @timed f()
        times_ms[i] = 1000.0 * m.time
        bytes[i] = m.bytes
    end
    sort!(times_ms)
    sort!(bytes)
    med_ms = times_ms[cld(reps, 2)]
    med_kib = bytes[cld(reps, 2)] / 1024.0
    println(rpad(name, 42), " median_time=", round(med_ms, digits=3),
            " ms  median_alloc=", round(med_kib, digits=1), " KiB")
    return (ms=med_ms, kib=med_kib)
end

function _random_poset(n::Int; p::Float64=0.03, seed::Int=0xF1F1)
    rng = Random.MersenneTwister(seed)
    leq = falses(n, n)
    @inbounds for i in 1:n
        leq[i, i] = true
        for j in (i + 1):n
            leq[i, j] = rand(rng) < p
        end
    end
    # Boolean transitive closure.
    @inbounds for k in 1:n, i in 1:n, j in 1:n
        leq[i, j] = leq[i, j] || (leq[i, k] && leq[k, j])
    end
    return FF.FinitePoset(leq; check=false)
end

function _fixture_uptight(P::FF.AbstractPoset, ny::Int; seed::Int=0x5151)
    rng = Random.MersenneTwister(seed)
    n = PM.nvertices(P)
    Y = Vector{FF.Upset}(undef, ny)
    @inbounds for i in 1:ny
        mask = BitVector(rand(rng, Bool, n))
        Y[i] = FF.upset_closure(P, mask)
    end
    return Y
end

function _uptight_regions_old(Q::FF.AbstractPoset, Y::Vector{FF.Upset})
    sigs = Dict{Tuple{Vararg{Bool}}, Vector{Int}}()
    @inbounds for q in 1:PM.nvertices(Q)
        key = ntuple(i -> Y[i].mask[q], length(Y))
        push!(get!(sigs, key, Int[]), q)
    end
    return collect(values(sigs))
end

function _canon_regions(regs::Vector{Vector{Int}})
    out = [sort(copy(r)) for r in regs]
    sort!(out; by = x -> (length(x), x))
    return out
end

@inline function _is_subset_mask(a::BitVector, b::BitVector)
    @inbounds for i in eachindex(a)
        if a[i] && !b[i]
            return false
        end
    end
    return true
end

function _random_fringe_module(P::FF.AbstractPoset, field::CM.AbstractCoeffField;
                               nu::Int=24, nd::Int=24, density::Float64=0.15, seed::Int=0x6161)
    rng = Random.MersenneTwister(seed)
    K = CM.coeff_type(field)
    n = PM.nvertices(P)

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
        if FF.intersects(U[i], D[j]) && rand(rng) < density
            v = rand(rng, -3:3)
            if v == 0
                v = 1
            end
            phi[j, i] = CM.coerce(field, v)
        end
    end

    return FF.FringeModule{K}(P, U, D, phi; field=field)
end

function _hom_dimension_old(M::FF.FringeModule{K}, N::FF.FringeModule{K}) where {K}
    M.P === N.P || error("Posets must match")
    P = M.P
    adj = FF._cover_undirected_adjacency(P)

    nUM = length(M.U); nDM = length(M.D)
    nUN = length(N.U); nDN = length(N.D)

    Ucomp_id_M    = Vector{Vector{Int}}(undef, nUM)
    Ucomp_masks_M = Vector{Vector{BitVector}}(undef, nUM)
    Ucomp_n_M     = Vector{Int}(undef, nUM)
    for i in 1:nUM
        comp_id, ncomp, comp_masks, _ = FF._component_data(adj, M.U[i].mask)
        Ucomp_id_M[i] = comp_id
        Ucomp_masks_M[i] = comp_masks
        Ucomp_n_M[i] = ncomp
    end

    Dcomp_id_N    = Vector{Vector{Int}}(undef, nDN)
    Dcomp_masks_N = Vector{Vector{BitVector}}(undef, nDN)
    Dcomp_n_N     = Vector{Int}(undef, nDN)
    for t in 1:nDN
        comp_id, ncomp, comp_masks, _ = FF._component_data(adj, N.D[t].mask)
        Dcomp_id_N[t] = comp_id
        Dcomp_masks_N[t] = comp_masks
        Dcomp_n_N[t] = ncomp
    end

    w_index = Dict{Tuple{Int,Int},Int}()
    w_rows_u = Vector{Vector{Vector{Int}}}()
    w_rows_d = Vector{Vector{Vector{Int}}}()
    W_dim = 0

    for iM in 1:nUM
        for tN in 1:nDN
            mask_int = M.U[iM].mask .& N.D[tN].mask
            if any(mask_int)
                _, ncomp_int, _, reps_int = FF._component_data(adj, mask_int)
                base = W_dim
                W_dim += ncomp_int

                rows_by_u = [Int[] for _ in 1:Ucomp_n_M[iM]]
                rows_by_d = [Int[] for _ in 1:Dcomp_n_N[tN]]
                for c in 1:ncomp_int
                    row = base + c
                    v = reps_int[c]
                    cu = Ucomp_id_M[iM][v]
                    cd = Dcomp_id_N[tN][v]
                    push!(rows_by_u[cu], row)
                    push!(rows_by_d[cd], row)
                end

                push!(w_rows_u, rows_by_u)
                push!(w_rows_d, rows_by_d)
                w_index[(iM, tN)] = length(w_rows_u)
            end
        end
    end

    V1 = Tuple{Int,Int,Int}[]
    for iM in 1:nUM
        for jN in 1:nUN
            for cU in 1:Ucomp_n_M[iM]
                if _is_subset_mask(Ucomp_masks_M[iM][cU], N.U[jN].mask)
                    push!(V1, (iM, jN, cU))
                end
            end
        end
    end
    V1_dim = length(V1)

    V2 = Tuple{Int,Int,Int}[]
    for sM in 1:nDM
        for tN in 1:nDN
            for cD in 1:Dcomp_n_N[tN]
                if _is_subset_mask(Dcomp_masks_N[tN][cD], M.D[sM].mask)
                    push!(V2, (sM, tN, cD))
                end
            end
        end
    end
    V2_dim = length(V2)

    T = zeros(K, W_dim, V1_dim)
    for (col, (iM, jN, cU)) in enumerate(V1)
        for tN in 1:nDN
            val = N.phi[tN, jN]
            if val != zero(K)
                pid = get(w_index, (iM, tN), 0)
                if pid != 0
                    rows = w_rows_u[pid][cU]
                    for r in rows
                        T[r, col] += val
                    end
                end
            end
        end
    end

    S = zeros(K, W_dim, V2_dim)
    for (col, (sM, tN, cD)) in enumerate(V2)
        for iM in 1:nUM
            val = M.phi[sM, iM]
            if val != zero(K)
                pid = get(w_index, (iM, tN), 0)
                if pid != 0
                    rows = w_rows_d[pid][cD]
                    for r in rows
                        S[r, col] += val
                    end
                end
            end
        end
    end

    big = hcat(T, -S)
    rT = FL.rank(M.field, T)
    rS = FL.rank(M.field, S)
    rBig = FL.rank(M.field, big)
    dimKer_big = (V1_dim + V2_dim) - rBig
    dimKer_T = V1_dim - rT
    dimKer_S = V2_dim - rS
    return dimKer_big - (dimKer_T + dimKer_S)
end

function main(args=ARGS)
    reps = _parse_int_arg(args, "--reps", 7)
    n = _parse_int_arg(args, "--n", 180)
    ny = _parse_int_arg(args, "--y", 96)
    nu = _parse_int_arg(args, "--nu", 24)
    nd = _parse_int_arg(args, "--nd", 24)

    println("FiniteFringe microbench")
    println("reps=$(reps), poset_n=$(n), y=$(ny), nu=$(nu), nd=$(nd)\n")

    P = _random_poset(n; p=0.03, seed=Int(0x7010))
    Y = _fixture_uptight(P, ny; seed=Int(0x7011))

    # Parity checks for region partition.
    r_old = _canon_regions(_uptight_regions_old(P, Y))
    r_new = _canon_regions(EN._uptight_regions(P, Y))
    r_old == r_new || error("_uptight_regions parity failed")

    println("== Uptight region partition ==")
    b_old_up = _bench("uptight old (tuple keys)", () -> _uptight_regions_old(P, Y); reps=reps)
    b_new_up = _bench("uptight new (bit-packed)", () -> EN._uptight_regions(P, Y); reps=reps)
    println("speedup(new/old): ", round(b_old_up.ms / b_new_up.ms, digits=2), "x")
    println()

    field = CM.QQField()
    M = _random_fringe_module(P, field; nu=nu, nd=nd, density=0.15, seed=Int(0x7021))
    N = _random_fringe_module(P, field; nu=nu, nd=nd, density=0.15, seed=Int(0x7022))

    # Parity checks for hom dimension.
    h_old = _hom_dimension_old(M, N)
    h_new = FF.hom_dimension(M, N)
    h_old == h_new || error("hom_dimension parity failed: old=$(h_old), new=$(h_new)")

    println("== Hom dimension assembly ==")
    b_old_h = _bench("hom_dimension old (dict+mask)", () -> _hom_dimension_old(M, N); reps=reps)
    b_new_h = _bench("hom_dimension new (dense idx)", () -> FF.hom_dimension(M, N); reps=reps)
    println("speedup(new/old): ", round(b_old_h.ms / b_new_h.ms, digits=2), "x")
end

main()
