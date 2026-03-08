#!/usr/bin/env julia

using Random
using JSON3

if isdefined(Main, :PosetModules)
    const PosetModules = getfield(Main, :PosetModules)
else
    try
        using PosetModules
    catch
        include(joinpath(@__DIR__, "..", "src", "PosetModules.jl"))
        using .PosetModules
    end
end

const SER = PosetModules.Serialization
const CM = PosetModules.CoreModules
const OPT = PosetModules.Options
const DT = PosetModules.DataTypes
const EC = PosetModules.EncodingCore
const RES = PosetModules.Results
const FF = PosetModules.FiniteFringe
const FZ = PosetModules.FlangeZn
const ZE = PosetModules.ZnEncoding

function _parse_int_arg(args, key::String, default::Int)
    for a in args
        startswith(a, key * "=") || continue
        return max(1, parse(Int, split(a, "=", limit=2)[2]))
    end
    return default
end

function _parse_bool_arg(args, key::String, default::Bool)
    for a in args
        startswith(a, key * "=") || continue
        v = lowercase(split(a, "=", limit=2)[2])
        if v in ("1", "true", "yes", "y", "on")
            return true
        elseif v in ("0", "false", "no", "n", "off")
            return false
        else
            error("Invalid bool value for $key: $v")
        end
    end
    return default
end

function _bench(name::AbstractString, f::Function; reps::Int=5, warmup::Int=1)
    for _ in 1:warmup
        f()
    end
    GC.gc()
    ts = Vector{Float64}(undef, reps)
    bs = Vector{Int}(undef, reps)
    for i in 1:reps
        m = @timed f()
        ts[i] = 1000.0 * m.time
        bs[i] = m.bytes
    end
    sort!(ts)
    sort!(bs)
    med_ms = ts[cld(reps, 2)]
    med_kib = bs[cld(reps, 2)] / 1024.0
    println(rpad(name, 52), " median=", round(med_ms, digits=3), " ms",
            " alloc=", round(med_kib, digits=1), " KiB")
    return (ms=med_ms, kib=med_kib)
end

function _axis_encoding_flange(field::CM.AbstractCoeffField; ncuts::Int=8, seed::Integer=0x5352454e)
    N = 2
    rng = Random.MersenneTwister(UInt(seed))
    tau_x = FZ.face(N, [false, true])
    tau_y = FZ.face(N, [true, false])
    thresholds = collect(-ncuts:ncuts)
    flats = FZ.IndFlat{N}[]
    injectives = FZ.IndInj{N}[]
    for (i, t) in enumerate(thresholds)
        push!(flats, FZ.IndFlat(tau_x, (t, 0); id=Symbol(:Fx, i)))
        push!(injectives, FZ.IndInj(tau_x, (t + 1, 0); id=Symbol(:Ex, i)))
        push!(flats, FZ.IndFlat(tau_y, (0, t); id=Symbol(:Fy, i)))
        push!(injectives, FZ.IndInj(tau_y, (0, t + 1); id=Symbol(:Ey, i)))
    end
    m = length(injectives)
    K = CM.coeff_type(field)
    phi = Matrix{K}(undef, m, m)
    @inbounds for i in 1:m, j in 1:m
        phi[i, j] = CM.coerce(field, i == j ? 1 : rand(rng, -1:1))
    end
    return FZ.Flange{K}(N, flats, injectives, phi; field=field)
end

function _make_graph(n::Int, rng::AbstractRNG)
    edges = Tuple{Int,Int}[]
    sizehint!(edges, 3n)
    for i in 1:(n - 1)
        push!(edges, (i, i + 1))
    end
    for _ in 1:(2n)
        u = rand(rng, 1:n)
        v = rand(rng, 1:n)
        u == v && continue
        push!(edges, (min(u, v), max(u, v)))
    end
    coords = [randn(rng, 2) for _ in 1:n]
    weights = rand(rng, length(edges))
    return DT.GraphData(n, edges; coords=coords, weights=weights, T=Float64)
end

function _write_legacy_pointcloud(path::AbstractString, pc::DT.PointCloud)
    obj = Dict("kind" => "PointCloud", "points" => [collect(p) for p in pc.points])
    open(path, "w") do io
        JSON3.write(io, obj)
    end
    return path
end

function _write_legacy_graph(path::AbstractString, g::DT.GraphData)
    obj = Dict(
        "kind" => "GraphData",
        "n" => g.n,
        "edges" => [collect(e) for e in g.edges],
        "coords" => g.coords === nothing ? nothing : [collect(c) for c in g.coords],
        "weights" => g.weights === nothing ? nothing : collect(g.weights),
    )
    open(path, "w") do io
        JSON3.write(io, obj)
    end
    return path
end

function _encoding_section(tmpdir::AbstractString; reps::Int)
    println("\n=== Encoding JSON fast paths ===")
    qf = CM.QQField()
    FG = _axis_encoding_flange(qf; ncuts=8)
    opts = OPT.EncodingOptions(backend=:zn, max_regions=500_000, field=qf)
    P, H, pi = ZE.encode_from_flange(FG, opts; poset_kind=:signature)
    path_auto = joinpath(tmpdir, "enc_auto.json")
    path_dense = joinpath(tmpdir, "enc_dense.json")
    path_pretty = joinpath(tmpdir, "enc_pretty.json")
    path_compact = joinpath(tmpdir, "enc_compact.json")

    SER.save_encoding_json(path_auto, P, H, pi; include_leq=:auto, pretty=false)
    SER.save_encoding_json(path_dense, P, H, pi; include_leq=true, pretty=false)
    SER.save_encoding_json(path_pretty, P, H, pi; include_leq=:auto, pretty=true)
    SER.save_encoding_json(path_compact, P, H, pi; include_leq=:auto, pretty=false)

    b_load_v1 = _bench("load schema_v1 (typed) strict",
                             () -> SER.load_encoding_json(path_auto; output=:fringe, validation=:strict);
                             reps=reps)
    b_load_trusted = _bench("load trusted",
                            () -> SER.load_encoding_json(path_auto; output=:fringe, validation=:trusted);
                            reps=reps)
    b_load_pi = _bench("load trusted + pi",
                       () -> SER.load_encoding_json(path_auto; output=:fringe_with_pi, validation=:trusted);
                       reps=reps)
    b_save_auto = _bench("save include_leq=:auto",
                         () -> SER.save_encoding_json(path_auto, P, H, pi; include_leq=:auto, pretty=false);
                         reps=reps)
    b_save_dense = _bench("save include_leq=true",
                          () -> SER.save_encoding_json(path_dense, P, H, pi; include_leq=true, pretty=false);
                          reps=reps)
    b_save_pretty = _bench("save pretty=true",
                           () -> SER.save_encoding_json(path_pretty, P, H, pi; include_leq=:auto, pretty=true);
                           reps=reps)
    b_save_compact = _bench("save pretty=false",
                            () -> SER.save_encoding_json(path_compact, P, H, pi; include_leq=:auto, pretty=false);
                            reps=reps)

    Kqq = CM.coeff_type(qf)
    mphi, kphi = size(H.phi)
    phi_big = Matrix{Kqq}(undef, mphi, kphi)
    @inbounds for i in 1:mphi, j in 1:kphi
        if H.phi[i, j] == 0
            phi_big[i, j] = 0
        else
            num = BigInt(10)^28 + BigInt(97 * i + j)
            den = BigInt(10)^19 + BigInt(31 * j + i + 1)
            phi_big[i, j] = num // den
        end
    end
    H_big = FF.FringeModule{Kqq}(P, H.U, H.D, phi_big; field=qf)
    path_big = joinpath(tmpdir, "enc_qq_big.json")
    SER.save_encoding_json(path_big, P, H_big, pi; include_leq=:auto, pretty=false)
    b_load_big = _bench("load qq_chunks heavy trusted",
                        () -> SER.load_encoding_json(path_big; output=:fringe, validation=:trusted);
                        reps=reps)

    size_auto = filesize(path_auto)
    size_dense = filesize(path_dense)
    size_pretty = filesize(path_pretty)
    size_compact = filesize(path_compact)
    println("file size include_leq=:auto: ", size_auto, " bytes")
    println("file size include_leq=true:  ", size_dense, " bytes",
            "  (x", round(size_dense / max(1, size_auto), digits=2), ")")
    println("file size pretty=true:       ", size_pretty, " bytes")
    println("file size pretty=false:      ", size_compact, " bytes",
            "  (", round(100 * (1 - size_compact / max(1, size_pretty)), digits=1), "% smaller)")
    println("trusted load speedup (trusted vs strict): x",
            round(b_load_v1.ms / max(1e-9, b_load_trusted.ms), digits=2))
    println("pi decode overhead (fringe_with_pi vs fringe): x",
            round(b_load_pi.ms / max(1e-9, b_load_trusted.ms), digits=2))
    println("auto leq save speedup (auto vs dense): x",
            round(b_save_dense.ms / max(1e-9, b_save_auto.ms), digits=2))
    println("compact save speedup (pretty=false vs true): x",
            round(b_save_pretty.ms / max(1e-9, b_save_compact.ms), digits=2))
    println("qq heavy/load baseline ratio (heavy vs regular trusted): x",
            round(b_load_big.ms / max(1e-9, b_load_trusted.ms), digits=2))
end

function _dataset_section(tmpdir::AbstractString; reps::Int)
    println("\n=== Dataset JSON columnar vs legacy ===")
    rng = Random.MersenneTwister(0x51525354)

    pc = DT.PointCloud(randn(rng, 5000, 3))
    graph = _make_graph(5000, rng)

    pc_col = joinpath(tmpdir, "pc_columnar.json")
    pc_legacy = joinpath(tmpdir, "pc_legacy.json")
    g_col = joinpath(tmpdir, "g_columnar.json")
    g_legacy = joinpath(tmpdir, "g_legacy.json")

    SER.save_dataset_json(pc_col, pc; pretty=false)
    _write_legacy_pointcloud(pc_legacy, pc)
    SER.save_dataset_json(g_col, graph; pretty=false)
    _write_legacy_graph(g_legacy, graph)

    b_pc_col = _bench("load PointCloud columnar", () -> SER.load_dataset_json(pc_col); reps=reps)
    b_pc_legacy = _bench("load PointCloud legacy", () -> SER.load_dataset_json(pc_legacy); reps=reps)
    b_g_col = _bench("load GraphData columnar", () -> SER.load_dataset_json(g_col); reps=reps)
    b_g_legacy = _bench("load GraphData legacy", () -> SER.load_dataset_json(g_legacy); reps=reps)

    println("pointcloud file size columnar: ", filesize(pc_col), " bytes")
    println("pointcloud file size legacy:   ", filesize(pc_legacy), " bytes")
    println("graph file size columnar:      ", filesize(g_col), " bytes")
    println("graph file size legacy:        ", filesize(g_legacy), " bytes")
    println("PointCloud load speedup (columnar vs legacy): x",
            round(b_pc_legacy.ms / max(1e-9, b_pc_col.ms), digits=2))
    println("GraphData load speedup (columnar vs legacy): x",
            round(b_g_legacy.ms / max(1e-9, b_g_col.ms), digits=2))
end

function main(args=ARGS)
    reps = _parse_int_arg(args, "--reps", 4)
    run_encoding = _parse_bool_arg(args, "--encoding", true)
    run_dataset = _parse_bool_arg(args, "--dataset", true)
    println("Serialization fast-path microbench (warm cached)")
    println("reps=", reps)
    println("encoding=", run_encoding, ", dataset=", run_dataset)
    tmpdir = mktempdir(prefix="serialization_fastpath_")
    println("tmpdir=", tmpdir)
    try
        run_encoding && _encoding_section(tmpdir; reps=reps)
        run_dataset && _dataset_section(tmpdir; reps=reps)
    finally
        rm(tmpdir; recursive=true, force=true)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
