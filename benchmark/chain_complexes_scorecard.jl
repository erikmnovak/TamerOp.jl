#!/usr/bin/env julia
#
# chain_complexes_scorecard.jl
#
# Purpose
# - Compare two `chain_complexes_microbench.jl` CSV outputs and summarize
#   before/after deltas by probe.
#
# Usage
#   julia --project=. benchmark/chain_complexes_scorecard.jl \
#       --before=benchmark/_tmp_chain_complexes_before.csv \
#       --after=benchmark/_tmp_chain_complexes_after.csv
#
# Output
# - Prints an aggregate scorecard to stdout.
# - Writes a per-probe delta CSV (default: benchmark/_tmp_chain_complexes_scorecard.csv).
#

using Printf

function _parse_int_arg(args, key::String, default::Int)
    for a in args
        startswith(a, key * "=") || continue
        return parse(Int, split(a, "=", limit=2)[2])
    end
    return default
end

function _parse_float_arg(args, key::String, default::Float64)
    for a in args
        startswith(a, key * "=") || continue
        return parse(Float64, split(a, "=", limit=2)[2])
    end
    return default
end

function _parse_string_arg(args, key::String, default::String)
    for a in args
        startswith(a, key * "=") || continue
        return String(strip(split(a, "=", limit=2)[2]))
    end
    return default
end

function _read_microbench_csv(path::AbstractString)
    isfile(path) || error("scorecard: file not found: $path")
    rows = Dict{String,NamedTuple{(:median_ms, :median_kib),Tuple{Float64,Float64}}}()

    open(path, "r") do io
        eof(io) && error("scorecard: empty CSV: $path")
        header = split(strip(readline(io)), ',')
        expected = ["probe", "median_ms", "median_kib"]
        header == expected || error("scorecard: unexpected header in $path: $(join(header, ','))")
        lineno = 1
        for line in eachline(io)
            lineno += 1
            s = strip(line)
            isempty(s) && continue
            parts = split(s, ',')
            length(parts) == 3 || error("scorecard: malformed line $lineno in $path")
            probe = String(parts[1])
            ms = parse(Float64, parts[2])
            kib = parse(Float64, parts[3])
            rows[probe] = (median_ms=ms, median_kib=kib)
        end
    end

    isempty(rows) && error("scorecard: no rows read from $path")
    return rows
end

@inline function _ratio(after::Float64, before::Float64)
    if before == 0.0
        return after == 0.0 ? 1.0 : Inf
    end
    return after / before
end

function _geomean(xs::Vector{Float64})
    good = Float64[]
    for x in xs
        if isfinite(x) && x > 0
            push!(good, log(x))
        end
    end
    isempty(good) && return NaN
    return exp(sum(good) / length(good))
end

function _status(ratio::Float64, noise::Float64)
    ratio > 1.0 + noise && return "regressed"
    ratio < 1.0 - noise && return "improved"
    return "flat"
end

function _write_scorecard_csv(path::AbstractString, rows)
    open(path, "w") do io
        println(io,
                "probe,before_ms,after_ms,time_ratio,delta_pct,before_kib,after_kib,alloc_ratio,alloc_delta_pct,status")
        for r in rows
            println(io,
                    string(r.probe, ",",
                           r.before_ms, ",",
                           r.after_ms, ",",
                           r.time_ratio, ",",
                           r.delta_pct, ",",
                           r.before_kib, ",",
                           r.after_kib, ",",
                           r.alloc_ratio, ",",
                           r.alloc_delta_pct, ",",
                           r.status))
        end
    end
end

function main(; before_path::String,
              after_path::String,
              out::String,
              label_before::String="before",
              label_after::String="after",
              noise::Float64=0.03,
              top::Int=8)
    noise >= 0.0 || error("scorecard: --noise must be >= 0")
    top >= 1 || error("scorecard: --top must be >= 1")

    before = _read_microbench_csv(before_path)
    after = _read_microbench_csv(after_path)

    probes_before = Set(keys(before))
    probes_after = Set(keys(after))
    common = sort!(collect(intersect(probes_before, probes_after)))

    isempty(common) && error("scorecard: no overlapping probes between before/after")

    missing_in_after = sort!(collect(setdiff(probes_before, probes_after)))
    missing_in_before = sort!(collect(setdiff(probes_after, probes_before)))

    rows = NamedTuple[]
    time_ratios = Float64[]
    alloc_ratios = Float64[]

    for probe in common
        b = before[probe]
        a = after[probe]
        tr = _ratio(a.median_ms, b.median_ms)
        ar = _ratio(a.median_kib, b.median_kib)

        push!(time_ratios, tr)
        push!(alloc_ratios, ar)

        push!(rows,
              (probe=probe,
               before_ms=b.median_ms,
               after_ms=a.median_ms,
               time_ratio=tr,
               delta_pct=100.0 * (tr - 1.0),
               before_kib=b.median_kib,
               after_kib=a.median_kib,
               alloc_ratio=ar,
               alloc_delta_pct=100.0 * (ar - 1.0),
               status=_status(tr, noise)))
    end

    regressed = count(r -> r.status == "regressed", rows)
    improved = count(r -> r.status == "improved", rows)
    flat = count(r -> r.status == "flat", rows)

    geomean_time = _geomean(time_ratios)
    geomean_alloc = _geomean(alloc_ratios)

    sort_reg = sort(rows; by=r -> r.time_ratio, rev=true)
    sort_imp = sort(rows; by=r -> r.time_ratio)

    println("ChainComplexes scorecard")
    println("  $label_before: $before_path")
    println("  $label_after : $after_path")
    println("  ratio policy : time_ratio = $label_after / $label_before (>1 means slower)")
    println("  overlap probes: $(length(common))")
    println("  geomean time ratio : ", @sprintf("%.4f", geomean_time),
            " (", @sprintf("%.2f", geomean_time < 1 ? (1 / geomean_time) : geomean_time),
            geomean_time < 1 ? "x faster" : "x slower", ")")
    println("  geomean alloc ratio: ", @sprintf("%.4f", geomean_alloc))
    println("  counts (noise=", noise, "): improved=", improved,
            ", regressed=", regressed, ", flat=", flat)

    if !isempty(missing_in_after)
        println("  missing in $label_after: ", join(missing_in_after, ", "))
    end
    if !isempty(missing_in_before)
        println("  missing in $label_before: ", join(missing_in_before, ", "))
    end

    println("\nTop regressions (time):")
    for r in first(sort_reg, min(top, length(sort_reg)))
        @printf("  %-44s ratio=%8.4f  delta=%8.2f%%\n", r.probe, r.time_ratio, r.delta_pct)
    end

    println("\nTop improvements (time):")
    for r in first(sort_imp, min(top, length(sort_imp)))
        @printf("  %-44s ratio=%8.4f  delta=%8.2f%%\n", r.probe, r.time_ratio, r.delta_pct)
    end

    _write_scorecard_csv(out, rows)
    println("\nWrote ", out)
end

if abspath(PROGRAM_FILE) == @__FILE__
    before_path = _parse_string_arg(ARGS, "--before", joinpath(@__DIR__, "_tmp_chain_complexes_before.csv"))
    after_path = _parse_string_arg(ARGS, "--after", joinpath(@__DIR__, "_tmp_chain_complexes_after.csv"))
    out = _parse_string_arg(ARGS, "--out", joinpath(@__DIR__, "_tmp_chain_complexes_scorecard.csv"))
    label_before = _parse_string_arg(ARGS, "--label_before", "before")
    label_after = _parse_string_arg(ARGS, "--label_after", "after")
    noise = _parse_float_arg(ARGS, "--noise", 0.03)
    top = _parse_int_arg(ARGS, "--top", 8)

    main(; before_path=before_path,
         after_path=after_path,
         out=out,
         label_before=label_before,
         label_after=label_after,
         noise=noise,
         top=top)
end
