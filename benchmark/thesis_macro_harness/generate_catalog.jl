#!/usr/bin/env julia

using TOML

include(joinpath(@__DIR__, "common.jl"))

const _DEFAULT_OUT = joinpath(@__DIR__, "manifest.toml")

function main(args::Vector{String})
    profile = Symbol(_arg(args, "--profile", "thesis"))
    out_path = String(_arg(args, "--out", _DEFAULT_OUT))
    fixtures_dir = String(_arg(args, "--fixtures_dir", "fixtures"))

    manifest = build_catalog(profile; fixtures_dir=fixtures_dir)
    mkpath(dirname(out_path))
    open(out_path, "w") do io
        TOML.print(io, manifest)
    end

    profile_name = String(manifest["profile"])
    n_source_cases = length(manifest["source_cases"])
    n_jobs = length(manifest["jobs"])
    println("wrote manifest: $(out_path)")
    println("  profile: $(profile_name)")
    println("  source cases: $(n_source_cases)")
    println("  jobs: $(n_jobs)")
end

main(ARGS)
