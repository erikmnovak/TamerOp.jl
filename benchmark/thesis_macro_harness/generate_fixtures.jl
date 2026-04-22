#!/usr/bin/env julia

using TOML

include(joinpath(@__DIR__, "common.jl"))

const _DEFAULT_MANIFEST = joinpath(@__DIR__, "manifest.toml")

function main(args::Vector{String})
    manifest_path = String(_arg(args, "--manifest", _DEFAULT_MANIFEST))
    force = lowercase(_arg(args, "--force", "false")) == "true"
    source_case_ids = _parse_symbol_list(_arg(args, "--source_case_ids", "all"))
    source_kinds = _parse_symbol_list(_arg(args, "--source_kinds", "all"))
    limit_raw = strip(_arg(args, "--limit", ""))
    limit = isempty(limit_raw) ? nothing : parse(Int, limit_raw)

    manifest = TOML.parsefile(manifest_path)
    manifest_dir = dirname(manifest_path)
    cases = filter_source_cases(manifest["source_cases"];
                                source_case_ids=source_case_ids === nothing ? nothing : String.(source_case_ids),
                                source_kinds=source_kinds,
                                limit=limit)

    generated = 0
    skipped = 0
    for source_case in cases
        fixture_path = joinpath(manifest_dir, String(source_case["fixture_relpath"]))
        if isfile(fixture_path) && !force
            skipped += 1
            continue
        end
        save_source_fixture(fixture_path, source_case)
        generated += 1
    end

    println("manifest: $(manifest_path)")
    println("selected source cases: $(length(cases))")
    println("generated: $(generated)")
    println("skipped_existing: $(skipped)")
end

main(ARGS)
