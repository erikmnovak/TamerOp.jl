#!/usr/bin/env julia

using Dates
using Statistics
using TOML

include(joinpath(@__DIR__, "common.jl"))

const _DEFAULT_MANIFEST = joinpath(@__DIR__, "manifest.toml")
const _DEFAULT_OUT = joinpath(@__DIR__, "results_tamer_macro.csv")

const _RESULT_COLUMNS = [
    "tool",
    "job_id",
    "source_case_id",
    "source_template",
    "source_kind",
    "source_variant",
    "size_group",
    "family",
    "family_case",
    "invariant_kind",
    "degree_label",
    "degree",
    "nparams",
    "requested_size",
    "size_tier",
    "status",
    "backend_label",
    "exact_supported",
    "reps",
    "warm_reps_completed",
    "cold_mode",
    "source_size",
    "ambient_dim",
    "source_aux",
    "source_shape",
    "encoding_vertices",
    "encoding_nparams",
    "encoding_axis_lengths",
    "cold_ms",
    "cold_encode_ms",
    "cold_invariant_ms",
    "cold_alloc_bytes",
    "cold_gc_ms",
    "warm_median_ms",
    "warm_mean_ms",
    "warm_min_ms",
    "warm_encode_median_ms",
    "warm_invariant_median_ms",
    "warm_alloc_median_bytes",
    "output_kind",
    "output_term_count",
    "output_abs_mass",
    "output_shape",
    "output_digest",
    "output_signature",
    "notes",
    "error_stage",
    "error_type",
    "error_message",
    "started_utc",
    "completed_utc",
]

@inline function _csv_escape(x)
    x === nothing && return ""
    s = x isa AbstractString ? x : string(x)
    if occursin(',', s) || occursin('"', s) || occursin('\n', s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function _write_csv_header(io)
    println(io, join(_RESULT_COLUMNS, ","))
end

function _write_csv_row(io, row::AbstractDict{String,<:Any})
    vals = [_csv_escape(get(row, col, "")) for col in _RESULT_COLUMNS]
    println(io, join(vals, ","))
end

function _timed_metrics(t)
    gc_ms = hasproperty(t, :gctime) ? 1000 * Float64(getproperty(t, :gctime)) : 0.0
    alloc = hasproperty(t, :bytes) ? Int(getproperty(t, :bytes)) : 0
    return alloc, gc_ms
end

function _run_once(data, source_case, job)
    t0 = time_ns()
    enc = build_encoding_result(data, source_case, job; cache=:auto)
    t1 = time_ns()
    result = run_invariant_on_encoding(enc, Symbol(job["invariant_kind"]))
    t2 = time_ns()
    return (
        enc=enc,
        result=result,
        encode_ms=(t1 - t0) / 1.0e6,
        invariant_ms=(t2 - t1) / 1.0e6,
        total_ms=(t2 - t0) / 1.0e6,
    )
end

function _timed_once(data, source_case, job)
    t = @timed _run_once(data, source_case, job)
    alloc, gc_ms = _timed_metrics(t)
    value = hasproperty(t, :value) ? getproperty(t, :value) : t[1]
    return (
        value=value,
        total_ms=Float64(value.total_ms),
        encode_ms=Float64(value.encode_ms),
        invariant_ms=Float64(value.invariant_ms),
        alloc_bytes=alloc,
        gc_ms=gc_ms,
    )
end

function _base_row(source_case, job)
    return Dict{String,Any}(
        "tool" => "tamer_op",
        "job_id" => String(job["id"]),
        "source_case_id" => String(job["source_case_id"]),
        "source_template" => String(source_case["source_template"]),
        "source_kind" => String(source_case["source_kind"]),
        "source_variant" => String(source_case["source_variant"]),
        "size_group" => String(source_case["size_group"]),
        "family" => String(job["family"]),
        "family_case" => String(job["family_case"]),
        "invariant_kind" => String(job["invariant_kind"]),
        "degree_label" => String(job["degree_label"]),
        "degree" => get(job, "degree", nothing),
        "nparams" => Int(job["nparams"]),
        "requested_size" => Int(job["requested_size"]),
        "size_tier" => String(job["size_tier"]),
        "status" => "pending",
        "backend_label" => "",
        "exact_supported" => false,
        "reps" => 0,
        "warm_reps_completed" => 0,
        "cold_mode" => "warm_uncached",
        "source_size" => 0,
        "ambient_dim" => 0,
        "source_aux" => 0,
        "source_shape" => "",
        "encoding_vertices" => 0,
        "encoding_nparams" => 0,
        "encoding_axis_lengths" => "",
        "cold_ms" => "",
        "cold_encode_ms" => "",
        "cold_invariant_ms" => "",
        "cold_alloc_bytes" => "",
        "cold_gc_ms" => "",
        "warm_median_ms" => "",
        "warm_mean_ms" => "",
        "warm_min_ms" => "",
        "warm_encode_median_ms" => "",
        "warm_invariant_median_ms" => "",
        "warm_alloc_median_bytes" => "",
        "output_kind" => "",
        "output_term_count" => "",
        "output_abs_mass" => "",
        "output_shape" => "",
        "output_digest" => "",
        "output_signature" => "",
        "notes" => "",
        "error_stage" => "",
        "error_type" => "",
        "error_message" => "",
        "started_utc" => string(now(UTC)),
        "completed_utc" => "",
    )
end

function _success_row!(row::Dict{String,Any}, data, source_case, job, cold, warms, reps::Int)
    enc = cold.value.enc
    invariant_kind = Symbol(job["invariant_kind"])
    degree_label = Symbol(job["degree_label"])
    src = source_complexity_summary(data)
    encs = encoding_complexity_summary(enc)
    out = summarize_output(cold.value.result, invariant_kind)
    exact = _supports_exact_path(enc, invariant_kind)

    row["status"] = "ok"
    row["backend_label"] = backend_label(enc, invariant_kind, degree_label)
    row["exact_supported"] = exact
    row["reps"] = reps
    row["warm_reps_completed"] = length(warms)
    row["source_size"] = src.source_size
    row["ambient_dim"] = src.ambient_dim
    row["source_aux"] = src.source_aux
    row["source_shape"] = src.source_shape
    row["encoding_vertices"] = encs.encoding_vertices
    row["encoding_nparams"] = encs.encoding_nparams
    row["encoding_axis_lengths"] = encs.encoding_axis_lengths
    row["cold_ms"] = round(cold.total_ms; digits=6)
    row["cold_encode_ms"] = round(cold.encode_ms; digits=6)
    row["cold_invariant_ms"] = round(cold.invariant_ms; digits=6)
    row["cold_alloc_bytes"] = cold.alloc_bytes
    row["cold_gc_ms"] = round(cold.gc_ms; digits=6)
    row["warm_median_ms"] = round(median(map(x -> x.total_ms, warms)); digits=6)
    row["warm_mean_ms"] = round(mean(map(x -> x.total_ms, warms)); digits=6)
    row["warm_min_ms"] = round(minimum(map(x -> x.total_ms, warms)); digits=6)
    row["warm_encode_median_ms"] = round(median(map(x -> x.encode_ms, warms)); digits=6)
    row["warm_invariant_median_ms"] = round(median(map(x -> x.invariant_ms, warms)); digits=6)
    row["warm_alloc_median_bytes"] = Int(round(median(map(x -> x.alloc_bytes, warms))))
    row["output_kind"] = out.output_kind
    row["output_term_count"] = out.output_term_count
    row["output_abs_mass"] = round(out.output_abs_mass; digits=6)
    row["output_shape"] = out.output_shape
    row["output_digest"] = out.output_digest
    row["output_signature"] = out.output_signature
    row["notes"] = benchmark_notes(enc, invariant_kind, degree_label)
    row["completed_utc"] = string(now(UTC))
    return row
end

function _error_row!(row::Dict{String,Any}, stage::AbstractString, err)
    row["status"] = "error"
    row["error_stage"] = stage
    row["error_type"] = string(typeof(err))
    row["error_message"] = sprint(showerror, err)
    row["completed_utc"] = string(now(UTC))
    return row
end

function _maybe_trim!(enabled::Bool)
    enabled && _memory_relief!()
    return nothing
end

function _load_data_cache!(cache::Dict{String,Any}, manifest_dir::String, source_case)
    sid = String(source_case["id"])
    if haskey(cache, sid)
        return cache[sid]
    end
    fixture_path = joinpath(manifest_dir, String(source_case["fixture_relpath"]))
    data = load_source_fixture(fixture_path, source_case)
    cache[sid] = data
    return data
end

function _fixture_path(manifest_dir::String, source_case)
    return joinpath(manifest_dir, String(source_case["fixture_relpath"]))
end

function _check_required_fixtures!(manifest_dir::String,
                                   jobs,
                                   source_cases_by_id::AbstractDict{String,<:Any})
    missing = String[]
    seen = Set{String}()
    for job in jobs
        sid = String(job["source_case_id"])
        sid in seen && continue
        push!(seen, sid)
        source_case = source_cases_by_id[sid]
        fixture_path = _fixture_path(manifest_dir, source_case)
        isfile(fixture_path) || push!(missing, fixture_path)
    end
    isempty(missing) && return nothing
    preview = join(missing[1:min(end, 8)], ", ")
    suffix = length(missing) > 8 ? " ..." : ""
    error("Missing required fixture files ($(length(missing))): $(preview)$(suffix). Run benchmark/thesis_macro_harness/generate_fixtures.jl first.")
end

function _write_progress(progress_path::String; status::AbstractString, job_id="", completed::Int=0, total::Int=0)
    open(progress_path, "w") do io
        println(io, "status=$(status)")
        println(io, "job_id=$(job_id)")
        println(io, "completed=$(completed)")
        println(io, "total=$(total)")
        println(io, "updated_utc=$(now(UTC))")
    end
    return nothing
end

function main(args::Vector{String})
    manifest_path = String(_arg(args, "--manifest", _DEFAULT_MANIFEST))
    out_path = String(_arg(args, "--out", _DEFAULT_OUT))
    profile = Symbol(_arg(args, "--profile", "probe"))
    fail_fast = lowercase(_arg(args, "--fail_fast", "false")) == "true"
    append = lowercase(_arg(args, "--append", "false")) == "true"
    job_ids = _parse_symbol_list(_arg(args, "--job_ids", "all"))
    source_case_ids = _parse_symbol_list(_arg(args, "--source_case_ids", "all"))
    families = _parse_symbol_list(_arg(args, "--families", "all"))
    invariants = _parse_symbol_list(_arg(args, "--invariants", "all"))
    degrees = _parse_symbol_list(_arg(args, "--degrees", "all"))
    source_kinds = _parse_symbol_list(_arg(args, "--source_kinds", "all"))
    limit_raw = strip(_arg(args, "--limit", ""))
    limit = isempty(limit_raw) ? nothing : parse(Int, limit_raw)

    manifest = TOML.parsefile(manifest_path)
    manifest_dir = dirname(manifest_path)
    source_cases_by_id = source_case_lookup(manifest)
    jobs = filter_jobs(manifest["jobs"], source_cases_by_id;
                       job_ids=job_ids === nothing ? nothing : String.(job_ids),
                       source_case_ids=source_case_ids === nothing ? nothing : String.(source_case_ids),
                       families=families,
                       invariants=invariants,
                       degrees=degrees,
                       source_kinds=source_kinds,
                       limit=limit)
    _check_required_fixtures!(manifest_dir, jobs, source_cases_by_id)
    defaults = _profile_defaults(profile)
    reps = Int(defaults.reps)

    mkpath(dirname(out_path))
    progress_path = out_path * ".progress"
    data_cache = Dict{String,Any}()
    _write_progress(progress_path; status="starting", completed=0, total=length(jobs))
    open(out_path, append ? "a" : "w") do io
        if !append || filesize(out_path) == 0
            _write_csv_header(io)
            flush(io)
        end
        for (job_idx, job) in enumerate(jobs)
            source_case = source_cases_by_id[String(job["source_case_id"])]
            row = _base_row(source_case, job)
            job_id = String(job["id"])
            println(stdout, "[$job_idx/$(length(jobs))] start $(job_id)")
            flush(stdout)
            _write_progress(progress_path; status="running", job_id=job_id, completed=job_idx - 1, total=length(jobs))
            try
                data = _load_data_cache!(data_cache, manifest_dir, source_case)
                _maybe_trim!(defaults.trim_between_cases)

                try
                    _run_once(data, source_case, job)
                catch err
                    _error_row!(row, "warmup", err)
                    _write_csv_row(io, row)
                    fail_fast && rethrow(err)
                    continue
                end

                _maybe_trim!(defaults.trim_between_reps)
                cold = _timed_once(data, source_case, job)

                warms = NamedTuple[]
                for _ in 1:reps
                    _maybe_trim!(defaults.trim_between_reps)
                    push!(warms, _timed_once(data, source_case, job))
                end
                _success_row!(row, data, source_case, job, cold, warms, reps)
            catch err
                _error_row!(row, isempty(String(get(row, "error_stage", ""))) ? "run" : String(row["error_stage"]), err)
                fail_fast && rethrow(err)
            end
            _write_csv_row(io, row)
            flush(io)
            status_str = String(row["status"])
            println(stdout, "[$job_idx/$(length(jobs))] done $(job_id) status=$(status_str)")
            flush(stdout)
            _write_progress(progress_path; status="running", job_id=job_id, completed=job_idx, total=length(jobs))
        end
    end
    _write_progress(progress_path; status="completed", completed=length(jobs), total=length(jobs))

    println("manifest: $(manifest_path)")
    println("output: $(out_path)")
    println("jobs_run: $(length(jobs))")
    println("profile: $(profile)")
end

main(ARGS)
