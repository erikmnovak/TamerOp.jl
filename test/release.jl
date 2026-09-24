# Full release verification, partitioned into fresh processes without omitting owners.
# Usage: julia --project=ENV test/release.jl --group=all --output=/outside/checkout
using Dates, Pkg, SHA, TOML

include("runner.jl")

const GROUPS = Dict(
    "core" => ["test_contracts.jl", "test_test_runner.jl", "test_field_linalg.jl",
               "test_finite_fringe.jl", "test_poset_interface.jl"],
    "encoding" => ["test_encoding.jl", "test_zn_backend.jl", "test_pl_backend.jl",
                   "test_synthetic_data.jl"],
    "geometry" => ["test_geometry.jl", "test_data_pipeline.jl", "test_ordinary_persistence.jl"],
    "resolutions" => ["test_indicator_resolutions.jl", "test_model_independent_ext_layer.jl",
                      "test_random_stress.jl"],
    "derived" => ["test_derived_functors.jl", "test_chain_complexes_homology.jl",
                 "test_functoriality_ext_tor_maps.jl"],
    "invariants" => ["test_invariants.jl"],
    "interfaces" => ["test_visualization.jl", "test_featurizers.jl", "test_extensions.jl",
                     "test_examples.jl"],
)
owners = reduce(vcat, values(GROUPS))
length(owners) == length(unique(owners)) &&
    Set(owners) == Set(TamerOpTestRunner._TEST_FILES) || error("Release groups must cover every owner exactly once")

options = Dict("group" => "all", "extensions" => "none")
seen = Set{String}()
for arg in ARGS
    pair = split(arg, '='; limit=2)
    length(pair) == 2 && startswith(pair[1], "--") || error("Expected --key=value: $arg")
    key, value = pair[1][3:end], pair[2]
    key in ("group", "output", "extensions") || error("Unknown option: $key")
    key in seen && error("Duplicate option: $key")
    isempty(value) && error("Empty option: $key")
    push!(seen, key)
    options[key] = value
end
group = options["group"]
group == "all" || haskey(GROUPS, group) || error("Unknown release group: $group")
options["extensions"] in ("none", "all") || error("extensions must be none or all")
haskey(options, "output") || error("An external --output directory is required")
root = realpath(joinpath(@__DIR__, ".."))
output = abspath(options["output"])
ispath(output) && (!isdir(output) || !isempty(readdir(output))) &&
    error("Output must be a new or empty directory")
mkpath(output)
relative = relpath(realpath(output), root)
(isabspath(relative) || first(splitpath(relative)) == "..") || error("Output must be outside the checkout")
git(args...) = readchomp(`git -C $root $args`)
isempty(git("status", "--porcelain", "--untracked-files=normal")) || error("Release verification requires a clean candidate checkout")
commit, tree = git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}")
project = TOML.parsefile(joinpath(root, "Project.toml"))
extensions = options["extensions"] == "all" ? sort!(collect(keys(project["extensions"]))) : String[]
files = group == "all" ? collect(TamerOpTestRunner._TEST_FILES) : GROUPS[group]
environment = dirname(Base.active_project())
for name in ("Project.toml", "Manifest.toml")
    path = joinpath(environment, name)
    isfile(path) && cp(path, joinpath(output, name); force=true)
end
environment_hashes() = Dict(name => (isfile(joinpath(environment, name)) ?
    bytes2hex(sha256(read(joinpath(environment, name)))) : "missing")
    for name in ("Project.toml", "Manifest.toml"))
initial_environment = environment_hashes()
status_path = joinpath(output, "environment.txt")
open(status_path, "w") do io
    println(io, "Julia ", VERSION, "; machine=", Sys.MACHINE)
    println(io, "Source commit: ", commit, "; tree: ", tree)
    Pkg.status(; io, mode=Pkg.PKGMODE_MANIFEST)
end
record = Dict{String,Any}(
    "commit" => commit, "tree" => tree, "julia" => string(VERSION),
    "machine" => Sys.MACHINE, "threads" => get(ENV, "JULIA_NUM_THREADS", string(Threads.nthreads())),
    "blas_threads" => 1, "fields" => collect(TamerOpTestRunner._FIELD_NAMES),
    "group" => group, "owners" => files, "required_extensions" => extensions,
    "seed" => 2677, "long_tests" => true, "started_utc" => string(now(UTC)),
    "threshold_profile_sha256" => bytes2hex(sha256(read(joinpath(root, "linalg_thresholds.toml")))),
    "environment_hashes" => initial_environment,
    "results" => Dict{String,Any}[], "complete" => false, "passed" => false,
)
save_record() = open(joinpath(output, "results.toml"), "w") do io
    TOML.print(io, record; sorted=true)
end
save_record()
try
    for file in files
        log = joinpath(output, replace(file, ".jl" => ".log"))
        args = ["--file=" * file, "--fields=QQ,F2,F3,F5,Real64"]
        append!(args, ["--require-extension=" * name for name in extensions])
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$environment $(joinpath(root, "test", "runtests.jl")) $args`
        command_text = string(cmd)
        cmd = addenv(cmd, "JULIA_LOAD_PATH" => (Sys.iswindows() ? "@;@stdlib" : "@:@stdlib"),
                     "OPENBLAS_NUM_THREADS" => "1", "POSETMODULES_LONG_TESTS" => "true",
                     "TAMEROP_TEST_SEED" => "2677")
        println("RUN ", file, "; log=", log)
        flush(stdout)
        started = time()
        code = open(log, "w") do io
            run(pipeline(ignorestatus(cmd); stdout=io, stderr=io)).exitcode
        end
        push!(record["results"], Dict("owner" => file, "exit_code" => code,
              "elapsed_seconds" => time() - started, "command" => command_text,
              "log" => basename(log), "log_sha256" => bytes2hex(sha256(read(log)))))
        save_record()
        println("DONE ", file, "; exit=", code)
        flush(stdout)
    end
    record["complete"] = true
catch err
    record["execution_error"] = sprint(showerror, err)
    rethrow()
finally
    record["finished_utc"] = string(now(UTC))
    record["source_unchanged"] = isempty(git("status", "--porcelain", "--untracked-files=normal")) &&
        git("rev-parse", "HEAD") == commit
    record["environment_unchanged"] = environment_hashes() == initial_environment
    record["passed"] = record["complete"] && record["source_unchanged"] && record["environment_unchanged"] && all(r["exit_code"] == 0 for r in record["results"])
    save_record()
end
record["passed"] || error("Release verification failed; inspect ", joinpath(output, "results.toml"))
println("PASS release group ", group, "; source=", commit)
