# Standalone release installation check; uses only Julia standard libraries.
using Dates, TOML

function main(args)
    options = Dict{String,String}()
    caches = String[]
    for arg in args
        if arg == "--help"
            length(args) == 1 || error("Use --help by itself")
            println("julia --startup-file=no test/installation/run.jl --repo=URL --rev=COMMIT --tree=TREE [--output=NEW_DIRECTORY] [--cache-depot=DIRECTORY ...]")
            println("COMMIT and TREE must be full Git SHA-1 hashes. The default depot is fresh; cache depots are optional and recorded.")
            return
        end
        pair = split(arg, '='; limit=2)
        length(pair) == 2 && !isempty(pair[2]) || error("Expected --name=value; use --help")
        key, value = pair
        if key == "--cache-depot"
            isdir(value) || error("Cache depot does not exist: $value")
            push!(caches, realpath(value))
        elseif key in ("--repo", "--rev", "--tree", "--output")
            haskey(options, key) && error("Repeated option: $key")
            options[key] = value
        else
            error("Unknown option: $key")
        end
    end
    for key in ("--repo", "--rev", "--tree")
        haskey(options, key) || error("Missing required option: $key")
    end
    for key in ("--rev", "--tree")
        occursin(r"^[0-9a-f]{40}$", options[key]) || error("$key must be a full lowercase Git SHA-1")
    end

    checkout = realpath(joinpath(@__DIR__, "..", ".."))
    output = haskey(options, "--output") ? abspath(options["--output"]) : mktempdir(; cleanup=false)
    ispath(output) && (!isdir(output) || !isempty(readdir(output))) &&
        error("Output must be a new or empty directory: $output")
    mkpath(output)
    output = realpath(output)
    relative = relpath(output, checkout)
    (!isabspath(relative) && (relative == "." || first(splitpath(relative)) != "..")) &&
        error("Output must be outside the checkout")
    environment = joinpath(output, "environment")
    depot = joinpath(output, "depot")
    harness = joinpath(output, "harness")
    work = joinpath(output, "work")
    foreach(mkpath, (environment, depot, harness, work))
    for name in ("common.jl", "install.jl", "verify.jl", "run.jl")
        cp(joinpath(@__DIR__, name), joinpath(harness, name))
    end
    config = Dict(
        "repository" => isdir(options["--repo"]) ? realpath(options["--repo"]) : options["--repo"],
        "revision" => options["--rev"],
        "tree" => options["--tree"], "checkout" => checkout, "output" => output,
        "environment" => environment, "primary_depot" => depot,
        "cache_depots" => caches, "fresh_primary_depot" => true,
        "cache_policy" => isempty(caches) ? "no reused depots" : "explicit fallback depots",
        "offline_requested" => get(ENV, "JULIA_PKG_OFFLINE", "false"),
        "started_utc" => string(now(UTC)), "state" => "running",
        "completed_phases" => String[],
    )
    config_path = joinpath(output, "verification.toml")
    save() = open(io -> TOML.print(io, config; sorted=true), config_path, "w")
    save()
    separator = Sys.iswindows() ? ';' : ':'
    process_env = copy(ENV)
    process_env["JULIA_DEPOT_PATH"] = join([depot; caches], separator)
    process_env["JULIA_LOAD_PATH"] = join(["@", "@stdlib"], separator)
    process_env["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
    process_env["JULIA_NUM_PRECOMPILE_TASKS"] = get(ENV, "JULIA_NUM_PRECOMPILE_TASKS", "1")
    println("Installation evidence: ", output)
    flush(stdout)
    try
        for (script, phase) in (("install", "core"), ("verify", "core"),
                                ("install", "plot"), ("verify", "plot"))
            label = "$(script)_$(phase)"
            println("Starting ", label, " in a fresh Julia process")
            flush(stdout)
            cmd = `$(Base.julia_cmd()) --startup-file=no --history-file=no --project=$environment $(joinpath(harness, script * ".jl")) $config_path $phase`
            open(joinpath(output, label * ".log"), "w") do log
                run(pipeline(setenv(Cmd(cmd; dir=work), process_env); stdout=log, stderr=log))
            end
            cp(joinpath(environment, "Project.toml"), joinpath(output, "$(label)_Project.toml"))
            cp(joinpath(environment, "Manifest.toml"), joinpath(output, "$(label)_Manifest.toml"))
            push!(config["completed_phases"], label)
            save()
            println("Passed ", label)
            flush(stdout)
        end
        config["state"] = "passed"
    catch
        config["state"] = "failed"
        rethrow()
    finally
        config["finished_utc"] = string(now(UTC))
        save()
    end
end

main(ARGS)
