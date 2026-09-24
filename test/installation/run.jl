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
    revision = options["--rev"]
    tree = readchomp(`git -C $checkout rev-parse $(revision * "^{tree}")`)
    tree == options["--tree"] || error("Supplied tree does not belong to the pinned commit")
    entries = Dict{String,String}[]
    for entry in split(read(`git -C $checkout ls-tree -r -z --full-tree $revision`, String), '\0'; keepempty=false)
        metadata, path = split(entry, '\t'; limit=2)
        mode, kind, blob = split(metadata)
        kind == "blob" && mode in ("100644", "100755", "120000") ||
            error("Unsupported candidate Git entry: $path ($kind, $mode)")
        push!(entries, Dict("path" => path, "mode" => mode, "blob" => blob))
    end
    inventory_path = joinpath(output, "candidate_files.toml")
    open(io -> TOML.print(io, Dict("tree" => tree, "files" => entries); sorted=true),
         inventory_path, "w")
    config = Dict(
        "repository" => isdir(options["--repo"]) ? realpath(options["--repo"]) : options["--repo"],
        "revision" => options["--rev"],
        "tree" => options["--tree"], "checkout" => checkout, "output" => output,
        "environment" => environment, "primary_depot" => depot, "inventory" => inventory_path,
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
            log_path = joinpath(output, label * ".log")
            process = open(log_path, "w") do log
                # A raw ProcessFailedException can print inherited environment
                # values. Preserve diagnostics without exposing that environment.
                child = ignorestatus(setenv(Cmd(cmd; dir=work), process_env))
                run(pipeline(child; stdout=log, stderr=log))
            end
            # Keep even partial resolver state when a later stage fails.
            for name in ("Project.toml", "Manifest.toml")
                path = joinpath(environment, name)
                isfile(path) && cp(path, joinpath(output, "$(label)_$name"))
            end
            if !success(process)
                println(stderr, join(last(readlines(log_path), 80), '\n'))
                error("Installation phase $label failed (exit $(process.exitcode), signal $(process.termsignal)); see $log_path")
            end
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
