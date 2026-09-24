# Shared checks for the standalone, installed-package verification processes.
using Pkg, TOML, UUIDs

const TAMEROP_UUID = UUID("40830518-3340-46e0-bdcb-56edef4036ff")

function inside(path, directory)
    relative = relpath(realpath(path), realpath(directory))
    return !isabspath(relative) && (relative == "." || first(splitpath(relative)) != "..")
end

write_report(path, values) = open(io -> TOML.print(io, values; sorted=true), path, "w")

function candidate_info(config)
    info = Pkg.dependencies()[TAMEROP_UUID]
    info.is_tracking_repo || error("TamerOp must be installed from a Git revision")
    !info.is_tracking_path || error("Path/develop installations are not release verification")
    info.git_revision == config["revision"] || error("Installed revision differs from candidate")
    info.tree_hash == config["tree"] || error("Manifest tree differs from candidate")
    inside(info.source, config["checkout"]) && error("Package source is inside the checkout")
    any(depot -> isdir(joinpath(depot, "packages")) &&
        inside(info.source, joinpath(depot, "packages")), DEPOT_PATH) ||
        error("TamerOp did not load from package-manager storage")
    # Pkg's own Git tree calculation includes file contents, modes and paths.
    # Check actual bytes as well as the manifest's declared source identity.
    bytes2hex(Pkg.GitTools.tree_hash(info.source)) == config["tree"] ||
        error("Installed package contents differ from the candidate Git tree")
    for name in ("common.jl", "install.jl", "verify.jl", "run.jl")
        read(joinpath(info.source, "test", "installation", name)) == read(joinpath(@__DIR__, name)) ||
            error("Installation harness differs from candidate: $name")
    end
    return info
end

function environment_report(config, info)
    dependencies = Dict(string(uuid) => Dict(
        "name" => dependency.name,
        "version" => string(something(dependency.version, "stdlib")),
        "tree" => something(dependency.tree_hash, ""),
    ) for (uuid, dependency) in Pkg.dependencies())
    return Dict(
        "julia_version" => string(VERSION),
        "machine" => Sys.MACHINE,
        "threads" => Threads.nthreads(),
        "revision" => config["revision"],
        "tree" => config["tree"],
        "installed_source" => info.source,
        "active_project" => Base.active_project(),
        "depots" => copy(DEPOT_PATH),
        "load_path" => copy(LOAD_PATH),
        "dependencies" => dependencies,
    )
end
