# Shared checks for the standalone, installed-package verification processes.
using Pkg, TOML, UUIDs, SHA

const TAMEROP_UUID = UUID("40830518-3340-46e0-bdcb-56edef4036ff")

function inside(path, directory)
    relative = relpath(realpath(path), realpath(directory))
    return !isabspath(relative) && (relative == "." || first(splitpath(relative)) != "..")
end

write_report(path, values) = open(io -> TOML.print(io, values; sorted=true), path, "w")

function _git_blob_hash(data)
    header = Vector{UInt8}(codeunits("blob $(length(data))\0"))
    return bytes2hex(sha1([header; data]))
end

function source_files(directory)
    files = Dict{String,String}()
    modes = Dict{String,String}()
    directories = Set{String}()
    function visit(parent)
        for path in readdir(parent; join=true)
            relative = replace(relpath(path, directory), '\\' => '/')
            if isdir(path) && !islink(path)
                push!(directories, relative)
                visit(path)
            else
                # Git hashes the exact bytes, including a symlink's target.
                data = islink(path) ? Vector{UInt8}(codeunits(readlink(path))) : read(path)
                files[relative] = _git_blob_hash(data)
                modes[relative] = islink(path) ? "120000" :
                    iszero(filemode(path) & 0o100) ? "100644" : "100755"
            end
        end
    end
    visit(directory)
    return files, modes, directories
end

function verify_source(config, source; windows::Bool=Sys.iswindows())
    inventory = TOML.parsefile(config["inventory"])
    inventory["tree"] == config["tree"] || error("Candidate inventory has the wrong tree")
    expected = Dict(entry["path"] => entry["blob"] for entry in inventory["files"])
    expected_modes = Dict(entry["path"] => entry["mode"] for entry in inventory["files"])
    expected_directories = Set{String}()
    for path in keys(expected)
        parts = split(path, '/')
        for count in 1:(length(parts) - 1)
            push!(expected_directories, join(parts[1:count], '/'))
        end
    end
    actual, modes, directories = source_files(source)
    missing = sort!(collect(setdiff(keys(expected), keys(actual))))
    extra = sort!(collect(setdiff(keys(actual), keys(expected))))
    changed = Dict{String,String}[]
    checkout_transforms = String[]
    for path in sort!(collect(intersect(keys(expected), keys(actual))))
        expected[path] == actual[path] && continue
        # Windows checkouts can leave the attributes file itself with CRLF
        # despite its LF policy. Accept only this verified metadata
        # transformation; code, data, symlinks and every other file stay exact.
        if windows && path == ".gitattributes" && modes[path] != "120000"
            normalized = codeunits(replace(read(joinpath(source, path), String), "\r\n" => "\n"))
            if _git_blob_hash(normalized) == expected[path]
                push!(checkout_transforms, ".gitattributes: CRLF to LF")
                continue
            end
        end
        push!(changed, Dict("path" => path, "expected" => expected[path], "actual" => actual[path]))
    end
    # NTFS ACLs do not represent Git's POSIX executable bit. On Windows the
    # authenticated Git inventory retains that metadata; raw file bytes and
    # symlink kinds still must match exactly. POSIX checks executable bits too.
    mode_changes = [path for path in intersect(keys(expected), keys(actual))
                    if (windows ? (expected_modes[path] == "120000") !=
                                          (modes[path] == "120000") :
                                          expected_modes[path] != modes[path])]
    if !isempty(missing) || !isempty(extra) || !isempty(changed) ||
       !isempty(mode_changes) || directories != expected_directories
        report = Dict("missing" => missing, "extra" => extra, "changed" => changed,
                      "mode_changes" => sort!(mode_changes),
                      "missing_directories" => sort!(collect(setdiff(expected_directories, directories))),
                      "extra_directories" => sort!(collect(setdiff(directories, expected_directories))),
                      "expected_tree" => config["tree"],
                      "filesystem_tree" => bytes2hex(Pkg.GitTools.tree_hash(source)))
        write_report(joinpath(config["output"], "source_mismatch.toml"), report)
        TOML.print(stderr, report; sorted=true)
        error("Installed source does not match the pinned Git file inventory")
    end
    return checkout_transforms
end

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
    verify_source(config, info.source)
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
        "filesystem_tree" => bytes2hex(Pkg.GitTools.tree_hash(info.source)),
        "source_verification" => "Git paths, raw blob bytes and symlink kinds; Windows attributes-file CRLF permitted",
        "checkout_transforms" => verify_source(config, info.source),
        "posix_executable_modes_checked" => !Sys.iswindows(),
        "installed_source" => info.source,
        "active_project" => Base.active_project(),
        "depots" => copy(DEPOT_PATH),
        "load_path" => copy(LOAD_PATH),
        "dependencies" => dependencies,
    )
end
