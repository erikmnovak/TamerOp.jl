# Run by run.jl in a fresh process, never in the developer's environment.
include("common.jl")
using InteractiveUtils

config = TOML.parsefile(ARGS[1])
phase = ARGS[2]
phase in ("core", "plot") || error("Unknown installation phase: $phase")
realpath(dirname(Base.active_project())) == realpath(config["environment"]) ||
    error("Incorrect installation environment")

if phase == "core"
    Pkg.add(PackageSpec(url=config["repository"], rev=config["revision"]))
else
    candidate_info(config)
    Pkg.add("CairoMakie")
end

info = candidate_info(config)
versioninfo()
Pkg.status(; mode=Pkg.PKGMODE_MANIFEST)
write_report(joinpath(config["output"], "environment_$phase.toml"),
             environment_report(config, info))
