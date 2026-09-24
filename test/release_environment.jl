# Build the release test environment outside the immutable candidate checkout.
using Pkg, TOML

length(ARGS) == 2 || error("Usage: release_environment.jl /external/environment core|extensions")
environment = abspath(ARGS[1])
mode = ARGS[2]
mode in ("core", "extensions") || error("Expected core or extensions")
root = realpath(joinpath(@__DIR__, ".."))
mkpath(environment)
relative = relpath(realpath(environment), root)
(isabspath(relative) || first(splitpath(relative)) == "..") || error("Environment must be outside the checkout")
isfile(joinpath(environment, "Project.toml")) && error("Use a fresh environment directory")
package = TOML.parsefile(joinpath(root, "Project.toml"))
deps = merge(package["deps"], package["extras"])
mode == "extensions" && merge!(deps, package["weakdeps"])
compat = Dict(k => v for (k, v) in package["compat"] if k == "julia" || haskey(deps, k))
open(joinpath(environment, "Project.toml"), "w") do io
    TOML.print(io, Dict("deps" => deps, "compat" => compat); sorted=true)
end
Pkg.activate(environment)
Pkg.develop(path=root)
Pkg.instantiate()
Pkg.precompile()
Pkg.status(; mode=Pkg.PKGMODE_MANIFEST)
