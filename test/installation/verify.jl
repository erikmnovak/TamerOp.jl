# Public computations from outside the checkout, with no test prelude or source includes.
include("common.jl")
using Test

config = TOML.parsefile(ARGS[1])
phase = ARGS[2]
phase in ("core", "plot") || error("Unknown verification phase: $phase")
info = candidate_info(config)
source_before = bytes2hex(Pkg.GitTools.tree_hash(info.source))
transforms_before = verify_source(config, info.source)
inside(pwd(), config["checkout"]) && error("Verification must run outside the checkout")
import TamerOp as OP

# Complete the import as a separate top-level expression before running tests.
# This also establishes the extension methods in Julia 1.12's current world.
if phase == "plot"
    import CairoMakie
end

@testset "Installed candidate identity" begin
    @test realpath(pathof(OP)) == realpath(joinpath(info.source, "src", "TamerOp.jl"))
    @test realpath(dirname(Base.active_project())) == realpath(config["environment"])
    @test LOAD_PATH == ["@", "@stdlib"]
    @test !isdir(joinpath(info.source, ".git"))
    for path in ("AGENTS.md", "examples", "audit", "benchmark")
        @test !ispath(joinpath(info.source, path))
    end
end

if phase == "core"
    @testset "Minimal installed environment" begin
        project = TOML.parsefile(joinpath(info.source, "Project.toml"))
        @test Set(keys(TOML.parsefile(Base.active_project())["deps"])) == Set(["TamerOp"])
        for extension in keys(project["extensions"])
            @test Base.get_extension(OP, Symbol(extension)) === nothing
        end
        @test Base.find_package("CairoMakie") === nothing
    end

    @testset "Installed image computation: a ring fills at five" begin
        diagram = OP.cubical_persistence([0 0 0; 0 5 0; 0 0 0])
        @test OP.finite_intervals(diagram; dim=1) == [(0, 5)]
        @test OP.essential_births(diagram; dim=0) == [0]
        @test isempty(OP.finite_intervals(diagram; dim=0))
        @test isempty(OP.essential_births(diagram; dim=1))
    end

    @testset "Installed algebra: the non-split two-point interval" begin
        # On 1 < 2, 0 -> S2 -> [1,2] -> S1 -> 0 generates Ext^1(S1,S2).
        # Hom(S1,S2)=0; Hom(S1,S1)=k and Ext^1(S1,S1)=0.
        for field in (OP.CoreModules.QQField(), OP.CoreModules.F3())
            K = OP.CoreModules.coeff_type(field)
            P = OP.Advanced.FinitePoset(Bool[1 1; 0 1])
            S1 = OP.Advanced.PModule(P, [1, 0], Dict((1, 2) => zeros(K, 0, 1)); field=field)
            S2 = OP.Advanced.PModule(P, [0, 1], Dict((1, 2) => zeros(K, 1, 0)); field=field)
            cross = OP.ext(S1, S2; maxdeg=1)
            self = OP.ext(S1, S1; maxdeg=1)
            @test OP.Advanced.dim(cross, 0) == 0
            @test OP.Advanced.dim(cross, 1) == 1
            @test OP.Advanced.dim(self, 0) == 1
            @test OP.Advanced.dim(self, 1) == 0
        end
    end
else
    @testset "Installed optional renderer and actual exports" begin
        @test Base.get_extension(OP, :TamerOpCairoMakieExt) !== nothing
        diagram = OP.cubical_persistence([0 0 0; 0 5 0; 0 0 0])
        @test OP.finite_intervals(diagram; dim=1) == [(0, 5)]
        figures = joinpath(config["output"], "figures")
        mkpath(figures)
        png_path = OP.save_visual(joinpath(figures, "ring_barcode.png"), diagram;
                                 kind=:barcode, dim=1, backend=:cairomakie)
        svg_path = OP.save_visual(joinpath(figures, "ring_diagram.svg"), diagram;
                                 kind=:persistence_diagram, dim=1, backend=:cairomakie)
        png = read(png_path)
        @test png[1:8] == UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
        @test length(png) > 1000
        svg = read(svg_path, String)
        @test occursin("<svg", svg)
        @test occursin("<path", svg)
        @test sort(readdir(figures)) == ["ring_barcode.png", "ring_diagram.svg"]
    end
end

@testset "Installed package remains unchanged" begin
    @test bytes2hex(Pkg.GitTools.tree_hash(info.source)) == source_before
    @test verify_source(config, info.source) == transforms_before
end
report = environment_report(config, info)
report["phase"] = phase
report["result"] = "passed"
report["source_unchanged"] = true
write_report(joinpath(config["output"], "verified_$phase.toml"), report)
