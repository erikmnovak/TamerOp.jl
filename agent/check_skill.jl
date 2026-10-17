using Test

include(joinpath(@__DIR__, "install_skill.jl"))

function skill_files(directory)
    return Dict(relpath(joinpath(root, name), directory) => read(joinpath(root, name))
                for (root, _, names) in walkdir(directory) for name in names)
end

@testset "Optional agent skill installation" begin
    source = joinpath(@__DIR__, "skills", "tamerop")
    expected = skill_files(source)
    @test haskey(expected, "SKILL.md")
    @test haskey(expected, joinpath("agents", "openai.yaml"))

    mktempdir() do project
        # The same portable skill is discovered at each client's project path.
        for (client, folder) in (("codex", ".agents"), ("claude", ".claude"))
            destination = TamerOpSkillInstaller.install(client, project)
            @test destination == joinpath(project, folder, "skills", "tamerop")
            @test skill_files(destination) == expected
            write(joinpath(destination, "local-notes.txt"), "Preserve my changes.\n")
            before = skill_files(destination)
            @test_throws ArgumentError TamerOpSkillInstaller.install(client, project)
            @test skill_files(destination) == before
        end
        @test sort(readdir(project)) == [".agents", ".claude"]
        @test_throws ArgumentError TamerOpSkillInstaller.install("unknown", project)
        @test_throws ArgumentError TamerOpSkillInstaller.install("codex", joinpath(project, "missing"))
        @test !ispath(joinpath(project, "missing"))
    end

    mktempdir() do project
        # A dangling link is still an existing skill and must not be overwritten.
        directory = joinpath(project, ".agents", "skills")
        mkpath(directory)
        destination = joinpath(directory, "tamerop")
        if !Sys.iswindows()
            symlink(joinpath(project, "missing-target"), destination)
            @test_throws ArgumentError TamerOpSkillInstaller.install("codex", project)
            @test islink(destination)
        end
    end

    if !Sys.iswindows()
        mktempdir() do project
            mktempdir() do external
                symlink(external, joinpath(project, ".agents"))
                @test_throws ArgumentError TamerOpSkillInstaller.install("codex", project)
                @test isempty(readdir(external))
            end
        end
    end

    installer = joinpath(@__DIR__, "install_skill.jl")
    julia = Base.julia_cmd()
    @test success(pipeline(`$julia --startup-file=no $installer --help`; stdout=devnull))
    @test !success(pipeline(`$julia --startup-file=no $installer`; stdout=devnull, stderr=devnull))
    mktempdir() do project
        # Exercise the documented command as well as the function used after include.
        @test success(pipeline(`$julia --startup-file=no $installer codex $project`; stdout=devnull))
        @test skill_files(joinpath(project, ".agents", "skills", "tamerop")) == expected
    end
end
