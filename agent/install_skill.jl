module TamerOpSkillInstaller

"""
    install(client, project_dir)

Copy the optional TamerOp skill into an existing project directory. `client`
is `"codex"` or `"claude"`; the corresponding location is
`.agents/skills/tamerop` or `.claude/skills/tamerop`. Existing skills are preserved.
No Julia packages or global agent settings are changed.
"""
function install(client::AbstractString, project_dir::AbstractString)
    container = if client == "codex"
        ".agents"
    elseif client == "claude"
        ".claude"
    else
        throw(ArgumentError("Choose codex or claude; received $(repr(client))."))
    end
    project = abspath(project_dir)
    isdir(project) || throw(ArgumentError("Project directory does not exist: $project"))
    source = joinpath(@__DIR__, "skills", "tamerop")
    isfile(joinpath(source, "SKILL.md")) || error("The packaged TamerOp skill is missing: $source")
    destination = joinpath(project, container, "skills", "tamerop")
    (ispath(destination) || islink(destination)) &&
        throw(ArgumentError("A skill already exists at $destination. Review it before replacing it."))

    # Keep the project installation inside its selected directory, including when
    # an existing configuration directory is a link to a shared/global location.
    for parent in (joinpath(project, container), dirname(destination))
        islink(parent) && throw(ArgumentError("Skill installation parent is a symbolic link: $parent. Copy the skill manually to your intended location."))
    end
    mkpath(dirname(destination))
    cp(source, destination; force=false, follow_symlinks=false)
    return destination
end

function main(args=ARGS)
    if args == ["--help"] || args == ["-h"]
        println("Usage: julia agent/install_skill.jl <codex|claude> <project-directory>")
        println("Copies the optional skill into this project; preserves any existing skill.")
        return 0
    end
    length(args) == 2 || throw(ArgumentError("Usage: julia agent/install_skill.jl <codex|claude> <project-directory>"))
    destination = install(args[1], args[2])
    println("Installed TamerOp skill: ", destination)
    return 0
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    try
        exit(TamerOpSkillInstaller.main())
    catch exception
        showerror(stderr, exception)
        println(stderr)
        exit(1)
    end
end
