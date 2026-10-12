#!/usr/bin/env julia
# Execute the canonical guide and derive figure and saved-session artifacts
# from its actual cells. The recipe is a separate bounded entry to the task.
module LiveInspectionNotebook
using JSON3
using Base64
const NOTEBOOK = normpath(joinpath(@__DIR__, "..", "tutorials", "live_inspection.ipynb"))
const RECIPE = normpath(joinpath(@__DIR__, "..", "recipes", "reopen_session.md"))

function check(; record=false, gallery=nothing, main_path=false)
    record && main_path && error("Record the complete notebook, not a partial execution.")
    original = read(NOTEBOOK, String)
    nb = JSON3.read(original, Dict{String,Any})
    workspace = Module(:LiveInspectionGuide)
    count = 0
    figures = 0
    cards = String[]
    retained = Set{String}()
    gallery === nothing || mkpath(gallery)
    escape(s) = replace(s, "&"=>"&amp;", "<"=>"&lt;", ">"=>"&gt;", "\""=>"&quot;")
    for (i, cell) in enumerate(nb["cells"])
        cell["cell_type"] == "code" || continue
        main_path && "optional" in get(cell["metadata"], "tags", String[]) && continue
        count += 1
        println("Executing live-inspection guide cell $i")
        flush(stdout)
        source = join(cell["source"])
        value = Base.include_string(workspace, source, "live_inspection.ipynb:cell-$i")
        data = Dict{String,Any}()
        value === nothing || (data["text/plain"] = [Base.invokelatest(sprint, show,
            MIME"text/plain"(), value; context=:limit=>true)])
        expected = get(get(cell["metadata"], "tamerop", Dict()), "figure_alt", String[])
        if !isempty(expected)
            Base.invokelatest(showable, MIME"image/png"(), value) ||
                error("Cell $i did not produce its promised figure.")
            length(expected) == 1 || error("Expected exactly one figure per guide cell.")
            buffer = IOBuffer()
            Base.invokelatest(show, buffer, MIME"image/png"(), value)
            bytes = take!(buffer)
            data["image/png"] = base64encode(bytes)
            figures += 1
            if gallery !== nothing
                stem = "session-$(lpad(figures, 2, '0'))"
                write(joinpath(gallery, stem * ".png"), bytes)
                Cairo = Base.invokelatest(getfield, workspace, :CairoMakie)
                Base.invokelatest(() -> Cairo.save(joinpath(gallery, stem * ".svg"), value))
                push!(cards, "<figure><a href=\"$stem.svg\"><img src=\"$stem.png\" alt=\"$(escape(only(expected)))\"></a><figcaption>$(escape(only(expected)))</figcaption></figure>")
            end
        end
        if gallery !== nothing
            for matched in eachmatch(r"OP\.save_visual_session\(\s*(json_path|html_path|ring_path)\s*,", source)
                name = Symbol(only(matched.captures))
                Base.invokelatest(isdefined, workspace, name) ||
                    error("Cell $i did not define its saved-artifact path $name.")
                path = Base.invokelatest(getfield, workspace, name)
                path isa AbstractString && isfile(path) && filesize(path) > 0 ||
                    error("Cell $i did not produce a nonempty saved artifact at $name.")
                destination = joinpath(gallery, basename(path))
                cp(path, destination; force=true)
                isfile(destination) && read(path) == read(destination) ||
                    error("The saved artifact $name was not retained faithfully in the gallery.")
                push!(retained, basename(path))
            end
        end
        if record
            cell["execution_count"] = count
            cell["outputs"] = isempty(data) ? Any[] : [Dict("output_type"=>"execute_result",
                "execution_count"=>count, "metadata"=>Dict(), "data"=>data)]
        end
    end
    if gallery !== nothing
        expected_artifacts = main_path ? ("square.json", "square.html") :
            ("square.json", "square.html", "ring.html")
        all(name -> name in retained && isfile(joinpath(gallery, name)), expected_artifacts) ||
            error("The gallery is missing a saved artifact from this execution: expected $(join(expected_artifacts, ", ")).")
    end
    if record
        read(NOTEBOOK, String) == original || error("The notebook changed during execution; rerun before recording outputs.")
        open(NOTEBOOK, "w") do io
            JSON3.pretty(io, nb)
            println(io)
        end
    end
    if gallery !== nothing
        links = "<p>Open the <a href=\"square.html\">square's three saved states</a>"
        "ring.html" in retained && (links *= " or the <a href=\"ring.html\">selected ordinary interval</a>")
        links *= " without Julia or network access. Select an image below for its SVG export.</p>"
        write(joinpath(gallery, "index.html"), "<!doctype html><meta charset=utf-8><title>Keeping a selected mathematical view</title><style>body{font:18px system-ui;max-width:1400px;margin:2em auto;padding:0 1em;background:#f6f8fa;color:#172b38}figure{margin:2em 0;background:white;padding:1em;border:1px solid #ccd7dd}img{width:100%;height:auto}figcaption{margin-top:1em}</style><h1>Keeping a selected mathematical view</h1><p>Figures and saved artifacts executed from the canonical live-inspection guide.</p>" * links * join(cards, "\n"))
    end
    println("Live-inspection guide: $count code cells and $figures figures executed successfully.")
    return (; cells=count, figures, workspace)
end

function check_recipe()
    workspace = Module(:ReopenSessionRecipe)
    blocks = collect(eachmatch(r"```julia\n(.*?)\n```"s, read(RECIPE, String)))
    isempty(blocks) && error("The reopen-session recipe has no Julia blocks.")
    mktempdir() do folder
        cd(folder) do
            for (i, block) in enumerate(blocks)
                Base.include_string(workspace, block.captures[1], "reopen_session.md:block-$i")
            end
        end
    end
    println("Reopen-session recipe: $(length(blocks)) blocks executed successfully.")
    return (; blocks=length(blocks))
end
end

if abspath(PROGRAM_FILE) == @__FILE__
    all(a -> a in ("--record", "--main-path", "--recipe") || startswith(a, "--gallery="), ARGS) ||
        error("Usage: check_live_inspection.jl [--record | --main-path] [--gallery=PATH] [--recipe]")
    galleries = filter(a -> startswith(a, "--gallery="), ARGS)
    length(galleries) <= 1 || error("Supply at most one gallery path.")
    LiveInspectionNotebook.check(; record="--record" in ARGS, main_path="--main-path" in ARGS,
        gallery=isempty(galleries) ? nothing : split(only(galleries), '='; limit=2)[2])
    "--recipe" in ARGS && LiveInspectionNotebook.check_recipe()
end
