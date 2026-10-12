#!/usr/bin/env julia
# Execute the canonical guide and derive figure and saved-session artifacts
# from its actual cells. The recipe is a separate bounded entry to the task.
module PresentationsNotebook
using JSON3
using Base64
const NOTEBOOK = normpath(joinpath(@__DIR__, "..", "tutorials", "presentations.ipynb"))
const RECIPE = normpath(joinpath(@__DIR__, "..", "recipes", "build_module.md"))

function check(; record=false, gallery=nothing, main_path=false)
    record && main_path && error("Record the complete notebook, not a partial execution.")
    original = read(NOTEBOOK, String)
    nb = JSON3.read(original, Dict{String,Any})
    workspace = Module(:PresentationsGuide)
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
        println("Executing presentations guide cell $i")
        flush(stdout)
        source = join(cell["source"])
        value = Base.include_string(workspace, source, "presentations.ipynb:cell-$i")
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
                stem = "presentation-$(lpad(figures, 2, '0'))"
                write(joinpath(gallery, stem * ".png"), bytes)
                Cairo = Base.invokelatest(getfield, workspace, :CairoMakie)
                Base.invokelatest(() -> Cairo.save(joinpath(gallery, stem * ".svg"), value))
                push!(cards, "<figure><a href=\"$stem.svg\"><img src=\"$stem.png\" alt=\"$(escape(only(expected)))\"></a><figcaption>$(escape(only(expected)))</figcaption></figure>")
            end
        end
        if gallery !== nothing
            for name in (:json_path, :html_path, :code_path)
                Base.invokelatest(isdefined, workspace, name) || continue
                path = Base.invokelatest(getfield, workspace, name)
                isfile(path) && filesize(path)>0 || error("Missing guide artifact: $path")
                cp(path, joinpath(gallery,basename(path));force=true)
                push!(retained,basename(path))
            end
        end
        if record
            cell["execution_count"] = count
            cell["outputs"] = isempty(data) ? Any[] : [Dict("output_type"=>"execute_result",
                "execution_count"=>count, "metadata"=>Dict(), "data"=>data)]
        end
    end
    if gallery !== nothing
        expected_artifacts = main_path ? () : ("diamond.json", "diamond.html", "diamond.jl")
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
        links = main_path ? "" : "<p><a href=\"diamond.html\">Open the saved diamond</a>; download its <a href=\"diamond.json\">draft</a> or <a href=\"diamond.jl\">Julia construction</a>.</p>"
        write(joinpath(gallery, "index.html"), "<!doctype html><meta charset=utf-8><title>Constructing modules without a filtration</title><style>body{font:18px system-ui;max-width:1400px;margin:2em auto;padding:0 1em;background:#f6f8fa;color:#172b38}figure{margin:2em 0;background:white;padding:1em;border:1px solid #ccd7dd}img{width:100%;height:auto}figcaption{margin-top:1em}</style><h1>Constructing modules without a filtration</h1><p>Figures executed from the canonical construction guide. Select a figure for its SVG.</p>" * links * join(cards, "\n"))
    end
    println("Presentations guide: $count code cells and $figures figures executed successfully.")
    return (; cells=count, figures, workspace)
end

function check_recipe()
    workspace = Module(:BuildModuleRecipe)
    blocks = collect(eachmatch(r"```julia\n(.*?)\n```"s, read(RECIPE, String)))
    isempty(blocks) && error("The build-module recipe has no Julia blocks.")
    mktempdir() do folder
        cd(folder) do
            for (i, block) in enumerate(blocks)
                Base.include_string(workspace, block.captures[1], "build_module.md:block-$i")
            end
        end
    end
    println("Build-module recipe: $(length(blocks)) blocks executed successfully.")
    return (; blocks=length(blocks))
end
end

if abspath(PROGRAM_FILE) == @__FILE__
    all(a -> a in ("--record", "--main-path", "--recipe") || startswith(a, "--gallery="), ARGS) ||
        error("Usage: check_presentations.jl [--record | --main-path] [--gallery=PATH] [--recipe]")
    galleries = filter(a -> startswith(a, "--gallery="), ARGS)
    length(galleries) <= 1 || error("Supply at most one gallery path.")
    PresentationsNotebook.check(; record="--record" in ARGS, main_path="--main-path" in ARGS,
        gallery=isempty(galleries) ? nothing : split(only(galleries), '='; limit=2)[2])
    "--recipe" in ARGS && PresentationsNotebook.check_recipe()
end
