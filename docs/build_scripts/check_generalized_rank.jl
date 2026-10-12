#!/usr/bin/env julia
# Execute the canonical region guide, capture actual figures, and check its recipe.
module GeneralizedRankNotebook
using JSON3, Base64
const NOTEBOOK=normpath(joinpath(@__DIR__,"..","tutorials","generalized_rank_queries.ipynb"))
const RECIPE=normpath(joinpath(@__DIR__,"..","recipes","query_region_invariant.md"))
function check(;record=false,gallery=nothing,main_path=false)
    record && main_path && error("Record the whole notebook, not a partial execution.")
    original=read(NOTEBOOK,String);nb=JSON3.read(original,Dict{String,Any})
    workspace=Module(:GeneralizedRankGuide);count=0;figures=0;cards=String[]
    gallery===nothing || mkpath(gallery)
    escape(s)=replace(s,"&"=>"&amp;","<"=>"&lt;",">"=>"&gt;","\""=>"&quot;")
    for (i,cell) in enumerate(nb["cells"])
        cell["cell_type"]=="code" || continue
        main_path && "optional" in get(cell["metadata"],"tags",String[]) && continue
        count+=1;println("Executing generalized-rank guide cell $i");flush(stdout)
        value=Base.include_string(workspace,join(cell["source"]),"generalized_rank_queries.ipynb:cell-$i")
        data=Dict{String,Any}()
        value===nothing || (data["text/plain"]=[Base.invokelatest(sprint,show,MIME"text/plain"(),value;context=:limit=>true)])
        expected=get(get(cell["metadata"],"tamerop",Dict()),"figure_alt",String[])
        if !isempty(expected)
            Base.invokelatest(showable,MIME"image/png"(),value) || error("Cell $i did not produce its promised figure.")
            length(expected)==1 || error("Expected one figure per cell.")
            buffer=IOBuffer();Base.invokelatest(show,buffer,MIME"image/png"(),value)
            bytes=take!(buffer);data["image/png"]=base64encode(bytes);figures+=1
            if gallery!==nothing
                stem="region-$(lpad(figures,2,'0'))";write(joinpath(gallery,stem*".png"),bytes)
                Cairo=Base.invokelatest(getfield,workspace,:CairoMakie)
                Base.invokelatest(() -> Cairo.save(joinpath(gallery,stem*".svg"),value))
                push!(cards,"<figure><a href=\"$stem.svg\"><img src=\"$stem.png\" alt=\"$(escape(only(expected)))\"></a><figcaption>$(escape(only(expected)))</figcaption></figure>")
            end
        end
        if record
            cell["execution_count"]=count
            cell["outputs"]=isempty(data) ? Any[] : [Dict("output_type"=>"execute_result","execution_count"=>count,
                "metadata"=>Dict(),"data"=>data)]
        end
    end
    if record
        read(NOTEBOOK,String)==original || error("Notebook changed during execution.")
        open(NOTEBOOK,"w") do io;JSON3.pretty(io,nb);println(io);end
    end
    if gallery!==nothing
        write(joinpath(gallery,"index.html"),"<!doctype html><meta charset=utf-8><title>Generalized rank and GRIL</title><style>body{font:18px system-ui;max-width:1400px;margin:2em auto;padding:0 1em;background:#f6f8fa;color:#172b38}figure{margin:2em 0;background:white;padding:1em;border:1px solid #ccd7dd}img{width:100%;height:auto}figcaption{margin-top:1em}</style><h1>Querying regions and building GRIL features</h1><p>Figures executed from the canonical guide. Select a figure for its SVG.</p>"*join(cards,"\n"))
    end
    println("Generalized-rank guide: $count code cells and $figures figures executed.")
    (;cells=count,figures,workspace)
end
function check_recipe()
    workspace=Module(:RegionRankRecipe)
    blocks=collect(eachmatch(r"```julia\n(.*?)\n```"s,read(RECIPE,String)))
    isempty(blocks) && error("No Julia recipe blocks")
    for (i,block) in enumerate(blocks)
        Base.include_string(workspace,block.captures[1],"query_region_invariant.md:block-$i")
    end
    println("Region recipe: $(length(blocks)) executable blocks passed.")
    (;blocks=length(blocks))
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    all(a->a in ("--record","--main-path","--recipe") || startswith(a,"--gallery="),ARGS) ||
        error("Usage: check_generalized_rank.jl [--record | --main-path] [--recipe] [--gallery=PATH]")
    galleries=filter(a->startswith(a,"--gallery="),ARGS);length(galleries)<=1 || error("Supply one gallery path")
    GeneralizedRankNotebook.check(;record="--record" in ARGS,main_path="--main-path" in ARGS,
        gallery=isempty(galleries) ? nothing : split(only(galleries),'=';limit=2)[2])
    "--recipe" in ARGS && GeneralizedRankNotebook.check_recipe()
end
