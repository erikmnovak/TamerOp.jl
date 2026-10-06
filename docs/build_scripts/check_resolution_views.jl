#!/usr/bin/env julia
# Execute the canonical guide and retain its real text/figure outputs. Gallery
# exports are derived from those same cells, never a second authored example.
module ResolutionNotebook
using JSON3
using Base64
const NOTEBOOK = normpath(joinpath(@__DIR__, "..", "tutorials", "resolutions.ipynb"))
function check(; record=false, gallery=nothing)
    nb=JSON3.read(read(NOTEBOOK,String),Dict{String,Any})
    workspace=Module(:ResolutionGuide)
    count=0;figures=0;cards=String[]
    gallery===nothing || mkpath(gallery)
    escape(s)=replace(s,"&"=>"&amp;","<"=>"&lt;",">"=>"&gt;","\""=>"&quot;")
    for (i,cell) in enumerate(nb["cells"])
        cell["cell_type"]=="code" || continue
        count+=1
        println("Executing resolution guide cell $i");flush(stdout)
        value=Base.include_string(workspace,join(cell["source"]),"resolutions.ipynb:cell-$i")
        data=Dict{String,Any}()
        value===nothing || (data["text/plain"]=[Base.invokelatest(sprint,show,MIME"text/plain"(),value;context=:limit=>true)])
        expected=get(get(cell["metadata"],"tamerop",Dict()),"figure_alt",String[])
        if !isempty(expected)
            Base.invokelatest(showable,MIME"image/png"(),value) || error("Cell $i did not produce its promised figure.")
            length(expected)==1 || error("Expected exactly one figure per guide cell.")
            buffer=IOBuffer();Base.invokelatest(show,buffer,MIME"image/png"(),value)
            bytes=take!(buffer);data["image/png"]=base64encode(bytes);figures+=1
            if gallery!==nothing
                stem="resolution-$(lpad(figures,2,'0'))"
                write(joinpath(gallery,stem*".png"),bytes)
                # SVG uses the same prepared figure with the same semantic data.
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
        open(NOTEBOOK,"w") do io;JSON3.pretty(io,nb);println(io);end
    end
    if gallery!==nothing
        write(joinpath(gallery,"index.html"),"<!doctype html><meta charset=utf-8><title>Resolution views</title><style>body{font:18px system-ui;max-width:1400px;margin:2em auto;padding:0 1em;background:#f6f8fa;color:#172b38}figure{margin:2em 0;background:white;padding:1em;border:1px solid #ccd7dd}img{width:100%;height:auto}figcaption{margin-top:1em}</style><h1>Resolution views</h1><p>Figures executed from the canonical resolution guide. Select an image for its SVG export.</p>"*join(cards,"\n"))
    end
    println("Resolution guide: $count code cells and $figures figures executed successfully.")
    return (;cells=count,figures,workspace)
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    all(a->a=="--record" || startswith(a,"--gallery="),ARGS) || error("Usage: check_resolution_views.jl [--record] [--gallery=PATH]")
    galleries=filter(a->startswith(a,"--gallery="),ARGS)
    length(galleries)<=1 || error("Supply at most one gallery path.")
    ResolutionNotebook.check(;record="--record" in ARGS,gallery=isempty(galleries) ? nothing : split(only(galleries),'=';limit=2)[2])
end
