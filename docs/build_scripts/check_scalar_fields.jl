#!/usr/bin/env julia
# Execute the canonical region guide, capture actual figures, and check its recipe.
module ScalarFieldsNotebook
using JSON3, Base64
const NOTEBOOK=normpath(joinpath(@__DIR__,"..","tutorials","features.ipynb"))

function check(;record=false,gallery=nothing,main_path=false)
    record && main_path && error("Record the whole notebook, not a partial execution.")
    original=read(NOTEBOOK,String);nb=JSON3.read(original,Dict{String,Any})
    workspace=Module(:ScalarFieldsGuide);count=0;figures=0;cards=String[]
    gallery===nothing || mkpath(gallery)
    escape(s)=replace(s,"&"=>"&amp;","<"=>"&lt;",">"=>"&gt;","\""=>"&quot;")
    for (i,cell) in enumerate(nb["cells"])
        cell["cell_type"]=="code" || continue
        main_path && "optional" in get(cell["metadata"],"tags",String[]) && continue
        count+=1;println("Executing scalar-fields guide cell $i");flush(stdout)
        value=Base.include_string(workspace,join(cell["source"]),"features.ipynb:cell-$i")
        data=Dict{String,Any}()
        value===nothing || (data["text/plain"]=[Base.invokelatest(sprint,show,MIME"text/plain"(),value;context=:limit=>true)])
        expected=get(get(cell["metadata"],"tamerop",Dict()),"figure_alt",String[])
        if !isempty(expected)
            Base.invokelatest(showable,MIME"image/png"(),value) || error("Cell $i did not produce its promised figure.")
            length(expected)==1 || error("Expected one figure per cell.")
            buffer=IOBuffer();Base.invokelatest(show,buffer,MIME"image/png"(),value)
            bytes=take!(buffer);data["image/png"]=base64encode(bytes);figures+=1
            if gallery!==nothing
                stem="field-$(lpad(figures,2,'0'))";write(joinpath(gallery,stem*".png"),bytes)
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
        write(joinpath(gallery,"index.html"),"<!doctype html><meta charset=utf-8><title>Scalar fields and ambient landscapes</title><style>body{font:18px system-ui;max-width:1400px;margin:2em auto;padding:0 1em;background:#f6f8fa;color:#172b38}figure{margin:2em 0;background:white;padding:1em;border:1px solid #ccd7dd}img{width:100%;height:auto}figcaption{margin-top:1em}</style><h1>Reading scalar fields in their original coordinates</h1><p>Figures executed from the canonical guide. Select a figure for its SVG.</p>"*join(cards,"\n"))
    end
    println("Scalar-fields guide: $count code cells and $figures figures executed.")
    (;cells=count,figures,workspace)
end

end
if abspath(PROGRAM_FILE)==@__FILE__
    all(a->a in ("--record","--main-path") || startswith(a,"--gallery="),ARGS) ||
        error("Usage: check_scalar_fields.jl [--record | --main-path] [--gallery=PATH]")
    galleries=filter(a->startswith(a,"--gallery="),ARGS);length(galleries)<=1 || error("Supply one gallery path")
    ScalarFieldsNotebook.check(;record="--record" in ARGS,main_path="--main-path" in ARGS,
        gallery=isempty(galleries) ? nothing : split(only(galleries),'=';limit=2)[2])
end
