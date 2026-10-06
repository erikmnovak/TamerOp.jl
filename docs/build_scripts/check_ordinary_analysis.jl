#!/usr/bin/env julia
# Execute the canonical tutorial cells in order; optionally retain text outputs.
module OrdinaryAnalysisNotebook
using JSON3
const NOTEBOOK = normpath(joinpath(@__DIR__, "..", "tutorials", "ordinary_analysis.ipynb"))
function check(; record::Bool=false)
    notebook = JSON3.read(read(NOTEBOOK,String),Dict{String,Any})
    workspace = Module(:OrdinaryAnalysisTutorial)
    count = 0
    for (i,cell) in enumerate(notebook["cells"])
        cell["cell_type"] == "code" || continue
        count += 1
        value = Base.include_string(workspace,join(cell["source"]),"ordinary_analysis.ipynb:cell-$i")
        if record
            cell["execution_count"] = count
            cell["outputs"] = value === nothing ? Any[] : [Dict(
                "output_type"=>"execute_result", "execution_count"=>count,
                "metadata"=>Dict(),"data"=>Dict("text/plain"=>[Base.invokelatest(sprint,show,MIME"text/plain"(),value;context=:limit=>true)]))]
        end
    end
    if record
        open(NOTEBOOK,"w") do io
            JSON3.pretty(io,notebook)
            println(io)
        end
    end
    return count
end
end
if abspath(PROGRAM_FILE) == @__FILE__
    all(==("--record"),ARGS) || error("Usage: check_ordinary_analysis.jl [--record]")
    n = OrdinaryAnalysisNotebook.check(;record="--record" in ARGS)
    println("Ordinary analysis tutorial: $n code cells executed successfully.")
end
