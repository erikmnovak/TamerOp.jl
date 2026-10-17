# Run with an environment containing TamerOp and CairoMakie.
# --core checks the two workflows that need only TamerOp.
import TamerOp

ARGS in (String[], ["--core"]) || error("Usage: check_examples.jl [--core]")
agent_examples_dir = joinpath(pkgdir(TamerOp), "agent", "examples")
include(joinpath(agent_examples_dir, "data_to_encoding.jl"))
include(joinpath(agent_examples_dir, "module_maps.jl"))
run_data_to_encoding_example()
run_module_maps_example()
println("Checked connected components, merge ranks, commuting maps and vector images.")

if isempty(ARGS)
    include(joinpath(agent_examples_dir, "visualization.jl"))
end

if isempty(ARGS)
    mktempdir() do output_dir
        run_visualization_example(; output_dir)
    end
    println("Checked the ring interval, SVG export and saved diagram round trip.")
end
