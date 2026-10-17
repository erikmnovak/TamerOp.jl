# Run with an environment containing TamerOp and CairoMakie.
# --core checks the two workflows that need only TamerOp.
import TamerOp

function check_agent_examples(args=ARGS)
    args in (String[], ["--core"]) || error("Usage: check_examples.jl [--core]")
    examples = joinpath(pkgdir(TamerOp), "agent", "examples")
    include(joinpath(examples, "data_to_encoding.jl"))
    include(joinpath(examples, "module_maps.jl"))
    Base.invokelatest(run_data_to_encoding_example)
    Base.invokelatest(run_module_maps_example)
    println("Checked connected components, merge ranks, commuting maps and vector images.")

    if isempty(args)
        include(joinpath(examples, "visualization.jl"))
        mktempdir() do output_dir
            Base.invokelatest(run_visualization_example; output_dir)
        end
        println("Checked the ring interval, SVG export and saved diagram round trip.")
    end
    return nothing
end

check_agent_examples()
