import TamerOp as OP
import TamerOp.Advanced as OA
import CairoMakie

function run_visualization_example(; output_dir::AbstractString)
    mkpath(output_dir)
    values = [0 0 0;
              0 5 0;
              0 0 0]
    diagram = OP.cubical_persistence(values; field=OP.CoreModules.F2())
    intervals = OP.finite_intervals(diagram; dim=1)
    @assert intervals == [(0, 5)]

    specification = OA.visual_spec(diagram;
        kind=:barcode, dim=1, interval=1, window=(-1, 6))
    @assert OA.check_visual_spec(specification; throw=true).valid
    figure_path = joinpath(output_dir, "ring.svg")
    OP.save_visual(figure_path, specification; backend=:cairomakie)
    @assert isfile(figure_path) && occursin("<svg", read(figure_path, String))

    data_path = joinpath(output_dir, "ring.json")
    OP.save_persistence_diagram_json(data_path, diagram)
    reopened = OP.load_persistence_diagram_json(data_path)
    @assert OP.finite_intervals(reopened; dim=1) == intervals

    return (; diagram, intervals, specification, figure_path, data_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    output_dir = isempty(ARGS) ? joinpath(pwd(), "tamerop-figures") : only(ARGS)
    result = run_visualization_example(; output_dir)
    println("Hole interval: ", only(result.intervals))
    println("Figure: ", abspath(result.figure_path))
    println("Diagram data: ", abspath(result.data_path))
end
