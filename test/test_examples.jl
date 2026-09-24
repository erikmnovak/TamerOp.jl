# Public onboarding behavior is part of the package test suite. Tutorial files
# under examples/ are local assets and are exercised only when present.
@testset "Package onboarding computation" begin
    values = [0 0 0; 0 5 0; 0 0 0]
    diagram = TamerOp.cubical_persistence(values)
    @test TamerOp.finite_intervals(diagram; dim=1) == [(0, 5)]
    @test TamerOp.essential_births(diagram; dim=0) == [0]
    @test isempty(TamerOp.finite_intervals(diagram; dim=0))
    @test isempty(TamerOp.essential_births(diagram; dim=1))
end

@testset "Package onboarding figure export" begin
    if Base.find_package("CairoMakie") === nothing
        @test_skip Base.find_package("CairoMakie") !== nothing
    else
        @eval import CairoMakie
        @test Base.get_extension(TamerOp, :TamerOpCairoMakieExt) !== nothing
        mktempdir() do directory
            Base.invokelatest() do
                diagram = TamerOp.cubical_persistence([0 0 0; 0 5 0; 0 0 0])
                @test TamerOp.finite_intervals(diagram; dim=1) == [(0, 5)]
                png_path = TamerOp.save_visual(joinpath(directory, "ring_barcode.png"), diagram;
                                               kind=:barcode, dim=1, backend=:cairomakie)
                svg_path = TamerOp.save_visual(joinpath(directory, "ring_diagram.svg"), diagram;
                                               kind=:persistence_diagram, dim=1, backend=:cairomakie)
                png = read(png_path)
                @test png[1:8] == UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
                @test length(png) > 1000
                svg = read(svg_path, String)
                @test occursin("<svg", svg)
                @test occursin("<path", svg)
                @test sort(readdir(directory)) == ["ring_barcode.png", "ring_diagram.svg"]
            end
        end
    end
end

@testset "A15 installed-user getting-started script" begin
    path = joinpath(@__DIR__, "..", "examples", "getting_started.jl")
    if !isfile(path)
        @test_skip isfile(path)
    else
        target = Module(gensym(:GettingStartedExample))
        mktempdir() do directory
            withenv("TAMEROP_EXAMPLE_OUTPUT_ROOT" => nothing) do
                cd(directory) do
                    Base.include(target, path)
                end
            end
            Base.invokelatest() do
                @test getfield(target, :OP) === TamerOp
                diagram = getfield(target, :diagram)
                @test OP.finite_intervals(diagram; dim=1) == [(0, 5)]
                @test OP.essential_births(diagram; dim=0) == [0]
                @test isempty(OP.finite_intervals(diagram; dim=0))
                @test isempty(OP.essential_births(diagram; dim=1))
                @test isempty(readdir(directory))
            end
        end
    end
end

@testset "A15 installed-user first figure script" begin
    path = joinpath(@__DIR__, "..", "examples", "first_plot.jl")
    if !isfile(path)
        @test_skip isfile(path)
    elseif Base.find_package("CairoMakie") === nothing
        @test_skip false
    else
        target = Module(gensym(:FirstFigureExample))
        mktempdir() do directory
            withenv("TAMEROP_EXAMPLE_OUTPUT_ROOT" => nothing) do
                cd(directory) do
                    Base.include(target, path)
                end
            end
            Base.invokelatest() do
                @test getfield(target, :OP) === TamerOp
                @test OP.finite_intervals(getfield(target, :diagram); dim=1) == [(0, 5)]
                output_dir = joinpath(directory, "tamerop_outputs")
                @test getfield(target, :output_dir) == output_dir
                png = read(joinpath(output_dir, "ring_barcode.png"))
                @test png[1:8] == UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
                @test length(png) > 1000
                svg = read(joinpath(output_dir, "ring_diagram.svg"), String)
                @test occursin("<svg", svg)
                @test occursin("<path", svg)
                @test sort(readdir(output_dir)) == ["ring_barcode.png", "ring_diagram.svg"]
            end
        end
    end
end

@testset "A15 notebook imports outside the checkout" begin
    for name in ("00_quickstart_module_to_features.ipynb",
                 "01_graded_complex_by_hand.ipynb",
                 "02_point_cloud_bifiltration_rips_density.ipynb",
                 "03_image_bifiltration_distance_intensity.ipynb")
        path = joinpath(@__DIR__, "..", "examples", name)
        if !isfile(path)
            @test_skip isfile(path)
            continue
        end
        @eval import JSON3
        notebook = JSON3.read(read(path, String))
        cell = first(filter(c -> c.cell_type == "code", notebook.cells))
        target = Module(gensym(:InstalledNotebookStartup))
        Core.eval(target, :(include(path) = Base.include(@__MODULE__, path)))
        previous_project = Base.active_project()
        mktempdir() do directory
            withenv("TAMEROP_EXAMPLE_OUTPUT_ROOT" => nothing) do
                cd(directory) do
                    Base.include_string(target, join(cell.source), "$path:startup")
                end
            end
            Base.invokelatest() do
                @test getfield(target, :TO) === TamerOp
                @test Base.active_project() == previous_project
                outdir = getfield(target, :outdir)
                @test dirname(outdir) == joinpath(directory, "tamerop_outputs")
                @test isdir(outdir)
                if isdefined(target, :example_outdir)
                    alternate = joinpath(directory, "selected_outputs")
                    withenv("TAMEROP_EXAMPLE_OUTPUT_ROOT" => alternate) do
                        @test getfield(target, :example_outdir)("chosen") == joinpath(alternate, "chosen")
                    end
                    @test isdir(joinpath(alternate, "chosen"))
                end
            end
        end
    end
end

@testset "A17 executable script example" begin
    path = joinpath(@__DIR__, "..", "examples", "01_graded_complex_by_hand.jl")
    if !isfile(path)
        @test_skip isfile(path)
    else
        # Execute the actual onboarding script in isolation, with disposable output.
        target = Module(gensym(:ScriptExample))
        Core.eval(target, :(include(path) = Base.include(@__MODULE__, path)))
        mktempdir() do directory
            withenv("TAMEROP_EXAMPLE_OUTPUT_ROOT" => directory) do
                Base.include(target, path)
            end
            # include created new bindings and helper methods. Inspect them in the
            # latest world, as required by Julia 1.12's global-binding semantics.
            Base.invokelatest() do
                @test getfield(target, :TO) === TamerOp
                enc = getfield(target, :enc)
                @test TamerOp.encoding_module(enc).field == CM.F2()
                @test FF.nvertices(enc.P) == 20
                @test size(getfield(target, :fs).X) == (1, 400)
                @test all(>(0), CC.dimensions(enc))
                @test maximum(CC.dimensions(enc)) == 3
                @test minimum(CC.dimensions(enc)) == 1
                sample = only(getfield(target, :to_encoding_samples)([enc], ["line"]))
                @test sample.M isa MD.PModule
                @test sample.M.dims == CC.dimensions(enc)
                @test sample.pi === TamerOp.encoding_map(enc)
                @test (sample.label, sample.id) == ("line", "sample_001")
                @test isfile(getfield(target, :paths).csv_wide)
                @test isfile(getfield(target, :paths).csv_long)
                restored = SER.load_dataset_json(getfield(target, :g_path))
                @test restored.boundaries == getfield(target, :G).boundaries
                @test restored.grades == getfield(target, :G).grades
            end
        end
    end
end

@testset "A17 executable geometry notebook" begin
    path = joinpath(@__DIR__, "..", "examples", "14_geometric_bifiltrations.ipynb")
    if !isfile(path)
        @test_skip isfile(path)
    else
        @eval import JSON3
        target = Module(gensym(:NotebookExample))
        notebook = JSON3.read(read(path, String))
        executed = 0
        @eval import Pkg
        previous_project = Base.active_project()
        try
            cd(joinpath(@__DIR__, "..")) do
                for (index, cell) in enumerate(notebook.cells)
                    cell.cell_type == "code" || continue
                    Base.include_string(target, join(cell.source), "$path:cell$index")
                    executed += 1
                end
            end
        finally
            Base.active_project() == previous_project ||
                Pkg.activate(dirname(previous_project); io=devnull)
        end
        Base.invokelatest() do
            @test executed == 13
            @test getfield(target, :TamerOp) === TamerOp
            # The planar lower point enters after the three outer vertices. Their
            # function-Delaunay bifiltration includes all faces of a tetrahedron.
            planar = getfield(target, :planar_complex)
            @test length.(planar.cells_by_dim) == [4, 6, 4, 1]
            graph = getfield(target, :graph_complex)
            @test length.(graph.cells_by_dim) == [4, 4]
            components = getfield(target, :multicover_components)
            @test maximum(CC.dimensions(components)) == 2
            @test 1 in CC.dimensions(components)
        end
    end
end

@testset "A16 visualization notebook package startup" begin
    if Base.find_package("CairoMakie") === nothing
        @test_skip false
    else
        for name in ("10_visualization_engine_basics.ipynb",
                     "11_ingestion_visualization_pipeline.ipynb",
                     "12_graph_ingestion_visualization_pipeline.ipynb")
            if startswith(name, "11_") && Base.find_package("CSV") === nothing
                @test_skip false
                continue
            end
            path = joinpath(@__DIR__, "..", "examples", name)
            if !isfile(path)
                @test_skip isfile(path)
                continue
            end
            @eval import JSON3
            notebook = JSON3.read(read(path, String))
            cell = first(filter(c -> c.cell_type == "code", notebook.cells))
            target = Module(gensym(:VisualNotebookStartup))
            mktempdir() do directory
                withenv("TAMEROP_EXAMPLE_OUTPUT_ROOT" => nothing) do
                    cd(directory) do
                        Base.include_string(target, join(cell.source), "$path:startup")
                    end
                end
                Base.invokelatest() do
                    @test getfield(target, :TO) === TamerOp
                    @test getfield(target, :DISPLAY_VIS_BACKEND) === :cairomakie
                    @test getfield(target, :EXPORT_VIS_BACKEND) === :cairomakie
                    @test getfield(target, :OUTPUT_ROOT) == joinpath(directory, "tamerop_outputs")
                    @test Base.get_extension(TamerOp, :TamerOpCairoMakieExt) !== nothing
                end
            end
        end
    end
end
