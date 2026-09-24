# Optional packages may be installed, but core loading must not activate them.
@testset "A16 optional dependency boundaries" begin
    project = TamerOpTestRunner.TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))
    for name in ("CairoMakie", "WGLMakie", "CSV", "DelaunayTriangulation", "Distances", "KernelFunctions")
        @test !haskey(project["deps"], name)
        @test haskey(project["weakdeps"], name)
    end
    for name in ("Nemo", "Polyhedra", "CDDLib")
        @test haskey(project["deps"], name)
    end
    @test !isdefined(TamerOp, :_try_load_source_extension!)
    @test !isdefined(TamerOp.Visualization, :_try_load_visual_backend!)
    ingestion = TamerOp.DataIngestion
    spec = TamerOp.Options.FiltrationSpec(kind=:alpha, delaunay_backend=:auto)
    active = ingestion._have_pointcloud_delaunay_backend()
    @test ingestion._pointcloud_delaunay_backend(spec) === (active ? :fast : :naive)
    if !active
        err = try
            ingestion._pointcloud_delaunay_backend(TamerOp.Options.FiltrationSpec(kind=:alpha, delaunay_backend=:fast))
            nothing
        catch caught
            caught
        end
        @test err isa ArgumentError
        @test occursin("using DelaunayTriangulation", sprint(showerror, err))
        @test !ingestion._have_pointcloud_delaunay_backend()
        @test Base.get_extension(TamerOp, :TamerOpDelaunayTriangulationExt) === nothing
    end
    if !ingestion._have_pointcloud_nn_backend()
        err = try
            ingestion._pointcloud_nn_backend(TamerOp.Options.FiltrationSpec(kind=:rips, nn_backend=:nearestneighbors))
            nothing
        catch caught
            caught
        end
        @test err isa ArgumentError
        @test occursin("using NearestNeighbors", sprint(showerror, err))
        @test !ingestion._have_pointcloud_nn_backend()
    end
    # The strict minimal environment checks accessibility as well as activation.
    if get(ENV, "TAMEROP_TEST_MINIMAL", "false") == "true"
        for name in ("CairoMakie", "WGLMakie", "CSV", "KernelFunctions", "DelaunayTriangulation")
            @test Base.find_package(name) === nothing
        end
        @test isempty(TamerOp.Visualization._VISUAL_RENDERERS)
    end
end

# Package activation is part of correctness: source-included extension modules
# do not establish that users can load the declared package extensions.
@testset "A17 package extension activation" begin
    project = TamerOpTestRunner.TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))
    @test Base.PkgId(TamerOp).uuid !== nothing
    @test normpath(pathof(TamerOp)) == normpath(joinpath(@__DIR__, "..", "src", "TamerOp.jl"))
    for name in sort!(collect(keys(project["extensions"])))
        dependencies = project["extensions"][name]
        dependencies isa String && (dependencies = [dependencies])
        @testset "$name" begin
            if all(dependency -> Base.find_package(dependency) !== nothing, dependencies)
                for dependency in dependencies
                    Core.eval(@__MODULE__, Expr(:import, Expr(:., Symbol(dependency))))
                end
                @test Base.get_extension(TamerOp, Symbol(name)) !== nothing
                println("ACTIVE extension ", name)
            else
                missing = filter(dependency -> Base.find_package(dependency) === nothing, dependencies)
                println("SKIP extension ", name, ": unavailable in active environment: ", join(missing, ", "))
                @test_skip false
            end
        end
    end
end

@testset "A17 extension table and IO oracle" begin
    fs = TamerOp.Featurizers.FeatureSet([1.0 2.0; 3.0 5.0], [:x, :y], ["a", "b"], nothing)
    if Base.find_package("CSV") === nothing
        @test_skip false
    else
        @eval import CSV
        @test Base.get_extension(TamerOp, :TamerOpCSVExt) !== nothing
        mktempdir() do directory
            path = joinpath(directory, "features.csv")
            TamerOp.Featurizers.save_features(path, fs; format=:csv, mode=:wide)
            restored = TamerOp.Featurizers.load_features(path; format=:csv, mode=:wide)
            @test restored.X == [1.0 2.0; 3.0 5.0]
            @test restored.names == [:x, :y]
            @test restored.ids == ["a", "b"]
        end
    end
    if Base.find_package("Tables") === nothing
        @test_skip false
    else
        @eval import Tables
        @test Base.get_extension(TamerOp, :TamerOpTablesExt) !== nothing
        @test Tables.columntable(fs) == (id=["a", "b"], x=[1.0, 3.0], y=[2.0, 5.0])
    end
end

@testset "A16 DataFrames constructors use native Tables columns" begin
    if Base.find_package("DataFrames") === nothing || Base.find_package("Tables") === nothing
        @test_skip false
    else
        @eval import DataFrames, Tables
        FEA = TamerOp.Featurizers
        @test Base.get_extension(TamerOp, :TamerOpDataFramesExt) !== nothing
        @test Base.get_extension(TamerOp, :TamerOpTablesExt) !== nothing
        fs = FEA.FeatureSet([1 2; 3 5], [:x, :y], ["a", "b"], nothing)
        wide = (id=["a", "b"], x=[1, 3], y=[2, 5])
        long = (sample_index=[1, 1, 2, 2], id=["a", "a", "b", "b"],
                feature=[:x, :y, :x, :y], value=[1, 2, 3, 5])
        @test Tables.columntable(DataFrames.DataFrame(fs)) == wide
        @test Tables.columntable(DataFrames.DataFrame(fs; format=:long)) == long
        @test Tables.columntable(DataFrames.DataFrame(FEA.feature_table(fs))) == wide
        @test Tables.columntable(DataFrames.DataFrame(FEA.feature_table(fs; format=:long))) == long
        @test_throws ArgumentError DataFrames.DataFrame(fs; format=:invalid)

        # These wrappers previously recursed into the same DataFrame method.
        # Each now uses the generic Tables constructor, retaining exact grades.
        euler = FEA.euler_surface_table([7 8; 9 10]; axes=([0//1, 1//3], [2//1, 5//2]), id="e")
        @test Tables.columntable(DataFrames.DataFrame(euler)) ==
            (id=["e", "e", "e", "e"], x=[0//1, 0//1, 1//3, 1//3],
             y=[2//1, 5//2, 2//1, 5//2], value=[7, 8, 9, 10])
        image = TamerOp.SliceInvariants.PersistenceImage1D([0.0, 1.0], [2.0], [3.0 4.0])
        landscape = TamerOp.MultiparameterImages.MPLandscape(1, [0.0, 1.0],
            reshape([2.0, 3.0], 1, 1, 1, 2), ones(1, 1), [(1.0, 1.0)], [0.0])
        measure = TamerOp.SignedMeasures.PointSignedMeasure(([0//1, 1//3], [2//1]),
            [(1, 1), (2, 1)], [2, -3])
        for table in (FEA.persistence_image_table(image), FEA.mp_landscape_table(landscape),
                      FEA.point_signed_measure_table(measure))
            columns = Tables.columntable(DataFrames.DataFrame(table))
            @test columns == Tables.columntable(table)
            @test length(first(columns)) == 2
        end
    end
end

@testset "A16 signed point kernel independent Gaussian oracle" begin
    if Base.find_package("KernelFunctions") === nothing
        @test_skip false
    else
        @eval import KernelFunctions
        FEA = TamerOp.Featurizers
        @test Base.get_extension(TamerOp, :TamerOpKernelFunctionsExt) !== nothing
        # mu = 2*delta_0 - delta_1, nu = -delta_0. Expanding the signed
        # Gaussian sum by hand gives K(mu,mu)=5-4a, K(mu,nu)=-2+a,
        # K(nu,nu)=1, where a=exp(-1/(2*sigma^2)). Negative cross terms
        # ensure the adapter preserves signs rather than absolute masses.
        mu = SM.PointSignedMeasure(([0.0, 1.0],), [(1,), (2,)], [2, -1])
        nu = SM.PointSignedMeasure(([0.0, 1.0],), [(1,)], [-1])
        for sigma in (1.0, 2.0)
            a = exp(-1 / (2 * sigma^2))
            expected = [(5 - 4a) (-2 + a); (-2 + a) 1.0]
            kernel = FEA.point_signed_measure_kernel_object(; sigma=sigma)
            @test isapprox(KernelFunctions.kappa(kernel, mu, mu), expected[1, 1]; atol=1e-12, rtol=0)
            @test isapprox(KernelFunctions.kappa(kernel, mu, nu), expected[1, 2]; atol=1e-12, rtol=0)
            @test isapprox(KernelFunctions.kappa(kernel, nu, mu), expected[2, 1]; atol=1e-12, rtol=0)
            @test KernelFunctions.kappa(kernel, nu, nu) == 1.0
            @test isapprox(kernel(mu, nu), expected[1, 2]; atol=1e-12, rtol=0)
            @test isapprox(KernelFunctions.kernelmatrix(kernel, [mu, nu]), expected; atol=1e-12, rtol=0)
            @test isapprox(KernelFunctions.kernelmatrix(kernel, [mu, nu], [nu]), expected[:, 2:2]; atol=1e-12, rtol=0)
            @test isapprox(KernelFunctions.kernelmatrix_diag(kernel, [mu, nu]), [expected[1, 1], 1.0]; atol=1e-12, rtol=0)
        end
    end
end
