# ---------------- ASCII-only source tree test -------------------------------

@testset "ASCII-only source tree" begin
    function jl_files_under(dir::AbstractString)
        files = String[]
        for (root, _, fs) in walkdir(dir)
            for f in fs
                endswith(f, ".jl") || continue
                push!(files, joinpath(root, f))
            end
        end
        sort!(files)
        return files
    end

    function first_nonascii_byte(path::AbstractString)
        data = read(path)
        for (i, b) in enumerate(data)
            if b > 0x7f
                return (i, b)
            end
        end
        return nothing
    end

    src_guess = _TO_SRC_DIR
    src_dir  = isdir(src_guess) ? src_guess : normpath(@__DIR__)
    test_dir = normpath(@__DIR__)

    for f in vcat(jl_files_under(src_dir), jl_files_under(test_dir))
        bad = first_nonascii_byte(f)
        if bad !== nothing
            pos, byte = bad
            @info "Non-ASCII byte detected" file=f pos=pos byte=byte
        end
        @test bad === nothing
    end
end

# ---------------- Export hygiene ----------------------------------------------

@testset "No submodule export blocks" begin
    src_dir = _TO_SRC_DIR

    function jl_files_under(dir::AbstractString)
        files = String[]
        for (root, _, fs) in walkdir(dir)
            for f in fs
                endswith(f, ".jl") || continue
                push!(files, normpath(joinpath(root, f)))
            end
        end
        sort!(files)
        return files
    end

    allowed_exports = Set([normpath(joinpath(src_dir, "TamerOp.jl"))])
    pat = r"(?m)^\s*export\b"

    for f in jl_files_under(src_dir)
        f in allowed_exports && continue
        has_export = occursin(pat, read(f, String))
        if has_export
            @info "Unexpected export statement outside TamerOp.jl" file=f
        end
        @test !has_export
    end
end

# ---------------- Public API smoke test --------------------------------------

@testset "Public API smoke test" begin
    # Finite-poset primitives
    @test isdefined(TOA, :FinitePoset)
    @test isdefined(TOA, :Upset)
    @test isdefined(TOA, :Downset)
    @test isdefined(TOA, :FringeModule)
    @test isdefined(TOA, :principal_upset)
    @test isdefined(TOA, :principal_downset)
    @test isdefined(TOA, :upset_from_generators)
    @test isdefined(TOA, :downset_from_generators)
    @test isdefined(TOA, :one_by_one_fringe)
    @test isdefined(TOA, :cover_edges)

    # Encoding-map layer
    @test isdefined(TOA, :EncodingMap)
    @test isdefined(TOA, :UptightEncoding)
    @test isdefined(TOA.Encoding, :build_uptight_encoding_from_fringe)
    @test isdefined(TOA.Encoding, :pullback_fringe_along_encoding)
    @test isdefined(TOA.Encoding, :pushforward_fringe_along_encoding)

    # JSON IO helpers
    @test isdefined(TOA, :parse_finite_fringe_json)
    @test isdefined(TOA, :finite_fringe_from_m2)
    @test isdefined(TOA, :save_flange_json)
    @test isdefined(TOA, :load_flange_json)
    @test isdefined(TOA, :parse_flange_json)
    @test isdefined(TOA, :save_pl_fringe_json)
    @test isdefined(TOA, :load_pl_fringe_json)
    @test isdefined(TOA, :parse_pl_fringe_json)
    @test isdefined(TOA.Serialization, :parse_finite_fringe_json)
    @test isdefined(TOA.Serialization, :finite_fringe_from_m2)
    @test isdefined(TOA.Serialization, :save_encoding_json)
    @test isdefined(TOA.Serialization, :load_encoding_json)
    @test isdefined(TOA.Serialization, :save_mpp_decomposition_json)
    @test isdefined(TOA.Serialization, :load_mpp_decomposition_json)
    @test isdefined(TOA.Serialization, :save_mpp_image_json)
    @test isdefined(TOA.Serialization, :load_mpp_image_json)
    @test isdefined(TOA, :save_dataset_json)
    @test isdefined(TOA, :load_dataset_json)
    @test isdefined(TOA, :save_pipeline_json)
    @test isdefined(TOA, :load_pipeline_json)

    # Data ingestion entrypoints
    @test isdefined(TOA, :encode)
    @test isdefined(TOA, :hom_dimension)
    @test !isdefined(TOA, :encode_from_data)
    @test !isdefined(TOA, :ingest)
    @test isdefined(TOA, :one_criticalify)
    @test isdefined(TOA, :criticality)
    @test isdefined(TOA, :normalize_multicritical)
    @test isdefined(TOA, :fringe_presentation)
    @test isdefined(TOA, :PipelineOptions)
    @test isdefined(TOA, :DataFileOptions)
    @test isdefined(TOA, :load_data)
    @test isdefined(TOA, :inspect_data_file)
    @test isdefined(TOA, :DataIngestion)
    @test isdefined(TOA, :DataFileIO)
    @test isdefined(TOA.DataIngestion, :AbstractFiltration)
    @test isdefined(TOA.DataIngestion, :RipsFiltration)
    @test isdefined(TOA.DataIngestion, :LandmarkRipsFiltration)
    @test isdefined(TOA.DataIngestion, :GraphLowerStarFiltration)
    @test isdefined(TOA.DataIngestion, :DelaunayLowerStarFiltration)
    @test isdefined(TOA.DataIngestion, :FunctionDelaunayFiltration)
    @test isdefined(TOA.DataIngestion, :CoreFiltration)
    @test isdefined(TOA.DataIngestion, :GraphCoreFiltration)
    @test isdefined(TOA.DataIngestion, :RhomboidFiltration)
    @test isdefined(TOA.DataIngestion, :to_filtration)
    @test isdefined(TOA.DataIngestion, :estimate_ingestion)
    @test isdefined(TOA, :IngestionPlan)
    @test isdefined(TOA, :IngestionEstimate)
    @test isdefined(TOA, :GradedComplexBuildResult)
    @test isdefined(TOA, :estimate_ingestion)
    @test isdefined(TOA, :plan_ingestion)
    @test isdefined(TOA, :run_ingestion)
    @test isdefined(TOA, :check_data_filtration)
    @test isdefined(TOA, :ingestion_plan_summary)
    @test isdefined(TOA, :OrdinaryPersistence)
    @test isdefined(TOA, :PersistenceDiagram)
    @test isdefined(TOA, :persistence_diagram)
    @test isdefined(TOA, :cubical_persistence)
    @test isdefined(TOA, :check_torus_persistence)

    # Indicator-resolution and module hot-path entrypoints.
    @test isdefined(TOA, :pmodule_from_fringe)
    @test isdefined(TOA, :projective_cover)
    @test isdefined(TOA, :injective_hull)
    @test isdefined(TOA, :upset_resolution)
    @test isdefined(TOA, :downset_resolution)
    @test isdefined(TOA, :indicator_resolutions)
    @test isdefined(TOA, :verify_upset_resolution)
    @test isdefined(TOA, :verify_downset_resolution)
    @test isdefined(TOA, :map_leq)
    @test isdefined(TOA, :map_leq_many)
    @test isdefined(TOA, :map_leq_many!)
    @test isdefined(TOA, :direct_sum_with_maps)

    # Core advanced options and deeper change-of-poset hooks.
    @test isdefined(TOA, :EncodingOptions)
    @test isdefined(TOA, :ResolutionOptions)
    @test isdefined(TOA, :InvariantOptions)
    @test isdefined(TOA, :DerivedFunctorOptions)
    @test isdefined(TOA, :left_kan_extension)
    @test isdefined(TOA, :right_kan_extension)
    @test isdefined(TOA, :derived_pushforward_left)
    @test isdefined(TOA, :derived_pushforward_right)

    # Resolution tables
    @test isdefined(TOA, :betti_table)
    @test isdefined(TOA, :bass_table)
end

@testset "API surface contracts" begin
    root_exports = Set(names(TamerOp; all=false, imported=false))
    adv_exports = Set(names(TamerOp.Advanced; all=false, imported=false))

    # Root exports are strictly the curated simple surface.
    for sym in TamerOp.SIMPLE_API
        @test sym in root_exports
    end
    for sym in TamerOp.ADVANCED_ONLY_API
        @test !(sym in root_exports)
    end

    # Advanced exports the full curated power-user superset.
    for sym in TamerOp.ADVANCED_API
        @test sym in adv_exports
    end
end

