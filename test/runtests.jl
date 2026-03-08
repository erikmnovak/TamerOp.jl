using Test
using Random
using LinearAlgebra
using SparseArrays

# -----------------------------------------------------------------------------
# Test bootstrap
#
# These tests can be run in two modes:
#
# 1) Package mode (recommended):
#    From the repo root:
#        julia --project=. -e 'import Pkg; Pkg.test()'
#
# 2) Script mode (convenient during development):
#        julia test/runtests.jl
#
# In script mode, `using PosetModules` fails unless the repo is on LOAD_PATH.
# So we fall back to a direct include of src/PosetModules.jl.
# -----------------------------------------------------------------------------

try
    using PosetModules
catch
    include(joinpath(@__DIR__, "..", "src", "PosetModules.jl"))
    using .PosetModules
end

# Keep API-surface contract checks explicit.
const PMA = PosetModules.Advanced

# Test-facing namespace:
# resolve symbols directly from owner modules to avoid coupling correctness tests
# to APISurface curation decisions.
struct _TestSurface end

const _TEST_SURFACE_MODULES = (
    PosetModules,
    PosetModules.CoreModules,
    PosetModules.Stats,
    PosetModules.Options,
    PosetModules.DataTypes,
    PosetModules.EncodingCore,
    PosetModules.Results,
    PosetModules.RegionGeometry,
    PosetModules.FiniteFringe,
    PosetModules.IndicatorTypes,
    PosetModules.Encoding,
    PosetModules.Modules,
    PosetModules.AbelianCategories,
    PosetModules.IndicatorResolutions,
    PosetModules.FlangeZn,
    PosetModules.ZnEncoding,
    PosetModules.PLPolyhedra,
    PosetModules.PLBackend,
    PosetModules.ChainComplexes,
    PosetModules.DerivedFunctors,
    PosetModules.ModuleComplexes,
    PosetModules.ChangeOfPosets,
    PosetModules.Serialization,
    PosetModules.DataFileIO,
    PosetModules.Invariants,
    PosetModules.Workflow,
    PosetModules.DataIngestion,
    PosetModules.Featurizers,
    PosetModules.Viz2D,
)

@inline function Base.getproperty(::_TestSurface, s::Symbol)
    for m in _TEST_SURFACE_MODULES
        if isdefined(m, s)
            return getfield(m, s)
        end
    end
    throw(UndefVarError(s))
end

const PM = _TestSurface()

# Convenient aliases used throughout the test suite.
const DF  = PosetModules.DerivedFunctors
const FF  = PosetModules.FiniteFringe
const EN  = PosetModules.Encoding
const HE  = DF.HomExtEngine
const MD  = PosetModules.Modules
const IR  = PosetModules.IndicatorResolutions
const FZ  = PosetModules.FlangeZn
const SER = PosetModules.Serialization
const PLP = PosetModules.PLPolyhedra
const PLB = PosetModules.PLBackend
const CC  = PosetModules.ChainComplexes
const OPT = PosetModules.Options
const DT  = PosetModules.DataTypes
const EC  = PosetModules.EncodingCore
const RES = PosetModules.Results
const QQ  = PosetModules.CoreModules.QQ
const CM  = PosetModules.CoreModules
const Inv = PosetModules.Invariants

using SparseArrays

# NOTE:
# PLBackend is now always loaded by src/PosetModules.jl, so the historical test-time
# "force include PLBackend.jl regardless of ENV toggles" hack is no longer needed.

# ---------------- Helpers used by multiple test files -------------------------

"""
    chain_poset(n::Integer; check::Bool=false) -> FF.FinitePoset

Return the chain poset on `n` elements labeled `1:n`, ordered by
`i <= j` iff `i <= j`.

Notes
-----
- This constructor is deterministic and the relation is known to be a valid
  partial order, so `check=false` is the default for speed.
- Set `check=true` if you want to force validation (useful when debugging).
- `n == 0` returns the empty poset.
"""
function chain_poset(n::Integer; check::Bool=false)::FF.FinitePoset
    n < 0 && throw(ArgumentError("chain_poset: n must be >= 0, got $n"))
    nn = Int(n)

    # BitMatrix with L[i,j] = true iff i <= j.
    L = falses(nn, nn)
    @inbounds for i in 1:nn
        for j in i:nn
            L[i, j] = true
        end
    end

    return FF.FinitePoset(L; check=check)
end

"Disjoint union of two chains of lengths m and n, with no relations across components."
function disjoint_two_chains_poset(m::Int, n::Int)
    (m >= 1 && n >= 1) || error("disjoint_two_chains_poset: need m >= 1 and n >= 1")
    N = m + n
    leq = falses(N, N)

    # First chain: 1 <= 2 <= ... <= m (store full transitive closure).
    for i in 1:m
        for j in i:m
            leq[i, j] = true
        end
    end

    # Second chain: (m+1) <= ... <= (m+n), again full transitive closure.
    off = m
    for i in 1:n
        for j in i:n
            leq[off + i, off + j] = true
        end
    end

    return FF.FinitePoset(leq)
end

"Default: two chains of length 2 (vertices {1,2} and {3,4})."
disjoint_two_chains_poset() = disjoint_two_chains_poset(2, 2)


"""
Diamond poset on {1,2,3,4} with relations

    1 < 2 < 4
    1 < 3 < 4

and with 2 incomparable to 3.

This is the smallest non-chain poset where "two different length-2 paths"
exist (1->2->4 and 1->3->4), which is exactly the situation where indicator
resolutions can have length > 1 and Ext^2 can be nonzero.
"""
function diamond_poset()
    leq = falses(4, 4)
    for i in 1:4
        leq[i, i] = true
    end
    leq[1, 2] = true
    leq[1, 3] = true
    leq[2, 4] = true
    leq[3, 4] = true
    leq[1, 4] = true  # transitive closure needed explicitly for FinitePoset
    return FF.FinitePoset(leq)
end

"""
Boolean lattice B3 on subsets of {1,2,3} ordered by inclusion.

Element numbering is by bitmask order:
1: {}
2: {1}
3: {2}
4: {3}
5: {1,2}
6: {1,3}
7: {2,3}
8: {1,2,3}
"""
function boolean_lattice_B3_poset()
    masks = Int[0, 1, 2, 4, 3, 5, 6, 7]
    n = length(masks)
    leq = falses(n, n)
    for i in 1:n
        mi = masks[i]
        for j in 1:n
            mj = masks[j]
            leq[i, j] = (mi & mj) == mi
        end
    end
    return FF.FinitePoset(leq)
end

"Convenience: 1x1 fringe module with scalar on the unique entry."
one_by_one_fringe(P::FF.AbstractPoset, U::FF.Upset, D::FF.Downset;
                  scalar=CM.QQ(1), field=CM.QQField()) =
    FF.one_by_one_fringe(P, U, D, scalar; field=field)

"Convenience: 1x1 fringe module with a specified scalar (positional)."
one_by_one_fringe(P::FF.AbstractPoset, U::FF.Upset, D::FF.Downset, scalar;
                  field=CM.QQField()) =
    FF.one_by_one_fringe(P, U, D, scalar; field=field)

"Simple modules on the chain 1 < 2: S1 supported at 1, S2 supported at 2."
function simple_modules_chain2()
    P = chain_poset(2)
    S1 = one_by_one_fringe(P, FF.principal_upset(P, 1), FF.principal_downset(P, 1))
    S2 = one_by_one_fringe(P, FF.principal_upset(P, 2), FF.principal_downset(P, 2))
    return P, S1, S2
end

# ---------------- Field test harness ----------------------------------------

const FIELD_QQ = CM.QQField()
const FIELD_F2 = CM.F2()
const FIELD_F3 = CM.F3()
const FIELD_F5 = CM.Fp(5)
const FIELD_R64 = CM.RealField(Float64; rtol=1e-10, atol=1e-12)

const FIELDS_FULL = (FIELD_QQ, FIELD_F2, FIELD_F3, FIELD_F5, FIELD_R64)

with_fields(fields, f::Function) = foreach(f, fields)
with_fields(f::Function, fields) = foreach(f, fields)

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

    src_guess = normpath(joinpath(@__DIR__, "..", "src"))
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
    src_dir = normpath(joinpath(@__DIR__, "..", "src"))

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

    allowed_exports = Set([normpath(joinpath(src_dir, "PosetModules.jl"))])
    pat = r"(?m)^\s*export\b"

    for f in jl_files_under(src_dir)
        f in allowed_exports && continue
        has_export = occursin(pat, read(f, String))
        if has_export
            @info "Unexpected export statement outside PosetModules.jl" file=f
        end
        @test !has_export
    end
end

# ---------------- Public API smoke test --------------------------------------

@testset "Public API smoke test" begin
    # Finite-poset primitives
    @test isdefined(PMA, :FinitePoset)
    @test isdefined(PMA, :Upset)
    @test isdefined(PMA, :Downset)
    @test isdefined(PMA, :FringeModule)
    @test isdefined(PMA, :principal_upset)
    @test isdefined(PMA, :principal_downset)
    @test isdefined(PMA, :upset_from_generators)
    @test isdefined(PMA, :downset_from_generators)
    @test isdefined(PMA, :one_by_one_fringe)
    @test isdefined(PMA, :cover_edges)

    # Encoding-map layer
    @test isdefined(PMA, :EncodingMap)
    @test isdefined(PMA, :UptightEncoding)
    @test isdefined(PMA.Encoding, :build_uptight_encoding_from_fringe)
    @test isdefined(PMA.Encoding, :pullback_fringe_along_encoding)
    @test isdefined(PMA.Encoding, :pushforward_fringe_along_encoding)

    # JSON IO helpers
    @test isdefined(PMA, :parse_finite_fringe_json)
    @test isdefined(PMA, :finite_fringe_from_m2)
    @test isdefined(PMA, :save_flange_json)
    @test isdefined(PMA, :load_flange_json)
    @test isdefined(PMA, :parse_flange_json)
    @test isdefined(PMA, :save_pl_fringe_json)
    @test isdefined(PMA, :load_pl_fringe_json)
    @test isdefined(PMA, :parse_pl_fringe_json)
    @test isdefined(PMA.Serialization, :parse_finite_fringe_json)
    @test isdefined(PMA.Serialization, :finite_fringe_from_m2)
    @test isdefined(PMA.Serialization, :save_encoding_json)
    @test isdefined(PMA.Serialization, :load_encoding_json)
    @test isdefined(PMA.Serialization, :save_mpp_decomposition_json)
    @test isdefined(PMA.Serialization, :load_mpp_decomposition_json)
    @test isdefined(PMA.Serialization, :save_mpp_image_json)
    @test isdefined(PMA.Serialization, :load_mpp_image_json)
    @test isdefined(PMA, :save_dataset_json)
    @test isdefined(PMA, :load_dataset_json)
    @test isdefined(PMA, :save_pipeline_json)
    @test isdefined(PMA, :load_pipeline_json)

    # Data ingestion entrypoints
    @test isdefined(PMA, :encode)
    @test isdefined(PMA, :hom_dimension)
    @test !isdefined(PMA, :encode_from_data)
    @test !isdefined(PMA, :ingest)
    @test isdefined(PMA, :one_criticalify)
    @test isdefined(PMA, :criticality)
    @test isdefined(PMA, :normalize_multicritical)
    @test isdefined(PMA, :fringe_presentation)
    @test isdefined(PMA, :PipelineOptions)
    @test isdefined(PMA, :DataFileOptions)
    @test isdefined(PMA, :load_data)
    @test isdefined(PMA, :inspect_data_file)
    @test isdefined(PMA, :DataIngestion)
    @test isdefined(PMA, :DataFileIO)
    @test isdefined(PMA.DataIngestion, :AbstractFiltration)
    @test isdefined(PMA.DataIngestion, :RipsFiltration)
    @test isdefined(PMA.DataIngestion, :LandmarkRipsFiltration)
    @test isdefined(PMA.DataIngestion, :GraphLowerStarFiltration)
    @test isdefined(PMA.DataIngestion, :DelaunayLowerStarFiltration)
    @test isdefined(PMA.DataIngestion, :FunctionDelaunayFiltration)
    @test isdefined(PMA.DataIngestion, :CoreFiltration)
    @test isdefined(PMA.DataIngestion, :RhomboidFiltration)
    @test isdefined(PMA.DataIngestion, :to_filtration)
    @test isdefined(PMA.DataIngestion, :estimate_ingestion)
    @test isdefined(PMA, :IngestionPlan)
    @test isdefined(PMA, :plan_ingestion)
    @test isdefined(PMA, :run_ingestion)

    # Indicator-resolution and module hot-path entrypoints.
    @test isdefined(PMA, :pmodule_from_fringe)
    @test isdefined(PMA, :projective_cover)
    @test isdefined(PMA, :injective_hull)
    @test isdefined(PMA, :upset_resolution)
    @test isdefined(PMA, :downset_resolution)
    @test isdefined(PMA, :indicator_resolutions)
    @test isdefined(PMA, :verify_upset_resolution)
    @test isdefined(PMA, :verify_downset_resolution)
    @test isdefined(PMA, :map_leq)
    @test isdefined(PMA, :map_leq_many)
    @test isdefined(PMA, :map_leq_many!)
    @test isdefined(PMA, :direct_sum_with_maps)

    # Core advanced options and deeper change-of-poset hooks.
    @test isdefined(PMA, :EncodingOptions)
    @test isdefined(PMA, :ResolutionOptions)
    @test isdefined(PMA, :InvariantOptions)
    @test isdefined(PMA, :DerivedFunctorOptions)
    @test isdefined(PMA, :left_kan_extension)
    @test isdefined(PMA, :right_kan_extension)
    @test isdefined(PMA, :derived_pushforward_left)
    @test isdefined(PMA, :derived_pushforward_right)

    # Resolution tables
    @test isdefined(PMA, :betti_table)
    @test isdefined(PMA, :bass_table)
end

@testset "API surface contracts" begin
    root_exports = Set(names(PosetModules; all=false, imported=false))
    adv_exports = Set(names(PosetModules.Advanced; all=false, imported=false))

    # Root exports are strictly the curated simple surface.
    for sym in PosetModules.SIMPLE_API
        @test sym in root_exports
    end
    for sym in PosetModules.ADVANCED_ONLY_API
        @test !(sym in root_exports)
    end

    # Advanced exports the full curated power-user superset.
    for sym in PosetModules.ADVANCED_API
        @test sym in adv_exports
    end
end

# ---------------- Run test files ---------------------------------------------
# Linear algebra engine
include("test_field_linalg.jl")

# Core functionality
include("test_finite_fringe.jl")
include("test_encoding.jl")
include("test_poset_interface.jl")

# Backends + geometry
include("test_pl_backend.jl")
include("test_zn_backend.jl")
include("test_geometry.jl")

# Data pipeline
include("test_data_pipeline.jl")

# Algebra
include("test_indicator_resolutions.jl")
include("test_derived_functors.jl")
include("test_model_independent_ext_layer.jl")
include("test_chain_complexes_homology.jl")
include("test_functoriality_ext_tor_maps.jl")

# Invariants
include("test_invariants.jl")
include("test_featurizers.jl")

# Stress tests last
include("test_random_stress.jl")
