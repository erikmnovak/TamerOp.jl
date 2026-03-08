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

include("test_finite_fringe.jl")
