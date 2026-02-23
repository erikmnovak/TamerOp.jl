module ModuleComplexes
using ..DerivedFunctors

function _resolution_offsets(res)
    return res.offsets
end

function _resolution_offsets(res::DerivedFunctors.Resolutions.InjectiveResolution)
    return _gens_offsets(res.gens)
end

function _gens_offsets(gens)
    offsets = Vector{Vector{Int}}(undef, length(gens))
    for i in eachindex(gens)
        gi = gens[i]
        if isempty(gi)
            offsets[i] = Int[]
            continue
        end
        if eltype(gi) <: AbstractVector
            lens = length.(gi)
            offs = Vector{Int}(undef, length(lens) + 1)
            offs[1] = 0
            for j in 1:length(lens)
                offs[j + 1] = offs[j] + lens[j]
            end
            offsets[i] = offs
        else
            offsets[i] = collect(0:length(gi))
        end
    end
    return offsets
end

using LinearAlgebra
using SparseArrays
import Base.Threads

using ..CoreModules: ResolutionOptions, _append_scaled_triplets!
using ..FieldLinAlg
using ..FiniteFringe
using ..FiniteFringe: AbstractPoset, FinitePoset, cover_edges, nvertices, upset_indices
using ..Modules: PModule, PMorphism, id_morphism,
                 zero_pmodule, zero_morphism,
                 direct_sum, direct_sum_with_maps,
                 map_leq

import ..IndicatorResolutions
import ..AbelianCategories
using ..AbelianCategories: kernel_with_inclusion, image_with_inclusion, _cokernel_module


import ..ChainComplexes:
    shift, extend_range, mapping_cone, mapping_cone_triangle,
    induced_map_on_cohomology, _map_at
using ..ChainComplexes:
    CochainComplex, DoubleComplex, CochainMap,
    total_complex, cohomology_data, spectral_sequence,
    cohomology_coordinates, cohomology_representative,
    solve_particular

using ..DerivedFunctors:
    HomSpace, Hom,
    HomSystemCache, hom_with_cache, precompose_matrix_cached, postcompose_matrix_cached,
    ProjectiveResolution, InjectiveResolution,
    injective_resolution, projective_resolution,
    lift_injective_chainmap,
    compose

# Internal Functoriality helpers live in DerivedFunctors.Functoriality.
using ..DerivedFunctors.Functoriality:
    _tensor_map_on_tor_chains_from_projective_coeff,
    _lift_pmodule_map_to_projective_resolution_chainmap_coeff

# Extend the DerivedFunctors.GradedSpaces interface for the "hyper" derived objects
# computed in this file (HyperExtSpace and HyperTorSpace).
import ..DerivedFunctors.GradedSpaces:
    degree_range, dim, basis, representative, coordinates, cycles, boundaries


import Base.Threads

const IR = IndicatorResolutions
const FF = FiniteFringe


"""
    DirectSumModuleMap{K}

Internal helper type used by certain lifting routines.

Some internal algorithms represent morphisms between direct sums of principal
downset modules by storing the induced linear maps on selected vertex bases.
This struct packages those vertexwise matrices in a concrete, documented way.

Fields
------
  - `comps::Vector{Matrix{K}}`:
      A list of matrices. The intended semantics of `comps[i]` depend on the
      caller; typically it is the matrix describing the map at the i-th selected
      base vertex.

Notes
-----
This type is *not* part of the public API and is not exported. It exists so that
method signatures that refer to `DirectSumModuleMap` are well-typed, and so that
internal callers can pass a small, allocation-free container.
"""
struct DirectSumModuleMap{K}
    comps::Vector{Matrix{K}}
end

# ============================================================
# Module cochain complexes + maps
# ============================================================

struct ModuleCochainComplex{K}
    tmin::Int
    tmax::Int
    terms::Vector{PModule{K}}      # length = tmax-tmin+1
    diffs::Vector{PMorphism{K}}    # length = tmax-tmin
end

"""
    ModuleCochainComplex(tmin, tmax, terms, diffs)

Internal positional constructor that normalizes abstract vector inputs into the
concrete storage used by `ModuleCochainComplex{K}`.
"""
function ModuleCochainComplex(
    tmin::Int,
    tmax::Int,
    terms::AbstractVector{<:PModule{K}},
    diffs::AbstractVector{<:PMorphism{K}},
) where {K}
    terms_vec = PModule{K}[]
    append!(terms_vec, terms)
    diffs_vec = PMorphism{K}[]
    append!(diffs_vec, diffs)
    return ModuleCochainComplex{K}(tmin, tmax, terms_vec, diffs_vec)
end

poset(C::ModuleCochainComplex) = C.terms[1].Q

# Max cohomological degree stored in a module cochain complex.
maxdeg_of_complex(C::ModuleCochainComplex) = C.tmax

@inline function _matrix_is_zero(M)
    # Fast exact check for "all entries are zero".
    for x in M
        if x != 0
            return false
        end
    end
    return true
end

@inline _field_of_complex(C::ModuleCochainComplex) = C.terms[1].field

@inline function _eye(::Type{K}, n::Int) where {K}
    M = zeros(K, n, n)
    for i in 1:n
        M[i, i] = one(K)
    end
    return M
end

# Internal structural equality for PModules used in validation.
# We avoid defining `==` globally. This is needed because zero modules are
# often constructed on the fly, so pointer-identity is too strict.
@inline function _pmodule_equal(M::PModule{K}, N::PModule{K}) where {K}
    M === N && return true
    M.Q === N.Q || return false
    M.dims == N.dims || return false
    # Compare cover-edge maps using store-aligned traversal (succs + maps_to_succ)
    # to avoid keyed lookups in the common validation path.
    storeM = M.edge_maps
    storeN = N.edge_maps
    succs = storeM.succs
    mapsM = storeM.maps_to_succ
    mapsN = storeN.maps_to_succ

    @inbounds for u in 1:nvertices(M.Q)
        Mu = mapsM[u]
        Nu = mapsN[u]
        for j in eachindex(succs[u])
            if Mu[j] != Nu[j]
                return false
            end
        end
    end
    return true
end

"""
    ModuleCochainComplex(terms, diffs; tmin=0, check=true)

Construct a bounded cochain complex of PModules

    C^tmin --d--> C^(tmin+1) --d--> ... --d--> C^tmax

with `terms[i] = C^(tmin + i - 1)` and `diffs[i] = d^(tmin + i - 1)`.

If `check=true`, we validate:
  * all terms live over the same poset;
  * each differential has the correct domain/codomain;
  * d^(t+1) circ d^t = 0 in every degree (fiberwise, vertex-by-vertex).
"""
function ModuleCochainComplex(
    terms::AbstractVector{<:PModule{K}},
    diffs::AbstractVector{<:PMorphism{K}};
    tmin::Int=0,
    check::Bool=true,
) where {K}
    @assert length(diffs) == length(terms) - 1
    tmax = tmin + length(terms) - 1

    if check
        Q = terms[1].Q
        for (i, T) in enumerate(terms)
            if T.Q !== Q
                error("ModuleCochainComplex: term $i lives over a different poset")
            end
        end

        # Check each differential's endpoints.
        for i in 1:length(diffs)
            d = diffs[i]
            if !_pmodule_equal(d.dom, terms[i]) || !_pmodule_equal(d.cod, terms[i+1])
                t = tmin + (i - 1)
                error("ModuleCochainComplex: differential at degree $t has wrong domain/codomain")
            end
        end

        # Check d^(t+1) * d^t = 0 fiberwise.
        Qn = nvertices(Q)
        for i in 1:(length(diffs) - 1)
            t = tmin + (i - 1)
            d1 = diffs[i]
            d2 = diffs[i+1]
            for u in 1:Qn
                prod = d2.comps[u] * d1.comps[u]
                if !_matrix_is_zero(prod)
                    error("ModuleCochainComplex: d^(t+1)*d^t != 0 at degree $t, vertex $u")
                end
            end
        end
    end

    return ModuleCochainComplex{K}(tmin, tmax, terms, diffs)
end

function ModuleCochainComplex(
    terms::AbstractVector{<:PModule},
    diffs::AbstractVector{<:PMorphism};
    tmin::Int=0,
    check::Bool=true,
)
    isempty(terms) && throw(ArgumentError("ModuleCochainComplex: terms must be nonempty"))
    K = typeof(terms[1]).parameters[1]
    return ModuleCochainComplex(
        Vector{PModule{K}}(terms),
        Vector{PMorphism{K}}(diffs);
        tmin = tmin,
        check = check,
    )
end




_term(C::ModuleCochainComplex{K}, t::Int) where {K} =
    (t < C.tmin || t > C.tmax) ? zero_pmodule(poset(C); field=_field_of_complex(C)) : C.terms[t - C.tmin + 1]

_diff(C::ModuleCochainComplex{K}, t::Int) where {K} =
    (t < C.tmin || t >= C.tmax) ? zero_morphism(_term(C,t), _term(C,t+1)) : C.diffs[t - C.tmin + 1]

"""
    ModuleCochainComplex(
        Hs::AbstractVector{<:FF.FringeModule},
        ds::AbstractVector{<:IR.PMorphism{K}};
        tmin::Integer = 0,
        check::Bool = true,
    )

Convenience constructor for building a cochain complex from finite fringe modules.

This is a mathematician-facing API feature: if you naturally specify terms using
"upset/downset + matrix" (finite fringe data), you should not have to manually
convert each term to an `IndicatorResolutions.PModule`.

Implementation details:
- Each `FringeModule` term is converted to a `PModule{K}` using
  `IndicatorResolutions.pmodule_from_fringe`.
- We then call the standard `ModuleCochainComplex(::Vector{PModule}, ::Vector{PMorphism}; ...)`
  constructor.
- The coefficient field is taken from each fringe module (and preserved by
  `pmodule_from_fringe`).

Parameters
- `Hs`: terms of the cochain complex as fringe modules.
- `ds`: differentials as `PMorphism{K}`. (Same convention as the PModule constructor.)
- `tmin`: cohomological starting degree (term `Hs[1]` is placed in degree `tmin`).
- `check`: if true, run structural checks (same meaning as in the PModule constructor).
"""
function ModuleCochainComplex(
    Hs::AbstractVector{<:FF.FringeModule},
    ds::AbstractVector{<:IR.PMorphism{K}};
    tmin::Integer = 0,
    check::Bool = true,
 ) where {K}
    # Convert each fringe module into a PModule{K}.
    Ms = Vector{IR.PModule{K}}(undef, length(Hs))
    for i in eachindex(Hs)
        Ms[i] = IR.pmodule_from_fringe(Hs[i])
    end

    # Delegate to the existing, fully-checked constructor.
    return ModuleCochainComplex(
        Ms,
        Vector{IR.PMorphism{K}}(ds);
        tmin = Int(tmin),
        check = check,
    )
end

function ModuleCochainComplex(
    Hs::AbstractVector{<:FF.FringeModule},
    ds::AbstractVector{<:IR.PMorphism};
    tmin::Integer = 0,
    check::Bool = true,
)
    isempty(Hs) && throw(ArgumentError("ModuleCochainComplex: Hs must be nonempty"))
    K = eltype(Hs[1].phi)
    Ms = Vector{IR.PModule{K}}(undef, length(Hs))
    for i in eachindex(Hs)
        Ms[i] = IR.pmodule_from_fringe(Hs[i])
    end
    return ModuleCochainComplex(
        Ms,
        Vector{IR.PMorphism{K}}(ds);
        tmin = Int(tmin),
        check = check,
    )
end

struct ModuleCochainMap{K}
    C::ModuleCochainComplex{K}
    D::ModuleCochainComplex{K}
    tmin::Int
    tmax::Int
    comps::Vector{PMorphism{K}}   # degreewise maps
end

"""
    ModuleCochainMap(C, D, comps; tmin, tmax, check=true)

A cochain map f : C -> D, i.e. degreewise morphisms

    f^t : C^t -> D^t

such that for every degree t:

    d_D^t circ f^t = f^(t+1) circ d_C^t

Outside the provided range `[tmin, tmax]`, the map is taken to be zero.

If `check=true`, we verify the chain-map equation in *all relevant degrees*,
including the boundary degrees where one side uses the implicit zero map.
"""
struct ModuleCochainMap{K}
    C::ModuleCochainComplex{K}
    D::ModuleCochainComplex{K}
    tmin::Int
    tmax::Int
    comps::Vector{PMorphism{K}}
end

_map(f::ModuleCochainMap{K}, t::Int) where {K} =
    (t < f.tmin || t > f.tmax) ?
        zero_morphism(_term(f.C, t), _term(f.D, t)) :
        f.comps[t - f.tmin + 1]

function ModuleCochainMap(
    C::ModuleCochainComplex{K},
    D::ModuleCochainComplex{K},
    comps::Vector{PMorphism{K}};
    tmin=nothing,
    tmax=nothing,
    check::Bool=true,
) where {K}
    tmin = isnothing(tmin) ? C.tmin : tmin
    tmax = isnothing(tmax) ? C.tmax : tmax
    @assert length(comps) == tmax - tmin + 1

    if check
        if poset(C) !== poset(D)
            error("ModuleCochainMap: domain and codomain complexes live over different posets")
        end

        # Degreewise domain/codomain checks (structural, not pointer-only).
        for t in tmin:tmax
            ft = comps[t - tmin + 1]
            if !_pmodule_equal(ft.dom, _term(C, t)) || !_pmodule_equal(ft.cod, _term(D, t))
                error("ModuleCochainMap: component f^$t has wrong domain/codomain")
            end
        end

        # Chain-map equation must also hold at boundary degrees:
        # we check t in [tmin-1, tmax].
        Q = poset(C)
        for t in (tmin - 1):tmax
            dD = _diff(D, t)
            dC = _diff(C, t)

            ft  = (t < tmin)  ? zero_morphism(_term(C, t), _term(D, t)) :
                                comps[t - tmin + 1]
            ftp = (t + 1 > tmax) ? zero_morphism(_term(C, t+1), _term(D, t+1)) :
                                  comps[t + 1 - tmin + 1]

            for u in 1:nvertices(Q)
                lhs = dD.comps[u] * ft.comps[u]
                rhs = ftp.comps[u] * dC.comps[u]
                if lhs != rhs
                    error("ModuleCochainMap: chain map equation fails at degree $t, vertex $u")
                end
            end
        end
    end

    return ModuleCochainMap{K}(C, D, tmin, tmax, comps)
end

function ModuleCochainMap(
    C::ModuleCochainComplex{K},
    D::ModuleCochainComplex{K},
    comps::AbstractVector{<:PMorphism};
    tmin=nothing,
    tmax=nothing,
    check::Bool=true,
) where {K}
    return ModuleCochainMap(
        C,
        D,
        Vector{PMorphism{K}}(comps);
        tmin = tmin,
        tmax = tmax,
        check = check,
    )
end

# Identity cochain map on a module complex.
function idmap(C::ModuleCochainComplex{K}) where {K}
    comps = [id_morphism(_term(C, t)) for t in C.tmin:C.tmax]
    return ModuleCochainMap(C, C, comps; tmin=C.tmin, tmax=C.tmax, check=true)
end


# ============================================================
# Cochain homotopies
# ============================================================

"""
    ModuleCochainHomotopy(f, g, hcomps; tmin, tmax, check=true)

A cochain homotopy between cochain maps f,g : C -> D.

The data is morphisms
    h^t : C^t -> D^(t-1)
such that for all t:

    f^t - g^t = d_D^(t-1) circ h^t + h^(t+1) circ d_C^t

Outside `[tmin,tmax]`, h^t is interpreted as 0.

If `check=true`, validation is performed fiberwise.
"""
struct ModuleCochainHomotopy{K}
    f::ModuleCochainMap{K}
    g::ModuleCochainMap{K}
    tmin::Int
    tmax::Int
    comps::Vector{PMorphism{K}}
end

_hcomp(H::ModuleCochainHomotopy{K}, t::Int) where {K} =
    (t < H.tmin || t > H.tmax) ?
        zero_morphism(_term(H.f.C, t), _term(H.f.D, t-1)) :
        H.comps[t - H.tmin + 1]

"""
    is_cochain_homotopy(H)

Return true iff H satisfies the cochain homotopy identity.
"""
function is_cochain_homotopy(H::ModuleCochainHomotopy{K}) where {K}
    f = H.f
    g = H.g
    C = f.C
    D = f.D

    Q = poset(C)
    tcheck_min = min(f.tmin, g.tmin, H.tmin) - 1
    tcheck_max = max(f.tmax, g.tmax, H.tmax)

    for t in tcheck_min:tcheck_max
        ft = _map(f, t)
        gt = _map(g, t)

        dD_prev = _diff(D, t-1)
        dC_t    = _diff(C, t)

        ht   = _hcomp(H, t)
        htp1 = _hcomp(H, t+1)

        for u in 1:nvertices(Q)
            left  = ft.comps[u] - gt.comps[u]
            right = dD_prev.comps[u] * ht.comps[u] + htp1.comps[u] * dC_t.comps[u]
            if left != right
                return false
            end
        end
    end

    return true
end

function ModuleCochainHomotopy(
    f::ModuleCochainMap{K},
    g::ModuleCochainMap{K},
    comps::Vector{PMorphism{K}};
    tmin=nothing,
    tmax=nothing,
    check::Bool=true,
) where {K}
    if f.C !== g.C || f.D !== g.D
        error("ModuleCochainHomotopy: maps must have the same domain and codomain complexes")
    end

    tmin = isnothing(tmin) ? f.C.tmin : tmin
    tmax = isnothing(tmax) ? f.C.tmax : tmax
    @assert length(comps) == tmax - tmin + 1

    if check
        for t in tmin:tmax
            ht = comps[t - tmin + 1]
            if !_pmodule_equal(ht.dom, _term(f.C, t)) || !_pmodule_equal(ht.cod, _term(f.D, t-1))
                error("ModuleCochainHomotopy: component h^$t has wrong domain/codomain")
            end
        end
    end

    H = ModuleCochainHomotopy{K}(f, g, tmin, tmax, comps)

    if check && !is_cochain_homotopy(H)
        error("ModuleCochainHomotopy: homotopy identity does not hold")
    end

    return H
end

function ModuleCochainHomotopy(
    f::ModuleCochainMap{K},
    g::ModuleCochainMap{K},
    comps::AbstractVector{<:PMorphism};
    tmin=nothing,
    tmax=nothing,
    check::Bool=true,
) where {K}
    return ModuleCochainHomotopy(
        f,
        g,
        Vector{PMorphism{K}}(comps);
        tmin = tmin,
        tmax = tmax,
        check = check,
    )
end



# ============================================================
# shift / extend_range for module complexes
# ============================================================

function shift(C::ModuleCochainComplex{K}, k::Int) where {K}
    if k == 0
        return C
    end
    diffs = C.diffs
    if isodd(k)
        diffs = [PMorphism{K}(d.dom, d.cod, [-M for M in d.comps]) for d in diffs]
    end
    return ModuleCochainComplex{K}(C.tmin + k, C.tmax + k, C.terms, diffs)
end

function extend_range(C::ModuleCochainComplex{K}, tmin::Int, tmax::Int) where {K}
    Q = poset(C)
    terms = PModule{K}[]
    diffs = PMorphism{K}[]
    for t in tmin:tmax
        push!(terms, _term(C,t))
    end
    for t in tmin:(tmax-1)
        push!(diffs, _diff(C,t))
    end
    return ModuleCochainComplex{K}(tmin,tmax,terms,diffs)
end

# ============================================================
# mapping cone for module cochain maps
# ============================================================

function mapping_cone(f::ModuleCochainMap{K}) where {K}
    C = f.C
    D = f.D
    tmin = min(D.tmin, C.tmin - 1)
    tmax = max(D.tmax, C.tmax - 1)

    terms = PModule{K}[]
    diffs = PMorphism{K}[]

    for t in tmin:tmax
        push!(terms, direct_sum(_term(D,t), _term(C,t+1)))
    end

    for t in tmin:(tmax-1)
        dom = terms[t - tmin + 1]
        cod = terms[t - tmin + 2]
        Dt  = _term(D,t); Dt1 = _term(D,t+1)
        Ct1 = _term(C,t+1); Ct2 = _term(C,t+2)

        dD  = _diff(D,t)
        dC  = _diff(C,t+1)
        ft1 = _map(f,t+1)

        comps = Vector{Matrix{K}}(undef, poset(C).n)
        for u in 1:poset(C).n
            a = Dt.dims[u]; b = Ct1.dims[u]
            c = Dt1.dims[u]; d = Ct2.dims[u]
            M = zeros(K, c+d, a+b)
            if c>0 && a>0; M[1:c, 1:a] .= dD.comps[u]; end
            if c>0 && b>0; M[1:c, a+1:a+b] .= ft1.comps[u]; end
            if d>0 && b>0; M[c+1:c+d, a+1:a+b] .= -dC.comps[u]; end
            comps[u] = M
        end
        push!(diffs, PMorphism{K}(dom,cod,comps))
    end

    return ModuleCochainComplex{K}(tmin,tmax,terms,diffs)
end

# Triangle object (optional but included)
struct ModuleDistinguishedTriangle{K}
    C::ModuleCochainComplex{K}
    D::ModuleCochainComplex{K}
    Cone::ModuleCochainComplex{K}
    f::ModuleCochainMap{K}
    i::ModuleCochainMap{K}
    p::ModuleCochainMap{K}
end

function mapping_cone_triangle(f::ModuleCochainMap{K}) where {K}
    C, D = f.C, f.D
    Cone = mapping_cone(f)
    # maps D -> Cone and Cone -> C[1]
    tmin = Cone.tmin
    tmax = Cone.tmax
    i_comps = PMorphism{K}[]
    p_comps = PMorphism{K}[]
    for t in tmin:tmax
        Dt = _term(D,t)
        Ct1 = _term(C,t+1)
        S = _term(Cone,t)
        # injection D^t -> D^t oplus C^{t+1}
        comps_i = Vector{Matrix{K}}(undef, poset(C).n)
        comps_p = Vector{Matrix{K}}(undef, poset(C).n)
        for u in 1:poset(C).n
            a = Dt.dims[u]; b = Ct1.dims[u]
            inj = zeros(K, a+b, a)
            proj = zeros(K, b, a+b)
            if a > 0
                inj[1:a, 1:a] .= _eye(K, a)
            end
            if b > 0
                proj[1:b, a+1:a+b] .= _eye(K, b)
            end
            comps_i[u] = inj
            comps_p[u] = proj
        end
        push!(i_comps, PMorphism{K}(Dt,S,comps_i))
        push!(p_comps, PMorphism{K}(S,Ct1,comps_p))
    end
    i = ModuleCochainMap(D,Cone,i_comps; tmin=tmin, tmax=tmax)
    p = ModuleCochainMap(Cone,shift(C,1),p_comps; tmin=tmin, tmax=tmax)
    return ModuleDistinguishedTriangle{K}(C,D,Cone,f,i,p)
end

# ============================================================
# Cohomology as PModules
# ============================================================

"""
    cohomology_module_data(C,t)

Return (Z,iZ,B,iB,j,H,q) where:
Z = ker d^t, B = im d^{t-1}, j: B->Z inclusion, H=Z/B, q:Z->H.
"""
function cohomology_module_data(C::ModuleCochainComplex{K}, t::Int) where {K}
    M  = _term(C,t)
    d0 = _diff(C,t-1)
    d1 = _diff(C,t)

    Z, iZ = kernel_with_inclusion(d1)
    B, iB = image_with_inclusion(d0)

    # j: B -> Z such that iZ circ j = iB
    Q = poset(C)
    jcomps = Vector{Matrix{K}}(undef, nvertices(Q))
    field = M.field
    for u in 1:nvertices(Q)
        if B.dims[u] == 0
            jcomps[u] = zeros(K, Z.dims[u], 0)
        elseif Z.dims[u] == 0
            jcomps[u] = zeros(K, 0, B.dims[u])
        else
            jcomps[u] = FieldLinAlg.solve_fullcolumn(field, iZ.comps[u], iB.comps[u]; check_rhs=false)
        end
    end
    j = PMorphism{K}(B,Z,jcomps)

    H, q = _cokernel_module(j)

    return (Z=Z, iZ=iZ, B=B, iB=iB, j=j, H=H, q=q)
end

cohomology_module(C::ModuleCochainComplex{K}, t::Int) where {K} = cohomology_module_data(C,t).H

# induced map on cohomology modules
function induced_map_on_cohomology_modules(f::ModuleCochainMap{K}, t::Int) where {K}
    field = _field_of_complex(f.C)
    Cd = cohomology_module_data(f.C,t)
    Dd = cohomology_module_data(f.D,t)

    ft = _map(f,t)
    # restrict to cycles: ZC -> D^t
    ZC_to_Dt = PMorphism{K}(Cd.Z, _term(f.D,t), [ft.comps[u] * Cd.iZ.comps[u] for u in 1:poset(f.C).n])

    # land in ZD by solving iZD * X = (ft*iZC)
    Q = poset(f.C)
    ZC_to_ZD = Vector{Matrix{K}}(undef, nvertices(Q))
    for u in 1:nvertices(Q)
        if Dd.Z.dims[u] == 0
            ZC_to_ZD[u] = zeros(K, 0, Cd.Z.dims[u])
        else
            ZC_to_ZD[u] = FieldLinAlg.solve_fullcolumn(field, Dd.iZ.comps[u], ZC_to_Dt.comps[u])
        end
    end
    hZ = PMorphism{K}(Cd.Z, Dd.Z, ZC_to_ZD)

    # Want H map: HD circ qC = qD circ hZ
    compsH = Vector{Matrix{K}}(undef, nvertices(Q))
    for u in 1:nvertices(Q)
        RHS = Dd.q.comps[u] * hZ.comps[u]
        qC = Cd.q.comps[u]
        if size(qC,1) == 0
            compsH[u] = zeros(K, size(RHS,1), 0)
        else
            # qC has full row rank; pick right inverse via (qC*qC')^{-1}
            A = qC * transpose(qC)
            invA = FieldLinAlg.solve_fullcolumn(field, A, _eye(K, size(A, 1)))
            rinv = transpose(qC) * invA
            compsH[u] = RHS * rinv
        end
    end
    return PMorphism{K}(Cd.H, Dd.H, compsH)
end

# quasi-isomorphism check
function is_isomorphism(f::PMorphism{K}) where {K}
    Q = f.dom.Q
    field = f.dom.field
    for u in 1:nvertices(Q)
        if f.dom.dims[u] != f.cod.dims[u]
            return false
        end
        if FieldLinAlg.rank(field, f.comps[u]) != f.dom.dims[u]
            return false
        end
    end
    return true
end

function is_quasi_isomorphism(f::ModuleCochainMap{K}) where {K}
    tmin = min(f.C.tmin, f.D.tmin)
    tmax = max(f.C.tmax, f.D.tmax)
    for t in tmin:tmax
        hf = induced_map_on_cohomology_modules(f,t)
        if !is_isomorphism(hf)
            return false
        end
    end
    return true
end

# ============================================================
# RHom and derived tensor (bicomplex + total complex)
# ============================================================

struct RHomComplex{K}
    C::ModuleCochainComplex{K}
    N::PModule{K}
    resN
    homs::Array{HomSpace{K},2}
    DC::DoubleComplex{K}
    tot::CochainComplex{K}
end


function RHomComplex(
    C::ModuleCochainComplex{K},
    N::PModule{K};
    maxlen::Int = 3,
    resN = nothing,
    cache::Union{Nothing,HomSystemCache}=nothing,
    threads::Bool = (Threads.nthreads() > 1),
) where {K}
    Q = N.Q
    maxlen = maxlen
    maxdeg = maxdeg_of_complex(C)
    resN = (resN === nothing) ? injective_resolution(N, ResolutionOptions(maxlen=maxlen)) : resN

    na, nb = maxdeg + 1, maxlen + 1
    homs = Array{HomSpace{K}}(undef, na, nb)
    dims = zeros(Int, na, nb)

    # Build Hom blocks (expensive) in parallel if requested.
    if threads && Threads.nthreads() > 1
        nT = Threads.nthreads()
        Threads.@threads for slot in 1:nT
            for idx in slot:nT:(na * nb)
                ia = div((idx - 1), nb) + 1
                ib = (idx - 1) % nb + 1
                p = ia - 1
                q = ib - 1

                Cp = _term(C, p)
                Eb = resN.Emods[q + 1]

                h = hom_with_cache(Cp, Eb; cache=cache)
                homs[ia, ib] = h
                dims[ia, ib] = dim(h)
            end
        end
    else
        for ia in 1:na, ib in 1:nb
            p = ia - 1
            q = ib - 1
            Cp = _term(C, p)
            Eb = resN.Emods[q + 1]
            homs[ia, ib] = hom_with_cache(Cp, Eb; cache=cache)
            dims[ia, ib] = dim(homs[ia, ib])
        end
    end

    # Vertical (C direction) differentials.
    dv = Array{SparseMatrixCSC{K, Int}}(undef, na, nb)
    if threads && Threads.nthreads() > 1
        nT = Threads.nthreads()
        Threads.@threads for slot in 1:nT
            for idx in slot:nT:(na * nb)
                ia = div((idx - 1), nb) + 1
                ib = (idx - 1) % nb + 1
                if ia == na
                    dv[ia, ib] = spzeros(K, dims[ia, ib], 0)
                else
                    p = ia - 1
                    dv[ia, ib] = precompose_matrix_cached(homs[ia, ib], homs[ia + 1, ib], _diff(C, p); cache=cache)
                end
            end
        end
    else
        for ia in 1:na, ib in 1:nb
            if ia == na
                dv[ia, ib] = spzeros(K, dims[ia, ib], 0)
            else
                p = ia - 1
                dv[ia, ib] = precompose_matrix_cached(homs[ia, ib], homs[ia + 1, ib], _diff(C, p); cache=cache)
            end
        end
    end

    # Horizontal (resolution direction) differentials.
    dh = Array{SparseMatrixCSC{K, Int}}(undef, na, nb)
    if threads && Threads.nthreads() > 1
        nT = Threads.nthreads()
        Threads.@threads for slot in 1:nT
            for idx in slot:nT:(na * nb)
                ia = div((idx - 1), nb) + 1
                ib = (idx - 1) % nb + 1
                if ib == nb
                    dh[ia, ib] = spzeros(K, dims[ia, ib], 0)
                else
                    q = ib - 1
                    dh[ia, ib] = postcompose_matrix_cached(homs[ia, ib + 1], homs[ia, ib], resN.d_mor[q + 1]; cache=cache)
                end
            end
        end
    else
        for ia in 1:na, ib in 1:nb
            if ib == nb
                dh[ia, ib] = spzeros(K, dims[ia, ib], 0)
            else
                q = ib - 1
                dh[ia, ib] = postcompose_matrix_cached(homs[ia, ib + 1], homs[ia, ib], resN.d_mor[q + 1]; cache=cache)
            end
        end
    end

    # Index convention: a = cochain degree in C (0..maxdeg), b = injective degree in resN (0..maxlen).
    DC = DoubleComplex{K}(0, maxdeg, 0, maxlen, dims, dv, dh)
    tot = total_complex(DC)
    return RHomComplex{K}(C, N, resN, homs, DC, tot)
end

function RHomComplex(
    C::ModuleCochainComplex{K},
    H::FF.FringeModule{K};
    kwargs...
) where {K}
    return RHomComplex(C, IndicatorResolutions.pmodule_from_fringe(H); kwargs...)
end


RHom(C::ModuleCochainComplex{K}, N::PModule{K}; kwargs...) where {K} = RHomComplex(C,N; kwargs...).tot
RHom(C::ModuleCochainComplex{K}, H::FF.FringeModule{K}; kwargs...) where {K} =
    RHom(C, IndicatorResolutions.pmodule_from_fringe(H); kwargs...)


# ------------------------------------------------------------
# Functoriality: induced maps on RHom(-, N)
# ------------------------------------------------------------

@inline function _cc_dim_at(C::CochainComplex, t::Int)
    if t < C.tmin || t > C.tmax
        return 0
    end
    return C.dims[t - C.tmin + 1]
end

function _tot_block_offsets(DC::DoubleComplex, t::Int)
    d = Dict{Tuple{Int,Int},Int}()
    off = 1
    for a in DC.amin:DC.amax
        b = t - a
        if DC.bmin <= b <= DC.bmax
            ai = a - DC.amin + 1
            bi = b - DC.bmin + 1
            d[(a,b)] = off
            off += DC.dims[ai,bi]
        end
    end
    return d
end

"""
    rhom_map_first(f, Rdom, Rcod; check=true)

Induced map on derived Hom complexes in the first variable.

If f : C -> D is a cochain map, then RHom(-,N) is contravariant, so we obtain:

    RHom(D,N) -> RHom(C,N)

`Rdom` must be `RHomComplex(D,N)` and `Rcod` must be `RHomComplex(C,N)`.

Strict functoriality (exact equality of matrices under composition) requires
that both RHom complexes were built using the same injective resolution
object (`Rdom.resN === Rcod.resN`).
"""
function rhom_map_first(
    f::ModuleCochainMap{K},
    Rdom::RHomComplex{K},
    Rcod::RHomComplex{K};
    check::Bool=true,
    cache::Union{Nothing,HomSystemCache}=nothing,
    threads::Bool = (Threads.nthreads() > 1),
) where {K}
    if f.C !== Rcod.C || f.D !== Rdom.C
        error("rhom_map_first: RHom complexes do not match the given map f : C -> D")
    end
    if Rdom.N !== Rcod.N
        error("rhom_map_first: codomain module N differs")
    end
    if Rdom.resN !== Rcod.resN
        error("rhom_map_first: strict functoriality requires the same injective resolution object")
    end

    tot_src = Rdom.tot
    tot_tgt = Rcod.tot

    tmin = min(tot_src.tmin, tot_tgt.tmin)
    tmax = max(tot_src.tmax, tot_tgt.tmax)

    maps = Vector{SparseMatrixCSC{K,Int}}(undef, tmax - tmin + 1)

    if threads && Threads.nthreads() > 1 && (tmax >= tmin)
        Threads.@threads for idx in 1:(tmax - tmin + 1)
            t = tmin + idx - 1
            dim_src = _cc_dim_at(tot_src, t)
            dim_tgt = _cc_dim_at(tot_tgt, t)

            if dim_src == 0 || dim_tgt == 0
                maps[idx] = spzeros(K, dim_tgt, dim_src)
                continue
            end

            off_src = _tot_block_offsets(Rdom.DC, t)
            off_tgt = _tot_block_offsets(Rcod.DC, t)

            I = Int[]
            J = Int[]
            V = K[]

            for ((a,b), col_off) in off_src
                if !haskey(off_tgt, (a,b))
                    continue
                end
                row_off = off_tgt[(a,b)]

                ai_src = a - Rdom.DC.amin + 1
                bi_src = b - Rdom.DC.bmin + 1
                ai_tgt = a - Rcod.DC.amin + 1
                bi_tgt = b - Rcod.DC.bmin + 1

                if ai_src < 1 || ai_src > size(Rdom.DC.dims,1) || bi_src < 1 || bi_src > size(Rdom.DC.dims,2)
                    continue
                end
                if ai_tgt < 1 || ai_tgt > size(Rcod.DC.dims,1) || bi_tgt < 1 || bi_tgt > size(Rcod.DC.dims,2)
                    continue
                end

                dim_block_src = Rdom.DC.dims[ai_src, bi_src]
                dim_block_tgt = Rcod.DC.dims[ai_tgt, bi_tgt]
                if dim_block_src == 0 || dim_block_tgt == 0
                    continue
                end

                # In RHom, the first index is the cochain degree a, so use p = a.
                p = a
                fp = _map(f, p)

                Hdom = Rdom.homs[ai_src, bi_src]
                Hcod = Rcod.homs[ai_tgt, bi_tgt]
                F = precompose_matrix_cached(Hdom, Hcod, fp; cache=cache)

                # Avoid allocating sparse(F) just to call findnz; append triplets directly.
                _append_scaled_triplets!(I, J, V, F, row_off - 1, col_off - 1)
            end

            maps[idx] = sparse(I, J, V, dim_tgt, dim_src)
        end
    else
        for t in tmin:tmax
            dim_src = _cc_dim_at(tot_src, t)
            dim_tgt = _cc_dim_at(tot_tgt, t)

            if dim_src == 0 || dim_tgt == 0
                maps[t - tmin + 1] = spzeros(K, dim_tgt, dim_src)
                continue
            end

            off_src = _tot_block_offsets(Rdom.DC, t)
            off_tgt = _tot_block_offsets(Rcod.DC, t)

            I = Int[]
            J = Int[]
            V = K[]

            for ((a,b), col_off) in off_src
                if !haskey(off_tgt, (a,b))
                    continue
                end
                row_off = off_tgt[(a,b)]

                ai_src = a - Rdom.DC.amin + 1
                bi_src = b - Rdom.DC.bmin + 1
                ai_tgt = a - Rcod.DC.amin + 1
                bi_tgt = b - Rcod.DC.bmin + 1

                if ai_src < 1 || ai_src > size(Rdom.DC.dims,1) || bi_src < 1 || bi_src > size(Rdom.DC.dims,2)
                    continue
                end
                if ai_tgt < 1 || ai_tgt > size(Rcod.DC.dims,1) || bi_tgt < 1 || bi_tgt > size(Rcod.DC.dims,2)
                    continue
                end

                dim_block_src = Rdom.DC.dims[ai_src, bi_src]
                dim_block_tgt = Rcod.DC.dims[ai_tgt, bi_tgt]
                if dim_block_src == 0 || dim_block_tgt == 0
                    continue
                end

                # In RHom, the first index is the cochain degree a, so use p = a.
                p = a
                fp = _map(f, p)

                Hdom = Rdom.homs[ai_src, bi_src]
                Hcod = Rcod.homs[ai_tgt, bi_tgt]
                F = precompose_matrix_cached(Hdom, Hcod, fp; cache=cache)

                # Avoid allocating sparse(F) just to call findnz; append triplets directly.
                _append_scaled_triplets!(I, J, V, F, row_off - 1, col_off - 1)
            end

            maps[t - tmin + 1] = sparse(I, J, V, dim_tgt, dim_src)
        end
    end

    return CochainMap(tot_src, tot_tgt, maps; tmin=tmin, tmax=tmax, check=check)
end

"""
    rhom_map_first(f, N; maxlen=3, resN=nothing, check=true)

Convenience wrapper: build RHom complexes using a shared injective resolution,
then return the induced map on totals.
"""
function rhom_map_first(
    f::ModuleCochainMap{K},
    N::PModule{K};
    maxlen::Int=3,
    resN=nothing,
    check::Bool=true,
    cache::Union{Nothing,HomSystemCache}=nothing,
    threads::Bool = (Threads.nthreads() > 1),
) where {K}
    resN = isnothing(resN) ? injective_resolution(N, ResolutionOptions(maxlen=maxlen)) : resN
    Rdom = RHomComplex(f.D, N; maxlen=maxlen, resN=resN, cache=cache, threads=threads)
    Rcod = RHomComplex(f.C, N; maxlen=maxlen, resN=resN, cache=cache, threads=threads)
    return rhom_map_first(f, Rdom, Rcod; check=check, cache=cache, threads=threads)
end

function rhom_map_first(
    f::ModuleCochainMap{K},
    H::FF.FringeModule{K};
    maxlen::Int=3,
    resN=nothing,
    check::Bool=true,
    cache::Union{Nothing,HomSystemCache}=nothing,
    threads::Bool = (Threads.nthreads() > 1),
) where {K}
    return rhom_map_first(f, IndicatorResolutions.pmodule_from_fringe(H);
        maxlen=maxlen, resN=resN, check=check, cache=cache, threads=threads)
end


################################################################################
# Internal utilities for a canonical (deterministic) chain map between injective
# resolutions. This is the key ingredient for rhom_map_second (covariant in N).
################################################################################

"""
    _bases_from_injective_gens(gens) -> Vector{Int}

Internal helper used by injective chain-map lifting.

In the current codebase, `DerivedFunctors.InjectiveResolution.gens[b+1]` is already a
flat `Vector{Int}` containing the *base vertices* (one per principal downset summand,
with repetition).

Older versions stored injective generators in a "gens_at" format:
a vector indexed by vertices, where `gens_at[u]` was a list of generators based at `u`.

This function accepts both layouts and returns the canonical flat list of base vertices.
"""
function _bases_from_injective_gens(gens)::Vector{Int}
    isempty(gens) && return Int[]

    # New format: already a flat list of base vertices.
    if gens isa Vector{Int}
        return gens
    end
    if eltype(gens) <: Integer
        return [Int(x) for x in gens]
    end

    # Older format: gens[u] is iterable, with one entry per generator based at u.
    bases = Int[]
    for u in 1:length(gens)
        for _ in gens[u]
            push!(bases, u)
        end
    end
    return bases
end

# Active indices at vertex i for a downset direct sum determined by bases.
# A generator based at u is active at i iff i <= u in the poset.
# Returned lists are in increasing global generator index order.
function _active_indices_from_bases(Q::AbstractPoset, bases::Vector{Int})
    n = nvertices(Q)
    by_base = [Int[] for _ in 1:n]
    for (j, b) in enumerate(bases)
        push!(by_base[b], j)
    end
    active = Vector{Vector{Int}}(undef, n)
    for i in 1:n
        idx = Int[]
        for b in upset_indices(Q, i)
            isempty(by_base[b]) && continue
            append!(idx, by_base[b])
        end
        active[i] = idx
    end
    return active
end

# Given coefficient matrix C (ncod x ndom) describing a morphism between direct
# sums of principal downsets, build the corresponding PMorphism.
function _pmorphism_from_downset_coeff(
    E::PModule{K},
    Ep::PModule{K},
    act_dom::Vector{Vector{Int}},
    act_cod::Vector{Vector{Int}},
    C::Matrix{K}
) where {K}
    comps = Vector{Matrix{K}}(undef, nvertices(E.Q))
    for i in 1:nvertices(E.Q)
        comps[i] = C[act_cod[i], act_dom[i]]
    end
    return PMorphism{K}(E, Ep, comps)
end

# NOTE: The authoritative implementation is DerivedFunctors.lift_injective_chainmap.

"""
    rhom_map_second(g, Rsrc, Rtgt; check=true)

Covariant functoriality of RHom in the second argument.

Given:
  * `g : N -> Np` a module morphism
  * `Rsrc = RHomComplex(C, N)`
  * `Rtgt = RHomComplex(C, Np)`

returns a cochain map of total complexes:

    RHom(C, N)  ->  RHom(C, Np)

This is implemented by canonically lifting `g` to a chain map between the
injective resolutions used inside `Rsrc` and `Rtgt`, then postcomposing on
each `Hom(C^p, E^b)` block. The result is assembled as a sparse block matrix.

The lift is deterministic (via solve_particular), so repeated calls are stable.
"""
function rhom_map_second(
    g::PMorphism{K},
    Rsrc::RHomComplex{K},
    Rtgt::RHomComplex{K};
    check::Bool = true,
    cache::Union{Nothing,HomSystemCache}=nothing,
    threads::Bool = (Threads.nthreads() > 1),
) where {K}
    @assert Rsrc.C === Rtgt.C
    @assert g.dom === Rsrc.N
    @assert g.cod === Rtgt.N
    @assert Rsrc.tot.tmin == Rtgt.tot.tmin
    @assert Rsrc.tot.tmax == Rtgt.tot.tmax
    @assert length(Rsrc.resN.d_mor) == length(Rtgt.resN.d_mor)

    # Canonical lift between injective resolutions.
    phis = lift_injective_chainmap(g, Rsrc.resN, Rtgt.resN; check=check)

    maps = Vector{SparseMatrixCSC{K, Int}}(undef, Rsrc.tot.tmax - Rsrc.tot.tmin + 1)

    if threads && Threads.nthreads() > 1 && (Rsrc.tot.tmax >= Rsrc.tot.tmin)
        Threads.@threads for idx in 1:(Rsrc.tot.tmax - Rsrc.tot.tmin + 1)
            t = Rsrc.tot.tmin + idx - 1
            offsets_src = _tot_block_offsets(Rsrc.DC, t)
            offsets_tgt = _tot_block_offsets(Rtgt.DC, t)

            dim_src = Rsrc.tot.dims[idx]
            dim_tgt = Rtgt.tot.dims[idx]

            I = Int[]
            J = Int[]
            V = K[]

            # Blocks are indexed by (A,B). For RHom, B is injective resolution degree.
            for ((A, B), off_tgt) in offsets_tgt
                if !haskey(offsets_src, (A, B))
                    continue
                end
                off_src = offsets_src[(A, B)]

                ia_src = A - Rsrc.DC.amin + 1
                ib_src = B - Rsrc.DC.bmin + 1
                ia_tgt = A - Rtgt.DC.amin + 1
                ib_tgt = B - Rtgt.DC.bmin + 1

                if ia_src < 1 || ia_src > size(Rsrc.DC.dims, 1) || ib_src < 1 || ib_src > size(Rsrc.DC.dims, 2)
                    continue
                end
                if ia_tgt < 1 || ia_tgt > size(Rtgt.DC.dims, 1) || ib_tgt < 1 || ib_tgt > size(Rtgt.DC.dims, 2)
                    continue
                end
                if Rsrc.DC.dims[ia_src, ib_src] == 0 || Rtgt.DC.dims[ia_tgt, ib_tgt] == 0
                    continue
                end

                # Postcompose on Hom(C^p, E^B).
                Mb = postcompose_matrix_cached(Rtgt.homs[ia_tgt, ib_tgt], Rsrc.homs[ia_src, ib_src], phis[B + 1]; cache=cache)
                _append_scaled_triplets!(I, J, V, Mb, off_tgt - 1, off_src - 1)
            end

            maps[idx] = sparse(I, J, V, dim_tgt, dim_src)
        end
    else
        for t in Rsrc.tot.tmin:Rsrc.tot.tmax
            offsets_src = _tot_block_offsets(Rsrc.DC, t)
            offsets_tgt = _tot_block_offsets(Rtgt.DC, t)

            dim_src = Rsrc.tot.dims[t - Rsrc.tot.tmin + 1]
            dim_tgt = Rtgt.tot.dims[t - Rtgt.tot.tmin + 1]

            I = Int[]
            J = Int[]
            V = K[]

            # Blocks are indexed by (A,B). For RHom, B is injective resolution degree.
            for ((A, B), off_tgt) in offsets_tgt
                if !haskey(offsets_src, (A, B))
                    continue
                end
                off_src = offsets_src[(A, B)]

                ia_src = A - Rsrc.DC.amin + 1
                ib_src = B - Rsrc.DC.bmin + 1
                ia_tgt = A - Rtgt.DC.amin + 1
                ib_tgt = B - Rtgt.DC.bmin + 1

                if ia_src < 1 || ia_src > size(Rsrc.DC.dims, 1) || ib_src < 1 || ib_src > size(Rsrc.DC.dims, 2)
                    continue
                end
                if ia_tgt < 1 || ia_tgt > size(Rtgt.DC.dims, 1) || ib_tgt < 1 || ib_tgt > size(Rtgt.DC.dims, 2)
                    continue
                end
                if Rsrc.DC.dims[ia_src, ib_src] == 0 || Rtgt.DC.dims[ia_tgt, ib_tgt] == 0
                    continue
                end

                # Postcompose on Hom(C^p, E^B).
                Mb = postcompose_matrix_cached(Rtgt.homs[ia_tgt, ib_tgt], Rsrc.homs[ia_src, ib_src], phis[B + 1]; cache=cache)
                _append_scaled_triplets!(I, J, V, Mb, off_tgt - 1, off_src - 1)
            end

            maps[t - Rsrc.tot.tmin + 1] = sparse(I, J, V, dim_tgt, dim_src)
        end
    end

    return CochainMap(Rsrc.tot, Rtgt.tot, maps; check=check)
end

function rhom_map_second(
    g::PMorphism{K},
    C::ModuleCochainComplex{K},
    Hsrc::FF.FringeModule{K},
    Htgt::FF.FringeModule{K};
    maxlen::Int=3,
    resN=nothing,
    check::Bool=true,
    cache::Union{Nothing,HomSystemCache}=nothing,
    threads::Bool = (Threads.nthreads() > 1),
) where {K}
    Nsrc = IndicatorResolutions.pmodule_from_fringe(Hsrc)
    Ntgt = IndicatorResolutions.pmodule_from_fringe(Htgt)
    if g.dom !== Nsrc || g.cod !== Ntgt
        g = PMorphism{K}(Nsrc, Ntgt, g.comps)
    end
    resNsrc = isnothing(resN) ? injective_resolution(Nsrc, ResolutionOptions(maxlen=maxlen)) : resN
    resNtgt = isnothing(resN) ? injective_resolution(Ntgt, ResolutionOptions(maxlen=maxlen)) : resN
    Rsrc = RHomComplex(C, Nsrc; maxlen=maxlen, resN=resNsrc, cache=cache, threads=threads)
    Rtgt = RHomComplex(C, Ntgt; maxlen=maxlen, resN=resNtgt, cache=cache, threads=threads)
    return rhom_map_second(g, Rsrc, Rtgt; check=check, cache=cache, threads=threads)
end

function rhom_map_second(
    gH::FF.FringeModule{K},
    C::ModuleCochainComplex{K},
    Hsrc::FF.FringeModule{K},
    Htgt::FF.FringeModule{K};
    maxlen::Int=3,
    resN=nothing,
    check::Bool=true,
    cache::Union{Nothing,HomSystemCache}=nothing,
    threads::Bool = (Threads.nthreads() > 1),
) where {K}
    g = IndicatorResolutions.pmodule_from_fringe(gH)
    return rhom_map_second(g, C, Hsrc, Htgt; maxlen=maxlen, resN=resN, check=check, cache=cache, threads=threads)
end




struct HyperExtSpace{K}
    R::RHomComplex{K}
    cohom
end

function hyperExt(C::ModuleCochainComplex{K}, N::PModule{K}; kwargs...) where {K}
    R = RHomComplex(C,N; kwargs...)
    return HyperExtSpace{K}(R, cohomology_data(R.tot))
end

function hyperExt(C::ModuleCochainComplex{K}, H::FF.FringeModule{K}; kwargs...) where {K}
    return hyperExt(C, IR.pmodule_from_fringe(H); kwargs...)
end

dim(H::HyperExtSpace, t::Int) = (t < H.R.tot.tmin || t > H.R.tot.tmax) ? 0 : H.cohom[t - H.R.tot.tmin + 1].dimH

"""
    induced_map_on_cohomology(f, HCdom, HCcod, t)

Convenience wrapper to compute the induced map on cohomology in degree t
from a full cochain map `f::CochainMap`, using cached cohomology data.

- `HCdom` and `HCcod` should be the outputs of `cohomology_data(f.C)` and
  `cohomology_data(f.D)` respectively.
- Returns a dense matrix representing H^t(f) with respect to the stored bases.

This method is used by hyperExt_map_* and hyperTor_map_* to provide a
mathematician-friendly API.
"""
function induced_map_on_cohomology(
    f::CochainMap{K},
    HCdom,
    HCcod,
    t::Int
) where {K}
    # If degree is outside either range, return a correctly-sized zero map.
    if t < f.C.tmin || t > f.C.tmax || t < f.D.tmin || t > f.D.tmax
        dim_dom = (t < f.C.tmin || t > f.C.tmax) ? 0 : HCdom[t - f.C.tmin + 1].dimH
        dim_cod = (t < f.D.tmin || t > f.D.tmax) ? 0 : HCcod[t - f.D.tmin + 1].dimH
        return zeros(K, dim_cod, dim_dom)
    end

    Ht_dom = HCdom[t - f.C.tmin + 1]
    Ht_cod = HCcod[t - f.D.tmin + 1]
    Ft = _map_at(f, t)
    return induced_map_on_cohomology(Ht_dom, Ht_cod, Ft)
end


"""
    hyperExt_map_first(f, Hcod, Hdom; t, check=true)

Given a cochain map `f : C -> D` of module complexes, this returns the induced
map on hyperExt in degree `t`:

    Ext^t(D, N) -> Ext^t(C, N)

Here `Hcod = hyperExt(C,N)` and `Hdom = hyperExt(D,N)`.

This is the mathematically expected contravariance in the first argument.
"""
function hyperExt_map_first(
    f::ModuleCochainMap{K},
    Hcod::HyperExtSpace{K},
    Hdom::HyperExtSpace{K};
    t::Int,
    check::Bool = true,
    cache::Union{Nothing,HomSystemCache}=nothing
) where {K}
    Rmap = rhom_map_first(f, Hcod.R, Hdom.R; check=check, cache=cache)
    return induced_map_on_cohomology(Rmap, Hdom.cohom, Hcod.cohom, t)
end

"""
    hyperExt_map_second(g, Hsrc, Htgt; t, check=true)

Given a module morphism `g : N -> Np`, this returns the induced map on hyperExt
in degree `t`:

    Ext^t(C, N) -> Ext^t(C, Np)

Here `Hsrc = hyperExt(C,N)` and `Htgt = hyperExt(C,Np)`.

This is the mathematically expected covariance in the second argument.
"""
function hyperExt_map_second(
    g::PMorphism{K},
    Hsrc::HyperExtSpace{K},
    Htgt::HyperExtSpace{K};
    t::Int,
    check::Bool = true,
    cache::Union{Nothing,HomSystemCache}=nothing
) where {K}
    Rmap = rhom_map_second(g, Hsrc.R, Htgt.R; check=check, cache=cache)
    return induced_map_on_cohomology(Rmap, Hsrc.cohom, Htgt.cohom, t)
end



# ------------------------------------------------------------

struct DerivedTensorComplex{K}
    Rop::PModule{K}
    C::ModuleCochainComplex{K}
    resR
    DC::DoubleComplex{K}
    tot::CochainComplex{K}
end

function DerivedTensorComplex(
    Rop::PModule{K},
    C::ModuleCochainComplex{K};
    maxlen::Int = 3,
    maxdeg::Int = C.tmax,
    threads::Bool = (Threads.nthreads() > 1),
    check::Bool = false,
) where {K}
    resR = projective_resolution(Rop, ResolutionOptions(maxlen=maxlen, minimal=true, check=check))

    # Double-complex bidegrees:
    #   A = -a  where a = 0..maxlen is the projective-resolution (homological) degree
    #   B = p   where p runs over cochain degrees of C
    # Total degree is t = A + B = p - a, so hyperTor_n corresponds to H^{-n}.
    amin, amax = -maxlen, 0
    bmin, bmax = C.tmin, maxdeg

    na = amax - amin + 1
    nb = bmax - bmin + 1

    dims = zeros(Int, na, nb)
    dv = Array{SparseMatrixCSC{K, Int}}(undef, na, nb)
    dh = Array{SparseMatrixCSC{K, Int}}(undef, na, nb)

    total_jobs = na * nb

    build_cell = function (ai::Int, bi::Int)
        A = amin + (ai - 1)
        B = bmin + (bi - 1)

        a = -A  # projective-resolution degree (>= 0)
        p = B   # cochain degree in C

        Mp = _term(C, p)
        gens_a = resR.gens[a + 1]

        offs_dom = _offs_for_gens(Mp, gens_a)
        dims[ai, bi] = offs_dom[end]

        # Vertical differential: dv(A,B): (A,B) -> (A,B+1)
        # Use sign (-1)^a so that total differential is dv + dh.
        if B < bmax
            Mp1 = _term(C, p + 1)
            dC = _diff(C, p)  # PMorphism Mp -> Mp1
            offs_cod = _offs_for_gens(Mp1, gens_a)

            sgn = isodd(a) ? -one(K) : one(K)

            Itrip = Int[]
            Jtrip = Int[]
            Vtrip = K[]
            for (i, u) in enumerate(gens_a)
                # Block: dC.comps[u] : Mp[u] -> Mp1[u]
                _append_scaled_triplets!(
                    Itrip, Jtrip, Vtrip,
                    dC.comps[u],
                    offs_cod[i],
                    offs_dom[i];
                    scale = sgn,
                )
            end
            dv[ai, bi] = sparse(Itrip, Jtrip, Vtrip, offs_cod[end], offs_dom[end])
        else
            # Unused by total_complex at the top boundary (B == bmax), but keep typed.
            dv[ai, bi] = spzeros(K, 0, offs_dom[end])
        end

        # Horizontal differential: dh(A,B): (A,B) -> (A+1,B)
        # This corresponds to a -> a-1 in the projective resolution.
        if A < amax
            gens_am1 = resR.gens[a]        # degree (a-1) generators
            dP = resR.d_mat[a]             # matrix from gens_a -> gens_am1 (rows = gens_am1, cols = gens_a)
            offs_cod = _offs_for_gens(Mp, gens_am1)

            Itrip = Int[]
            Jtrip = Int[]
            Vtrip = K[]
            for (i, u) in enumerate(gens_a)
                for (j, v) in enumerate(gens_am1)
                    c = dP[j, i]
                    c == 0 && continue
                    Muv = map_leq(Mp, u, v)
                    _append_scaled_triplets!(
                        Itrip, Jtrip, Vtrip,
                        Muv,
                        offs_cod[j],
                        offs_dom[i];
                        scale = c,
                    )
                end
            end
            dh[ai, bi] = sparse(Itrip, Jtrip, Vtrip, offs_cod[end], offs_dom[end])
        else
            # Unused by total_complex at the right boundary (A == amax), but keep typed.
            dh[ai, bi] = spzeros(K, 0, offs_dom[end])
        end

        return nothing
    end

    if threads && total_jobs > 1
        Threads.@threads for idx in 1:total_jobs
            bi = Int(div(idx - 1, na)) + 1
            ai = (idx - 1) % na + 1
            build_cell(ai, bi)
        end
    else
        for bi in 1:nb
            for ai in 1:na
                build_cell(ai, bi)
            end
        end
    end

    DC = DoubleComplex(amin, amax, bmin, bmax, dims, dv, dh)
    tot = total_complex(DC)

    return DerivedTensorComplex{K}(Rop, C, resR, DC, tot)
end



DerivedTensor(Rop::PModule{K}, C::ModuleCochainComplex{K}; kwargs...) where {K} =
    DerivedTensorComplex(Rop, C; kwargs...).tot

struct HyperTorSpace{K}
    T::DerivedTensorComplex{K}
    cohom
end

function hyperTor(Rop::PModule{K}, C::ModuleCochainComplex{K}; kwargs...) where {K}
    T = DerivedTensorComplex(Rop,C; kwargs...)
    return HyperTorSpace{K}(T, cohomology_data(T.tot))
end

function hyperTor(H::FF.FringeModule{K}, C::ModuleCochainComplex{K}; kwargs...) where {K}
    return hyperTor(IR.pmodule_from_fringe(H), C; kwargs...)
end

"""
    dim(H::HyperTorSpace, n::Int) -> Int

Dimension of `hyperTor_n`, computed as `H^{-n}` of the total cochain complex.

By convention:
- returns 0 for `n < 0`,
- returns 0 when the required total degree `t = -n` lies outside the stored range.
"""
function dim(H::HyperTorSpace, n::Int)
    if n < 0
        return 0
    end
    # hyperTor_n = H^{-n}(Tot)
    t = -n
    tmin = H.T.tot.tmin
    tmax = H.T.tot.tmax
    if t < tmin || t > tmax
        return 0
    end
    return H.cohom[t - tmin + 1].dimH
end

# ---------------------------------------------------------------------------
# GradedSpaces interface for HyperTorSpace
# ---------------------------------------------------------------------------

"""
    degree_range(H::HyperTorSpace) -> UnitRange{Int}

Tor degrees `n` for which this `HyperTorSpace` stores data.

Internally, the total complex is a cochain complex in degrees `t`, and
`hyperTor_n` corresponds to cohomology degree `t = -n`. This function returns
the induced nonnegative range of `n` values.
"""
function degree_range(H::HyperTorSpace)
    tmin = H.T.tot.tmin
    tmax = H.T.tot.tmax

    # We need t = -n in [tmin, tmax], so n in [-tmax, -tmin].
    n_lo = max(0, -tmax)
    n_hi = -tmin
    if n_lo > n_hi
        return 0:-1  # empty range
    end
    return n_lo:n_hi
end

"""
    cycles(H::HyperTorSpace, n::Int) -> Matrix{K}

Columns form a basis of the cycle space in the relevant total cochain degree
`t = -n`. Returns a 0 times 0 matrix if `n` is outside `degree_range(H)`.
"""
function cycles(H::HyperTorSpace, n::Int)
    if n < 0
        return zeros(K, 0, 0)
    end
    t = -n
    tmin = H.T.tot.tmin
    tmax = H.T.tot.tmax
    if t < tmin || t > tmax
        return zeros(K, 0, 0)
    end
    return H.cohom[t - tmin + 1].K
end

"""
    boundaries(H::HyperTorSpace, n::Int) -> Matrix{K}

Columns form a basis of the boundary space in the relevant total cochain degree
`t = -n`. Returns a 0 times 0 matrix if `n` is outside `degree_range(H)`.
"""
function boundaries(H::HyperTorSpace, n::Int)
    if n < 0
        return zeros(K, 0, 0)
    end
    t = -n
    tmin = H.T.tot.tmin
    tmax = H.T.tot.tmax
    if t < tmin || t > tmax
        return zeros(K, 0, 0)
    end
    return H.cohom[t - tmin + 1].B
end

"""
    representative(H::HyperTorSpace, n::Int, coords::AbstractVector{K}) -> Vector{K}

Given coordinates of a class in `hyperTor_n` with respect to the fixed basis,
return a cocycle representative in total cochain degree `t = -n`.

Requirements:
- `n` must lie in `degree_range(H)`.
- `length(coords)` must equal `dim(H,n)`.
"""
function representative(H::HyperTorSpace, n::Int, coords::AbstractVector{K}) where {K}
    if n < 0
        throw(DomainError(n, "Tor degree n must be nonnegative."))
    end
    t = -n
    tmin = H.T.tot.tmin
    tmax = H.T.tot.tmax
    if t < tmin || t > tmax
        throw(DomainError(n, "n must lie in degree_range(H) = $(degree_range(H))."))
    end
    data = H.cohom[t - tmin + 1]
    d = size(data.Hrep, 2)
    if length(coords) != d
        throw(DimensionMismatch("Expected coordinates of length $d, got $(length(coords))."))
    end
    return cohomology_representative(data, coords)
end

"""
    coordinates(H::HyperTorSpace, n::Int, cocycle::AbstractVector{K}) -> Vector{K}

Compute coordinates of a cocycle representative in `hyperTor_n` relative to the
fixed basis, via total cochain degree `t = -n`.
"""
function coordinates(H::HyperTorSpace, n::Int, cocycle::AbstractVector{K}) where {K}
    if n < 0
        throw(DomainError(n, "Tor degree n must be nonnegative."))
    end
    t = -n
    tmin = H.T.tot.tmin
    tmax = H.T.tot.tmax
    if t < tmin || t > tmax
        throw(DomainError(n, "n must lie in degree_range(H) = $(degree_range(H))."))
    end
    data = H.cohom[t - tmin + 1]
    x = cohomology_coordinates(data, cocycle)
    return vec(x)
end

"""
    basis(H::HyperTorSpace, n::Int) -> Vector{Vector{K}}

Return a list of cocycle representatives forming a basis of `hyperTor_n`.
If `n` is outside `degree_range(H)` or `dim(H,n) == 0`, returns an empty vector.
"""
function basis(H::HyperTorSpace{K}, n::Int) where {K}
    if n < 0
        return Vector{Vector{K}}()
    end
    t = -n
    tmin = H.T.tot.tmin
    tmax = H.T.tot.tmax
    if t < tmin || t > tmax
        return Vector{Vector{K}}()
    end
    Hrep = H.cohom[t - tmin + 1].Hrep
    d = size(Hrep, 2)
    B = Vector{Vector{K}}(undef, d)
    for i in 1:d
        B[i] = Hrep[:, i]
    end
    return B
end


# ---------------------------------------------------------------------------
# GradedSpaces interface for HyperExtSpace
# ---------------------------------------------------------------------------

"""
    degree_range(H::HyperExtSpace) -> UnitRange{Int}

Inclusive range of total degrees `t` for which this `HyperExtSpace` stores
cohomology data of the total cochain complex `H.R.tot`.

This is the canonical iterator for graded-space queries:
`dim(H,t)`, `basis(H,t)`, `representative(H,t,coords)`, `coordinates(H,t,z)`,
`cycles(H,t)`, and `boundaries(H,t)`.
"""
degree_range(H::HyperExtSpace) = H.R.tot.tmin:H.R.tot.tmax

"""
    cycles(H::HyperExtSpace, t::Int) -> Matrix{K}

Columns form a basis of the cycle space `ker(d^t)` inside the total cochain group
in degree `t`. Returns a 0 times 0 matrix if `t` is outside `degree_range(H)`.
"""
function cycles(H::HyperExtSpace, t::Int)
    tmin = H.R.tot.tmin
    tmax = H.R.tot.tmax
    if t < tmin || t > tmax
        return zeros(K, 0, 0)
    end
    return H.cohom[t - tmin + 1].K
end

"""
    boundaries(H::HyperExtSpace, t::Int) -> Matrix{K}

Columns form a basis of the boundary space `im(d^(t-1))` inside the total cochain
group in degree `t`. Returns a 0 times 0 matrix if `t` is outside `degree_range(H)`.
"""
function boundaries(H::HyperExtSpace, t::Int)
    tmin = H.R.tot.tmin
    tmax = H.R.tot.tmax
    if t < tmin || t > tmax
        return zeros(K, 0, 0)
    end
    return H.cohom[t - tmin + 1].B
end

"""
    representative(H::HyperExtSpace, t::Int, coords::AbstractVector{K}) -> Vector{K}

Given coordinates of a class in `HyperExt^t` with respect to the fixed basis
chosen by `cohomology_data`, return a cocycle representative in the ambient
total cochain group in degree `t`.

Requirements:
- `t` must lie in `degree_range(H)`.
- `length(coords)` must equal `dim(H,t)`.
"""
function representative(H::HyperExtSpace, t::Int, coords::AbstractVector{K}) where {K}
    tmin = H.R.tot.tmin
    tmax = H.R.tot.tmax
    if t < tmin || t > tmax
        throw(DomainError(t, "t must lie in degree_range(H) = $(tmin):$(tmax)."))
    end
    data = H.cohom[t - tmin + 1]
    d = size(data.Hrep, 2)
    if length(coords) != d
        throw(DimensionMismatch("Expected coordinates of length $d, got $(length(coords))."))
    end
    return cohomology_representative(data, coords)
end

"""
    coordinates(H::HyperExtSpace, t::Int, cocycle::AbstractVector{K}) -> Vector{K}

Compute the coordinate vector of the cohomology class of `cocycle` in `HyperExt^t`,
relative to the fixed basis chosen by `cohomology_data`.

Notes:
- This assumes `cocycle` is in the correct ambient cochain group and is a cocycle.
- If `cocycle` is not a cocycle, the underlying solver may error or return
  coordinates for an implicitly projected class depending on consistency.
"""
function coordinates(H::HyperExtSpace, t::Int, cocycle::AbstractVector{K}) where {K}
    tmin = H.R.tot.tmin
    tmax = H.R.tot.tmax
    if t < tmin || t > tmax
        throw(DomainError(t, "t must lie in degree_range(H) = $(tmin):$(tmax)."))
    end
    data = H.cohom[t - tmin + 1]
    x = cohomology_coordinates(data, cocycle)
    return vec(x)
end

"""
    basis(H::HyperExtSpace, t::Int) -> Vector{Vector{K}}

Return a list of cocycle representatives forming a basis of `HyperExt^t`.
If `t` is outside `degree_range(H)` or `dim(H,t) == 0`, returns an empty vector.

Each basis element is a cochain vector in the ambient total cochain group.
"""
function basis(H::HyperExtSpace{K}, t::Int) where {K}
    tmin = H.R.tot.tmin
    tmax = H.R.tot.tmax
    if t < tmin || t > tmax
        return Vector{Vector{K}}()
    end
    Hrep = H.cohom[t - tmin + 1].Hrep
    d = size(Hrep, 2)
    B = Vector{Vector{K}}(undef, d)
    for i in 1:d
        B[i] = Hrep[:, i]
    end
    return B
end


# ============================================================
# Chain-level maps for derived tensor (covariant in both vars)
# ============================================================

@inline function _offs_for_gens(M::PModule{K}, gens::Vector{Int}) where {K}
    o = zeros(Int, length(gens) + 1)
    for i in 1:length(gens)
        u = gens[i]
        o[i+1] = o[i] + M.dims[u]
    end
    return o
end

"""
    derived_tensor_map_first(f, Tdom, Tcod; check=true)

Chain-level map on total derived tensor complexes induced by a morphism
in the *first* argument (right module variable):

    f : Rop -> Rop'

This produces a cochain map:

    Tot( Rop  otimes^L C ) -> Tot( Rop' otimes^L C )

Strict functoriality at the chain level uses the deterministic lift
provided by `_lift_pmodule_map_to_projective_resolution_chainmap_coeff`.
"""
function derived_tensor_map_first(
    f::PMorphism{K},
    Tdom::DerivedTensorComplex{K},
    Tcod::DerivedTensorComplex{K};
    check::Bool = true,
    threads::Bool = (Threads.nthreads() > 1),
) where {K}
    f.dom === Tdom.Rop || error("derived_tensor_map_first: f.dom must equal Tdom.Rop")
    f.cod === Tcod.Rop || error("derived_tensor_map_first: f.cod must equal Tcod.Rop")
    Tdom.C === Tcod.C || error("derived_tensor_map_first: complexes must be identical objects for strict functoriality")

    # Lift module map to a chain map between projective resolutions (coeff matrices per degree).
    upto = min(length(Tdom.resR.Pmods), length(Tcod.resR.Pmods)) - 1
    coeffs = _lift_pmodule_map_to_projective_resolution_chainmap_coeff(Tdom.resR, Tcod.resR, f; upto=upto)

    # Optional validation of chain map relation: d_cod[a]*F_a == F_{a-1}*d_dom[a]
    if check
        upto = length(coeffs) - 1
        for a in 1:upto
            lhs = Tcod.resR.d_mat[a] * coeffs[a+1]
            rhs = coeffs[a] * Tdom.resR.d_mat[a]
            lhs == rhs || error("derived_tensor_map_first: lifted coefficients fail chain map check at degree $a")
        end
    end

    # Assemble total cochain map degree-by-degree using block structure.
    tmin = min(Tdom.tot.tmin, Tcod.tot.tmin)
    tmax = max(Tdom.tot.tmax, Tcod.tot.tmax)
    maps = Vector{SparseMatrixCSC{K,Int}}(undef, tmax - tmin + 1)

    if threads && Threads.nthreads() > 1 && (tmax >= tmin)
        Threads.@threads for idx in 1:(tmax - tmin + 1)
            t = tmin + idx - 1
            off_dom = _tot_block_offsets(Tdom.DC, t)
            off_cod = _tot_block_offsets(Tcod.DC, t)
            dim_dom = _cc_dim_at(Tdom.tot, t)
            dim_cod = _cc_dim_at(Tcod.tot, t)

            I = Int[]; J = Int[]; V = K[]

            for (key, col_off) in off_dom
                haskey(off_cod, key) || continue
                row_off = off_cod[key]
                A, p = key
                a = -A
                (a < 0 || a > (length(Tdom.resR.Pmods) - 1)) && continue
                (a < 0 || a > (length(Tcod.resR.Pmods) - 1)) && continue

                Mp = _term(Tdom.C, p)
                dom_gens = Tdom.resR.gens[a+1]
                cod_gens = Tcod.resR.gens[a+1]
                offs_dom = _offs_for_gens(Mp, dom_gens)
                offs_cod = _offs_for_gens(Mp, cod_gens)

                block = _tensor_map_on_tor_chains_from_projective_coeff(
                    Mp, dom_gens, cod_gens, offs_dom, offs_cod, coeffs[a+1]
                )

                ii, jj, vv = findnz(block)
                for k in eachindex(vv)
                    push!(I, row_off - 1 + ii[k])
                    push!(J, col_off - 1 + jj[k])
                    push!(V, vv[k])
                end
            end

            maps[idx] = sparse(I, J, V, dim_cod, dim_dom)
        end
    else
        for t in tmin:tmax
            off_dom = _tot_block_offsets(Tdom.DC, t)
            off_cod = _tot_block_offsets(Tcod.DC, t)
            dim_dom = _cc_dim_at(Tdom.tot, t)
            dim_cod = _cc_dim_at(Tcod.tot, t)

            I = Int[]; J = Int[]; V = K[]

            for (key, col_off) in off_dom
                haskey(off_cod, key) || continue
                row_off = off_cod[key]
                A, p = key
                a = -A
                (a < 0 || a > (length(Tdom.resR.Pmods) - 1)) && continue
                (a < 0 || a > (length(Tcod.resR.Pmods) - 1)) && continue

                Mp = _term(Tdom.C, p)
                dom_gens = Tdom.resR.gens[a+1]
                cod_gens = Tcod.resR.gens[a+1]
                offs_dom = _offs_for_gens(Mp, dom_gens)
                offs_cod = _offs_for_gens(Mp, cod_gens)

                block = _tensor_map_on_tor_chains_from_projective_coeff(
                    Mp, dom_gens, cod_gens, offs_dom, offs_cod, coeffs[a+1]
                )

                ii, jj, vv = findnz(block)
                for k in eachindex(vv)
                    push!(I, row_off - 1 + ii[k])
                    push!(J, col_off - 1 + jj[k])
                    push!(V, vv[k])
                end
            end

            maps[t - tmin + 1] = sparse(I, J, V, dim_cod, dim_dom)
        end
    end

    return CochainMap(Tdom.tot, Tcod.tot, maps; tmin=tmin, tmax=tmax, check=check)
end


"""
    derived_tensor_map_second(g, Tsrc, Ttgt; check=true)

Chain-level map on total derived tensor complexes induced by a cochain map
in the *second* argument (complex variable):

    g : C -> C'

with fixed right module Rop.

Produces:

    Tot(Rop otimes^L C) -> Tot(Rop otimes^L C')

Strict chain-level functoriality requires that Tsrc and Ttgt use the same
projective resolution object (or at least same gens ordering). Here we
require identical gens lists for safety.
"""
function derived_tensor_map_second(
    g::ModuleCochainMap{K},
    Tsrc::DerivedTensorComplex{K},
    Ttgt::DerivedTensorComplex{K};
    check::Bool = true,
    threads::Bool = (Threads.nthreads() > 1),
) where {K}
    g.C === Tsrc.C || error("derived_tensor_map_second: g.C must equal Tsrc.C")
    g.D === Ttgt.C || error("derived_tensor_map_second: g.D must equal Ttgt.C")
    Tsrc.Rop === Ttgt.Rop || error("derived_tensor_map_second: right modules must match")
    Tsrc.resR.gens == Ttgt.resR.gens || error("derived_tensor_map_second: resolutions must have identical gens ordering")

    tmin = min(Tsrc.tot.tmin, Ttgt.tot.tmin)
    tmax = max(Tsrc.tot.tmax, Ttgt.tot.tmax)
    maps = Vector{SparseMatrixCSC{K,Int}}(undef, tmax - tmin + 1)

    if threads && Threads.nthreads() > 1 && (tmax >= tmin)
        Threads.@threads for idx in 1:(tmax - tmin + 1)
            t = tmin + idx - 1
            off_src = _tot_block_offsets(Tsrc.DC, t)
            off_tgt = _tot_block_offsets(Ttgt.DC, t)
            dim_src = _cc_dim_at(Tsrc.tot, t)
            dim_tgt = _cc_dim_at(Ttgt.tot, t)

            I = Int[]; J = Int[]; V = K[]

            for (key, col_off) in off_src
                haskey(off_tgt, key) || continue
                row_off = off_tgt[key]
                A, p = key
                a = -A
                (a < 0 || a > (length(Tsrc.resR.Pmods) - 1)) && continue

                Mp = _term(Tsrc.C, p)
                Mp1 = _term(Ttgt.C, p)
                gp = _map(g, p)
                gens = Tsrc.resR.gens[a+1]

                offs_dom = _offs_for_gens(Mp, gens)
                offs_cod = _offs_for_gens(Mp1, gens)

                # Sparse block-diagonal assembly
                for (i,u) in enumerate(gens)
                    row0 = offs_cod[i]
                    col0 = offs_dom[i]
                    block = gp.comps[u]
                    nr = size(block, 1)
                    nc = size(block, 2)
                    for r in 1:nr
                        for c in 1:nc
                            x = block[r,c]
                            if x != 0
                                push!(I, row_off - 1 + row0 + r)
                                push!(J, col_off - 1 + col0 + c)
                                push!(V, x)
                            end
                        end
                    end
                end
            end

            maps[idx] = sparse(I, J, V, dim_tgt, dim_src)
        end
    else
        for t in tmin:tmax
            off_src = _tot_block_offsets(Tsrc.DC, t)
            off_tgt = _tot_block_offsets(Ttgt.DC, t)
            dim_src = _cc_dim_at(Tsrc.tot, t)
            dim_tgt = _cc_dim_at(Ttgt.tot, t)

            I = Int[]; J = Int[]; V = K[]

            for (key, col_off) in off_src
                haskey(off_tgt, key) || continue
                row_off = off_tgt[key]
                A, p = key
                a = -A
                (a < 0 || a > (length(Tsrc.resR.Pmods) - 1)) && continue

                Mp = _term(Tsrc.C, p)
                Mp1 = _term(Ttgt.C, p)
                gp = _map(g, p)
                gens = Tsrc.resR.gens[a+1]

                offs_dom = _offs_for_gens(Mp, gens)
                offs_cod = _offs_for_gens(Mp1, gens)

                # Sparse block-diagonal assembly
                for (i,u) in enumerate(gens)
                    row0 = offs_cod[i]
                    col0 = offs_dom[i]
                    block = gp.comps[u]
                    nr = size(block, 1)
                    nc = size(block, 2)
                    for r in 1:nr
                        for c in 1:nc
                            x = block[r,c]
                            if x != 0
                                push!(I, row_off - 1 + row0 + r)
                                push!(J, col_off - 1 + col0 + c)
                                push!(V, x)
                            end
                        end
                    end
                end
            end

            maps[t - tmin + 1] = sparse(I, J, V, dim_tgt, dim_src)
        end
    end

    return CochainMap(Tsrc.tot, Ttgt.tot, maps; tmin=tmin, tmax=tmax, check=check)
end


# ============================================================
# Induced maps on hyperTor_n (mathematician-friendly API)
# ============================================================

"""
    hyperTor_map_first(f, Hdom, Hcod; n, check=true)

Given a morphism of right modules f : Rop -> Rop', return the induced map:

    Tor_n(Rop, C) -> Tor_n(Rop', C)

Here `Hdom = hyperTor(Rop, C)` and `Hcod = hyperTor(Rop', C)`.

Convention: Tor_n = H^{-n}(Tot).
"""
function hyperTor_map_first(
    f::PMorphism{K},
    Hdom::HyperTorSpace{K},
    Hcod::HyperTorSpace{K};
    n::Int,
    check::Bool = true
) where {K}
    Tmap = derived_tensor_map_first(f, Hdom.T, Hcod.T; check=check)
    return induced_map_on_cohomology(Tmap, Hdom.cohom, Hcod.cohom, -n)
end


"""
    hyperTor_map_second(g, Hsrc, Htgt; n, check=true)

Given a cochain map g : C -> C', return the induced map:

    Tor_n(Rop, C) -> Tor_n(Rop, C')

Here `Hsrc = hyperTor(Rop, C)` and `Htgt = hyperTor(Rop, C')`.

Convention: Tor_n = H^{-n}(Tot).
"""
function hyperTor_map_second(
    g::ModuleCochainMap{K},
    Hsrc::HyperTorSpace{K},
    Htgt::HyperTorSpace{K};
    n::Int,
    check::Bool = true
) where {K}
    Tmap = derived_tensor_map_second(g, Hsrc.T, Htgt.T; check=check)
    return induced_map_on_cohomology(Tmap, Hsrc.cohom, Htgt.cohom, -n)
end


end # module
function _resolution_offsets(res)
    return res.offsets
end

function _resolution_offsets(res::DerivedFunctors.Resolutions.InjectiveResolution)
    return _gens_offsets(res.gens)
end

function _gens_offsets(gens)
    offsets = Vector{Vector{Int}}(undef, length(gens))
    for i in eachindex(gens)
        gi = gens[i]
        if isempty(gi)
            offsets[i] = Int[]
            continue
        end
        if eltype(gi) <: AbstractVector
            lens = length.(gi)
            offs = Vector{Int}(undef, length(lens) + 1)
            offs[1] = 0
            for j in 1:length(lens)
                offs[j + 1] = offs[j] + lens[j]
            end
            offsets[i] = offs
        else
            offsets[i] = collect(0:length(gi))
        end
    end
    return offsets
end
