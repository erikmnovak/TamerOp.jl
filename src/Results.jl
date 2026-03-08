# =============================================================================
# Results.jl
#
# Workflow-facing result wrappers and lightweight result utilities.
# =============================================================================
module Results

using ..CoreModules: AbstractCoeffField, QQField, SessionCache
using ..Options: EncodingOptions, ResolutionOptions, InvariantOptions
using ..EncodingCore: CompiledEncoding, _compile_encoding_cached
import ..CoreModules: change_field
import ..EncodingCore: compile_encoding

"""
    EncodingResult(P, M, pi; H=nothing, presentation=nothing,
                  opts=EncodingOptions(), backend=opts.backend, meta=NamedTuple())

Small workflow object representing the output of a user-facing `encode(...)`.
"""
struct EncodingResult{PType,MType,PiType,HType,PresType,MetaType}
    P::PType
    M::MType
    pi::PiType
    H::HType
    presentation::PresType
    opts::EncodingOptions
    backend::Symbol
    meta::MetaType
end

EncodingResult(P, M, pi;
               H=nothing,
               presentation=nothing,
               opts::EncodingOptions=EncodingOptions(),
               backend::Symbol=opts.backend,
               meta=NamedTuple()) =
    EncodingResult(P, M, pi, H, presentation, opts, backend, meta)

"""
    CohomologyDimsResult(P, dims, pi; degree=0, field=QQField(), meta=NamedTuple())

Workflow object for dims-only cohomology output on an encoding poset.
"""
struct CohomologyDimsResult{PType,DType,PiType,FType,MetaType}
    P::PType
    dims::DType
    pi::PiType
    degree::Int
    field::FType
    meta::MetaType
end

CohomologyDimsResult(P, dims, pi;
                     degree::Int=0,
                     field::AbstractCoeffField=QQField(),
                     meta=NamedTuple()) =
    CohomologyDimsResult(P, dims, pi, degree, field, meta)

materialize_module(M) = M
module_dims(M) = M.dims

compile_encoding(enc::EncodingResult; kwargs...) =
    enc.pi isa CompiledEncoding ? enc.pi : compile_encoding(enc.P, enc.pi; kwargs...)

compile_encoding(enc::CohomologyDimsResult; kwargs...) =
    enc.pi isa CompiledEncoding ? enc.pi : compile_encoding(enc.P, enc.pi; kwargs...)

Base.length(::EncodingResult) = 3
Base.IteratorSize(::Type{<:EncodingResult}) = Base.HasLength()

function Base.iterate(enc::EncodingResult, state::Int=1)
    state == 1 && return (enc.P, 2)
    state == 2 && return (enc.M, 3)
    state == 3 && return (enc.pi, 4)
    return nothing
end

Base.length(::CohomologyDimsResult) = 3
Base.IteratorSize(::Type{<:CohomologyDimsResult}) = Base.HasLength()

function Base.iterate(enc::CohomologyDimsResult, state::Int=1)
    state == 1 && return (enc.P, 2)
    state == 2 && return (enc.dims, 3)
    state == 3 && return (enc.pi, 4)
    return nothing
end

@inline function _encoding_with_session_cache(enc::EncodingResult,
                                              session_cache::Union{Nothing,SessionCache})
    session_cache === nothing && return enc
    raw_pi = enc.pi isa CompiledEncoding ? enc.pi.pi : enc.pi
    pi2 = _compile_encoding_cached(enc.P, raw_pi, session_cache)
    return EncodingResult(enc.P, enc.M, pi2;
                          H=enc.H,
                          presentation=enc.presentation,
                          opts=enc.opts,
                          backend=enc.backend,
                          meta=enc.meta)
end

@inline function _encoding_with_session_cache(enc::CohomologyDimsResult,
                                              session_cache::Union{Nothing,SessionCache})
    session_cache === nothing && return enc
    raw_pi = enc.pi isa CompiledEncoding ? enc.pi.pi : enc.pi
    pi2 = _compile_encoding_cached(enc.P, raw_pi, session_cache)
    return CohomologyDimsResult(enc.P, enc.dims, pi2;
                                degree=enc.degree,
                                field=enc.field,
                                meta=enc.meta)
end

function change_field(enc::EncodingResult, field::AbstractCoeffField)
    M2 = change_field(enc.M, field)
    H2 = enc.H === nothing ? nothing : change_field(enc.H, field)
    pres2 = enc.presentation
    if pres2 !== nothing && hasmethod(change_field, (typeof(pres2), AbstractCoeffField))
        pres2 = change_field(pres2, field)
    end
    return EncodingResult(enc.P, M2, enc.pi;
                          H=H2,
                          presentation=pres2,
                          opts=enc.opts,
                          backend=enc.backend,
                          meta=enc.meta)
end

function change_field(enc::CohomologyDimsResult, field::AbstractCoeffField)
    return CohomologyDimsResult(enc.P, copy(enc.dims), enc.pi;
                                degree=enc.degree,
                                field=field,
                                meta=enc.meta)
end

unwrap(enc::EncodingResult) = (enc.P, enc.M, enc.pi)
unwrap(enc::CohomologyDimsResult) = (enc.P, enc.dims, enc.pi)

"""
    ResolutionResult(res; enc=nothing, betti=nothing, minimality=nothing,
                     opts=ResolutionOptions(), meta=NamedTuple())

Workflow object storing a resolution computation plus provenance.
"""
struct ResolutionResult{ResType,EncType,BettiType,MinType,MetaType}
    res::ResType
    enc::EncType
    betti::BettiType
    minimality::MinType
    opts::ResolutionOptions
    meta::MetaType
end

ResolutionResult(res;
                 enc=nothing,
                 betti=nothing,
                 minimality=nothing,
                 opts::ResolutionOptions=ResolutionOptions(),
                 meta=NamedTuple()) =
    ResolutionResult(res, enc, betti, minimality, opts, meta)

unwrap(res::ResolutionResult) = res.res

"""
    InvariantResult(enc, which, value; opts=InvariantOptions(), meta=NamedTuple())

Workflow object storing a computed invariant plus provenance.
"""
struct InvariantResult{EncType,WhichType,ValType,MetaType}
    enc::EncType
    which::WhichType
    value::ValType
    opts::InvariantOptions
    meta::MetaType
end

InvariantResult(enc, which, value;
                opts::InvariantOptions=InvariantOptions(),
                meta=NamedTuple()) =
    InvariantResult(enc, which, value, opts, meta)

unwrap(inv::InvariantResult) = inv.value

function change_field(res::ResolutionResult, field::AbstractCoeffField)
    enc2 = res.enc === nothing ? nothing : change_field(res.enc, field)
    res2 = res.res
    if res2 !== nothing && hasmethod(change_field, (typeof(res2), AbstractCoeffField))
        res2 = change_field(res2, field)
    end
    return ResolutionResult(res2;
                            enc=enc2,
                            betti=res.betti,
                            minimality=res.minimality,
                            opts=res.opts,
                            meta=res.meta)
end

function change_field(inv::InvariantResult, field::AbstractCoeffField)
    enc2 = change_field(inv.enc, field)
    return InvariantResult(enc2, inv.which, inv.value; opts=inv.opts, meta=inv.meta)
end

end # module Results
