function _supports_exact_euler_signed_measure(enc::EncodingResult{PType,MType};
                                              opts::InvariantOptions=InvariantOptions(),
                                              kwargs...) where {PType,MType<:_LazyEncodedModule}
    _ = kwargs
    return enc.M.dims !== nothing || _supports_exact_restricted_hilbert(enc; opts=opts)
end

function _exact_euler_signed_measure(enc::EncodingResult{PType,MType};
                                     opts::InvariantOptions=InvariantOptions(),
                                     kwargs...) where {PType,MType<:_LazyEncodedModule}
    # EncodingResult denotes one chosen homology module, whose Euler surface
    # is its dimension function. Alternating cellular dimensions belong to
    # EncodedComplexResult/the full complex, not to this selected degree.
    dims = _exact_restricted_hilbert(enc; opts=opts)
    dims === nothing && return nothing
    return SignedMeasures.euler_signed_measure(dims, enc.pi, opts; kwargs...)
end
