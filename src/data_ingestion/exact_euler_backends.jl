function _supports_exact_euler_signed_measure(enc::EncodingResult{PType,MType};
                                              kwargs...) where {PType,MType<:_LazyEncodedModule}
    _ = kwargs
    return true
end

function _exact_euler_signed_measure(enc::EncodingResult{PType,MType};
                                     opts::InvariantOptions=InvariantOptions(),
                                     kwargs...) where {PType,MType<:_LazyEncodedModule}
    return euler_signed_measure(enc.M.lazy, enc.pi, opts; kwargs...)
end
