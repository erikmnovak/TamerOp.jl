# Mathematical provenance for workflow results. Inspection reads stored data only.

@inline function _recorded_provenance(meta)
    p = meta isa NamedTuple ? get(meta, :provenance, nothing) :
        meta isa AbstractDict ? get(meta, :provenance, nothing) : nothing
    return p isa NamedTuple ? p : NamedTuple()
end

function _encoding_provenance(P, field, pi, meta; backend=:not_recorded,
                              degree=nothing, degree_convention=:not_applicable)
    recorded = _recorded_provenance(meta)
    defaults = (category=:finite_poset_representations, base_poset=P,
        field=field, degree=degree, degree_convention=degree_convention,
        encoding=(map_type=typeof(pi), backend=backend),
        window=:not_recorded, orientation=:not_recorded,
        construction=(requested=:not_recorded, effective=:not_recorded, substitution=:not_recorded),
        discretization=:not_recorded, approximation=:not_recorded,
        backend=(requested=:not_recorded, effective=backend),
        ambient_identification=:not_asserted)
    # Actual field/poset come from the current object, never stale producer metadata.
    return merge(defaults, recorded, (category=:finite_poset_representations, base_poset=P, field=field,
        encoding=(map_type=typeof(pi), backend=backend)))
end

function provenance(enc::EncodingResult)
    field = _provenance_field(enc.M)
    return _encoding_provenance(enc.P, field, enc.pi, enc.meta; backend=enc.backend)
end

function provenance(enc::CohomologyDimsResult)
    p = _encoding_provenance(enc.P, enc.field, enc.pi, enc.meta;
        degree=enc.degree, degree_convention=:cohomological)
    return merge(p, (degree=enc.degree, output=:dimension_function))
end

function _presentation_encoding_meta(opts::EncodingOptions, backend::Symbol; joint::Bool=false)
    return (provenance=(window=:unrestricted, orientation=:coordinatewise,
        construction=(requested=:presentation_encoding, effective=:presentation_encoding,
                      substitution=:none),
        refinement=joint ? :joint_presentation_encoding : :single_encoding,
        discretization=(kind=:finite_encoding, filtration_values=:not_resampled),
        approximation=(construction=:none_requested,
            geometry=backend === :zn ? :integer_coordinates : :backend_numerical_contract,
            strict_eps=opts.strict_eps,
            effective_strict_eps=backend === :pl ? opts.strict_eps : nothing,
            feasibility=backend === :pl ?
                (opts.strict_eps === nothing ? :exact_rational : :fixed_margin) : :not_applicable),
        backend=(requested=opts.backend, effective=backend)),)
end

function provenance(enc::EncodedComplexResult)
    p = _encoding_provenance(enc.P, enc.field, enc.pi, enc.meta;
        degree=nothing, degree_convention=:cohomological)
    return merge(p, (degree=nothing, degree_range=_try_complex_degree_range(enc.C),
                     degree_convention=:cohomological, output=:cochain_complex))
end

function provenance(res::ModuleTranslationResult)
    source = res.source === nothing ? NamedTuple() : provenance(res.source)
    P = getproperty(res.M, :Q)
    p = _encoding_provenance(P, _provenance_field(res.M), res.classifier, res.meta)
    derived_degree = res.meta isa NamedTuple || res.meta isa AbstractDict ?
        get(res.meta, :derived_degree, nothing) : nothing
    convention = res.kind === :derived_pushforward_left ? :homological :
                 res.kind === :derived_pushforward_right ? :cohomological : :not_applicable
    return merge(p, (category=:finite_poset_representations,
        translation=res.kind, translation_map=res.map,
        degree=derived_degree, degree_convention=convention,
        window=res.classifier === nothing ? :not_applicable : p.window,
        orientation=res.classifier === nothing ? :not_applicable : p.orientation,
        reconstruction=res.classifier === nothing ? :none : :translated_module_pullback,
        construction=(requested=res.kind, effective=res.kind, substitution=:none),
        discretization=:not_applicable, approximation=:source_contract,
        backend=(requested=:finite_linear_algebra, effective=:operation_specific),
        source=source, ambient_identification=:not_asserted))
end

function provenance(res::ResolutionResult)
    source = res.enc === nothing ? NamedTuple() : provenance(res.enc)
    native = applicable(provenance, res.res) ? provenance(res.res) : NamedTuple()
    return merge((category=:finite_poset_representations, base_poset=nothing,
        field=nothing, degree=:not_recorded, degree_convention=:not_recorded),
        source, native, (source=source, ambient_identification=:not_asserted))
end

function provenance(inv::InvariantResult)
    source = provenance(inv.enc)
    recorded = _recorded_provenance(inv.meta)
    return merge(source, recorded, (category=source.category,
        base_poset=source.base_poset, field=source.field, invariant=inv.which,
        source=source,
        query=(axes=inv.opts.axes, axes_policy=inv.opts.axes_policy,
               max_axis_len=inv.opts.max_axis_len, box=inv.opts.box,
               pl_mode=inv.opts.pl_mode,
               parameters=inv.meta isa NamedTuple ? get(inv.meta, :parameters, NamedTuple()) : :not_recorded),
        # The source window is known; requested query options alone do not
        # certify the evaluation window of arbitrary user-supplied invariants.
        evaluation_window=get(recorded, :evaluation_window, :not_recorded),
        window=source.window))
end

function _field_reinterpretation_meta(enc::EncodingResult, field)
    p = merge(provenance(enc), (field=field,
        source=provenance(enc), degree=nothing, degree_convention=:not_applicable,
        reconstruction=:stored_matrix_reinterpretation,
        construction=(requested=:coefficient_reinterpretation,
                      effective=:coefficient_reinterpretation, substitution=:none),
        coefficient_change=(from=_provenance_field(enc.M), to=field,
            semantics=:reinterpret_stored_module_matrices),
        ambient_identification=:not_asserted))
    if enc.meta isa NamedTuple
        return merge(enc.meta, (provenance=p,))
    elseif enc.meta isa AbstractDict
        out = copy(enc.meta)
        out[:provenance] = p
        return out
    end
    return (provenance=p, source_meta=enc.meta)
end

function _complex_field_reinterpretation_meta(enc::EncodedComplexResult, field)
    source = provenance(enc)
    p = merge(source, (field=field, source=source,
        reconstruction=:stored_complex_matrix_reinterpretation,
        construction=(requested=:coefficient_reinterpretation,
                      effective=:coefficient_reinterpretation, substitution=:none),
        coefficient_change=(from=enc.field, to=field,
            semantics=:reinterpret_stored_complex_matrices),
        ambient_identification=:not_asserted))
    # The original presentation is historical source data, not a witness for
    # the reinterpreted complex. Preserve it only inside the source metadata.
    if enc.meta isa NamedTuple
        kept = (; (k => v for (k, v) in pairs(enc.meta) if k !== :presentation)...)
        return merge(kept, (provenance=p, source_meta=enc.meta))
    elseif enc.meta isa AbstractDict
        out = copy(enc.meta)
        delete!(out, :presentation)
        out[:provenance] = p
        out[:source_meta] = enc.meta
        return out
    end
    return (provenance=p, source_meta=enc.meta)
end

function _show_result_provenance(io, p)
    print(io, "\n  category: ", p.category,
          "\n  field: ", p.field)
    haskey(p, :degree) && print(io, "\n  degree: ", p.degree,
        " (", get(p, :degree_convention, :not_recorded), ")")
    haskey(p, :window) && print(io, "\n  window: ", p.window)
    haskey(p, :orientation) && print(io, "\n  orientation: ", p.orientation)
    if haskey(p, :construction)
        c = p.construction
        print(io, "\n  construction: ", c)
    end
    haskey(p, :discretization) && print(io, "\n  discretization: ", p.discretization)
    haskey(p, :approximation) && print(io, "\n  approximation: ", p.approximation)
    return nothing
end
