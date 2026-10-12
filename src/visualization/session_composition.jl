# Capture the already-checked relation behind a composed figure without keeping
# the live diagram/chain. Reopening validates the archive, not a new theorem.

function _frozen_composition_anchor(spec, anchor)
    anchor.target === nothing && return anchor
    anchor.target[1] === :source_cell || return anchor
    i = findfirst(==(anchor.panel), spec.metadata.panel_names)
    i === nothing && throw(ArgumentError("Unknown source anchor panel $(anchor.panel)."))
    resolved = _composition_anchor(spec.panels[i], anchor)
    resolved.kind === :data || throw(ArgumentError("Source anchors must resolve to data coordinates."))
    return VisualAnchor(anchor.panel; point=resolved.point)
end

function _freeze_visual_composition(spec::VisualizationSpec)
    # Check source identity and representative equations while the actual live
    # objects still exist. Validation below will subsequently be structural.
    spec.kind === :composition && check_visual_spec(spec; throw=true)
    metadata = spec.metadata
    if spec.kind === :composition
        archived = NamedTuple[]
        annotations = map(metadata.annotations) do annotation
            from = _frozen_composition_anchor(spec, annotation.from)
            to = _frozen_composition_anchor(spec, annotation.to)
            for (original, resolved) in ((annotation.from, from), (annotation.to, to))
                original === resolved || push!(archived, (; original, resolved))
            end
            evidence = annotation.evidence
            if evidence !== nothing && get(evidence, :mode, nothing) !== :frozen_correspondence
                cycle = evidence.cycle
                data = DataIngestion.source_chain_data(cycle)
                support = (; data.cell_indices, data.cell_ids, data.coefficients, data.source_ids)
                evidence = (; mode=:frozen_correspondence, verification=:checked_before_capture,
                    relation=:ordinary_source_cycle_to_interval, field=cycle.field,
                    degree=cycle.dimension, parameter=cycle.parameter, interval=cycle.metadata.interval,
                    member=(; kind=evidence.kind, index=evidence.index), support,
                    original_from=annotation.from, original_to=annotation.to,
                    resolved_from=from, resolved_to=to)
            end
            VisualAnnotation(from, to, annotation.kind, annotation.label, annotation.curve, evidence)
        end
        old_anchors = get(metadata, :captured_source_anchors, ())
        metadata = merge(metadata, (; annotations=Tuple(annotations),
            captured_source_anchors=(old_anchors..., archived...),
            correspondence_verification=any(a -> a.evidence !== nothing, annotations) ?
                :checked_before_capture : :no_computed_correspondence))
    end
    if haskey(metadata, :overlay_specs)
        metadata = merge(metadata, (; overlay_specs=map(_freeze_visual_composition, metadata.overlay_specs)))
    end
    panels = VisualizationSpec[_freeze_visual_composition(panel) for panel in spec.panels]
    return _composition_copy(spec; panels, metadata)
end

function _check_frozen_correspondence(spec, annotation)
    evidence = annotation.evidence
    expected = (:mode, :verification, :relation, :field, :degree, :parameter, :interval,
        :member, :support, :original_from, :original_to, :resolved_from, :resolved_to)
    evidence isa NamedTuple && keys(evidence) == expected ||
        throw(ArgumentError("Unrecognized captured correspondence record."))
    evidence.mode === :frozen_correspondence && evidence.verification === :checked_before_capture &&
        evidence.relation === :ordinary_source_cycle_to_interval ||
        throw(ArgumentError("Captured correspondence must declare its original check and relation."))
    evidence.field isa CoreModules.AbstractCoeffField ||
        throw(ArgumentError("Captured correspondence needs its coefficient field."))
    evidence.degree isa Integer && !(evidence.degree isa Bool) && evidence.degree >= 0 ||
        throw(ArgumentError("Captured correspondence needs a nonnegative degree."))
    evidence.parameter isa Tuple && length(evidence.parameter) == 1 &&
        only(evidence.parameter) isa Real && isfinite(only(evidence.parameter)) ||
        throw(ArgumentError("Captured ordinary correspondence needs one finite parameter."))
    evidence.interval isa Tuple && length(evidence.interval) == 2 &&
        all(x -> x isa Real && !isnan(x), evidence.interval) ||
        throw(ArgumentError("Captured correspondence needs its exact interval endpoints."))
    member = evidence.member
    member isa NamedTuple && keys(member) == (:kind, :index) && member.kind in (:finite, :essential) &&
        member.index isa Integer && !(member.index isa Bool) && member.index > 0 ||
        throw(ArgumentError("Captured correspondence has an invalid interval member."))
    for name in (:original_from, :original_to, :resolved_from, :resolved_to)
        getproperty(evidence, name) isa VisualAnchor || throw(ArgumentError("Captured endpoints must be declarative anchors."))
    end
    isequal(describe(evidence.resolved_from), describe(annotation.from)) &&
        isequal(describe(evidence.resolved_to), describe(annotation.to)) ||
        throw(ArgumentError("Captured endpoints disagree with the selected drawing."))
    original = evidence.original_from
    original.space === :data && original.panel === annotation.from.panel &&
        original.target isa Tuple && length(original.target) == 2 && original.target[1] === :source_cell &&
        original.target[2] isa Integer && !(original.target[2] isa Bool) && original.target[2] > 0 ||
        throw(ArgumentError("Captured correspondence lacks its original source-cell identity."))
    evidence.original_to.target == (:interval, evidence.degree, member.kind, member.index) &&
        isequal(describe(evidence.original_to), describe(annotation.to)) ||
        throw(ArgumentError("Captured correspondence disagrees with its interval selector."))
    support = evidence.support
    support isa NamedTuple && keys(support) == (:cell_indices, :cell_ids, :coefficients, :source_ids) &&
        all(x -> x isa Tuple, values(support)) &&
        all(x -> length(x) == length(support.cell_indices), values(support)) &&
        allunique(support.cell_indices) && all(x -> x isa Integer && !(x isa Bool) && x > 0, support.cell_indices) &&
        original.target[2] in support.cell_indices && all(!iszero, support.coefficients) ||
        throw(ArgumentError("Captured correspondence has inconsistent source support."))
    source = spec.panels[findfirst(==(annotation.from.panel), spec.metadata.panel_names)]
    retained = get(source.metadata, :source_chain, nothing)
    retained isa NamedTuple || throw(ArgumentError("Captured correspondence needs retained source-chain drawing data."))
    for name in keys(support)
        haskey(retained, name) && isequal(getproperty(support, name), getproperty(retained, name)) ||
            throw(ArgumentError("Captured correspondence source $name disagrees with its panel."))
    end
    retained.kind === :cycle && isequal(retained.field, evidence.field) && retained.dimension == evidence.degree &&
        isequal(retained.parameter, evidence.parameter) ||
        throw(ArgumentError("Captured correspondence field, degree or parameter disagrees with its panel."))
    context = get(retained, :metadata, NamedTuple())
    isequal(get(context, :interval, nothing), evidence.interval) &&
        get(context, :interval_kind, nothing) === member.kind && get(context, :member, nothing) == member.index ||
        throw(ArgumentError("Captured correspondence interval disagrees with its source record."))
    support_index = findfirst(==(original.target[2]), retained.cell_indices)
    targets = get(source.metadata, :pick_targets, ())
    support_index <= length(targets) && annotation.from.target === nothing &&
        isequal(annotation.from.point, targets[support_index].point) ||
        throw(ArgumentError("Captured source endpoint disagrees with its retained drawing geometry."))
    return annotation
end
