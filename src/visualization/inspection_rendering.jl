# Live inspection discovery and rendering contracts; algebra stays in the session.

available_visuals(::_AbstractInspectionSession) = (:linked_inspector,)
_visual_request_keywords(::_AbstractInspectionSession, ::Symbol) = ()

function _append_visual_request_issues!(issues::Vector{String}, session::_AbstractInspectionSession,
                                        ::Symbol; kwargs...)
    inspection_summary(session).closed && push!(issues,
        "This inspection session is closed. Its inspection_snapshot remains available for static export.")
    return issues
end

function _visual_request_cost(session::InspectionSession, ::Symbol)
    return (; work=:linked_stalk_map_rank_section_or_slice, timing=:not_measured,
        cache_reuse=:bounded_session_cache, geometry=:prepared_once_per_session,
        default=:module_dimensions_without_structure_maps,
        basis=:explicit_single_stalk_opt_in,
        map_queries=session.state.view in (:rank_from, :rank_to) ? :anchor_rank_queries_and_selected_pair : :selected_pair_only,
        retained_matrices=:selected_pair_readouts, slice_queries=:explicit_bounded_stratified_restriction,
        rank_queries=:one_anchor_row_or_column_in_rank_views,
        requires_live_julia=true)
end

function _visual_spec(session::InspectionSession, kind::Symbol; kwargs...)
    kind === :linked_inspector || throw(ArgumentError("InspectionSession supports kind=:linked_inspector."))
    isempty(kwargs) || throw(ArgumentError("Change a session with select_inspection!; renderer sizing belongs to visualize."))
    snapshot = inspection_snapshot(session)
    return VisualizationSpec(:linked_inspector;
        title="Explore the finite encoding", subtitle="Live selection with Julia",
        panels=snapshot.panels,
        interaction=(; hover=true, clicks=true, labels=true,
            widgets=(:inspection_selection, :inspection_view, :inspection_basis, :inspection_slice),
            notebook=:linked_inspector, mode=:live_julia,
            requires_live_julia=true, offline_widgets=false),
        metadata=(; session, category=:finite_poset_representations,
            selected_fibers_only=!(inspection_selection(session).view in (:rank_from, :rank_to))))
end

function _check_linked_inspection!(issues::Vector{String}, spec::VisualizationSpec)
    session = get(spec.metadata, :session, nothing)
    if !(session isa _AbstractInspectionSession)
        push!(issues, "A linked inspector requires an inspection session.")
        return issues
    end
    inspection_summary(session).closed && push!(issues,
        "A live inspector cannot use a closed session; use its static inspection_snapshot.")
    get(spec.interaction, :mode, nothing) === :live_julia &&
        get(spec.interaction, :requires_live_julia, false) &&
        !get(spec.interaction, :offline_widgets, true) ||
        push!(issues, "A linked inspector requires live Julia and has no offline callbacks.")
    get(spec.interaction, :notebook, nothing) === :linked_inspector ||
        push!(issues, "A linked inspector must identify its live notebook renderer.")
    expected_widgets = session isa MatchingInspectionSession ? (:matching_pair,:matching_slice,:matching_optimum) : session isa IntervalInspectionSession ?
        (:interval_selection, :interval_member, :interval_representative) :
        (:inspection_selection, :inspection_view, :inspection_basis, :inspection_slice)
    get(spec.interaction, :widgets, ()) == expected_widgets ||
        push!(issues, "Unsupported linked-inspector widget contract.")
    return issues
end

function _visual_request_cost(session::IntervalInspectionSession, ::Symbol)
    return (; work=:retained_intervals_and_selected_representative, timing=:not_measured,
        default=:interval_metadata, representative=:explicit_retained_member_opt_in,
        requires_live_julia=true)
end

function _visual_spec(session::IntervalInspectionSession, kind::Symbol; kwargs...)
    kind === :linked_inspector || throw(ArgumentError("IntervalInspectionSession supports kind=:linked_inspector."))
    isempty(kwargs) || throw(ArgumentError("Change a session with select_inspection!; renderer sizing belongs to visualize."))
    snapshot = inspection_snapshot(session)
    return VisualizationSpec(:linked_inspector; title="Explore persistence intervals",
        subtitle="Live interval selection with Julia", panels=snapshot.panels,
        interaction=(; hover=true, clicks=true, labels=true,
            widgets=(:interval_selection,:interval_member,:interval_representative),
            notebook=:linked_inspector, mode=:live_julia,
            requires_live_julia=true, offline_widgets=false),
        metadata=(; session, inspector_kind=:intervals))
end

function _visual_spec(session::MatchingInspectionSession,kind::Symbol;kwargs...)
    snapshot=inspection_snapshot(session)
    return VisualizationSpec(:linked_inspector;title="Explore a matching witness",panels=snapshot.panels,
        interaction=(;hover=true,clicks=true,labels=true,widgets=(:matching_pair,:matching_slice,:matching_optimum),
            notebook=:linked_inspector,mode=:live_julia,requires_live_julia=true,offline_widgets=false),
        metadata=(;session,inspector_kind=:matching))
end
_visual_request_cost(::MatchingInspectionSession,::Symbol) = (;work=:selected_matching,
    selection=:retained_witness,optimizer=:explicit_opt_in,requires_live_julia=true,timing=:not_measured)
