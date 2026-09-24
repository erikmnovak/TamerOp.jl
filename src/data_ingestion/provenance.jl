# Execution evidence is scoped to one ingestion call and inherited by its tasks.
# This records successful branches; it never changes backend selection.
mutable struct _IngestionExecutionTrace
    backends::Vector{Pair{Symbol,Symbol}}
    lock::ReentrantLock
end
_IngestionExecutionTrace() = _IngestionExecutionTrace(Pair{Symbol,Symbol}[], ReentrantLock())
const _INGESTION_EXECUTION_TRACE = ScopedValue{Union{Nothing,_IngestionExecutionTrace}}(nothing)

function _record_ingestion_backend(operation::Symbol, backend::Symbol)
    trace = _INGESTION_EXECUTION_TRACE[]
    trace === nothing && return nothing
    entry = operation => backend
    lock(trace.lock) do
        entry in trace.backends || push!(trace.backends, entry)
    end
    return nothing
end

function _ingestion_construction_provenance(data, spec, filtration)
    requested = spec isa FiltrationSpec ? spec.kind : filtration_kind(typeof(filtration))
    effective = requested
    substitution = :none
    projection = :none
    if data isa PointCloud && !isempty(data.points) && length(first(data.points)) > 2 &&
       requested in (:alpha, :delaunay_lower_star)
        effective = requested === :alpha ? :rips : :function_rips
        projection = requested === :alpha ? :none : :function_coordinate
        substitution = :explicit_highdim_rips
    elseif data isa GradedComplex || data isa MultiCriticalGradedComplex || data isa SimplexTreeMulti
        effective = :provided_graded_complex
    end
    grade_scale = projection === :function_coordinate ? :filtration_values :
                  effective === :alpha ? :squared_radius :
                  effective in (:core, :core_delaunay, :function_delaunay, :rhomboid) ? :radius :
                  effective in (:rips, :function_rips, :rips_density, :rips_codensity, :degree_rips,
                                :landmark_rips) ? :diameter : :filtration_values
    return (; requested, effective, substitution, projection, grade_scale)
end

function _ingestion_provenance(plan, P, axes, orientation, degree, stage, trace;
                               grades, cell_counts)
    params = plan.spec isa FiltrationSpec ? plan.spec.params : NamedTuple()
    construction = merge(_ingestion_construction_provenance(plan.data, plan.spec, plan.filtration),
                         (requested_max_dim=get(params, :max_dim, nothing),
                          retained_max_dim=length(cell_counts) - 1))
    if construction.requested === :rhomboid
        construction = merge(construction, (
            depth_range=get(params, :depth_range, nothing),
            model=get(params, :backend, :auto) === :subdivision_cech ? :subdivision_cech :
                  get(params, :depth_range, nothing) === nothing ? :unsliced_rhomboid : :sliced_rhomboid))
    end
    N = length(axes)
    # Checking membership, rather than merely checking the selected axis policy,
    # distinguishes a genuinely unchanged critical grid from a snapped grid.
    preserves_grades = all(grades) do grade
        all(1:N) do i
            value = orientation[i] == 1 ? grade[i] : -grade[i]
            j = searchsortedfirst(axes[i], value)
            j <= length(axes[i]) && axes[i][j] == value
        end
    end
    recorded = lock(trace.lock) do
        Tuple(sort!(copy(trace.backends); by=x -> (string(first(x)), string(last(x)))))
    end
    custom = !(construction.requested in _BUILTIN_FILTRATION_KINDS)
    effective_backend = isempty(recorded) ? (custom ? :not_recorded : :native) : recorded
    numerical_grades = plan.data isa PointCloud || plan.data isa ImageNd ||
                       plan.data isa GraphData || plan.data isa EmbeddedPlanarGraph2D ||
                       plan.data isa AbstractMatrix
    neighbor_records = filter(x -> first(x) in (:knn_graph, :radius_graph, :knn_distances, :dtm), recorded)
    nn_approximation = custom ? :not_recorded : isempty(neighbor_records) ? :not_used :
                       any(x -> last(x) === :approx, neighbor_records) ? :approximate :
                       any(x -> last(x) === :not_recorded, neighbor_records) ? :not_recorded : :exact_search
    return (
        category=:finite_poset_representations,
        base_poset=P,
        field=plan.field,
        degree=stage === :encoded_complex ? nothing : degree,
        degree_convention=:homological,
        chain_storage=:cochain_reindexed_homology,
        encoding=:oriented_grid,
        window=(lower=ntuple(i -> first(axes[i]), N), upper=ntuple(i -> last(axes[i]), N),
                coordinates=:oriented, outside_lower=:unrepresented, upper_extension=:constant),
        orientation=Tuple(orientation),
        construction,
        discretization=(axes_policy=plan.pipeline.axes_policy,
                        axes_source=get(params, :axes, nothing) === nothing ? :computed_grades : :provided,
                        axis_kind=plan.pipeline.axis_kind,
                        grade_placement=preserves_grades ? :critical_grades : :floor_snapped,
                        sampling=false,
                        quantization=plan.pipeline.eps === nothing ? :none : :nearest_multiple,
                        eps=plan.pipeline.eps),
        approximation=(sparsify=plan.construction.sparsify,
                       input_point_count=plan.data isa PointCloud ? length(plan.data.points) : nothing,
                       retained_vertex_count=isempty(cell_counts) ? 0 : first(cell_counts),
                       radius_cutoff=get(params, :radius, nothing),
                       collapse=plan.construction.collapse,
                       multicritical=plan.multicritical,
                       onecritical_selector=plan.multicritical === :one_critical ? plan.onecritical_selector : nothing,
                       onecritical_enforce_boundary=plan.multicritical === :one_critical ? plan.onecritical_enforce_boundary : nothing,
                       grade_arithmetic=any(a -> eltype(a) <: AlgebraicReal, axes) ? :exact_real_algebraic :
                                        numerical_grades ? :float64 : :input_arithmetic,
                       neighbor_search=nn_approximation,
                       construction_substitution=construction.substitution),
        backend=(requested=(ingestion=:data,
                            neighbors=get(params, :nn_backend, :auto),
                            multicover=get(params, :backend, :auto),
                            delaunay=get(params, :delaunay_backend, :auto)),
                 effective=effective_backend,
                 linear_algebra=:operation_specific),
        reconstruction=preserves_grades ? :computed_graded_complex : :floor_snapped_graded_complex,
    )
end
