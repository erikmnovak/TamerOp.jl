# Owned dataset and pipeline JSON formats.

# -----------------------------------------------------------------------------
# A0) Datasets + pipeline specs (Workflow)
# -----------------------------------------------------------------------------

# Nonempty coordinate rows determine their arity and encoded scalar kind.
# Empty graded datasets need that information explicitly; it is mathematical
# shape data, not a Julia type name to evaluate while loading.
function _store_dataset_grade_rows!(obj, key::String, rows::AbstractVector{NTuple{N,T}}) where {N,T}
    isempty(rows) || return _store_coordinate_rows!(obj, key, rows)
    N > 0 || throw(ArgumentError("Empty graded datasets need a positive parameter dimension."))
    kind = if T <: Union{Float16,Float32,Float64}
        "float64"
    elseif T <: AlgebraicReal
        "algebraic_real"
    elseif T <: Union{Integer,Rational,BigFloat}
        "rational"
    else
        throw(ArgumentError("Unsupported empty grade coordinate type: $T"))
    end
    obj[key] = []
    obj["empty_grade_type"] = Dict("parameter_dim" => N, "scalar_kind" => kind)
    return obj
end

function _dataset_grade_rows(obj, key::String)
    obj[key] isa AbstractVector || throw(ArgumentError("Dataset $key must be an array of coordinate rows."))
    rows = _coordinate_rows_from_obj(obj[key], get(obj, "exact_" * key, nothing))
    if isempty(rows)
        metadata = get(obj, "empty_grade_type", nothing)
        metadata isa AbstractDict && length(metadata) == 2 &&
            haskey(metadata, "parameter_dim") && haskey(metadata, "scalar_kind") ||
            throw(ArgumentError("Empty graded datasets require empty_grade_type with parameter_dim and scalar_kind."))
        get(obj, "exact_" * key, nothing) === nothing ||
            throw(ArgumentError("Empty grades use empty_grade_type, not an additional exact-coordinate payload."))
        dimension = metadata["parameter_dim"]
        dimension isa Integer && !(dimension isa Bool) && 0 < dimension <= typemax(Int) ||
            throw(ArgumentError("empty_grade_type.parameter_dim must be a positive integer."))
        kind = metadata["scalar_kind"]
        kind isa AbstractString || throw(ArgumentError("empty_grade_type.scalar_kind must be a string."))
        T = if kind == "float64"
            Float64
        elseif kind == "rational"
            QQ
        elseif kind == "algebraic_real"
            AlgebraicReal
        else
            throw(ArgumentError("Unknown empty grade scalar_kind: $kind"))
        end
        return NTuple{Int(dimension),T}[]
    end
    haskey(obj, "empty_grade_type") && throw(ArgumentError(
        "empty_grade_type is only valid when the graded dataset has no grade rows."))
    return rows
end

function _obj_from_dataset(data)
    if data isa PointCloud
        pts = point_matrix(data)
        npts, d = size(pts)
        obj = Dict{String,Any}("kind" => "PointCloud",
                    "layout" => _DATASET_COLUMN_LAYOUT,
                    "n" => npts,
                    "d" => d,
                    "points_flat" => vec(pts))
        return _store_coordinate_vector!(obj, "points_flat", vec(pts))
    elseif data isa ImageNd
        obj = Dict{String,Any}("kind" => "ImageNd",
                    "size" => collect(size(data.data)),
                    "data" => collect(vec(data.data)))
        return _store_coordinate_vector!(obj, "data", vec(data.data))
    elseif data isa GraphData
        edges_u, edges_v = edge_columns(data)
        coords_dim = nothing
        coords_flat = nothing
        coords = coord_matrix(data)
        if coords !== nothing
            ncoords, d = size(coords)
            ncoords == data.n || error("GraphData coords row count must equal n for columnar serialization.")
            coords_dim = d
            coords_flat = vec(coords)
        end
        obj = Dict{String,Any}("kind" => "GraphData",
                    "layout" => _DATASET_COLUMN_LAYOUT,
                    "n" => data.n,
                    "edges_u" => edges_u,
                    "edges_v" => edges_v,
                    "coords_dim" => coords_dim,
                    "coords_flat" => coords_flat,
                    "weights" => getfield(data, :weights))
        coords_flat === nothing || _store_coordinate_vector!(obj, "coords_flat", coords_flat)
        data.weights === nothing || _store_coordinate_vector!(obj, "weights", data.weights)
        return obj
    elseif data isa EmbeddedPlanarGraph2D
        obj = Dict{String,Any}("kind" => "EmbeddedPlanarGraph2D",
                    "vertices" => [collect(v) for v in data.vertices],
                    "edges" => [collect(e) for e in data.edges],
                    "polylines" => data.polylines === nothing ? nothing : _coordinate_parameter_obj([[collect(p) for p in poly] for poly in data.polylines]),
                    "bbox" => data.bbox === nothing ? nothing : _coordinate_parameter_obj(collect(data.bbox)))
        return _store_coordinate_rows!(obj, "vertices", data.vertices)
    elseif data isa GradedComplex
        bnds = Any[]
        for B in data.boundaries
            Ii, Jj, Vv = findnz(B)
            push!(bnds, Dict(
                "m" => size(B, 1),
                "n" => size(B, 2),
                "I" => collect(Ii),
                "J" => collect(Jj),
                "V" => collect(Vv),
            ))
        end
        obj = Dict{String,Any}("kind" => "GradedComplex",
                    "cells_by_dim" => [collect(c) for c in data.cells_by_dim],
                    "boundaries" => bnds,
                    "grades" => [collect(g) for g in data.grades],
                    "cell_dims" => collect(data.cell_dims))
        return _store_dataset_grade_rows!(obj, "grades", data.grades)
    elseif data isa MultiCriticalGradedComplex
        bnds = Any[]
        for B in data.boundaries
            Ii, Jj, Vv = findnz(B)
            push!(bnds, Dict(
                "m" => size(B, 1),
                "n" => size(B, 2),
                "I" => collect(Ii),
                "J" => collect(Jj),
                "V" => collect(Vv),
            ))
        end
        obj = Dict{String,Any}("kind" => "MultiCriticalGradedComplex",
                    "cells_by_dim" => [collect(c) for c in data.cells_by_dim],
                    "boundaries" => bnds,
                    "grades" => [[collect(g) for g in gs] for gs in data.grades],
                    "cell_dims" => collect(data.cell_dims))
        exact = _exact_coordinate_rows(data.grade_data)
        if exact !== nothing
            obj["grades"] = []
            obj["exact_grades"] = exact
            obj["grade_offsets"] = collect(data.grade_offsets)
        end
        return obj
    elseif data isa SimplexTreeMulti
        obj = Dict{String,Any}("kind" => "SimplexTreeMulti",
                    "simplex_offsets" => collect(data.simplex_offsets),
                    "simplex_vertices" => collect(data.simplex_vertices),
                    "simplex_dims" => collect(data.simplex_dims),
                    "dim_offsets" => collect(data.dim_offsets),
                    "grade_offsets" => collect(data.grade_offsets),
                    "grade_data" => [collect(g) for g in data.grade_data])
        return _store_dataset_grade_rows!(obj, "grade_data", data.grade_data)
    else
        error("Unsupported dataset type for serialization.")
    end
end

function _dataset_from_obj(obj)
    kind = String(obj["kind"])
    if kind == "PointCloud"
        haskey(obj, "points_flat") || error("PointCloud JSON missing canonical `points_flat` payload.")
        haskey(obj, "layout") || error("PointCloud JSON missing canonical `layout` payload.")
        haskey(obj, "n") || error("PointCloud JSON missing canonical `n` payload.")
        haskey(obj, "d") || error("PointCloud JSON missing canonical `d` payload.")
        _require_dataset_layout(String(obj["layout"]), kind)
        n = Int(obj["n"])
        d = Int(obj["d"])
        return _pointcloud_from_flat(n, d, _coordinate_vector_from_obj(obj["points_flat"], get(obj, "exact_points_flat", nothing)))
    elseif kind == "ImageNd"
        sz = Vector{Int}(obj["size"])
        flat = _coordinate_vector_from_obj(obj["data"], get(obj, "exact_data", nothing))
        data = reshape(flat, Tuple(sz))
        return ImageNd(data)
    elseif kind == "GraphData"
        haskey(obj, "n") || error("GraphData JSON missing canonical `n` payload.")
        n = Int(obj["n"])
        obj["weights"] === nothing && get(obj, "exact_weights", nothing) !== nothing && error("exact_weights requires an empty weights array.")
        weights = obj["weights"] === nothing ? nothing : _coordinate_vector_from_obj(obj["weights"], get(obj, "exact_weights", nothing))
        haskey(obj, "edges_u") || error("GraphData JSON missing canonical `edges_u` payload.")
        haskey(obj, "edges_v") || error("GraphData JSON missing canonical `edges_v` payload.")
        haskey(obj, "layout") || error("GraphData JSON missing canonical `layout` payload.")
        _require_dataset_layout(String(obj["layout"]), kind)
        coords_dim = haskey(obj, "coords_dim") && obj["coords_dim"] !== nothing ? Int(obj["coords_dim"]) : nothing
        coords_flat = haskey(obj, "coords_flat") && obj["coords_flat"] !== nothing ?
            _coordinate_vector_from_obj(obj["coords_flat"], get(obj, "exact_coords_flat", nothing)) : nothing
        coords_flat === nothing && get(obj, "exact_coords_flat", nothing) !== nothing && error("exact_coords_flat requires an empty coords_flat array.")
        return _graph_from_columns(n,
                                   Vector{Int}(obj["edges_u"]),
                                   Vector{Int}(obj["edges_v"]);
                                   coords_dim=coords_dim,
                                   coords_flat=coords_flat,
                                   weights=weights)
    elseif kind == "EmbeddedPlanarGraph2D"
        verts = _coordinate_rows_from_obj(obj["vertices"], get(obj, "exact_vertices", nothing))
        edges = [ (Int(e[1]), Int(e[2])) for e in obj["edges"] ]
        polylines = obj["polylines"] === nothing ? nothing :
            _coordinate_parameter_from_obj(obj["polylines"])
        T = eltype(eltype(verts))
        bbox = obj["bbox"] === nothing ? nothing : Tuple(T.(_coordinate_parameter_from_obj(obj["bbox"])))
        return EmbeddedPlanarGraph2D(verts, edges; polylines=polylines, bbox=bbox)
    elseif kind == "GradedComplex"
        cells = [Vector{Int}(c) for c in obj["cells_by_dim"]]
        boundaries = SparseMatrixCSC{Int,Int}[]
        for b in obj["boundaries"]
            m = Int(b["m"]); n = Int(b["n"])
            I = Vector{Int}(b["I"])
            J = Vector{Int}(b["J"])
            V = Vector{Int}(b["V"])
            push!(boundaries, sparse(I, J, V, m, n))
        end
        grades = _dataset_grade_rows(obj, "grades")
        cell_dims = Vector{Int}(obj["cell_dims"])
        return GradedComplex(cells, boundaries, grades; cell_dims=cell_dims)
    elseif kind == "MultiCriticalGradedComplex"
        cells = [Vector{Int}(c) for c in obj["cells_by_dim"]]
        boundaries = SparseMatrixCSC{Int,Int}[]
        for b in obj["boundaries"]
            m = Int(b["m"]); n = Int(b["n"])
            I = Vector{Int}(b["I"])
            J = Vector{Int}(b["J"])
            V = Vector{Int}(b["V"])
            push!(boundaries, sparse(I, J, V, m, n))
        end
        grades = if get(obj, "exact_grades", nothing) === nothing
            [[Vector{Float64}(g) for g in gs] for gs in obj["grades"]]
        else
            flat = _coordinate_rows_from_obj(obj["grades"], obj["exact_grades"])
            offsets = Int.(obj["grade_offsets"])
            length(offsets) == sum(length, cells) + 1 && first(offsets) == 1 &&
                last(offsets) == length(flat) + 1 && issorted(offsets) ||
                throw(ArgumentError("multicritical grade_offsets must partition the exact grades by cell"))
            [flat[offsets[i]:(offsets[i + 1] - 1)] for i in 1:(length(offsets) - 1)]
        end
        cell_dims = Vector{Int}(obj["cell_dims"])
        return MultiCriticalGradedComplex(cells, boundaries, grades; cell_dims=cell_dims)
    elseif kind == "SimplexTreeMulti"
        simplex_offsets = Vector{Int}(obj["simplex_offsets"])
        simplex_vertices = Vector{Int}(obj["simplex_vertices"])
        simplex_dims = Vector{Int}(obj["simplex_dims"])
        dim_offsets = Vector{Int}(obj["dim_offsets"])
        grade_offsets = Vector{Int}(obj["grade_offsets"])
        raw_grades = _dataset_grade_rows(obj, "grade_data")
        grade_data = if isempty(raw_grades)
            raw_grades # The empty-shape metadata already supplies NTuple{N,T}.
        else
            N, T = length(first(raw_grades)), eltype(first(raw_grades))
            typed = Vector{NTuple{N,T}}(undef, length(raw_grades))
            for i in eachindex(raw_grades)
                g = raw_grades[i]
                length(g) == N || error("SimplexTreeMulti JSON grade arity mismatch at index $i.")
                typed[i] = ntuple(k -> T(g[k]), N)
            end
            typed
        end
        return SimplexTreeMulti(simplex_offsets, simplex_vertices, simplex_dims,
                                dim_offsets, grade_offsets, grade_data)
    else
        error("Unknown dataset kind: $kind")
    end
end

@inline function _construction_budget_obj(b::ConstructionBudget)
    return Dict(
        "max_simplices" => b.max_simplices,
        "max_edges" => b.max_edges,
        "memory_budget_bytes" => b.memory_budget_bytes,
    )
end

@inline function _construction_options_obj(c::ConstructionOptions)
    return Dict(
        "sparsify" => String(c.sparsify),
        "collapse" => String(c.collapse),
        "output_stage" => String(c.output_stage),
        "budget" => _construction_budget_obj(c.budget),
    )
end

function _spec_obj(spec::FiltrationSpec)
    params = Dict{String,Any}()
    for (k, v) in pairs(spec.params)
        if k == :construction
            if v isa ConstructionOptions
                params["construction"] = _construction_options_obj(v)
            elseif v isa ConstructionBudget
                params["construction"] = Dict("budget" => _construction_budget_obj(v))
            else
                params["construction"] = v
            end
        elseif k == :field
            params["field"] = v === nothing ? nothing : _field_to_obj(v)
        else
            params[String(k)] = _coordinate_parameter_obj(v)
        end
    end
    return Dict("kind" => String(spec.kind), "params" => params)
end

# Arrays represent canonical per-axis tuples at the JSON boundary. Validate
# before promotion so [true, 0.5] cannot become an accepted numeric step tuple.
function _pipeline_eps_from_obj(raw)
    if raw isa AbstractVector && any(value -> value isa Bool, raw)
        throw(ArgumentError("pipeline eps must contain positive real steps, not booleans."))
    end
    decoded = _coordinate_parameter_from_obj(raw)
    steps = decoded isa AbstractVector ? Tuple(decoded) : decoded
    return PipelineOptions(eps=steps).eps
end

function _spec_from_obj(obj)
    kind = Symbol(String(obj["kind"]))
    params_obj = obj["params"]
    construction = get(params_obj, "construction", nothing)
    if construction !== nothing
        construction isa AbstractDict ||
            throw(ArgumentError("pipeline spec construction must be a JSON object."))
        # Validate the canonical selectors at the serialized boundary. Retain
        # the serialized parameter representation for subsequent spec decoding.
        ConstructionOptions(;
            sparsify=Symbol(get(construction, "sparsify", "none")),
            collapse=Symbol(get(construction, "collapse", "none")),
            output_stage=Symbol(get(construction, "output_stage", "encoding_result")),
        )
    end
    # Validate before generic numeric promotion can turn Bool/Int mixtures
    # into an integer vector. Serialized selectors retain the in-memory contract.
    orientation = if haskey(params_obj, "orientation")
        raw = params_obj["orientation"]
        (raw === nothing || raw isa AbstractVector) ||
            throw(ArgumentError("pipeline orientation must be null or an array of integer signs."))
        PipelineOptions(orientation=raw === nothing ? nothing : Tuple(raw)).orientation
    else
        nothing
    end
    params = (; (Symbol(k) => _coordinate_parameter_from_obj(params_obj[k]) for k in keys(params_obj))...)
    if haskey(params_obj, "orientation")
        params = merge(params, (orientation=orientation,))
    end
    if haskey(params_obj, "eps")
        params = merge(params, (eps=_pipeline_eps_from_obj(params_obj["eps"]),))
    end
    if haskey(params_obj, "field")
        raw = params_obj["field"]
        (raw === nothing || raw isa AbstractDict) ||
            throw(ArgumentError("pipeline field must be null or a coefficient-field JSON object."))
        params = merge(params, (field=raw === nothing ? nothing : _field_from_obj(raw),))
    end
    if kind === :rhomboid
        if haskey(params_obj, "depth_range")
            # Validate before generic numeric-vector promotion: promotion would
            # erase the distinction between a Boolean and an integer endpoint.
            raw = params_obj["depth_range"]
            depths = if raw === nothing
                nothing
            else
                raw isa AbstractVector && length(raw) == 2 &&
                    all(k -> k isa Integer && !(k isa Bool), raw) && 0 <= raw[1] <= raw[2] <= typemax(Int) ||
                    throw(ArgumentError("rhomboid depth_range must be a JSON array of two nonnegative increasing integer endpoints representable as Int"))
                (Int(raw[1]), Int(raw[2]))
            end
            params = merge(params, (depth_range=depths,))
        end
        if haskey(params_obj, "backend")
            backend = params_obj["backend"]
            backend isa AbstractString && backend in ("auto", "exhaustive", "incremental", "subdivision_cech") ||
                throw(ArgumentError("unknown rhomboid backend in pipeline JSON"))
            params = merge(params, (backend=Symbol(backend),))
        end
    end
    return FiltrationSpec(; kind=kind, params...)
end

function _pipeline_options_from_spec(spec::FiltrationSpec)
    p = spec.params
    return PipelineOptions(;
        orientation = get(p, :orientation, nothing),
        axes_policy = Symbol(get(p, :axes_policy, :encoding)),
        axis_kind = get(p, :axis_kind, nothing),
        eps = get(p, :eps, nothing),
        poset_kind = Symbol(get(p, :poset_kind, :signature)),
        field = get(p, :field, nothing),
        max_axis_len = get(p, :max_axis_len, nothing),
    )
end

function _pipeline_options_from_any(spec::FiltrationSpec, x)
    if x === nothing
        return _pipeline_options_from_spec(spec)
    elseif x isa PipelineOptions
        return x
    elseif x isa NamedTuple
        return PipelineOptions(; x...)
    elseif x isa AbstractDict
        vals = (; (Symbol(k) => x[k] for k in keys(x))...)
        return PipelineOptions(; vals...)
    end
    throw(ArgumentError("pipeline_opts must be nothing, PipelineOptions, NamedTuple, or AbstractDict."))
end

function _pipeline_options_obj(opts::PipelineOptions)
    return Dict(
        "orientation" => opts.orientation,
        "axes_policy" => String(opts.axes_policy),
        "axis_kind" => opts.axis_kind,
        "eps" => _coordinate_parameter_obj(opts.eps),
        "poset_kind" => String(opts.poset_kind),
        "field" => opts.field === nothing ? nothing : _field_to_obj(opts.field),
        "max_axis_len" => opts.max_axis_len,
    )
end

function _pipeline_options_from_obj(obj)::PipelineOptions
    orient_raw = get(obj, "orientation", nothing)
    orientation = if orient_raw isa AbstractVector
        Tuple(orient_raw)
    else
        orient_raw
    end
    axis_kind_raw = get(obj, "axis_kind", nothing)
    axis_kind = axis_kind_raw isa AbstractString ? Symbol(axis_kind_raw) : axis_kind_raw
    field_raw = get(obj, "field", nothing)
    (field_raw === nothing || field_raw isa AbstractDict) ||
        throw(ArgumentError("pipeline field must be null or a coefficient-field JSON object."))
    field = field_raw === nothing ? nothing : _field_from_obj(field_raw)
    quantization = _pipeline_eps_from_obj(get(obj, "eps", nothing))
    return PipelineOptions(;
        orientation = orientation,
        axes_policy = Symbol(get(obj, "axes_policy", "encoding")),
        axis_kind = axis_kind,
        eps = quantization,
        poset_kind = Symbol(get(obj, "poset_kind", "signature")),
        field = field,
        max_axis_len = get(obj, "max_axis_len", nothing),
    )
end

"""
    save_dataset_json(path, data; profile=:compact, pretty=nothing)

Serialize a dataset into the stable TamerOp-owned dataset schema.

Exact coordinates and grades use tagged rational or real-algebraic payloads.
Algebraic values store an integer minimal polynomial and its ordered real-root
index; displayed floating approximations are never authoritative. Ordinary
floating coordinates use Float64 JSON arrays (Float16/Float32 are widened
without changing their represented values).

Empty `GradedComplex` and `SimplexTreeMulti` datasets retain their parameter
dimension and scalar coordinate contract in `empty_grade_type`. Its canonical
`scalar_kind` is `"float64"`, `"rational"`, or `"algebraic_real"`; no arbitrary
Julia type is reconstructed. Nonempty grade rows retain their existing format.

This is the canonical owned write path for datasets such as `PointCloud`,
`GraphData`, `ImageNd`, `EmbeddedPlanarGraph2D`, `GradedComplex`,
`MultiCriticalGradedComplex`, and `SimplexTreeMulti`.

This is not the cheap inspection path. Use [`dataset_json_summary`](@ref) or
[`inspect_json`](@ref) to inspect an existing artifact cheaply, and use
[`check_dataset_json`](@ref) when you need strict schema validation before
calling [`load_dataset_json`](@ref).

Use `profile=:compact` (default) for compact writes and `profile=:debug` for a
pretty-printed artifact.
"""
function save_dataset_json(path::AbstractString, data;
                           profile::Symbol=:compact,
                           pretty::Union{Nothing,Bool}=nothing)
    return _json_write(path, _obj_from_dataset(data);
                       pretty=_resolve_owned_json_pretty(profile, pretty))
end

"""
    load_dataset_json(path; validation=:strict)

Load a dataset serialized by [`save_dataset_json`](@ref).

This is a strict owned-schema loader. Prefer [`dataset_json_summary`](@ref) for
cheap-first inspection and [`check_dataset_json`](@ref) when you need explicit
validation before loading an existing artifact.

Use `validation=:trusted` only for TamerOp-produced files that already
passed schema validation and need the lighter hot load path.
"""
function load_dataset_json(path::AbstractString; validation::Symbol=:strict)
    validation === :strict && return _load_dataset_json_strict(path)
    validation === :trusted && return _load_dataset_json_trusted(path)
    _resolve_validation_mode(validation)
    error("unreachable validation mode")
end

"""
    save_pipeline_json(path, data, spec; degree=nothing, pipeline_opts=nothing, profile=:compact, pretty=nothing)

Serialize a dataset, filtration spec, degree, and structured `PipelineOptions`
into the stable TamerOp-owned pipeline schema.

This is the canonical owned artifact for replaying a workflow setup. Use
[`pipeline_json_summary`](@ref) or [`inspect_json`](@ref) to inspect an
existing artifact cheaply before deciding whether to validate or load it.

Use `profile=:compact` (default) for compact writes and `profile=:debug` for a
pretty-printed artifact.
"""
function save_pipeline_json(path::AbstractString, data, spec::FiltrationSpec;
                            degree=nothing,
                            pipeline_opts=nothing,
                            profile::Symbol=:compact,
                            pretty::Union{Nothing,Bool}=nothing)
    popts = _pipeline_options_from_any(spec, pipeline_opts)
    obj = Dict(
        "schema_version" => PIPELINE_SCHEMA_VERSION,
        "dataset" => _obj_from_dataset(data),
        "spec" => _spec_obj(spec),
        "degree" => degree,
        "pipeline_options" => _pipeline_options_obj(popts),
    )
    return _json_write(path, obj; pretty=_resolve_owned_json_pretty(profile, pretty))
end

"""
    load_pipeline_json(path; validation=:strict) -> (data, spec, degree, pipeline_opts)

Load a pipeline artifact written by [`save_pipeline_json`](@ref).

This is a strict owned-schema loader returning the dataset, filtration spec,
degree, and `PipelineOptions`. Prefer [`pipeline_json_summary`](@ref) for a
cheap-first family check and [`check_pipeline_json`](@ref) for explicit schema
validation before loading.

Use `validation=:trusted` only for TamerOp-produced artifacts when you
want the lighter replay path and are willing to trust the stored schema.
"""
function load_pipeline_json(path::AbstractString; validation::Symbol=:strict)
    validation === :strict && return _load_pipeline_json_strict(path)
    validation === :trusted && return _load_pipeline_json_trusted(path)
    _resolve_validation_mode(validation)
    error("unreachable validation mode")
end
