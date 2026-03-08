# =============================================================================
# DataFileIO.jl
#
# File-oriented dataset ingestion adapters.
# Converts on-disk formats into typed ingestion objects used by DataIngestion.
# =============================================================================

module DataFileIO

using ..DataTypes: PointCloud, GraphData, ImageNd, GradedComplex
using ..Options: ConstructionOptions, DataFileOptions
import ..Serialization

const _MISSING_STRINGS = Set(["", "na", "nan", "null", "missing"])

@inline function _resolve_file_kind(kind::Symbol, opts::DataFileOptions)::Symbol
    if kind == :auto
        return opts.kind
    end
    return kind
end

@inline function _infer_file_format(path::AbstractString)::Symbol
    ext = lowercase(splitext(path)[2])
    if ext == ".json"
        return :dataset_json
    elseif ext == ".csv"
        return :csv
    elseif ext == ".tsv"
        return :tsv
    elseif ext == ".txt"
        return :txt
    elseif ext == ".bin"
        return :ripser_binary_lower_distance
    end
    throw(ArgumentError("load_data: could not infer format from extension $(ext). Pass format explicitly."))
end

@inline function _resolve_file_format(format::Symbol, path::AbstractString, opts::DataFileOptions)::Symbol
    f = format == :auto ? opts.format : format
    return f == :auto ? _infer_file_format(path) : f
end

@inline function _delimiter_for_format(fmt::Symbol, opts::DataFileOptions)
    if opts.delimiter !== nothing
        return opts.delimiter
    elseif fmt == :csv
        return ','
    elseif fmt == :tsv
        return '\t'
    elseif fmt == :txt
        return nothing
    end
    return nothing
end

@inline function _is_missing_token(tok::AbstractString)::Bool
    return lowercase(strip(tok)) in _MISSING_STRINGS
end

@inline function _split_line(line::AbstractString, delim)
    if delim === nothing
        return split(strip(line))
    end
    return [strip(t) for t in split(line, delim)]
end

@inline function _all_numeric_tokens(tokens::Vector{String})::Bool
    @inbounds for t in tokens
        tryparse(Float64, t) === nothing && return false
    end
    return true
end

@inline function _sanitize_name(s::AbstractString)::Symbol
    str = strip(String(s))
    isempty(str) && return Symbol("_")
    str = replace(str, r"\s+" => "_")
    return Symbol(str)
end

@inline function _uniquify_names(names::Vector{Symbol})
    seen = Dict{Symbol,Int}()
    out = Vector{Symbol}(undef, length(names))
    @inbounds for i in eachindex(names)
        nm = names[i]
        k = get!(seen, nm, 0) + 1
        seen[nm] = k
        out[i] = k == 1 ? nm : Symbol(string(nm), "_", k)
    end
    return out
end

@inline function _parse_delimited_table(path::AbstractString,
                                        fmt::Symbol,
                                        opts::DataFileOptions)
    delim = _delimiter_for_format(fmt, opts)
    rows = Vector{Vector{String}}()
    open(path, "r") do io
        for raw in eachline(io)
            line = strip(raw)
            isempty(line) && continue
            if opts.comment_prefix !== nothing
                startswith(line, string(opts.comment_prefix)) && continue
            end
            toks = _split_line(line, delim)
            isempty(toks) && continue
            push!(rows, toks)
        end
    end
    isempty(rows) && throw(ArgumentError("load_data: file has no data rows after filtering comments/empty lines: $(path)"))
    width = length(rows[1])
    @inbounds for i in 2:length(rows)
        length(rows[i]) == width || throw(ArgumentError("load_data: non-rectangular delimited table at row $(i)."))
    end

    header_used = if opts.header === nothing
        !_all_numeric_tokens(rows[1])
    else
        Bool(opts.header)
    end

    names = if header_used
        _uniquify_names([_sanitize_name(t) for t in rows[1]])
    else
        [Symbol("x", i) for i in 1:width]
    end
    data_rows = header_used ? rows[2:end] : rows
    isempty(data_rows) && throw(ArgumentError("load_data: no data rows after header in $(path)."))
    return names, data_rows, header_used
end

@inline function _col_selector_vec(x)
    if x === nothing
        return nothing
    elseif x isa Tuple
        return collect(x)
    elseif x isa AbstractVector
        return collect(x)
    else
        return Any[x]
    end
end

@inline function _column_index(names::Vector{Symbol}, key)::Int
    if key isa Integer
        idx = Int(key)
        1 <= idx <= length(names) || throw(ArgumentError("column index $(idx) is out of bounds 1:$(length(names))."))
        return idx
    elseif key isa Symbol
        idx = findfirst(==(key), names)
        idx === nothing && throw(ArgumentError("column $(key) not found. Available columns: $(names)."))
        return idx
    elseif key isa AbstractString
        return _column_index(names, Symbol(key))
    end
    throw(ArgumentError("unsupported column selector type $(typeof(key)); use Int/Symbol/String."))
end

@inline function _parse_float_token(tok::AbstractString)
    _is_missing_token(tok) && return nothing
    v = tryparse(Float64, tok)
    v === nothing && throw(ArgumentError("non-numeric token encountered: '$(tok)'"))
    return v
end

@inline function _parse_int_token(tok::AbstractString)
    _is_missing_token(tok) && return nothing
    vi = tryparse(Int, tok)
    if vi !== nothing
        return vi
    end
    vf = tryparse(Float64, tok)
    vf === nothing && throw(ArgumentError("non-integer token encountered: '$(tok)'"))
    isinteger(vf) || throw(ArgumentError("expected integer token but found non-integer value: '$(tok)'"))
    return Int(round(vf))
end

@inline function _drop_row_on_missing(policy::Symbol)::Bool
    policy == :drop_rows && return true
    policy == :error && return false
    throw(ArgumentError("unsupported missing_policy $(policy)."))
end

function _load_point_cloud_table(path::AbstractString, fmt::Symbol, opts::DataFileOptions)
    names, rows, _ = _parse_delimited_table(path, fmt, opts)
    cols_raw = _col_selector_vec(opts.cols)
    col_idx = if cols_raw === nothing
        collect(1:length(names))
    else
        [_column_index(names, c) for c in cols_raw]
    end
    isempty(col_idx) && throw(ArgumentError("point-cloud parsing requires at least one coordinate column."))
    points = Vector{Vector{Float64}}()
    drop_missing = _drop_row_on_missing(opts.missing_policy)
    @inbounds for row in rows
        p = Vector{Float64}(undef, length(col_idx))
        keep = true
        for j in eachindex(col_idx)
            v = _parse_float_token(row[col_idx[j]])
            if v === nothing
                if drop_missing
                    keep = false
                    break
                end
                throw(ArgumentError("missing value in point-cloud row with missing_policy=:error."))
            end
            p[j] = v
        end
        keep && push!(points, p)
    end
    isempty(points) && throw(ArgumentError("no point rows survived parsing."))
    return PointCloud(points)
end

function _load_graph_table(path::AbstractString, fmt::Symbol, opts::DataFileOptions)
    names, rows, _ = _parse_delimited_table(path, fmt, opts)
    u_idx = if opts.u_col == :u && !(:u in names)
        1
    else
        _column_index(names, opts.u_col)
    end
    v_idx = if opts.v_col == :v && !(:v in names)
        2
    else
        _column_index(names, opts.v_col)
    end
    length(names) >= 2 || throw(ArgumentError("graph parsing requires at least two columns for u/v."))
    w_idx = opts.weight_col === nothing ? nothing : _column_index(names, opts.weight_col)
    edges = Tuple{Int,Int}[]
    weights = w_idx === nothing ? nothing : Float64[]
    drop_missing = _drop_row_on_missing(opts.missing_policy)
    maxv = 0
    @inbounds for row in rows
        u = _parse_int_token(row[u_idx])
        v = _parse_int_token(row[v_idx])
        if u === nothing || v === nothing
            if drop_missing
                continue
            end
            throw(ArgumentError("missing u/v in graph row with missing_policy=:error."))
        end
        u >= 1 || throw(ArgumentError("graph edges must be 1-based positive indices (got u=$(u))."))
        v >= 1 || throw(ArgumentError("graph edges must be 1-based positive indices (got v=$(v))."))
        push!(edges, (u, v))
        if w_idx !== nothing
            w = _parse_float_token(row[w_idx])
            if w === nothing
                if drop_missing
                    pop!(edges)
                    continue
                end
                throw(ArgumentError("missing weight in graph row with missing_policy=:error."))
            end
            push!(weights, w)
        end
        maxv = max(maxv, u, v)
    end
    isempty(edges) && throw(ArgumentError("no graph edges survived parsing."))
    return GraphData(maxv, edges; weights=weights, T=Float64)
end

function _load_image_table(path::AbstractString, fmt::Symbol, opts::DataFileOptions)
    _, rows, _ = _parse_delimited_table(path, fmt, opts)
    m = length(rows)
    n = length(rows[1])
    A = Matrix{Float64}(undef, m, n)
    drop_missing = _drop_row_on_missing(opts.missing_policy)
    kept = 0
    @inbounds for i in 1:m
        if drop_missing
            has_missing = false
            for j in 1:n
                _is_missing_token(rows[i][j]) && (has_missing = true; break)
            end
            has_missing && continue
        end
        kept += 1
        for j in 1:n
            v = _parse_float_token(rows[i][j])
            if v === nothing
                throw(ArgumentError("missing value in image row with missing_policy=:error."))
            end
            A[kept, j] = v
        end
    end
    kept > 0 || throw(ArgumentError("no image rows survived parsing."))
    return ImageNd(A[1:kept, :])
end

function _load_distance_matrix_table(path::AbstractString, fmt::Symbol, opts::DataFileOptions;
                                     max_dim::Int=1,
                                     radius::Union{Nothing,Real}=nothing,
                                     knn::Union{Nothing,Int}=nothing,
                                     construction::ConstructionOptions=ConstructionOptions(),
                                     check_symmetric::Bool=true,
                                     symmetry_tol::Real=1.0e-10)
    _, rows, _ = _parse_delimited_table(path, fmt, opts)
    n = length(rows)
    all(length(r) == n for r in rows) ||
        throw(ArgumentError("distance_matrix parsing expects an n x n numeric table; got $(n) rows with non-square width."))
    dist = Matrix{Float64}(undef, n, n)
    drop_missing = _drop_row_on_missing(opts.missing_policy)
    drop_missing && throw(ArgumentError("distance_matrix parsing does not support missing_policy=:drop_rows."))
    @inbounds for i in 1:n
        for j in 1:n
            v = _parse_float_token(rows[i][j])
            v === nothing && throw(ArgumentError("distance_matrix parsing encountered missing values."))
            dist[i, j] = v
        end
    end
    if check_symmetric
        tol = Float64(symmetry_tol)
        @inbounds for i in 1:n
            for j in (i+1):n
                abs(dist[i, j] - dist[j, i]) <= tol ||
                    throw(ArgumentError("distance matrix is not symmetric within tolerance $(tol) at ($(i),$(j))."))
            end
        end
    end
    return Serialization._graded_complex_from_distance_matrix(
        dist;
        max_dim=max_dim,
        radius=radius,
        knn=knn,
        construction=construction,
    )
end

@inline function _kind_matches_data(data, kind::Symbol)::Bool
    kind == :auto && return true
    if kind == :point_cloud
        return data isa PointCloud
    elseif kind == :graph
        return data isa GraphData
    elseif kind == :image
        return data isa ImageNd
    elseif kind == :distance_matrix
        return data isa GradedComplex
    end
    return false
end

"""
    load_data(path; kind=:auto, format=:auto, opts=DataFileOptions(), kwargs...)

Load a dataset-like object from a file path into canonical typed ingestion data.
"""
function load_data(path::AbstractString;
                   kind::Symbol=:auto,
                   format::Symbol=:auto,
                   opts::DataFileOptions=DataFileOptions(),
                   kwargs...)
    k = _resolve_file_kind(kind, opts)
    f = _resolve_file_format(format, path, opts)

    if f == :dataset_json
        if !isempty(kwargs)
            bad = join(string.(keys(kwargs)), ", ")
            throw(ArgumentError("load_data: dataset_json does not accept extra kwargs ($(bad))."))
        end
        data = Serialization.load_dataset_json(path)
        _kind_matches_data(data, k) || throw(ArgumentError("load_data: expected kind=$(k), but dataset JSON decoded to $(typeof(data))."))
        return data
    elseif f == :ripser_point_cloud
        data = Serialization.load_ripser_point_cloud(path)
        _kind_matches_data(data, k) || throw(ArgumentError("load_data: expected kind=$(k), but ripser point-cloud loader returns PointCloud."))
        return data
    elseif f == :ripser_distance
        data = Serialization.load_ripser_distance(path; kwargs...)
        _kind_matches_data(data, k == :auto ? :distance_matrix : k) ||
            throw(ArgumentError("load_data: expected kind=$(k), but ripser distance loader returns GradedComplex."))
        return data
    elseif f == :ripser_lower_distance
        data = Serialization.load_ripser_lower_distance(path; kwargs...)
        _kind_matches_data(data, k == :auto ? :distance_matrix : k) ||
            throw(ArgumentError("load_data: expected kind=$(k), but ripser lower-distance loader returns GradedComplex."))
        return data
    elseif f == :ripser_upper_distance
        data = Serialization.load_ripser_upper_distance(path; kwargs...)
        _kind_matches_data(data, k == :auto ? :distance_matrix : k) ||
            throw(ArgumentError("load_data: expected kind=$(k), but ripser upper-distance loader returns GradedComplex."))
        return data
    elseif f == :ripser_sparse_triplet
        data = Serialization.load_ripser_sparse_triplet(path; kwargs...)
        _kind_matches_data(data, k == :auto ? :distance_matrix : k) ||
            throw(ArgumentError("load_data: expected kind=$(k), but ripser sparse-triplet loader returns GradedComplex."))
        return data
    elseif f == :ripser_binary_lower_distance
        data = Serialization.load_ripser_binary_lower_distance(path; kwargs...)
        _kind_matches_data(data, k == :auto ? :distance_matrix : k) ||
            throw(ArgumentError("load_data: expected kind=$(k), but ripser binary loader returns GradedComplex."))
        return data
    elseif f == :ripser_lower_distance_streaming
        data = Serialization.load_ripser_lower_distance_streaming(path; kwargs...)
        _kind_matches_data(data, k == :auto ? :distance_matrix : k) ||
            throw(ArgumentError("load_data: expected kind=$(k), but ripser streaming loader returns GradedComplex."))
        return data
    elseif f == :csv || f == :tsv || f == :txt
        k == :auto &&
            throw(ArgumentError("load_data: kind=:auto is ambiguous for $(f). Pass kind=:point_cloud, :graph, :image, or :distance_matrix."))
        if k == :point_cloud
            if !isempty(kwargs)
                bad = join(string.(keys(kwargs)), ", ")
                throw(ArgumentError("load_data: point_cloud table parsing does not accept extra kwargs ($(bad))."))
            end
            return _load_point_cloud_table(path, f, opts)
        elseif k == :graph
            if !isempty(kwargs)
                bad = join(string.(keys(kwargs)), ", ")
                throw(ArgumentError("load_data: graph table parsing does not accept extra kwargs ($(bad))."))
            end
            return _load_graph_table(path, f, opts)
        elseif k == :image
            if !isempty(kwargs)
                bad = join(string.(keys(kwargs)), ", ")
                throw(ArgumentError("load_data: image table parsing does not accept extra kwargs ($(bad))."))
            end
            return _load_image_table(path, f, opts)
        elseif k == :distance_matrix
            return _load_distance_matrix_table(path, f, opts; kwargs...)
        end
        throw(ArgumentError("load_data: unsupported kind=$(k) for format=$(f)."))
    end
    throw(ArgumentError("load_data: unsupported format $(f)."))
end

"""
    inspect_data_file(path; format=:auto, sample_rows=10, opts=DataFileOptions())

Lightweight inspector for file-ingestion planning.
"""
function inspect_data_file(path::AbstractString;
                           format::Symbol=:auto,
                           sample_rows::Int=10,
                           opts::DataFileOptions=DataFileOptions())
    sample_rows >= 0 || throw(ArgumentError("inspect_data_file: sample_rows must be >= 0."))
    f = _resolve_file_format(format, path, opts)
    if f == :dataset_json
        info = Serialization.inspect_json(path)
        kind = hasproperty(info, :kind) ? Symbol(getproperty(info, :kind)) : :unknown
        schema = hasproperty(info, :schema_version) ? getproperty(info, :schema_version) : nothing
        return (
            path=abspath(path),
            format=f,
            kind=kind,
            schema_version=schema,
            candidate_kinds=(kind,),
            detail=info,
        )
    elseif f == :ripser_point_cloud
        return (
            path=abspath(path),
            format=f,
            kind=:point_cloud,
            candidate_kinds=(:point_cloud,),
            detail=nothing,
        )
    elseif f == :ripser_distance || f == :ripser_lower_distance || f == :ripser_upper_distance ||
           f == :ripser_sparse_triplet || f == :ripser_binary_lower_distance || f == :ripser_lower_distance_streaming
        return (
            path=abspath(path),
            format=f,
            kind=:distance_matrix,
            candidate_kinds=(:distance_matrix,),
            detail=nothing,
        )
    elseif f == :csv || f == :tsv || f == :txt
        names, rows, header_used = _parse_delimited_table(path, f, opts)
        nr = length(rows)
        nc = length(names)
        cands = Symbol[:point_cloud]
        nc >= 2 && push!(cands, :graph)
        push!(cands, :image)
        nr == nc && push!(cands, :distance_matrix)
        nshow = min(sample_rows, nr)
        sample = nshow == 0 ? Vector{Vector{String}}() : rows[1:nshow]
        return (
            path=abspath(path),
            format=f,
            kind=:table,
            header_used=header_used,
            nrows=nr,
            ncols=nc,
            columns=Tuple(names),
            candidate_kinds=Tuple(cands),
            sample=sample,
        )
    end
    throw(ArgumentError("inspect_data_file: unsupported format $(f)."))
end

end # module DataFileIO
