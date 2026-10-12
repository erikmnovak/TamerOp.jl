# The view envelope owns versioning, integrity and portable source references.
# Mathematical object schemas remain separate from declarative view values.

const VISUAL_SESSION_SCHEMA_VERSION = 1
const _VISUAL_SESSION_MAX_BYTES = 256 * 1024 * 1024

function _visual_session_byte_budget(max_bytes)
    max_bytes isa Integer && !(max_bytes isa Bool) && 0 < max_bytes <= typemax(Int) ||
        throw(ArgumentError("max_bytes must be a positive integer"))
    return Int(max_bytes)
end

function _visual_session_read(path; max_bytes=_VISUAL_SESSION_MAX_BYTES)
    _visual_session_byte_budget(max_bytes)
    isfile(path) || throw(ArgumentError("Saved visual resource is missing: $path"))
    filesize(path) <= max_bytes || throw(ArgumentError("Saved visual resource exceeds max_bytes: $path"))
    return read(path, String)
end

"""Encode one declarative view record in the versioned, integrity-checked envelope."""
function visual_session_to_json(record::NamedTuple; max_bytes=_VISUAL_SESSION_MAX_BYTES)
    _visual_session_byte_budget(max_bytes)
    payload = String(JSON3.write(visual_data_to_json(record)))
    result = String(JSON3.write((; kind="TamerOpVisualSession", schema_version=VISUAL_SESSION_SCHEMA_VERSION,
        sha256=bytes2hex(SHA.sha256(payload)), payload)))
    sizeof(result) <= max_bytes || throw(ArgumentError("Saved visual session exceeds max_bytes"))
    return result
end

"""Read a versioned view envelope; this validates data and does not execute a view request."""
function visual_session_from_json(text::AbstractString; max_bytes=_VISUAL_SESSION_MAX_BYTES)
    _visual_session_byte_budget(max_bytes)
    sizeof(text) <= max_bytes || throw(ArgumentError("Saved visual session exceeds max_bytes"))
    try
        obj = JSON3.read(text)
        _visual_data_keys(obj, ("kind", "schema_version", "sha256", "payload"), "session")
        obj["kind"] == "TamerOpVisualSession" && obj["schema_version"] isa Integer &&
            !(obj["schema_version"] isa Bool) && obj["schema_version"] == VISUAL_SESSION_SCHEMA_VERSION ||
            throw(ArgumentError("Unsupported visual-session schema"))
        obj["payload"] isa AbstractString && obj["sha256"] isa AbstractString ||
            throw(ArgumentError("Invalid visual-session envelope"))
        bytes2hex(SHA.sha256(obj["payload"])) == obj["sha256"] ||
            throw(ArgumentError("Saved visual-session content identity has changed"))
        result = visual_data_from_json(JSON3.read(obj["payload"]))
        result isa NamedTuple || throw(ArgumentError("A saved visual session requires a named record"))
        return result
    catch err
        err isa InterruptException && rethrow()
        err isa ArgumentError && rethrow()
        throw(ArgumentError("Invalid saved visual session: $(sprint(showerror, err))"))
    end
end

"""Validate a portable relative source path lexically without reading the resource."""
function visual_session_resource_path(relative)
    relative isa AbstractString && !isempty(relative) && !isabspath(relative) ||
        throw(ArgumentError("Saved source references must be nonempty relative paths"))
    # Store slash-separated paths so moving a bundle also works across platforms.
    parts = split(replace(String(relative), '\\'=>'/'), '/')
    any(p -> p in ("..", "") || occursin(':', p) || occursin('\0',p), parts) &&
        throw(ArgumentError("Saved source references cannot escape the bundle directory"))
    return join(parts,'/')
end

function _visual_relative_resource(base_dir, relative; must_exist=true)
    parts = split(visual_session_resource_path(relative),'/')
    root = realpath(base_dir)
    path = joinpath(root, parts...)
    if ispath(path)
        resolved = realpath(path)
        relative_resolved = relpath(resolved, root)
        first(splitpath(relative_resolved)) == ".." &&
            throw(ArgumentError("Saved source symlink escapes the bundle directory"))
    elseif must_exist
        throw(ArgumentError("Saved source resource is missing: $relative"))
    end
    return path
end

function _visual_source_artifact(text::AbstractString)
    kind = get(JSON3.read(text), "kind", nothing)
    if kind == "PersistenceDiagram"
        return mktempdir() do dir
            path = joinpath(dir, "diagram.json")
            write(path, text)
            load_persistence_diagram_json(path)
        end
    elseif kind == "FiniteEncodingFringe"
        return _load_encoding_json_v1(codeunits(text); output=:encoding_result, validation=:strict)
    end
    return visual_source_from_json(text)
end

"""Build an explicit embedded or relative mathematical source reference for a saved view."""
function visual_session_reference(; object=nothing, object_path=nothing, base_dir=pwd(), max_bytes=_VISUAL_SESSION_MAX_BYTES)
    _visual_session_byte_budget(max_bytes)
    object === nothing && object_path === nothing && return (; mode=:none, identity=nothing)
    if object_path === nothing
        payload = visual_source_to_json(object)
        sizeof(payload) <= max_bytes || throw(ArgumentError("Embedded source exceeds max_bytes"))
        return (; mode=:embedded, identity=bytes2hex(SHA.sha256(payload)), payload)
    end
    object_path isa AbstractString || throw(ArgumentError("object_path must be a path"))
    relative = isabspath(object_path) ? relpath(object_path, base_dir) : String(object_path)
    relative = replace(relative, '\\'=>'/')
    path = _visual_relative_resource(base_dir, relative)
    text = _visual_session_read(path; max_bytes)
    source = _visual_source_artifact(text)
    identity = visual_source_identity(source)
    object === nothing || visual_source_identity(object) == identity || throw(ArgumentError(
        "object_path describes a different source or different module basis coordinates"))
    return (; mode=:relative, identity, path=relative, sha256=bytes2hex(SHA.sha256(text)))
end

"""Restore a checked owned source, or check an explicitly supplied object's content identity."""
function visual_session_source(reference::NamedTuple, base_dir; object=nothing, max_bytes=_VISUAL_SESSION_MAX_BYTES)
    _visual_session_byte_budget(max_bytes)
    reference.mode === :none && throw(ArgumentError(
        "This session contains precomputed views only. Save with object=... to resume supported Julia queries."))
    if object !== nothing
        visual_source_identity(object) == reference.identity || throw(ArgumentError(
            "The supplied mathematical object's content identity differs from the saved source"))
        return object
    end
    text = if reference.mode === :embedded
        sizeof(reference.payload) <= max_bytes || throw(ArgumentError("Embedded source exceeds max_bytes"))
        reference.payload
    elseif reference.mode === :relative
        path = _visual_relative_resource(base_dir, reference.path)
        data = _visual_session_read(path; max_bytes)
        bytes2hex(SHA.sha256(data)) == reference.sha256 || throw(ArgumentError(
            "Saved source resource has changed: $(reference.path)"))
        data
    else
        throw(ArgumentError("Unknown saved source reference mode"))
    end
    result = _visual_source_artifact(text)
    visual_source_identity(result) == reference.identity || throw(ArgumentError(
        "Saved mathematical source content identity is inconsistent"))
    return result
end

"""Read JSON or the inert canonical record embedded in a self-contained HTML export."""
function read_visual_session(path::AbstractString; max_bytes=_VISUAL_SESSION_MAX_BYTES)
    text = _visual_session_read(path; max_bytes)
    if lowercase(splitext(path)[2]) == ".html"
        matches = collect(eachmatch(r"<script id=\"tamerop-visual-session\" type=\"application/json\">(.*?)</script>"s, text))
        length(matches) == 1 || throw(ArgumentError("HTML must contain exactly one TamerOp visual-session record"))
        text = only(matches).captures[1]
    end
    return visual_session_from_json(text; max_bytes)
end
