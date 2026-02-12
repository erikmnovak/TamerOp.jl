module TamerOpArrowExt

using Arrow
using Tables

const PM = let pm = nothing
    if isdefined(Main, :PosetModules)
        pm = getfield(Main, :PosetModules)
    else
        @eval import PosetModules
        pm = PosetModules
    end
    pm
end

const WF = PM.Workflow

function WF.save_features_arrow(path::AbstractString,
                                fs::WF.FeatureSet;
                                format::Symbol=:wide,
                                include_metadata::Bool=true,
                                metadata_path::Union{Nothing,AbstractString}=nothing,
                                compress=nothing)
    tbl = WF.feature_table(fs; format=format)
    if compress === nothing
        Arrow.write(path, tbl)
    else
        Arrow.write(path, tbl; compress=compress)
    end

    if include_metadata
        mpath = metadata_path === nothing ? WF.default_feature_metadata_path(path) : String(metadata_path)
        WF.save_metadata_json(mpath, WF.feature_metadata(fs; format=format))
    end
    return path
end

function WF.load_features_arrow(path::AbstractString;
                                format::Symbol=:wide,
                                ids_col::Symbol=:id,
                                metadata_path::Union{Nothing,AbstractString}=nothing,
                                require_metadata::Bool=false)
    table = Arrow.Table(path)
    cols = Tables.columntable(table)
    mpath = metadata_path === nothing ? WF.default_feature_metadata_path(path) : String(metadata_path)
    md = if isfile(mpath)
        WF.load_metadata_json(mpath; validate_feature_schema=true)
    elseif require_metadata
        throw(ArgumentError("load_features_arrow: metadata sidecar not found at $(mpath)"))
    else
        nothing
    end
    meta = md === nothing ? NamedTuple() : (metadata=md,)
    return WF._featureset_from_columntable(cols; format=format, ids_col=ids_col, meta=meta)
end

end # module
