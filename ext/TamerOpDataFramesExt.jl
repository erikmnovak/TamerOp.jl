module TamerOpDataFramesExt

using DataFrames

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

"""
    DataFrame(fs::FeatureSet; format=:wide, copycols=true)

Convenience DataFrame constructor for feature outputs.
"""
function DataFrames.DataFrame(fs::WF.FeatureSet; format::Symbol=:wide, copycols::Bool=true)
    return DataFrames.DataFrame(WF.feature_table(fs; format=format); copycols=copycols)
end

DataFrames.DataFrame(t::WF.FeatureSetWideTable; copycols::Bool=true) =
    DataFrames.DataFrame(WF.feature_table(t.fs; format=:wide); copycols=copycols)

DataFrames.DataFrame(t::WF.FeatureSetLongTable; copycols::Bool=true) =
    DataFrames.DataFrame(WF.feature_table(t.fs; format=:long); copycols=copycols)

DataFrames.DataFrame(t::WF.EulerSurfaceLongTable; copycols::Bool=true) =
    DataFrames.DataFrame(t; copycols=copycols)

DataFrames.DataFrame(t::WF.PersistenceImageLongTable; copycols::Bool=true) =
    DataFrames.DataFrame(t; copycols=copycols)

DataFrames.DataFrame(t::WF.MPLandscapeLongTable; copycols::Bool=true) =
    DataFrames.DataFrame(t; copycols=copycols)

DataFrames.DataFrame(t::WF.PointSignedMeasureLongTable; copycols::Bool=true) =
    DataFrames.DataFrame(t; copycols=copycols)

end # module
