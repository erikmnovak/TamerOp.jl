module TamerOpDataFramesExt

using DataFrames
using Tables
import TamerOp

const FEA = TamerOp.Featurizers

"""
    DataFrame(fs::FeatureSet; format=:wide, copycols=true)

Construct a DataFrame in wide or long form from a feature result. Table wrappers
use DataFrames' native Tables interface directly.
"""
function DataFrames.DataFrame(fs::FEA.FeatureSet; format::Symbol=:wide, copycols::Bool=true)
    columns = Tables.columntable(FEA.feature_table(fs; format=format))
    return DataFrames.DataFrame(columns; copycols=copycols)
end

end # module
