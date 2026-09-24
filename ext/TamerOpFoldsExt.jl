module TamerOpFoldsExt

using Folds

import TamerOp

const TO = TamerOp

const FEA = TO.Featurizers

function _foreach_indexed(n::Int, f; chunk_size::Int=0, deterministic::Bool=true)
    n <= 1 && return nothing
    if chunk_size > 0
        Folds.foreach(1:n; basesize=chunk_size) do i
            f(i)
        end
    else
        Folds.foreach(1:n) do i
            f(i)
        end
    end
    return nothing
end

function __init__()
    FEA._set_batch_impl!((foreach_indexed=_foreach_indexed,))
    return nothing
end

end # module
