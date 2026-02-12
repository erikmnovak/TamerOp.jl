module TamerOpFoldsExt

using Folds

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

WF._set_batch_impl!((foreach_indexed=_foreach_indexed,))

end # module
