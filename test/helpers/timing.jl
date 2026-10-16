# Paired measurements for short, repeated queries with retained preparation.
# Warm the complete timed wrapper; keep collections and compilation in the data.
function _timed_query_batch(f::F, evals::Int) where F
    stats = @timed for _ in 1:evals
        f()
    end
    return (seconds=stats.time, bytes=stats.bytes, gc_seconds=stats.gctime,
            compile_seconds=stats.compile_time, recompile_seconds=stats.recompile_time)
end

function _paired_query_samples(left::F, right::G; reps::Int=9, evals::Int=16) where {F,G}
    left_warm = _timed_query_batch(left, evals)
    right_warm = _timed_query_batch(right, evals)
    left_samples = Vector{typeof(left_warm)}(undef, reps)
    right_samples = Vector{typeof(right_warm)}(undef, reps)
    for sample in 1:reps
        GC.gc()
        if isodd(sample)
            left_samples[sample] = _timed_query_batch(left, evals)
            right_samples[sample] = _timed_query_batch(right, evals)
        else
            right_samples[sample] = _timed_query_batch(right, evals)
            left_samples[sample] = _timed_query_batch(left, evals)
        end
    end
    return (; left=left_samples, right=right_samples, evals)
end
