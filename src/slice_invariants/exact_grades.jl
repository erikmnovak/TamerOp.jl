# Sampling remains sampling, but evaluating the declared sample points must not
# round exact coordinates or merge distinct parameter values.
function _oriented_algebraic_direction(pi, direction)
    length(direction) == dimension(pi) || throw(ArgumentError("slice direction dimension mismatch"))
    orientation = hasproperty(pi, :orientation) ? pi.orientation : ntuple(_ -> 1, length(direction))
    oriented = [orientation[i]*direction[i] for i in eachindex(direction)]
    all(>=(0), oriented) && any(>(0), oriented) || throw(ArgumentError(
        "slice direction must be nonzero and monotone in the encoding's coordinate orientation"))
    return oriented
end

function _algebraic_sampled_chain(pi, x0, dir, opts;
        ts=nothing, tmin=nothing, tmax=nothing, nsteps::Int=1001,
        box2=nothing, drop_unknown::Bool=true, dedup::Bool=true,
        check_chain::Bool=false)
    x, d = AlgebraicReal.(collect(x0)), AlgebraicReal.(collect(dir))
    n = dimension(pi)
    length(x) == length(d) == n || throw(ArgumentError("slice coordinate dimension mismatch"))
    _oriented_algebraic_direction(pi, d)
    nsteps > 0 || throw(ArgumentError("nsteps must be positive"))
    lower = tmin === nothing ? nothing : AlgebraicReal(tmin)
    upper = tmax === nothing ? nothing : AlgebraicReal(tmax)
    boxes = (opts.box, box2)
    for raw in boxes
        raw === nothing && continue
        box = raw === :auto ? encoding_box(pi, InvariantOptions(); margin=0) : raw
        length(box[1]) == length(box[2]) == n || throw(ArgumentError("slice window dimension mismatch"))
        all(box[1] .<= box[2]) || throw(ArgumentError("slice window endpoints must be ordered"))
        for i in 1:n
            a, b = AlgebraicReal(box[1][i]), AlgebraicReal(box[2][i])
            if iszero(d[i])
                a <= x[i] <= b || return Int[], AlgebraicReal[]
            else
                u,v = minmax((a-x[i])/d[i], (b-x[i])/d[i])
                lower = lower === nothing ? u : max(lower,u)
                upper = upper === nothing ? v : min(upper,v)
            end
        end
    end
    lower !== nothing && upper !== nothing && lower > upper && return Int[], AlgebraicReal[]
    times = if ts === nothing
        lower !== nothing && upper !== nothing || throw(ArgumentError("slice_chain needs finite tmin/tmax or a clipping box"))
        nsteps == 1 ? [lower] : [lower+(upper-lower)*((i-1)//(nsteps-1)) for i in 1:nsteps]
    else
        AlgebraicReal.(collect(ts))
    end
    chain = Int[]; values = AlgebraicReal[]
    strict = opts.strict === nothing ? true : opts.strict
    for t in times
        lower !== nothing && t < lower && continue
        upper !== nothing && t > upper && continue
        p = x .+ t .* d
        rid = locate(pi,p)
        if rid == 0
            strict && throw(ArgumentError("slice sample is outside the represented region"))
            drop_unknown && continue
        end
        dedup && !isempty(chain) && last(chain) == rid && continue
        push!(chain,rid); push!(values,t)
    end
    check_chain && !isempty(chain) && _check_chain_monotone(pi,x,d,chain,values;strict=strict)
    return chain,values
end

function _compile_algebraic_slices(pi, dirs, offs, opts, normalize_dirs,
        direction_weight, offset_weights, normalize_weights, drop_unknown, filtered)
    directions = map(collect(dirs)) do raw
        d = AlgebraicReal.(collect(raw))
        _oriented_algebraic_direction(pi, d)
        scale = normalize_dirs === :none ? one(AlgebraicReal) :
            normalize_dirs === :L1 ? sum(abs,d) :
            normalize_dirs === :Linf ? maximum(abs,d) :
            throw(ArgumentError("normalize_dirs must be :none, :L1, or :Linf"))
        iszero(scale) && throw(ArgumentError("zero slice direction"))
        d ./ scale
    end
    offsets = [AlgebraicReal.(collect(x)) for x in offs]
    nd,no = length(directions),length(offsets)
    nd > 0 && no > 0 || throw(ArgumentError("directions and offsets must be nonempty"))
    weights = [SliceInvariants.direction_weight(_oriented_algebraic_direction(pi,d),direction_weight) for d in directions] *
        _offset_sample_weights(offsets,offset_weights)'
    if normalize_weights
        sum(weights) > 0 || throw(ArgumentError("total slice weight must be positive"))
        weights ./= sum(weights)
    end
    chains = Vector{Int}[]; values = Vector{AlgebraicReal}[]
    for d in directions, x in offsets
        chain, vals = _algebraic_sampled_chain(pi,x,d,opts;drop_unknown=drop_unknown,filtered...)
        push!(chains,chain); push!(values,vals)
    end
    return _compiled_slice_plan_from_vectors(directions,offsets,weights,chains,values,nd,no)
end
