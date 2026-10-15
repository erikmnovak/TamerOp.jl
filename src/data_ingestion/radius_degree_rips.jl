# Degree-Rips births from full-cloud neighbor distances; independent of display.
function _validate_radius_degree_params(p)
    allowed=(:max_dim,:radius,:construction,:orientation,:axes,:axes_policy,:axis_kind,
        :eps,:poset_kind,:field,:max_axis_len,:multicritical)
    all(k->k in allowed,keys(p)) || throw(ArgumentError("Unsupported radius-degree Rips option."))
    d=get(p,:max_dim,1);r=get(p,:radius,nothing)
    d isa Int && d>=0 || throw(ArgumentError("max_dim must be a nonnegative Int."))
    (r===nothing || (r isa Real && !(r isa Bool) && isfinite(r) && 0<=r && isfinite(Float64(r)))) ||
        throw(ArgumentError("radius must be nothing or a finite nonnegative diameter cap."))
    (get(p,:orientation,nothing)===nothing || Tuple(p.orientation)==(1,-1)) || throw(ArgumentError("Degree\u2013Rips requires orientation=(1,-1)."))
    get(p,:multicritical,:union)===:union || throw(ArgumentError("Degree\u2013Rips requires multicritical=:union."))
    c=_construction_from_params(p)
    c.sparsify in (:none,:radius) && c.collapse===:none ||
        throw(ArgumentError("Degree\u2013Rips supports the full radius graph, without sparsification or collapse."))
    c.sparsify===:radius && r===nothing && throw(ArgumentError("Radius construction requires a diameter cap."))
    nothing
end

function _radius_degree_complex(data,spec;return_simplex_tree=false,cache=nothing)
    _validate_radius_degree_params(spec.params)
    return_simplex_tree && throw(ArgumentError("Degree\u2013Rips has multiple births per simplex; request the graded complex."))
    n=length(data.points);n>0 || throw(ArgumentError("Degree\u2013Rips requires at least one point."))
    c=_construction_from_params(spec.params)
    _construction_check_memory_budget!(big(8)*n*n,spec)
    # Column k+1 contains the kth other-neighbor distance, with column 1 zero.
    points=point_matrix(data)
    all(v->isfinite(Float64(v)),points) || throw(ArgumentError("Degree\u2013Rips coordinates must be finite Float64 values."))
    distances=[_euclidean_distance(points,v,w) for v in 1:n,w in 1:n]
    all(isfinite,distances) || throw(ArgumentError("Pair distances overflow Float64; rescale the input."))
    for v in 1:n;sort!(view(distances,v,:));end
    requested_cap=get(spec.params,:radius,nothing)
    cap=requested_cap===nothing ? nothing : Float64(requested_cap)
    base=FiltrationSpec(;kind=:rips,max_dim=get(spec.params,:max_dim,1),radius=cap,
        nn_backend=:bruteforce,construction=c)
    G,_,_=_graded_complex_from_point_cloud(data,base;cache)
    grades=Vector{Vector{NTuple{2,Float64}}}()
    grade_count=0
    for (i,vertices) in enumerate(Iterators.flatten(_simplices_from_complex(G)))
        births=NTuple{2,Float64}[]
        for j in 1:n
            r=max(Float64(G.grades[i][1]),maximum(v->distances[v,j],vertices))
            cap!==nothing && r>cap && break
            if !isempty(births) && r==last(births)[1]
                births[end]=(r,Float64(j-1))
            else
                push!(births,(r,Float64(j-1)))
            end
        end
        grade_count+=length(births)
        _construction_check_memory_budget!(big(8)*n*n+big(32)*grade_count,spec)
        push!(grades,births)
    end
    out=MultiCriticalGradedComplex(G.cells_by_dim,G.boundaries,grades)
    axes=get(spec.params,:axes,nothing)
    out,axes===nothing ? _axes_from_complex_grades(out,(1,-1)) : axes,(1,-1)
end
