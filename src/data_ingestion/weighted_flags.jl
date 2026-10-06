# One-parameter flag filtrations with explicit vertex and edge birth times.

function _weighted_grade(x; absent=false)
    x isa Real && !(x isa Bool) || throw(ArgumentError("filtration values must be real numbers."))
    y = Float64(x)
    (isfinite(x) && isfinite(y)) || (absent && x == Inf) ||
        throw(ArgumentError("filtration values must be finite and representable as Float64; only absent edges may use +Inf."))
    return y
end

function _validate_weighted_params(p)
    d = get(p,:max_dim,1)
    d isa Integer && !(d isa Bool) && 0 <= d < typemax(Int)-1 ||
        throw(ArgumentError("max_dim must be a nonnegative simplex dimension fitting Int."))
    t = get(p,:threshold,nothing)
    t === nothing || _weighted_grade(t;absent=true)
    construction = _construction_from_params(p)
    construction.sparsify === :none && construction.collapse === :none ||
        throw(ArgumentError("weighted-vertex flag input requires sparsify=:none and collapse=:none; use threshold to restrict its parameter range."))
    return nothing
end

# Vertex extraction is also used to identify original vertices in explicit results.
function _weighted_vertex_births(data::GraphData, spec)
    values = get(spec.params,:vertex_births,nothing)
    values === nothing && return zeros(data.n)
    values isa AbstractVector && length(values)==data.n ||
        throw(ArgumentError("supply exactly one vertex birth per graph vertex."))
    Base.require_one_based_indexing(values)
    return [_weighted_grade(v) for v in values]
end
_weighted_vertex_births(data::AbstractMatrix, spec) = [_weighted_grade(data[i,i]) for i in axes(data,1)]

function _weighted_flag_graph(data,spec;check_budget=true)
    _validate_weighted_params(spec.params)
    weights = get(spec.params,:edge_weights,nothing)
    vertex_births = get(spec.params,:vertex_births,nothing)
    edges = NTuple{2,Int}[]
    grades = Float64[]
    slots = Dict{NTuple{2,Int},Int}()
    function add_edge(u,v,value,n)
        1 <= u <= n && 1 <= v <= n && u != v ||
            throw(ArgumentError("graph edges must join distinct vertices in 1:$n."))
        w = _weighted_grade(value;absent=true)
        key = minmax(u,v)
        slot = get(slots,key,0)
        if slot == 0
            push!(edges,key); push!(grades,w); slots[key]=length(edges)
        else
            grades[slot] == w || throw(ArgumentError("conflicting birth times for edge $key."))
        end
    end
    if data isa GraphData
        n = data.n
        n >= 0 || throw(ArgumentError("graph vertex count must be nonnegative."))
        length(data.edge_u)==length(data.edge_v) || throw(ArgumentError("graph edge columns must have equal length."))
        births = _weighted_vertex_births(data,spec)
        weights = weights === nothing ? data.weights : weights
        weights === nothing && isempty(data.edge_u) && (weights=Float64[])
        weights isa AbstractVector && length(weights)==length(data.edge_u) ||
            throw(ArgumentError("supply one edge birth per graph edge, using GraphData weights or edge_weights."))
        Base.require_one_based_indexing(weights)
        for i in eachindex(data.edge_u)
            add_edge(data.edge_u[i],data.edge_v[i],weights[i],n)
        end
    elseif data isa AbstractMatrix{<:Real}
        Base.require_one_based_indexing(data)
        n = size(data,1)
        size(data,2)==n || throw(ArgumentError("weighted flag matrix must be square."))
        weights === nothing && vertex_births === nothing ||
            throw(ArgumentError("matrix input supplies vertex births on its diagonal and edge births off diagonal; do not also pass birth vectors."))
        if issparse(data)
            A = data isa SparseMatrixCSC ? data : sparse(data)
            length(A.colptr)==n+1 && first(A.colptr)==1 && issorted(A.colptr) &&
                last(A.colptr)==length(A.nzval)+1 && length(A.rowval)==length(A.nzval) ||
                throw(ArgumentError("invalid sparse weighted-graph storage."))
            births = zeros(n) # omitted diagonal means vertex birth zero
            for j in 1:n
                previous=0
                for k in nzrange(A,j)
                    i=A.rowval[k]
                    previous < i <= n || throw(ArgumentError("sparse rows must be sorted, distinct and in range."))
                    previous=i
                    if i==j
                        births[i]=_weighted_grade(A.nzval[k])
                    else
                        add_edge(i,j,A.nzval[k],n)
                    end
                end
            end
        else
            births = [_weighted_grade(data[i,i]) for i in 1:n]
            for j in 2:n, i in 1:j-1
                a,b = _weighted_grade(data[i,j];absent=true),_weighted_grade(data[j,i];absent=true)
                a==b || throw(ArgumentError("weighted flag matrix must be symmetric off diagonal."))
                add_edge(i,j,a,n)
            end
        end
    else
        throw(ArgumentError("EdgeWeightedFiltration expects GraphData or a real weighted adjacency matrix."))
    end
    births isa AbstractVector && length(births)==n ||
        throw(ArgumentError("supply exactly one vertex birth per graph vertex."))
    for (i,(u,v)) in enumerate(edges)
        grades[i] >= max(births[u],births[v]) ||
            throw(ArgumentError("edge ($u,$v) appears before an endpoint; its birth must be at least both vertex births."))
    end
    threshold = Float64(something(get(spec.params,:threshold,nothing),Inf))
    active = findall(<=(threshold),births)
    mapping = zeros(Int,n)
    for (i,v) in enumerate(active)
        mapping[v]=i
    end
    keep = get(spec.params,:max_dim,1)==0 ? Int[] :
        findall(i -> isfinite(grades[i]) && grades[i]<=threshold,eachindex(edges))
    sort!(keep;by=i -> edges[i])
    retained_edges = [(mapping[edges[i][1]],mapping[edges[i][2]]) for i in keep]
    check_budget && _construction_check_max_edges!(length(keep),spec)
    check_budget && _construction_check_max_simplices!(length(active),0,spec)
    return (n=length(active),edges=retained_edges,dists=grades[keep],births=births[active],
        source_indices=active,original_n=n,input_edges=length(keep),spec=spec,
        graph_selection=:none,collapse=:none)
end

function _graded_complex_from_weighted_graph(data,spec;return_simplex_tree=false)
    p = _weighted_flag_graph(data,spec)
    return _materialize_flag_output(p.edges,[(b,) for b in p.births],[(d,) for d in p.dists],spec;
        return_simplex_tree)
end

function _estimate_weighted_flag_counts(data,spec;warnings,strict)
    p = _weighted_flag_graph(data,spec;check_budget=false)
    d = Int(get(spec.params,:max_dim,1))
    d >= 2 && _ingestion_warn!(warnings,
        "Weighted-graph simplex counts above edges are upper bounds (complete-graph assumption).",strict)
    return BigInt[k==0 ? p.n : k==1 ? length(p.edges) : binomial(big(p.n),k+1) for k in 0:d]
end
