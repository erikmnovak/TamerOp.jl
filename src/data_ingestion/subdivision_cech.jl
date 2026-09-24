# Exact subdivision-Cech multicover for arbitrary finite indexed Euclidean balls.
# This explicit small-data backend preserves coincident labels and never perturbs
# coordinates. Radius grades are algebraic square roots of rational optima.

function _subdivision_cech_budget(spec, n)
    budget = _construction_budget(spec)
    budget.max_simplices === nothing && throw(ArgumentError(
        "backend=:subdivision_cech requires an explicit ConstructionBudget(max_simplices=...)."))
    _construction_check_max_simplices!(n, 0, spec)
    lo = get(spec.params, :depth_range, nothing) === nothing ? 0 : spec.params.depth_range[1]
    candidates = sum(binomial(big(n), j) for j in lo:n; init=big(0))
    _construction_check_max_simplices!(candidates, 0, spec)
    _construction_check_memory_budget!(256candidates, spec)
    return nothing
end

# A minimum ball is realized by <=d+1 affinely independent input sites.
# Enumerating their circumcenters includes that optimum. Every accepted sphere
# contains the requested subset, so taking the first (smallest) such radius
# cannot underestimate its minimum enclosing radius. No floating predicates.
function _subdivision_cech_spheres(X, spec, ::Type{M};
                                   dimension::Int=_rhomboid_affine_dimension(X)) where {M<:Integer}
    n = size(X,1)
    dim = dimension
    count = sum(binomial(big(n), j) for j in 1:dim+1; init=big(0))
    _construction_check_max_simplices!(count, dim, spec)
    _construction_check_memory_budget!(256count, spec)
    candidates = Tuple{QQ,M}[]
    for cardinality in 1:dim+1
        combination = collect(1:cardinality)
        while true
            support = foldl((mask,site)->mask | (one(M) << (site-1)), combination; init=zero(M))
            sphere = _rhomboid_sphere(X, support)
            sphere === nothing || push!(candidates, (sphere.radius2, sphere.inside | sphere.on))
            _next_combination!(combination,n,cardinality) || break
        end
    end
    sort!(candidates)
    unique!(candidates)
    return candidates
end

struct _CompiledSubdivisionCechGeometry{M<:Integer}
    masks::Vector{M}
    radii2::Vector{QQ}
    sphere_count::BigInt
    dimension::Int
end

function _compile_subdivision_cech_vertices(X::Matrix{QQ}, spec; cutoff2=nothing)
    n = size(X,1)
    _subdivision_cech_budget(spec,n)
    requested_depth = get(spec.params,:depth_range,nothing)
    lo = requested_depth === nothing ? 0 : first(requested_depth)
    M = n <= 64 ? UInt64 : BigInt
    dimension = _rhomboid_affine_dimension(X)
    spheres = _subdivision_cech_spheres(X,spec,M;dimension)
    masks = M[]
    radii2 = QQ[]
    for cardinality in lo:n
        if cardinality == 0
            push!(masks,zero(M)); push!(radii2,zero(QQ))
            continue
        end
        combination = collect(1:cardinality)
        while true
            mask = foldl((m,s)->m | (one(M) << (s-1)),combination;init=zero(M))
            index = findfirst(s -> iszero(mask & ~s[2]),spheres)
            index === nothing && error("Subdivision-Cech minimum ball has no support certificate.")
            radius2 = spheres[index][1]
            if cutoff2 === nothing || radius2 <= cutoff2
                push!(masks,mask); push!(radii2,radius2)
            end
            _next_combination!(combination,n,cardinality) || break
        end
    end
    order = sortperm(masks)
    masks, radii2 = masks[order], radii2[order]
    count = sum(binomial(big(n), j) for j in 1:dimension+1; init=big(0))
    return _CompiledSubdivisionCechGeometry(masks,radii2,count,dimension)
end

function _subdivision_cech_vertices(X::Matrix{QQ}, spec;
                                    cache::Union{Nothing,EncodingCache}=nothing)
    n = size(X,1)
    cutoff = get(spec.params,:radius,nothing)
    cutoff2 = cutoff === nothing ? nothing : cutoff isa AlgebraicReal ? cutoff^2 : QQ(cutoff)^2
    if cache === nothing
        geometry = _compile_subdivision_cech_vertices(X,spec;cutoff2)
        return geometry.masks,geometry.radii2
    end
    _subdivision_cech_budget(spec,n)
    depths = get(spec.params,:depth_range,nothing)
    lo = depths === nothing ? 0 : first(depths)
    key = (:subdivision_cech_geometry,_structural_cache_key((X,lo)))
    geometry = _rhomboid_cached_geometry(cache,key) do
        _compile_subdivision_cech_vertices(X,spec)
    end
    _construction_check_max_simplices!(geometry.sphere_count,geometry.dimension,spec)
    _construction_check_memory_budget!(256geometry.sphere_count,spec)
    # Return fresh arrays, including on unfiltered calls. Their ownership must
    # remain independent of the shared minimum-enclosing-ball certificates.
    cutoff === nothing && return copy(geometry.masks),copy(geometry.radii2)
    keep = findall(r -> r <= cutoff2,geometry.radii2)
    return geometry.masks[keep],geometry.radii2[keep]
end

function _subdivision_cech_complex(X::Matrix{QQ}, spec; return_simplex_tree=false,
                                    cache::Union{Nothing,EncodingCache}=nothing)
    n = size(X,1)
    requested_depth = get(spec.params,:depth_range,nothing)
    lo, hi = requested_depth === nothing ? (0,n) : requested_depth
    requested_dim = get(spec.params,:max_dim,nothing)
    maxdim = requested_dim === nothing ? n-lo : Int(min(requested_dim,n-lo))
    masks, radii2 = _subdivision_cech_vertices(X,spec;cache)
    successors = [Int[] for _ in masks]
    edge_count = 0
    if maxdim > 0
        for i in eachindex(masks),j in i+1:length(masks)
            masks[i] & masks[j] == masks[i] || continue
            edge_count += 1
            _construction_check_max_edges!(edge_count,spec)
            _construction_check_max_simplices!(length(masks)+edge_count,1,spec)
            _construction_check_memory_budget!(big(256length(masks))+32big(edge_count),spec)
            push!(successors[i],j)
        end
    end
    # Count all flags before allocating them, including lower-dimensional faces.
    current = ones(BigInt,length(masks))
    counts = BigInt[sum(current;init=big(0))]
    for dim in 1:maxdim
        next = zeros(BigInt,length(masks))
        for i in eachindex(masks),j in successors[i]
            next[j] += current[i]
        end
        current = next
        push!(counts,sum(current;init=big(0)))
    end
    _construction_check_max_simplices!(sum(counts),maxdim,spec)
    _construction_check_memory_budget!(sum(big(192+16d)*counts[d] for d in eachindex(counts)),spec)
    simplices = [Vector{Vector{Int}}() for _ in 0:maxdim]
    chain = Int[]
    function extend!(vertex)
        push!(chain,vertex)
        push!(simplices[length(chain)],copy(chain))
        if length(chain) <= maxdim
            foreach(extend!,successors[vertex])
        end
        pop!(chain)
    end
    foreach(extend!,eachindex(masks))
    foreach(sort!,simplices)
    # A subset can occur in exponentially many flags. Construct its exact
    # algebraic coordinates once, rather than taking one square root per flag.
    vertex_radii = _rhomboid_radius.(radii2)
    vertex_depths = AlgebraicReal[min(count_ones(mask),hi) for mask in masks]
    grades = NTuple{2,AlgebraicReal}[]
    for group in simplices,simplex in group
        push!(grades,(vertex_radii[last(simplex)],vertex_depths[first(simplex)]))
    end
    # Empty windows represent the zero module, not a phantom component.
    tree = _simplex_tree_multi_from_simplices(simplices,grades)
    result = return_simplex_tree ? tree : _graded_complex_from_simplex_tree(tree)
    radius_axis = sort!(unique!(vcat(AlgebraicReal[0],first.(grades))))
    cutoff = get(spec.params,:radius,nothing)
    cutoff === nothing || sort!(unique!(push!(radius_axis, AlgebraicReal(cutoff))))
    depth_axis = AlgebraicReal[-k for k in hi:-1:lo]
    axes = get(spec.params,:axes,(radius_axis,depth_axis))
    _record_ingestion_backend(:multicover,:subdivision_cech)
    return result,axes,(1,-1)
end
