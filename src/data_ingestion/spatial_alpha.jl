# Spatial alpha complexes: a triangulated Delaunay complex with exact rational
# empty-ball predicates and squared radii. Gabriel faces enter at their own
# circumsphere; other faces inherit the smallest coface value (Edelsbrunner and
# Mucke, Three-dimensional alpha shapes, 1994, doi:10.1145/174462.156635).
const _SPATIAL_DELAUNAY_IMPL = Ref{Any}(nothing)

function _spatial_alpha_points(data)
    all(isfinite,point_matrix(data)) || throw(ArgumentError("Alpha coordinates must be finite."))
    X = Rational{BigInt}.(point_matrix(data))
    size(X, 2) == 3 || throw(ArgumentError("Spatial alpha requires three coordinates."))
    allunique(Tuple(row) for row in eachrow(X)) ||
        throw(ArgumentError("Alpha input contains duplicate sites; remove duplicates explicitly to preserve source identities."))
    return X
end

function _alpha_affine_basis(X)
    ids = Int[1]
    for i in 2:size(X, 1)
        E = X[[ids[2:end]; i], :] .- transpose(X[1, :])
        iszero(det(E * transpose(E))) || push!(ids, i)
        length(ids) == 4 && break
    end
    return ids
end

function _alpha_sphere(X, s)
    a = X[first(s), :]
    length(s) == 1 && return (a, zero(eltype(X)))
    E = X[s[2:end], :] .- transpose(a)
    gram = E * transpose(E)
    iszero(det(gram)) && return nothing
    h = gram \ (diag(gram) ./ 2)
    v = transpose(E) * h
    return (a + v, sum(abs2, v))
end

function _alpha_native_top(X, affine)
    n, k = size(X, 1), length(affine)
    binomial(big(n), k) <= 50_000 || throw(ArgumentError(
        "Exact enumerative alpha geometry exceeds 50,000 candidate simplices. Load MiniQhull for full-dimensional 3D input."))
    top = Vector{Int}[]
    # Exact coordinates in an affine basis; their metric is used only through
    # the original ambient coordinates. No rounded planar projection is used.
    E = X[affine[2:end], :] .- transpose(X[1, :])
    Y = k == 1 ? zeros(eltype(X), n, 0) :
        (X .- transpose(X[1, :])) * transpose(E) * inv(E * transpose(E))
    s = collect(1:k)
    while true
        ball = _alpha_sphere(X, s)
        if ball !== nothing
            c, r2 = ball
            good = true
            B = vcat(ones(eltype(X), 1, k), transpose(Y[s, :]))
            for q in 1:n
                q in s && continue
                delta = sum(abs2, X[q, :] - c) - r2
                if delta < 0
                    good = false; break
                elseif iszero(delta)
                    # Lexicographic infinitesimal lifted heights epsilon^id
                    # select one coherent regular triangulation of cospheres.
                    w = B \ [one(eltype(X)); Y[q, :]]
                    coeff = Dict(s[i] => -w[i] for i in 1:k)
                    coeff[q] = one(eltype(X))
                    firstid = minimum(id for (id, value) in coeff if !iszero(value))
                    if coeff[firstid] < 0
                        good = false; break
                    end
                end
            end
            good && push!(top, copy(s))
        end
        j = k
        while j > 0 && s[j] == n - k + j; j -= 1; end
        j == 0 && break
        s[j] += 1
        for l in j+1:k; s[l] = s[l-1] + 1; end
    end
    return top
end

function _alpha_hull_volume6(X, faces)
    # Count each supporting polygon once, even if a numerical candidate
    # covers a cospherical hull with two different triangulations.
    planes=Dict{NTuple{4,eltype(X)},Vector{Int}}()
    for (f,opposite) in faces
        length(opposite)==1 || continue
        a,b,c=(X[i,:] for i in f)
        normal=cross(b-a,c-a);pivot=findfirst(!iszero,normal)
        key=Tuple([normal./normal[pivot];dot(normal,a)/normal[pivot]])
        haskey(planes,key) || (planes[key]=findall(i->iszero(dot(normal,X[i,:]-a)),axes(X,1)))
    end
    center=vec(sum(X;dims=1))./size(X,1);volume=zero(eltype(X))
    for (plane,ids) in planes
        drop=findfirst(!iszero,plane[1:3]);cols=[i for i in 1:3 if i!=drop]
        order=sort(ids;by=i->Tuple(X[i,cols]))
        turn(a,b,c)=(X[b,cols[1]]-X[a,cols[1]])*(X[c,cols[2]]-X[a,cols[2]])-
                    (X[b,cols[2]]-X[a,cols[2]])*(X[c,cols[1]]-X[a,cols[1]])
        function chain(order)
            hull=Int[]
            for i in order
                while length(hull)>=2 && turn(hull[end-1],hull[end],i)<=0;pop!(hull);end
                push!(hull,i)
            end
            hull
        end
        lo,hi=chain(order),chain(reverse(order));polygon=[lo[1:end-1];hi[1:end-1]]
        for i in 2:length(polygon)-1
            E=X[[polygon[1],polygon[i],polygon[i+1]],:].-transpose(center)
            volume+=abs(det(E))
        end
    end
    volume
end

function _alpha_check_tetrahedra(X, top)
    n = size(X, 1)
    isempty(top) && throw(ArgumentError("Geometry backend returned no tetrahedra."))
    allunique(Tuple(s) for s in top) || throw(ArgumentError("Repeated backend tetrahedra."))
    faces = Dict{NTuple{3,Int},Vector{Int}}()
    for s in top
        length(s) == 4 && issorted(s) && allunique(s) && all(i -> 1 <= i <= n, s) ||
            throw(ArgumentError("Invalid backend tetrahedron identities."))
        ball = _alpha_sphere(X, s)
        ball === nothing && throw(ArgumentError("Geometry backend returned a flat tetrahedron."))
        c, r2 = ball
        all(q -> sum(abs2, X[q, :] - c) >= r2, 1:n) ||
            throw(ArgumentError("Numerical tetrahedralization failed exact empty-sphere verification; try delaunay_backend=:naive."))
        for i in 1:4
            face = Tuple(s[j] for j in 1:4 if j != i)
            push!(get!(faces, face, Int[]), s[i])
        end
    end
    sort!(unique!(reduce(vcat, top))) == collect(1:n) ||
        throw(ArgumentError("Geometry backend omitted input sites."))
    # Interior faces must join opposite tetrahedra; boundary faces must be
    # supporting hull faces. Together with exact nondegenerate empty spheres
    # this rejects holes, overlaps and incomplete numerical triangulations.
    for (f, opposite) in faces
        a, b, c = (X[i, :] for i in f)
        normal = cross(b-a, c-a)
        side(q) = dot(normal, X[q, :] - a)
        if length(opposite) == 2
            side(opposite[1]) * side(opposite[2]) < 0 ||
                throw(ArgumentError("Geometry backend returned overlapping tetrahedra."))
        elseif length(opposite) == 1
            sign0 = sign(side(only(opposite)))
            all(q -> sign0 * side(q) >= 0, 1:n) ||
                throw(ArgumentError("Geometry backend returned an incomplete convex hull."))
        else
            throw(ArgumentError("Geometry backend returned a nonmanifold face."))
        end
    end
    volume=sum(s->abs(det(X[s[2:end],:].-transpose(X[s[1],:]))),top)
    volume==_alpha_hull_volume6(X,faces) ||
        throw(ArgumentError("Geometry backend does not cover the convex hull exactly once."))
    return nothing
end

function _graded_complex_from_spatial_alpha(data, spec; return_simplex_tree=false)
    X = _spatial_alpha_points(data)
    _construction_check_counts!([size(X,1)],spec)
    affine = _alpha_affine_basis(X)
    dim = length(affine)-1
    requested = min(Int(get(spec.params, :max_dim, 3)), dim)
    backend = get(spec.params, :delaunay_backend, :auto)
    if requested == 0
        top = [[i] for i in axes(X, 1)]
        dim = 0
        _record_ingestion_backend(:spatial_alpha, :exact_vertices)
    elseif dim == 3 && size(X, 1) > 4 && backend !== :naive && _SPATIAL_DELAUNAY_IMPL[] !== nothing
        top = _SPATIAL_DELAUNAY_IMPL[](Float64.(X))
        _alpha_check_tetrahedra(X, top)
        _record_ingestion_backend(:spatial_alpha, :miniqhull_exactly_checked)
    else
        backend === :fast && dim == 3 && size(X, 1) > 4 &&
            throw(ArgumentError("Spatial delaunay_backend=:fast requires MiniQhull; run using MiniQhull."))
        top = _alpha_native_top(X, affine)
        _record_ingestion_backend(:spatial_alpha, :native_exact)
    end
    _construction_check_max_simplices!(length(top), dim, spec)
    sets = [Set{Tuple{Vararg{Int}}}() for _ in 0:dim]
    for s in top
        push!(sets[end], Tuple(s))
    end
    for k in dim+1:-1:2
        for s in sets[k], omit in eachindex(s)
            push!(sets[k-1], Tuple(s[j] for j in eachindex(s) if j != omit))
        end
        _construction_check_counts!(length.(sets), spec)
    end
    simplices = [[collect(s) for s in sort!(collect(ss))] for ss in sets]
    values = Dict{Tuple{Vararg{Int}},Rational{BigInt}}()
    for k in dim+1:-1:1, s in simplices[k]
        c, r2 = _alpha_sphere(X, s)
        key = Tuple(s)
        if all(q -> sum(abs2, X[q, :] - c) >= r2, axes(X, 1))
            values[key] = r2
        else
            haskey(values, key) || error("Non-Gabriel simplex lacks a Delaunay coface.")
        end
        for omit in eachindex(s)
            k == 1 && break
            face = Tuple(s[j] for j in eachindex(s) if j != omit)
            values[face] = min(get(values, face, values[key]), values[key])
        end
    end
    resize!(simplices, requested+1)
    grades = [(values[Tuple(s)],) for ss in simplices for s in ss]
    return _materialize_simplicial_output(simplices, grades, spec; return_simplex_tree)
end
