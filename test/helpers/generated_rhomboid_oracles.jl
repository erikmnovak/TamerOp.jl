# Finite, deterministic A77 references. This file deliberately does not call
# production sphere predicates, CDD, linear algebra, or radius propagation.
# The A72 elimination, minimum-ball certificate and simplicial-chain helpers
# are independent test implementations shared by these generated families.

function _a77_subsets(n, cardinality)
    return [findall(i -> !iszero(mask & (1 << (i-1))), 1:n)
            for mask in 0:(1 << n)-1 if count_ones(mask) == cardinality]
end

function _a77_clouds()
    cases = NamedTuple[]
    line = QQ[-1//2, 0, 1//2, 3//2]
    for cardinality in (2, 3), ids in _a77_subsets(4, cardinality)
        push!(cases, (name="line_$(join(ids, '_'))", X=reshape(line[ids], :, 1), family=:line))
    end
    planar = QQ[0 0; 1 0; 0 1; 1 1; 1//2 3//2]
    for ids in _a77_subsets(5, 3)
        push!(cases, (name="plane_$(join(ids, '_'))", X=planar[ids, :], family=:plane))
    end
    for a in 0:2, b in a:2, c in b:2
        push!(cases, (name="multiset_$(a)_$(b)_$(c)", X=reshape(QQ[a,b,c], :, 1), family=:multiset))
    end
    for scale in QQ[1//2, 3//2]
        square = scale .* QQ[0 0; 1 0; 1 1; 0 1]
        tetra = scale .* QQ[1 1 1; 1 -1 -1; -1 1 -1; -1 -1 1]
        push!(cases, (name="square_$scale", X=square, family=:square))
        push!(cases, (name="tetra_$scale", X=tetra, family=:tetra))
    end
    # Lower affine dimension in an oblique ambient subspace. The ambient norm
    # is retained; these embeddings are intentionally not all isometries.
    for scale in QQ[1//2, 3//2]
        X = scale .* QQ[0 0; 2 0; 1 2; 3 3]
        embedded = hcat(X, X[:,1] + 2X[:,2])
        push!(cases, (name="oblique_$scale", X=embedded, family=:oblique))
        constrained = scale .* QQ[0 0; 4 0; 1 1; 0 3]
        push!(cases, (name="constrained_$scale", X=constrained, family=:constrained))
    end
    return cases
end

function _a77_affine_solution(E, f)
    d = size(E, 2)
    R, pivots = _a72_rref(hcat(E, f))
    any(==(d+1), pivots) && return nothing
    particular = zeros(QQ, d)
    for (row, column) in enumerate(pivots)
        particular[column] = R[row, end]
    end
    return particular, _a72_nullspace_from_rref(R[:,1:d], pivots)
end

# Strict rational Fourier--Motzkin elimination. Equalities are eliminated
# first; every remaining inequality is strict. Unbounded systems are allowed.
function _a77_strict_feasible(E, f, A, b)
    solved = _a77_affine_solution(E, f)
    solved === nothing && return false
    particular, kernel = solved
    A, b = A * kernel, b - A * particular
    while size(A, 2) > 0
        positive = findall(>(0), A[:,end])
        negative = findall(<(0), A[:,end])
        zero_rows = findall(iszero, A[:,end])
        rows = [collect(A[i,1:end-1]) for i in zero_rows]
        rhs = QQ[b[i] for i in zero_rows]
        for p in positive, m in negative
            push!(rows, collect(A[p,1:end-1] / A[p,end] - A[m,1:end-1] / A[m,end]))
            push!(rhs, b[p] / A[p,end] - b[m] / A[m,end])
        end
        columns = size(A, 2)-1
        A = isempty(rows) ? zeros(QQ, 0, columns) : reduce(vcat, permutedims.(rows))
        b = rhs
        for row in axes(A,1)
            all(iszero, A[row,:]) && b[row] <= 0 && return false
        end
    end
    return all(>(0), b)
end

function _a77_carrier_feasible(X, inside, on)
    n, d = size(X)
    E, A = zeros(QQ, 0, d+1), zeros(QQ, 0, d+1)
    f, b = QQ[], QQ[]
    for i in 1:n
        row = vcat(2collect(X[i,:]), one(QQ))
        norm2 = sum(abs2, X[i,:])
        bit = 1 << (i-1)
        if !iszero(on & bit)
            E = vcat(E, permutedims(row)); push!(f, norm2)
        else
            sign = iszero(inside & bit) ? 1 : -1
            A = vcat(A, permutedims(sign .* row)); push!(b, sign * norm2)
        end
    end
    return _a77_strict_feasible(E, f, A, b)
end

# Minimize ||c-p||^2 on a rational affine polyhedron by enumerating all active
# inequality faces. The minimizer lies in the relative interior of one face;
# its affine projection is therefore included. Every retained candidate is
# primal feasible, so taking the minimum cannot underestimate the optimum.
function _a77_projected_minimum(p, E, f, A, b)
    best = nothing
    for active in 0:(1 << size(A,1))-1
        ids = findall(i -> !iszero(active & (1 << (i-1))), axes(A,1))
        C, y = vcat(E, A[ids,:]), vcat(f, b[ids])
        solved = _a77_affine_solution(C, y)
        solved === nothing && continue
        c0, V = solved
        center = isempty(V) ? c0 : c0 + V * _a72_solve(transpose(V)*V,
                                                      reshape(transpose(V)*(p-c0), :, 1))[:,1]
        all(A * center .<= b) || continue
        value = sum(abs2, center-p)
        best = best === nothing ? value : min(best, value)
    end
    return best
end

function _a77_carrier_radius_squared(X, inside, on)
    n, d = size(X)
    iszero(inside | on) && return zero(QQ)
    # At a vertex minimum the radius can be reduced until an inside site is
    # tight. Try every such site; a nonempty on-set already supplies one.
    anchors = findall(i -> !iszero((iszero(on) ? inside : on) & (1 << (i-1))), 1:n)
    best = nothing
    for anchor in (iszero(on) ? anchors : anchors[1:1])
        p = collect(X[anchor,:])
        E, A = zeros(QQ,0,d), zeros(QQ,0,d)
        f, b = QQ[], QQ[]
        for i in 1:n
            i == anchor && continue
            row = 2 .* (p - collect(X[i,:]))
            rhs = sum(abs2,p) - sum(abs2,X[i,:])
            bit = 1 << (i-1)
            if !iszero(on & bit)
                E = vcat(E,permutedims(row)); push!(f,rhs)
            else
                sign = iszero(inside & bit) ? -1 : 1
                A = vcat(A,permutedims(sign .* row)); push!(b,sign * rhs)
            end
        end
        # Some possible inside anchors cannot be active on this closure.
        closed = _a77_affine_solution(E,f)
        closed === nothing && continue
        value = _a77_projected_minimum(p,E,f,A,b)
        value === nothing && continue
        best = best === nothing ? value : min(best,value)
    end
    best === nothing && error("No independent active-site radius certificate")
    return best
end

function _a77_carriers(X)
    n = size(X,1)
    result = Dict{Tuple{Int,Int},QQ}()
    for inside in 0:(1 << n)-1, on in 0:(1 << n)-1
        iszero(inside & on) || continue
        _a77_carrier_feasible(X,inside,on) || continue
        result[(inside,on)] = _a77_carrier_radius_squared(X,inside,on)
    end
    return result
end

function _a77_native_admissible(X)
    n = size(X,1)
    allunique(Tuple.(eachrow(X))) || return false
    _, pivots = _a72_rref(X[2:end,:] .- X[1:1,:])
    dimension = length(pivots)
    # Deliberately sufficient general-position hypotheses. Rejected generated
    # clouds use the exact fallback; no production error is silently skipped.
    for cardinality in 2:min(n,dimension+1), ids in _a77_subsets(n,cardinality)
        _, columns = _a72_rref(X[ids[2:end],:] .- X[ids[1:1],:])
        length(columns) == cardinality-1 || return false
    end
    for ids in _a77_subsets(n,dimension+2)
        p = X[first(ids),:]
        E = 2 .* (X[ids[2:end],:] .- permutedims(p))
        f = [sum(abs2,X[i,:])-sum(abs2,p) for i in ids[2:end]]
        _a77_affine_solution(E,f) === nothing || return false
    end
    return true
end

function _a77_radius_probes(values)
    levels = sort!(unique!(QQ[0; collect(values)]))
    squared = sort!(unique!(vcat(levels, [(levels[i]+levels[i+1])/2 for i in 1:length(levels)-1])))
    return sqrt.(TamerOp.ExactReals.AlgebraicReal.(squared))
end

function _a77_check_cube_boundary(cells, G)
    for slot in 2:length(cells)
        rows = Dict((Int(c.inside),Int(c.on)) => i for (i,c) in enumerate(cells[slot-1]))
        expected = zeros(Int,length(rows),length(cells[slot]))
        for (column,c) in enumerate(cells[slot])
            generators = findall(i -> !iszero(Int(c.on) & (1 << (i-1))), 1:64)
            for (position,site) in enumerate(generators)
                bit = 1 << (site-1)
                lower = (Int(c.inside),xor(Int(c.on), bit))
                upper = (Int(c.inside) | bit,xor(Int(c.on), bit))
                expected[rows[upper],column] += (-1)^(position-1)
                expected[rows[lower],column] -= (-1)^(position-1)
            end
        end
        @test G.boundaries[slot-1] == expected
    end
end

function _a77_expected_depth_cells(carriers, window, n, maxdim)
    lo, hi = window
    expected = [Dict{Tuple,Tuple{QQ,Int}}() for _ in 0:maxdim]
    for ((inside,on),radius2) in carriers, slab in (false,true), level in lo:hi
        slab && level == hi && continue
        vertices = _a76_slice_vertices(inside,on,level,slab,n)
        isempty(vertices) && continue
        slab && length(unique(count_ones.(vertices))) == 1 && continue
        dimension = size(_a76_slice_basis(vertices,n),2)
        dimension <= maxdim || continue
        signature = Tuple(vertices)
        existing = get(expected[dimension+1],signature,nothing)
        expected[dimension+1][signature] = (existing === nothing ? radius2 : min(first(existing),radius2),level)
    end
    return expected
end

function _a77_check_native_windows(X, carriers; family)
    n = size(X,1)
    dimension = length(last(_a72_rref(X[2:end,:] .- X[1:1,:])))
    top = dimension+1
    windows = [(lo,hi) for lo in 0:n for hi in lo:n]
    critical = sort!(unique!(collect(values(carriers))))
    counts = (windows=0, cutoff_builds=0, carriers=length(carriers))
    for backend in (:exhaustive,:incremental)
        spec = OPT.FiltrationSpec(kind=:rhomboid,backend=backend)
        cells,actual_radii = DI._rhomboid_geometry(X,dimension,spec,UInt64;backend)
        actual = Dict((Int(c.inside),Int(c.on)) => actual_radii[c] for group in cells for c in group)
        @test actual == carriers
        full = DI.encode(DT.PointCloud(X),spec;stage=:graded_complex)
        _a77_check_cube_boundary(cells,full)
        @test all(full.grades[i][1]^2 == carriers[(Int(c.inside),Int(c.on))] &&
                  full.grades[i][2] == count_ones(c.inside)
                  for (i,c) in enumerate(Iterators.flatten(cells)))
        for window in windows
            bounded = OPT.FiltrationSpec(kind=:rhomboid,backend=backend,depth_range=window)
            retained,radii = DI._rhomboid_geometry(X,dimension,bounded,UInt64;backend)
            groups,G = DI._rhomboid_depth_model(retained,radii,n,bounded)
            expected = _a77_expected_depth_cells(carriers,window,n,top)
            actual_groups = [Dict(Tuple(_a76_slice_vertices(c.carrier.inside,c.carrier.on,c.level,c.slab,n)) =>
                (radii[c.carrier],c.level) for c in group) for group in groups]
            @test actual_groups == expected
            _a76_check_depth_boundaries(groups,G,n)
            counts = merge(counts,(windows=counts.windows+1,))
            # Every exact native critical radius, including zero and equality.
            # Expected restriction is read from independent squared grades.
            for cutoff2 in critical
                cutoff = sqrt(TamerOp.ExactReals.AlgebraicReal(cutoff2))
                request = OPT.FiltrationSpec(kind=:rhomboid,backend=backend,
                    depth_range=window,radius=cutoff)
                built = DI.encode(DT.PointCloud(X),request;stage=:graded_complex)
                active = [[i for (i,c) in enumerate(group) if
                    carriers[(Int(c.carrier.inside),Int(c.carrier.on))] <= cutoff2] for group in groups]
                @test DT.cell_counts(built) == length.(active)
                offsets = cumsum([0;length.(groups)])
                @test built.grades == [G.grades[offsets[slot]+i] for slot in eachindex(groups) for i in active[slot]]
                @test built.boundaries == [G.boundaries[slot][active[slot],active[slot+1]] for slot in eachindex(G.boundaries)]
                counts = merge(counts,(cutoff_builds=counts.cutoff_builds+1,))
            end
        end
    end
    println("A77 native geometry family=",family," sites=",n," ambient=",size(X,2)," coverage=",counts)
    return counts
end

function _a77_check_fallback_windows(X, reference; family)
    n = size(X,1)
    construction = OPT.ConstructionOptions(budget=OPT.ConstructionBudget(max_simplices=50_000))
    critical = sort!(unique(reference.radii2))
    counts = 0
    for lo in 0:n, hi in lo:n, cutoff2 in critical
        cutoff = sqrt(TamerOp.ExactReals.AlgebraicReal(cutoff2))
        spec = OPT.FiltrationSpec(kind=:rhomboid,backend=:subdivision_cech,
            depth_range=(lo,hi),max_dim=3,radius=cutoff,construction=construction)
        G,flags,masks,radii2 = _a68_flag_fixture(X,spec)
        expected_masks = [mask for mask in 0:(1 << n)-1 if count_ones(mask) >= lo && reference.radii2[mask+1] <= cutoff2]
        @test Int.(masks) == expected_masks
        @test radii2 == reference.radii2[expected_masks .+ 1]
        expected_flags = [[flag for flag in group if count_ones(first(flag)) >= lo &&
            reference.radii2[last(flag)+1] <= cutoff2] for group in reference.simplices[1:length(flags)]]
        @test flags == expected_flags
        @test G.boundaries == _a72_boundaries(expected_flags)
        @test [(g[1]^2,Int(g[2])) for g in G.grades] ==
            [(reference.radii2[last(flag)+1],min(count_ones(first(flag)),hi)) for group in expected_flags for flag in group]
        counts += 1
    end
    println("A77 fallback geometry family=",family," sites=",n," cutoff_windows=",counts)
    return counts
end

function _a77_check_generated_persistence(case,index)
    X = case.X
    n = size(X,1)
    reference = _a72_subdivision_cech(X;maxdim=3)
    radii = _a77_radius_probes(reference.radii2)
    admissible = _a77_native_admissible(X)
    backend = admissible ? (isodd(index) ? :exhaustive : :incremental) : :subdivision_cech
    construction = OPT.ConstructionOptions(budget=OPT.ConstructionBudget(max_simplices=50_000))
    degrees = case.family === :tetra ? (0,1,2) : size(X,2) == 1 ? (0,) : (0,1)
    # Every cloud gets the full two-parameter grid and a single positive cap.
    # A deterministic cutoff retains all lower critical radii and their open
    # intervals while testing a proper finite radius window when possible.
    critical = sort!(unique(reference.radii2))
    cutoff2 = critical[max(1,length(critical)-Int(iseven(index) && case.family !== :tetra))]
    cutoff = sqrt(TamerOp.ExactReals.AlgebraicReal(cutoff2))
    radius_queries = filter(<=(cutoff),radii)
    cap = mod1(index,n)
    windows = [(0,n),(cap,cap)]
    (case.name == "plane_1_2_5" || case.name == "tetra_1//2") && push!(windows,(1,2))
    results = Dict{Tuple{Int,Int},Any}()
    for (window_index,window) in enumerate(windows)
        spec = OPT.FiltrationSpec(kind=:rhomboid,backend=backend,depth_range=window,
            max_dim=3,radius=cutoff,construction=construction)
        if admissible
            G,unions = _a66_native_fixture(X,spec)
            comparison = _a66_carrier_comparison(G.boundaries,unions,reference)
        else
            G,flags,_,_ = _a68_flag_fixture(X,spec)
            comparison = _a66_flag_carrier_comparison(flags,reference)
        end
        for slot in eachindex(G.boundaries)
            @test reference.boundaries[slot] * comparison[slot+1] == comparison[slot] * G.boundaries[slot]
        end
        _a66_check_carrier_grades(G,reference,comparison)
        queries = [(r,k) for r in radius_queries for k in first(window):last(window)]
        active_reference = _a66_active_reference(reference,queries)
        active_source = [_a72_source_active(G,p,(1,-1)) for p in queries]
        # A high lower-depth cutoff can leave only vertices. Higher chain
        # groups then mean zero, including in comparisons of zero homology.
        while length(comparison) < maximum(degrees)+1
            push!(comparison,spzeros(Int,length(reference.simplices[length(comparison)+1]),0))
            foreach(a -> push!(a,Int[]),active_source)
        end
        for (field_index,field) in enumerate((CM.QQField(),CM.F2(),CM.F3())), degree in degrees
            expected = [size(_a72_reference_homology(reference,a,degree,field).basis,2) for a in active_reference]
            if window_index == 1
                if all(row -> row == X[1,:],eachrow(X))
                    @test all(==(1),expected) # multiplicity=n remains nonempty at radius zero
                elseif (degree == 2 && case.family === :tetra) ||
                       (degree == 1 && (case.family === :square || case.name == "plane_1_2_5"))
                    @test any(>(0),expected) # ensure the generated map test is not vacuous
                    @test all(expected[j] == 0 for j in eachindex(queries) if first(queries[j]) == cutoff)
                end
            end
            computed = _a72_check_module_comparison(DT.PointCloud(X),spec,G,reference,comparison,
                queries,active_reference,active_source,degree,expected,field)
            key = (field_index,degree)
            if window_index == 1
                results[key] = (computed,queries)
            else
                full,full_queries = results[key]
                ids = [findfirst(==(q),full_queries) for q in queries]
                restricted = (module_object=full.module_object,labels=full.labels[ids],comparison_maps=full.comparison_maps[ids])
                _a66_compare_through_reference(restricted,computed,queries)
            end
        end
    end
    println("A77 persistence cloud=",case.name," backend=",backend," cutoff2=",cutoff2,
        " radii=",length(radius_queries)," cap=",cap," fields=QQ,F2,F3 degrees=",degrees)
    return nothing
end
