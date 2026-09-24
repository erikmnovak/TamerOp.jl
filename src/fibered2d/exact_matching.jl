# The two slope charts use the weighted line parameter u: in the shallow chart
# x = u/q, y = u+h, with 0 < q <= 1. Vertical and horizontal grid crossings
# therefore have affine parameters q*x and y-h. The second chart swaps x,y.
# Geometry and bottleneck comparisons use exact ordered-field arithmetic; only the
# returned distance is rounded to Float64.
const _ExactMatchingPoint2D{T} = NTuple{2,T}
const _ExactMatchingAffine2D{T} = NTuple{3,T}
const _ExactMatchingBar2D{T} = NTuple{2,_ExactMatchingAffine2D{T}}

mutable struct _ExactMatchingBudget2D
    used::Int
    limit::Int
end

function _exact_matching_charge!(budget::_ExactMatchingBudget2D, count::Integer=1)
    count <= budget.limit - budget.used || throw(ArgumentError(
        "matching_distance_exact_2d: max_candidates=$(budget.limit) exceeded " *
        "while constructing the exact arrangement; increase max_candidates " *
        "or use matching_distance_sampled_2d explicitly"))
    budget.used += Int(count)
    return nothing
end

@inline _exact_matching_value(f::_ExactMatchingAffine2D{T}, p::_ExactMatchingPoint2D{T}) where {T} =
    f[1]*p[1] + f[2]*p[2] + f[3]

# A line and its negative have the same normalized key. Constant differences
# have no zero line and are discarded before calling this helper.
function _exact_matching_line(f::_ExactMatchingAffine2D{T}) where {T}
    scale = !iszero(f[1]) ? f[1] : f[2]
    return (f[1]/scale, f[2]/scale, f[3]/scale)
end

function _exact_matching_halfplane(poly::Vector{_ExactMatchingPoint2D{T}},
                                   line::_ExactMatchingAffine2D{T}, positive::Bool) where {T}
    out = _ExactMatchingPoint2D{T}[]
    prev = poly[end]
    vp = _exact_matching_value(line, prev)
    for curr in poly
        vc = _exact_matching_value(line, curr)
        inp = positive ? vp >= 0 : vp <= 0
        inc = positive ? vc >= 0 : vc <= 0
        if inp != inc
            t = vp/(vp-vc)
            point = (prev[1]+t*(curr[1]-prev[1]), prev[2]+t*(curr[2]-prev[2]))
            (isempty(out) || out[end] != point) && push!(out, point)
        end
        inc && (isempty(out) || out[end] != curr) && push!(out, curr)
        prev, vp = curr, vc
    end
    length(out) > 1 && out[end] == out[1] && pop!(out)
    return out
end

# Every mixed crossing q*x = y-h is a barcode-combinatorics wall. Including
# the window sides here also handles intersections of grid lines with a window
# whose corners are not on the encoding grid.
function _exact_matching_cells(xs::Vector{T},
                               ys::Vector{T},
                               budget::_ExactMatchingBudget2D) where {T}
    z, o = zero(T), one(T)
    xmin, xmax, ymin, ymax = xs[1], xs[end], ys[1], ys[end]
    cells = Vector{_ExactMatchingPoint2D{T}}[
        [(z,ymin), (o,ymin-xmax), (o,ymax-xmin), (z,ymax)],
    ]
    for x in xs, y in ys
        line = (x,o,-y)
        next = Vector{_ExactMatchingPoint2D{T}}[]
        for poly in cells
            _exact_matching_charge!(budget)
            vals = [_exact_matching_value(line,p) for p in poly]
            if minimum(vals) < 0 < maximum(vals)
                push!(next, _exact_matching_halfplane(poly,line,false))
                push!(next, _exact_matching_halfplane(poly,line,true))
            else
                push!(next,poly)
            end
        end
        cells = next
    end
    return cells
end

function _exact_matching_coordinates(coords, lo::T,
                                      hi::T) where {T}
    cuts = T[lo,hi]
    for x in coords
        isnan(x) && throw(ArgumentError("matching_distance_exact_2d: NaN grid coordinate"))
        isfinite(x) || continue
        q = T(x)
        lo < q < hi && push!(cuts,q)
    end
    sort!(cuts)
    unique!(cuts)
    return cuts
end

# Native box location normally converts to Float64 (including its fast uniform
# grid route). Bypass that conversion for exact rational/algebraic queries.
function _exact_matching_locate(pi, x::_ExactMatchingPoint2D{T}) where {T}
    boxowner = getfield(parentmodule(@__MODULE__), :PLBackend)
    if pi isa boxowner.PLEncodingMapBoxes
        idx = 1
        for axis in 1:2
            coords = pi.coords[axis]
            lo, hi = 0, length(coords)+1
            while lo+1 < hi
                mid = (lo+hi) >>> 1
                cut = coords[mid]
                if cut == -Inf || (isfinite(cut) && T(cut) < x[axis])
                    lo = mid
                else
                    hi = mid
                end
            end
            idx += lo*pi.cell_strides[axis]
        end
        return pi.cell_to_region[idx]
    end
    # Custom box encodings must locate interior points in the exact coordinate
    # field without loss and be constant on their declared open axis cells.
    return locate(pi,collect(x))
end

function _exact_matching_cell_bars(cacheM::FiberedBarcodeCache2D,
                                  cacheN::FiberedBarcodeCache2D,
                                  poly::Vector{_ExactMatchingPoint2D{T}},
                                  xs::Vector{T},
                                  ys::Vector{T}, swapped::Bool,
                                  budget::_ExactMatchingBudget2D) where {T}
    # The centroid lies strictly inside this full-dimensional geometric cell.
    # Successive event midpoints then lie in open grid cells even when distinct
    # exact coordinates have the same floating-point representation.
    p = (sum(v[1] for v in poly)/length(poly), sum(v[2] for v in poly)/length(poly))
    q,h = p
    z, o = zero(q), one(q)
    events = _ExactMatchingAffine2D{T}[(x,z,z) for x in xs]
    append!(events,[(z,-o,y) for y in ys])
    lower, upper = max(q*xs[1],ys[1]-h), min(q*xs[end],ys[end]-h)
    filter!(f -> lower <= _exact_matching_value(f,p) <= upper,events)
    sort!(events; by=f -> _exact_matching_value(f,p))
    chain = Int[]
    bounds = _ExactMatchingAffine2D{T}[events[1]]
    for i in 1:length(events)-1
        u = (_exact_matching_value(events[i],p)+_exact_matching_value(events[i+1],p))/2
        point = swapped ? (u+h,u/q) : (u/q,u+h)
        region = _exact_matching_locate(cacheM.arrangement.pi,point)
        1 <= region <= length(cacheM.M.dims) || throw(ArgumentError(
            "matching_distance_exact_2d: the window contains an unrepresented grid cell; " *
            "choose a fully represented window"))
        if isempty(chain)
            push!(chain,region)
        elseif chain[end] != region
            leq(cacheM.M.Q,chain[end],region) || throw(ArgumentError(
                "matching_distance_exact_2d: the encoding does not map positive slices to poset chains"))
            push!(bounds,events[i])
            push!(chain,region)
        end
    end
    push!(bounds,events[end])
    cid = _arr2d_chain_id!(cacheM.arrangement,chain)
    bars = map((cacheM,cacheN)) do cache
        packed = _index_packed_for_chain!(cache,cid)
        _exact_matching_charge!(budget,sum(big(m) for m in packed.mults;init=big(0)))
        result = _ExactMatchingBar2D{T}[]
        for i in eachindex(packed.pairs)
            interval = packed.pairs[i]
            for _ in 1:packed.mults[i]
                push!(result,(bounds[interval.b],bounds[interval.d]))
            end
        end
        result
    end
    return bars
end

function _exact_matching_candidates(poly::Vector{_ExactMatchingPoint2D{T}},
                                   barsM::Vector{_ExactMatchingBar2D{T}},
                                   barsN::Vector{_ExactMatchingBar2D{T}},
                                   budget::_ExactMatchingBudget2D) where {T}
    z = zero(T)
    forms = Set{_ExactMatchingAffine2D{T}}([(z,z,z)])
    _exact_matching_charge!(budget,2*big(length(barsM))*length(barsN))
    for bars in (barsM,barsN), (birth,death) in bars
        push!(forms,ntuple(i -> (death[i]-birth[i])/2,3))
    end
    for a in barsM, b in barsN, endpoint in 1:2
        f = ntuple(i -> a[endpoint][i]-b[endpoint][i],3)
        push!(forms,f)
        push!(forms,ntuple(i -> -f[i],3))
    end
    fs = sort!(collect(forms))
    lines = Set{_ExactMatchingAffine2D{T}}()
    _exact_matching_charge!(budget,div(big(length(fs))*(length(fs)-1),2))
    for i in eachindex(fs), j in 1:i-1
        line = ntuple(k -> fs[i][k]-fs[j][k],3)
        iszero(line[1]) && iszero(line[2]) && continue
        # Lines merely touching the polygon at a vertex, or coinciding with
        # an edge, add no subdivision and need no additional intersections.
        vals = [_exact_matching_value(line,p) for p in poly]
        minimum(vals) < 0 < maximum(vals) || continue
        push!(lines,_exact_matching_line(line))
    end
    # Add the polygon's supporting lines, so switch/border intersections are
    # included alongside switch/switch intersections and geometric vertices.
    for i in eachindex(poly)
        a,b = poly[i],poly[mod1(i+1,length(poly))]
        line = (a[2]-b[2], b[1]-a[1], a[1]*b[2]-a[2]*b[1])
        push!(lines,_exact_matching_line(line))
    end
    ls = sort!(collect(lines))
    _exact_matching_charge!(budget,div(big(length(ls))*(length(ls)-1),2))
    candidates = Set(poly)
    for i in eachindex(ls), j in 1:i-1
        a,b = ls[i],ls[j]
        det = a[1]*b[2]-b[1]*a[2]
        iszero(det) && continue
        p = ((a[2]*b[3]-b[2]*a[3])/det, (a[3]*b[1]-b[3]*a[1])/det)
        inside = true
        for k in eachindex(poly)
            u,v = poly[k],poly[mod1(k+1,length(poly))]
            if (v[1]-u[1])*(p[2]-u[2])-(v[2]-u[2])*(p[1]-u[1]) < 0
                inside = false
                break
            end
        end
        inside && push!(candidates,p)
    end
    return sort!(collect(candidates))
end

# Use the same scalar/witness engine as ordinary slice barcodes, preserving
# exact thresholds throughout. Finite-window endpoints cannot be infinite.
function _exact_matching_bottleneck(A::Vector{_ExactMatchingPoint2D{T}},
                                   B::Vector{_ExactMatchingPoint2D{T}}) where {T}
    owner = getfield(parentmodule(@__MODULE__), :SliceInvariants)
    return owner._bottleneck_distance_points(A, B)
end

struct _ExactMatchingWork2D{T<:Real}
    candidates::Vector{_ExactMatchingPoint2D{T}}
    barsM::Vector{_ExactMatchingBar2D{T}}
    barsN::Vector{_ExactMatchingBar2D{T}}
end

function _exact_matching_cell_max(work::_ExactMatchingWork2D{T}) where {T}
    best = zero(T)
    for p in work.candidates
        # Compactifying the slope chart contributes zero: clipped bars have
        # weighted lifespan at most q times the finite horizontal box width.
        iszero(p[1]) && continue
        # On a geometric wall, coalescing crossings compresses the chain;
        # functoriality preserves its composite maps and only zero-length bars
        # disappear. The closed-cell formulas therefore give the boundary
        # barcode, without a rounded boundary query. See docs/exact_matching.md.
        A = [(_exact_matching_value(a,p),_exact_matching_value(b,p)) for (a,b) in work.barsM]
        B = [(_exact_matching_value(a,p),_exact_matching_value(b,p)) for (a,b) in work.barsN]
        best = max(best,_exact_matching_bottleneck(A,B))
    end
    return best
end

function _matching_distance_box_exact_2d(cacheM::FiberedBarcodeCache2D,
                                        cacheN::FiberedBarcodeCache2D;
                                        max_candidates::Int=200_000,
                                        threads::Bool=false)::Float64
    arr = cacheM.arrangement
    arr === cacheN.arrangement || throw(ArgumentError("matching_distance_exact_2d: caches must share an arrangement"))
    arr.backend === :boxes && hasproperty(arr.pi,:coords) || throw(ArgumentError(
        "matching_distance_exact_2d requires an axis-aligned box encoding"))
    dimension(arr.pi) == 2 && length(arr.pi.coords) == 2 || throw(ArgumentError(
        "matching_distance_exact_2d requires exactly two coordinate axes"))
    arr.pi isa ZnEncodingMap && throw(ArgumentError(
        "matching_distance_exact_2d does not support rounded lattice point location; " *
        "use a continuous box encoding"))
    if hasproperty(arr.pi,:orientation) && any(!=(1),arr.pi.orientation)
        throw(ArgumentError("matching_distance_exact_2d requires positively oriented grid coordinates"))
    end
    max_candidates > 0 || throw(ArgumentError("max_candidates must be positive"))
    a,b = arr.input_box
    length(a) == 2 && length(b) == 2 || throw(ArgumentError(
        "matching_distance_exact_2d requires two-dimensional window endpoints"))
    all(isfinite,a) && all(isfinite,b) || throw(ArgumentError(
        "matching_distance_exact_2d requires a finite window"))
    all(a .<= b) || throw(ArgumentError("matching_distance_exact_2d requires ordered window endpoints"))
    any(a .== b) && return 0.0
    T = eltype(a)
    ax = [T(x) for x in a]
    bx = [T(x) for x in b]
    xs = _exact_matching_coordinates(arr.pi.coords[1],ax[1],bx[1])
    ys = _exact_matching_coordinates(arr.pi.coords[2],ax[2],bx[2])
    budget = _ExactMatchingBudget2D(0,max_candidates)
    work = _ExactMatchingWork2D{T}[]
    for swapped in (false,true)
        cx,cy = swapped ? (ys,xs) : (xs,ys)
        for poly in _exact_matching_cells(cx,cy,budget)
            barsM,barsN = _exact_matching_cell_bars(cacheM,cacheN,poly,cx,cy,swapped,budget)
            isempty(barsM) && isempty(barsN) && continue
            candidates = _exact_matching_candidates(poly,barsM,barsN,budget)
            push!(work,_ExactMatchingWork2D(candidates,barsM,barsN))
        end
    end
    # All cache mutation and algebra happens above. Workers own their diagrams
    # and matching graphs, and write only to their deterministic result index.
    maxima = fill(zero(T),length(work))
    if threads && Threads.nthreads() > 1
        Threads.@threads for i in eachindex(work)
            maxima[i] = _exact_matching_cell_max(work[i])
        end
    else
        for i in eachindex(work)
            maxima[i] = _exact_matching_cell_max(work[i])
        end
    end
    return Float64(maximum(maxima;init=zero(T)))
end
