# Continuous GRIL worms (Xin et al., PMLR 221, 2023, Definition 3.2):
# |x-px| <= ell*delta, |y-py| <= ell*delta,
# |x+y-px-py| <= 2*delta. They are not a finite union of 2ell-1 squares.
# See docs/implementation/generalized_rank.md for the restriction and event arguments.

"""
    GRILResult

Exact GRIL values at fixed user-supplied centers, positive rank levels, and
worm lengths. Array order is `(center, level, length)`. The ambient module is
the right-continuous extension of a positively oriented two-dimensional grid:
zero below either first axis coordinate, constant above the last coordinates.
Coordinates and returned widths are exact rationals (floating inputs retain
their exact binary value). No learned probes or derivatives are computed.
"""
struct GRILResult
    centers::Vector{NTuple{2,QQ}}
    levels::Vector{Int}
    lengths::Vector{Int}
    values::Array{QQ,3}
    queries::Int
end
"""Return copies of the exact rational probe centers in result order."""
gril_centers(r::GRILResult) = copy(r.centers)
"""Return the requested positive rank levels in result order."""
gril_levels(r::GRILResult) = copy(r.levels)
"""Return the requested positive integer worm lengths in result order."""
gril_lengths(r::GRILResult) = copy(r.lengths)
"""Return a copy of exact widths in `(center, level, length)` array order."""
gril_values(r::GRILResult) = copy(r.values)
"""Flatten exact GRIL values with center varying fastest, then level, then length."""
feature_vector(r::GRILResult) = vec(copy(r.values))
"""Query a GRIL width by center index and the actual requested rank level and worm length."""
function value_at(r::GRILResult; center::Integer, level::Integer, length::Integer)
    k = findfirst(==(level),r.levels)
    ell = findfirst(==(length),r.lengths)
    isnothing(k) && throw(ArgumentError("rank level was not requested"))
    isnothing(ell) && throw(ArgumentError("worm length was not requested"))
    return r.values[center,k,ell]
end
"""Summarize a fixed-probe GRIL result, including shape, exactness, extension and rank-query count."""
gril_summary(r::GRILResult) =
    (kind=:gril, ncenters=length(r.centers), levels=copy(r.levels), lengths=copy(r.lengths),
     shape=size(r.values), queries=r.queries, exact=true, worm=:continuous,
     extension=:right_continuous_grid_zero_below_constant_above)
describe(r::GRILResult) = gril_summary(r)
Base.show(io::IO, r::GRILResult) =
    print(io, "GRILResult(shape=", size(r.values), ", exact=true, queries=", r.queries, ")")

function _gr_coordinate(x)
    x isa Union{Integer,Rational,AbstractFloat} && isfinite(x) ||
        throw(ArgumentError("GRIL coordinates must be finite integers, rationals, or floating-point numbers"))
    return QQ(x)
end
function _gr_center(center)
    center isa Union{Tuple,AbstractVector} && length(center) == 2 ||
        throw(ArgumentError("a GRIL center must have exactly two coordinates"))
    return (_gr_coordinate(center[1]), _gr_coordinate(center[2]))
end
function _gr_positive_indices(values, name)
    values isa Union{Tuple,AbstractVector} && !isempty(values) ||
        throw(ArgumentError("$name must be a nonempty finite vector or tuple"))
    all(v -> v isa Integer && !(v isa Bool) && 0 < v <= typemax(Int), values) ||
        throw(ArgumentError("$name must contain positive machine integers"))
    result = Int.(collect(values))
    allunique(result) || throw(ArgumentError("$name must be distinct"))
    return result
end

function _gr_grid(M::PModule, pi, budget)
    _gr_field(M)
    grid = _unwrap_compiled(pi)
    grid isa GridEncodingMap{2} ||
        throw(ArgumentError("GRIL requires a two-dimensional GridEncodingMap (or its compiled wrapper)"))
    grid.orientation == (1,1) || throw(ArgumentError("GRIL requires positively oriented grid axes"))
    grid.P isa AbstractPoset || throw(ArgumentError("the grid encoding must have a finite poset"))
    ax = map(a -> _gr_coordinate.(a), grid.coords)
    all(a -> !isempty(a) && issorted(a) && allunique(a), ax) ||
        throw(ArgumentError("GRIL axes must be nonempty and strictly increasing"))
    nx, ny = length.(ax)
    grid.sizes == (nx,ny) && grid.strides == (1,nx) ||
        throw(ArgumentError("grid size/stride metadata must agree with its axes"))
    n = big(nx)*ny
    n == nvertices(M.Q) == nvertices(grid.P) ||
        throw(ArgumentError("grid axes and module poset have incompatible sizes"))
    _gr_bound(n*n, budget.max_order_checks, "grid order validation")
    # Check the actual orders, not just equal vertex counts or object identity.
    for b in 1:Int(n), a in 1:Int(n)
        expected = (a-1)%nx <= (b-1)%nx && div(a-1,nx) <= div(b-1,nx)
        (leq(M.Q,a,b) == expected && leq(grid.P,a,b) == expected) ||
            throw(ArgumentError("module and encoding must use the grid's product order and column-major labels"))
    end
    return ax
end

function _gr_request(M, pi, centers, levels, lengths, budget)
    ax = _gr_grid(M,pi,budget)
    centers isa Union{AbstractVector,Tuple} || throw(ArgumentError("centers must be a finite vector or tuple of points"))
    _gr_bound(length(centers), budget.max_queries, "centers")
    points = NTuple{2,QQ}[_gr_center(p) for p in centers]
    ks = _gr_positive_indices(levels,"rank levels")
    ls = _gr_positive_indices(lengths,"worm lengths")
    _gr_bound(big(length(points))*length(ks)*length(ls), budget.max_queries, "output entries")
    return ax, points, ks, ls
end

"""
    check_gril_query(M, grid; centers, levels=(1,), lengths=(1,),
                     budget=GeneralizedRankBudget(), throw=false)

Check the exact field, positive 2D grid order, coordinates and probe/output
contract without computing ranks. Data-dependent event/algebra budgets are
checked during computation. Returns a `GeneralizedRankValidationSummary`.
"""
function check_gril_query(M::PModule, pi; centers, levels=(1,), lengths=(1,),
        budget::GeneralizedRankBudget=GeneralizedRankBudget(), throw::Bool=false)
    issues = String[]
    try
        _gr_request(M,pi,centers,levels,lengths,budget)
    catch err
        err isa ArgumentError || rethrow()
        push!(issues,err.msg)
    end
    result = GeneralizedRankValidationSummary((valid=isempty(issues),issues=issues,
                                               interpretation=:continuous_grid_worms))
    throw && !result.valid && Base.throw(ArgumentError(join(issues,"; ")))
    return result
end

# Closed lower / possibly open upper grid intervals, intersected with a CLOSED
# worm. Both endpoint flags matter: touching an excluded cell boundary does
# not give a fiber, nor an identity relation between two fibers.
struct _GRProjection
    lo::QQ
    hi::QQ
    lo_closed::Bool
    hi_closed::Bool
end
struct _GRCell
    label::Int
    i::Int
    j::Int
    x::_GRProjection
    y::_GRProjection
end
function _gr_projection(lo,hi,hi_closed,other_lo,other_hi,other_hi_closed,slo,shi)
    a = max(lo,slo-other_hi)
    b = min(hi,shi-other_lo)
    ac = a > slo-other_hi || other_hi_closed
    bc = b < hi || hi_closed
    return _GRProjection(a,b,ac,bc)
end
_gr_nonempty(p::_GRProjection) = p.lo < p.hi || (p.lo == p.hi && p.lo_closed && p.hi_closed)
_gr_ordered(a::_GRProjection,b::_GRProjection) =
    a.lo < b.hi || (a.lo == b.hi && a.lo_closed && b.hi_closed)

function _gr_worm_cells(ax, p, delta, ell, budget)
    nx, ny = length.(ax)
    # Iterate only cells meeting the bounding box, with no full grid allocation.
    xl, xh = p[1]-ell*delta, p[1]+ell*delta
    yl, yh = p[2]-ell*delta, p[2]+ell*delta
    slo, shi = sum(p)-2*delta, sum(p)+2*delta
    ir = max(1,searchsortedlast(ax[1],xl)):searchsortedlast(ax[1],xh)
    jr = max(1,searchsortedlast(ax[2],yl)):searchsortedlast(ax[2],yh)
    _gr_bound(big(length(ir))*length(jr), budget.max_order_checks, "worm cell candidates")
    cells = _GRCell[]
    for j in jr, i in ir
        a = max(ax[1][i],xl); b = i == nx ? xh : min(ax[1][i+1],xh)
        c = max(ax[2][j],yl); d = j == ny ? yh : min(ax[2][j+1],yh)
        bc = i == nx || xh < ax[1][i+1]
        dc = j == ny || yh < ax[2][j+1]
        xp = _gr_projection(a,b,bc,c,d,dc,slo,shi)
        yp = _gr_projection(c,d,dc,a,b,bc,slo,shi)
        _gr_nonempty(xp) && _gr_nonempty(yp) || continue
        _gr_bound(length(cells)+1, budget.max_vertices, "worm vertices")
        push!(cells,_GRCell(i+(j-1)*nx,i,j,xp,yp))
    end
    return cells
end

function _gr_worm_rank(M, ax, p, delta, ell, budget)
    # A zero stalk anywhere in the connected worm kills its comparison map.
    (p[1]-ell*delta < first(ax[1]) || p[2]-ell*delta < first(ax[2])) && return 0
    if iszero(delta)
        i = searchsortedlast(ax[1],p[1]); j = searchsortedlast(ax[2],p[2])
        return M.dims[i+(j-1)*length(ax[1])]
    end
    cells = _gr_worm_cells(ax,p,delta,ell,budget)
    labels = [c.label for c in cells]
    any(v -> iszero(M.dims[v]), labels) && return 0
    _gr_bound(big(length(cells))^2, budget.max_order_checks, "worm comparisons")
    edges = Tuple{Int,Int}[]
    for (a,A) in enumerate(cells), (b,B) in enumerate(cells)
        a == b && continue
        A.i <= B.i && A.j <= B.j || continue
        # Equal coordinate bins require an actual comparable pair in the worm.
        # Merely comparing finite grid labels gives spurious relations here.
        A.i == B.i && !_gr_ordered(A.x,B.x) && continue
        A.j == B.j && !_gr_ordered(A.y,B.y) && continue
        push!(edges,(a,b))
    end
    return _gr_diagram(M,labels,edges,budget)
end

"""
    worm_rank(M, grid; center, width, length=1, budget=GeneralizedRankBudget())

Exact generalized rank on a continuous closed GRIL worm in the right-continuous
grid extension. A length-`ell`, width-`d` worm is the set satisfying
`abs(x-px) <= ell*d`, `abs(y-py) <= ell*d`, and
`abs(x+y-px-py) <= 2*d`. Width zero queries the center stalk.
"""
function worm_rank(M::PModule, pi; center, width, length=1,
        budget::GeneralizedRankBudget=GeneralizedRankBudget())
    ax = _gr_grid(M,pi,budget)
    p = _gr_center(center)
    delta = _gr_coordinate(width)
    delta >= 0 || throw(ArgumentError("worm width must be nonnegative"))
    ell = only(_gr_positive_indices((length,),"worm length"))
    return _gr_worm_rank(M,ax,p,delta,ell,budget)
end

# Every fiber/projection endpoint is a min/max of these affine functions.
# Their pairwise crossings include every possible change of the contracted
# diagram, including changes of its relations with an unchanged vertex set.
function _gr_events(ax,p,ell,cap,budget)
    events = Set{QQ}((zero(QQ),cap))
    _gr_bound(length(events), budget.max_events, "critical widths")
    for coordinate in 1:2
        other = 3-coordinate
        lines = Tuple{QQ,BigInt}[(x,big(0)) for x in ax[coordinate]]
        for sign in (-1,1)
            push!(lines,(p[coordinate],big(sign)*ell))
            push!(lines,(p[coordinate],big(sign)*(big(ell)+2)))
            for x in ax[other]
                push!(lines,(sum(p)-x,big(sign)*2))
            end
        end
        unique!(lines)
        _gr_bound(big(length(lines))^2, budget.max_order_checks, "critical-width comparisons")
        for i in eachindex(lines), j in 1:i-1
            a,b = lines[i]; c,d = lines[j]
            b == d && continue
            width = (c-a)/(b-d)
            0 < width < cap || continue
            push!(events,width)
            _gr_bound(length(events), budget.max_events, "critical widths")
        end
    end
    return sort!(collect(events))
end

"""
    gril(M, grid; centers, levels=(1,), lengths=(1,), budget=GeneralizedRankBudget())

Compute exact generalized-rank invariant landscape values at specified probes:
`lambda(p,k,ell) = sup {d >= 0 : worm_rank(M,grid;p,d,ell) >= k}`.
The supremum of an empty set of admissible nonnegative widths is reported as
zero. Uses the continuous worms of Xin et al. (2023), not a sampled-radius
approximation. On each open interval between exact critical widths the finite
restriction diagram is constant; binary search reuses ranks across levels.

Only positively oriented 2D grid encodings are supported. The extension is
zero below the first grid coordinates and constant in the upper tails. This
explicit boundary convention gives a finite upper bound for each probe.
No interval enumeration, training, learned selection, or differentiation occurs.
"""
function gril(M::PModule, pi; centers, levels=(1,), lengths=(1,),
        budget::GeneralizedRankBudget=GeneralizedRankBudget())
    ax, points, ks, ls = _gr_request(M,pi,centers,levels,lengths,budget)
    values = zeros(QQ,length(points),length(ks),length(ls))
    queries = 0
    events_used = 0
    for (c,p) in enumerate(points), (l,ell) in enumerate(ls)
        cap = min(p[1]-first(ax[1]), p[2]-first(ax[2]))/ell
        cap <= 0 && continue
        breaks = _gr_events(ax,p,ell,cap,budget)
        events_used += length(breaks)
        _gr_bound(events_used, budget.max_events, "total critical widths")
        memo = Dict{Int,Int}()
        for (k,level) in enumerate(ks)
            lo = 0; hi = length(breaks)-1
            while lo < hi
                mid = lo + div(hi-lo+1,2)
                rank = get!(memo,mid) do
                    queries += 1
                    _gr_bound(queries, budget.max_queries, "rank queries")
                    delta = (breaks[mid]+breaks[mid+1])/2
                    _gr_worm_rank(M,ax,p,delta,ell,budget)
                end
                if rank >= level
                    lo = mid
                else
                    hi = mid-1
                end
            end
            values[c,k,l] = iszero(lo) ? zero(QQ) : breaks[lo+1]
        end
    end
    return GRILResult(points,ks,ls,values,queries)
end
