# An explicit geometric adapter for retained one-dimensional samples.
"""
    AmbientLandscapeGrid

Values of a selected parallel slice family on a declared Cartesian grid.
`landscape_values` has one axis per ambient coordinate, followed by the retained
landscape layer. Missing lines and out-of-window queries are `NaN`, never zero.
`describe` records the line parameter, source windows and interpolation contract.
Construct with `ambient_landscape`.
"""
struct AmbientLandscapeGrid{N}
    axes::NTuple{N,Vector{Float64}}
    values::Array{Float64}
    coverage::BitArray{N}
    errors::Array{Float64,N}
    metadata::NamedTuple
end
landscape_values(g::AmbientLandscapeGrid) = g.values
landscape_grid(g::AmbientLandscapeGrid) = g.axes
landscape_layers(g::AmbientLandscapeGrid) = size(g.values,ndims(g.values))
"""Return the ambient grid's availability mask; missing samples are not zeros."""
ambient_coverage(g::AmbientLandscapeGrid) = g.coverage
"""Return pointwise conditional interpolation bounds, or NaN where unknown."""
ambient_errors(g::AmbientLandscapeGrid) = g.errors
"""Return the structured domain, coverage and parameterization summary."""
ambient_landscape_summary(g::AmbientLandscapeGrid) = describe(g)
"""
    check_ambient_landscape(g; throw=false)

Check a hand-built grid's coordinate axes, shapes, coverage and error entries.
This validates the stored result, not the caller's source-window or error claims.
"""
function check_ambient_landscape(g::AmbientLandscapeGrid;throw=false)
    issues=String[];n=length(g.axes);shape=Tuple(length.(g.axes))
    n>=2&&all(a->!isempty(a)&&all(isfinite,a)&&all(>(0),diff(a)),g.axes)||push!(issues,"Invalid ambient coordinate axes.")
    ndims(g.values)==n+1&&size(g.values)[1:n]==shape&&size(g.values,n+1)>0||push!(issues,"Values do not match axes and layers.")
    size(g.coverage)==shape&&size(g.errors)==shape||push!(issues,"Coverage/error shapes do not match axes.")
    all(x->isnan(x)||(isfinite(x)&&x>=0),g.values)||push!(issues,"Invalid landscape height.")
    all(x->isnan(x)||(isfinite(x)&&x>=0),g.errors)||push!(issues,"Invalid error bound.")
    if isempty(issues)
        for I in CartesianIndices(g.coverage)
            vals=view(g.values,Tuple(I)...,:)
            (g.coverage[I] ? all(isfinite,vals) : all(isnan,vals)&&isnan(g.errors[I]))||push!(issues,"Coverage disagrees with values/errors at $(Tuple(I)).")
        end
    end
    valid=isempty(issues)
    throw&&!valid&&Base.throw(ArgumentError(join(issues," ")))
    (;kind=:ambient_landscape_validation,valid,issues)
end
describe(g::AmbientLandscapeGrid) = (;kind=:ambient_landscape_grid,shape=size(g.coverage),
    layers=landscape_layers(g),covered=count(g.coverage),total=length(g.coverage),g.metadata...)
Base.show(io::IO,g::AmbientLandscapeGrid) = print(io,"AmbientLandscapeGrid(",size(g.coverage),
    ", layers=",landscape_layers(g),", covered=",count(g.coverage),"/",length(g.coverage),")")

"""
    ambient_landscape(L; axes, basepoints, source_window, direction_index=1,
        sample_error=nothing, line_tolerance=1e-10, max_points=250_000)

Place a retained `MPLandscape` family on ambient coordinates. `basepoints[j]`
must be the actual point at parameter `t=0` of stored offset `j`; direction
`d=L.directions[direction_index]` must be a positive vector. Thus the stored
parameterization is `x=basepoints[j]+t*d`. No offset convention is guessed.

Linearly interpolate in `t` within the stored grid and the declared closed
`source_window=(lo,hi)` (or one window per offset). Do not interpolate between
lines or extrapolate. Coverage uses the stated absolute geometric tolerance.
Multiply heights by `maximum(d)` to express distance along the line in ambient
infinity-norm units. Equal direction components give the diagonal landscape;
other directions give a directional landscape, not the diagonal invariant.
Slice weights are not used: they describe aggregation, not pointwise heights.

`sample_error` optionally declares a uniform error bound on the stored heights
against the intended one-dimensional landscape, in original `t` units. The
reported pointwise interpolation bound is `maximum(d)*(sample_error+2a*b/(a+b))`,
where `a,b` are distances to neighboring samples. Without that input, numerical
error against the underlying module is unknown. This does not infer a bound
for an earlier discretization, window censoring, or a confidence interval.
"""
function ambient_landscape(L::MPLandscape; axes, basepoints, source_window,
        direction_index=1, sample_error=nothing, line_tolerance=1e-10, max_points=250_000)
    check_mp_landscape(L;throw=true)
    size(L.values,4)==length(landscape_grid(L)) && all(>=(0),L.values) ||
        throw(ArgumentError("Landscape values must be nonnegative and match the stored parameter grid."))
    direction_index isa Integer && !(direction_index isa Bool) && 1<=direction_index<=ndirections(L) ||
        throw(ArgumentError("direction_index must select a stored direction."))
    d=try Float64.(collect(slice_directions(L)[direction_index])) catch
        throw(ArgumentError("Ambient queries require a stored vector direction.")) end
    n=length(d)
    n>=2 && all(x->isfinite(x)&&x>0,d) || throw(ArgumentError("The direction must have at least two positive finite coordinates."))
    ax=Tuple(Float64.(collect(a)) for a in axes)
    length(ax)==n && all(a->!isempty(a)&&all(isfinite,a)&&issorted(a)&&all(>(0),diff(a)),ax) ||
        throw(ArgumentError("axes must give one finite strictly increasing nonempty grid per coordinate."))
    max_points isa Integer && !(max_points isa Bool) && max_points>0 || throw(ArgumentError("max_points must be positive."))
    prod(BigInt(length(a)) for a in ax)*landscape_layers(L)<=max_points || throw(ArgumentError("Ambient grid exceeds max_points, counting layers."))
    line_tolerance isa Real && !(line_tolerance isa Bool) && isfinite(line_tolerance) && line_tolerance>=0 ||
        throw(ArgumentError("line_tolerance must be finite and nonnegative."))
    sample_error===nothing || (sample_error isa Real && !(sample_error isa Bool) && isfinite(sample_error) && sample_error>=0) ||
        throw(ArgumentError("sample_error must be nothing or a finite nonnegative bound."))
    bases=[Float64.(collect(b)) for b in basepoints]
    length(bases)==noffsets(L) && all(b->length(b)==n&&all(isfinite,b),bases) ||
        throw(ArgumentError("Supply one finite basepoint of the correct dimension per stored offset."))
    windows=source_window isa Tuple && length(source_window)==2 && all(x->x isa Real,source_window) ?
        fill(source_window,length(bases)) : collect(source_window)
    length(windows)==length(bases) && all(w->length(w)==2&&all(isfinite,w)&&w[1]<w[2],windows) ||
        throw(ArgumentError("source_window must be a finite increasing pair, or one pair per offset."))
    pivot=argmax(d);factor=d[pivot];tg=landscape_grid(L)
    # Coincident lines would give ambiguous occurrence identities and heights.
    for j in eachindex(bases),i in 1:j-1
        t=(bases[j][pivot]-bases[i][pivot])/d[pivot]
        maximum(abs.(bases[j].-bases[i].-t.*d))>2line_tolerance ||
            throw(ArgumentError("Basepoints $i and $j describe coincident or tolerance-overlapping lines."))
    end
    shape=Tuple(length.(ax));values=fill(NaN,(shape...,landscape_layers(L)))
    coverage=falses(shape);errors=fill(NaN,shape)
    for I in CartesianIndices(coverage)
        x=[ax[q][I[q]] for q in 1:n]
        for j in eachindex(bases)
            t=(x[pivot]-bases[j][pivot])/d[pivot]
            maximum(abs.(x.-bases[j].-t.*d))<=line_tolerance || continue
            max(first(tg),windows[j][1])<=t<=min(last(tg),windows[j][2]) || continue
            right=searchsortedfirst(tg,t)
            left=right<=length(tg)&&tg[right]==t ? right : right-1
            1<=left<=right<=length(tg) || continue
            windows[j][1]<=tg[left] && tg[right]<=windows[j][2] || continue
            a=t-tg[left];b=tg[right]-t;w=left==right ? 0.0 : a/(a+b)
            for k in 1:landscape_layers(L)
                values[Tuple(I)...,k]=factor*((1-w)*L.values[direction_index,j,k,left]+w*L.values[direction_index,j,k,right])
            end
            coverage[I]=true
            sample_error===nothing || (errors[I]=factor*(sample_error+(left==right ? 0.0 : 2a*b/(a+b))))
            break
        end
    end
    all(v->isnan(v)||isfinite(v),values) || throw(ArgumentError("Ambient values overflowed display precision."))
    metadata=(;direction=Tuple(d),direction_index,basepoints=Tuple(Tuple.(bases)),source_windows=Tuple(Tuple.(windows)),
        t_window=(first(tg),last(tg)),normalization=:ambient_infinity_norm,height_factor=factor,
        interpretation=all(==(first(d)),d) ? :diagonal_landscape : :directional_landscape,
        interpolation=:linear_in_t,no_transverse_interpolation=true,line_tolerance=Float64(line_tolerance),
        sample_error,error_bound=sample_error===nothing ? :unknown : :conditional_on_sample_bound,
        error_domain=:projected_line_point)
    AmbientLandscapeGrid(ax,values,coverage,errors,metadata)
end
