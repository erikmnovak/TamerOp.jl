# Signed convolution on an explicit Cartesian query grid, with atom attribution.
"""Sampled signed convolution, retaining its grid, kernel and canonical input atoms."""
struct SignedConvolution{N}
    axes::NTuple{N,Vector{_TransportQ}}
    values::Array{Float64,N}
    points::Vector{NTuple{N,_TransportQ}}
    weights::Vector{_TransportQ}
    sigma::_TransportQ
    scale::Vector{_TransportQ}
    kernel::Symbol
end
"""Return copies of the query axes or the sampled values of a signed convolution."""
convolution_axes(r::SignedConvolution)=map(copy,r.axes)
convolution_values(r::SignedConvolution)=copy(r.values)
function _convolution_axes(axes,N,budget)
    axes isa Tuple && length(axes)==N || throw(ArgumentError("query axes must be a tuple of ambient dimension"))
    _transport_bound(prod(big(length(a)) for a in axes),budget.max_queries,"query grid")
    ax=map(a->_transport_rational.(collect(a)),axes)
    all(a->!isempty(a) && _axes_sorted_unique(a),ax) || throw(ArgumentError("query axes must be nonempty, increasing and unique"))
    return ax
end
function _convolution_term(x,y,w,sigma,scale,kernel)
    # Subtract before conversion to preserve small offsets at large exact grades.
    r=0.0
    for k in eachindex(x)
        r=hypot(r,Float64((x[k]-y[k])*scale[k]/sigma))
    end
    kernel_value=kernel===:gaussian ? exp(-0.5*r*r) : exp(-r)
    term=Float64(w)*kernel_value
    isfinite(term) || throw(ArgumentError("convolution contribution exceeds Float64 range; rescale the weights"))
    return kernel_value,term
end
"""
    signed_convolution(mu; axes, sigma=1, scale=nothing, kernel=:gaussian, budget=SignedMeasureBudget())

Evaluate `sum(w * K(x - atom))` on the supplied Cartesian grid. Gaussian kernels
use `exp(-r^2/2)` and Laplacian kernels use `exp(-r)`, where
`r = norm(scale .* (x - atom)) / sigma`. Both have peak one; no density
normalization, atom clipping or pixel integration is applied. Values use Float64
arithmetic; axes, coordinates and weights retain their exact stored values.
The query grid does not truncate the measure. `convolution_terms` explains one
sample through the contributing canonical signed atoms.
"""
function signed_convolution(mu::PointSignedMeasure{N};axes,sigma=1,scale=nothing,
        kernel::Symbol=:gaussian,budget::SignedMeasureBudget=SignedMeasureBudget()) where N
    kernel in (:gaussian,:laplacian) || throw(ArgumentError("kernel must be :gaussian or :laplacian"))
    width=_transport_rational(sigma);width>0 || throw(ArgumentError("sigma must be positive"))
    s=_transport_scale(scale,N);ax=_convolution_axes(axes,N,budget)
    atoms=_transport_atoms(mu,budget);points=sort!(collect(keys(atoms)));weights=[atoms[x] for x in points]
    _transport_bound(prod(big(length(a)) for a in ax)*length(points)*N,budget.max_evaluations,"kernel evaluations")
    vals=zeros(Float64,map(length,ax))
    for I in CartesianIndices(vals)
        x=ntuple(k->ax[k][I[k]],N);total=0.0;correction=0.0
        for (y,w) in zip(points,weights)
            _,term=_convolution_term(x,y,w,width,s,kernel)
            z=term-correction;t=total+z;correction=(t-total)-z;total=t
        end
        isfinite(total) || throw(ArgumentError("convolution value exceeds Float64 range"))
        vals[I]=total
    end
    return SignedConvolution(ax,vals,points,weights,width,s,kernel)
end
"""Return canonical atoms, signed weights and their kernel contributions at one grid index."""
function convolution_terms(r::SignedConvolution{N};index) where N
    length(index)==N || throw(ArgumentError("index must have ambient dimension"))
    all(k->index[k] isa Integer && 1<=index[k]<=length(r.axes[k]),1:N) || throw(ArgumentError("query index is out of bounds"))
    x=ntuple(k->r.axes[k][index[k]],N)
    terms=[_convolution_term(x,y,w,r.sigma,r.scale,r.kernel) for (y,w) in zip(r.points,r.weights)]
    return (point=x,atoms=copy(r.points),weights=copy(r.weights),
        kernel_values=first.(terms),contributions=last.(terms))
end
"""Describe a sampled convolution, including its finite query grid and peak normalization."""
convolution_summary(r::SignedConvolution)=describe(r)
describe(r::SignedConvolution)=(kind=:signed_convolution,kernel=r.kernel,sigma=r.sigma,
    scale=copy(r.scale),shape=size(r.values),atoms=length(r.weights),arithmetic=:Float64,
    normalization=:peak,window=:full_finite_support,grid_role=:point_samples)
Base.show(io::IO,r::SignedConvolution)=print(io,"SignedConvolution(kernel=",r.kernel,", shape=",size(r.values),")")
"""Check sampled-convolution shape, source data and finite numerical values."""
function check_signed_convolution(r::SignedConvolution{N};throw::Bool=false) where N
    errors=String[]
    size(r.values)==map(length,r.axes) || push!(errors,"grid and value shapes disagree")
    all(a->!isempty(a) && _axes_sorted_unique(a),r.axes) || push!(errors,"invalid query axes")
    length(r.points)==length(r.weights) || push!(errors,"atom and weight counts disagree")
    length(r.scale)==N && all(>(0),r.scale) && r.sigma>0 || push!(errors,"invalid scale or bandwidth")
    r.kernel in (:gaussian,:laplacian) || push!(errors,"invalid kernel")
    all(isfinite,r.weights) && all(p->all(isfinite,p),r.points) && all(a->all(isfinite,a),r.axes) || push!(errors,"nonfinite source or grid")
    all(isfinite,r.values) || push!(errors,"nonfinite convolution value")
    throw && !isempty(errors) && Base.throw(ArgumentError(join(errors,"; ")))
    return SignedTransportValidationSummary(isempty(errors),errors)
end
