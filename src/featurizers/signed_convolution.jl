# Fixed Cartesian signed-convolution features and their portable specification.
"""
    SignedConvolutionSpec(; axes, sigma=1, scale=nothing, kernel=:gaussian, ...)

Evaluate signed point measures on a fixed Float64 Cartesian grid using the
peak-one kernel of `signed_convolution`. Features follow Julia column-major
grid order. The grid and bandwidth are shared across samples; there is no
sample-dependent normalization or fitted support window.
"""
struct SignedConvolutionSpec{N} <: AbstractFeaturizerSpec
    axes::NTuple{N,Vector{Float64}}
    sigma::Float64
    scale::Vector{Float64}
    kernel::Symbol
    max_atoms::Int
    max_queries::Int
    max_evaluations::Int
end
function SignedConvolutionSpec(;axes::Tuple,sigma=1,scale=nothing,kernel::Symbol=:gaussian,
        max_atoms=4096,max_queries=1_000_000,max_evaluations=10_000_000)
    N=length(axes);N>0 || throw(ArgumentError("convolution requires positive ambient dimension"))
    budget=SignedMeasures.SignedMeasureBudget(;max_atoms,max_queries,max_evaluations)
    ax=map(a->Float64.(collect(a)),axes)
    s=scale===nothing ? ones(N) : Float64.(collect(scale))
    spec=SignedConvolutionSpec{N}(ax,Float64(sigma),s,kernel,budget.max_atoms,budget.max_queries,budget.max_evaluations)
    check_featurizer_spec(spec;throw=true)
    return spec
end
function _check_spec_issues(spec::SignedConvolutionSpec{N}) where N
    issues=String[]
    _check_axes_payload!(issues,spec.axes)
    N>0 || push!(issues,"axes must have positive dimension")
    isfinite(spec.sigma) && spec.sigma>0 || push!(issues,"sigma must be positive and finite")
    length(spec.scale)==N && all(x->isfinite(x)&&x>0,spec.scale) || push!(issues,"scale must be positive, finite and match dimension")
    spec.kernel in (:gaussian,:laplacian) || push!(issues,"unsupported convolution kernel")
    all(>=(0),(spec.max_atoms,spec.max_queries,spec.max_evaluations)) || push!(issues,"negative convolution budget")
    prod(big(length(a)) for a in spec.axes)<=spec.max_queries || push!(issues,"convolution query budget exceeded")
    return issues
end
supports(spec::SignedConvolutionSpec{N},obj) where N = obj isa PointSignedMeasure{N}
nfeatures(spec::SignedConvolutionSpec)=prod(length,spec.axes)
feature_axes(spec::SignedConvolutionSpec)=(axes=map(copy,spec.axes),order=:column_major,kernel=spec.kernel,sigma=spec.sigma,scale=copy(spec.scale),normalization=:peak)
feature_names(spec::SignedConvolutionSpec)=[Symbol("signed_convolution__",join(Tuple(I),"_")) for I in CartesianIndices(map(length,spec.axes))][:]
function transform(spec::SignedConvolutionSpec,cache::PointMeasureInvariantCache;
        opts::InvariantOptions=InvariantOptions(),threaded::Bool=true)
    check_featurizer_spec(spec;throw=true)
    supports(spec,cache.pm) || throw(ArgumentError("convolution sample dimension disagrees with grid"))
    budget=SignedMeasures.SignedMeasureBudget(max_atoms=spec.max_atoms,max_queries=spec.max_queries,max_evaluations=spec.max_evaluations)
    out=SignedMeasures.signed_convolution(cache.pm;axes=spec.axes,sigma=spec.sigma,
        scale=spec.scale,kernel=spec.kernel,budget=budget)
    return vec(SignedMeasures.convolution_values(out))
end
function transform(spec::SignedConvolutionSpec,obj;opts::InvariantOptions=InvariantOptions(),threaded::Bool=true)
    return transform(spec,build_cache(obj,spec;opts,threaded);opts,threaded)
end
