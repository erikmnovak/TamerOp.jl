@testset "Signed transport and convolution" begin
    function measure(points,weights)
        N=length(first(points))
        axes=ntuple(k->sort!(unique([p[k] for p in points])),N)
        inds=[ntuple(k->findfirst(==(p[k]),axes[k]),N) for p in points]
        return SM.PointSignedMeasure(axes,inds,collect(weights))
    end
    function empty_measure(N)
        SM.PointSignedMeasure(ntuple(_->Int[],N),NTuple{N,Int}[],Int[])
    end
    # Independent assignment oracle. Repeated unit masses occur only in tiny tests.
    function assignment_cost(a,b,metric)
        isempty(a) && return 0
        best=nothing
        for j in eachindex(b)
            ds=abs.(collect(a[1]).-collect(b[j]))
            cost=(metric===:l1 ? sum(ds) : maximum(ds))+assignment_cost(a[2:end],b[[i for i in eachindex(b) if i!=j]],metric)
            best=best===nothing ? cost : min(best,cost)
        end
        return best
    end
    @testset "Balanced exact costs and certificates" begin
        z=empty_measure(1)
        mu=measure([(0,),(2,),(5,)],[2,-3,1])
        result=TO.signed_transport(mu,z;witness=true)
        @test TO.transport_cost(result)==7
        @test TO.transport_summary(result).bounds==(7,7)
        @test SM.check_signed_transport(result).valid
        @test sum(SM.transport_plan(result).supply)==3
        @test SM.check_signed_transport(SM.transport_plan(result)).valid
        @test TO.transport_summary(result).boundary==:none
        @test TO.transport_cost(TO.signed_transport(z,mu))==7
        @test TO.transport_cost(TO.signed_transport(mu,mu))==0
        shared=measure([(0,),(2,),(5,),(9,)],[2,-3,1,7])
        @test TO.transport_cost(TO.signed_transport(shared,measure([(9,)],[7])))==7
        @test TO.transport_cost(TO.signed_transport(z,z;witness=true))==0
        @test SM.check_signed_transport(TO.signed_transport(z,z;witness=true)).valid
        p=SM.transport_plan(result);p.flow[1,1]+=1
        @test !SM.check_signed_transport(p).valid
        @test SM.check_signed_transport(result).valid
        @test_throws ArgumentError SM.check_signed_transport(p;throw=true)
        @test_throws ArgumentError SM.transport_plan(TO.signed_transport(mu,z))
        # Exact fractional masses and duplicate cancellation, including beyond Int.
        h=big(10)^40
        huge=measure([(0,),(0,),(1,)],[h+2,-h,-2])
        @test TO.transport_cost(TO.signed_transport(huge,z;witness=true))==2
        frac=measure([(0//1,),(1//3,)],[2//7,-2//7])
        @test TO.transport_cost(TO.signed_transport(frac,z))==2//21
        near=measure([(h//1,),(h//1+1//1000,)],[1,-1])
        @test TO.transport_cost(TO.signed_transport(near,z))==1//1000
        @test TO.transport_cost(TO.signed_transport(measure([(0.0,),(0.25,)],[1.0,-1.0]),z))==1//4
        # Unit multiplicities are never expanded: one edge for arbitrarily large mass.
        large=TO.signed_transport(measure([(0,),(1,)],[h,-h]),z;witness=true)
        @test size(SM.transport_plan(large).flow)==(1,1)
        @test large.augmentations==1 && TO.transport_cost(large)==h
        for metric in (:l1,:linf)
            rng=MersenneTwister(8110)
            for n in 1:5,trial in 1:8
                a=[(rand(rng,-3:3),rand(rng,-3:3)) for _ in 1:n]
                b=[(rand(rng,-3:3),rand(rng,-3:3)) for _ in 1:n]
                mu=measure(a,ones(Int,n));nu=measure(b,ones(Int,n))
                r=TO.signed_transport(mu,nu;metric,witness=true)
                @test TO.transport_cost(r)==assignment_cost(a,b,metric)
                @test SM.check_signed_transport(r).valid
                @test TO.transport_cost(TO.signed_transport(nu,mu;metric))==TO.transport_cost(r)
                c=measure([(0,0)], [n])
                @test TO.transport_cost(r)<=TO.transport_cost(TO.signed_transport(mu,c;metric))+TO.transport_cost(TO.signed_transport(c,nu;metric))
            end
        end
        a=measure([(0,0)],[1]);b=measure([(3,4)],[1])
        @test TO.transport_cost(TO.signed_transport(a,b;metric=:l1))==7
        @test TO.transport_cost(TO.signed_transport(a,b;metric=:linf))==4
        @test TO.transport_cost(TO.signed_transport(a,b;metric=:euclidean))==5
        @test TO.transport_cost(TO.signed_transport(a,b;scale=[2,3]))==18
        r=TO.signed_transport(a,measure([(1,1)],[1]);metric=:euclidean,witness=true)
        @test r.lower_bound<=sqrt(big(2))<=r.upper_bound
        @test TO.transport_summary(r).status==:bounded_euclidean
        @test SM.check_signed_transport(r).valid
        # Independent Euclidean assignment with irrational costs.
        aa=[(0,0),(2,0),(0,3)];bb=[(1,1),(3,2),(2,4)]
        eu=TO.signed_transport(measure(aa,[1,1,1]),measure(bb,[1,1,1]);metric=:euclidean,witness=true)
        costs=BigFloat[]
        for i in 1:3,j in 1:3,k in 1:3
            length(Set((i,j,k)))==3 || continue
            push!(costs,sum(sqrt(sum(big(aa[t][d]-bb[u][d])^2 for d in 1:2)) for (t,u) in enumerate((i,j,k))))
        end
        @test eu.lower_bound<=minimum(costs)<=eu.upper_bound
        @test SM.check_signed_transport(eu).valid
        @test occursin("SignedTransport",sprint(show,r))
        @test_throws ArgumentError TO.signed_transport(a,empty_measure(2))
        @test_throws ArgumentError TO.signed_transport(a,empty_measure(1))
        @test_throws ArgumentError TO.signed_transport(a,b;metric=:l2)
        @test_throws ArgumentError TO.signed_transport(a,b;scale=[0,1])
        @test_throws ArgumentError TO.signed_transport(a,b;scale=[1])
        @test_throws ArgumentError TO.signed_transport(measure([(Inf,)],[1]),measure([(0,)],[1]))
        @test_throws ArgumentError TO.signed_transport(measure([(0,)],[NaN]),z)
        @test_throws ArgumentError TO.signed_transport(a,b;budget=SM.SignedMeasureBudget(max_atoms=0))
        @test_throws ArgumentError TO.signed_transport(a,b;budget=SM.SignedMeasureBudget(max_pairs=0))
        @test_throws ArgumentError TO.signed_transport(a,b;budget=SM.SignedMeasureBudget(max_augmentations=0))
        @test_throws ArgumentError SM.SignedMeasureBudget(max_atoms=-1)
        @test_throws ArgumentError SM.SignedMeasureBudget(max_pairs=true)
        @test_throws ArgumentError TO.signed_transport(SM.PointSignedMeasure(([0],),[(2,)],[1]),z)
    end
    @testset "Declared projected quadrature" begin
        a=measure([(0,0),(2,1)],[2,-1]);b=measure([(1,0)],[1])
        dirs=[1 0;0 1]
        r=TO.sliced_signed_transport(a,b;directions=dirs)
        @test r.costs==[3,1]
        @test TO.transport_cost(r)==2
        @test SM.check_signed_transport(r).valid
        @test TO.transport_summary(r).status==:sampled_projections
        @test TO.transport_summary(r).sampling_error===nothing
        r2=TO.sliced_signed_transport(a,b;directions=dirs,weights=[1,3])
        @test TO.transport_cost(r2)==3//2
        @test TO.transport_cost(TO.sliced_signed_transport(b,a;directions=dirs))==2
        @test TO.transport_cost(TO.sliced_signed_transport(a,a;directions=dirs))==0
        @test TO.transport_cost(TO.sliced_signed_transport(a,b;directions=2dirs))==2
        @test TO.transport_cost(TO.sliced_signed_transport(a,b;directions=dirs,scale=[2,3]))==9//2
        invisible=TO.sliced_signed_transport(measure([(0,0)],[1]),measure([(0,3)],[1]);directions=[1 0])
        @test TO.transport_cost(invisible)==0
        @test_throws ArgumentError TO.sliced_signed_transport(a,b;directions=zeros(1,2))
        @test_throws ArgumentError TO.sliced_signed_transport(a,b;directions=ones(2,3))
        @test_throws ArgumentError TO.sliced_signed_transport(a,b;directions=dirs,weights=[1,-1])
        @test_throws ArgumentError TO.sliced_signed_transport(a,b;directions=dirs,budget=SM.SignedMeasureBudget(max_queries=1))
        @test_throws ArgumentError TO.sliced_signed_transport(a,b;directions=dirs,budget=SM.SignedMeasureBudget(max_evaluations=0))
    end
    @testset "Convolution values, cancellation and feature integration" begin
        a=measure([(0,0),(1,0),(0,0)],[3,-1,-1])
        axes=([0,1,2],[0,2]);r=TO.signed_convolution(a;axes,sigma=2)
        @test size(TO.convolution_values(r))==(3,2)
        for i in 1:3,j in 1:2
            x=axes[1][i];y=axes[2][j]
            expected=2exp(-(x*x+y*y)/8)-exp(-((x-1)^2+y*y)/8)
            @test isapprox(r.values[i,j], expected; atol=1e-12, rtol=1e-12)
            terms=SM.convolution_terms(r;index=(i,j))
            @test isapprox(sum(terms.contributions), expected; atol=1e-12, rtol=1e-12)
            @test terms.weights==[2,-1]
        end
        @test SM.check_signed_convolution(r).valid
        @test TO.convolution_summary(r).normalization==:peak
        copyvals=TO.convolution_values(r);copyvals[1]=999
        @test r.values[1]!=999
        @test all(iszero,TO.convolution_values(TO.signed_convolution(empty_measure(2);axes)))
        # Query windows do not discard off-grid atoms.
        offgrid=TO.signed_convolution(measure([(2,)],[1]);axes=([0],))
        @test isapprox(offgrid.values[1], exp(-2); atol=1e-12, rtol=1e-12)
        h=big(10)^40
        exact=TO.signed_convolution(measure([(h//1,)],[1]);axes=([h//1+1//2],),sigma=1//2)
        @test isapprox(exact.values[1], exp(-0.5); atol=1e-12, rtol=1e-12)
        lap=TO.signed_convolution(measure([(0,)],[2]);axes=([1],),kernel=:laplacian,scale=[2])
        @test isapprox(lap.values[1], 2exp(-2); atol=1e-12, rtol=1e-12)
        @test_throws ArgumentError TO.signed_convolution(a;axes,sigma=0)
        @test_throws ArgumentError TO.signed_convolution(a;axes,kernel=:bad)
        @test_throws ArgumentError TO.signed_convolution(a;axes=([0,0],[1]))
        @test_throws ArgumentError TO.signed_convolution(a;axes=([Inf],[1]))
        @test_throws ArgumentError TO.signed_convolution(a;axes,budget=SM.SignedMeasureBudget(max_queries=1))
        @test_throws ArgumentError TO.signed_convolution(a;axes,budget=SM.SignedMeasureBudget(max_evaluations=1))
        @test_throws ArgumentError SM.convolution_terms(r;index=(4,1))
        spec=TO.SignedConvolutionSpec(;axes,sigma=2)
        @test TamerOp.Featurizers.check_featurizer_spec(spec).valid
        @test TO.nfeatures(spec)==6
        @test length(unique(TO.feature_names(spec)))==6
        @test isapprox(TO.transform(spec,a), vec(r.values); atol=1e-12, rtol=1e-12)
        cached=TO.build_cache(a,spec)
        @test isapprox(TO.transform(spec,cached), vec(r.values); atol=1e-12, rtol=1e-12)
        @test_throws ArgumentError TO.transform(spec,empty_measure(1))
        fs=TO.featurize([a,a],spec;batch=TO.BatchOptions(threaded=false))
        @test size(fs)==(2,6)
        @test isapprox(TamerOp.Featurizers.feature_matrix(fs)[1,:], vec(r.values); atol=1e-12, rtol=1e-12)
        md=TamerOp.Featurizers.feature_metadata(fs;format=:wide)
        serialized=TamerOp.Featurizers.JSON3.read(TamerOp.Featurizers.JSON3.write(md))
        restored=TamerOp.Featurizers.spec_from_metadata(serialized["spec"])
        @test restored isa TO.SignedConvolutionSpec
        @test isapprox(TO.transform(restored,a), vec(r.values); atol=1e-12, rtol=1e-12)
        @test TO.feature_names(restored)==TO.feature_names(spec)
        @test TamerOp.Featurizers.feature_axes(restored)==TamerOp.Featurizers.feature_axes(spec)
        opposite=measure([(0,0),(1,0)],[-2,1])
        @test isapprox(TO.transform(spec,opposite),-vec(r.values);atol=1e-12)
        @test_throws ArgumentError TO.signed_convolution(measure([(0,)],[big(10)^400]);axes=([0],))
        composite=TO.CompositeSpec((spec,spec))
        @test TO.transform(composite,a)==vcat(vec(r.values),vec(r.values))
        fs_threaded=TO.featurize([a,a],spec;batch=TO.BatchOptions(threaded=true))
        @test TamerOp.Featurizers.feature_matrix(fs_threaded)==TamerOp.Featurizers.feature_matrix(fs)
        @test_throws ArgumentError TO.SignedConvolutionSpec(axes=([0,0],))
        @test_throws ArgumentError TO.SignedConvolutionSpec(axes=([0],),sigma=Inf)
    end
end
