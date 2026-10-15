"""Overlapping delay-coordinate clouds with shared sample identities.
`window` counts embedded points, `delay` counts samples, and `stride` advances
between windows. No interpolation, denoising or periodic continuation is applied.
"""
struct TemporalClouds
    clouds::Vector{DataTypes.PointCloud{Float64}}
    point_ids::Vector{Vector{Int}}
    times::Vector{Rational{BigInt}}
    provenance::NamedTuple
end
function delay_clouds(signal;dimension=2,delay=1,window=16,stride=1,times=1:length(signal),budget::TemporalBudget=TemporalBudget())
    for (value,name) in ((dimension,"dimension"),(delay,"delay"),(window,"window"),(stride,"stride"))
        _index(value,typemax(Int),name)
    end
    length(signal)==length(times) || throw(ArgumentError("Supply one time label per sample."))
    ts=_axis(times,"sample times";strict=false)
    all(x->x isa Real && !(x isa Bool) && isfinite(x) && isfinite(Float64(x)),signal) || throw(ArgumentError("Signal values must be finite real numbers representable for geometry."))
    count=big(length(signal))-big(dimension-1)*delay
    count>=window || throw(ArgumentError("The signal is too short for the requested embedding and window."))
    _bound(window,budget.max_points,"points per window")
    _bound(dimension*big(window),budget.max_matrix_entries,"embedding coordinates")
    starts=1:stride:Int(count)-window+1
    _bound(2big(length(starts))-1,budget.max_occurrences,"occurrences")
    clouds=DataTypes.PointCloud{Float64}[];ids=Vector{Int}[];labels=Rational{BigInt}[]
    for a in starts
        indices=collect(a:a+window-1)
        coords=[Float64(signal[i+(k-1)*delay]) for i in indices,k in 1:dimension]
        push!(clouds,DataTypes.PointCloud(coords));push!(ids,indices)
        push!(labels,ts[a+(dimension-1)*delay])
    end
    TemporalClouds(clouds,ids,labels,(;construction=:delay_windows,dimension,delay,window,stride,sample_times=ts,signal=Float64.(signal)))
end
describe(c::TemporalClouds)=(;kind=:temporal_clouds,windows=length(c.clouds),points=length.(c.point_ids),provenance=c.provenance)
Base.show(io::IO,c::TemporalClouds)=print(io,"TemporalClouds(",length(c.clouds)," windows)")

function _state_model(x,scale,d,F)
    G=x.complex;counts=DataTypes.cell_counts(G);offsets=cumsum([0;counts]);K=coeff_type(F)
    active=[[i for i in 1:c if any(g->g[1]<=scale,DataTypes.cell_grade_set(G,offsets[k]+i))] for (k,c) in enumerate(counts)]
    ids(k)=0<=k<length(active) ? active[k+1] : Int[]
    boundary(k)=1<=k<length(counts) ? K.(G.boundaries[k][ids(k-1),ids(k)]) : zeros(K,length(ids(k-1)),length(ids(k)))
    H=ChainComplexes.homology_data(boundary(d+1),boundary(d),d;field=F)
    (;active,H,ids=ids(d))
end
function _inclusion(a,b,K)
    index=Dict(id=>i for (i,id) in enumerate(b))
    all(id->haskey(index,id),a) || throw(ArgumentError("A declared inclusion omits source cells."))
    sparse([index[id] for id in a],collect(eachindex(a)),fill(one(K),length(a)),length(b),length(a))
end
function _time_order(directions,n)
    order=BitMatrix([i==j for i in 1:n,j in 1:n])
    for (e,d) in enumerate(directions)
        d in (:forward,:backward) || throw(ArgumentError("Time directions must be :forward or :backward."))
        a,b=d===:forward ? (e,e+1) : (e+1,e);order[a,b]=true
    end
    for k in 1:n,j in 1:n,i in 1:n;order[i,j]|=order[i,k] && order[k,j];end
    order
end

"""
    temporal_persistence(sources; scales, directions, chain_maps, times=1:length(sources),
                         degree=0, field=F2(), representatives=false, budget=TemporalBudget())

Construct a finite time-by-scale module from scalar source filtrations and
actual directed chain maps. Each `chain_maps[e]` contains one full coefficient
matrix per chain degree, in source-cell order. Every boundary equation and each
sampled sublevel restriction is checked. Vertical maps are source inclusions;
all squares therefore commute. Sources must retain their original cells.

The separate point-cloud method constructs the union bridges and their maps.
This workflow concerns extended zigzags, not extended relative persistence.
"""
Base.@constprop :none Base.@noinline Base.@nospecializeinfer function temporal_persistence(
        sources::AbstractVector{<:DataIngestion.FilteredComplex};scales,directions,chain_maps,
        times=1:length(sources),degree=0,field=F2(),representatives=false,budget::TemporalBudget=TemporalBudget(),
        provenance=(;construction=:supplied_filtered_chain_maps))
    F=_field(field);K=coeff_type(F);n=length(sources);ts=_axis(times,"times";strict=false);ss=_axis(scales,"scales");m=length(ss)
    n>0 && length(ts)==n || throw(ArgumentError("Supply a nonempty source sequence with one time per occurrence."))
    degree isa Integer && !(degree isa Bool) && 0<=degree<budget.max_cells || throw(ArgumentError("degree must be a bounded nonnegative integer."))
    representatives isa Bool || throw(ArgumentError("representatives must be Bool."))
    _bound(n,budget.max_occurrences,"occurrences");_bound(m,budget.max_scales,"scales")
    _bound((big(n)*m)^2,budget.max_order_entries,"order entries")
    length(directions)==length(chain_maps)==n-1 || throw(ArgumentError("Supply one direction and chain map per adjacent occurrence pair."))
    order=_time_order(directions,n)
    counts=[DataTypes.cell_counts(s.complex) for s in sources]
    _bound(sum(c->sum(big,c),counts),budget.max_cells,"source cells")
    owned=DataIngestion.FilteredComplex[deepcopy(s) for s in sources]
    slots=maximum(length,counts)
    estimate=sum(c->sum(x->big(x)^2,c),counts)*m*8
    _bound(estimate,budget.max_matrix_entries,"homology entries")
    for s in owned
        DataIngestion.check_filtered_complex(s;throw=true)
        DataTypes.parameter_dim(s.complex)==1 && s.orientation==(1,) || throw(ArgumentError("Temporal sources require a scalar sublevel filtration."))
        s.provenance.model===:original || throw(ArgumentError("Supply original temporal cells or explicit unreduced models with their own chain maps."))
    end
    fullmaps=Vector{Matrix{K}}[]
    dim(i,d)=d<length(counts[i]) ? counts[i][d+1] : 0
    boundary(i,d)=1<=d<length(counts[i]) ? K.(owned[i].complex.boundaries[d]) : zeros(K,dim(i,d-1),dim(i,d))
    for (e,direction) in enumerate(directions)
        a,b=direction===:forward ? (e,e+1) : (e+1,e)
        length(chain_maps[e])==slots || throw(ArgumentError("Each time map needs a component in every retained chain degree."))
        maps=Matrix{K}[]
        for d in 0:slots-1
            A=chain_maps[e][d+1]
            A isa AbstractMatrix && size(A)==(dim(b,d),dim(a,d)) || throw(ArgumentError("Time map $e has the wrong shape or direction in degree $d."))
            estimate+=length(A);_bound(estimate,budget.max_matrix_entries,"chain-map entries")
            all(x->x isa Union{Integer,Rational} || x isa K,A) || throw(ArgumentError("Temporal maps require coefficients in the declared exact field."))
            push!(maps,Matrix{K}(map(x->coerce(F,x),A)))
        end
        for d in 1:slots-1
            boundary(b,d)*maps[d+1]==maps[d]*boundary(a,d) || throw(ArgumentError("Time map $e does not commute with the boundary."))
        end
        push!(fullmaps,maps)
    end
    models=[_state_model(owned[i],ss[j],degree,F) for i in 1:n,j in 1:m]
    vid(i,j)=i+(j-1)*n
    edges=Dict{Tuple{Int,Int},Matrix{K}}()
    for j in 1:m,i in 1:n
        if j<m
            A=_inclusion(models[i,j].ids,models[i,j+1].ids,K)
            edges[(vid(i,j),vid(i,j+1))]=ChainComplexes.induced_map_on_homology(models[i,j].H,models[i,j+1].H,A)
        end
    end
    for j in 1:m,e in 1:n-1
        a,b=directions[e]===:forward ? (e,e+1) : (e+1,e)
        aa,bb=models[a,j],models[b,j]
        for d in 0:slots-1
            ia=d<length(aa.active) ? aa.active[d+1] : Int[]
            ib=d<length(bb.active) ? bb.active[d+1] : Int[]
            all(iszero,fullmaps[e][d+1][setdiff(1:dim(b,d),ib),ia]) ||
                throw(ArgumentError("Time map $e fails to preserve sampled scale $(ss[j])."))
        end
        A=degree<slots ? fullmaps[e][degree+1][bb.ids,aa.ids] : zeros(K,0,0)
        edges[(vid(a,j),vid(b,j))]=ChainComplexes.induced_map_on_homology(aa.H,bb.H,A)
    end
    dims=vec([model.H.dimH for model in models])
    estimate+=sum(length,values(edges);init=0);_bound(estimate,budget.max_matrix_entries,"retained module entries")
    Q=FiniteFringe.FinitePoset(BitMatrix([order[mod1(a,n),mod1(b,n)] && cld(a,n)<=cld(b,n) for a in 1:n*m,b in 1:n*m]);check=false)
    M=Modules.PModule{K}(Q,dims,edges;field=F)
    # Adjacent squares generate all relations of this product category.
    for j in 1:m-1,e in 1:n-1
        a,b=directions[e]===:forward ? (e,e+1) : (e+1,e)
        edges[(vid(b,j),vid(b,j+1))]*edges[(vid(a,j),vid(b,j))]==
            edges[(vid(a,j+1),vid(b,j+1))]*edges[(vid(a,j),vid(a,j+1))] || error("Temporal square failed to commute.")
    end
    zb=_zbudget(budget)
    bars=ZigzagPersistence.ZigzagResult[]
    for j in 1:m
        maps=[directions[e]===:forward ? edges[(vid(e,j),vid(e+1,j))] : edges[(vid(e+1,j),vid(e,j))] for e in 1:n-1]
        z=ZigzagPersistence.ZigzagModule(dims[(1:n).+(j-1)*n],maps;directions,field=F,times=ts,budget=zb)
        push!(bars,ZigzagPersistence.zigzag_persistence(z;representatives,budget=zb))
    end
    hs=representatives ? Matrix{Any}([x.H for x in models]) : nothing
    TemporalPersistenceResult(M,ts,ss,Symbol.(directions),owned,bars,hs,[copy(x.ids) for x in models],Int(degree),F,provenance,budget)
end

function _cloud_sources(clouds,point_ids,scales,degree,budget)
    n=length(clouds);n>0 || throw(ArgumentError("Provide at least one point cloud."))
    _bound(2big(n)-1,budget.max_occurrences,"union occurrences")
    matrices=[c isa DataTypes.PointCloud ? c.coords : DataTypes.PointCloud(c).coords for c in clouds]
    dimension=size(first(matrices),2);dimension>0 || throw(ArgumentError("Clouds need coordinates."))
    all(C->size(C,2)==dimension && all(x->x isa Real && isfinite(x) && isfinite(Float64(x)),C),matrices) ||
        throw(ArgumentError("Clouds need finite coordinates of a common dimension."))
    _bound(sum(C->big(length(C)),matrices),budget.max_matrix_entries,"input coordinates")
    ids=if point_ids===nothing
        offset=0
        map(matrices) do C
            out=collect(offset+1:offset+size(C,1));offset+=size(C,1);out
        end
    else
        length(point_ids)==n || throw(ArgumentError("Supply one point-ID vector per cloud."))
        [Int[_index(id,typemax(Int),"point ID") for id in row] for row in point_ids]
    end
    pool=Dict{Int,Vector{Float64}}();exact=Dict{Int,Tuple}()
    for (C,row) in zip(matrices,ids)
        length(row)==size(C,1) && allunique(row) || throw(ArgumentError("Each cloud needs one distinct point ID per row."))
        for (i,id) in enumerate(row)
            coordinates=Tuple(C[i,:])
            haskey(exact,id) && exact[id]!=coordinates && throw(ArgumentError("Point ID $id moved between clouds. Use different IDs for distinct samples; moving-point correspondence is not an inclusion."))
            exact[id]=coordinates;pool[id]=dimension==1 ? [Float64(coordinates[1]),0.] : Float64.(collect(coordinates))
        end
    end
    occurrences=Vector{Int}[]
    for i in 1:n
        push!(occurrences,sort(ids[i]))
        i<n && push!(occurrences,sort!(union(ids[i],ids[i+1])))
    end
    sources=DataIngestion.FilteredComplex[];total=0
    for row in occurrences
        _bound(length(row),budget.max_points,"points in a source or bridge")
        remaining=budget.max_cells-total
        if isempty(row)
            G=DataTypes.GradedComplex([Int[] for _ in 0:degree+1],[spzeros(Int,0,0) for _ in 0:degree],NTuple{1,Float64}[])
            push!(sources,DataIngestion.filtered_complex(G;geometry=Vector{NTuple{2,Float64}}[],source_ids=Tuple{Vararg{Int}}[],provenance=(;ambient_dim=max(2,dimension))))
        else
            C=reduce(vcat,[permutedims(pool[id]) for id in row])
            options=Options.ConstructionOptions(budget=Options.ConstructionBudget(max_simplices=remaining,max_edges=remaining,memory_budget_bytes=min(typemax(Int),16big(budget.max_matrix_entries)) ))
            x=DataIngestion.filtered_complex(DataTypes.PointCloud(C),DataIngestion.RipsFiltration(max_dim=degree+1,radius=Float64(last(scales)),construction=options))
            global_ids=[Tuple(row[v] for v in source) for source in x.sources]
            push!(sources,DataIngestion.filtered_complex(x.complex;geometry=x.geometry,source_ids=global_ids,
                faces=DataIngestion.face_incidence(x),provenance=merge(x.provenance,(;temporal_vertex_ids=copy(row),input_coordinate_dimension=dimension))))
        end
        total+=sum(DataTypes.cell_counts(last(sources).complex));_bound(total,budget.max_cells,"source cells")
    end
    F_sources=(;sources,occurrences,exact)
end

"""
    temporal_persistence(clouds; scales, times=1:length(clouds), point_ids=nothing, ...)

Build Rips(source), Rips(union), Rips(next source), with actual inclusions
`source -> union <- next source` at every supplied spatial scale. Rips grades
are maximum edge distances in Float64. Union complexes include cross edges.
By default samples from different clouds have distinct identities. Shared
`point_ids` require unchanged coordinates; moving points need distinct IDs.

`temporal_persistence(delay_clouds(signal; ...); scales, ...)` preserves the
identities of embedded samples shared by overlapping windows. The requested
homology degree constructs the skeleton through degree+1, including deaths.
"""
function temporal_persistence(clouds::Union{TemporalClouds,AbstractVector};scales,
        times=clouds isa TemporalClouds ? clouds.times : 1:length(clouds),
        point_ids=clouds isa TemporalClouds ? clouds.point_ids : nothing,
        degree=0,field=F2(),representatives=false,budget::TemporalBudget=TemporalBudget())
    input=clouds isa TemporalClouds ? clouds.clouds : clouds
    ts=_axis(times,"source times";strict=false);length(ts)==length(input) || throw(ArgumentError("Supply one time label per source cloud."))
    _bound(length(scales),budget.max_scales,"scales")
    ss=_axis(scales,"scales");first(ss)>=0 || throw(ArgumentError("Rips scales must be nonnegative."))
    degree isa Integer && !(degree isa Bool) && 0<=degree<budget.max_cells || throw(ArgumentError("degree must be a bounded nonnegative integer."))
    _field(field);built=_cloud_sources(input,point_ids,ss,Int(degree),budget);sources=built.sources;n=length(sources);K=coeff_type(field)
    times_=Rational{BigInt}[]
    for i in eachindex(ts);push!(times_,ts[i]);i<length(ts) && push!(times_,(ts[i]+ts[i+1])/2);end
    directions=[isodd(e) ? :forward : :backward for e in 1:n-1]
    slots=maximum(s->length(DataTypes.cell_counts(s.complex)),sources)
    chainmaps=Vector{SparseMatrixCSC{K,Int}}[]
    for (e,d) in enumerate(directions)
        a,b=d===:forward ? (e,e+1) : (e+1,e)
        cells(s,k)=[s.sources[i] for i in eachindex(s.sources) if s.complex.cell_dims[i]==k]
        push!(chainmaps,[_inclusion(cells(sources[a],k),cells(sources[b],k),K) for k in 0:slots-1])
    end
    provenance=(;construction=:rips_union_zigzag,occurrence_kinds=[isodd(i) ? :source : :union for i in 1:n],
        point_ids=built.occurrences,original_coordinates=built.exact,rips_convention=:maximum_edge_distance,
        grade_precision=:float64,embedding=clouds isa TemporalClouds ? clouds.provenance : nothing)
    temporal_persistence(sources;scales=ss,directions,chain_maps=chainmaps,times=times_,degree,field,representatives,budget,provenance)
end
