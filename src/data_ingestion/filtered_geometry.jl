# Source geometry is opt-in and travels with the actual constructed complex.
"""
    FilteredComplex

A graded complex with cell geometry, source identities and construction metadata.
Create with `filtered_complex(data, filtration)` or attach explicit geometry to
a supplied graded complex. This stores cells, not homology representatives.
Use `graded_complex`, `cell_geometry`, `cell_source`, `grade_orientation` and
`describe`; `check_filtered_complex` checks the chain and filtration contracts.
Treat retained arrays as read-only while an inspection session is using them.
"""
struct FilteredComplex{G,V,S,O,P}
    complex::G
    geometry::V
    sources::S
    orientation::O
    provenance::P
end
graded_complex(x::FilteredComplex)=x.complex
grade_orientation(x::FilteredComplex)=x.orientation
"""`cell_geometry(x, i)` returns the ordered geometric vertices of global cell `i`."""
cell_geometry(x::FilteredComplex,i::Integer)=x.geometry[i]
"""`cell_source(x, i)` returns the source identity of global cell `i`, before drawing."""
cell_source(x::FilteredComplex,i::Integer)=x.sources[i]
function describe(x::FilteredComplex)
    (;kind=:filtered_complex,cell_counts=DataTypes.cell_counts(x.complex),
      parameter_dim=DataTypes.parameter_dim(x.complex),orientation=x.orientation,
      ambient_dim=x.provenance.ambient_dim,construction=x.provenance,
      representatives=:not_computed)
end
Base.show(io::IO,x::FilteredComplex)=print(io,"FilteredComplex(",sum(DataTypes.cell_counts(x.complex)),
    " cells, ",DataTypes.parameter_dim(x.complex)," parameters, ",x.provenance.effective,
    ", ",x.provenance.model,")")

"""
    check_filtered_complex(x; throw=false)

Check cell identities, finite geometry, boundary shapes, integer boundary-square
zero and filtration closure. For a multicritical cell, every appearance of it
must have an active occurrence of each incident face. Geometry of supplied
cellular attaching maps is user data; this does not certify an embedding.
"""
function check_filtered_complex(x::FilteredComplex;throw=false)
    issues=String[]
    try
        G=x.complex;counts=DataTypes.cell_counts(G);n=sum(counts);N=DataTypes.parameter_dim(G)
        length(x.orientation)==N && all(o->o isa Int && abs(o)==1,x.orientation) ||
            error("orientation must have one +1/-1 per grade coordinate.")
        length(x.geometry)==length(x.sources)==n || error("geometry and source identities must cover every cell.")
        all(d->length(unique(DataTypes.cells(G;dim=d)))==counts[d+1],0:length(counts)-1) ||
            error("cell IDs must be unique within each dimension.")
        D=x.provenance.ambient_dim
        D>=2 || error("source geometry needs at least two coordinates.")
        dims=G.cell_dims
        for i in 1:n
            vertices=x.geometry[i];d=dims[i]
            (d==0 ? length(vertices)==1 : d==1 ? length(vertices)>=2 : length(vertices)>=d+1) ||
                error("cell $i has insufficient geometric vertices for dimension $d.")
            all(p->length(p)==D && all(v->v isa Real && isfinite(v),p),vertices) ||
                error("cell $i has invalid geometric coordinates.")
            grades=DataTypes.cell_grade_set(G,i)
            !isempty(grades) && all(g->length(g)==N && all(v->v isa Real && isfinite(v),g),grades) ||
                error("cell $i has invalid grades.")
        end
        B=DataTypes.boundary_maps(G);length(B)==length(counts)-1 || error("wrong number of boundary maps.")
        offsets=cumsum(vcat(0,counts))
        for d in eachindex(B)
            size(B[d])==(counts[d],counts[d+1]) || error("boundary $d has the wrong shape.")
            # BigInt prevents overflow from certifying a false chain identity.
            d>1 && !iszero(SparseMatrixCSC{BigInt,Int}(B[d-1])*SparseMatrixCSC{BigInt,Int}(B[d])) &&
                error("boundary squared is nonzero in dimension $d.")
            rows,cols,vals=findnz(B[d])
            for (r,c,v) in zip(rows,cols,vals)
                iszero(v) && continue
                face=DataTypes.cell_grade_set(G,offsets[d]+r)
                coface=DataTypes.cell_grade_set(G,offsets[d+1]+c)
                all(g->any(f->all(k->x.orientation[k]==1 ? f[k]<=g[k] : f[k]>=g[k],1:N),face),coface) ||
                    error("face $(offsets[d]+r) is not active whenever cell $(offsets[d+1]+c) is active.")
            end
        end
    catch err
        err isa InterruptException && rethrow();push!(issues,sprint(showerror,err))
    end
    valid=isempty(issues)
    throw && !valid && Base.throw(ArgumentError(join(issues,"\n")))
    return (;kind=:filtered_complex_validation,valid,issues)
end

"""
    filtered_complex(G; geometry, orientation=(1,...), source_ids=nothing,
                     construction="supplied cellular complex", model=:original)
    filtered_complex(tree; coordinates, orientation=(1,...))
    filtered_complex(data, filtration; cache=:auto)

Retain the actual filtered cells with source geometry for static or live inspection.
The first form accepts a graded cellular complex and one ordered vertex list per
global cell (dimension-major order). Polygon vertices must be cyclic; edges may
be polylines. Higher cells retain vertices but are displayed through a labelled
2-skeleton. Coordinates may have more than three components; projection is an
explicit visualization choice. `source_ids` defaults to `(dimension, cell_id)`.

The data form constructs supported point-cloud simplicial, embedded graph or
vertex-valued cubical image filtrations. It records actual construction choices,
including sparse/reduced models and source vertex IDs. Periodic cubes use an
unwrapped fundamental-domain drawing with boundary identifications. It never
substitutes a radius graph or recomputes a source correspondence by proximity.
Specialized cell models can use the explicit geometry form. No homology, chain
lifting or representative optimization is performed.
"""
function filtered_complex(G::Union{GradedComplex,MultiCriticalGradedComplex};geometry=nothing,
        cubical_shape=nothing,periodic=false,
        orientation=ntuple(_->1,DataTypes.parameter_dim(G)),source_ids=nothing,
        construction="supplied cellular complex",model=:original,provenance=(;))
    model in (:original,:reduced) || throw(ArgumentError("model must be :original or :reduced."))
    if cubical_shape !== nothing
        geometry === nothing || throw(ArgumentError("Supply either geometry or cubical_shape."))
        shape=Tuple(cubical_shape);N=length(shape)
        all(n->n isa Int && n>0,shape) || throw(ArgumentError("cubical_shape must contain positive vertex counts."))
        per=_cubical_periodic_tuple(periodic,Val(N))
        structure=_cubical_structure(shape;periodic=per)
        map(length,structure.cells_by_dim)==DataTypes.cell_counts(G) || throw(ArgumentError("Cubical shape does not match the constructed cells."))
        all(pair->abs.(pair[1])==abs.(pair[2]),zip(structure.boundaries,DataTypes.boundary_maps(G))) ||
            throw(ArgumentError("Cubical incidences do not match the supplied complex."))
        geometry,identities=_filtered_cubical_geometry(structure.cells_by_dim,Val(N))
        source_ids === nothing && (source_ids=identities)
        provenance=merge(provenance,(;periodic=per,domain=ntuple(i->(1,shape[i]+Int(per[i])),N),source_mapping=:cubical_origin_mask))
    end
    geometry isa AbstractVector || throw(ArgumentError("geometry must be a vector of cell vertex lists."))
    vertices=[[Tuple(p) for p in cell] for cell in geometry]
    !isempty(vertices) && isempty(first(vertices)) && throw(ArgumentError("The first cell needs geometric vertices."))
    D=isempty(vertices) ? get(provenance,:ambient_dim,0) : length(first(first(vertices)))
    D>=2 || throw(ArgumentError("Empty geometry requires provenance with ambient_dim >= 2."))
    sources=source_ids === nothing ? [(d,id) for (d,ids) in enumerate(DataTypes.cells(G)) for id in ids] : collect(source_ids)
    source_ids === nothing && (sources=[(d-1,id) for (d,id) in sources])
    p=merge((;requested=String(construction),effective=String(construction),grade_scale=:supplied,
        source_mapping=:supplied,periodic=(),domain=nothing,lifting_maps=nothing),provenance,
        (;ambient_dim=D,model))
    x=FilteredComplex(G,vertices,sources,Tuple(orientation),p)
    check_filtered_complex(x;throw=true)
    return x
end

function filtered_complex(ST::SimplexTreeMulti;coordinates,orientation=ntuple(_->1,DataTypes.parameter_dim(ST)),
        construction="supplied simplicial complex",model=:original,provenance=(;))
    C=coordinates isa PointCloud ? point_matrix(coordinates) : coordinates
    C isa AbstractMatrix || throw(ArgumentError("coordinates must be a vertex-by-coordinate matrix or PointCloud."))
    simplices=[collect(simplex_vertices(ST,i)) for i in 1:simplex_count(ST)]
    all(s->all(v->1<=v<=size(C,1),s),simplices) || throw(ArgumentError("A simplex has no supplied vertex coordinate."))
    return filtered_complex(_graded_complex_from_simplex_tree(ST);
        geometry=[[Tuple(C[v,:]) for v in s] for s in simplices],source_ids=Tuple.(simplices),
        orientation,construction,model,provenance)
end

function _filtered_simplicial_geometry(G,C,vertex_ids)
    simplices=_simplices_from_complex(G)
    maximum((maximum(s;init=0) for group in simplices for s in group);init=0)<=size(C,1) ||
        throw(ArgumentError("Constructed vertices do not have retained source coordinates."))
    geometry=[[Tuple(C[v,:]) for v in s] for group in simplices for s in group]
    sources=[Tuple(vertex_ids[v] for v in s) for group in simplices for s in group]
    return geometry,sources
end

function filtered_complex(data::Union{PointCloud,GraphData,EmbeddedPlanarGraph2D,ImageNd},
        filtration::AbstractFiltration;cache=:auto)
    cache = _workflow_encoding_cache(_resolve_workflow_session_cache(cache))
    spec=_filtration_spec(filtration)
    construction=_construction_from_params(spec.params)
    provenance=_ingestion_construction_provenance(data,spec,filtration)
    provenance=merge(provenance,(;filtration_parameters=filtration_parameters(filtration),
        sparsify=construction.sparsify,collapse=construction.collapse,
        source_mapping=:retained_cell_vertices))
    model=construction.collapse==:none ? :original : :reduced
    if data isa PointCloud
        spec.kind===:rhomboid && return _filtered_rhomboid(data,spec,provenance;cache)
        spec.kind in (:landmark_rips,:witness,:dowker) &&
            throw(ArgumentError("This cell model needs explicit retained cell geometry; use filtered_complex(G;geometry,...)."))
        reduced,actual,idx=_maybe_greedy_perm_reduce(data,spec)
        G,axes,orientation=_graded_complex_from_point_cloud(reduced,actual;cache)
        C=point_matrix(reduced);ids=idx === nothing ? collect(1:size(C,1)) : idx
        geometry,sources=_filtered_simplicial_geometry(G,C,ids)
        provenance=merge(provenance,(;source_vertex_ids=ids,sampled=idx !== nothing))
        if spec.kind in (:core,:core_delaunay) && idx === nothing && model===:original
            provenance=merge(provenance,(;geometric=(;sites=QQ.(C),family=:core,
                representation=:source_simplices,backend=spec.kind,axes,
                radius_cutoff=get(spec.params,:radius,nothing),depth_window=(1,size(C,1)),
                grade_precision=:floating_point,max_dim=get(spec.params,:max_dim,2),
                beta=get(spec.params,:beta,1.0),k_values=get(spec.params,:k_values,nothing)),ambient_dim=size(C,2)))
        end
    elseif data isa ImageNd
        G,axes,orientation=_graded_complex_from_image(data,spec;cache)
        dims=size(data.data);N=length(dims);N>=2 || throw(ArgumentError("Image geometry needs at least two spatial axes."))
        periodic=_cubical_periodic_from_spec(spec,Val(N))
        cells=_cubical_structure(dims;periodic,cache).cells_by_dim
        geometry,sources=_filtered_cubical_geometry(cells,Val(N))
        provenance=merge(provenance,(;source_mapping=:cubical_origin_mask,periodic,
            domain=ntuple(i->(1,dims[i]+Int(periodic[i])),N),image_values=:vertices,
            source_field=:actual_vertex_grades))
    else
        data isa EmbeddedPlanarGraph2D && spec.kind==:wing_vein_bifiltration &&
            throw(ArgumentError("A sampled wing field needs explicit cell geometry; a graph drawing is not its cubical complex."))
        G,axes,orientation=_graded_complex_from_data(data,spec;cache)
        C=DataTypes.vertex_coordinates(data)
        C === nothing && throw(ArgumentError("Graph input needs source coordinates."))
        C=C isa AbstractMatrix ? C : reduce(vcat,permutedims.(collect.(C)))
        vertex_ids=spec.kind==:edge_weighted ? _weighted_flag_graph(data,spec).source_indices : collect(1:size(C,1))
        geometry,sources=_filtered_simplicial_geometry(G,C[vertex_ids,:],vertex_ids)
        if data isa EmbeddedPlanarGraph2D && data.polylines !== nothing && DataTypes.max_dim(G)==1
            # The graph owner preserves edge order in its non-clique route.
            spec.kind in (:graph_lower_star,:graph_weight_threshold) ||
                throw(ArgumentError("Polyline geometry needs the non-clique graph route or explicit cell geometry."))
            n=DataTypes.cell_counts(G)[1]
            for (i,poly) in enumerate(data.polylines);geometry[n+i]=Tuple.(poly);end
        end
    end
    return filtered_complex(G;geometry,source_ids=sources,orientation,
        construction=string(spec.kind),model,provenance)
end

function _filtered_cubical_geometry(cells,::Val{N}) where {N}
    geometry=Vector{Vector{NTuple{N,Int}}}();sources=NamedTuple[]
    for group in cells, (origin,mask) in group
        varying=findall(==(1),collect(mask));d=length(varying)
        # Gray order is cyclic on each square; higher cells keep corner data.
        bits=d==2 ? (0,1,3,2) : (0:(2^d-1))
        vertices=[ntuple(a->origin[a]+(a in varying ? (b >> (findfirst(==(a),varying)-1)) & 1 : 0),N) for b in bits]
        push!(geometry,vertices);push!(sources,(;origin,mask))
    end
    return geometry,sources
end
