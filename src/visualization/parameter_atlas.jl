# A finite restriction of one module, with actual maps between sampled states.
"""
    ParameterAtlas

A budgeted restriction to declared parameter states, constructed with
`parameter_atlas`. `atlas_states` describes the sampled spaces;
`atlas_transition(atlas; source, target)` returns the actual map and its
kernel/cokernel dimensions, or explicitly reports that no map is defined.
Indices refer to the declared states, not to cell IDs or finite-poset labels.
Retained arrays and the input object are read-only. Sampling does not certify
an exhaustive event subdivision or a decomposition into persistent classes.
"""
struct ParameterAtlas{O,F,S,T}
    object::O
    field::F
    states::S
    transitions::T
    options::NamedTuple
end

"""Return the parameters, finite labels, dimensions and active source IDs of an atlas."""
atlas_states(a::ParameterAtlas) = [Base.structdiff(s,(;view=s.view)) for s in a.states]
"""Describe the sampling, budgets, field and retained maps of a parameter atlas."""
function atlas_summary(a::ParameterAtlas)
    (;kind=:parameter_atlas,states=length(a.states),maps=count(t->t.defined,values(a.transitions)),
      field=a.field,degree=a.options.degree,sampling=a.options.sampling,grid_shape=a.options.grid_shape,
      exhaustive=false,witnesses=a.options.witnesses,budgets=a.options.budgets,workspace_estimate=a.options.workspace_estimate,
      matrix_entries=sum(t->t.defined ? length(t.matrix) : 0,values(a.transitions)),
      source_cells=a.object isa DataIngestion.FilteredComplex)
end
describe(a::ParameterAtlas)=atlas_summary(a)
Base.show(io::IO,a::ParameterAtlas)=print(io,"ParameterAtlas(",length(a.states)," states, ",atlas_summary(a).maps," maps, sampled)")

"""Check atlas map shapes, identities and every supported composition equation.
This optional check multiplies retained matrices; it does not reconstruct source homology.
"""
function check_parameter_atlas(a::ParameterAtlas;throw=false)
    issues=String[]
    try
        for i in eachindex(a.states),j in eachindex(a.states)
            t=a.transitions[(i,j)];t.defined || continue
            size(t.matrix)==(a.states[j].dimension,a.states[i].dimension) || error("Wrong map shape at ($i,$j).")
            if i==j
                all(t.matrix[r,c]==(r==c ? one(eltype(t.matrix)) : zero(eltype(t.matrix))) for r in axes(t.matrix,1),c in axes(t.matrix,2)) || error("Identity equation failed at $i.")
            end
            for k in eachindex(a.states)
                u=a.transitions[(j,k)];u.defined || continue
                v=a.transitions[(i,k)]
                v.defined && u.matrix*t.matrix==v.matrix || error("Composition equation failed at ($i,$j,$k).")
            end
        end
    catch err
        err isa InterruptException && rethrow();push!(issues,sprint(showerror,err))
    end
    valid=isempty(issues);throw && !valid && _throw_invalid_visual(:check_parameter_atlas,issues)
    (;kind=:parameter_atlas,valid,issues)
end

function _atlas_index(a,i,name)
    i isa Integer && !(i isa Bool) && 1<=i<=length(a.states) ||
        throw(ArgumentError("$name must be a state index in 1:$(length(a.states))."))
    Int(i)
end

"""
    atlas_transition(atlas; source, target)

Inspect the map from one declared state to another. `defined=false` and
`matrix=nothing` distinguish unavailable, reversed and incomparable queries
from a genuine zero map. Rows use the target basis; columns use the source
basis. `entries` records newly active source cells, independently of homology.
With `witnesses=true` at construction, `kernel` and `cokernel` contain chosen
coordinate bases; source complexes additionally retain `killed` cycle/filling
pairs and `born` cycles. These explain the endpoint map, not individual event
times inside an unsampled interval, and are not canonical generators.
"""
function atlas_transition(a::ParameterAtlas;source,target)
    a.transitions[(_atlas_index(a,source,:source),_atlas_index(a,target,:target))]
end

function _atlas_parameters(x,parameters,grid,max_states)
    (parameters===nothing) != (grid===nothing) || throw(ArgumentError("Supply parameters or grid, exactly one."))
    if grid!==nothing
        x isa Modules.PModule && throw(ArgumentError("A finite module takes vertex IDs in parameters, not a coordinate grid."))
        grid isa Tuple && length(grid)==2 || throw(ArgumentError("grid must be (xcoordinates, ycoordinates)."))
        all(v->v isa Union{AbstractVector,Tuple,AbstractRange} && !isempty(v),grid) ||
            throw(ArgumentError("Grid axes must be nonempty finite vectors, ranges or tuples."))
        prod(Int128(length(v)) for v in grid)<=max_states || throw(ArgumentError("Grid exceeds max_states=$max_states."))
        all(allunique,grid) || throw(ArgumentError("Grid coordinates must be distinct on each axis."))
        # Rows run from the last y coordinate to the first, like a Cartesian plot.
        return [(u,v) for v in reverse(grid[2]) for u in grid[1]],(length(grid[2]),length(grid[1])),:sampled_grid
    end
    parameters isa Union{AbstractVector,Tuple,AbstractRange} && 0<length(parameters)<=max_states ||
        throw(ArgumentError("parameters must contain 1:$max_states declared states."))
    collect(parameters),nothing,:sampled_states
end

function _atlas_relation(x,p,q)
    if x isa Modules.PModule
        xy=leq(x.Q,p,q);yx=leq(x.Q,q,p)
    else
        orientation=x isa DataIngestion.FilteredComplex ? x.orientation :
            _inspection_parameter_orientation(_inspection_classifier(encoding_map(x)))
        xy=all(i->orientation[i]==1 ? p[i]<=q[i] : p[i]>=q[i],eachindex(p))
        yx=all(i->orientation[i]==1 ? q[i]<=p[i] : q[i]>=p[i],eachindex(p))
    end
    xy ? (yx ? :equal : :comparable) : (yx ? :reverse_comparable : :incomparable)
end

function _atlas_source_model(x,p,degree,field)
    G=DataIngestion.graded_complex(x);counts=DataTypes.cell_counts(G);offsets=cumsum(vcat(0,counts))
    active=[[i for i in 1:counts[d+1] if _complex_active(x,offsets[d+1]+i,p)] for d in 0:length(counts)-1]
    K=CoreModules.coeff_type(field)
    ids(d)=0<=d<length(counts) ? active[d+1] : Int[]
    boundary(d)=d==0 || d>=length(counts) ? zeros(K,length(ids(d-1)),length(ids(d))) :
        Matrix{K}(map(v->CoreModules.coerce(field,v),Matrix(DataTypes.boundary_map(G;dim=d)[ids(d-1),ids(d)])))
    current,next=boundary(degree),boundary(degree+1)
    H=ChainComplexes.homology_data(next,current,degree;field)
    (;active,ids=ids(degree),next_ids=ids(degree+1),next,H,
      global_ids=sort(vcat([offsets[d].+a for (d,a) in enumerate(active)]...)))
end

function _atlas_inclusion(a,b,K)
    rows=Dict(i=>j for (j,i) in enumerate(b.ids))
    sparse([rows[i] for i in a.ids],collect(eachindex(a.ids)),fill(one(K),length(a.ids)),length(b.ids),length(a.ids))
end

function _atlas_cokernel(field,A,r)
    K=eltype(A);n=size(A,1)
    r==n && return zeros(K,n,0)
    identity=zeros(K,n,n)
    for i in 1:n;identity[i,i]=one(K);end
    # A single elimination extends the image with coordinate vectors in order.
    _,pivots=FieldLinAlg.rref(field,hcat(A,identity))
    identity[:,[j-size(A,2) for j in pivots if j>size(A,2)]]
end

function _atlas_source_chain(x,p,field,degree,ids,coefficients,kind)
    counts=DataTypes.cell_counts(DataIngestion.graded_complex(x))
    full=zeros(CoreModules.coeff_type(field),degree<length(counts) ? counts[degree+1] : 0)
    full[ids]=coefficients
    c=DataIngestion.SourceChain(x,full,degree,p,field,kind,(;choice=:homology_quotient,optimization=:none))
    DataIngestion.check_source_chain(c;throw=true)
    c
end

function _atlas_witnesses(x,p,q,field,degree,src,tgt,inclusion,kernel,cokernel)
    killed=NamedTuple[];born=DataIngestion.SourceChain[]
    for c in eachcol(kernel)
        z=vec(ChainComplexes.basis(src.H)*c);image=inclusion*z
        filling=ChainComplexes.solve_particular(field,tgt.next,reshape(image,:,1))
        filling===nothing && error("An atlas kernel class has no target filling.")
        tgt.next*filling==reshape(image,:,1) || error("Atlas filling equation failed.")
        cycle=_atlas_source_chain(x,p,field,degree,src.ids,z,:cycle)
        chain=_atlas_source_chain(x,q,field,degree+1,tgt.next_ids,vec(filling),:chain)
        push!(killed,(;cycle,filling=chain))
    end
    for c in eachcol(cokernel)
        z=vec(ChainComplexes.basis(tgt.H)*c)
        push!(born,_atlas_source_chain(x,q,field,degree,tgt.ids,z,:cycle))
    end
    (;killed,born)
end

"""
    parameter_atlas(object; parameters, degree=0, witnesses=false, ...)
    parameter_atlas(object; grid=(xs, ys), degree=0, witnesses=false, ...)

Restrict a retained `FilteredComplex`, planar `EncodingResult`, or finite
`PModule` to a small, explicit set of states. A source complex computes degree
`degree` homology over `field=QQField()`; a module keeps its stored field and
stalks (omit `degree` and `field`). For a finite module, parameters are vertex
IDs. Grid coordinates are exact query coordinates, not pixels.

All ordered endpoint maps are computed, including nonadjacent pairs and
identities. Incomparable parameters have no map even if their encoding labels
coincide. Label zero is unavailable, not a zero stalk. Use `atlas_transition`
to inspect maps, `visualize(atlas)` for an aligned contact sheet, and
`visualize(inspection_session(atlas))` for linked selection.

`witnesses=true` retains chosen kernel and cokernel bases and, for source
complexes, actual cycles and fillings. It increases construction and storage
cost. Only exact rational and prime fields are supported. Limits are explicit:
`max_states=16`, `max_transitions=256`, `max_cells=5000`, and
`max_matrix_entries=2_000_000`. The matrix budget covers a conservative algebra
workspace estimate plus retained maps and witnesses; it is not a byte limit.
`view_dims=nothing` uses the source's two or three spatial coordinates; select
a projection explicitly for higher-dimensional source geometry. A grid/path
is always labelled sampled, without claiming exhaustive event coverage.
"""
function parameter_atlas(x::Union{DataIngestion.FilteredComplex,EncodingResult,Modules.PModule};
        parameters=nothing,grid=nothing,degree=nothing,field=nothing,witnesses=false,
        max_states=16,max_transitions=256,max_cells=5000,max_matrix_entries=2_000_000,view_dims=nothing)
    for (name,v) in pairs((;max_states,max_transitions,max_cells,max_matrix_entries))
        v isa Int && v>0 || throw(ArgumentError("$name must be a positive Int."))
    end
    witnesses isa Bool || throw(ArgumentError("witnesses must be Bool."))
    source=x isa DataIngestion.FilteredComplex
    if source
        degree=degree===nothing ? 0 : degree
        degree isa Int && 0<=degree<=DataTypes.max_dim(x.complex) || throw(ArgumentError("degree must be a retained nonnegative chain degree."))
        field=field===nothing ? CoreModules.QQField() : field
    else
        degree===nothing && field===nothing || throw(ArgumentError("A module atlas uses its stored stalks and field; omit degree and field."))
        field=_inspection_field(x)
        view_dims===nothing || throw(ArgumentError("view_dims applies to source geometry only."))
        x isa EncodingResult && !_inspection_has_geometry(x) &&
            throw(ArgumentError("Coordinate atlases support the planar classifiers used by module inspection. Use encoding_module with finite vertex IDs otherwise."))
    end
    field isa Union{CoreModules.QQField,CoreModules.PrimeField} || throw(ArgumentError("Parameter atlases require an exact rational or prime field."))
    points,shape,sampling=_atlas_parameters(x,parameters,grid,max_states)
    if source
        points=[_complex_options(x;parameter=p,view_dims,max_cells).parameter for p in points]
    elseif x isa Modules.PModule
        all(p->p isa Int && 1<=p<=nvertices(x.Q),points) || throw(ArgumentError("Finite-module states must be valid vertex IDs."))
    else
        points=[only(_collect_query_points(;point=p)) for p in points]
    end
    allunique(points) || throw(ArgumentError("Atlas states must be distinct."))
    shape===nothing && all(i->_atlas_relation(x,points[i],points[i+1]) in (:equal,:comparable),1:length(points)-1) && (sampling=:sampled_path)
    relations=[_atlas_relation(x,p,q) for p in points,q in points]
    count(r->r in (:equal,:comparable),relations)<=max_transitions || throw(ArgumentError("Ordered pairs exceed max_transitions=$max_transitions."))
    size_bound=source ? sum(DataTypes.cell_counts(x.complex)) : nvertices(_inspection_poset(x))
    size_bound<=max_cells || throw(ArgumentError("Input has $size_bound cells/vertices, exceeding max_cells=$max_cells."))
    dims=source ? nothing : _inspection_dimensions(x)
    labels=source ? fill(nothing,length(points)) : x isa Modules.PModule ? points :
        [_visual_locate(_inspection_classifier(encoding_map(x)),p) for p in points]
    maxdim=source ? size_bound : maximum(dims;init=0)
    estimate=source ? Int128(6)*length(points)*Int128(size_bound)^2 : Int128(0)
    estimate+=Int128(maxdim)^2*count(r->r in (:equal,:comparable),relations)*(witnesses ? 5 : 1)
    # A lazy encoding may materialize maps outside the selected restriction.
    # Include its complete possible block storage in the preflight estimate.
    if x isa EncodingResult && !Results.result_summary(x).materialized
        estimate+=sum(Int128(d) for d in dims)^2
    end
    estimate<=max_matrix_entries || throw(ArgumentError("Conservative algebra estimate $estimate exceeds max_matrix_entries=$max_matrix_entries; reduce states/input or raise the explicit budget."))
    prepared=source ? nothing : _prepare_hasse(_inspection_poset(x))
    models=source ? [_atlas_source_model(x,p,degree,field) for p in points] : nothing
    states=map(eachindex(points)) do i
        p=points[i];label=labels[i]
        dimension=source ? dimensions(models[i].H).homology : label==0 ? nothing : dims[label]
        active=source ? models[i].global_ids : Int[]
        counts=source ? length.(models[i].active) : Int[]
        view=source ? first(visual_spec(x;parameter=p,view_dims,max_cells).panels) :
            _hasse_spec(_inspection_poset(x);dims,vertex=label==0 ? nothing : label,prepared,field_label=_inspection_field_label(field))
        (;index=i,parameter=p,label,dimension,active,counts,view)
    end
    transitions=Dict{Tuple{Int,Int},NamedTuple}();K=CoreModules.coeff_type(field)
    for i in eachindex(points),j in eachindex(points)
        relation=relations[i,j]
        defined=relation in (:equal,:comparable) && states[i].dimension!==nothing && states[j].dimension!==nothing
        unavailable=states[i].dimension===nothing || states[j].dimension===nothing
        entries=NamedTuple[]
        if source && defined
            active=falses(size_bound);active[states[j].active].=true
            entries=[_complex_cell_record(x,id,active) for id in setdiff(states[j].active,states[i].active)]
        end
        A=if !defined
            nothing
        elseif source
            inc=_atlas_inclusion(models[i],models[j],K)
            ChainComplexes.induced_map_on_homology(models[i].H,models[j].H,inc)
        else
            M=x isa EncodingResult ? Results.encoding_module(x) : x
            copy(Modules.structure_map(M;source=labels[i],target=labels[j]))
        end
        r=defined ? FieldLinAlg.rank(field,A) : nothing
        kernel=defined && witnesses ? FieldLinAlg.nullspace(field,A) : nothing
        cokernel=defined && witnesses ? _atlas_cokernel(field,A,r) : nothing
        chains=source && defined && witnesses ? _atlas_witnesses(x,points[i],points[j],field,degree,models[i],models[j],
            _atlas_inclusion(models[i],models[j],K),kernel,cokernel) : (;killed=NamedTuple[],born=DataIngestion.SourceChain[])
        transitions[(i,j)]=(;source=i,target=j,relation=unavailable ? :outside : relation,defined,matrix=A,rank=r,
            kernel_dimension=defined ? size(A,2)-r : nothing,cokernel_dimension=defined ? size(A,1)-r : nothing,
            entries,kernel,cokernel,chains...)
    end
    orientation=source ? x.orientation : x isa Modules.PModule ? nothing : _inspection_parameter_orientation(_inspection_classifier(encoding_map(x)))
    options=(;degree,witnesses,view_dims,sampling,grid_shape=shape,orientation,
        budgets=(;max_states,max_transitions,max_cells,max_matrix_entries),workspace_estimate=estimate)
    ParameterAtlas(x,field,states,transitions,options)
end
