"""
A minimal projective presentation of graph H0 over a specified finite poset.
`generator_labels`, `relation_labels`, and `presentation_matrix` expose the
compact algebraic data. `presentation_module` and `presentation_map` explicitly
materialize stalks. On products of at most two chains, `betti_table` also uses
computed second Betti labels; higher-dimensional bases use general resolutions.
"""
struct H0Presentation{K,F<:AbstractCoeffField,P<:AbstractPoset}
    poset::P
    field::F
    generators::Vector{Int}
    relations::Vector{Int}
    edges::Vector{NTuple{2,Int}}
    matrix::SparseMatrixCSC{K,Int}
    second::Union{Nothing,Vector{Int}}
    syzygies::Union{Nothing,SparseMatrixCSC{K,Int}}
    original_vertices::Vector{Int}
    vertex_images::Vector{Int}
    source_vertices::Vector{Int}
    source_edges::Vector{NTuple{2,Int}}
    source_grades::Vector{Int}
    algorithm::Symbol
end

generator_labels(p::H0Presentation)=copy(p.generators)
relation_labels(p::H0Presentation)=copy(p.relations)
presentation_matrix(p::H0Presentation)=copy(p.matrix)
provenance(p::H0Presentation)=(category=:finite_poset_representations,base_poset=p.poset,field=p.field,homology_degree=0)
"""Return compact generator, relation and second Betti counts without materializing stalks."""
function h0_presentation_summary(p::H0Presentation)
    return (kind=:h0_presentation,algorithm=p.algorithm,field=p.field,
        nvertices=nvertices(p.poset),generators=length(p.generators),relations=length(p.relations),
        second_betti=p.second===nothing ? nothing : length(p.second),
        syzygies_retained=p.syzygies!==nothing,provenance=provenance(p))
end
describe(p::H0Presentation)=h0_presentation_summary(p)
presentation_summary(p::H0Presentation)=h0_presentation_summary(p)
Base.show(io::IO,p::H0Presentation)=print(io,"H0Presentation(",length(p.generators)," generators, ",length(p.relations)," relations; ",p.algorithm,")")

function _h0_validate_graph(P,vg,edges,eg)
    length(edges)==length(eg) || throw(DimensionMismatch("h0_presentation: one grade per edge is required"))
    n=nvertices(P)
    all(g->1<=g<=n,vg) && all(g->1<=g<=n,eg) || throw(ArgumentError("h0_presentation: grades must be labels of the finite poset"))
    for (i,(u,v)) in enumerate(edges)
        1<=u<=length(vg) && 1<=v<=length(vg) || throw(ArgumentError("h0_presentation: invalid endpoint"))
        leq(P,vg[u],eg[i]) && leq(P,vg[v],eg[i]) || throw(ArgumentError("h0_presentation: edge precedes an endpoint"))
    end
end

"""
    h0_presentation(P; vertex_grades, edges, edge_grades, field=QQField(),
                    syzygies=false, max_witness_entries=10_000_000)
    h0_presentation(enc; syzygies=false, max_expanded_cells=1_000_000, ...)

Compute an exact minimal H0 presentation. Supplied graph grades are finite-poset
labels; edges are pairs of vertex indices, including loops and parallel edges.
Encoding inputs use retained degree-zero graph boundaries. No geometric source
or ambient polynomial-ring interpretation is inferred from a finite category.
Products of one or two chains use an ordered graph sweep. `syzygies=true`
retains its second differential, with an explicit bound on stored entries.
General posets use filtered graph deletion and general higher resolutions.
"""
function h0_presentation(P::AbstractPoset; vertex_grades,edges,edge_grades,
                         field::AbstractCoeffField=QQField(),syzygies::Bool=false,
                         max_witness_entries::Int=10_000_000)
    field isa Union{QQField,PrimeField,RealField} || throw(ArgumentError("h0_presentation: unsupported field"))
    max_witness_entries>=0 || throw(ArgumentError("h0_presentation: max_witness_entries must be nonnegative"))
    vg=Int[vertex_grades...]; eg=Int[edge_grades...]
    es=NTuple{2,Int}[Tuple(e) for e in edges]
    _h0_validate_graph(P,vg,es,eg)
    roots,images,reduced,grades=_h0_collapse(P,vg,es,eg)
    shape=_h0_grid_shape(P)
    sweep=shape!==nothing && length(shape[1])<=2
    syzygies && !sweep && throw(ArgumentError("h0_presentation: retained syzygies require a product of one or two chains; use resolve for a general poset"))
    if sweep
        kept,second,S=_h0_sweep(P,length(roots),reduced,grades;witnesses=syzygies,max_witness_entries)
    else
        kept=_h0_minimal_edges(P,length(roots),reduced,grades)
        second=nothing; S=nothing
    end
    K=coeff_type(field); rows=Int[]; cols=Int[]; values=K[]
    for (j,e) in enumerate(kept)
        u,v=reduced[e]
        append!(rows,(u,v)); append!(cols,(j,j)); append!(values,(-one(K),one(K)))
    end
    matrix=sparse(rows,cols,values,length(roots),length(kept))
    return H0Presentation(P,field,vg[roots],grades[kept],reduced[kept],matrix,second,
        S===nothing ? nothing : SparseMatrixCSC{K,Int}(S),roots,images,vg,es,eg,
        sweep ? :bigraded_forest : :poset_graph)
end

# Materialization is explicit: no dense |P| expansion is needed for a compact
# presentation or its first two Betti rows.
function _h0_projective(P,labels,field)
    K=coeff_type(field)
    active=[findall(g->leq(P,g,v),labels) for v in 1:nvertices(P)]
    maps=Dict{Tuple{Int,Int},SparseMatrixCSC{K,Int}}()
    for (u,v) in cover_edges(P)
        positions=Dict(g=>i for (i,g) in enumerate(active[v]))
        maps[u,v]=sparse([positions[g] for g in active[u]],collect(eachindex(active[u])),
                          fill(one(K),length(active[u])),length(active[v]),length(active[u]))
    end
    return PModule{K}(P,length.(active),maps;field),active
end
function presentation_map(p::H0Presentation{K}) where {K}
    P0,a0=_h0_projective(p.poset,p.generators,p.field)
    P1,a1=_h0_projective(p.poset,p.relations,p.field)
    return PMorphism(P1,P0,[p.matrix[a0[v],a1[v]] for v in 1:nvertices(p.poset)])
end
# Component coordinates give an exact combinatorial quotient, even when the
# chosen scalar type is floating point. Numerical cokernel bases can introduce
# roundoff into a differential that should vanish identically.
function _h0_component_module(p::H0Presentation{K}) where {K}
    P=p.poset; n=nvertices(P)
    components=Vector{Vector{Int}}(undef,n)
    representatives=Vector{Vector{Int}}(undef,n)
    projections=Vector{SparseMatrixCSC{K,Int}}(undef,n)
    for v in 1:n
        roots=_h0_components_at(P,p.generators,p.edges,p.relations,v)
        labels=sort!(unique(filter(!iszero,roots)))
        positions=Dict(r=>i for (i,r) in enumerate(labels))
        components[v]=[r==0 ? 0 : positions[r] for r in roots]
        representatives[v]=labels
        active=findall(!iszero,roots)
        projections[v]=sparse(components[v][active],collect(eachindex(active)),
            fill(one(K),length(active)),length(labels),length(active))
    end
    maps=Dict{Tuple{Int,Int},SparseMatrixCSC{K,Int}}()
    for (u,v) in cover_edges(P)
        reps=representatives[u]
        maps[u,v]=sparse(components[v][reps],collect(eachindex(reps)),
            fill(one(K),length(reps)),length(representatives[v]),length(reps))
    end
    return PModule{K}(P,length.(representatives),maps;field=p.field),projections
end
presentation_module(p::H0Presentation)=first(_h0_component_module(p))

function projective_resolution(p::H0Presentation,opts::ResolutionOptions=ResolutionOptions())
    if p.second===nothing
        return DerivedFunctors.projective_resolution(presentation_module(p),opts)
    end
    if opts.maxlen>=2 && p.syzygies===nothing && !isempty(p.second)
        p=h0_presentation(p.poset;vertex_grades=p.source_vertices,edges=p.source_edges,
            edge_grades=p.source_grades,field=p.field,syzygies=true)
    end
    K=coeff_type(p.field)
    d=presentation_map(p)
    M,comps=_h0_component_module(p)
    aug=PMorphism(d.cod,M,comps)
    terms=PModule{K}[d.cod]; gens=[copy(p.generators)]
    maps=PMorphism{K}[]; matrices=SparseMatrixCSC{K,Int}[]
    if opts.maxlen>=1
        push!(terms,d.dom); push!(gens,copy(p.relations)); push!(maps,d); push!(matrices,copy(p.matrix))
    end
    if opts.maxlen>=2
        P2,a2=_h0_projective(p.poset,p.second,p.field)
        a1=[findall(g->leq(p.poset,g,v),p.relations) for v in 1:nvertices(p.poset)]
        S=p.syzygies===nothing ? spzeros(K,length(p.relations),0) : p.syzygies
        d2=PMorphism(P2,d.dom,[S[a1[v],a2[v]] for v in 1:nvertices(p.poset)])
        push!(terms,P2); push!(gens,copy(p.second)); push!(maps,d2); push!(matrices,copy(S))
    end
    res=DerivedFunctors.Resolutions.ProjectiveResolution{K}(M,terms,gens,maps,matrices,aug)
    DerivedFunctors.Resolutions._pad_projective_resolution!(res,opts.maxlen)
    if opts.check
        DerivedFunctors.check_projective_resolution(res;throw=true)
        opts.minimal && DerivedFunctors.assert_minimal(res;check_cover=true)
    end
    return res
end

function betti_table(p::H0Presentation;opts::ResolutionOptions=ResolutionOptions(),pad_to::Union{Nothing,Int}=nothing)
    pad_to===nothing || pad_to>=0 || throw(ArgumentError("betti_table: pad_to must be nonnegative"))
    if p.second===nothing && opts.maxlen>=2
        return DerivedFunctors.betti_table(projective_resolution(p,opts);pad_to)
    end
    labels=p.second===nothing ? [p.generators,p.relations] : [p.generators,p.relations,p.second]
    B=zeros(Int,opts.maxlen+1,nvertices(p.poset))
    for d in 1:min(length(labels),size(B,1)),v in labels[d]; B[d,v]+=1; end
    # Match the general resolution formatter's trailing-zero and padding policy.
    return pad_to===nothing ? DerivedFunctors.Resolutions._trim_trailing_zero_rows(B) :
           DerivedFunctors.Resolutions._pad_or_truncate_rows(B,pad_to+1)
end

"""A validity flag and diagnostic issues from `check_h0_presentation`."""
struct H0PresentationValidationSummary
    valid::Bool
    issues::Vector{String}
end
Base.show(io::IO,s::H0PresentationValidationSummary)=print(io,"H0PresentationValidationSummary(valid=",s.valid,", issues=",length(s.issues),")")
Base.:(==)(a::H0PresentationValidationSummary,b::H0PresentationValidationSummary)=a.valid==b.valid && a.issues==b.issues
Base.isequal(a::H0PresentationValidationSummary,b::H0PresentationValidationSummary)=isequal(a.valid,b.valid) && isequal(a.issues,b.issues)
Base.hash(s::H0PresentationValidationSummary,h::UInt)=hash((s.valid,s.issues),h)
describe(s::H0PresentationValidationSummary)=(valid=s.valid,issues=copy(s.issues))

function _h0_components_at(P,vg,es,eg,v)
    parent=collect(eachindex(vg))
    for (i,(a,b)) in enumerate(es)
        leq(P,eg[i],v) || continue
        a,b=_h0_find!(parent,a),_h0_find!(parent,b)
        parent[b]=a
    end
    return [leq(P,g,v) ? _h0_find!(parent,i) : 0 for (i,g) in enumerate(vg)]
end

"""Validate graph presentation coefficients, minimality and the source isomorphism at every finite label."""
function check_h0_presentation(p::H0Presentation;throw::Bool=false)
    issues=String[]
    try
        P=p.poset
        _h0_validate_graph(P,p.generators,p.edges,p.relations)
        _h0_validate_graph(P,p.source_vertices,p.source_edges,p.source_grades)
        length(p.vertex_images)==length(p.source_vertices) || error("wrong source image count")
        length(p.original_vertices)==length(p.generators) || error("wrong generator source count")
        all(i->1<=i<=length(p.source_vertices),p.original_vertices) || error("invalid generator sources")
        p.source_vertices[p.original_vertices]==p.generators || error("wrong generator source grades")
        p.vertex_images[p.original_vertices]==collect(eachindex(p.generators)) || error("wrong generator source images")
        all(i->1<=i<=length(p.generators),p.vertex_images) || error("invalid source images")
        size(p.matrix)==(length(p.generators),length(p.relations)) || error("wrong matrix shape")
        K=coeff_type(p.field)
        for (j,(u,v)) in enumerate(p.edges)
            expected=zeros(K,length(p.generators)); expected[u]-=one(K); expected[v]+=one(K)
            p.matrix[:,j]==expected || error("invalid graph incidence column")
            p.generators[u]!=p.relations[j] && p.generators[v]!=p.relations[j] || error("collapsible relation")
        end
        length(_h0_minimal_edges(P,length(p.generators),p.edges,p.relations))==length(p.edges) || error("redundant relation")
        for v in 1:nvertices(P)
            a=_h0_components_at(P,p.source_vertices,p.source_edges,p.source_grades,v)
            b=_h0_components_at(P,p.generators,p.edges,p.relations,v)
            forward=Dict{Int,Int}(); backward=Dict{Int,Int}()
            for i in eachindex(a)
                a[i]==0 && continue
                z=b[p.vertex_images[i]]; z!=0 || error("source image is not active")
                get!(forward,a[i],z)==z && get!(backward,z,a[i])==a[i] || error("source map is not a component bijection")
            end
            length(backward)==length(Set(filter(!iszero,b))) || error("source map is not onto")
            if p.second!==nothing
                euler=count(g->leq(P,g,v),p.generators)-count(g->leq(P,g,v),p.relations)+count(g->leq(P,g,v),p.second)
                euler==length(backward) || error("second Betti labels violate the Hilbert identity")
            end
        end
        if p.syzygies!==nothing
            p.second!==nothing || error("syzygies need second Betti labels")
            size(p.syzygies)==(length(p.relations),length(p.second)) || error("wrong syzygy shape")
            iszero(p.matrix*p.syzygies) || error("nonzero differential composite")
            for j in axes(p.syzygies,2),k in nzrange(p.syzygies,j)
                i=p.syzygies.rowval[k]
                leq(P,p.relations[i],p.second[j]) && p.relations[i]!=p.second[j] || error("nonminimal or ungraded syzygy")
            end
            FieldLinAlg.rank(p.field,p.syzygies)==length(p.second) || error("dependent syzygies")
        end
    catch err
        err isa InterruptException && rethrow()
        push!(issues,sprint(showerror,err))
    end
    report=H0PresentationValidationSummary(isempty(issues),issues)
    throw && !report.valid && Base.throw(ArgumentError(join(issues,"; ")))
    return report
end
