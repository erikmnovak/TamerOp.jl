# The editor owns an input draft, not an unchecked mathematical module.
"""
    ModuleEditorSession

A bounded mutable input draft from `module_editor`. Mathematical mistakes remain
visible and undoable; `editor_object` and `editor_code` require a valid draft.
Use `editor_draft` for an independent copy, `check_module_editor` for the
validation report, and the shared inspection selection/snapshot/close methods.
No caller-owned module or presentation is modified. Live editing needs Julia.
"""
mutable struct ModuleEditorSession <: _AbstractInspectionSession
    draft::Union{Nothing,NamedTuple}
    initial::Union{Nothing,NamedTuple}
    validation::NamedTuple
    state::NamedTuple
    snapshot::VisualizationSpec
    history::Vector{NamedTuple}
    future::Vector{NamedTuple}
    history_limit::Int
    limits::NamedTuple
    listeners::Dict{Int,Any}
    next_listener::Int
    listener_errors::Vector{String}
    closed::Bool
    busy::Bool
    lock::ReentrantLock
end

function _editor_limits(max_vertices,max_dimension,max_entries)
    (;max_vertices=_inspection_integer(max_vertices,:max_vertices;lower=1),
        max_dimension=_inspection_integer(max_dimension,:max_dimension;lower=1),
        max_entries=_inspection_integer(max_entries,:max_entries;lower=1))
end
function _editor_matrix(A,limits)
    A isa AbstractMatrix || throw(ArgumentError("A coefficient matrix must have two dimensions."))
    maximum(size(A);init=0)<=limits.max_dimension || throw(ArgumentError("Matrix axes exceed max_dimension."))
    length(A)<=limits.max_entries || throw(ArgumentError("Matrix exceeds max_entries."))
    all(x->x isa Union{Integer,Rational,AbstractFloat,CoreModules.FpElem},A) ||
        throw(ArgumentError("Matrix entries must be numerical coefficients."))
    return copy(Matrix(A))
end
function _editor_normalize(d,limits)
    d.kind in (:module,:presentation) || throw(ArgumentError("Editor kind must be :module or :presentation."))
    d.field isa Union{CoreModules.QQField,CoreModules.PrimeField,CoreModules.RealField} ||
        throw(ArgumentError("The editor supports rational, prime and real coefficient fields."))
    d.order isa AbstractMatrix && all(x->x in (0,1),d.order) ||
        throw(ArgumentError("order must be a zero/one relation matrix, including its diagonal."))
    maximum(size(d.order);init=0)<=limits.max_vertices || throw(ArgumentError("Order exceeds max_vertices."))
    order=BitMatrix(d.order)
    if d.kind===:module
        d.dimensions isa AbstractVector && all(x->x isa Integer && !(x isa Bool),d.dimensions) ||
            throw(ArgumentError("dimensions must be an integer vector."))
        length(d.dimensions)<=limits.max_vertices && all(x->abs(big(x))<=limits.max_dimension,d.dimensions) ||
            throw(ArgumentError("Stalk dimensions exceed the editor budget."))
        d.maps isa AbstractDict || throw(ArgumentError("maps must assign matrices to (source,target) pairs."))
        length(d.maps)<=limits.max_vertices^2 || throw(ArgumentError("Too many edge maps."))
        maps=Dict{Tuple{Int,Int},Any}()
        for (edge,A) in d.maps
            edge isa Tuple && length(edge)==2 && all(x->x isa Integer && !(x isa Bool) && 1<=x<=limits.max_vertices,edge) ||
                throw(ArgumentError("Map keys must be pairs of bounded positive vertex IDs."))
            maps[Int.(edge)]=_editor_matrix(A,limits)
        end
        sum(length,values(maps);init=0)<=limits.max_entries || throw(ArgumentError("All maps together exceed max_entries."))
        return (;kind=d.kind,order,dimensions=Int.(d.dimensions),maps,field=d.field)
    end
    supports=map((d.upsets,d.downsets)) do family
        family isa AbstractVector && length(family)<=limits.max_dimension ||
            throw(ArgumentError("Support lists exceed max_dimension."))
        [begin
            ids isa AbstractVector && length(ids)<=limits.max_vertices &&
                all(x->x isa Integer && !(x isa Bool) && abs(big(x))<=limits.max_vertices,ids) ||
                throw(ArgumentError("Each support is a bounded vector of integer vertex IDs."))
            Int.(ids)
        end for ids in family]
    end
    (;kind=d.kind,order,upsets=supports[1],downsets=supports[2],matrix=_editor_matrix(d.matrix,limits),field=d.field)
end
_editor_coefficient(field,x)=CoreModules.coerce(field,x isa CoreModules.FpElem ? x.val : x)
function _editor_coefficients(field,A)
    result=map(x->_editor_coefficient(field,x),A)
    K=CoreModules.coeff_type(field)
    result=Matrix{K}(result)
    all(isfinite,result) || throw(ArgumentError("Coefficients must be finite."))
    result
end
function _editor_evaluate(d)
    issues=String[];P=nothing;object=nothing;witness=nothing
    try
        size(d.order,1)==size(d.order,2) || throw(ArgumentError("The order matrix must be square."))
        P=FiniteFringe.FinitePoset(copy(d.order);check=false)
        append!(issues,FiniteFringe.check_poset(P).issues)
        if isempty(issues) && d.kind===:module
            n=nvertices(P)
            length(d.dimensions)==n || push!(issues,"One stalk dimension is required per vertex.")
            all(>=(0),d.dimensions) || push!(issues,"Stalk dimensions must be nonnegative.")
            expected=Set(FiniteFringe.cover_edges(P));actual=Set(keys(d.maps))
            missing=setdiff(expected,actual);extra=setdiff(actual,expected)
            isempty(missing) || push!(issues,"Missing cover matrices: $(sort!(collect(missing))).")
            isempty(extra) || push!(issues,"Maps must correspond exactly to covers; extra entries: $(sort!(collect(extra))).")
            if isempty(issues)
                K=CoreModules.coeff_type(d.field)
                maps=Dict{Tuple{Int,Int},Matrix{K}}(e=>_editor_coefficients(d.field,A) for (e,A) in d.maps)
                for ((u,v),A) in maps
                    size(A)==(d.dimensions[v],d.dimensions[u]) || push!(issues,
                        "Map $u -> $v has size $(size(A)); expected $((d.dimensions[v],d.dimensions[u])).")
                end
                if isempty(issues)
                    candidate=Modules.PModule{CoreModules.coeff_type(d.field)}(P,copy(d.dimensions),maps;field=d.field)
                    witness=Modules.module_path_witness(candidate)
                    if witness===nothing
                        object=candidate
                    else
                        push!(issues,"Paths $(join(witness.first_path," -> ")) and $(join(witness.second_path," -> ")) have different composites.")
                    end
                end
            end
        elseif isempty(issues)
            n=nvertices(P)
            for (name,family) in (("upset",d.upsets),("downset",d.downsets)), (i,ids) in enumerate(family)
                length(unique(ids))==length(ids) || push!(issues,"$name $i repeats a vertex.")
                all(v->1<=v<=n,ids) || push!(issues,"$name $i contains a vertex outside 1:$n.")
            end
            if isempty(issues)
                U=[FiniteFringe.Upset(P,BitVector([v in ids for v in 1:n])) for ids in d.upsets]
                D=[FiniteFringe.Downset(P,BitVector([v in ids for v in 1:n])) for ids in d.downsets]
                A=_editor_coefficients(d.field,d.matrix)
                report=FiniteFringe.check_fringe_data(P,U,D,A;field=d.field)
                append!(issues,report.issues)
                isempty(issues) && (object=FiniteFringe.FringeModule{CoreModules.coeff_type(d.field)}(P,U,D,A;field=d.field))
            end
        end
    catch err
        err isa InterruptException && rethrow()
        push!(issues,sprint(showerror,err))
    end
    report=(;kind=:module_editor,valid=isempty(issues),issues,witness,field=d.field,input_kind=d.kind)
    return (;report,poset=P===nothing || !FiniteFringe.check_poset(P).valid ? nothing : P,object)
end

function _editor_snapshot(d,e,state)
    report=e.report;panels=VisualizationSpec[]
    if e.poset!==nothing
        dims=d.kind===:module && length(d.dimensions)==nvertices(e.poset) && all(>=(0),d.dimensions) ? d.dimensions : nothing
        graph=_hasse_spec(e.poset;dims,vertex=state.vertex,field_label=_inspection_field_label(d.field))
        if report.witness!==nothing
            layers=copy(graph.layers);w=report.witness
            for (path,color,linestyle) in ((w.first_path,:steelblue,:solid),(w.second_path,:darkorange,:dash))
                xy=graph.metadata.positions
                segments=NTuple{4,Float64}[(xy[u]...,xy[v]...) for (u,v) in zip(path[1:end-1],path[2:end])]
                push!(layers,SegmentLayer(segments,color,1.0,4.0,linestyle))
            end
            graph=VisualizationSpec(graph.kind;title="Conflicting cover paths",subtitle="Blue: first path. Dashed orange: second path.",
                layers,axes=graph.axes,legend=graph.legend,metadata=graph.metadata)
        end
        push!(panels,graph)
    else
        push!(panels,_presentation_matrix_panel(Int.(d.order),"Draft order (not a poset)",
            string.(1:size(d.order,1)),string.(1:size(d.order,2)),(12,12)))
    end
    lines=report.valid ? ["All defining relations passed validation.","Field: $(_inspection_field_label(d.field)).",
        "Export produces an independent mathematical object."] : ["Invalid draft: object and code export are disabled.";report.issues]
    push!(panels,_inspection_text_panel(report.valid ? "Valid $(d.kind)" : "Repair the draft",lines))
    if report.witness!==nothing
        w=report.witness
        push!(panels,_algebra_matrix_panel(w.first_composite,"First composite",d.field,(12,12);subtitle=join(w.first_path," -> ")))
        push!(panels,_algebra_matrix_panel(w.second_composite,"Second composite",d.field,(12,12);subtitle=join(w.second_path," -> ")))
    elseif d.kind===:module && state.edge!==nothing && haskey(d.maps,state.edge)
        A=d.maps[state.edge]
        push!(panels,_presentation_matrix_panel(A,"Entered cover matrix $(state.edge)",string.(1:size(A,1)),string.(1:size(A,2)),(12,12)))
    elseif d.kind===:presentation
        A=d.matrix
        push!(panels,_presentation_matrix_panel(A,"Entered fringe coefficients",string.(1:size(A,1)),string.(1:size(A,2)),(12,12)))
        if report.valid && state.vertex!==nothing
            stalk=IndicatorResolutions.presentation_stalk(e.object;vertex=state.vertex)
            push!(panels,_inspection_text_panel("Image at vertex $(state.vertex)",
                ["Dimension: $(IndicatorResolutions.presentation_summary(stalk).dimension)",
                 "Active rows: $(IndicatorResolutions.active_rows(stalk))", "Active columns: $(IndicatorResolutions.active_columns(stalk))"]))
        end
    end
    positions=[(1:1,1:1),(1:1,2:2)]
    length(panels)>2 && push!(positions,(2:2,1:1))
    length(panels)>3 && push!(positions,(2:2,2:2))
    weights=length(panels)>2 ? [390,270] : [390]
    VisualizationSpec(:module_editor;title="Author a finite $(d.kind)",panels,
        metadata=(;editor_validation=report,editor_draft=deepcopy(d),selection=state,category=:finite_poset_representations,
            panel_columns=2,panel_positions=positions,panel_row_weights=weights,figure_size=(1160,sum(weights)+90)))
end

"""
    module_editor(M; history_limit=32, max_vertices=32, max_dimension=32, max_entries=20000)
    module_editor(; order, dimensions, maps, field=QQField(), ...)
    module_editor(; order, upsets, downsets, matrix, field=QQField(), ...)

Open an independent draft of a finite PModule or FringeModule, or enter raw
spaces/maps or fringe data. `order[i,j]` means i <= j and includes the diagonal;
`maps[(i,j)]` is a target-by-source matrix for each cover, exactly once. Fringe
supports are lists of vertex IDs, not automatically completed closures.

Invalid mathematical drafts are retained with explicit diagnostics. Malformed
input types and budget violations throw transactionally. Field edits explicitly
reinterpret coefficient representatives and revalidate; they are not an
assertion of invariance under coefficient change. Matrices are never resized
or repaired implicitly. Graph drawing is schematic, not an ambient encoding.
"""
function module_editor(;order,dimensions=nothing,maps=nothing,upsets=nothing,downsets=nothing,matrix=nothing,
                        field=CoreModules.QQField(),history_limit=32,max_vertices=32,max_dimension=32,max_entries=20000)
    limits=_editor_limits(max_vertices,max_dimension,max_entries)
    history_limit=_inspection_integer(history_limit,:history_limit)
    (dimensions!==nothing || maps!==nothing) && (upsets!==nothing || downsets!==nothing || matrix!==nothing) &&
        throw(ArgumentError("Choose spaces/maps or a fringe presentation."))
    raw=dimensions!==nothing || maps!==nothing ? (;kind=:module,order,dimensions,maps,field) :
        (;kind=:presentation,order,upsets,downsets,matrix,field)
    draft=_editor_normalize(raw,limits);e=_editor_evaluate(draft)
    n=size(draft.order,1)
    edge=draft.kind===:module && !isempty(draft.maps) ? first(sort!(collect(keys(draft.maps)))) : nothing
    state=(;vertex=e.poset===nothing || n==0 ? nothing : 1,edge,revision=0)
    snapshot=_editor_snapshot(draft,e,state)
    ModuleEditorSession(draft,deepcopy(draft),e.report,state,snapshot,NamedTuple[],NamedTuple[],history_limit,limits,
        Dict{Int,Any}(),1,String[],false,false,ReentrantLock())
end
function module_editor(M::Modules.PModule;kwargs...)
    P=_inspection_poset(M)
    module_editor(;order=[leq(P,i,j) for i in 1:nvertices(P),j in 1:nvertices(P)],dimensions=copy(_inspection_dimensions(M)),
        maps=Dict(e=>copy(Modules.structure_map(M;source=e[1],target=e[2])) for e in FiniteFringe.cover_edges(P)),field=_inspection_field(M),kwargs...)
end
function module_editor(H::FiniteFringe.FringeModule;kwargs...)
    P=FiniteFringe.ambient_poset(H);n=nvertices(P)
    module_editor(;order=[leq(P,i,j) for i in 1:n,j in 1:n],field=FiniteFringe.field(H),
        upsets=[[q for q in 1:n if FiniteFringe.contains(U,q)] for U in FiniteFringe.birth_upsets(H)],
        downsets=[[q for q in 1:n if FiniteFringe.contains(D,q)] for D in FiniteFringe.death_downsets(H)],
        matrix=copy(FiniteFringe.fringe_coefficients(H)),kwargs...)
end
"""Return an independent copy of the current input draft, including invalid input."""
editor_draft(s::ModuleEditorSession)=lock(s.lock) do;_inspection_require_open(s);deepcopy(s.draft);end
inspection_selection(s::ModuleEditorSession)=lock(s.lock) do;s.state;end
inspection_snapshot(s::ModuleEditorSession)=lock(s.lock) do;s.snapshot;end
function inspection_summary(s::ModuleEditorSession)
    lock(s.lock) do
        (;kind=:module_editor,valid=s.validation.valid,input_kind=s.validation.input_kind,
            field=s.validation.field,selection=s.state,revision=s.state.revision,closed=s.closed,updating=s.busy,
            undo_count=length(s.history),redo_count=length(s.future),history_limit=s.history_limit,
            listener_count=length(s.listeners),listener_errors=Tuple(s.listener_errors),limits=s.limits)
    end
end
describe(s::ModuleEditorSession)=inspection_summary(s)
Base.show(io::IO,s::ModuleEditorSession)=print(io,"ModuleEditorSession(",s.validation.input_kind,", valid=",s.validation.valid,", revision=",s.state.revision,", closed=",s.closed,")")
"""Return validation issues and a conflicting-path witness. Wrap the report in `VisualizationValidationSummary` for compact display."""
function check_module_editor(s::ModuleEditorSession;throw=false)
    lock(s.lock) do
        report=deepcopy(s.validation)
        throw && !report.valid && _throw_invalid_visual(:check_module_editor,report.issues)
        report
    end
end
"""Construct an independent ordinary PModule or FringeModule from a valid open draft."""
function editor_object(s::ModuleEditorSession)
    lock(s.lock) do
        _inspection_require_open(s)
        e=_editor_evaluate(s.draft)
        e.report.valid || throw(ArgumentError("Repair the draft before exporting an object: "*join(e.report.issues,"; ")))
        e.object
    end
end
function _editor_transition!(build::Function,s,mode)
    lock(s.lock)
    try
        _inspection_require_open(s);s.busy && throw(ArgumentError("An editor update or callback is already in progress."))
        draft,state=build()
        draft=_editor_normalize(draft,s.limits);e=_editor_evaluate(draft)
        vertex=state.vertex
        (e.poset===nothing || vertex!==nothing && vertex>nvertices(e.poset)) && (vertex=nothing)
        edge=state.edge
        draft.kind===:module && edge!==nothing && !haskey(draft.maps,edge) && (edge=nothing)
        state=(;vertex,edge,revision=s.state.revision+1)
        snapshot=_editor_snapshot(draft,e,state)
        if mode===:edit
            s.history_limit>0 && push!(s.history,(;draft=deepcopy(s.draft),selection=s.state))
            length(s.history)>s.history_limit && popfirst!(s.history)
            empty!(s.future)
        elseif mode===:undo
            pop!(s.history);push!(s.future,(;draft=deepcopy(s.draft),selection=s.state))
        elseif mode===:redo
            pop!(s.future);push!(s.history,(;draft=deepcopy(s.draft),selection=s.state))
        end
        s.draft=draft;s.validation=e.report;s.state=state;s.snapshot=snapshot;s.busy=true
    finally;unlock(s.lock);end
    try;_inspection_notify(s)
    finally;lock(s.lock) do;s.busy=false;end;end
    s
end
"""
    edit_module!(editor; order, dimensions, maps, field, upsets, downsets, matrix)
    edit_module!(editor; map=((source,target), A))

Replace supplied parts of the draft as one undoable edit. Unspecified data stay
unchanged. `map=(edge,nothing)` removes an entered cover map. Invalid relations
remain visible and block object/code export; syntax/type/budget errors preserve
the previous draft. Changing a stalk does not resize any adjacent matrix.
"""
function edit_module!(s::ModuleEditorSession;kwargs...)
    _editor_transition!(s,:edit) do
        d=s.draft
        allowed=d.kind===:module ? (:order,:dimensions,:maps,:map,:field) : (:order,:upsets,:downsets,:matrix,:field)
        all(k->k in allowed,keys(kwargs)) || throw(ArgumentError("Unsupported edit for $(d.kind)."))
        haskey(kwargs,:map) && haskey(kwargs,:maps) && throw(ArgumentError("Supply map or maps, not both."))
        updates=(;[k=>v for (k,v) in kwargs if k!==:map]...)
        next=merge(d,updates)
        if haskey(kwargs,:map)
            change=kwargs[:map]
            change isa Tuple && length(change)==2 || throw(ArgumentError("map must be (edge,matrix) or (edge,nothing)."))
            edge,A=change;maps=copy(d.maps)
            edge isa Tuple && length(edge)==2 && all(x->x isa Integer && !(x isa Bool) && 1<=x<=s.limits.max_vertices,edge) ||
                throw(ArgumentError("Map keys must be pairs of bounded positive vertex IDs."))
            A===nothing ? delete!(maps,edge) : (maps[edge]=A)
            next=merge(next,(;maps))
        end
        next,s.state
    end
end
"""Undo one draft edit. Selection-only navigation is not added to the edit history."""
function undo_module_edit!(s::ModuleEditorSession)
    _editor_transition!(s,:undo) do
        isempty(s.history) && throw(ArgumentError("No edit remains to undo."))
        last(s.history).draft,last(s.history).selection
    end
end
"""Redo one undone edit; a new edit discards the redo branch."""
function redo_module_edit!(s::ModuleEditorSession)
    _editor_transition!(s,:redo) do
        isempty(s.future) && throw(ArgumentError("No edit remains to redo."))
        last(s.future).draft,last(s.future).selection
    end
end
function select_inspection!(s::ModuleEditorSession;vertex=s.state.vertex,edge=s.state.edge)
    _editor_transition!(s,:select) do
        vertex===nothing || (vertex=_inspection_integer(vertex,:vertex;lower=1))
        vertex===nothing || vertex<=size(s.draft.order,1) || throw(ArgumentError("Vertex is outside the draft."))
        edge===nothing || s.draft.kind===:module && haskey(s.draft.maps,edge) || throw(ArgumentError("Select an entered cover matrix."))
        s.draft,(;vertex,edge)
    end
end
reset_inspection!(s::ModuleEditorSession)=_editor_transition!(() -> (s.initial,(;vertex=nothing,edge=nothing)),s,:edit)
function close_inspection!(s::ModuleEditorSession)
    lock(s.lock)
    try
        s.closed && return s
        s.busy && throw(ArgumentError("Cannot close an editor during an update."))
        s.closed=true;s.busy=true
    finally;unlock(s.lock);end
    try;_inspection_notify(s)
    finally
        lock(s.lock) do
            empty!(s.listeners);empty!(s.history);empty!(s.future)
            s.draft=nothing;s.initial=nothing;s.busy=false
        end
    end
    s
end
function check_inspection_session(s::ModuleEditorSession;throw=false)
    lock(s.lock) do
        issues=copy(check_visual_spec(s.snapshot).issues)
        s.snapshot.metadata.selection==s.state || push!(issues,"Selection and snapshot differ.")
        length(s.history)<=s.history_limit || push!(issues,"Undo history exceeds its budget.")
        s.closed && (s.draft!==nothing || !isempty(s.listeners) || !isempty(s.history) || !isempty(s.future)) && push!(issues,"Closed editor retains runtime state.")
        valid=isempty(issues)
        throw && !valid && _throw_invalid_visual(:check_inspection_session,issues)
        (;kind=:inspection_session,valid,issues,closed=s.closed,revision=s.state.revision)
    end
end
function _visual_spec(s::ModuleEditorSession,::Symbol;kwargs...)
    isempty(kwargs) || throw(ArgumentError("Edit with edit_module! and select with select_inspection!."))
    VisualizationSpec(:linked_inspector;title="Edit a finite mathematical diagram",panels=inspection_snapshot(s).panels,
        interaction=(;hover=false,clicks=true,labels=true,widgets=(:editor_draft,:editor_history,:editor_export),
            notebook=:linked_inspector,mode=:live_julia,requires_live_julia=true,offline_widgets=false),
        metadata=(;session=s,inspector_kind=:module_editor))
end
_visual_request_cost(::ModuleEditorSession,::Symbol)=(;work=:bounded_draft_validation,requires_live_julia=true,timing=:not_measured)

# Shared literal-only input contract for text controls. Rows are separated by
# semicolons; optional explicit shape preserves matrices with an empty axis.
function _editor_parse_matrix(text::AbstractString;shape=nothing,max_entries=20000,max_dimension=max_entries)
    ncodeunits(text)<=1024*1024 || throw(ArgumentError("Matrix text exceeds 1 MiB."))
    rows=isempty(strip(text)) ? Vector{SubString{String}}[] : [split(strip(row),r"[ ,]+";keepempty=false) for row in split(String(text),';')]
    if shape===nothing
        nr=length(rows);nc=isempty(rows) ? 0 : length(first(rows))
    else
        nr,nc=shape
        nr isa Integer && nc isa Integer && nr>=0 && nc>=0 || throw(ArgumentError("Matrix shape must be nonnegative."))
    end
    big(nr)*nc<=max_entries || throw(ArgumentError("Matrix text exceeds max_entries."))
    max(nr,nc)<=max_dimension || throw(ArgumentError("Matrix axes exceed max_dimension."))
    if nr*nc==0
        isempty(strip(text)) || throw(ArgumentError("An empty-axis matrix must have empty coefficient text."))
        return Matrix{CoreModules.QQ}(undef,nr,nc)
    end
    length(rows)==nr && all(r->length(r)==nc,rows) || throw(ArgumentError("Rows must match the requested rectangular shape."))
    A=Matrix{CoreModules.QQ}(undef,nr,nc)
    for i in 1:nr,j in 1:nc
        A[i,j]=_parse_inspection_coordinate(rows[i][j])
    end
    A
end
_editor_matrix_text(A)=join((join((_matrix_coefficient_text(A[i,j]) for j in axes(A,2))," ") for i in axes(A,1)),"; ")
