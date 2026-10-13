# Contact sheets retain one coordinate frame and expose endpoint-map evidence.
available_visuals(::ParameterAtlas)=(:parameter_atlas,)
_visual_request_keywords(::ParameterAtlas,::Symbol)=(:source,:target,:event,:witness)
_visual_request_cost(::ParameterAtlas,::Symbol)=(;work=:retained_states_and_maps,homology=:already_computed,
    exhaustive=false,hover=:retained_state_summary,timing=:not_measured)

function _atlas_selection(a;source=1,target=nothing,event=:summary,witness=1)
    source=_atlas_index(a,source,:source)
    target===nothing || (target=_atlas_index(a,target,:target))
    event in (:summary,:entries,:kernel,:cokernel) || throw(ArgumentError("event must be :summary, :entries, :kernel or :cokernel."))
    witness isa Int && witness>=1 || throw(ArgumentError("witness must be a positive Int."))
    if event!==:summary
        target===nothing && throw(ArgumentError("An event requires a target state."))
        t=atlas_transition(a;source,target)
        t.defined || throw(ArgumentError("This pair has no map or forward entry event."))
        event===:entries && !(a.object isa DataIngestion.FilteredComplex) && throw(ArgumentError("Source-cell entries require a retained filtered complex."))
        if event in (:kernel,:cokernel)
            a.options.witnesses || throw(ArgumentError("Build the atlas with witnesses=true to select a class."))
            n=event===:kernel ? t.kernel_dimension : t.cokernel_dimension
            witness<=n || throw(ArgumentError("This transition has $n $(event) witnesses."))
        end
    end
    (;source,target,event,witness)
end
function _append_visual_request_issues!(issues::Vector{String},a::ParameterAtlas,::Symbol;kwargs...)
    try;_atlas_selection(a;kwargs...)
    catch err;err isa InterruptException && rethrow();push!(issues,sprint(showerror,err));end
    issues
end

function _atlas_matrix_panel(a,t)
    A=t.matrix;nr,nc=min(size(A,1),12),min(size(A,2),12)
    subtitle="$(_inspection_field_label(a.field)); rank $(t.rank), kernel $(t.kernel_dimension), cokernel $(t.cokernel_dimension)\nRows: target; columns: source. $(size(A,1)) \u00d7 $(size(A,2)) matrix."
    (nr<size(A,1) || nc<size(A,2)) && (subtitle*="\nShowing the first $nr rows and $nc columns; the full matrix is retained.")
    VisualizationSpec(:structure_matrix;title="State $(t.source) \u2192 $(t.target)",
        subtitle,
        layers=AbstractVisualizationLayer[MatrixLayer([_matrix_coefficient_text(A[i,j]) for i in 1:nr,j in 1:nc],
            ["e$i" for i in 1:nr],["e$j" for j in 1:nc])],
        metadata=(;panel_style=:matrix,matrix=A,matrix_size=size(A),truncated=nr<size(A,1) || nc<size(A,2),
            row_roles=fill(:target,nr),column_roles=fill(:source,nc)))
end

function _atlas_entries_panel(a,t)
    x=a.object;state=a.states[t.target];base=state.view;layers=AbstractVisualizationLayer[]
    vd=_complex_options(x;parameter=state.parameter,view_dims=a.options.view_dims).view_dims
    for r in sort(t.entries;by=r->(-r.dimension,r.index))
        r.dimension<=2 || continue
        vertices=[Tuple(Float64(p[d]) for d in vd) for p in r.geometry]
        _complex_layers!(layers,vertices,r.dimension,:selected,r.dimension==2 ? 0.5 : 1.0)
    end
    _composition_copy(base;title="Cells entering between the selected states",
        subtitle="$(length(t.entries)) source cells. Shared source coordinates; no individual birth times inferred.",layers,
        metadata=merge(base.metadata,(;entries=t.entries,pick_targets=NamedTuple[],minimal_axes=true)))
end

function _atlas_transition_metadata(t)
    merge(Base.structdiff(t,(;killed=t.killed,born=t.born)),
        (;killed=[(;cycle=DataIngestion.source_chain_data(w.cycle),filling=DataIngestion.source_chain_data(w.filling)) for w in t.killed],
          born=DataIngestion.source_chain_data.(t.born)))
end

function _visual_spec(a::ParameterAtlas,::Symbol;kwargs...)
    selection=_atlas_selection(a;kwargs...);panels=VisualizationSpec[]
    for s in a.states
        role=s.index==selection.source ? " \u00b7 source" : s.index==selection.target ? " \u00b7 target" : ""
        dim=s.dimension===nothing ? "unavailable stalk" : a.object isa DataIngestion.FilteredComplex ?
            "dim H$(a.options.degree) = $(s.dimension)" : "stalk dimension $(s.dimension)"
        detail=isempty(s.counts) ? "finite label $(s.label)" : join(["C$(i-1): $n" for (i,n) in enumerate(s.counts)]," \u00b7 ")
        get(s.view.metadata,:projection,false) && (detail*="\nCoordinate projection $(a.options.view_dims).")
        length(s.counts)>3 && (detail*="\nOnly the 0-2 skeleton is drawn; higher cells enter the homology computation.")
        isempty(get(s.view.metadata,:warnings,())) || (detail*="\nDrawing precision warning; inspect the exact retained coordinates.")
        push!(panels,_composition_copy(s.view;title="$(s.index). $(s.parameter)$role",subtitle="$dim\n$detail",
            metadata=merge(s.view.metadata,(;atlas_state=s.index))))
    end
    lines=[a.options.sampling===:sampled_grid ? "Declared grid; only its listed states are sampled." :
        a.options.sampling===:sampled_path ? "Declared monotone path; only its listed states are sampled." : "Declared states; their order need not be monotone.",
        "This is not an exhaustive event subdivision.",a.object isa DataIngestion.FilteredComplex ?
        "Each cell keeps its source ID and spatial coordinates." : "Repeated poset panels share one layout; drawing coordinates are not original parameters.",
        "Matching colors do not identify classes across parameters."]
    a.options.orientation===nothing || push!(lines,"Parameter order: "*join(["coordinate $i "*(o==1 ? "increases" : "decreases") for (i,o) in enumerate(a.options.orientation)],"; ")*".")
    t=selection.target===nothing ? nothing : atlas_transition(a;source=selection.source,target=selection.target)
    if t!==nothing && t.defined
        push!(panels,_atlas_matrix_panel(a,t))
        push!(lines,"Selected endpoint map: rank $(t.rank), kernel dimension $(t.kernel_dimension), cokernel dimension $(t.cokernel_dimension).")
        push!(lines,"Cell entries and homology changes are different quantities.")
        if !isempty(t.entries)
            push!(lines,"Newly active cells: $(length(t.entries)). Simultaneous entries are evaluated together.")
            for r in Iterators.take(t.entries,6)
                push!(lines,"Cell $(r.index), dimension $(r.dimension), grades $(r.grades).")
            end
            length(t.entries)>6 && push!(lines,"Further entries are retained in atlas_transition(...).entries.")
        end
        if selection.event===:entries
            push!(panels,_atlas_entries_panel(a,t))
        elseif selection.event in (:kernel,:cokernel)
            n=selection.witness;kernel=selection.event===:kernel
            vector=kernel ? t.kernel[:,n] : t.cokernel[:,n]
            push!(lines,"Chosen $(selection.event) vector $n: $(Tuple(vector)).")
            push!(lines,"Representatives depend on the chosen quotient basis; no minimal-support claim.")
            if a.object isa DataIngestion.FilteredComplex
                chains=kernel ? [t.killed[n].cycle,t.killed[n].filling] : [t.born[n]]
                titles=kernel ? ["A class in the kernel","Its filling at the target"] : ["A target class outside the image"]
                for (c,title) in zip(chains,titles)
                    if c.dimension<=2
                        chart=first(visual_spec(c;view_dims=a.options.view_dims).panels)
                        push!(panels,_composition_copy(chart;title,axes=a.states[selection.target].view.axes))
                    else
                        push!(lines,"Degree $(c.dimension) witness retained as coefficients; source overlays display degrees 0\u20132.")
                    end
                end
            end
        end
    elseif t!==nothing
        push!(lines,t.relation===:outside ? "A classifier returned label 0: the stalk is unavailable." :
            t.relation===:reverse_comparable ? "The requested direction reverses the parameter order." : "The selected parameters are incomparable.")
        push!(lines,"No map is defined for this selection.")
    else
        push!(lines,"Select a source and target to inspect their actual map.")
    end
    push!(panels,_inspection_text_panel("What changes between states?",lines))
    columns=a.options.grid_shape===nothing ? min(3,length(a.states)) : a.options.grid_shape[2]
    state_rows=cld(length(a.states),columns)
    detail_count=length(panels)-length(a.states)-1
    detail_columns=max(1,detail_count)
    layout_columns=lcm(columns,detail_columns)
    state_span=div(layout_columns,columns);detail_span=div(layout_columns,detail_columns)
    positions=Tuple{UnitRange{Int},UnitRange{Int}}[]
    for i in eachindex(a.states)
        row=div(i-1,columns)+1;col=mod(i-1,columns)*state_span+1
        push!(positions,(row:row,col:col+state_span-1))
    end
    for i in 1:detail_count
        col=(i-1)*detail_span+1
        push!(positions,(state_rows+1:state_rows+1,col:col+detail_span-1))
    end
    last_row=state_rows+(detail_count>0 ? 1 : 0)+1
    push!(positions,(last_row:last_row,1:layout_columns))
    weights=vcat(ones(state_rows),detail_count==0 ? Float64[] : [detail_count==1 ? 0.65 : 1.0],
        [max(0.5,(length(lines)*34+110)/360)])
    VisualizationSpec(:parameter_atlas;title="A parameter atlas",subtitle="Aligned states and actual transition maps \u00b7 $(_inspection_field_label(a.field))",
        panels,metadata=(;states=atlas_states(a),selection,transition=t===nothing ? nothing : _atlas_transition_metadata(t),
            sampling=a.options.sampling,grid_shape=a.options.grid_shape,orientation=a.options.orientation,exhaustive=false,degree=a.options.degree,field=a.field,
            panel_columns=columns,panel_positions=positions,panel_row_weights=weights,
            figure_size=(max(900,max(columns,detail_columns)*420),ceil(Int,sum(weights)*360)+140),html_mode=:responsive_fixed_svg,
            state_panel_count=length(a.states)))
end
