# Selected columns, their images, and optional classifier fibers.
available_visuals(::AbelianCategories.VectorPropagation) = (:vector_persistence,)
_visual_request_keywords(::AbelianCategories.VectorPropagation,::Symbol) =
    (:target,:box,:matrix_limit,:max_vertices)
_visual_request_cost(::AbelianCategories.VectorPropagation,::Symbol) =
    (;work=:retained_selected_images,selected_map=:computed_only_for_target,
      generated_submodule=:explicit_separate_call,timing=:not_measured)
function _append_visual_request_issues!(issues::Vector{String},r::AbelianCategories.VectorPropagation,::Symbol;kwargs...)
    _algebra_selection_issues!(issues,r.ambient.Q;vertex=get(kwargs,:target,nothing),
        matrix_limit=get(kwargs,:matrix_limit,(12,12)))
    budget=get(kwargs,:max_vertices,200)
    budget isa Integer && !(budget isa Bool) && budget>0 || push!(issues,"max_vertices must be a positive integer.")
    budget isa Integer && nvertices(r.ambient.Q)>budget && push!(issues,"Finite graph exceeds max_vertices; increase the explicit budget.")
    enc=r.context.encoding
    box=get(kwargs,:box,nothing)
    box===nothing || (enc!==nothing && _inspection_has_geometry(enc)) || push!(issues,"box requires a retained planar encoding.")
    box===nothing || _visual_box_2d(box)
    return issues
end

"""
    visual_spec(r::VectorPropagation; target=nothing, box=nothing,
                matrix_limit=(12,12), max_vertices=200)

Display image dimensions of the selected columns on the finite order and, when
retained, actual planar classifier fibers. Colors distinguish no forward map,
zero image, and unrepresented parameters. Fibers show pullback of the finite
generated submodule; they do not assert an ambient lifetime from a chosen point.
Select a target label for the actual full map and propagated columns. Cropping
only affects matrix display. Submodules and quotients are separate opt-in calls.
"""
function _visual_spec(r::AbelianCategories.VectorPropagation,::Symbol;
        target=nothing,box=nothing,matrix_limit=(12,12),max_vertices=200)
    AbelianCategories.check_propagation(r;throw=true)
    M=r.ambient;p=r.source;ds=r.image_dimensions
    maximum_rank=maximum((d===nothing ? 0 : d for d in ds);init=0)
    graph=_hasse_spec(M.Q;vertex=p,pair=target===nothing ? nothing : (p,target))
    layers=AbstractVisualizationLayer[]
    for layer in graph.layers
        if layer isa PointLayer && layer.color isa _VisualRole && layer.color.kind===:categorical
            d=ds[layer.color.index]
            color=d===nothing ? _VisualRole(:absent) : _rank_section_color(d,maximum_rank)
            push!(layers,PointLayer(layer.points,color,layer.alpha,layer.markersize))
        elseif layer isa TextLayer && length(layer.labels)==length(ds)
            labels=["$q: "*(ds[q]===nothing ? "no map" : string(ds[q])) for q in eachindex(ds)]
            push!(layers,TextLayer(labels,layer.positions,layer.color,11.0))
        else
            push!(layers,layer)
        end
    end
    entries=NamedTuple[(;label="image dimension $d",color=_rank_section_color(d,maximum_rank),style=:patch)
        for d in sort!(unique(Int[d for d in ds if d!==nothing]))]
    push!(entries,(;label="no forward map",color=_VisualRole(:absent),style=:patch))
    panels=VisualizationSpec[VisualizationSpec(:propagation_hasse;title="Images from source $p",
        subtitle="Schematic order; labels: vertex: image dimension",layers,axes=graph.axes,
        legend=_default_legend(;visible=true,entries),metadata=graph.metadata)]
    enc=r.context.encoding
    if enc!==nothing && _inspection_has_geometry(enc)
        pi=_inspection_classifier(encoding_map(enc));geo=_region_geometry_2d(pi;box)
        colors=Dict(q => q==0 ? _VisualRole(:unrepresented) : ds[q]===nothing ? _VisualRole(:absent) :
            _rank_section_color(ds[q],maximum_rank) for q in geo.region_ids)
        base=_visual_spec(pi,:region_labels;box,prepared_geometry=geo)
        regionlayers=_region_geometry_layers(geo;colors)
        for layer in base.layers
            layer isa TextLayer || continue
            keep=findall(label->!startswith(label,"outside"),layer.labels)
            push!(regionlayers,TextLayer(layer.labels[keep],layer.positions[keep],layer.color,layer.textsize))
        end
        subtitle="Finite-label selection; labels identify fibers\nSolid/included, dashed/excluded, dotted/window cut"
        geo.geometry_kind===:nearest_lattice_tiles && (subtitle*="\nNearest-lattice tiles; ties round-to-even")
        isempty(base.metadata.warnings) || (subtitle*="\nDrawing precision warning: inspect exact geometry metadata")
        push!(panels,VisualizationSpec(:propagation_regions;title="Pullback to parameter regions",
            subtitle,
            layers=regionlayers,axes=geo.axes,legend=_default_legend(;visible=true,
                entries=[entries;[(;label="unrepresented",color=_VisualRole(:unrepresented),style=:patch)]]),
            metadata=(;geometry=geo,warnings=base.metadata.warnings,scope=:finite_submodule_pullback)))
    end
    spatial_panels=length(panels)
    push!(panels,_algebra_matrix_panel(r.vectors,"Chosen columns at $p",M.field,matrix_limit;
        subtitle="Ambient stalk coordinates; columns need not be independent"))
    selected=nothing
    if target!==nothing
        A=AbelianCategories.propagation_image(r;target)
        if A!==nothing
            fullmap=copy(Modules.structure_map(M;source=p,target))
            push!(panels,_algebra_matrix_panel(fullmap,"Structure map $p -> $target",M.field,matrix_limit))
            push!(panels,_algebra_matrix_panel(A,"Images of the chosen columns",M.field,matrix_limit;
                subtitle="Image dimension $(ds[target]); zero columns are retained"))
            selected=(;target,defined=true,matrix=fullmap,images=A,dimension=ds[target])
        else
            selected=(;target,defined=false,matrix=nothing,images=nothing,dimension=nothing)
        end
    end
    lines=["Field: $(_inspection_field_label(M.field)). Source: finite label $p.",
        "Image ranks refer to the chosen columns, not the full structure map.",
        "No map is different from a defined map whose image is zero.",
        "The generated submodule is not certified to be a direct summand."]
    selected===nothing || selected.defined || push!(lines,"Selected target $target has no forward map from $p.")
    enc===nothing || push!(lines,"Region colors pull back the finite submodule; no ambient source point was chosen.")
    push!(panels,_inspection_text_panel("Interpreting the selection",lines))
    matrix_panels=length(panels)-spatial_panels-1
    positions=spatial_panels==2 ? [(1:1,1:3),(1:1,4:6)] : [(1:1,1:6)]
    if matrix_panels==3
        append!(positions,[(2:2,1:2),(2:2,3:4),(2:2,5:6),(3:3,1:6)])
        weights=[400,240,180];size=(1400,940)
    else
        append!(positions,[(2:2,1:2),(2:2,3:6)])
        weights=[400,250];size=(1400,770)
    end
    return VisualizationSpec(:vector_persistence;title="Following selected vectors",panels,
        metadata=(;field=M.field,source=p,selected,image_dimensions=copy(ds),
            nonzero_support=AbelianCategories.propagation_summary(r).nonzero_support,
            basis_convention=r.context.basis_convention,summand=:not_certified,
            panel_columns=6,panel_positions=positions,panel_row_weights=weights,
            figure_size=size))
end
