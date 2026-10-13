# Paired source views share spatial axes; barcode views share diameter and row
# scales. Draft geometry never masquerades as rebuilt topology.
function _lab_axes(b,c,d)
    X=vcat(b.input.points,c.input.points,d.points)
    x=extrema(X[:,1]);y=extrema(X[:,2]);span=max(x[2]-x[1],y[2]-y[1],0.1)
    pad=0.12span
    _default_axes_2d(xlabel="Source x",ylabel="Source y",xlimits=(x[1]-pad,x[2]+pad),ylimits=(y[1]-pad,y[2]+pad))
end
function _lab_source_panel(build,draft,state,axes;edited=false,dirty=false)
    d=draft;points=Tuple.(eachrow(d.points));layers=AbstractVisualizationLayer[];active=Int[];sites=Int[]
    if !dirty
        source=build.source;G=source.complex
        for i in eachindex(G.grades)
            if G.grades[i][1]<=min(state.diameter,d.cap) && G.grades[i][2]<=state.codensity
                push!(active,i)
                if G.cell_dims[i]==0
                    push!(sites,only(source.sources[i]))
                else
                    _complex_layers!(layers,[Tuple(Float64.(q)) for q in source.geometry[i]],G.cell_dims[i],:source,0.45)
                end
            end
        end
    end
    push!(layers,PointLayer(points,_VisualRole(dirty ? :foreground : :muted),dirty ? 1. : 0.3,8.))
    dirty || push!(layers,PointLayer(points[sites],_VisualRole(:source),1.,10.))
    push!(layers,TextLayer(string.(d.ids),points,_VisualRole(:foreground),12.))
    subtitle=dirty ? "Draft points only; rebuild to replace the committed cells and barcode." :
        "Diameter $(min(state.diameter,d.cap)), codensity $(state.codensity); faint points are inactive."
    !dirty && state.diameter>d.cap && (subtitle*=" Diameter is beyond this construction cap; displayed cells stop at $(d.cap).")
    VisualizationSpec(:laboratory_source;title=edited ? (dirty ? "Edited source (draft)" : "Edited source (committed)") : "Original source",
        subtitle,layers,axes,metadata=(;source_ids=copy(d.ids),points=copy(d.points),active_cells=active,
            draft=dirty,active_source_ids=copy(d.ids[sites]),editable=edited,pick_targets=[(;id,point) for (id,point) in zip(d.ids,points)]))
end
function _lab_bar(build,diagram,state,o,title,window)
    payload=_interval_payload(diagram;dim=o.degree)
    records=Tuple(isfinite(r.death) ? r : merge(r,(;death=build.input.cap,right_closed=true,right_status=:censored,
        right_clipped=true,singleton=r.birth==build.input.cap)) for r in payload.records)
    bar,_=_interval_panels(records;window,max_intervals=o.max_points+o.max_cells,
        barcode_title=title)
    layers=copy(bar.layers);limits=bar.axes.ylimits
    push!(layers,SegmentLayer([(state.diameter,limits[1],state.diameter,limits[2])],_VisualRole(:selected),0.8,1.,:dash))
    _composition_copy(bar;layers,subtitle="H$(o.degree), $(_inspection_field_label(o.field)); fixed codensity $(state.codensity). ? marks survival at cap $(build.input.cap).",
        axes=merge(bar.axes,(;xlabel="Rips diameter",xlimits=(-0.03window[2],1.06window[2]))),
        metadata=merge(bar.metadata,(;input_version=build.input.version,construction_cap=build.input.cap,
            codensity=state.codensity,degree=o.degree,essential_interpretation=:survives_diameter_cap)))
end
function _lab_snapshot(b,c,d,state,o,diagrams)
    dirty=!_lab_same(d,c.input);axes=_lab_axes(b,c,d);window=(0.,max(b.input.cap,c.input.cap,state.diameter,0.1))
    left=_lab_source_panel(b,b.input,state,axes)
    right=_lab_source_panel(c,d,state,axes;edited=true,dirty)
    bars=[_lab_bar(b,diagrams[1],state,o,"Original barcode",window),_lab_bar(c,diagrams[2],state,o,"Committed edited barcode",window)]
    ymax=max(bars[1].axes.ylimits[2],bars[2].axes.ylimits[2])
    bars=[_composition_copy(p;axes=merge(p.axes,(;ylimits=(0.,ymax)))) for p in bars]
    lines=["$(dirty ? "Unbuilt draft" : "Committed input") $(d.version); committed input $(c.input.version), rebuild generation $(state.generation).",
        "Original: $(length(b.input.ids)) points, DTM mass $(b.input.mass), diameter cap $(b.input.cap).",
        "Draft: $(length(d.ids)) points, DTM mass $(d.mass), diameter cap $(d.cap). Committed: mass $(c.input.mass), cap $(c.input.cap).",
        "DTM uses floor(mass * point count) + 1 neighbours, including the point itself. Distances and grades use Float64.",
        "Point labels identify source inputs. Rebuilds create new cells and homology; matching labels do not identify homology classes.",
        "Bars compare the same codensity and diameter scales. No matching or correspondence between bars is asserted.",
        "Undo and reset restore a draft. Rebuild explicitly to update its mathematics. The original reference stays fixed."]
    panels=[left,right,bars...,_inspection_text_panel("Inputs and interpretation",lines)]
    VisualizationSpec(:source_laboratory;title="Source and construction sensitivity",panels,
        metadata=(;selection=state,dirty,baseline=_lab_data(b.input),current=_lab_data(c.input),draft=_lab_data(d),
            generation=state.generation,degree=o.degree,field=_inspection_field_label(o.field),
            current_grades=copy(c.source.complex.grades),baseline_grades=copy(b.source.complex.grades),
            panel_columns=2,figure_size=(1120,1300),recomputation=:full,homology_identity=:local_to_rebuild))
end
