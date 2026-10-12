# A figure-local owner of drawing buffers. No process-wide scene cache and no
# mathematical-result cache is introduced. Structural changes return `nothing`
# before mutating the displayed figure, so callers can replace it offscreen.
mutable struct _VisualRetainedFigure
    figure::Any
    spec::Any
    panels::Vector{Any}
    headings::Vector{Any}
end

_visual_command_shape(c) = (c.kind, length(c.args), keys(c.kwargs))
function _visual_panel_shape(V, p)
    (;kind=p.kind, axis3=get(p.metadata,:source_view_dimension,0)==3,
      title=!isempty(p.title), subtitle=!isempty(p.subtitle), legend=p.legend,
      axiskeys=keys(p.axes), minimal=get(p.metadata,:minimal_axes,false),
      hide=get(p.metadata,:hide_decorations,false),
      legend_position=get(p.metadata,:legend_position,:bottom),
      plane=get(p.metadata,:plane,nothing),field_id=get(p.metadata,:field_id,nothing))
end

function _visual_retained_plan(H,V,spec,style)
    panels=isempty(spec.panels) ? [spec] : spec.panels
    # Literal coefficient tables and composed figure anchors own a different
    # layout contract; their existing renderer remains the structural fallback.
    any(p->!isempty(p.panels)||get(p.metadata,:panel_style,nothing) in (:matrix,:text_only)||
        any(l->l isa V.MatrixLayer,p.layers),panels) && return nothing
    (spec.kind===:composition || haskey(spec.metadata,:panel_positions) || haskey(spec.metadata,:panel_row_weights) || get(spec.legend,:visible,false)) && return nothing
    return [H.plan(p;style) for p in panels]
end

function _visual_retained_figure(M,V,H,spec;size=nothing,style=V.VisualStyle())
    plans=_visual_retained_plan(H,V,spec,style)
    plans===nothing && return _VisualRetainedFigure(H.render(spec;size,style),spec,Any[],Any[])
    actualsize=size===nothing ? get(spec.metadata,:figure_size,(1000,700)) : size
    fig=M.Figure(;size=actualsize,figure_padding=style.padding,
        backgroundcolor=V._visual_color(style,V._VisualRole(:background)))
    panels=isempty(spec.panels) ? [spec] : spec.panels
    columns=isempty(spec.panels) ? 1 : get(spec.metadata,:panel_columns,min(3,length(panels)))
    handles=Any[]; headings=Any[]; row=1
    if !isempty(spec.panels)
        for (text,fontsize,role) in ((spec.title,22,:foreground),(spec.subtitle,14,:muted))
            isempty(text) && continue
            push!(headings,M.Label(fig[row,1:columns],text;font=style.font,fontsize=fontsize*style.fontsize/16,
                color=V._visual_color(style,V._VisualRole(role)),tellwidth=false,word_wrap=true))
            row+=1
        end
    end
    for (i,(p,plan)) in enumerate(zip(panels,plans))
        grid=fig[row+div(i-1,columns),mod1(i,columns)]=M.GridLayout()
        before=length(fig.content);plotrow=H.heading(grid,p,style)
        labels=copy(fig.content[before+1:end])
        ax=H.make_axis(grid[plotrow,1],p;style,headings=false)
        H.apply_limits(ax,p)
        plots=Any[]
        for c in plan.commands
            plot=c.kind===:interval_annotations ? _visual_interval_annotations!(M,ax,c.args...;c.kwargs...) :
                getproperty(M,Symbol(c.kind,:!))(ax,c.args...;c.kwargs...)
            push!(plots,plot)
        end
        for (j,cb) in enumerate(plan.colorbars)
            H.colorbar(grid[plotrow,1+j],plots[cb.plot],cb.label,style)
        end
        position=get(p.metadata,:legend_position,:bottom)
        position===:none || H.legend(fig,position===:right ? grid[plotrow,2+length(plan.colorbars)] : grid[plotrow+1,1],p,style)
        H.style_grid(grid,style)
        push!(handles,(;axis=ax,plots,plan,labels))
    end
    H.style_figure(fig,style)
    return _VisualRetainedFigure(fig,spec,handles,headings)
end

function _visual_update_retained!(M,V,H,state,spec;style=V.VisualStyle())
    isempty(state.panels) && return false
    old=state.spec
    old.kind==spec.kind && isempty(old.panels)==isempty(spec.panels) || return false
    length(old.panels)==length(spec.panels) || return false
    for key in (:panel_columns,:panel_positions,:panel_row_weights,:figure_size)
        get(old.metadata,key,nothing)==get(spec.metadata,key,nothing) || return false
    end
    (!isempty(old.title),!isempty(old.subtitle),old.legend)==(!isempty(spec.title),!isempty(spec.subtitle),spec.legend) || return false
    ps=isempty(spec.panels) ? [spec] : spec.panels
    os=isempty(old.panels) ? [old] : old.panels
    all(_visual_panel_shape(V,a)==_visual_panel_shape(V,b) for (a,b) in zip(os,ps)) || return false
    plans=_visual_retained_plan(H,V,spec,style);plans===nothing && return false
    for (handle,plan) in zip(state.panels,plans)
        map(_visual_command_shape,handle.plan.commands)==map(_visual_command_shape,plan.commands) || return false
        handle.plan.colorbars==plan.colorbars || return false
    end
    # Validate all shapes first. Each plot's arguments and attributes then move
    # together through Makie's public coordinated update API.
    for (h,p,o,plan) in zip(state.panels,ps,os,plans)
        for (plot,c,prev) in zip(h.plots,plan.commands,h.plan.commands)
            isequal(c,prev) && continue
            if c.kind===:interval_annotations
                plot.update(c.args...;c.kwargs...)
            else
                M.update!(plot;(Symbol(:arg,i)=>a for (i,a) in enumerate(c.args))...,c.kwargs...)
            end
        end
        for (label,text) in zip(h.labels,filter(!isempty,[p.title,p.subtitle]))
            label.text[]==text || (label.text[]=text)
        end
        p.axes==o.axes || begin
            h.axis.xlabel=p.axes.xlabel;h.axis.ylabel=p.axes.ylabel
            h.axis.xticks=something(get(p.axes,:xticks,nothing),M.automatic)
            h.axis.yticks=something(get(p.axes,:yticks,nothing),M.automatic)
            H.apply_limits(h.axis,p)
        end
    end
    for (label,text) in zip(state.headings,filter(!isempty,[spec.title,spec.subtitle]))
        label.text[]==text || (label.text[]=text)
    end
    state.panels=Any[merge(h,(;plan)) for (h,plan) in zip(state.panels,plans)]
    state.spec=spec
    return true
end
