# Annotation endpoints follow final scene viewports and coefficient-label boxes.
# This is renderer work only; semantic selectors have already been validated.
function _render_visual_composition!(TO,M,fig,spec,handles,style)
    V=TO.Visualization
    names=spec.metadata.panel_names
    for group in spec.metadata.linked_axes
        axes=[handles[findfirst(==(id),names)].axis for id in group]
        all(ax->ax isa M.Axis,axes) || throw(ArgumentError("Shared axes require planar axis panels."))
        M.linkaxes!(axes...)
    end
    # WGL draws child scenes after root plots. A final transparent child keeps
    # connectors above panel backgrounds as well as in the gutters between them.
    overlay=M.Scene(fig.scene;camera=M.campixel!,clear=false)
    function endpoint(a)
        if a.space===:figure
            return M.lift(fig.scene,M.viewport(fig.scene)) do rect
                M.Point2d(minimum(rect) .+ a.point .* M.widths(rect))
            end
        end
        i=findfirst(==(a.panel),names);h=handles[i]
        resolved=V._composition_anchor(spec.panels[i],a)
        if resolved.kind===:matrix
            cell=h.cells[resolved.entry]
            return M.lift(fig.scene,cell.positions) do positions
                M.Point2d(positions[cell.index])
            end
        elseif resolved.kind===:panel
            box=h.axis===nothing ? h.grid.layoutobservables.computedbbox : M.viewport(h.axis.scene)
            return M.lift(fig.scene,box) do rect
                M.Point2d(minimum(rect) .+ resolved.point .* M.widths(rect))
            end
        end
        h.axis isa M.Axis || throw(ArgumentError("Data-point anchors require a planar axis; select a matrix coefficient by target."))
        ax=h.axis
        camera=M.Makie.camera(ax.scene)
        # Limits can notify before the camera projection is updated. Observe the
        # projection as well so the final endpoint uses the final transformation.
        return M.lift(fig.scene,M.viewport(ax.scene),ax.finallimits,camera.projectionview,
                ax.xreversed,ax.yreversed) do viewport,limits,projection,xrev,yrev
            p=resolved.point
            lo=minimum(limits);hi=maximum(limits)
            # Cropping hides the annotation instead of attaching it to the frame.
            all(k->lo[k]<=p[k]<=hi[k],1:2) || return M.Point2d(NaN,NaN)
            M.Point2d(M.Makie.project(ax.scene,M.Point2d(p)) + minimum(viewport))
        end
    end
    for a in spec.metadata.annotations
        from=endpoint(a.from);to=endpoint(a.to)
        selected=a.evidence!==nothing
        color=M.to_color(V._visual_color(style,V._VisualRole(selected ? :selected : :muted)))
        pattern=selected ? :solid : :dash
        shape=M.lift(fig.scene,from,to) do p,q
            delta=q-p;normal=M.Vec2d(-delta[2],delta[1]);bend=a.curve*normal
            if a.kind===:highlight
                x1,x2=minmax(p[1],q[1]);y1,y2=minmax(p[2],q[2])
                return M.Point2d[(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)]
            elseif a.kind===:bracket
                len=sqrt(sum(abs2,delta));n=len>0 ? 8normal/len : zero(normal)
                return M.Point2d[p+n,p,q,q+n]
            end
            return [M.Point2d((1-t)*p+t*q+4t*(1-t)*bend) for t in range(0,1;length=49)]
        end
        line=M.lines!(overlay,shape;space=:pixel,color,linewidth=2style.linewidth_scale,
            linestyle=pattern,inspectable=false,overdraw=true)
        M.translate!(line,0,0,500)
        if a.kind in (:arrow,:callout)
            head=M.lift(fig.scene,shape) do pts
                q=last(pts);v=q-pts[end-1];len=sqrt(sum(abs2,v))
                u=len>0 ? v/len : M.Vec2d(0,0);n=M.Vec2d(-u[2],u[1])
                M.Point2d[q-11u+5n,q,q-11u-5n]
            end
            plot=M.lines!(overlay,head;space=:pixel,color,linewidth=2style.linewidth_scale,inspectable=false,overdraw=true)
            M.translate!(plot,0,0,500)
        end
        labelpoint=M.lift(fig.scene,shape) do pts
            p=a.kind===:highlight ? (pts[3]+pts[4])/2 :
                a.kind===:bracket ? (pts[2]+pts[3])/2 : pts[cld(length(pts),2)]
            [p+M.Vec2d(0,8)]
        end
        halo=M.text!(overlay,labelpoint;text=[a.label],space=:pixel,
            color=V._visual_color(style,V._VisualRole(:background)),
            align=(:center,:bottom),font=style.font,fontsize=12style.fontsize/16,
            strokecolor=V._visual_color(style,V._VisualRole(:background)),strokewidth=1.0,
            inspectable=false,overdraw=true)
        M.translate!(halo,0,0,501)
        label=M.text!(overlay,labelpoint;text=[a.label],space=:pixel,color,
            align=(:center,:bottom),font=style.font,fontsize=12style.fontsize/16,
            strokewidth=0,
            inspectable=false,overdraw=true)
        M.translate!(label,0,0,502)
    end
    return fig
end

function _save_composition_html(TO,M,path,spec;kwargs...)
    escape(s)=replace(string(s),'&'=>"&amp;",'<'=>"&lt;",'>'=>"&gt;",'\"'=>"&quot;")
    # Cairo's vector output fixes the complete selected scene. Preserve its
    # intrinsic text size; narrow browsers scroll panels and annotations together.
    svg=mktempdir() do dir
        file=joinpath(dir,"figure.svg")
        TO.Visualization._VISUAL_SAVERS[:cairomakie](file,spec;kwargs...)
        read(file,String)
    end
    svg=replace(svg,r"<\?xml[^>]*\?>"=>"")
    svg=replace(svg,"<svg "=>"<svg role=\"img\" aria-labelledby=\"figure-title figure-description\" ";count=1)
    descriptions=[(a.evidence===nothing ? "Authored explanation: " : "Checked representative correspondence: ")*
        a.label*" ("*string(a.from.panel)*" "*string(a.from.target===nothing ? a.from.point : a.from.target)*
        " to "*string(a.to.panel)*" "*string(a.to.target===nothing ? a.to.point : a.to.target)*")" for a in spec.metadata.annotations]
    caption=join(filter(!isempty,[spec.subtitle;descriptions])," ")
    write(path,"<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"*
        "<title>"*escape(spec.title)*"</title><style>*{box-sizing:border-box}body{font:17px system-ui;margin:1.5rem auto;padding:0 1rem;max-width:1600px;color:#172b38;overflow-wrap:anywhere}.figure-scroll{max-width:100%;overflow:auto}.figure-scroll:focus-visible{outline:3px solid #0072b2;outline-offset:3px}svg{display:block;max-width:none;height:auto}figure{margin:0}summary{cursor:pointer}@media print{.figure-scroll{overflow:visible}svg{max-width:100%}}</style>"*
        "<h1 id=\"figure-title\">"*escape(spec.title)*"</h1><p id=\"figure-description\">"*escape(caption)*"</p><figure><div class=\"figure-scroll\" tabindex=\"0\" role=\"region\" aria-label=\"Composed figure\">"*svg*"</div>"*
        "<figcaption>Fixed selected figure. Scroll to inspect wide panels at their original text size; panels and annotations stay together. This file needs no Julia session.</figcaption></figure>"*
        "<details><summary>Annotation meanings and coordinate context</summary><ul>"*join(["<li>"*escape(d)*"</li>" for d in descriptions])*"</ul>"*
        join(["<p>"*escape(string(id))*": "*escape(string(get(p.metadata,:composition_context,nothing)===nothing ?
            "independent panel" : TO.Visualization.describe(p.metadata.composition_context)))*"</p>" for (id,p) in zip(spec.metadata.panel_names,spec.panels)])*
        "</details></html>")
    return path
end
