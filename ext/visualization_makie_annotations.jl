# Screen-space placement owns only labels and leaders. The interval anchors and
# exact records stay in the specification. Two plots suffice for any label count.
function _visual_label_layout(M, points, boxes, extent, gap, previous)
    placed = NTuple{4,Float64}[]
    positions = M.Vec2d[]
    leaders = M.Point2d[]
    for (index,(point, box)) in enumerate(zip(points, boxes))
        lo, hi = minimum(box), maximum(box)
        w, h = Float64(hi[1]-lo[1]), Float64(hi[2]-lo[2])
        if !all(isfinite, (point..., w, h))
            push!(positions, M.Vec2d(NaN,NaN)); continue
        end
        function fit(x,y)
            x = w+4 <= extent[1] ? clamp(x,2.,extent[1]-2-w) : (extent[1]-w)/2
            y = h+4 <= extent[2] ? clamp(y,2.,extent[2]-2-h) : (extent[2]-h)/2
            (x,y,x+w,y+h)
        end
        function score(b)
            overlap = sum(placed;init=0.) do a
                max(0.,min(a[3]+2,b[3])-max(a[1]-2,b[1])) *
                    max(0.,min(a[4]+2,b[4])-max(a[2]-2,b[2]))
            end
            covered = count(p->b[1]-gap<p[1]<b[3]+gap && b[2]-gap<p[2]<b[4]+gap,points)
            distance = hypot(clamp(point[1],b[1],b[3])-point[1],clamp(point[2],b[2],b[4])-point[2])
            (overlap, covered, distance)
        end
        best = index<=length(previous) && all(isfinite,previous[index]) ?
            fit(point[1]+previous[index][1]+lo[1],point[2]+previous[index][2]+lo[2]) :
            fit(point[1]+gap,point[2]+gap)
        cost = score(best)
        # Stable candidate order makes redraws reproducible. A bounded search
        # also keeps crowded stress views finite; no labels are discarded.
        for radius in (gap, gap+h+3, gap+2h+6, gap+4h+12)
            cost[1]==0 && cost[2]==0 && break
            for (dx,dy) in ((radius,radius),(-w-radius,radius),
                           (radius,-h-radius),(-w-radius,-h-radius),
                           (-w/2,radius),(-w/2,-h-radius),
                           (radius,-h/2),(-w-radius,-h/2))
                b=fit(point[1]+dx,point[2]+dy); c=score(b)
                if c < cost; best=b;cost=c;end
            end
            cost[1]==0 && cost[2]==0 && break
        end
        push!(placed,best)
        push!(positions,M.Vec2d(best[1]-lo[1]-point[1],best[2]-lo[2]-point[2]))
        # Clip the leader at the measured glyph rectangle and shrink both ends.
        q=M.Point2d(clamp(point[1],best[1],best[3]),clamp(point[2],best[2],best[4]))
        delta=point-q; distance=hypot(delta...)
        distance > 1.5gap && append!(leaders,(q+0.5gap*delta/distance,point-gap*delta/distance))
    end
    return positions,leaders
end

function _visual_interval_annotations!(M, ax, anchors; text, color, font, fontsize,
                                      linewidth, shrink, kwargs...)
    # Glyph bounds are measured once per text change, independently of the
    # offset we are about to choose. Updating offsets cannot restart placement.
    label=M.text!(ax,anchors;text,color,font,fontsize,align=(:left,:bottom),
        justification=:left,offset=fill(M.Vec2d(0),length(anchors)),inspectable=false)
    line=M.linesegments!(ax.scene,M.Point2d[];space=:pixel,color,linewidth,inspectable=false)
    boxes=Ref(M.Makie.raw_string_boundingboxes(label))
    points=Ref(anchors)
    previous=Ref(M.Vec2d[])
    labels=Ref(text)
    function layout()
        screen=[M.Point2d(M.Makie.project(ax.scene,p)) for p in points[]]
        offsets,segments=_visual_label_layout(M,screen,boxes[],M.widths(M.viewport(ax.scene)[]),Float64(last(shrink)),previous[])
        previous[]=offsets
        M.update!(label;offset=offsets)
        M.update!(line;arg1=segments)
        nothing
    end
    M.onany(ax.scene,M.viewport(ax.scene),M.Makie.camera(ax.scene).projectionview;update=true) do _...
        layout()
    end
    function update(nextanchors; text, color, font, fontsize, linewidth, shrink, kwargs...)
        text==labels[] || (previous[]=M.Vec2d[])
        labels[]=text
        points[]=nextanchors
        M.update!(label;arg1=nextanchors,text,color,font,fontsize,offset=fill(M.Vec2d(0),length(nextanchors)))
        M.update!(line; color,linewidth)
        boxes[]=M.Makie.raw_string_boundingboxes(label)
        layout()
    end
    return (;label,line,update)
end
