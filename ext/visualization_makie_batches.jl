# Conservative overlap test: false guarantees disjoint interiors. Touching
# boundaries retain their common stroke; uncertain concave boxes stay ordered.
function _visual_polygon_overlap(polygons)
    boxes = [(minimum(first,p), maximum(first,p), minimum(last,p), maximum(last,p))
             for p in polygons if !isempty(p)]
    length(boxes) < 2 && return false
    spanx = maximum(b->b[2],boxes) - minimum(b->b[1],boxes)
    spany = maximum(b->b[4],boxes) - minimum(b->b[3],boxes)
    # Sweep on the less crowded dimension (no all-pairs geometry reconstruction).
    xcost = sum(b->b[2]-b[1],boxes) / max(spanx,eps(Float64))
    ycost = sum(b->b[4]-b[3],boxes) / max(spany,eps(Float64))
    lo,hi,otherlo,otherhi = xcost <= ycost ? (1,2,3,4) : (3,4,1,2)
    sort!(boxes;by=b->b[lo])
    active = typeof(first(boxes))[]
    for b in boxes
        filter!(a->a[hi]>b[lo],active)
        any(a->a[otherlo]<b[otherhi] && b[otherlo]<a[otherhi],active) && return true
        push!(active,b)
    end
    return false
end

# One layout block and one text plot for the whole table. Measured glyph extents
# determine column widths and row heights; literal coefficients never become
# scalar colors. Cell anchors share the positions observable, not per-cell plots.
function _visual_matrix_table!(M, V, parent, layer, meta, style;
                               color, role_color, textsize, composition_cells)
    nr,nc = size(layer.entries)
    labels = String[get(meta,:matrix_corner,"target / source")]
    colors = [M.to_color(role_color(:muted))]
    fonts = String[style.font]
    sizes = Float64[textsize(11)]
    cells = Tuple{Int,Int}[(1,1)]
    row_colors = get(meta,:row_colors,fill(V._VisualRole(:foreground),nr))
    col_colors = get(meta,:column_colors,fill(V._VisualRole(:foreground),nc))
    entry_colors = get(meta,:cell_colors,fill(V._VisualRole(:foreground),nr,nc))
    row_roles = get(meta,:row_roles,nothing)
    col_roles = get(meta,:column_roles,nothing)
    entry_roles = get(meta,:cell_roles,nothing)
    function add(text,i,j,c,font,fontsize)
        push!(labels,text);push!(cells,(i,j));push!(colors,M.to_color(c))
        push!(fonts,font);push!(sizes,fontsize)
    end
    for j in 1:nc
        role = col_roles === nothing ? nothing : col_roles[j]
        add(layer.column_labels[j] * (role===nothing ? "" : V._visual_role_label(role)),
            1,j+1,role===nothing ? color(col_colors[j]) : role_color(role),style.font,textsize(12))
    end
    for i in 1:nr
        role = row_roles === nothing ? nothing : row_roles[i]
        add(layer.row_labels[i] * (role===nothing ? "" : V._visual_role_label(role)),
            i+1,1,role===nothing ? color(row_colors[i]) : role_color(role),style.font,textsize(12))
        for j in 1:nc
            c = entry_roles===nothing ? color(entry_colors[i,j]) : role_color(entry_roles[i,j])
            add(layer.entries[i,j],i+1,j+1,c,style.mono_font,textsize(nc>8 ? 12 : 14))
        end
    end
    # The empty Label provides only a native, disposable layout anchor. Its
    # explicit dimensions avoid creating a GridLayout/Label per coefficient.
    holder = M.Label(parent,"";width=1.,height=1.,tellwidth=false,halign=:center,valign=:top)
    positions = M.Observable(fill(M.Point2d(0),length(labels)))
    plot = M.text!(holder.blockscene,positions;text=labels,color=colors,font=fonts,
        fontsize=sizes,align=(:center,:center),markerspace=:data,inspectable=false)
    boxes = M.Makie.raw_string_boundingboxes(plot)
    widths = zeros(nc+1);heights = zeros(nr+1)
    for (box,(i,j)) in zip(boxes,cells)
        extent = M.widths(box)
        isfinite(extent[1]) && (widths[j]=max(widths[j],extent[1]))
        isfinite(extent[2]) && (heights[i]=max(heights[i],extent[2]))
    end
    dx,dy = style.gap*7/6,style.gap*2/3
    holder.width = sum(widths)+nc*dx
    holder.height = sum(heights)+nr*dy
    x = cumsum(widths) .- widths./2 .+ (0:nc).*dx
    y = cumsum(heights) .- heights./2 .+ (0:nr).*dy
    M.on(holder.blockscene,holder.layoutobservables.computedbbox;update=true) do box
        origin = minimum(box);top = maximum(box)[2]
        positions[] = [M.Point2d(origin[1]+x[j],top-y[i]) for (i,j) in cells]
    end
    if composition_cells !== nothing
        for (index,(i,j)) in enumerate(cells)
            i>1 && j>1 && (composition_cells[(i-1,j-1)]=(;positions,index))
        end
    end
    return nothing
end
