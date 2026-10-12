# Inspect rendered strings independently of the number of layout/plot objects.
# Native batching may change those objects; literal text and coordinates must not.
function visual_text_records(fig, M)
    records = NamedTuple[]
    function visit(scene)
        for p in scene.plots
            p isa M.Text || continue
            strings = p.input_text[]
            for (i,text) in enumerate(strings)
                attr(name) = begin
                    v = getproperty(p,name)[]
                    v isa AbstractVector && !(v isa M.Makie.StaticVector) ? v[i] : v
                end
                push!(records,(;text=string(text),plot=p,index=i,position=p[1][][i],
                    fontsize=attr(:fontsize),font=attr(:font),color=attr(:color)))
            end
        end
        foreach(visit,scene.children)
    end
    visit(fig.scene)
    return records
end
