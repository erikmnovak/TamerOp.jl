# Source panels consume the same retained records as static export. Telemetry
# observes product state; selection is driven only by actual browser events.
function _source_browser_app(sessions,tick;periodic=false)
    instance=Ref{Any}(nothing)
    get_session=()->begin
        if instance[]===nothing
            values=periodic ? fill(2//3,1,1) : [0 0 0;0 5 0;0 0 0]
            diagram=TamerOp.cubical_persistence(values;periodic,field=TamerOp.CoreModules.F3(),
                representatives=true,cocycles=true,source=true)
            instance[]=_observe_fixture!(sessions,V.inspection_session(diagram;dim=1),tick)
        end
        instance[]
    end
    return _browser_app(get_session,"A43 source representatives",tick;
        style=TamerOp.VisualStyle(fontsize=18),fixture=:ordinary)
end
