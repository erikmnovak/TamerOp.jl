# Self-contained browser fixtures. Start from the repository root with:
#   julia --project=test/browser test/browser/serve.jl
# Stop by creating TAMEROP_BROWSER_STOP_FILE, or interrupt the Julia process.
# Browser tests interact with the ordinary inspector controls and canvases.
# The hidden fixture readout is read-only; it never invokes a selection callback.
import TamerOp
using JSON3, SparseArrays, WGLMakie

const V = TamerOp.Visualization
const B = WGLMakie.Bonito
const M = WGLMakie.Makie
const QQ = Rational{BigInt}

function _triangle_diagram()
    complex = TamerOp.DataTypes.GradedComplex([[11,12,13],[21,22,23],[31]],
        [sparse([-1 -1 0;1 0 -1;0 1 1]),sparse(reshape([1,-1,1],3,1))],
        [(0,),(0,),(0,),(1,),(1,),(1,),(2,)])
    return TamerOp.persistence_diagram(complex;representatives=true)
end

function _band_encoding()
    PL = TamerOp.PLPolyhedra
    # The regions are x+y<0, 0<=x+y<2, and x+y>=2. One class spans
    # the whole line, one dies at x+y=2, and one starts at x+y=0.
    regions = [PL.HPoly(2,QQ[1 1],QQ[0],nothing,BitVector([true]),zero(QQ)),
        PL.HPoly(2,QQ[-1 -1;1 1],QQ[0,2],nothing,BitVector([false,true]),zero(QQ)),
        PL.HPoly(2,QQ[-1 -1],QQ[-2],nothing,BitVector([false]),zero(QQ))]
    classifier = PL.PLEncodingMap(2,[BitVector() for _ in regions],
        [BitVector() for _ in regions],regions,[(-1,0),(1,0),(3,0)])
    poset = TamerOp.FiniteFringe.FinitePoset(BitMatrix([i <= j for i in 1:3,j in 1:3]))
    module_ = TamerOp.Modules.PModule{QQ}(poset,[2,3,2],
        Dict((1,2)=>QQ[1 0;0 1;0 0],(2,3)=>QQ[1 0 0;0 0 1]);
        field=TamerOp.CoreModules.QQField())
    return TamerOp.Results.EncodingResult(poset,module_,
        TamerOp.EncodingCore.compile_encoding(poset,classifier))
end

function _squares_encoding()
    A = TamerOp.Advanced
    options = A.EncodingOptions(;backend=:pl_backend,poset_kind=:signature,
        field=TamerOp.CoreModules.QQField())
    # k_[0,2]^2 direct-sum k_[1,3]^2, with its retained indicator presentation.
    return TamerOp.encode([A.BoxUpset([0,0]),A.BoxUpset([1,1])],
        [A.BoxDownset([2,2]),A.BoxDownset([3,3])],QQ[1 0;0 1],options)
end

# JSON retains rational values exactly and represents infinities as strings.
_json_value(x::Union{Nothing,Bool,AbstractString,Integer}) = x
_json_value(x::Symbol) = string(x)
_json_value(x::Rational) = denominator(x) == 1 ? string(numerator(x)) : string(x)
_json_value(x::AbstractFloat) = isfinite(x) ? x : string(x)
_json_value(x::NamedTuple) = Dict(string(k)=>_json_value(v) for (k,v) in pairs(x))
_json_value(x::AbstractDict) = Dict(string(k)=>_json_value(v) for (k,v) in pairs(x))
_json_value(x::Union{Tuple,AbstractArray}) = [_json_value(v) for v in x]
_json_value(x) = string(x)

# Preserve matrix shape, including the 1-by-0 image of an active zero block.
_matrix_state(A) = A === nothing ? nothing :
    (;size=size(A),rows=[[A[i,j] for j in axes(A,2)] for i in axes(A,1)])

function _presentation_state(snapshot)
    haskey(snapshot.metadata,:stalks) || return nothing
    IR = TamerOp.IndicatorResolutions
    stalks = map(snapshot.metadata.stalks) do stalk
        stalk === nothing && return nothing
        return (;vertex=IR.presentation_vertex(stalk),
            dimension=IR.presentation_summary(stalk).dimension,
            active_rows=IR.active_rows(stalk),active_columns=IR.active_columns(stalk),
            matrix=_matrix_state(IR.presentation_matrix(stalk)),
            basis=_matrix_state(IR.image_basis(stalk)))
    end
    map_ = snapshot.metadata.presentation_map
    return (;relation=snapshot.metadata.relation,defined=snapshot.metadata.defined,stalks,
        map=map_ === nothing ? nothing : _matrix_state(IR.induced_map(map_)),
        ambient_projection=map_ === nothing ? nothing : _matrix_state(IR.ambient_projection(map_)))
end

function _navigation_data(session,fixture)
    session isa V.IntervalInspectionSession && return nothing
    prepared = V._inspection_scene(session)
    positions = copy(prepared.hasse.metadata.positions)
    dimensions = copy(session.prepared.dims)
    points = if fixture in (:squares,:squares_slices,:squares_grayscale)
        ((:first,(1//2,1//2)),(:overlap,(3//2,3//2)),
         (:second,(5//2,5//2)),(:active_zero,(1//2,5//2)))
    else
        ()
    end
    regions = map(points) do (id,point)
        vertex = TamerOp.EncodingCore.locate(TamerOp.Results.encoding_map(session.object),point)
        (;id,point,vertex,dimension=dimensions[vertex])
    end
    return (;positions,dimensions,regions)
end

function _navigation_state(ui,data)
    data === nothing && return nothing
    figure = ui.figure
    width,height = M.widths(M.viewport(figure.scene)[])
    targets = NamedTuple[]
    if !ui.closed[]
        for (vertex,point) in enumerate(data.positions)
            pixel = M.project(ui.axes.hasse.scene,M.Point2d(point)) + minimum(M.viewport(ui.axes.hasse.scene)[])
            push!(targets,(;id=vertex,panel=:hasse,point,vertex,
                dimension=data.dimensions[vertex],x=Float64(pixel[1]/width),y=Float64(1-pixel[2]/height)))
        end
        for item in data.regions
            pixel = M.project(ui.axes.region.scene,M.Point2d(item.point)) + minimum(M.viewport(ui.axes.region.scene)[])
            push!(targets,merge(item,(;panel=:region,x=Float64(pixel[1]/width),y=Float64(1-pixel[2]/height))))
        end
    end
    markers = map((:region,:hasse)) do panel
        group = getproperty(ui.markers,panel)
        group === nothing ? nothing : (;stalk=group.stalk[],source=group.source[],
            target=group.target[],both=group.both[])
    end
    return (;hover=ui.hover_text[],mouseposition=M.events(figure.scene).mouseposition[],
        figure_size=(width,height),pick_targets=targets,markers=(;region=markers[1],hasse=markers[2]))
end

function _pick_targets(figure, axes, data)
    (figure === nothing || axes === nothing || data === nothing) && return ()
    width,height = M.widths(M.viewport(figure.scene)[])
    targets = NamedTuple[]
    for (i,id) in enumerate(data.interval_ids)
        segment = data.bar_segments[i]
        for (panel,point) in ((:diagram,data.diagram_points[i]),
                              (:barcode,((segment[1]+segment[3])/2,segment[2])))
            axis = getproperty(axes,panel)
            pixel = M.project(axis.scene,M.Point2d(point)) + minimum(M.viewport(axis.scene)[])
            # DOM canvas coordinates start at the top left; Makie starts below.
            push!(targets,(;id,panel,x=Float64(pixel[1]/width),y=Float64(1-pixel[2]/height)))
        end
    end
    return targets
end

function _rank_navigation_state(ui, data, snapshot)
    data === nothing && return nothing
    haskey(snapshot.metadata, :domain) || return nothing
    figure = ui.rank_figure[]
    axis = ui.rank_axis[]
    (figure === nothing || axis === nothing || ui.closed[]) && return nothing
    width,height = M.widths(M.viewport(figure.scene)[])
    targets = NamedTuple[]
    items = snapshot.metadata.domain === :parameter ? data.regions :
        [(;id=q,point,vertex=q,dimension=data.dimensions[q]) for (q,point) in enumerate(data.positions)]
    for item in items
        pixel = M.project(axis.scene,M.Point2d(item.point)) + minimum(M.viewport(axis.scene)[])
        push!(targets,merge(item,(;x=Float64(pixel[1]/width),y=Float64(1-pixel[2]/height))))
    end
    return (;pick_targets=targets,figure_size=(width,height),mouseposition=M.events(figure.scene).mouseposition[])
end

function _browser_state(session,ui,last_figure,navigation,fixture)
    summary = V.inspection_summary(session)
    snapshot = V.inspection_snapshot(session)
    interval_fixture = session isa V.IntervalInspectionSession
    result = interval_fixture ? nothing : get(snapshot.metadata,:slice_result,nothing)
    data = interval_fixture ? snapshot.metadata.interval_view : get(snapshot.metadata,:slice_view,nothing)
    records = interval_fixture ? snapshot.metadata.interval_payload.records :
        result === nothing ? () : result.intervals
    selected = findfirst(r -> r.id == summary.selection.interval,records)
    figure = interval_fixture ? ui.figure : ui.slice.figure[]
    axes = interval_fixture ? ui.axes[] : ui.slice.axes[]
    if figure !== nothing && figure !== last_figure[] && !ui.closed[]
        M.update_state_before_display!(figure)
        last_figure[] = figure
    end
    bar = interval_fixture ? ui.selected_bar[] :
        ui.slice.selected_bar[] === nothing ? () : ui.slice.selected_bar[][]
    point = interval_fixture ? ui.selected_point[] :
        ui.slice.selected_point[] === nothing ? () : ui.slice.selected_point[][]
    return JSON3.write(_json_value((;
        fixture,style=(;palette=ui.style.palette,fontsize=ui.style.fontsize),
        summary,selection=summary.selection,
        # One session listener belongs to this fixture's observation probe.
        viewer_count=max(0,summary.listener_count-(summary.closed ? 0 : 1)),
        records,selected_record=selected === nothing ? nothing : records[selected],
        representative=get(snapshot.metadata,:selected_representative,nothing),
        inspection=get(snapshot.metadata,:inspection,nothing),
        rank_sections=get(snapshot.metadata,:rank_sections,nothing),
        rank_anchors=get(snapshot.metadata,:anchors,nothing),
        presentation=_presentation_state(snapshot),
        panels=[(;title=panel.title,kind=panel.kind,
            matrix=_matrix_state(get(panel.metadata,:matrix,nothing))) for panel in snapshot.panels],
        main_ui=_navigation_state(ui,navigation),
        rank_ui=interval_fixture ? nothing : _rank_navigation_state(ui,navigation,snapshot),
        slice=result === nothing ? nothing : (;
            scope=result.scope,domain=result.domain,window=result.window,
            line=result.line,endpoint_semantics=result.endpoint_semantics,
            essential_status=result.essential_status),
        ui=(;dom_id=ui.dom_id,closed=ui.closed[],
            rebuild_count=interval_fixture ? ui.rebuild_count[] : ui.slice.rebuild_count[],
            displayed_ids=data === nothing ? () : data.interval_ids,
            selected_bar=bar,selected_point=point,
            selected_region=interval_fixture ? () : ui.slice.selected_region[],
            selected_region_point=interval_fixture ? () : ui.slice.selected_region_point[],
            line_points=interval_fixture ? () : ui.slice.line_points[],
            status=ui.status_text[],error=ui.error_text[],
            mouseposition=figure === nothing ? nothing : M.events(figure.scene).mouseposition[],
            figure_size=figure === nothing ? nothing : Tuple(M.widths(M.viewport(figure.scene)[])),
            pick_targets=ui.closed[] ? () : _pick_targets(figure,axes,data)))))
end

function _browser_app(get_session::Function,title,tick;style,fixture)
    return B.App(;title) do client
        session = get_session()
        V.inspection_summary(session).closed && return B.DOM.p(
            "This fixture session is closed. Restart the browser fixture server.")
        viewer = TamerOp.visualize(session;backend=:wglmakie,style)
        extension = Base.get_extension(TamerOp,:TamerOpWGLMakieExt)
        ui = extension._inspection_ui(viewer)
        navigation = _navigation_data(session,fixture)
        navigation === nothing || M.update_state_before_display!(ui.figure)
        last_figure = Ref{Any}(nothing)
        state = map(client,tick) do _
            _browser_state(session,ui,last_figure,navigation,fixture)
        end
        return B.DOM.div(B.jsrender(client,viewer),
            B.DOM.pre(state;var"data-testid"="julia-session-state",style="display:none"))
    end
end

function _observe_fixture!(sessions,session,tick)
    push!(sessions,session)
    V._on_inspection(session,_ -> begin
        # Notify after all product observers finish the committed update.
        @async begin
            yield()
            tick[] += 1
        end
        nothing
    end)
    return session
end

function _lazy_squares_app(sessions,tick;title,style,fixture,slice)
    instance = Ref{Any}(nothing)
    fixture_lock = ReentrantLock()
    get_session() = lock(fixture_lock) do
        if instance[] === nothing
            session = V.inspection_session(_squares_encoding();box=([-1,-1],[4,4]),
                slice=slice ? (basepoint=(0,0),direction=(1,1)) : nothing)
            instance[] = _observe_fixture!(sessions,session,tick)
        end
        instance[]
    end
    return _browser_app(get_session,title,tick;style,fixture)
end

function _port(name,default)
    value = tryparse(Int,get(ENV,name,string(default)))
    value !== nothing && 1 <= value <= 65535 || error("$name must be a port from 1 to 65535")
    return value
end

include("matching-fixtures.jl")

function _main()
    interval_port = _port("TAMEROP_BROWSER_INTERVAL_PORT",8848)
    slice_port = _port("TAMEROP_BROWSER_SLICE_PORT",8849)
    interval_port != slice_port || error("Browser fixture ports must differ")
    fontsize = tryparse(Float64,get(ENV,"TAMEROP_BROWSER_FONTSIZE","18"))
    fontsize !== nothing && isfinite(fontsize) && fontsize > 0 ||
        error("TAMEROP_BROWSER_FONTSIZE must be a positive finite number")
    style = TamerOp.VisualStyle(;fontsize)
    stop_file = get(ENV,"TAMEROP_BROWSER_STOP_FILE",joinpath(@__DIR__,".stop-server"))
    isfile(stop_file) && rm(stop_file)
    initial_sessions = (V.inspection_session(_triangle_diagram();dim=0),
        V.inspection_session(_band_encoding();box=([-1,-1],[2,2]),
            slice=(basepoint=(0,0),direction=(1,1)),slice_scope=:global))
    tick = M.Observable(0)
    sessions = Any[]
    servers = B.Server[]
    try
        for session in initial_sessions
            _observe_fixture!(sessions,session,tick)
        end
        for (session,title,port,fixture) in zip(initial_sessions,
                ("A41 ordinary intervals","A41 whole-line slices"),(interval_port,slice_port),
                (:ordinary,:slices))
            server = B.Server(_browser_app(() -> session,title,tick;style,fixture),"127.0.0.1",port)
            push!(servers,server)
            # Bonito may try the next port when one is occupied. Tests require
            # the explicitly requested address, so fail instead of drifting.
            server.port == port || error("Requested browser fixture port $port is occupied")
            # Readiness probes must not allocate an inspector or its listeners.
            B.route!(server,"/health" => (_ -> B.HTTP.Response(200,
                ["Content-Type" => "application/json"],JSON3.write((;pid=getpid())))))
            # Full Chromium requests a tab icon; this fixture has none. Answer
            # explicitly so a missing icon does not pollute the console checks.
            B.route!(server,"/favicon.ico" => (_ -> B.HTTP.Response(204)))
        end
        # Independent sessions keep each historical acceptance scenario isolated.
        # Construct them only when its first browser requests the corresponding URL.
        for (path,fixture,title,slice,route_style) in (
                ("/squares",:squares,"A37 spaces, maps and presentations",false,TamerOp.VisualStyle(;fontsize=18)),
                ("/rank-sections",:squares,"A84 anchored rank sections",false,TamerOp.VisualStyle(;fontsize=18)),
                ("/squares-slices",:squares_slices,"A40 finite-window square slices",true,TamerOp.VisualStyle(;fontsize=18)),
                ("/squares-grayscale",:squares_grayscale,"A35 large grayscale inspector",true,TamerOp.VisualStyle(;fontsize=24,palette=:grayscale)))
            B.route!(servers[2],path => _lazy_squares_app(sessions,tick;
                title,style=route_style,fixture,slice))
        end
        B.route!(servers[2],"/matching" => _matching_browser_app(sessions,tick))
        B.route!(servers[2],"/matching-slices" => _matching_browser_app(sessions,tick;fibered=true))
        println("TAMEROP_BROWSER_READY intervals=http://127.0.0.1:$interval_port slices=http://127.0.0.1:$slice_port")
        flush(stdout)
        while !isfile(stop_file)
            sleep(0.2)
            # Read-only sampling also exposes pointer delivery and viewer
            # disconnection, neither of which changes mathematical selection.
            tick[] += 1
        end
    finally
        foreach(close,servers)
        foreach(V.close_inspection!,sessions)
        isfile(stop_file) && rm(stop_file)
        println("TAMEROP_BROWSER_STOPPED")
        flush(stdout)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    Base.exit_on_sigint(false)
    try
        _main()
    catch err
        err isa InterruptException || rethrow()
    end
end
