# Labelled browser controls for array slices; Cairo renders the supplied slice.
function _render_volume_viewer(spec; display=:inline, figure=nothing, size=nothing, style=VIZ.VisualStyle())
    VIZ._check_visual_render_options(; display, figure, size, style)
    volume = get(spec.metadata, :volume, nothing)
    view_dims = Tuple(spec.metadata.view_dims)
    control_dims = sort!(collect(setdiff(1:ndims(volume), collect(view_dims))))
    isempty(control_dims) && return Base.invokelatest(_BASE_WGL_RENDER, spec; display, figure, size, style)
    B, M = WGLMakie.Bonito, WGLMakie.Makie
    _NEXT_INSPECTION_UI_ID[] += 1
    dom_id = "tamerop-volume-viewer-$(_NEXT_INSPECTION_UI_ID[])"
    fixed = Dict{Int,Int}(spec.metadata.fixed_indices)
    source = Ref{Any}(volume)
    metadata = Base.structdiff(spec.metadata, (; volume=nothing, requires_live_julia=nothing))
    static = VIZ.VisualizationSpec(:image; title=spec.title, subtitle="",
        layers=spec.layers, axes=spec.axes, legend=spec.legend, metadata)
    old_blocks = figure === nothing ? Any[] : copy(figure.content)
    fig = Base.invokelatest(_BASE_WGL_RENDER, static; display, figure, size, style)
    blocks = [block for block in fig.content if !(block in old_blocks)]
    axis = only(filter(block -> block isa M.Axis, blocks))
    heatmap = only(filter(plot -> plot isa M.Heatmap, axis.scene.plots))
    closed = Ref(false)
    disabled = M.Observable(false)
    status = M.Observable("")
    content = M.Observable{Any}(fig)
    observers = Any[]
    browser_client = Ref{Any}(nothing)
    browser_lock = ReentrantLock()
    app_ref = Ref(WeakRef(nothing))
    sliders = Dict(dim => B.Slider(1:Base.size(volume, dim); value=fixed[dim],
        id="$dom_id-dim-$dim", disabled, style=_inspection_control_style(style)) for dim in control_dims)
    function update_status()
        status[] = "Displayed axes $(view_dims[1]) and $(view_dims[2]); " *
            join(("axis $dim index $(fixed[dim])" for dim in control_dims), "; ") * "."
    end
    for dim in control_dims
        push!(observers, M.on(sliders[dim].value) do value
            closed[] && return nothing
            fixed[dim] = Int(value)
            heatmap[3][] = permutedims(VIZ._image_slice_values(source[], view_dims, fixed))
            update_status()
            nothing
        end)
    end
    function dispose(; close_session=false)
        closed[] && return nothing
        closed[] = true
        disabled[] = true
        foreach(M.off, observers)
        empty!(observers)
        source[] = nothing
        foreach(delete!, blocks)
        empty!(blocks)
        content[] = B.DOM.p("Volume viewer closed. Render the image again to open another view.")
        lock(browser_lock) do
            browser_client[] = nothing
        end
        app = app_ref[].value
        app === nothing || delete!(_INSPECTION_UI_HANDLES, app)
        nothing
    end
    close_button = B.Button("Close viewer"; id="$dom_id-close", disabled, style=_inspection_control_style(style))
    push!(observers, M.on(_ -> dispose(), close_button.value))
    update_status()
    labels = [B.DOM.label(B.DOM.span("Slice index on axis $dim"; style="display:block;font-weight:600"),
        sliders[dim]; style="display:block;min-width:12em;max-width:100%") for dim in control_dims]
    dom = B.DOM.div(B.DOM.p("Move a slider with the arrow keys or pointer. Indices select array slices; live Julia is required."),
        B.DOM.div(labels..., close_button; style="display:flex;flex-wrap:wrap;gap:$(style.gap)px;align-items:end"),
        B.DOM.p(status; id="$dom_id-status", role="status"),
        B.DOM.div(content; style="max-width:100%;overflow:auto", tabindex="0", var"aria-label"="Volume slice chart");
        id=dom_id, style="font-family:'$(style.font)',sans-serif;font-size:$(style.fontsize)px;line-height:1.45;max-width:100%;padding:$(style.padding)px;box-sizing:border-box")
    app = B.App(; title="TamerOp volume slice viewer") do client
        lock(browser_lock) do
            closed[] && return B.DOM.p("This volume viewer has been closed.")
            if browser_client[] !== nothing && browser_client[] !== client
                existing = app_ref[].value
                existing === nothing || (existing.session[] = browser_client[])
                return B.DOM.p("This widget already has a browser client. Render the image again for an independent view.")
            end
            if browser_client[] === nothing
                browser_client[] = client
                _inspection_keyboard_navigation!(client, dom)
                push!(observers, M.on(client.on_close) do ended
                    ended && dispose()
                    nothing
                end)
            end
            dom
        end
    end
    app_ref[] = WeakRef(app)
    _INSPECTION_UI_HANDLES[app] = (; session=nothing, figure=fig, axis, heatmap,
        controls=(; sliders, close=close_button), fixed_indices=fixed, view_dims,
        dispose, closed, disabled, observers, browser_client, dom_id, status, source)
    app
end
