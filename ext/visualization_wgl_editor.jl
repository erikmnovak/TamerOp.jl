# Table-first editing makes every operation available to a keyboard. Algebra
# and history live in the core; this extension only parses explicit input.
function _render_module_editor(spec;display=:inline,figure=nothing,size=nothing,style=VIZ.VisualStyle())
    VIZ._check_visual_render_options(;display,figure,size,style)
    figure===nothing || throw(ArgumentError("The editor owns its figure."))
    session=spec.metadata.session
    VIZ.inspection_summary(session).closed && throw(ArgumentError("This editor is closed."))
    B,M=WGLMakie.Bonito,WGLMakie.Makie
    _NEXT_INSPECTION_UI_ID[]+=1
    dom_id="tamerop-module-editor-$(_NEXT_INSPECTION_UI_ID[])"
    disabled=M.Observable(false);export_disabled=M.Observable(false)
    error_text=M.Observable("");status_text=M.Observable("")
    code_text=M.Observable("");code_link=M.Observable("#");archive_link=M.Observable("#")
    content=M.Observable{Any}(B.DOM.div());readout=M.Observable{Any}(B.DOM.div())
    cs=_inspection_control_style(style)
    text(suffix,value="")=B.TextField(value;id="$dom_id-$suffix",disabled,style=cs)
    button(suffix,label)=B.Button(label;id="$dom_id-$suffix",disabled,style=cs)
    order=text("order");dims=text("dimensions");field=text("field")
    source=text("source","1");target=text("target","2");coefficients=text("matrix")
    rows=text("rows","1");columns=text("columns","1");vertex=text("vertex","1")
    upsets=text("upsets");downsets=text("downsets")
    apply_order=button("apply-order","Apply order and dimensions")
    apply_field=button("apply-field","Apply coefficient field")
    select=button("select","Select vertex and edge")
    apply_matrix=button("apply-matrix","Apply matrix")
    remove_map=button("remove-map","Remove cover matrix")
    apply_supports=button("apply-supports","Apply supports")
    undo=button("undo","Undo edit");redo=button("redo","Redo edit")
    reset=button("reset","Restore initial draft");close_button=button("close","Close editor")
    generate=B.Button("Generate Julia code";id="$dom_id-generate",disabled=export_disabled,style=cs)
    archive=button("archive","Prepare saved draft")
    fig=Ref{Any}(nothing);closed=Ref(false);updating=Ref(false)
    observers=Any[];session_token=Ref(0);browser_client=Ref{Any}(nothing);app_ref=Ref(WeakRef(nothing));browser_lock=ReentrantLock()
    integer(c)=begin
        n=tryparse(Int,strip(c.value[]));n===nothing && throw(ArgumentError("Enter an integer index or dimension."));n
    end
    integers(t)=isempty(strip(t)) ? Int[] : [begin
        x=tryparse(Int,s);x===nothing && throw(ArgumentError("Use integer vertex IDs."));x
    end for s in split(strip(t),r"[ ,]+";keepempty=false)]
    support_lists(t)=isempty(strip(t)) ? Vector{Int}[] : [strip(row)=="-" ? Int[] : integers(row) for row in split(t,';')]
    field_name(f)=f isa TO.CoreModules.QQField ? "QQ" : f isa TO.CoreModules.PrimeField ? "F$(f.p)" :
        "Real("*string(TO.CoreModules.coeff_type(f))*")"
    function chosen_field()
        value=strip(field.value[]);old=VIZ.editor_draft(session).field
        value==field_name(old) && return old
        value=="QQ" && return TO.CoreModules.QQField()
        for T in (Float16,Float32,Float64)
            value=="Real($T)" && return TO.CoreModules.RealField(T)
        end
        match_=match(r"^F([0-9]+)$",value)
        match_===nothing && throw(ArgumentError("Use QQ, F followed by a prime (such as F3), or Real(Float16), Real(Float32), Real(Float64)."))
        TO.CoreModules.Fp(parse(Int,only(match_.captures)))
    end
    function commit(f)
        closed[] && return false
        disabled[]=true
        try;f();error_text[]="";true
        catch err;err isa InterruptException && rethrow();error_text[]=sprint(showerror,err);false
        finally;disabled[]=closed[];end
    end
    function listen(f,observable)
        push!(observers,M.on(observable) do value
            updating[] || closed[] || f(value)
            nothing
        end)
    end
    function dispose(;close_session=false)
        closed[] && return nothing
        closed[]=true;disabled[]=true;export_disabled[]=true
        session_token[]==0 || VIZ._off_inspection!(session,session_token[])
        session_token[]=0;foreach(M.off,observers);empty!(observers)
        fig[]===nothing || empty!(fig[])
        content[]=B.DOM.p("Editor closed. The final static snapshot remains available.")
        lock(browser_lock) do;browser_client[]=nothing;end
        app=app_ref[].value;app===nothing || delete!(_INSPECTION_UI_HANDLES,app)
        close_session && VIZ.close_inspection!(session)
        nothing
    end
    function refresh()
        summary=VIZ.inspection_summary(session)
        summary.closed && return dispose()
        d=VIZ.editor_draft(session);state=VIZ.inspection_selection(session)
        updating[]=true
        try
            order.value[]=VIZ._editor_matrix_text(Int.(d.order));field.value[]=field_name(d.field)
            vertex.value[]=state.vertex===nothing ? "" : string(state.vertex)
            if d.kind===:module
                dims.value[]=join(d.dimensions," ")
                if state.edge!==nothing
                    source.value[],target.value[]=string.(state.edge)
                    A=d.maps[state.edge];coefficients.value[]=VIZ._editor_matrix_text(A)
                    rows.value[],columns.value[]=string.(Base.size(A))
                end
            else
                upsets.value[]=join((isempty(ids) ? "-" : join(ids," ") for ids in d.upsets),"; ")
                downsets.value[]=join((isempty(ids) ? "-" : join(ids," ") for ids in d.downsets),"; ")
                coefficients.value[]=VIZ._editor_matrix_text(d.matrix)
                rows.value[],columns.value[]=string.(Base.size(d.matrix))
            end
            export_disabled[]=!summary.valid
            code_text[]="";code_link[]="#";archive_link[]="#"
            status_text[]="$(summary.valid ? "Valid" : "Invalid") $(d.kind) draft. Revision $(summary.revision). Undo $(summary.undo_count); redo $(summary.redo_count). Live Julia."
            snapshot=VIZ.inspection_snapshot(session)
            old=fig[];fig[]=Base.invokelatest(_BASE_WGL_RENDER,snapshot;size,style)
            content[]=fig[];old===nothing || empty!(old)
            readout[]=_inspection_dom_readout(snapshot;style)
        finally;updating[]=false;end
        nothing
    end
    listen(_->commit() do
        d=VIZ.editor_draft(session)
        relation=VIZ._editor_parse_matrix(order.value[];max_entries=session.limits.max_vertices^2)
        d.kind===:module ? VIZ.edit_module!(session;order=relation,dimensions=integers(dims.value[])) : VIZ.edit_module!(session;order=relation)
    end,apply_order.value)
    listen(_->commit(()->VIZ.edit_module!(session;field=chosen_field())),apply_field.value)
    listen(_->commit() do
        d=VIZ.editor_draft(session)
        q=isempty(strip(vertex.value[])) ? nothing : integer(vertex)
        e=d.kind===:module && !isempty(strip(source.value[])) && !isempty(strip(target.value[])) ? (integer(source),integer(target)) : nothing
        VIZ.select_inspection!(session;vertex=q,edge=e)
    end,select.value)
    listen(_->commit() do
        A=VIZ._editor_parse_matrix(coefficients.value[];shape=(integer(rows),integer(columns)),max_entries=session.limits.max_entries,max_dimension=session.limits.max_dimension)
        d=VIZ.editor_draft(session)
        if d.kind===:module
            edge=(integer(source),integer(target))
            VIZ.edit_module!(session;map=(edge,A))
            VIZ.select_inspection!(session;edge)
        else
            VIZ.edit_module!(session;matrix=A)
        end
    end,apply_matrix.value)
    listen(_->commit(()->VIZ.edit_module!(session;map=((integer(source),integer(target)),nothing))),remove_map.value)
    listen(_->commit(()->VIZ.edit_module!(session;upsets=support_lists(upsets.value[]),downsets=support_lists(downsets.value[]))),apply_supports.value)
    listen(_->commit(()->VIZ.undo_module_edit!(session)),undo.value)
    listen(_->commit(()->VIZ.redo_module_edit!(session)),redo.value)
    listen(_->commit(()->VIZ.reset_inspection!(session)),reset.value)
    listen(_->commit(()->VIZ.close_inspection!(session)),close_button.value)
    listen(_->commit() do
        code_text[]=VIZ.editor_code(session)
        code_link[]="data:text/plain;base64,"*VIZ.base64encode(code_text[])
    end,generate.value)
    listen(_->commit() do
        payload=mktempdir() do dir
            path=joinpath(dir,"draft.json");VIZ.save_visual_session(path,session;caption="Finite input draft. Inspect validation before exporting mathematics.");read(path,String)
        end
        archive_link[]="data:application/json;base64,"*VIZ.base64encode(payload)
    end,archive.value)
    session_token[]=VIZ._on_inspection(session,_ -> refresh())
    refresh()
    label(t,c)=B.DOM.label(B.DOM.span(t;style="display:block;font-weight:600"),c;style="display:block;min-width:0;max-width:100%")
    row(children...)=B.DOM.div(children...;style="display:flex;flex-wrap:wrap;gap:$(style.gap)px;align-items:end;margin:$(style.gap)px 0")
    ismodule=VIZ.editor_draft(session).kind===:module
    dom=B.DOM.div(B.DOM.h2("Author a finite $(ismodule ? "module" : "presentation")"),
        B.DOM.p("Edit the tables and apply explicitly. Invalid drafts remain visible; only validated data can become a mathematical object. Matrices use rows separated by semicolons and exact fractions such as 1/3. Editing requires Julia."),
        row(label("Coefficient field",field),apply_field),
        B.DOM.details(B.DOM.summary("Edit the finite order and stalks"),label("Order matrix (1 means source <= target)",order),
            ismodule ? label("Stalk dimensions",dims) : B.DOM.p("Support membership is not completed automatically."),apply_order),
        ismodule ? row(label("Source vertex",source),label("Target vertex",target),label("Selected vertex",vertex),select) : row(label("Selected vertex",vertex),select),
        ismodule ? B.DOM.div() : B.DOM.details(B.DOM.summary("Edit indicator supports"),
            label("Upsets (vertex IDs; separate supports with semicolons; - is empty)",upsets),
            label("Downsets (vertex IDs; separate supports with semicolons; - is empty)",downsets),apply_supports),
        row(label("Matrix rows",rows),label("Matrix columns",columns)),label("Matrix coefficients",coefficients),
        row(apply_matrix,ismodule ? remove_map : B.DOM.div(),undo,redo,reset,close_button),
        B.DOM.p(error_text;id="$dom_id-error",role="alert"),B.DOM.p(status_text;id="$dom_id-status",role="status"),
        B.DOM.div(content;style="max-width:100%;overflow:auto",tabindex="0",var"aria-label"="Module editor diagram"),
        B.DOM.div(readout;id="$dom_id-readout"),
        B.DOM.details(B.DOM.summary("Export checked code or retain this draft"),
            row(generate,B.DOM.a("Download Julia construction";href=code_link,download="module.jl",id="$dom_id-code-download",
                    style=map(link->link=="#" ? "display:none" : "",code_link)),
                archive,B.DOM.a("Download saved draft";href=archive_link,download="draft.json",id="$dom_id-archive-download",
                    style=map(link->link=="#" ? "display:none" : "",archive_link))),
            B.DOM.pre(code_text;id="$dom_id-code",style="white-space:pre-wrap;overflow-wrap:anywhere"));
        id=dom_id,style="box-sizing:border-box;font-family:'$(style.font)',sans-serif;font-size:$(style.fontsize)px;line-height:1.45;padding:$(style.padding)px;background:white;color:#172b38;max-width:100%;min-width:0;overflow-wrap:anywhere")
    app=B.App(;title="TamerOp module editor") do client
        lock(browser_lock) do
            closed[] && return B.DOM.p("This editor has been closed.")
            if browser_client[]!==nothing && browser_client[]!==client
                existing=app_ref[].value;existing===nothing || (existing.session[]=browser_client[])
                return B.DOM.p("Call visualize(editor) again for an independent linked viewer.")
            end
            if browser_client[]===nothing
                browser_client[]=client;_inspection_keyboard_navigation!(client,dom)
                push!(observers,M.on(client.on_close) do ended;ended && dispose();nothing;end)
            end
            dom
        end
    end
    app_ref[]=WeakRef(app)
    _INSPECTION_UI_HANDLES[app]=(;session,figure=fig,dispose,closed,disabled,observers,session_token,browser_client,
        error_text,status_text,code_text,dom_id,controls=(;order,dims,field,source,target,coefficients,rows,columns,vertex,
            upsets,downsets,apply_order,apply_field,select,apply_matrix,undo,redo,reset,close=close_button,generate,archive))
    app
end
