# Native Bonito/Makie event checks. A real browser remains a separate acceptance
# step; these exercise rendered DOM serialization, selection and cleanup.

@testset "A41 live interval member selection and retained cycles" begin
    if Base.find_package("WGLMakie") === nothing
        @test_skip false
    else
        @eval import WGLMakie
        V = TamerOp.Visualization
        E = Base.get_extension(TamerOp,:TamerOpWGLMakieExt)
        B,M = WGLMakie.Bonito,WGLMakie.Makie
        d1 = sparse([-1 -1 0;1 0 -1;0 1 1])
        d2 = sparse(reshape([1,-1,1],3,1))
        G = DT.GradedComplex([[11,12,13],[21,22,23],[31]],[d1,d2],
            [(0,),(0,),(0,),(1,),(1,),(1,),(2,)])
        diagram = OP.persistence_diagram(G;representatives=true)
        s = V.inspection_session(diagram;dim=0)
        app = V.visualize(s;backend=:wglmakie,size=(1000,500))
        sibling = V.visualize(s;backend=:wglmakie,size=(900,480))
        ui,other = E._inspection_ui(app),E._inspection_ui(sibling)
        client = B.Session(B.NoConnection();asset_server=B.NoServer())
        try
            dom = B.session_dom(client,app;init=false)
            html = sprint(show,MIME"text/html"(),dom)
            for suffix in ("interval","member","apply-member","representative","reset","close")
                @test occursin(ui.dom_id*"-"*suffix,html)
            end
            @test !occursin("Bonito.Dropdown(",html)
            @test !isempty(B.serialize_binary(client,B.fused_messages!(client)))
            @test ui.representative_disabled[] && ui.member_disabled[]
            @test V.inspection_summary(s).listener_count == 2
            snapshot = V.inspection_snapshot(s)
            records = snapshot.metadata.interval_payload.records
            duplicate = only(filter(r -> r.multiplicity==2,records)).id
            essential = only(filter(r -> r.right_status===:essential,records)).id
            ui.controls.interval.value[] = duplicate
            @test V.inspection_selection(s).interval == duplicate
            @test V.inspection_selection(s).member === nothing
            @test !ui.member_disabled[] && !ui.representative_disabled[]
            @test ui.selected_bar[] == other.selected_bar[]
            @test ui.selected_point[] == other.selected_point[]
            builds = ui.rebuild_count[]
            ui.controls.representative.value[] = true
            @test occursin("member",lowercase(ui.error_text[]))
            @test !V.inspection_selection(s).representative
            @test !ui.controls.representative.value[]
            for text in ("abc","0","3")
                previous = V.inspection_selection(s)
                ui.controls.member.value[] = text
                ui.controls.apply_member.value[] = true
                @test !isempty(ui.error_text[])
                @test V.inspection_selection(s) == previous
            end
            ui.controls.member.value[] = "1"
            ui.controls.apply_member.value[] = true
            @test isempty(ui.error_text[])
            @test V.inspection_selection(s).member == 1
            @test V.inspection_snapshot(s).metadata.selected_representative === nothing
            ui.controls.representative.value[] = true
            rep = V.inspection_snapshot(s).metadata.selected_representative
            @test rep.available && rep.cycle.cell_ids == (11,12)
            @test other.controls.representative.value[]
            @test other.controls.member.value[] == "1"
            @test ui.rebuild_count[] == builds
            ui.controls.member.value[] = "2"
            ui.controls.apply_member.value[] = true
            @test !V.inspection_selection(s).representative
            @test V.inspection_snapshot(s).metadata.selected_representative === nothing
            ui.controls.representative.value[] = true
            @test V.inspection_snapshot(s).metadata.selected_representative.cycle.cell_ids != rep.cycle.cell_ids

            # Native clicks use the drawn essential lane, never a finite death
            # substituted into the underlying mathematical record.
            M.update_state_before_display!(ui.figure)
            view = V.inspection_snapshot(s).metadata.interval_view
            at = findfirst(==(essential),view.interval_ids)
            point = view.diagram_points[at]
            ax = ui.axes[].diagram
            events = M.events(ui.figure.scene)
            pixel = M.project(ax.scene,M.Point2d(point))+minimum(M.viewport(ax.scene)[])
            events.mouseposition[] = (Float64(pixel[1]),Float64(pixel[2]))
            @test M.is_mouseinside(ax.scene)
            before_hover = V.inspection_selection(s)
            @test ui.callbacks.hover(:diagram,point).interval_ids == (essential,)
            @test V.inspection_selection(s) == before_hover
            events.mousebutton[] = M.MouseButtonEvent(M.Mouse.left,M.Mouse.press)
            events.mousebutton[] = M.MouseButtonEvent(M.Mouse.left,M.Mouse.release)
            @test V.inspection_selection(s).interval == essential
            @test V.inspection_selection(s).member == 1
            @test !V.inspection_selection(s).representative
            @test ui.selected_point[] == [M.Point2d(point)]
            @test ui.selected_point[] == other.selected_point[]
            @test V.inspection_snapshot(s).metadata.interval_view.selected_record.death == Inf
            @test ui.rebuild_count[] == builds
            ui.controls.representative.value[] = true
            @test V.inspection_snapshot(s).metadata.selected_representative.bounding_chain === nothing
            ui.controls.reset.value[] = true
            @test V.inspection_selection(s).interval === nothing
            @test isempty(ui.selected_point[]) && isempty(other.selected_bar[])
            @test V.inspection_snapshot(s).metadata.selected_representative === nothing
            @test isempty(V.inspection_summary(s).listener_errors)
            close(client)
            @test ui.closed[] && !other.closed[]
            @test !V.inspection_summary(s).closed
            @test V.inspection_summary(s).listener_count == 1
            other.controls.close.value[] = true
            @test other.closed[] && V.inspection_summary(s).closed
            @test V.inspection_summary(s).listener_count == 0
            @test isempty(other.observers)
        finally
            close(client)
            E._dispose_inspection_ui!(app;close_session=false)
            E._dispose_inspection_ui!(sibling;close_session=false)
            V.close_inspection!(s)
        end
    end
end

@testset "A41 live interval clipping budget and coincident picks" begin
    if Base.find_package("WGLMakie") === nothing
        @test_skip false
    else
        @eval import WGLMakie
        V = TamerOp.Visualization
        E = Base.get_extension(TamerOp,:TamerOpWGLMakieExt)
        M = WGLMakie.Makie
        s = V.inspection_session([(0,2),(0,3),(10,11)];window=(0,1),max_intervals=2)
        app = V.visualize(s;backend=:wglmakie,size=(1000,500))
        ui = E._inspection_ui(app)
        try
            M.update_state_before_display!(ui.figure)
            @test ui.controls.interval.options[] == Any[0,1,2,3]
            view = V.inspection_snapshot(s).metadata.interval_view
            @test view.diagram_points[1] == view.diagram_points[2]
            ax = ui.axes[].diagram
            events = M.events(ui.figure.scene)
            point = view.diagram_points[1]
            pixel = M.project(ax.scene,M.Point2d(point))+minimum(M.viewport(ax.scene)[])
            events.mouseposition[] = (Float64(pixel[1]),Float64(pixel[2]))
            @test M.is_mouseinside(ax.scene)
            for id in (1,2,1)
                events.mousebutton[] = M.MouseButtonEvent(M.Mouse.left,M.Mouse.press)
                events.mousebutton[] = M.MouseButtonEvent(M.Mouse.left,M.Mouse.release)
                @test V.inspection_selection(s).interval == id
            end
            @test ui.member_disabled[]
            ui.controls.representative.value[] = true
            absent = V.inspection_snapshot(s).metadata.selected_representative
            @test !absent.available && absent.reason===:no_source_member_correspondence
            ui.controls.interval.value[] = 3
            @test V.inspection_selection(s).interval == 3
            @test isempty(ui.selected_bar[]) && isempty(ui.selected_point[])
            @test occursin("outside",ui.status_text[])
            @test V.inspection_snapshot(s).metadata.interval_view.selected_record.birth == 10
            @test isempty(V.inspection_summary(s).listener_errors)
        finally
            E._dispose_inspection_ui!(app)
            V.close_inspection!(s)
        end
        bounded = V.inspection_session([(0,2),(1,3),(2,4)];window=(0,4),max_intervals=1)
        limited = V.visualize(bounded;backend=:wglmakie,size=(900,500))
        limited_ui = E._inspection_ui(limited)
        try
            @test limited_ui.chart_data[].interval_ids == (1,)
            initial_builds = limited_ui.rebuild_count[]
            limited_ui.controls.interval.value[] = 3
            @test limited_ui.chart_data[].interval_ids == (3,)
            @test limited_ui.rebuild_count[] == initial_builds+1
            @test !isempty(limited_ui.selected_bar[])
            @test length(V.inspection_snapshot(bounded).metadata.interval_payload.records) == 3
        finally
            E._dispose_inspection_ui!(limited)
            V.close_inspection!(bounded)
        end
    end
end

@testset "A41 live slice scope commits certified finite and infinite endpoints" begin
    if Base.find_package("WGLMakie") === nothing
        @test_skip false
    else
        @eval import WGLMakie
        V = TamerOp.Visualization
        E = Base.get_extension(TamerOp,:TamerOpWGLMakieExt)
        B,M = WGLMakie.Bonito,WGLMakie.Makie
        F = TamerOp.CoreModules.QQField()
        K = TamerOp.CoreModules.coeff_type(F)
        A = TamerOp.Advanced
        FF0,MD0 = TamerOp.FiniteFringe,TamerOp.Modules
        EC0,PL0 = TamerOp.EncodingCore,TamerOp.PLPolyhedra
        R0 = TamerOp.Results
        line = (basepoint=(0,0),direction=(1,1))

        # The local window lies inside both closed squares. Its one censored
        # multiplicity-two interval must split into the actual global intervals.
        squares = TamerOp.encode([A.BoxUpset([0.0,0.0]),A.BoxUpset([1.0,1.0])],
            [A.BoxDownset([2.0,2.0]),A.BoxDownset([3.0,3.0])],K[1 0;0 1],
            A.EncodingOptions(;backend=:pl_backend,poset_kind=:signature,field=F))
        s = V.inspection_session(squares;box=([3//2,3//2],[7//4,7//4]),slice=line)
        app = V.visualize(s;backend=:wglmakie,size=(1000,500))
        ui = E._inspection_ui(app)
        client = B.Session(B.NoConnection();asset_server=B.NoServer())
        try
            dom = B.session_dom(client,app;init=false)
            @test occursin(ui.dom_id*"-slice-scope",sprint(show,MIME"text/html"(),dom))
            @test !isempty(B.serialize_binary(client,B.fused_messages!(client)))
            initial = V.inspection_snapshot(s)
            @test only(initial.metadata.slice_result.intervals).multiplicity == 2
            ui.controls.slice.interval.value[] = 1
            before = V.inspection_snapshot(s)
            ui.controls.slice.scope.value[] = :global
            @test V.inspection_snapshot(s) === before
            @test V.inspection_selection(s).slice_scope === :window
            ui.controls.slice.apply.value[] = true
            @test isempty(ui.error_text[])
            @test V.inspection_selection(s).slice_scope === :global
            @test V.inspection_selection(s).interval === nothing
            global_result = V.inspection_snapshot(s).metadata.slice_result
            @test global_result.endpoint_semantics === :decorated_global
            @test global_result.essential_status === :certified
            @test Set((r.birth,r.death,r.multiplicity) for r in global_result.intervals) ==
                Set(((0,2,1),(1,3,1)))
            @test all(r -> !r.left_clipped && !r.right_clipped,global_result.intervals)
            ui.controls.slice.interval.value[] = 2
            @test ui.slice.selected_region[] == [M.Point2d(1.5,1.5),M.Point2d(1.75,1.75)]
            ui.controls.slice.scope.value[] = :window
            ui.controls.slice.apply.value[] = true
            @test isempty(ui.error_text[])
            @test V.inspection_selection(s).interval === nothing
            local_result = V.inspection_snapshot(s).metadata.slice_result
            @test local_result.endpoint_semantics === :decorated_finite_window
            @test local_result.essential_status === :not_inferred
            @test only(local_result.intervals).left_clipped && only(local_result.intervals).right_clipped
            @test isempty(ui.slice.selected_region[])
            @test isempty(V.inspection_summary(s).listener_errors)
        finally
            close(client)
            E._dispose_inspection_ui!(app)
            V.close_inspection!(s)
        end

        # An event-free full-plane classifier certifies both infinite tails.
        # A vertical line additionally guards against 0*Inf in region drawing.
        hp = PL0.HPoly(2,zeros(K,0,2),K[],nothing,falses(0),zero(K))
        pi = PL0.PLEncodingMap(2,[BitVector()],[BitVector()],[hp],[(0,0)])
        P = FF0.ProductOfChainsPoset((1,))
        module0 = MD0.PModule{K}(P,[1],Dict{Tuple{Int,Int},Matrix{K}}();field=F)
        full = R0.EncodingResult(P,module0,EC0.compile_encoding(P,pi))
        infinite = V.inspection_session(full;box=([-1,-1],[1,1]),
            slice=(basepoint=(0,0),direction=(0,1)))
        infinite_app = V.visualize(infinite;backend=:wglmakie,size=(1000,500))
        inf_ui = E._inspection_ui(infinite_app)
        try
            inf_ui.controls.slice.scope.value[] = :global
            inf_ui.controls.slice.apply.value[] = true
            @test isempty(inf_ui.error_text[])
            result = V.inspection_snapshot(infinite).metadata.slice_result
            r = only(result.intervals)
            @test (r.birth,r.death) == (-Inf,Inf)
            @test result.essential_status === :certified
            inf_ui.controls.slice.interval.value[] = r.id
            @test inf_ui.slice.selected_region[] == [M.Point2d(0,-1),M.Point2d(0,1)]
            @test all(p -> all(isfinite,p),inf_ui.slice.selected_bar[][])
            @test all(p -> all(isfinite,p),inf_ui.slice.selected_point[][])
            @test length(inf_ui.slice.selected_point[][]) == 1
            @test isempty(V.inspection_summary(infinite).listener_errors)
        finally
            E._dispose_inspection_ui!(infinite_app)
            V.close_inspection!(infinite)
        end

        # The same UI must preserve the previous result when the encoding does
        # not represent a lower tail; it cannot silently treat missing data as 0.
        P = FF0.ProductOfChainsPoset((2,2))
        module0 = MD0.PModule{K}(P,ones(Int,4),
            Dict(edge=>reshape(K[1],1,1) for edge in FF0.cover_edges(P));field=F)
        grid = EC0.GridEncodingMap(P,([0,2],[0,2]))
        partial = R0.EncodingResult(P,module0,EC0.compile_encoding(P,grid))
        bounded = V.inspection_session(partial;box=([1//2,1//2],[3//2,3//2]),slice=line)
        bounded_app = V.visualize(bounded;backend=:wglmakie,size=(1000,500))
        bounded_ui = E._inspection_ui(bounded_app)
        try
            bounded_ui.controls.slice.interval.value[] = 1
            before,state = V.inspection_snapshot(bounded),V.inspection_selection(bounded)
            bounded_ui.controls.slice.scope.value[] = :global
            bounded_ui.controls.slice.apply.value[] = true
            @test occursin("unrepresented",bounded_ui.error_text[])
            @test V.inspection_snapshot(bounded) === before
            @test V.inspection_selection(bounded) == state
            @test !isempty(bounded_ui.slice.selected_region[])
            @test isempty(V.inspection_summary(bounded).listener_errors)
            @test V.check_inspection_session(bounded).valid
        finally
            E._dispose_inspection_ui!(bounded_app)
            V.close_inspection!(bounded)
        end
    end
end
