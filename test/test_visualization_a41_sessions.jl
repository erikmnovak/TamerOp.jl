# Linked interval sessions use the same retained records as static views. The
# representative oracle is the independently tested ordinary owner accessor.

function _a41_session_triangle(;order=:sublevel,representatives=true)
    d1 = sparse([-1 -1 0;1 0 -1;0 1 1])
    d2 = sparse(reshape([1,-1,1],3,1))
    grade = order === :sublevel ? identity : x -> 2-x
    G = DT.GradedComplex([[11,12,13],[21,22,23],[31]],[d1,d2],
        [(grade(0),),(grade(0),),(grade(0),),(grade(1),),(grade(1),),(grade(1),),(grade(2),)])
    return OP.persistence_diagram(G;order,representatives)
end

@testset "A41 interval sessions preserve all IDs under clipping and budgets" begin
    V = TamerOp.Visualization
    bars = Dict((0,1)=>2,(2,Inf)=>1,(3,4)=>1)
    s = V.inspection_session(bars;window=(1//2,7//2),max_intervals=1)
    try
        snap = V.inspection_snapshot(s)
        @test s isa V.IntervalInspectionSession
        @test V.inspection_selection(s) ==
            (interval=nothing,member=nothing,representative=false,revision=0)
        @test snap.metadata.interval_payload.order === :sublevel
        @test snap.metadata.interval_view.total_groups == 3
        @test snap.metadata.interval_view.total_multiplicity == 4
        @test snap.metadata.interval_view.interval_ids == (1,)
        @test snap.metadata.interval_view.all_interval_ids == (1,2,3)
        @test V.inspection_summary(s).window == (1//2,7//2)
        @test V.inspection_summary(s).max_intervals == 1
        @test !V.inspection_summary(s).representative_available
        for id in (2,3,1)
            V.select_inspection!(s;interval=id)
            snap = V.inspection_snapshot(s)
            @test snap.metadata.interval_view.interval_ids == (id,)
            @test snap.metadata.interval_view.selected_record.id == id
            @test snap.metadata.interval_view.total_groups == 3
            @test snap.metadata.interval_payload.records[id].multiplicity == (id == 1 ? 2 : 1)
            @test snap.metadata.selected_representative === nothing
            @test V.inspection_selection(s).member === nothing
            @test V.check_visual_spec(snap).valid
            @test V.check_inspection_session(s).valid
        end
        # Raw endpoint multiplicities do not invent source class identities.
        V.select_inspection!(s;representative=true)
        unavailable = V.inspection_snapshot(s).metadata.selected_representative
        @test !unavailable.available
        @test unavailable.reason === :no_source_member_correspondence
        @test unavailable.cycle === nothing
        before = V.inspection_snapshot(s)
        @test_throws ArgumentError V.select_inspection!(s;member=1)
        @test V.inspection_snapshot(s) === before
        V.reset_inspection!(s)
        @test V.inspection_selection(s).interval === nothing
        @test !V.inspection_selection(s).representative
        @test V.inspection_snapshot(s).metadata.selected_representative === nothing
        @test V.inspection_summary(s).window == (1//2,7//2)
    finally
        V.close_inspection!(s)
    end
    # An offscreen record still has a selectable exact readout.
    offscreen = V.inspection_session(bars;window=(5,6),max_intervals=1)
    try
        V.select_inspection!(offscreen;interval=1)
        view = V.inspection_snapshot(offscreen).metadata.interval_view
        @test view.selected_record.birth == 0 && view.selected_record.death == 1
        @test !(1 in view.interval_ids)
        @test view.offscreen_groups == 2
        @test V.check_visual_spec(V.inspection_snapshot(offscreen)).valid
    finally
        V.close_inspection!(offscreen)
    end
end

@testset "A41 interval representatives require a distinct original member" begin
    V = TamerOp.Visualization
    for order in (:sublevel,:superlevel)
        diagram = _a41_session_triangle(;order)
        s = V.inspection_session(diagram;dim=0,max_intervals=1)
        try
            records = V.inspection_snapshot(s).metadata.interval_payload.records
            duplicates = only(r.id for r in records if r.multiplicity == 2)
            essential = only(r.id for r in records if any(m -> m.kind === :essential,r.members))
            @test V.inspection_summary(s).representative_available
            V.select_inspection!(s;interval=duplicates)
            @test V.inspection_selection(s).member === nothing
            before = V.inspection_snapshot(s)
            @test !V.check_inspection_selection(s;representative=true).valid
            @test_throws ArgumentError V.select_inspection!(s;representative=true)
            @test V.inspection_snapshot(s) === before
            cycles = []
            for member in 1:2
                V.select_inspection!(s;member,representative=true)
                selected = V.inspection_snapshot(s).metadata.selected_representative
                source = records[duplicates].members[member]
                expected = OP.persistence_representative(diagram;
                    dim=source.dim,kind=source.kind,index=source.index)
                @test selected == expected
                @test selected.available
                @test selected.cycle.cell_ids != ()
                @test selected.bounding_chain !== nothing
                @test V.inspection_selection(s).member == member
                @test V.inspection_snapshot(s).metadata.interval_view.selected_interval == duplicates
                @test V.check_visual_spec(V.inspection_snapshot(s)).valid
                push!(cycles,selected.cycle.cell_ids)
            end
            @test cycles[1] != cycles[2]
            V.select_inspection!(s;interval=essential)
            @test V.inspection_selection(s).member == 1
            @test !V.inspection_selection(s).representative
            @test V.inspection_snapshot(s).metadata.selected_representative === nothing
            V.select_inspection!(s;representative=true)
            selected = V.inspection_snapshot(s).metadata.selected_representative
            @test selected.kind === :essential
            @test selected.bounding_chain === nothing
            @test selected.order === order
            @test V.check_inspection_session(s).valid
            final_snapshot = V.inspection_snapshot(s)
            V.close_inspection!(s)
            @test s.object === nothing
            @test V.inspection_snapshot(s) === final_snapshot
            @test V.inspection_snapshot(s).metadata.selected_representative == selected
            @test V.check_inspection_session(s).valid
        finally
            V.close_inspection!(s)
        end
    end

    plain = V.inspection_session(_a41_session_triangle(;representatives=false);dim=1)
    try
        V.select_inspection!(plain;interval=1,representative=true)
        @test V.inspection_selection(plain).member == 1
        @test !V.inspection_summary(plain).representative_available
        report = V.inspection_snapshot(plain).metadata.selected_representative
        @test !report.available && report.reason === :not_retained
        @test report.cycle === nothing
    finally
        V.close_inspection!(plain)
    end
end

@testset "A41 interval session lifecycle and validation are transactional" begin
    V = TamerOp.Visualization
    s = V.inspection_session([(0,1),(0,1),(2,3)])
    calls = Ref(0)
    token = V._on_inspection(s,current -> (calls[] += 1))
    failure = V._on_inspection(s,current -> error("A41 listener error"))
    reentrant = V._on_inspection(s,current -> V.select_inspection!(current;interval=1))
    try
        before = V.inspection_snapshot(s)
        for request in ((interval=10,),(interval=-1,),(interval=true,),
                        (representative=true,),(representative=:yes,),(member=1,))
            @test !V.check_inspection_selection(s;request...).valid
            @test_throws ArgumentError V.select_inspection!(s;request...)
            @test V.inspection_snapshot(s) === before
            @test V.inspection_summary(s).revision == 0
            @test calls[] == 0
        end
        V.select_inspection!(s;interval=2)
        @test calls[] == 1
        @test V.inspection_selection(s).interval == 2
        @test length(V.inspection_summary(s).listener_errors) == 2
        @test !V.inspection_summary(s).updating
        V._off_inspection!(s,failure)
        V._off_inspection!(s,reentrant)
        V.select_inspection!(s;interval=0)
        @test calls[] == 2
        @test V.inspection_selection(s).interval === nothing
        @test V.check_inspection_session(s).valid
        retained = V.inspection_snapshot(s)
        V.close_inspection!(s)
        @test calls[] == 3
        @test V.inspection_summary(s).closed
        @test V.inspection_summary(s).listener_count == 0
        @test V.inspection_snapshot(s) === retained
        @test V.close_inspection!(s) === s
        @test_throws ArgumentError V.select_inspection!(s;interval=1)
        @test_throws ArgumentError V.reset_inspection!(s)
        @test_throws ArgumentError V._on_inspection(s,identity)
        @test V.check_inspection_session(s).valid
    finally
        V._off_inspection!(s,token)
        V.close_inspection!(s)
    end
    for window in ((2,1),(0,Inf),(false,1),(0,1,2))
        @test_throws ArgumentError V.inspection_session([(0,1)];window)
    end
    for max_intervals in (0,-1,true,1.5)
        @test_throws ArgumentError V.inspection_session([(0,1)];max_intervals)
    end
    empty_session = V.inspection_session(Tuple{Int,Int}[])
    try
        @test V.inspection_summary(empty_session).total_groups == 0
        @test V.inspection_summary(empty_session).total_multiplicity == 0
        @test_throws ArgumentError V.select_inspection!(empty_session;interval=1)
        @test V.check_inspection_session(empty_session).valid
    finally
        V.close_inspection!(empty_session)
    end
end
