# Executable code is generated from checked scalar data; importing saved views
# never evaluates it. The source is for the user to review and run explicitly.
_editor_literal(x::CoreModules.FpElem)=string(x.val)
_editor_literal(x::Integer)="big\"$(x)\""
_editor_literal(x::Rational)="big\"$(numerator(x))\"//big\"$(denominator(x))\""
_editor_literal(x::AbstractFloat)=repr(x)
function _editor_matrix_code(A)
    values=join((_editor_literal(x) for x in A),", ")
    "reshape([C.coerce(field, x) for x in [$values]], $(size(A,1)), $(size(A,2)))"
end
function _editor_field_code(field)
    field isa CoreModules.QQField && return "C.QQField()"
    field isa CoreModules.PrimeField && return "C.Fp($(field.p))"
    T=CoreModules.coeff_type(field)
    T in (Float16,Float32,Float64) || throw(ArgumentError("Code export supports Float16/32/64 real fields; use owned view data for other precision types."))
    "C.RealField($T; atol=$(repr(field.atol)), rtol=$(repr(field.rtol)))"
end
"""
    editor_code(editor) -> String

Generate a self-contained Julia construction of the current validated PModule
or finite FringeModule using exact scalar literals and qualified owner APIs.
No text entered in the editor is evaluated. Save with `write(path, editor_code(e))`
and review/run it explicitly. Field changes are coefficient reinterpretations.
Code export supports QQ, prime fields and Float16/32/64 real fields.
"""
function editor_code(s::ModuleEditorSession)
    lock(s.lock) do
        object=editor_object(s);d=s.draft
        code="import TamerOp\nC = TamerOp.CoreModules\nFF = TamerOp.FiniteFringe\nfield = "*_editor_field_code(d.field)*"\nK = C.coeff_type(field)\n"
        n=size(d.order,1)
        code*="order = reshape(Bool["*join(Int.(vec(d.order)),", ")*"], $n, $n)\nP = FF.FinitePoset(order)\n"
        if d.kind===:module
            code*="maps = Dict{Tuple{Int,Int},Matrix{K}}()\n"
            for edge in sort!(collect(keys(d.maps)))
                A=Modules.structure_map(object;source=edge[1],target=edge[2])
                code*="maps[$edge] = Matrix{K}("*_editor_matrix_code(A)*")\n"
            end
            code*="object = TamerOp.Modules.PModule{K}(P, Int["*join(d.dimensions,", ")*"], maps; field)\nTamerOp.Modules.check_module_functoriality(object; throw=true)\n"
        else
            code*="upsets = "*repr(d.upsets)*"\ndownsets = "*repr(d.downsets)*"\n"
            code*="U = [FF.Upset(P, BitVector([q in ids for q in 1:$n])) for ids in upsets]\nD = [FF.Downset(P, BitVector([q in ids for q in 1:$n])) for ids in downsets]\n"
            code*="matrix = Matrix{K}("*_editor_matrix_code(FiniteFringe.fringe_coefficients(object))*")\nobject = FF.FringeModule{K}(P, U, D, matrix; field)\nFF.check_fringe_module(object; throw=true)\n"
        end
        code*"object\n"
    end
end

"""
    module_editor(saved::SavedVisualSession; state=saved.selected, kwargs...)

Reopen the declarative input draft retained by an editor snapshot. Revalidate
in Julia and start a new bounded edit history; no old callbacks or undo history
are restored. Invalid drafts remain invalid. Capture with
`save_visual_session(path, editor)` and read with `load_visual_session(path)`.
"""
function module_editor(saved::SavedVisualSession;state=saved.selected,kwargs...)
    frame=inspection_snapshot(saved;state)
    d=get(frame.metadata,:editor_draft,nothing)
    d isa NamedTuple || throw(ArgumentError("The saved state contains no module-editor draft."))
    haskey(d,:kind) || throw(ArgumentError("Missing editor draft kind."))
    d.kind in (:module,:presentation) || throw(ArgumentError("Unknown editor draft kind."))
    fields=Base.structdiff(d,(;kind=d.kind))
    s=module_editor(;fields...,kwargs...)
    s.draft.kind===d.kind || throw(ArgumentError("Saved editor kind does not match its input fields."))
    selected=get(frame.metadata,:selection,(;vertex=nothing,edge=nothing))
    select_inspection!(s;vertex=selected.vertex,edge=selected.edge)
    s
end
