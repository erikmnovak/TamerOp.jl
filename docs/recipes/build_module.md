# Build and validate a small module from spaces and maps

Suppose four vector spaces form a diamond: $1<2<4$ and $1<3<4$.
You know the four cover matrices and want an ordinary finite module whose
comparable maps are independent of the chosen path. This recipe enters the
data over the rationals, detects a failed relation and repairs it.

The input is already on a finite poset. No filtration or ambient parameter
classifier is inferred. For the surrounding choices, see
[constructing modules without a filtration](../tutorials/presentations.ipynb).

```julia
import TamerOp as OP
A = OP.Advanced
order = Bool[1 1 1 1; 0 1 0 1; 0 0 1 1; 0 0 0 1]
editor = OP.module_editor(; order, dimensions=ones(Int, 4),
    maps=Dict((1,2)=>[1;;], (1,3)=>[1;;], (2,4)=>[1;;], (3,4)=>[2;;]))
report = A.check_module_editor(editor)
@assert !report.valid
@assert report.witness.first_composite == [1;;]
@assert report.witness.second_composite == [2;;]
```

The order matrix includes identities and the transitive relation $1\leq4$;
the coefficient dictionary contains exactly the covers. Each matrix has target
dimension rows and source dimension columns. Here one path multiplies to 1
and the other to 2. The editor retains this invalid draft for inspection,
but `editor_object` refuses to construct a module from it.

Replace the last map and check the recovered composite:

```julia
OP.edit_module!(editor; map=((3,4), [1;;]))
A.check_module_editor(editor; throw=true)
M = OP.editor_object(editor)
@assert A.structure_map(M; source=1, target=4) == [1;;]
```

`M` is an independent `PModule` ready for stalk, rank or algebraic operations
on this finite base. Later edits to the draft do not change `M`. See the
[editor contract](../src/reference/visual_sessions.md#author-a-finite-module-or-presentation)
for field changes, undo and input limits.

To inspect the two paths visually, load CairoMakie and display
`OP.visualize(A.inspection_snapshot(editor))`. With WGLMakie loaded,
`OP.visualize(editor; backend=:wglmakie)` opens the live table editor; Julia
must remain running. The [guide](../tutorials/presentations.ipynb) shows this
workflow and an indicator-presentation alternative.
