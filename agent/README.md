# Using TamerOp with coding agents

TamerOp is a Julia environment for multiparameter persistence. It connects
constructions from data, invariants and comparisons with algebraic computations
on persistence modules. A persistence module assigns vector spaces to parameter
values and linear maps to comparable pairs of values. A **finite encoding**
describes those spaces and maps through a finite set of labels, so we can retain
the relationships needed for further computations.

Use this guide to find a working example for your task, then consult the
documentation of the operations you need. It is written for agents carrying out
an analysis and for people directing them. You can use it directly or install
the optional skill below.

## Choose a starting computation

| Your task | Start here | What the example explains |
| --- | --- | --- |
| Construct persistence from data with two parameters | [Data to a finite encoding](workflows/data_to_encoding.md) | How a point cloud and a value on each point determine a filtration, and how to query the resulting module. |
| Compute with a module described by spaces and maps | [Spaces and maps](workflows/module_maps.md) | How to supply a small mathematical example and interpret its dimensions and structure maps. |
| Inspect, draw and save a result | [Visualization and saving](workflows/visualization.md) | How to select a view, inspect its mathematical values and export a figure. |

Each guide links a complete Julia script with a known small answer. Choose the
closest example and adapt its input and mathematical choices to the user's
question. The [API reference](https://tamerop.com/collections/api.html) routes
other tasks to the relevant operations, including ordinary barcodes,
comparisons, Hom, Ext and resolutions. The
[Using TamerOp overview](https://tamerop.com/collections/using.html) explains how
the different computations fit together.

## Work in the analysis environment

Use Julia from the directory containing the analysis's `Project.toml`. The
`--project=.` option selects that environment, while `--startup-file=no` makes
the command independent of personal Julia startup code:

```sh
julia --startup-file=no --project=. -e 'import TamerOp; println("TamerOp ", Base.pkgversion(TamerOp)); println("Project: ", Base.active_project()); println("Guide: ", joinpath(pkgdir(TamerOp), "agent", "README.md"))'
```

This locates the guide accompanying the installed code. For an existing
analysis, retain its selected package version. The website follows the latest
TamerOp sources; local function documentation describes the installed version.
If an older installation has no `agent` directory, use its local operation help
and the corresponding repository revision, or update the package when that is
part of the user's task.

For a new analysis, create its directory and run the following there to install
the latest TamerOp:

```sh
julia --startup-file=no --project=. -e 'import Pkg; Pkg.add(url="https://github.com/erikmnovak/TamerOp.jl", rev="main")'
```

Keep the resulting `Project.toml` and `Manifest.toml` with the analysis. If Julia
is unavailable, the [installation guide](https://tamerop.com/start/install.html)
explains how to set it up. The visualization example gives its additional
plotting dependency at the point where it is needed.

## Find the calling form and mathematical meaning

Import the main operations as `OP` and the more detailed interface as `OA`:

```julia
import TamerOp as OP
import TamerOp.Advanced as OA

println(@doc OP.encode)
println(methods(OP.encode))
```

Replace `OP.encode` with the operation you need. Its documentation explains
accepted inputs, options and results; `methods` shows the available calling
forms. Read these before adapting an unfamiliar call. If Julia reports a
`MethodError`, compare the actual argument types with those calling forms.

For a calculation, identify what each parameter means, the homological degree
(for example, connected components or loops), and the coefficient field used
for linear algebra. Retain the finite encoding when later queries need spaces
and maps. A summary such as a barcode or a list of dimensions answers a
particular question and does not retain every relationship in the module.
The task guides explain these choices on concrete inputs.

Check the small example's answer before replacing its data. Then explain the
new result in terms of the user's question, keeping the parameter conventions
and any consequential approximation explicit. Distinguish a computed zero map
from the absence of a map between incomparable parameters. When an analysis
needs source cells or representatives, select a construction that retains
them; the relevant operation help describes that choice.

## Install the optional skill

The same `tamerop` skill works with Codex and Claude Code. It directs the agent
to this guide in the active Julia environment, so the mathematical instructions
stay with the package being used. From Julia in the analysis environment:

```julia
import TamerOp
include(joinpath(pkgdir(TamerOp), "agent", "install_skill.jl"))
TamerOpSkillInstaller.install("codex", pwd())
```

For Claude Code, use `"claude"` in the last line. The installer places the skill
in `.agents/skills/tamerop` for Codex or `.claude/skills/tamerop` for Claude Code
within the chosen project. It leaves an existing installation intact. To
replace one, review and remove that skill directory before installing again.

Ask Codex to use `$tamerop`, or invoke `/tamerop` in Claude Code, with the
calculation you want. For example:

> Use TamerOp to construct persistence from these points and their measured
> values. Follow its data workflow, explain the two parameters, and show how
> the connected components change.

The skill uses Julia through the agent's terminal. A session needs access to
that runtime and the analysis files to execute the examples. Agents without
skill support can read this guide and the selected workflow directly.

The skill format follows the published guidance for
[Codex](https://learn.chatgpt.com/docs/build-skills) and
[Claude Code](https://code.claude.com/docs/en/skills).
