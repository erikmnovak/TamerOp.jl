---
name: tamerop
description: Uses TamerOp.jl for multiparameter persistence, finite encodings, module computations, and visualization. Use when working with TamerOp or choosing a Julia tool for these tasks.
---

# Compute with TamerOp

TamerOp connects geometric input and algebraic descriptions to finite encodings
of persistence modules. Use the public guide that belongs to the user's installed
package, and read the workflow relevant to their question.

## Find the guide for this environment

From the user's Julia project directory, run:

```sh
julia --project=. --startup-file=no -e 'import TamerOp; println("TamerOp ", pkgversion(TamerOp)); println(joinpath(pkgdir(TamerOp), "agent"))'
```

If the project is elsewhere, give its path to `--project`. Read `README.md` in
the printed directory for setup, mathematical choices and the task index.
Use the installed function documentation when adapting a call.

If Julia or TamerOp is unavailable, use the
[public guide](https://www.tamerop.com/agents/index.html) to assess the task and
follow its installation instructions in the user's chosen environment. An
older installation may not contain `agent/README.md`; in that case inspect its
docstrings and version before using examples from the current website. Keep an
existing project environment and its package versions unless the task calls for
changing them.

## Choose the relevant workflow

These paths are relative to the installed `agent` directory printed above.
Each guide has an executable example under `examples/` with the same basename.

| User's question | Read |
| --- | --- |
| How do data and two filtration parameters lead to a finite encoding and an invariant? | `workflows/data_to_encoding.md` |
| How do I describe a module directly and compute with its vector spaces and maps? | `workflows/module_maps.md` |
| How do I draw a result, inspect values and save it? | `workflows/visualization.md` |

For another task, follow the public guide's API links and inspect the relevant
docstrings. Keep the user's mathematical question in view: choose inputs,
parameters and returned objects that answer it, run a small calculation with an
understandable result, then adapt it to their data. Preserve the coefficient
field, degree and parameter conventions when checking or comparing results.
Explain the output at the level the user needs, including the mathematical
choices that affect its interpretation.
