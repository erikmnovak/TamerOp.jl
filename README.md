# TamerOp

Toolkit for Algebraic Module Encodings over $\mathbb{R}^n$ and Other Posets.

[Documentation](https://tamerop.com/) ·
[Installation](https://tamerop.com/start/install.html) ·
[Using TamerOp](https://tamerop.com/collections/using.html) ·
[Coding agents](agent/README.md) ·
[Benchmark results](https://tamerop.com/benchmarks/index.html)

TamerOp is a Julia library for studying persistence modules: vector spaces
attached to parameters, with linear maps describing how their vectors change
as the parameters increase. A **finite encoding** describes such a module
using spaces and maps on a finite partially ordered set of labels, together
with an assignment from the original parameters to those labels. We can use
this finite encoding to recover spaces and maps, compare modules, and compute
invariants and algebraic constructions.

The project began with Ezra Miller's
[Homological algebra of modules over posets](https://arxiv.org/abs/2008.00063).
Its theory of *tameness* explains when a module admits these finite descriptions.
TamerOp connects this algebra with computations starting from points, images,
graphs and filtered complexes. The [Using TamerOp overview](https://tamerop.com/collections/using.html)
shows how those inputs lead to finite encodings and the questions we can ask
of them.

## Install TamerOp

Use Julia 1.12 or a compatible later 1.x version. In Julia, create an environment
for your work and install the latest version of TamerOp:

```julia
import Pkg

workdir = joinpath(homedir(), "tamerop-work")
mkpath(workdir)
cd(workdir)
Pkg.activate(workdir)
Pkg.add(url="https://github.com/erikmnovak/TamerOp.jl", rev="main")
```

The environment's `Project.toml` and `Manifest.toml` record the packages and
versions used by your analysis. Keep them with your code. The
[installation guide](https://tamerop.com/start/install.html) explains Julia setup,
reopening an environment, optional plotting packages and downloadable notebooks.

## Your first computation

A small image gives us a hole whose lifetime we can see by hand:

```julia
import TamerOp as OP

values = [0 0 0;
          0 5 0;
          0 0 0]

diagram = OP.cubical_persistence(values)
println(OP.finite_intervals(diagram; dim=1))
```

The result is:

```text
[(0, 5)]
```

Each matrix entry gives the appearance value of a square. At parameter `0`,
the eight outer squares form a ring. At parameter `5`, the central square
fills its hole. The interval `[0,5)` records that one-dimensional homology class.
Here `dim=1` asks about loops around holes; `dim=0` asks about connected
components. This calculation uses coefficients in F₂, the field with two
elements.

The [ring lesson](https://tamerop.com/tutorials/ring.html) follows the squares and
their barcode as the parameter changes. Continue with
[finite encodings](https://tamerop.com/finite_encodings.html) to see how we retain
spaces and maps when several parameters vary, or use the
[API reference](https://tamerop.com/collections/api.html) to look up an operation.

If you work with a coding agent, the [agent guide](agent/README.md) gives it
task examples, local API lookup and an optional skill for Codex or Claude Code.
The guide and its runnable examples accompany the package, so an agent can
read the instructions for the version used by your analysis.

## The three repositories

| Repository | What lives there |
| --- | --- |
| **TamerOp.jl** (this repository) | The Julia library, its tests and package metadata. |
| [TamerOp-site](https://github.com/erikmnovak/TamerOp-site/blob/main/README.md) | The lessons, guides, figures and sources for [tamerop.com](https://tamerop.com/). |
| [TamerOp-benchmarks](https://github.com/erikmnovak/TamerOp-benchmarks/blob/main/README.md) | Executable comparisons, frozen inputs and results used in the benchmark reports. |

For questions and examples, use [Discussions](https://github.com/erikmnovak/TamerOp.jl/discussions).
Report bugs or propose changes in [Issues](https://github.com/erikmnovak/TamerOp.jl/issues),
identifying the project concerned. The [contributor guide](CONTRIBUTING.md)
explains how to propose a library change and check it. Contributions to lessons
and benchmarks have their own short guides in the corresponding repositories.

## Releases and citation

[CHANGELOG.md](CHANGELOG.md) describes changes to the library, and
[RELEASING.md](RELEASING.md) documents the maintainer's release procedure.
TamerOp is distributed under the [MIT license](LICENSE).

If TamerOp contributes to your research, cite **Erik Novak, TamerOp.jl: Toolkit
for Algebraic Module Encodings over R^n and Other Posets**, with the
[repository URL](https://github.com/erikmnovak/TamerOp.jl) and the version or Git
commit used. [CITATION.cff](CITATION.cff) supplies the citation metadata for
GitHub's **Cite this repository** menu. Keep the analysis environment's
`Project.toml` and `Manifest.toml` with your results so others can recover its
dependency versions.
