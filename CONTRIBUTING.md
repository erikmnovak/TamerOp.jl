# Contributing to TamerOp

Contributions can start with a mathematical question, an unclear explanation,
or a small example whose answer does not look right. You do not need to know
the whole library to help improve one part of it.

TamerOp is built around finite encodings of persistence modules. The
[finite-encoding introduction](docs/finite_encodings.md) explains that common
object and the questions it supports. Keep this connection visible when
proposing new capabilities or explaining existing ones.

## Ask a question or report a problem

Use the [GitHub issues page](https://github.com/erikmnovak/TamerOp.jl/issues)
for questions, bug reports, and feature proposals. Check existing issues
first; a related report may already contain an answer or useful context.

For a bug report, include:

- A small, complete Julia example that reproduces the problem, with its input.
- What you expected, what happened, and the full error message if one appeared.
- Your Julia version, operating system, and TamerOp version or Git commit.
- Relevant choices such as the coefficient field, filtration, parameter window,
  and optional packages.

At the Julia prompt, `VERSION` shows the Julia version. Run `import Pkg`
followed by `Pkg.status()` in your analysis environment to list package versions.
For a source checkout, `git rev-parse HEAD` in a terminal identifies the commit.

If the issue concerns a mathematical result, explain how you obtained the
expected answer. A hand calculation, a cited theorem with its assumptions,
or an independent computation helps us understand the discrepancy. A
performance report should follow the [benchmarking manual](docs/benchmarking.md):
state the requested output, input size, thread count, compilation state and
which mathematical results or caches were already available.

For a feature proposal, start with the question you want to answer and an
example input and desired result. Discuss substantial algorithm or API changes
in an issue before spending time on an implementation.

## Set up a development checkout

Use Julia 1.12, the version used by the current tests, and Git. If you only
want to use the library, follow the [installation guide](README.md#install-tamerop)
instead.

On GitHub, use **Fork** to create a copy of the repository in your account.
Then run these commands in a **terminal**, replacing `YOUR-USERNAME` with your
GitHub username:

```sh
git clone https://github.com/YOUR-USERNAME/TamerOp.jl.git
cd TamerOp.jl
git switch -c improve-docs
julia --project=. -e "import Pkg; Pkg.instantiate()"
```

Choose a branch name that describes your change. The final command installs
the dependencies for this checkout; its first run can take several minutes.
The dot in `--project=.` selects the project in the current directory.

To explore the checkout interactively, start Julia from that directory:

```sh
julia --project=.
```

Then, at the **Julia prompt**:

```julia
import TamerOp as OP
pathof(OP)
```

The printed path should point to `src/TamerOp.jl` inside your clone. After
editing source files, start a fresh Julia session to load the revised code.

## Check the part you changed

The [testing guide](docs/testing.md) describes the maintained test runner.
From the repository root, list the available test files with:

```sh
julia --project=. test/runtests.jl --list
```

Select the file that exercises your change. For example, these commands run
the package and API contract checks, and a selected family of ingestion tests:

```sh
julia --project=. test/runtests.jl --file=test_contracts.jl
julia --project=. test/runtests.jl --file=test_data_pipeline.jl --prefix=A14
```

The second command is an example of selecting a related test family; it is
not a required check for every contribution. The testing guide explains
selection by field, threaded runs, and checks requiring optional extensions.
Report the commands you actually ran, their results, and any skipped checks.

For an algorithm change, add a small test with an independently known answer.
When the claim concerns a persistence module, check its maps as well as its
dimensions. For floating-point results, explain the tolerance. Place tests
with the relevant subsystem so another contributor can find and rerun them.
Keep required checks self-contained in the public repository.

For documentation edits, check links, mathematical assumptions, and the
explanation around each example. Run examples whose executable content you
change. A prose correction does not require rerunning the entire test suite.

## Measure performance changes

Read the [benchmarking manual](docs/benchmarking.md) before planning or reporting
performance work. Its primary comparison measures compiled code recomputing the
mathematics without prior results; startup and reuse have separate measurements.
Use its study brief to define outputs, valid inputs, cache state, correctness
checks and evidence. External tools may use different algorithms for the same
verified answer. The guide also plans a future reproducible benchmark release;
local benchmark and audit files remain excluded from the public package.

## Write explanations that teach

Follow the [writing guide](docs/writing.md). Begin with the reader's question,
explain the mathematical object being used, and introduce terms and notation
when they become useful. Show what a result means and why the next step follows.

Use the public API in introductory examples. Explain the returned object
before listing more operations. Preserve the distinction between a finite
encoded module and summaries of it, and state the assumptions under which
each computation is interpreted.

## Send a pull request

Keep a contribution focused on one purpose, with its related tests and
documentation. Inspect `git diff` before committing, then stage the files
you intend to include. Commit them with a short description of the change
and push your branch to your fork.

Open a pull request from that branch to `main` in
[the TamerOp repository](https://github.com/erikmnovak/TamerOp.jl/pulls).
Explain the problem, the resulting behavior or explanation, and how you
checked it. Link a related issue if there is one. A draft pull request is
welcome when you want feedback before the work is complete.

Discuss disagreements in terms of the example, mathematical assumptions,
and evidence. Questions about an explanation are useful feedback: they help
make the library approachable to its next reader.
