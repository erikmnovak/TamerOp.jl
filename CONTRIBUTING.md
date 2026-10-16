# Contributing to TamerOp

A contribution can begin with an example whose answer seems wrong, a faster
computation, or a mathematical question the library does not yet address.
Use [Discussions](https://github.com/erikmnovak/TamerOp.jl/discussions) for questions
and examples, and [Issues](https://github.com/erikmnovak/TamerOp.jl/issues) for bug
reports and proposals across the three projects.

This guide covers the Julia library. For teaching material, see
[TamerOp-site's contribution guide](https://github.com/erikmnovak/TamerOp-site/blob/main/CONTRIBUTING.md);
for a comparison or measurements from your computer, see
[TamerOp-benchmarks' contribution guide](https://github.com/erikmnovak/TamerOp-benchmarks/blob/main/CONTRIBUTING.md).
Pull requests belong in the repository they change. Link the central issue when
there is one.

## Describe the change

For a bug, provide a small Julia example with its input, the expected answer,
the actual result or full error, and your Julia and TamerOp versions. Include
relevant choices such as the coefficient field, parameter range and optional
packages. `VERSION` and `import Pkg; Pkg.status()` show the Julia and package versions;
`git rev-parse HEAD` identifies a source checkout.

For a mathematical discrepancy, explain how you obtained the expected answer.
A hand calculation, a theorem with its assumptions, or an independent
computation gives us something concrete to investigate together. For a new
feature, describe the question, an example input and the desired result.
Discuss substantial algorithm or API changes in an issue before investing in
the implementation. Small corrections can arrive directly as pull requests.

## Work from a fork

Fork this repository on GitHub, then run the following in a terminal, replacing
`YOUR-USERNAME` with your GitHub username:

```sh
git clone https://github.com/YOUR-USERNAME/TamerOp.jl.git
cd TamerOp.jl
git switch -c describe-your-change
julia --project=. -e 'import Pkg; Pkg.instantiate()'
```

Use Julia 1.12 or a compatible later 1.x version. Start an interactive session
with `julia --project=.`; after `import TamerOp as OP`, `pathof(OP)` should point
inside your clone. Start a fresh Julia session after editing source files.

## Check the affected computation

The [testing guide](test/README.md) explains how to select test files, fields and
optional integrations. From the repository root:

```sh
julia --project=. test/runtests.jl --list
julia --project=. test/runtests.jl --file=test_contracts.jl
```

Choose the additional files that exercise your change. Add a small independent
check for a new algorithm or a regression: for a persistence module, check its
maps as well as dimensions; for approximate results, explain the tolerance.
Keep required tests in the public repository. Report the commands you ran and
their results, including any failures or skipped checks. A focused contribution
needs the relevant checks; full release verification is a maintainer task.

For a performance improvement, provide a reproducible comparison on the same
mathematical task. Explain the requested output, inputs, software versions,
threads, preparation and any reused results. Check the answers before comparing
timings. The [benchmark protocol](https://github.com/erikmnovak/TamerOp-benchmarks/blob/main/PROTOCOL.md)
provides the measurement procedure and evidence requirements.

For docstrings and examples, introduce unfamiliar terms, explain the returned
object and state the mathematical assumptions where they matter. Execute any
example whose code you change. Lessons and longer explanations belong in
[TamerOp-site](https://github.com/erikmnovak/TamerOp-site/blob/main/README.md).

## Review and integration

Push your branch to your fork and open a pull request to `main`. Describe the
problem, what your change does, and how you checked it. A draft pull request
is welcome for work that would benefit from discussion.

Erik Novak reviews contributions and makes the final decisions about the API,
integration and releases. He may help adapt a proposal to the library, request
changes, or explain why it does not fit. Substantial changes to its meaning
should be discussed with its author, and contributions retain appropriate
credit in history, acknowledgements and citations.

If you would like to become a maintainer, email
[erikmendesnovak@gmail.com](mailto:erikmendesnovak@gmail.com) to discuss your
interests and the responsibilities you would like to take on.
