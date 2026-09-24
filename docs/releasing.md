# Releasing TamerOp

This guide is for the package maintainer. To use the library, follow
[the installation walkthrough](../README.md#install-tamerop).

The first planned release is **0.1.0**. Until its registration has merged,
users install from the GitHub URL. A GitHub repository, a Git tag, and a General
registry entry serve different purposes: General is what makes name-only
`Pkg.add("TamerOp")` work. TagBot creates tags and GitHub releases after
registration; it does not register the package.
[General registration](https://github.com/JuliaRegistries/General#registering-a-package-in-general),
[TagBot](https://github.com/JuliaRegistries/TagBot).

## 1. Prepare one release candidate

Keep these package identifiers unchanged:

| Setting | Value |
| --- | --- |
| Julia package name | `TamerOp` |
| Repository | `https://github.com/erikmnovak/TamerOp.jl` |
| Package UUID | `40830518-3340-46e0-bdcb-56edef4036ff` |
| First planned version | `0.1.0` |
| License | MIT, in the root `LICENSE` file |

Review `Project.toml`, the beginner instructions, and `CHANGELOG.md` together.
For each new required or optional dependency, update its UUID, compatibility
bound, extension declaration where applicable, and installation guidance.
Exercise the affected integration through Julia's native loader. Keep supported
Julia versions consistent with the tested release matrix; do not widen the
compatibility declaration merely to make registration checks pass.

The public tree must work independently of the maintainer's local material.
The following **terminal** command should print nothing:

```sh
git ls-files -- AGENTS.md examples audit benchmark
```

Commit and push the candidate, then record its full commit and tree identifiers:

```sh
git status --short
git rev-parse HEAD
git rev-parse 'HEAD^{tree}'
```

The first command should show a clean tracked working tree. Record the other
two outputs with the test evidence. A changed candidate requires a new record
and checks appropriate to that change. Keep unrelated projects outside this
repository and outside the release test inputs.

## 2. Verify before registration

Run the maintained [release checks](testing.md) on that same candidate and keep
the environment versions, commands, complete logs, skips and exclusions.
Ordinary focused CI passing is not evidence that the full suite passed. Include
the mathematical oracles and the supported field/thread/backend/extension
combinations in the declared release matrix. Resolve failures before publishing
the release; a skipped optional integration is not a passing integration test.

The [Installed package workflow](../.github/workflows/Installation.yml)
separately installs the published commit with
`Pkg.add(url=..., rev=...)`, outside the checkout, and checks core computation
and actual CairoMakie figure exports on its configured platforms. Its report
records the pinned Git tree, installed source and dependency environment. The
harness compares every installed file's raw bytes, path and symlink kind with
an inventory from that commit, and checks again after computation and export.
On POSIX systems it also checks executable bits; Windows filesystem permissions
do not represent those Git modes. The recorded `filesystem_tree` is diagnostic
and can therefore differ on Windows even when the source verification passes.
This covers a different path from running tests in a developer checkout.

To reproduce that check, use a checkout of the same candidate with Git available
in your terminal's `PATH`. Replace `COMMIT` and `TREE` with the full 40-character
identifiers above, and `OUTPUT` with a new absolute directory for this run's
evidence:

```sh
julia --startup-file=no test/installation/run.jl --repo=https://github.com/erikmnovak/TamerOp.jl.git --rev=COMMIT --tree=TREE --output=OUTPUT
```

The candidate must already be accessible at that URL. The default uses a fresh
primary depot and fresh environments. Explicit `--cache-depot=PATH` options
allow reuse of dependency downloads and are recorded as such; do not describe
that run as a completely fresh download. Keep the reports and exported figures
with the release evidence. These checks need no local examples or audit files.

## 3. Configure the GitHub services

The repository owner needs to complete account settings which are not supplied
by committing files:

- Confirm that the [Registrator GitHub App](https://github.com/apps/juliateam-registrator/installations/new)
  is enabled for `erikmnovak/TamerOp.jl`. Enabling the app does not request
  registration; that is the separate step below, after validation. Without app
  installation, a registration comment does not activate the service. The
  commenter must be a repository collaborator.
  See [Registrator's setup](https://github.com/JuliaRegistries/Registrator.jl#via-the-github-app).
- Enable Actions and check that the repository's default workflow permissions
  allow TagBot to create releases. The maintained
  [TagBot workflow](../.github/workflows/TagBot.yml) follows the upstream default
  token configuration. A permission failure requires checking repository or
  organization settings, not changing a package dependency.
- If tags must trigger other workflows, configure an SSH deploy key as described
  by [TagBot](https://github.com/JuliaRegistries/TagBot#ssh-deploy-keys), then
  enable the documented `ssh` input. The ordinary configuration does not need
  a personal access token.

Current TagBot guidance warns that `GITHUB_TOKEN` may not tag a commit which
changes workflow files. Commit workflow preparation before the final release
metadata commit, validate the final candidate, and register that exact commit.
If tagging still fails, follow
[TagBot's troubleshooting instructions](https://github.com/JuliaRegistries/TagBot#commits-that-modify-workflow-files)
and use the commit recorded by Registrator. Do not replace an existing public
tag to hide a mismatch.

## 4. Request General registration

Read the [General requirements](https://github.com/JuliaRegistries/General)
and [AutoMerge checks](https://juliaregistries.github.io/RegistryCI.jl/stable/guidelines/)
before requesting registration. They check the package name, repository URL,
license, dependency bounds, download, installation and loading. A source check
of General on 24 September 2026 found neither the case-insensitive name
`TamerOp` nor this UUID. That is not a reservation: check again when submitting,
and address any name-similarity review.

Open the **validated commit's page** on GitHub, rather than an arbitrary issue
or a moving branch. After the release checks pass, add this comment:

```text
@JuliaRegistrator register

Release notes:
Initial TamerOp 0.1.0 release. See CHANGELOG.md in this commit for capabilities,
mathematical scope, installation and compatibility information.
```

Registrator reads that commit's `Project.toml` and opens a registration pull
request in General. Follow its link and inspect the checks and review feedback.
If a repair is needed, commit it, repeat the affected validation, and trigger
registration on the replacement commit. Do not announce name-only installation
while the pull request is pending.
[Registrator instructions](https://github.com/JuliaRegistries/Registrator.jl#via-the-github-app).

New packages have a minimum three-day AutoMerge waiting period for community
feedback; passing automated checks does not guarantee immediate acceptance.
[General's waiting periods](https://github.com/JuliaRegistries/General#automatic-merging-of-pull-requests).

## 5. Confirm the released package

After the registration pull request merges, allow the registry and package
servers to update. Check the TagBot run and the GitHub release. The `v0.1.0`
tag must refer to the registered commit, and the registry's source-tree hash
must match the validated tree. If necessary, use **Actions → TagBot → Run
workflow** to retry after correcting its configuration.

In a fresh Julia session, run this **only after registration has merged**:

```julia
import Pkg
Pkg.activate(; temp=true)
Pkg.Registry.update()
Pkg.add(Pkg.PackageSpec(name="TamerOp", version="0.1.0"))
import TamerOp as OP
@assert Base.pkgversion(OP) == v"0.1.0"
values = zeros(Int, 3, 3)
values[2, 2] = 5
diagram = OP.cubical_persistence(values)
@assert OP.finite_intervals(diagram; dim=1) == [(0, 5)]
@assert OP.essential_births(diagram; dim=0) == [0]
Pkg.status()
```

Use an ordinary permanent analysis environment for subsequent work; the
temporary environment here is only a release check. Confirm the reported
package source and dependency versions. A missing registry entry immediately
after a merge can reflect propagation delay; update the registry and retry.

Once this succeeds, update the README's distribution status and make the main
installation command `Pkg.add("TamerOp")`. Existing URL installations can use
`Pkg.free("TamerOp")` in their analysis environment to return to registry-managed
versions. Keep a manifest for reproducible analyses. Instructions and version
selection follow the [Pkg guide](https://pkgdocs.julialang.org/v1/managing-packages/).

Change the changelog's candidate heading to the released version and actual
date. Add version-specific citation metadata or an archive DOI only when it
exists; retain the repository citation and tell users to record their analysis
version or commit. A date or DOI must never be inferred from a planned release.

## Later versions

Keep the name and UUID. Select the next version, describe user-visible changes
and migration steps in the changelog, review dependency bounds, and repeat this
process. Test installation from the exact new commit and then from the registry.
Register a new version for repairs to a released tree. Maintain
`CITATION.cff` alongside the software metadata and release notes.
