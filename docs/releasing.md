# Releasing TamerOp

Use this guide to prepare and verify a release candidate, register that exact
commit, and confirm that users can install the registered release. It is for the
package maintainer; library users should follow
[the installation walkthrough](src/start/install.md).

When release notes or a paper make performance claims, follow the
[benchmarking manual](benchmarking.md), including its evidence and reproducible
bundle requirements. [Benchmark reports and result downloads](benchmarks/index.md)
are already tracked documentation. Releasing portable executable reproduction
bundles is separate work; the local `benchmark/` and `audit/` directories remain
excluded from the public package.

TamerOp is registered in
[Julia's General registry](https://github.com/JuliaRegistries/General/tree/master/T/TamerOp),
so ordinary installation uses `Pkg.add("TamerOp")`. A GitHub repository, a Git tag,
and a General registry entry serve different purposes: pushing changes to
`main` does not update the registered release. Each new version needs its own
registration. TagBot creates tags and GitHub releases after registration; it
does not register the package.
[General registration](https://github.com/JuliaRegistries/General#registering-a-package-in-general),
[TagBot](https://github.com/JuliaRegistries/TagBot).

## 1. Prepare one release candidate

Keep these package identifiers unchanged:

| Setting | Value |
| --- | --- |
| Julia package name | `TamerOp` |
| Repository | `https://github.com/erikmnovak/TamerOp.jl` |
| Package UUID | `40830518-3340-46e0-bdcb-56edef4036ff` |
| License | MIT, in the root `LICENSE` file |

Select a new version in `Project.toml`; do not reuse a version already listed in
[General's release entries](https://github.com/JuliaRegistries/General/blob/master/T/TamerOp/Versions.toml).
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
harness compares installed paths, raw file bytes and symlink kinds with an
inventory from that commit. Windows checkouts can leave `.gitattributes` itself
with CRLF despite its LF policy; that specific metadata conversion is
accepted only when normalization reproduces the expected Git blob, and is
recorded in the report. Code and data files must match their raw Git bytes.
The harness checks again after computation and export, including an unchanged
raw filesystem fingerprint.
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
license, dependency bounds, download, installation and loading. For an existing
package, keep its registered name and UUID and request the new version from
the validated commit.

Open the **validated commit's page** on GitHub, rather than an arbitrary issue
or a moving branch. After the release checks pass, add this comment:

```text
@JuliaRegistrator register

Release notes:
See CHANGELOG.md in this commit for the changes, mathematical scope and
compatibility information for this version.
```

Registrator reads that commit's `Project.toml` and opens a registration pull
request in General. Follow its link and inspect the checks and review feedback.
If a repair is needed, commit it, repeat the affected validation, and trigger
registration on the replacement commit. Do not announce that the new version
is available through General while its pull request is pending.
[Registrator instructions](https://github.com/JuliaRegistries/Registrator.jl#via-the-github-app).

Follow the waiting period and review requirements for a new version of an
existing package; passing automated checks does not guarantee immediate acceptance.
[General's waiting periods](https://github.com/JuliaRegistries/General#automatic-merging-of-pull-requests).

## 5. Confirm the released package

After the registration pull request merges, allow the registry and package
servers to update. Check the TagBot run and the GitHub release. The version's
tag must refer to the registered commit, and the registry's source-tree hash
must match the validated tree. If necessary, use **Actions → TagBot → Run
workflow** to retry after correcting its configuration.

After that version's registration has merged, verify it in a fresh Julia session.
Set `release_version` to the version being checked; `0.1.0` below is an existing
registered release:

```julia
import Pkg
release_version = v"0.1.0"
Pkg.activate(; temp=true)
Pkg.Registry.update()
Pkg.add(Pkg.PackageSpec(name="TamerOp", version=release_version))
import TamerOp as OP
@assert Base.pkgversion(OP) == release_version
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

Keep the README and installation page aligned with the released package, with
`Pkg.add("TamerOp")` as the ordinary installation command. Existing URL installations can use
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
