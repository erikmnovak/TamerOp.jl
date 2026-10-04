# Build the teaching site

The canonical ring lesson is [tutorials/ring.ipynb](tutorials/ring.ipynb).
One execution produces a website lesson with static figures and a downloadable
executed notebook containing those same results. Edit the notebook, not its
generated Markdown. Existing mathematical chapters remain in their current
tracked locations and are copied into the build according to
`publication.toml`; they are not independently authored website copies.

## Prepare the tools

Run these commands from the repository root with Julia 1.12 and Python 3.11
or newer. The isolated documentation environment uses the current checkout of
TamerOp. It does not add plotting or website dependencies to the core package.

```sh
python -m venv docs/.venv
docs/.venv/bin/python -m pip install -r docs/requirements.txt
JULIA_NUM_PRECOMPILE_TASKS=1 julia --startup-file=no --threads=1 --project=docs -e 'using Pkg; Pkg.instantiate()'
```

On Windows, use `docs\.venv\Scripts\python.exe` and set the environment
variable using your shell's syntax. `docs/Project.toml` pins the direct Julia
tool versions; the resolved local `Manifest.toml` is retained in the generated
downloads. The Python direct dependencies are pinned in `requirements.txt`.

## Build and review

```sh
docs/.venv/bin/python -m unittest discover -s docs/build_scripts -p 'test_*.py'
docs/.venv/bin/python docs/build_scripts/publish.py
docs/.venv/bin/python docs/build_scripts/check_site.py
python -m http.server 8000 --directory docs/build
```

Open <http://localhost:8000/> and follow the ring lesson. The HTML also uses
ordinary `.html` links so `docs/build/index.html` can be opened directly.
Check the appearance-grade image, all three masks, the interval's included
birth/excluded death, the essential component, and the original/changed bars
on the same axis. Check narrow-screen layout, equations, syntax highlighting,
figure descriptions, downloads and the transition into finite encodings.

The notebook's assertions check `[0,5)`, the essential component born at zero,
and the changed interval `[0,3)`. The build fails on a cell exception, missing
execution, missing declared figure or missing caption metadata. Code is run
once in a fresh IJulia kernel with one Julia/BLAS thread. The default ring
budget is 900 seconds per cell and 20 MiB for its executed download; first-use
compilation is included in recorded execution time. No global kernel is installed.

The build stages sources under `docs/.build/src`, then calls `docs/make.jl`.
For a formatting-only iteration, rerun `julia --project=docs docs/make.jl` on
that captured stage; this does not execute cells or incorporate edited notebook
sources. Run the complete publication command after changing lesson code/prose.
`check_site.py` rejects a download built from an older notebook source.

Outputs:

- `docs/build/tutorials/ring.html`: website lesson.
- `docs/build/reading_map.html`: linked overview of the available reading routes.
- `docs/build/explanations/two_parameters.html`: short bridge into the existing
  mathematical chapters, staged from `docs/two_parameters.md`.
- `docs/build/downloads/ring.ipynb`: notebook with saved figures.
- `docs/build/downloads/figures/`: separate exports made by the notebook.
- `docs/build/downloads/publication.json`: source hashes, checkout/dirty status,
  execution timing, figure counts and tool versions.
- `docs/build/downloads/environment/`: the resolved build environment. Its
  TamerOp path is relative to the original repository's `docs/` directory;
  restore it there in the recorded checkout to reproduce those dependency
  versions. For ordinary notebook use, create your own environment as the
  installation lesson explains. These snapshots are kept away from the
  notebook so Jupyter cannot accidentally activate their relative build paths.

An executed download rewrites relative reading links to repository URLs so
they still work when the file is moved. Site links point to staged chapters
where available. Links to supporting material not yet integrated into the
site lead to its canonical repository source. A dirty build records the
package's individual source hashes; the commit alone does not identify it.

## Maintain the reading map

The reader-facing [map page](src/reading_map.md) contains its introduction.
`reading_map.toml` supplies the cards, questions, reading arrows, and grouped
links to further guides. Each lesson node also declares `next` and an optional
`alternate`, using canonical paths relative to `docs/`. The publication build
checks that the last section's links (or the last notebook Markdown cell's links)
match these destinations in order. Keep this ending to the motivated continuation,
with broader reference links in the relevant earlier prose.

The clickable diagram and its linked outline share this graph. The site footer
uses the declared next lesson rather than Documenter's linear sidebar order;
sidebar and footer titles use the map's titles. It also links to the reading map
instead of inventing a single previous page for a branching route. Add a real source
path when introducing a page; the build uses the local site destination when
that source is published and labels other destinations as repository links.

The [map design principles](writing.md#preserve-choices-as-the-reading-map-grows)
govern future additions: equal entry choices, meaningful optional branches,
finite encodings as the shared destination, and a selective, accessible diagram.
Preserve these choices in the rendering as well as the introductory prose.

Arrows suggest next readings. Incoming arrows offer alternative routes,
not a list of mandatory prerequisites. Keep cycles out of this progression
even when the chapters contain reciprocal reference links. Unwritten lessons
stay in the authoring backlog.

Keep status and development history out of lessons. A clearly labeled figure
placeholder may describe a missing visual and its mathematical purpose.
Record publication evidence here or in the testing
guide, and future work in the backlog. Operational and mathematical constraints
remain in the lesson where they affect the reader's choices.

The builder checks source destinations, graph cycles, duplicate nodes/edges,
and overlapping or out-of-bounds cards. When changing the layout, review the
actual arrows, line wrapping, keyboard focus, horizontal scrolling on a narrow
screen, the linked outline, and the light/dark themes. Native links and SVG
connectors work without a JavaScript renderer or external diagram service.
A small focus handler keeps keyboard-selected cards fully visible when the
map has been scrolled sideways; the linked outline also works without scripts.

### Map expansion checklist

1. State the new page's reader question, the knowledge it needs, and the next
   question it makes meaningful. Decide whether it belongs on the main route,
   beside a lesson as support, or among the grouped references. Add a diagram
   card only when it improves a reader's choice.
2. Update the canonical source and routes in `reading_map.toml`, together with
   affected lesson endings. Set entry status explicitly: an optional incoming
   arrow must not demote a starting point. Use `display = "support"` for setup
   beside a lesson and `display = "reference"` for a node retained in the outline
   and navigation without a diagram card. Optional edges use `kind = "optional"`
   with a purposeful label.
3. Check that both starts retain equal prominence, direct routes remain visible,
   and no new arrow makes setup or an optional treatment look compulsory.
   Prefer a grouped reference or a focused branch map to crowded cards,
   crossing connectors, repeated instructions, or smaller text.
4. Build from the authored sources and run the publication checks. Review the
   actual diagram and outline at desktop and narrow widths, in light and dark
   themes. Check card text and arrow labels for clipping and overlap. Verify
   that both starts are visible before sideways scrolling and that keyboard
   focus brings linked cards fully into view.
5. Follow the changed routes and their footer links. Confirm that the lesson's
   prose explains the same choice, including repository destinations. Ask the
   newcomer questions in the writing guide; passing link, cycle, and bounding-box
   checks does not establish that the map communicates the intended choices.

Edit authored Markdown, TOML, styles, and the renderer; generated HTML is an
output. Keep review evidence in contributor material rather than adding status
reports or maintenance instructions to the reader's map.

## Optional sections

Follow [the writing guide](writing.md#reveal-options-when-they-become-useful):
keep mathematical choices and required comparisons visible, and teach
styling/export after the main result. Mark every cell in a contiguous optional
section with the same metadata, for example:

```json
{"tamerop": {"optional_section": "Optional: Prepare a figure for sharing"}}
```

Keep an explicit optional heading and explanation in its first Markdown cell.
The website wraps the group in a native, initially closed disclosure; the
executed download keeps the original cells and all saved figures. Figure cells
still require `tamerop.figure_alt`. All cells, including optional ones, execute
during publication. Later required cells must work if optional cells are skipped.
Review both disclosure states, keyboard access, narrow-screen layout, and the
complete notebook; do not use folding to hide an assumption needed for a result.

## Benchmark result pages

[Benchmark results](benchmarks/index.md) collects completed comparisons, with
[QPA](benchmarks/qpa.md) for finite-module algebra and
[PHAT](benchmarks/phat.md) for ordinary barcodes on the expanded medium/large
workloads. Each report has its own scope,
versions, machine record, timing tables, figures and compact downloadable data.
Keep these accounts separate from the mathematical learning path.

Follow the [benchmark presentation guidance](benchmarking.md#practical-significance-and-presentation):
lead with TamerOp's demonstrated strengths, practical differences and actual
times. Describe close performance as similar with the measured edge explicit;
use counts as supporting detail. Preserve the same practical band for both
tools, all measurements and material losses. Edit these canonical sources so
the GitHub reports and generated website carry the same account.

Author reports and data under `docs/benchmarks/`; list their Markdown pages in
`publication.toml` and the benchmark navigation group in `make.jl`. Publication
copies linked figures, CSV/JSON data and checksum files into the site, so the
reports work on GitHub and in the built website without local audit directories.
Do not replace a dated result set silently when later measurements are available.

## Fit with the wider documentation plan

The separate [implementation reference](implementation/index.md) answers how
the algorithms work and why their implementation choices were made. Its first
account covers [exact rational coordinates](implementation/qq_coordinates.md),
with a [shared bibliography](implementation/references.md). These pages are
staged through `publication.toml` and have their own site navigation category;
they are not part of the reading map or the supporting usage guides. This
first account is a format pilot, not a prescribed template for later accounts.

This is the first site scaffold, with installation, the ring lesson, the
short [two-parameter bridge](two_parameters.md), the existing mathematical
chapters and contributor writing guidance. The bridge joins the original
persistence modules → finite encodings → indicator presentations → tameness
sequence, now followed by [why finite computations stay tame](practical_tameness.md).
That chapter explains finiteness supplied by construction and closure under
compatible algebraic operations. The [learning-path brief](learning_path.md#keep-the-mathematical-chapters-distinct)
records each treatment's pedagogical boundary. Those chapters remain the
canonical full explanations; the bridge
reuses their parameter-order diagram and motivates the square lesson.
The site leaves room for the full finite-encoding learning path, API reference
and galleries.
The square notebook remains a separate canonical lesson; adding it to
`publication.toml` will require its own static-output and frontend acceptance.
Live widgets are optional enhancements to a readable static lesson.

The documentation CI builds and uploads review artifacts without deployment
credentials. Nothing here publishes to GitHub Pages or establishes a `stable`
release site. Generated output and environments are ignored by Git; public
builds need no `audit/`, `examples/`, `benchmark/` or sibling-project files.
