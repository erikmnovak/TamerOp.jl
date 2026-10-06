# Build the teaching site

The canonical notebooks are the lessons [the ring](tutorials/ring.ipynb) and
[inspecting spaces and maps](tutorials/inspect_encoding.ipynb), the library
guides [From inputs to computed objects](tutorials/inputs_to_objects.ipynb) and
[Choosing and inspecting resolutions](tutorials/resolutions.ipynb), and the
recipe [Explain a distance through its matching](tutorials/distance_witness.ipynb).
One execution of each produces a website page with static figures and a
downloadable executed notebook containing those same results. Edit the notebook,
not its generated Markdown. Existing mathematical chapters remain in their current
tracked locations and are copied into the build according to
`publication.toml`; they are not independently authored website copies.

The [documentation programme](documentation_plan.md) is the authoring plan for
the full mathematical, computational and visual scope. It coordinates the five
main article families, examples, future map views and existing-page migrations.
Use it with the [article inventory](article_inventory.md) and
[writing guide](writing.md); a planned record creates no publication route.

## Prepare the tools

Run these commands from the repository root with Julia 1.12 and Python 3.11
or newer. The isolated documentation environment uses the current checkout of
TamerOp. It does not add plotting or website dependencies to the core package.

Reader-facing setup starts with `Pkg.add("TamerOp")` from General. A website
build against the checkout does not establish compatibility with the registered
release: check the registry's source tree when changing installation examples.
Keep the README's first computation usable with that release. If a downloadable
lesson needs newer APIs, explain its separate development environment in the
[installation page](src/start/install.md); do not quietly make source tracking
the default for ordinary users. Revisit that distinction when a new release
includes those APIs.

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
python docs/build_scripts/preview.py
```

Open <http://localhost:8000/> and follow the lessons and the input-routes guide.
The HTML also uses ordinary `.html` links so `docs/build/index.html` can be opened directly.
The preview disables browser caching and serves the same introduction at `/`
and `/index.html`. If an older preview left a cached copy, reload that URL once
with Ctrl+Shift+R (Cmd+Shift+R on macOS). Use `--port 8765` for another port.
Rebuild after editing sources; refreshing alone does not regenerate HTML.
Check the appearance-grade image, all three masks, the interval's included
birth/excluded death, the essential component, and the original/changed bars
on the same axis. Check narrow-screen layout, equations, syntax highlighting,
figure descriptions, downloads and the transition through finite encodings
into square inspection. For the square, follow a parameter into its finite
label, space and selected map; compare the three overlapping-square maps on
the same window, and check that the active-zero presentation has zero image.
Review collapsed and expanded optional sections, including the saved static
exports. Live inspector instructions require their own frontend review.

For the input-routes guide, follow both the filtration and diagonal-strip
construction into parameter queries. Check that the displayed spaces and maps
agree with the stated support and coefficients, then read the kernel example
over its declared finite base. Review the strip boundary and the distinction
between its unbounded mathematical support and the finite plotting window.

The ring's assertions check `[0,5)`, the essential component born at zero,
and the changed interval `[0,3)`. The square's assertions check the closed
support, correctly shaped maps, and the zero composite of the two nonzero
successive maps. The build fails on a cell exception, missing
execution, missing declared figure or missing caption metadata. Code is run
once per notebook in a fresh IJulia kernel with one Julia/BLAS thread. Each
lesson has a budget of 900 seconds per cell and 20 MiB for its executed download.
First-use compilation is included in recorded execution time. No global kernel is installed.

The build stages sources under `docs/.build/src`. It calls
`site_catalog.py`'s `prepare_catalog` to generate collection and topic pages,
then `docs/make.jl` runs Documenter. The postprocessing order is
`navigation.py` for learning continuations, followed by `site_shell.py` for the
shared navigation, local heading outline and search metadata.

For site prose, navigation or appearance changes with unchanged notebooks,
reuse the verified execution outputs while restaging the current sources:

```sh
docs/.venv/bin/python docs/build_scripts/publish.py --reuse-executed
docs/.venv/bin/python docs/build_scripts/check_site.py
```

This checks notebook source/download hashes, the selected notebook set, package
sources and the declared documentation execution environment before reusing
saved outputs. It preserves their original execution evidence. Run the normal
publication command after changing notebook code or prose, package sources or
that environment. Both modes accept `--julia /path/to/julia` when Julia is not
on `PATH` or a particular executable is needed.

`check_site.py` requires a matching publication record, page, executed download,
figures and optional sections for every notebook in `publication.toml`. It
rejects a download built from an older source and checks each lesson's download link.
It also requires the built HTML to match the catalog and every page to be
reachable from the homepage, catching abandoned pages and disconnected groups.

Outputs:

- `docs/build/tutorials/ring.html` and `docs/build/tutorials/inspect_encoding.html`:
  website lessons; `docs/build/guides/inputs_to_objects.html` is the practical
  input-routes guide.
- `docs/build/reading_map.html`: linked overview of the available reading routes.
- `docs/build/topic_map.html` and `docs/build/topics/`: subject overview and
  topic pages linking related treatments across article families.
- `docs/build/collections/`: Mathematics, Using TamerOp, Task recipes and API reference
  landings; Implementation and Benchmarks use their existing index pages.
- `docs/build/contributing/index.html`: contributor procedures and project credit.
- `docs/build/explanations/two_parameters.html`: short bridge into the existing
  mathematical chapters, staged from `docs/two_parameters.md`.
- `docs/build/downloads/ring.ipynb`, `docs/build/downloads/inspect_encoding.ipynb`
  and `docs/build/downloads/inputs_to_objects.ipynb`: notebooks with saved figures.
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

## Maintain site navigation

The sidebar keeps Introduction, Installation, Learning map and Topic map above
six collections: Mathematics, Using TamerOp, Task recipes, API reference,
Implementation and Benchmarks. Only the current collection's branch starts expanded. Contributor
guidance is available from the sidebar footer. A separate local outline follows each
article's H2/H3 headings; it must not inflate the collection hierarchy.

Maintain one article record in [the inventory](article_inventory.md), with its
canonical source, type, reader question and topics. `publication.toml` and
authored pages under `docs/src/` identify published destinations.
`navigation.toml` controls labels, grouping and order only. The catalog builder
joins these sources to generate collection listings, the topic map, topic
landings and metadata for navigation and search. Do not add article-source
lists to `navigation.toml`, `make.jl` or hand-authored topic pages.

Public listings contain authored, usable pages. Keep planned articles and
migration notes in the inventory. An article appearing under several subjects
still has one canonical source and destination. Use shallow topic groups and
landing pages as a collection grows; do not solve growth by exposing every
article and section at once. Relevant guides can also be linked thematically
from the page that needs them.

Sidebar grouping must not determine whether a published URL continues to exist.
Set `topic_pages = true` on a collection with published subject landings to keep
those pages when its article count falls below `sidebar_threshold`. The smaller
collection keeps its direct article list and offers its subject indexes under
a collapsed “Browse by subject” section. If a subject's articles move to another
collection, `related_topics` can retain an index pointing to those treatments
with their current editorial labels. Both options use inventory topics, never
another list of article sources. Keep the pages useful and reachable, without
duplicate articles or empty placeholders.

Learning routes remain independent. `reading_map.toml` owns their arrows and
lesson footer continuations; collection order and topic membership cannot
silently change the next lesson. The contributor entrance in the sidebar footer
is separate from those learning continuations.

After a navigation change, run the documentation tooling checks and rebuild
the site. The dedicated static-site browser check uses the existing Playwright
installation described in the [browser guide](../test/browser/README.md):

```sh
node test/browser/documentation.mjs
```

It starts its own loopback server and Chromium, without starting Julia. It checks
desktop and mobile navigation, keyboard access, the local outline, actual search
labels and navigation with JavaScript disabled. Screenshots go to
`test/browser/test-results/documentation/`. Review those images alongside the
interaction checks; a passing script alone does not establish visual clarity.

Review these behaviors in the browser:

1. Follow all four permanent entrances and the sidebar's contributor footer link from a
   lesson, a guide and a deeply nested page.
2. Check the six collection labels and initial disclosure state. Open other
   branches with the keyboard and verify visible focus and the current-page
   indication.
3. Follow a subject from the topic overview to treatments of different types.
   Confirm the question, type and canonical destination agree with the inventory;
   planned articles must not appear as empty destinations.
4. Check that the local outline contains the current page's H2/H3 sections and
   reaches their anchors without duplicating article-navigation entries.
5. Search for a mathematical term shared by several families. Verify that
   results identify article type and topic and lead to the intended treatment.
6. Repeat at narrow widths and in light and dark themes. Check labels,
   disclosure controls, scrolling and focus for clipping or overflow; follow
   lesson footer links separately to confirm the learning route is unchanged.

Edit the authored sources and metadata rather than generated HTML. Structural
checks establish consistency; rendered review establishes whether readers can
recognize and use these choices.

## Maintain the homepage

The introduction lives in `src/index.md`; its scoped presentation lives in
`src/assets/home.css`. Keep the motivating question, finite encoding, and the
equally prominent example-first and definitions-first entrances together.
Installation supports execution; reading saved results does not require it.
The overview has its own compact navigation instead of an article outline.
`/` and `/index.html` serve this one page. If they appear different, check the
preview's browser cache and build before changing navigation or making another
introduction. Compare the local and deployed sites separately: rebuilding a
local preview does not publish it to GitHub Pages.

Its theory-to-application spectrum offers independent entrances: Mathematics
develops understanding, Using TamerOp explores capabilities and choices, and
Task recipes give focused procedures for defined results. Follow the
[editorial boundaries](writing.md#move-from-understanding-to-exploration-to-a-defined-task)
when expanding those introductions; do not turn them into a required sequence.

Its responsive workflow figure adapts the thesis's many-inputs, finite-object,
many-questions architecture. Make both starting routes visible: filtered data
and directly specified modules. Pair familiar filtration inputs with supported
geometric presentations and general finite posets, then show the spaces, maps,
algebra and summaries available afterward. The inline SVG gives the finite
model a concrete square example:
the selected parameter maps to the one-dimensional space in the four-label
encoding. Keep the caption, space labels, upward order arrows, and accessible
description consistent. The figure describes the finite-encoding workflow;
direct ordinary-barcode routines need not pass through it. All figure content
is authored here; the site has no dependency on thesis files.

The compact capability passages should explain both routes through concrete
questions. A sloped strip illustrates exact geometric support beyond a finite
axis-aligned grid; its plotted window must not be mistaken for a truncation of
the module. Filtration-derived encodings also support finite-module algebra.
Link these claims to [the executable input-routes guide](tutorials/inputs_to_objects.ipynb),
keeping the finite base explicit for derived operations. Preserve the equally
prominent learning entrances and keep this practical guide outside the
mathematical reading map.

Distinguish supported capabilities from measured performance. Link scoped
comparison reports at the point of a performance claim and preserve results
favoring other tools. Review the
homepage in light and dark themes, at narrow widths, and with keyboard access;
the diagram should reflow without requiring horizontal scrolling. Its styles
must not change the layout of lessons or reports.

## Maintain the task-recipe collection

`src/collections/recipes.md` introduces the collection. Its article list is
generated from published `recipe` records in `article_inventory.toml`, while
Using TamerOp lists `library_guide` records. Keep Installation pinned among the
permanent upper links and linked from the recipe landing; it has one canonical
page and does not need a duplicate sidebar item inside the collection.

Keep intended recipes in the inventory, with a defined task and scope, rather
than adding empty public pages. Follow the
[recipe-writing guidance](writing.md#write-recipes-around-an-attainable-result)
when authoring or adapting one. Confirm that its collection, topic links and
search label agree, that its procedure yields the stated result, and that
necessary assumptions remain beside the code. A changed navigation label alone
does not complete an existing page's editorial migration.

## Library guide examples

The **Using TamerOp** site section publishes the
[input-routes guide](tutorials/inputs_to_objects.ipynb) from its canonical
notebook, followed by the
[spaces-and-maps pilot](spaces_and_maps.md),
[deferred-computation guide](lazy_inspection.md), and
[view and interval guide](visualization.md) from their canonical Markdown.
These independently consulted guides share the site's navigation without
adding required stops to the mathematical route.

The input guide compares two constructions and returns to the same space/map
questions before continuing with a supplied morphism and its kernel. It owns
that choice of starting route; the spaces-and-maps guide owns the deeper query
and inspection workflow. Keep both under their existing article identities
and route the notebook's page and executed download through `publication.toml`.
Publish this notebook with the normal build after editing it; its captured
figures and assertions are reviewed together with the displayed explanation.

The pilot's figures come from its own static Julia blocks. To check the
hand-computable answers and regenerate both figures through the package API:

```sh
julia --startup-file=no --threads=1 --project=docs docs/build_scripts/render_spaces_maps.jl
```

The renderer reads the blocks preceding the optional live-session section;
there is no second copy of the example. It saves PNGs under
`docs/assets/guides/`, which the publication build copies into the site.
Run it after changing those examples, then review both figures and the page.
Live browser controls remain a separate frontend check.

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
its continuation titles come from the reading map. The sidebar uses the article
catalog's collection metadata. The lesson footer also links to the reading map
instead of inventing a single previous page for a branching route. Add a real source
path when introducing a page; the build uses the local site destination when
that source is published and labels other destinations as repository links.

The [map design principles](writing.md#preserve-choices-as-the-reading-map-grows)
govern future additions: equal entry choices, meaningful optional branches,
finite encodings as the shared destination, and accessible overview/focused views.
Preserve these choices in the rendering as well as the introductory prose.

The [full map plan](documentation_plan.md#one-graph-several-views-complete-mathematical-coverage)
extends coverage to every published mathematical lesson, including advanced
chapters, through focused views and a complete linked outline of one graph.
The current fixed-coordinate renderer does not yet implement that expansion.
Join article identity, publication availability and learning-route metadata;
do not maintain independent route graphs for each view or add missing pages.

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
   question it makes meaningful. Give each mathematical lesson a node in a
   focused view and the shared outline; select only orienting junctions for
   the overview. Supporting setup and references remain separately identified.
2. Update the canonical source and routes in `reading_map.toml`, together with
   affected lesson endings. Set entry status explicitly: an optional incoming
   arrow must not demote a starting point. Use `display = "support"` for setup
   beside a lesson and `display = "reference"` for a node retained in the outline
   and navigation without a diagram card. Optional edges use `kind = "optional"`
   with a purposeful label.
3. Check that both starts retain equal prominence, direct routes remain visible,
   and no new arrow makes setup or an optional treatment look compulsory.
   Prefer a focused branch map to crowded cards, crossing connectors,
   repeated instructions, or smaller text. Group nonlesson references separately.
4. Build from the authored sources and run the publication checks. Review the
   actual diagram and outline at desktop and narrow widths, in light and dark
   themes. Check card text and arrow labels for clipping and overlap. Verify
   that both starts are visible before sideways scrolling and that keyboard
   focus brings linked cards fully into view.
5. Follow the changed routes and their footer links. Confirm that the lesson's
   prose explains the same choice, including repository destinations. Ask the
   newcomer questions in the writing guide; passing link, cycle, and bounding-box
   checks does not establish that the map communicates the intended choices.
6. As the expanded route model is implemented, check complete published-lesson
   coverage against the article catalog, unique identities across views, and
   agreement of diagram links, accessible outline and local neighborhoods.
   Keep the knowledge relation separate from suggested reading edges and the
   topic map. Cover definition-led chapters such as `math_categories`, not
   only the executed notebook route.

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
still require `tamerop.figure_alt`. All code cells, including optional ones,
execute during publication. Later required cells must work if optional cells are skipped.
Review both disclosure states, keyboard access, narrow-screen layout, and the
complete notebook; do not use folding to hide an assumption needed for a result.

Live WGLMakie examples in the square lesson are Julia fences in Markdown cells,
with instructions to copy them into a code cell in a running notebook. They
remain visible in the optional section and downloadable notebook but are not
executed by the static publication build. Required figures and the separate
PNG/SVG exports use CairoMakie. The export section selects the endpoint map
directly from the encoding; it does not depend on a live session or snapshot.
The optional snapshot example is also Markdown only, so the documentation
environment needs no WGLMakie dependency. Do not tag an executable live widget
as optional and expect publication to skip it: optional metadata controls
website folding only.

## Benchmark result pages

[Benchmark results](benchmarks/index.md) collects completed comparisons, with
[QPA](benchmarks/qpa.md) for finite-module algebra,
[PHAT](benchmarks/phat.md) for ordinary barcodes on supplied boundary data, and
[Ripser.py](benchmarks/ripser.md) for Rips barcodes, cocycles and landmark workflows.
Each report has its own scope,
versions, machine record, timing tables, figures and compact downloadable data.
Keep these accounts separate from the mathematical learning path.

Follow the [benchmark presentation guidance](benchmarking.md#practical-significance-and-presentation):
lead with TamerOp's demonstrated strengths, practical differences and actual
times. Describe close performance as similar with the measured edge explicit;
use counts as supporting detail. Preserve the same practical band for both
tools, all measurements and material losses. Edit these canonical sources so
the GitHub reports and generated website carry the same account.

Use the [QPA-based visual standard](benchmark_style.md) and its shared
[plotting preset](assets/benchmarks/report.mplstyle) for report appearance.
Keep tool colors, typography, grids, legends and tables consistent while
choosing plot types and panel layouts for each study's evidence. Review exported
figures at their displayed size; the preset alone does not establish conformity.

Author reports and data under `docs/benchmarks/`; record each canonical article
in the inventory and its Markdown route in `publication.toml`. The catalog
derives the benchmark collection membership. Publication
copies linked figures, CSV/JSON data and checksum files into the site, so the
reports work on GitHub and in the built website without local audit directories.
Do not replace a dated result set silently when later measurements are available.

PHAT's figures can be regenerated from its public summaries with
[render_phat_figures.py](build_scripts/render_phat_figures.py). With Matplotlib
installed (the recorded presentation uses 3.9.3), run from the repository root:

```sh
MPLCONFIGDIR=/tmp/tamerop-matplotlib python docs/build_scripts/render_phat_figures.py --preview-dir /tmp/tamerop-phat-figures
```

This updates `docs/benchmarks/phat_v2/presentation/` and its separate checksums,
leaving the original sealed results intact. It does not rerun the benchmarks.
Review the SVG exports and page after regeneration. Ordinary site builds use
the saved figures and do not require Matplotlib.

## Fit with the wider documentation plan

Use the [article inventory and maintenance guide](article_inventory.md) to
find existing and planned treatments across the documentation. The
[editorial families](writing.md#choose-an-articles-purpose-before-its-format)
distinguish mathematical lessons, library guides, task recipes, API reference,
implementation accounts, benchmark reports and contributor guides. Supporting
indexes and project records have their own entries. Classification describes
the intended purpose; it does not certify a page's quality or require moving
all current files at once.

The inventory owns article identity, topic membership and gradual migration
intent. Keep learning continuations in `reading_map.toml`, publication routes
in `publication.toml`, and binding coverage in `api_coverage.toml`. Notebook
exports and generated website pages are presentations of their canonical
article, not additional authored entries.

The [topic-map proposal](writing.md#separate-learning-routes-from-topic-exploration)
adds subject clusters containing related treatments across editorial families.
It complements the mathematical reading map without implying prerequisites
or crowding that diagram with every guide and reference. Its future linked
index and visual view should derive membership from the article inventory;
the topic map is not implemented by adding inventory records.

The separate [implementation reference](implementation/index.md) answers how
the algorithms work and why their implementation choices were made. Its first
account covers [exact rational coordinates](implementation/qq_coordinates.md),
with a [shared bibliography](implementation/references.md). These pages are
staged through `publication.toml` and have their own site navigation category;
they are not part of the reading map or the supporting usage guides. The
first account is the accepted model for depth and explanatory style, with
structure adapted to each topic rather than copied as a fixed template.
Follow the [implementation-writing guidance](writing.md#explain-implementation-rather-than-its-development-history)
when planning and reviewing later accounts: follow a coherent mathematical
operation through its representations, actual execution choices, correctness,
reuse and costs. Choose examples and figures that explain consequential choices;
cite academic ideas and software where used. Keep development, benchmark and
review records in their own material.

This site scaffold includes installation, the ring and square lessons, the
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
All canonical notebooks are listed in `publication.toml` and use the same
execution, figure capture, download and navigation checks. A fresh publication
build and rendered review must establish acceptance for a changed lesson;
earlier notebook execution or separate exports do not establish it. Live widgets
are optional enhancements to a readable static lesson and need separate
notebook-frontend acceptance.

## Publish on GitHub Pages

As new computational families arrive, publish small coherent groups through
this existing pipeline. Their article records and API-family backlog can be
planned first; add routes only after canonical sources are ready. Keep the
current collections and topic pages, using their generated listings instead of
adding a sidebar item for every implementation task. A guide, mathematical
lesson and algorithm account about one feature answer different questions and
should link to each other without repeating their contents.

With parallel implementation, one integration owner updates shared publication,
navigation, reading-route and API manifests after the code and examples merge.
The feature owner supplies the examples and article changes. Regenerate and
check the site from that integrated tree; a successful build on either feature
branch does not establish the combined result. Existing-but-unpublished
sources, such as a newly executed notebook, still need routing and rendered
review. See the [catalog integration guidance](article_inventory.md#plan-documentation-with-a-computational-addition)
and [later teaching branches](learning_path.md#teaching-the-next-computational-families).

The [Documentation workflow](../.github/workflows/Documentation.yml) builds and
checks the site before publishing `main` at
<https://erikmnovak.github.io/TamerOp.jl/>. Pull requests and other branches
produce a `teaching-site` review artifact without deployment privileges.
The separate deployment job uses GitHub's short-lived workflow token and
the `github-pages` environment; no personal token or SSH key is needed.
Only the generated `docs/build` directory is uploaded for hosting.

For the first deployment, a repository administrator selects **GitHub Actions**
under **Settings → Pages → Build and deployment → Source**. The workflow is
already provided here; no additional starter workflow is needed. If the first
build finishes before this setting is enabled, enable it and choose **Re-run
failed jobs** on the Documentation run. Its Pages artifact is retained for
seven days. After that, use **Run workflow** on `main` to make a fresh build.
See GitHub's [custom Pages workflow guide](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

After a successful deployment, open the public homepage, both maps, a nested
guide, search results, figures, and notebook downloads. Set the repository's
About website link to the public address. Static lessons work without Julia;
the live inspectors described in them still require a running Julia session.
The maintained browser check (`npm run test:docs` in `test/browser`) serves the
local build under `/TamerOp.jl/` to exercise the same project-site URL layout.

This publishes the current documentation without versioned `dev` or `stable`
directories. Release-specific documentation needs a separate versioning policy.
Generated output and environments stay ignored by Git; public builds need no
`audit/`, `examples/`, `benchmark/` or sibling-project files.

## Check the resolution guide

`tutorials/resolutions.ipynb` is the canonical usage guide for finite-poset
resolution tables, selected incidence, support and grade views. Its article
home is `guide_resolutions`; the precise contracts are in
`src/reference/visualization.md`, and the shared implementation explanation is
`implementation/visual_specs.md`. This is a usage guide, not another
mathematical lesson or a duplicate general plotting manual.

Run its maintained executor against the checkout's documentation environment:

```sh
julia --startup-file=no --project=docs docs/build_scripts/check_resolution_views.jl --record
```

The command runs the code cells in order, including independent expected
multiplicities and equations, and retains actual text/PNG outputs. An optional
`--gallery=PATH` exports PNG/SVG pairs from those same figure cells for review.
Article source, execution, site routing and public deployment remain distinct;
add a route only with the corresponding publication review.
